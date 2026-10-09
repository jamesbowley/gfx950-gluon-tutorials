# Stream-K partials through one XCD's L2

Can the programs of a stream-K tile exchange their partials through their XCD's L2 instead of
memory? The reduce-scatter ending of v20
([WORKLOG](../../kernels/gemm/intra_wave/a16w16/v20_streamk_reduce_scatter/WORKLOG.md)) writes
every partial through to memory (`.wt`), meets at a GPU-scope acquire-release barrier, and reads
the partials back past L2 (`.cv`). Section 10 of
[STREAMK_EXPLAINED.md](../../kernels/gemm/intra_wave/a16w16/v14_streamk/STREAMK_EXPLAINED.md)
explains why: L2 is per XCD and not coherent across XCDs. But if every program of a tile runs on
one XCD, that L2 is the point of coherence for all of them. This probe tests that before
[v21](../../kernels/gemm/intra_wave/a16w16/v21_streamk_l2_partials/WORKLOG.md) uses it.

```bash
HIP_VISIBLE_DEVICES=5 python bench.py                          # check, asm, warm timing (about 25 s)
HIP_VISIBLE_DEVICES=5 python bench.py --sweep time --cold      # a 1 GiB read before every launch
HIP_VISIBLE_DEVICES=6 python bench.py --sweep end              # launch time against the stamped span
HIP_VISIBLE_DEVICES=5 python bench.py --sweep pmc              # rocprofv3 TCC counters (about 4 min)
HIP_VISIBLE_DEVICES=5 python bench.py --sweep pmc_detail       # where the memory reads come from
```

Machine: MI355X (gfx950), 256 CUs, Triton `gfx950-tutorial-v2.2` + deepbind shim, no LLIR or
amdgcnas plugins. Raw output: [results/check_time_warm.md](results/check_time_warm.md),
[results/time_cold.md](results/time_cold.md), [results/end.md](results/end.md),
[results/pmc.md](results/pmc.md), [results/pmc_detail.md](results/pmc_detail.md). One k-step is
1.21 us.

## TL;DR

- **Exchanging through one XCD's L2 is correct, and only on one XCD.**
  - With a group's programs on one XCD, every L2 variant summed exactly in 20 of 20 launches, at every
    footprint from 1 to 8 MiB per XCD. `HW_REG_XCC_ID` confirmed the placement.
  - With the same programs spread over XCDs, every L2 variant was wrong in 20 of 20 launches.
  - The write-through variants were right both ways.
- **The barrier is the big win, not the partials.** v20's barrier costs 3.6-16 us. A flag per
  program holding the launch's epoch, polled with `sc0` loads that hit L2, costs 0.9-2.6 us.
  Over the whole ending (`l2_flags` against v20's `mem`) that is a 29-46% cut at every footprint.
  - Keeping the partials themselves in L2 saves 1-1.5 us on the store, at every footprint.
  - It saves nothing on the read: a read of a line that was written without first being
    read fetches the line from memory anyway (next point).
- **L2 holds the written lines, but they are not valid for reads.** The stores allocate lines
  in L2 without fetching them (the store-only run shows no memory reads). A later read of such a line
  counts as an L2 hit but still reads the whole line from memory, as many memory reads as the
  write-through path. Reading the slot's old contents into L2 first (`l2_pre`) removes those
  reads and halves the slice at 2-4 MiB per XCD (5.2 to 2.9 us).
- **Capacity: 4 MiB per XCD.** Up to 4 MiB of partials per XCD (16 programs) the stores and
  reads hit L2. At 8 MiB the hit rate falls from 50% to 30-37%, and the pre-read costs more than
  it saves.
- **The partials are still written to memory, at the end of the kernel.** They stay dirty in L2
  and are written back when the kernel ends: the write requests to memory do not change. After
  the last stamp, the L2 launch spends 1.2-2.1 us more than the write-through one at 2-4 MiB
  per XCD.
- **Cache bits.** On CDNA4 Triton's `.cg` load is `sc0 nt` (not `sc0` alone), `.wt` and `.cv` are
  `sc0 sc1`, and no modifier means no bits. `sc0 nt` misses L1 and hits L2: it works for the flag
  poll and the partial reads.

## Method

**Kernel** ([kernel.py](kernel.py)). A synthetic reduce-scatter ending, no GEMM. Groups of n
programs stand in for the n programs of a split tile; every XCD runs T groups, so an XCD holds
T n partials of 256 KiB (`T n / 4` MiB, against 4 MiB of L2). Each program:

1. loads its "accumulator" (a 256x256 fp32 slot of `src` plus the launch's epoch, so every launch's
   partials differ) in v14's MFMA layout and meets the grid at a barrier;
2. stores it to its slot of `P` (v20's lane-contiguous map), drains (`s_waitcnt vmcnt(0)` and a
   workgroup barrier);
3. meets its group at a tile barrier;
4. sums its 1/n of the tile over the group's n partials and stores that slice.

The host checks every group's output against the exact sum (small integers, exact in fp32).

**Placement.** `PLACE = 0` puts a group on one XCD: pid `p` runs on XCD `p % 8`
(the repo's measured dispatch, checked here with `HW_REG_XCC_ID`), so a group is `n` pids with
the same `p % 8`. `PLACE = 1` uses the same active pids grouped consecutively, so a group spans
`min(n, 8)` XCDs.

**Variants.**

| variant | partial store | barrier | slice loads |
|---|---|---|---|
| `mem` | `.wt` (`sc0 sc1`) | v20's counter: `acq_rel` arrival, release generation, acquire after the spin (`buffer_wbl2 sc1`, `buffer_inv sc1`) | `.cv` (`sc0 sc1`) |
| `wt_relaxed` | `.wt` | the same counter, relaxed atomics only | `.cv` |
| `l2_cg` | no bits | relaxed counter, then `buffer_inv sc0` (a NOP, see below) | `.cg` (`sc0 nt`) |
| `l2_def` | no bits | relaxed counter, `buffer_inv sc0` | no bits |
| `l2_cv` | no bits | relaxed counter, `buffer_inv sc0` | `.cv` |
| `l2_cg_noinv` | no bits | relaxed counter | `.cg` |
| `l2_nts` | `.cs` (`sc0 nt`) | relaxed counter, `buffer_inv sc0` | `.cg` |
| `l2_pre` | as `l2_cg`, after reading the program's own slot into L2 before the clock starts | | |
| `l2_flags` | no bits | flag barrier: store the epoch to this program's flag, poll the group's n flags with `sc0 nt` loads until all hold it; `buffer_inv sc0` | `.cg` |
| `l2_flags_pre` | `l2_flags` after the pre-read | | |

**`buffer_inv sc0` does nothing here.** It was added as an L1 invalidate, but the CDNA4 ISA
(Table 52) defines `buffer_inv sc0` as a NOP outside TG-split mode (TG-split lets one workgroup's
waves run on different CUs; it is off). The `.cg` (`sc0 nt`) and `.cv` (`sc0 sc1`) loads miss L1 by
themselves (Table 49: "Miss Evict"), so those variants were correct for that reason. `l2_def`, with
default loads, was correct only because the reading CU never had the partial's lines in its L1;
it is not a safe pattern. v21 no longer emits the instruction; see
[SYNC_REVIEW.md](../../kernels/gemm/intra_wave/a16w16/v21_streamk_l2_partials/SYNC_REVIEW.md).

The relaxed counter keeps the generation read ahead of the arrival with an explicit
`s_waitcnt vmcnt(0)`, which the `acq_rel` arrival used to provide. Without it the last arrival
and the generation bump can overtake the read, and the program waits for a generation that never
comes. The flag barrier needs no atomics and no re-arming: each flag has one writer and the host
passes a new epoch every launch.

**Timing.** `s_memrealtime` stamps (100 MHz) after `s_waitcnt vmcnt(0)` and a workgroup barrier:
start (after the grid barrier), stored, synced, done. Medians over programs and 20 launches.

## Results

### Correctness

Every variant, 9 configurations (T, n) from (1, 4) to (4, 8), 20 launches each, both placements:

- XCD-local placement: 0 wrong launches for every variant.
- Spread placement: every `l2_*` variant wrong in 20 of 20 launches; `mem` and `wt_relaxed`
  correct. The flag barrier is not run spread, because it would hang: a flag that sits in another
  XCD's L2 never arrives.
- `HW_REG_XCC_ID`: every XCD-local group ran on one XCD.

The relaxed counter barrier itself works across XCDs (`wt_relaxed`, spread): device-scope atomics
are performed past L2, as the atomics experiment found.

### Timing (warm), us

At 4 MiB per XCD, the default capped split of the GEMM (16 programs per XCD):

| T | n | variant | store | barrier | slice | ending | span |
|---|---|---|---|---|---|---|---|
| 1 | 16 | `mem` (v20) | 4.48 | 7.28 | 5.28 | 17.32 | 19.47 |
| 1 | 16 | `wt_relaxed` | 4.48 | 4.84 | 5.28 | 14.80 | 15.92 |
| 1 | 16 | `l2_cg` | 3.32 | 6.20 | 6.08 | 15.88 | 19.14 |
| 1 | 16 | `l2_flags` | 3.32 | 1.64 | 6.20 | 11.20 | 14.00 |
| 1 | 16 | `l2_flags_pre` | 3.28 | 1.72 | 4.68 | 9.60 | 11.16 |
| 2 | 8 | `mem` (v20) | 4.52 | 7.68 | 4.40 | 16.86 | 18.95 |
| 2 | 8 | `wt_relaxed` | 4.52 | 4.64 | 4.40 | 13.80 | 15.52 |
| 2 | 8 | `l2_flags` | 3.36 | 1.44 | 5.16 | 10.16 | 12.66 |
| 2 | 8 | `l2_flags_pre` | 3.28 | 1.56 | 3.00 | 7.84 | 9.82 |

The whole sweep, 1 to 8 MiB per XCD, is in [results/check_time_warm.md](results/check_time_warm.md).
- **Barrier.** v20's costs 3.6-16 us, more as the groups grow. Relaxed atomics alone save a third
  to a half; the flag barrier costs 0.9-2.6 us at every size.
- **Store.** 1-1.5 us faster in L2 (3.3 against 4.5 us at 4 MiB). One CU writes 256 KiB at
  about 75-125 GB/s either way, close to what one CU can move.
- **Slice.** No faster from L2 unless the slot was read first: 5.2-6.2 us for `l2_cg` against 4.4-5.3 for
  the write-through path at 4 MiB, and 2.9-4.7 us after the pre-read.
- **At 8 MiB per XCD** the L2 partials no longer fit: `l2_cg` is slower than `wt_relaxed`, and only
  the flag barrier still pays.

**Cold** ([results/time_cold.md](results/time_cold.md), a 1 GiB read before every launch). The
`l2_*` slices without the pre-read get slower, for example 7.2 us against 5.2 for `wt_relaxed` at (1, 16):
the fills now come from HBM rather than the last-level cache. The pre-read variants are
unaffected.

### Counters ([results/pmc.md](results/pmc.md), [results/pmc_detail.md](results/pmc_detail.md))

Per launch, at (1, 8), 2 MiB per XCD, 131K 128-byte lines of partials:

| variant | L2 hit rate | memory reads | memory writes |
|---|---|---|---|
| `mem`, `wt_relaxed` | 4.5% | 263K | 295K |
| `l2_cg`, `l2_def`, `l2_cv`, `l2_flags` | 50.7% | 263K | 295K |
| `l2_pre`, `l2_flags_pre` | 60.1% | 263K | 295K |

- Memory reads are 131K lines of `src` plus 131K lines of partials in every variant.
- Without the slice (store and barrier only) every variant reads 132K: the stores fetch nothing.
- So in `l2_cg` the partial reads count as hits but each still fetches its line from memory.
- `l2_pre` moves that fetch to before the clock and its slice reads hit outright.
- All traffic is `TCC_RW_REQ` (the write-back cacheable type); the `l2_*` variants have no
  `TCC_PROBE` or `TCC_BYPASS_REQ`, while `mem` has 262K of each, from its write-through stores.
- `TCC_ALL_TC_OP_WB_WRITEBACK` is 131K lines for `l2_cg` against 16K for `mem`: the dirty partials
  are written back at kernel end.

### Kernel end ([results/end.md](results/end.md))

Launch time (CUDA events) minus the stamped span, `l2_flags` against `wt_relaxed`: 1.2-2.1 us more
at 2-4 MiB per XCD, 0.4-1.1 us at 8 MiB, where most partials were evicted (and written back)
during the kernel. Part of the in-kernel gain moves to the kernel's end.

## What this means for v21

- Exchange the partials and the flags through L2 only with every program of a tile on one XCD,
  and check the placement on the host: a mistake is a hang or wrong results, never an error.
- Use the flag barrier: it is where most of the gain is.
- Keep the store in L2 (no `.wt`), read with `.cg`. A pre-read helps only if it is hidden and
  the footprint fits; v21 tests that in the GEMM.
- Expect part of the gain to go to the write-back at kernel end.
