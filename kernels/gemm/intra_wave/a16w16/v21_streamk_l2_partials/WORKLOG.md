# v21_streamk_l2_partials work log

One entry per change: what changed and what it did to performance and spills. Results come
from the published v10+ stack unless noted: Triton v2.2 + deepbind shim on `PYTHONPATH`,
`LLVM_PASS_PLUGIN_PATH=plugins/llir_scheduler/libLlirSched.so`, `TRITON_AMDGCNAS_PLUGIN=1`, run
from `a16w16/`. Raw tables are in `results/`. One k-step is about 1.21 us.

## 1. Partials and flags through one XCD's L2 (2026-10-03)

v20 with a second way to exchange the reduce-scatter partials. v20 writes every partial
through to memory (`.wt`), meets at a GPU-scope acquire-release barrier and reads the partials
back past L2 (`.cv`), because L2 is per XCD and not coherent across XCDs. If every program of a
tile runs on one XCD, that XCD's L2 is the coherence point for all of them, and the partials
and the flags can stay in it. [experiments/streamk_l2_partials](../../../../../experiments/streamk_l2_partials/README.md)
tested the idea on a synthetic ending first; this entry puts it in the GEMM.

**`STREAMK_PARTIALS`** (constexpr `PARTIALS`; default `mem`):

| mode | partial store | barrier | slice loads |
|---|---|---|---|
| `mem` | `.wt` (`sc0 sc1`) | v20's counter: `acq_rel` arrival, release generation, acquire after the spin (`buffer_wbl2 sc1`, `buffer_inv sc1`) | `.cv` (`sc0 sc1`) |
| `wt` | `.wt` | the same counter with relaxed atomics only | `.cv` |
| `l2` | no cache bits | a flag per program, no atomics | `.cg` (`sc0 nt`) |

- **`wt`**: the drain before the arrival (`s_waitcnt vmcnt(0)` per wave, then a workgroup
  barrier) already waits for the write-through stores, so the acquire-release adds only the L2
  write-back and invalidate. An explicit `s_waitcnt vmcnt(0)` keeps the generation read ahead
  of the arrival, which the `acq_rel` arrival used to do: if the last arrival and the bump
  overtook that read, the program would wait for a generation that never comes. The owner
  ending's flags become relaxed too.
- **`l2`**: program `spid` stores the launch's epoch to its flag `tile_sync[2 * 256 + spid]`,
  then polls the tile's n flags with `sc0 nt` loads, which miss L1 and hit the XCD's L2, until
  all hold it.
  - The host passes the epoch (`do_not_specialize`) and bumps it every launch, so the flags
    need no re-arming.
  - An asm memory clobber in the poll loop keeps the loads inside it. Another, after the loop,
    keeps the slice loads below it.
  - The slice loads are `sc0 nt` too, which always miss L1 (CDNA4 ISA Table 49, "Miss Evict"),
    so no L1 invalidate is needed.
  - The owner ending is unchanged apart from relaxed flags, as in `wt`.

  The first version also ended the wait with an inline-asm `buffer_inv sc0`, meant as an L1
  invalidate. The ISA (Table 52) defines it as a NOP outside TG-split mode, so it did nothing;
  it was removed and the stress test repeated (`results/check_streamk_no_inv.md`, 12
  configurations x 50 launches, correct). [SYNC_REVIEW.md](SYNC_REVIEW.md) shows the source and
  assembly of every mode's synchronization, and the assumptions behind it, for hardware review.

**Placement.** `l2` needs every program of a tile on one XCD, so it turns on the XCD-local
split (`STREAMK_XCD_LOCAL`, constexpr `XCD_LOCAL`, `xcd_local_chunk_of`):
- XCD x owns `S // 8` consecutive stream-K tiles, one more for `x < S % 8`.
- Its local slot `l = spid % 32` works on its tile `l // n`, as chunk `l % n`.
- So a tile's n programs are consecutive spids on one XCD, its slots and flags are
  `b .. b + n - 1` as before, and the XCD of a slot never changes between launches.
- n is `STREAMK_NUM_PROGRAMS // S` (the capped rule's split), capped at `32 // ceil(S / 8)`
  and at one program per pair.
- For S < 8 only S XCDs work. For S > 128, n = 1 and the tiles run whole (the capped rule
  picks data-parallel there anyway).

v20's balanced placement already puts each tile on one XCD when S is a multiple of 8 and n
divides `256 / S`. Under the capped rule it spans XCDs for every other reduce-scatter S
(`check_partition.py`, `results/partition.md`): every tile for S = 1-7, and 3-7 of the tiles
for S = 9-31 that are not multiples of 8.

**Host.**
- In `l2` mode the host replays the placement (`tile_groups`) and asserts that no tile spans
  XCDs. A tile that did would read stale partials or hang.
- `STREAMK_UNSAFE_PLACEMENT=1` skips the assertion (the negative control).
- The check is cached per configuration (`first_straddling_tile`). Uncached, the replay's Python
  loop over 256 programs ran on every launch: 4352x4096x4096 (a 120 us kernel) then lost 15-20% under
  do_bench and back-to-back launches, because the GPU waited on the host.
- `tile_sync` grows to `5 * 256` words: counters and generations, L2 flags, and the debug
  records of `STREAMK_RECORD_XCC=1`, which writes each slot's tile and `HW_REG_XCC_ID`.

**Assembly** (`asm_check.py`, `results/asm_check_published_stack.md`, 4352x4096x8192).
- All three modes: 484 VGPRs, no spills.
- `mem` has v20's two `buffer_wbl2 sc1` and two `buffer_inv sc1`, and `sc0 sc1` partial stores
  and loads.
- `wt` has no cache maintenance.
- `l2` has no `sc1` access, no atomics and no cache maintenance at all. Its slice and
  flag-poll loads are `sc0 nt`.

On CDNA3/4 Triton's `.cg` load is `sc0 nt`, not `sc0` alone (`getCtrlBitsForCacheModifierOn_CDNA3_CDNA4`).

**Correctness.**
- `check_streamk.py`: 30 rotating launches per configuration, on the four plan shapes and the
  9 stream-K shapes:
  - all three modes, capped and full-split one-tile, with and without bias: 132
    configurations;
  - all correct, every atomic flag and count re-armed, every `l2` flag holding the last epoch
    (`results/check_streamk_published_stack.md`; without the plugins, `results/check_streamk.md`);
  - 200 launches on the four plan shapes in `l2`: correct (`results/check_streamk_l2_long.md`).
- `STREAMK_RECORD_XCC` confirms the placement on the hardware: `l2` never had a split tile on
  two XCDs. `mem` and `wt` had all 4 at S = 4 and 1-3 of 33 at S = 33.
- **Negative control** (`--unsafe-placement`): `l2` with v20's placement on S = 4 and S = 2, where
  each tile spans 2-4 XCDs. All four configurations hang: a flag written into one XCD's L2 never
  reaches the other. FAIL, as required (`results/check_streamk_negative.md`). The GPU
  recovered after the hung processes were killed. The probe showed the other failure, wrong
  results, when the flags are coherent but the partials are not.

### Where the ending's time goes

`fixup_timing.py` (`results/fixup_timing.md`): `s_memrealtime` stamps in a patched copy, one-tile
reduce-scatter with n programs per tile. Medians over the split programs, in us:

| shape | S | n | MiB per XCD | mode | MAC | store | barrier | slice | ending | tail |
|---|---|---|---|---|---|---|---|---|---|---|
| 3072x5632x8192 | 8 | 16 | 4 | mem | 13.1 | 5.8 | 4.3 | 8.8 | 19.5 | 45.2 |
| | | | | wt | 13.4 | 5.6 | 2.3 | 8.2 | 16.8 | 45.7 |
| | | | | l2 | 12.9 | 3.4 | 1.2 | 9.4 | 14.7 | 42.2 |
| 4352x4096x8192 | 16 | 8 | 4 | mem | 20.2 | 5.5 | 4.4 | 8.9 | 18.9 | 54.6 |
| | | | | wt | 20.2 | 5.0 | 2.2 | 8.6 | 16.2 | 51.4 |
| | | | | l2 | 20.3 | 3.3 | 1.0 | 9.0 | 13.7 | 49.1 |
| 4096x4608x8192 | 32 | 4 | 4 | mem | 39.2 | 4.6 | 5.9 | 9.8 | 20.3 | 69.9 |
| | | | | wt | 39.6 | 4.4 | 2.5 | 9.8 | 17.0 | 66.6 |
| | | | | l2 | 39.2 | 3.3 | 0.8 | 10.0 | 14.1 | 60.2 |
| 3328x5120x8192 | 4 | 16 | 4 | mem | 11.0 | 3.4 | 3.4 | 8.2 | 15.0 | 37.6 |
| | | | | l2 | 11.0 | 3.1 | 1.1 | 8.9 | 13.2 | 35.9 |
| 4352x4096x8192 | 16 | 16 | 8 | mem | 18.6 | 11.6 | 7.5 | 9.6 | 28.0 | 59.6 |
| | | | | l2 | 18.5 | 11.1 | 1.3 | 10.3 | 23.2 | 55.6 |

(The capped rule's n for each shape is the first block. The table has every split from 1 to
8 MiB per XCD, and K = 4096.)

1. **The barrier is most of the gain.** v20's costs 3-9 us (more as tiles are split further).
   Relaxed atomics halve it (`wt`); the L2 flags cut it to 0.5-1.7 us at every size.
2. **The store is 0.3-2.4 us faster in L2** at up to 4 MiB per XCD (the least at S = 4).
3. **The slice does not get faster: 8-11 us in every mode.**
   - The probe explains part of it. A read of a line that was written into L2 without being
     read first still fetches the line from memory, so the partial reads cost the same as from
     memory.
   - But it is not the main limit. In the probe the same slice takes 3-6 us, and here the
     write-through slice is no faster than the L2 one.
   - So `reduce_scatter_slice` itself sets its 8 k-steps. Entry 2 found the cause: the lane
     order of its loads, not the bias or the C store.
4. **The ending gets 2-6 us shorter** (`l2` against `mem`), 12-31%. The stream-K tail changes
   by between 1 us longer (S = 4, n = 32, where only 4 XCDs run the MAC) and 10 us shorter.
5. **8 MiB per XCD** (n = 16 at S = 16, n = 32 at S = 8, n = 8 at S = 32): the partials no
   longer fit in L2 and the store gains at most 2 us, but the flags still save 6-7 us.

**Pre-touching the partial slot does not pay.** In the probe, reading the slot into L2 before
storing removes the read-time fill and halves the slice at 2-4 MiB. In the GEMM (an experimental
copy, not kept) the touch of one dword per 128-byte line, at the end of the MAC loop or in its
last pair:
- cut the slice by about 1 us (9.2 to 8.2);
- lengthened the MAC by 5-6 us, so the tail got longer and the kernel did not change.

### Kernel time

**Cold rocprof** (`rocprof_check.py`, `results/rocprof.md`: 1000 back-to-back launches over
rotating inputs, GPU 3), us:

| shape | S | v19 | v20 | v21 wt | v21 l2 | l2 against v20 |
|---|---|---|---|---|---|---|
| 3072x5632x8192 | 8 | 214.4 | 218.1 | 215.8 | 214.4 | -1.7% |
| 4352x4096x8192 | 16 | 221.2 | 227.0 | 225.4 | 224.4 | -1.1% |
| 4096x4608x8192 | 32 | 231.6 | 243.5 | 239.9 | 237.7 | -2.4% |
| 4352x4352x8192 (owner ending) | 33 | 233.6 | 245.2 | 245.0 | 243.4 | -0.7% |
| 4352x4096x4096 | 16 | 126.3 | 128.6 | 124.6 | 123.8 | -3.7% |
| 3328x5120x8192 | 4 | 214.2 | 214.4 | 212.6 | 199.0 | -7.2% |
| 1536x11008x8192 | 2 | 203.6 | 215.1 | 213.7 | 200.3 | -6.9% |

**The S = 2 and 4 gains are GPU 3's, not `l2`'s.** At S < 8 the XCD-local split runs the tail on
only S XCDs. GPU 3 is about 10% slower than GPUs 4 and 6 on these shapes, which fits some of its
XCDs being slower. On GPUs 4 and 6, against v20:
- 3328x5120x8192: `l2` +0.5-0.6% (slower), `wt` -0.4-0.8%;
- 1536x11008x8192: `l2` -1.6-2.2%;
- 3072x5632x8192: `l2` -1.0-1.4%;
- 4352x4096x4096: `l2` -0.8-0.9%.

**do_bench** (`results/do_bench.md`, median of 3, GPU 3), TFLOPS:

| shape | S | v19 | v20 | v21 mem | v21 wt | v21 l2 |
|---|---|---|---|---|---|---|
| 3072x5632x8192 | 8 | 1274 | 1281 | 1279 | 1300 | 1302 |
| 4352x4096x8192 | 16 | 1267 | 1289 | 1283 | 1298 | 1299 |
| 4096x4608x8192 | 32 | 1278 | 1255 | 1255 | 1275 | 1281 |
| 4352x4352x8192 | 33 | 1234 | 1262 | 1261 | 1265 | 1270 |
| 3072x5632x16384 | 8 | 1381 | 1363 | 1363 | 1372 | 1378 |

The K = 4096 and S < 8 rows are left out:
- At K = 4096 the kernel (about 120 us) is shorter than the host's 165 us per `matmul` call,
  so do_bench measures the launch path.
- At S < 8 the row is GPU 3's slow XCDs again.

**Between launches** (`between_launches.py`, `results/between_launches.md`, GPU 6, CUDA events,
K = 8192 shapes):
- after a 256 MB read, a 64 MB zero or an idle gap, `wt` and `l2` are level with v20 or up to
  1.4% ahead, except `l2` at S = 4 (0.9-1.3% behind);
- after do_bench's 256 MB zero, `l2` is 1.5% ahead at 3072x5632x8192, level at S = 16 and 32,
  and 1.8% behind at S = 4;
- back to back, and at K = 4096 after a zero, the GEMMs wait on the host's launch path.

**The partials still reach memory, at the end of the kernel.**
- In `l2` they are dirty in L2 when the kernel ends and are written back then: the memory write
  requests do not change (probe, `TCC_EA0_WRREQ`).
- The probe measured 1.2-2.1 us more after the last stamp at 2-4 MiB per XCD.
- So part of the shorter ending moves to the kernel's end, and kernel time gains less than
  the tail.

### Results

1. **Same-XCD exchange through L2 works and is correct.** On MI355X, with every program of a
   tile on one XCD:
   - default-policy stores, an epoch flag polled with `sc0 nt` loads and `sc0 nt` reads gave
     exact results in every stress configuration;
   - the same code across XCDs hangs.
   - It leans on hardware behaviour the memory model does not promise:
     - L1 is write-through, and `sc0 nt` loads always miss it (ISA Table 49);
     - a store's `vmcnt` completion means it is in L2;
     - pid `p` runs on XCD `p % 8`.
   - The stress tests, the `HW_REG_XCC_ID` records and the negative control are what back it.
     [SYNC_REVIEW.md](SYNC_REVIEW.md) lists these assumptions for a hardware engineer to check.
2. **The win is the barrier, not keeping partials out of memory.**
   - The flag barrier cuts v20's 3-9 us to about 1 us.
   - The store gains up to 2.4 us. The slice gains nothing.
   - In the kernel the ending is 2-6 us shorter. Under cold rocprof, kernel time is 1-2.4%
     faster at K = 8192 on the S = 8-32 shapes, about 1% at K = 4096, about 2% at S = 2, and
     level to 0.6% slower at S = 4.
3. **`wt` is a safe middle step**: write-through partials with a relaxed counter. It needs no
   placement, works across XCDs and gets about half of `l2`'s gain.
4. **v19 still wins on some shapes.** Under cold rocprof, two-tile is ahead of `l2` on
   4352x4096x8192, 4096x4608x8192 and 4352x4352x8192, and level on 3072x5632x8192. v21 does not
   change v20's conclusions about one-tile against two-tile.

**Decision.**
- The default stays `STREAMK_PARTIALS=mem`, so v21 is v20 unless asked. `l2` gains 1-2%, and
  it depends on placement and on hardware behaviour beyond the memory model. Flip it after a
  wider sweep (the leftover-S shapes and the real-model GEMMs of `experiments/streamk_capped_vs_auto`)
  on more than one GPU.
- **No quadrant staging.** It would keep the footprint under 4 MiB. But at the capped rule's
  split the footprint is 4 MiB or less except for S = 9-14, 17-21 and 25 (up to 7 MiB on the
  busiest XCD). The slice, the only phase that staging could speed up, is not limited by where
  the partials live. And every extra round costs a barrier.
- **No two-level reduction for S < 8.** Running the tail on fewer XCDs cost 1-2 us of MAC at
  S = 4, n = 32, and `l2` is within 0.6% of v20 there.

### Next

- **The memory system, measured** ([experiments/l2_memsys](../../../../../experiments/l2_memsys/README.md), GPU 1):
  - The first read of a line written into L2, but not valid there before, always fetches the
    whole line from wherever its old contents live (HBM at 369 ns, the LLC at 225 ns, against
    85 ns for an L2 hit), whatever part of the line was written. That is why the `l2` slice is
    no faster.
  - Streaming stores (`.cs`) to absent lines are dropped from L2 almost at once.
  - The L2 keeps at most about 2 MiB per XCD dirty. Writing it back at kernel end costs 1-3 us
    after the last instruction, and nothing is deferred to the next kernel.
- **The slice**: done in entry 2. What is left there is the `l2` fill (6.1-6.6 us of reads
  against 4.5-5 under `mem`) and the C store's 4-rows-per-cycle pattern.
- **A flag barrier without atomics for `wt`**: write-through flags polled with `.cv` would work
  across XCDs. Does it get close to `l2`'s barrier without the placement constraint?
- **GPU 3's XCDs.** Time a fixed tail per XCD to see whether parts differ between XCDs, and
  whether the XCD-local split should avoid slow ones.

## 2. Slice reads in lane order (2026-10-04)

All on GPU 1. `slice_timing.py` adds a stamp inside `reduce_scatter_slice`, after its reads, and
can swap in the streamk_l2_partials probe's loop (no C store, timing only)
([results/slice_timing.md](results/slice_timing.md)).

1. **The slice is almost all reads.**
   - Reads take about 8 us in every mode; the bias and C store take 0.6-2.9 us.
   - The probe's loop reads the same partials, in the same kernel, in 3.9-4.4 us under `mem`.
   - Both loops issue the same 16-byte loads with the same waits, so the difference is the
     addresses, not the instruction count.
2. **The cause is lane order.**
   - The slice's layout (`BlockedLayout([1,1,4], [1,16,4], [1,2,2], [2,1,0])`) put columns first.
     Lane `l` read float `4 (l // 4) + 64 (l % 4)` of each 1 KiB block.
   - So the 4 lanes the L1 handles per cycle hit 4 lines 256 bytes apart. The L1 looked each
     line up 8 times instead of 2, and lines 256 bytes apart share a tag-RAM set.
   - The counters, against the probe loop:
     - L1-to-L2 read requests are the same (18.48M);
     - L1 accesses are 1.64M higher (predicted: 1.57M);
     - there are 262K read tag-conflict stalls, one per partial line, against 0;
     - address-stall cycles are 0.76M higher.
3. **The fix.** Order `[1, 2, 0]`: lane `l` gets row `l % 16` and columns `4 (l // 16) .. + 3`,
   which `in_blk` puts at `4 l .. 4 l + 3`. The same one-line change is in v20.
   - Reads fall to 4.5-5.0 us under `mem` (within 0.6 us of the probe loop) and to 6.1-6.6 us
     under `l2`.
   - Bias + C rises 0.3-1.3 us, because 4 lanes now write 4 rows.
   - The slice under `mem` drops from 9.1-10.2 us to 6.1-8.1 us.
   - Tried and dropped: rows-first loads, then `gl.convert_layout` to columns first for the C
     store. The conversion goes through LDS, and bias + C rose to 2.4-10.5 us.
4. **Kernel time** ([results/rocprof_lane_order.md](results/rocprof_lane_order.md); cold rocprof;
   two runs in opposite orders; v19 as the drift reference, within 1.4 us):

   | | gain on the 6 slice shapes | 4352x4352 (S = 33, owner fixup) |
   |---|---|---|
   | v20 | 1.9-4.5 us | 0 |
   | v21 `mem` | 2.8-3.7 us | -0.2 |
   | v21 `l2` | 1.2-2.4 us | 0.6 |

   - `l2` gains less because its reads are limited by the L2 fill, not the L1.
   - `l2` and `mem` are now within a few us of each other either way, so `mem` stays the default.
5. **Correctness.**
   - `check_streamk.py --policies capped one_tile` passes for v21 in all three modes.
   - v20's `check_streamk.py` passes 108 configurations.
   - `asm_check.py`: the slice loads are still `buffer_load_dwordx4`, and the `l2` checks pass.

## 3. One-wave data-parallel calls v9 (2026-10-06)

The same host change as [v20 WORKLOG entry 10](../v20_streamk_reduce_scatter/WORKLOG.md), which has
the reasoning and the v20 numbers.
- When `capped` or `auto` picks dp and `total_tiles <= NUM_PROGRAMS`, `matmul` returns
  `v9_beyond_hotloop`'s `matmul`. An explicit `STREAMK_POLICY=dp` still runs the persistent loop.
- Only when `c.dtype == a.dtype`, as in v20: v9 stores C in A's dtype, and v21 stores it in C's.
  On 4096x4096x8192, an fp32 C now stays on the persistent loop and matches the reference, with
  and without bias; a bf16 C still goes to v9.
- `rocprof_check.py` traces v9's kernel for v21 too (`kernel_regex`).
- **Kernel time** ([v20 results/v9_fallback.md](../v20_streamk_reduce_scatter/results/v9_fallback.md),
  GPUs 3-5): -2.7% and -2.9% at 16 k-tiles (2048x2048x1024, 3840x4096x1024), and -1.1% to +0.6%
  (noise) at 128 k-tiles (4096x4096x8192, 3840x4096x8192, 256x49152x8192).
- **Correctness:**
  - v20's `check_v9_fallback.py` covers v21 too: every case is right and runs the expected kernel.
  - `check_streamk.py` with its default configurations: 66 pass (11 shapes, all three partials modes,
    with and without bias), with no stale flags or counts.

## 4. `wt` becomes the default; every version syncs this way (2026-10-08)

[memory_model_litmus](../../../../../experiments/memory_model_litmus/WRITEUP.md) recommends `wt`'s
sequence for any placement: write-through partials, a drain in every wave, a workgroup barrier,
relaxed `gpu`-scope flags or counters, and L1-missing (`.cv`) reads with no acquire. So:
- `STREAMK_PARTIALS` now defaults to `wt`; `mem` (the compiler's release and acquire) and `l2`
  stay available. Entry 1's "the default stays `mem`" no longer holds.
- v15-v20 and v22 use the same sequence in their owner flags and reduce-scatter barriers, so v20
  is now what `wt` was, and `mem` is v20 before this date.
- `rocprof_check.py` knows v22, and `sync_asm_check.py` counts `buffer_wbl2` / `buffer_inv` in
  every version's compiled kernel (none are left except under `mem`).
- `sync_wt_compare.py` builds each version's `results/sync_wt.md` from the raw tables in
  [results/sync_wt](results/sync_wt/): old against new code, GPU 0, no amdgcnas hook, two
  interleaved rounds.

For v21 itself ([results/sync_wt.md](results/sync_wt.md)):
- **Correct:** `check_streamk.py`'s 66 default configurations (all three modes) and 132 with
  `--partials wt --policies one_tile two_tile --fixups rs owner --orders v9 split` pass.
- **The ending** (`fixup_timing.py --modes mem wt`, one run): `wt`'s barrier takes 1.8-3.6 us
  against `mem`'s 3.1-8.0 us, and the ending is 0.6-5.0 us shorter at every split count.
- **Kernel time:** with clean caches, 0-2.9 us faster. Cold rocprof: the default (now `wt`)
  against the old default (`mem`) moves by -1.3 to -0.5 us, within the spread of the rounds.
