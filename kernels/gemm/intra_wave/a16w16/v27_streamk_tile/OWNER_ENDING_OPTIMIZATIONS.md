# v27 owner ending: optimizations to try (2026-10-19)

On 1024x1024x7168, v27's owner ending is slower than hipBLASLt's at the same split. Both use a 128x128x128 tile
with each tile split over n programs, and the same kind of ending: the program that starts the tile owns it, and
the others write fp32 partials and exit. The main loops are level, so the gap is in the owner's fix-up.

| Programs per tile | hipBLASLt 1.2.2 | v27 owner ending | v27 vs hipBLASLt |
|---:|---:|---:|---:|
| 1 (data-parallel, no fix-up) | 256.2 | 258.5 | +0.9% |
| 2 | 407.0 | 384.4 | -5.6% |
| 4 | 546.0 | 484.2 | -11.3% |

TFLOPS, cold, torch 2.10 / hipBLASLt 1.2.2, physical GPU 1 (`amd-smi` 1, PCI bus `0x15`). hipBLASLt's split was
forced with `TENSILE_STREAMK_FIXED_GRID`. v27's reduce-scatter ending on 4 programs runs 541.3 (-0.9%).

The measurements, both kernels' ISA and the source comparison are in
`/home/jbowley/repos/aiter/streamk_study/2026-10-19/README.md` (with `results/isa/`).

Line numbers below are this directory's `matmul_kernel.py` (sha256 `1175b73525d3cc3f`, the snapshot timed).

## 1. Read each peer's partial in large batches (the main one)

**Problem.** The owner reads the partials one quadrant at a time, and each quadrant's peer loop ends every peer
with `s_waitcnt vmcnt(0)`.
- At 256x256 a quadrant is 64 KB, 16 `buffer_load_dwordx4` per wave per trip. That is the same as hipBLASLt's
  256x256 SK3 kernel (4 batches of 16 loads per peer, no spills) and is fine.
- At 128x128 a quadrant is 16 KB, only 4 loads per trip. With 3 peers that is 12 dependent trips, against
  hipBLASLt's 3 trips of 16: it reads a peer's whole 64 KB partial in one batch.
- Loads in flight set the read rate: one CU reads at 15 GB/s with 1 load per step, 58 GB/s with 8 and 71 GB/s with 16
  ([memory_model_litmus](../../../../../experiments/memory_model_litmus/WRITEUP.md), section 10).

**Change.** For tiles of 128x128 and below, issue the loads for a peer's whole tile (all 4 quadrants, 16 loads)
before waiting. The alternative is several peers' quadrants at once. Then sum into the peer sums (`psum`) and add
those to the accumulator once, as now.
- Keep the accumulator out of any loop or branch. That is what removed v16's spills (v17 WORKLOG, entry 1), and it
  doesn't prevent larger batches.
- Leave the 256x256 path as it is.

**Registers.** The fp32 `owall` build uses 256 of the 512 registers per lane (176 VGPRs + 80 AGPRs). A 16-load
batch needs about 64 more VGPRs.

**Where:** `fixup_quadrant` (lines 728-819), and the owner branch that calls it once per quadrant (lines 1500-1528).

**Expected:** most of the 3.5 us gap at 4 programs per tile and the 2.2 us at 2. With one peer this is 4 trips
against hipBLASLt's 1.

## 2. Poll with `gl.atomic_poll` instead of a scalar `atomic_add`

**Problem.** `gl.atomic_add(flag, 0, sem="relaxed", scope="gpu")` (lines 1504-1506) is lowered as one elected lane
doing `global_atomic_or ... sc0`. The result is broadcast through LDS, so each poll iteration costs an atomic round
trip, `ds_write_b32` / `ds_read_b32` and 2 `s_barrier`s, plus 4 bytes of LDS.

**Change.** `gl.atomic_poll(flag, 1, sem="relaxed", scope="gpu")`, as the small-tile owner already does
(lines 2001-2004).
- It compiles to a 4-instruction spin on lane 0 of wave 0 (`global_load_dword ... sc1`, wait, compare, branch),
  with one `s_barrier` after it and no LDS.
- The litmus write-up lists this relaxed `gpu` spin as correct for any placement. The `.cv` partial reads need no
  acquire.
- It is a vector load on one lane, not a scalar load. hipBLASLt polls with `s_load_dword ... glc` on every wave,
  which LLVM will not emit for a flag that another program writes.

**Side effect.** It removes the atomics on the shared flag line. Flags are one int32 per stream-K program, and an
XCD's 32 programs are consecutive spids, so one XCD's flags fill one 128-byte line. On this shape that line has
8 owners polling it with atomics and 24 peers setting their flags with atomic swaps.

**Cost.** v27 at 256x256x64 will no longer compile to v26's exact instructions (WORKLOG section 3, the gate), so
the quadrant builds' checks and timings need re-running.

**Expected:** smaller than change 1, probably well under 1 us.

## 3. Optionally, wait for and read one peer at a time

**Change.** Rather than waiting for all peers' flags before the first read (lines 1504-1506), wait for peer 1,
read it, then wait for peer 2, as hipBLASLt does. Reading the early peers then overlaps with the later peers
still storing.

**When.** Try it after change 1. It fits naturally if change 1 batches peer by peer.

> **Comment (2026-10-19):** We think this was tried before and was slower, but no WORKLOG from v15 to v27 records
> it, so the details are lost.
> - If it was slower, the poll is a likely reason: each wait is the elected-lane atomic with an LDS broadcast and
>   2 barriers per iteration, and splitting the waits between reads puts that cost between every peer.
> - The reads were also only 4 loads per trip at the time, so there was little read time to overlap.
> - With `gl.atomic_poll` (change 2) and 16-load batches (change 1), the result may be different, so re-measure
>   it with both in place.
> - The wait is a real share of the ending. v20's stamped timings at 256x256 (v20 WORKLOG, "Where the tail's time
>   goes") put the owner's wait for the last flag at 6-21 k-steps, against 7-125 k-steps for its reads and C store.

## 4. Optionally, a compile-time peer count

**Change.** One-tile splits already have `N_LO` and `N_HI` at compile time. Looping over the peers with
`static_range` would unroll the loop and let the compiler put several peers' loads in flight.

**When.** This is an alternative route to change 1's loads in flight, not something hipBLASLt does: its peer loop
is a run-time loop, and only the loads within a peer are generated at compile time. Today the fp32 peer loop is
not unrolled (one peer per iteration). That is LLVM's choice: Triton's unroll pass acts only on loops marked
`loop_unroll_factor`.

## Keep as is

**The `.cv` loads and `.wt` stores on the partials.**
- On gfx950 write-through data is read back from the LLC whatever bits the load has, so the bits cost nothing here.
- hipBLASLt's plain loads are correct only because of its access pattern: one workgroup per CU, L1 invalidated at
  launch, and each slot read once, after its flag.
- `.cv` keeps v27 correct whatever a CU read earlier in the launch.

## How to measure

- **Correctness:** `check_streamk.py` stress checks at 128x128x128 and the other quadrant tiles, owner ending,
  n = 2-8, with every flag re-armed.
- **ISA** (`asm_check.py`, or the harness's Triton cache):
  - the `owall` fix-up should show about 16 `buffer_load_dwordx4` per `vmcnt(0)`, and a `global_load_dword ... sc1`
    poll with no `ds_write` / `ds_read`;
  - 256x256 builds should be unchanged, except for the poll if change 2 is made.
- **Performance:** re-run the 2026-10-19 sessions (that README's "How to reproduce" section,
  `results/sk1024_picks.csv`, with `TENSILE_STREAMK_FIXED_GRID` = 128 and 64).
  - The target is `owall` at about 546 TFLOPS and `ow2` at about 407 (hipBLASLt at the same split).
  - `dp` 128x128x128 should stay at about 258, and `rsall` (541) is the control.
- **Wider check:** a faster owner ending may move `capped_policy`'s crossover, which today picks reduce-scatter
  whenever there are 4 or more programs per leftover tile below one wave. Re-run the owner-against-reduce-scatter
  comparison by n from the 2026-10-17 worst-shape sweep.
