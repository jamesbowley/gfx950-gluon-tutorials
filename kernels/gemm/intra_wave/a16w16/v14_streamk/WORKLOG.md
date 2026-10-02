# v14_streamk work log

One entry per change: what changed and what it did to performance and spills. Results come
from `HIP_VISIBLE_DEVICES=2 python regress.py ...` (bf16 unless noted), run from `a16w16/`
under the published v10+ stack: Triton v2.2 + deepbind shim on `PYTHONPATH`,
`LLVM_PASS_PLUGIN_PATH=plugins/llir_scheduler/libLlirSched.so`, `TRITON_AMDGCNAS_PLUGIN=1`.
The four regression shapes (4096x4096x8192, 8192x8192x8192, 4096x8192x4096,
4096x106496x16384) all have `STREAMK_TILES = 0`; 4352x4096x8192 (272 tiles,
`STREAMK_TILES = 16`) exercises the tail. Run-to-run noise on 4352x4096x8192 is about
±1.5% (v13 alone: 1029-1043 TFLOPS over four runs).

## 1. Peel the last wave out of v13's persistent loop (2026-09-29)

Copy of v13. The per-tile body moves into `persistent_tile`; the persistent loop stops at
`total_full_tiles = total_tiles - STREAMK_TILES`, and programs with `start < STREAMK_TILES`
run one more full-K tile after it. `STREAMK_TILES = total_tiles % NUM_PROGRAMS` is computed
on the host; `STREAMK_NUM_PROGRAMS` is passed but unused. Same work and order as v13.

Result on 4352x4096x8192 (only shape run so far): correct, no spills (0 VGPR / SGPR spills,
no scratch ops), VGPRs 480 -> 488 (bias 488 -> 496), AGPRs 256 both. TFLOPS vs v13 over
three runs: -1.8%, -0.4%, +0.6% (bias, one run: -1.8%), i.e. within noise.

| shape | version | TFLOPS | VGPR | AGPR | spills |
|---|---|---|---|---|---|
| 4352x4096x8192 | v13 | 1041.3 / 1028.9 / 1032.1 | 480 | 256 | 0 |
| 4352x4096x8192 | v14 | 1022.7 / 1025.2 / 1038.5 | 488 | 256 | 0 |
| 4352x4096x8192 bias | v13 | 1043.4 | 488 | 256 | 0 |
| 4352x4096x8192 bias | v14 | 1024.3 | 496 | 256 | 0 |

Full regression run (bf16, no bias): all correct, no spills. With `STREAMK_TILES = 0` the
tail is compiled out and v14 has v13's register counts; the +8 VGPRs only appear when the
tail is present. All deltas are within noise.

| shape | v13 TFLOPS | v14 TFLOPS | vs v13 | VGPR v13 / v14 | AGPR | spills |
|---|---|---|---|---|---|---|
| 4096x4096x8192 | 1488.4 | 1499.8 | +0.8% | 480 / 480 | 256 | 0 |
| 8192x8192x8192 | 1612.6 | 1615.6 | +0.2% | 480 / 480 | 256 | 0 |
| 4096x8192x4096 | 1465.8 | 1482.0 | +1.1% | 480 / 480 | 256 | 0 |
| 4096x106496x16384 | 1672.9 | 1688.9 | +1.0% | 480 / 480 | 256 | 0 |
| 4352x4096x8192 | 1040.4 | 1034.8 | -0.5% | 480 / 488 | 256 | 0 |

## 2. Design notes for the stream-K tail (2026-09-30)

Not a code change: open questions assessed against the TensorAtlas reference and the schedule
model in `streamk_diagrams.py`. Details, figures and numbers are in `STREAMK_EXPLAINED.md`.

- **Fixup waits.** The owner does not need all peers before it starts; it can add partials as
  they land (a fixed order keeps results bit-reproducible). Under two-tile SK + DP
  (`STREAMK_TILES = total % P + P`, which needs at least one full wave) every tile has at most
  one peer, `pid + 1`, which published its partial before the owner finishes, so waits are ~0
  and no peer-count precompute is needed. The cost left is the 256 KiB read. It cannot sit next
  to the 256-AGPR accumulator (one 128x128 fp32 quadrant is 64 VGPRs; v14 is at 488), so read
  it back quadrant by quadrant (or half-quadrants, or staged through the free 128 KiB of A/B
  LDS), interleaved with the 4-quadrant C store.
- **Split policy** (doc section 9). Tail time = compute your chunk + collect the partials, and
  today collection is ~4x the compute (4352x4096x8192 at c = 2 k-steps per partial: 8 + 2 + 30
  = 40, against 128 for the data-parallel last wave). Options: two-tile (at most one peer per
  tile; same total partials read, shorter serial chain); (a) tile-aligned cuts (removes tile
  switches and 1-3 k-step segments; small gain on its own); (b) fewer programs per tile (best
  s about sqrt(128 / c), 32 at s = 8); (c) reduce-scatter, where each program sums 1/s of the
  tile from all partials (about 12). Model numbers, not measurements; c still to be measured.
- **Prefetching the first stream-K segment.** The stream-K first segment can be prefetched (its
  address depends only on pid), but not with v14's k-step-0/1 pair pipeline. Segments start at
  any k, can be odd-length (L = 17 on 4352x4352) and can be 1-3 k-steps long. Fix: partition
  stream-K work in units of 2 k-steps, so every segment is even-length and starts on an even
  k-step and the pair loop runs unchanged. No CUs are given up; the longest range on 4352x4352
  one-tile goes from 17 to 18 k-steps (about 0.7% under two-tile). A 2-step segment is the
  epilogue block alone, behind a uniform branch. Per-lane masking of the second half of the
  pair is worse: ~12 VGPRs (as measured for MASK_TAIL_PREFETCH), wasted MFMAs, broken schedule.
- **Cache warming the first k-step of every stream-K program.** 64 KiB per program, 16 MiB in
  total. It saves about one k-step of cold latency; the epilogue hand-off above already covers
  it.
- **Stream-K first, then persistent tiles.** Correct (all waits are inside the stream-K phase),
  and it makes the stream-K prefetch the kernel prologue. Drift into the persistent tiles comes
  from asymmetric stream-K work: large for one-tile (the owner reads 15 partials, each peer does
  one store), small for two-tile (every program does about one store and one read). L2 holds
  about 2 k-steps of an XCD's traffic (32 x 64 KiB = 2 MiB of 4 MiB), so drift beyond that
  moves shared A/B hits to the LLC.
- **Separate stream-K workgroups after the persistent ones.** They dispatch as persistent WGs
  exit, roughly in pid order, so consecutive pids in time can take consecutive tiles in space.
  It gives a fresh prologue and a free stream-K grid size. It loses the epilogue-to-stream-K
  overlap and relies on observed (not guaranteed) dispatch order; waiting on lower pids (the
  reversed variant) or a last-arriver reduction avoids depending on it.
- **Split-K + stream-K.** Any K split needs a reduction; the reference already does it in
  kernel with no atomics on C, and a tile-aligned split is what 4352x4096 does today (16-way).
  The variant that removes spinning is last-arriver: each contributor writes its partial and
  bumps a per-tile counter (release); the one that sees `splits - 1` reduces in pid order and
  writes C. It is deadlock-free under any dispatch order; it needs a counter reset or an epoch,
  and `P` indexed by seam (a program can hold two partials).
- **Memory ordering** (doc section 11). Issue order is not enough: stores are routed by address
  into independent L2 channels and memory slices, so the 1-line flag can overtake the 2048-line
  partial. Each wave must wait for its own stores (`vmcnt`) before the barrier and the flag
  store. A GPU-scope release is a full `vmcnt(0)`, which also drains the next segment's
  prefetch at a tile switch; Triton atomics default to `acq_rel` / `gpu`. Start with release /
  acquire atomics; only hand-roll (stores before prefetch, partial `vmcnt(N)`) if the drain
  shows up, and stress-test it.

## 3. Measure the partial and tile-switch costs (2026-09-30)

Not a v14 change. Details are in `experiments/streamk_costs/README.md`.

**Method.** `experiments/streamk_costs` copies v14 and adds knobs:
- tiles run as K segments;
- segment endings can store fp32 partials, run a fixup (contributors publish, owners read), or
  switch tiles.

The shape is 4096x4096x8192, one tile per program. Each cost is the slope over the number of
partials or switches. Timing is back-to-back launches behind a GPU sleep, because `do_bench`'s
flush memset and host launch overhead put a 105-150 us floor under a 170 us kernel.

**Results**, in k-steps (1.27 us in these builds, 1.21 for v14 with amdgcnas), with
lane-contiguous partials:
- **Store one partial:** 1.5 with 16 storers, 4-11 for the first partial with 240-256 storers.
- **Owner reads a partial in the fixup:** 1.8.
- **Stream-K tile switch** (fp32 partial, drain, release), 16 switchers: 3.4. Of that, 1.2 is
  the pipeline restart and 0.5 is the drain.
- **Whole 16-way fixup:** 36, as the model predicts (6 + 15 x 2).

**Layout.** Row-major partials (addressed from the MFMA layout, 16 half lines per store
instruction) cost 2-3x more: 4.1 per store, 4.9 per read, 6.0 per switch. The port should write
partials lane-contiguous, `[16 vectors][4 warps][64 lanes][4 fp32]`, which is a bit permutation
of the accumulator's layout.

**No effect:**
- `.wt` / `.cv` against default caching.
- Flag polling, as long as it is relaxed with one acquire. An acquire in the spin loop doubled
  the fixup intercept.

**Model.** `streamk_diagrams.py` gains a "measured" column: a burst store of 6 for the one-tile
policies (all contributors publish at once), 1.5 for two-tile, and reads of 2. With it,
two-tile (139.5 / 148.5) edges out tile-aligned reduce-scatter (144 / 155) on 4352x4096 /
4352x4352.

**Registers.** A run-time loop or branch that carries the AGPR accumulator spills heavily, so:
- partial stores are unrolled and guarded;
- peers are summed in VGPRs and added into the accumulator once;
- the owner's C store is late, and there is only one C-store path per build.

The experiment builds still spill 9-26 VGPRs, none of it in the K loop or the measured loops.

**Tooling.**
- The amdgcnas peephole put a comma before buffer ops' cache-policy modifiers, so any `.cv` /
  `.wt` buffer op failed to assemble. It is fixed in `amdgcnas_ext.py`.
- The peephole also faults on spilling builds; unresolved, so the experiment runs without it.
