# v25_streamk_ragged_mn work log

One entry per change: what changed and what it did to performance and spills. Results come from
Triton v2.2 + deepbind shim on `PYTHONPATH` and `LLVM_PASS_PLUGIN_PATH=plugins/llir_scheduler/libLlirSched.so`,
without the amdgcnas hook (as v24), run from `a16w16/`, on GPU 0. Timings are `regress.py`'s
(`triton.testing.do_bench` median, warm inputs, a fresh Triton cache per case). Raw tables are in `results/`.

## 1. M and N need not be multiples of 256 (2026-10-10)

v24 only checks K (`static_assert`, a multiple of 128). M and N were assumed to be multiples of the
256x256 tile, unchecked: the tile grid uses `cdiv`, but loads, C stores and bias loads were all unmasked,
and the buffer descriptors' record count is `INT_MAX - 1`, so the hardware does not zero out-of-range
lanes. A ragged N wrapped C stores into the next row; a ragged M wrote past the end of C.

**The change** (two constexprs, `EDGE_M` / `EDGE_N`; with both off the kernel is v24's):
- **Loads: clamp, not modulus.** Every tile's A rows past M (B columns past N) are clamped to its last
  in-bounds one, `min(r, M - 1 - m0)` (`clamped_tile_offsets`). Clamped lanes read valid memory and only
  feed C elements that the store masks drop. The tile's position stays in the 64-bit scalar base, so the
  32-bit offsets still span one tile (a modulus would put absolute rows in them and overflow on large B,
  e.g. 106496x16384). The offsets become per-tile, so the bottom / right half is no longer v24's scalar
  shift of the top / left offsets; instead the odd K-step becomes a scalar shift (`operand_offsets`), so a
  tile still holds four offset tensors. The persistent loop builds the next tile's offsets where it issues
  its prefetch and carries them to the next trip; a stream-K segment builds its own.
- **Stores: mask.** The persistent epilogue's four quadrant stores, the owner's `store_c`, both
  reduce-scatter slice stores (band and slice), the atomic ending's adds and its zeroing, and every bias
  load are masked to M x N. Partials stay whole 256x256 tiles, so no other part of the stream-K endings
  changes, and with masks no two tiles share a C element, so the atomic ending stays valid.
- **Host.** Both flags are set whenever either dimension is ragged (below). `STREAMK_FORCE_EDGE=1` sets
  them for any shape, to measure their cost where they change nothing. Ragged shapes skip the v9 fallback,
  which has no masks (until entry 2).

**One flag alone spilled.** With only `EDGE_M` on (8100x8192x8192, 128x8192x8192) the build reached 512
VGPRs with 5-15 spills; both on, the same shapes build at 492-496 with none. Clamping both operands'
loads whenever either flag was on made it worse (13 spills M-only, 5 N-only), so it was reverted; the
host now sets both together. The mask on the aligned dimension is always true and costs nothing measurable.

**Checks** (`results/`):
- `asm_check.py` ([asm_check.md](results/asm_check.md)): all 34 configurations of v24's asm check (v23's and
  v24's options) on 4096x8192xK. Flags off: instruction streams identical to v24's, same VGPRs and spills.
  Flags forced on: 492-504 VGPRs (v24: 440-484), no spills, except the two end-to-end builds (two-tile and
  spread), which already spill 15 in v24 and go to 25 and 21.
- `check_streamk.py` ([check_streamk.md](results/check_streamk.md)): v24's stress check with C as a window
  of a sentinel-filled buffer 256 rows and columns larger (a launch fails if it writes outside M x N) and
  per-tile mean errors over the in-bounds elements. 1000x2000x8192 (32 tiles, all stream-K, so the edge
  tiles go through every ending) and 4300x4000x8192 (S = 16 after a full wave), every valid ending, partial
  type, reader mode, placement and exchange, one-tile and two-tile, with and without bias, 10 launches
  each: 224 configurations, no wrong results, no stray writes, every flag and count re-armed.
- `regress.py`: every shape below is correct against torch.

**Timing** ([timing.md](results/timing.md), [timing_ragged_final.md](results/timing_ragged_final.md),
[timing_8400_attribution.md](results/timing_8400_attribution.md); two rounds each, us). On identical code
(v24 against v25 with the flags off) rounds differ by up to 0.7% (one 1.9% outlier in
[baseline_noise.md](results/baseline_noise.md)); 256x8192x8192 (about 80 us) varies by 10%.

| aligned shape | v24 | v25, flags off | v25, flags forced on |
|---|---|---|---|
| 8192x8192x8192 (4 waves, data-parallel) | 680.2 / 669.9 | 677.0 / 667.7 | 670.2 / 670.8 |
| 4352x4096x8192 (S = 16) | 204.4 / 204.7 | 205.9 / 205.3 | 206.3 / 203.9 |
| 8192x8448x8192 (4 waves + 32) | 713.7 / 712.8 | 712.1 / 714.1 | 713.8 / 715.1 |
| 4096x6144x8192 (S = 128) | 270.3 / 272.9 | 271.3 / 274.5 | 273.0 / 269.9 |

| ragged shape (same tile grid as) | v25 | aligned twin, v25 |
|---|---|---|
| 8100x8192x8192, M (8192x8192x8192) | 669.6 / 670.6 | 677.0 / 667.7 |
| 8192x8128x8192, N (8192x8192x8192) | 675.2 / 676.5 | 677.0 / 667.7 |
| 4300x4000x8192, both (4352x4096x8192) | 204.6 / 204.1 | 205.9 / 205.3 |
| 8150x8400x8192, both (8192x8448x8192) | 724.3 / 729.4 | 712.1 / 714.1 |
| 4000x6080x8192, both (4096x6144x8192) | 266.8 / 268.1 | 271.3 / 274.5 |
| 128x8192x8192, M < 256 (256x8192x8192, v9) | 76.6 / 76.7 | 79.9-86.0 |

- **The masks and clamps cost nothing measurable.** Forced on, every aligned shape ties with flags off;
  every ragged shape ties with its aligned twin or is slightly faster. Under `do_bench` 8150x8400 looked
  1.5% slower than 8192x8448 (and 8192x8400 likewise); cold rocprof (below) shows a tie, so that was the
  harness (contiguous C, `do_bench`'s cache flush), not the shape or the masks.

**Kernel-only check** ([rocprof_cold.md](results/rocprof_cold.md), `rocprof_cold.py`: `rocprofv3
--kernel-trace` over `bench.py --rocprof`, 1000 launches over a rotating 512 MB, padded B and C rows, median
dispatch, us, one round). `regress.py`'s `do_bench` zero-fills a 256 MB buffer before every launch and
times each launch with events, which can favour one kernel over another (entry 2), so the comparisons
were repeated this way:

| shape | v24 | v25, flags off | v25, flags forced on | ragged twin, v25 |
|---|---|---|---|---|
| 8192x8192x8192 | 686.0 | 685.5 | 680.4 | 8100x8192: 681.0, 8192x8128: 680.8 |
| 4352x4096x8192 | 206.4 | 206.4 | 207.0 | 4300x4000: 205.9 |
| 8192x8448x8192 | 726.8 | | 724.6 | 8150x8400: 726.1, 8150x8448: 724.5, 8192x8400: 725.5 |
| 4096x6144x8192 | 275.3 | | 274.6 | 4000x6080: 273.8 |

Every edge case is within 1% of its aligned twin.
- **N should still be a multiple of 16** (K already is): Triton marks an integer argument 16-divisible or
  not at all, and N is C's row stride. 8192x8100x8192 is correct but takes 741.7 us against 674.6 us for
  8192x8128x8192, with 12 VGPR and 336 SGPR spills. M is only ever a count and can be anything.

## 2. The one-wave fallback goes to v9_ragged (2026-10-10)

When the host's rule picks data-parallel and the tiles fit in one wave, v24 and v25 hand the call to v9.
Entry 1 kept ragged shapes on the persistent path instead, because v9 has no edge masks. The fallback now
goes to [v9_ragged](../v9_ragged/WORKLOG.md), v9 with the same edge handling, for every shape: with its
flags off it compiles to v9's instructions, so aligned shapes run exactly what v24 runs.

- 4000x4000x8192 (256 tiles, one wave) is correct and runs v9_ragged: 175.0 us cold kernel time, against
  176.1 us for v9 on 4096x4096x8192 ([rocprof_cold.md](results/rocprof_cold.md)). 1000x2000x8192 (32 tiles)
  stays on stream-K.
- **The fallback and the persistent loop tie; `do_bench` said otherwise.** Under `regress.py` the
  persistent loop (`STREAMK_POLICY=dp`) looked 4-7% faster than the v9 fallback (4096x4096x8192: 174.6 /
  176.4 us against 182.6 / 182.3 us, v24). Kernel-only rocprof (v20's `rocprof_check.py`, 1000 launches)
  shows a tie, cold and warm:

  | shape | v9 | v24 (falls back to v9) | v24, `STREAMK_POLICY=dp` | v13 |
  |---|---|---|---|---|
  | 4096x4096x8192, cold | 176.2 | 176.2 | 176.6 | 176.8 |
  | 3840x4096x8192, cold | 172.8 | 171.8 | 172.7 | 174.0 |
  | 4096x4096x8192, warm, contiguous | 170.1 | 172.7 | 170.5 | 171.4 |
  | 3840x4096x8192, warm, contiguous | 165.4 | 165.5 | 166.7 | 166.4 |

  A rocprof trace of `do_bench` itself (4096x4096x8192) shows why: each launch follows the zero-fill of
  its 256 MB L2-flush buffer, and under that v9's kernel runs 189.6 us (median) against 180.8 us for v24's
  persistent kernel, with the same 5 us gap after the fill. The difference is in how the two kernels
  run right after that fill, not in normal operation, so the fallback rule stands. `regress.py` is fine
  for comparing builds of one kernel; across kernels use rocprof.

## 3. Sweep G and the AITER re-score: host rules below one wave (2026-10-10)

No kernel change. v25 replaces v24 in the split-count experiments: on aligned shapes its edge flags are off and its
code is v24's, and re-timing 20 stored sweep F and AITER cases (v24 and v25 alternately, cold rocprof) gave v25 / v24
a median of 0.999, range 0.996-1.012 ([spot_v25.md](../../../../../experiments/streamk_split_count/results/spot_v25.md)),
including one AITER shape that falls back to data-parallel (v9 under v24, v9_ragged under v25). The tools register
v25 and v9_ragged (`rocprof_check.py`: `DELEGATES[25] = ("9r",)`), and `STREAMK_FORCE_EDGE` / `V9_FORCE_EDGE` are
in every `ENV_KEYS`, so no case inherits them.

- **Sweep G** ([experiments/streamk_split_count](../../../../../experiments/streamk_split_count/README.md), section 6):
  4096x8192xK with `GEMM_TILE_LIMIT = S` (no full wave), 2,003 cases, all checked first. Below one wave reduce-scatter
  beats owner up to S = 48-56 and its best n falls as about 32 / sqrt(S); `capped` switches to owner at S = 33 (4-24%
  slower at S = 36-48) and runs every K = 1024 shape data-parallel (up to 40% slower at S = 1). Under
  `GEMM_TILE_LIMIT` a data-parallel choice stays on v25's persistent loop; v9_ragged and v13 tie there (median 1.000).
- **Version 2 of the frozen rules**, fitted on sweep G below one wave (two variants: version 1's family, and the
  same with n <= Q / sqrt(S)), version 1's constants with one wave or more.
- **AITER re-score** ([streamk_aiter_test](../../../../../experiments/streamk_aiter_test/README.md)): below one
  wave, 314 shapes, loss against the per-shape best (median / 90th percentile / worst) `capped` 1.035 / 1.165 / 1.277,
  version 2 1.013 / 1.102 / 1.221; at S = 32-63 the median goes from 1.123 to 1.004. The sqrt(S) variant came second
  (1.117 at the 90th percentile). Shapes with a full wave are unchanged.
- **Proposed, not applied:** `capped_policy` below one wave (tiles < programs) with version 2's constants: data-parallel
  if (256 - S) k / 256 < 14, reduce-scatter while 256 // S >= 4 with n = min(192 // S, k / 4), else owner with
  n = round(sqrt(k / 1.5)). S > 128 below one wave keeps losing up to 18% with any threshold on k alone.

## 4. Tile-dependent C row offsets under EDGE_M (2026-10-10)

From [experiments/ragged_spills](../../../../../experiments/ragged_spills/README.md) (`crow`). The persistent epilogue's
C store offsets are `stride_cm * row + stride_cn * col`; the column keeps them tile-dependent, but the row half of the
product is loop-invariant, so LLVM hoists it above the tile loop and holds it across the loop, and once the ragged
builds' other values fill the 256 architectural VGPRs it is spilled (v26's M, N and K builds reloaded it in the tile
loop). Under `EDGE_M` the rows are now `min(row, rows_left - 1)`: only rows the store masks drop change (in the top
and the bottom quadrants), and the offsets become tile-dependent, so they are computed in the epilogue. With the edge
flags off the code is unchanged.

- `asm_check.py` ([asm_check.md](results/asm_check.md)): flags off, all 34 builds identical to v24's; forced on,
  484-500 VGPRs (488-504 before), no spills, except the two end-to-end builds (25 and 21, as before).
- Real shapes ([survey_v25_final.md](../../../../../experiments/ragged_spills/results/survey_v25_final.md)): M / N-ragged
  builds 8 VGPRs smaller (4300x4000x8192 496 -> 488, owner 500 -> 492), no spills. Two kinds of builds still spill,
  as before: M / N ragged with a bias (4 spills, outside the tile loop) and N below 256 with many M tiles
  (131072x200x8192: 12 before, 11 now).
- `check_streamk.py` ([check_streamk.md](results/check_streamk.md)): 1000x2000x8192 and 4300x4000x8192, every ending,
  partial type, reader mode, placement and exchange, with and without bias: 224 configurations, no wrong results, no
  stray writes, every flag and count re-armed.
- Cold kernel time ([timing_v25_v26_final.md](../../../../../experiments/ragged_spills/results/timing_v25_v26_final.md),
  [timing_bias_final.md](../../../../../experiments/ragged_spills/results/timing_bias_final.md), row stride 8256, two
  rounds, before / after): 4300x4000x8192 214.8 / 213.8 and 214.6 / 215.2 us; 8150x8400x8192 741.0 / 744.0 and
  741.6 / 742.1; with bias 215.8 / 216.6 and 216.4 / 217.6. Ties.
