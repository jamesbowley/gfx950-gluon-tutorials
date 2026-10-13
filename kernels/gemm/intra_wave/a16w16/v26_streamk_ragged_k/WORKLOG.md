# v26_streamk_ragged_k work log

v25 (v24 for ragged M and N) for K that is not a multiple of 128. Results come from Triton v2.2 + deepbind
shim and the LLIR scheduler plugin, without the amdgcnas hook (as v24 and v25), run from `a16w16/`, on
GPU 1. Kernel times are cold rocprof kernel-only medians unless noted; raw tables are in `results/`.

## 1. Ragged K: the last pair first (2026-10-10)

v25 requires K to be a multiple of 128 (`static_assert`): the peeled pair, the 2x-unrolled loop and the
2-step epilogue consume K-steps of 64 in pairs. Its one-wave fallback (v9_ragged) did not check K at all
before this change, and v9 and v9_tile accepted K with an odd number of K-steps and computed a wrong C
(4096x4096x8128 through `v9_tile` and v9: `regress.py` reports wrong results).

**The change** (constexpr `EDGE_K`, set by the host when K % 128 != 0; with it off the kernel is v25's):
- K is rounded up to K_PAD, a whole number of pairs, and **every tile's last, partial pair goes first**, as
  its pair 0, with the lanes at or past K masked to zero (`k_tail_mask`, `tail_pair_masks`: K-step 1 of
  that pair always reaches past K, K-step 0 only when K_PAD - K > 64). The masked loads are in the three
  places a tile's first pair is loaded: the kernel prologue, the persistent epilogue's prefetch of the next
  tile, and the stream-K segment that starts at k0 = 0 (its mask bound is a scalar select on k0, so other
  segments load unmasked). The main loops are unchanged.
- Every other pair p is at columns 128 (p - 1), so main-loop reads start at whole pairs from the row's start.
  A stream-K segment's pair i >= 1 is at `a_tile + 128 i` with `a_tile` one pair before its first; for the
  segment at k0 = 0 the clamped prefetch of its last pair is held at pair 1 so it never reads before the row.
- Host: K must be a multiple of 16 (Triton only marks a stride 16-aligned if it is a multiple of 16; the
  async copies need 16-byte rows) and at least 256 after rounding up; the policy rules and the split count
  pairs of K_PAD.

**Residue first with a shifted loop was 55% slower.** The first version (in v9_ragged) shifted the whole loop
to start K_PAD - K before the matrix and masked the first pair. With rows on 128-byte lines that misaligns
every main-loop read by 2 (K_PAD - K) bytes: on 4096x4096xK with a fixed row stride of 8256, K = 8128
(shift 128 bytes) tied with K = 8192, while K = 8176 and 8080 took 287.9 and 283.8 us against 183.5
([stride_timing.md](results/stride_timing.md)). Putting the partial pair last in K order but first in the
schedule keeps the masks out of the loop without moving it.

**Row strides.** With contiguous inputs a K that is not a multiple of 64 also makes A's and B's rows start off
the 128-byte lines, which costs about 20% whatever the kernel: unchanged v9 at K = 8192 with a row stride of
8208 takes 220.7 us against 184.7 us at 8256, the same as v9_ragged at K = 8176 with stride 8176 (221.4 us).
So the timings below hold the row stride fixed (8256, 129 lines) across K, and compare the kernel alone
(`stride_rp.py` / `stride_bench.py`: rocprofv3 over 300 launches on rotating copies > 512 MB, median). Callers
with such K should pad the row stride to a multiple of 64 elements.

**Checks** (`results/`):
- `asm_check.py` ([asm_check.md](results/asm_check.md)): all 34 configurations of v24's asm check on
  4096x8192xK. Aligned K: instruction streams identical to v25's, same VGPRs and spills. K - 16 and K - 112
  (EDGE_K): 448-488 VGPRs, no spills, except the two end-to-end builds (two-tile, spread), which spill 15 in
  v25 and 16 here. K - 112 with M and N edges forced on too: 496-504, no spills (end-to-end 25 and 23).
- `regress.py`, against torch: 4352x4096xK for K = 8176, 8128, 8080 (K_PAD - K = 16, 64, 112), 4300x4000x8080,
  8192x8448x8080, 4096x4096x8080 (one wave: v9_ragged), 1000x2000x4032, 4352x4096x400 (K_PAD = 512),
  128x8192x8176. With all three edges, some real shapes (4300x4000x8080, 1000x2000x4032, 128x8192x8176)
  build at 512 VGPRs with 9-11 spills; the timings below include 4300x4000 and show no cost.
- `check_streamk.py` ([check_streamk.md](results/check_streamk.md)): v25's stress check (C in a sentinel
  guard buffer, per-tile mean errors over in-bounds elements), with pair counts rounded up, on
  1000x2000x8080 (all 32 tiles stream-K, K_PAD - K = 112, so the k0 = 0 segment masks both K-steps) and
  4300x4000x8176 (S = 16 after a full wave), every valid ending, partial type, reader mode, placement and
  exchange, one-tile and two-tile, with and without bias, 10 launches each: 224 configurations, no wrong
  results, no stray writes, every flag and count re-armed.

**Timing** ([stride_timing.md](results/stride_timing.md), row stride 8256 for every K, us):

| shape | v25, K = 8192 | v26, K = 8192 | v26, K = 8176 | v26, K = 8128 | v26, K = 8080 |
|---|---|---|---|---|---|
| 4352x4096 (S = 16) | 223.7 | 224.4 | 220.6 | 220.4 | 220.0 |
| 8192x8448 (4 waves + 32) | 735.2 | | 734.3 | | 735.2 |
| 8192x8192 (4 waves, data-parallel) | 692.1 | | 690.3 | | 687.0 |
| 4096x6144 (S = 128) | 277.4 | | | | 276.8 |
| 4300x4000, M, N and K ragged (v25 4352x4096x8192 with M / N edges forced) | 223.6 | | 216.7 | | 214.6 |

Ragged K costs nothing measurable: every ragged K ties with, or is slightly faster than, K = 8192 on the same
rows (it loads up to 112 fewer columns).

## 2. Fewer spills on ragged shapes (2026-10-10)

From [experiments/ragged_spills](../../../../../experiments/ragged_spills/README.md), two changes, each compiled only
with an edge flag on (aligned builds are unchanged):
- **Tile-dependent C row offsets under `EDGE_M`** (as v25 entry 4): the persistent epilogue's `stride_cm * row` is no
  longer hoisted above the tile loop and held (spilled) across it.
- **`HAS_FULL`**: with an edge flag on and no full tiles (fewer tiles than one wave, all stream-K), the persistent loop
  and its first-tile prologue are not compiled; they never run, and their registers made the stream-K phase spill.
  With full tiles, or on aligned shapes, the code is entry 1's.

A third candidate, making the persistent epilogue's K-tail masks depend on the next tile, was dropped: it saves
registers on some builds but adds 6-14 spills to the `GEMM_TILE_LIMIT` builds of the asm check.

- `asm_check.py` ([asm_check.md](results/asm_check.md)): aligned K, all 34 builds identical to v25's. K - 16 and
  K - 112: 448-488 VGPRs, no spills (end-to-end builds 16, as before). K - 112 with M and N forced on too: 488-500
  (496-504 before), no spills; end-to-end 16 and 23 (25 and 23 before). Only-K-ragged real shapes compile to the
  same instructions as before (4352x4096x8080, 4352x4096x8176, 8192x8192x8080).
- Real shapes ([survey_v26_final.md](../../../../../experiments/ragged_spills/results/survey_v26_final.md), VGPR
  spills before -> after, scratch ops in the tile loop in brackets):

  | build | before | after |
  |---|---|---|
  | M, N and K ragged (4300x4000x8080, 8150x8400x8080, 4000x6080x8080, owner ending) | 9-12 (3-6) | 5 (0) |
  | ... 16 columns left over (4300x4000x8176) | 11 (6) | 8 (0) |
  | ... data-parallel only | 2 (2) | 0 |
  | ... l2 exchange | 10 (3) | 9 (2) |
  | ... with bias (4300x4000x8080 / 8176) | 18 / 23 (12 / 11) | 22 / 28 (12 / 14) |
  | fewer tiles than one wave (128x8192x8176, 1000x2000x4032, 4096x128x8080) | 9-11 (3-6) | **0** (412-424 VGPRs) |
  | a dimension below 256, many tiles (128x98304x8176) | 11 (5) | 9 (2) |
  | N below 256, many M tiles (131072x200x8176) | 44 (22) | 47 (24) |
  | only K ragged, aligned | 0 | 0 |

- `check_streamk.py` ([check_streamk.md](results/check_streamk.md)): 1000x2000x8080 (fewer tiles than one wave: no
  persistent loop) and 4300x4000x8176, every ending, partial type, reader mode, placement and exchange, one-tile
  and two-tile, with and without bias: 224 configurations, no wrong results, no stray writes, every flag and count
  re-armed.
- Cold kernel time ([timing_v25_v26_final.md](../../../../../experiments/ragged_spills/results/timing_v25_v26_final.md),
  [timing_bias_final.md](../../../../../experiments/ragged_spills/results/timing_bias_final.md), row stride 8256, two
  rounds, before / after, us):

  | shape | round 1 | round 2 |
  |---|---|---|
  | 4300x4000x8080 | 215.0 / 214.3 | 215.0 / 215.0 |
  | 4000x6080x8080 | 278.0 / 274.1 | 276.0 / 274.3 |
  | 4300x4000x8176 | 215.9 / 215.9 | 216.7 / 216.9 |
  | 1000x2000x4032 (row stride 4160) | 36.8 / 35.2 | 37.0 / 35.4 |
  | 4096x128x8080 | 39.7 / 37.5 | 40.1 / 37.5 |
  | 128x8192x8176 | 64.5 / 63.0 | 64.3 / 63.1 |
  | 4300x4000x8080, bias | 218.4 / 218.4 | 216.8 / 217.5 |
  | 4300x4000x8176, bias | 218.7 / 218.2 | 218.6 / 219.5 |

  No regression; 1% faster at 4000x6080x8080 and 2-6% below one wave. The spills that remain (and the bias builds'
  extra ones) do not show in time; the experiment's README has the shapes where they are largest.

## 3. Fewer spills in the end-to-end segment loop (2026-10-12)

From [experiments/endtoend_spills](../../../../../experiments/endtoend_spills/README.md), on GPU 3. Two-tile and
spread (`END_TO_END`) have spilled since v19: 14 VGPRs on aligned shapes, 15 in the asm check. All are thread-id
values computed once before the segment loop and held across every segment's K loop: the C store's row offsets
(8) and column offset (1), and 5 LDS addresses of `store_c`'s layout conversion. No K loop has a scratch op; the
segment ending reloads 29 values, around the C stores.

**The change:** `streamk_segment` takes a constexpr `SEG_LOOP`, set only by the end-to-end call. With it the C row
offsets are clamped to the tile's last row, as `persistent_tile` does under `EDGE_M` (entry 2): only rows the
stores mask off change, and `stride_cm * row` becomes tile-dependent, so it is not hoisted above the segment loop.
The K loops compile to the same instructions and the ending keeps the same loads and stores; the spills go from
14 to 5 and the ending's reloads from 29 to 9 (with a bias 22 -> 31 spills, but 43 -> 34 reloads).

- `asm_check.py` ([asm_check.md](results/asm_check.md)): every build but the two end-to-end ones is still v25's.
  End-to-end, VGPR spills: two-tile 15 -> 6 (K - 16, K - 112: 16 -> 9, 9; all edges forced: 16 -> 14), spread
  15 -> 6 (16 -> 7, 5; 23 -> 22).
- `check_streamk.py --policies two_tile spread --orders v9 split`
  ([check_streamk_end_to_end.md](../../../../../experiments/endtoend_spills/results/check_streamk_end_to_end.md)):
  4352x4096x8192, 4352x4352x8192, 3328x5120x8192, 4096x6144x8192, 4300x4000x8176, 1000x2000x8080, 4096x7168x8192,
  fp32 and bf16-pair partials, with and without bias: 112 configurations, no wrong results, no stray writes, every
  flag re-armed.
- Cold kernel time (median of 3 rounds, us, before / after; row stride K, 8256 for the ragged shapes):

  | shape | policy | before | after |
  |---|---|---|---|
  | 4352x4096x8192 | two-tile | 220.1 | 215.2 |
  | 4352x4096x4096 | two-tile | 125.3 | 121.0 |
  | 8192x8448x8192 | two-tile | 788.8 | 784.3 |
  | 4096x7168x8192 | spread | 346.6 | 341.9 |
  | 4096x7168x32768 | spread | 1289.1 | 1286.7 |
  | 4096x7936x32768 | spread | 1427.7 | 1422.1 |
  | 4352x4096x8192, bias | two-tile | 225.8 | 222.8 |
  | 4096x7168x8192, bias | spread | 350.8 | 346.7 |
  | 4300x4000x8176 | two-tile | 225.4 | 220.1 |
  | 4300x4000x8176 | spread | 265.7 | 261.3 |
  | 1000x2000x8080 | spread | 71.5 | 67.9 |

  4-6 us faster on every end-to-end shape timed (0.2-5%). The removed reloads run right after a K loop, which has
  likely evicted their spill slots, and they come in dependent batches, so each waits for memory.

With this, two-tile beats the default policy (`capped`, which never picks it) on all nine of v19's sweep shapes
timed, S = 2 to 64 with a full wave to borrow, by 0.5-7.5% (e.g. 4352x4352x8192: 228.4 against 246.9 us; the
README's table). Making it the default there needs a policy sweep first. The other 5 spills (4 partial slot
offsets, 1 column offset) did not go without breaking the 16-byte accesses or moving spills elsewhere.

## 4. Entry 3 on the split-count grid: spread for S > 128, two-tile (2026-10-12)

No kernel change. Entry 3's end-to-end builds, timed on sweep C's matrix
([experiments/streamk_split_count](../../../../../experiments/streamk_split_count/README.md), section 7), after the
tile-limit check passed for every case (v26 is registered in `rocprof_check.py`, delegating to v9_ragged, and in
`check_tile_limit.py`):
- **Spread** (sweep H, S = 132-255, K = 2048-32768, two rounds): 1.1% faster than v25's (median of 40 cells, same
  session; 0.931-1.004). Spread beats data-parallel when (256 - S) k >= about 10,500, in both regimes; on AITER's 91
  shapes with S > 128 that test (frozen first) loses 0.9% / 0.5% at the 90th percentile below one wave / with one,
  against 12.9% / 1.6% for `capped`'s k >= 512
  ([streamk_aiter_test](../../../../../experiments/streamk_aiter_test/README.md)).
- **Two-tile** (sweep F's 105 cells): 3.2% faster than v24's (median; 0.855-0.998), but slower than the best one-tile
  or data-parallel choice in 104 of them (median 6.2%). It beats `capped`'s choice by more than 2% only in 9 of 90 cells
  (S <= 6, mostly K = 1024), where a better one-tile n beats it too, so a policy sweep for two-tile is not worth it on
  this matrix.
- **Proposed, not applied:** `capped_policy` for S > programs / 2: spread if (programs - S) k >= 10,493.

## 5. The host rule from sweeps G and H (2026-10-12)

`capped_policy` (the default) now applies rules_frozen.json version 3
([experiments/streamk_split_count](../../../../../experiments/streamk_split_count/README.md), sections 6-7):
- **S > programs / 2**, both regimes: spread if (programs - S) k >= 10493 (was k >= 512). `split_rule` hands these S
  to `capped_policy`, so it follows.
- **Fewer tiles than programs, S <= 128**: data-parallel if (programs - S) k < 14 programs; reduce-scatter while
  programs // S >= 4 with n = min(192 // S, round(k / 4)); otherwise owner with n = round(sqrt(k / 1.5)); both at
  most programs // S and k // 2, a split at that cap on all programs (as the rule was scored).
- **One wave or more, S <= 128**: unchanged (it did better on AITER than the fitted constants).

Host code only; no kernel build changes. **Checks:** the function's choices match the frozen rule (and the old one
where unchanged) on all 440 AITER shapes and on every (tiles, K) for tiles 1-511, K = 256-39,936 in steps of 128
(158,610 points). v26 as deployed is correct on all 440 AITER shapes against torch
([check_v26_deployed.md](../../../../../experiments/streamk_aiter_test/results/check_v26_deployed.md)) and on
`regress.py` ragged shapes, with and without bias (1000x2000x8192, 2000x2000x4000, 128x8192x8192, 3000x3000x16384,
1000x9000x8080, 3800x4000x16384, 4300x4000x8176, 4096x7168x16384).

**AITER, timed as deployed** (`v26 deployed` in [test.md](../../../../../experiments/streamk_aiter_test/results/test.md);
loss against the per-shape best, median / 90th percentile / worst): less than one wave 1.012 / 1.105 / 1.231 (v24's
`capped` 1.043 / 1.165 / 1.277), of which S > 128 1.002 / 1.014 / 1.061 (1.005 / 1.131 / 1.212); one full wave or more
1.003 / 1.015 / 1.039 (1.003 / 1.016 / 1.058).

**Ragged M / N: not covered by the fit.** Sweeps G and H and the AITER shapes are all aligned. On ragged shapes the
new spread test sends more of them to spread, whose edge build spills (35 VGPRs at 3000x3000x16384, 127 with bias;
aligned 4096x7168x16384: 5, and 31 with bias). Cold rocprof, spread / data-parallel (`STREAMK_POLICY=dp`), without / with bias:
3000x3000x16384 1.038 / 1.166 (aligned twin 3072x3072: 0.842 / 0.855); 1000x9000x8080 0.853 / 0.989 (1024x9216x8192:
0.928 / 0.940); 3000x3000x32768, which the old rule already sent to spread, 1.028 / 1.102. Spread on ragged shapes,
especially with bias, needs its own check before the rule is trusted there.

**Ragged shapes may need their own rule (open).** The spread test was fitted and tested on aligned shapes only. A
first sweep ([ragged_spread.md](../../../../../experiments/streamk_split_count/results/ragged_spread.md),
`ragged_spread.py`: S = 136-248 below one wave and with one, K = 4096-32768, aligned / M and N ragged / K ragged /
all three, with and without bias, spread against data-parallel as deployed, 640 cases, all correct first) says the
answer differs by what is ragged, and is not simply "the same rule with the spilling build disabled":
- **Aligned** (new real shapes, not the fit's matrix): the rule holds, 90th-percentile regret 1.000-1.010, worst
  1.019.
- **M / N ragged, no bias**: the same rule is fine (1.014 at the 90th percentile; worst 1.083). Disabling spread is
  worse at the 90th percentile (1.044; worst 1.066), because spread still wins where the idle share is large.
- **M / N ragged, with bias** (the build that spills most): the rule's worst is 1.110. Disabling spread gives
  1.030 / 1.048 (90th percentile / worst); raising C to about 30,000 does better in-sample (1.013 / 1.037). So rather
  than switching the spilling build off, it wants a higher bar, or the spills fixed.
- **K ragged** (K - 112): the opposite. Data-parallel is 23-47% slower than on the aligned twin and spread 4-27%, so
  spread should run far more often (in-sample C about 2,000; the rule as is loses 13-15% at the 90th percentile,
  32-35% at worst, and disabling spread 32-35% / 54-55%). Part of this is the layout, not the kernel: bench.py pads B's and C's
  rows but not A's, so with K not a multiple of 64 every A row is misaligned (v9_ragged entry 2), which hurts
  data-parallel more. It needs re-timing with A's rows padded before a K-ragged rule is chosen.
- **All three ragged**: in between (in-sample C about 4,000-6,000).
The C values above are fitted and scored on the same 40 cells per variant, so they are only a direction: a ragged
rule needs its own sweep (with padded A for ragged K) and a held-out check, as sweeps G and H had.

**Parked (2026-10-12): a ragged host rule, and the bias spills behind it.** Not started; what it needs:
- **Bias spills come first.** The losses on ragged shapes come mostly from the edge builds that spill, and a bias
  makes them much worse. `regress.py` (VGPR / SGPR spills, scratch ops in K loop / tile or segment loop / outside):

  | build | no bias | with bias |
  |---|---|---|
  | spread, 3000x3000x16384 (M / N ragged) | 35 / 58, 0 / 86 / 19 | 127 / 223, 0 / 266 / 55 |
  | spread, 1000x9000x8080 (M, N, K ragged) | 36 / 59, 0 / 91 / 20 | 126 / 229, 0 / 261 / 54 |
  | spread, 4096x7168x16384 (aligned) | 5 / 25, 0 / 0 / 14 | 31 / 27, 0 / 40 / 7 |
  | default (one-tile), 4300x4000x8176 (M, N, K ragged) | 8 / 49, 0 / 0 / 16 | 28 / 42, 0 / 14 / 18 |

  None is in the K loop; they sit around the segment / tile ending, as in entry 3 and in
  [experiments/aiter_bias_spills](../../../../../experiments/aiter_bias_spills/results.md) (the persistent kernels).
  The cost shows in time: spread / data-parallel at 3000x3000x16384 is 1.038 without a bias and 1.166 with one
  (3000x3000x32768: 1.028 / 1.102), and in the ragged sweep the rule's worst case with M / N ragged goes from 1.083 to
  1.110 with a bias. If these spills can be removed (entry 3's approach: stop hoisting thread-id values across the
  segment loop, here with the bias and edge masks too), ragged shapes may simply follow the aligned rule, and a ragged
  sweep should wait for that.
- **A held-out ragged set has to be built.** None of AITER's 617 shapes is ragged (M, N multiples of 256, K of 128).
  Plan: AITER's (N, K) with M made ragged (e.g. M - 72, plus small odd M), about 400 shapes, frozen before any fit.
- **Ragged K needs A padded.** bench.py pads B's and C's rows (against 512-byte camping) but not A's, so K not a
  multiple of 64 misaligns every A row. Add an option to pad A's rows to 128 bytes, and time contiguous A as well
  (what callers like AITER pass); the rule may need A's row stride.
- **Sweep design (about 2,500 cases, 3-4 h per raggedness class, twice that with and without bias):** real shapes,
  edges on; S <= 128 at S = 1, 2, 4, 8, 16, 24, 32, 40, 48, 64, 96, 128 and K = 1024-16384, below one wave and with
  one, data-parallel and reduce-scatter / owner at n = 2, 3, 4, 6, 8, 12, 16, 32 and the default; S > 128 on sweep H's
  grid. M / N ragged first (one build covers both, and ragged M is the common case), ragged K once A padding is in.
  Then freeze, re-score on the held-out set, and apply, as for version 3.
