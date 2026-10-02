# v19_streamk_two_tile_reversed work log

One entry per change: what changed and what it did to performance and spills. Results come
from `HIP_VISIBLE_DEVICES=2 python regress.py ...` (bf16 unless noted), run from `a16w16/`
under the published v10+ stack: Triton v2.2 + deepbind shim on `PYTHONPATH`,
`LLVM_PASS_PLUGIN_PATH=plugins/llir_scheduler/libLlirSched.so`, `TRITON_AMDGCNAS_PLUGIN=1`.
Raw tables are in `results/`.

## 1. Reversed two-tile (2026-10-01)

v17 plus a second stream-K path, chosen on the host.

**Host policy**, with `S = total_tiles % 256` leftover tiles:
- If there is a full wave to borrow (`total_tiles >= 256`), use two-tile:
  `STREAMK_TILES = S + 256`.
- Otherwise, v17's tile-aligned one-tile path, unchanged.

The kernel selects the path with a constexpr (`STREAMK_TILES > NUM_PROGRAMS`).

**Two-tile path.**
- The stream-K tiles' pairs of k-steps are laid end to end and cut into 256 ranges, as in v15.
  Each range is one tile plus `d = iters_per_tile * S / 256` k-steps, so every tile has at most
  one peer.
- Each program processes its segments last to first (STREAMK_EXPLAINED section 8):
  1. the head of its last tile, from k-step 0. This is its only partial, and it is published
     first;
  2. any whole tile, from k-step 0, stored straight to C;
  3. the tail of its first tile, which it owns. The owner waits on `spid - 1`, which processed
     that tile's head first.
- Peers are counted down (`PEER_STEP = -1` in the poll, `fixup_quadrant` and the reset). Waits
  point only to lower spids, and every program publishes before it waits.
- Every program starts at k-step 0, so the programs of an XCD read k-steps at most `d` apart.

**Correctness.**
- `check_partition.py` (`results/partition.md`), pure Python on all 17 two-tile shapes:
  - every pair is covered once;
  - every tile has one owner, with at most one peer, which is `spid - 1`;
  - every program has at most one partial, processed first;
  - an XCD's spread of k-steps read at once is at most `d`.
- `check_mapping.py` still passes.
- `check_streamk.py` passes on the 9 stream-K shapes, the 10 other sweep shapes and the capped
  shapes (now two-tile), in both orders, with and without bias, with no amdgcnas fallbacks
  (`results/check_streamk.md`).

**Registers.** The two-tile builds bring back the segment loop, and with it v16's register state:
512 VGPRs, 14 VGPR spills, 22 SGPR spills, and no scratch ops in the K loop. Builds without a
full wave are v17's (no VGPR spills). Regression shapes: within 0.4% of v17.

### Leftover-tile sweep

`results/sweep_summary.md` (do_bench TFLOPS, v9 order). "distinct k" is the average number of
distinct k-steps an XCD's 32 programs read at the same time in the lockstep model, for forward
and reversed order (`results/partition.md`). "best" is the fastest version if it leads by more
than 1%, otherwise "tie".

| leftover tiles S | shape | full waves | lag d (k-steps) | distinct k per XCD, fwd / rev | v13 | v17 | v19 | v17 vs v13 | v19 vs v13 | best |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 1536x11008x8192 | 1 | 2 | 17 / 1.2 | 964 | 789 | 1336 | -18.2% | +38.5% | v19 |
| 4 | 3328x5120x8192 | 1 | 2 | 32 / 1.5 | 1002 | 795 | 1322 | -20.6% | +32.0% | v19 |
| 16 | 4352x4096x4096 | 1 | 4 | 16 / 1.9 | 981 | 950 | 1126 | -3.1% | +14.8% | v19 |
| 16 | 4352x4096x8192 | 1 | 8 | 16 / 1.9 | 1043 | 1224 | 1284 | +17.4% | +23.2% | v19 |
| 16 | 4352x4096x16384 | 1 | 16 | 16 / 1.9 | 900 | 1212 | 1302 | +34.7% | +44.7% | v19 |
| 32 | 8192x8448x8192 | 4 | 16 | 8 / 1.8 | 1464 | 1571 | 1581 | +7.3% | +7.9% | tie |
| 33 | 4352x4352x8192 | 1 | 18 | 14 / 2.7 | 1084 | 1306 | 1281 | +20.5% | +18.2% | v17 |
| 64 | 4096x5120x8192 | 1 | 32 | 4 / 1.6 | 1147 | 1404 | 1269 | +22.5% | +10.7% | v17 |
| 128 | 4096x6144x8192 | 1 | 64 | 2 / 1.3 | 1325 | 1511 | 1388 | +14.1% | +4.8% | v17 |
| 192 | 4096x7168x4096 | 1 | 48 | 4 / 2.3 | 1363 | 1372 | 1174 | +0.7% | -13.9% | tie |
| 192 | 4096x7168x8192 | 1 | 96 | 4 / 2.3 | 1457 | 1478 | 1266 | +1.5% | -13.1% | v17 |
| 192 | 4096x7168x16384 | 1 | 192 | 4 / 2.3 | 1497 | 1495 | 1365 | -0.2% | -8.8% | tie |
| 200 | 9472x10240x8192 | 5 | 100 | 32 / 14.6 | 1541 | 1556 | 1454 | +1.0% | -5.7% | v17 |
| 224 | 8192x7936x8192 | 3 | 112 | 8 / 4.3 | 1611 | 1599 | 1376 | -0.7% | -14.6% | tie |
| 240 | 3840x4096x8192 | 0 | - | - | 1440 | 1467 | 1464 | +1.9% | +1.7% | tie |
| 240 | 4096x7936x8192 | 1 | 120 | 16 / 8.3 | 1540 | 1556 | 1136 | +1.1% | -26.2% | v17 |
| 248 | 4608x7168x8192 | 1 | 124 | 32 / 16.3 | 1567 | 1582 | 1154 | +1.0% | -26.3% | v17 |
| 255 | 1792x18688x8192 | 1 | 128 | 9 / 4.9 | 1501 | 1545 | 1376 | +2.9% | -8.4% | v17 |

Cold-cache rocprof (`results/rocprof.md`), against v13:

| leftover tiles | v19 | v17 |
|---|---|---|
| S = 4 | +37.6% | -18.6% |
| S = 16 | +30.4% | +16.3% |
| S = 128 | +0.1% | +7.3% |
| S = 240 | -28.4% | -1.5% |

Split order and bias (`results/leftover_sweep_split_order.md`, `results/leftover_sweep_bias.md`)
show the same crossover:
- with few leftover tiles, v19 leads v17 with bias by 61-67% (S = 2 and 4) and by 2.5%
  (S = 16);
- from S = 64 up, v17 leads.

**Reversed against forward order** (`results/forward_vs_reversed.md`, a scratch build that
processes the same two-tile segments in k order):

| S | 2 | 4 | 16 | 32 | 64 | 128 | 192 | 200 | 224 | 240 | 248 | 255 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reversed vs forward | +8.5% | +10.5% | +55.0% | +17.9% | +11.9% | +2.4% | +7.6% | +4.7% | +19.5% | +21.2% | +13.6% | +8.9% |

**L2 counters** (`results/rocprof.md`), hit rate:

| S | v13 | v17 | v19 | forward |
|---|---|---|---|---|
| 4 | 74% | 69% | 68% | 29% |
| 16 | 73% | 68% | 62% | 5% |
| 64 | 77% | 73% | 63% | 26% |
| 128 | 78% | 76% | 60% | 48% |
| 240 | 79% | 77% | 28% | 5% |

What the numbers say:

1. **Reversed order is what makes two-tile work.** It beats forward order on every shape, by
   2-55%. For S up to 64 it lifts two-tile's L2 hit rate from 5-29% to 61-68%, close to v17
   and data-parallel. The lockstep model agrees: an XCD's programs read about 2 distinct
   k-steps at once instead of 16-32.
2. **With few leftover tiles (S <= 16), v19 is the best version by a wide margin.**
   - +38.5% over v13 at S = 2, +32% at S = 4, +23% at S = 16;
   - +45% at S = 16 with K = 16384, and +15% at K = 4096, where v17 loses 3%;
   - this is exactly where one-tile v17 is collection-bound (63 serial reads per owner at S = 4).
     Two-tile has at most one peer per tile.
3. **Around S = 32 the two are level** (8192x8448x8192: v19 +7.9%, v17 +7.3%; 4352x4352x8192 at
   S = 33: v17 +20.5%, v19 +18.2%).
4. **From S = 64 v17 is better, and from S = 192 v19 loses to v13 by 6-26%.**
   - A nearly full last wave has little idle time to recover; v17 and v13 are within 3%
     there.
   - Two-tile, in contrast, moves a whole extra wave into stream-K. Every program then runs 2-3
     segments, each with its own pipeline restart, a half-tile partial store and read, and the
     segment loop's spills.
   - Its lag `d` grows to 100-128 k-steps. The L2 hit rate falls to 28% at S = 240 (v17: 77%),
     with 3.4 times v17's memory reads.
   - The no-full-wave control (3840x4096x8192) uses v17's path and matches v17.

**Proposed host rule.** If there is a full wave to borrow and `S <= 32` (at most 1/8 of a
wave left over), use reversed two-tile; otherwise use v17's tile-aligned one-tile split.
- The crossover is between S = 32 and S = 64 at K = 8192.
- At S = 16 two-tile wins at K = 4096, 8192 and 16384.
- The data cannot tell whether the limit is really on `S` or on the lag `d`: at K = 8192 the two
  are the same measure. S = 32 is the conservative choice.

### Next

v20: apply the host rule. Both paths are already in v19's kernel, so it is a host-only change:
choose two-tile for `S <= 32` and the tile-aligned path otherwise. Then rerun the sweep, which
should give the best of v17 and v19 on every shape.

## 2. Is the limit S or the lag d? (2026-10-01)

Not a code change. At K = 8192, S and `d = iters_per_tile * S / 256` are the same measure, so
entry 1 could not tell which limits two-tile. These shapes separate them: S fixed while K
varies, and `d` fixed while S varies. They also add a one-wave S = 32 shape and a four-wave
S = 16 shape (`results/s_vs_d.md`, do_bench TFLOPS against v13). The other rows are from entry
1's sweep.

"v17 collection" is the owner's serial reads in v17: `n - 1` partials at about 2 k-steps each,
with `n = 256 // S` programs per tile. "v17 chunk" is each program's compute, in k-steps:
`iters_per_tile / n`. "Rule" picks v19 when the collection is more than 1.25 times the chunk,
v17 when the chunk is more than 1.25 times the collection, and otherwise calls a tie.
"Measured" is the version more than 1.5 points ahead.

| S | shape | d | n | v17 collection | v17 chunk | v17 vs v13 | v19 vs v13 | measured | rule |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 4352x4096x4096 | 4 | 16 | 30 | 4 | -3.1% | +14.8% | v19 | v19 |
| 16 | 4352x4096x8192 | 8 | 16 | 30 | 8 | +17.4% | +23.2% | v19 | v19 |
| 16 | 4352x4096x16384 | 16 | 16 | 30 | 16 | +34.7% | +44.7% | v19 | v19 |
| 16 | 4096x16640x8192 (4 waves) | 8 | 16 | 30 | 8 | +6.5% | +9.5% | v19 | v19 |
| 32 | 4096x4608x4096 | 8 | 8 | 14 | 8 | +12.6% | +16.4% | v19 | v19 |
| 32 | 4096x4608x8192 | 16 | 8 | 14 | 16 | +25.4% | +23.6% | v17 | tie |
| 32 | 4096x4608x16384 | 32 | 8 | 14 | 32 | +39.7% | +28.9% | v17 | v17 |
| 32 | 8192x8448x8192 (4 waves) | 16 | 8 | 14 | 16 | +7.3% | +7.9% | tie | tie |
| 33 | 4352x4352x4096 | 8.25 | 7 | 12 | 9.1 | +5.7% | +11.0% | v19 | v19 |
| 33 | 4352x4352x8192 | 16.5 | 7 | 12 | 18.3 | +20.5% | +18.2% | v17 | v17 |
| 33 | 4352x4352x16384 | 33 | 7 | 12 | 36.6 | +34.3% | +33.1% | tie | v17 |
| 64 | 4096x5120x4096 | 16 | 4 | 6 | 16 | +14.2% | +8.8% | v17 | v17 |
| 64 | 4096x5120x8192 | 32 | 4 | 6 | 32 | +22.5% | +10.7% | v17 | v17 |
| 64 | 4096x5120x16384 | 64 | 4 | 6 | 64 | +32.9% | +16.6% | v17 | v17 |

What the numbers say:

1. **The lag alone is not the limit.** At the same `d` = 16 k-steps, v19 leads v17 by 10 points
   at S = 16 (K = 16384), ties at S = 32 (K = 8192) and trails by 5 points at S = 64
   (K = 4096). At S = 192 (entry 1), quadrupling `d` made v19's loss smaller, not larger.
2. **K moves the S = 32 boundary.** At S = 32-33, v19 wins at K = 4096 and loses or ties at
   K = 8192-16384.
3. **What decides is v17's serial collection against its own compute.**
   - v17's tail is its chunk (`iters_per_tile / n`) plus `n - 1` serial partial reads, which
     cost the same at any K.
   - Two-tile replaces those reads with at most one per tile, and pays a roughly fixed cost
     for its extra segments instead.
   - So two-tile wins when the collection outweighs the chunk. The rule gets 12 of 14 points;
     the other two are within 2 points either way.
4. **Waves before the tail only dilute the effect.** The four-wave S = 16 shape gives the same
   order (v19 +9.5%, v17 +6.5%) with smaller percentages.

**Revised host rule (one-tile vs two-tile).** With `n = 256 // S` and a full wave to borrow,
use reversed two-tile if `2 * (n - 1) > iters_per_tile / n`, and v17's tile-aligned split
otherwise. In words: two-tile when v17's owner would spend longer collecting partials than
computing its chunk. This gives two-tile for S <= 16 at every K tested, for S = 32-33 at
K = 4096, and never for S >= 64.

Reduce-scatter in v17 would cut that collection to about one partial's read, which removes
the case for two-tile. It is the next thing to test before fixing a host rule.
