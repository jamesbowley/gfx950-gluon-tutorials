# v20_streamk_reduce_scatter work log

One entry per change: what changed and what it did to performance and spills. Results come
from `HIP_VISIBLE_DEVICES=2 python regress.py ...` (bf16 unless noted), run from `a16w16/`
under the published v10+ stack: Triton v2.2 + deepbind shim on `PYTHONPATH`,
`LLVM_PASS_PLUGIN_PATH=plugins/llir_scheduler/libLlirSched.so`, `TRITON_AMDGCNAS_PLUGIN=1`.
Raw tables are in `results/`. One k-step is about 1.21 us.

## 1. Reduce-scatter fixup for the tile-aligned one-tile split (2026-10-01)

v19 with a second ending for v17's tile-aligned one-tile path. v19's two-tile path is
unchanged. v19 WORKLOG entry 2 asked whether reduce-scatter would cut v17's serial collection
enough to remove the case for two-tile; this entry answers that.

**The ending** (`REDUCE_SCATTER`, `STREAMK_FIXUP=rs`; `owner` keeps v17's):
1. Every program of a split tile, chunk 0 included, stores its four lane-contiguous quadrants
   to its slot `P[spid]`. A tile with one program stores C directly. The accumulator is dead
   after this, so there are no run-time role guards around it.
2. **Per-tile barrier** in a new `tile_sync` buffer: a count and a generation per tile, at the
   tile's first spid `b`.
   - Each program reads the generation, then arrives with an `acq_rel` add on the count.
   - The last arriver resets the count and releases `generation + 1`.
   - The others poll relaxed until the generation changes, then acquire once.

   The barrier re-arms itself, with no host state, so it is safe under graph capture.
3. **Slice reduce** (`reduce_scatter_slice`).
   - A slot is 64 blocks of 4 KiB, one per (quadrant, register vector). Each block is a
     32x32 square of the tile.
   - Program j of n sums blocks `[64 j / n, 64 (j + 1) / n)` over the tile's n slots with
     16-byte `.cv` loads, 16 loads per lane in flight. It adds the bias and stores bf16 C,
     4 columns per lane.
   - It reads its own partial back too, because its slice is a run-time subset of its registers.

**Scalar atomics.** The barrier relies on a scalar `gl.atomic_add` running once per workgroup.
It does: Triton predicates it to thread 0 and broadcasts the result. A scratch kernel with 8
programs incremented each count exactly once, and every thread saw the same old value.

**Host.**
- `STREAMK_POLICY` = `dp` | `one_tile` | `two_tile` | `auto`.
- `STREAMK_FIXUP` = `rs` | `owner` | `auto`.
- In this entry the defaults were `one_tile` and `rs`; entry 2 makes `auto` the default.
- `reduce_scatter_shape` picks the block group and the unroll from the smallest split count.

**Correctness** (`results/check_streamk.md`).
- `check_streamk.py` runs 30 rotating launches per configuration:
  - one-tile with both endings, and two-tile;
  - both tile orders, with and without bias;
  - the 9 stream-K shapes, 10 sweep shapes (including S = 2 at K = 16384, where n = 128 > 64
    blocks), and the capped shapes 256x65792x8192 and 256x67328x4096.

  All 244 configurations are correct, and every flag and barrier count is back to 0.
- **Negative test** (`--stale-count`): tile 0's count is preset to n - 1, so its first arriver
  takes itself for the last. One shape hangs, the other leaves a count up: FAIL, as it must.
  - A stale count of 1 does not fail. The tile's arrivals land so close together that the
    early release almost never overtakes the last partial, and the reset lands after the
    last arrival.
  - The GPU recovers after the hung process is killed.
- `check_blocks.py` (pure Python):
  - the block decode inverts `lane_contiguous_offsets` for all 65536 elements;
  - the slices cover each tile exactly once for every n from 2 to 128;
  - the host's group shape is valid for every S and K.

**Registers.** 484-488 VGPRs, **no VGPR spills and no scratch ops** (v19: 14 VGPR spills, 43
scratch ops). There are 3-5 SGPR spills, and 44 when n = 2-4 (NB = 16 blocks per group);
they go to VGPR lanes, not scratch. Regression shapes (`STREAMK_TILES = 0`): -0.6% to +1.8%
against v19 (`results/regression_shapes.md`).

### Where the tail's time goes

`fixup_timing.py` (`results/fixup_timing.md`) builds a copy of the kernel with
`s_memrealtime` stamps patched in. Medians are in k-steps; for `owner`, the wait is for the
last peer flag and the fixup is the serial reads plus the C store.

| shape | n | ending | MAC | partial store | barrier / flag wait | slice or owner fixup | tail |
|---|---|---|---|---|---|---|---|
| 3328x5120x8192 | 64 | owner | 7.5 | - | 20.7 | 125.4 | 157.2 |
| 3328x5120x8192 | 64 | rs | 6.4 | 7.5 | 9.1 | 8.6 | 37.4 |
| 4352x4096x8192 | 16 | owner | 15.4 | - | 14.5 | 33.3 | 70.3 |
| 4352x4096x8192 | 16 | rs | 15.6 | 8.9 | 6.2 | 8.0 | 47.9 |
| 8192x8448x8192 | 8 | owner | 20.8 | - | 11.8 | 19.4 | 75.7 |
| 8192x8448x8192 | 8 | rs | 22.5 | 8.5 | 7.9 | 8.4 | 72.9 |
| 4096x5120x8192 | 4 | owner | 35.4 | - | 9.2 | 11.2 | 63.6 |
| 4096x5120x8192 | 4 | rs | 35.6 | 7.1 | 8.2 | 8.5 | 70.6 |
| 4096x6144x8192 | 2 | owner | 66.3 | - | 6.2 | 7.4 | 90.4 |
| 4096x6144x8192 | 2 | rs | 67.1 | 6.7 | 13.6 | 9.7 | 109.6 |

1. **Reduce-scatter's ending costs about the same at every n**: store about 8, barrier wait
   7-9, slice 8-10, so 23-26 k-steps in all. The owner's grows with n: about 10 to wait, plus
   2.2 per peer read. They cross at n = 8.
2. **The slice reduce costs about 8 k-steps, not the 2 of an owner reading alone** (the
   risk noted in the plan). All programs read 64 MiB at once, so the read is a burst, like the
   store. More loads in flight (8, 16 or 32 per lane) changed nothing within noise.
3. **The MAC phase is HBM-bound.** On 4352x4096x8192 the 8 k-steps of each chunk take 15.6
   contended k-steps: the 16 tiles' 64 MiB of B is unique and is read in that window, at about
   3.6 TB/s.
4. **At n = 2 the barrier wait doubles** (13.6), and reduce-scatter moves twice the owner's
   partial bytes (n stores and reads instead of n - 1).

### Results

do_bench TFLOPS, v9 order (`results/leftover_sweep.md`, `results/s_vs_d.md`,
`results/streamk_shapes.md`). v20 is `one_tile` + `rs`; "owner" is the same build with
`STREAMK_FIXUP=owner` (`results/leftover_sweep_owner.md`, `results/s_vs_d_owner.md`).
`d = iters_per_tile * S / 256` is the two-tile lag, which is also the one-tile chunk length.

| S | shape | n | d | v13 | v17 | v19 | v20 rs | v20 owner |
|---|---|---|---|---|---|---|---|---|
| 2 | 1536x11008x8192 | 64 | 1 | 965 | 789 | **1331** | 1297 | 777 |
| 4 | 3328x5120x8192 | 64 | 2 | 1001 | 801 | **1314** | 1309 | 797 |
| 16 | 4352x4096x4096 | 16 | 4 | 976 | 944 | 1120 | 1120 | - |
| 16 | 4352x4096x8192 | 16 | 8 | 1041 | 1212 | 1278 | **1349** | 1211 |
| 16 | 4352x4096x16384 | 16 | 16 | 898 | 1210 | **1301** | 1288 | - |
| 16 | 4096x16640x8192 (4 waves) | 16 | 8 | 1457 | 1540 | **1592** | 1581 | 1545 |
| 32 | 4096x4608x4096 | 8 | 8 | 994 | 1109 | **1156** | 1143 | 1083 |
| 32 | 4096x4608x8192 | 8 | 16 | 1053 | 1318 | 1313 | **1364** | 1320 |
| 32 | 4096x4608x16384 | 8 | 32 | 1026 | 1432 | 1325 | **1457** | 1440 |
| 32 | 8192x8448x8192 (4 waves) | 8 | 16 | 1469 | 1572 | 1582 | 1582 | 1568 |
| 33 | 4352x4352x4096 | 7-8 | 8.25 | 1020 | 1098 | **1152** | 1126 | 1104 |
| 33 | 4352x4352x8192 | 7-8 | 16.5 | 1084 | 1301 | 1297 | 1312 | - |
| 33 | 4352x4352x16384 | 7-8 | 33 | 1020 | **1290** | 1283 | 1240 | 1287 |
| 64 | 4096x5120x4096 | 4 | 16 | 1084 | **1237** | 1181 | 1174 | 1232 |
| 64 | 4096x5120x8192 | 4 | 32 | 1143 | **1415** | 1257 | 1365 | 1425 |
| 64 | 4096x5120x16384 | 4 | 64 | 1147 | **1512** | 1341 | 1483 | 1510 |
| 128 | 4096x6144x8192 | 2 | 64 | 1312 | **1497** | 1408 | 1386 | 1496 |

From S = 192 to 255 (8 shapes) v20 is within 1% of v17 and 0.2-1.9% above v13. The tiles
there are split at most 2 ways, and the no-full-wave control 3840x4096x8192 is level with
v17.

With bias (`results/leftover_sweep_bias.md`), v20 against v19: +4.6% at S = 2, +0.8% at S = 4,
+6.5% at S = 16. In split order (`results/leftover_sweep_split_order.md`) the order of the
versions is the same.

What the do_bench numbers say:
1. **Reduce-scatter fixes the collection-bound one-tile shapes.** Against v17: +64% at S = 2
   and S = 4, +11% at S = 16 (+19% at K = 4096), +3.5% at S = 32. It wins wherever every tile
   is split at least 8 ways.
2. **It loses where n <= 4**: -3.5% at S = 64 and -7.4% at S = 128. There the owner reads only 1-3
   partials, and reduce-scatter pays a flat ~25 k-steps (finding 1 above) and doubles the
   partial traffic at n = 2.
3. **Against two-tile it is level or ahead under do_bench**: within 3% at S <= 4 and ahead at
   S = 16-32 with K = 8192. But do_bench is not a neutral judge here (next section).

### Clean or dirty caches decide between v19 and v20

Cold-cache rocprof (`rocprof_check.py`, `results/rocprof.md`: 1000 back-to-back launches over
rotating inputs) disagreed with do_bench:

| shape | S | v13 (us) | v17 (us) | v19 (us) | v20 rs (us) | v20 owner (us) |
|---|---|---|---|---|---|---|
| 3328x5120x8192 | 4 | 288.7 | 355.8 | **202.8** | 216.2 | 353.6 |
| 4352x4096x8192 | 16 | 289.4 | 247.3 | **211.7** | 221.0 | 247.5 |
| 4352x4352x8192 | 33 | 290.2 | 242.2 | **224.7** | 237.3 | 240.7 |
| 4096x6144x8192 | 128 | 295.0 | **271.3** | 288.9 | 293.0 | 272.0 |

The L2 hit rates are v17's (67.6% against 67.8% on 4352x4096x8192), so reduce-scatter costs
no A/B reuse; v19's are 6 points lower.

`between_launches.py` (`results/between_launches.md`) times each GEMM with CUDA events and
varies only what runs between launches. On 4352x4096x8192, in us:

| version | nothing | zero 256 MB (do_bench) | read 256 MB | zero 64 MB | sleep 100 us |
|---|---|---|---|---|---|
| v17 | 236.6 | 247.8 | 243.5 | 240.3 | 237.6 |
| v19 | 211.6 | **240.4** | 206.4 | 207.7 | 200.0 |
| v20 | 217.7 | 221.8 | 216.4 | 213.0 | 209.8 |

- **Only do_bench's flush hurts v19.** do_bench zeroes 256 MB before every launch, which leaves
  L2 and the last-level cache full of dirty lines that are written back while the GEMM starts.
  That costs v19 14% and v20 2%.
  - Reading 256 MB (cold but clean caches), zeroing 64 MB, an idle gap and back-to-back
    launches all leave v19 at 200-212 us.
  - My reading: v19 publishes its partials and polls its flags right at the start of the
    kernel, while the write-back saturates HBM. v20's stores come at the end.
- Serialising kernels (`AMD_SERIALIZE_KERNEL=3`) keeps the gap, so overlapping launches are not
  the cause.
- **With clean or mildly dirty caches, two-tile still beats one-tile reduce-scatter by 3-15%**
  wherever S <= 33 and d <= 18:
  - S = 2-4: +3-5%;
  - S = 16: +4-11%, the most at K = 4096;
  - S = 32-33 at K <= 8192: +5-15%;
  - with four full waves before the tail (4096x16640x8192, 8192x8448x8192): +1-2%.

  v20 is ahead at S = 32-33 with d >= 32 (K = 16384: v20 402-417 us against v19 430-435 at
  S = 32), and v17 is best from S = 64.
- With nothing between launches, the K = 4096 kernels (about 150 us) wait on the Python launch
  path, so that column is only valid for the longer kernels.

**So reduce-scatter does not remove the case for two-tile.** It cuts the one-tile
collection from up to 125 k-steps to a flat ~25, which makes one-tile the right choice
wherever two-tile's lag is long or there is no wave to borrow. But two-tile's single peer per
tile is still cheaper than a 25-k-step burst of all-to-all traffic when the lag is short.
Entry 2 turns this into the host rule.

## 2. Host rule: `STREAMK_POLICY=auto`, `STREAMK_FIXUP=auto` (2026-10-01)

Host-only change. `streamk_policy(total_tiles, iters_per_tile, 256)` picks the path, and both
env vars now default to `auto`. With `S = total_tiles % 256`, `n = 256 // S` programs per
tile in the one-tile split, and `d = iters_per_tile * S / 256`:

1. **`S = 0` or `S > 128`: data-parallel** (`STREAMK_TILES = 0`; the persistent loop also runs the
   partial last wave). Most leftover tiles would run whole anyway, so stream-K can only add
   traffic (v17 entry 3). The host no longer asserts a whole number of waves when
   `STREAMK_TILES = 0`.
2. **A full wave to borrow, `n >= 7` and `d <= 18`: two-tile** (v19's reversed path).
3. **Otherwise one-tile**, with reduce-scatter if `min(n, iters_per_tile / 2) >= 8` and the
   owner's ending below that.

**Where the thresholds come from.**
- **The fixup threshold (n >= 8)** is where the two endings cross in the timing table of entry 1:
  reduce-scatter's flat ~25 k-steps against the owner's ~10 + 2.2 (n - 1). The do_bench rows
  agree: at n = 8 reduce-scatter leads by 1-6%, at n <= 4 the owner by 3-7%.
- **The two-tile threshold** follows the clean-cache comparison (`results/between_launches.md`),
  not do_bench. With clean or mildly dirty caches, two-tile wins wherever n >= 7 and d <= 18
  (S = 2 to 33, K <= 8192 at S = 32-33). It is level or behind from d = 32 or n = 4.
- **Where do_bench disagrees.** Its 256 MB dirty flush moves two shapes to one-tile
  reduce-scatter: 4352x4096x8192 (1349 against 1268-1284) and 4096x4608x8192 (1364 against
  1298-1313). The rule keeps two-tile there, because only that flush, which a real
  predecessor kernel would rarely leave behind, makes two-tile lose.

**Correctness.** `check_streamk.py --policies auto --fixups auto` passes 68 configurations: 17
shapes covering every branch of the rule, both orders, with and without bias
(`results/check_streamk.md`).

**Results** (do_bench TFLOPS, v9 order, `results/leftover_sweep_auto.md`,
`results/s_vs_d_auto.md`; "clean best" is the fastest version with clean caches, read 256 MB
between launches, from `results/between_launches.md`):

| S | shape | auto picks | v13 | v17 | v19 | v20 auto | clean best |
|---|---|---|---|---|---|---|---|
| 2 | 1536x11008x8192 | two-tile | 956 | 788 | 1314 | 1357 | v19 |
| 4 | 3328x5120x8192 | two-tile | 996 | 801 | 1326 | 1312 | v19 |
| 16 | 4352x4096x4096 | two-tile | 974 | 953 | 1124 | 1136 | v19 |
| 16 | 4352x4096x8192 | two-tile | 1044 | 1216 | 1284 | 1268 | v19 |
| 16 | 4352x4096x16384 | two-tile | 892 | 1208 | 1292 | 1292 | v19 |
| 16 | 4096x16640x8192 | two-tile | 1451 | 1552 | 1589 | 1589 | v19 |
| 32 | 4096x4608x4096 | two-tile | 1002 | 1105 | 1153 | 1149 | v19 |
| 32 | 4096x4608x8192 | two-tile | 1054 | 1320 | 1298 | 1313 | v19 |
| 32 | 4096x4608x16384 | one-tile rs | 1007 | 1380 | 1322 | 1459 | v20 rs |
| 32 | 8192x8448x8192 | two-tile | 1464 | 1570 | 1581 | 1580 | v19 |
| 33 | 4352x4352x4096 | two-tile | 1018 | 1097 | 1150 | 1131 | v19 |
| 33 | 4352x4352x8192 | two-tile | 1083 | 1312 | 1290 | 1298 | v19 |
| 33 | 4352x4352x16384 | one-tile owner | 956 | 1289 | 1280 | 1291 (rerun) | v17 / v20 rs, level |
| 64 | 4096x5120x4096 | one-tile owner | 1081 | 1227 | 1180 | 1226 | v17 |
| 64 | 4096x5120x8192 | one-tile owner | 1145 | 1417 | 1252 | 1418 | v17 |
| 64 | 4096x5120x16384 | one-tile owner | 1146 | 1522 | 1347 | 1504 | v17 |
| 128 | 4096x6144x8192 | one-tile owner | 1301 | 1493 | 1389 | 1492 | v17 |
| 192 | 4096x7168x8192 | data-parallel | 1476 | 1491 | 1260 | 1503 | v17 by 3% (idle gap: v13) |
| 224 | 8192x7936x8192 | data-parallel | 1604 | 1602 | 1390 | 1620 | v13 / v20 dp |
| 240 | 4096x7936x8192 | data-parallel | 1541 | 1563 | 1128 | 1590 | v13 / v20 dp |
| 248 | 4608x7168x8192 | data-parallel | 1566 | 1584 | 1155 | 1609 | v13 / v20 dp |
| 255 | 1792x18688x8192 | data-parallel | 1496 | 1544 | 1360 | 1560 | v13 / v20 dp |
| 240 | 3840x4096x8192 (no full wave) | data-parallel | 1433 | 1467 | 1472 | 1495 | - |

The other S = 192-200 shapes give the same order (`results/leftover_sweep_auto.md`).

What the numbers say:
1. **auto picks the clean-cache best on every measured shape but one** (4096x7168x8192, where
   v17's split and data-parallel trade places depending on the gap between launches). Under
   do_bench it is within 2% of the best of v13, v17 and v19 on every shape; only one-tile
   reduce-scatter beats it there, on the two dirty-flush cases above. Where v20 auto and v19
   run the same path, they differ by 1-3% between runs.
2. **Reduce-scatter's place is narrow:** one-tile shapes where every tile is split at least 8
   ways but two-tile's lag is long (S <= 32 at K >= 16384) or there is no full wave to borrow
   (for example 256x65792x8192 and other single-wave tails). 4096x4608x16384 is the measured
   case: +5.7% over v17 and +10% over v19.
3. **The data-parallel branch is level with v13 with clean caches** (`between_launches.md`,
   last table), and 1-3% above it under do_bench. v17's split of S > 128 is level or 1-3%
   behind, which supports v17 entry 3's rule.

### Next

- The tail's MAC phase is HBM-bound (entry 1, finding 3): with one-tile, 16 leftover tiles'
  B columns are read in 8 k-steps. Ordering the leftover tiles so that tiles sharing B
  (same pid_n) are split together, or giving the tail a deeper prefetch, would attack the
  15-k-step MAC phase, which is now the largest part of the S = 16 tail.
- Two-tile is sensitive to a dirty memory system at kernel start. Moving its first partial
  store later, or publishing the head piece after the first whole tile, could remove the
  do_bench loss.
- The reduce-scatter burst (store about 8 k-steps, slice about 8) is bandwidth-bound. Keeping
  each program's own slice in registers saves 1/n of it; storing partials in a narrower
  format would halve it, at a cost in precision.

## 3. Experiment: end-to-end stream-K for S > 128 at large K (2026-10-01)

**The question.** Entry 2 uses data-parallel for S > 128, because with v17's tile-aligned split
most leftover tiles get one program and run whole. An even split avoids that: the
end-to-end partition of v15/v16 gives every program `S / 256` of a tile.
- At S = 192 that is each 3 tiles to 4 programs, at offsets 0, 0.75, 0.5 and 0.25 of a tile,
  with one peer per seam.
- On a one-wave shape the ideal gain is a quarter of a wave, 12.5%; at S = 224 and 240 it is
  about 3%.
- The fixed costs (fixup, restarts) shrink relative to the work as K grows. L2 loss and the
  power limit do not.

**Change.**
- A constexpr `END_TO_END` now selects v19's end-to-end reversed path, replacing
  `STREAMK_TILES > NUM_PROGRAMS`.
- A new `STREAMK_POLICY=spread` runs that path on the one-tile leftover tiles (`STREAMK_TILES = S`).
  Each program processes its head piece first, so every program that crosses a tile boundary
  starts at k-step 0.
- `two_tile` sets `END_TO_END` when it has a wave to borrow, so its behaviour is unchanged.
- `auto` never picks `spread`.
- v16 is the same split in forward order, as a comparison.

**Correctness.**
- `check_partition.py` (pure Python, `results/partition.md`) replicates the end-to-end
  partition for both policies, S = 1-255 and K = 8192-65536 (2040 cases). It checks that:
  - every pair is covered once;
  - every tile has one owner, and its peers are the contiguous lower spids (at most 2 for
    S > 128, at most 1 for two-tile);
  - every program has at most one partial, processed first.
- `check_streamk.py --policies spread` passes 48 configurations: S = 4 to 240, K up to 65536,
  both orders, with and without bias (`results/check_streamk.md`).
- `spread` builds have v19's register state: 512 VGPRs and 14 spills.

**Distinct k-steps per XCD** (lockstep model, `results/partition.md`). With q the denominator of
`S / 256`, forward order reads q distinct k-steps at once (4 at S = 192, 8 at S = 224, 16 at
S = 240). Reversed order reads about 2 at every S > 128.

**Results.** Clean caches (`between_launches.py`, read 256 MB between launches,
`results/spread_between_launches.md`). The K >= 32768 rows are the reruns with the version
order reversed, on GPU 2, with GPU 4's value in brackets. Times are in us; "gain" is spread
against the faster of v13 and v20 `dp`.

| S | shape | v13 | v20 dp | v16 (forward) | v17 | v19 | v20 spread | gain | ideal |
|---|---|---|---|---|---|---|---|---|---|
| 128 | 4096x6144x8192 (control) | 287.0 | 284.8 | 264.2 | 261.6 | 276.8 | 260.2 | = v17 | - |
| 192 | 4096x7168x8192 | 299.2 | 290.3 | 332.7 | 310.1 | 354.8 | 302.4 | -4.2% | 12.5% |
| 192 | 4096x7168x16384 | 580.4 | 592.9 | 610.1 | 619.8 | 678.4 | 583.5 | -0.5% | 12.5% |
| 192 | 4096x7168x32768 | 1246.3 | 1252.0 | 1239.2 | 1258.7 | 1302.0 | 1159.4 (1244.2) | +7.0% (+6.6%) | 12.5% |
| 192 | 4096x7168x65536 | 2771.9 | 2770.6 | 2580.2 | 2737.1 | 2671.9 | 2497.3 (2510.1) | +9.9% (+9.4%) | 12.5% |
| 224 | 8192x7936x32768 (3 waves) | 2640.4 | 2628.7 | 3078.3 | 3117.3* | 3340.7* | 2599.4 (2605.7) | +1.1% (+1.4%) | 3.1% |
| 240 | 4096x7936x32768 | 1318.0 | 1324.1 | 1851.3 | 1323.6 | 1912.5 | 1286.5 (1309.4) | +2.4% (+2.4%) | 3.1% |

The v17 and v19 columns are from the first run. \* That run's 8192x7936x32768 rows are inflated for
every version (v13 3111 us).
Cold rocprof (`results/spread_rocprof.md`) has v17 and v19 there at 2710 and 3151 us, against
v13's 2681.

- **Power** (`power_check.py`, `results/spread_power.md`). Every kernel runs at the board's
  limit of about 1.41 kW at K = 32768, v13 included.
  - S = 192: spread is 5.3% faster at the same power (1218 against 1286 us) and uses 4.8%
    less energy per launch.
  - S = 240: level (+0.3%).
- **do_bench** (`results/spread_do_bench.md`) on S = 192 against v13: +1% at K = 8192, +3% at
  16384, +6% at 32768, +8.4% at 65536. At S = 240 it is level. do_bench's numbers for these
  2-3 ms kernels vary by up to 5% between runs of the same data-parallel kernel (v13 against
  v20 `dp`), so they are only a cross-check here.
- **Cold rocprof** (back to back, rotating inputs) on S = 192, K = 32768: spread 1231 against
  1306-1317 us (+6%). At S = 224 it is 1.3% behind, at S = 240 level.

What the numbers say:
1. **Yes: at large K, spreading the S > 128 tail beats data-parallel.** At S = 192 the gain goes
   from -4.2% at K = 8192, through level at 16384, to +7% at 32768 and +10% at 65536, which is
   80% of the ideal 12.5%.
   - The fixed cost is about 40-75 us at every K: the restart, the partial store and read, and
     a few points of L2 hits.
   - The ideal saving is a quarter of a wave and grows with K, so it overtakes that cost
     between K = 16384 and 32768.
2. **The reversed order is what makes it work, as in v19.**
   - Forward order (v16) gains at S = 192 only at K = 65536 (+7%). It is 17% behind at
     S = 224 and 40% behind at S = 240: its q = 8 and 16 distinct offsets cost the L2 reuse
     (40-62% L2 hits).
   - Reversed spread keeps 68-75% L2 hits, against data-parallel's 78-80%.
   - Two-tile (v19) moves a whole extra wave into stream-K and loses at every point here.
3. **Small leftovers gain little.** At S = 224 and 240 the ideal is about 3%. Spread captures
   1.1-2.4% with clean caches, and nothing when the board is power-capped (S = 240) or in cold
   rocprof (S = 224).
4. **The S = 128 control matches v17** (260.2 against 261.6 us). There the end-to-end split and
   the tile-aligned split are the same partition.

**Proposed rule (not applied).** Add to `streamk_policy`: for `128 < S` and
`iters_per_tile >= 512` (K >= 32768 with 64-wide k-steps), use `spread`; otherwise keep
data-parallel. That covers every measured win, including power-capped S = 192, and keeps
data-parallel at S = 192 with K <= 16384.
- The points are few: S = 224 and 240 were only measured at K = 32768, and S = 160 not at all.
- A rule on the ideal saving, `(256 - S) * iters_per_tile / 256` k-steps, would separate S
  better, but the fixed cost differs between S values (about 10 us at S = 240, about 45-50 us
  at S = 224, 40-75 us at S = 192). One threshold does not fit.
- So the rule should wait for a sweep over S = 136-248 at K = 16384-65536.

## 4. Split count and small K (2026-10-01)

Full write-up: [experiments/streamk_split_count](../../../../../experiments/streamk_split_count/README.md).

**Change, in v15-v20.**
- The `STREAMK_NUM_PROGRAMS` constexpr, unused until now, takes its value from the environment
  variable of the same name (default 256). Only that many programs take part in the stream-K phase.
- `streamk_compact_pid` gives each XCD the same number of active programs, numbered
  contiguously; `STREAMK_BALANCE_XCDS=0` idles the last spids instead.
- In the tile-aligned split, `STREAMK_NUM_PROGRAMS = S * n` gives n programs per tile. In the
  end-to-end split it gives each program 1/n of a tile.
- v20's `reduce_scatter_shape` and `streamk_policy` (its fixup choice) follow the program count.
  `auto` is unchanged.
- Correctness: `check_all.py` runs every version's `check_streamk.py`, 258 configurations, all
  correct with flags and counts re-armed. `check_partition.py` covers the compacted ids and the
  reduced partitions.

**Findings** (rocprof kernel time):
1. **The default split (all 256 programs) over-splits when S is small.**
   - Capping n speeds up the owner ending by 39% at S = 4, 10% at S = 16 and 4% at S = 38
     (K = 8192). Reduce-scatter gains 6%, 3% and 3%.
   - The owner's best n follows `sqrt(iters_per_tile / c)` with `c = max(2.2, S / 5)`.
   - Reduce-scatter wants more splits (n = 16 at S = 4).
2. **With the split capped, one-tile ties or beats two-tile.**
   - In rocprof, by 2-5% at K = 8192 and level at K = 4096.
   - With clean caches it is level, within about 4%.
   - So most of entry 1's gap to two-tile was the split count. Fewer programs only slow
     two-tile down (213 to 315 us from 256 to 128 programs at S = 16).
3. **Below K = 2048 no split pays.** At K = 512-1024 data-parallel beats every stream-K
   configuration by 0.5-6%, and the default splits by 30-80%.
   - At K = 2048 a split pays only for S <= 64, and only 2-3 ways (4-16 for reduce-scatter at
     S = 4).
   - The threshold that fits: split when `(256 - S) / 256 * iters_per_tile >= ~20` k-steps.
4. **Spreading the idle programs over the XCDs is worth 11-15%** against idling whole XCDs
   (S = 38, n = 2).

**Proposed rule (not applied).**
- Data-parallel when `(256 - S) * iters_per_tile < 20 * 256`.
- Otherwise one-tile with a capped split:
  - reduce-scatter if `256 // S >= 8`, with `n = min(256 // S, max(4, iters_per_tile / 8))`;
  - otherwise the owner's ending, with `n = min(256 // S, round(sqrt(iters_per_tile / max(2.2, S / 5))))`.
- Drop two-tile.
- For S > 128 at K >= 32768, entry 3's `spread`.

## 5. `STREAMK_POLICY=capped` (2026-10-02)

Host-only change. Entry 4's rule is now selectable, refit to the version-table sweep
([experiments/streamk_version_tables](../../../../../experiments/streamk_version_tables/sweep.py)),
so that it can be compared with `auto` on every shape. `auto` stays the default and two-tile
stays available.

**The rule** (`capped_policy(total_tiles, iters_per_tile, 256)`), with S leftover tiles and
k = iters_per_tile:
1. S = 0: data-parallel.
2. S > 128: `spread` if k >= 512 (entry 3), otherwise data-parallel.
3. `(256 - S) * k < 20 * 256`: data-parallel (entry 4, finding 3).
4. `256 // S >= 8`: one-tile reduce-scatter with `n = min(128 // S, max(8, k // 8))`.
5. Otherwise one-tile owner with `n = min(256 // S, round(sqrt(k / max(2.2, S / 5))))`.

It returns `STREAMK_NUM_PROGRAMS = S * n`, or all programs when n reaches the default split.
An explicit `STREAMK_NUM_PROGRAMS` in the environment still overrides it, and `STREAMK_FIXUP`
overrides the ending as with `auto`.

**Why the rs split changed from entry 4.** Entry 4's `n = min(256 // S, max(4, k / 8))` fails in
two places:
- At S = 16 and K = 8192 it gives n = 16, the default split, where n = 8 is 3.8% faster.
- At S = 4 and K = 2048 it gives n = 4, where n = 8 is 7% faster.

In the version-table sweep, reduce-scatter's best n is 16 for S = 1-4 at K = 8192, and 8 for S = 7-16 at K >= 4096 and for S = 4 at K = 2048. `min(128 // S, max(8, k // 8))` gives each of these. The one shape it misses is 4352x4096x2048, where it gives 8 but v17 at n = 3 is fastest.

**Correctness** (`results/check_streamk.md`). `check_streamk.py --policies capped` covers 16 shapes and every branch of the rule, including no full wave. With both tile orders, and with and without bias, that is 64 configurations of 30 rotating launches. All are correct, with every flag and count re-armed. The kernel is unchanged, so registers and spills are those of the path picked.

**Results.** Cold rocprof, median of 3 processes, GPU 0, from
[results/tables.md](../../../../../experiments/streamk_version_tables/results/tables.md).
"Two-tile" is the faster of v19 and v20 `two_tile`. "Best other" is the fastest of every other
configuration, including every n of the split-count sweep.

| shape | S | k | capped picks | capped | auto | two-tile | best other | capped vs auto | capped vs best other |
|---|---|---|---|---|---|---|---|---|---|
| 4096x4096x8192 | 0 | 128 | dp | 1558 | 1560 | 1559 | 1566 (v9) | -0.1% | -0.5% |
| 8192x8192x8192 | 0 | 128 | dp | 1619 | 1621 | 1621 | 1627 (v12) | -0.1% | -0.5% |
| 1536x11008x8192 | 2 | 128 | rs, n = 16 | 1415 | 1384 | 1387 | 1411 (rs n 16) | +2.3% | +0.3% |
| 3328x5120x8192 | 4 | 128 | rs, n = 16 | 1426 | 1366 | 1371 | 1431 (rs n 16) | +4.3% | -0.4% |
| 4352x4096x4096 | 16 | 64 | rs, n = 8 | 1208 | 1217 | 1216 | 1217 (auto) | -0.8% | -0.8% |
| 4352x4096x8192 | 16 | 128 | rs, n = 8 | 1393 | 1349 | 1349 | 1390 (rs n 8) | +3.2% | +0.2% |
| 4352x4096x16384 | 16 | 256 | rs, n = 8 | 1452 | 1428 | 1448 | 1451 (rs n 8) | +1.7% | +0.1% |
| 4096x16640x8192 (4 waves) | 16 | 128 | rs, n = 8 | 1568 | 1570 | 1572 | 1572 (two_tile) | -0.1% | -0.2% |
| 4096x4608x8192 | 32 | 128 | rs, n = 4 | 1378 | 1363 | 1364 | 1393 (rs n 6) | +1.1% | -1.0% |
| 8192x8448x8192 (4 waves) | 32 | 128 | rs, n = 4 | 1552 | 1573 | 1575 | 1575 (v19) | -1.3% | -1.5% |
| 4352x4352x8192 | 33 | 128 | owner, n = 4 | 1384 | 1351 | 1351 | 1383 (v17 n 4) | +2.4% | +0.1% |
| 3584x5376x8192 | 38 | 128 | owner, n = 4 | 1397 | 1334 | 1353 | 1398 (v17 n 4) | +4.7% | -0.1% |
| 4096x5120x8192 | 64 | 128 | owner, n = 3 | 1444 | 1424 | 1360 | 1452 (owner n 3) | +1.4% | -0.6% |
| 4096x6144x8192 | 128 | 128 | owner, default | 1495 | 1494 | 1424 | 1494 (auto) | +0.1% | +0.1% |
| 4096x7168x8192 | 192 | 128 | dp | 1499 | 1498 | 1312 | 1504 (v10) | +0.1% | -0.4% |
| 4096x7168x32768 | 192 | 512 | spread | 1557 | 1474 | 1419 | 1561 (spread) | +5.6% | -0.2% |
| 8192x7936x8192 | 224 | 128 | dp | 1595 | 1596 | 1373 | 1603 (v12) | -0.1% | -0.5% |
| 4096x7936x8192 | 240 | 128 | dp | 1575 | 1577 | 1144 | 1578 (v11) | -0.1% | -0.2% |
| 1792x18688x8192 | 255 | 128 | dp | 1557 | 1556 | 1392 | 1558 (dp) | +0.1% | -0.1% |
| 3840x4096x8192 (no full wave) | 240 | 128 | dp | 1501 | 1501 | 1472 | 1512 (v10) | 0.0% | -0.8% |
| 4352x4096x1024 | 16 | 16 | dp | 781 | 640 | 640 | 790 (v11) | +22.0% | -1.2% |
| 4352x4096x2048 | 16 | 32 | rs, n = 8 | 965 | 888 | 893 | 998 (v17 n 3) | +8.7% | -3.3% |
| 3328x5120x2048 | 4 | 32 | rs, n = 8 | 992 | 946 | 948 | 1015 (rs n 8) | +4.8% | -2.3% |
| 4096x5120x2048 | 64 | 32 | owner, n = 2 | 1048 | 990 | 954 | 1058 (v17 n 2) | +5.8% | -1.0% |
| 4096x6144x2048 | 128 | 32 | dp | 1158 | 1146 | 1038 | 1205 (v11) | +1.1% | -3.9% |
| 4096x7168x1024 | 192 | 16 | dp | 1078 | 1079 | 745 | 1092 (v11) | -0.1% | -1.4% |
| 256x65792x8192 | 1 | 128 | rs, n = 16 | 1101 | 975 | 973 | 1078 (rs n 16) | +12.9% | +2.1% |
| 256x67328x4096 | 7 | 64 | rs, n = 8 | 857 | 825 | 810 | 853 (rs n 8) | +3.9% | +0.4% |
| 256x32768x8192 (no full wave) | 128 | 128 | owner, default | 756 | 774 | 774 | 787 (v18) | -2.4% | -3.9% |
| 256x49152x8192 (no full wave) | 192 | 128 | dp | 1088 | 1073 | 925 | 1083 (v13) | +1.4% | +0.5% |

What the numbers say:
1. **`capped` is within 1.5% of the best on 26 of the 30 shapes; `auto` is within 2.5% on 19.**
   `capped` gains the most on short K (+22% where `auto` picks two-tile at K = 1024), on skinny
   shapes (+13% at S = 1) and through `spread` (+5.6% at S = 192, K = 32768). On compute shapes
   with S = 2-38 it is 1.1-4.7% ahead of `auto` on 7 of 10.
2. **Only one of its four misses is a wrong pick:** 4352x4096x2048, -3.3% (rs at n = 8 where v17 at n = 3 is best). The others:
   - 3328x5120x2048 (-2.3%): the same configuration as the best, so this is run-to-run variation.
   - 4096x6144x2048 (-3.9%): `capped` picks `dp`, which compiles to the same kernel as v20 `dp` (1200).
   - 256x32768x8192: inside that shape's 30% process-to-process spread.
3. **Two-tile still wins with more waves before the tail.** On 4352x4096x4096 and both four-wave shapes, two-tile is 0.2-1.5% ahead of `capped`. On the other 6 compute shapes with S = 2-33, `capped` is 0.3-4.0% ahead.

   This is why two-tile stays: the four-wave result rests on two shapes, and the coefficients of `capped` are fit to these same shapes. Shapes with 8-16 full waves, held-out S and K (S = 8, 24, 48 and 96; K = 3072 and 6144) and real model GEMMs should decide whether `capped` replaces `auto`.

## 6. C in its own dtype (2026-10-02)

The kernel cast C to A's dtype, so it could not write an fp32 C from bf16 inputs. The four
stores of `persistent_tile` and the `out_dtype` of `streamk_segment` (both endings) now use
`c_ptr.dtype.element_ty`. With a bf16 C the kernel is unchanged. `matmul` still allocates C in
A's dtype when none is passed.

- `check_streamk.py --policies auto --fixups auto` passes on 5 shapes (K = 512 to 16384), bf16.
- fp32 C matches an fp32 reference on all three paths, one-tile owner (512x6144x2048), one-tile
  reduce-scatter (512x1024x5120) and data-parallel (2048x6144x3072), with the flags and counts
  back at 0 after graph replay.

This is for the comparison against AITER's picks on the 630 shapes in
`experiments/aiter_lixun_shapes/` (32 of them have an fp32 output).
