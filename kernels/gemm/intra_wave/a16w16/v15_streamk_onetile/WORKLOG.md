# v15_streamk_onetile work log

One entry per change: what changed and what it did to performance and spills. Results come
from `HIP_VISIBLE_DEVICES=2 python regress.py ...` (bf16 unless noted), run from `a16w16/`
under the published v10+ stack: Triton v2.2 + deepbind shim on `PYTHONPATH`,
`LLVM_PASS_PLUGIN_PATH=plugins/llir_scheduler/libLlirSched.so`, `TRITON_AMDGCNAS_PLUGIN=1`.
Raw tables are in `results/`. One k-step is about 1.21 us (v14 with amdgcnas, measured in
`experiments/streamk_costs`).

## 1. TensorAtlas one-tile stream-K (2026-09-30)

v14 with the peeled last wave replaced by TensorAtlas's stream-K tail
(`TensorAtlas/kernels/gemm/streamk_matmul.py`, explained in
`../v14_streamk/STREAMK_EXPLAINED.md`). Same algorithm:

- `STREAMK_TILES = total_tiles % 256`;
- the stream-K tiles' k-steps are split evenly over all 256 programs;
- the program that computes a tile's k-step 0 owns it, walks `pid + 1, pid + 2, ...`, adds
  their partials and the bias, and stores C;
- row-major fp32 partials, `.wt` stores and `.cv` loads;
- every stream-K segment starts with its own prologue.

Changes from TensorAtlas, all forced:

- **Work is handed out in pairs of k-steps**, so every segment is even-length and starts on an
  even k-step, which v14's pair pipeline needs (doc section 12). 4352x4096x8192 gets exactly
  TensorAtlas's split.
- **Memory ordering** (doc section 11). Each wave drains its stores, then a barrier, then a
  GPU-scope release flag store. The owner polls relaxed and acquires once.
- **Flags are re-armed**: the owner resets each peer's flag after reading its partial. The
  reference never resets them (doc section 13).
- **No zero-init of `P`.**

Implementation:

- **Segment body.** `streamk_segment` runs a run-time loop of `k_step_pair` (1 or more pairs)
  from zero accumulators. The last pair's prefetch is clamped to reload the last pair, and
  the drain discards it. There is no peeled pair and no interleaved epilogue, so 2-k-step
  segments need no special path.
- **Endings.** Contributor and owner run the same guarded code (from
  `experiments/streamk_costs`):
  - the peers are summed in VGPRs and added into the accumulator once;
  - the owner stores C late;
  - `n_peers` comes from the closed-form `streamk_owner_of`, not a walk.
- **Host.** `P` (64 MiB) and `locks` are allocated once per device and cached.

**Correctness.** `check_streamk.py` runs 30 launches on rotating inputs for every stream-K
shape: v9 and split tile order, with and without bias, 36 configurations. Every result is
correct and every flag is back to 0 afterwards. With the flags preset to 1 (a stale-lock
bug), the check fails as it should.

**Registers.** 512 VGPRs, 14 VGPR and 16-22 SGPR spills. There are no scratch ops in the
persistent loop or the K loops. 29 scratch ops (41 with bias) sit in the stream-K fixup,
around the four peer-read loops, and 14 sit once before the stream-K loop. With
`STREAMK_TILES = 0` the tail is compiled out, and v15 has v14's 480 VGPRs and no spills.

**Regression shapes** (`STREAMK_TILES = 0`, `results/regression_shapes.md`): all within noise
of v14 (-1.1% to +4.7%).

### Results, stream-K shapes

> Measured with the double XCD grouping of the stream-K tiles (v9 order); see entry 3
> for the fix and the corrected numbers. The old raw tables are in `results/pre_xcd_fix/`.

`results/streamk_shapes.md`, v9 tile order, TFLOPS; the deltas are against v13, the
persistent baseline.

| shape | what it tests | v13 | v9 | v14 | v15 | v15 vs v13 |
|---|---|---|---|---|---|---|
| 4352x4096x8192 | 16-way split, 15 peers | 1040 | 1026 | 1035 | 922 | -11.3% |
| 4352x4352x8192 | drifting seams, 8 crossing, 2-k-step segments | 1089 | 1083 | 1069 | 1123 | +3.2% |
| 4352x4096x4096 | as the first, 4 k-steps per program | 989 | 977 | 969 | 636 | -35.7% |
| 4352x4096x16384 | as the first, 16 k-steps per program | 944 | 951 | 944 | 1108 | +17.3% |
| 3328x5120x8192 | 4 SK tiles, 2-k-step segments, 63 peers | 995 | 1009 | 995 | 438 | -56.0% |
| 4096x6144x8192 | 128 SK tiles, 2-way, 1 peer | 1317 | 1314 | 1313 | 1422 | +7.9% |
| 3840x4096x8192 | 240 tiles, no full wave | 1426 | 1452 | 1433 | 915 | -35.8% |
| 8192x8448x8192 | 4 full waves + 32 SK tiles | 1462 | 1460 | 1466 | 1500 | +2.7% |
| 8192x7936x8192 | 3 full waves + 224 SK tiles | 1613 | 1597 | 1595 | 1496 | -7.3% |

With bias (`results/streamk_shapes_bias.md`) the deltas are the same within 1-2 points, except
4352x4096x16384 (+7.7% instead of +17.3%) and 4352x4352x8192 (+0.2% instead of +3.2%).

rocprof with cold caches (`results/rocprof.md`) matches do_bench within about 3%. The one
shape where the verdict changes is 4352x4352x8192: v15 is 1.1% slower than v13 there.

### Tail time against the model

> Measured with the double XCD grouping of the stream-K tiles (v9 order); see entry 3
> for the fix and the corrected numbers. The old raw tables are in `results/pre_xcd_fix/`.

- **Tail:** the kernel time minus the full waves' time. The full-wave time is v13 on a
  whole-wave shape with the same K: 4096x4096xK for one wave, 8192x6144x8192 for three,
  8192x8192x8192 for four (`results/full_wave_refs.md`).
- **Model:** `streamk_predict.py`, the v14 lockstep model in 2-k-step units, with row-major
  costs (burst store 7, read 4.9 k-steps) and lane-contiguous costs (6 and 2).
- **Share of the gain captured:** `(v15 - v13) / (ideal - v13)` in TFLOPS, where "ideal" is
  v13 on the whole-wave shape.

| shape | v13 tail (us) | v15 tail (us) | model, row-major (us) | model, lane (us) | share of the gain captured |
|---|---|---|---|---|---|
| 4352x4096x8192 | 97 | 133 | 107 | 53 | -26% |
| 4352x4352x8192 | 101 | 92 | 72 | 46 | +8% |
| 4352x4096x4096 | 36 | 118 | 102 | 48 | -147% |
| 4352x4096x16384 | 236 | 145 | 117 | 63 | +33% |
| 3328x5120x8192 | 97 | 454 | 384 | 162 | -112% |
| 4096x6144x8192 | 129 | 106 | 92 | 87 | +59% |
| 3840x4096x8192 | 181 | 282 | 160 | 155 | -734% |
| 8192x8448x8192 | 98 | 78 | 69 | 44 | +24% |
| 8192x7936x8192 | 146 | 197 | 150 | 145 | n/a (v13 above ideal) |

What the numbers say:

1. **The data-parallel last wave costs far less than the doc's model assumed.** The model
   charges it 128 k-steps at the contended rate: 155 us at K=8192. Measured, 16 tiles on 16
   CUs take 97 us (0.76 us per k-step), because 16 CUs alone get the whole memory system. At
   K=4096 it is 36 us. So every "data-parallel last wave" row in the doc's tables overstates
   what stream-K can win. One-tile stream-K only wins where the last wave is long (K=16384:
   +17%) or big (128 of 256 tiles: +8%).
2. **Where the fixup dominates, v15's tail follows the row-major model**, 10-30% above it. The
   gap is segment prologues, C stores and spills, none of which the model counts. The serial
   collection is the tail:
   - 15 peers at 4.9 k-steps a read on 4352x4096x8192;
   - 63 peers on 3328x5120x8192, 454 us, against 97 us for the data-parallel last wave.

   Lane-contiguous partials (next version) should halve these tails.
3. **Long ranges lose their L2 reuse.** On 3840x4096x8192 and 8192x7936x8192 every program
   runs 112-120 k-steps and 192-224 of them cross a tile boundary. The tails are 1.3-1.8x the
   model, and the fixup is small there (1 peer), so the fixup is not the cause. The L2
   counters (`results/rocprof.md`) show the cause:
   - on 3840x4096x8192 the L2 hit rate falls from 76% (v13) to 16% (v15), and memory read
     requests go from 3.65M to 14.88M;
   - on 4352x4096x8192, where every program's chunk starts on a boundary shared by all
     tiles, v15 keeps 67%.

   In data-parallel, the programs of an XCD read the same k-slice of shared A rows and B
   columns at about the same time. In stream-K each program starts at its own k offset
   (`start % tile`), so those loads no longer coincide.
4. **Tile order** (`results/streamk_shapes_split_order.md`, `PERSISTENT_TILE_ORDER=split`,
   against v13 in the same order):
   - 3840x4096x8192 gets worse (812 against 915 TFLOPS in v9 order), so scattering the
     stream-K tiles across XCDs is not what costs the L2 reuse;
   - 4096x6144x8192 (1472) and 4352x4352x8192 (1137) are the best v15 numbers in either order;
   - v13 itself is 2-5% slower in split order.

## 2. Two-tile on v15's kernel (2026-09-30)

> Measured with the double XCD grouping of the stream-K tiles (v9 order); see entry 3
> for the fix and the corrected numbers. The old raw tables are in `results/pre_xcd_fix/`.

Not a v15 change: `two_tile_check.py` launches the unchanged kernel with the two-tile host
policy, `STREAMK_TILES = total % 256 + 256`. The kernel is correct under it with no changes:
- ranges now have up to three segments;
- the middle segment is a whole tile, stored straight to C;
- each tile has at most one peer.

All 7 shapes with at least one full wave are correct over 10 launches, and the flags are
re-armed (`results/two_tile.md`).

| shape | v13 | one-tile | two-tile |
|---|---|---|---|
| 4352x4096x8192 | 1035 | 928 | 913 |
| 4352x4352x8192 | 1082 | 1100 | 805 |
| 4352x4096x16384 | 958 | 1101 | 949 |
| 3328x5120x8192 | 999 | 436 | 890 |
| 4096x6144x8192 | 1322 | 1430 | 1346 |
| 8192x8448x8192 | 1461 | 1488 | 1465 |
| 8192x7936x8192 | 1570 | 1489 | 1291 |

- **Two-tile loses to one-tile on 6 of 7 shapes, and never clearly beats v13.** It only helps
  where one-tile's fixup is pathological: 3328x5120x8192 goes from 436 to 890, still below
  v13's 999.
- **The cause is L2, not the fixup.** Each tile has at most one peer, so collecting partials
  is cheap. But every program now runs 136-145 k-steps from its own k offset, and on
  4352x4352x8192 the L2 hit rate falls from 65% (one-tile) to 11%. Memory read requests go
  from 6.47M to 18.59M. This is the same drift that costs one-tile on its long-range shapes
  (entry 1, finding 3). Two-tile applies it to a whole wave of tiles instead of the tail.
- **So two-tile is not worth pursuing until the programs of an XCD read the same A/B k-slices
  again.** Lane-contiguous partials barely help it, with only one peer per tile.
- **Measurement note:** the first `do_bench` of a process can read 25-30% low (v13 at 986 or
  1112 against its usual 1322 or 1461), so `two_tile_check.py` throws one away.

### Next

- v16: lane-contiguous partials. The model halves the fixup-bound one-tile tails.
- The k-offset drift is the bigger lever, and it gates two-tile. Options: cuts aligned
  across tiles so that chunk j of every tile covers the same k-steps (tile-aligned policy
  (a) with a common split count), or the reversed order, which starts crossing programs at
  k-step 0. Each can be checked with the L2 counters in `results/rocprof.md`.
- A host-side rule for when to run stream-K at all: v13 wins wherever the last wave is short
  (K=4096) or holds most of a wave.

## 3. XCD grouping fix (2026-10-01)

**What was wrong.**
- The stream-K phase numbers its programs with the XCD-grouped id `spid`
  (`get_logical_chiplet_mapped_pids`: 32 consecutive spids per XCD).
- It then looked up stream-K tile `t` with `persistent_tile_id(total_full_tiles + t)`. Under the
  default v9 order that function applies the same XCD grouping again (`xcd_remap_tiles`), which
  is meant for raw program ids.
- So consecutive stream-K tiles, which belong to consecutive spids on one XCD, were spread
  across XCDs. Under split order the grouping was applied only once, which is why split order
  never had the problem.

**Why it happened.**
- In the v14 scaffold the peeled tile ran on program `start` (the raw pid under v9 order) with
  the same lookup, which was correct.
- v15 switched the stream-K numbering to `spid`, because the TensorAtlas partition needs a
  tile's programs on consecutive numbers and the owner counts up `spid + 1, ...`. It kept the
  lookup because the lookup guaranteed that the stream-K tiles are exactly the tiles the
  persistent loop leaves over. Only that set was checked, not which XCD each tile lands on.
- It went unnoticed for two reasons:
  - correctness tests cannot see it, since any assignment of tiles to programs is correct;
  - the v15 split-order run in entry 1 made 3840x4096x8192 slower, which suggested the tile
    order did not matter. With this partition's drifting k offsets it doesn't, as the results
    below show. It only became visible in v17, after the cuts were aligned.

**The fix**, in v15, v16 and v17:
- `streamk_tile_id(t, ...)` replaces the lookup.
- Split order: `total_full_tiles + t`, as before, so split-order builds are unchanged.
- v9 order: the `t`-th smallest of the tiles the persistent loop leaves over. That is the same
  set as before, in ascending tile-id order, so consecutive spids get neighbouring tiles. It
  has a closed form, because the leftover tiles are the top of each XCD's block of tile ids.

`check_mapping.py` checks the leftover-set equality for 3898 shape and policy cases (both
orders, one-tile and two-tile) and draws the before and after XCD maps
(`results/mapping.md`). `check_streamk.py` passes all 36 configurations, with no amdgcnas
fallbacks.

### Results

`results/streamk_shapes.md` (v9 order, TFLOPS). "Before" is from `results/pre_xcd_fix/`; each
v13 column is from its own run.

| shape | v13 | v15 before | v15 after | after vs before | after vs v13 |
|---|---|---|---|---|---|
| 4352x4096x8192 | 1046 | 922 | 932 | +1.1% | -10.8% |
| 4352x4352x8192 | 1081 | 1123 | 1129 | +0.5% | +4.5% |
| 4352x4096x4096 | 977 | 636 | 640 | +0.7% | -34.5% |
| 4352x4096x16384 | 923 | 1108 | 1047 | -5.4% | +13.4% |
| 3328x5120x8192 | 996 | 438 | 439 | +0.4% | -55.9% |
| 4096x6144x8192 | 1324 | 1422 | 1444 | +1.5% | +9.1% |
| 3840x4096x8192 | 1436 | 915 | 817 | -10.7% | -43.1% |
| 8192x8448x8192 | 1457 | 1500 | 1505 | +0.3% | +3.3% |
| 8192x7936x8192 | 1584 | 1496 | 1355 | -9.4% | -14.5% |

- **The long-range shapes get about 10% slower.** On 3840x4096x8192 and 8192x7936x8192 the
  programs' ranges are about a tile long and start at drifting k offsets.
  - The fixed v9 order now reproduces the split-order numbers of entry 1 (812 and 1364 there),
    which confirms the two mappings are now identical.
  - With drifting offsets, no mapping gives L2 reuse, and the grouped-once map measures slightly
    lower (`results/rocprof.md`): 10% L2 hits on 3840x4096x8192 against 16% before, and 60%
    against 66% on 8192x7936x8192.
  - Each XCD now works on ~30 neighbouring tiles that need the same A rows, but at ~30 different
    k offsets, so a row's k-slices pass through one L2 at different times.
- **The other shapes are within noise.** K=16384 varies by ±5% between runs (v16 gained 2.6%
  there).
- **rocprof with cold caches agrees:** -11.8% on 4352x4096x8192, +0.9% on 4352x4352x8192 and
  -55.7% on 3328x5120x8192 against v13.

**Two-tile** (`results/two_tile.md`) changes much more:

| shape | v13 | two-tile before | two-tile after |
|---|---|---|---|
| 4352x4096x8192 | 1034 | 913 | 812 |
| 4352x4352x8192 | 1085 | 805 | 833 |
| 4352x4096x16384 | 948 | 949 | 845 |
| 3328x5120x8192 | 991 | 890 | **1136** |
| 4096x6144x8192 | 1327 | 1346 | 1319 |
| 8192x8448x8192 | 1472 | 1465 | 1345 |
| 8192x7936x8192 | 1626 | 1291 | 1166 |

- 3328x5120x8192 under two-tile is now 15% faster than v13. That is the best number on that
  shape so far, and a shape every one-tile version loses badly.
  - Its two-tile ranges are 130 k-steps long, so neighbouring programs' offsets differ by only
    2 k-steps: they are nearly aligned.
  - With neighbouring tiles on one XCD (the fix), that near-alignment turns into L2 reuse.
- Where the offsets drift faster (8 k-steps or more per program), two-tile gets worse, like the
  long-range one-tile shapes.

**Kept anyway.** The mapping is the same in every version that follows. v17 needs it, and so
does any aligned two-tile scheme (the reversed order in `STREAMK_EXPLAINED.md` section 8 keeps
offsets within a fixed lag). The losses above are on a partition whose L2 behaviour is poor
under either mapping.
