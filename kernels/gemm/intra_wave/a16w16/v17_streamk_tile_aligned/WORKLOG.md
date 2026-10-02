# v17_streamk_tile_aligned work log

One entry per change: what changed and what it did to performance and spills. Results come
from `HIP_VISIBLE_DEVICES=2 python regress.py ...` (bf16 unless noted), run from `a16w16/`
under the published v10+ stack: Triton v2.2 + deepbind shim on `PYTHONPATH`,
`LLVM_PASS_PLUGIN_PATH=plugins/llir_scheduler/libLlirSched.so`, `TRITON_AMDGCNAS_PLUGIN=1`.
Raw tables are in `results/`. One k-step is about 1.21 us.

## 1. Tile-aligned cuts (2026-09-30)

v16 with one change: the stream-K partition. The host policy
(`STREAMK_TILES = total_tiles % 256`), the pipeline, the lane-contiguous partials and the
endings are unchanged.

- **v16** (TensorAtlas) laid the stream-K tiles' k-step pairs end to end and cut the line
  into 256 equal ranges. A program's range can start anywhere in a tile and cross into the
  next, so the programs of an XCD run at different k offsets.
- **v17** cuts inside tiles only:
  - tile `t` gets `256 // STREAMK_TILES` consecutive programs, the first
    `256 % STREAMK_TILES` tiles one more, and at most one per pair of k-steps (extra
    programs idle);
  - each tile's pairs are split evenly among its programs (`streamk_chunk_of`).

  Tiles with the same program count have the same chunk boundaries, so their chunk `j` covers
  the same k-steps at the same time. Every program runs exactly one segment. Chunk 0 owns the
  tile, and its peers are the next `n - 1` spids.

Six of the nine test shapes already had tile-aligned ranges under v16, so v17 gives them the
same partition; they are the control. It changes three:
- **3840x4096x8192:** 224 tiles whole on one program each, 16 split in two;
- **8192x7936x8192:** 192 whole, 32 split in two;
- **4352x4352x8192:** 25 tiles in 8 chunks and 8 tiles in 7, with no boundary crossings.

The lockstep model (`streamk_predict.py`, `results/predict.md`) predicts no gain on these
three, because it assumes L2 reuse is unaffected.

**Registers.**
- The segment `while` loop and `streamk_owner_of` are gone.
- VGPRs fall from 512 to 492, with **no VGPR spills and no scratch ops** (v16: 14 VGPR
  spills, 43 scratch ops). 2-4 SGPR spills remain.
- Regression shapes (`STREAMK_TILES = 0`): within 0.7% of v16.

**Correctness.**
- `check_streamk.py`: all 36 configurations pass, and the flags are re-armed
  (`results/check_streamk.md`).
- The capped case is checked too: 256x65792x8192 (1 tile, 64 working programs) and
  256x67328x4096 (7 tiles of 32 programs, 32 idle).

**Tooling fix (amdgcnas).**
- On 3328x5120x8192 (every program has work), LLVM placed an early-exit block, with its own
  `s_endpgm`, ahead of other blocks. The peephole took the first `s_endpgm` as the end of the
  program, lost the blocks after it (`KeyError('.LBB0_49')`), and fell back to un-optimised
  assembly.
- `plugins/amdgcnas/amdgcnas_ext.py` now ends the program at the last `s_endpgm`, and gives
  earlier ones no successor in the CFG.
- v13 and v16 compile to byte-identical assembly with the fix (4352x4096x8192 and
  3328x5120x8192), and no v17 build falls back any more. The plugin's cache key hashes this
  file, so the old fallback builds are not reused.

### Results

> Measured with the double XCD grouping of the stream-K tiles (v9 order); see entry 2
> for the fix and the corrected numbers. The old raw tables are in `results/pre_xcd_fix/`.

TFLOPS (`results/streamk_shapes.md`, `results/streamk_shapes_split_order.md`); each order's
deltas are against v13 in the same order.

| shape | v13 (v9) | v16 (v9) | v17 (v9) | v17 vs v13 (v9) | v13 (split) | v17 (split) | v17 vs v13 (split) |
|---|---|---|---|---|---|---|---|
| 4352x4096x8192 | 1033 | 1175 | 1203 | +16.5% | 975 | 1197 | +22.8% |
| 4352x4352x8192 | 1085 | 1256 | 1297 | +19.5% | 1041 | 1341 | +28.8% |
| 4352x4096x4096 | 975 | 910 | 935 | -4.1% | 938 | 940 | +0.2% |
| 4352x4096x16384 | 894 | 1172 | 1199 | +34.0% | 863 | 1189 | +37.8% |
| 3328x5120x8192 | 1003 | 792 | 800 | -20.2% | 1000 | 802 | -19.7% |
| 4096x6144x8192 | 1320 | 1481 | 1500 | +13.7% | 1271 | 1520 | +19.7% |
| 3840x4096x8192 | 1436 | 893 | 1434 | -0.2% | 1455 | 1475 | +1.4% |
| 8192x8448x8192 | 1472 | 1571 | 1565 | +6.3% | 1472 | 1579 | +7.3% |
| 8192x7936x8192 | 1579 | 1485 | 1508 | -4.5% | 1592 | 1604 | +0.8% |

rocprof with cold caches, v9 order (`results/rocprof.md`): +14.5% and +17.3% on
4352x4096x8192 and 4352x4352x8192, -18.5% on 3328x5120x8192, and -5.4% on 3840x4096x8192
(180 against 171 us).

With bias (`results/streamk_shapes_bias.md`), v17 against v16 gains 1-4% on the control
shapes and 59% on 3840x4096x8192.

What the numbers say:

1. **Aligned cuts remove the long-range loss.** 3840x4096x8192 goes from -38% (v16) to level
   with v13. The L2 counters show where it comes from (`results/rocprof.md`):

   | build | L2 hit rate | memory reads |
   |---|---|---|
   | v16, v9 order | 14% | 14.8M |
   | v17, v9 order | 53% | 7.7M |
   | v17, split order | 76% | 3.7M |
   | v13 (data-parallel) | 76% | 3.7M |

2. **The rest of the L2 gap is XCD locality, set by the tile order.**
   - Stream-K tile `t` is `persistent_tile_id(total_full_tiles + t)`. Under v9 order that
     spreads the tiles of consecutive spids, which are on one XCD, across the whole matrix;
     under split order they are contiguous.
   - v17 in split order matches data-parallel's L2 hit rate on all three changed shapes, and
     it matches or beats v13 on 8 of 9 shapes in that order.
   - v13's persistent phase prefers v9 order, though (split costs v13 2-6%). So the best
     combination is not available from one order: v17 split against v13 v9 is +16% and +24%
     on the two main shapes, but still -3.6% at K=4096.
3. **The control shapes gain 1-4% from the register change.** Their partition is unchanged, but
   the build has no spills and 20 fewer VGPRs.
4. **Left as before:**
   - 3328x5120x8192, with 63 serial reads per owner: -20%;
   - K=4096, where the data-parallel last wave takes only about 39 us.

   Both are collection costs, not L2.

### Next

- **v18:** map the stream-K tiles contiguously per XCD under the v9 order too, so the
  persistent phase keeps v9 order and the tail gets split order's locality.
- **v19:** cap the split count (fewer programs per tile), for the collection-bound cases
  (3328x5120x8192, K=4096).
- **Then** a host rule for when to skip stream-K.

## 2. XCD grouping fix (2026-10-01)

The stream-K tiles were grouped by XCD twice under v9 order: once in `spid`, and again in the
tile lookup `persistent_tile_id(total_full_tiles + t)`. Entry 1 found this as the remaining L2
gap in v9 order (finding 2). `streamk_tile_id` now applies the grouping once: the same
leftover tiles, in ascending tile-id order, so consecutive spids get neighbouring tiles. The
persistent phase keeps v9 order. What went wrong and why is in v15's WORKLOG, entry 3.

**Correctness.** `check_streamk.py` passes all 36 configurations, plus the capped shapes
256x65792x8192 and 256x67328x4096, with no amdgcnas fallbacks (`results/check_streamk.md`).
A split-order spot check (3840x4096x8192 at 1468 TFLOPS, 4352x4352x8192 at 1336) matches
entry 1's split-order run, as expected.

`results/streamk_shapes.md` (v9 order, TFLOPS), against `results/pre_xcd_fix/`:

| shape | v13 | v17 before | v17 after | after vs before | after vs v13 |
|---|---|---|---|---|---|
| 4352x4096x8192 | 1038 | 1203 | 1211 | +0.7% | +16.7% |
| 4352x4352x8192 | 1076 | 1297 | 1322 | +1.9% | +22.9% |
| 4352x4096x4096 | 972 | 935 | 949 | +1.5% | -2.4% |
| 4352x4096x16384 | 959 | 1199 | 1252 | +4.5% | +30.6% |
| 3328x5120x8192 | 1001 | 800 | 793 | -0.9% | -20.8% |
| 4096x6144x8192 | 1329 | 1500 | 1510 | +0.6% | +13.6% |
| 3840x4096x8192 | 1409 | 1434 | 1477 | +3.0% | +4.8% |
| 8192x8448x8192 | 1466 | 1565 | 1569 | +0.2% | +7.0% |
| 8192x7936x8192 | 1605 | 1508 | 1602 | +6.2% | -0.2% |

- **v17 now gets split order's stream-K locality with v9 order's persistent phase.**
  - The L2 hit rate on 3840x4096x8192 goes from 53% to 76%, data-parallel's level, and
    8192x7936x8192 from 72% to 77% (`results/rocprof.md`).
  - 3840x4096x8192 moves from level with v13 to +4.8%, and 8192x7936x8192 from -4.5% to level.
- **rocprof with cold caches:**
  - +17.7% on 4352x4096x8192 and +19.4% on 4352x4352x8192 (before: +14.5%, +17.3%);
  - -18.1% on 3328x5120x8192;
  - -2.4% on 3840x4096x8192 (before: -5.4%).
- **With bias**, the long-range shapes gain the same way: 3840x4096x8192 from 1414 to 1458, and
  8192x7936x8192 from 1521 to 1595 (`results/streamk_shapes_bias.md`).
- **Still losing:** 3328x5120x8192 (63 serial reads per owner) and K=4096 (-2.4%). Both are
  collection costs.

### Next

v18: chunk-major mapping. Chunk j of neighbouring tiles goes on the same XCD, so programs
reading the same k-steps share an L2, with a rotating owner so the fixups are spread over the
XCDs.

## 3. Launch heuristic: data-parallel when any tile would stay whole (2026-10-01)

Not a code change.

**The rule.** A stream-K split only shortens the tail if every leftover tile is split, so that
the longest chunk is shorter than a tile. With the tile-aligned split that means
`S = total_tiles % 256 <= 128`.

**Why it holds.** For `S > 128`, most tiles get one program and run whole from k-step 0, exactly
as in data-parallel, so they set the tail's length.
- If the split tiles finish earlier, the kernel takes exactly as long as data-parallel.
- If they finished later, it would be slower.
- Their extra traffic (A/B loads, partial stores and reads) runs alongside the whole tiles
  that make up the critical path, so it can only cost time.

**Measured.** v17 is within noise of v13 for S = 192-255 (-0.7% to +2.9% with do_bench, -1.5%
cold-cache at S = 240), against +14% at S = 128, where every tile is split in two (entry 2;
v19 `results/sweep_summary.md`). So use data-parallel for `S > 128`, and stream-K only below.

**Energy and power** (`power_check.py`, `results/power.md`: 15 s of back-to-back launches,
amd-smi energy counter):

| shape | S | v13 power | v17 power | v13 energy per launch | v17 energy per launch |
|---|---|---|---|---|---|
| 4096x7936x8192 | 240 | 1412 W | 1412 W | 456.8 mJ | 461.4 mJ (+1.0%) |
| 4352x4096x8192 | 16 | 1157 W | 1370 W | 311.5 mJ | 305.6 mJ (-1.9%) |

- **S = 240:** both kernels run at the board's power limit (about 1.4 kW). Spreading the
  leftover tiles over the idle CUs lowers neither power nor energy; v17 is about 1% slower
  and uses about 1% more energy per launch. That supports the rule above.
- **S = 16:** v13's last wave keeps 240 CUs idle, so its average power is lower. v17 keeps every
  CU busy and draws 18% more power, but it finishes 17% sooner, so its energy per launch is 2%
  lower.

So spreading work raises power draw, and it saves energy only when it also shortens the kernel.
