# v16_streamk_lane_partials work log

One entry per change: what changed and what it did to performance and spills. Results come
from `HIP_VISIBLE_DEVICES=2 python regress.py ...` (bf16 unless noted), run from `a16w16/`
under the published v10+ stack: Triton v2.2 + deepbind shim on `PYTHONPATH`,
`LLVM_PASS_PLUGIN_PATH=plugins/llir_scheduler/libLlirSched.so`, `TRITON_AMDGCNAS_PLUGIN=1`.
Raw tables are in `results/`. One k-step is about 1.21 us.

## 1. Lane-contiguous partials (2026-09-30)

v15 with one change: the address map of the fp32 partials in the workspace `P`
(`lane_contiguous_offsets`, used for both the contributor's store and the owner's load).
C, the algorithm and everything else are unchanged.

**What changes.** Partials are still stored straight from the accumulator's MFMA registers,
with no layout conversion. Only the addresses differ:

- v15 wrote them row-major (`row * 128 + col`). The 64 lanes of one store instruction hold a
  16 x 16 block, so the instruction wrote 64 bytes into each of 16 rows: 16 half cache lines.
- v16 lays each 128x128 quadrant out as `[16 register vectors][4 warps][64 lanes][4 fp32]`.
  Lane `l` writes bytes `16l .. 16l+15` of a 1 KiB block, so one instruction covers 8 whole
  cache lines.

The owner holds the same register layout and loads with the same map, so every value comes
back to the lane and register it left from.

The map is the one measured in `experiments/streamk_costs` (`P_LAYOUT=1`). Its column term
has to stay `c +` terms constant over 4 columns, or the accesses stop being 16 bytes wide.

**Correctness and registers.**
- `check_streamk.py`: all 36 configurations pass (9 shapes, 2 tile orders, with and without
  bias, 30 rotating launches each), and the flags are re-armed.
- The build is the same as v15's:
  - 512 VGPRs, 14 VGPR spills, 21 SGPR spills, and the same scratch ops (none in the
    persistent or K loops);
  - 64 16-byte partial stores and 192 16-byte partial loads, with no single-dword accesses.
- Regression shapes (`STREAMK_TILES = 0`, `results/regression_shapes.md`): within 1% of v15.

### Results, stream-K shapes

> Measured with the double XCD grouping of the stream-K tiles (v9 order); see entry 2
> for the fix and the corrected numbers. The old raw tables are in `results/pre_xcd_fix/`.

`results/streamk_shapes.md`, v9 tile order, TFLOPS. Tails are the kernel time minus v13's
full-wave time (v15 `results/full_wave_refs.md`). The model is v15's `streamk_predict.py`
with lane-contiguous costs.

| shape | v13 | v15 | v16 | v16 vs v13 | v13 tail (us) | v15 tail | v16 tail | lane model |
|---|---|---|---|---|---|---|---|---|
| 4352x4096x8192 | 1042 | 927 | 1166 | +11.9% | 96 | 131 | 67 | 53 |
| 4352x4352x8192 | 1078 | 1104 | 1272 | +18.1% | 104 | 97 | 60 | 46 |
| 4352x4096x4096 | 970 | 632 | 918 | -5.4% | 39 | 119 | 47 | 48 |
| 4352x4096x16384 | 894 | 1032 | 1167 | +30.5% | 271 | 184 | 118 | 63 |
| 3328x5120x8192 | 1001 | 439 | 792 | -20.9% | 95 | 452 | 169 | 162 |
| 4096x6144x8192 | 1328 | 1456 | 1485 | +11.8% | 127 | 99 | 94 | 87 |
| 3840x4096x8192 | 1423 | 889 | 918 | -35.5% | 181 | 290 | 281 | 155 |
| 8192x8448x8192 | 1468 | 1515 | 1571 | +7.0% | 94 | 70 | 44 | 44 |
| 8192x7936x8192 | 1595 | 1502 | 1517 | -4.9% | 153 | 195 | 188 | 145 |

v13's 4352x4096x16384 figure is low in this run: 894 against 944 in v15's run. Against 944,
v16 is +24%.

rocprof with cold caches (`results/rocprof.md`) agrees: +13.0% and +14.3% on 4352x4096x8192
and 4352x4352x8192, and -19.4% on 3328x5120x8192. With bias
(`results/streamk_shapes_bias.md`), v16 against v15 is +28% and +14% on the first two
shapes, and +80% on 3328x5120x8192.

What the numbers say:

1. **The fixup-bound tails shrank to about the lane model.**
   - 4352x4096x8192: 131 to 67 us (model 53);
   - 3328x5120x8192: 452 to 169 us (model 162);
   - 4352x4096x4096: 119 to 47 us (model 48).

   That makes v16 the first version to beat data-parallel on the main shape: its tail is
   67 us against v13's 96.
2. **Where the fixup still dominates, one-tile still loses.** 3328x5120x8192 (63 serial reads
   per owner) is 21% below v13. At K=4096 the data-parallel last wave takes only 39 us, and
   v16 is 5% below. The serial collection, and not the partial layout, is now the limit: the
   remaining split options of section 9 (fewer programs per tile, reduce-scatter) address it.
3. **The long-range shapes barely moved:** 3840x4096x8192 and 8192x7936x8192 gained 1-3%. Their
   loss is L2 reuse (v15 entry 1, finding 3), which the partial layout does not touch.
4. **The model still underestimates K=16384** (118 us against 63). It uses the K=8192 k-step
   time, and at K=16384 a k-step is slower (1.42 us per k-step in `experiments/streamk_costs`).
5. **Two-tile barely changes** (`results/two_tile.md`): 915 and 831 TFLOPS on 4352x4096x8192 and
   4352x4352x8192, against 913 and 805 in v15. Its tiles have at most one peer, so the
   layout hardly matters; it stays below both one-tile and v13.

### Next

- Restore L2 reuse on long ranges: cuts aligned across tiles, so the programs of an XCD read
  the same k-slices. This gates both two-tile and the 3840x4096x8192 and 8192x7936x8192 losses.
- Shorten the serial collection: fewer programs per tile, or reduce-scatter.
- A host rule for when to skip stream-K: v13 still wins at K=4096, with very few leftover
  tiles, and with nearly full last waves.

## 2. XCD grouping fix (2026-10-01)

The stream-K tiles were grouped by XCD twice under v9 order. Stream-K numbers its programs
with the XCD-grouped `spid`, and the tile lookup `persistent_tile_id(total_full_tiles + t)`
grouped again. `streamk_tile_id` now applies the grouping once: the same leftover tiles, in
ascending tile-id order. What went wrong and why is in v15's WORKLOG, entry 3. The split-order
results are unchanged by the fix. `check_streamk.py` passes all 36 configurations.

`results/streamk_shapes.md` (v9 order, TFLOPS), against `results/pre_xcd_fix/`:

| shape | v13 | v16 before | v16 after | after vs before | after vs v13 |
|---|---|---|---|---|---|
| 4352x4096x8192 | 1034 | 1166 | 1206 | +3.4% | +16.6% |
| 4352x4352x8192 | 1074 | 1272 | 1247 | -2.0% | +16.1% |
| 4352x4096x4096 | 980 | 918 | 928 | +1.1% | -5.3% |
| 4352x4096x16384 | 962 | 1167 | 1197 | +2.6% | +24.3% |
| 3328x5120x8192 | 1003 | 792 | 780 | -1.4% | -22.2% |
| 4096x6144x8192 | 1311 | 1485 | 1482 | -0.2% | +13.1% |
| 3840x4096x8192 | 1432 | 918 | 818 | -10.8% | -42.9% |
| 8192x8448x8192 | 1473 | 1571 | 1570 | -0.1% | +6.6% |
| 8192x7936x8192 | 1586 | 1517 | 1370 | -9.7% | -13.6% |

- **As in v15, the long-range shapes lose about 10%; the rest is within noise.** The fix makes v9
  order identical to split order, and this partition's drifting k offsets get no L2 reuse under
  either mapping. The L2 hit rates (`results/rocprof.md`) go from 14% to 8% on
  3840x4096x8192, and from 66% to 60% on 8192x7936x8192.
- **rocprof with cold caches is unchanged on the main shapes:** +14.5%, +14.2% and -19.3% against
  v13.
- **Two-tile** (`results/two_tile.md`): 3328x5120x8192 goes from 914 to 1160 TFLOPS, 17% above
  v13's 996, because its offsets drift by only 2 k-steps per program. The other shapes get
  6-12% slower, for example 4352x4096x8192 from 915 to 801. See v15 entry 3.
