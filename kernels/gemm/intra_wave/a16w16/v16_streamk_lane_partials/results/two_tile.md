# One-tile vs two-tile on v16's kernel (HIP_VISIBLE_DEVICES=2, bf16, v9 tile order, XCD grouping fix)

`python v16_streamk_lane_partials/two_tile_check.py`, do_bench TFLOPS. Two-tile results correct over 10 launches, flags re-armed. Before the fix: `pre_xcd_fix/two_tile.md`.

| shape | full waves left | v13 | one-tile | two-tile | two-tile correct |
|---|---|---|---|---|---|
| 4352x4096x8192 | 0 | 1034 | 1205 | 801 | yes |
| 4352x4352x8192 | 0 | 1086 | 1218 | 832 | yes |
| 4352x4096x16384 | 0 | 959 | 1279 | 848 | yes |
| 3328x5120x8192 | 0 | 996 | 781 | 1160 | yes |
| 4096x6144x8192 | 0 | 1332 | 1468 | 1347 | yes |
| 8192x8448x8192 | 3 | 1469 | 1568 | 1349 | yes |
| 8192x7936x8192 | 2 | 1625 | 1360 | 1162 | yes |
