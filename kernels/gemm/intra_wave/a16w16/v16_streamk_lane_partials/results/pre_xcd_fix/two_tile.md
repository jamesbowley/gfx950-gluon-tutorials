# One-tile vs two-tile on v16's kernel (HIP_VISIBLE_DEVICES=2, bf16, v9 tile order)

`python v16_streamk_lane_partials/two_tile_check.py`, do_bench TFLOPS. Two-tile results were
correct over 10 launches, and the flags were re-armed afterwards.

| shape | full waves left | v13 | one-tile | two-tile | two-tile correct |
|---|---|---|---|---|---|
| 4352x4096x8192 | 0 | 1034 | 1187 | 915 | yes |
| 4352x4352x8192 | 0 | 1084 | 1222 | 831 | yes |
| 4352x4096x16384 | 0 | 946 | 1246 | 953 | yes |
| 3328x5120x8192 | 0 | 995 | 782 | 914 | yes |
| 4096x6144x8192 | 0 | 1329 | 1458 | 1382 | yes |
| 8192x8448x8192 | 3 | 1469 | 1569 | 1490 | yes |
| 8192x7936x8192 | 2 | 1623 | 1495 | 1321 | yes |
