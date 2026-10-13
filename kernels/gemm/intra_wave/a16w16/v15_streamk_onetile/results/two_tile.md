# One-tile vs two-tile on v15's kernel (HIP_VISIBLE_DEVICES=2, bf16, v9 tile order, XCD grouping fix)

`python v15_streamk_onetile/two_tile_check.py`, do_bench TFLOPS. Two-tile results correct over 10 launches, flags re-armed. Before the fix: `pre_xcd_fix/two_tile.md`.

| shape | full waves left | v13 | one-tile | two-tile | two-tile correct |
|---|---|---|---|---|---|
| 4352x4096x8192 | 0 | 1034 | 941 | 812 | yes |
| 4352x4352x8192 | 0 | 1085 | 1073 | 833 | yes |
| 4352x4096x16384 | 0 | 948 | 1105 | 845 | yes |
| 3328x5120x8192 | 0 | 991 | 432 | 1136 | yes |
| 4096x6144x8192 | 0 | 1327 | 1435 | 1319 | yes |
| 8192x8448x8192 | 3 | 1472 | 1498 | 1345 | yes |
| 8192x7936x8192 | 2 | 1626 | 1367 | 1166 | yes |
