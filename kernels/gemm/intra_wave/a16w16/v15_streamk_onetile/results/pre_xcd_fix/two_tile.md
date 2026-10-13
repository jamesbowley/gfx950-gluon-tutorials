# One-tile vs two-tile on v15's kernel (HIP_VISIBLE_DEVICES=2, bf16, v9 tile order)

`python v15_streamk_onetile/two_tile_check.py`, do_bench TFLOPS. The 8192x8448x8192,
8192x7936x8192 and 4352x4352x8192 rows are from a second run: the first run's
8192x8448x8192 row had v13 at 1112, well below its usual 1461, so it was noise.

| shape | full waves left | v13 | one-tile | two-tile | two-tile correct (10 launches, flags re-armed) |
|---|---|---|---|---|---|
| 4352x4096x8192 | 0 | 1035 | 928 | 913 | yes |
| 4352x4352x8192 | 0 | 1082 | 1100 | 805 | yes |
| 4352x4096x16384 | 0 | 958 | 1101 | 949 | yes |
| 3328x5120x8192 | 0 | 999 | 436 | 890 | yes |
| 4096x6144x8192 | 0 | 1322 | 1430 | 1346 | yes |
| 8192x8448x8192 | 3 | 1461 | 1488 | 1465 | yes |
| 8192x7936x8192 | 2 | 1570 | 1489 | 1291 | yes |

L2 counters, 4352x4352x8192, 10 launches each on the same inputs, median of launches 3-10:

| policy | L2 hit rate | L2 misses | memory read requests |
|---|---|---|---|
| one-tile | 65.0% | 7.74M | 6.47M |
| two-tile | 11.4% | 19.86M | 18.59M |
