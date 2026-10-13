# v25 against v24 on stored cases (spot check)

rocprof kernel times, us (1000 launches over rotating inputs, median), HIP device 0. Sweep F cases: 4096x8192xK, GEMM_TILE_LIMIT = 256 + S, padded B and C; AITER cases: contiguous, no limit. `stored`: the v24 row in sweep_f.jsonl / test.jsonl; v24 and v25 timed now, alternating which goes first. Edge flags unset (all shapes aligned).

| source | shape | S | config | n | stored v24 | v24 now | v25 now | v25 / v24 now | v25 / stored |
|---|---|---|---|---|---|---|---|---|---|
| sweep F | 4096x8192x16384 | 2 | v24 rs | 64 | 364.2 | 364.3 | 364.2 | 1.000 | 1.000 |
| sweep F | 4096x8192x16384 | 8 | v24 rs | 4 | 425.5 | 426.9 | 426.2 | 0.998 | 1.002 |
| sweep F | 4096x8192x8192 | 38 | v24 rs | 6 | 222.8 | 222.7 | 221.9 | 0.996 | 0.996 |
| sweep F | 4096x8192x16384 | 2 | v24 owner | 64 | 518.6 | 517.6 | 517.3 | 0.999 | 0.997 |
| sweep F | 4096x8192x16384 | 8 | v24 owner | 4 | 430.4 | 430.2 | 429.8 | 0.999 | 0.999 |
| sweep F | 4096x8192x8192 | 38 | v24 owner | 6 | 228.8 | 229.8 | 229.0 | 0.997 | 1.001 |
| sweep F | 4096x8192x2048 | 6 | v24 two_tile | - | 71.4 | 71.9 | 71.9 | 1.000 | 1.008 |
| sweep F | 4096x8192x512 | 96 | v24 two_tile | - | 48.1 | 48.1 | 48.5 | 1.007 | 1.007 |
| sweep F | 4096x8192x8192 | 192 | v24 spread | - | 321.6 | 321.5 | 321.3 | 0.999 | 0.999 |
| sweep F | 4096x8192x32768 | 240 | v24 spread | - | 1343.8 | 1340.8 | 1339.1 | 0.999 | 0.996 |
| AITER | 512x6144x2048 | 48 | v24 capped | - | 36.1 | 36.1 | 36.3 | 1.007 | 1.006 |
| AITER | 4096x14336x8192 | 128 | v24 capped | - | 610.7 | 611.6 | 611.4 | 1.000 | 1.001 |
| AITER | 2048x5120x3200 | 160 | v24 capped | - | 63.8 | 63.9 | 63.8 | 0.999 | 1.001 |
| AITER | 1024x1024x5120 | 16 | v24 rs | 3 | 49.7 | 49.8 | 49.6 | 0.997 | 0.999 |
| AITER | 512x5120x3200 | 40 | v24 rs | 1 | 63.8 | 63.7 | 63.6 | 0.998 | 0.997 |
| AITER | 2048x10240x8192 | 64 | v24 rs | 1 | 294.1 | 294.4 | 293.4 | 0.996 | 0.998 |
| AITER | 512x16384x8192 | 128 | v24 rs | 2 | 121.9 | 120.3 | 121.8 | 1.012 | 0.999 |
| AITER | 1280x1024x4096 | 20 | v24 owner | 2 | 51.4 | 51.0 | 51.6 | 1.011 | 1.005 |
| AITER | 512x7168x8192 | 56 | v24 owner | 2 | 86.4 | 85.6 | 86.0 | 1.005 | 0.995 |
| AITER | 4096x6144x4608 | 128 | v24 owner | 1 | 179.7 | 179.0 | 179.4 | 1.002 | 0.998 |

v25 / v24 timed now: median 0.999, range 0.996-1.012, 18 of 20 within 1%.
