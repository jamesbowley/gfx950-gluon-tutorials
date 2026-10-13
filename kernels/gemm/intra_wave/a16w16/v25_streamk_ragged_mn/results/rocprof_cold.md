# Cold kernel-only times (rocprofv3 --kernel-trace over bench.py --rocprof: 1000 launches over a rotating 512 MB, padded B and C rows, median dispatch), GPU 0

| shape | version | env | cold kernel us | kernel |
|---|---|---|---|---|
| 8192x8192x8192 | 24 | - | 686.0 | v24_streamk_combined |
| 8192x8192x8192 | 25 | - | 685.5 | v25_streamk_ragged_mn |
| 8192x8192x8192 | 25 | STREAMK_FORCE_EDGE=1 | 680.4 | v25_streamk_ragged_mn |
| 8100x8192x8192 | 25 | - | 681.0 | v25_streamk_ragged_mn |
| 8192x8128x8192 | 25 | - | 680.8 | v25_streamk_ragged_mn |
| 4352x4096x8192 | 24 | - | 206.4 | v24_streamk_combined |
| 4352x4096x8192 | 25 | - | 206.4 | v25_streamk_ragged_mn |
| 4352x4096x8192 | 25 | STREAMK_FORCE_EDGE=1 | 207.0 | v25_streamk_ragged_mn |
| 4300x4000x8192 | 25 | - | 205.9 | v25_streamk_ragged_mn |
| 8192x8448x8192 | 24 | - | 726.8 | v24_streamk_combined |
| 8192x8448x8192 | 25 | STREAMK_FORCE_EDGE=1 | 724.6 | v25_streamk_ragged_mn |
| 8150x8400x8192 | 25 | - | 726.1 | v25_streamk_ragged_mn |
| 8150x8448x8192 | 25 | - | 724.5 | v25_streamk_ragged_mn |
| 8192x8400x8192 | 25 | - | 725.5 | v25_streamk_ragged_mn |
| 4096x6144x8192 | 24 | - | 275.3 | v24_streamk_combined |
| 4096x6144x8192 | 25 | STREAMK_FORCE_EDGE=1 | 274.6 | v25_streamk_ragged_mn |
| 4000x6080x8192 | 25 | - | 273.8 | v25_streamk_ragged_mn |
| 4096x4096x8192 | 9 | - | 176.1 | v9_beyond_hotloop |
| 4096x4096x8192 | 9r | - | 176.7 | v9_ragged |
| 4096x4096x8192 | 9r | V9_FORCE_EDGE=1 | 175.4 | v9_ragged |
| 4000x4000x8192 | 9r | - | 175.0 | v9_ragged |
| 4000x4000x8192 | 25 | - | 175.0 | v9_ragged |
| 4096x4096x8192 | 9t | - | 176.6 | v9_tile |
| 4000x4000x8192 | 9t | - | 174.2 | v9_tile |
