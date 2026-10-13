# rocprofv3 checks (HIP_VISIBLE_DEVICES=2, bf16, amdgcnas on)

## Kernel time, cold caches, v9 tile order

`rocprofv3 --kernel-trace -- python bench.py --version V --M M --N N --K K --dtype bf16 --rocprof`
(1000 launches over rotating 512 MB of inputs), median dispatch duration. The v13, v15 and v16
columns other than 3840x4096x8192 are from the v15 and v16 `results/rocprof.md`.

| shape | v13 (us) | v16 (us) | v17 (us) | v17 vs v13 |
|---|---|---|---|---|
| 4352x4096x8192 | 278.9 | 246.9 | 243.5 | +14.5% |
| 4352x4352x8192 | 279.2 | 244.2 | 238.0 | +17.3% |
| 3328x5120x8192 | 278.4 | 345.6 | 341.8 | -18.5% |
| 3840x4096x8192 | 170.5 | | 180.3 | -5.4% |

## L2 counters

`rocprofv3 --pmc TCC_HIT_sum TCC_MISS_sum TCC_EA0_RDREQ_sum` over 10 launches on the same
inputs (warm), median of launches 3-10. `PERSISTENT_TILE_ORDER` as given.

| shape | build | L2 hit rate | memory read requests |
|---|---|---|---|
| 3840x4096x8192 | v13, v9 order | 76.4% | 3.65M |
| 3840x4096x8192 | v13, split order | 79.4% | 3.16M |
| 3840x4096x8192 | v16, v9 order | 13.8% | 14.82M |
| 3840x4096x8192 | v17, v9 order | 52.8% | 7.66M |
| 3840x4096x8192 | v17, split order | 76.4% | 3.67M |
| 8192x7936x8192 | v13, v9 order | 77.8% | 13.92M |
| 8192x7936x8192 | v13, split order | 76.4% | 14.91M |
| 8192x7936x8192 | v16, v9 order | 66.2% | 21.51M |
| 8192x7936x8192 | v17, v9 order | 71.7% | 18.05M |
| 8192x7936x8192 | v17, split order | 76.1% | 15.07M |
| 4352x4352x8192 | v13, v9 order | 69.9% | 5.67M |
| 4352x4352x8192 | v13, split order | 76.1% | 4.45M |
| 4352x4352x8192 | v16, v9 order | 64.1% | 6.50M |
| 4352x4352x8192 | v17, v9 order | 64.3% | 6.46M |
| 4352x4352x8192 | v17, split order | 69.1% | 5.45M |
