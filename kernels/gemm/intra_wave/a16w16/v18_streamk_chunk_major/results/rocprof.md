# rocprofv3 checks (HIP_VISIBLE_DEVICES=2, bf16, v9 tile order, amdgcnas on)

## Kernel time, cold caches

`rocprofv3 --kernel-trace -- python bench.py --version V --M M --N N --K K --dtype bf16 --rocprof`
(1000 launches over rotating 512 MB of inputs), median dispatch duration. The v13 and v17
times are from `../../v17_streamk_tile_aligned/results/rocprof.md` (v17 after the XCD grouping
fix).

| shape | v13 (us) | v17 (us) | v18 (us) | v18 vs v17 | v18 vs v13 |
|---|---|---|---|---|---|
| 4352x4096x8192 | 278.9 | 237.0 | 245.3 | -3.4% | +13.7% |
| 4352x4352x8192 | 279.2 | 233.8 | 231.7 | +0.9% | +20.5% |
| 3328x5120x8192 | 278.4 | 340.1 | 350.4 | -2.9% | -20.5% |
| 3840x4096x8192 | 170.5 | 174.7 | 172.8 | +1.1% | -1.3% |

## L2 counters

`rocprofv3 --pmc TCC_HIT_sum TCC_MISS_sum TCC_EA0_RDREQ_sum` over 10 launches on the same
inputs (warm), median of launches 3-10. Whole-kernel counts, so the persistent phase is included.

| shape | build | L2 hit rate | L2 requests | memory read requests |
|---|---|---|---|---|
| 4352x4096x8192 | v17 | 67.8% | 20.46M | 5.34M |
| 4352x4096x8192 | v18 | 67.0% | 20.48M | 5.50M |
| 4352x4096x16384 | v17 | 64.1% | 38.29M | 12.49M |
| 4352x4096x16384 | v18 | 64.3% | 38.31M | 12.41M |
| 8192x8448x8192 | v17 | 75.8% | 73.35M | 15.78M |
| 8192x8448x8192 | v18 | 75.4% | 73.37M | 16.09M |
| 3328x5120x8192 | v17 | 68.9% | 19.72M | 4.84M |
| 3328x5120x8192 | v18 | 68.7% | 19.72M | 4.87M |
