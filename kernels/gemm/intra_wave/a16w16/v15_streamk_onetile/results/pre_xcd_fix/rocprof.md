# rocprofv3 checks (HIP_VISIBLE_DEVICES=2, bf16, v9 tile order, amdgcnas on)

## Kernel time, cold caches

`rocprofv3 --kernel-trace --stats -- python bench.py --version V --M M --N N --K K --dtype bf16 --rocprof`
(1000 launches over rotating 512 MB of inputs). Median dispatch duration, and the do_bench time
from `streamk_shapes.md` for comparison.

| shape | v13 rocprof (us) | v13 do_bench (us) | v15 rocprof (us) | v15 do_bench (us) | v15 vs v13, rocprof |
|---|---|---|---|---|---|
| 4352x4096x8192 | 278.9 | 280.7 | 319.7 | 316.7 | -12.8% |
| 4352x4352x8192 | 279.2 | 286.8 | 282.2 | 278.0 | -1.1% |
| 3328x5120x8192 | 278.4 | 280.5 | 627.7 | 637.9 | -55.6% |

## L2 counters

`rocprofv3 --pmc TCC_HIT_sum TCC_MISS_sum TCC_EA0_RDREQ_sum` over 10 launches on the same
inputs (warm), median of launches 3-10.

| shape | version | L2 hit rate | L2 requests | L2 misses | memory read requests |
|---|---|---|---|---|---|
| 3840x4096x8192 | v13 | 76.4% | 16.5M | 3.89M | 3.65M |
| 3840x4096x8192 | v15 | 15.7% | 19.1M | 16.13M | 14.88M |
| 4352x4096x8192 | v13 | 72.8% | 18.7M | 5.09M | 4.81M |
| 4352x4096x8192 | v15 | 67.2% | 21.1M | 6.91M | 5.62M |
