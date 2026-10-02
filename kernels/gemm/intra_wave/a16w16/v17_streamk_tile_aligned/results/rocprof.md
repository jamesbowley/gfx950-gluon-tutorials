# rocprofv3 checks (HIP_VISIBLE_DEVICES=2, bf16, amdgcnas on), after the XCD grouping fix

Before the fix: `pre_xcd_fix/rocprof.md`.

## Kernel time, cold caches, v9 tile order

`rocprofv3 --kernel-trace -- python bench.py --version V --M M --N N --K K --dtype bf16 --rocprof`
(1000 launches over rotating 512 MB of inputs), median dispatch duration. The v13 times are from
the earlier runs; the fix does not touch v13.

| shape | v13 (us) | v17 before fix (us) | v17 (us) | v17 vs v13 |
|---|---|---|---|---|
| 4352x4096x8192 | 278.9 | 243.5 | 237.0 | +17.7% |
| 4352x4352x8192 | 279.2 | 238.0 | 233.8 | +19.4% |
| 3328x5120x8192 | 278.4 | 341.8 | 340.1 | -18.1% |
| 3840x4096x8192 | 170.5 | 180.3 | 174.7 | -2.4% |

## L2 counters

`rocprofv3 --pmc TCC_HIT_sum TCC_MISS_sum TCC_EA0_RDREQ_sum` over 10 launches on the same
inputs (warm), median of launches 3-10. v9 order unless noted; the split-order rows are from
before the fix (the fix does not change split order).

| shape | build | L2 hit rate | memory read requests |
|---|---|---|---|
| 3840x4096x8192 | v13 | 76.4% | 3.65M |
| 3840x4096x8192 | v17 before fix | 52.8% | 7.66M |
| 3840x4096x8192 | v17 | 76.4% | 3.68M |
| 3840x4096x8192 | v17, split order | 76.4% | 3.67M |
| 8192x7936x8192 | v13 | 77.8% | 13.91M |
| 8192x7936x8192 | v17 before fix | 71.7% | 18.05M |
| 8192x7936x8192 | v17 | 76.6% | 14.71M |
| 8192x7936x8192 | v17, split order | 76.1% | 15.07M |
| 4352x4352x8192 | v13 | 69.9% | 5.67M |
| 4352x4352x8192 | v17 before fix | 64.3% | 6.46M |
| 4352x4352x8192 | v17 | 65.7% | 6.17M |
| 4352x4352x8192 | v17, split order | 69.1% | 5.45M |
