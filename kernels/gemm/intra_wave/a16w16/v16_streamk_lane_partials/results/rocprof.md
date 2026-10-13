# rocprofv3 checks (HIP_VISIBLE_DEVICES=2, bf16, v9 tile order, amdgcnas on), after the XCD grouping fix

Before the fix: `pre_xcd_fix/rocprof.md`.

## Kernel time, cold caches

`rocprofv3 --kernel-trace -- python bench.py --version V --M M --N N --K K --dtype bf16 --rocprof`
(1000 launches over rotating 512 MB of inputs), median dispatch duration. The v13 times are from
the earlier runs; the fix does not touch v13.

| shape | v13 (us) | v16 before fix (us) | v16 (us) | v16 vs v13 |
|---|---|---|---|---|
| 4352x4096x8192 | 278.9 | 246.9 | 243.6 | +14.5% |
| 4352x4352x8192 | 279.2 | 244.2 | 244.5 | +14.2% |
| 3328x5120x8192 | 278.4 | 345.6 | 345.2 | -19.3% |
| 3840x4096x8192 | 170.5 | | 307.2 | -44.5% |

## L2 counters

`rocprofv3 --pmc TCC_HIT_sum TCC_MISS_sum TCC_EA0_RDREQ_sum` over 10 launches on the same
inputs (warm), median of launches 3-10, v9 order.

| shape | build | L2 hit rate | memory read requests |
|---|---|---|---|
| 3840x4096x8192 | v13 | 76.4% | 3.65M |
| 3840x4096x8192 | v16 before fix | 13.8% | 14.82M |
| 3840x4096x8192 | v16 | 8.0% | 15.90M |
| 8192x7936x8192 | v13 | 77.8% | 13.91M |
| 8192x7936x8192 | v16 before fix | 66.2% | 21.51M |
| 8192x7936x8192 | v16 | 59.8% | 25.91M |
| 4352x4352x8192 | v13 | 69.9% | 5.67M |
| 4352x4352x8192 | v16 before fix | 64.1% | 6.50M |
| 4352x4352x8192 | v16 | 64.8% | 6.36M |
