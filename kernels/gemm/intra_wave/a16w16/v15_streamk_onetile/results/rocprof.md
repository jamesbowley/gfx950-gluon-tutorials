# rocprofv3 checks (HIP_VISIBLE_DEVICES=2, bf16, v9 tile order, amdgcnas on), after the XCD grouping fix

Before the fix: `pre_xcd_fix/rocprof.md`.

## Kernel time, cold caches

`rocprofv3 --kernel-trace -- python bench.py --version V --M M --N N --K K --dtype bf16 --rocprof`
(1000 launches over rotating 512 MB of inputs), median dispatch duration. The v13 times are from
the earlier runs (`pre_xcd_fix/rocprof.md` and v17's `results/pre_xcd_fix/rocprof.md`); the fix
does not touch v13.

| shape | v13 (us) | v15 before fix (us) | v15 (us) | v15 vs v13 |
|---|---|---|---|---|
| 4352x4096x8192 | 278.9 | 319.7 | 316.3 | -11.8% |
| 4352x4352x8192 | 279.2 | 282.2 | 276.8 | +0.9% |
| 3328x5120x8192 | 278.4 | 627.7 | 628.7 | -55.7% |
| 3840x4096x8192 | 170.5 | | 309.7 | -45.0% |

## L2 counters

`rocprofv3 --pmc TCC_HIT_sum TCC_MISS_sum TCC_EA0_RDREQ_sum` over 10 launches on the same
inputs (warm), median of launches 3-10, v9 order.

| shape | build | L2 hit rate | memory read requests |
|---|---|---|---|
| 3840x4096x8192 | v13 | 76.4% | 3.65M |
| 3840x4096x8192 | v15 before fix | 15.7% | 14.88M |
| 3840x4096x8192 | v15 | 10.4% | 15.89M |
| 8192x7936x8192 | v13 | 77.8% | 13.91M |
| 8192x7936x8192 | v15 | 60.3% | 25.78M |
| 4352x4352x8192 | v13 | 69.9% | 5.67M |
| 4352x4352x8192 | v15 | 65.6% | 6.34M |
