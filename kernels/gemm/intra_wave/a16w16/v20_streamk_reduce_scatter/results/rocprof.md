# rocprofv3 checks (rocprof_check.py, HIP_VISIBLE_DEVICES=4, bf16, v9 tile order, amdgcnas on)

Cold kernel time: rocprofv3 --kernel-trace over `bench.py --rocprof` (1000 back-to-back launches over
a rotating 512 MB of inputs), median. L2: --pmc TCC_HIT_sum TCC_MISS_sum TCC_EA0_RDREQ_sum over 10
warm launches, median of launches 3-10. v20 is one-tile with the reduce-scatter fixup (default).

| shape | version | cold kernel time (us) | L2 hit rate | memory read requests |
|---|---|---|---|---|
| 3328x5120x8192 | v13 | 288.7 | 74.4% | 4.32M |
| 3328x5120x8192 | v17 | 355.8 | 68.6% | 4.92M |
| 3328x5120x8192 | v19 | 202.8 | 68.3% | 5.07M |
| 3328x5120x8192 | v20 | 216.2 | 68.6% | 4.90M |
| 4352x4096x8192 | v13 | 289.4 | 72.8% | 4.81M |
| 4352x4096x8192 | v17 | 247.3 | 67.8% | 5.32M |
| 4352x4096x8192 | v19 | 211.7 | 61.8% | 6.70M |
| 4352x4096x8192 | v20 | 221.0 | 67.6% | 5.34M |
| 4352x4352x8192 | v13 | 290.2 | 69.9% | 5.68M |
| 4352x4352x8192 | v17 | 242.2 | 65.8% | 6.15M |
| 4352x4352x8192 | v19 | 224.7 | 60.0% | 7.53M |
| 4352x4352x8192 | v20 | 237.3 | 65.3% | 6.24M |
| 4096x6144x8192 | v13 | 295.0 | 78.3% | 5.30M |
| 4096x6144x8192 | v17 | 271.3 | 76.3% | 5.57M |
| 4096x6144x8192 | v19 | 288.9 | 61.3% | 9.86M |
| 4096x6144x8192 | v20 | 293.0 | 74.3% | 5.85M |

v20 with STREAMK_FIXUP=owner (v17's ending in the v20 build), same session:

| shape | version | cold kernel time (us) | L2 hit rate | memory read requests |
|---|---|---|---|---|
| 3328x5120x8192 | v20 | 353.6 | 68.9% | 4.86M |
| 4352x4096x8192 | v20 | 247.5 | 67.9% | 5.31M |
| 4352x4352x8192 | v20 | 240.7 | 65.9% | 6.12M |
| 4096x6144x8192 | v20 | 272.0 | 76.3% | 5.57M |
