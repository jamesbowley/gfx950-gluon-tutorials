# rocprofv3 checks (HIP_VISIBLE_DEVICES=2, bf16, v9 tile order, amdgcnas on)

## Kernel time, cold caches

`rocprofv3 --kernel-trace -- python bench.py --version V --M M --N N --K K --dtype bf16 --rocprof`
(1000 launches over rotating 512 MB of inputs), median dispatch duration. All in one session.

| shape | leftover tiles S | v13 (us) | v17 (us) | v19 (us) | v17 vs v13 | v19 vs v13 |
|---|---|---|---|---|---|---|
| 3328x5120x8192 | 4 | 278.3 | 342.0 | 202.2 | -18.6% | +37.6% |
| 4352x4096x8192 | 16 | 278.9 | 239.9 | 213.8 | +16.3% | +30.4% |
| 4096x6144x8192 | 128 | 292.0 | 272.2 | 291.8 | +7.3% | +0.1% |
| 4096x7936x8192 | 240 | 335.2 | 340.2 | 468.0 | -1.5% | -28.4% |

## L2 counters

`rocprofv3 --pmc TCC_HIT_sum TCC_MISS_sum TCC_EA0_RDREQ_sum` over 10 launches on the same
inputs (warm), median of launches 3-10. "forward" is the scratch build with v19's two-tile
segments processed in k order (`forward_vs_reversed.md`).

| shape | S | v13 hit / reads | v17 hit / reads | v19 hit / reads | forward hit / reads |
|---|---|---|---|---|---|
| 3328x5120x8192 | 4 | 74.2% / 4.34M | 68.7% / 4.87M | 68.0% / 5.10M | 28.6% / 13.04M |
| 4352x4096x8192 | 16 | 72.6% / 4.84M | 67.8% / 5.33M | 61.6% / 6.71M | 4.5% / 18.64M |
| 4096x5120x8192 | 64 | 76.7% / 4.79M | 73.2% / 5.17M | 62.6% / 7.76M | 26.3% / 16.39M |
| 4096x6144x8192 | 128 | 78.3% / 5.30M | 76.3% / 5.57M | 60.4% / 10.05M | 48.3% / 13.40M |
| 4096x7936x8192 | 240 | 78.7% / 6.71M | 77.3% / 7.22M | 27.5% / 24.77M | 5.1% / 32.90M |
