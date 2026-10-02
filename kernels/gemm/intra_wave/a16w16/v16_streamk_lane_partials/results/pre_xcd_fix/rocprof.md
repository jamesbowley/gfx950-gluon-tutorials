# rocprofv3 kernel time, cold caches (HIP_VISIBLE_DEVICES=2, bf16, v9 tile order)

`rocprofv3 --kernel-trace -- python bench.py --version V --M M --N N --K K --dtype bf16 --rocprof`
(1000 launches over rotating 512 MB of inputs), median dispatch duration. The v13 and v15
columns are from `../../v15_streamk_onetile/results/rocprof.md`.

| shape | v13 (us) | v15 (us) | v16 (us) | v16 vs v13 |
|---|---|---|---|---|
| 4352x4096x8192 | 278.9 | 319.7 | 246.9 | +13.0% |
| 4352x4352x8192 | 279.2 | 282.2 | 244.2 | +14.3% |
| 3328x5120x8192 | 278.4 | 627.7 | 345.6 | -19.4% |
