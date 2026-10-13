# streamk_wip (v26, capped) vs AITER, v9 and hipBLASLt on the 630 shapes

Shapes: `supported_status_scoped.csv` (630 shapes, no bias; 598 bf16 and 32 fp32 outputs). Inputs from `torch.manual_seed(0)`, every output checked against an fp32 `F.linear` reference, `do_bench_cudagraph` with 3 interleaved rounds and the median taken.

- Stack: torch 2.10.0+rocm7.2.4.git3d3aa833, hipBLASLt 100202 (dabb6df2b9), Triton 3.8.0 @ /home/jbowley/repos/triton_gfx950-tutorial-v2.2/python/triton.
- GPU: HIP device 3, PCI bus 21, gfx950:sramecc+:xnack-.
- AITER: e2c7aa0aa [Triton/Gluon] gfx950 a16w16: skip the Gluon kernel on Triton without cd_regclass, on top of a1455f1c4 2026-10-02 [CK] [FlyDSL] add mxfp8 a8w8 blockscale for gfx950 (#4951).
- streamk_wip: tutorial `v26_streamk_ragged_k`, snapshot sha256 96f09f5c62943d6b (newest WORKLOG entry: "5. The host rule from sweeps G and H (2026-10-12)"), `STREAMK_POLICY=capped` (the kernel's default), v9 tile order. The configuration capped ran on each shape is in the appendix and in `v20_label`.
- hipBLASLt: the call AITER's `torch` libtype makes, `F.linear(x, w)` for bf16 and `torch.mm(x, w.t(), out_dtype=torch.float32)` for fp32, timed the same way. Its tile is the MT of the first `Cijk_` kernel it launches.
- Cache state: back to back on the same buffers (inputs stay cached between launches).
- "Faster" and "slower" mean a gap of more than 1%; "within 1%" holds the rest.

**Against AITER's pick, by the configuration capped chose**

| | Shapes | v9: faster | v9: within 1% | v9: slower | v9: median | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All shapes | 630 | 327 | 20 | 283 | +1.8% | 403 | 19 | 208 | +3.4% |
| Data-parallel | 287 | 239 | 15 | 33 | +3.8% | 244 | 11 | 32 | +3.9% |
| One-tile reduce-scatter, K >= 2048 | 147 | 5 | 0 | 142 | -53.0% | 37 | 4 | 106 | -13.9% |
| One-tile owner, K >= 2048 | 155 | 64 | 4 | 87 | -5.9% | 102 | 3 | 50 | +7.6% |
| One-tile reduce-scatter, K < 2048 | 16 | 0 | 0 | 16 | -51.3% | 0 | 0 | 16 | -50.9% |
| One-tile owner, K < 2048 | 3 | 0 | 0 | 3 | -25.6% | 0 | 0 | 3 | -34.7% |
| Spread | 22 | 19 | 1 | 2 | +2.9% | 20 | 1 | 1 | +8.0% |

Each pair of columns compares that kernel with AITER's pick; positive medians mean it is faster.

**Against AITER's pick, by the tile AITER uses**

| | Shapes | v9: faster | v9: within 1% | v9: slower | v9: median | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All shapes | 630 | 327 | 20 | 283 | +1.8% | 403 | 19 | 208 | +3.4% |
| AITER uses a 256x256 tile | 248 | 232 | 11 | 5 | +3.7% | 242 | 6 | 0 | +4.1% |
| AITER uses another tile | 382 | 95 | 9 | 278 | -24.1% | 161 | 13 | 208 | -3.3% |

Each pair of columns compares that kernel with AITER's pick; positive medians mean it is faster.

**Against AITER's pick, by v9 grid size (number of 256x256 output tiles)**

| | Shapes | v9: faster | v9: within 1% | v9: slower | v9: median | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All shapes | 630 | 327 | 20 | 283 | +1.8% | 403 | 19 | 208 | +3.4% |
| Under 256 (less than one wave) | 321 | 60 | 7 | 254 | -33.2% | 105 | 13 | 203 | -7.8% |
| 256 to 1023 | 152 | 120 | 5 | 27 | +4.2% | 144 | 3 | 5 | +7.1% |
| 1024 or more | 157 | 147 | 8 | 2 | +3.7% | 154 | 3 | 0 | +3.9% |

Each pair of columns compares that kernel with AITER's pick; positive medians mean it is faster.

**Against hipBLASLt (torch), by v9 grid size**

| | Shapes | AITER's pick: faster | AITER's pick: within 1% | AITER's pick: slower | AITER's pick: median | v9: faster | v9: within 1% | v9: slower | v9: median | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All shapes | 630 | 216 | 235 | 179 | -0.2% | 307 | 35 | 288 | +0.6% | 402 | 31 | 197 | +3.1% |
| Under 256 (less than one wave) | 321 | 119 | 111 | 91 | -0.1% | 67 | 6 | 248 | -36.1% | 121 | 11 | 189 | -6.3% |
| 256 to 1023 | 152 | 43 | 69 | 40 | -0.2% | 116 | 3 | 33 | +3.5% | 143 | 3 | 6 | +7.3% |
| 1024 or more | 157 | 54 | 55 | 48 | -0.3% | 124 | 26 | 7 | +3.1% | 138 | 17 | 2 | +4.0% |

Each pair of columns compares that kernel with hipBLASLt; positive medians mean it is faster.

**hipBLASLt's tile choice**

| hipBLASLt tile (MxNxK) | Shapes | hipBLASLt median TFLOPS | streamk_wip vs hipBLASLt: faster | within 1% | slower | median |
|---|---:|---:|---:|---:|---:|---:|
| 256x256x64 | 225 | 1411 | 205 | 18 | 2 | +3.7% |
| 128x128x256 | 40 | 736 | 10 | 2 | 28 | -7.9% |
| 192x256x64 | 34 | 1194 | 25 | 4 | 5 | +4.9% |
| 224x256x64 | 32 | 1261 | 32 | 0 | 0 | +10.5% |
| 256x128x64 | 27 | 751 | 1 | 1 | 25 | -8.7% |
| 160x256x64 | 27 | 1179 | 22 | 2 | 3 | +6.0% |
| 256x224x64 | 20 | 1278 | 20 | 0 | 0 | +9.2% |
| 256x240x64 | 18 | 1161 | 18 | 0 | 0 | +21.2% |
| 144x256x64 | 15 | 1156 | 13 | 1 | 1 | +6.8% |
| 128x128x128 | 14 | 640 | 0 | 0 | 14 | -21.0% |
| 160x128x128 | 13 | 898 | 2 | 0 | 11 | -5.4% |
| 96x128x128 | 12 | 640 | 0 | 0 | 12 | -40.9% |
| 192x128x128 | 12 | 839 | 6 | 1 | 5 | +2.4% |
| 256x208x64 | 12 | 1161 | 12 | 0 | 0 | +14.4% |
| 256x192x64 | 11 | 1213 | 8 | 1 | 2 | +4.6% |
| 208x256x64 | 11 | 1234 | 11 | 0 | 0 | +15.3% |
| 256x160x64 | 10 | 1181 | 3 | 0 | 7 | -3.1% |
| 64x64x256 | 9 | 484 | 0 | 0 | 9 | -40.5% |
| 128x192x128 | 7 | 983 | 2 | 0 | 5 | -13.1% |
| 128x160x128 | 7 | 960 | 0 | 0 | 7 | -16.0% |
| 240x256x64 | 7 | 1199 | 7 | 0 | 0 | +20.4% |
| 128x80x128 | 6 | 714 | 0 | 0 | 6 | -21.4% |
| 128x96x128 | 6 | 796 | 0 | 0 | 6 | -21.0% |
| 224x128x64 | 5 | 875 | 0 | 0 | 5 | -20.2% |
| 256x176x64 | 4 | 793 | 3 | 0 | 1 | +3.8% |
| 128x64x128 | 4 | 572 | 0 | 0 | 4 | -59.1% |
| 160x64x128 | 4 | 510 | 0 | 0 | 4 | -28.4% |
| 64x80x256 | 4 | 536 | 0 | 0 | 4 | -35.9% |
| 64x96x256 | 4 | 564 | 0 | 0 | 4 | -27.4% |
| 64x32x256 | 3 | 329 | 0 | 0 | 3 | -40.4% |
| 64x48x256 | 3 | 432 | 0 | 0 | 3 | -44.8% |
| 160x192x128 | 3 | 948 | 0 | 1 | 2 | -1.4% |
| 192x128x64 | 2 | 918 | 0 | 0 | 2 | -29.2% |
| 128x192x64 | 2 | 894 | 0 | 0 | 2 | -7.7% |
| 256x112x64 | 2 | 479 | 0 | 0 | 2 | -13.6% |
| 192x64x128 | 2 | 428 | 0 | 0 | 2 | -21.8% |
| 128x128x64 | 2 | 500 | 0 | 0 | 2 | -30.2% |
| 256x96x64 | 2 | 443 | 0 | 0 | 2 | -18.7% |
| 256x64x64 | 2 | 343 | 0 | 0 | 2 | -19.8% |
| 96x64x128 | 1 | 474 | 0 | 0 | 1 | -57.4% |
| 224x160x64 | 1 | 1088 | 1 | 0 | 0 | +12.6% |
| 192x192x64 | 1 | 1042 | 0 | 0 | 1 | -3.7% |
| 64x96x128 | 1 | 600 | 0 | 0 | 1 | -24.7% |
| 256x80x64 | 1 | 406 | 0 | 0 | 1 | -24.9% |
| 256x80x128 | 1 | 586 | 0 | 0 | 1 | -12.8% |
| 320x192x64 | 1 | 716 | 1 | 0 | 0 | +9.4% |

Solution-name tokens (stream-K `SK`, global split-U `GSU`, kernels launched): SK3 GSU0 x1: 604, SK3 GSU0 x2: 24, none GSU- x1: 2. `SK3` marks a kernel built with stream-K support, not that this launch split K.

**streamk_wip against v9, by configuration**

| | Shapes | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|
| All shapes | 630 | 360 | 240 | 30 | +2.4% |
| Data-parallel | 287 | 44 | 229 | 14 | +0.1% |
| One-tile reduce-scatter, K >= 2048 | 147 | 146 | 1 | 0 | +100.8% |
| One-tile owner, K >= 2048 | 155 | 149 | 3 | 3 | +12.8% |
| One-tile reduce-scatter, K < 2048 | 16 | 8 | 3 | 5 | +2.4% |
| One-tile owner, K < 2048 | 3 | 0 | 0 | 3 | -12.2% |
| Spread | 22 | 13 | 4 | 5 | +2.5% |

Each pair of columns compares that kernel with v9; positive medians mean it is faster.

**Diagnostic: capped's split against forcing data-parallel**

| Configuration | Shapes | split faster | within 1% | split slower | median |
|---|---:|---:|---:|---:|---:|
| One-tile owner, K < 2048 | 3 | 0 | 0 | 3 | -8.9% |
| One-tile owner, K >= 2048 | 155 | 149 | 3 | 3 | +12.7% |
| One-tile reduce-scatter, K < 2048 | 16 | 12 | 1 | 3 | +4.9% |
| One-tile reduce-scatter, K >= 2048 | 147 | 146 | 1 | 0 | +101.7% |
| Spread | 22 | 14 | 5 | 3 | +2.5% |

**Selected shapes (TFLOPS)**

| Shape (MxNxK) | Output tiles | AITER pick | AITER | hipBLASLt | hipBLASLt tile | v9 | streamk_wip | streamk_wip config | vs v9 | vs AITER | vs hipBLASLt |
|---|---:|---|---:|---:|---|---:|---:|---|---:|---:|---:|
| 512x2304x16384 | 18 | torch | 744 | 737 | 256x128x64 | 162 | 728 | one-tile reduce-scatter, n = 10 | +350% | -2% | -1% |
| 512x5120x25600 | 40 | flydsl | 436 | 909 | 160x128x128 | 327 | 930 | one-tile reduce-scatter, n = 4 | +184% | +113% | +2% |
| 512x7168x35840 | 56 | flydsl | 818 | 965 | 256x176x64 | 436 | 1059 | one-tile reduce-scatter, n = 3 | +143% | +29% | +10% |
| 8192x2304x16384 | 288 | torch | 1196 | 1200 | 144x256x64 | 1009 | 1387 | one-tile reduce-scatter, n = 4 | +37% | +16% | +16% |
| 4096x5120x25600 | 320 | torch | 1263 | 1251 | 160x256x64 | 1138 | 1499 | one-tile owner, n = 4 | +32% | +19% | +20% |
| 4096x4608x8192 | 288 | torch | 1175 | 1174 | 144x256x64 | 1021 | 1351 | one-tile reduce-scatter, n = 4 | +32% | +15% | +15% |
| 1024x3072x512 | 48 | flydsl | 459 | 400 | 96x128x128 | 226 | 226 | data-parallel | -0% | -51% | -43% |
| 2048x2048x512 | 64 | flydsl | 523 | 466 | 128x128x128 | 281 | 282 | data-parallel | +0% | -46% | -40% |
| 4096x5120x640 | 320 | flydsl | 818 | 762 | 256x176x64 | 740 | 747 | data-parallel | +1% | -9% | -2% |
| 8192x2048x512 | 256 | torch | 797 | 799 | 256x256x64 | 847 | 839 | data-parallel | -1% | +5% | +5% |

**Against `results/details_v24.csv`** (this run / that run - 1, median; for v20 the baseline may use a different policy or kernel snapshot)

| | Shapes | AITER's pick | v9 | v20 | hipBLASLt |
|---|---:|---:|---:|---:|---:|
| All shapes | 630 | -0.0% | +0.0% | +0.0% | +0.0% |
| Under 256 (less than one wave) | 321 | +0.0% | +0.0% | +0.2% | +0.0% |
| 256 to 1023 | 152 | -0.0% | -0.0% | +0.0% | -0.0% |
| 1024 or more | 157 | -0.0% | -0.0% | -0.0% | +0.0% |
| K < 2048 | 91 | +0.0% | +0.0% | -0.0% | +0.0% |
| K >= 2048 | 539 | -0.0% | -0.0% | +0.1% | +0.0% |

AITER's library and kernel are the same on 630 of 630 shapes.

**Wrong outputs, failures and notes**

| Shape | out | pick ok (max err) | v9 ok | hipBLASLt ok | streamk_wip ok | after timing | flags left | note |
|---|---|---|---|---|---|---|---|---|
| 1024x6144x2048 | bfloat16 | False (1) | True | True | True (1) | True | 0 |  |
| 8192x6144x2048 | bfloat16 | False (1) | True | True | True (1) | True | 0 |  |
| 1024x6144x3072 | bfloat16 | False (2) | True | True | True (1) | True | 0 |  |
| 32768x6144x3072 | bfloat16 | False (2) | True | True | True (1) | True | 0 |  |
| 512x1024x4096 | bfloat16 | False (2) | True | True | True (1) | True | 0 |  |
| 768x8704x4096 | bfloat16 | False (2) | True | True | True (1) | True | 0 |  |
| 1280x512x4096 | bfloat16 | False (2) | True | True | True (1) | True | 0 |  |
| 512x6400x5120 | bfloat16 | False (2) | True | True | True (1) | True | 0 |  |
| 512x12800x5120 | bfloat16 | False (2) | True | True | True (1) | True | 0 |  |
| 1024x6400x5120 | bfloat16 | False (2) | True | True | True (1) | True | 0 |  |
| 512x1280x6144 | bfloat16 | False (2) | True | True | True (1) | True | 0 |  |
| 512x1536x6144 | bfloat16 | False (2) | True | True | True (1) | True | 0 |  |
| 512x2304x6144 | bfloat16 | False (2) | True | True | True (1) | True | 0 |  |
| 512x3072x6144 | bfloat16 | False (2) | True | True | True (1) | True | 0 |  |
| 512x5120x6400 | bfloat16 | False (2) | True | True | True (2) | True | 0 |  |
| 768x2048x7168 | bfloat16 | False (2) | True | True | True (1) | True | 0 |  |
| 1024x512x7168 | bfloat16 | False (2) | True | True | True (1) | True | 0 |  |
| 1024x1536x7168 | bfloat16 | False (2) | True | True | True (2) | True | 0 |  |
| 4096x512x7168 | bfloat16 | False (2) | True | True | True (2) | True | 0 |  |
| 512x4096x8192 | bfloat16 | False (2) | True | True | True (1) | True | 0 |  |
| 512x16384x8192 | bfloat16 | False (2) | True | True | True (2) | True | 0 |  |
| 1024x10240x8192 | bfloat16 | False (2) | True | True | True (2) | True | 0 |  |
| 512x13312x16384 | bfloat16 | False (4) | True | True | True (2) | True | 0 |  |
| 512x6144x18432 | bfloat16 | False (4) | True | True | True (2) | True | 0 |  |
| 512x5120x25600 | bfloat16 | False (4) | True | True | True (2) | True | 0 |  |
| 512x16384x26624 | bfloat16 | False (4) | True | True | True (2) | True | 0 |  |
| 2048x16384x26624 | bfloat16 | False (4) | True | True | True (4) | True | 0 |  |
| 512x8192x28672 | bfloat16 | False (4) | True | True | True (4) | True | 0 |  |
| 512x7168x35840 | bfloat16 | False (4) | True | True | True (4) | True | 0 |  |

29 of these are AITER's pick alone outside the tolerance; every other kernel passed.

## Appendix: every shape

| Shape | out | tiles | streamk_wip config | streamk_wip | v9 | AITER pick | AITER | hipBLASLt | hipBLASLt tile |
|---|---|---:|---|---:|---:|---|---:|---:|---|
| 1024x3072x512 | bfloat16 | 48 | data-parallel | 226.1 | 226.3 | flydsl | 459.1 | 400.1 | 96x128x128 |
| 1024x4096x512 | bfloat16 | 64 | data-parallel | 282.5 | 282.3 | flydsl | 523.9 | 464.7 | 128x128x128 |
| 1024x7168x512 | bfloat16 | 112 | data-parallel | 453.1 | 452.6 | flydsl | 609.8 | 568.6 | 256x128x64 |
| 2048x2048x512 | bfloat16 | 64 | data-parallel | 281.5 | 281.1 | flydsl | 522.8 | 465.7 | 128x128x128 |
| 2048x4096x512 | bfloat16 | 128 | data-parallel | 490.3 | 491.1 | torch | 603.9 | 604.7 | 256x128x64 |
| 4096x2048x512 | bfloat16 | 128 | data-parallel | 488.8 | 488.7 | torch | 660.3 | 661.1 | 256x128x64 |
| 4096x4096x512 | bfloat16 | 256 | data-parallel | 846.0 | 853.9 | torch | 803.4 | 803.6 | 256x256x64 |
| 4096x7168x512 | bfloat16 | 448 | data-parallel | 873.3 | 866.8 | flydsl | 849.7 | 773.1 | 224x256x64 |
| 8192x2048x512 | bfloat16 | 256 | data-parallel | 839.3 | 847.1 | torch | 796.7 | 798.6 | 256x256x64 |
| 8192x4096x512 | bfloat16 | 512 | data-parallel | 940.3 | 922.9 | flydsl | 861.3 | 863.4 | 256x256x64 |
| 16384x2048x512 | bfloat16 | 512 | data-parallel | 935.9 | 924.5 | torch | 859.8 | 863.2 | 256x256x64 |
| 16384x4096x512 | bfloat16 | 1024 | data-parallel | 1020.5 | 978.6 | torch | 919.0 | 929.5 | 256x256x64 |
| 16384x7168x512 | bfloat16 | 1792 | data-parallel | 1051.2 | 1010.1 | torch | 975.0 | 981.2 | 256x256x64 |
| 32768x2048x512 | bfloat16 | 1024 | data-parallel | 1014.3 | 971.2 | torch | 919.5 | 925.8 | 256x256x64 |
| 1024x5120x640 | bfloat16 | 80 | data-parallel | 376.3 | 375.4 | torch | 516.2 | 519.7 | 160x128x128 |
| 2048x5120x640 | bfloat16 | 160 | data-parallel | 672.4 | 672.4 | flydsl | 670.3 | 530.5 | 192x256x64 |
| 4096x5120x640 | bfloat16 | 320 | data-parallel | 747.2 | 740.2 | flydsl | 817.8 | 762.5 | 256x176x64 |
| 8192x5120x640 | bfloat16 | 640 | data-parallel | 965.0 | 936.9 | torch | 897.1 | 905.6 | 256x224x64 |
| 16384x5120x640 | bfloat16 | 1280 | data-parallel | 1113.6 | 1073.9 | torch | 1036.2 | 1041.6 | 256x256x64 |
| 32768x5120x640 | bfloat16 | 2560 | data-parallel | 1118.4 | 1066.6 | torch | 1080.3 | 1087.1 | 256x256x64 |
| 1024x7168x768 | bfloat16 | 112 | data-parallel | 535.2 | 534.6 | flydsl | 708.6 | 662.1 | 224x128x64 |
| 10240x7168x768 | bfloat16 | 1120 | data-parallel | 1095.5 | 1058.0 | opus | 404.6 | 855.9 | 256x240x64 |
| 512x4096x1024 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 194.0 | 199.9 | torch | 482.5 | 483.6 | 128x64x128 |
| 512x6144x1024 | bfloat16 | 48 | data-parallel | 293.1 | 292.2 | flydsl | 584.1 | 554.0 | 96x128x128 |
| 512x7168x1024 | bfloat16 | 56 | data-parallel | 334.5 | 334.0 | flydsl | 641.0 | 598.0 | 128x128x128 |
| 512x8192x1024 | bfloat16 | 64 | data-parallel | 372.5 | 372.2 | flydsl | 691.7 | 651.2 | 128x128x128 |
| 1024x4096x1024 | bfloat16 | 64 | data-parallel | 372.2 | 372.0 | torch | 649.0 | 650.7 | 128x128x128 |
| 1024x7168x1024 | bfloat16 | 112 | data-parallel | 605.1 | 604.4 | flydsl | 775.8 | 758.3 | 224x128x64 |
| 1024x8192x1024 | bfloat16 | 128 | data-parallel | 661.9 | 661.1 | torch | 817.1 | 817.6 | 256x128x64 |
| 2048x4096x1024 | bfloat16 | 128 | data-parallel | 661.0 | 660.2 | torch | 818.7 | 820.1 | 256x128x64 |
| 2048x7168x1024 | bfloat16 | 224 | data-parallel | 1048.5 | 1056.5 | flydsl | 1028.4 | 998.5 | 256x256x64 |
| 2048x8192x1024 | bfloat16 | 256 | data-parallel | 1109.6 | 1100.9 | flydsl | 1059.8 | 1056.7 | 256x256x64 |
| 3072x4096x1024 | bfloat16 | 192 | data-parallel | 934.8 | 940.2 | torch | 953.2 | 952.1 | 256x192x64 |
| 4096x4096x1024 | bfloat16 | 256 | data-parallel | 1116.3 | 1125.2 | torch | 1102.9 | 1104.7 | 256x256x64 |
| 4096x7168x1024 | bfloat16 | 448 | data-parallel | 1115.9 | 1118.2 | flydsl | 1088.0 | 1011.9 | 224x256x64 |
| 4096x8192x1024 | bfloat16 | 512 | data-parallel | 1170.3 | 1170.1 | asm | 1160.0 | 1128.3 | 256x256x64 |
| 5120x4096x1024 | bfloat16 | 320 | data-parallel | 885.6 | 881.2 | torch | 1023.9 | 1025.6 | 256x160x64 |
| 7680x4096x1024 | bfloat16 | 480 | data-parallel | 1154.2 | 1151.0 | torch | 959.5 | 963.6 | 256x240x64 |
| 8192x7168x1024 | bfloat16 | 896 | data-parallel | 1167.1 | 1150.2 | asm | 1090.9 | 1047.4 | 224x256x64 |
| 8192x8192x1024 | bfloat16 | 1024 | data-parallel | 1237.4 | 1211.6 | asm | 1191.0 | 1188.7 | 256x256x64 |
| 16384x4096x1024 | bfloat16 | 1024 | data-parallel | 1241.5 | 1223.2 | torch | 1189.7 | 1195.6 | 256x256x64 |
| 16384x7168x1024 | bfloat16 | 1792 | data-parallel | 1277.7 | 1250.9 | asm | 1193.8 | 1229.0 | 256x256x64 |
| 16384x8192x1024 | bfloat16 | 2048 | data-parallel | 1261.0 | 1231.5 | torch | 1224.9 | 1230.5 | 256x256x64 |
| 32768x7168x1024 | bfloat16 | 3584 | data-parallel | 1256.8 | 1232.4 | torch | 1239.4 | 1250.6 | 256x256x64 |
| 32768x8192x1024 | bfloat16 | 4096 | data-parallel | 1256.5 | 1220.3 | torch | 1242.0 | 1253.5 | 256x256x64 |
| 512x5120x1280 | bfloat16 | 40 | one-tile reduce-scatter, n = 4 | 257.2 | 263.9 | flydsl | 537.8 | 553.8 | 160x64x128 |
| 1024x5120x1280 | bfloat16 | 80 | data-parallel | 482.7 | 482.5 | torch | 705.9 | 706.8 | 160x128x128 |
| 2048x5120x1280 | bfloat16 | 160 | data-parallel | 867.8 | 870.1 | torch | 919.5 | 925.4 | 160x256x64 |
| 4096x5120x1280 | bfloat16 | 320 | data-parallel | 927.8 | 923.8 | torch | 981.7 | 983.3 | 160x256x64 |
| 8192x5120x1280 | bfloat16 | 640 | data-parallel | 1174.3 | 1157.8 | flydsl | 1129.3 | 1105.7 | 256x224x64 |
| 16384x5120x1280 | bfloat16 | 1280 | data-parallel | 1310.9 | 1288.2 | flydsl | 1228.3 | 1264.8 | 256x256x64 |
| 32768x5120x1280 | bfloat16 | 2560 | data-parallel | 1310.4 | 1282.1 | torch | 1287.8 | 1295.5 | 256x256x64 |
| 512x1536x1536 | bfloat16 | 12 | one-tile reduce-scatter, n = 6 | 115.7 | 85.9 | flydsl | 347.5 | 304.9 | 64x64x256 |
| 512x2048x1536 | bfloat16 | 16 | one-tile reduce-scatter, n = 6 | 150.2 | 113.7 | torch | 401.7 | 400.3 | 64x64x256 |
| 512x2304x1536 | bfloat16 | 18 | one-tile reduce-scatter, n = 6 | 162.8 | 127.1 | flydsl | 393.2 | 380.7 | 64x80x256 |
| 512x4096x1536 | bfloat16 | 32 | one-tile reduce-scatter, n = 6 | 234.6 | 224.4 | torch | 570.5 | 571.8 | 128x64x128 |
| 512x6144x1536 | bfloat16 | 48 | one-tile reduce-scatter, n = 4 | 324.7 | 323.5 | flydsl | 638.1 | 633.4 | 96x128x128 |
| 512x7168x1536 | bfloat16 | 56 | one-tile reduce-scatter, n = 3 | 358.8 | 373.4 | flydsl | 675.1 | 584.4 | 128x128x256 |
| 1024x1536x1536 | bfloat16 | 24 | one-tile reduce-scatter, n = 6 | 201.7 | 168.6 | flydsl | 504.9 | 473.5 | 96x64x128 |
| 1024x2048x1536 | bfloat16 | 32 | one-tile reduce-scatter, n = 6 | 233.1 | 220.6 | torch | 570.1 | 573.0 | 128x64x128 |
| 1024x2304x1536 | bfloat16 | 36 | one-tile reduce-scatter, n = 5 | 259.2 | 248.4 | torch | 514.0 | 513.2 | 128x80x128 |
| 1024x3072x1536 | bfloat16 | 48 | one-tile reduce-scatter, n = 4 | 324.3 | 324.4 | flydsl | 635.8 | 633.1 | 96x128x128 |
| 1024x4096x1536 | bfloat16 | 64 | one-tile reduce-scatter, n = 3 | 387.9 | 418.2 | torch | 626.0 | 624.4 | 128x128x256 |
| 1024x7168x1536 | bfloat16 | 112 | data-parallel | 682.4 | 680.9 | torch | 873.2 | 875.1 | 224x128x64 |
| 2048x1536x1536 | bfloat16 | 48 | one-tile reduce-scatter, n = 4 | 323.2 | 324.6 | torch | 630.9 | 631.5 | 96x128x128 |
| 2048x2048x1536 | bfloat16 | 64 | one-tile reduce-scatter, n = 3 | 388.7 | 417.6 | torch | 625.3 | 624.6 | 128x128x256 |
| 2048x4096x1536 | bfloat16 | 128 | data-parallel | 751.7 | 751.1 | torch | 936.6 | 939.8 | 256x128x64 |
| 3072x2048x1536 | bfloat16 | 96 | one-tile owner, n = 2-3 | 524.0 | 596.5 | opus | 862.1 | 797.8 | 128x192x128 |
| 3072x4096x1536 | bfloat16 | 192 | data-parallel | 1047.7 | 1054.2 | opus | 1083.5 | 1091.6 | 256x192x64 |
| 4096x1536x1536 | bfloat16 | 96 | one-tile owner, n = 2-3 | 522.5 | 595.4 | torch | 799.9 | 804.7 | 192x128x128 |
| 4096x2048x1536 | bfloat16 | 128 | data-parallel | 752.1 | 752.4 | opus | 1000.2 | 941.1 | 256x128x64 |
| 4096x3072x1536 | bfloat16 | 192 | data-parallel | 1048.5 | 1058.8 | flydsl | 1014.3 | 1054.9 | 192x256x64 |
| 4096x4096x1536 | bfloat16 | 256 | data-parallel | 1224.8 | 1229.5 | torch | 1205.8 | 1210.6 | 256x256x64 |
| 5120x2048x1536 | bfloat16 | 160 | data-parallel | 909.7 | 911.2 | torch | 1003.6 | 1008.6 | 256x160x64 |
| 5120x4096x1536 | bfloat16 | 320 | data-parallel | 962.4 | 959.3 | torch | 1068.8 | 1073.2 | 256x160x64 |
| 7680x2048x1536 | bfloat16 | 240 | data-parallel | 1194.4 | 1202.8 | torch | 1021.1 | 1021.0 | 256x240x64 |
| 7680x4096x1536 | bfloat16 | 480 | data-parallel | 1252.8 | 1249.4 | torch | 1065.2 | 1070.4 | 256x240x64 |
| 8192x1536x1536 | bfloat16 | 192 | data-parallel | 1044.6 | 1054.6 | torch | 1057.1 | 1057.8 | 192x256x64 |
| 10240x7168x1536 | bfloat16 | 1120 | data-parallel | 1248.1 | 1233.1 | asm | 1180.3 | 1145.0 | 224x256x64 |
| 16384x1536x1536 | bfloat16 | 384 | data-parallel | 1119.2 | 1124.3 | torch | 1115.0 | 1120.1 | 192x256x64 |
| 16384x2048x1536 | bfloat16 | 512 | data-parallel | 1282.2 | 1276.9 | torch | 1251.9 | 1258.6 | 256x256x64 |
| 16384x3072x1536 | bfloat16 | 768 | data-parallel | 1320.8 | 1304.5 | torch | 1278.1 | 1284.6 | 256x256x64 |
| 16384x4096x1536 | bfloat16 | 1024 | data-parallel | 1340.1 | 1322.8 | torch | 1293.6 | 1300.5 | 256x256x64 |
| 32768x1536x1536 | bfloat16 | 768 | data-parallel | 1325.9 | 1308.8 | flydsl | 1254.0 | 1284.2 | 256x256x64 |
| 512x7168x1792 | bfloat16 | 56 | one-tile reduce-scatter, n = 3 | 419.0 | 389.1 | flydsl | 716.9 | 596.0 | 128x128x256 |
| 1024x7168x1792 | bfloat16 | 112 | one-tile owner, n = 2-3 | 663.5 | 713.8 | flydsl | 852.2 | 916.8 | 224x128x64 |
| 2048x7168x1792 | bfloat16 | 224 | data-parallel | 1207.9 | 1215.2 | flydsl | 1162.5 | 1083.0 | 224x256x64 |
| 4096x7168x1792 | bfloat16 | 448 | data-parallel | 1252.8 | 1254.3 | flydsl | 1208.6 | 1137.7 | 224x256x64 |
| 8192x7168x1792 | bfloat16 | 896 | data-parallel | 1279.5 | 1271.1 | asm | 1215.2 | 1161.2 | 224x256x64 |
| 16384x7168x1792 | bfloat16 | 1792 | data-parallel | 1380.3 | 1370.5 | torch | 1339.6 | 1344.0 | 256x256x64 |
| 32768x7168x1792 | bfloat16 | 3584 | data-parallel | 1386.2 | 1371.8 | torch | 1362.8 | 1369.3 | 256x256x64 |
| 512x4096x2048 | bfloat16 | 32 | one-tile reduce-scatter, n = 6 | 314.4 | 237.5 | torch | 600.6 | 603.1 | 128x64x128 |
| 512x6144x2048 | float32 | 48 | one-tile reduce-scatter, n = 4 | 380.6 | 314.5 | asm | 415.3 | 645.9 | 96x128x128 |
| 512x6144x2048 | bfloat16 | 48 | one-tile reduce-scatter, n = 4 | 407.5 | 348.6 | flydsl | 673.6 | 687.4 | 96x128x128 |
| 512x8192x2048 | bfloat16 | 64 | one-tile reduce-scatter, n = 3 | 486.4 | 450.0 | flydsl | 790.6 | 675.2 | 128x128x256 |
| 512x16384x2048 | bfloat16 | 128 | one-tile owner, n = 2 | 819.3 | 812.5 | flydsl | 821.8 | 993.6 | 256x128x64 |
| 1024x4096x2048 | bfloat16 | 64 | one-tile reduce-scatter, n = 3 | 485.7 | 448.1 | torch | 673.2 | 673.7 | 128x128x256 |
| 1024x6144x2048 | float32 | 96 | one-tile owner, n = 2-3 | 586.5 | 583.8 | opus | 667.1 | 880.3 | 192x128x64 |
| 1024x6144x2048 | bfloat16 | 96 | one-tile owner, n = 2-3 | 593.9 | 645.2 | flydsl | 802.2 | 872.7 | 192x128x128 |
| 1024x8192x2048 | bfloat16 | 128 | one-tile owner, n = 2 | 820.9 | 812.3 | flydsl | 1033.8 | 999.9 | 256x128x64 |
| 1024x16384x2048 | bfloat16 | 256 | data-parallel | 1288.6 | 1292.7 | flydsl | 911.1 | 1267.8 | 256x256x64 |
| 2048x4096x2048 | bfloat16 | 128 | one-tile owner, n = 2 | 817.5 | 814.0 | torch | 1001.6 | 1004.1 | 256x128x64 |
| 2048x6144x2048 | float32 | 192 | data-parallel | 1024.6 | 1030.2 | asm | 979.6 | 1034.1 | 192x256x64 |
| 2048x6144x2048 | bfloat16 | 192 | data-parallel | 1135.0 | 1142.6 | flydsl | 1113.0 | 1117.2 | 192x256x64 |
| 2048x8192x2048 | bfloat16 | 256 | data-parallel | 1290.8 | 1294.1 | flydsl | 1240.5 | 1264.9 | 256x256x64 |
| 2048x16384x2048 | bfloat16 | 512 | data-parallel | 1340.5 | 1336.3 | flydsl | 1049.3 | 1291.7 | 256x256x64 |
| 3072x4096x2048 | bfloat16 | 192 | data-parallel | 1130.3 | 1135.0 | torch | 1123.2 | 1128.0 | 256x192x64 |
| 4096x4096x2048 | bfloat16 | 256 | data-parallel | 1303.7 | 1309.1 | torch | 1278.1 | 1281.0 | 256x256x64 |
| 4096x6144x2048 | float32 | 384 | data-parallel | 1045.6 | 1058.1 | asm | 1011.6 | 1084.0 | 192x256x64 |
| 4096x6144x2048 | bfloat16 | 384 | data-parallel | 1191.0 | 1189.7 | flydsl | 1007.8 | 1169.8 | 192x256x64 |
| 4096x8192x2048 | bfloat16 | 512 | data-parallel | 1338.2 | 1342.4 | flydsl | 1279.1 | 1297.7 | 256x256x64 |
| 4096x16384x2048 | bfloat16 | 1024 | data-parallel | 1388.2 | 1375.3 | flydsl | 942.7 | 1339.1 | 256x256x64 |
| 5120x4096x2048 | bfloat16 | 320 | one-tile owner, n = 2 | 1060.1 | 1013.0 | torch | 1171.4 | 1182.8 | 256x160x64 |
| 7680x4096x2048 | bfloat16 | 480 | data-parallel | 1318.3 | 1313.8 | torch | 1129.3 | 1136.3 | 256x240x64 |
| 8192x6144x2048 | float32 | 768 | data-parallel | 1213.7 | 1210.3 | asm | 1161.3 | 1105.9 | 256x256x64 |
| 8192x6144x2048 | bfloat16 | 768 | data-parallel | 1381.5 | 1375.8 | flydsl | 865.1 | 1335.5 | 256x256x64 |
| 8192x8192x2048 | bfloat16 | 1024 | data-parallel | 1388.1 | 1374.4 | asm | 1335.7 | 1339.7 | 256x256x64 |
| 8192x16384x2048 | bfloat16 | 2048 | data-parallel | 1377.5 | 1349.5 | triton | 1029.7 | 1371.3 | 256x256x64 |
| 16384x4096x2048 | bfloat16 | 1024 | data-parallel | 1394.2 | 1383.3 | torch | 1326.9 | 1330.8 | 256x256x64 |
| 16384x6144x2048 | float32 | 1536 | data-parallel | 1220.5 | 1216.7 | asm | 1170.6 | 1130.1 | 256x256x64 |
| 16384x6144x2048 | bfloat16 | 1536 | data-parallel | 1411.4 | 1401.1 | torch | 1366.9 | 1369.7 | 256x256x64 |
| 16384x8192x2048 | bfloat16 | 2048 | data-parallel | 1411.8 | 1402.0 | torch | 1373.0 | 1376.6 | 256x256x64 |
| 16384x16384x2048 | bfloat16 | 4096 | data-parallel | 1383.5 | 1352.0 | flydsl | 959.0 | 1399.4 | 256x256x64 |
| 32768x6144x2048 | bfloat16 | 3072 | data-parallel | 1411.2 | 1392.3 | flydsl | 993.8 | 1396.5 | 256x256x64 |
| 32768x8192x2048 | bfloat16 | 4096 | data-parallel | 1424.5 | 1402.2 | torch | 1393.1 | 1404.7 | 256x256x64 |
| 32768x16384x2048 | bfloat16 | 8192 | data-parallel | 1389.7 | 1349.8 | flydsl | 828.5 | 1411.2 | 256x256x64 |
| 512x6144x3072 | float32 | 48 | one-tile reduce-scatter, n = 4 | 495.0 | 345.1 | opus | 386.2 | 729.3 | 96x128x128 |
| 512x6144x3072 | bfloat16 | 48 | one-tile reduce-scatter, n = 4 | 524.4 | 373.1 | flydsl | 728.3 | 761.3 | 96x128x128 |
| 1024x6144x3072 | float32 | 96 | one-tile owner, n = 2-3 | 716.5 | 648.3 | opus | 612.9 | 956.1 | 192x128x64 |
| 1024x6144x3072 | bfloat16 | 96 | one-tile owner, n = 2-3 | 705.6 | 691.0 | flydsl | 874.8 | 944.4 | 192x128x128 |
| 2048x6144x3072 | float32 | 192 | data-parallel | 1124.6 | 1127.3 | asm | 1078.8 | 1108.5 | 192x256x64 |
| 2048x6144x3072 | bfloat16 | 192 | data-parallel | 1230.2 | 1230.8 | flydsl | 1185.4 | 1176.8 | 192x256x64 |
| 4096x6144x3072 | float32 | 384 | data-parallel | 1140.7 | 1146.7 | asm | 1100.5 | 1148.6 | 192x256x64 |
| 4096x6144x3072 | bfloat16 | 384 | data-parallel | 1242.7 | 1249.5 | flydsl | 1072.3 | 1213.1 | 192x256x64 |
| 8192x6144x3072 | float32 | 768 | data-parallel | 1302.8 | 1298.5 | asm | 1248.1 | 1183.8 | 256x256x64 |
| 8192x6144x3072 | bfloat16 | 768 | data-parallel | 1424.7 | 1432.7 | flydsl | 847.9 | 1385.3 | 256x256x64 |
| 16384x6144x3072 | float32 | 1536 | data-parallel | 1313.3 | 1300.5 | asm | 1271.0 | 1220.7 | 256x256x64 |
| 16384x6144x3072 | bfloat16 | 1536 | data-parallel | 1443.5 | 1439.5 | torch | 1404.3 | 1408.6 | 256x256x64 |
| 32768x6144x3072 | bfloat16 | 3072 | data-parallel | 1468.9 | 1454.0 | flydsl | 860.8 | 1437.2 | 256x256x64 |
| 512x5120x3200 | bfloat16 | 40 | one-tile reduce-scatter, n = 4 | 485.7 | 317.2 | flydsl | 625.6 | 690.1 | 160x64x128 |
| 1024x5120x3200 | bfloat16 | 80 | one-tile owner, n = 3-4 | 719.8 | 594.4 | torch | 926.2 | 927.6 | 160x128x128 |
| 2048x5120x3200 | bfloat16 | 160 | data-parallel | 1121.6 | 1122.4 | torch | 1106.9 | 1109.5 | 160x256x64 |
| 4096x5120x3200 | bfloat16 | 320 | one-tile owner, n = 2 | 1176.8 | 1063.5 | torch | 1130.2 | 1134.9 | 160x256x64 |
| 8192x5120x3200 | bfloat16 | 640 | data-parallel | 1319.3 | 1319.3 | asm | 1262.3 | 1240.4 | 256x224x64 |
| 16384x5120x3200 | bfloat16 | 1280 | data-parallel | 1440.2 | 1434.3 | torch | 1394.1 | 1397.4 | 256x256x64 |
| 32768x5120x3200 | bfloat16 | 2560 | data-parallel | 1482.7 | 1461.8 | torch | 1416.5 | 1420.7 | 256x256x64 |
| 512x7168x3584 | bfloat16 | 56 | one-tile reduce-scatter, n = 3 | 613.7 | 438.3 | flydsl | 835.5 | 701.3 | 128x128x256 |
| 512x8192x3584 | bfloat16 | 64 | one-tile reduce-scatter, n = 3 | 667.4 | 493.2 | flydsl | 783.1 | 772.0 | 128x128x256 |
| 1024x7168x3584 | bfloat16 | 112 | one-tile owner, n = 2-3 | 885.1 | 816.2 | torch | 1032.3 | 1035.2 | 224x128x64 |
| 1024x8192x3584 | bfloat16 | 128 | one-tile owner, n = 2 | 1021.5 | 909.2 | flydsl | 883.1 | 1084.3 | 256x128x64 |
| 2048x8192x3584 | bfloat16 | 256 | data-parallel | 1375.5 | 1374.8 | flydsl | 1304.6 | 1337.0 | 256x256x64 |
| 4096x8192x3584 | bfloat16 | 512 | data-parallel | 1417.4 | 1417.0 | flydsl | 1027.3 | 1371.3 | 256x256x64 |
| 8192x8192x3584 | bfloat16 | 1024 | data-parallel | 1441.1 | 1434.6 | torch | 1393.2 | 1393.6 | 256x256x64 |
| 10240x7168x3584 | bfloat16 | 1120 | one-tile owner, n = 2-3 | 1365.2 | 1346.3 | flydsl | 826.1 | 1256.5 | 224x256x64 |
| 16384x8192x3584 | bfloat16 | 2048 | data-parallel | 1469.9 | 1459.9 | flydsl | 845.5 | 1416.8 | 256x256x64 |
| 32768x8192x3584 | bfloat16 | 4096 | data-parallel | 1491.6 | 1476.1 | flydsl | 869.4 | 1451.2 | 256x256x64 |
| 512x1024x4096 | bfloat16 | 8 | one-tile reduce-scatter, n = 16 | 196.1 | 66.7 | flydsl | 341.6 | 329.4 | 64x32x256 |
| 512x2048x4096 | bfloat16 | 16 | one-tile reduce-scatter, n = 12 | 310.6 | 133.3 | flydsl | 515.7 | 509.0 | 64x64x256 |
| 512x4096x4096 | bfloat16 | 32 | one-tile reduce-scatter, n = 6 | 496.7 | 263.2 | flydsl | 703.7 | 626.5 | 128x128x128 |
| 512x8192x4096 | bfloat16 | 64 | one-tile reduce-scatter, n = 3 | 733.6 | 506.5 | flydsl | 877.7 | 801.9 | 128x128x256 |
| 512x8704x4096 | bfloat16 | 68 | one-tile owner, n = 3-4 | 715.3 | 538.9 | torch | 892.2 | 897.7 | 160x128x128 |
| 512x10240x4096 | bfloat16 | 80 | one-tile owner, n = 3-4 | 821.6 | 622.0 | torch | 875.2 | 876.8 | 128x192x64 |
| 512x16384x4096 | bfloat16 | 128 | one-tile owner, n = 2 | 1059.9 | 940.2 | flydsl | 802.3 | 699.0 | 256x256x64 |
| 512x129280x4096 | bfloat16 | 1010 | data-parallel | 1352.8 | 1351.5 | asm | 1287.8 | 1147.3 | 256x256x64 |
| 768x1024x4096 | bfloat16 | 12 | one-tile reduce-scatter, n = 16 | 238.6 | 100.4 | torch | 431.0 | 432.1 | 64x48x256 |
| 768x2048x4096 | bfloat16 | 24 | one-tile reduce-scatter, n = 8 | 410.5 | 199.6 | flydsl | 551.9 | 564.2 | 64x96x256 |
| 768x4096x4096 | bfloat16 | 48 | one-tile reduce-scatter, n = 4 | 628.7 | 392.7 | flydsl | 757.1 | 796.5 | 128x96x128 |
| 768x8192x4096 | bfloat16 | 96 | one-tile owner, n = 2-3 | 891.1 | 740.6 | torch | 981.3 | 983.0 | 128x192x128 |
| 768x8704x4096 | bfloat16 | 102 | one-tile owner, n = 2-3 | 916.0 | 782.7 | flydsl | 948.0 | 947.6 | 160x192x128 |
| 768x10240x4096 | bfloat16 | 120 | one-tile owner, n = 2-3 | 1033.4 | 916.1 | flydsl | 1141.9 | 1047.6 | 160x192x128 |
| 768x129280x4096 | bfloat16 | 1515 | data-parallel | 1428.1 | 1398.7 | torch | 1349.4 | 1356.1 | 256x256x64 |
| 1024x512x4096 | bfloat16 | 8 | one-tile reduce-scatter, n = 16 | 196.4 | 66.7 | flydsl | 330.7 | 329.3 | 64x32x256 |
| 1024x1024x4096 | bfloat16 | 16 | one-tile reduce-scatter, n = 12 | 310.4 | 133.2 | flydsl | 498.2 | 524.4 | 64x64x256 |
| 1024x2048x4096 | bfloat16 | 32 | one-tile reduce-scatter, n = 6 | 492.9 | 264.7 | flydsl | 696.4 | 536.5 | 128x128x256 |
| 1024x4096x4096 | bfloat16 | 64 | one-tile reduce-scatter, n = 3 | 738.9 | 507.2 | flydsl | 895.6 | 801.2 | 128x128x256 |
| 1024x8192x4096 | bfloat16 | 128 | one-tile owner, n = 2 | 1058.0 | 951.2 | flydsl | 1152.8 | 717.2 | 256x256x64 |
| 1024x8704x4096 | bfloat16 | 136 | data-parallel | 1047.7 | 1037.7 | torch | 1069.4 | 1068.6 | 144x256x64 |
| 1024x10240x4096 | bfloat16 | 160 | data-parallel | 1165.2 | 1170.7 | torch | 1138.6 | 1139.9 | 160x256x64 |
| 1024x16384x4096 | bfloat16 | 256 | data-parallel | 1402.9 | 1406.0 | flydsl | 896.5 | 1372.0 | 256x256x64 |
| 1024x129280x4096 | bfloat16 | 2020 | data-parallel | 1442.5 | 1429.5 | torch | 1372.5 | 1380.5 | 256x256x64 |
| 1280x512x4096 | bfloat16 | 10 | one-tile reduce-scatter, n = 16 | 218.4 | 83.5 | flydsl | 335.9 | 376.0 | 64x48x256 |
| 1280x1024x4096 | bfloat16 | 20 | one-tile reduce-scatter, n = 9 | 364.5 | 165.9 | torch | 549.9 | 552.6 | 64x80x256 |
| 1280x2048x4096 | bfloat16 | 40 | one-tile reduce-scatter, n = 4 | 570.4 | 329.6 | flydsl | 651.7 | 714.4 | 128x80x128 |
| 1280x129280x4096 | bfloat16 | 2525 | data-parallel | 1412.5 | 1404.3 | torch | 1388.9 | 1399.0 | 256x256x64 |
| 1536x512x4096 | bfloat16 | 12 | one-tile reduce-scatter, n = 16 | 236.3 | 100.3 | torch | 430.7 | 431.9 | 64x48x256 |
| 1536x1024x4096 | bfloat16 | 24 | one-tile reduce-scatter, n = 8 | 408.6 | 200.1 | flydsl | 545.1 | 564.4 | 64x96x256 |
| 1536x2048x4096 | bfloat16 | 48 | one-tile reduce-scatter, n = 4 | 629.0 | 391.1 | flydsl | 757.9 | 796.9 | 128x96x128 |
| 1536x129280x4096 | bfloat16 | 3030 | data-parallel | 1433.6 | 1417.1 | torch | 1389.4 | 1396.9 | 256x256x64 |
| 1792x512x4096 | bfloat16 | 14 | one-tile reduce-scatter, n = 13 | 270.9 | 117.1 | flydsl | 418.2 | 473.9 | 64x64x256 |
| 1792x1024x4096 | bfloat16 | 28 | one-tile reduce-scatter, n = 6 | 457.3 | 232.3 | flydsl | 606.0 | 501.4 | 256x128x64 |
| 1792x2048x4096 | bfloat16 | 56 | one-tile reduce-scatter, n = 3 | 675.7 | 449.8 | flydsl | 823.7 | 724.4 | 128x128x256 |
| 1792x129280x4096 | bfloat16 | 3535 | data-parallel | 1447.7 | 1440.6 | asm | 1414.5 | 1361.4 | 256x256x64 |
| 2048x512x4096 | bfloat16 | 16 | one-tile reduce-scatter, n = 12 | 310.2 | 133.2 | flydsl | 478.0 | 521.5 | 64x64x256 |
| 2048x1024x4096 | bfloat16 | 32 | one-tile reduce-scatter, n = 6 | 492.0 | 264.0 | flydsl | 700.8 | 542.0 | 128x128x256 |
| 2048x2048x4096 | bfloat16 | 64 | one-tile reduce-scatter, n = 3 | 735.7 | 506.2 | flydsl | 877.9 | 805.2 | 128x128x256 |
| 2048x4096x4096 | bfloat16 | 128 | one-tile owner, n = 2 | 1059.2 | 971.9 | opus | 1137.3 | 737.8 | 256x256x64 |
| 2048x8192x4096 | bfloat16 | 256 | data-parallel | 1400.7 | 1396.4 | flydsl | 1324.4 | 1360.4 | 256x256x64 |
| 2048x8704x4096 | bfloat16 | 272 | one-tile reduce-scatter, n = 8 | 1224.7 | 931.1 | torch | 1069.2 | 1088.0 | 224x160x64 |
| 2048x10240x4096 | bfloat16 | 320 | one-tile owner, n = 2 | 1220.6 | 1086.8 | torch | 1155.8 | 1159.5 | 160x256x64 |
| 2048x16384x4096 | bfloat16 | 512 | data-parallel | 1427.2 | 1426.6 | flydsl | 1353.0 | 1378.6 | 256x256x64 |
| 2048x129280x4096 | bfloat16 | 4040 | data-parallel | 1481.8 | 1473.2 | flydsl | 1421.6 | 1381.8 | 256x256x64 |
| 2304x512x4096 | bfloat16 | 18 | one-tile reduce-scatter, n = 10 | 327.3 | 149.7 | flydsl | 438.7 | 518.8 | 64x80x256 |
| 2304x1024x4096 | bfloat16 | 36 | one-tile reduce-scatter, n = 5 | 530.6 | 294.1 | flydsl | 643.2 | 713.9 | 128x80x128 |
| 2304x2048x4096 | bfloat16 | 72 | one-tile owner, n = 3-4 | 780.4 | 564.5 | opus | 866.7 | 933.6 | 128x160x128 |
| 2560x512x4096 | bfloat16 | 20 | one-tile reduce-scatter, n = 9 | 361.0 | 165.9 | flydsl | 481.2 | 554.5 | 64x80x256 |
| 2560x1024x4096 | bfloat16 | 40 | one-tile reduce-scatter, n = 4 | 562.1 | 329.6 | flydsl | 686.7 | 714.5 | 128x80x128 |
| 2560x2048x4096 | bfloat16 | 80 | one-tile owner, n = 3-4 | 818.6 | 620.0 | opus | 960.5 | 968.9 | 128x160x128 |
| 2816x512x4096 | bfloat16 | 22 | one-tile reduce-scatter, n = 8 | 393.9 | 182.7 | flydsl | 509.7 | 528.0 | 64x96x256 |
| 2816x1024x4096 | bfloat16 | 44 | one-tile reduce-scatter, n = 4 | 596.5 | 361.6 | flydsl | 721.6 | 749.4 | 128x96x128 |
| 2816x2048x4096 | bfloat16 | 88 | one-tile owner, n = 2-3 | 833.2 | 675.5 | opus | 1001.0 | 894.6 | 192x128x128 |
| 3072x512x4096 | bfloat16 | 24 | one-tile reduce-scatter, n = 8 | 407.5 | 199.3 | flydsl | 559.4 | 565.8 | 64x96x256 |
| 3072x1024x4096 | bfloat16 | 48 | one-tile reduce-scatter, n = 4 | 628.1 | 391.0 | flydsl | 750.0 | 794.9 | 128x96x128 |
| 3072x2048x4096 | bfloat16 | 96 | one-tile owner, n = 2-3 | 807.0 | 727.8 | opus | 1068.2 | 983.3 | 128x192x128 |
| 3328x512x4096 | bfloat16 | 26 | one-tile reduce-scatter, n = 7 | 430.4 | 215.5 | flydsl | 585.0 | 472.8 | 256x112x64 |
| 3328x1024x4096 | bfloat16 | 52 | one-tile reduce-scatter, n = 3 | 641.3 | 423.1 | flydsl | 767.7 | 693.8 | 128x128x256 |
| 3328x2048x4096 | bfloat16 | 104 | one-tile owner, n = 2-3 | 926.2 | 788.0 | opus | 1067.8 | 798.2 | 256x208x64 |
| 3584x512x4096 | bfloat16 | 28 | one-tile reduce-scatter, n = 6 | 454.6 | 233.5 | flydsl | 605.9 | 471.2 | 256x128x64 |
| 3584x1024x4096 | bfloat16 | 56 | one-tile reduce-scatter, n = 3 | 677.0 | 450.8 | flydsl | 786.3 | 736.9 | 128x128x256 |
| 3584x2048x4096 | bfloat16 | 112 | one-tile owner, n = 2-3 | 928.9 | 829.6 | opus | 1101.8 | 935.7 | 160x192x128 |
| 3840x512x4096 | bfloat16 | 30 | one-tile reduce-scatter, n = 6 | 471.0 | 248.7 | flydsl | 620.8 | 531.1 | 128x128x256 |
| 3840x1024x4096 | bfloat16 | 60 | one-tile reduce-scatter, n = 3 | 699.0 | 480.3 | flydsl | 815.5 | 773.6 | 128x128x256 |
| 3840x2048x4096 | bfloat16 | 120 | one-tile owner, n = 2-3 | 1064.9 | 896.8 | opus | 1117.6 | 771.5 | 256x240x64 |
| 4096x512x4096 | bfloat16 | 32 | one-tile reduce-scatter, n = 6 | 493.6 | 263.0 | flydsl | 679.7 | 553.2 | 128x128x256 |
| 4096x1024x4096 | bfloat16 | 64 | one-tile reduce-scatter, n = 3 | 737.7 | 503.6 | flydsl | 880.5 | 805.2 | 128x128x256 |
| 4096x2048x4096 | bfloat16 | 128 | one-tile owner, n = 2 | 1057.8 | 957.7 | opus | 1144.0 | 731.3 | 256x256x64 |
| 4096x4096x4096 | bfloat16 | 256 | data-parallel | 1401.8 | 1400.6 | flydsl | 1327.9 | 1346.7 | 256x256x64 |
| 4096x8192x4096 | bfloat16 | 512 | data-parallel | 1425.0 | 1425.6 | asm | 1374.6 | 1379.7 | 256x256x64 |
| 4096x8704x4096 | bfloat16 | 544 | one-tile reduce-scatter, n = 4 | 1335.9 | 1155.1 | torch | 1194.9 | 1177.7 | 192x256x64 |
| 4096x10240x4096 | bfloat16 | 640 | one-tile owner, n = 2 | 1375.8 | 1347.2 | flydsl | 1290.6 | 1226.3 | 224x256x64 |
| 4096x16384x4096 | bfloat16 | 1024 | data-parallel | 1451.5 | 1447.2 | flydsl | 1381.5 | 1407.0 | 256x256x64 |
| 4096x129280x4096 | bfloat16 | 8080 | data-parallel | 1485.6 | 1483.4 | asm | 1417.2 | 1397.5 | 256x256x64 |
| 4352x512x4096 | bfloat16 | 34 | one-tile reduce-scatter, n = 5 | 514.3 | 280.8 | flydsl | 533.8 | 633.5 | 128x80x128 |
| 4352x1024x4096 | bfloat16 | 68 | one-tile owner, n = 3-4 | 740.1 | 533.8 | opus | 832.5 | 894.3 | 128x160x128 |
| 4352x2048x4096 | bfloat16 | 136 | data-parallel | 1004.1 | 1015.6 | torch | 1041.6 | 1042.5 | 192x192x64 |
| 5120x512x4096 | bfloat16 | 40 | one-tile reduce-scatter, n = 4 | 560.8 | 328.0 | flydsl | 568.2 | 714.6 | 128x80x128 |
| 5120x1024x4096 | bfloat16 | 80 | one-tile owner, n = 3-4 | 816.0 | 616.4 | opus | 988.1 | 971.5 | 128x160x128 |
| 5120x2048x4096 | bfloat16 | 160 | data-parallel | 1156.2 | 1162.0 | torch | 1175.4 | 1178.3 | 256x160x64 |
| 6144x512x4096 | bfloat16 | 48 | one-tile reduce-scatter, n = 4 | 627.6 | 392.4 | flydsl | 759.8 | 797.1 | 128x96x128 |
| 6144x1024x4096 | bfloat16 | 96 | one-tile owner, n = 2-3 | 863.8 | 728.4 | opus | 1070.8 | 994.5 | 128x192x128 |
| 6144x2048x4096 | bfloat16 | 192 | data-parallel | 1269.2 | 1273.5 | opus | 1256.5 | 1212.7 | 256x192x64 |
| 6144x129280x4096 | bfloat16 | 12120 | one-tile owner, n = 2-3 | 1487.5 | 1467.4 | asm | 1412.7 | 1403.8 | 256x256x64 |
| 6400x512x4096 | bfloat16 | 50 | one-tile reduce-scatter, n = 3 | 613.6 | 406.4 | flydsl | 580.9 | 665.3 | 128x128x256 |
| 6400x1024x4096 | bfloat16 | 100 | one-tile owner, n = 2-3 | 913.4 | 758.7 | opus | 1057.9 | 804.8 | 256x208x64 |
| 6400x2048x4096 | bfloat16 | 200 | data-parallel | 1286.3 | 1289.3 | torch | 1151.1 | 1152.4 | 256x208x64 |
| 6912x512x4096 | bfloat16 | 54 | one-tile reduce-scatter, n = 3 | 662.2 | 436.9 | flydsl | 587.0 | 709.3 | 128x128x256 |
| 6912x1024x4096 | bfloat16 | 108 | one-tile owner, n = 2-3 | 967.5 | 837.3 | opus | 1095.0 | 699.1 | 256x224x64 |
| 6912x2048x4096 | bfloat16 | 216 | data-parallel | 1321.5 | 1326.9 | torch | 1226.0 | 1226.8 | 256x224x64 |
| 7168x512x4096 | bfloat16 | 56 | one-tile reduce-scatter, n = 3 | 673.9 | 448.4 | flydsl | 587.0 | 729.7 | 128x128x256 |
| 7168x1024x4096 | bfloat16 | 112 | one-tile owner, n = 2-3 | 999.4 | 843.1 | opus | 1118.8 | 1055.7 | 256x128x64 |
| 7168x2048x4096 | bfloat16 | 224 | data-parallel | 1345.7 | 1346.9 | torch | 1243.6 | 1247.5 | 256x224x64 |
| 7424x512x4096 | bfloat16 | 58 | one-tile reduce-scatter, n = 3 | 689.3 | 464.4 | flydsl | 588.8 | 747.8 | 128x128x256 |
| 7424x1024x4096 | bfloat16 | 116 | one-tile owner, n = 2-3 | 1029.4 | 887.4 | opus | 1113.6 | 777.7 | 256x240x64 |
| 7424x2048x4096 | bfloat16 | 232 | data-parallel | 1361.2 | 1361.4 | torch | 1180.9 | 1183.6 | 256x240x64 |
| 7680x512x4096 | bfloat16 | 60 | one-tile reduce-scatter, n = 3 | 707.6 | 476.8 | flydsl | 593.3 | 768.5 | 128x128x256 |
| 7680x1024x4096 | bfloat16 | 120 | one-tile owner, n = 2-3 | 1032.4 | 923.2 | opus | 1137.9 | 778.8 | 256x240x64 |
| 7680x2048x4096 | bfloat16 | 240 | data-parallel | 1375.8 | 1378.1 | torch | 1197.2 | 1200.0 | 256x240x64 |
| 7936x512x4096 | bfloat16 | 62 | one-tile reduce-scatter, n = 3 | 723.2 | 493.2 | flydsl | 802.5 | 787.3 | 128x128x256 |
| 7936x1024x4096 | bfloat16 | 124 | one-tile owner, n = 2-3 | 1061.6 | 942.7 | opus | 1137.2 | 716.2 | 256x256x64 |
| 7936x2048x4096 | bfloat16 | 248 | data-parallel | 1390.3 | 1388.2 | torch | 1353.8 | 1356.3 | 256x256x64 |
| 8192x512x4096 | bfloat16 | 64 | one-tile reduce-scatter, n = 3 | 737.9 | 504.6 | flydsl | 865.2 | 806.0 | 128x128x256 |
| 8192x1024x4096 | bfloat16 | 128 | one-tile owner, n = 2 | 1061.8 | 956.7 | opus | 1158.5 | 691.2 | 256x256x64 |
| 8192x2048x4096 | bfloat16 | 256 | data-parallel | 1409.2 | 1409.0 | torch | 1372.2 | 1373.6 | 256x256x64 |
| 8192x4096x4096 | bfloat16 | 512 | data-parallel | 1434.1 | 1435.2 | flydsl | 1362.4 | 1394.3 | 256x256x64 |
| 8192x8192x4096 | bfloat16 | 1024 | data-parallel | 1456.4 | 1452.1 | asm | 1403.9 | 1408.3 | 256x256x64 |
| 8192x8704x4096 | bfloat16 | 1088 | one-tile owner, n = 2 | 1389.1 | 1329.8 | asm | 1290.7 | 1277.6 | 256x224x64 |
| 8192x10240x4096 | bfloat16 | 1280 | data-parallel | 1465.6 | 1464.9 | torch | 1421.3 | 1421.7 | 256x256x64 |
| 8192x16384x4096 | bfloat16 | 2048 | data-parallel | 1489.4 | 1485.8 | torch | 1431.9 | 1450.0 | 256x256x64 |
| 8192x129280x4096 | bfloat16 | 16160 | one-tile reduce-scatter, n = 4 | 1499.9 | 1490.7 | flydsl | 1420.3 | 1394.9 | 256x256x64 |
| 10240x512x4096 | bfloat16 | 80 | one-tile owner, n = 3-4 | 822.0 | 615.1 | torch | 977.5 | 980.6 | 128x160x128 |
| 10240x1024x4096 | bfloat16 | 160 | data-parallel | 1158.5 | 1159.8 | torch | 1185.6 | 1191.3 | 256x160x64 |
| 10240x2048x4096 | bfloat16 | 320 | one-tile owner, n = 2 | 1229.3 | 1092.5 | torch | 1203.9 | 1206.1 | 256x160x64 |
| 10240x129280x4096 | bfloat16 | 20200 | data-parallel | 1465.8 | 1468.0 | asm | 1435.3 | 1377.3 | 256x256x64 |
| 12288x512x4096 | bfloat16 | 96 | one-tile owner, n = 2-3 | 808.0 | 727.9 | opus | 1087.6 | 991.9 | 128x192x128 |
| 12288x1024x4096 | bfloat16 | 192 | data-parallel | 1277.0 | 1281.7 | opus | 1271.5 | 1226.9 | 256x192x64 |
| 12288x2048x4096 | bfloat16 | 384 | one-tile owner, n = 2 | 1322.3 | 1287.1 | opus | 1288.5 | 1250.1 | 256x192x64 |
| 12288x129280x4096 | bfloat16 | 24240 | data-parallel | 1489.0 | 1497.2 | asm | 1436.7 | 1388.6 | 256x256x64 |
| 14336x512x4096 | bfloat16 | 112 | one-tile owner, n = 2-3 | 1002.4 | 853.8 | opus | 1139.6 | 675.5 | 256x224x64 |
| 14336x1024x4096 | bfloat16 | 224 | data-parallel | 1356.8 | 1361.1 | torch | 1263.2 | 1264.7 | 256x224x64 |
| 14336x2048x4096 | bfloat16 | 448 | data-parallel | 1376.4 | 1380.8 | flydsl | 1320.7 | 1291.3 | 256x224x64 |
| 14336x129280x4096 | bfloat16 | 28280 | one-tile owner, n = 2-3 | 1445.7 | 1485.4 | asm | 1422.3 | 1384.5 | 256x256x64 |
| 16384x512x4096 | bfloat16 | 128 | one-tile owner, n = 2 | 1065.6 | 951.5 | opus | 1173.4 | 785.3 | 256x256x64 |
| 16384x1024x4096 | bfloat16 | 256 | data-parallel | 1420.6 | 1420.0 | flydsl | 1339.4 | 1323.0 | 256x256x64 |
| 16384x2048x4096 | bfloat16 | 512 | data-parallel | 1443.4 | 1447.2 | torch | 1385.6 | 1393.2 | 256x256x64 |
| 16384x16384x4096 | bfloat16 | 4096 | data-parallel | 1513.2 | 1510.9 | torch | 1449.7 | 1499.9 | 256x256x64 |
| 16384x129280x4096 | bfloat16 | 32320 | one-tile owner, n = 2 | 1466.3 | 1491.7 | asm | 1424.4 | 1396.3 | 256x256x64 |
| 32768x16384x4096 | bfloat16 | 8192 | data-parallel | 1513.4 | 1506.5 | flydsl | 1421.7 | 1469.3 | 256x256x64 |
| 512x7168x4224 | bfloat16 | 56 | one-tile reduce-scatter, n = 3 | 660.8 | 449.7 | flydsl | 828.1 | 856.7 | 128x128x128 |
| 1024x7168x4224 | bfloat16 | 112 | one-tile owner, n = 2-3 | 1024.2 | 846.0 | flydsl | 1110.9 | 818.0 | 240x256x64 |
| 10240x7168x4224 | bfloat16 | 1120 | one-tile owner, n = 2-3 | 1404.3 | 1356.4 | opus | 948.4 | 1266.6 | 224x256x64 |
| 512x6144x4608 | bfloat16 | 48 | one-tile reduce-scatter, n = 4 | 665.8 | 397.0 | flydsl | 781.4 | 671.4 | 192x128x128 |
| 1024x6144x4608 | bfloat16 | 96 | one-tile owner, n = 2-3 | 913.0 | 745.9 | torch | 987.1 | 990.1 | 192x128x128 |
| 2048x6144x4608 | bfloat16 | 192 | data-parallel | 1281.1 | 1283.6 | flydsl | 1232.3 | 1217.9 | 192x256x64 |
| 4096x6144x4608 | bfloat16 | 384 | one-tile owner, n = 2 | 1329.7 | 1290.5 | torch | 1234.0 | 1237.9 | 192x256x64 |
| 8192x6144x4608 | bfloat16 | 768 | data-parallel | 1452.5 | 1451.3 | torch | 1409.8 | 1412.8 | 256x256x64 |
| 16384x6144x4608 | bfloat16 | 1536 | data-parallel | 1473.0 | 1471.1 | torch | 1427.0 | 1429.1 | 256x256x64 |
| 32768x6144x4608 | bfloat16 | 3072 | data-parallel | 1507.9 | 1495.5 | torch | 1453.5 | 1467.8 | 256x256x64 |
| 512x1024x5120 | float32 | 8 | one-tile reduce-scatter, n = 20 | 208.1 | 65.6 | asm | 333.4 | 338.5 | 64x32x256 |
| 512x5120x5120 | bfloat16 | 40 | one-tile reduce-scatter, n = 4 | 629.0 | 334.9 | torch | 767.8 | 770.1 | 96x128x128 |
| 512x6400x5120 | bfloat16 | 50 | one-tile reduce-scatter, n = 3 | 667.8 | 416.1 | flydsl | 700.0 | 700.7 | 128x128x256 |
| 512x12800x5120 | bfloat16 | 100 | one-tile owner, n = 2-3 | 987.0 | 798.7 | flydsl | 779.8 | 813.3 | 208x256x64 |
| 512x37888x5120 | bfloat16 | 296 | one-tile owner, n = 3 | 1153.9 | 938.1 | torch | 1049.3 | 1050.5 | 160x256x64 |
| 512x51200x5120 | bfloat16 | 400 | data-parallel | 1200.0 | 1200.4 | flydsl | 1132.9 | 1082.4 | 208x256x64 |
| 768x1024x5120 | float32 | 12 | one-tile reduce-scatter, n = 16 | 280.6 | 98.1 | asm | 407.4 | 380.9 | 64x64x256 |
| 768x37888x5120 | bfloat16 | 444 | data-parallel | 1310.9 | 1311.8 | torch | 1215.8 | 1224.3 | 224x256x64 |
| 1024x1024x5120 | float32 | 16 | one-tile reduce-scatter, n = 12 | 339.7 | 130.6 | asm | 462.8 | 483.6 | 64x64x256 |
| 1024x5120x5120 | bfloat16 | 80 | one-tile owner, n = 3-4 | 884.6 | 641.1 | torch | 973.0 | 974.6 | 160x128x128 |
| 1024x6400x5120 | bfloat16 | 100 | one-tile owner, n = 2-3 | 986.6 | 794.8 | flydsl | 939.0 | 856.0 | 256x208x64 |
| 1024x12800x5120 | bfloat16 | 200 | data-parallel | 1307.3 | 1309.7 | flydsl | 757.1 | 1169.4 | 256x208x64 |
| 1024x37888x5120 | bfloat16 | 592 | one-tile owner, n = 2 | 1316.7 | 1207.2 | torch | 1063.6 | 1064.0 | 256x208x64 |
| 1024x51200x5120 | bfloat16 | 800 | one-tile reduce-scatter, n = 4 | 1362.2 | 1199.5 | asm | 1148.7 | 1070.3 | 256x208x64 |
| 1536x1024x5120 | float32 | 24 | one-tile reduce-scatter, n = 8 | 452.2 | 194.4 | asm | 582.2 | 600.5 | 64x96x128 |
| 1536x37888x5120 | bfloat16 | 888 | one-tile owner, n = 2-3 | 1383.0 | 1322.7 | torch | 1244.4 | 1247.4 | 224x256x64 |
| 2048x1024x5120 | float32 | 32 | one-tile reduce-scatter, n = 6 | 540.9 | 256.2 | opus | 637.9 | 629.7 | 128x128x128 |
| 2048x5120x5120 | bfloat16 | 160 | data-parallel | 1191.9 | 1194.1 | torch | 1159.1 | 1160.8 | 160x256x64 |
| 2048x6400x5120 | bfloat16 | 200 | data-parallel | 1311.3 | 1314.4 | flydsl | 1226.9 | 1177.2 | 256x208x64 |
| 2048x12800x5120 | bfloat16 | 400 | data-parallel | 1329.4 | 1331.4 | flydsl | 834.7 | 1198.4 | 256x208x64 |
| 2048x37888x5120 | bfloat16 | 1184 | data-parallel | 1430.2 | 1423.7 | torch | 1194.8 | 1198.6 | 240x256x64 |
| 2048x51200x5120 | bfloat16 | 1600 | one-tile owner, n = 2 | 1436.5 | 1379.5 | asm | 222.9 | 1172.5 | 240x256x64 |
| 4096x1024x5120 | float32 | 64 | one-tile reduce-scatter, n = 3 | 753.8 | 483.2 | opus | 947.4 | 888.1 | 128x128x128 |
| 4096x5120x5120 | bfloat16 | 320 | one-tile owner, n = 2 | 1260.4 | 1098.9 | torch | 1180.0 | 1182.0 | 160x256x64 |
| 4096x6400x5120 | bfloat16 | 400 | data-parallel | 1325.2 | 1332.5 | flydsl | 1285.3 | 1202.8 | 256x208x64 |
| 4096x12800x5120 | bfloat16 | 800 | one-tile reduce-scatter, n = 4 | 1404.1 | 1253.1 | asm | 1212.9 | 1197.9 | 256x208x64 |
| 4096x51200x5120 | bfloat16 | 3200 | one-tile owner, n = 2 | 1490.7 | 1463.2 | asm | 1380.2 | 1382.1 | 256x256x64 |
| 8192x1024x5120 | float32 | 128 | one-tile owner, n = 2 | 1060.5 | 976.4 | opus | 1166.6 | 1079.5 | 256x128x64 |
| 8192x5120x5120 | bfloat16 | 640 | one-tile owner, n = 2 | 1404.3 | 1366.6 | torch | 1275.6 | 1279.3 | 256x224x64 |
| 8192x6400x5120 | bfloat16 | 800 | one-tile reduce-scatter, n = 4 | 1405.5 | 1254.7 | flydsl | 1211.5 | 1201.2 | 256x208x64 |
| 8192x12800x5120 | bfloat16 | 1600 | one-tile owner, n = 2 | 1443.1 | 1387.5 | asm | 1335.3 | 1204.1 | 256x240x64 |
| 8192x51200x5120 | bfloat16 | 6400 | data-parallel | 1532.2 | 1528.5 | flydsl | 1450.3 | 1418.9 | 256x256x64 |
| 16384x1024x5120 | float32 | 256 | data-parallel | 1358.5 | 1360.7 | opus | 1267.7 | 1240.7 | 256x256x64 |
| 16384x5120x5120 | bfloat16 | 1280 | data-parallel | 1486.2 | 1482.5 | torch | 1443.3 | 1449.5 | 256x256x64 |
| 16384x6400x5120 | bfloat16 | 1600 | one-tile owner, n = 2 | 1450.2 | 1382.9 | flydsl | 1324.8 | 1219.5 | 256x240x64 |
| 16384x12800x5120 | bfloat16 | 3200 | one-tile owner, n = 2 | 1512.1 | 1482.0 | torch | 1461.0 | 1480.7 | 256x256x64 |
| 16384x51200x5120 | bfloat16 | 12800 | data-parallel | 1547.4 | 1525.2 | flydsl | 1478.6 | 1419.8 | 256x256x64 |
| 32768x5120x5120 | bfloat16 | 2560 | data-parallel | 1524.6 | 1515.5 | torch | 1475.8 | 1494.6 | 256x256x64 |
| 32768x6400x5120 | bfloat16 | 3200 | one-tile owner, n = 2 | 1513.4 | 1478.3 | torch | 1471.2 | 1484.1 | 256x256x64 |
| 32768x12800x5120 | bfloat16 | 6400 | data-parallel | 1539.3 | 1522.1 | torch | 1487.5 | 1524.2 | 256x256x64 |
| 32768x51200x5120 | bfloat16 | 25600 | data-parallel | 1504.0 | 1516.3 | flydsl | 1432.3 | 1397.7 | 256x256x64 |
| 512x1280x6144 | bfloat16 | 10 | one-tile reduce-scatter, n = 19 | 277.9 | 86.4 | flydsl | 403.9 | 375.8 | 160x64x128 |
| 512x1536x6144 | bfloat16 | 12 | one-tile reduce-scatter, n = 16 | 313.9 | 103.6 | flydsl | 432.7 | 410.7 | 192x64x128 |
| 512x2048x6144 | float32 | 16 | one-tile reduce-scatter, n = 12 | 392.2 | 133.7 | asm | 491.9 | 509.5 | 64x64x256 |
| 512x2304x6144 | bfloat16 | 18 | one-tile reduce-scatter, n = 10 | 440.8 | 154.5 | flydsl | 511.3 | 409.7 | 192x128x128 |
| 512x2560x6144 | bfloat16 | 20 | one-tile reduce-scatter, n = 9 | 466.5 | 171.4 | torch | 507.0 | 507.8 | 256x128x64 |
| 512x3072x6144 | float32 | 24 | one-tile reduce-scatter, n = 8 | 514.6 | 199.7 | opus | 488.3 | 607.3 | 96x128x128 |
| 512x3072x6144 | bfloat16 | 24 | one-tile reduce-scatter, n = 8 | 525.4 | 206.2 | flydsl | 600.7 | 491.1 | 192x128x128 |
| 512x6144x6144 | bfloat16 | 48 | one-tile reduce-scatter, n = 4 | 769.3 | 406.5 | flydsl | 804.3 | 727.6 | 192x128x128 |
| 512x9216x6144 | bfloat16 | 72 | one-tile owner, n = 3-4 | 899.1 | 596.9 | flydsl | 865.0 | 618.6 | 256x256x64 |
| 1024x2048x6144 | float32 | 32 | one-tile reduce-scatter, n = 6 | 595.6 | 262.4 | opus | 529.6 | 684.0 | 128x128x128 |
| 1024x2304x6144 | bfloat16 | 36 | one-tile reduce-scatter, n = 5 | 672.7 | 305.0 | torch | 664.9 | 662.2 | 256x176x64 |
| 1024x2560x6144 | bfloat16 | 40 | one-tile reduce-scatter, n = 4 | 691.6 | 338.8 | torch | 730.0 | 750.2 | 160x128x128 |
| 1024x3072x6144 | float32 | 48 | one-tile reduce-scatter, n = 4 | 735.7 | 385.3 | opus | 562.4 | 828.9 | 96x128x128 |
| 1024x6144x6144 | bfloat16 | 96 | one-tile owner, n = 2-3 | 1002.3 | 790.8 | torch | 988.4 | 990.5 | 192x256x64 |
| 1024x9216x6144 | bfloat16 | 144 | spread | 1143.2 | 1148.9 | torch | 1127.6 | 1130.3 | 144x256x64 |
| 2048x2048x6144 | float32 | 64 | one-tile reduce-scatter, n = 3 | 813.1 | 499.5 | opus | 874.1 | 916.7 | 128x128x128 |
| 2048x2304x6144 | bfloat16 | 72 | one-tile owner, n = 3-4 | 898.6 | 599.3 | torch | 960.2 | 960.5 | 128x160x128 |
| 2048x2560x6144 | bfloat16 | 80 | one-tile owner, n = 3-4 | 978.3 | 663.1 | torch | 999.9 | 1004.7 | 160x128x128 |
| 2048x3072x6144 | float32 | 96 | one-tile owner, n = 2-3 | 917.3 | 767.2 | opus | 783.1 | 939.5 | 192x256x64 |
| 2048x6144x6144 | bfloat16 | 192 | data-parallel | 1307.1 | 1312.2 | torch | 1238.5 | 1238.0 | 192x256x64 |
| 2048x9216x6144 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 | 1294.6 | 1001.9 | torch | 1142.5 | 1144.2 | 144x256x64 |
| 4096x2048x6144 | float32 | 128 | one-tile owner, n = 2 | 1115.2 | 1004.1 | opus | 1124.2 | 806.9 | 256x256x64 |
| 4096x2304x6144 | bfloat16 | 144 | spread | 1143.5 | 1148.3 | flydsl | 751.6 | 1135.8 | 144x256x64 |
| 4096x2560x6144 | bfloat16 | 160 | data-parallel | 1214.0 | 1220.6 | flydsl | 699.0 | 1187.7 | 160x256x64 |
| 4096x3072x6144 | float32 | 192 | data-parallel | 1256.6 | 1259.6 | asm | 1197.1 | 1210.5 | 192x256x64 |
| 4096x6144x6144 | bfloat16 | 384 | one-tile owner, n = 2 | 1379.2 | 1314.6 | torch | 1259.0 | 1260.9 | 192x256x64 |
| 4096x9216x6144 | bfloat16 | 576 | one-tile owner, n = 3 | 1383.5 | 1245.1 | torch | 1260.2 | 1261.4 | 192x256x64 |
| 8192x2048x6144 | float32 | 256 | data-parallel | 1383.3 | 1384.6 | opus | 1321.8 | 1248.1 | 256x256x64 |
| 8192x2304x6144 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 | 1300.5 | 1005.9 | torch | 1150.8 | 1150.4 | 144x256x64 |
| 8192x2560x6144 | bfloat16 | 320 | one-tile owner, n = 3 | 1318.5 | 1116.1 | asm | 526.3 | 1196.1 | 160x256x64 |
| 8192x3072x6144 | float32 | 384 | one-tile owner, n = 2 | 1311.1 | 1261.2 | asm | 1205.9 | 1232.2 | 192x256x64 |
| 8192x6144x6144 | bfloat16 | 768 | data-parallel | 1478.6 | 1483.1 | flydsl | 1413.4 | 1437.7 | 256x256x64 |
| 8192x9216x6144 | bfloat16 | 1152 | one-tile owner, n = 2 | 1474.8 | 1429.7 | asm | 1385.8 | 1222.6 | 256x240x64 |
| 16384x2048x6144 | float32 | 512 | data-parallel | 1361.2 | 1360.5 | opus | 1304.8 | 1271.4 | 256x256x64 |
| 16384x2304x6144 | bfloat16 | 576 | one-tile owner, n = 3 | 1396.3 | 1250.8 | flydsl | 1021.9 | 1288.2 | 192x256x64 |
| 16384x2560x6144 | bfloat16 | 640 | one-tile owner, n = 2 | 1439.8 | 1377.5 | flydsl | 1324.0 | 1299.2 | 256x224x64 |
| 16384x3072x6144 | float32 | 768 | data-parallel | 1400.1 | 1396.0 | asm | 1377.6 | 1306.4 | 256x256x64 |
| 16384x6144x6144 | bfloat16 | 1536 | data-parallel | 1522.1 | 1512.5 | asm | 1468.6 | 1492.1 | 256x256x64 |
| 16384x9216x6144 | bfloat16 | 2304 | data-parallel | 1527.6 | 1527.5 | torch | 1479.5 | 1524.5 | 256x256x64 |
| 32768x2304x6144 | bfloat16 | 1152 | one-tile owner, n = 2 | 1465.5 | 1408.4 | flydsl | 653.7 | 1204.4 | 256x240x64 |
| 32768x2560x6144 | bfloat16 | 1280 | data-parallel | 1494.6 | 1490.7 | flydsl | 1433.1 | 1484.5 | 256x256x64 |
| 32768x6144x6144 | bfloat16 | 3072 | data-parallel | 1552.7 | 1541.4 | torch | 1494.0 | 1534.7 | 256x256x64 |
| 32768x9216x6144 | bfloat16 | 4608 | data-parallel | 1545.3 | 1541.4 | torch | 1495.5 | 1537.7 | 256x256x64 |
| 512x5120x6400 | bfloat16 | 40 | one-tile reduce-scatter, n = 4 | 712.9 | 341.6 | flydsl | 713.9 | 751.1 | 160x128x128 |
| 1024x5120x6400 | bfloat16 | 80 | one-tile owner, n = 3-4 | 1005.3 | 670.0 | flydsl | 669.7 | 1015.9 | 160x128x128 |
| 2048x5120x6400 | bfloat16 | 160 | data-parallel | 1218.3 | 1219.9 | flydsl | 746.8 | 1184.9 | 160x256x64 |
| 4096x5120x6400 | bfloat16 | 320 | one-tile owner, n = 3 | 1318.1 | 1112.8 | torch | 1196.6 | 1199.7 | 160x256x64 |
| 8192x5120x6400 | bfloat16 | 640 | one-tile owner, n = 2 | 1424.9 | 1380.6 | flydsl | 1323.9 | 1287.3 | 256x224x64 |
| 16384x5120x6400 | bfloat16 | 1280 | data-parallel | 1507.7 | 1499.6 | torch | 1451.0 | 1453.9 | 256x256x64 |
| 32768x5120x6400 | bfloat16 | 2560 | data-parallel | 1530.9 | 1524.9 | torch | 1480.7 | 1503.0 | 256x256x64 |
| 512x16384x6656 | bfloat16 | 128 | one-tile owner, n = 2 | 1161.1 | 1043.9 | flydsl | 920.5 | 907.9 | 256x256x64 |
| 1024x16384x6656 | bfloat16 | 256 | data-parallel | 1447.4 | 1455.6 | asm | 453.1 | 1422.6 | 256x256x64 |
| 2048x16384x6656 | bfloat16 | 512 | data-parallel | 1511.0 | 1482.0 | asm | 1397.9 | 1431.7 | 256x256x64 |
| 4096x16384x6656 | bfloat16 | 1024 | data-parallel | 1504.7 | 1501.7 | torch | 1451.6 | 1456.2 | 256x256x64 |
| 8192x16384x6656 | bfloat16 | 2048 | data-parallel | 1529.2 | 1519.8 | asm | 1471.6 | 1496.9 | 256x256x64 |
| 16384x16384x6656 | bfloat16 | 4096 | data-parallel | 1538.9 | 1537.4 | asm | 1505.6 | 1532.2 | 256x256x64 |
| 32768x16384x6656 | bfloat16 | 8192 | data-parallel | 1542.6 | 1539.6 | asm | 290.7 | 1499.3 | 256x256x64 |
| 512x1024x7168 | bfloat16 | 8 | one-tile reduce-scatter, n = 24 | 252.4 | 70.0 | torch | 423.9 | 429.3 | 128x128x64 |
| 512x1536x7168 | bfloat16 | 12 | one-tile reduce-scatter, n = 16 | 356.2 | 104.9 | torch | 446.8 | 445.7 | 192x64x128 |
| 512x2048x7168 | bfloat16 | 16 | one-tile reduce-scatter, n = 12 | 460.0 | 139.0 | torch | 575.3 | 576.6 | 128x128x128 |
| 512x3584x7168 | bfloat16 | 28 | one-tile reduce-scatter, n = 6 | 628.2 | 242.2 | flydsl | 615.6 | 621.3 | 128x128x256 |
| 512x6144x7168 | bfloat16 | 48 | one-tile reduce-scatter, n = 4 | 825.9 | 411.1 | flydsl | 819.3 | 776.4 | 192x128x128 |
| 512x8192x7168 | bfloat16 | 64 | one-tile reduce-scatter, n = 3 | 923.3 | 543.7 | flydsl | 849.3 | 866.2 | 128x128x256 |
| 512x8448x7168 | bfloat16 | 66 | one-tile owner, n = 3-4 | 930.7 | 560.5 | flydsl | 723.4 | 469.7 | 256x256x64 |
| 512x20480x7168 | bfloat16 | 160 | spread | 1115.8 | 1109.3 | flydsl | 1035.0 | 1149.6 | 160x256x64 |
| 768x1024x7168 | bfloat16 | 12 | one-tile reduce-scatter, n = 16 | 363.2 | 104.9 | torch | 440.7 | 442.4 | 256x96x64 |
| 768x2048x7168 | bfloat16 | 24 | one-tile reduce-scatter, n = 8 | 574.8 | 207.9 | flydsl | 609.4 | 621.9 | 256x128x64 |
| 1024x512x7168 | bfloat16 | 8 | one-tile reduce-scatter, n = 24 | 252.3 | 70.1 | flydsl | 382.6 | 321.0 | 256x64x64 |
| 1024x1024x7168 | bfloat16 | 16 | one-tile reduce-scatter, n = 12 | 460.5 | 139.2 | torch | 602.5 | 585.3 | 128x128x128 |
| 1024x1536x7168 | bfloat16 | 24 | one-tile reduce-scatter, n = 8 | 569.8 | 208.1 | flydsl | 625.7 | 620.6 | 256x128x64 |
| 1024x2048x7168 | bfloat16 | 32 | one-tile reduce-scatter, n = 6 | 675.3 | 275.9 | torch | 749.0 | 751.2 | 256x128x64 |
| 1024x3584x7168 | bfloat16 | 56 | one-tile reduce-scatter, n = 3 | 843.3 | 478.6 | flydsl | 859.7 | 807.7 | 128x128x256 |
| 1024x6144x7168 | bfloat16 | 96 | one-tile owner, n = 2-3 | 925.5 | 799.4 | torch | 977.5 | 984.9 | 192x256x64 |
| 1024x8192x7168 | bfloat16 | 128 | one-tile owner, n = 2 | 1209.6 | 1053.0 | flydsl | 1036.4 | 952.7 | 256x256x64 |
| 1024x8448x7168 | bfloat16 | 132 | spread | 1161.9 | 1090.3 | torch | 1106.9 | 1125.2 | 144x256x64 |
| 1024x20480x7168 | bfloat16 | 320 | one-tile owner, n = 3 | 1290.5 | 1067.3 | flydsl | 780.2 | 1178.7 | 160x256x64 |
| 1280x512x7168 | bfloat16 | 10 | one-tile reduce-scatter, n = 19 | 305.4 | 87.5 | torch | 406.1 | 406.5 | 256x80x64 |
| 1280x1024x7168 | bfloat16 | 20 | one-tile reduce-scatter, n = 9 | 510.9 | 173.4 | torch | 593.8 | 585.7 | 256x80x128 |
| 1280x2048x7168 | bfloat16 | 40 | one-tile reduce-scatter, n = 4 | 732.5 | 344.4 | torch | 810.0 | 807.0 | 128x160x128 |
| 1536x512x7168 | bfloat16 | 12 | one-tile reduce-scatter, n = 16 | 357.3 | 104.9 | torch | 446.1 | 443.8 | 256x96x64 |
| 1536x1024x7168 | bfloat16 | 24 | one-tile reduce-scatter, n = 8 | 570.0 | 208.5 | torch | 719.1 | 717.7 | 128x96x128 |
| 1536x2048x7168 | bfloat16 | 48 | one-tile reduce-scatter, n = 4 | 828.3 | 411.3 | torch | 906.0 | 911.1 | 128x192x64 |
| 1792x512x7168 | bfloat16 | 14 | one-tile reduce-scatter, n = 13 | 395.7 | 122.3 | torch | 485.0 | 484.5 | 256x112x64 |
| 1792x1024x7168 | bfloat16 | 28 | one-tile reduce-scatter, n = 6 | 620.4 | 242.2 | flydsl | 633.7 | 654.5 | 256x128x64 |
| 1792x2048x7168 | bfloat16 | 56 | one-tile reduce-scatter, n = 3 | 841.8 | 476.6 | flydsl | 898.4 | 797.1 | 128x128x256 |
| 2048x512x7168 | bfloat16 | 16 | one-tile reduce-scatter, n = 12 | 461.7 | 139.3 | torch | 572.0 | 570.8 | 128x128x64 |
| 2048x1024x7168 | bfloat16 | 32 | one-tile reduce-scatter, n = 6 | 676.1 | 274.4 | torch | 757.2 | 755.4 | 128x128x128 |
| 2048x2048x7168 | bfloat16 | 64 | one-tile reduce-scatter, n = 3 | 914.7 | 541.6 | flydsl | 949.6 | 865.5 | 128x128x256 |
| 2048x3584x7168 | bfloat16 | 112 | one-tile owner, n = 2-3 | 1078.8 | 933.5 | opus | 1159.6 | 981.2 | 224x256x64 |
| 2048x8192x7168 | bfloat16 | 256 | data-parallel | 1445.3 | 1450.1 | asm | 1390.6 | 1401.4 | 256x256x64 |
| 2048x20480x7168 | bfloat16 | 640 | one-tile owner, n = 2 | 1441.0 | 1377.9 | flydsl | 823.3 | 1250.5 | 224x256x64 |
| 4096x512x7168 | bfloat16 | 32 | one-tile reduce-scatter, n = 6 | 678.3 | 276.3 | flydsl | 703.5 | 675.2 | 128x128x256 |
| 4096x1024x7168 | bfloat16 | 64 | one-tile reduce-scatter, n = 3 | 926.9 | 542.0 | torch | 938.7 | 941.8 | 256x128x64 |
| 4096x2048x7168 | bfloat16 | 128 | one-tile owner, n = 2 | 1213.8 | 1048.8 | torch | 1216.2 | 1220.8 | 256x128x64 |
| 4096x3584x7168 | bfloat16 | 224 | data-parallel | 1395.9 | 1398.7 | flydsl | 1325.5 | 1272.0 | 224x256x64 |
| 4096x8192x7168 | bfloat16 | 512 | data-parallel | 1472.4 | 1476.9 | flydsl | 1402.9 | 1430.9 | 256x256x64 |
| 4096x20480x7168 | bfloat16 | 1280 | data-parallel | 1507.2 | 1508.0 | flydsl | 1439.9 | 1453.3 | 256x256x64 |
| 6144x512x7168 | bfloat16 | 48 | one-tile reduce-scatter, n = 4 | 824.1 | 410.4 | flydsl | 797.8 | 689.3 | 128x192x128 |
| 6144x1024x7168 | bfloat16 | 96 | one-tile owner, n = 2-3 | 1047.9 | 799.0 | opus | 1136.9 | 866.6 | 256x192x64 |
| 6144x2048x7168 | bfloat16 | 192 | data-parallel | 1321.7 | 1324.8 | opus | 1320.5 | 1263.1 | 256x192x64 |
| 8192x512x7168 | bfloat16 | 64 | one-tile reduce-scatter, n = 3 | 922.8 | 542.2 | flydsl | 936.0 | 870.7 | 128x128x256 |
| 8192x1024x7168 | bfloat16 | 128 | one-tile owner, n = 2 | 1222.1 | 1056.6 | torch | 1102.7 | 1104.3 | 256x128x64 |
| 8192x2048x7168 | bfloat16 | 256 | data-parallel | 1466.5 | 1466.2 | torch | 1400.3 | 1406.4 | 256x256x64 |
| 8192x3584x7168 | bfloat16 | 448 | data-parallel | 1416.7 | 1426.7 | flydsl | 1360.8 | 1296.3 | 224x256x64 |
| 8192x8192x7168 | bfloat16 | 1024 | data-parallel | 1505.0 | 1507.0 | flydsl | 1429.5 | 1467.0 | 256x256x64 |
| 8192x20480x7168 | bfloat16 | 2560 | data-parallel | 1533.4 | 1534.9 | asm | 1499.1 | 1474.6 | 256x256x64 |
| 10240x512x7168 | bfloat16 | 80 | one-tile owner, n = 3-4 | 1040.2 | 671.6 | torch | 843.1 | 814.6 | 256x160x64 |
| 10240x1024x7168 | bfloat16 | 160 | spread | 1181.0 | 1225.5 | torch | 1220.2 | 1223.4 | 256x160x64 |
| 10240x2048x7168 | bfloat16 | 320 | one-tile owner, n = 3 | 1343.9 | 1123.5 | torch | 1235.4 | 1235.0 | 256x160x64 |
| 10240x8448x7168 | bfloat16 | 1320 | one-tile owner, n = 4 | 1461.1 | 1347.6 | asm | 1297.0 | 1326.2 | 256x224x64 |
| 10240x20480x7168 | bfloat16 | 3200 | one-tile owner, n = 2 | 1499.9 | 1469.8 | asm | 1430.4 | 1438.3 | 256x256x64 |
| 12288x512x7168 | bfloat16 | 96 | one-tile owner, n = 2-3 | 1043.8 | 801.2 | opus | 1146.1 | 861.4 | 256x192x64 |
| 12288x1024x7168 | bfloat16 | 192 | data-parallel | 1327.6 | 1328.5 | opus | 1326.8 | 1271.6 | 256x192x64 |
| 12288x2048x7168 | bfloat16 | 384 | one-tile owner, n = 2 | 1408.2 | 1330.5 | opus | 1339.7 | 1281.3 | 256x192x64 |
| 14336x512x7168 | bfloat16 | 112 | one-tile owner, n = 2-3 | 1120.0 | 929.7 | opus | 1192.0 | 829.1 | 256x224x64 |
| 14336x1024x7168 | bfloat16 | 224 | data-parallel | 1397.0 | 1406.3 | torch | 1309.5 | 1312.0 | 256x224x64 |
| 14336x2048x7168 | bfloat16 | 448 | data-parallel | 1422.8 | 1420.7 | flydsl | 1358.5 | 1333.0 | 256x224x64 |
| 16384x512x7168 | bfloat16 | 128 | one-tile owner, n = 2 | 1178.6 | 1020.7 | opus | 1236.5 | 1016.4 | 256x256x64 |
| 16384x1024x7168 | bfloat16 | 256 | data-parallel | 1469.5 | 1473.8 | torch | 1440.5 | 1438.0 | 256x256x64 |
| 16384x2048x7168 | bfloat16 | 512 | data-parallel | 1508.3 | 1494.8 | torch | 1452.1 | 1452.7 | 256x256x64 |
| 16384x3584x7168 | bfloat16 | 896 | one-tile owner, n = 2 | 1486.6 | 1416.5 | flydsl | 1369.2 | 1336.8 | 224x256x64 |
| 16384x8192x7168 | bfloat16 | 2048 | data-parallel | 1551.0 | 1547.5 | torch | 1488.9 | 1538.7 | 256x256x64 |
| 16384x20480x7168 | bfloat16 | 5120 | data-parallel | 1574.1 | 1560.2 | flydsl | 1495.0 | 1471.3 | 256x256x64 |
| 32768x512x7168 | bfloat16 | 256 | data-parallel | 1368.9 | 1378.2 | torch | 1372.1 | 1375.3 | 256x256x64 |
| 32768x1024x7168 | bfloat16 | 512 | data-parallel | 1484.9 | 1483.4 | torch | 1438.7 | 1444.9 | 256x256x64 |
| 32768x2048x7168 | bfloat16 | 1024 | data-parallel | 1526.8 | 1514.4 | torch | 1486.3 | 1491.7 | 256x256x64 |
| 32768x3584x7168 | bfloat16 | 1792 | data-parallel | 1533.4 | 1516.9 | torch | 1507.0 | 1517.5 | 256x256x64 |
| 32768x8192x7168 | bfloat16 | 4096 | data-parallel | 1569.3 | 1556.5 | torch | 1513.4 | 1548.0 | 256x256x64 |
| 32768x20480x7168 | bfloat16 | 10240 | data-parallel | 1530.0 | 1551.5 | asm | 1488.7 | 1438.8 | 256x256x64 |
| 512x1280x8192 | bfloat16 | 10 | one-tile reduce-scatter, n = 19 | 338.4 | 88.4 | torch | 464.3 | 465.3 | 160x64x128 |
| 512x2560x8192 | bfloat16 | 20 | one-tile reduce-scatter, n = 9 | 548.4 | 174.4 | torch | 598.5 | 598.0 | 256x128x64 |
| 512x4096x8192 | bfloat16 | 32 | one-tile reduce-scatter, n = 6 | 746.3 | 276.9 | flydsl | 734.2 | 743.2 | 128x128x256 |
| 512x4608x8192 | bfloat16 | 36 | one-tile reduce-scatter, n = 5 | 776.7 | 311.8 | torch | 797.1 | 800.6 | 160x128x128 |
| 512x7168x8192 | bfloat16 | 56 | one-tile reduce-scatter, n = 3 | 873.7 | 483.5 | flydsl | 893.5 | 824.2 | 256x176x64 |
| 512x8192x8192 | bfloat16 | 64 | one-tile reduce-scatter, n = 3 | 934.5 | 548.5 | flydsl | 842.5 | 890.9 | 128x128x256 |
| 512x10240x8192 | bfloat16 | 80 | one-tile owner, n = 3-4 | 1024.2 | 683.1 | flydsl | 658.5 | 572.6 | 256x256x64 |
| 512x14336x8192 | bfloat16 | 112 | one-tile owner, n = 2-3 | 1148.4 | 945.8 | torch | 1017.3 | 1023.9 | 224x256x64 |
| 512x16384x8192 | bfloat16 | 128 | one-tile owner, n = 2 | 1192.1 | 1008.7 | flydsl | 682.5 | 918.5 | 256x256x64 |
| 512x57344x8192 | bfloat16 | 448 | data-parallel | 1328.3 | 1319.8 | flydsl | 1266.8 | 1250.3 | 224x256x64 |
| 768x4096x8192 | bfloat16 | 48 | one-tile reduce-scatter, n = 4 | 876.1 | 414.1 | flydsl | 822.8 | 751.3 | 128x192x128 |
| 1024x512x8192 | bfloat16 | 8 | one-tile reduce-scatter, n = 24 | 297.9 | 70.7 | torch | 366.9 | 364.2 | 256x64x64 |
| 1024x1280x8192 | bfloat16 | 20 | one-tile reduce-scatter, n = 9 | 549.6 | 175.0 | torch | 604.5 | 610.7 | 256x128x64 |
| 1024x2560x8192 | bfloat16 | 40 | one-tile reduce-scatter, n = 4 | 784.2 | 347.9 | torch | 805.6 | 829.0 | 160x128x128 |
| 1024x4096x8192 | bfloat16 | 64 | one-tile reduce-scatter, n = 3 | 951.8 | 549.0 | flydsl | 939.9 | 889.9 | 128x128x256 |
| 1024x4608x8192 | bfloat16 | 72 | one-tile owner, n = 3-4 | 1001.6 | 617.5 | torch | 941.3 | 944.5 | 160x256x64 |
| 1024x7168x8192 | bfloat16 | 112 | one-tile owner, n = 2-3 | 1199.8 | 949.3 | flydsl | 828.2 | 1018.1 | 224x256x64 |
| 1024x8192x8192 | bfloat16 | 128 | one-tile owner, n = 2 | 1255.5 | 1067.7 | flydsl | 1086.2 | 978.7 | 256x256x64 |
| 1024x10240x8192 | bfloat16 | 160 | spread | 1198.4 | 1241.0 | flydsl | 837.5 | 1203.5 | 160x256x64 |
| 1024x14336x8192 | bfloat16 | 224 | data-parallel | 1391.0 | 1395.9 | flydsl | 1332.8 | 1266.4 | 224x256x64 |
| 1024x16384x8192 | bfloat16 | 256 | data-parallel | 1488.8 | 1473.9 | torch | 1420.3 | 1420.3 | 256x256x64 |
| 1024x57344x8192 | bfloat16 | 896 | one-tile owner, n = 2 | 1480.2 | 1402.9 | asm | 1354.0 | 1308.3 | 224x256x64 |
| 2048x512x8192 | bfloat16 | 16 | one-tile reduce-scatter, n = 12 | 484.6 | 140.6 | torch | 530.4 | 530.6 | 256x128x64 |
| 2048x1280x8192 | bfloat16 | 40 | one-tile reduce-scatter, n = 4 | 783.9 | 347.5 | torch | 715.5 | 716.4 | 320x192x64 |
| 2048x2560x8192 | bfloat16 | 80 | one-tile owner, n = 3-4 | 1092.2 | 683.8 | torch | 965.9 | 937.0 | 160x256x64 |
| 2048x4096x8192 | bfloat16 | 128 | one-tile owner, n = 2 | 1255.4 | 1062.7 | opus | 1210.1 | 1001.8 | 256x256x64 |
| 2048x4608x8192 | bfloat16 | 144 | spread | 1212.5 | 1179.8 | torch | 1153.3 | 1156.0 | 144x256x64 |
| 2048x7168x8192 | bfloat16 | 224 | data-parallel | 1406.3 | 1416.2 | asm | 1361.2 | 1286.9 | 224x256x64 |
| 2048x8192x8192 | bfloat16 | 256 | data-parallel | 1476.9 | 1477.2 | torch | 1431.4 | 1421.7 | 256x256x64 |
| 2048x10240x8192 | bfloat16 | 320 | one-tile owner, n = 3 | 1361.9 | 1127.6 | flydsl | 977.7 | 1214.6 | 160x256x64 |
| 2048x14336x8192 | bfloat16 | 448 | data-parallel | 1419.8 | 1428.7 | flydsl | 1366.6 | 1301.4 | 224x256x64 |
| 2048x16384x8192 | bfloat16 | 512 | data-parallel | 1526.5 | 1508.8 | asm | 1438.1 | 1445.3 | 256x256x64 |
| 2048x57344x8192 | bfloat16 | 1792 | data-parallel | 1561.6 | 1555.1 | flydsl | 1500.2 | 1457.1 | 256x256x64 |
| 4096x512x8192 | bfloat16 | 32 | one-tile reduce-scatter, n = 6 | 742.8 | 279.2 | flydsl | 745.2 | 734.6 | 128x128x256 |
| 4096x1280x8192 | bfloat16 | 80 | one-tile owner, n = 3-4 | 1024.7 | 683.2 | torch | 936.2 | 938.2 | 160x256x64 |
| 4096x2560x8192 | bfloat16 | 160 | spread | 1218.1 | 1242.2 | torch | 1207.2 | 1209.2 | 160x256x64 |
| 4096x4096x8192 | bfloat16 | 256 | data-parallel | 1483.2 | 1485.2 | torch | 1438.6 | 1441.1 | 256x256x64 |
| 4096x4608x8192 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 | 1350.7 | 1021.3 | torch | 1175.1 | 1174.4 | 144x256x64 |
| 4096x7168x8192 | bfloat16 | 448 | data-parallel | 1425.7 | 1432.0 | flydsl | 1366.4 | 1308.9 | 224x256x64 |
| 4096x8192x8192 | bfloat16 | 512 | data-parallel | 1490.4 | 1496.3 | asm | 1446.1 | 1447.7 | 256x256x64 |
| 4096x10240x8192 | bfloat16 | 640 | one-tile owner, n = 2 | 1476.8 | 1398.4 | flydsl | 1344.8 | 1292.3 | 224x256x64 |
| 4096x14336x8192 | bfloat16 | 896 | one-tile owner, n = 2 | 1499.9 | 1437.4 | flydsl | 1376.9 | 1336.3 | 224x256x64 |
| 4096x16384x8192 | bfloat16 | 1024 | data-parallel | 1533.4 | 1535.0 | flydsl | 1458.0 | 1487.6 | 256x256x64 |
| 4096x57344x8192 | bfloat16 | 3584 | data-parallel | 1580.5 | 1558.0 | flydsl | 1518.0 | 1478.2 | 256x256x64 |
| 8192x512x8192 | bfloat16 | 64 | one-tile reduce-scatter, n = 3 | 947.7 | 548.7 | flydsl | 944.3 | 901.8 | 128x128x256 |
| 8192x1280x8192 | bfloat16 | 160 | spread | 1230.0 | 1249.3 | torch | 1211.5 | 1213.4 | 160x256x64 |
| 8192x2560x8192 | bfloat16 | 320 | one-tile owner, n = 3 | 1376.9 | 1136.0 | torch | 1221.9 | 1222.3 | 160x256x64 |
| 8192x4096x8192 | bfloat16 | 512 | data-parallel | 1495.7 | 1495.7 | torch | 1452.4 | 1451.2 | 256x256x64 |
| 8192x4608x8192 | bfloat16 | 576 | one-tile owner, n = 3 | 1438.6 | 1273.8 | torch | 1294.4 | 1296.0 | 192x256x64 |
| 8192x7168x8192 | bfloat16 | 896 | one-tile owner, n = 2 | 1500.6 | 1438.0 | asm | 1391.1 | 1354.1 | 224x256x64 |
| 8192x8192x8192 | bfloat16 | 1024 | data-parallel | 1535.9 | 1547.5 | asm | 1478.2 | 1521.6 | 256x256x64 |
| 8192x10240x8192 | bfloat16 | 1280 | data-parallel | 1543.7 | 1553.0 | flydsl | 1464.4 | 1542.5 | 256x256x64 |
| 8192x14336x8192 | bfloat16 | 1792 | data-parallel | 1551.9 | 1556.3 | flydsl | 1479.2 | 1537.4 | 256x256x64 |
| 8192x16384x8192 | bfloat16 | 2048 | data-parallel | 1555.8 | 1557.5 | asm | 1513.8 | 1505.1 | 256x256x64 |
| 8192x57344x8192 | bfloat16 | 7168 | data-parallel | 1552.5 | 1566.3 | asm | 1506.4 | 1450.5 | 256x256x64 |
| 16384x512x8192 | bfloat16 | 128 | one-tile owner, n = 2 | 1196.1 | 990.5 | torch | 1055.3 | 1060.7 | 256x256x64 |
| 16384x1280x8192 | bfloat16 | 320 | one-tile owner, n = 3 | 1367.1 | 1131.1 | torch | 1251.5 | 1241.2 | 160x256x64 |
| 16384x2560x8192 | bfloat16 | 640 | one-tile owner, n = 2 | 1476.6 | 1376.7 | flydsl | 1337.1 | 1304.5 | 256x224x64 |
| 16384x4608x8192 | bfloat16 | 1152 | one-tile owner, n = 2 | 1520.2 | 1453.5 | flydsl | 1400.5 | 1212.2 | 256x240x64 |
| 16384x7168x8192 | bfloat16 | 1792 | data-parallel | 1564.9 | 1552.1 | torch | 1501.3 | 1546.4 | 256x256x64 |
| 16384x8192x8192 | bfloat16 | 2048 | data-parallel | 1555.5 | 1561.2 | torch | 1507.9 | 1551.5 | 256x256x64 |
| 16384x10240x8192 | bfloat16 | 2560 | data-parallel | 1577.5 | 1568.7 | torch | 1515.0 | 1555.8 | 256x256x64 |
| 16384x14336x8192 | bfloat16 | 3584 | data-parallel | 1551.0 | 1563.8 | asm | 1529.1 | 1521.9 | 256x256x64 |
| 16384x16384x8192 | bfloat16 | 4096 | data-parallel | 1565.7 | 1565.4 | asm | 1528.8 | 1513.0 | 256x256x64 |
| 16384x57344x8192 | bfloat16 | 14336 | data-parallel | 1551.0 | 1571.4 | asm | 1496.3 | 1441.7 | 256x256x64 |
| 32768x1280x8192 | bfloat16 | 640 | one-tile owner, n = 2 | 1450.9 | 1348.0 | flydsl | 1309.0 | 1320.2 | 256x224x64 |
| 32768x2560x8192 | bfloat16 | 1280 | data-parallel | 1528.9 | 1522.7 | torch | 1525.1 | 1525.0 | 256x256x64 |
| 32768x7168x8192 | bfloat16 | 3584 | data-parallel | 1573.9 | 1560.9 | torch | 1539.9 | 1552.5 | 256x256x64 |
| 32768x8192x8192 | bfloat16 | 4096 | data-parallel | 1579.8 | 1571.9 | flydsl | 1508.1 | 1556.9 | 256x256x64 |
| 32768x10240x8192 | bfloat16 | 5120 | data-parallel | 1580.9 | 1569.7 | torch | 1514.9 | 1555.5 | 256x256x64 |
| 32768x14336x8192 | bfloat16 | 7168 | data-parallel | 1572.2 | 1565.8 | asm | 1522.0 | 1501.5 | 256x256x64 |
| 32768x16384x8192 | bfloat16 | 8192 | data-parallel | 1526.1 | 1560.1 | asm | 1515.2 | 1467.7 | 256x256x64 |
| 32768x57344x8192 | bfloat16 | 28672 | data-parallel | 1527.0 | 1560.6 | asm | 1470.4 | 1430.2 | 256x256x64 |
| 512x6144x12288 | bfloat16 | 48 | one-tile reduce-scatter, n = 4 | 1002.1 | 421.4 | torch | 888.0 | 896.3 | 192x128x128 |
| 1024x6144x12288 | bfloat16 | 96 | one-tile owner, n = 2-3 | 1151.7 | 833.4 | torch | 1075.9 | 1101.6 | 192x256x64 |
| 2048x6144x12288 | bfloat16 | 192 | spread | 1392.7 | 1362.8 | torch | 1282.9 | 1284.9 | 192x256x64 |
| 4096x6144x12288 | bfloat16 | 384 | one-tile owner, n = 2 | 1462.4 | 1345.6 | torch | 1296.1 | 1298.0 | 192x256x64 |
| 8192x6144x12288 | bfloat16 | 768 | data-parallel | 1551.8 | 1539.9 | torch | 1484.9 | 1512.1 | 256x256x64 |
| 16384x6144x12288 | bfloat16 | 1536 | data-parallel | 1559.1 | 1545.1 | torch | 1508.3 | 1537.8 | 256x256x64 |
| 32768x6144x12288 | bfloat16 | 3072 | data-parallel | 1567.5 | 1563.0 | asm | 1535.6 | 1542.6 | 256x256x64 |
| 512x16384x13312 | bfloat16 | 128 | one-tile owner, n = 2 | 1276.9 | 928.5 | flydsl | 714.8 | 1096.6 | 256x256x64 |
| 1024x16384x13312 | bfloat16 | 256 | data-parallel | 1509.7 | 1515.6 | asm | 1457.5 | 1456.0 | 256x256x64 |
| 2048x16384x13312 | bfloat16 | 512 | data-parallel | 1556.2 | 1560.3 | flydsl | 1013.1 | 1483.3 | 256x256x64 |
| 4096x16384x13312 | bfloat16 | 1024 | data-parallel | 1554.4 | 1550.2 | flydsl | 1482.6 | 1483.9 | 256x256x64 |
| 8192x16384x13312 | bfloat16 | 2048 | data-parallel | 1579.4 | 1573.0 | flydsl | 1517.3 | 1491.4 | 256x256x64 |
| 16384x16384x13312 | bfloat16 | 4096 | data-parallel | 1571.5 | 1573.7 | asm | 1500.4 | 1476.3 | 256x256x64 |
| 32768x16384x13312 | bfloat16 | 8192 | data-parallel | 1564.7 | 1565.8 | flydsl | 1487.4 | 1447.0 | 256x256x64 |
| 512x2304x16384 | bfloat16 | 18 | one-tile reduce-scatter, n = 10 | 728.2 | 162.0 | torch | 744.0 | 736.6 | 256x128x64 |
| 512x4608x16384 | bfloat16 | 36 | one-tile reduce-scatter, n = 5 | 993.6 | 319.9 | flydsl | 580.8 | 918.7 | 160x128x128 |
| 512x9216x16384 | bfloat16 | 72 | one-tile owner, n = 3-4 | 1091.8 | 591.1 | flydsl | 646.4 | 756.5 | 256x256x64 |
| 512x13312x16384 | bfloat16 | 104 | one-tile owner, n = 2-3 | 1170.5 | 781.8 | flydsl | 542.4 | 1076.5 | 208x256x64 |
| 512x26624x16384 | bfloat16 | 208 | spread | 1267.8 | 1297.8 | flydsl | 773.1 | 1176.2 | 208x256x64 |
| 512x53248x16384 | bfloat16 | 416 | spread | 1331.8 | 1294.9 | flydsl | 642.6 | 1191.8 | 208x256x64 |
| 1024x2304x16384 | bfloat16 | 36 | one-tile reduce-scatter, n = 5 | 978.9 | 321.4 | torch | 890.3 | 892.7 | 256x256x64 |
| 1024x4608x16384 | bfloat16 | 72 | one-tile owner, n = 3-4 | 1182.5 | 635.6 | flydsl | 786.7 | 1058.3 | 144x256x64 |
| 1024x9216x16384 | bfloat16 | 144 | spread | 1219.7 | 1122.2 | flydsl | 680.8 | 1171.2 | 144x256x64 |
| 1024x13312x16384 | bfloat16 | 208 | spread | 1349.3 | 1353.2 | asm | 1323.2 | 1234.2 | 208x256x64 |
| 1024x26624x16384 | bfloat16 | 416 | spread | 1432.7 | 1365.8 | flydsl | 927.7 | 1242.2 | 208x256x64 |
| 1024x53248x16384 | bfloat16 | 832 | one-tile owner, n = 4 | 1505.8 | 1335.0 | triton | 995.2 | 1246.7 | 208x256x64 |
| 2048x2304x16384 | bfloat16 | 72 | one-tile owner, n = 3-4 | 1165.9 | 631.8 | torch | 939.0 | 939.5 | 256x256x64 |
| 2048x4608x16384 | bfloat16 | 144 | spread | 1278.3 | 1215.3 | flydsl | 692.1 | 1197.0 | 144x256x64 |
| 2048x9216x16384 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 | 1409.6 | 1032.2 | flydsl | 910.3 | 1160.1 | 144x256x64 |
| 2048x13312x16384 | bfloat16 | 416 | spread | 1476.0 | 1403.3 | flydsl | 1363.8 | 1234.7 | 208x256x64 |
| 2048x26624x16384 | bfloat16 | 832 | one-tile owner, n = 4 | 1526.1 | 1363.3 | flydsl | 909.4 | 1234.0 | 208x256x64 |
| 2048x53248x16384 | bfloat16 | 1664 | one-tile owner, n = 2 | 1536.3 | 1485.7 | flydsl | 783.0 | 1271.5 | 240x256x64 |
| 4096x2304x16384 | bfloat16 | 144 | spread | 1282.5 | 1216.0 | torch | 1201.0 | 1200.5 | 144x256x64 |
| 4096x4608x16384 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 | 1417.2 | 1038.1 | torch | 1214.3 | 1210.6 | 144x256x64 |
| 4096x9216x16384 | bfloat16 | 576 | one-tile owner, n = 4 | 1517.0 | 1293.5 | torch | 1307.1 | 1310.3 | 192x256x64 |
| 4096x13312x16384 | bfloat16 | 832 | one-tile owner, n = 4 | 1529.9 | 1365.5 | flydsl | 1311.5 | 1248.6 | 208x256x64 |
| 4096x26624x16384 | bfloat16 | 1664 | one-tile owner, n = 2 | 1544.6 | 1506.1 | flydsl | 1448.9 | 1316.2 | 240x256x64 |
| 4096x53248x16384 | bfloat16 | 3328 | data-parallel | 1578.7 | 1563.6 | flydsl | 1530.1 | 1466.8 | 256x256x64 |
| 8192x2304x16384 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 | 1386.6 | 1008.8 | torch | 1196.4 | 1200.1 | 144x256x64 |
| 8192x4608x16384 | bfloat16 | 576 | one-tile owner, n = 4 | 1503.5 | 1269.7 | torch | 1344.0 | 1338.8 | 192x256x64 |
| 8192x9216x16384 | bfloat16 | 1152 | one-tile owner, n = 2 | 1545.3 | 1450.5 | flydsl | 1402.4 | 1148.0 | 256x240x64 |
| 8192x13312x16384 | bfloat16 | 1664 | one-tile owner, n = 2 | 1543.2 | 1477.2 | flydsl | 1437.2 | 1281.7 | 240x256x64 |
| 8192x26624x16384 | bfloat16 | 3328 | data-parallel | 1554.8 | 1559.0 | flydsl | 1494.7 | 1461.1 | 256x256x64 |
| 8192x53248x16384 | bfloat16 | 6656 | data-parallel | 1524.3 | 1552.4 | flydsl | 1481.2 | 1440.3 | 256x256x64 |
| 16384x2304x16384 | bfloat16 | 576 | one-tile owner, n = 4 | 1475.3 | 1272.6 | torch | 1330.7 | 1323.1 | 192x256x64 |
| 16384x4608x16384 | bfloat16 | 1152 | one-tile owner, n = 2 | 1527.0 | 1440.5 | flydsl | 1403.5 | 1174.6 | 256x240x64 |
| 16384x9216x16384 | bfloat16 | 2304 | data-parallel | 1538.8 | 1552.2 | flydsl | 1492.0 | 1478.9 | 256x256x64 |
| 16384x13312x16384 | bfloat16 | 3328 | data-parallel | 1532.8 | 1550.7 | asm | 1483.9 | 1460.6 | 256x256x64 |
| 16384x26624x16384 | bfloat16 | 6656 | data-parallel | 1522.1 | 1545.1 | flydsl | 1493.9 | 1442.5 | 256x256x64 |
| 16384x53248x16384 | bfloat16 | 13312 | data-parallel | 1495.6 | 1531.1 | asm | 1458.3 | 1426.5 | 256x256x64 |
| 32768x2304x16384 | bfloat16 | 1152 | one-tile owner, n = 2 | 1484.8 | 1433.3 | flydsl | 1398.1 | 1179.0 | 256x240x64 |
| 32768x4608x16384 | bfloat16 | 2304 | data-parallel | 1531.3 | 1545.5 | torch | 1520.4 | 1527.0 | 256x256x64 |
| 32768x9216x16384 | bfloat16 | 4608 | data-parallel | 1513.3 | 1542.1 | asm | 1445.1 | 1446.7 | 256x256x64 |
| 32768x13312x16384 | bfloat16 | 6656 | data-parallel | 1506.9 | 1542.0 | asm | 1455.1 | 1428.8 | 256x256x64 |
| 32768x26624x16384 | bfloat16 | 13312 | data-parallel | 1499.4 | 1528.1 | flydsl | 1454.8 | 1430.0 | 256x256x64 |
| 32768x53248x16384 | bfloat16 | 26624 | data-parallel | 1492.2 | 1517.7 | flydsl | 1438.1 | 1441.6 | 256x256x64 |
| 512x6144x18432 | bfloat16 | 48 | one-tile reduce-scatter, n = 4 | 1011.6 | 427.9 | flydsl | 910.0 | 942.1 | 192x128x128 |
| 1024x6144x18432 | bfloat16 | 96 | one-tile owner, n = 2-3 | 1148.4 | 795.8 | torch | 1171.6 | 1169.6 | 192x256x64 |
| 2048x6144x18432 | bfloat16 | 192 | spread | 1444.0 | 1335.8 | torch | 1305.1 | 1309.2 | 192x256x64 |
| 4096x6144x18432 | bfloat16 | 384 | one-tile owner, n = 2 | 1500.5 | 1353.6 | torch | 1325.7 | 1313.2 | 192x256x64 |
| 8192x6144x18432 | bfloat16 | 768 | data-parallel | 1567.2 | 1553.8 | torch | 1525.6 | 1515.9 | 256x256x64 |
| 16384x6144x18432 | bfloat16 | 1536 | data-parallel | 1560.7 | 1560.0 | asm | 1520.5 | 1518.5 | 256x256x64 |
| 32768x6144x18432 | bfloat16 | 3072 | data-parallel | 1576.3 | 1579.5 | asm | 1520.1 | 1502.2 | 256x256x64 |
| 512x5120x25600 | bfloat16 | 40 | one-tile reduce-scatter, n = 4 | 930.1 | 327.4 | flydsl | 436.1 | 909.1 | 160x128x128 |
| 1024x5120x25600 | bfloat16 | 80 | one-tile owner, n = 3-4 | 1266.7 | 653.8 | torch | 1137.6 | 1147.1 | 160x256x64 |
| 2048x5120x25600 | bfloat16 | 160 | spread | 1393.3 | 1165.4 | torch | 1256.8 | 1261.0 | 160x256x64 |
| 4096x5120x25600 | bfloat16 | 320 | one-tile owner, n = 4 | 1498.9 | 1138.2 | torch | 1263.0 | 1251.2 | 160x256x64 |
| 8192x5120x25600 | bfloat16 | 640 | one-tile owner, n = 2 | 1539.3 | 1389.2 | asm | 1373.8 | 1332.9 | 256x224x64 |
| 16384x5120x25600 | bfloat16 | 1280 | data-parallel | 1573.4 | 1574.1 | torch | 1528.7 | 1532.8 | 256x256x64 |
| 32768x5120x25600 | bfloat16 | 2560 | data-parallel | 1564.7 | 1572.7 | torch | 1506.3 | 1515.3 | 256x256x64 |
| 512x16384x26624 | bfloat16 | 128 | one-tile owner, n = 2 | 1361.5 | 936.8 | flydsl | 649.3 | 1187.3 | 256x256x64 |
| 1024x16384x26624 | bfloat16 | 256 | data-parallel | 1540.3 | 1545.2 | flydsl | 784.9 | 1494.4 | 256x256x64 |
| 2048x16384x26624 | bfloat16 | 512 | data-parallel | 1588.4 | 1552.6 | flydsl | 815.0 | 1499.3 | 256x256x64 |
| 4096x16384x26624 | bfloat16 | 1024 | data-parallel | 1597.3 | 1572.0 | flydsl | 1538.7 | 1503.3 | 256x256x64 |
| 8192x16384x26624 | bfloat16 | 2048 | data-parallel | 1590.7 | 1567.2 | asm | 235.9 | 1485.9 | 256x256x64 |
| 16384x16384x26624 | bfloat16 | 4096 | data-parallel | 1582.1 | 1588.0 | asm | 1518.1 | 1464.2 | 256x256x64 |
| 32768x16384x26624 | bfloat16 | 8192 | data-parallel | 1564.5 | 1579.1 | asm | 1506.7 | 1454.8 | 256x256x64 |
| 512x8192x28672 | bfloat16 | 64 | one-tile reduce-scatter, n = 3 | 1107.7 | 479.6 | flydsl | 778.0 | 923.2 | 240x256x64 |
| 1024x8192x28672 | bfloat16 | 128 | one-tile owner, n = 2 | 1442.2 | 943.7 | triton | 694.7 | 1296.3 | 256x256x64 |
| 2048x8192x28672 | bfloat16 | 256 | data-parallel | 1537.3 | 1537.4 | flydsl | 1035.9 | 1496.5 | 256x256x64 |
| 4096x8192x28672 | bfloat16 | 512 | data-parallel | 1560.6 | 1556.4 | flydsl | 1525.6 | 1519.5 | 256x256x64 |
| 8192x8192x28672 | bfloat16 | 1024 | data-parallel | 1586.9 | 1589.1 | asm | 1545.3 | 1525.9 | 256x256x64 |
| 16384x8192x28672 | bfloat16 | 2048 | data-parallel | 1591.2 | 1591.1 | asm | 1530.9 | 1507.2 | 256x256x64 |
| 32768x8192x28672 | bfloat16 | 4096 | data-parallel | 1578.1 | 1593.2 | asm | 1521.6 | 1464.7 | 256x256x64 |
| 512x7168x35840 | bfloat16 | 56 | one-tile reduce-scatter, n = 3 | 1058.8 | 436.4 | flydsl | 818.0 | 965.1 | 256x176x64 |
| 1024x7168x35840 | bfloat16 | 112 | one-tile owner, n = 2-3 | 1362.4 | 861.0 | flydsl | 565.5 | 1269.0 | 224x256x64 |
| 2048x7168x35840 | bfloat16 | 224 | spread | 1492.9 | 1467.6 | asm | 1438.8 | 1369.7 | 224x256x64 |
| 4096x7168x35840 | bfloat16 | 448 | spread | 1547.6 | 1482.6 | flydsl | 1442.8 | 1366.9 | 224x256x64 |
| 8192x7168x35840 | bfloat16 | 896 | one-tile owner, n = 2 | 1580.6 | 1430.5 | asm | 1373.0 | 1376.3 | 224x256x64 |
| 16384x7168x35840 | bfloat16 | 1792 | data-parallel | 1572.6 | 1582.5 | torch | 1501.8 | 1503.1 | 256x256x64 |
| 32768x7168x35840 | bfloat16 | 3584 | data-parallel | 1574.3 | 1582.6 | flydsl | 1496.8 | 1458.7 | 256x256x64 |
