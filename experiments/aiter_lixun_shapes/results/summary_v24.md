# streamk_wip (v24, capped) vs AITER, v9 and hipBLASLt on the 630 shapes

Shapes: `supported_status_scoped.csv` (630 shapes, no bias; 598 bf16 and 32 fp32 outputs). Inputs from `torch.manual_seed(0)`, every output checked against an fp32 `F.linear` reference, `do_bench_cudagraph` with 3 interleaved rounds and the median taken.

- Stack: torch 2.10.0+rocm7.2.4.git3d3aa833, hipBLASLt 100202 (dabb6df2b9), Triton 3.8.0 @ /home/jbowley/repos/triton_gfx950-tutorial-v2.2/python/triton.
- GPU: HIP device 3, PCI bus 21, gfx950:sramecc+:xnack-.
- AITER: e2c7aa0aa [Triton/Gluon] gfx950 a16w16: skip the Gluon kernel on Triton without cd_regclass, on top of a1455f1c4 2026-10-02 [CK] [FlyDSL] add mxfp8 a8w8 blockscale for gfx950 (#4951).
- streamk_wip: tutorial `v24_streamk_combined`, snapshot sha256 07108e9456bf44c4 (newest WORKLOG entry: "1. Every ending, placement and partial type in one kernel (2026-10-08)"), `STREAMK_POLICY=capped` (the kernel's default), v9 tile order. The configuration capped ran on each shape is in the appendix and in `v20_label`.
- hipBLASLt: the call AITER's `torch` libtype makes, `F.linear(x, w)` for bf16 and `torch.mm(x, w.t(), out_dtype=torch.float32)` for fp32, timed the same way. Its tile is the MT of the first `Cijk_` kernel it launches.
- Cache state: back to back on the same buffers (inputs stay cached between launches).
- "Faster" and "slower" mean a gap of more than 1%; "within 1%" holds the rest.

**Against AITER's pick, by the configuration capped chose**

| | Shapes | v9: faster | v9: within 1% | v9: slower | v9: median | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All shapes | 630 | 324 | 23 | 283 | +1.5% | 396 | 20 | 214 | +3.3% |
| Data-parallel | 324 | 255 | 18 | 51 | +3.6% | 259 | 16 | 49 | +3.7% |
| One-tile reduce-scatter, K >= 2048 | 77 | 5 | 0 | 72 | -65.8% | 14 | 0 | 63 | -17.6% |
| One-tile owner, K >= 2048 | 219 | 62 | 5 | 152 | -19.7% | 121 | 4 | 94 | +3.1% |
| One-tile reduce-scatter, K < 2048 | 6 | 0 | 0 | 6 | -67.3% | 0 | 0 | 6 | -59.5% |
| One-tile owner, K < 2048 | 2 | 0 | 0 | 2 | -48.8% | 0 | 0 | 2 | -47.8% |
| Spread | 2 | 2 | 0 | 0 | +2.4% | 2 | 0 | 0 | +5.1% |

Each pair of columns compares that kernel with AITER's pick; positive medians mean it is faster.

**Against AITER's pick, by the tile AITER uses**

| | Shapes | v9: faster | v9: within 1% | v9: slower | v9: median | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All shapes | 630 | 324 | 23 | 283 | +1.5% | 396 | 20 | 214 | +3.3% |
| AITER uses a 256x256 tile | 248 | 231 | 12 | 5 | +3.7% | 242 | 6 | 0 | +4.1% |
| AITER uses another tile | 382 | 93 | 11 | 278 | -24.2% | 154 | 14 | 214 | -5.7% |

Each pair of columns compares that kernel with AITER's pick; positive medians mean it is faster.

**Against AITER's pick, by v9 grid size (number of 256x256 output tiles)**

| | Shapes | v9: faster | v9: within 1% | v9: slower | v9: median | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All shapes | 630 | 324 | 23 | 283 | +1.5% | 396 | 20 | 214 | +3.3% |
| Under 256 (less than one wave) | 321 | 58 | 9 | 254 | -33.1% | 98 | 14 | 209 | -9.6% |
| 256 to 1023 | 152 | 119 | 6 | 27 | +4.1% | 144 | 3 | 5 | +6.5% |
| 1024 or more | 157 | 147 | 8 | 2 | +3.7% | 154 | 3 | 0 | +4.1% |

Each pair of columns compares that kernel with AITER's pick; positive medians mean it is faster.

**Against hipBLASLt (torch), by v9 grid size**

| | Shapes | AITER's pick: faster | AITER's pick: within 1% | AITER's pick: slower | AITER's pick: median | v9: faster | v9: within 1% | v9: slower | v9: median | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All shapes | 630 | 215 | 233 | 182 | -0.2% | 310 | 34 | 286 | +0.7% | 395 | 28 | 207 | +2.9% |
| Under 256 (less than one wave) | 321 | 119 | 111 | 91 | -0.1% | 68 | 7 | 246 | -36.1% | 114 | 8 | 199 | -8.3% |
| 256 to 1023 | 152 | 45 | 66 | 41 | -0.2% | 116 | 3 | 33 | +3.2% | 143 | 3 | 6 | +7.1% |
| 1024 or more | 157 | 51 | 56 | 50 | -0.4% | 126 | 24 | 7 | +3.1% | 138 | 17 | 2 | +3.8% |

Each pair of columns compares that kernel with hipBLASLt; positive medians mean it is faster.

**hipBLASLt's tile choice**

| hipBLASLt tile (MxNxK) | Shapes | hipBLASLt median TFLOPS | streamk_wip vs hipBLASLt: faster | within 1% | slower | median |
|---|---:|---:|---:|---:|---:|---:|
| 256x256x64 | 225 | 1411 | 205 | 18 | 2 | +3.5% |
| 128x128x256 | 40 | 735 | 8 | 0 | 32 | -12.8% |
| 192x256x64 | 34 | 1196 | 26 | 4 | 4 | +5.2% |
| 224x256x64 | 32 | 1262 | 31 | 1 | 0 | +10.5% |
| 256x128x64 | 27 | 748 | 1 | 0 | 26 | -11.7% |
| 160x256x64 | 27 | 1181 | 23 | 0 | 4 | +3.9% |
| 256x224x64 | 20 | 1279 | 20 | 0 | 0 | +8.9% |
| 256x240x64 | 18 | 1158 | 18 | 0 | 0 | +20.9% |
| 144x256x64 | 15 | 1158 | 10 | 2 | 3 | +1.3% |
| 128x128x128 | 14 | 637 | 0 | 0 | 14 | -21.5% |
| 160x128x128 | 13 | 897 | 1 | 1 | 11 | -13.3% |
| 96x128x128 | 12 | 639 | 0 | 0 | 12 | -42.9% |
| 192x128x128 | 12 | 837 | 5 | 0 | 7 | -5.3% |
| 256x208x64 | 12 | 1162 | 12 | 0 | 0 | +14.4% |
| 256x192x64 | 11 | 1215 | 8 | 1 | 2 | +4.5% |
| 208x256x64 | 11 | 1234 | 11 | 0 | 0 | +10.2% |
| 256x160x64 | 10 | 1182 | 3 | 1 | 6 | -2.5% |
| 64x64x256 | 9 | 484 | 0 | 0 | 9 | -35.2% |
| 128x192x128 | 7 | 983 | 2 | 0 | 5 | -14.0% |
| 128x160x128 | 7 | 959 | 0 | 0 | 7 | -17.0% |
| 240x256x64 | 7 | 1199 | 7 | 0 | 0 | +20.7% |
| 128x80x128 | 6 | 714 | 0 | 0 | 6 | -28.8% |
| 128x96x128 | 6 | 796 | 0 | 0 | 6 | -23.7% |
| 224x128x64 | 5 | 874 | 0 | 0 | 5 | -20.3% |
| 256x176x64 | 4 | 792 | 2 | 0 | 2 | +2.9% |
| 128x64x128 | 4 | 573 | 0 | 0 | 4 | -54.0% |
| 160x64x128 | 4 | 508 | 0 | 0 | 4 | -30.3% |
| 64x80x256 | 4 | 536 | 0 | 0 | 4 | -33.5% |
| 64x96x256 | 4 | 563 | 0 | 0 | 4 | -24.9% |
| 64x32x256 | 3 | 329 | 0 | 0 | 3 | -39.3% |
| 64x48x256 | 3 | 432 | 0 | 0 | 3 | -36.4% |
| 160x192x128 | 3 | 945 | 0 | 0 | 3 | -1.7% |
| 192x128x64 | 2 | 920 | 0 | 0 | 2 | -30.0% |
| 128x192x64 | 2 | 886 | 0 | 0 | 2 | -10.6% |
| 256x112x64 | 2 | 479 | 0 | 0 | 2 | -11.7% |
| 192x64x128 | 2 | 432 | 0 | 0 | 2 | -15.1% |
| 128x128x64 | 2 | 501 | 0 | 0 | 2 | -26.1% |
| 256x96x64 | 2 | 443 | 0 | 0 | 2 | -12.7% |
| 256x64x64 | 2 | 345 | 0 | 0 | 2 | -10.6% |
| 96x64x128 | 1 | 475 | 0 | 0 | 1 | -55.9% |
| 224x160x64 | 1 | 1091 | 1 | 0 | 0 | +11.9% |
| 192x192x64 | 1 | 1043 | 0 | 0 | 1 | -2.5% |
| 64x96x128 | 1 | 596 | 0 | 0 | 1 | -24.8% |
| 256x80x64 | 1 | 406 | 0 | 0 | 1 | -16.9% |
| 256x80x128 | 1 | 591 | 0 | 0 | 1 | -15.1% |
| 320x192x64 | 1 | 715 | 1 | 0 | 0 | +1.1% |

Solution-name tokens (stream-K `SK`, global split-U `GSU`, kernels launched): SK3 GSU0 x1: 604, SK3 GSU0 x2: 24, none GSU- x1: 2. `SK3` marks a kernel built with stream-K support, not that this launch split K.

**streamk_wip against v9, by configuration**

| | Shapes | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|
| All shapes | 630 | 350 | 262 | 18 | +1.7% |
| Data-parallel | 324 | 49 | 259 | 16 | +0.0% |
| One-tile reduce-scatter, K >= 2048 | 77 | 76 | 1 | 0 | +150.2% |
| One-tile owner, K >= 2048 | 219 | 216 | 2 | 1 | +20.7% |
| One-tile reduce-scatter, K < 2048 | 6 | 6 | 0 | 0 | +24.0% |
| One-tile owner, K < 2048 | 2 | 1 | 0 | 1 | +1.6% |
| Spread | 2 | 2 | 0 | 0 | +2.5% |

Each pair of columns compares that kernel with v9; positive medians mean it is faster.

**Diagnostic: capped's split against forcing data-parallel**

| Configuration | Shapes | split faster | within 1% | split slower | median |
|---|---:|---:|---:|---:|---:|
| One-tile owner, K < 2048 | 2 | 2 | 0 | 0 | +4.4% |
| One-tile owner, K >= 2048 | 219 | 214 | 3 | 2 | +21.1% |
| One-tile reduce-scatter, K < 2048 | 6 | 6 | 0 | 0 | +27.1% |
| One-tile reduce-scatter, K >= 2048 | 77 | 76 | 1 | 0 | +152.2% |
| Spread | 2 | 2 | 0 | 0 | +2.6% |

**Selected shapes (TFLOPS)**

| Shape (MxNxK) | Output tiles | AITER pick | AITER | hipBLASLt | hipBLASLt tile | v9 | streamk_wip | streamk_wip config | vs v9 | vs AITER | vs hipBLASLt |
|---|---:|---|---:|---:|---|---:|---:|---|---:|---:|---:|
| 512x2304x16384 | 18 | torch | 740 | 734 | 256x128x64 | 162 | 675 | one-tile reduce-scatter, n = 7 | +316% | -9% | -8% |
| 512x5120x25600 | 40 | flydsl | 431 | 908 | 160x128x128 | 325 | 1022 | one-tile owner, n = 6-7 | +214% | +137% | +13% |
| 512x7168x35840 | 56 | flydsl | 817 | 967 | 256x176x64 | 437 | 1197 | one-tile owner, n = 4-5 | +174% | +47% | +24% |
| 8192x2304x16384 | 288 | torch | 1198 | 1199 | 144x256x64 | 1006 | 1384 | one-tile reduce-scatter, n = 4 | +38% | +16% | +15% |
| 4096x5120x25600 | 320 | torch | 1248 | 1254 | 160x256x64 | 1130 | 1495 | one-tile owner, n = 4 | +32% | +20% | +19% |
| 4096x4608x8192 | 288 | torch | 1183 | 1180 | 144x256x64 | 1024 | 1347 | one-tile reduce-scatter, n = 4 | +32% | +14% | +14% |
| 1024x3072x512 | 48 | flydsl | 462 | 394 | 96x128x128 | 226 | 225 | data-parallel | -0% | -51% | -43% |
| 2048x2048x512 | 64 | flydsl | 521 | 465 | 128x128x128 | 281 | 282 | data-parallel | +0% | -46% | -39% |
| 4096x5120x640 | 320 | flydsl | 816 | 762 | 256x176x64 | 741 | 747 | data-parallel | +1% | -9% | -2% |
| 8192x2048x512 | 256 | torch | 797 | 798 | 256x256x64 | 846 | 842 | data-parallel | -0% | +6% | +6% |

**Against `results/details_aisched.csv`** (this run / that run - 1, median; for v20 the baseline may use a different policy or kernel snapshot)

| | Shapes | AITER's pick | v9 | v20 | hipBLASLt |
|---|---:|---:|---:|---:|---:|
| All shapes | 630 | -2.3% | -1.5% | -1.5% | -2.4% |
| Under 256 (less than one wave) | 321 | -2.1% | -0.9% | -0.7% | -2.5% |
| 256 to 1023 | 152 | -3.0% | -3.0% | -3.0% | -3.8% |
| 1024 or more | 157 | -2.1% | -1.6% | -1.7% | -1.5% |
| K < 2048 | 91 | -1.7% | -1.3% | -1.2% | -2.0% |
| K >= 2048 | 539 | -2.6% | -1.6% | -1.7% | -2.6% |

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
| 1024x1536x7168 | bfloat16 | False (2) | True | True | True (1) | True | 0 |  |
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
| 512x7168x35840 | bfloat16 | False (4) | True | True | True (2) | True | 0 |  |

29 of these are AITER's pick alone outside the tolerance; every other kernel passed.

## Appendix: every shape

| Shape | out | tiles | streamk_wip config | streamk_wip | v9 | AITER pick | AITER | hipBLASLt | hipBLASLt tile |
|---|---|---:|---|---:|---:|---|---:|---:|---|
| 1024x3072x512 | bfloat16 | 48 | data-parallel | 225.3 | 225.9 | flydsl | 461.9 | 394.2 | 96x128x128 |
| 1024x4096x512 | bfloat16 | 64 | data-parallel | 282.2 | 282.1 | flydsl | 524.7 | 465.0 | 128x128x128 |
| 1024x7168x512 | bfloat16 | 112 | data-parallel | 453.3 | 452.4 | flydsl | 609.8 | 568.2 | 256x128x64 |
| 2048x2048x512 | bfloat16 | 64 | data-parallel | 281.8 | 280.9 | flydsl | 520.8 | 465.0 | 128x128x128 |
| 2048x4096x512 | bfloat16 | 128 | data-parallel | 489.7 | 489.3 | torch | 603.4 | 603.9 | 256x128x64 |
| 4096x2048x512 | bfloat16 | 128 | data-parallel | 489.1 | 487.8 | torch | 658.3 | 659.6 | 256x128x64 |
| 4096x4096x512 | bfloat16 | 256 | data-parallel | 840.9 | 855.2 | torch | 802.2 | 804.4 | 256x256x64 |
| 4096x7168x512 | bfloat16 | 448 | data-parallel | 878.3 | 865.9 | flydsl | 847.8 | 770.0 | 224x256x64 |
| 8192x2048x512 | bfloat16 | 256 | data-parallel | 841.9 | 846.1 | torch | 797.1 | 798.0 | 256x256x64 |
| 8192x4096x512 | bfloat16 | 512 | data-parallel | 938.4 | 926.7 | flydsl | 857.4 | 864.8 | 256x256x64 |
| 16384x2048x512 | bfloat16 | 512 | data-parallel | 936.0 | 925.8 | torch | 858.5 | 862.7 | 256x256x64 |
| 16384x4096x512 | bfloat16 | 1024 | data-parallel | 1020.9 | 978.8 | torch | 923.2 | 925.6 | 256x256x64 |
| 16384x7168x512 | bfloat16 | 1792 | data-parallel | 1050.6 | 1007.6 | torch | 974.3 | 980.5 | 256x256x64 |
| 32768x2048x512 | bfloat16 | 1024 | data-parallel | 1012.8 | 969.4 | torch | 916.9 | 925.2 | 256x256x64 |
| 1024x5120x640 | bfloat16 | 80 | data-parallel | 375.8 | 375.6 | torch | 518.1 | 519.3 | 160x128x128 |
| 2048x5120x640 | bfloat16 | 160 | data-parallel | 672.2 | 670.2 | flydsl | 670.4 | 530.3 | 192x256x64 |
| 4096x5120x640 | bfloat16 | 320 | data-parallel | 746.6 | 740.6 | flydsl | 816.1 | 762.1 | 256x176x64 |
| 8192x5120x640 | bfloat16 | 640 | data-parallel | 964.0 | 936.4 | torch | 898.8 | 909.2 | 256x224x64 |
| 16384x5120x640 | bfloat16 | 1280 | data-parallel | 1113.8 | 1073.1 | torch | 1035.0 | 1041.1 | 256x256x64 |
| 32768x5120x640 | bfloat16 | 2560 | data-parallel | 1117.7 | 1065.6 | torch | 1080.7 | 1089.1 | 256x256x64 |
| 1024x7168x768 | bfloat16 | 112 | data-parallel | 533.4 | 535.3 | flydsl | 709.1 | 661.8 | 224x128x64 |
| 10240x7168x768 | bfloat16 | 1120 | data-parallel | 1092.4 | 1055.1 | opus | 404.1 | 853.8 | 256x240x64 |
| 512x4096x1024 | bfloat16 | 32 | data-parallel | 200.2 | 200.2 | torch | 482.2 | 485.2 | 128x64x128 |
| 512x6144x1024 | bfloat16 | 48 | data-parallel | 292.7 | 292.6 | flydsl | 582.0 | 551.7 | 96x128x128 |
| 512x7168x1024 | bfloat16 | 56 | data-parallel | 333.8 | 334.0 | flydsl | 639.7 | 598.1 | 128x128x128 |
| 512x8192x1024 | bfloat16 | 64 | data-parallel | 371.8 | 371.8 | flydsl | 696.9 | 649.5 | 128x128x128 |
| 1024x4096x1024 | bfloat16 | 64 | data-parallel | 371.7 | 371.9 | torch | 649.0 | 651.0 | 128x128x128 |
| 1024x7168x1024 | bfloat16 | 112 | data-parallel | 601.1 | 601.8 | flydsl | 777.2 | 754.3 | 224x128x64 |
| 1024x8192x1024 | bfloat16 | 128 | data-parallel | 661.4 | 659.7 | torch | 813.6 | 815.9 | 256x128x64 |
| 2048x4096x1024 | bfloat16 | 128 | data-parallel | 661.7 | 660.6 | torch | 816.7 | 819.7 | 256x128x64 |
| 2048x7168x1024 | bfloat16 | 224 | data-parallel | 1054.0 | 1055.7 | flydsl | 1027.2 | 999.8 | 256x256x64 |
| 2048x8192x1024 | bfloat16 | 256 | data-parallel | 1106.9 | 1113.6 | flydsl | 1063.6 | 1054.7 | 256x256x64 |
| 3072x4096x1024 | bfloat16 | 192 | data-parallel | 937.8 | 938.5 | torch | 953.2 | 952.8 | 256x192x64 |
| 4096x4096x1024 | bfloat16 | 256 | data-parallel | 1118.3 | 1124.5 | torch | 1102.1 | 1105.1 | 256x256x64 |
| 4096x7168x1024 | bfloat16 | 448 | data-parallel | 1122.2 | 1115.7 | flydsl | 1088.4 | 1013.0 | 224x256x64 |
| 4096x8192x1024 | bfloat16 | 512 | data-parallel | 1171.6 | 1161.5 | asm | 1161.5 | 1123.5 | 256x256x64 |
| 5120x4096x1024 | bfloat16 | 320 | data-parallel | 885.5 | 881.1 | torch | 1022.4 | 1022.9 | 256x160x64 |
| 7680x4096x1024 | bfloat16 | 480 | data-parallel | 1155.2 | 1149.5 | torch | 958.4 | 963.6 | 256x240x64 |
| 8192x7168x1024 | bfloat16 | 896 | data-parallel | 1168.5 | 1149.1 | asm | 1091.8 | 1046.2 | 224x256x64 |
| 8192x8192x1024 | bfloat16 | 1024 | data-parallel | 1237.7 | 1214.1 | asm | 1188.4 | 1187.6 | 256x256x64 |
| 16384x4096x1024 | bfloat16 | 1024 | data-parallel | 1241.4 | 1223.6 | torch | 1188.0 | 1193.2 | 256x256x64 |
| 16384x7168x1024 | bfloat16 | 1792 | data-parallel | 1278.9 | 1251.5 | asm | 1189.8 | 1227.2 | 256x256x64 |
| 16384x8192x1024 | bfloat16 | 2048 | data-parallel | 1259.4 | 1228.9 | torch | 1224.3 | 1227.3 | 256x256x64 |
| 32768x7168x1024 | bfloat16 | 3584 | data-parallel | 1255.1 | 1232.1 | torch | 1240.0 | 1248.7 | 256x256x64 |
| 32768x8192x1024 | bfloat16 | 4096 | data-parallel | 1257.5 | 1219.0 | torch | 1242.5 | 1250.7 | 256x256x64 |
| 512x5120x1280 | bfloat16 | 40 | data-parallel | 263.1 | 262.5 | flydsl | 546.5 | 550.5 | 160x64x128 |
| 1024x5120x1280 | bfloat16 | 80 | data-parallel | 482.7 | 483.0 | torch | 707.6 | 706.9 | 160x128x128 |
| 2048x5120x1280 | bfloat16 | 160 | data-parallel | 867.9 | 871.3 | torch | 922.3 | 924.1 | 160x256x64 |
| 4096x5120x1280 | bfloat16 | 320 | data-parallel | 929.8 | 923.6 | torch | 981.5 | 985.3 | 160x256x64 |
| 8192x5120x1280 | bfloat16 | 640 | data-parallel | 1176.2 | 1161.7 | flydsl | 1129.5 | 1106.2 | 256x224x64 |
| 16384x5120x1280 | bfloat16 | 1280 | data-parallel | 1311.3 | 1286.8 | flydsl | 1230.8 | 1266.6 | 256x256x64 |
| 32768x5120x1280 | bfloat16 | 2560 | data-parallel | 1307.8 | 1281.0 | torch | 1288.7 | 1294.9 | 256x256x64 |
| 512x1536x1536 | bfloat16 | 12 | one-tile reduce-scatter, n = 8 | 119.9 | 85.8 | flydsl | 348.4 | 305.3 | 64x64x256 |
| 512x2048x1536 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 | 153.1 | 113.7 | torch | 398.5 | 400.5 | 64x64x256 |
| 512x2304x1536 | bfloat16 | 18 | one-tile reduce-scatter, n = 7 | 157.2 | 127.0 | flydsl | 394.6 | 380.8 | 64x80x256 |
| 512x4096x1536 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 264.3 | 224.5 | torch | 570.1 | 572.5 | 128x64x128 |
| 512x6144x1536 | bfloat16 | 48 | data-parallel | 322.8 | 323.3 | flydsl | 639.3 | 632.0 | 96x128x128 |
| 512x7168x1536 | bfloat16 | 56 | data-parallel | 372.7 | 373.4 | flydsl | 674.8 | 584.9 | 128x128x256 |
| 1024x1536x1536 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 | 209.5 | 168.6 | flydsl | 508.0 | 474.8 | 96x64x128 |
| 1024x2048x1536 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 263.2 | 221.0 | torch | 570.6 | 573.3 | 128x64x128 |
| 1024x2304x1536 | bfloat16 | 36 | one-tile owner, n = 2 | 245.4 | 248.7 | torch | 514.7 | 510.0 | 128x80x128 |
| 1024x3072x1536 | bfloat16 | 48 | data-parallel | 324.4 | 325.4 | flydsl | 644.3 | 632.8 | 96x128x128 |
| 1024x4096x1536 | bfloat16 | 64 | data-parallel | 417.8 | 418.5 | torch | 626.0 | 625.7 | 128x128x256 |
| 1024x7168x1536 | bfloat16 | 112 | data-parallel | 680.5 | 681.4 | torch | 875.9 | 874.0 | 224x128x64 |
| 2048x1536x1536 | bfloat16 | 48 | data-parallel | 324.2 | 324.4 | torch | 630.2 | 629.9 | 96x128x128 |
| 2048x2048x1536 | bfloat16 | 64 | data-parallel | 417.4 | 418.3 | torch | 624.8 | 624.8 | 128x128x256 |
| 2048x4096x1536 | bfloat16 | 128 | data-parallel | 752.3 | 752.2 | torch | 934.8 | 937.4 | 256x128x64 |
| 3072x2048x1536 | bfloat16 | 96 | data-parallel | 596.2 | 598.0 | opus | 862.7 | 801.4 | 128x192x128 |
| 3072x4096x1536 | bfloat16 | 192 | data-parallel | 1048.3 | 1052.2 | opus | 1086.4 | 1089.7 | 256x192x64 |
| 4096x1536x1536 | bfloat16 | 96 | data-parallel | 595.5 | 595.9 | torch | 802.7 | 802.0 | 192x128x128 |
| 4096x2048x1536 | bfloat16 | 128 | data-parallel | 751.6 | 750.9 | opus | 994.5 | 939.6 | 256x128x64 |
| 4096x3072x1536 | bfloat16 | 192 | data-parallel | 1052.5 | 1057.0 | flydsl | 1013.4 | 1056.4 | 192x256x64 |
| 4096x4096x1536 | bfloat16 | 256 | data-parallel | 1227.4 | 1232.1 | torch | 1208.6 | 1212.2 | 256x256x64 |
| 5120x2048x1536 | bfloat16 | 160 | data-parallel | 909.7 | 911.7 | torch | 1006.6 | 1010.2 | 256x160x64 |
| 5120x4096x1536 | bfloat16 | 320 | data-parallel | 964.7 | 960.1 | torch | 1069.9 | 1074.5 | 256x160x64 |
| 7680x2048x1536 | bfloat16 | 240 | data-parallel | 1201.6 | 1204.2 | torch | 1018.5 | 1020.3 | 256x240x64 |
| 7680x4096x1536 | bfloat16 | 480 | data-parallel | 1253.3 | 1248.8 | torch | 1064.4 | 1069.1 | 256x240x64 |
| 8192x1536x1536 | bfloat16 | 192 | data-parallel | 1049.7 | 1053.1 | torch | 1056.7 | 1061.7 | 192x256x64 |
| 10240x7168x1536 | bfloat16 | 1120 | data-parallel | 1248.4 | 1231.3 | asm | 1180.8 | 1143.8 | 224x256x64 |
| 16384x1536x1536 | bfloat16 | 384 | data-parallel | 1121.6 | 1123.3 | torch | 1114.2 | 1118.2 | 192x256x64 |
| 16384x2048x1536 | bfloat16 | 512 | data-parallel | 1285.8 | 1278.1 | torch | 1251.7 | 1257.2 | 256x256x64 |
| 16384x3072x1536 | bfloat16 | 768 | data-parallel | 1325.3 | 1308.8 | torch | 1278.2 | 1285.6 | 256x256x64 |
| 16384x4096x1536 | bfloat16 | 1024 | data-parallel | 1340.9 | 1322.9 | torch | 1293.4 | 1301.5 | 256x256x64 |
| 32768x1536x1536 | bfloat16 | 768 | data-parallel | 1325.3 | 1307.5 | flydsl | 1254.0 | 1280.8 | 256x256x64 |
| 512x7168x1792 | bfloat16 | 56 | one-tile owner, n = 2 | 406.5 | 388.9 | flydsl | 717.8 | 596.5 | 128x128x256 |
| 1024x7168x1792 | bfloat16 | 112 | data-parallel | 711.1 | 710.8 | flydsl | 850.5 | 916.6 | 224x128x64 |
| 2048x7168x1792 | bfloat16 | 224 | data-parallel | 1212.5 | 1215.1 | flydsl | 1164.0 | 1084.7 | 224x256x64 |
| 4096x7168x1792 | bfloat16 | 448 | data-parallel | 1254.3 | 1254.7 | flydsl | 1211.7 | 1139.8 | 224x256x64 |
| 8192x7168x1792 | bfloat16 | 896 | data-parallel | 1282.7 | 1271.6 | asm | 1215.9 | 1166.0 | 224x256x64 |
| 16384x7168x1792 | bfloat16 | 1792 | data-parallel | 1382.3 | 1370.6 | torch | 1343.4 | 1345.4 | 256x256x64 |
| 32768x7168x1792 | bfloat16 | 3584 | data-parallel | 1386.1 | 1367.0 | torch | 1364.2 | 1371.0 | 256x256x64 |
| 512x4096x2048 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 331.2 | 237.3 | torch | 601.4 | 603.1 | 128x64x128 |
| 512x6144x2048 | float32 | 48 | one-tile owner, n = 2 | 346.8 | 314.3 | asm | 413.9 | 645.5 | 96x128x128 |
| 512x6144x2048 | bfloat16 | 48 | one-tile owner, n = 2 | 391.1 | 348.7 | flydsl | 671.6 | 685.6 | 96x128x128 |
| 512x8192x2048 | bfloat16 | 64 | one-tile owner, n = 2 | 494.9 | 449.9 | flydsl | 793.6 | 672.2 | 128x128x256 |
| 512x16384x2048 | bfloat16 | 128 | data-parallel | 812.6 | 812.0 | flydsl | 820.9 | 992.7 | 256x128x64 |
| 1024x4096x2048 | bfloat16 | 64 | one-tile owner, n = 2 | 492.2 | 447.8 | torch | 673.1 | 672.7 | 128x128x256 |
| 1024x6144x2048 | float32 | 96 | data-parallel | 584.9 | 583.1 | opus | 662.9 | 882.7 | 192x128x64 |
| 1024x6144x2048 | bfloat16 | 96 | data-parallel | 646.7 | 645.4 | flydsl | 804.2 | 872.3 | 192x128x128 |
| 1024x8192x2048 | bfloat16 | 128 | data-parallel | 813.5 | 815.0 | flydsl | 1035.3 | 998.9 | 256x128x64 |
| 1024x16384x2048 | bfloat16 | 256 | data-parallel | 1291.3 | 1296.2 | flydsl | 911.3 | 1267.7 | 256x256x64 |
| 2048x4096x2048 | bfloat16 | 128 | data-parallel | 809.1 | 810.2 | torch | 1001.6 | 1003.3 | 256x128x64 |
| 2048x6144x2048 | float32 | 192 | data-parallel | 1026.3 | 1028.2 | asm | 979.9 | 1035.2 | 192x256x64 |
| 2048x6144x2048 | bfloat16 | 192 | data-parallel | 1138.0 | 1137.2 | flydsl | 1112.4 | 1123.3 | 192x256x64 |
| 2048x8192x2048 | bfloat16 | 256 | data-parallel | 1296.6 | 1303.2 | flydsl | 1238.0 | 1268.8 | 256x256x64 |
| 2048x16384x2048 | bfloat16 | 512 | data-parallel | 1343.6 | 1334.9 | flydsl | 1050.3 | 1293.7 | 256x256x64 |
| 3072x4096x2048 | bfloat16 | 192 | data-parallel | 1131.8 | 1133.3 | torch | 1124.5 | 1129.5 | 256x192x64 |
| 4096x4096x2048 | bfloat16 | 256 | data-parallel | 1305.9 | 1310.1 | torch | 1275.6 | 1281.6 | 256x256x64 |
| 4096x6144x2048 | float32 | 384 | data-parallel | 1046.8 | 1057.9 | asm | 1011.1 | 1082.2 | 192x256x64 |
| 4096x6144x2048 | bfloat16 | 384 | data-parallel | 1187.9 | 1192.6 | flydsl | 1006.8 | 1170.6 | 192x256x64 |
| 4096x8192x2048 | bfloat16 | 512 | data-parallel | 1337.0 | 1329.0 | flydsl | 1277.3 | 1295.5 | 256x256x64 |
| 4096x16384x2048 | bfloat16 | 1024 | data-parallel | 1388.6 | 1375.8 | flydsl | 943.9 | 1339.7 | 256x256x64 |
| 5120x4096x2048 | bfloat16 | 320 | one-tile owner, n = 2 | 1057.0 | 1015.8 | torch | 1180.6 | 1186.4 | 256x160x64 |
| 7680x4096x2048 | bfloat16 | 480 | data-parallel | 1319.7 | 1310.4 | torch | 1128.8 | 1132.7 | 256x240x64 |
| 8192x6144x2048 | float32 | 768 | data-parallel | 1215.4 | 1211.7 | asm | 1164.2 | 1106.7 | 256x256x64 |
| 8192x6144x2048 | bfloat16 | 768 | data-parallel | 1382.7 | 1378.1 | flydsl | 866.9 | 1337.3 | 256x256x64 |
| 8192x8192x2048 | bfloat16 | 1024 | data-parallel | 1393.1 | 1376.5 | asm | 1336.9 | 1344.1 | 256x256x64 |
| 8192x16384x2048 | bfloat16 | 2048 | data-parallel | 1376.9 | 1347.5 | triton | 1029.7 | 1372.6 | 256x256x64 |
| 16384x4096x2048 | bfloat16 | 1024 | data-parallel | 1396.0 | 1386.4 | torch | 1325.3 | 1328.7 | 256x256x64 |
| 16384x6144x2048 | float32 | 1536 | data-parallel | 1219.8 | 1214.3 | asm | 1171.0 | 1128.8 | 256x256x64 |
| 16384x6144x2048 | bfloat16 | 1536 | data-parallel | 1407.1 | 1399.9 | torch | 1363.3 | 1367.3 | 256x256x64 |
| 16384x8192x2048 | bfloat16 | 2048 | data-parallel | 1407.6 | 1402.3 | torch | 1374.6 | 1377.7 | 256x256x64 |
| 16384x16384x2048 | bfloat16 | 4096 | data-parallel | 1384.8 | 1352.5 | flydsl | 961.1 | 1404.3 | 256x256x64 |
| 32768x6144x2048 | bfloat16 | 3072 | data-parallel | 1411.5 | 1392.5 | flydsl | 994.2 | 1397.3 | 256x256x64 |
| 32768x8192x2048 | bfloat16 | 4096 | data-parallel | 1420.3 | 1397.8 | torch | 1392.0 | 1404.6 | 256x256x64 |
| 32768x16384x2048 | bfloat16 | 8192 | data-parallel | 1389.1 | 1348.3 | flydsl | 828.0 | 1411.3 | 256x256x64 |
| 512x6144x3072 | float32 | 48 | one-tile owner, n = 2 | 426.4 | 345.5 | opus | 390.4 | 727.8 | 96x128x128 |
| 512x6144x3072 | bfloat16 | 48 | one-tile owner, n = 2 | 477.8 | 373.6 | flydsl | 725.7 | 759.9 | 96x128x128 |
| 1024x6144x3072 | float32 | 96 | one-tile owner, n = 2-3 | 706.3 | 647.1 | opus | 611.1 | 956.6 | 192x128x64 |
| 1024x6144x3072 | bfloat16 | 96 | one-tile owner, n = 2-3 | 698.4 | 689.8 | flydsl | 876.0 | 942.6 | 192x128x128 |
| 2048x6144x3072 | float32 | 192 | data-parallel | 1127.4 | 1128.4 | asm | 1077.2 | 1109.1 | 192x256x64 |
| 2048x6144x3072 | bfloat16 | 192 | data-parallel | 1234.7 | 1236.2 | flydsl | 1190.6 | 1177.0 | 192x256x64 |
| 4096x6144x3072 | float32 | 384 | data-parallel | 1138.7 | 1148.0 | asm | 1102.4 | 1149.5 | 192x256x64 |
| 4096x6144x3072 | bfloat16 | 384 | data-parallel | 1248.2 | 1249.2 | flydsl | 1075.1 | 1214.4 | 192x256x64 |
| 8192x6144x3072 | float32 | 768 | data-parallel | 1302.0 | 1299.5 | asm | 1248.1 | 1183.6 | 256x256x64 |
| 8192x6144x3072 | bfloat16 | 768 | data-parallel | 1426.4 | 1427.7 | flydsl | 846.1 | 1385.2 | 256x256x64 |
| 16384x6144x3072 | float32 | 1536 | data-parallel | 1312.6 | 1299.9 | asm | 1273.5 | 1221.3 | 256x256x64 |
| 16384x6144x3072 | bfloat16 | 1536 | data-parallel | 1446.1 | 1441.2 | torch | 1387.9 | 1406.7 | 256x256x64 |
| 32768x6144x3072 | bfloat16 | 3072 | data-parallel | 1470.5 | 1450.4 | flydsl | 857.0 | 1441.3 | 256x256x64 |
| 512x5120x3200 | bfloat16 | 40 | one-tile owner, n = 2 | 420.3 | 317.9 | flydsl | 622.3 | 689.3 | 160x64x128 |
| 1024x5120x3200 | bfloat16 | 80 | one-tile owner, n = 2 | 751.0 | 586.1 | torch | 925.4 | 927.8 | 160x128x128 |
| 2048x5120x3200 | bfloat16 | 160 | data-parallel | 1126.7 | 1122.5 | torch | 1108.6 | 1112.2 | 160x256x64 |
| 4096x5120x3200 | bfloat16 | 320 | one-tile owner, n = 2 | 1173.0 | 1063.7 | torch | 1133.3 | 1136.9 | 160x256x64 |
| 8192x5120x3200 | bfloat16 | 640 | data-parallel | 1324.6 | 1320.7 | asm | 1264.1 | 1241.8 | 256x224x64 |
| 16384x5120x3200 | bfloat16 | 1280 | data-parallel | 1437.5 | 1436.3 | torch | 1399.2 | 1398.2 | 256x256x64 |
| 32768x5120x3200 | bfloat16 | 2560 | data-parallel | 1475.0 | 1459.8 | torch | 1418.4 | 1425.9 | 256x256x64 |
| 512x7168x3584 | bfloat16 | 56 | one-tile owner, n = 2 | 592.3 | 438.9 | flydsl | 832.2 | 699.5 | 128x128x256 |
| 512x8192x3584 | bfloat16 | 64 | one-tile owner, n = 2 | 645.7 | 492.2 | flydsl | 772.6 | 770.2 | 128x128x256 |
| 1024x7168x3584 | bfloat16 | 112 | one-tile owner, n = 2-3 | 872.3 | 814.1 | torch | 1030.9 | 1036.1 | 224x128x64 |
| 1024x8192x3584 | bfloat16 | 128 | data-parallel | 898.8 | 923.1 | flydsl | 877.3 | 1086.8 | 256x128x64 |
| 2048x8192x3584 | bfloat16 | 256 | data-parallel | 1377.7 | 1384.6 | flydsl | 1312.6 | 1342.5 | 256x256x64 |
| 4096x8192x3584 | bfloat16 | 512 | data-parallel | 1417.2 | 1417.7 | flydsl | 1030.0 | 1375.6 | 256x256x64 |
| 8192x8192x3584 | bfloat16 | 1024 | data-parallel | 1444.2 | 1437.5 | torch | 1395.6 | 1397.5 | 256x256x64 |
| 10240x7168x3584 | bfloat16 | 1120 | one-tile owner, n = 2-3 | 1361.3 | 1347.8 | flydsl | 826.0 | 1257.4 | 224x256x64 |
| 16384x8192x3584 | bfloat16 | 2048 | data-parallel | 1470.1 | 1464.2 | flydsl | 845.4 | 1417.7 | 256x256x64 |
| 32768x8192x3584 | bfloat16 | 4096 | data-parallel | 1493.0 | 1477.8 | flydsl | 868.7 | 1457.5 | 256x256x64 |
| 512x1024x4096 | bfloat16 | 8 | one-tile reduce-scatter, n = 8 | 199.3 | 66.8 | flydsl | 342.9 | 329.1 | 64x32x256 |
| 512x2048x4096 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 | 333.2 | 133.2 | flydsl | 515.9 | 506.1 | 64x64x256 |
| 512x4096x4096 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 503.7 | 264.1 | flydsl | 702.7 | 617.3 | 128x128x128 |
| 512x8192x4096 | bfloat16 | 64 | one-tile owner, n = 2 | 690.5 | 506.1 | flydsl | 868.7 | 796.3 | 128x128x256 |
| 512x8704x4096 | bfloat16 | 68 | one-tile owner, n = 2 | 727.2 | 534.4 | torch | 891.0 | 896.7 | 160x128x128 |
| 512x10240x4096 | bfloat16 | 80 | one-tile owner, n = 2 | 801.6 | 622.1 | torch | 870.6 | 868.3 | 128x192x64 |
| 512x16384x4096 | bfloat16 | 128 | one-tile owner, n = 2 | 1039.4 | 947.2 | flydsl | 800.6 | 701.2 | 256x256x64 |
| 512x129280x4096 | bfloat16 | 1010 | data-parallel | 1349.3 | 1353.3 | asm | 1281.1 | 1142.2 | 256x256x64 |
| 768x1024x4096 | bfloat16 | 12 | one-tile reduce-scatter, n = 8 | 274.7 | 100.2 | torch | 428.0 | 431.5 | 64x48x256 |
| 768x2048x4096 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 | 423.8 | 199.5 | flydsl | 551.9 | 562.0 | 64x96x256 |
| 768x4096x4096 | bfloat16 | 48 | one-tile owner, n = 3 | 617.2 | 393.1 | flydsl | 760.2 | 795.7 | 128x96x128 |
| 768x8192x4096 | bfloat16 | 96 | one-tile owner, n = 2-3 | 888.1 | 740.5 | torch | 978.2 | 983.4 | 128x192x128 |
| 768x8704x4096 | bfloat16 | 102 | one-tile owner, n = 2-3 | 919.6 | 791.1 | flydsl | 946.2 | 945.0 | 160x192x128 |
| 768x10240x4096 | bfloat16 | 120 | one-tile owner, n = 2-3 | 1027.6 | 925.3 | flydsl | 1141.3 | 1045.6 | 160x192x128 |
| 768x129280x4096 | bfloat16 | 1515 | data-parallel | 1422.0 | 1404.2 | torch | 1350.2 | 1357.4 | 256x256x64 |
| 1024x512x4096 | bfloat16 | 8 | one-tile reduce-scatter, n = 8 | 199.6 | 66.6 | flydsl | 325.4 | 328.9 | 64x32x256 |
| 1024x1024x4096 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 | 333.9 | 133.1 | flydsl | 502.8 | 520.6 | 64x64x256 |
| 1024x2048x4096 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 501.0 | 263.2 | flydsl | 695.9 | 537.6 | 128x128x256 |
| 1024x4096x4096 | bfloat16 | 64 | one-tile owner, n = 2 | 703.2 | 505.7 | flydsl | 896.7 | 800.8 | 128x128x256 |
| 1024x8192x4096 | bfloat16 | 128 | one-tile owner, n = 2 | 1040.8 | 935.9 | flydsl | 1147.1 | 700.4 | 256x256x64 |
| 1024x8704x4096 | bfloat16 | 136 | data-parallel | 1051.0 | 1047.5 | torch | 1073.9 | 1072.8 | 144x256x64 |
| 1024x10240x4096 | bfloat16 | 160 | data-parallel | 1162.6 | 1166.8 | torch | 1137.1 | 1140.8 | 160x256x64 |
| 1024x16384x4096 | bfloat16 | 256 | data-parallel | 1410.3 | 1410.4 | flydsl | 899.5 | 1378.4 | 256x256x64 |
| 1024x129280x4096 | bfloat16 | 2020 | data-parallel | 1451.7 | 1431.6 | torch | 1370.6 | 1377.9 | 256x256x64 |
| 1280x512x4096 | bfloat16 | 10 | one-tile reduce-scatter, n = 8 | 239.5 | 83.7 | flydsl | 336.2 | 376.7 | 64x48x256 |
| 1280x1024x4096 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 | 369.7 | 166.0 | torch | 551.1 | 552.3 | 64x80x256 |
| 1280x2048x4096 | bfloat16 | 40 | one-tile owner, n = 3 | 517.7 | 329.3 | flydsl | 651.3 | 715.6 | 128x80x128 |
| 1280x129280x4096 | bfloat16 | 2525 | data-parallel | 1409.1 | 1403.0 | torch | 1387.6 | 1395.6 | 256x256x64 |
| 1536x512x4096 | bfloat16 | 12 | one-tile reduce-scatter, n = 8 | 274.0 | 100.1 | torch | 430.9 | 431.6 | 64x48x256 |
| 1536x1024x4096 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 | 423.6 | 199.3 | flydsl | 545.2 | 563.1 | 64x96x256 |
| 1536x2048x4096 | bfloat16 | 48 | one-tile owner, n = 3 | 610.0 | 391.6 | flydsl | 758.8 | 796.6 | 128x96x128 |
| 1536x129280x4096 | bfloat16 | 3030 | data-parallel | 1440.0 | 1415.2 | torch | 1390.6 | 1397.9 | 256x256x64 |
| 1792x512x4096 | bfloat16 | 14 | one-tile reduce-scatter, n = 8 | 306.6 | 117.1 | flydsl | 415.5 | 473.4 | 64x64x256 |
| 1792x1024x4096 | bfloat16 | 28 | one-tile reduce-scatter, n = 4 | 455.8 | 233.0 | flydsl | 607.0 | 500.9 | 256x128x64 |
| 1792x2048x4096 | bfloat16 | 56 | one-tile owner, n = 2 | 630.9 | 449.7 | flydsl | 821.8 | 728.1 | 128x128x256 |
| 1792x129280x4096 | bfloat16 | 3535 | data-parallel | 1458.2 | 1442.0 | asm | 1415.1 | 1364.0 | 256x256x64 |
| 2048x512x4096 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 | 334.0 | 133.1 | flydsl | 477.9 | 522.1 | 64x64x256 |
| 2048x1024x4096 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 503.9 | 263.7 | flydsl | 699.9 | 541.1 | 128x128x256 |
| 2048x2048x4096 | bfloat16 | 64 | one-tile owner, n = 2 | 699.5 | 505.3 | flydsl | 881.4 | 801.2 | 128x128x256 |
| 2048x4096x4096 | bfloat16 | 128 | one-tile owner, n = 2 | 1042.9 | 925.4 | opus | 1141.6 | 695.6 | 256x256x64 |
| 2048x8192x4096 | bfloat16 | 256 | data-parallel | 1400.7 | 1400.3 | flydsl | 1324.2 | 1361.8 | 256x256x64 |
| 2048x8704x4096 | bfloat16 | 272 | one-tile reduce-scatter, n = 8 | 1220.4 | 932.1 | torch | 1079.2 | 1090.7 | 224x160x64 |
| 2048x10240x4096 | bfloat16 | 320 | one-tile owner, n = 2 | 1218.2 | 1087.7 | torch | 1160.7 | 1163.9 | 160x256x64 |
| 2048x16384x4096 | bfloat16 | 512 | data-parallel | 1430.3 | 1429.7 | flydsl | 1361.6 | 1381.7 | 256x256x64 |
| 2048x129280x4096 | bfloat16 | 4040 | data-parallel | 1484.6 | 1476.2 | flydsl | 1426.5 | 1380.3 | 256x256x64 |
| 2304x512x4096 | bfloat16 | 18 | one-tile reduce-scatter, n = 7 | 343.9 | 149.8 | flydsl | 438.5 | 519.4 | 64x80x256 |
| 2304x1024x4096 | bfloat16 | 36 | one-tile owner, n = 3 | 469.7 | 296.2 | flydsl | 643.6 | 712.8 | 128x80x128 |
| 2304x2048x4096 | bfloat16 | 72 | one-tile owner, n = 2 | 742.9 | 561.3 | opus | 867.3 | 933.5 | 128x160x128 |
| 2560x512x4096 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 | 370.0 | 165.8 | flydsl | 479.0 | 554.5 | 64x80x256 |
| 2560x1024x4096 | bfloat16 | 40 | one-tile owner, n = 3 | 516.7 | 329.2 | flydsl | 688.1 | 715.4 | 128x80x128 |
| 2560x2048x4096 | bfloat16 | 80 | one-tile owner, n = 2 | 803.3 | 619.0 | opus | 955.7 | 967.8 | 128x160x128 |
| 2816x512x4096 | bfloat16 | 22 | one-tile reduce-scatter, n = 5 | 396.9 | 182.1 | flydsl | 515.1 | 529.2 | 64x96x256 |
| 2816x1024x4096 | bfloat16 | 44 | one-tile owner, n = 3 | 566.0 | 361.4 | flydsl | 726.5 | 747.1 | 128x96x128 |
| 2816x2048x4096 | bfloat16 | 88 | one-tile owner, n = 2-3 | 821.4 | 680.5 | opus | 999.3 | 895.6 | 192x128x128 |
| 3072x512x4096 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 | 422.1 | 199.7 | flydsl | 547.2 | 565.6 | 64x96x256 |
| 3072x1024x4096 | bfloat16 | 48 | one-tile owner, n = 3 | 605.7 | 392.1 | flydsl | 752.0 | 795.4 | 128x96x128 |
| 3072x2048x4096 | bfloat16 | 96 | one-tile owner, n = 2-3 | 803.8 | 728.8 | opus | 1071.0 | 983.6 | 128x192x128 |
| 3328x512x4096 | bfloat16 | 26 | one-tile reduce-scatter, n = 4 | 430.4 | 216.0 | flydsl | 573.5 | 472.7 | 256x112x64 |
| 3328x1024x4096 | bfloat16 | 52 | one-tile owner, n = 2 | 587.0 | 423.2 | flydsl | 775.2 | 691.6 | 128x128x256 |
| 3328x2048x4096 | bfloat16 | 104 | one-tile owner, n = 2-3 | 911.6 | 781.6 | opus | 1071.6 | 790.5 | 256x208x64 |
| 3584x512x4096 | bfloat16 | 28 | one-tile reduce-scatter, n = 4 | 456.0 | 231.8 | flydsl | 599.1 | 472.3 | 256x128x64 |
| 3584x1024x4096 | bfloat16 | 56 | one-tile owner, n = 2 | 630.0 | 449.7 | flydsl | 781.2 | 737.7 | 128x128x256 |
| 3584x2048x4096 | bfloat16 | 112 | one-tile owner, n = 2-3 | 923.0 | 844.7 | opus | 1099.1 | 934.3 | 160x192x128 |
| 3840x512x4096 | bfloat16 | 30 | one-tile reduce-scatter, n = 4 | 477.0 | 249.2 | flydsl | 624.3 | 534.6 | 128x128x256 |
| 3840x1024x4096 | bfloat16 | 60 | one-tile owner, n = 2 | 670.8 | 481.8 | flydsl | 811.2 | 771.9 | 128x128x256 |
| 3840x2048x4096 | bfloat16 | 120 | one-tile owner, n = 2-3 | 1070.5 | 883.8 | opus | 1119.0 | 779.0 | 256x240x64 |
| 4096x512x4096 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 498.5 | 264.8 | flydsl | 679.9 | 548.4 | 128x128x256 |
| 4096x1024x4096 | bfloat16 | 64 | one-tile owner, n = 2 | 702.9 | 506.8 | flydsl | 882.6 | 806.2 | 128x128x256 |
| 4096x2048x4096 | bfloat16 | 128 | one-tile owner, n = 2 | 1037.5 | 966.2 | opus | 1145.5 | 726.9 | 256x256x64 |
| 4096x4096x4096 | bfloat16 | 256 | data-parallel | 1402.1 | 1402.1 | flydsl | 1325.8 | 1344.5 | 256x256x64 |
| 4096x8192x4096 | bfloat16 | 512 | data-parallel | 1426.5 | 1422.0 | asm | 1372.9 | 1377.6 | 256x256x64 |
| 4096x8704x4096 | bfloat16 | 544 | one-tile reduce-scatter, n = 4 | 1331.2 | 1157.5 | torch | 1196.2 | 1177.5 | 192x256x64 |
| 4096x10240x4096 | bfloat16 | 640 | one-tile owner, n = 2 | 1370.8 | 1346.6 | flydsl | 1291.8 | 1226.7 | 224x256x64 |
| 4096x16384x4096 | bfloat16 | 1024 | data-parallel | 1450.2 | 1448.7 | flydsl | 1384.4 | 1411.1 | 256x256x64 |
| 4096x129280x4096 | bfloat16 | 8080 | data-parallel | 1485.2 | 1483.5 | asm | 1418.1 | 1396.5 | 256x256x64 |
| 4352x512x4096 | bfloat16 | 34 | one-tile owner, n = 3 | 443.8 | 281.9 | flydsl | 534.1 | 632.9 | 128x80x128 |
| 4352x1024x4096 | bfloat16 | 68 | one-tile owner, n = 2 | 715.2 | 534.3 | opus | 835.8 | 894.5 | 128x160x128 |
| 4352x2048x4096 | bfloat16 | 136 | data-parallel | 1017.2 | 1011.2 | torch | 1043.4 | 1043.2 | 192x192x64 |
| 5120x512x4096 | bfloat16 | 40 | one-tile owner, n = 3 | 518.0 | 329.7 | flydsl | 568.3 | 716.0 | 128x80x128 |
| 5120x1024x4096 | bfloat16 | 80 | one-tile owner, n = 2 | 807.6 | 620.4 | opus | 984.0 | 970.4 | 128x160x128 |
| 5120x2048x4096 | bfloat16 | 160 | data-parallel | 1157.8 | 1160.3 | torch | 1175.5 | 1178.5 | 256x160x64 |
| 6144x512x4096 | bfloat16 | 48 | one-tile owner, n = 3 | 606.5 | 390.5 | flydsl | 762.8 | 796.2 | 128x96x128 |
| 6144x1024x4096 | bfloat16 | 96 | one-tile owner, n = 2-3 | 854.8 | 726.2 | opus | 1074.5 | 994.1 | 128x192x128 |
| 6144x2048x4096 | bfloat16 | 192 | data-parallel | 1271.6 | 1273.8 | opus | 1261.3 | 1214.7 | 256x192x64 |
| 6144x129280x4096 | bfloat16 | 12120 | one-tile owner, n = 2-3 | 1469.8 | 1468.2 | asm | 1410.6 | 1398.9 | 256x256x64 |
| 6400x512x4096 | bfloat16 | 50 | one-tile owner, n = 3 | 624.8 | 405.8 | flydsl | 582.1 | 665.9 | 128x128x256 |
| 6400x1024x4096 | bfloat16 | 100 | one-tile owner, n = 2-3 | 916.6 | 768.1 | opus | 1054.2 | 807.4 | 256x208x64 |
| 6400x2048x4096 | bfloat16 | 200 | data-parallel | 1288.2 | 1288.1 | torch | 1151.0 | 1155.1 | 256x208x64 |
| 6912x512x4096 | bfloat16 | 54 | one-tile owner, n = 2 | 606.8 | 435.8 | flydsl | 588.8 | 708.6 | 128x128x256 |
| 6912x1024x4096 | bfloat16 | 108 | one-tile owner, n = 2-3 | 957.0 | 825.8 | opus | 1093.6 | 699.1 | 256x224x64 |
| 6912x2048x4096 | bfloat16 | 216 | data-parallel | 1324.0 | 1325.1 | torch | 1228.6 | 1228.7 | 256x224x64 |
| 7168x512x4096 | bfloat16 | 56 | one-tile owner, n = 2 | 627.7 | 447.6 | flydsl | 590.1 | 732.8 | 128x128x256 |
| 7168x1024x4096 | bfloat16 | 112 | one-tile owner, n = 2-3 | 1013.7 | 845.7 | opus | 1120.2 | 1053.2 | 256x128x64 |
| 7168x2048x4096 | bfloat16 | 224 | data-parallel | 1347.8 | 1348.7 | torch | 1245.7 | 1247.7 | 256x224x64 |
| 7424x512x4096 | bfloat16 | 58 | one-tile owner, n = 2 | 649.4 | 465.5 | flydsl | 591.7 | 749.1 | 128x128x256 |
| 7424x1024x4096 | bfloat16 | 116 | one-tile owner, n = 2-3 | 1027.1 | 890.4 | opus | 1112.7 | 778.8 | 256x240x64 |
| 7424x2048x4096 | bfloat16 | 232 | data-parallel | 1359.3 | 1360.5 | torch | 1179.8 | 1182.4 | 256x240x64 |
| 7680x512x4096 | bfloat16 | 60 | one-tile owner, n = 2 | 660.9 | 477.5 | flydsl | 594.8 | 765.5 | 128x128x256 |
| 7680x1024x4096 | bfloat16 | 120 | one-tile owner, n = 2-3 | 1028.3 | 922.0 | opus | 1137.6 | 755.1 | 256x240x64 |
| 7680x2048x4096 | bfloat16 | 240 | data-parallel | 1381.4 | 1378.9 | torch | 1196.8 | 1202.0 | 256x240x64 |
| 7936x512x4096 | bfloat16 | 62 | one-tile owner, n = 2 | 673.1 | 491.1 | flydsl | 795.8 | 787.2 | 128x128x256 |
| 7936x1024x4096 | bfloat16 | 124 | one-tile owner, n = 2-3 | 1069.3 | 929.3 | opus | 1136.3 | 721.3 | 256x256x64 |
| 7936x2048x4096 | bfloat16 | 248 | data-parallel | 1389.6 | 1389.7 | torch | 1355.8 | 1356.2 | 256x256x64 |
| 8192x512x4096 | bfloat16 | 64 | one-tile owner, n = 2 | 686.4 | 504.8 | flydsl | 866.8 | 808.3 | 128x128x256 |
| 8192x1024x4096 | bfloat16 | 128 | one-tile owner, n = 2 | 1044.0 | 929.1 | opus | 1161.9 | 666.2 | 256x256x64 |
| 8192x2048x4096 | bfloat16 | 256 | data-parallel | 1407.9 | 1409.3 | torch | 1374.0 | 1372.4 | 256x256x64 |
| 8192x4096x4096 | bfloat16 | 512 | data-parallel | 1435.0 | 1435.9 | flydsl | 1365.2 | 1393.6 | 256x256x64 |
| 8192x8192x4096 | bfloat16 | 1024 | data-parallel | 1451.9 | 1451.2 | asm | 1402.2 | 1409.3 | 256x256x64 |
| 8192x8704x4096 | bfloat16 | 1088 | one-tile owner, n = 2 | 1387.0 | 1330.3 | asm | 1289.1 | 1277.6 | 256x224x64 |
| 8192x10240x4096 | bfloat16 | 1280 | data-parallel | 1465.5 | 1466.8 | torch | 1420.3 | 1423.2 | 256x256x64 |
| 8192x16384x4096 | bfloat16 | 2048 | data-parallel | 1495.0 | 1489.6 | torch | 1434.8 | 1450.4 | 256x256x64 |
| 8192x129280x4096 | bfloat16 | 16160 | one-tile reduce-scatter, n = 4 | 1497.5 | 1492.8 | flydsl | 1423.4 | 1387.2 | 256x256x64 |
| 10240x512x4096 | bfloat16 | 80 | one-tile owner, n = 2 | 801.3 | 614.3 | torch | 977.5 | 980.1 | 128x160x128 |
| 10240x1024x4096 | bfloat16 | 160 | data-parallel | 1152.1 | 1158.8 | torch | 1180.9 | 1190.6 | 256x160x64 |
| 10240x2048x4096 | bfloat16 | 320 | one-tile owner, n = 2 | 1223.7 | 1092.6 | torch | 1202.8 | 1202.8 | 256x160x64 |
| 10240x129280x4096 | bfloat16 | 20200 | data-parallel | 1469.3 | 1470.9 | asm | 1431.5 | 1389.8 | 256x256x64 |
| 12288x512x4096 | bfloat16 | 96 | one-tile owner, n = 2-3 | 805.4 | 721.6 | opus | 1082.2 | 991.5 | 128x192x128 |
| 12288x1024x4096 | bfloat16 | 192 | data-parallel | 1275.5 | 1280.0 | opus | 1269.5 | 1227.0 | 256x192x64 |
| 12288x2048x4096 | bfloat16 | 384 | one-tile owner, n = 2 | 1312.7 | 1284.6 | opus | 1288.0 | 1249.6 | 256x192x64 |
| 12288x129280x4096 | bfloat16 | 24240 | data-parallel | 1499.4 | 1500.5 | asm | 1439.7 | 1382.9 | 256x256x64 |
| 14336x512x4096 | bfloat16 | 112 | one-tile owner, n = 2-3 | 1000.4 | 834.7 | opus | 1140.9 | 693.7 | 256x224x64 |
| 14336x1024x4096 | bfloat16 | 224 | data-parallel | 1357.2 | 1358.2 | torch | 1261.4 | 1262.6 | 256x224x64 |
| 14336x2048x4096 | bfloat16 | 448 | data-parallel | 1376.9 | 1381.3 | flydsl | 1321.4 | 1288.4 | 256x224x64 |
| 14336x129280x4096 | bfloat16 | 28280 | one-tile owner, n = 2-3 | 1460.3 | 1448.0 | asm | 1422.2 | 1397.1 | 256x256x64 |
| 16384x512x4096 | bfloat16 | 128 | one-tile owner, n = 2 | 1046.4 | 946.9 | opus | 1176.3 | 772.8 | 256x256x64 |
| 16384x1024x4096 | bfloat16 | 256 | data-parallel | 1418.4 | 1417.4 | flydsl | 1344.8 | 1322.4 | 256x256x64 |
| 16384x2048x4096 | bfloat16 | 512 | data-parallel | 1443.7 | 1444.7 | torch | 1386.5 | 1392.4 | 256x256x64 |
| 16384x16384x4096 | bfloat16 | 4096 | data-parallel | 1515.4 | 1508.9 | torch | 1447.8 | 1494.5 | 256x256x64 |
| 16384x129280x4096 | bfloat16 | 32320 | one-tile owner, n = 2 | 1474.1 | 1492.4 | asm | 1425.1 | 1400.0 | 256x256x64 |
| 32768x16384x4096 | bfloat16 | 8192 | data-parallel | 1528.3 | 1504.7 | flydsl | 1430.4 | 1471.4 | 256x256x64 |
| 512x7168x4224 | bfloat16 | 56 | one-tile owner, n = 2 | 646.2 | 448.9 | flydsl | 830.1 | 856.1 | 128x128x128 |
| 1024x7168x4224 | bfloat16 | 112 | one-tile owner, n = 2-3 | 1036.0 | 838.2 | flydsl | 1101.8 | 819.6 | 240x256x64 |
| 10240x7168x4224 | bfloat16 | 1120 | one-tile owner, n = 2-3 | 1399.0 | 1349.0 | opus | 946.3 | 1264.7 | 224x256x64 |
| 512x6144x4608 | bfloat16 | 48 | one-tile owner, n = 3 | 616.9 | 396.2 | flydsl | 784.6 | 663.7 | 192x128x128 |
| 1024x6144x4608 | bfloat16 | 96 | one-tile owner, n = 2-3 | 900.5 | 746.8 | torch | 986.3 | 990.1 | 192x128x128 |
| 2048x6144x4608 | bfloat16 | 192 | data-parallel | 1282.5 | 1281.8 | flydsl | 1231.9 | 1214.1 | 192x256x64 |
| 4096x6144x4608 | bfloat16 | 384 | one-tile owner, n = 2 | 1321.7 | 1289.7 | torch | 1226.1 | 1237.2 | 192x256x64 |
| 8192x6144x4608 | bfloat16 | 768 | data-parallel | 1455.4 | 1451.5 | torch | 1408.9 | 1411.6 | 256x256x64 |
| 16384x6144x4608 | bfloat16 | 1536 | data-parallel | 1475.2 | 1468.3 | torch | 1427.7 | 1430.6 | 256x256x64 |
| 32768x6144x4608 | bfloat16 | 3072 | data-parallel | 1506.5 | 1495.6 | torch | 1452.6 | 1465.1 | 256x256x64 |
| 512x1024x5120 | float32 | 8 | one-tile reduce-scatter, n = 10 | 220.5 | 65.5 | asm | 332.7 | 338.6 | 64x32x256 |
| 512x5120x5120 | bfloat16 | 40 | one-tile owner, n = 3 | 560.4 | 334.8 | torch | 769.4 | 771.6 | 96x128x128 |
| 512x6400x5120 | bfloat16 | 50 | one-tile owner, n = 3 | 676.8 | 415.3 | flydsl | 701.3 | 700.6 | 128x128x256 |
| 512x12800x5120 | bfloat16 | 100 | one-tile owner, n = 2-3 | 972.5 | 806.6 | flydsl | 780.8 | 813.7 | 208x256x64 |
| 512x37888x5120 | bfloat16 | 296 | one-tile owner, n = 3 | 1153.5 | 936.9 | torch | 1048.1 | 1051.8 | 160x256x64 |
| 512x51200x5120 | bfloat16 | 400 | data-parallel | 1196.2 | 1200.1 | flydsl | 1126.9 | 1085.8 | 208x256x64 |
| 768x1024x5120 | float32 | 12 | one-tile reduce-scatter, n = 10 | 303.0 | 98.2 | asm | 406.6 | 381.1 | 64x64x256 |
| 768x37888x5120 | bfloat16 | 444 | data-parallel | 1308.7 | 1312.1 | torch | 1218.8 | 1222.4 | 224x256x64 |
| 1024x1024x5120 | float32 | 16 | one-tile reduce-scatter, n = 8 | 369.6 | 130.4 | asm | 462.4 | 483.5 | 64x64x256 |
| 1024x5120x5120 | bfloat16 | 80 | one-tile owner, n = 2 | 878.2 | 641.9 | torch | 971.9 | 974.3 | 160x128x128 |
| 1024x6400x5120 | bfloat16 | 100 | one-tile owner, n = 2-3 | 976.3 | 803.7 | flydsl | 937.3 | 846.3 | 256x208x64 |
| 1024x12800x5120 | bfloat16 | 200 | data-parallel | 1308.4 | 1309.7 | flydsl | 759.9 | 1169.1 | 256x208x64 |
| 1024x37888x5120 | bfloat16 | 592 | one-tile owner, n = 2 | 1316.9 | 1211.8 | torch | 1058.6 | 1058.8 | 256x208x64 |
| 1024x51200x5120 | bfloat16 | 800 | one-tile reduce-scatter, n = 4 | 1363.0 | 1201.6 | asm | 1141.7 | 1072.0 | 256x208x64 |
| 1536x1024x5120 | float32 | 24 | one-tile reduce-scatter, n = 5 | 448.3 | 194.6 | asm | 578.9 | 596.4 | 64x96x128 |
| 1536x37888x5120 | bfloat16 | 888 | one-tile owner, n = 2-3 | 1385.7 | 1318.6 | torch | 1242.1 | 1249.6 | 224x256x64 |
| 2048x1024x5120 | float32 | 32 | one-tile reduce-scatter, n = 4 | 525.5 | 255.9 | opus | 637.8 | 624.0 | 128x128x128 |
| 2048x5120x5120 | bfloat16 | 160 | data-parallel | 1190.9 | 1193.7 | torch | 1160.3 | 1162.0 | 160x256x64 |
| 2048x6400x5120 | bfloat16 | 200 | data-parallel | 1312.5 | 1315.8 | flydsl | 1224.9 | 1179.3 | 256x208x64 |
| 2048x12800x5120 | bfloat16 | 400 | data-parallel | 1325.2 | 1331.4 | flydsl | 832.2 | 1197.5 | 256x208x64 |
| 2048x37888x5120 | bfloat16 | 1184 | data-parallel | 1434.5 | 1423.6 | torch | 1195.5 | 1199.0 | 240x256x64 |
| 2048x51200x5120 | bfloat16 | 1600 | one-tile owner, n = 2 | 1437.1 | 1379.2 | asm | 212.2 | 1170.6 | 240x256x64 |
| 4096x1024x5120 | float32 | 64 | one-tile owner, n = 2 | 694.9 | 485.5 | opus | 946.8 | 887.4 | 128x128x128 |
| 4096x5120x5120 | bfloat16 | 320 | one-tile owner, n = 2 | 1256.2 | 1098.6 | torch | 1178.7 | 1182.5 | 160x256x64 |
| 4096x6400x5120 | bfloat16 | 400 | data-parallel | 1322.7 | 1330.5 | flydsl | 1280.6 | 1202.6 | 256x208x64 |
| 4096x12800x5120 | bfloat16 | 800 | one-tile reduce-scatter, n = 4 | 1398.9 | 1254.9 | asm | 1213.4 | 1197.2 | 256x208x64 |
| 4096x51200x5120 | bfloat16 | 3200 | one-tile owner, n = 2 | 1488.7 | 1463.8 | asm | 1382.7 | 1375.9 | 256x256x64 |
| 8192x1024x5120 | float32 | 128 | one-tile owner, n = 2 | 1051.9 | 974.0 | opus | 1165.2 | 1078.8 | 256x128x64 |
| 8192x5120x5120 | bfloat16 | 640 | one-tile owner, n = 2 | 1400.4 | 1366.0 | torch | 1272.9 | 1280.9 | 256x224x64 |
| 8192x6400x5120 | bfloat16 | 800 | one-tile reduce-scatter, n = 4 | 1402.3 | 1257.0 | flydsl | 1214.1 | 1199.1 | 256x208x64 |
| 8192x12800x5120 | bfloat16 | 1600 | one-tile owner, n = 2 | 1439.5 | 1385.9 | asm | 1333.9 | 1204.0 | 256x240x64 |
| 8192x51200x5120 | bfloat16 | 6400 | data-parallel | 1532.6 | 1525.9 | flydsl | 1446.8 | 1416.5 | 256x256x64 |
| 16384x1024x5120 | float32 | 256 | data-parallel | 1360.7 | 1364.3 | opus | 1266.0 | 1241.6 | 256x256x64 |
| 16384x5120x5120 | bfloat16 | 1280 | data-parallel | 1480.1 | 1478.4 | torch | 1446.4 | 1449.9 | 256x256x64 |
| 16384x6400x5120 | bfloat16 | 1600 | one-tile owner, n = 2 | 1447.1 | 1383.9 | flydsl | 1326.8 | 1218.7 | 256x240x64 |
| 16384x12800x5120 | bfloat16 | 3200 | one-tile owner, n = 2 | 1519.7 | 1483.3 | torch | 1461.4 | 1482.0 | 256x256x64 |
| 16384x51200x5120 | bfloat16 | 12800 | data-parallel | 1530.1 | 1526.6 | flydsl | 1480.9 | 1414.9 | 256x256x64 |
| 32768x5120x5120 | bfloat16 | 2560 | data-parallel | 1520.9 | 1511.6 | torch | 1478.1 | 1495.6 | 256x256x64 |
| 32768x6400x5120 | bfloat16 | 3200 | one-tile owner, n = 2 | 1511.7 | 1474.9 | torch | 1470.3 | 1481.4 | 256x256x64 |
| 32768x12800x5120 | bfloat16 | 6400 | data-parallel | 1535.0 | 1519.9 | torch | 1488.8 | 1523.0 | 256x256x64 |
| 32768x51200x5120 | bfloat16 | 25600 | data-parallel | 1540.6 | 1522.7 | flydsl | 1436.9 | 1396.9 | 256x256x64 |
| 512x1280x6144 | bfloat16 | 10 | one-tile reduce-scatter, n = 12 | 305.4 | 86.4 | flydsl | 404.5 | 379.1 | 160x64x128 |
| 512x1536x6144 | bfloat16 | 12 | one-tile reduce-scatter, n = 10 | 349.5 | 103.7 | flydsl | 432.2 | 408.6 | 192x64x128 |
| 512x2048x6144 | float32 | 16 | one-tile reduce-scatter, n = 8 | 414.1 | 133.8 | asm | 492.4 | 509.2 | 64x64x256 |
| 512x2304x6144 | bfloat16 | 18 | one-tile reduce-scatter, n = 7 | 437.1 | 154.5 | flydsl | 508.1 | 410.7 | 192x128x128 |
| 512x2560x6144 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 | 466.6 | 171.4 | torch | 506.4 | 507.4 | 256x128x64 |
| 512x3072x6144 | float32 | 24 | one-tile reduce-scatter, n = 5 | 499.9 | 199.4 | opus | 488.7 | 608.2 | 96x128x128 |
| 512x3072x6144 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 | 519.2 | 205.6 | flydsl | 598.1 | 495.1 | 192x128x128 |
| 512x6144x6144 | bfloat16 | 48 | one-tile owner, n = 3 | 702.6 | 406.2 | flydsl | 804.4 | 728.5 | 192x128x128 |
| 512x9216x6144 | bfloat16 | 72 | one-tile owner, n = 3-4 | 882.2 | 600.0 | flydsl | 866.3 | 580.1 | 256x256x64 |
| 1024x2048x6144 | float32 | 32 | one-tile reduce-scatter, n = 4 | 571.8 | 262.3 | opus | 529.4 | 697.7 | 128x128x128 |
| 1024x2304x6144 | bfloat16 | 36 | one-tile owner, n = 4 | 580.0 | 306.2 | torch | 665.4 | 661.3 | 256x176x64 |
| 1024x2560x6144 | bfloat16 | 40 | one-tile owner, n = 3 | 600.1 | 340.5 | torch | 724.3 | 722.3 | 160x128x128 |
| 1024x3072x6144 | float32 | 48 | one-tile owner, n = 3 | 654.1 | 389.8 | opus | 561.0 | 827.7 | 96x128x128 |
| 1024x6144x6144 | bfloat16 | 96 | one-tile owner, n = 2-3 | 1004.2 | 788.0 | torch | 980.0 | 983.8 | 192x256x64 |
| 1024x9216x6144 | bfloat16 | 144 | data-parallel | 1141.7 | 1146.9 | torch | 1130.5 | 1130.5 | 144x256x64 |
| 2048x2048x6144 | float32 | 64 | one-tile owner, n = 3 | 814.8 | 502.4 | opus | 873.2 | 916.1 | 128x128x128 |
| 2048x2304x6144 | bfloat16 | 72 | one-tile owner, n = 3-4 | 878.4 | 596.5 | torch | 959.4 | 958.9 | 128x160x128 |
| 2048x2560x6144 | bfloat16 | 80 | one-tile owner, n = 2 | 947.5 | 663.4 | torch | 1000.5 | 999.9 | 160x128x128 |
| 2048x3072x6144 | float32 | 96 | one-tile owner, n = 2-3 | 908.9 | 767.4 | opus | 785.4 | 892.7 | 192x256x64 |
| 2048x6144x6144 | bfloat16 | 192 | data-parallel | 1306.5 | 1309.0 | torch | 1237.3 | 1237.5 | 192x256x64 |
| 2048x9216x6144 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 | 1282.7 | 1002.6 | torch | 1145.7 | 1147.8 | 144x256x64 |
| 4096x2048x6144 | float32 | 128 | one-tile owner, n = 2 | 1102.9 | 1007.5 | opus | 1124.4 | 829.8 | 256x256x64 |
| 4096x2304x6144 | bfloat16 | 144 | data-parallel | 1146.0 | 1151.4 | flydsl | 752.9 | 1136.4 | 144x256x64 |
| 4096x2560x6144 | bfloat16 | 160 | data-parallel | 1212.2 | 1221.8 | flydsl | 699.0 | 1188.5 | 160x256x64 |
| 4096x3072x6144 | float32 | 192 | data-parallel | 1254.6 | 1257.8 | asm | 1197.5 | 1212.1 | 192x256x64 |
| 4096x6144x6144 | bfloat16 | 384 | one-tile owner, n = 2 | 1374.2 | 1315.4 | torch | 1250.5 | 1261.3 | 192x256x64 |
| 4096x9216x6144 | bfloat16 | 576 | one-tile owner, n = 3 | 1380.5 | 1246.5 | torch | 1262.3 | 1261.7 | 192x256x64 |
| 8192x2048x6144 | float32 | 256 | data-parallel | 1385.9 | 1387.6 | opus | 1325.3 | 1250.6 | 256x256x64 |
| 8192x2304x6144 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 | 1293.7 | 1007.0 | torch | 1152.3 | 1152.3 | 144x256x64 |
| 8192x2560x6144 | bfloat16 | 320 | one-tile owner, n = 3 | 1314.2 | 1116.1 | asm | 525.6 | 1198.6 | 160x256x64 |
| 8192x3072x6144 | float32 | 384 | one-tile owner, n = 2 | 1308.3 | 1261.6 | asm | 1206.7 | 1231.4 | 192x256x64 |
| 8192x6144x6144 | bfloat16 | 768 | data-parallel | 1478.3 | 1483.3 | flydsl | 1411.8 | 1437.9 | 256x256x64 |
| 8192x9216x6144 | bfloat16 | 1152 | one-tile owner, n = 2 | 1474.2 | 1430.9 | asm | 1383.9 | 1222.5 | 256x240x64 |
| 16384x2048x6144 | float32 | 512 | data-parallel | 1354.0 | 1361.0 | opus | 1303.4 | 1265.9 | 256x256x64 |
| 16384x2304x6144 | bfloat16 | 576 | one-tile owner, n = 3 | 1389.2 | 1252.1 | flydsl | 1022.0 | 1284.9 | 192x256x64 |
| 16384x2560x6144 | bfloat16 | 640 | one-tile owner, n = 2 | 1435.1 | 1372.8 | flydsl | 1324.4 | 1297.7 | 256x224x64 |
| 16384x3072x6144 | float32 | 768 | data-parallel | 1395.7 | 1386.5 | asm | 1377.1 | 1303.8 | 256x256x64 |
| 16384x6144x6144 | bfloat16 | 1536 | data-parallel | 1522.7 | 1515.3 | asm | 1455.3 | 1497.7 | 256x256x64 |
| 16384x9216x6144 | bfloat16 | 2304 | data-parallel | 1531.0 | 1526.0 | torch | 1476.0 | 1509.5 | 256x256x64 |
| 32768x2304x6144 | bfloat16 | 1152 | one-tile owner, n = 2 | 1458.3 | 1405.4 | flydsl | 653.9 | 1204.1 | 256x240x64 |
| 32768x2560x6144 | bfloat16 | 1280 | data-parallel | 1495.8 | 1495.8 | flydsl | 1439.6 | 1485.3 | 256x256x64 |
| 32768x6144x6144 | bfloat16 | 3072 | data-parallel | 1551.7 | 1540.9 | torch | 1491.1 | 1532.4 | 256x256x64 |
| 32768x9216x6144 | bfloat16 | 4608 | data-parallel | 1549.2 | 1545.2 | torch | 1498.6 | 1537.7 | 256x256x64 |
| 512x5120x6400 | bfloat16 | 40 | one-tile owner, n = 4 | 643.0 | 340.9 | flydsl | 711.7 | 741.5 | 160x128x128 |
| 1024x5120x6400 | bfloat16 | 80 | one-tile owner, n = 2 | 961.1 | 666.0 | flydsl | 667.3 | 1015.4 | 160x128x128 |
| 2048x5120x6400 | bfloat16 | 160 | data-parallel | 1219.3 | 1221.4 | flydsl | 746.8 | 1186.2 | 160x256x64 |
| 4096x5120x6400 | bfloat16 | 320 | one-tile owner, n = 3 | 1317.4 | 1113.3 | torch | 1197.5 | 1200.4 | 160x256x64 |
| 8192x5120x6400 | bfloat16 | 640 | one-tile owner, n = 2 | 1422.7 | 1379.0 | flydsl | 1324.0 | 1287.0 | 256x224x64 |
| 16384x5120x6400 | bfloat16 | 1280 | data-parallel | 1503.7 | 1497.0 | torch | 1452.2 | 1455.5 | 256x256x64 |
| 32768x5120x6400 | bfloat16 | 2560 | data-parallel | 1541.0 | 1525.1 | torch | 1477.4 | 1496.7 | 256x256x64 |
| 512x16384x6656 | bfloat16 | 128 | one-tile owner, n = 2 | 1152.2 | 1040.0 | flydsl | 917.9 | 900.1 | 256x256x64 |
| 1024x16384x6656 | bfloat16 | 256 | data-parallel | 1447.1 | 1462.8 | asm | 453.5 | 1422.4 | 256x256x64 |
| 2048x16384x6656 | bfloat16 | 512 | data-parallel | 1488.6 | 1488.3 | asm | 1399.3 | 1435.0 | 256x256x64 |
| 4096x16384x6656 | bfloat16 | 1024 | data-parallel | 1501.6 | 1502.6 | torch | 1451.7 | 1454.4 | 256x256x64 |
| 8192x16384x6656 | bfloat16 | 2048 | data-parallel | 1527.3 | 1521.3 | asm | 1468.5 | 1493.0 | 256x256x64 |
| 16384x16384x6656 | bfloat16 | 4096 | data-parallel | 1541.9 | 1535.0 | asm | 1475.7 | 1524.2 | 256x256x64 |
| 32768x16384x6656 | bfloat16 | 8192 | data-parallel | 1540.3 | 1540.0 | asm | 290.1 | 1500.9 | 256x256x64 |
| 512x1024x7168 | bfloat16 | 8 | one-tile reduce-scatter, n = 14 | 286.7 | 70.0 | torch | 426.4 | 430.1 | 128x128x64 |
| 512x1536x7168 | bfloat16 | 12 | one-tile reduce-scatter, n = 10 | 383.8 | 104.7 | torch | 452.0 | 455.5 | 192x64x128 |
| 512x2048x7168 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 | 464.0 | 139.2 | torch | 574.3 | 573.9 | 128x128x128 |
| 512x3584x7168 | bfloat16 | 28 | one-tile reduce-scatter, n = 4 | 576.2 | 242.7 | flydsl | 616.8 | 626.3 | 128x128x256 |
| 512x6144x7168 | bfloat16 | 48 | one-tile owner, n = 3 | 788.7 | 411.9 | flydsl | 819.6 | 768.1 | 192x128x128 |
| 512x8192x7168 | bfloat16 | 64 | one-tile owner, n = 3 | 927.8 | 542.6 | flydsl | 849.2 | 863.6 | 128x128x256 |
| 512x8448x7168 | bfloat16 | 66 | one-tile owner, n = 3-4 | 913.5 | 560.2 | flydsl | 722.8 | 468.3 | 256x256x64 |
| 512x20480x7168 | bfloat16 | 160 | data-parallel | 1098.5 | 1107.6 | flydsl | 1031.4 | 1143.4 | 160x256x64 |
| 768x1024x7168 | bfloat16 | 12 | one-tile reduce-scatter, n = 10 | 388.6 | 104.9 | torch | 442.9 | 442.7 | 256x96x64 |
| 768x2048x7168 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 | 544.5 | 208.0 | flydsl | 609.0 | 617.7 | 256x128x64 |
| 1024x512x7168 | bfloat16 | 8 | one-tile reduce-scatter, n = 14 | 285.5 | 70.1 | flydsl | 376.3 | 324.1 | 256x64x64 |
| 1024x1024x7168 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 | 461.5 | 139.2 | torch | 601.0 | 587.0 | 128x128x128 |
| 1024x1536x7168 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 | 544.1 | 208.2 | flydsl | 624.0 | 616.4 | 256x128x64 |
| 1024x2048x7168 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 634.7 | 275.7 | torch | 749.7 | 748.1 | 256x128x64 |
| 1024x3584x7168 | bfloat16 | 56 | one-tile owner, n = 3 | 861.4 | 476.5 | flydsl | 849.5 | 805.2 | 128x128x256 |
| 1024x6144x7168 | bfloat16 | 96 | one-tile owner, n = 2-3 | 913.2 | 808.1 | torch | 977.4 | 934.1 | 192x256x64 |
| 1024x8192x7168 | bfloat16 | 128 | one-tile owner, n = 2 | 1201.7 | 1055.2 | flydsl | 1037.2 | 929.1 | 256x256x64 |
| 1024x8448x7168 | bfloat16 | 132 | data-parallel | 1088.0 | 1090.8 | torch | 1111.8 | 1129.6 | 144x256x64 |
| 1024x20480x7168 | bfloat16 | 320 | one-tile owner, n = 3 | 1290.4 | 1066.6 | flydsl | 778.8 | 1180.7 | 160x256x64 |
| 1280x512x7168 | bfloat16 | 10 | one-tile reduce-scatter, n = 12 | 336.9 | 87.5 | torch | 405.5 | 405.5 | 256x80x64 |
| 1280x1024x7168 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 | 501.2 | 173.5 | torch | 592.0 | 590.6 | 256x80x128 |
| 1280x2048x7168 | bfloat16 | 40 | one-tile owner, n = 4 | 679.1 | 344.3 | torch | 794.9 | 797.0 | 128x160x128 |
| 1536x512x7168 | bfloat16 | 12 | one-tile reduce-scatter, n = 10 | 385.1 | 104.9 | torch | 444.4 | 443.7 | 256x96x64 |
| 1536x1024x7168 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 | 546.1 | 207.8 | torch | 714.7 | 713.8 | 128x96x128 |
| 1536x2048x7168 | bfloat16 | 48 | one-tile owner, n = 3 | 780.0 | 409.6 | torch | 903.0 | 902.9 | 128x192x64 |
| 1792x512x7168 | bfloat16 | 14 | one-tile reduce-scatter, n = 9 | 415.0 | 122.0 | torch | 484.7 | 485.4 | 256x112x64 |
| 1792x1024x7168 | bfloat16 | 28 | one-tile reduce-scatter, n = 4 | 583.9 | 242.4 | flydsl | 632.3 | 652.8 | 256x128x64 |
| 1792x2048x7168 | bfloat16 | 56 | one-tile owner, n = 3 | 858.7 | 475.4 | flydsl | 900.0 | 795.3 | 128x128x256 |
| 2048x512x7168 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 | 464.2 | 139.1 | torch | 571.9 | 572.2 | 128x128x64 |
| 2048x1024x7168 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 632.9 | 275.1 | torch | 750.7 | 752.0 | 128x128x128 |
| 2048x2048x7168 | bfloat16 | 64 | one-tile owner, n = 3 | 916.5 | 542.7 | flydsl | 947.8 | 862.4 | 128x128x256 |
| 2048x3584x7168 | bfloat16 | 112 | one-tile owner, n = 2-3 | 1064.9 | 934.4 | opus | 1161.4 | 969.7 | 224x256x64 |
| 2048x8192x7168 | bfloat16 | 256 | data-parallel | 1450.4 | 1448.8 | asm | 1391.7 | 1403.5 | 256x256x64 |
| 2048x20480x7168 | bfloat16 | 640 | one-tile owner, n = 2 | 1435.6 | 1375.3 | flydsl | 823.0 | 1250.8 | 224x256x64 |
| 4096x512x7168 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 638.3 | 275.8 | flydsl | 703.8 | 669.5 | 128x128x256 |
| 4096x1024x7168 | bfloat16 | 64 | one-tile owner, n = 3 | 925.7 | 542.7 | torch | 937.1 | 939.0 | 256x128x64 |
| 4096x2048x7168 | bfloat16 | 128 | one-tile owner, n = 2 | 1202.3 | 1047.4 | torch | 1218.6 | 1219.7 | 256x128x64 |
| 4096x3584x7168 | bfloat16 | 224 | data-parallel | 1397.2 | 1401.4 | flydsl | 1329.0 | 1274.1 | 224x256x64 |
| 4096x8192x7168 | bfloat16 | 512 | data-parallel | 1472.0 | 1477.8 | flydsl | 1404.0 | 1430.7 | 256x256x64 |
| 4096x20480x7168 | bfloat16 | 1280 | data-parallel | 1505.1 | 1512.1 | flydsl | 1438.4 | 1455.6 | 256x256x64 |
| 6144x512x7168 | bfloat16 | 48 | one-tile owner, n = 3 | 783.4 | 411.7 | flydsl | 799.3 | 710.8 | 128x192x128 |
| 6144x1024x7168 | bfloat16 | 96 | one-tile owner, n = 2-3 | 1035.6 | 803.8 | opus | 1135.4 | 854.5 | 256x192x64 |
| 6144x2048x7168 | bfloat16 | 192 | data-parallel | 1320.1 | 1323.1 | opus | 1319.9 | 1263.2 | 256x192x64 |
| 8192x512x7168 | bfloat16 | 64 | one-tile owner, n = 3 | 929.6 | 544.2 | flydsl | 935.8 | 871.7 | 128x128x256 |
| 8192x1024x7168 | bfloat16 | 128 | one-tile owner, n = 2 | 1210.9 | 1050.7 | torch | 1101.0 | 1107.5 | 256x128x64 |
| 8192x2048x7168 | bfloat16 | 256 | data-parallel | 1467.7 | 1469.7 | torch | 1401.5 | 1404.0 | 256x256x64 |
| 8192x3584x7168 | bfloat16 | 448 | data-parallel | 1420.9 | 1427.4 | flydsl | 1363.6 | 1296.7 | 224x256x64 |
| 8192x8192x7168 | bfloat16 | 1024 | data-parallel | 1506.9 | 1508.0 | flydsl | 1431.9 | 1463.7 | 256x256x64 |
| 8192x20480x7168 | bfloat16 | 2560 | data-parallel | 1537.7 | 1533.9 | asm | 1505.7 | 1474.1 | 256x256x64 |
| 10240x512x7168 | bfloat16 | 80 | one-tile owner, n = 3-4 | 1022.4 | 673.6 | torch | 817.5 | 821.6 | 256x160x64 |
| 10240x1024x7168 | bfloat16 | 160 | data-parallel | 1221.2 | 1224.3 | torch | 1224.2 | 1226.5 | 256x160x64 |
| 10240x2048x7168 | bfloat16 | 320 | one-tile owner, n = 3 | 1337.5 | 1123.3 | torch | 1234.8 | 1232.8 | 256x160x64 |
| 10240x8448x7168 | bfloat16 | 1320 | one-tile owner, n = 4 | 1466.6 | 1352.0 | asm | 1302.6 | 1330.4 | 256x224x64 |
| 10240x20480x7168 | bfloat16 | 3200 | one-tile owner, n = 2 | 1502.5 | 1469.0 | asm | 1440.0 | 1447.1 | 256x256x64 |
| 12288x512x7168 | bfloat16 | 96 | one-tile owner, n = 2-3 | 1001.1 | 803.2 | opus | 1140.7 | 824.3 | 256x192x64 |
| 12288x1024x7168 | bfloat16 | 192 | data-parallel | 1327.6 | 1330.3 | opus | 1327.0 | 1271.4 | 256x192x64 |
| 12288x2048x7168 | bfloat16 | 384 | one-tile owner, n = 2 | 1401.1 | 1330.1 | opus | 1340.0 | 1282.8 | 256x192x64 |
| 14336x512x7168 | bfloat16 | 112 | one-tile owner, n = 2-3 | 1125.0 | 928.7 | opus | 1190.5 | 911.6 | 256x224x64 |
| 14336x1024x7168 | bfloat16 | 224 | data-parallel | 1405.6 | 1408.1 | torch | 1311.2 | 1311.3 | 256x224x64 |
| 14336x2048x7168 | bfloat16 | 448 | data-parallel | 1418.7 | 1421.8 | flydsl | 1361.3 | 1333.7 | 256x224x64 |
| 16384x512x7168 | bfloat16 | 128 | one-tile owner, n = 2 | 1173.4 | 1050.7 | opus | 1235.3 | 1016.9 | 256x256x64 |
| 16384x1024x7168 | bfloat16 | 256 | data-parallel | 1466.3 | 1473.1 | torch | 1437.2 | 1436.2 | 256x256x64 |
| 16384x2048x7168 | bfloat16 | 512 | data-parallel | 1496.8 | 1495.8 | torch | 1454.9 | 1453.3 | 256x256x64 |
| 16384x3584x7168 | bfloat16 | 896 | one-tile owner, n = 2 | 1488.5 | 1415.9 | flydsl | 1368.9 | 1337.9 | 224x256x64 |
| 16384x8192x7168 | bfloat16 | 2048 | data-parallel | 1551.5 | 1544.8 | torch | 1484.0 | 1536.8 | 256x256x64 |
| 16384x20480x7168 | bfloat16 | 5120 | data-parallel | 1576.4 | 1559.8 | flydsl | 1511.3 | 1478.2 | 256x256x64 |
| 32768x512x7168 | bfloat16 | 256 | data-parallel | 1371.3 | 1378.0 | torch | 1363.2 | 1376.5 | 256x256x64 |
| 32768x1024x7168 | bfloat16 | 512 | data-parallel | 1481.3 | 1481.9 | torch | 1439.3 | 1444.8 | 256x256x64 |
| 32768x2048x7168 | bfloat16 | 1024 | data-parallel | 1523.4 | 1511.3 | torch | 1484.3 | 1490.8 | 256x256x64 |
| 32768x3584x7168 | bfloat16 | 1792 | data-parallel | 1516.9 | 1516.4 | torch | 1505.0 | 1513.4 | 256x256x64 |
| 32768x8192x7168 | bfloat16 | 4096 | data-parallel | 1554.5 | 1550.5 | torch | 1507.4 | 1546.6 | 256x256x64 |
| 32768x20480x7168 | bfloat16 | 10240 | data-parallel | 1546.1 | 1546.6 | asm | 1490.5 | 1440.2 | 256x256x64 |
| 512x1280x8192 | bfloat16 | 10 | one-tile reduce-scatter, n = 12 | 364.4 | 88.3 | torch | 463.7 | 464.7 | 160x64x128 |
| 512x2560x8192 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 | 546.6 | 174.6 | torch | 598.3 | 595.3 | 256x128x64 |
| 512x4096x8192 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 681.5 | 278.8 | flydsl | 738.8 | 726.0 | 128x128x256 |
| 512x4608x8192 | bfloat16 | 36 | one-tile owner, n = 4 | 667.7 | 311.9 | torch | 799.9 | 806.7 | 160x128x128 |
| 512x7168x8192 | bfloat16 | 56 | one-tile owner, n = 3 | 887.7 | 482.5 | flydsl | 892.4 | 822.7 | 256x176x64 |
| 512x8192x8192 | bfloat16 | 64 | one-tile owner, n = 3 | 978.1 | 548.1 | flydsl | 845.9 | 892.2 | 128x128x256 |
| 512x10240x8192 | bfloat16 | 80 | one-tile owner, n = 3-4 | 1001.8 | 683.1 | flydsl | 658.2 | 570.5 | 256x256x64 |
| 512x14336x8192 | bfloat16 | 112 | one-tile owner, n = 2-3 | 1137.9 | 945.2 | torch | 1005.8 | 1023.4 | 224x256x64 |
| 512x16384x8192 | bfloat16 | 128 | one-tile owner, n = 2 | 1187.0 | 1008.6 | flydsl | 686.2 | 921.8 | 256x256x64 |
| 512x57344x8192 | bfloat16 | 448 | data-parallel | 1336.6 | 1334.3 | flydsl | 1278.5 | 1259.0 | 224x256x64 |
| 768x4096x8192 | bfloat16 | 48 | one-tile owner, n = 4 | 827.0 | 415.6 | flydsl | 823.3 | 759.7 | 128x192x128 |
| 1024x512x8192 | bfloat16 | 8 | one-tile reduce-scatter, n = 16 | 332.3 | 70.7 | torch | 366.0 | 366.1 | 256x64x64 |
| 1024x1280x8192 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 | 542.6 | 175.1 | torch | 602.9 | 605.1 | 256x128x64 |
| 1024x2560x8192 | bfloat16 | 40 | one-tile owner, n = 4 | 727.5 | 347.4 | torch | 823.9 | 832.2 | 160x128x128 |
| 1024x4096x8192 | bfloat16 | 64 | one-tile owner, n = 3 | 975.1 | 548.3 | flydsl | 939.7 | 891.4 | 128x128x256 |
| 1024x4608x8192 | bfloat16 | 72 | one-tile owner, n = 3-4 | 1020.4 | 616.7 | torch | 941.8 | 942.7 | 160x256x64 |
| 1024x7168x8192 | bfloat16 | 112 | one-tile owner, n = 2-3 | 1185.0 | 947.3 | flydsl | 829.0 | 1033.2 | 224x256x64 |
| 1024x8192x8192 | bfloat16 | 128 | one-tile owner, n = 2 | 1243.3 | 1070.0 | flydsl | 1087.3 | 916.0 | 256x256x64 |
| 1024x10240x8192 | bfloat16 | 160 | data-parallel | 1233.0 | 1238.6 | flydsl | 839.6 | 1203.3 | 160x256x64 |
| 1024x14336x8192 | bfloat16 | 224 | data-parallel | 1395.2 | 1396.9 | flydsl | 1337.6 | 1268.0 | 224x256x64 |
| 1024x16384x8192 | bfloat16 | 256 | data-parallel | 1487.7 | 1482.8 | torch | 1426.7 | 1425.6 | 256x256x64 |
| 1024x57344x8192 | bfloat16 | 896 | one-tile owner, n = 2 | 1485.6 | 1405.4 | asm | 1360.1 | 1314.3 | 224x256x64 |
| 2048x512x8192 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 | 502.1 | 140.6 | torch | 530.1 | 529.6 | 256x128x64 |
| 2048x1280x8192 | bfloat16 | 40 | one-tile owner, n = 4 | 722.6 | 347.2 | torch | 713.5 | 714.7 | 320x192x64 |
| 2048x2560x8192 | bfloat16 | 80 | one-tile owner, n = 3-4 | 1080.8 | 682.1 | torch | 950.0 | 953.6 | 160x256x64 |
| 2048x4096x8192 | bfloat16 | 128 | one-tile owner, n = 2 | 1245.9 | 1070.7 | opus | 1214.9 | 995.8 | 256x256x64 |
| 2048x4608x8192 | bfloat16 | 144 | data-parallel | 1175.7 | 1178.7 | torch | 1160.6 | 1160.2 | 144x256x64 |
| 2048x7168x8192 | bfloat16 | 224 | data-parallel | 1415.3 | 1421.6 | asm | 1366.6 | 1290.4 | 224x256x64 |
| 2048x8192x8192 | bfloat16 | 256 | data-parallel | 1480.4 | 1484.2 | torch | 1436.6 | 1430.0 | 256x256x64 |
| 2048x10240x8192 | bfloat16 | 320 | one-tile owner, n = 3 | 1365.8 | 1133.8 | flydsl | 982.6 | 1221.2 | 160x256x64 |
| 2048x14336x8192 | bfloat16 | 448 | data-parallel | 1423.0 | 1428.6 | flydsl | 1369.7 | 1303.1 | 224x256x64 |
| 2048x16384x8192 | bfloat16 | 512 | data-parallel | 1527.7 | 1511.2 | asm | 1436.1 | 1443.1 | 256x256x64 |
| 2048x57344x8192 | bfloat16 | 1792 | data-parallel | 1560.6 | 1557.9 | flydsl | 1502.8 | 1468.9 | 256x256x64 |
| 4096x512x8192 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 670.3 | 279.1 | flydsl | 746.3 | 749.0 | 128x128x256 |
| 4096x1280x8192 | bfloat16 | 80 | one-tile owner, n = 3-4 | 1005.9 | 682.4 | torch | 953.9 | 967.7 | 160x256x64 |
| 4096x2560x8192 | bfloat16 | 160 | data-parallel | 1241.0 | 1244.7 | torch | 1208.9 | 1209.7 | 160x256x64 |
| 4096x4096x8192 | bfloat16 | 256 | data-parallel | 1487.8 | 1484.9 | torch | 1444.1 | 1442.1 | 256x256x64 |
| 4096x4608x8192 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 | 1347.3 | 1024.4 | torch | 1183.1 | 1180.1 | 144x256x64 |
| 4096x7168x8192 | bfloat16 | 448 | data-parallel | 1434.3 | 1436.8 | flydsl | 1370.7 | 1313.9 | 224x256x64 |
| 4096x8192x8192 | bfloat16 | 512 | data-parallel | 1497.7 | 1497.3 | asm | 1448.6 | 1455.9 | 256x256x64 |
| 4096x10240x8192 | bfloat16 | 640 | one-tile owner, n = 2 | 1475.1 | 1400.0 | flydsl | 1348.2 | 1295.6 | 224x256x64 |
| 4096x14336x8192 | bfloat16 | 896 | one-tile owner, n = 2 | 1501.2 | 1436.6 | flydsl | 1377.0 | 1335.9 | 224x256x64 |
| 4096x16384x8192 | bfloat16 | 1024 | data-parallel | 1538.1 | 1538.2 | flydsl | 1457.3 | 1496.0 | 256x256x64 |
| 4096x57344x8192 | bfloat16 | 3584 | data-parallel | 1580.4 | 1559.9 | flydsl | 1509.5 | 1480.2 | 256x256x64 |
| 8192x512x8192 | bfloat16 | 64 | one-tile owner, n = 3 | 969.5 | 548.9 | flydsl | 945.5 | 906.2 | 128x128x256 |
| 8192x1280x8192 | bfloat16 | 160 | data-parallel | 1248.7 | 1253.2 | torch | 1213.7 | 1216.8 | 160x256x64 |
| 8192x2560x8192 | bfloat16 | 320 | one-tile owner, n = 3 | 1371.7 | 1137.8 | torch | 1225.8 | 1222.4 | 160x256x64 |
| 8192x4096x8192 | bfloat16 | 512 | data-parallel | 1495.7 | 1500.1 | torch | 1453.2 | 1453.9 | 256x256x64 |
| 8192x4608x8192 | bfloat16 | 576 | one-tile owner, n = 3 | 1434.1 | 1274.6 | torch | 1296.5 | 1296.7 | 192x256x64 |
| 8192x7168x8192 | bfloat16 | 896 | one-tile owner, n = 2 | 1499.9 | 1437.9 | asm | 1390.1 | 1359.0 | 224x256x64 |
| 8192x8192x8192 | bfloat16 | 1024 | data-parallel | 1537.4 | 1545.8 | asm | 1474.1 | 1517.9 | 256x256x64 |
| 8192x10240x8192 | bfloat16 | 1280 | data-parallel | 1548.5 | 1551.8 | flydsl | 1466.9 | 1545.1 | 256x256x64 |
| 8192x14336x8192 | bfloat16 | 1792 | data-parallel | 1549.2 | 1559.9 | flydsl | 1476.2 | 1535.1 | 256x256x64 |
| 8192x16384x8192 | bfloat16 | 2048 | data-parallel | 1556.3 | 1556.4 | asm | 1516.3 | 1509.3 | 256x256x64 |
| 8192x57344x8192 | bfloat16 | 7168 | data-parallel | 1553.2 | 1563.6 | asm | 1512.6 | 1449.8 | 256x256x64 |
| 16384x512x8192 | bfloat16 | 128 | one-tile owner, n = 2 | 1189.0 | 971.2 | torch | 1044.7 | 1036.6 | 256x256x64 |
| 16384x1280x8192 | bfloat16 | 320 | one-tile owner, n = 3 | 1367.1 | 1130.6 | torch | 1244.2 | 1239.6 | 160x256x64 |
| 16384x2560x8192 | bfloat16 | 640 | one-tile owner, n = 2 | 1480.0 | 1375.9 | flydsl | 1336.1 | 1306.9 | 256x224x64 |
| 16384x4608x8192 | bfloat16 | 1152 | one-tile owner, n = 2 | 1521.5 | 1452.7 | flydsl | 1403.6 | 1218.4 | 256x240x64 |
| 16384x7168x8192 | bfloat16 | 1792 | data-parallel | 1552.9 | 1552.4 | torch | 1498.9 | 1548.1 | 256x256x64 |
| 16384x8192x8192 | bfloat16 | 2048 | data-parallel | 1571.0 | 1563.3 | torch | 1509.1 | 1553.2 | 256x256x64 |
| 16384x10240x8192 | bfloat16 | 2560 | data-parallel | 1556.4 | 1568.5 | torch | 1508.1 | 1555.1 | 256x256x64 |
| 16384x14336x8192 | bfloat16 | 3584 | data-parallel | 1573.9 | 1563.3 | asm | 1530.2 | 1516.6 | 256x256x64 |
| 16384x16384x8192 | bfloat16 | 4096 | data-parallel | 1551.5 | 1568.7 | asm | 1534.6 | 1510.0 | 256x256x64 |
| 16384x57344x8192 | bfloat16 | 14336 | data-parallel | 1546.1 | 1569.8 | asm | 1505.0 | 1432.8 | 256x256x64 |
| 32768x1280x8192 | bfloat16 | 640 | one-tile owner, n = 2 | 1444.3 | 1347.4 | flydsl | 1312.8 | 1318.1 | 256x224x64 |
| 32768x2560x8192 | bfloat16 | 1280 | data-parallel | 1530.0 | 1524.8 | torch | 1515.3 | 1522.8 | 256x256x64 |
| 32768x7168x8192 | bfloat16 | 3584 | data-parallel | 1569.3 | 1559.6 | torch | 1513.7 | 1551.9 | 256x256x64 |
| 32768x8192x8192 | bfloat16 | 4096 | data-parallel | 1580.6 | 1574.4 | flydsl | 1498.1 | 1557.4 | 256x256x64 |
| 32768x10240x8192 | bfloat16 | 5120 | data-parallel | 1579.6 | 1571.9 | torch | 1515.2 | 1556.0 | 256x256x64 |
| 32768x14336x8192 | bfloat16 | 7168 | data-parallel | 1569.8 | 1562.9 | asm | 1516.3 | 1501.3 | 256x256x64 |
| 32768x16384x8192 | bfloat16 | 8192 | data-parallel | 1555.4 | 1561.9 | asm | 1516.2 | 1481.1 | 256x256x64 |
| 32768x57344x8192 | bfloat16 | 28672 | data-parallel | 1540.9 | 1565.8 | asm | 1474.1 | 1420.4 | 256x256x64 |
| 512x6144x12288 | bfloat16 | 48 | one-tile owner, n = 4 | 971.1 | 421.0 | torch | 874.6 | 894.4 | 192x128x128 |
| 1024x6144x12288 | bfloat16 | 96 | one-tile owner, n = 2-3 | 1159.1 | 829.7 | torch | 1097.2 | 1099.7 | 192x256x64 |
| 2048x6144x12288 | bfloat16 | 192 | data-parallel | 1363.7 | 1365.1 | torch | 1288.3 | 1287.5 | 192x256x64 |
| 4096x6144x12288 | bfloat16 | 384 | one-tile owner, n = 2 | 1462.8 | 1342.8 | torch | 1300.0 | 1297.6 | 192x256x64 |
| 8192x6144x12288 | bfloat16 | 768 | data-parallel | 1551.3 | 1542.6 | torch | 1483.4 | 1503.6 | 256x256x64 |
| 16384x6144x12288 | bfloat16 | 1536 | data-parallel | 1560.2 | 1548.7 | torch | 1512.3 | 1537.8 | 256x256x64 |
| 32768x6144x12288 | bfloat16 | 3072 | data-parallel | 1562.9 | 1561.3 | asm | 1537.9 | 1539.3 | 256x256x64 |
| 512x16384x13312 | bfloat16 | 128 | one-tile owner, n = 2 | 1278.4 | 924.7 | flydsl | 713.8 | 1093.6 | 256x256x64 |
| 1024x16384x13312 | bfloat16 | 256 | data-parallel | 1511.3 | 1515.1 | asm | 1463.4 | 1468.5 | 256x256x64 |
| 2048x16384x13312 | bfloat16 | 512 | data-parallel | 1532.5 | 1540.4 | flydsl | 1008.4 | 1472.2 | 256x256x64 |
| 4096x16384x13312 | bfloat16 | 1024 | data-parallel | 1556.3 | 1558.0 | flydsl | 1478.0 | 1483.8 | 256x256x64 |
| 8192x16384x13312 | bfloat16 | 2048 | data-parallel | 1578.3 | 1571.5 | flydsl | 1515.6 | 1491.6 | 256x256x64 |
| 16384x16384x13312 | bfloat16 | 4096 | data-parallel | 1561.5 | 1554.8 | asm | 1516.9 | 1477.1 | 256x256x64 |
| 32768x16384x13312 | bfloat16 | 8192 | data-parallel | 1555.1 | 1559.3 | flydsl | 1487.7 | 1457.8 | 256x256x64 |
| 512x2304x16384 | bfloat16 | 18 | one-tile reduce-scatter, n = 7 | 674.7 | 162.0 | torch | 739.9 | 734.1 | 256x128x64 |
| 512x4608x16384 | bfloat16 | 36 | one-tile owner, n = 6 | 919.8 | 320.6 | flydsl | 581.2 | 917.7 | 160x128x128 |
| 512x9216x16384 | bfloat16 | 72 | one-tile owner, n = 3-4 | 1090.9 | 588.5 | flydsl | 642.0 | 753.3 | 256x256x64 |
| 512x13312x16384 | bfloat16 | 104 | one-tile owner, n = 2-3 | 1139.0 | 774.7 | flydsl | 546.4 | 1068.0 | 208x256x64 |
| 512x26624x16384 | bfloat16 | 208 | data-parallel | 1280.8 | 1283.6 | flydsl | 774.4 | 1179.1 | 208x256x64 |
| 512x53248x16384 | bfloat16 | 416 | data-parallel | 1294.7 | 1294.9 | flydsl | 636.5 | 1188.1 | 208x256x64 |
| 1024x2304x16384 | bfloat16 | 36 | one-tile owner, n = 6 | 911.8 | 320.8 | torch | 890.8 | 893.1 | 256x256x64 |
| 1024x4608x16384 | bfloat16 | 72 | one-tile owner, n = 3-4 | 1072.2 | 635.1 | flydsl | 788.1 | 1055.1 | 144x256x64 |
| 1024x9216x16384 | bfloat16 | 144 | data-parallel | 1125.0 | 1125.4 | flydsl | 690.1 | 1173.1 | 144x256x64 |
| 1024x13312x16384 | bfloat16 | 208 | data-parallel | 1342.8 | 1349.5 | asm | 1325.6 | 1236.7 | 208x256x64 |
| 1024x26624x16384 | bfloat16 | 416 | data-parallel | 1361.5 | 1364.8 | flydsl | 928.1 | 1244.1 | 208x256x64 |
| 1024x53248x16384 | bfloat16 | 832 | one-tile owner, n = 4 | 1513.0 | 1336.2 | triton | 996.7 | 1248.4 | 208x256x64 |
| 2048x2304x16384 | bfloat16 | 72 | one-tile owner, n = 3-4 | 1189.3 | 633.5 | torch | 940.7 | 936.1 | 256x256x64 |
| 2048x4608x16384 | bfloat16 | 144 | data-parallel | 1214.0 | 1215.9 | flydsl | 693.3 | 1199.0 | 144x256x64 |
| 2048x9216x16384 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 | 1409.3 | 1034.3 | flydsl | 910.4 | 1157.7 | 144x256x64 |
| 2048x13312x16384 | bfloat16 | 416 | data-parallel | 1398.0 | 1402.3 | flydsl | 1362.6 | 1233.8 | 208x256x64 |
| 2048x26624x16384 | bfloat16 | 832 | one-tile owner, n = 4 | 1541.6 | 1363.8 | flydsl | 909.7 | 1233.9 | 208x256x64 |
| 2048x53248x16384 | bfloat16 | 1664 | one-tile owner, n = 2 | 1540.1 | 1488.4 | flydsl | 779.7 | 1277.1 | 240x256x64 |
| 4096x2304x16384 | bfloat16 | 144 | data-parallel | 1214.8 | 1216.6 | torch | 1203.7 | 1202.1 | 144x256x64 |
| 4096x4608x16384 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 | 1410.2 | 1037.9 | torch | 1215.7 | 1218.3 | 144x256x64 |
| 4096x9216x16384 | bfloat16 | 576 | one-tile owner, n = 4 | 1519.5 | 1294.9 | torch | 1322.5 | 1314.1 | 192x256x64 |
| 4096x13312x16384 | bfloat16 | 832 | one-tile owner, n = 4 | 1536.4 | 1364.0 | flydsl | 1309.9 | 1252.6 | 208x256x64 |
| 4096x26624x16384 | bfloat16 | 1664 | one-tile owner, n = 2 | 1546.8 | 1508.7 | flydsl | 1455.5 | 1316.1 | 240x256x64 |
| 4096x53248x16384 | bfloat16 | 3328 | data-parallel | 1576.5 | 1561.2 | flydsl | 1511.2 | 1469.1 | 256x256x64 |
| 8192x2304x16384 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 | 1384.1 | 1006.4 | torch | 1197.5 | 1199.4 | 144x256x64 |
| 8192x4608x16384 | bfloat16 | 576 | one-tile owner, n = 4 | 1493.7 | 1270.3 | torch | 1333.5 | 1333.8 | 192x256x64 |
| 8192x9216x16384 | bfloat16 | 1152 | one-tile owner, n = 2 | 1519.8 | 1448.8 | flydsl | 1396.9 | 1147.5 | 256x240x64 |
| 8192x13312x16384 | bfloat16 | 1664 | one-tile owner, n = 2 | 1547.2 | 1478.4 | flydsl | 1436.6 | 1282.1 | 240x256x64 |
| 8192x26624x16384 | bfloat16 | 3328 | data-parallel | 1555.6 | 1559.1 | flydsl | 1501.0 | 1465.5 | 256x256x64 |
| 8192x53248x16384 | bfloat16 | 6656 | data-parallel | 1534.1 | 1553.1 | flydsl | 1479.9 | 1437.0 | 256x256x64 |
| 16384x2304x16384 | bfloat16 | 576 | one-tile owner, n = 4 | 1484.9 | 1273.5 | torch | 1331.7 | 1323.3 | 192x256x64 |
| 16384x4608x16384 | bfloat16 | 1152 | one-tile owner, n = 2 | 1528.7 | 1444.3 | flydsl | 1404.4 | 1167.5 | 256x240x64 |
| 16384x9216x16384 | bfloat16 | 2304 | data-parallel | 1551.1 | 1553.2 | flydsl | 1503.0 | 1482.6 | 256x256x64 |
| 16384x13312x16384 | bfloat16 | 3328 | data-parallel | 1533.5 | 1546.2 | asm | 1485.2 | 1464.4 | 256x256x64 |
| 16384x26624x16384 | bfloat16 | 6656 | data-parallel | 1520.1 | 1545.2 | flydsl | 1492.2 | 1437.2 | 256x256x64 |
| 16384x53248x16384 | bfloat16 | 13312 | data-parallel | 1514.3 | 1541.5 | asm | 1456.0 | 1444.3 | 256x256x64 |
| 32768x2304x16384 | bfloat16 | 1152 | one-tile owner, n = 2 | 1491.2 | 1432.0 | flydsl | 1397.1 | 1176.5 | 256x240x64 |
| 32768x4608x16384 | bfloat16 | 2304 | data-parallel | 1530.0 | 1547.4 | torch | 1529.5 | 1528.4 | 256x256x64 |
| 32768x9216x16384 | bfloat16 | 4608 | data-parallel | 1523.3 | 1544.7 | asm | 1447.4 | 1459.1 | 256x256x64 |
| 32768x13312x16384 | bfloat16 | 6656 | data-parallel | 1511.6 | 1547.8 | asm | 1453.7 | 1422.8 | 256x256x64 |
| 32768x26624x16384 | bfloat16 | 13312 | data-parallel | 1500.8 | 1537.4 | flydsl | 1458.8 | 1425.6 | 256x256x64 |
| 32768x53248x16384 | bfloat16 | 26624 | data-parallel | 1495.6 | 1528.5 | flydsl | 1437.5 | 1450.4 | 256x256x64 |
| 512x6144x18432 | bfloat16 | 48 | one-tile owner, n = 5-6 | 1017.5 | 426.7 | flydsl | 905.1 | 924.4 | 192x128x128 |
| 1024x6144x18432 | bfloat16 | 96 | one-tile owner, n = 2-3 | 1148.0 | 791.0 | torch | 1144.7 | 1180.3 | 192x256x64 |
| 2048x6144x18432 | bfloat16 | 192 | data-parallel | 1330.3 | 1333.9 | torch | 1310.4 | 1314.7 | 192x256x64 |
| 4096x6144x18432 | bfloat16 | 384 | one-tile owner, n = 2 | 1508.5 | 1349.2 | torch | 1330.9 | 1316.2 | 192x256x64 |
| 8192x6144x18432 | bfloat16 | 768 | data-parallel | 1562.3 | 1557.4 | torch | 1525.7 | 1521.4 | 256x256x64 |
| 16384x6144x18432 | bfloat16 | 1536 | data-parallel | 1578.8 | 1577.5 | asm | 1533.4 | 1526.7 | 256x256x64 |
| 32768x6144x18432 | bfloat16 | 3072 | data-parallel | 1578.4 | 1582.0 | asm | 1518.8 | 1500.9 | 256x256x64 |
| 512x5120x25600 | bfloat16 | 40 | one-tile owner, n = 6-7 | 1022.2 | 325.3 | flydsl | 431.4 | 908.2 | 160x128x128 |
| 1024x5120x25600 | bfloat16 | 80 | one-tile owner, n = 3-4 | 1300.0 | 648.9 | torch | 1135.6 | 1140.3 | 160x256x64 |
| 2048x5120x25600 | bfloat16 | 160 | data-parallel | 1167.0 | 1160.5 | torch | 1256.8 | 1261.9 | 160x256x64 |
| 4096x5120x25600 | bfloat16 | 320 | one-tile owner, n = 4 | 1495.2 | 1129.6 | torch | 1247.6 | 1254.2 | 160x256x64 |
| 8192x5120x25600 | bfloat16 | 640 | one-tile owner, n = 2 | 1545.8 | 1390.0 | asm | 1378.1 | 1332.7 | 256x224x64 |
| 16384x5120x25600 | bfloat16 | 1280 | data-parallel | 1579.1 | 1559.6 | torch | 1530.2 | 1528.0 | 256x256x64 |
| 32768x5120x25600 | bfloat16 | 2560 | data-parallel | 1571.4 | 1579.3 | torch | 1512.2 | 1519.1 | 256x256x64 |
| 512x16384x26624 | bfloat16 | 128 | one-tile owner, n = 2 | 1361.1 | 944.3 | flydsl | 648.2 | 1182.9 | 256x256x64 |
| 1024x16384x26624 | bfloat16 | 256 | data-parallel | 1541.9 | 1549.3 | flydsl | 783.5 | 1498.4 | 256x256x64 |
| 2048x16384x26624 | bfloat16 | 512 | data-parallel | 1556.1 | 1558.8 | flydsl | 820.6 | 1497.2 | 256x256x64 |
| 4096x16384x26624 | bfloat16 | 1024 | data-parallel | 1597.5 | 1575.3 | flydsl | 1542.0 | 1495.8 | 256x256x64 |
| 8192x16384x26624 | bfloat16 | 2048 | data-parallel | 1593.4 | 1566.9 | asm | 233.5 | 1491.9 | 256x256x64 |
| 16384x16384x26624 | bfloat16 | 4096 | data-parallel | 1583.2 | 1592.1 | asm | 1530.8 | 1468.4 | 256x256x64 |
| 32768x16384x26624 | bfloat16 | 8192 | data-parallel | 1562.0 | 1590.8 | asm | 1503.1 | 1451.3 | 256x256x64 |
| 512x8192x28672 | bfloat16 | 64 | one-tile owner, n = 4 | 1235.6 | 474.3 | flydsl | 790.9 | 923.7 | 240x256x64 |
| 1024x8192x28672 | bfloat16 | 128 | one-tile owner, n = 2 | 1440.4 | 943.1 | triton | 693.2 | 1254.5 | 256x256x64 |
| 2048x8192x28672 | bfloat16 | 256 | data-parallel | 1545.2 | 1538.8 | flydsl | 1033.5 | 1502.4 | 256x256x64 |
| 4096x8192x28672 | bfloat16 | 512 | data-parallel | 1562.6 | 1561.8 | flydsl | 1537.3 | 1522.7 | 256x256x64 |
| 8192x8192x28672 | bfloat16 | 1024 | data-parallel | 1571.9 | 1586.0 | asm | 1540.6 | 1525.6 | 256x256x64 |
| 16384x8192x28672 | bfloat16 | 2048 | data-parallel | 1589.7 | 1568.9 | asm | 1538.5 | 1508.6 | 256x256x64 |
| 32768x8192x28672 | bfloat16 | 4096 | data-parallel | 1585.4 | 1592.8 | asm | 1526.0 | 1477.5 | 256x256x64 |
| 512x7168x35840 | bfloat16 | 56 | one-tile owner, n = 4-5 | 1197.1 | 436.6 | flydsl | 816.7 | 966.8 | 256x176x64 |
| 1024x7168x35840 | bfloat16 | 112 | one-tile owner, n = 2-3 | 1266.6 | 859.3 | flydsl | 564.0 | 1269.0 | 224x256x64 |
| 2048x7168x35840 | bfloat16 | 224 | spread | 1493.7 | 1470.6 | asm | 1442.1 | 1370.6 | 224x256x64 |
| 4096x7168x35840 | bfloat16 | 448 | spread | 1535.5 | 1483.2 | flydsl | 1441.1 | 1369.3 | 224x256x64 |
| 8192x7168x35840 | bfloat16 | 896 | one-tile owner, n = 2 | 1580.9 | 1435.8 | asm | 1383.1 | 1372.1 | 224x256x64 |
| 16384x7168x35840 | bfloat16 | 1792 | data-parallel | 1575.4 | 1557.7 | torch | 1498.9 | 1513.4 | 256x256x64 |
| 32768x7168x35840 | bfloat16 | 3584 | data-parallel | 1567.4 | 1553.6 | flydsl | 1503.4 | 1471.1 | 256x256x64 |
