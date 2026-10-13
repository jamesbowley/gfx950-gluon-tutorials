# streamk_wip (v20, capped) vs AITER, v9 and hipBLASLt on the 630 shapes

Shapes: `supported_status_scoped.csv` (630 shapes, no bias; 598 bf16 and 32 fp32 outputs). Inputs from `torch.manual_seed(0)`, every output checked against an fp32 `F.linear` reference, `do_bench_cudagraph` with 2 interleaved rounds and the median taken.

- Stack: torch 2.10.0+rocm7.2.4.git3d3aa833, hipBLASLt 100202 (dabb6df2b9), Triton 3.8.0 @ /home/jbowley/repos/triton_gfx950-tutorial-v2.2/python/triton.
- GPU: HIP device 0, PCI bus 117, gfx950:sramecc+:xnack-.
- AITER: e2c7aa0aa [Triton/Gluon] gfx950 a16w16: skip the Gluon kernel on Triton without cd_regclass, on top of a1455f1c4 2026-10-02 [CK] [FlyDSL] add mxfp8 a8w8 blockscale for gfx950 (#4951).
- streamk_wip: tutorial `v20_streamk_reduce_scatter`, snapshot sha256 7c55c4017abfc6cc (newest WORKLOG entry: "9. A split rule from a systematic grid: not adopted (2026-10-04)"), `STREAMK_POLICY=capped` (the kernel's default), v9 tile order. The configuration capped ran on each shape is in the appendix and in `v20_label`.
- hipBLASLt: the call AITER's `torch` libtype makes, `F.linear(x, w)` for bf16 and `torch.mm(x, w.t(), out_dtype=torch.float32)` for fp32, timed the same way. Its tile is the MT of the first `Cijk_` kernel it launches.
- Cache state: back to back on the same buffers (inputs stay cached between launches).
- "Faster" and "slower" mean a gap of more than 1%; "within 1%" holds the rest.

**Against AITER's pick, by the configuration capped chose**

| | Shapes | v9: faster | v9: within 1% | v9: slower | v9: median | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All shapes | 630 | 286 | 42 | 302 | -0.3% | 327 | 37 | 266 | +1.6% |
| Data-parallel | 324 | 225 | 33 | 66 | +2.7% | 209 | 27 | 88 | +2.5% |
| One-tile reduce-scatter, K >= 2048 | 77 | 5 | 1 | 71 | -66.1% | 12 | 1 | 64 | -27.7% |
| One-tile owner, K >= 2048 | 219 | 54 | 8 | 157 | -21.0% | 104 | 9 | 106 | -0.4% |
| One-tile reduce-scatter, K < 2048 | 6 | 0 | 0 | 6 | -67.0% | 0 | 0 | 6 | -65.8% |
| One-tile owner, K < 2048 | 2 | 0 | 0 | 2 | -49.4% | 0 | 0 | 2 | -49.8% |
| Spread | 2 | 2 | 0 | 0 | +2.5% | 2 | 0 | 0 | +5.4% |

Each pair of columns compares that kernel with AITER's pick; positive medians mean it is faster.

**Against AITER's pick, by the tile AITER uses**

| | Shapes | v9: faster | v9: within 1% | v9: slower | v9: median | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All shapes | 630 | 286 | 42 | 302 | -0.3% | 327 | 37 | 266 | +1.6% |
| AITER uses a 256x256 tile | 248 | 208 | 32 | 8 | +2.8% | 202 | 23 | 23 | +3.2% |
| AITER uses another tile | 382 | 78 | 10 | 294 | -23.6% | 125 | 14 | 243 | -10.2% |

Each pair of columns compares that kernel with AITER's pick; positive medians mean it is faster.

**Against AITER's pick, by v9 grid size (number of 256x256 output tiles)**

| | Shapes | v9: faster | v9: within 1% | v9: slower | v9: median | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All shapes | 630 | 286 | 42 | 302 | -0.3% | 327 | 37 | 266 | +1.6% |
| Under 256 (less than one wave) | 321 | 45 | 8 | 268 | -34.0% | 62 | 15 | 244 | -15.3% |
| 256 to 1023 | 152 | 115 | 7 | 30 | +3.0% | 122 | 9 | 21 | +5.7% |
| 1024 or more | 157 | 126 | 27 | 4 | +2.6% | 143 | 13 | 1 | +3.7% |

Each pair of columns compares that kernel with AITER's pick; positive medians mean it is faster.

**Against hipBLASLt (torch), by v9 grid size**

| | Shapes | AITER's pick: faster | AITER's pick: within 1% | AITER's pick: slower | AITER's pick: median | v9: faster | v9: within 1% | v9: slower | v9: median | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All shapes | 630 | 219 | 236 | 175 | -0.2% | 263 | 53 | 314 | -0.9% | 333 | 39 | 258 | +1.6% |
| Under 256 (less than one wave) | 321 | 113 | 102 | 106 | -0.2% | 46 | 11 | 264 | -36.7% | 77 | 10 | 234 | -15.8% |
| 256 to 1023 | 152 | 43 | 73 | 36 | -0.2% | 102 | 10 | 40 | +3.4% | 119 | 11 | 22 | +5.8% |
| 1024 or more | 157 | 63 | 61 | 33 | -0.0% | 115 | 32 | 10 | +2.9% | 137 | 18 | 2 | +4.3% |

Each pair of columns compares that kernel with hipBLASLt; positive medians mean it is faster.

**hipBLASLt's tile choice**

| hipBLASLt tile (MxNxK) | Shapes | hipBLASLt median TFLOPS | streamk_wip vs hipBLASLt: faster | within 1% | slower | median |
|---|---:|---:|---:|---:|---:|---:|
| 256x256x64 | 225 | 1452 | 181 | 29 | 15 | +3.5% |
| 128x128x256 | 40 | 754 | 8 | 0 | 32 | -18.0% |
| 192x256x64 | 34 | 1279 | 13 | 3 | 18 | -2.1% |
| 224x256x64 | 32 | 1320 | 32 | 0 | 0 | +8.2% |
| 256x128x64 | 27 | 733 | 1 | 0 | 26 | -21.1% |
| 160x256x64 | 27 | 1252 | 13 | 2 | 12 | -0.8% |
| 256x224x64 | 20 | 1351 | 17 | 3 | 0 | +8.0% |
| 256x240x64 | 18 | 1163 | 18 | 0 | 0 | +20.6% |
| 144x256x64 | 15 | 1232 | 7 | 0 | 8 | -4.7% |
| 128x128x128 | 14 | 665 | 0 | 0 | 14 | -29.5% |
| 160x128x128 | 13 | 903 | 1 | 0 | 12 | -18.2% |
| 96x128x128 | 12 | 668 | 0 | 0 | 12 | -47.2% |
| 192x128x128 | 12 | 859 | 2 | 0 | 10 | -10.4% |
| 256x208x64 | 12 | 1236 | 12 | 0 | 0 | +7.1% |
| 256x192x64 | 11 | 1301 | 4 | 0 | 7 | -2.7% |
| 208x256x64 | 11 | 1237 | 11 | 0 | 0 | +10.6% |
| 256x160x64 | 10 | 1234 | 2 | 0 | 8 | -10.2% |
| 64x64x256 | 9 | 485 | 0 | 0 | 9 | -42.9% |
| 128x192x128 | 7 | 1007 | 1 | 1 | 5 | -16.8% |
| 128x160x128 | 7 | 989 | 0 | 0 | 7 | -21.0% |
| 240x256x64 | 7 | 1220 | 7 | 0 | 0 | +21.7% |
| 128x80x128 | 6 | 718 | 0 | 0 | 6 | -31.6% |
| 128x96x128 | 6 | 805 | 0 | 0 | 6 | -27.5% |
| 224x128x64 | 5 | 894 | 0 | 0 | 5 | -22.6% |
| 256x176x64 | 4 | 824 | 2 | 1 | 1 | +0.7% |
| 128x64x128 | 4 | 578 | 0 | 0 | 4 | -61.7% |
| 160x64x128 | 4 | 522 | 0 | 0 | 4 | -36.6% |
| 64x80x256 | 4 | 545 | 0 | 0 | 4 | -42.9% |
| 64x96x256 | 4 | 571 | 0 | 0 | 4 | -35.6% |
| 64x32x256 | 3 | 330 | 0 | 0 | 3 | -45.9% |
| 64x48x256 | 3 | 436 | 0 | 0 | 3 | -43.6% |
| 160x192x128 | 3 | 995 | 0 | 0 | 3 | -7.2% |
| 192x128x64 | 2 | 931 | 0 | 0 | 2 | -31.7% |
| 128x192x64 | 2 | 914 | 0 | 0 | 2 | -14.2% |
| 256x112x64 | 2 | 483 | 0 | 0 | 2 | -21.3% |
| 192x64x128 | 2 | 433 | 0 | 0 | 2 | -25.9% |
| 128x128x64 | 2 | 511 | 0 | 0 | 2 | -34.8% |
| 256x96x64 | 2 | 448 | 0 | 0 | 2 | -24.1% |
| 256x64x64 | 2 | 346 | 0 | 0 | 2 | -19.6% |
| 96x64x128 | 1 | 480 | 0 | 0 | 1 | -63.9% |
| 224x160x64 | 1 | 1154 | 1 | 0 | 0 | +9.4% |
| 192x192x64 | 1 | 1091 | 0 | 0 | 1 | -8.4% |
| 64x96x128 | 1 | 612 | 0 | 0 | 1 | -31.8% |
| 256x80x64 | 1 | 407 | 0 | 0 | 1 | -26.4% |
| 256x80x128 | 1 | 588 | 0 | 0 | 1 | -23.6% |
| 320x192x64 | 1 | 726 | 0 | 0 | 1 | -1.3% |

Solution-name tokens (stream-K `SK`, global split-U `GSU`, kernels launched): SK3 GSU0 x1: 604, SK3 GSU0 x2: 24, none GSU- x1: 2. `SK3` marks a kernel built with stream-K support, not that this launch split K.

**streamk_wip against v9, by configuration**

| | Shapes | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|
| All shapes | 630 | 377 | 111 | 142 | +1.9% |
| Data-parallel | 324 | 78 | 107 | 139 | -0.3% |
| One-tile reduce-scatter, K >= 2048 | 77 | 76 | 1 | 0 | +121.2% |
| One-tile owner, K >= 2048 | 219 | 217 | 2 | 0 | +19.1% |
| One-tile reduce-scatter, K < 2048 | 6 | 4 | 0 | 2 | +3.6% |
| One-tile owner, K < 2048 | 2 | 1 | 0 | 1 | -1.0% |
| Spread | 2 | 1 | 1 | 0 | +2.8% |

Each pair of columns compares that kernel with v9; positive medians mean it is faster.

**Diagnostic: capped's split against forcing data-parallel**

| Configuration | Shapes | split faster | within 1% | split slower | median |
|---|---:|---:|---:|---:|---:|
| One-tile owner, K < 2048 | 2 | 1 | 0 | 1 | +0.8% |
| One-tile owner, K >= 2048 | 219 | 211 | 8 | 0 | +19.0% |
| One-tile reduce-scatter, K < 2048 | 6 | 4 | 0 | 2 | +5.3% |
| One-tile reduce-scatter, K >= 2048 | 77 | 77 | 0 | 0 | +118.8% |
| Spread | 2 | 1 | 1 | 0 | +2.1% |

**Selected shapes (TFLOPS)**

| Shape (MxNxK) | Output tiles | AITER pick | AITER | hipBLASLt | hipBLASLt tile | v9 | streamk_wip | streamk_wip config | vs v9 | vs AITER | vs hipBLASLt |
|---|---:|---|---:|---:|---|---:|---:|---|---:|---:|---:|
| 512x2304x16384 | 18 | torch | 734 | 733 | 256x128x64 | 162 | 635 | one-tile reduce-scatter, n = 7 | +292% | -14% | -13% |
| 512x5120x25600 | 40 | flydsl | 439 | 910 | 160x128x128 | 328 | 994 | one-tile owner, n = 6-7 | +203% | +126% | +9% |
| 512x7168x35840 | 56 | flydsl | 820 | 967 | 256x176x64 | 440 | 1172 | one-tile owner, n = 4-5 | +166% | +43% | +21% |
| 8192x2304x16384 | 288 | torch | 1203 | 1206 | 144x256x64 | 1036 | 1399 | one-tile reduce-scatter, n = 4 | +35% | +16% | +16% |
| 4096x5120x25600 | 320 | torch | 1280 | 1282 | 160x256x64 | 1162 | 1548 | one-tile owner, n = 4 | +33% | +21% | +21% |
| 4096x4608x8192 | 288 | torch | 1271 | 1274 | 144x256x64 | 1088 | 1397 | one-tile reduce-scatter, n = 4 | +28% | +10% | +10% |
| 1024x3072x512 | 48 | flydsl | 459 | 404 | 96x128x128 | 226 | 212 | data-parallel | -6% | -54% | -47% |
| 2048x2048x512 | 64 | flydsl | 528 | 471 | 128x128x128 | 285 | 273 | data-parallel | -4% | -48% | -42% |
| 4096x5120x640 | 320 | flydsl | 811 | 778 | 256x176x64 | 753 | 779 | data-parallel | +3% | -4% | +0% |
| 8192x2048x512 | 256 | torch | 805 | 807 | 256x256x64 | 867 | 788 | data-parallel | -9% | -2% | -2% |

**Against `results/details_main.csv`** (this run / that run - 1, median; for v20 the baseline may use a different policy or kernel snapshot)

| | Shapes | AITER's pick | v9 | v20 |
|---|---:|---:|---:|---:|
| All shapes | 630 | -0.1% | -0.1% | -0.1% |
| Under 256 (less than one wave) | 321 | +0.0% | -0.1% | +0.4% |
| 256 to 1023 | 152 | -0.1% | -0.0% | -0.2% |
| 1024 or more | 157 | -0.1% | -0.0% | -0.2% |
| K < 2048 | 91 | -0.1% | -0.1% | +6.3% |
| K >= 2048 | 539 | -0.0% | -0.0% | -0.2% |

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
| 1024x3072x512 | bfloat16 | 48 | data-parallel | 212.3 | 226.1 | flydsl | 458.8 | 403.8 | 96x128x128 |
| 1024x4096x512 | bfloat16 | 64 | data-parallel | 273.1 | 285.7 | flydsl | 532.2 | 471.1 | 128x128x128 |
| 1024x7168x512 | bfloat16 | 112 | data-parallel | 437.1 | 458.9 | flydsl | 614.3 | 581.7 | 256x128x64 |
| 2048x2048x512 | bfloat16 | 64 | data-parallel | 272.6 | 285.3 | flydsl | 528.4 | 470.9 | 128x128x128 |
| 2048x4096x512 | bfloat16 | 128 | data-parallel | 482.6 | 497.7 | torch | 621.0 | 621.5 | 256x128x64 |
| 4096x2048x512 | bfloat16 | 128 | data-parallel | 480.3 | 495.8 | torch | 670.3 | 669.5 | 256x128x64 |
| 4096x4096x512 | bfloat16 | 256 | data-parallel | 787.0 | 863.7 | torch | 811.2 | 811.0 | 256x256x64 |
| 4096x7168x512 | bfloat16 | 448 | data-parallel | 851.6 | 880.8 | flydsl | 857.0 | 764.4 | 224x256x64 |
| 8192x2048x512 | bfloat16 | 256 | data-parallel | 787.8 | 866.9 | torch | 804.8 | 807.0 | 256x256x64 |
| 8192x4096x512 | bfloat16 | 512 | data-parallel | 906.7 | 924.0 | flydsl | 858.6 | 864.9 | 256x256x64 |
| 16384x2048x512 | bfloat16 | 512 | data-parallel | 913.5 | 928.8 | torch | 862.4 | 866.1 | 256x256x64 |
| 16384x4096x512 | bfloat16 | 1024 | data-parallel | 1000.9 | 977.7 | torch | 946.4 | 952.6 | 256x256x64 |
| 16384x7168x512 | bfloat16 | 1792 | data-parallel | 1042.3 | 1013.5 | torch | 992.7 | 996.1 | 256x256x64 |
| 32768x2048x512 | bfloat16 | 1024 | data-parallel | 989.6 | 978.0 | torch | 941.1 | 951.7 | 256x256x64 |
| 1024x5120x640 | bfloat16 | 80 | data-parallel | 363.8 | 378.2 | torch | 531.8 | 532.2 | 160x128x128 |
| 2048x5120x640 | bfloat16 | 160 | data-parallel | 653.1 | 680.4 | flydsl | 683.6 | 565.9 | 192x256x64 |
| 4096x5120x640 | bfloat16 | 320 | data-parallel | 778.7 | 753.4 | flydsl | 810.6 | 778.4 | 256x176x64 |
| 8192x5120x640 | bfloat16 | 640 | data-parallel | 946.3 | 938.6 | torch | 895.0 | 899.2 | 256x224x64 |
| 16384x5120x640 | bfloat16 | 1280 | data-parallel | 1096.9 | 1079.5 | torch | 1055.6 | 1059.8 | 256x256x64 |
| 32768x5120x640 | bfloat16 | 2560 | data-parallel | 1121.8 | 1081.9 | torch | 1095.3 | 1101.1 | 256x256x64 |
| 1024x7168x768 | bfloat16 | 112 | data-parallel | 525.6 | 544.3 | flydsl | 694.4 | 679.0 | 224x128x64 |
| 10240x7168x768 | bfloat16 | 1120 | data-parallel | 1089.2 | 1066.5 | opus | 400.4 | 863.3 | 256x240x64 |
| 512x4096x1024 | bfloat16 | 32 | data-parallel | 196.2 | 200.3 | torch | 485.3 | 493.0 | 128x64x128 |
| 512x6144x1024 | bfloat16 | 48 | data-parallel | 287.5 | 293.7 | flydsl | 591.0 | 574.5 | 96x128x128 |
| 512x7168x1024 | bfloat16 | 56 | data-parallel | 325.7 | 335.6 | flydsl | 637.6 | 606.6 | 128x128x128 |
| 512x8192x1024 | bfloat16 | 64 | data-parallel | 367.6 | 374.8 | flydsl | 696.1 | 667.1 | 128x128x128 |
| 1024x4096x1024 | bfloat16 | 64 | data-parallel | 366.6 | 374.2 | torch | 663.0 | 665.4 | 128x128x128 |
| 1024x7168x1024 | bfloat16 | 112 | data-parallel | 598.7 | 612.8 | flydsl | 772.3 | 770.9 | 224x128x64 |
| 1024x8192x1024 | bfloat16 | 128 | data-parallel | 663.0 | 672.9 | torch | 840.0 | 847.3 | 256x128x64 |
| 2048x4096x1024 | bfloat16 | 128 | data-parallel | 663.1 | 672.4 | torch | 842.6 | 850.1 | 256x128x64 |
| 2048x7168x1024 | bfloat16 | 224 | data-parallel | 990.5 | 1076.4 | flydsl | 1036.9 | 1013.2 | 256x256x64 |
| 2048x8192x1024 | bfloat16 | 256 | data-parallel | 1055.3 | 1137.7 | flydsl | 1092.9 | 1058.5 | 256x256x64 |
| 3072x4096x1024 | bfloat16 | 192 | data-parallel | 898.6 | 958.5 | torch | 972.8 | 976.3 | 256x192x64 |
| 4096x4096x1024 | bfloat16 | 256 | data-parallel | 1069.9 | 1152.6 | torch | 1126.6 | 1132.6 | 256x256x64 |
| 4096x7168x1024 | bfloat16 | 448 | data-parallel | 1093.2 | 1139.4 | flydsl | 1108.7 | 1032.7 | 224x256x64 |
| 4096x8192x1024 | bfloat16 | 512 | data-parallel | 1161.3 | 1189.9 | asm | 1197.0 | 1150.5 | 256x256x64 |
| 5120x4096x1024 | bfloat16 | 320 | data-parallel | 896.9 | 900.6 | torch | 1022.2 | 1024.3 | 256x160x64 |
| 7680x4096x1024 | bfloat16 | 480 | data-parallel | 1137.3 | 1172.0 | torch | 984.9 | 986.9 | 256x240x64 |
| 8192x7168x1024 | bfloat16 | 896 | data-parallel | 1166.5 | 1166.3 | asm | 1100.6 | 1063.4 | 224x256x64 |
| 8192x8192x1024 | bfloat16 | 1024 | data-parallel | 1244.7 | 1230.5 | asm | 1208.6 | 1206.3 | 256x256x64 |
| 16384x4096x1024 | bfloat16 | 1024 | data-parallel | 1259.4 | 1245.6 | torch | 1207.0 | 1210.7 | 256x256x64 |
| 16384x7168x1024 | bfloat16 | 1792 | data-parallel | 1283.3 | 1263.9 | asm | 1201.1 | 1242.7 | 256x256x64 |
| 16384x8192x1024 | bfloat16 | 2048 | data-parallel | 1270.4 | 1242.5 | torch | 1239.5 | 1245.3 | 256x256x64 |
| 32768x7168x1024 | bfloat16 | 3584 | data-parallel | 1267.3 | 1242.7 | torch | 1257.9 | 1263.0 | 256x256x64 |
| 32768x8192x1024 | bfloat16 | 4096 | data-parallel | 1269.0 | 1227.8 | torch | 1257.3 | 1264.9 | 256x256x64 |
| 512x5120x1280 | bfloat16 | 40 | data-parallel | 255.4 | 263.5 | flydsl | 529.2 | 570.1 | 160x64x128 |
| 1024x5120x1280 | bfloat16 | 80 | data-parallel | 473.3 | 488.0 | torch | 721.4 | 722.0 | 160x128x128 |
| 2048x5120x1280 | bfloat16 | 160 | data-parallel | 849.1 | 884.2 | torch | 942.5 | 944.4 | 160x256x64 |
| 4096x5120x1280 | bfloat16 | 320 | data-parallel | 945.9 | 948.0 | torch | 994.0 | 997.6 | 160x256x64 |
| 8192x5120x1280 | bfloat16 | 640 | data-parallel | 1172.6 | 1177.8 | flydsl | 1144.7 | 1129.1 | 256x224x64 |
| 16384x5120x1280 | bfloat16 | 1280 | data-parallel | 1338.2 | 1312.2 | flydsl | 1264.4 | 1294.0 | 256x256x64 |
| 32768x5120x1280 | bfloat16 | 2560 | data-parallel | 1323.1 | 1308.4 | torch | 1308.8 | 1314.2 | 256x256x64 |
| 512x1536x1536 | bfloat16 | 12 | one-tile reduce-scatter, n = 8 | 103.5 | 85.8 | flydsl | 345.9 | 306.2 | 64x64x256 |
| 512x2048x1536 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 | 129.1 | 113.5 | torch | 402.3 | 403.9 | 64x64x256 |
| 512x2304x1536 | bfloat16 | 18 | one-tile reduce-scatter, n = 7 | 133.0 | 126.9 | flydsl | 391.9 | 383.9 | 64x80x256 |
| 512x4096x1536 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 213.4 | 224.8 | torch | 571.1 | 579.0 | 128x64x128 |
| 512x6144x1536 | bfloat16 | 48 | data-parallel | 321.0 | 325.5 | flydsl | 644.4 | 667.8 | 96x128x128 |
| 512x7168x1536 | bfloat16 | 56 | data-parallel | 366.2 | 374.3 | flydsl | 676.3 | 591.4 | 128x128x256 |
| 1024x1536x1536 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 | 173.2 | 169.0 | flydsl | 503.6 | 479.7 | 96x64x128 |
| 1024x2048x1536 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 212.5 | 221.8 | torch | 570.4 | 577.2 | 128x64x128 |
| 1024x2304x1536 | bfloat16 | 36 | one-tile owner, n = 2 | 238.9 | 249.6 | torch | 531.1 | 535.4 | 128x80x128 |
| 1024x3072x1536 | bfloat16 | 48 | data-parallel | 319.9 | 328.1 | flydsl | 635.1 | 667.0 | 96x128x128 |
| 1024x4096x1536 | bfloat16 | 64 | data-parallel | 414.2 | 420.3 | torch | 636.6 | 638.3 | 128x128x256 |
| 1024x7168x1536 | bfloat16 | 112 | data-parallel | 675.6 | 687.6 | torch | 887.4 | 893.8 | 224x128x64 |
| 2048x1536x1536 | bfloat16 | 48 | data-parallel | 322.9 | 327.9 | torch | 655.8 | 668.5 | 96x128x128 |
| 2048x2048x1536 | bfloat16 | 64 | data-parallel | 413.2 | 420.2 | torch | 636.6 | 640.0 | 128x128x256 |
| 2048x4096x1536 | bfloat16 | 128 | data-parallel | 753.0 | 762.5 | torch | 955.8 | 958.5 | 256x128x64 |
| 3072x2048x1536 | bfloat16 | 96 | data-parallel | 587.5 | 601.7 | opus | 873.2 | 818.4 | 128x192x128 |
| 3072x4096x1536 | bfloat16 | 192 | data-parallel | 997.3 | 1086.2 | opus | 1137.7 | 1100.0 | 256x192x64 |
| 4096x1536x1536 | bfloat16 | 96 | data-parallel | 587.8 | 603.1 | torch | 820.3 | 824.8 | 192x128x128 |
| 4096x2048x1536 | bfloat16 | 128 | data-parallel | 739.1 | 762.1 | opus | 1018.8 | 958.8 | 256x128x64 |
| 4096x3072x1536 | bfloat16 | 192 | data-parallel | 1009.2 | 1078.8 | flydsl | 1038.6 | 1099.7 | 192x256x64 |
| 4096x4096x1536 | bfloat16 | 256 | data-parallel | 1183.1 | 1263.6 | torch | 1228.6 | 1226.9 | 256x256x64 |
| 5120x2048x1536 | bfloat16 | 160 | data-parallel | 891.5 | 927.4 | torch | 1036.1 | 1036.7 | 256x160x64 |
| 5120x4096x1536 | bfloat16 | 320 | data-parallel | 980.2 | 987.6 | torch | 1099.4 | 1101.8 | 256x160x64 |
| 7680x2048x1536 | bfloat16 | 240 | data-parallel | 1134.9 | 1206.1 | torch | 1038.8 | 1043.4 | 256x240x64 |
| 7680x4096x1536 | bfloat16 | 480 | data-parallel | 1233.9 | 1257.4 | torch | 1087.4 | 1088.3 | 256x240x64 |
| 8192x1536x1536 | bfloat16 | 192 | data-parallel | 999.0 | 1073.9 | torch | 1098.2 | 1098.7 | 192x256x64 |
| 10240x7168x1536 | bfloat16 | 1120 | data-parallel | 1280.5 | 1258.2 | asm | 1215.4 | 1175.7 | 224x256x64 |
| 16384x1536x1536 | bfloat16 | 384 | data-parallel | 1123.0 | 1137.0 | torch | 1150.4 | 1159.5 | 192x256x64 |
| 16384x2048x1536 | bfloat16 | 512 | data-parallel | 1290.7 | 1307.2 | torch | 1286.9 | 1292.7 | 256x256x64 |
| 16384x3072x1536 | bfloat16 | 768 | data-parallel | 1337.9 | 1336.0 | torch | 1309.6 | 1312.6 | 256x256x64 |
| 16384x4096x1536 | bfloat16 | 1024 | data-parallel | 1350.3 | 1346.9 | torch | 1328.2 | 1331.9 | 256x256x64 |
| 32768x1536x1536 | bfloat16 | 768 | data-parallel | 1337.0 | 1333.1 | flydsl | 1281.2 | 1306.6 | 256x256x64 |
| 512x7168x1792 | bfloat16 | 56 | one-tile owner, n = 2 | 399.2 | 390.5 | flydsl | 720.9 | 603.6 | 128x128x256 |
| 1024x7168x1792 | bfloat16 | 112 | data-parallel | 712.7 | 720.5 | flydsl | 852.1 | 935.6 | 224x128x64 |
| 2048x7168x1792 | bfloat16 | 224 | data-parallel | 1165.1 | 1251.0 | flydsl | 1192.8 | 1149.0 | 224x256x64 |
| 4096x7168x1792 | bfloat16 | 448 | data-parallel | 1251.8 | 1288.6 | flydsl | 1265.9 | 1194.5 | 224x256x64 |
| 8192x7168x1792 | bfloat16 | 896 | data-parallel | 1313.3 | 1311.5 | asm | 1264.6 | 1214.7 | 224x256x64 |
| 16384x7168x1792 | bfloat16 | 1792 | data-parallel | 1403.2 | 1388.4 | torch | 1379.1 | 1384.2 | 256x256x64 |
| 32768x7168x1792 | bfloat16 | 3584 | data-parallel | 1398.4 | 1376.2 | torch | 1384.6 | 1393.3 | 256x256x64 |
| 512x4096x2048 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 280.2 | 237.9 | torch | 607.2 | 617.6 | 128x64x128 |
| 512x6144x2048 | float32 | 48 | one-tile owner, n = 2 | 341.5 | 315.2 | asm | 419.5 | 657.5 | 96x128x128 |
| 512x6144x2048 | bfloat16 | 48 | one-tile owner, n = 2 | 378.9 | 353.3 | flydsl | 686.0 | 714.3 | 96x128x128 |
| 512x8192x2048 | bfloat16 | 64 | one-tile owner, n = 2 | 486.3 | 452.5 | flydsl | 790.1 | 681.9 | 128x128x256 |
| 512x16384x2048 | bfloat16 | 128 | data-parallel | 811.8 | 825.6 | flydsl | 818.4 | 1028.8 | 256x128x64 |
| 1024x4096x2048 | bfloat16 | 64 | one-tile owner, n = 2 | 485.4 | 451.6 | torch | 683.6 | 684.0 | 128x128x256 |
| 1024x6144x2048 | float32 | 96 | data-parallel | 582.1 | 602.0 | opus | 670.5 | 900.7 | 192x128x64 |
| 1024x6144x2048 | bfloat16 | 96 | data-parallel | 647.0 | 651.9 | flydsl | 810.4 | 893.2 | 192x128x128 |
| 1024x8192x2048 | bfloat16 | 128 | data-parallel | 815.8 | 825.7 | flydsl | 1063.1 | 1036.7 | 256x128x64 |
| 1024x16384x2048 | bfloat16 | 256 | data-parallel | 1286.8 | 1371.1 | flydsl | 933.9 | 1325.5 | 256x256x64 |
| 2048x4096x2048 | bfloat16 | 128 | data-parallel | 821.2 | 826.3 | torch | 1036.8 | 1037.5 | 256x128x64 |
| 2048x6144x2048 | float32 | 192 | data-parallel | 978.9 | 1075.8 | asm | 1030.9 | 1080.5 | 192x256x64 |
| 2048x6144x2048 | bfloat16 | 192 | data-parallel | 1119.4 | 1162.6 | flydsl | 1132.2 | 1179.7 | 192x256x64 |
| 2048x8192x2048 | bfloat16 | 256 | data-parallel | 1276.5 | 1362.5 | flydsl | 1308.0 | 1303.2 | 256x256x64 |
| 2048x16384x2048 | bfloat16 | 512 | data-parallel | 1383.3 | 1405.4 | flydsl | 1102.5 | 1361.1 | 256x256x64 |
| 3072x4096x2048 | bfloat16 | 192 | data-parallel | 1101.5 | 1175.0 | torch | 1173.8 | 1174.7 | 256x192x64 |
| 4096x4096x2048 | bfloat16 | 256 | data-parallel | 1276.3 | 1370.0 | torch | 1322.1 | 1323.7 | 256x256x64 |
| 4096x6144x2048 | float32 | 384 | data-parallel | 1044.3 | 1065.6 | asm | 1022.3 | 1125.6 | 192x256x64 |
| 4096x6144x2048 | bfloat16 | 384 | data-parallel | 1183.3 | 1208.7 | flydsl | 1054.8 | 1211.6 | 192x256x64 |
| 4096x8192x2048 | bfloat16 | 512 | data-parallel | 1383.3 | 1391.5 | flydsl | 1348.4 | 1354.6 | 256x256x64 |
| 4096x16384x2048 | bfloat16 | 1024 | data-parallel | 1435.9 | 1427.9 | flydsl | 971.7 | 1393.9 | 256x256x64 |
| 5120x4096x2048 | bfloat16 | 320 | one-tile owner, n = 2 | 1056.6 | 1048.7 | torch | 1218.5 | 1221.8 | 256x160x64 |
| 7680x4096x2048 | bfloat16 | 480 | data-parallel | 1335.7 | 1358.3 | torch | 1147.0 | 1148.1 | 256x240x64 |
| 8192x6144x2048 | float32 | 768 | data-parallel | 1253.3 | 1262.9 | asm | 1228.4 | 1165.1 | 256x256x64 |
| 8192x6144x2048 | bfloat16 | 768 | data-parallel | 1421.0 | 1411.7 | flydsl | 856.5 | 1378.0 | 256x256x64 |
| 8192x8192x2048 | bfloat16 | 1024 | data-parallel | 1428.2 | 1424.9 | asm | 1399.9 | 1387.2 | 256x256x64 |
| 8192x16384x2048 | bfloat16 | 2048 | data-parallel | 1396.4 | 1364.1 | triton | 1012.9 | 1410.5 | 256x256x64 |
| 16384x4096x2048 | bfloat16 | 1024 | data-parallel | 1439.7 | 1419.9 | torch | 1362.4 | 1367.6 | 256x256x64 |
| 16384x6144x2048 | float32 | 1536 | data-parallel | 1233.7 | 1226.5 | asm | 1182.4 | 1146.0 | 256x256x64 |
| 16384x6144x2048 | bfloat16 | 1536 | data-parallel | 1443.0 | 1421.7 | torch | 1396.5 | 1403.4 | 256x256x64 |
| 16384x8192x2048 | bfloat16 | 2048 | data-parallel | 1431.5 | 1417.0 | torch | 1402.8 | 1409.4 | 256x256x64 |
| 16384x16384x2048 | bfloat16 | 4096 | data-parallel | 1401.6 | 1371.2 | flydsl | 973.1 | 1417.4 | 256x256x64 |
| 32768x6144x2048 | bfloat16 | 3072 | data-parallel | 1428.1 | 1403.6 | flydsl | 1004.2 | 1414.4 | 256x256x64 |
| 32768x8192x2048 | bfloat16 | 4096 | data-parallel | 1444.9 | 1413.6 | torch | 1408.4 | 1415.7 | 256x256x64 |
| 32768x16384x2048 | bfloat16 | 8192 | data-parallel | 1409.5 | 1360.2 | flydsl | 836.5 | 1421.2 | 256x256x64 |
| 512x6144x3072 | float32 | 48 | one-tile owner, n = 2 | 422.8 | 345.9 | opus | 397.6 | 737.2 | 96x128x128 |
| 512x6144x3072 | bfloat16 | 48 | one-tile owner, n = 2 | 460.9 | 379.0 | flydsl | 738.9 | 781.9 | 96x128x128 |
| 1024x6144x3072 | float32 | 96 | one-tile owner, n = 2-3 | 692.7 | 682.2 | opus | 626.7 | 962.2 | 192x128x64 |
| 1024x6144x3072 | bfloat16 | 96 | one-tile owner, n = 2-3 | 742.7 | 707.5 | flydsl | 883.8 | 957.3 | 192x128x128 |
| 2048x6144x3072 | float32 | 192 | data-parallel | 1106.4 | 1194.5 | asm | 1142.4 | 1188.0 | 192x256x64 |
| 2048x6144x3072 | bfloat16 | 192 | data-parallel | 1195.7 | 1277.2 | flydsl | 1217.8 | 1252.3 | 192x256x64 |
| 4096x6144x3072 | float32 | 384 | data-parallel | 1143.1 | 1166.5 | asm | 1123.5 | 1229.7 | 192x256x64 |
| 4096x6144x3072 | bfloat16 | 384 | data-parallel | 1257.8 | 1270.0 | flydsl | 1108.3 | 1284.9 | 192x256x64 |
| 8192x6144x3072 | float32 | 768 | data-parallel | 1362.1 | 1366.8 | asm | 1323.9 | 1253.4 | 256x256x64 |
| 8192x6144x3072 | bfloat16 | 768 | data-parallel | 1510.2 | 1514.0 | flydsl | 851.9 | 1452.0 | 256x256x64 |
| 16384x6144x3072 | float32 | 1536 | data-parallel | 1327.1 | 1321.6 | asm | 1284.6 | 1241.7 | 256x256x64 |
| 16384x6144x3072 | bfloat16 | 1536 | data-parallel | 1498.5 | 1488.4 | torch | 1467.0 | 1476.7 | 256x256x64 |
| 32768x6144x3072 | bfloat16 | 3072 | data-parallel | 1514.9 | 1490.7 | flydsl | 871.2 | 1475.2 | 256x256x64 |
| 512x5120x3200 | bfloat16 | 40 | one-tile owner, n = 2 | 407.8 | 319.0 | flydsl | 629.0 | 695.7 | 160x64x128 |
| 1024x5120x3200 | bfloat16 | 80 | one-tile owner, n = 2 | 736.0 | 599.2 | torch | 955.3 | 957.5 | 160x128x128 |
| 2048x5120x3200 | bfloat16 | 160 | data-parallel | 1093.0 | 1160.9 | torch | 1168.2 | 1176.1 | 160x256x64 |
| 4096x5120x3200 | bfloat16 | 320 | one-tile owner, n = 2 | 1191.8 | 1109.8 | torch | 1196.4 | 1201.3 | 160x256x64 |
| 8192x5120x3200 | bfloat16 | 640 | data-parallel | 1373.4 | 1378.3 | asm | 1321.7 | 1308.6 | 256x224x64 |
| 16384x5120x3200 | bfloat16 | 1280 | data-parallel | 1511.8 | 1501.7 | torch | 1467.5 | 1473.2 | 256x256x64 |
| 32768x5120x3200 | bfloat16 | 2560 | data-parallel | 1514.8 | 1489.4 | torch | 1474.0 | 1479.0 | 256x256x64 |
| 512x7168x3584 | bfloat16 | 56 | one-tile owner, n = 2 | 572.4 | 446.8 | flydsl | 847.2 | 711.5 | 128x128x256 |
| 512x8192x3584 | bfloat16 | 64 | one-tile owner, n = 2 | 634.3 | 498.2 | flydsl | 776.2 | 773.7 | 128x128x256 |
| 1024x7168x3584 | bfloat16 | 112 | one-tile owner, n = 2-3 | 903.0 | 853.9 | torch | 1081.7 | 1084.6 | 224x128x64 |
| 1024x8192x3584 | bfloat16 | 128 | data-parallel | 930.1 | 950.0 | flydsl | 880.9 | 1154.3 | 256x128x64 |
| 2048x8192x3584 | bfloat16 | 256 | data-parallel | 1437.2 | 1480.1 | flydsl | 1429.9 | 1434.6 | 256x256x64 |
| 4096x8192x3584 | bfloat16 | 512 | data-parallel | 1504.3 | 1510.6 | flydsl | 1032.9 | 1474.3 | 256x256x64 |
| 8192x8192x3584 | bfloat16 | 1024 | data-parallel | 1539.0 | 1528.4 | torch | 1483.3 | 1485.1 | 256x256x64 |
| 10240x7168x3584 | bfloat16 | 1120 | one-tile owner, n = 2-3 | 1454.4 | 1423.1 | flydsl | 834.2 | 1342.9 | 224x256x64 |
| 16384x8192x3584 | bfloat16 | 2048 | data-parallel | 1542.4 | 1522.8 | flydsl | 846.3 | 1494.5 | 256x256x64 |
| 32768x8192x3584 | bfloat16 | 4096 | data-parallel | 1540.4 | 1516.4 | flydsl | 872.1 | 1481.1 | 256x256x64 |
| 512x1024x4096 | bfloat16 | 8 | one-tile reduce-scatter, n = 8 | 178.0 | 66.7 | flydsl | 337.7 | 330.1 | 64x32x256 |
| 512x2048x4096 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 | 293.6 | 133.2 | flydsl | 515.5 | 509.4 | 64x64x256 |
| 512x4096x4096 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 442.4 | 264.6 | flydsl | 686.0 | 632.9 | 128x128x128 |
| 512x8192x4096 | bfloat16 | 64 | one-tile owner, n = 2 | 682.8 | 511.0 | flydsl | 902.9 | 808.8 | 128x128x256 |
| 512x8704x4096 | bfloat16 | 68 | one-tile owner, n = 2 | 720.1 | 549.3 | torch | 901.3 | 902.8 | 160x128x128 |
| 512x10240x4096 | bfloat16 | 80 | one-tile owner, n = 2 | 812.0 | 630.4 | torch | 865.0 | 864.6 | 128x192x64 |
| 512x16384x4096 | bfloat16 | 128 | one-tile owner, n = 2 | 1061.6 | 960.6 | flydsl | 836.3 | 687.0 | 256x256x64 |
| 512x129280x4096 | bfloat16 | 1010 | data-parallel | 1379.5 | 1373.9 | asm | 1309.2 | 1147.2 | 256x256x64 |
| 768x1024x4096 | bfloat16 | 12 | one-tile reduce-scatter, n = 8 | 245.9 | 100.1 | torch | 434.6 | 436.3 | 64x48x256 |
| 768x2048x4096 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 | 369.0 | 199.4 | flydsl | 548.9 | 572.5 | 64x96x256 |
| 768x4096x4096 | bfloat16 | 48 | one-tile owner, n = 3 | 584.1 | 394.3 | flydsl | 761.3 | 807.5 | 128x96x128 |
| 768x8192x4096 | bfloat16 | 96 | one-tile owner, n = 2-3 | 835.1 | 756.3 | torch | 1005.6 | 1010.1 | 128x192x128 |
| 768x8704x4096 | bfloat16 | 102 | one-tile owner, n = 2-3 | 889.7 | 819.2 | flydsl | 954.7 | 986.3 | 160x192x128 |
| 768x10240x4096 | bfloat16 | 120 | one-tile owner, n = 2-3 | 1002.5 | 943.4 | flydsl | 1194.0 | 1079.8 | 160x192x128 |
| 768x129280x4096 | bfloat16 | 1515 | data-parallel | 1439.7 | 1422.7 | torch | 1353.7 | 1354.9 | 256x256x64 |
| 1024x512x4096 | bfloat16 | 8 | one-tile reduce-scatter, n = 8 | 178.3 | 66.6 | flydsl | 332.6 | 329.7 | 64x32x256 |
| 1024x1024x4096 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 | 293.3 | 133.1 | flydsl | 492.5 | 529.6 | 64x64x256 |
| 1024x2048x4096 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 442.7 | 264.4 | flydsl | 700.6 | 551.2 | 128x128x256 |
| 1024x4096x4096 | bfloat16 | 64 | one-tile owner, n = 2 | 684.1 | 516.9 | flydsl | 917.5 | 812.0 | 128x128x256 |
| 1024x8192x4096 | bfloat16 | 128 | one-tile owner, n = 2 | 1064.9 | 971.4 | flydsl | 1230.2 | 766.8 | 256x256x64 |
| 1024x8704x4096 | bfloat16 | 136 | data-parallel | 994.7 | 1062.8 | torch | 1120.5 | 1133.8 | 144x256x64 |
| 1024x10240x4096 | bfloat16 | 160 | data-parallel | 1141.2 | 1183.3 | torch | 1217.5 | 1218.0 | 160x256x64 |
| 1024x16384x4096 | bfloat16 | 256 | data-parallel | 1473.2 | 1517.7 | flydsl | 889.9 | 1476.1 | 256x256x64 |
| 1024x129280x4096 | bfloat16 | 2020 | data-parallel | 1474.7 | 1452.0 | torch | 1377.8 | 1378.3 | 256x256x64 |
| 1280x512x4096 | bfloat16 | 10 | one-tile reduce-scatter, n = 8 | 213.6 | 83.5 | flydsl | 338.9 | 375.4 | 64x48x256 |
| 1280x1024x4096 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 | 325.5 | 166.1 | torch | 559.8 | 564.2 | 64x80x256 |
| 1280x2048x4096 | bfloat16 | 40 | one-tile owner, n = 3 | 498.5 | 331.0 | flydsl | 656.1 | 720.3 | 128x80x128 |
| 1280x129280x4096 | bfloat16 | 2525 | data-parallel | 1439.4 | 1427.2 | torch | 1391.5 | 1390.0 | 256x256x64 |
| 1536x512x4096 | bfloat16 | 12 | one-tile reduce-scatter, n = 8 | 244.5 | 100.1 | torch | 432.2 | 436.2 | 64x48x256 |
| 1536x1024x4096 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 | 367.5 | 199.8 | flydsl | 544.9 | 570.4 | 64x96x256 |
| 1536x2048x4096 | bfloat16 | 48 | one-tile owner, n = 3 | 585.4 | 394.7 | flydsl | 758.5 | 805.0 | 128x96x128 |
| 1536x129280x4096 | bfloat16 | 3030 | data-parallel | 1462.4 | 1444.0 | torch | 1393.9 | 1398.7 | 256x256x64 |
| 1792x512x4096 | bfloat16 | 14 | one-tile reduce-scatter, n = 8 | 269.6 | 116.8 | flydsl | 408.7 | 478.0 | 64x64x256 |
| 1792x1024x4096 | bfloat16 | 28 | one-tile reduce-scatter, n = 4 | 397.8 | 232.6 | flydsl | 601.4 | 504.2 | 256x128x64 |
| 1792x2048x4096 | bfloat16 | 56 | one-tile owner, n = 2 | 609.8 | 457.8 | flydsl | 825.2 | 745.1 | 128x128x256 |
| 1792x129280x4096 | bfloat16 | 3535 | data-parallel | 1480.8 | 1456.6 | asm | 1441.4 | 1357.0 | 256x256x64 |
| 2048x512x4096 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 | 293.8 | 132.8 | flydsl | 480.5 | 514.6 | 64x64x256 |
| 2048x1024x4096 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 442.8 | 264.1 | flydsl | 707.1 | 559.2 | 128x128x256 |
| 2048x2048x4096 | bfloat16 | 64 | one-tile owner, n = 2 | 690.0 | 518.1 | flydsl | 906.8 | 812.1 | 128x128x256 |
| 2048x4096x4096 | bfloat16 | 128 | one-tile owner, n = 2 | 1084.1 | 1000.1 | opus | 1210.5 | 684.6 | 256x256x64 |
| 2048x8192x4096 | bfloat16 | 256 | data-parallel | 1491.3 | 1518.0 | flydsl | 1463.2 | 1465.4 | 256x256x64 |
| 2048x8704x4096 | bfloat16 | 272 | one-tile reduce-scatter, n = 8 | 1263.0 | 995.7 | torch | 1145.5 | 1154.5 | 224x160x64 |
| 2048x10240x4096 | bfloat16 | 320 | one-tile owner, n = 2 | 1262.9 | 1140.4 | torch | 1238.0 | 1244.3 | 160x256x64 |
| 2048x16384x4096 | bfloat16 | 512 | data-parallel | 1546.5 | 1541.4 | flydsl | 1494.4 | 1489.0 | 256x256x64 |
| 2048x129280x4096 | bfloat16 | 4040 | data-parallel | 1511.9 | 1490.1 | flydsl | 1439.4 | 1377.2 | 256x256x64 |
| 2304x512x4096 | bfloat16 | 18 | one-tile reduce-scatter, n = 7 | 300.5 | 149.8 | flydsl | 440.8 | 525.9 | 64x80x256 |
| 2304x1024x4096 | bfloat16 | 36 | one-tile owner, n = 3 | 457.6 | 297.7 | flydsl | 643.8 | 730.5 | 128x80x128 |
| 2304x2048x4096 | bfloat16 | 72 | one-tile owner, n = 2 | 753.5 | 578.2 | opus | 885.0 | 954.1 | 128x160x128 |
| 2560x512x4096 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 | 322.4 | 166.2 | flydsl | 484.9 | 564.2 | 64x80x256 |
| 2560x1024x4096 | bfloat16 | 40 | one-tile owner, n = 3 | 503.1 | 331.3 | flydsl | 685.5 | 719.2 | 128x80x128 |
| 2560x2048x4096 | bfloat16 | 80 | one-tile owner, n = 2 | 805.7 | 633.2 | opus | 968.1 | 1017.1 | 128x160x128 |
| 2816x512x4096 | bfloat16 | 22 | one-tile reduce-scatter, n = 5 | 345.1 | 182.4 | flydsl | 515.0 | 532.8 | 64x96x256 |
| 2816x1024x4096 | bfloat16 | 44 | one-tile owner, n = 3 | 548.6 | 363.5 | flydsl | 724.1 | 753.4 | 128x96x128 |
| 2816x2048x4096 | bfloat16 | 88 | one-tile owner, n = 2-3 | 823.4 | 695.6 | opus | 999.0 | 927.2 | 192x128x128 |
| 3072x512x4096 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 | 366.0 | 199.4 | flydsl | 556.7 | 571.4 | 64x96x256 |
| 3072x1024x4096 | bfloat16 | 48 | one-tile owner, n = 3 | 588.1 | 394.2 | flydsl | 753.1 | 806.7 | 128x96x128 |
| 3072x2048x4096 | bfloat16 | 96 | one-tile owner, n = 2-3 | 846.4 | 754.4 | opus | 1114.9 | 1006.8 | 128x192x128 |
| 3328x512x4096 | bfloat16 | 26 | one-tile reduce-scatter, n = 4 | 378.6 | 216.1 | flydsl | 578.2 | 475.1 | 256x112x64 |
| 3328x1024x4096 | bfloat16 | 52 | one-tile owner, n = 2 | 566.2 | 426.0 | flydsl | 780.6 | 695.8 | 128x128x256 |
| 3328x2048x4096 | bfloat16 | 104 | one-tile owner, n = 2-3 | 904.8 | 821.2 | opus | 1093.6 | 852.5 | 256x208x64 |
| 3584x512x4096 | bfloat16 | 28 | one-tile reduce-scatter, n = 4 | 399.8 | 232.8 | flydsl | 597.5 | 475.3 | 256x128x64 |
| 3584x1024x4096 | bfloat16 | 56 | one-tile owner, n = 2 | 604.5 | 457.3 | flydsl | 798.7 | 745.5 | 128x128x256 |
| 3584x2048x4096 | bfloat16 | 112 | one-tile owner, n = 2-3 | 961.7 | 880.8 | opus | 1154.1 | 994.8 | 160x192x128 |
| 3840x512x4096 | bfloat16 | 30 | one-tile reduce-scatter, n = 4 | 417.4 | 248.8 | flydsl | 629.4 | 543.9 | 128x128x256 |
| 3840x1024x4096 | bfloat16 | 60 | one-tile owner, n = 2 | 644.1 | 488.9 | flydsl | 813.4 | 785.9 | 128x128x256 |
| 3840x2048x4096 | bfloat16 | 120 | one-tile owner, n = 2-3 | 1007.1 | 936.2 | opus | 1178.8 | 834.8 | 256x240x64 |
| 4096x512x4096 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 439.2 | 265.1 | flydsl | 682.5 | 556.0 | 128x128x256 |
| 4096x1024x4096 | bfloat16 | 64 | one-tile owner, n = 2 | 678.1 | 516.7 | flydsl | 908.3 | 816.6 | 128x128x256 |
| 4096x2048x4096 | bfloat16 | 128 | one-tile owner, n = 2 | 1075.6 | 996.0 | opus | 1234.6 | 732.0 | 256x256x64 |
| 4096x4096x4096 | bfloat16 | 256 | data-parallel | 1473.7 | 1521.7 | flydsl | 1466.4 | 1403.2 | 256x256x64 |
| 4096x8192x4096 | bfloat16 | 512 | data-parallel | 1532.7 | 1540.6 | asm | 1517.0 | 1488.3 | 256x256x64 |
| 4096x8704x4096 | bfloat16 | 544 | one-tile reduce-scatter, n = 4 | 1392.5 | 1256.0 | torch | 1262.3 | 1273.6 | 192x256x64 |
| 4096x10240x4096 | bfloat16 | 640 | one-tile owner, n = 2 | 1454.1 | 1413.6 | flydsl | 1366.4 | 1305.8 | 224x256x64 |
| 4096x16384x4096 | bfloat16 | 1024 | data-parallel | 1553.9 | 1537.9 | flydsl | 1491.9 | 1507.9 | 256x256x64 |
| 4096x129280x4096 | bfloat16 | 8080 | data-parallel | 1542.1 | 1507.1 | asm | 1474.5 | 1409.5 | 256x256x64 |
| 4352x512x4096 | bfloat16 | 34 | one-tile owner, n = 3 | 429.0 | 282.0 | flydsl | 486.6 | 634.0 | 128x80x128 |
| 4352x1024x4096 | bfloat16 | 68 | one-tile owner, n = 2 | 715.3 | 537.6 | opus | 844.8 | 905.9 | 128x160x128 |
| 4352x2048x4096 | bfloat16 | 136 | data-parallel | 999.2 | 1052.4 | torch | 1095.3 | 1091.1 | 192x192x64 |
| 5120x512x4096 | bfloat16 | 40 | one-tile owner, n = 3 | 497.5 | 330.3 | flydsl | 622.5 | 716.2 | 128x80x128 |
| 5120x1024x4096 | bfloat16 | 80 | one-tile owner, n = 2 | 799.4 | 629.4 | opus | 987.5 | 1011.3 | 128x160x128 |
| 5120x2048x4096 | bfloat16 | 160 | data-parallel | 1119.2 | 1180.9 | torch | 1240.5 | 1246.8 | 256x160x64 |
| 6144x512x4096 | bfloat16 | 48 | one-tile owner, n = 3 | 580.6 | 394.8 | flydsl | 757.6 | 804.8 | 128x96x128 |
| 6144x1024x4096 | bfloat16 | 96 | one-tile owner, n = 2-3 | 835.1 | 762.8 | opus | 1110.7 | 1018.0 | 128x192x128 |
| 6144x2048x4096 | bfloat16 | 192 | data-parallel | 1263.5 | 1313.8 | opus | 1376.2 | 1303.3 | 256x192x64 |
| 6144x129280x4096 | bfloat16 | 12120 | one-tile owner, n = 2-3 | 1514.1 | 1490.0 | asm | 1482.6 | 1409.6 | 256x256x64 |
| 6400x512x4096 | bfloat16 | 50 | one-tile owner, n = 3 | 599.0 | 410.6 | flydsl | 513.8 | 689.3 | 128x128x256 |
| 6400x1024x4096 | bfloat16 | 100 | one-tile owner, n = 2-3 | 871.9 | 779.9 | opus | 1072.6 | 835.5 | 256x208x64 |
| 6400x2048x4096 | bfloat16 | 200 | data-parallel | 1277.7 | 1325.8 | torch | 1232.7 | 1234.6 | 256x208x64 |
| 6912x512x4096 | bfloat16 | 54 | one-tile owner, n = 2 | 582.8 | 442.8 | flydsl | 731.8 | 718.4 | 128x128x256 |
| 6912x1024x4096 | bfloat16 | 108 | one-tile owner, n = 2-3 | 915.8 | 859.5 | opus | 1106.5 | 719.5 | 256x224x64 |
| 6912x2048x4096 | bfloat16 | 216 | data-parallel | 1326.8 | 1376.4 | torch | 1326.3 | 1326.7 | 256x224x64 |
| 7168x512x4096 | bfloat16 | 56 | one-tile owner, n = 2 | 609.4 | 456.3 | flydsl | 524.6 | 747.3 | 128x128x256 |
| 7168x1024x4096 | bfloat16 | 112 | one-tile owner, n = 2-3 | 960.8 | 896.3 | opus | 1173.8 | 1063.8 | 256x128x64 |
| 7168x2048x4096 | bfloat16 | 224 | data-parallel | 1349.5 | 1399.5 | torch | 1345.5 | 1352.4 | 256x224x64 |
| 7424x512x4096 | bfloat16 | 58 | one-tile owner, n = 2 | 624.5 | 471.6 | flydsl | 586.9 | 760.5 | 128x128x256 |
| 7424x1024x4096 | bfloat16 | 116 | one-tile owner, n = 2-3 | 981.5 | 922.5 | opus | 1132.7 | 814.1 | 256x240x64 |
| 7424x2048x4096 | bfloat16 | 232 | data-parallel | 1379.5 | 1426.8 | torch | 1215.0 | 1219.6 | 256x240x64 |
| 7680x512x4096 | bfloat16 | 60 | one-tile owner, n = 2 | 643.8 | 485.9 | flydsl | 613.3 | 779.0 | 128x128x256 |
| 7680x1024x4096 | bfloat16 | 120 | one-tile owner, n = 2-3 | 995.5 | 950.0 | opus | 1189.1 | 827.8 | 256x240x64 |
| 7680x2048x4096 | bfloat16 | 240 | data-parallel | 1404.3 | 1456.3 | torch | 1242.0 | 1244.9 | 256x240x64 |
| 7936x512x4096 | bfloat16 | 62 | one-tile owner, n = 2 | 662.0 | 499.7 | flydsl | 789.0 | 797.1 | 128x128x256 |
| 7936x1024x4096 | bfloat16 | 124 | one-tile owner, n = 2-3 | 1037.0 | 968.0 | opus | 1166.9 | 735.1 | 256x256x64 |
| 7936x2048x4096 | bfloat16 | 248 | data-parallel | 1441.1 | 1487.5 | torch | 1436.5 | 1443.4 | 256x256x64 |
| 8192x512x4096 | bfloat16 | 64 | one-tile owner, n = 2 | 683.7 | 510.3 | flydsl | 855.5 | 819.1 | 128x128x256 |
| 8192x1024x4096 | bfloat16 | 128 | one-tile owner, n = 2 | 1062.1 | 993.4 | opus | 1243.6 | 703.7 | 256x256x64 |
| 8192x2048x4096 | bfloat16 | 256 | data-parallel | 1468.1 | 1519.7 | torch | 1467.0 | 1468.1 | 256x256x64 |
| 8192x4096x4096 | bfloat16 | 512 | data-parallel | 1538.4 | 1543.4 | flydsl | 1497.2 | 1490.2 | 256x256x64 |
| 8192x8192x4096 | bfloat16 | 1024 | data-parallel | 1564.6 | 1553.8 | asm | 1530.3 | 1507.8 | 256x256x64 |
| 8192x8704x4096 | bfloat16 | 1088 | one-tile owner, n = 2 | 1472.8 | 1425.8 | asm | 1398.3 | 1358.4 | 256x224x64 |
| 8192x10240x4096 | bfloat16 | 1280 | data-parallel | 1545.0 | 1530.3 | torch | 1501.0 | 1502.3 | 256x256x64 |
| 8192x16384x4096 | bfloat16 | 2048 | data-parallel | 1539.4 | 1520.6 | torch | 1511.4 | 1522.9 | 256x256x64 |
| 8192x129280x4096 | bfloat16 | 16160 | one-tile reduce-scatter, n = 4 | 1524.0 | 1520.6 | flydsl | 1461.0 | 1341.8 | 256x256x64 |
| 10240x512x4096 | bfloat16 | 80 | one-tile owner, n = 2 | 795.2 | 638.2 | torch | 1004.1 | 1007.5 | 128x160x128 |
| 10240x1024x4096 | bfloat16 | 160 | data-parallel | 1124.6 | 1177.3 | torch | 1249.6 | 1251.8 | 256x160x64 |
| 10240x2048x4096 | bfloat16 | 320 | one-tile owner, n = 2 | 1254.8 | 1138.8 | torch | 1274.6 | 1278.2 | 256x160x64 |
| 10240x129280x4096 | bfloat16 | 20200 | data-parallel | 1511.9 | 1500.1 | asm | 1495.5 | 1431.4 | 256x256x64 |
| 12288x512x4096 | bfloat16 | 96 | one-tile owner, n = 2-3 | 839.2 | 749.4 | opus | 1098.0 | 1008.9 | 128x192x128 |
| 12288x1024x4096 | bfloat16 | 192 | data-parallel | 1252.7 | 1312.5 | opus | 1375.4 | 1301.1 | 256x192x64 |
| 12288x2048x4096 | bfloat16 | 384 | one-tile owner, n = 2 | 1365.5 | 1311.0 | opus | 1389.1 | 1327.7 | 256x192x64 |
| 12288x129280x4096 | bfloat16 | 24240 | data-parallel | 1548.9 | 1513.4 | asm | 1487.0 | 1423.7 | 256x256x64 |
| 14336x512x4096 | bfloat16 | 112 | one-tile owner, n = 2-3 | 958.0 | 871.4 | opus | 1170.1 | 663.2 | 256x224x64 |
| 14336x1024x4096 | bfloat16 | 224 | data-parallel | 1350.7 | 1399.1 | torch | 1347.6 | 1352.2 | 256x224x64 |
| 14336x2048x4096 | bfloat16 | 448 | data-parallel | 1424.0 | 1428.1 | flydsl | 1390.7 | 1382.7 | 256x224x64 |
| 14336x129280x4096 | bfloat16 | 28280 | one-tile owner, n = 2-3 | 1521.2 | 1498.6 | asm | 1493.4 | 1418.4 | 256x256x64 |
| 16384x512x4096 | bfloat16 | 128 | one-tile owner, n = 2 | 1065.6 | 980.6 | opus | 1247.2 | 809.9 | 256x256x64 |
| 16384x1024x4096 | bfloat16 | 256 | data-parallel | 1460.7 | 1513.4 | flydsl | 1463.7 | 1334.2 | 256x256x64 |
| 16384x2048x4096 | bfloat16 | 512 | data-parallel | 1533.4 | 1537.3 | torch | 1422.8 | 1423.6 | 256x256x64 |
| 16384x16384x4096 | bfloat16 | 4096 | data-parallel | 1549.9 | 1522.2 | torch | 1526.0 | 1532.1 | 256x256x64 |
| 16384x129280x4096 | bfloat16 | 32320 | one-tile owner, n = 2 | 1524.9 | 1526.6 | asm | 1487.1 | 1379.5 | 256x256x64 |
| 32768x16384x4096 | bfloat16 | 8192 | data-parallel | 1556.3 | 1518.2 | flydsl | 1502.8 | 1512.0 | 256x256x64 |
| 512x7168x4224 | bfloat16 | 56 | one-tile owner, n = 2 | 625.8 | 455.7 | flydsl | 839.3 | 875.2 | 128x128x128 |
| 1024x7168x4224 | bfloat16 | 112 | one-tile owner, n = 2-3 | 974.1 | 874.4 | flydsl | 1143.9 | 842.8 | 240x256x64 |
| 10240x7168x4224 | bfloat16 | 1120 | one-tile owner, n = 2-3 | 1482.0 | 1442.8 | opus | 950.2 | 1361.9 | 224x256x64 |
| 512x6144x4608 | bfloat16 | 48 | one-tile owner, n = 3 | 610.3 | 399.9 | flydsl | 776.8 | 674.8 | 192x128x128 |
| 1024x6144x4608 | bfloat16 | 96 | one-tile owner, n = 2-3 | 868.7 | 782.7 | torch | 1012.0 | 1010.4 | 192x128x128 |
| 2048x6144x4608 | bfloat16 | 192 | data-parallel | 1289.8 | 1340.8 | flydsl | 1269.3 | 1316.7 | 192x256x64 |
| 4096x6144x4608 | bfloat16 | 384 | one-tile owner, n = 2 | 1398.0 | 1326.2 | torch | 1332.3 | 1334.6 | 192x256x64 |
| 8192x6144x4608 | bfloat16 | 768 | data-parallel | 1569.5 | 1566.8 | torch | 1512.8 | 1511.4 | 256x256x64 |
| 16384x6144x4608 | bfloat16 | 1536 | data-parallel | 1565.1 | 1551.4 | torch | 1519.7 | 1527.4 | 256x256x64 |
| 32768x6144x4608 | bfloat16 | 3072 | data-parallel | 1568.9 | 1546.0 | torch | 1503.1 | 1503.8 | 256x256x64 |
| 512x1024x5120 | float32 | 8 | one-tile reduce-scatter, n = 10 | 206.0 | 65.4 | asm | 333.9 | 338.5 | 64x32x256 |
| 512x5120x5120 | bfloat16 | 40 | one-tile owner, n = 3 | 558.9 | 335.8 | torch | 790.3 | 794.6 | 96x128x128 |
| 512x6400x5120 | bfloat16 | 50 | one-tile owner, n = 3 | 661.8 | 418.4 | flydsl | 701.1 | 706.9 | 128x128x256 |
| 512x12800x5120 | bfloat16 | 100 | one-tile owner, n = 2-3 | 945.8 | 821.3 | flydsl | 798.4 | 830.0 | 208x256x64 |
| 512x37888x5120 | bfloat16 | 296 | one-tile owner, n = 3 | 1155.5 | 982.8 | torch | 1058.8 | 1061.0 | 160x256x64 |
| 512x51200x5120 | bfloat16 | 400 | data-parallel | 1205.4 | 1204.6 | flydsl | 1139.8 | 1095.5 | 208x256x64 |
| 768x1024x5120 | float32 | 12 | one-tile reduce-scatter, n = 10 | 271.8 | 98.0 | asm | 407.2 | 381.3 | 64x64x256 |
| 768x37888x5120 | bfloat16 | 444 | data-parallel | 1334.6 | 1336.6 | torch | 1239.9 | 1245.7 | 224x256x64 |
| 1024x1024x5120 | float32 | 16 | one-tile reduce-scatter, n = 8 | 330.9 | 130.3 | asm | 458.5 | 484.8 | 64x64x256 |
| 1024x5120x5120 | bfloat16 | 80 | one-tile owner, n = 2 | 867.9 | 653.1 | torch | 1006.6 | 1011.0 | 160x128x128 |
| 1024x6400x5120 | bfloat16 | 100 | one-tile owner, n = 2-3 | 942.3 | 819.0 | flydsl | 945.7 | 881.9 | 256x208x64 |
| 1024x12800x5120 | bfloat16 | 200 | data-parallel | 1321.5 | 1359.3 | flydsl | 793.1 | 1237.5 | 256x208x64 |
| 1024x37888x5120 | bfloat16 | 592 | one-tile owner, n = 2 | 1337.9 | 1262.0 | torch | 1064.1 | 1064.8 | 256x208x64 |
| 1024x51200x5120 | bfloat16 | 800 | one-tile reduce-scatter, n = 4 | 1382.9 | 1270.8 | asm | 1232.1 | 1074.8 | 256x208x64 |
| 1536x1024x5120 | float32 | 24 | one-tile reduce-scatter, n = 5 | 416.9 | 195.3 | asm | 576.3 | 611.5 | 64x96x128 |
| 1536x37888x5120 | bfloat16 | 888 | one-tile owner, n = 2-3 | 1405.2 | 1361.4 | torch | 1276.6 | 1281.8 | 224x256x64 |
| 2048x1024x5120 | float32 | 32 | one-tile reduce-scatter, n = 4 | 487.2 | 257.4 | opus | 631.5 | 665.0 | 128x128x128 |
| 2048x5120x5120 | bfloat16 | 160 | data-parallel | 1174.5 | 1224.5 | torch | 1246.6 | 1251.6 | 160x256x64 |
| 2048x6400x5120 | bfloat16 | 200 | data-parallel | 1321.2 | 1353.3 | flydsl | 1242.7 | 1243.8 | 256x208x64 |
| 2048x12800x5120 | bfloat16 | 400 | data-parallel | 1353.5 | 1378.7 | flydsl | 800.7 | 1261.6 | 256x208x64 |
| 2048x37888x5120 | bfloat16 | 1184 | data-parallel | 1460.4 | 1445.4 | torch | 1214.8 | 1219.8 | 240x256x64 |
| 2048x51200x5120 | bfloat16 | 1600 | one-tile owner, n = 2 | 1465.6 | 1418.2 | asm | 213.5 | 1188.6 | 240x256x64 |
| 4096x1024x5120 | float32 | 64 | one-tile owner, n = 2 | 685.6 | 488.2 | opus | 960.1 | 902.2 | 128x128x128 |
| 4096x5120x5120 | bfloat16 | 320 | one-tile owner, n = 2 | 1307.1 | 1160.2 | torch | 1263.2 | 1270.3 | 160x256x64 |
| 4096x6400x5120 | bfloat16 | 400 | data-parallel | 1363.8 | 1381.2 | flydsl | 1330.3 | 1264.5 | 256x208x64 |
| 4096x12800x5120 | bfloat16 | 800 | one-tile reduce-scatter, n = 4 | 1460.0 | 1366.5 | asm | 1326.5 | 1272.6 | 256x208x64 |
| 4096x51200x5120 | bfloat16 | 3200 | one-tile owner, n = 2 | 1533.8 | 1499.2 | asm | 1461.3 | 1381.9 | 256x256x64 |
| 8192x1024x5120 | float32 | 128 | one-tile owner, n = 2 | 1081.0 | 982.1 | opus | 1243.7 | 1139.5 | 256x128x64 |
| 8192x5120x5120 | bfloat16 | 640 | one-tile owner, n = 2 | 1487.1 | 1423.9 | torch | 1354.3 | 1357.4 | 256x224x64 |
| 8192x6400x5120 | bfloat16 | 800 | one-tile reduce-scatter, n = 4 | 1477.3 | 1363.8 | flydsl | 1323.8 | 1273.8 | 256x208x64 |
| 8192x12800x5120 | bfloat16 | 1600 | one-tile owner, n = 2 | 1505.3 | 1448.7 | asm | 1418.8 | 1235.8 | 256x240x64 |
| 8192x51200x5120 | bfloat16 | 6400 | data-parallel | 1579.2 | 1550.9 | flydsl | 1497.3 | 1409.0 | 256x256x64 |
| 16384x1024x5120 | float32 | 256 | data-parallel | 1416.8 | 1446.7 | opus | 1336.2 | 1335.7 | 256x256x64 |
| 16384x5120x5120 | bfloat16 | 1280 | data-parallel | 1559.7 | 1547.1 | torch | 1524.5 | 1530.9 | 256x256x64 |
| 16384x6400x5120 | bfloat16 | 1600 | one-tile owner, n = 2 | 1490.9 | 1441.9 | flydsl | 1385.9 | 1242.2 | 256x240x64 |
| 16384x12800x5120 | bfloat16 | 3200 | one-tile owner, n = 2 | 1539.0 | 1500.2 | torch | 1485.6 | 1486.9 | 256x256x64 |
| 16384x51200x5120 | bfloat16 | 12800 | data-parallel | 1578.8 | 1545.2 | flydsl | 1487.6 | 1409.9 | 256x256x64 |
| 32768x5120x5120 | bfloat16 | 2560 | data-parallel | 1554.6 | 1535.0 | torch | 1491.5 | 1496.4 | 256x256x64 |
| 32768x6400x5120 | bfloat16 | 3200 | one-tile owner, n = 2 | 1532.6 | 1492.8 | torch | 1470.3 | 1477.2 | 256x256x64 |
| 32768x12800x5120 | bfloat16 | 6400 | data-parallel | 1561.3 | 1536.8 | torch | 1512.3 | 1514.9 | 256x256x64 |
| 32768x51200x5120 | bfloat16 | 25600 | data-parallel | 1574.6 | 1545.8 | flydsl | 1493.8 | 1384.3 | 256x256x64 |
| 512x1280x6144 | bfloat16 | 10 | one-tile reduce-scatter, n = 12 | 267.9 | 86.3 | flydsl | 404.2 | 374.0 | 160x64x128 |
| 512x1536x6144 | bfloat16 | 12 | one-tile reduce-scatter, n = 10 | 304.2 | 103.5 | flydsl | 427.5 | 414.9 | 192x64x128 |
| 512x2048x6144 | float32 | 16 | one-tile reduce-scatter, n = 8 | 379.4 | 133.4 | asm | 486.1 | 510.1 | 64x64x256 |
| 512x2304x6144 | bfloat16 | 18 | one-tile reduce-scatter, n = 7 | 383.2 | 154.1 | flydsl | 515.8 | 410.8 | 192x128x128 |
| 512x2560x6144 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 | 416.4 | 171.0 | torch | 511.2 | 512.3 | 256x128x64 |
| 512x3072x6144 | float32 | 24 | one-tile reduce-scatter, n = 5 | 459.4 | 199.5 | opus | 493.9 | 637.0 | 96x128x128 |
| 512x3072x6144 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 | 464.8 | 205.8 | flydsl | 601.9 | 492.2 | 192x128x128 |
| 512x6144x6144 | bfloat16 | 48 | one-tile owner, n = 3 | 689.7 | 408.1 | flydsl | 799.9 | 776.7 | 192x128x128 |
| 512x9216x6144 | bfloat16 | 72 | one-tile owner, n = 3-4 | 871.9 | 607.8 | flydsl | 891.8 | 580.0 | 256x256x64 |
| 1024x2048x6144 | float32 | 32 | one-tile reduce-scatter, n = 4 | 530.9 | 264.3 | opus | 534.6 | 728.1 | 128x128x128 |
| 1024x2304x6144 | bfloat16 | 36 | one-tile owner, n = 4 | 569.4 | 307.5 | torch | 673.8 | 674.7 | 256x176x64 |
| 1024x2560x6144 | bfloat16 | 40 | one-tile owner, n = 3 | 599.5 | 342.6 | torch | 786.8 | 762.7 | 160x128x128 |
| 1024x3072x6144 | float32 | 48 | one-tile owner, n = 3 | 652.6 | 394.4 | opus | 563.3 | 850.6 | 96x128x128 |
| 1024x6144x6144 | bfloat16 | 96 | one-tile owner, n = 2-3 | 979.5 | 801.9 | torch | 1029.7 | 1030.7 | 192x256x64 |
| 1024x9216x6144 | bfloat16 | 144 | data-parallel | 1120.8 | 1165.2 | torch | 1229.1 | 1232.2 | 144x256x64 |
| 2048x2048x6144 | float32 | 64 | one-tile owner, n = 3 | 802.8 | 505.3 | opus | 886.9 | 947.0 | 128x128x128 |
| 2048x2304x6144 | bfloat16 | 72 | one-tile owner, n = 3-4 | 866.7 | 604.5 | torch | 988.5 | 988.7 | 128x160x128 |
| 2048x2560x6144 | bfloat16 | 80 | one-tile owner, n = 2 | 922.3 | 670.2 | torch | 1051.6 | 1051.2 | 160x128x128 |
| 2048x3072x6144 | float32 | 96 | one-tile owner, n = 2-3 | 924.1 | 775.4 | opus | 816.5 | 954.9 | 192x256x64 |
| 2048x6144x6144 | bfloat16 | 192 | data-parallel | 1323.3 | 1352.9 | torch | 1346.1 | 1352.6 | 192x256x64 |
| 2048x9216x6144 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 | 1334.9 | 1074.0 | torch | 1240.3 | 1247.1 | 144x256x64 |
| 4096x2048x6144 | float32 | 128 | one-tile owner, n = 2 | 1154.7 | 1005.6 | opus | 1152.0 | 868.6 | 256x256x64 |
| 4096x2304x6144 | bfloat16 | 144 | data-parallel | 1128.3 | 1161.5 | flydsl | 782.3 | 1230.7 | 144x256x64 |
| 4096x2560x6144 | bfloat16 | 160 | data-parallel | 1209.3 | 1243.8 | flydsl | 708.4 | 1278.8 | 160x256x64 |
| 4096x3072x6144 | float32 | 192 | data-parallel | 1276.9 | 1317.3 | asm | 1228.5 | 1315.2 | 192x256x64 |
| 4096x6144x6144 | bfloat16 | 384 | one-tile owner, n = 2 | 1463.0 | 1351.9 | torch | 1360.1 | 1363.7 | 192x256x64 |
| 4096x9216x6144 | bfloat16 | 576 | one-tile owner, n = 3 | 1470.9 | 1344.3 | torch | 1368.2 | 1372.3 | 192x256x64 |
| 8192x2048x6144 | float32 | 256 | data-parallel | 1460.9 | 1500.6 | opus | 1394.3 | 1365.0 | 256x256x64 |
| 8192x2304x6144 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 | 1332.3 | 1074.2 | torch | 1235.8 | 1239.0 | 144x256x64 |
| 8192x2560x6144 | bfloat16 | 320 | one-tile owner, n = 3 | 1378.1 | 1176.8 | asm | 521.4 | 1283.7 | 160x256x64 |
| 8192x3072x6144 | float32 | 384 | one-tile owner, n = 2 | 1390.0 | 1293.4 | asm | 1242.6 | 1323.9 | 192x256x64 |
| 8192x6144x6144 | bfloat16 | 768 | data-parallel | 1597.7 | 1593.5 | flydsl | 1548.1 | 1558.4 | 256x256x64 |
| 8192x9216x6144 | bfloat16 | 1152 | one-tile owner, n = 2 | 1539.1 | 1474.9 | asm | 1461.6 | 1261.0 | 256x240x64 |
| 16384x2048x6144 | float32 | 512 | data-parallel | 1413.1 | 1430.6 | opus | 1350.7 | 1334.7 | 256x256x64 |
| 16384x2304x6144 | bfloat16 | 576 | one-tile owner, n = 3 | 1435.1 | 1323.9 | flydsl | 1043.1 | 1334.6 | 192x256x64 |
| 16384x2560x6144 | bfloat16 | 640 | one-tile owner, n = 2 | 1478.6 | 1410.3 | flydsl | 1374.6 | 1361.4 | 256x224x64 |
| 16384x3072x6144 | float32 | 768 | data-parallel | 1430.9 | 1446.4 | asm | 1393.6 | 1353.0 | 256x256x64 |
| 16384x6144x6144 | bfloat16 | 1536 | data-parallel | 1568.9 | 1563.1 | asm | 1526.1 | 1558.7 | 256x256x64 |
| 16384x9216x6144 | bfloat16 | 2304 | data-parallel | 1569.9 | 1556.4 | torch | 1540.1 | 1536.9 | 256x256x64 |
| 32768x2304x6144 | bfloat16 | 1152 | one-tile owner, n = 2 | 1478.7 | 1421.6 | flydsl | 655.0 | 1212.9 | 256x240x64 |
| 32768x2560x6144 | bfloat16 | 1280 | data-parallel | 1521.8 | 1514.8 | flydsl | 1465.4 | 1490.6 | 256x256x64 |
| 32768x6144x6144 | bfloat16 | 3072 | data-parallel | 1578.9 | 1565.3 | torch | 1520.4 | 1533.8 | 256x256x64 |
| 32768x9216x6144 | bfloat16 | 4608 | data-parallel | 1575.3 | 1557.7 | torch | 1560.4 | 1551.0 | 256x256x64 |
| 512x5120x6400 | bfloat16 | 40 | one-tile owner, n = 4 | 637.1 | 341.5 | flydsl | 703.2 | 778.9 | 160x128x128 |
| 1024x5120x6400 | bfloat16 | 80 | one-tile owner, n = 2 | 922.7 | 673.5 | flydsl | 699.5 | 1061.4 | 160x128x128 |
| 2048x5120x6400 | bfloat16 | 160 | data-parallel | 1204.0 | 1234.5 | flydsl | 789.1 | 1283.1 | 160x256x64 |
| 4096x5120x6400 | bfloat16 | 320 | one-tile owner, n = 3 | 1390.0 | 1178.7 | torch | 1290.8 | 1295.1 | 160x256x64 |
| 8192x5120x6400 | bfloat16 | 640 | one-tile owner, n = 2 | 1513.8 | 1443.7 | flydsl | 1409.8 | 1375.2 | 256x224x64 |
| 16384x5120x6400 | bfloat16 | 1280 | data-parallel | 1571.9 | 1559.5 | torch | 1556.6 | 1564.3 | 256x256x64 |
| 32768x5120x6400 | bfloat16 | 2560 | data-parallel | 1583.0 | 1563.6 | torch | 1537.8 | 1565.5 | 256x256x64 |
| 512x16384x6656 | bfloat16 | 128 | one-tile owner, n = 2 | 1158.0 | 1046.3 | flydsl | 954.1 | 919.0 | 256x256x64 |
| 1024x16384x6656 | bfloat16 | 256 | data-parallel | 1539.1 | 1549.8 | asm | 420.8 | 1553.5 | 256x256x64 |
| 2048x16384x6656 | bfloat16 | 512 | data-parallel | 1535.0 | 1534.1 | asm | 1482.6 | 1560.3 | 256x256x64 |
| 4096x16384x6656 | bfloat16 | 1024 | data-parallel | 1564.4 | 1560.2 | torch | 1572.9 | 1581.9 | 256x256x64 |
| 8192x16384x6656 | bfloat16 | 2048 | data-parallel | 1599.7 | 1581.5 | asm | 1528.9 | 1554.1 | 256x256x64 |
| 16384x16384x6656 | bfloat16 | 4096 | data-parallel | 1602.0 | 1574.9 | asm | 1543.4 | 1570.0 | 256x256x64 |
| 32768x16384x6656 | bfloat16 | 8192 | data-parallel | 1607.8 | 1578.9 | asm | 292.4 | 1566.8 | 256x256x64 |
| 512x1024x7168 | bfloat16 | 8 | one-tile reduce-scatter, n = 14 | 259.8 | 69.9 | torch | 435.0 | 433.2 | 128x128x64 |
| 512x1536x7168 | bfloat16 | 12 | one-tile reduce-scatter, n = 10 | 338.0 | 104.8 | torch | 452.5 | 451.8 | 192x64x128 |
| 512x2048x7168 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 | 417.7 | 138.9 | torch | 587.3 | 588.0 | 128x128x128 |
| 512x3584x7168 | bfloat16 | 28 | one-tile reduce-scatter, n = 4 | 525.5 | 242.1 | flydsl | 616.0 | 632.9 | 128x128x256 |
| 512x6144x7168 | bfloat16 | 48 | one-tile owner, n = 3 | 747.7 | 412.3 | flydsl | 821.9 | 811.3 | 192x128x128 |
| 512x8192x7168 | bfloat16 | 64 | one-tile owner, n = 3 | 919.5 | 545.8 | flydsl | 868.8 | 893.1 | 128x128x256 |
| 512x8448x7168 | bfloat16 | 66 | one-tile owner, n = 3-4 | 893.4 | 564.5 | flydsl | 750.7 | 469.7 | 256x256x64 |
| 512x20480x7168 | bfloat16 | 160 | data-parallel | 1084.4 | 1106.1 | flydsl | 1036.3 | 1160.2 | 160x256x64 |
| 768x1024x7168 | bfloat16 | 12 | one-tile reduce-scatter, n = 10 | 341.5 | 104.8 | torch | 445.7 | 447.0 | 256x96x64 |
| 768x2048x7168 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 | 499.7 | 208.5 | flydsl | 614.2 | 640.7 | 256x128x64 |
| 1024x512x7168 | bfloat16 | 8 | one-tile reduce-scatter, n = 14 | 258.9 | 69.9 | flydsl | 382.5 | 327.1 | 256x64x64 |
| 1024x1024x7168 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 | 418.2 | 139.1 | torch | 585.9 | 601.0 | 128x128x128 |
| 1024x1536x7168 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 | 500.5 | 208.8 | flydsl | 633.1 | 636.9 | 256x128x64 |
| 1024x2048x7168 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 573.5 | 277.4 | torch | 744.9 | 733.1 | 256x128x64 |
| 1024x3584x7168 | bfloat16 | 56 | one-tile owner, n = 3 | 842.3 | 480.6 | flydsl | 871.3 | 827.0 | 128x128x256 |
| 1024x6144x7168 | bfloat16 | 96 | one-tile owner, n = 2-3 | 1028.1 | 810.7 | torch | 1042.3 | 1024.8 | 192x256x64 |
| 1024x8192x7168 | bfloat16 | 128 | one-tile owner, n = 2 | 1261.7 | 1059.6 | flydsl | 1095.0 | 984.4 | 256x256x64 |
| 1024x8448x7168 | bfloat16 | 132 | data-parallel | 1069.8 | 1098.5 | torch | 1185.7 | 1183.8 | 144x256x64 |
| 1024x20480x7168 | bfloat16 | 320 | one-tile owner, n = 3 | 1322.0 | 1128.7 | flydsl | 781.4 | 1216.7 | 160x256x64 |
| 1280x512x7168 | bfloat16 | 10 | one-tile reduce-scatter, n = 12 | 299.2 | 87.4 | torch | 410.4 | 406.7 | 256x80x64 |
| 1280x1024x7168 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 | 449.5 | 173.3 | torch | 589.7 | 588.4 | 256x80x128 |
| 1280x2048x7168 | bfloat16 | 40 | one-tile owner, n = 4 | 665.7 | 345.3 | torch | 834.1 | 851.8 | 128x160x128 |
| 1536x512x7168 | bfloat16 | 12 | one-tile reduce-scatter, n = 10 | 338.0 | 104.8 | torch | 447.6 | 448.6 | 256x96x64 |
| 1536x1024x7168 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 | 507.1 | 208.3 | torch | 748.8 | 752.9 | 128x96x128 |
| 1536x2048x7168 | bfloat16 | 48 | one-tile owner, n = 3 | 748.3 | 412.4 | torch | 964.1 | 963.6 | 128x192x64 |
| 1792x512x7168 | bfloat16 | 14 | one-tile reduce-scatter, n = 9 | 381.5 | 122.1 | torch | 494.6 | 491.5 | 256x112x64 |
| 1792x1024x7168 | bfloat16 | 28 | one-tile reduce-scatter, n = 4 | 526.4 | 243.0 | flydsl | 635.2 | 658.9 | 256x128x64 |
| 1792x2048x7168 | bfloat16 | 56 | one-tile owner, n = 3 | 851.0 | 479.1 | flydsl | 902.1 | 806.7 | 128x128x256 |
| 2048x512x7168 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 | 414.5 | 138.9 | torch | 587.7 | 588.2 | 128x128x64 |
| 2048x1024x7168 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 579.0 | 276.3 | torch | 814.0 | 812.8 | 128x128x128 |
| 2048x2048x7168 | bfloat16 | 64 | one-tile owner, n = 3 | 942.4 | 546.6 | flydsl | 986.8 | 893.7 | 128x128x256 |
| 2048x3584x7168 | bfloat16 | 112 | one-tile owner, n = 2-3 | 1158.1 | 946.3 | opus | 1220.2 | 1056.6 | 224x256x64 |
| 2048x8192x7168 | bfloat16 | 256 | data-parallel | 1577.6 | 1583.4 | asm | 1557.5 | 1546.7 | 256x256x64 |
| 2048x20480x7168 | bfloat16 | 640 | one-tile owner, n = 2 | 1486.8 | 1397.9 | flydsl | 825.0 | 1294.0 | 224x256x64 |
| 4096x512x7168 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 581.3 | 277.9 | flydsl | 712.3 | 691.3 | 128x128x256 |
| 4096x1024x7168 | bfloat16 | 64 | one-tile owner, n = 3 | 910.4 | 546.1 | torch | 970.3 | 973.9 | 256x128x64 |
| 4096x2048x7168 | bfloat16 | 128 | one-tile owner, n = 2 | 1264.8 | 1065.5 | torch | 1302.5 | 1307.9 | 256x128x64 |
| 4096x3584x7168 | bfloat16 | 224 | data-parallel | 1448.9 | 1467.2 | flydsl | 1401.0 | 1396.4 | 224x256x64 |
| 4096x8192x7168 | bfloat16 | 512 | data-parallel | 1604.7 | 1599.7 | flydsl | 1551.0 | 1571.0 | 256x256x64 |
| 4096x20480x7168 | bfloat16 | 1280 | data-parallel | 1586.5 | 1576.8 | flydsl | 1528.5 | 1484.3 | 256x256x64 |
| 6144x512x7168 | bfloat16 | 48 | one-tile owner, n = 3 | 746.0 | 413.4 | flydsl | 795.0 | 743.0 | 128x192x128 |
| 6144x1024x7168 | bfloat16 | 96 | one-tile owner, n = 2-3 | 1024.9 | 813.9 | opus | 1191.1 | 928.2 | 256x192x64 |
| 6144x2048x7168 | bfloat16 | 192 | data-parallel | 1334.8 | 1359.3 | opus | 1477.7 | 1367.6 | 256x192x64 |
| 8192x512x7168 | bfloat16 | 64 | one-tile owner, n = 3 | 919.8 | 547.3 | flydsl | 969.2 | 899.5 | 128x128x256 |
| 8192x1024x7168 | bfloat16 | 128 | one-tile owner, n = 2 | 1262.2 | 1066.5 | torch | 1153.9 | 1164.5 | 256x128x64 |
| 8192x2048x7168 | bfloat16 | 256 | data-parallel | 1563.2 | 1585.4 | torch | 1449.3 | 1454.2 | 256x256x64 |
| 8192x3584x7168 | bfloat16 | 448 | data-parallel | 1490.2 | 1487.5 | flydsl | 1451.7 | 1396.6 | 224x256x64 |
| 8192x8192x7168 | bfloat16 | 1024 | data-parallel | 1593.7 | 1587.6 | flydsl | 1540.1 | 1580.5 | 256x256x64 |
| 8192x20480x7168 | bfloat16 | 2560 | data-parallel | 1609.6 | 1594.9 | asm | 1542.9 | 1496.9 | 256x256x64 |
| 10240x512x7168 | bfloat16 | 80 | one-tile owner, n = 3-4 | 1008.1 | 680.1 | torch | 842.1 | 811.3 | 256x160x64 |
| 10240x1024x7168 | bfloat16 | 160 | data-parallel | 1216.7 | 1243.7 | torch | 1311.4 | 1318.1 | 256x160x64 |
| 10240x2048x7168 | bfloat16 | 320 | one-tile owner, n = 3 | 1409.4 | 1181.0 | torch | 1326.3 | 1330.7 | 256x160x64 |
| 10240x8448x7168 | bfloat16 | 1320 | one-tile owner, n = 4 | 1505.7 | 1419.8 | asm | 1418.8 | 1371.8 | 256x224x64 |
| 10240x20480x7168 | bfloat16 | 3200 | one-tile owner, n = 2 | 1525.5 | 1486.4 | asm | 1513.1 | 1479.6 | 256x256x64 |
| 12288x512x7168 | bfloat16 | 96 | one-tile owner, n = 2-3 | 1033.1 | 814.2 | opus | 1190.3 | 857.8 | 256x192x64 |
| 12288x1024x7168 | bfloat16 | 192 | data-parallel | 1332.9 | 1361.3 | opus | 1474.7 | 1370.0 | 256x192x64 |
| 12288x2048x7168 | bfloat16 | 384 | one-tile owner, n = 2 | 1472.7 | 1349.1 | opus | 1480.0 | 1372.5 | 256x192x64 |
| 14336x512x7168 | bfloat16 | 112 | one-tile owner, n = 2-3 | 1081.6 | 945.2 | opus | 1248.2 | 922.3 | 256x224x64 |
| 14336x1024x7168 | bfloat16 | 224 | data-parallel | 1439.1 | 1465.3 | torch | 1405.5 | 1416.5 | 256x224x64 |
| 14336x2048x7168 | bfloat16 | 448 | data-parallel | 1463.5 | 1454.0 | flydsl | 1418.1 | 1400.7 | 256x224x64 |
| 16384x512x7168 | bfloat16 | 128 | one-tile owner, n = 2 | 1168.6 | 1063.5 | opus | 1331.1 | 1040.1 | 256x256x64 |
| 16384x1024x7168 | bfloat16 | 256 | data-parallel | 1497.5 | 1528.4 | torch | 1525.9 | 1531.0 | 256x256x64 |
| 16384x2048x7168 | bfloat16 | 512 | data-parallel | 1535.1 | 1535.1 | torch | 1555.0 | 1562.2 | 256x256x64 |
| 16384x3584x7168 | bfloat16 | 896 | one-tile owner, n = 2 | 1512.3 | 1441.4 | flydsl | 1399.5 | 1390.9 | 224x256x64 |
| 16384x8192x7168 | bfloat16 | 2048 | data-parallel | 1591.6 | 1573.2 | torch | 1554.4 | 1581.4 | 256x256x64 |
| 16384x20480x7168 | bfloat16 | 5120 | data-parallel | 1610.0 | 1585.9 | flydsl | 1541.7 | 1511.5 | 256x256x64 |
| 32768x512x7168 | bfloat16 | 256 | data-parallel | 1362.0 | 1401.7 | torch | 1389.2 | 1396.9 | 256x256x64 |
| 32768x1024x7168 | bfloat16 | 512 | data-parallel | 1500.3 | 1498.5 | torch | 1457.6 | 1473.6 | 256x256x64 |
| 32768x2048x7168 | bfloat16 | 1024 | data-parallel | 1552.2 | 1537.6 | torch | 1504.8 | 1513.7 | 256x256x64 |
| 32768x3584x7168 | bfloat16 | 1792 | data-parallel | 1557.7 | 1542.2 | torch | 1525.3 | 1549.9 | 256x256x64 |
| 32768x8192x7168 | bfloat16 | 4096 | data-parallel | 1596.2 | 1575.2 | torch | 1561.2 | 1591.3 | 256x256x64 |
| 32768x20480x7168 | bfloat16 | 10240 | data-parallel | 1567.0 | 1585.4 | asm | 1536.0 | 1400.7 | 256x256x64 |
| 512x1280x8192 | bfloat16 | 10 | one-tile reduce-scatter, n = 12 | 322.3 | 88.2 | torch | 471.8 | 473.0 | 160x64x128 |
| 512x2560x8192 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 | 484.8 | 175.2 | torch | 609.3 | 607.3 | 256x128x64 |
| 512x4096x8192 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 627.5 | 279.2 | flydsl | 756.7 | 769.8 | 128x128x256 |
| 512x4608x8192 | bfloat16 | 36 | one-tile owner, n = 4 | 657.2 | 313.5 | torch | 851.8 | 839.1 | 160x128x128 |
| 512x7168x8192 | bfloat16 | 56 | one-tile owner, n = 3 | 881.0 | 485.6 | flydsl | 927.3 | 868.8 | 256x176x64 |
| 512x8192x8192 | bfloat16 | 64 | one-tile owner, n = 3 | 960.9 | 552.6 | flydsl | 883.3 | 911.6 | 128x128x256 |
| 512x10240x8192 | bfloat16 | 80 | one-tile owner, n = 3-4 | 1056.3 | 688.4 | flydsl | 693.6 | 573.2 | 256x256x64 |
| 512x14336x8192 | bfloat16 | 112 | one-tile owner, n = 2-3 | 1106.3 | 953.3 | torch | 1045.8 | 1050.8 | 224x256x64 |
| 512x16384x8192 | bfloat16 | 128 | one-tile owner, n = 2 | 1174.4 | 1016.6 | flydsl | 677.1 | 921.6 | 256x256x64 |
| 512x57344x8192 | bfloat16 | 448 | data-parallel | 1328.0 | 1342.5 | flydsl | 1304.6 | 1272.4 | 224x256x64 |
| 768x4096x8192 | bfloat16 | 48 | one-tile owner, n = 4 | 823.1 | 416.0 | flydsl | 829.8 | 787.8 | 128x192x128 |
| 1024x512x8192 | bfloat16 | 8 | one-tile reduce-scatter, n = 16 | 298.4 | 70.6 | torch | 367.5 | 365.4 | 256x64x64 |
| 1024x1280x8192 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 | 491.5 | 174.8 | torch | 621.9 | 623.0 | 256x128x64 |
| 1024x2560x8192 | bfloat16 | 40 | one-tile owner, n = 4 | 713.0 | 348.3 | torch | 862.7 | 871.1 | 160x128x128 |
| 1024x4096x8192 | bfloat16 | 64 | one-tile owner, n = 3 | 962.4 | 551.9 | flydsl | 992.3 | 911.6 | 128x128x256 |
| 1024x4608x8192 | bfloat16 | 72 | one-tile owner, n = 3-4 | 976.4 | 619.3 | torch | 982.2 | 985.0 | 160x256x64 |
| 1024x7168x8192 | bfloat16 | 112 | one-tile owner, n = 2-3 | 1181.2 | 955.2 | flydsl | 832.3 | 1097.7 | 224x256x64 |
| 1024x8192x8192 | bfloat16 | 128 | one-tile owner, n = 2 | 1300.3 | 1074.0 | flydsl | 1157.7 | 1020.3 | 256x256x64 |
| 1024x10240x8192 | bfloat16 | 160 | data-parallel | 1241.5 | 1274.2 | flydsl | 871.0 | 1309.1 | 160x256x64 |
| 1024x14336x8192 | bfloat16 | 224 | data-parallel | 1402.0 | 1438.0 | flydsl | 1419.3 | 1360.9 | 224x256x64 |
| 1024x16384x8192 | bfloat16 | 256 | data-parallel | 1482.8 | 1527.9 | torch | 1489.6 | 1497.9 | 256x256x64 |
| 1024x57344x8192 | bfloat16 | 896 | one-tile owner, n = 2 | 1501.6 | 1425.6 | asm | 1386.5 | 1333.8 | 224x256x64 |
| 2048x512x8192 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 | 447.5 | 140.6 | torch | 533.3 | 537.5 | 256x128x64 |
| 2048x1280x8192 | bfloat16 | 40 | one-tile owner, n = 4 | 716.9 | 349.3 | torch | 727.2 | 726.4 | 320x192x64 |
| 2048x2560x8192 | bfloat16 | 80 | one-tile owner, n = 3-4 | 1055.6 | 687.7 | torch | 1012.4 | 981.9 | 160x256x64 |
| 2048x4096x8192 | bfloat16 | 128 | one-tile owner, n = 2 | 1302.5 | 1078.1 | opus | 1298.7 | 1065.8 | 256x256x64 |
| 2048x4608x8192 | bfloat16 | 144 | data-parallel | 1162.3 | 1189.6 | torch | 1259.6 | 1265.3 | 144x256x64 |
| 2048x7168x8192 | bfloat16 | 224 | data-parallel | 1465.8 | 1489.1 | asm | 1456.8 | 1408.8 | 224x256x64 |
| 2048x8192x8192 | bfloat16 | 256 | data-parallel | 1588.3 | 1604.7 | torch | 1565.3 | 1568.2 | 256x256x64 |
| 2048x10240x8192 | bfloat16 | 320 | one-tile owner, n = 3 | 1416.3 | 1189.1 | flydsl | 990.7 | 1318.9 | 160x256x64 |
| 2048x14336x8192 | bfloat16 | 448 | data-parallel | 1451.4 | 1455.6 | flydsl | 1413.9 | 1358.6 | 224x256x64 |
| 2048x16384x8192 | bfloat16 | 512 | data-parallel | 1556.4 | 1554.0 | asm | 1503.3 | 1565.8 | 256x256x64 |
| 2048x57344x8192 | bfloat16 | 1792 | data-parallel | 1587.9 | 1574.1 | flydsl | 1524.4 | 1489.1 | 256x256x64 |
| 4096x512x8192 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 617.5 | 278.4 | flydsl | 736.1 | 772.6 | 128x128x256 |
| 4096x1280x8192 | bfloat16 | 80 | one-tile owner, n = 3-4 | 1059.0 | 688.2 | torch | 983.5 | 1012.1 | 160x256x64 |
| 4096x2560x8192 | bfloat16 | 160 | data-parallel | 1245.1 | 1270.3 | torch | 1303.8 | 1309.0 | 160x256x64 |
| 4096x4096x8192 | bfloat16 | 256 | data-parallel | 1586.9 | 1603.5 | torch | 1532.9 | 1549.1 | 256x256x64 |
| 4096x4608x8192 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 | 1396.8 | 1087.9 | torch | 1271.2 | 1273.5 | 144x256x64 |
| 4096x7168x8192 | bfloat16 | 448 | data-parallel | 1500.6 | 1495.8 | flydsl | 1466.4 | 1416.8 | 224x256x64 |
| 4096x8192x8192 | bfloat16 | 512 | data-parallel | 1616.4 | 1605.6 | asm | 1582.6 | 1580.9 | 256x256x64 |
| 4096x10240x8192 | bfloat16 | 640 | one-tile owner, n = 2 | 1528.7 | 1433.4 | flydsl | 1394.6 | 1361.2 | 224x256x64 |
| 4096x14336x8192 | bfloat16 | 896 | one-tile owner, n = 2 | 1550.9 | 1469.4 | flydsl | 1428.6 | 1378.2 | 224x256x64 |
| 4096x16384x8192 | bfloat16 | 1024 | data-parallel | 1601.2 | 1588.8 | flydsl | 1535.7 | 1551.9 | 256x256x64 |
| 4096x57344x8192 | bfloat16 | 3584 | data-parallel | 1616.0 | 1598.1 | flydsl | 1552.1 | 1512.2 | 256x256x64 |
| 8192x512x8192 | bfloat16 | 64 | one-tile owner, n = 3 | 956.1 | 552.9 | flydsl | 979.4 | 915.4 | 128x128x256 |
| 8192x1280x8192 | bfloat16 | 160 | data-parallel | 1240.4 | 1270.7 | torch | 1305.2 | 1309.7 | 160x256x64 |
| 8192x2560x8192 | bfloat16 | 320 | one-tile owner, n = 3 | 1442.1 | 1188.4 | torch | 1304.7 | 1311.0 | 160x256x64 |
| 8192x4096x8192 | bfloat16 | 512 | data-parallel | 1618.5 | 1608.7 | torch | 1586.7 | 1577.0 | 256x256x64 |
| 8192x4608x8192 | bfloat16 | 576 | one-tile owner, n = 3 | 1508.0 | 1357.1 | torch | 1375.1 | 1380.1 | 192x256x64 |
| 8192x7168x8192 | bfloat16 | 896 | one-tile owner, n = 2 | 1567.2 | 1488.9 | asm | 1442.5 | 1419.1 | 224x256x64 |
| 8192x8192x8192 | bfloat16 | 1024 | data-parallel | 1604.6 | 1604.3 | asm | 1578.9 | 1574.0 | 256x256x64 |
| 8192x10240x8192 | bfloat16 | 1280 | data-parallel | 1613.0 | 1603.7 | flydsl | 1558.4 | 1591.6 | 256x256x64 |
| 8192x14336x8192 | bfloat16 | 1792 | data-parallel | 1616.5 | 1607.7 | flydsl | 1561.0 | 1569.6 | 256x256x64 |
| 8192x16384x8192 | bfloat16 | 2048 | data-parallel | 1615.2 | 1602.8 | asm | 1557.0 | 1544.4 | 256x256x64 |
| 8192x57344x8192 | bfloat16 | 7168 | data-parallel | 1552.5 | 1609.4 | asm | 1561.2 | 1442.6 | 256x256x64 |
| 16384x512x8192 | bfloat16 | 128 | one-tile owner, n = 2 | 1166.9 | 998.6 | torch | 1063.2 | 1066.1 | 256x256x64 |
| 16384x1280x8192 | bfloat16 | 320 | one-tile owner, n = 3 | 1383.7 | 1182.7 | torch | 1266.6 | 1272.5 | 160x256x64 |
| 16384x2560x8192 | bfloat16 | 640 | one-tile owner, n = 2 | 1490.7 | 1393.0 | flydsl | 1352.5 | 1325.9 | 256x224x64 |
| 16384x4608x8192 | bfloat16 | 1152 | one-tile owner, n = 2 | 1540.2 | 1475.7 | flydsl | 1438.2 | 1223.7 | 256x240x64 |
| 16384x7168x8192 | bfloat16 | 1792 | data-parallel | 1598.6 | 1588.4 | torch | 1589.4 | 1587.0 | 256x256x64 |
| 16384x8192x8192 | bfloat16 | 2048 | data-parallel | 1615.3 | 1593.3 | torch | 1590.5 | 1597.3 | 256x256x64 |
| 16384x10240x8192 | bfloat16 | 2560 | data-parallel | 1614.1 | 1597.8 | torch | 1597.6 | 1596.9 | 256x256x64 |
| 16384x14336x8192 | bfloat16 | 3584 | data-parallel | 1620.5 | 1603.0 | asm | 1566.3 | 1561.6 | 256x256x64 |
| 16384x16384x8192 | bfloat16 | 4096 | data-parallel | 1621.1 | 1602.7 | asm | 1571.5 | 1549.9 | 256x256x64 |
| 16384x57344x8192 | bfloat16 | 14336 | data-parallel | 1548.9 | 1607.0 | asm | 1573.9 | 1424.9 | 256x256x64 |
| 32768x1280x8192 | bfloat16 | 640 | one-tile owner, n = 2 | 1450.4 | 1359.2 | flydsl | 1325.3 | 1347.4 | 256x224x64 |
| 32768x2560x8192 | bfloat16 | 1280 | data-parallel | 1554.3 | 1542.8 | torch | 1549.5 | 1561.2 | 256x256x64 |
| 32768x7168x8192 | bfloat16 | 3584 | data-parallel | 1607.8 | 1585.9 | torch | 1593.5 | 1600.9 | 256x256x64 |
| 32768x8192x8192 | bfloat16 | 4096 | data-parallel | 1612.7 | 1597.1 | flydsl | 1562.4 | 1596.7 | 256x256x64 |
| 32768x10240x8192 | bfloat16 | 5120 | data-parallel | 1617.8 | 1597.0 | torch | 1596.1 | 1596.6 | 256x256x64 |
| 32768x14336x8192 | bfloat16 | 7168 | data-parallel | 1598.3 | 1601.6 | asm | 1567.1 | 1522.8 | 256x256x64 |
| 32768x16384x8192 | bfloat16 | 8192 | data-parallel | 1593.9 | 1602.8 | asm | 1562.9 | 1466.6 | 256x256x64 |
| 32768x57344x8192 | bfloat16 | 28672 | data-parallel | 1548.4 | 1577.9 | asm | 1563.6 | 1444.7 | 256x256x64 |
| 512x6144x12288 | bfloat16 | 48 | one-tile owner, n = 4 | 946.8 | 424.1 | torch | 929.8 | 924.6 | 192x128x128 |
| 1024x6144x12288 | bfloat16 | 96 | one-tile owner, n = 2-3 | 1159.6 | 832.8 | torch | 1173.9 | 1191.6 | 192x256x64 |
| 2048x6144x12288 | bfloat16 | 192 | data-parallel | 1395.2 | 1395.7 | torch | 1396.4 | 1396.8 | 192x256x64 |
| 4096x6144x12288 | bfloat16 | 384 | one-tile owner, n = 2 | 1529.8 | 1368.8 | torch | 1391.8 | 1396.1 | 192x256x64 |
| 8192x6144x12288 | bfloat16 | 768 | data-parallel | 1602.7 | 1591.2 | torch | 1578.1 | 1600.6 | 256x256x64 |
| 16384x6144x12288 | bfloat16 | 1536 | data-parallel | 1586.3 | 1575.1 | torch | 1588.4 | 1588.9 | 256x256x64 |
| 32768x6144x12288 | bfloat16 | 3072 | data-parallel | 1598.7 | 1578.1 | asm | 1573.4 | 1587.4 | 256x256x64 |
| 512x16384x13312 | bfloat16 | 128 | one-tile owner, n = 2 | 1270.2 | 933.2 | flydsl | 712.1 | 1103.3 | 256x256x64 |
| 1024x16384x13312 | bfloat16 | 256 | data-parallel | 1527.3 | 1539.4 | asm | 1494.7 | 1496.8 | 256x256x64 |
| 2048x16384x13312 | bfloat16 | 512 | data-parallel | 1587.8 | 1588.3 | flydsl | 1017.3 | 1517.2 | 256x256x64 |
| 4096x16384x13312 | bfloat16 | 1024 | data-parallel | 1624.1 | 1614.1 | flydsl | 1563.4 | 1520.9 | 256x256x64 |
| 8192x16384x13312 | bfloat16 | 2048 | data-parallel | 1608.4 | 1590.6 | flydsl | 1536.2 | 1523.0 | 256x256x64 |
| 16384x16384x13312 | bfloat16 | 4096 | data-parallel | 1603.5 | 1588.2 | asm | 1543.0 | 1492.0 | 256x256x64 |
| 32768x16384x13312 | bfloat16 | 8192 | data-parallel | 1604.2 | 1584.9 | flydsl | 1535.6 | 1455.7 | 256x256x64 |
| 512x2304x16384 | bfloat16 | 18 | one-tile reduce-scatter, n = 7 | 635.1 | 161.9 | torch | 734.4 | 732.6 | 256x128x64 |
| 512x4608x16384 | bfloat16 | 36 | one-tile owner, n = 6 | 923.0 | 321.0 | flydsl | 582.5 | 951.5 | 160x128x128 |
| 512x9216x16384 | bfloat16 | 72 | one-tile owner, n = 3-4 | 1061.3 | 592.4 | flydsl | 645.4 | 759.9 | 256x256x64 |
| 512x13312x16384 | bfloat16 | 104 | one-tile owner, n = 2-3 | 1137.9 | 752.9 | flydsl | 548.2 | 1080.9 | 208x256x64 |
| 512x26624x16384 | bfloat16 | 208 | data-parallel | 1248.0 | 1315.4 | flydsl | 779.4 | 1203.0 | 208x256x64 |
| 512x53248x16384 | bfloat16 | 416 | data-parallel | 1313.5 | 1309.3 | flydsl | 658.4 | 1209.0 | 208x256x64 |
| 1024x2304x16384 | bfloat16 | 36 | one-tile owner, n = 6 | 896.7 | 321.8 | torch | 895.0 | 900.3 | 256x256x64 |
| 1024x4608x16384 | bfloat16 | 72 | one-tile owner, n = 3-4 | 1189.9 | 640.6 | flydsl | 835.6 | 1136.1 | 144x256x64 |
| 1024x9216x16384 | bfloat16 | 144 | data-parallel | 1124.9 | 1135.5 | flydsl | 722.1 | 1180.9 | 144x256x64 |
| 1024x13312x16384 | bfloat16 | 208 | data-parallel | 1368.0 | 1353.8 | asm | 1334.0 | 1237.0 | 208x256x64 |
| 1024x26624x16384 | bfloat16 | 416 | data-parallel | 1389.2 | 1387.0 | flydsl | 952.7 | 1260.6 | 208x256x64 |
| 1024x53248x16384 | bfloat16 | 832 | one-tile owner, n = 4 | 1539.6 | 1382.9 | triton | 992.4 | 1266.2 | 208x256x64 |
| 2048x2304x16384 | bfloat16 | 72 | one-tile owner, n = 3-4 | 1201.7 | 640.4 | torch | 945.6 | 949.6 | 256x256x64 |
| 2048x4608x16384 | bfloat16 | 144 | data-parallel | 1209.6 | 1219.6 | flydsl | 737.3 | 1314.8 | 144x256x64 |
| 2048x9216x16384 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 | 1430.9 | 1094.7 | flydsl | 919.8 | 1164.1 | 144x256x64 |
| 2048x13312x16384 | bfloat16 | 416 | data-parallel | 1423.3 | 1414.4 | flydsl | 1368.8 | 1246.0 | 208x256x64 |
| 2048x26624x16384 | bfloat16 | 832 | one-tile owner, n = 4 | 1569.0 | 1408.0 | flydsl | 912.0 | 1248.3 | 208x256x64 |
| 2048x53248x16384 | bfloat16 | 1664 | one-tile owner, n = 2 | 1601.5 | 1522.7 | flydsl | 790.6 | 1291.4 | 240x256x64 |
| 4096x2304x16384 | bfloat16 | 144 | data-parallel | 1212.1 | 1219.1 | torch | 1305.9 | 1310.8 | 144x256x64 |
| 4096x4608x16384 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 | 1453.4 | 1104.8 | torch | 1258.6 | 1260.8 | 144x256x64 |
| 4096x9216x16384 | bfloat16 | 576 | one-tile owner, n = 4 | 1564.2 | 1367.7 | torch | 1333.9 | 1338.4 | 192x256x64 |
| 4096x13312x16384 | bfloat16 | 832 | one-tile owner, n = 4 | 1584.9 | 1426.3 | flydsl | 1378.3 | 1268.0 | 208x256x64 |
| 4096x26624x16384 | bfloat16 | 1664 | one-tile owner, n = 2 | 1609.5 | 1538.8 | flydsl | 1502.8 | 1322.0 | 240x256x64 |
| 4096x53248x16384 | bfloat16 | 3328 | data-parallel | 1580.9 | 1613.4 | flydsl | 1544.3 | 1435.9 | 256x256x64 |
| 8192x2304x16384 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 | 1399.1 | 1035.5 | torch | 1202.9 | 1205.6 | 144x256x64 |
| 8192x4608x16384 | bfloat16 | 576 | one-tile owner, n = 4 | 1520.1 | 1318.0 | torch | 1363.4 | 1367.5 | 192x256x64 |
| 8192x9216x16384 | bfloat16 | 1152 | one-tile owner, n = 2 | 1565.3 | 1475.9 | flydsl | 1421.4 | 1149.2 | 256x240x64 |
| 8192x13312x16384 | bfloat16 | 1664 | one-tile owner, n = 2 | 1553.5 | 1499.4 | flydsl | 1465.8 | 1310.5 | 240x256x64 |
| 8192x26624x16384 | bfloat16 | 3328 | data-parallel | 1595.9 | 1580.9 | flydsl | 1519.8 | 1488.9 | 256x256x64 |
| 8192x53248x16384 | bfloat16 | 6656 | data-parallel | 1602.2 | 1560.7 | flydsl | 1538.0 | 1440.7 | 256x256x64 |
| 16384x2304x16384 | bfloat16 | 576 | one-tile owner, n = 4 | 1499.4 | 1309.2 | torch | 1356.0 | 1356.8 | 192x256x64 |
| 16384x4608x16384 | bfloat16 | 1152 | one-tile owner, n = 2 | 1555.8 | 1467.3 | flydsl | 1436.2 | 1177.7 | 256x240x64 |
| 16384x9216x16384 | bfloat16 | 2304 | data-parallel | 1586.9 | 1573.5 | flydsl | 1530.7 | 1524.9 | 256x256x64 |
| 16384x13312x16384 | bfloat16 | 3328 | data-parallel | 1576.8 | 1573.8 | asm | 1532.3 | 1456.1 | 256x256x64 |
| 16384x26624x16384 | bfloat16 | 6656 | data-parallel | 1578.7 | 1578.0 | flydsl | 1537.7 | 1442.9 | 256x256x64 |
| 16384x53248x16384 | bfloat16 | 13312 | data-parallel | 1558.7 | 1577.9 | asm | 1530.2 | 1458.1 | 256x256x64 |
| 32768x2304x16384 | bfloat16 | 1152 | one-tile owner, n = 2 | 1521.9 | 1454.3 | flydsl | 1417.0 | 1190.7 | 256x240x64 |
| 32768x4608x16384 | bfloat16 | 2304 | data-parallel | 1572.6 | 1565.8 | torch | 1551.2 | 1555.6 | 256x256x64 |
| 32768x9216x16384 | bfloat16 | 4608 | data-parallel | 1536.1 | 1576.1 | asm | 1509.8 | 1433.2 | 256x256x64 |
| 32768x13312x16384 | bfloat16 | 6656 | data-parallel | 1536.9 | 1558.4 | asm | 1480.1 | 1436.2 | 256x256x64 |
| 32768x26624x16384 | bfloat16 | 13312 | data-parallel | 1553.2 | 1563.1 | flydsl | 1472.3 | 1434.9 | 256x256x64 |
| 32768x53248x16384 | bfloat16 | 26624 | data-parallel | 1575.3 | 1581.1 | flydsl | 1532.6 | 1453.0 | 256x256x64 |
| 512x6144x18432 | bfloat16 | 48 | one-tile owner, n = 5-6 | 1008.4 | 430.1 | flydsl | 921.4 | 970.6 | 192x128x128 |
| 1024x6144x18432 | bfloat16 | 96 | one-tile owner, n = 2-3 | 1171.8 | 817.4 | torch | 1207.0 | 1208.4 | 192x256x64 |
| 2048x6144x18432 | bfloat16 | 192 | data-parallel | 1346.7 | 1344.4 | torch | 1351.1 | 1355.0 | 192x256x64 |
| 4096x6144x18432 | bfloat16 | 384 | one-tile owner, n = 2 | 1562.5 | 1376.3 | torch | 1341.7 | 1348.0 | 192x256x64 |
| 8192x6144x18432 | bfloat16 | 768 | data-parallel | 1596.1 | 1588.0 | torch | 1510.5 | 1526.0 | 256x256x64 |
| 16384x6144x18432 | bfloat16 | 1536 | data-parallel | 1607.2 | 1593.5 | asm | 1553.8 | 1528.4 | 256x256x64 |
| 32768x6144x18432 | bfloat16 | 3072 | data-parallel | 1614.5 | 1594.1 | asm | 1552.8 | 1518.1 | 256x256x64 |
| 512x5120x25600 | bfloat16 | 40 | one-tile owner, n = 6-7 | 993.6 | 328.2 | flydsl | 439.2 | 909.9 | 160x128x128 |
| 1024x5120x25600 | bfloat16 | 80 | one-tile owner, n = 3-4 | 1287.3 | 656.3 | torch | 1166.7 | 1166.5 | 160x256x64 |
| 2048x5120x25600 | bfloat16 | 160 | data-parallel | 1187.5 | 1176.6 | torch | 1290.5 | 1293.7 | 160x256x64 |
| 4096x5120x25600 | bfloat16 | 320 | one-tile owner, n = 4 | 1547.8 | 1161.5 | torch | 1280.2 | 1282.1 | 160x256x64 |
| 8192x5120x25600 | bfloat16 | 640 | one-tile owner, n = 2 | 1577.6 | 1427.0 | asm | 1402.3 | 1350.7 | 256x224x64 |
| 16384x5120x25600 | bfloat16 | 1280 | data-parallel | 1608.1 | 1591.6 | torch | 1518.2 | 1513.7 | 256x256x64 |
| 32768x5120x25600 | bfloat16 | 2560 | data-parallel | 1495.6 | 1537.9 | torch | 1479.3 | 1460.0 | 256x256x64 |
| 512x16384x26624 | bfloat16 | 128 | one-tile owner, n = 2 | 1369.9 | 945.5 | flydsl | 651.2 | 1192.0 | 256x256x64 |
| 1024x16384x26624 | bfloat16 | 256 | data-parallel | 1566.4 | 1573.0 | flydsl | 786.2 | 1529.1 | 256x256x64 |
| 2048x16384x26624 | bfloat16 | 512 | data-parallel | 1620.8 | 1601.7 | flydsl | 826.0 | 1537.6 | 256x256x64 |
| 4096x16384x26624 | bfloat16 | 1024 | data-parallel | 1638.4 | 1617.7 | flydsl | 1575.9 | 1542.0 | 256x256x64 |
| 8192x16384x26624 | bfloat16 | 2048 | data-parallel | 1627.1 | 1608.3 | asm | 239.6 | 1521.8 | 256x256x64 |
| 16384x16384x26624 | bfloat16 | 4096 | data-parallel | 1629.4 | 1609.9 | asm | 1574.9 | 1490.0 | 256x256x64 |
| 32768x16384x26624 | bfloat16 | 8192 | data-parallel | 1631.0 | 1608.1 | asm | 1570.5 | 1443.7 | 256x256x64 |
| 512x8192x28672 | bfloat16 | 64 | one-tile owner, n = 4 | 1219.5 | 483.1 | flydsl | 799.6 | 917.2 | 240x256x64 |
| 1024x8192x28672 | bfloat16 | 128 | one-tile owner, n = 2 | 1447.8 | 933.2 | triton | 693.7 | 1349.0 | 256x256x64 |
| 2048x8192x28672 | bfloat16 | 256 | data-parallel | 1586.1 | 1600.0 | flydsl | 1031.7 | 1557.7 | 256x256x64 |
| 4096x8192x28672 | bfloat16 | 512 | data-parallel | 1620.7 | 1613.9 | flydsl | 1568.9 | 1569.3 | 256x256x64 |
| 8192x8192x28672 | bfloat16 | 1024 | data-parallel | 1616.4 | 1605.1 | asm | 1573.4 | 1561.1 | 256x256x64 |
| 16384x8192x28672 | bfloat16 | 2048 | data-parallel | 1612.2 | 1605.2 | asm | 1569.3 | 1500.0 | 256x256x64 |
| 32768x8192x28672 | bfloat16 | 4096 | data-parallel | 1623.4 | 1608.1 | asm | 1571.1 | 1463.3 | 256x256x64 |
| 512x7168x35840 | bfloat16 | 56 | one-tile owner, n = 4-5 | 1171.5 | 440.4 | flydsl | 820.2 | 966.7 | 256x176x64 |
| 1024x7168x35840 | bfloat16 | 112 | one-tile owner, n = 2-3 | 1353.7 | 868.3 | flydsl | 576.7 | 1297.5 | 224x256x64 |
| 2048x7168x35840 | bfloat16 | 224 | spread | 1502.2 | 1493.2 | asm | 1453.6 | 1411.2 | 224x256x64 |
| 4096x7168x35840 | bfloat16 | 448 | spread | 1579.8 | 1504.8 | flydsl | 1470.1 | 1409.9 | 224x256x64 |
| 8192x7168x35840 | bfloat16 | 896 | one-tile owner, n = 2 | 1609.2 | 1474.1 | asm | 1433.6 | 1413.3 | 224x256x64 |
| 16384x7168x35840 | bfloat16 | 1792 | data-parallel | 1626.0 | 1585.2 | torch | 1484.3 | 1549.7 | 256x256x64 |
| 32768x7168x35840 | bfloat16 | 3584 | data-parallel | 1625.0 | 1608.2 | flydsl | 1555.8 | 1433.4 | 256x256x64 |
