# streamk_wip (v20, capped) vs AITER, v9 and hipBLASLt on the 630 shapes (2048 MB rotating buffer)

Shapes: `supported_status_scoped.csv` (630 shapes, no bias; 598 bf16 and 32 fp32 outputs). Inputs from `torch.manual_seed(0)`, every output checked against an fp32 `F.linear` reference, `do_bench_cudagraph` with 2 interleaved rounds and the median taken.

- Stack: torch 2.10.0+rocm7.2.4.git3d3aa833, hipBLASLt 100202 (dabb6df2b9), Triton 3.8.0 @ /home/jbowley/repos/triton_gfx950-tutorial-v2.2/python/triton.
- GPU: HIP device 0, PCI bus 117, gfx950:sramecc+:xnack-.
- AITER: e2c7aa0aa [Triton/Gluon] gfx950 a16w16: skip the Gluon kernel on Triton without cd_regclass, on top of a1455f1c4 2026-10-02 [CK] [FlyDSL] add mxfp8 a8w8 blockscale for gfx950 (#4951).
- streamk_wip: tutorial `v20_streamk_reduce_scatter`, snapshot sha256 7c55c4017abfc6cc (newest WORKLOG entry: "9. A split rule from a systematic grid: not adopted (2026-10-04)"), `STREAMK_POLICY=capped` (the kernel's default), v9 tile order. The configuration capped ran on each shape is in the appendix and in `v20_label`.
- hipBLASLt: the call AITER's `torch` libtype makes, `F.linear(x, w)` for bf16 and `torch.mm(x, w.t(), out_dtype=torch.float32)` for fp32, timed the same way. Its tile is the MT of the first `Cijk_` kernel it launches.
- Cache state: each callable rotates over max(2, ceil(2048 MB / set)) copies of x, w and C inside the captured graph, so no launch reads the inputs of the previous one. Outputs that a callable allocates itself (AITER's pick, hipBLASLt) are kept alive in a ring so they rotate too.
- "Faster" and "slower" mean a gap of more than 1%; "within 1%" holds the rest.

Copies per shape: median 15, minimum 2; 70 shapes at the minimum of 2.
`do_bench_cudagraph` sizes its graph from a host-side estimate, so on 56 small shapes a timed graph touched fewer copies than it had. What matters is the data a graph cycles through: copies touched x set size is at least 638 MB on every shape (median 2107 MB), against a 256 MB last-level cache; 0 shapes are under 512 MB.

**Against AITER's pick, by the configuration capped chose**

| | Shapes | v9: faster | v9: within 1% | v9: slower | v9: median | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All shapes | 630 | 295 | 32 | 303 | -0.4% | 356 | 26 | 248 | +2.5% |
| Data-parallel | 324 | 230 | 24 | 70 | +2.6% | 227 | 18 | 79 | +3.1% |
| One-tile reduce-scatter, K >= 2048 | 77 | 6 | 0 | 71 | -68.4% | 14 | 0 | 63 | -26.5% |
| One-tile owner, K >= 2048 | 219 | 57 | 8 | 154 | -20.3% | 113 | 8 | 98 | +2.1% |
| One-tile reduce-scatter, K < 2048 | 6 | 0 | 0 | 6 | -66.0% | 0 | 0 | 6 | -62.1% |
| One-tile owner, K < 2048 | 2 | 0 | 0 | 2 | -49.4% | 0 | 0 | 2 | -47.9% |
| Spread | 2 | 2 | 0 | 0 | +2.4% | 2 | 0 | 0 | +5.4% |

Each pair of columns compares that kernel with AITER's pick; positive medians mean it is faster.

**Against AITER's pick, by the tile AITER uses**

| | Shapes | v9: faster | v9: within 1% | v9: slower | v9: median | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All shapes | 630 | 295 | 32 | 303 | -0.4% | 356 | 26 | 248 | +2.5% |
| AITER uses a 256x256 tile | 248 | 214 | 21 | 13 | +2.7% | 216 | 18 | 14 | +3.7% |
| AITER uses another tile | 382 | 81 | 11 | 290 | -21.8% | 140 | 8 | 234 | -7.1% |

Each pair of columns compares that kernel with AITER's pick; positive medians mean it is faster.

**Against AITER's pick, by v9 grid size (number of 256x256 output tiles)**

| | Shapes | v9: faster | v9: within 1% | v9: slower | v9: median | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All shapes | 630 | 295 | 32 | 303 | -0.4% | 356 | 26 | 248 | +2.5% |
| Under 256 (less than one wave) | 321 | 47 | 6 | 268 | -34.1% | 76 | 13 | 232 | -11.2% |
| 256 to 1023 | 152 | 116 | 11 | 25 | +2.8% | 130 | 8 | 14 | +6.3% |
| 1024 or more | 157 | 132 | 15 | 10 | +2.5% | 150 | 5 | 2 | +4.1% |

Each pair of columns compares that kernel with AITER's pick; positive medians mean it is faster.

**Against hipBLASLt (torch), by v9 grid size**

| | Shapes | AITER's pick: faster | AITER's pick: within 1% | AITER's pick: slower | AITER's pick: median | v9: faster | v9: within 1% | v9: slower | v9: median | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All shapes | 630 | 209 | 238 | 183 | -0.3% | 255 | 39 | 336 | -2.3% | 346 | 24 | 260 | +2.1% |
| Under 256 (less than one wave) | 321 | 109 | 107 | 105 | -0.4% | 35 | 6 | 280 | -35.3% | 80 | 4 | 237 | -14.7% |
| 256 to 1023 | 152 | 38 | 71 | 43 | -0.5% | 98 | 15 | 39 | +2.8% | 119 | 12 | 21 | +5.5% |
| 1024 or more | 157 | 62 | 60 | 35 | -0.1% | 122 | 18 | 17 | +3.4% | 147 | 8 | 2 | +4.6% |

Each pair of columns compares that kernel with hipBLASLt; positive medians mean it is faster.

**hipBLASLt's tile choice**

| hipBLASLt tile (MxNxK) | Shapes | hipBLASLt median TFLOPS | streamk_wip vs hipBLASLt: faster | within 1% | slower | median |
|---|---:|---:|---:|---:|---:|---:|
| 256x256x64 | 225 | 1431 | 192 | 19 | 14 | +3.8% |
| 128x128x256 | 40 | 661 | 9 | 0 | 31 | -14.9% |
| 192x256x64 | 34 | 1221 | 13 | 1 | 20 | -1.9% |
| 224x256x64 | 32 | 1302 | 32 | 0 | 0 | +6.6% |
| 256x128x64 | 27 | 735 | 3 | 0 | 24 | -19.4% |
| 160x256x64 | 27 | 1180 | 14 | 0 | 13 | +1.8% |
| 256x224x64 | 20 | 1299 | 18 | 0 | 2 | +8.6% |
| 256x240x64 | 18 | 1146 | 18 | 0 | 0 | +22.8% |
| 144x256x64 | 15 | 1154 | 7 | 0 | 8 | -7.9% |
| 128x128x128 | 14 | 565 | 0 | 0 | 14 | -31.1% |
| 160x128x128 | 13 | 796 | 2 | 0 | 11 | -16.0% |
| 96x128x128 | 12 | 561 | 0 | 0 | 12 | -41.7% |
| 192x128x128 | 12 | 775 | 2 | 0 | 10 | -12.8% |
| 256x208x64 | 12 | 1074 | 10 | 1 | 1 | +15.6% |
| 256x192x64 | 11 | 1213 | 3 | 1 | 7 | -2.8% |
| 208x256x64 | 11 | 1244 | 10 | 0 | 1 | +10.2% |
| 256x160x64 | 10 | 1149 | 2 | 1 | 7 | -8.5% |
| 64x64x256 | 9 | 403 | 0 | 0 | 9 | -35.6% |
| 128x192x128 | 7 | 880 | 1 | 0 | 6 | -17.8% |
| 128x160x128 | 7 | 860 | 0 | 0 | 7 | -16.8% |
| 240x256x64 | 7 | 1215 | 7 | 0 | 0 | +21.4% |
| 128x80x128 | 6 | 606 | 0 | 0 | 6 | -25.7% |
| 128x96x128 | 6 | 662 | 0 | 0 | 6 | -21.1% |
| 224x128x64 | 5 | 787 | 0 | 0 | 5 | -21.0% |
| 256x176x64 | 4 | 700 | 2 | 1 | 1 | +4.2% |
| 128x64x128 | 4 | 455 | 0 | 0 | 4 | -55.3% |
| 160x64x128 | 4 | 424 | 0 | 0 | 4 | -34.2% |
| 64x80x256 | 4 | 466 | 0 | 0 | 4 | -39.0% |
| 64x96x256 | 4 | 504 | 0 | 0 | 4 | -34.0% |
| 64x32x256 | 3 | 265 | 0 | 0 | 3 | -38.8% |
| 64x48x256 | 3 | 369 | 0 | 0 | 3 | -39.5% |
| 160x192x128 | 3 | 914 | 0 | 0 | 3 | -2.2% |
| 192x128x64 | 2 | 818 | 0 | 0 | 2 | -30.7% |
| 128x192x64 | 2 | 850 | 0 | 0 | 2 | -18.3% |
| 256x112x64 | 2 | 444 | 0 | 0 | 2 | -24.7% |
| 192x64x128 | 2 | 393 | 0 | 0 | 2 | -31.1% |
| 128x128x64 | 2 | 449 | 0 | 0 | 2 | -32.4% |
| 256x96x64 | 2 | 410 | 0 | 0 | 2 | -28.6% |
| 256x64x64 | 2 | 313 | 0 | 0 | 2 | -18.1% |
| 96x64x128 | 1 | 392 | 0 | 0 | 1 | -61.3% |
| 224x160x64 | 1 | 1094 | 1 | 0 | 0 | +11.4% |
| 192x192x64 | 1 | 1047 | 0 | 0 | 1 | -9.9% |
| 64x96x128 | 1 | 512 | 0 | 0 | 1 | -26.9% |
| 256x80x64 | 1 | 362 | 0 | 0 | 1 | -28.1% |
| 256x80x128 | 1 | 549 | 0 | 0 | 1 | -28.2% |
| 320x192x64 | 1 | 704 | 0 | 0 | 1 | -5.1% |

Solution-name tokens (stream-K `SK`, global split-U `GSU`, kernels launched): SK3 GSU0 x1: 604, SK3 GSU0 x2: 24, none GSU- x1: 2. `SK3` marks a kernel built with stream-K support, not that this launch split K.

**streamk_wip against v9, by configuration**

| | Shapes | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|
| All shapes | 630 | 422 | 93 | 115 | +2.6% |
| Data-parallel | 324 | 119 | 91 | 114 | +0.4% |
| One-tile reduce-scatter, K >= 2048 | 77 | 77 | 0 | 0 | +128.8% |
| One-tile owner, K >= 2048 | 219 | 217 | 1 | 1 | +27.0% |
| One-tile reduce-scatter, K < 2048 | 6 | 6 | 0 | 0 | +11.6% |
| One-tile owner, K < 2048 | 2 | 2 | 0 | 0 | +2.7% |
| Spread | 2 | 1 | 1 | 0 | +2.9% |

Each pair of columns compares that kernel with v9; positive medians mean it is faster.

**Diagnostic: capped's split against forcing data-parallel**

| Configuration | Shapes | split faster | within 1% | split slower | median |
|---|---:|---:|---:|---:|---:|
| One-tile owner, K < 2048 | 2 | 2 | 0 | 0 | +6.3% |
| One-tile owner, K >= 2048 | 219 | 211 | 8 | 0 | +28.2% |
| One-tile reduce-scatter, K < 2048 | 6 | 6 | 0 | 0 | +16.8% |
| One-tile reduce-scatter, K >= 2048 | 77 | 76 | 1 | 0 | +130.9% |
| Spread | 2 | 1 | 1 | 0 | +1.9% |

**Selected shapes (TFLOPS)**

| Shape (MxNxK) | Output tiles | AITER pick | AITER | hipBLASLt | hipBLASLt tile | v9 | streamk_wip | streamk_wip config | vs v9 | vs AITER | vs hipBLASLt |
|---|---:|---|---:|---:|---|---:|---:|---|---:|---:|---:|
| 512x2304x16384 | 18 | torch | 705 | 706 | 256x128x64 | 139 | 545 | one-tile reduce-scatter, n = 7 | +292% | -23% | -23% |
| 512x5120x25600 | 40 | flydsl | 430 | 876 | 160x128x128 | 296 | 968 | one-tile owner, n = 6-7 | +227% | +125% | +10% |
| 512x7168x35840 | 56 | flydsl | 820 | 966 | 256x176x64 | 439 | 1162 | one-tile owner, n = 4-5 | +165% | +42% | +20% |
| 8192x2304x16384 | 288 | torch | 1182 | 1186 | 144x256x64 | 1012 | 1382 | one-tile reduce-scatter, n = 4 | +37% | +17% | +17% |
| 4096x5120x25600 | 320 | torch | 1275 | 1282 | 160x256x64 | 1160 | 1544 | one-tile owner, n = 4 | +33% | +21% | +20% |
| 4096x4608x8192 | 288 | torch | 1180 | 1186 | 144x256x64 | 1076 | 1362 | one-tile reduce-scatter, n = 4 | +27% | +15% | +15% |
| 1024x3072x512 | 48 | flydsl | 366 | 324 | 96x128x128 | 203 | 192 | data-parallel | -5% | -47% | -41% |
| 2048x2048x512 | 64 | flydsl | 431 | 394 | 128x128x128 | 262 | 254 | data-parallel | -3% | -41% | -36% |
| 4096x5120x640 | 320 | flydsl | 616 | 654 | 256x176x64 | 641 | 656 | data-parallel | +2% | +6% | +0% |
| 8192x2048x512 | 256 | torch | 759 | 769 | 256x256x64 | 790 | 747 | data-parallel | -5% | -2% | -3% |

**Against `results/details_capped_hbl.csv`** (this run / that run - 1, median; for v20 the baseline may use a different policy or kernel snapshot)

| | Shapes | AITER's pick | v9 | v20 | hipBLASLt |
|---|---:|---:|---:|---:|---:|
| All shapes | 630 | -5.6% | -6.5% | -5.3% | -4.7% |
| Under 256 (less than one wave) | 321 | -11.2% | -10.3% | -8.8% | -9.8% |
| 256 to 1023 | 152 | -4.4% | -3.7% | -3.3% | -3.5% |
| 1024 or more | 157 | -0.9% | -0.5% | -0.4% | -1.3% |
| K < 2048 | 91 | -8.8% | -8.4% | -7.5% | -7.8% |
| K >= 2048 | 539 | -5.1% | -5.6% | -4.3% | -4.1% |

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
| 1024x3072x512 | bfloat16 | 48 | data-parallel | 192.4 | 202.8 | flydsl | 365.9 | 323.9 | 96x128x128 |
| 1024x4096x512 | bfloat16 | 64 | data-parallel | 251.2 | 259.7 | flydsl | 440.8 | 393.6 | 128x128x128 |
| 1024x7168x512 | bfloat16 | 112 | data-parallel | 407.6 | 420.4 | flydsl | 512.5 | 505.5 | 256x128x64 |
| 2048x2048x512 | bfloat16 | 64 | data-parallel | 254.0 | 261.8 | flydsl | 431.4 | 394.3 | 128x128x128 |
| 2048x4096x512 | bfloat16 | 128 | data-parallel | 452.2 | 464.9 | torch | 559.5 | 558.2 | 256x128x64 |
| 4096x2048x512 | bfloat16 | 128 | data-parallel | 447.8 | 465.1 | torch | 644.3 | 654.8 | 256x128x64 |
| 4096x4096x512 | bfloat16 | 256 | data-parallel | 738.9 | 789.9 | torch | 766.4 | 779.3 | 256x256x64 |
| 4096x7168x512 | bfloat16 | 448 | data-parallel | 769.9 | 768.3 | flydsl | 748.3 | 726.4 | 224x256x64 |
| 8192x2048x512 | bfloat16 | 256 | data-parallel | 747.0 | 789.9 | torch | 758.6 | 768.8 | 256x256x64 |
| 8192x4096x512 | bfloat16 | 512 | data-parallel | 834.7 | 817.8 | flydsl | 809.7 | 839.4 | 256x256x64 |
| 16384x2048x512 | bfloat16 | 512 | data-parallel | 844.8 | 833.1 | torch | 835.2 | 839.1 | 256x256x64 |
| 16384x4096x512 | bfloat16 | 1024 | data-parallel | 894.2 | 843.9 | torch | 896.9 | 900.8 | 256x256x64 |
| 16384x7168x512 | bfloat16 | 1792 | data-parallel | 977.5 | 907.2 | torch | 988.3 | 993.7 | 256x256x64 |
| 32768x2048x512 | bfloat16 | 1024 | data-parallel | 904.1 | 856.4 | torch | 881.4 | 881.4 | 256x256x64 |
| 1024x5120x640 | bfloat16 | 80 | data-parallel | 315.7 | 325.4 | torch | 447.5 | 447.5 | 160x128x128 |
| 2048x5120x640 | bfloat16 | 160 | data-parallel | 589.1 | 609.8 | flydsl | 597.3 | 510.3 | 192x256x64 |
| 4096x5120x640 | bfloat16 | 320 | data-parallel | 655.7 | 640.8 | flydsl | 615.8 | 653.7 | 256x176x64 |
| 8192x5120x640 | bfloat16 | 640 | data-parallel | 831.9 | 800.6 | torch | 867.4 | 867.9 | 256x224x64 |
| 16384x5120x640 | bfloat16 | 1280 | data-parallel | 1006.7 | 961.8 | torch | 1000.8 | 1018.8 | 256x256x64 |
| 32768x5120x640 | bfloat16 | 2560 | data-parallel | 1051.9 | 1002.2 | torch | 1039.7 | 1038.9 | 256x256x64 |
| 1024x7168x768 | bfloat16 | 112 | data-parallel | 467.5 | 488.0 | flydsl | 597.7 | 591.7 | 224x128x64 |
| 10240x7168x768 | bfloat16 | 1120 | data-parallel | 1037.8 | 1015.2 | opus | 390.4 | 862.2 | 256x240x64 |
| 512x4096x1024 | bfloat16 | 32 | data-parallel | 177.1 | 181.6 | torch | 388.3 | 393.5 | 128x64x128 |
| 512x6144x1024 | bfloat16 | 48 | data-parallel | 260.5 | 269.1 | flydsl | 502.0 | 476.9 | 96x128x128 |
| 512x7168x1024 | bfloat16 | 56 | data-parallel | 300.3 | 311.1 | flydsl | 536.4 | 499.6 | 128x128x128 |
| 512x8192x1024 | bfloat16 | 64 | data-parallel | 339.2 | 346.6 | flydsl | 581.4 | 551.9 | 128x128x128 |
| 1024x4096x1024 | bfloat16 | 64 | data-parallel | 338.8 | 347.5 | torch | 559.1 | 564.1 | 128x128x128 |
| 1024x7168x1024 | bfloat16 | 112 | data-parallel | 557.6 | 573.6 | flydsl | 668.8 | 697.2 | 224x128x64 |
| 1024x8192x1024 | bfloat16 | 128 | data-parallel | 623.6 | 631.6 | torch | 757.0 | 769.6 | 256x128x64 |
| 2048x4096x1024 | bfloat16 | 128 | data-parallel | 626.4 | 635.5 | torch | 768.3 | 779.3 | 256x128x64 |
| 2048x7168x1024 | bfloat16 | 224 | data-parallel | 927.5 | 1014.0 | flydsl | 959.7 | 966.7 | 256x256x64 |
| 2048x8192x1024 | bfloat16 | 256 | data-parallel | 999.8 | 1078.3 | flydsl | 1011.3 | 1017.1 | 256x256x64 |
| 3072x4096x1024 | bfloat16 | 192 | data-parallel | 846.4 | 901.9 | torch | 913.0 | 927.8 | 256x192x64 |
| 4096x4096x1024 | bfloat16 | 256 | data-parallel | 987.8 | 1081.4 | torch | 1025.6 | 1040.2 | 256x256x64 |
| 4096x7168x1024 | bfloat16 | 448 | data-parallel | 1019.5 | 1044.2 | flydsl | 995.2 | 982.8 | 224x256x64 |
| 4096x8192x1024 | bfloat16 | 512 | data-parallel | 1091.9 | 1103.9 | asm | 1114.6 | 1109.9 | 256x256x64 |
| 5120x4096x1024 | bfloat16 | 320 | data-parallel | 857.6 | 860.5 | torch | 976.8 | 981.2 | 256x160x64 |
| 7680x4096x1024 | bfloat16 | 480 | data-parallel | 1070.8 | 1095.6 | torch | 946.9 | 952.5 | 256x240x64 |
| 8192x7168x1024 | bfloat16 | 896 | data-parallel | 1091.6 | 1054.8 | asm | 1048.8 | 1027.6 | 224x256x64 |
| 8192x8192x1024 | bfloat16 | 1024 | data-parallel | 1173.3 | 1142.4 | asm | 1146.1 | 1161.2 | 256x256x64 |
| 16384x4096x1024 | bfloat16 | 1024 | data-parallel | 1176.3 | 1147.3 | torch | 1152.6 | 1162.7 | 256x256x64 |
| 16384x7168x1024 | bfloat16 | 1792 | data-parallel | 1225.4 | 1191.4 | asm | 1141.9 | 1205.8 | 256x256x64 |
| 16384x8192x1024 | bfloat16 | 2048 | data-parallel | 1230.7 | 1181.2 | torch | 1195.9 | 1215.3 | 256x256x64 |
| 32768x7168x1024 | bfloat16 | 3584 | data-parallel | 1252.9 | 1215.8 | torch | 1228.7 | 1228.3 | 256x256x64 |
| 32768x8192x1024 | bfloat16 | 4096 | data-parallel | 1258.3 | 1209.7 | torch | 1217.0 | 1230.6 | 256x256x64 |
| 512x5120x1280 | bfloat16 | 40 | data-parallel | 194.0 | 203.8 | flydsl | 450.0 | 424.0 | 160x64x128 |
| 1024x5120x1280 | bfloat16 | 80 | data-parallel | 378.3 | 395.9 | torch | 604.0 | 610.0 | 160x128x128 |
| 2048x5120x1280 | bfloat16 | 160 | data-parallel | 723.1 | 766.1 | torch | 846.6 | 852.8 | 160x256x64 |
| 4096x5120x1280 | bfloat16 | 320 | data-parallel | 805.9 | 800.0 | torch | 880.8 | 882.7 | 160x256x64 |
| 8192x5120x1280 | bfloat16 | 640 | data-parallel | 1040.1 | 1021.0 | flydsl | 1004.2 | 1073.4 | 256x224x64 |
| 16384x5120x1280 | bfloat16 | 1280 | data-parallel | 1231.4 | 1211.3 | flydsl | 1182.3 | 1234.0 | 256x256x64 |
| 32768x5120x1280 | bfloat16 | 2560 | data-parallel | 1272.2 | 1240.4 | torch | 1253.5 | 1265.1 | 256x256x64 |
| 512x1536x1536 | bfloat16 | 12 | one-tile reduce-scatter, n = 8 | 90.4 | 70.1 | flydsl | 263.5 | 238.0 | 64x64x256 |
| 512x2048x1536 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 | 115.3 | 93.0 | torch | 310.9 | 313.3 | 64x64x256 |
| 512x2304x1536 | bfloat16 | 18 | one-tile reduce-scatter, n = 7 | 120.8 | 104.7 | flydsl | 318.4 | 312.8 | 64x80x256 |
| 512x4096x1536 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 199.7 | 186.8 | torch | 443.1 | 449.1 | 128x64x128 |
| 512x6144x1536 | bfloat16 | 48 | data-parallel | 265.1 | 278.6 | flydsl | 541.5 | 545.6 | 96x128x128 |
| 512x7168x1536 | bfloat16 | 56 | data-parallel | 305.7 | 322.0 | flydsl | 588.8 | 496.6 | 128x128x256 |
| 1024x1536x1536 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 | 151.6 | 140.7 | flydsl | 400.7 | 392.1 | 96x64x128 |
| 1024x2048x1536 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 199.9 | 186.1 | torch | 455.4 | 461.3 | 128x64x128 |
| 1024x2304x1536 | bfloat16 | 36 | one-tile owner, n = 2 | 212.0 | 208.6 | torch | 457.9 | 459.3 | 128x80x128 |
| 1024x3072x1536 | bfloat16 | 48 | data-parallel | 263.6 | 278.2 | flydsl | 554.9 | 555.4 | 96x128x128 |
| 1024x4096x1536 | bfloat16 | 64 | data-parallel | 348.7 | 365.5 | torch | 561.0 | 561.6 | 128x128x256 |
| 1024x7168x1536 | bfloat16 | 112 | data-parallel | 586.6 | 623.7 | torch | 776.8 | 787.4 | 224x128x64 |
| 2048x1536x1536 | bfloat16 | 48 | data-parallel | 266.8 | 283.1 | torch | 553.0 | 556.8 | 96x128x128 |
| 2048x2048x1536 | bfloat16 | 64 | data-parallel | 352.8 | 371.5 | torch | 563.4 | 564.9 | 128x128x256 |
| 2048x4096x1536 | bfloat16 | 128 | data-parallel | 669.9 | 696.5 | torch | 849.1 | 854.3 | 256x128x64 |
| 3072x2048x1536 | bfloat16 | 96 | data-parallel | 515.6 | 540.3 | opus | 689.9 | 777.1 | 128x192x128 |
| 3072x4096x1536 | bfloat16 | 192 | data-parallel | 934.5 | 990.7 | opus | 996.3 | 959.2 | 256x192x64 |
| 4096x1536x1536 | bfloat16 | 96 | data-parallel | 514.8 | 541.3 | torch | 767.3 | 760.4 | 192x128x128 |
| 4096x2048x1536 | bfloat16 | 128 | data-parallel | 668.2 | 692.1 | opus | 856.2 | 850.0 | 256x128x64 |
| 4096x3072x1536 | bfloat16 | 192 | data-parallel | 934.9 | 996.5 | flydsl | 940.2 | 1009.6 | 192x256x64 |
| 4096x4096x1536 | bfloat16 | 256 | data-parallel | 1110.8 | 1205.8 | torch | 1176.3 | 1182.9 | 256x256x64 |
| 5120x2048x1536 | bfloat16 | 160 | data-parallel | 800.1 | 847.9 | torch | 953.0 | 959.6 | 256x160x64 |
| 5120x4096x1536 | bfloat16 | 320 | data-parallel | 940.9 | 947.4 | torch | 1005.0 | 1014.0 | 256x160x64 |
| 7680x2048x1536 | bfloat16 | 240 | data-parallel | 1063.3 | 1148.0 | torch | 957.7 | 963.6 | 256x240x64 |
| 7680x4096x1536 | bfloat16 | 480 | data-parallel | 1167.3 | 1198.1 | torch | 1034.0 | 1044.1 | 256x240x64 |
| 8192x1536x1536 | bfloat16 | 192 | data-parallel | 919.3 | 984.2 | torch | 996.6 | 1006.6 | 192x256x64 |
| 10240x7168x1536 | bfloat16 | 1120 | data-parallel | 1232.0 | 1220.4 | asm | 1153.0 | 1138.9 | 224x256x64 |
| 16384x1536x1536 | bfloat16 | 384 | data-parallel | 1038.3 | 1068.9 | torch | 1064.5 | 1067.1 | 192x256x64 |
| 16384x2048x1536 | bfloat16 | 512 | data-parallel | 1191.7 | 1220.1 | torch | 1217.3 | 1226.5 | 256x256x64 |
| 16384x3072x1536 | bfloat16 | 768 | data-parallel | 1259.2 | 1254.4 | torch | 1248.8 | 1254.0 | 256x256x64 |
| 16384x4096x1536 | bfloat16 | 1024 | data-parallel | 1268.6 | 1247.2 | torch | 1264.0 | 1276.5 | 256x256x64 |
| 32768x1536x1536 | bfloat16 | 768 | data-parallel | 1240.0 | 1237.3 | flydsl | 1206.0 | 1223.3 | 256x256x64 |
| 512x7168x1792 | bfloat16 | 56 | one-tile owner, n = 2 | 359.4 | 346.1 | flydsl | 621.1 | 506.8 | 128x128x256 |
| 1024x7168x1792 | bfloat16 | 112 | data-parallel | 648.0 | 673.3 | flydsl | 784.5 | 833.1 | 224x128x64 |
| 2048x7168x1792 | bfloat16 | 224 | data-parallel | 1095.0 | 1178.5 | flydsl | 1118.6 | 1076.3 | 224x256x64 |
| 4096x7168x1792 | bfloat16 | 448 | data-parallel | 1195.5 | 1211.9 | flydsl | 1156.3 | 1130.2 | 224x256x64 |
| 8192x7168x1792 | bfloat16 | 896 | data-parallel | 1246.3 | 1227.9 | asm | 1200.5 | 1172.7 | 224x256x64 |
| 16384x7168x1792 | bfloat16 | 1792 | data-parallel | 1369.5 | 1342.7 | torch | 1348.9 | 1354.0 | 256x256x64 |
| 32768x7168x1792 | bfloat16 | 3584 | data-parallel | 1392.7 | 1356.5 | torch | 1354.3 | 1369.3 | 256x256x64 |
| 512x4096x2048 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 259.3 | 215.5 | torch | 489.5 | 493.3 | 128x64x128 |
| 512x6144x2048 | float32 | 48 | one-tile owner, n = 2 | 314.2 | 289.3 | asm | 333.4 | 564.9 | 96x128x128 |
| 512x6144x2048 | bfloat16 | 48 | one-tile owner, n = 2 | 346.3 | 322.8 | flydsl | 597.1 | 596.9 | 96x128x128 |
| 512x8192x2048 | bfloat16 | 64 | one-tile owner, n = 2 | 443.4 | 413.2 | flydsl | 708.3 | 579.4 | 128x128x256 |
| 512x16384x2048 | bfloat16 | 128 | data-parallel | 738.3 | 756.6 | flydsl | 719.7 | 875.3 | 256x128x64 |
| 1024x4096x2048 | bfloat16 | 64 | one-tile owner, n = 2 | 447.4 | 419.3 | torch | 616.0 | 618.0 | 128x128x256 |
| 1024x6144x2048 | float32 | 96 | data-parallel | 536.1 | 559.1 | opus | 563.0 | 788.1 | 192x128x64 |
| 1024x6144x2048 | bfloat16 | 96 | data-parallel | 598.9 | 609.2 | flydsl | 738.6 | 811.4 | 192x128x128 |
| 1024x8192x2048 | bfloat16 | 128 | data-parallel | 759.1 | 774.4 | flydsl | 883.5 | 933.2 | 256x128x64 |
| 1024x16384x2048 | bfloat16 | 256 | data-parallel | 1184.4 | 1271.0 | flydsl | 750.2 | 1235.8 | 256x256x64 |
| 2048x4096x2048 | bfloat16 | 128 | data-parallel | 774.3 | 785.6 | torch | 939.5 | 947.3 | 256x128x64 |
| 2048x6144x2048 | float32 | 192 | data-parallel | 918.4 | 976.4 | asm | 923.8 | 1025.5 | 192x256x64 |
| 2048x6144x2048 | bfloat16 | 192 | data-parallel | 1041.5 | 1103.8 | flydsl | 1046.4 | 1103.7 | 192x256x64 |
| 2048x8192x2048 | bfloat16 | 256 | data-parallel | 1227.4 | 1292.6 | flydsl | 1230.3 | 1240.8 | 256x256x64 |
| 2048x16384x2048 | bfloat16 | 512 | data-parallel | 1300.4 | 1317.8 | flydsl | 1044.2 | 1290.4 | 256x256x64 |
| 3072x4096x2048 | bfloat16 | 192 | data-parallel | 1044.5 | 1099.9 | torch | 1100.0 | 1110.5 | 256x192x64 |
| 4096x4096x2048 | bfloat16 | 256 | data-parallel | 1224.6 | 1296.0 | torch | 1260.2 | 1273.3 | 256x256x64 |
| 4096x6144x2048 | float32 | 384 | data-parallel | 992.2 | 1002.0 | asm | 960.9 | 1065.9 | 192x256x64 |
| 4096x6144x2048 | bfloat16 | 384 | data-parallel | 1129.1 | 1133.9 | flydsl | 893.1 | 1152.4 | 192x256x64 |
| 4096x8192x2048 | bfloat16 | 512 | data-parallel | 1312.3 | 1320.2 | flydsl | 1274.7 | 1315.9 | 256x256x64 |
| 4096x16384x2048 | bfloat16 | 1024 | data-parallel | 1358.4 | 1344.5 | flydsl | 862.2 | 1350.9 | 256x256x64 |
| 5120x4096x2048 | bfloat16 | 320 | one-tile owner, n = 2 | 1039.1 | 1034.2 | torch | 1103.1 | 1137.3 | 256x160x64 |
| 7680x4096x2048 | bfloat16 | 480 | data-parallel | 1274.9 | 1296.1 | torch | 1104.4 | 1113.4 | 256x240x64 |
| 8192x6144x2048 | float32 | 768 | data-parallel | 1189.7 | 1187.9 | asm | 1154.6 | 1114.8 | 256x256x64 |
| 8192x6144x2048 | bfloat16 | 768 | data-parallel | 1368.4 | 1349.9 | flydsl | 835.3 | 1339.2 | 256x256x64 |
| 8192x8192x2048 | bfloat16 | 1024 | data-parallel | 1373.6 | 1352.9 | asm | 1326.2 | 1359.1 | 256x256x64 |
| 8192x16384x2048 | bfloat16 | 2048 | data-parallel | 1388.1 | 1348.7 | triton | 1019.7 | 1381.1 | 256x256x64 |
| 16384x4096x2048 | bfloat16 | 1024 | data-parallel | 1379.3 | 1352.9 | torch | 1327.8 | 1336.4 | 256x256x64 |
| 16384x6144x2048 | float32 | 1536 | data-parallel | 1222.4 | 1202.4 | asm | 1171.6 | 1143.5 | 256x256x64 |
| 16384x6144x2048 | bfloat16 | 1536 | data-parallel | 1386.8 | 1363.9 | torch | 1361.3 | 1376.6 | 256x256x64 |
| 16384x8192x2048 | bfloat16 | 2048 | data-parallel | 1405.4 | 1371.0 | torch | 1376.2 | 1380.3 | 256x256x64 |
| 16384x16384x2048 | bfloat16 | 4096 | data-parallel | 1412.6 | 1370.0 | flydsl | 967.2 | 1398.4 | 256x256x64 |
| 32768x6144x2048 | bfloat16 | 3072 | data-parallel | 1414.4 | 1395.2 | flydsl | 995.4 | 1379.4 | 256x256x64 |
| 32768x8192x2048 | bfloat16 | 4096 | data-parallel | 1434.5 | 1398.5 | torch | 1379.5 | 1396.2 | 256x256x64 |
| 32768x16384x2048 | bfloat16 | 8192 | data-parallel | 1413.8 | 1370.7 | flydsl | 830.5 | 1409.2 | 256x256x64 |
| 512x6144x3072 | float32 | 48 | one-tile owner, n = 2 | 367.3 | 294.8 | opus | 375.4 | 626.5 | 96x128x128 |
| 512x6144x3072 | bfloat16 | 48 | one-tile owner, n = 2 | 396.8 | 317.4 | flydsl | 639.1 | 658.7 | 96x128x128 |
| 1024x6144x3072 | float32 | 96 | one-tile owner, n = 2-3 | 598.6 | 580.2 | opus | 567.9 | 848.4 | 192x128x64 |
| 1024x6144x3072 | bfloat16 | 96 | one-tile owner, n = 2-3 | 633.2 | 621.5 | flydsl | 790.1 | 855.2 | 192x128x128 |
| 2048x6144x3072 | float32 | 192 | data-parallel | 1010.1 | 1048.3 | asm | 1016.9 | 1119.1 | 192x256x64 |
| 2048x6144x3072 | bfloat16 | 192 | data-parallel | 1118.1 | 1179.1 | flydsl | 1119.8 | 1180.7 | 192x256x64 |
| 4096x6144x3072 | float32 | 384 | data-parallel | 1071.4 | 1085.0 | asm | 1047.3 | 1158.1 | 192x256x64 |
| 4096x6144x3072 | bfloat16 | 384 | data-parallel | 1200.8 | 1205.6 | flydsl | 929.8 | 1223.0 | 192x256x64 |
| 8192x6144x3072 | float32 | 768 | data-parallel | 1288.9 | 1289.9 | asm | 1263.6 | 1210.0 | 256x256x64 |
| 8192x6144x3072 | bfloat16 | 768 | data-parallel | 1442.5 | 1430.5 | flydsl | 783.9 | 1404.3 | 256x256x64 |
| 16384x6144x3072 | float32 | 1536 | data-parallel | 1319.8 | 1302.1 | asm | 1273.6 | 1235.9 | 256x256x64 |
| 16384x6144x3072 | bfloat16 | 1536 | data-parallel | 1484.5 | 1459.5 | torch | 1426.2 | 1436.2 | 256x256x64 |
| 32768x6144x3072 | bfloat16 | 3072 | data-parallel | 1503.3 | 1470.4 | flydsl | 868.6 | 1450.5 | 256x256x64 |
| 512x5120x3200 | bfloat16 | 40 | one-tile owner, n = 2 | 359.9 | 270.1 | flydsl | 574.8 | 565.0 | 160x64x128 |
| 1024x5120x3200 | bfloat16 | 80 | one-tile owner, n = 2 | 687.2 | 534.0 | torch | 821.5 | 828.1 | 160x128x128 |
| 2048x5120x3200 | bfloat16 | 160 | data-parallel | 1017.4 | 1054.7 | torch | 1092.1 | 1100.1 | 160x256x64 |
| 4096x5120x3200 | bfloat16 | 320 | one-tile owner, n = 2 | 1158.6 | 1065.2 | torch | 1130.4 | 1137.7 | 160x256x64 |
| 8192x5120x3200 | bfloat16 | 640 | data-parallel | 1332.0 | 1314.9 | asm | 1266.7 | 1276.6 | 256x224x64 |
| 16384x5120x3200 | bfloat16 | 1280 | data-parallel | 1485.5 | 1455.9 | torch | 1425.5 | 1437.6 | 256x256x64 |
| 32768x5120x3200 | bfloat16 | 2560 | data-parallel | 1505.6 | 1473.2 | torch | 1440.2 | 1448.2 | 256x256x64 |
| 512x7168x3584 | bfloat16 | 56 | one-tile owner, n = 2 | 503.3 | 378.6 | flydsl | 717.0 | 603.6 | 128x128x256 |
| 512x8192x3584 | bfloat16 | 64 | one-tile owner, n = 2 | 570.0 | 431.5 | flydsl | 676.0 | 655.2 | 128x128x256 |
| 1024x7168x3584 | bfloat16 | 112 | one-tile owner, n = 2-3 | 840.3 | 745.9 | torch | 959.3 | 964.0 | 224x128x64 |
| 1024x8192x3584 | bfloat16 | 128 | data-parallel | 825.8 | 833.9 | flydsl | 791.0 | 1023.4 | 256x128x64 |
| 2048x8192x3584 | bfloat16 | 256 | data-parallel | 1373.9 | 1404.9 | flydsl | 1365.6 | 1362.2 | 256x256x64 |
| 4096x8192x3584 | bfloat16 | 512 | data-parallel | 1444.7 | 1445.9 | flydsl | 984.6 | 1418.8 | 256x256x64 |
| 8192x8192x3584 | bfloat16 | 1024 | data-parallel | 1482.3 | 1462.0 | torch | 1444.1 | 1455.1 | 256x256x64 |
| 10240x7168x3584 | bfloat16 | 1120 | one-tile owner, n = 2-3 | 1420.4 | 1387.0 | flydsl | 785.8 | 1310.3 | 224x256x64 |
| 16384x8192x3584 | bfloat16 | 2048 | data-parallel | 1499.7 | 1476.4 | flydsl | 809.7 | 1470.9 | 256x256x64 |
| 32768x8192x3584 | bfloat16 | 4096 | data-parallel | 1531.1 | 1498.4 | flydsl | 861.1 | 1473.2 | 256x256x64 |
| 512x1024x4096 | bfloat16 | 8 | one-tile reduce-scatter, n = 8 | 160.8 | 60.3 | flydsl | 284.0 | 264.7 | 64x32x256 |
| 512x2048x4096 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 | 276.2 | 120.7 | flydsl | 439.0 | 428.9 | 64x64x256 |
| 512x4096x4096 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 411.8 | 241.9 | flydsl | 601.3 | 618.0 | 128x128x128 |
| 512x8192x4096 | bfloat16 | 64 | one-tile owner, n = 2 | 616.9 | 469.5 | flydsl | 807.5 | 690.9 | 128x128x256 |
| 512x8704x4096 | bfloat16 | 68 | one-tile owner, n = 2 | 642.6 | 497.9 | torch | 751.9 | 754.2 | 160x128x128 |
| 512x10240x4096 | bfloat16 | 80 | one-tile owner, n = 2 | 737.3 | 580.7 | torch | 859.9 | 863.9 | 128x192x64 |
| 512x16384x4096 | bfloat16 | 128 | one-tile owner, n = 2 | 958.3 | 857.5 | flydsl | 634.5 | 678.5 | 256x256x64 |
| 512x129280x4096 | bfloat16 | 1010 | data-parallel | 1373.9 | 1367.0 | asm | 1305.8 | 1149.9 | 256x256x64 |
| 768x1024x4096 | bfloat16 | 12 | one-tile reduce-scatter, n = 8 | 224.0 | 91.3 | torch | 371.0 | 372.0 | 64x48x256 |
| 768x2048x4096 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 | 333.2 | 182.2 | flydsl | 526.6 | 502.4 | 64x96x256 |
| 768x4096x4096 | bfloat16 | 48 | one-tile owner, n = 3 | 531.8 | 362.4 | flydsl | 690.1 | 659.1 | 128x96x128 |
| 768x8192x4096 | bfloat16 | 96 | one-tile owner, n = 2-3 | 749.1 | 688.0 | torch | 876.5 | 879.9 | 128x192x128 |
| 768x8704x4096 | bfloat16 | 102 | one-tile owner, n = 2-3 | 799.8 | 727.1 | flydsl | 869.8 | 844.4 | 160x192x128 |
| 768x10240x4096 | bfloat16 | 120 | one-tile owner, n = 2-3 | 912.0 | 836.4 | flydsl | 973.4 | 932.3 | 160x192x128 |
| 768x129280x4096 | bfloat16 | 1515 | data-parallel | 1439.9 | 1420.4 | torch | 1346.9 | 1354.5 | 256x256x64 |
| 1024x512x4096 | bfloat16 | 8 | one-tile reduce-scatter, n = 8 | 160.7 | 60.5 | flydsl | 283.0 | 262.6 | 64x32x256 |
| 1024x1024x4096 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 | 278.4 | 121.7 | flydsl | 443.0 | 442.8 | 64x64x256 |
| 1024x2048x4096 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 411.0 | 242.4 | flydsl | 604.3 | 507.5 | 128x128x256 |
| 1024x4096x4096 | bfloat16 | 64 | one-tile owner, n = 2 | 625.4 | 471.3 | flydsl | 830.1 | 733.5 | 128x128x256 |
| 1024x8192x4096 | bfloat16 | 128 | one-tile owner, n = 2 | 997.9 | 881.4 | flydsl | 1022.0 | 668.5 | 256x256x64 |
| 1024x8704x4096 | bfloat16 | 136 | data-parallel | 927.7 | 941.9 | torch | 998.0 | 1006.8 | 144x256x64 |
| 1024x10240x4096 | bfloat16 | 160 | data-parallel | 1055.8 | 1079.8 | torch | 1122.8 | 1124.2 | 160x256x64 |
| 1024x16384x4096 | bfloat16 | 256 | data-parallel | 1361.9 | 1404.7 | flydsl | 824.2 | 1336.2 | 256x256x64 |
| 1024x129280x4096 | bfloat16 | 2020 | data-parallel | 1470.2 | 1448.8 | torch | 1369.4 | 1378.2 | 256x256x64 |
| 1280x512x4096 | bfloat16 | 10 | one-tile reduce-scatter, n = 8 | 192.6 | 75.9 | flydsl | 321.4 | 313.5 | 64x48x256 |
| 1280x1024x4096 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 | 299.2 | 152.3 | torch | 487.2 | 487.7 | 64x80x256 |
| 1280x2048x4096 | bfloat16 | 40 | one-tile owner, n = 3 | 457.8 | 304.7 | flydsl | 622.5 | 614.0 | 128x80x128 |
| 1280x129280x4096 | bfloat16 | 2525 | data-parallel | 1437.2 | 1418.8 | torch | 1387.2 | 1390.7 | 256x256x64 |
| 1536x512x4096 | bfloat16 | 12 | one-tile reduce-scatter, n = 8 | 223.3 | 90.8 | torch | 366.0 | 369.3 | 64x48x256 |
| 1536x1024x4096 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 | 334.1 | 183.3 | flydsl | 520.2 | 508.1 | 64x96x256 |
| 1536x2048x4096 | bfloat16 | 48 | one-tile owner, n = 3 | 541.6 | 366.3 | flydsl | 700.6 | 687.5 | 128x96x128 |
| 1536x129280x4096 | bfloat16 | 3030 | data-parallel | 1460.4 | 1439.2 | torch | 1394.0 | 1398.1 | 256x256x64 |
| 1792x512x4096 | bfloat16 | 14 | one-tile reduce-scatter, n = 8 | 250.8 | 106.2 | flydsl | 395.7 | 384.6 | 64x64x256 |
| 1792x1024x4096 | bfloat16 | 28 | one-tile reduce-scatter, n = 4 | 370.8 | 213.9 | flydsl | 577.1 | 468.6 | 256x128x64 |
| 1792x2048x4096 | bfloat16 | 56 | one-tile owner, n = 2 | 563.2 | 424.1 | flydsl | 752.0 | 668.8 | 128x128x256 |
| 1792x129280x4096 | bfloat16 | 3535 | data-parallel | 1477.8 | 1448.9 | asm | 1437.8 | 1355.7 | 256x256x64 |
| 2048x512x4096 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 | 277.0 | 122.0 | flydsl | 426.0 | 430.3 | 64x64x256 |
| 2048x1024x4096 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 412.7 | 244.3 | flydsl | 603.3 | 507.8 | 128x128x256 |
| 2048x2048x4096 | bfloat16 | 64 | one-tile owner, n = 2 | 630.7 | 477.1 | flydsl | 834.2 | 741.3 | 128x128x256 |
| 2048x4096x4096 | bfloat16 | 128 | one-tile owner, n = 2 | 1028.9 | 890.3 | opus | 1034.5 | 713.7 | 256x256x64 |
| 2048x8192x4096 | bfloat16 | 256 | data-parallel | 1405.5 | 1430.1 | flydsl | 1391.5 | 1390.2 | 256x256x64 |
| 2048x8704x4096 | bfloat16 | 272 | one-tile reduce-scatter, n = 8 | 1218.6 | 984.0 | torch | 1083.4 | 1094.0 | 224x160x64 |
| 2048x10240x4096 | bfloat16 | 320 | one-tile owner, n = 2 | 1212.8 | 1108.5 | torch | 1161.6 | 1167.3 | 160x256x64 |
| 2048x16384x4096 | bfloat16 | 512 | data-parallel | 1458.6 | 1456.6 | flydsl | 1407.8 | 1412.7 | 256x256x64 |
| 2048x129280x4096 | bfloat16 | 4040 | data-parallel | 1512.7 | 1487.0 | flydsl | 1431.2 | 1376.4 | 256x256x64 |
| 2304x512x4096 | bfloat16 | 18 | one-tile reduce-scatter, n = 7 | 271.7 | 136.3 | flydsl | 350.9 | 445.1 | 64x80x256 |
| 2304x1024x4096 | bfloat16 | 36 | one-tile owner, n = 3 | 416.7 | 274.1 | flydsl | 511.5 | 647.1 | 128x80x128 |
| 2304x2048x4096 | bfloat16 | 72 | one-tile owner, n = 2 | 685.9 | 532.1 | opus | 687.8 | 894.3 | 128x160x128 |
| 2560x512x4096 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 | 297.4 | 151.6 | flydsl | 390.5 | 487.7 | 64x80x256 |
| 2560x1024x4096 | bfloat16 | 40 | one-tile owner, n = 3 | 459.4 | 304.7 | flydsl | 559.9 | 610.8 | 128x80x128 |
| 2560x2048x4096 | bfloat16 | 80 | one-tile owner, n = 2 | 747.9 | 586.3 | opus | 770.2 | 887.7 | 128x160x128 |
| 2816x512x4096 | bfloat16 | 22 | one-tile reduce-scatter, n = 5 | 311.5 | 165.8 | flydsl | 418.7 | 471.1 | 64x96x256 |
| 2816x1024x4096 | bfloat16 | 44 | one-tile owner, n = 3 | 497.6 | 335.0 | flydsl | 600.1 | 642.8 | 128x96x128 |
| 2816x2048x4096 | bfloat16 | 88 | one-tile owner, n = 2-3 | 773.3 | 640.8 | opus | 824.3 | 854.9 | 192x128x128 |
| 3072x512x4096 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 | 333.0 | 181.9 | flydsl | 455.1 | 505.1 | 64x96x256 |
| 3072x1024x4096 | bfloat16 | 48 | one-tile owner, n = 3 | 533.6 | 365.8 | flydsl | 644.4 | 675.1 | 128x96x128 |
| 3072x2048x4096 | bfloat16 | 96 | one-tile owner, n = 2-3 | 761.9 | 693.6 | opus | 915.0 | 930.1 | 128x192x128 |
| 3328x512x4096 | bfloat16 | 26 | one-tile reduce-scatter, n = 4 | 351.8 | 197.3 | flydsl | 485.4 | 446.3 | 256x112x64 |
| 3328x1024x4096 | bfloat16 | 52 | one-tile owner, n = 2 | 519.8 | 393.4 | flydsl | 682.6 | 628.7 | 128x128x256 |
| 3328x2048x4096 | bfloat16 | 104 | one-tile owner, n = 2-3 | 820.4 | 744.8 | opus | 941.6 | 785.2 | 256x208x64 |
| 3584x512x4096 | bfloat16 | 28 | one-tile reduce-scatter, n = 4 | 368.3 | 211.8 | flydsl | 510.2 | 452.9 | 256x128x64 |
| 3584x1024x4096 | bfloat16 | 56 | one-tile owner, n = 2 | 557.1 | 421.0 | flydsl | 714.6 | 666.0 | 128x128x256 |
| 3584x2048x4096 | bfloat16 | 112 | one-tile owner, n = 2-3 | 901.9 | 793.5 | opus | 1006.5 | 914.0 | 160x192x128 |
| 3840x512x4096 | bfloat16 | 30 | one-tile reduce-scatter, n = 4 | 387.8 | 228.1 | flydsl | 539.1 | 487.3 | 128x128x256 |
| 3840x1024x4096 | bfloat16 | 60 | one-tile owner, n = 2 | 586.6 | 446.3 | flydsl | 741.9 | 700.8 | 128x128x256 |
| 3840x2048x4096 | bfloat16 | 120 | one-tile owner, n = 2-3 | 966.5 | 842.6 | opus | 1037.7 | 770.6 | 256x240x64 |
| 4096x512x4096 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 407.5 | 241.2 | flydsl | 581.6 | 501.3 | 128x128x256 |
| 4096x1024x4096 | bfloat16 | 64 | one-tile owner, n = 2 | 624.8 | 472.1 | flydsl | 832.1 | 738.7 | 128x128x256 |
| 4096x2048x4096 | bfloat16 | 128 | one-tile owner, n = 2 | 1034.1 | 887.1 | opus | 1093.7 | 747.9 | 256x256x64 |
| 4096x4096x4096 | bfloat16 | 256 | data-parallel | 1396.5 | 1438.2 | flydsl | 1381.8 | 1392.4 | 256x256x64 |
| 4096x8192x4096 | bfloat16 | 512 | data-parallel | 1481.4 | 1481.4 | asm | 1448.9 | 1433.4 | 256x256x64 |
| 4096x8704x4096 | bfloat16 | 544 | one-tile reduce-scatter, n = 4 | 1351.9 | 1227.5 | torch | 1215.2 | 1219.3 | 192x256x64 |
| 4096x10240x4096 | bfloat16 | 640 | one-tile owner, n = 2 | 1405.8 | 1358.8 | flydsl | 1311.6 | 1267.0 | 224x256x64 |
| 4096x16384x4096 | bfloat16 | 1024 | data-parallel | 1502.6 | 1489.6 | flydsl | 1459.9 | 1448.3 | 256x256x64 |
| 4096x129280x4096 | bfloat16 | 8080 | data-parallel | 1542.7 | 1503.3 | asm | 1475.6 | 1401.3 | 256x256x64 |
| 4352x512x4096 | bfloat16 | 34 | one-tile owner, n = 3 | 389.4 | 253.5 | flydsl | 447.5 | 525.3 | 128x80x128 |
| 4352x1024x4096 | bfloat16 | 68 | one-tile owner, n = 2 | 650.5 | 498.0 | opus | 644.5 | 814.5 | 128x160x128 |
| 4352x2048x4096 | bfloat16 | 136 | data-parallel | 943.4 | 980.2 | torch | 1046.7 | 1046.6 | 192x192x64 |
| 5120x512x4096 | bfloat16 | 40 | one-tile owner, n = 3 | 454.1 | 300.9 | flydsl | 536.9 | 601.7 | 128x80x128 |
| 5120x1024x4096 | bfloat16 | 80 | one-tile owner, n = 2 | 735.9 | 582.4 | opus | 769.0 | 884.5 | 128x160x128 |
| 5120x2048x4096 | bfloat16 | 160 | data-parallel | 1071.7 | 1085.7 | torch | 1166.5 | 1169.9 | 256x160x64 |
| 6144x512x4096 | bfloat16 | 48 | one-tile owner, n = 3 | 527.1 | 360.7 | flydsl | 681.6 | 664.9 | 128x96x128 |
| 6144x1024x4096 | bfloat16 | 96 | one-tile owner, n = 2-3 | 752.4 | 687.2 | opus | 928.0 | 924.6 | 128x192x128 |
| 6144x2048x4096 | bfloat16 | 192 | data-parallel | 1196.2 | 1233.8 | opus | 1294.0 | 1240.5 | 256x192x64 |
| 6144x129280x4096 | bfloat16 | 12120 | one-tile owner, n = 2-3 | 1514.2 | 1484.1 | asm | 1478.1 | 1412.4 | 256x256x64 |
| 6400x512x4096 | bfloat16 | 50 | one-tile owner, n = 3 | 541.9 | 375.0 | flydsl | 561.1 | 608.1 | 128x128x256 |
| 6400x1024x4096 | bfloat16 | 100 | one-tile owner, n = 2-3 | 777.2 | 710.9 | opus | 891.5 | 785.6 | 256x208x64 |
| 6400x2048x4096 | bfloat16 | 200 | data-parallel | 1218.3 | 1251.9 | torch | 1110.9 | 1119.6 | 256x208x64 |
| 6912x512x4096 | bfloat16 | 54 | one-tile owner, n = 2 | 530.1 | 401.1 | flydsl | 648.6 | 647.4 | 128x128x256 |
| 6912x1024x4096 | bfloat16 | 108 | one-tile owner, n = 2-3 | 856.1 | 762.6 | opus | 938.5 | 709.5 | 256x224x64 |
| 6912x2048x4096 | bfloat16 | 216 | data-parallel | 1268.4 | 1309.6 | torch | 1249.2 | 1254.8 | 256x224x64 |
| 7168x512x4096 | bfloat16 | 56 | one-tile owner, n = 2 | 546.7 | 413.5 | flydsl | 582.0 | 666.5 | 128x128x256 |
| 7168x1024x4096 | bfloat16 | 112 | one-tile owner, n = 2-3 | 894.3 | 789.3 | opus | 1005.8 | 1065.8 | 256x128x64 |
| 7168x2048x4096 | bfloat16 | 224 | data-parallel | 1310.9 | 1338.6 | torch | 1273.0 | 1284.9 | 256x224x64 |
| 7424x512x4096 | bfloat16 | 58 | one-tile owner, n = 2 | 563.0 | 428.5 | flydsl | 649.1 | 661.9 | 128x128x256 |
| 7424x1024x4096 | bfloat16 | 116 | one-tile owner, n = 2-3 | 921.9 | 814.4 | opus | 984.8 | 766.8 | 256x240x64 |
| 7424x2048x4096 | bfloat16 | 232 | data-parallel | 1327.2 | 1362.2 | torch | 1135.5 | 1144.0 | 256x240x64 |
| 7680x512x4096 | bfloat16 | 60 | one-tile owner, n = 2 | 575.8 | 442.5 | flydsl | 571.5 | 680.4 | 128x128x256 |
| 7680x1024x4096 | bfloat16 | 120 | one-tile owner, n = 2-3 | 955.3 | 845.7 | opus | 1038.2 | 771.4 | 256x240x64 |
| 7680x2048x4096 | bfloat16 | 240 | data-parallel | 1348.8 | 1387.4 | torch | 1162.4 | 1170.6 | 256x240x64 |
| 7936x512x4096 | bfloat16 | 62 | one-tile owner, n = 2 | 596.1 | 457.9 | flydsl | 715.7 | 697.6 | 128x128x256 |
| 7936x1024x4096 | bfloat16 | 124 | one-tile owner, n = 2-3 | 984.5 | 860.2 | opus | 1018.0 | 727.9 | 256x256x64 |
| 7936x2048x4096 | bfloat16 | 248 | data-parallel | 1376.0 | 1414.3 | torch | 1369.2 | 1375.9 | 256x256x64 |
| 8192x512x4096 | bfloat16 | 64 | one-tile owner, n = 2 | 611.0 | 468.4 | flydsl | 778.2 | 714.8 | 128x128x256 |
| 8192x1024x4096 | bfloat16 | 128 | one-tile owner, n = 2 | 1003.0 | 879.2 | opus | 1082.9 | 717.8 | 256x256x64 |
| 8192x2048x4096 | bfloat16 | 256 | data-parallel | 1397.4 | 1440.4 | torch | 1401.3 | 1404.4 | 256x256x64 |
| 8192x4096x4096 | bfloat16 | 512 | data-parallel | 1481.2 | 1475.7 | flydsl | 1429.8 | 1451.0 | 256x256x64 |
| 8192x8192x4096 | bfloat16 | 1024 | data-parallel | 1512.3 | 1500.6 | asm | 1475.9 | 1474.4 | 256x256x64 |
| 8192x8704x4096 | bfloat16 | 1088 | one-tile owner, n = 2 | 1450.0 | 1388.8 | asm | 1342.7 | 1334.0 | 256x224x64 |
| 8192x10240x4096 | bfloat16 | 1280 | data-parallel | 1529.1 | 1508.4 | torch | 1471.3 | 1476.9 | 256x256x64 |
| 8192x16384x4096 | bfloat16 | 2048 | data-parallel | 1526.0 | 1502.4 | torch | 1465.5 | 1474.0 | 256x256x64 |
| 8192x129280x4096 | bfloat16 | 16160 | one-tile reduce-scatter, n = 4 | 1568.0 | 1520.8 | flydsl | 1472.1 | 1391.2 | 256x256x64 |
| 10240x512x4096 | bfloat16 | 80 | one-tile owner, n = 2 | 724.9 | 574.6 | torch | 856.1 | 859.5 | 128x160x128 |
| 10240x1024x4096 | bfloat16 | 160 | data-parallel | 1047.3 | 1100.7 | torch | 1153.7 | 1161.4 | 256x160x64 |
| 10240x2048x4096 | bfloat16 | 320 | one-tile owner, n = 2 | 1212.9 | 1118.4 | torch | 1208.1 | 1217.3 | 256x160x64 |
| 10240x129280x4096 | bfloat16 | 20200 | data-parallel | 1532.6 | 1499.8 | asm | 1491.3 | 1431.0 | 256x256x64 |
| 12288x512x4096 | bfloat16 | 96 | one-tile owner, n = 2-3 | 729.3 | 678.8 | opus | 896.4 | 887.2 | 128x192x128 |
| 12288x1024x4096 | bfloat16 | 192 | data-parallel | 1153.5 | 1203.2 | opus | 1275.7 | 1213.0 | 256x192x64 |
| 12288x2048x4096 | bfloat16 | 384 | one-tile owner, n = 2 | 1308.1 | 1268.7 | opus | 1315.0 | 1246.8 | 256x192x64 |
| 12288x129280x4096 | bfloat16 | 24240 | data-parallel | 1538.4 | 1512.9 | asm | 1494.3 | 1416.0 | 256x256x64 |
| 14336x512x4096 | bfloat16 | 112 | one-tile owner, n = 2-3 | 871.9 | 776.6 | opus | 967.4 | 685.1 | 256x224x64 |
| 14336x1024x4096 | bfloat16 | 224 | data-parallel | 1266.9 | 1301.9 | torch | 1244.2 | 1252.4 | 256x224x64 |
| 14336x2048x4096 | bfloat16 | 448 | data-parallel | 1354.1 | 1351.3 | flydsl | 1316.2 | 1287.9 | 256x224x64 |
| 14336x129280x4096 | bfloat16 | 28280 | one-tile owner, n = 2-3 | 1521.3 | 1497.3 | asm | 1489.4 | 1398.9 | 256x256x64 |
| 16384x512x4096 | bfloat16 | 128 | one-tile owner, n = 2 | 951.8 | 859.1 | opus | 1022.6 | 808.2 | 256x256x64 |
| 16384x1024x4096 | bfloat16 | 256 | data-parallel | 1363.8 | 1399.2 | flydsl | 1357.0 | 1330.0 | 256x256x64 |
| 16384x2048x4096 | bfloat16 | 512 | data-parallel | 1459.2 | 1444.0 | torch | 1407.2 | 1413.8 | 256x256x64 |
| 16384x16384x4096 | bfloat16 | 4096 | data-parallel | 1550.5 | 1514.8 | torch | 1493.3 | 1507.0 | 256x256x64 |
| 16384x129280x4096 | bfloat16 | 32320 | one-tile owner, n = 2 | 1481.6 | 1525.9 | asm | 1489.8 | 1369.5 | 256x256x64 |
| 32768x16384x4096 | bfloat16 | 8192 | data-parallel | 1553.5 | 1519.1 | flydsl | 1490.2 | 1509.0 | 256x256x64 |
| 512x7168x4224 | bfloat16 | 56 | one-tile owner, n = 2 | 533.5 | 377.5 | flydsl | 725.6 | 723.4 | 128x128x128 |
| 1024x7168x4224 | bfloat16 | 112 | one-tile owner, n = 2-3 | 910.4 | 742.9 | flydsl | 933.8 | 808.7 | 240x256x64 |
| 10240x7168x4224 | bfloat16 | 1120 | one-tile owner, n = 2-3 | 1451.5 | 1406.1 | opus | 914.6 | 1332.8 | 224x256x64 |
| 512x6144x4608 | bfloat16 | 48 | one-tile owner, n = 3 | 545.1 | 339.2 | flydsl | 686.5 | 625.9 | 192x128x128 |
| 1024x6144x4608 | bfloat16 | 96 | one-tile owner, n = 2-3 | 784.7 | 667.9 | torch | 894.8 | 897.8 | 192x128x128 |
| 2048x6144x4608 | bfloat16 | 192 | data-parallel | 1223.4 | 1240.9 | flydsl | 1192.3 | 1241.3 | 192x256x64 |
| 4096x6144x4608 | bfloat16 | 384 | one-tile owner, n = 2 | 1355.4 | 1283.4 | torch | 1271.0 | 1279.0 | 192x256x64 |
| 8192x6144x4608 | bfloat16 | 768 | data-parallel | 1523.3 | 1503.2 | torch | 1463.0 | 1472.6 | 256x256x64 |
| 16384x6144x4608 | bfloat16 | 1536 | data-parallel | 1542.6 | 1521.7 | torch | 1478.1 | 1485.5 | 256x256x64 |
| 32768x6144x4608 | bfloat16 | 3072 | data-parallel | 1560.3 | 1528.5 | torch | 1483.3 | 1491.9 | 256x256x64 |
| 512x1024x5120 | float32 | 8 | one-tile reduce-scatter, n = 10 | 177.1 | 51.8 | asm | 248.6 | 275.3 | 64x32x256 |
| 512x5120x5120 | bfloat16 | 40 | one-tile owner, n = 3 | 482.7 | 270.1 | torch | 651.1 | 651.9 | 96x128x128 |
| 512x6400x5120 | bfloat16 | 50 | one-tile owner, n = 3 | 588.3 | 339.8 | flydsl | 614.5 | 567.2 | 128x128x256 |
| 512x12800x5120 | bfloat16 | 100 | one-tile owner, n = 2-3 | 794.6 | 659.7 | flydsl | 662.3 | 826.1 | 208x256x64 |
| 512x37888x5120 | bfloat16 | 296 | one-tile owner, n = 3 | 1142.5 | 949.1 | torch | 1056.6 | 1054.8 | 160x256x64 |
| 512x51200x5120 | bfloat16 | 400 | data-parallel | 1201.2 | 1201.8 | flydsl | 1140.3 | 1088.4 | 208x256x64 |
| 768x1024x5120 | float32 | 12 | one-tile reduce-scatter, n = 10 | 243.7 | 78.6 | asm | 313.2 | 313.6 | 64x64x256 |
| 768x37888x5120 | bfloat16 | 444 | data-parallel | 1322.2 | 1320.5 | torch | 1226.4 | 1233.7 | 224x256x64 |
| 1024x1024x5120 | float32 | 16 | one-tile reduce-scatter, n = 8 | 304.7 | 105.9 | asm | 333.9 | 403.4 | 64x64x256 |
| 1024x5120x5120 | bfloat16 | 80 | one-tile owner, n = 2 | 754.6 | 538.9 | torch | 847.5 | 853.8 | 160x128x128 |
| 1024x6400x5120 | bfloat16 | 100 | one-tile owner, n = 2-3 | 816.2 | 670.3 | flydsl | 824.8 | 818.7 | 256x208x64 |
| 1024x12800x5120 | bfloat16 | 200 | data-parallel | 1210.9 | 1236.8 | flydsl | 722.6 | 1053.6 | 256x208x64 |
| 1024x37888x5120 | bfloat16 | 592 | one-tile owner, n = 2 | 1335.3 | 1254.4 | torch | 1058.4 | 1060.4 | 256x208x64 |
| 1024x51200x5120 | bfloat16 | 800 | one-tile reduce-scatter, n = 4 | 1380.9 | 1274.0 | asm | 1224.6 | 1071.8 | 256x208x64 |
| 1536x1024x5120 | float32 | 24 | one-tile reduce-scatter, n = 5 | 374.6 | 158.3 | asm | 436.7 | 512.5 | 64x96x128 |
| 1536x37888x5120 | bfloat16 | 888 | one-tile owner, n = 2-3 | 1391.3 | 1343.5 | torch | 1271.2 | 1275.5 | 224x256x64 |
| 2048x1024x5120 | float32 | 32 | one-tile reduce-scatter, n = 4 | 437.0 | 211.5 | opus | 430.0 | 566.7 | 128x128x128 |
| 2048x5120x5120 | bfloat16 | 160 | data-parallel | 1059.6 | 1080.2 | torch | 1152.8 | 1148.5 | 160x256x64 |
| 2048x6400x5120 | bfloat16 | 200 | data-parallel | 1235.3 | 1269.5 | flydsl | 1133.4 | 1088.7 | 256x208x64 |
| 2048x12800x5120 | bfloat16 | 400 | data-parallel | 1297.8 | 1296.7 | flydsl | 802.9 | 1076.8 | 256x208x64 |
| 2048x37888x5120 | bfloat16 | 1184 | data-parallel | 1454.7 | 1438.9 | torch | 1210.5 | 1215.4 | 240x256x64 |
| 2048x51200x5120 | bfloat16 | 1600 | one-tile owner, n = 2 | 1464.5 | 1413.0 | asm | 212.7 | 1192.6 | 240x256x64 |
| 4096x1024x5120 | float32 | 64 | one-tile owner, n = 2 | 590.7 | 414.7 | opus | 714.7 | 772.2 | 128x128x128 |
| 4096x5120x5120 | bfloat16 | 320 | one-tile owner, n = 2 | 1250.8 | 1098.6 | torch | 1179.0 | 1185.9 | 160x256x64 |
| 4096x6400x5120 | bfloat16 | 400 | data-parallel | 1309.4 | 1306.9 | flydsl | 1247.7 | 1126.3 | 256x208x64 |
| 4096x12800x5120 | bfloat16 | 800 | one-tile reduce-scatter, n = 4 | 1424.9 | 1316.3 | asm | 1291.3 | 1170.7 | 256x208x64 |
| 4096x51200x5120 | bfloat16 | 3200 | one-tile owner, n = 2 | 1529.9 | 1490.4 | asm | 1449.8 | 1381.2 | 256x256x64 |
| 8192x1024x5120 | float32 | 128 | one-tile owner, n = 2 | 999.6 | 800.6 | opus | 1025.0 | 987.2 | 256x128x64 |
| 8192x5120x5120 | bfloat16 | 640 | one-tile owner, n = 2 | 1434.2 | 1366.1 | torch | 1305.9 | 1315.7 | 256x224x64 |
| 8192x6400x5120 | bfloat16 | 800 | one-tile reduce-scatter, n = 4 | 1429.3 | 1317.4 | flydsl | 1255.8 | 1222.7 | 256x208x64 |
| 8192x12800x5120 | bfloat16 | 1600 | one-tile owner, n = 2 | 1486.1 | 1439.2 | asm | 1388.0 | 1189.7 | 256x240x64 |
| 8192x51200x5120 | bfloat16 | 6400 | data-parallel | 1577.9 | 1544.3 | flydsl | 1493.8 | 1408.9 | 256x256x64 |
| 16384x1024x5120 | float32 | 256 | data-parallel | 1305.0 | 1332.5 | opus | 1221.5 | 1239.6 | 256x256x64 |
| 16384x5120x5120 | bfloat16 | 1280 | data-parallel | 1534.7 | 1510.3 | torch | 1474.9 | 1486.8 | 256x256x64 |
| 16384x6400x5120 | bfloat16 | 1600 | one-tile owner, n = 2 | 1477.6 | 1420.4 | flydsl | 1369.6 | 1205.2 | 256x240x64 |
| 16384x12800x5120 | bfloat16 | 3200 | one-tile owner, n = 2 | 1536.8 | 1494.7 | torch | 1459.3 | 1468.0 | 256x256x64 |
| 16384x51200x5120 | bfloat16 | 12800 | data-parallel | 1576.8 | 1545.8 | flydsl | 1494.9 | 1411.6 | 256x256x64 |
| 32768x5120x5120 | bfloat16 | 2560 | data-parallel | 1545.5 | 1519.3 | torch | 1485.5 | 1488.6 | 256x256x64 |
| 32768x6400x5120 | bfloat16 | 3200 | one-tile owner, n = 2 | 1528.3 | 1485.0 | torch | 1465.4 | 1469.1 | 256x256x64 |
| 32768x12800x5120 | bfloat16 | 6400 | data-parallel | 1568.1 | 1533.3 | torch | 1501.5 | 1505.0 | 256x256x64 |
| 32768x51200x5120 | bfloat16 | 25600 | data-parallel | 1576.8 | 1544.5 | flydsl | 1502.6 | 1419.8 | 256x256x64 |
| 512x1280x6144 | bfloat16 | 10 | one-tile reduce-scatter, n = 12 | 237.8 | 67.5 | flydsl | 339.7 | 350.7 | 160x64x128 |
| 512x1536x6144 | bfloat16 | 12 | one-tile reduce-scatter, n = 10 | 252.9 | 81.2 | flydsl | 383.8 | 375.8 | 192x64x128 |
| 512x2048x6144 | float32 | 16 | one-tile reduce-scatter, n = 8 | 333.5 | 105.4 | asm | 342.3 | 423.8 | 64x64x256 |
| 512x2304x6144 | bfloat16 | 18 | one-tile reduce-scatter, n = 7 | 324.6 | 122.0 | flydsl | 440.7 | 373.8 | 192x128x128 |
| 512x2560x6144 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 | 359.3 | 136.1 | torch | 476.8 | 476.8 | 256x128x64 |
| 512x3072x6144 | float32 | 24 | one-tile reduce-scatter, n = 5 | 376.2 | 158.5 | opus | 404.9 | 543.2 | 96x128x128 |
| 512x3072x6144 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 | 381.2 | 163.0 | flydsl | 542.3 | 464.3 | 192x128x128 |
| 512x6144x6144 | bfloat16 | 48 | one-tile owner, n = 3 | 607.3 | 327.9 | flydsl | 687.2 | 688.5 | 192x128x128 |
| 512x9216x6144 | bfloat16 | 72 | one-tile owner, n = 3-4 | 748.8 | 484.5 | flydsl | 609.9 | 593.4 | 256x256x64 |
| 1024x2048x6144 | float32 | 32 | one-tile reduce-scatter, n = 4 | 469.9 | 212.8 | opus | 411.2 | 646.5 | 128x128x128 |
| 1024x2304x6144 | bfloat16 | 36 | one-tile owner, n = 4 | 504.9 | 246.7 | torch | 604.0 | 603.9 | 256x176x64 |
| 1024x2560x6144 | bfloat16 | 40 | one-tile owner, n = 3 | 521.1 | 273.8 | torch | 693.9 | 702.0 | 160x128x128 |
| 1024x3072x6144 | float32 | 48 | one-tile owner, n = 3 | 578.0 | 317.8 | opus | 461.5 | 724.9 | 96x128x128 |
| 1024x6144x6144 | bfloat16 | 96 | one-tile owner, n = 2-3 | 855.5 | 651.5 | torch | 1023.5 | 1026.0 | 192x256x64 |
| 1024x9216x6144 | bfloat16 | 144 | data-parallel | 946.4 | 956.8 | torch | 1066.1 | 1067.3 | 144x256x64 |
| 2048x2048x6144 | float32 | 64 | one-tile owner, n = 3 | 746.9 | 428.3 | opus | 677.8 | 813.5 | 128x128x128 |
| 2048x2304x6144 | bfloat16 | 72 | one-tile owner, n = 3-4 | 827.1 | 503.2 | torch | 845.6 | 849.5 | 128x160x128 |
| 2048x2560x6144 | bfloat16 | 80 | one-tile owner, n = 2 | 832.1 | 555.4 | torch | 909.4 | 916.7 | 160x128x128 |
| 2048x3072x6144 | float32 | 96 | one-tile owner, n = 2-3 | 884.8 | 644.2 | opus | 697.2 | 929.5 | 192x256x64 |
| 2048x6144x6144 | bfloat16 | 192 | data-parallel | 1218.3 | 1248.7 | torch | 1267.2 | 1275.5 | 192x256x64 |
| 2048x9216x6144 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 | 1264.5 | 1005.0 | torch | 1101.5 | 1104.5 | 144x256x64 |
| 4096x2048x6144 | float32 | 128 | one-tile owner, n = 2 | 1089.1 | 835.1 | opus | 1030.8 | 859.4 | 256x256x64 |
| 4096x2304x6144 | bfloat16 | 144 | data-parallel | 964.5 | 984.4 | flydsl | 687.7 | 1095.0 | 144x256x64 |
| 4096x2560x6144 | bfloat16 | 160 | data-parallel | 1058.3 | 1086.5 | flydsl | 639.2 | 1180.5 | 160x256x64 |
| 4096x3072x6144 | float32 | 192 | data-parallel | 1160.2 | 1184.8 | asm | 1148.0 | 1245.4 | 192x256x64 |
| 4096x6144x6144 | bfloat16 | 384 | one-tile owner, n = 2 | 1400.1 | 1298.0 | torch | 1301.4 | 1310.0 | 192x256x64 |
| 4096x9216x6144 | bfloat16 | 576 | one-tile owner, n = 3 | 1423.5 | 1300.4 | torch | 1307.1 | 1317.7 | 192x256x64 |
| 8192x2048x6144 | float32 | 256 | data-parallel | 1366.6 | 1391.4 | opus | 1275.0 | 1293.8 | 256x256x64 |
| 8192x2304x6144 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 | 1250.2 | 999.9 | torch | 1105.8 | 1109.5 | 144x256x64 |
| 8192x2560x6144 | bfloat16 | 320 | one-tile owner, n = 3 | 1305.8 | 1108.4 | asm | 409.1 | 1202.6 | 160x256x64 |
| 8192x3072x6144 | float32 | 384 | one-tile owner, n = 2 | 1314.6 | 1209.3 | asm | 1174.9 | 1233.4 | 192x256x64 |
| 8192x6144x6144 | bfloat16 | 768 | data-parallel | 1547.2 | 1534.0 | flydsl | 1489.5 | 1497.4 | 256x256x64 |
| 8192x9216x6144 | bfloat16 | 1152 | one-tile owner, n = 2 | 1516.4 | 1447.7 | asm | 1425.1 | 1204.5 | 256x240x64 |
| 16384x2048x6144 | float32 | 512 | data-parallel | 1398.8 | 1407.4 | opus | 1249.1 | 1322.1 | 256x256x64 |
| 16384x2304x6144 | bfloat16 | 576 | one-tile owner, n = 3 | 1394.0 | 1280.3 | flydsl | 974.8 | 1311.1 | 192x256x64 |
| 16384x2560x6144 | bfloat16 | 640 | one-tile owner, n = 2 | 1436.7 | 1355.5 | flydsl | 1314.2 | 1312.7 | 256x224x64 |
| 16384x3072x6144 | float32 | 768 | data-parallel | 1417.6 | 1416.9 | asm | 1379.4 | 1338.4 | 256x256x64 |
| 16384x6144x6144 | bfloat16 | 1536 | data-parallel | 1556.6 | 1543.9 | asm | 1512.7 | 1510.1 | 256x256x64 |
| 16384x9216x6144 | bfloat16 | 2304 | data-parallel | 1567.7 | 1543.3 | torch | 1507.6 | 1510.3 | 256x256x64 |
| 32768x2304x6144 | bfloat16 | 1152 | one-tile owner, n = 2 | 1469.3 | 1410.9 | flydsl | 651.9 | 1195.3 | 256x240x64 |
| 32768x2560x6144 | bfloat16 | 1280 | data-parallel | 1514.5 | 1508.0 | flydsl | 1462.0 | 1490.3 | 256x256x64 |
| 32768x6144x6144 | bfloat16 | 3072 | data-parallel | 1573.0 | 1550.6 | torch | 1511.2 | 1521.4 | 256x256x64 |
| 32768x9216x6144 | bfloat16 | 4608 | data-parallel | 1574.4 | 1553.5 | torch | 1532.1 | 1535.4 | 256x256x64 |
| 512x5120x6400 | bfloat16 | 40 | one-tile owner, n = 4 | 560.7 | 281.1 | flydsl | 542.5 | 691.3 | 160x128x128 |
| 1024x5120x6400 | bfloat16 | 80 | one-tile owner, n = 2 | 815.9 | 562.1 | flydsl | 600.5 | 892.8 | 160x128x128 |
| 2048x5120x6400 | bfloat16 | 160 | data-parallel | 1089.7 | 1106.9 | flydsl | 744.5 | 1204.4 | 160x256x64 |
| 4096x5120x6400 | bfloat16 | 320 | one-tile owner, n = 3 | 1346.8 | 1138.1 | torch | 1226.6 | 1234.4 | 160x256x64 |
| 8192x5120x6400 | bfloat16 | 640 | one-tile owner, n = 2 | 1481.5 | 1395.7 | flydsl | 1355.3 | 1350.0 | 256x224x64 |
| 16384x5120x6400 | bfloat16 | 1280 | data-parallel | 1560.3 | 1545.1 | torch | 1502.5 | 1506.4 | 256x256x64 |
| 32768x5120x6400 | bfloat16 | 2560 | data-parallel | 1576.1 | 1550.5 | torch | 1504.3 | 1513.6 | 256x256x64 |
| 512x16384x6656 | bfloat16 | 128 | one-tile owner, n = 2 | 1106.6 | 880.6 | flydsl | 839.6 | 905.3 | 256x256x64 |
| 1024x16384x6656 | bfloat16 | 256 | data-parallel | 1428.2 | 1441.2 | asm | 363.3 | 1383.0 | 256x256x64 |
| 2048x16384x6656 | bfloat16 | 512 | data-parallel | 1526.5 | 1507.4 | asm | 1471.5 | 1448.2 | 256x256x64 |
| 4096x16384x6656 | bfloat16 | 1024 | data-parallel | 1557.1 | 1541.6 | torch | 1480.5 | 1484.5 | 256x256x64 |
| 8192x16384x6656 | bfloat16 | 2048 | data-parallel | 1586.4 | 1566.7 | asm | 1527.0 | 1505.5 | 256x256x64 |
| 16384x16384x6656 | bfloat16 | 4096 | data-parallel | 1599.0 | 1568.8 | asm | 1541.4 | 1544.3 | 256x256x64 |
| 32768x16384x6656 | bfloat16 | 8192 | data-parallel | 1606.7 | 1577.0 | asm | 292.5 | 1549.8 | 256x256x64 |
| 512x1024x7168 | bfloat16 | 8 | one-tile reduce-scatter, n = 14 | 235.1 | 56.7 | torch | 387.3 | 387.9 | 128x128x64 |
| 512x1536x7168 | bfloat16 | 12 | one-tile reduce-scatter, n = 10 | 289.7 | 86.3 | torch | 411.9 | 411.1 | 192x64x128 |
| 512x2048x7168 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 | 376.7 | 114.5 | torch | 545.5 | 545.5 | 128x128x128 |
| 512x3584x7168 | bfloat16 | 28 | one-tile reduce-scatter, n = 4 | 461.6 | 203.3 | flydsl | 556.1 | 538.3 | 128x128x256 |
| 512x6144x7168 | bfloat16 | 48 | one-tile owner, n = 3 | 637.2 | 350.8 | flydsl | 715.2 | 723.6 | 192x128x128 |
| 512x8192x7168 | bfloat16 | 64 | one-tile owner, n = 3 | 802.5 | 462.1 | flydsl | 709.3 | 735.2 | 128x128x256 |
| 512x8448x7168 | bfloat16 | 66 | one-tile owner, n = 3-4 | 737.7 | 476.4 | flydsl | 672.4 | 469.6 | 256x256x64 |
| 512x20480x7168 | bfloat16 | 160 | data-parallel | 1045.0 | 1066.1 | flydsl | 1032.9 | 1155.0 | 160x256x64 |
| 768x1024x7168 | bfloat16 | 12 | one-tile reduce-scatter, n = 10 | 293.3 | 87.0 | torch | 408.7 | 411.8 | 256x96x64 |
| 768x2048x7168 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 | 441.0 | 174.0 | flydsl | 560.5 | 570.7 | 256x128x64 |
| 1024x512x7168 | bfloat16 | 8 | one-tile reduce-scatter, n = 14 | 234.5 | 57.4 | flydsl | 336.0 | 292.7 | 256x64x64 |
| 1024x1024x7168 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 | 378.8 | 116.8 | torch | 561.8 | 551.5 | 128x128x128 |
| 1024x1536x7168 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 | 439.2 | 173.7 | flydsl | 564.3 | 570.9 | 256x128x64 |
| 1024x2048x7168 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 518.5 | 232.2 | torch | 735.0 | 734.8 | 256x128x64 |
| 1024x3584x7168 | bfloat16 | 56 | one-tile owner, n = 3 | 732.4 | 408.7 | flydsl | 777.8 | 683.5 | 128x128x256 |
| 1024x6144x7168 | bfloat16 | 96 | one-tile owner, n = 2-3 | 913.8 | 695.0 | torch | 981.4 | 994.1 | 192x256x64 |
| 1024x8192x7168 | bfloat16 | 128 | one-tile owner, n = 2 | 1158.5 | 901.2 | flydsl | 1016.1 | 942.5 | 256x256x64 |
| 1024x8448x7168 | bfloat16 | 132 | data-parallel | 915.8 | 930.7 | torch | 1039.1 | 1040.8 | 144x256x64 |
| 1024x20480x7168 | bfloat16 | 320 | one-tile owner, n = 3 | 1313.0 | 1121.1 | flydsl | 746.9 | 1205.7 | 160x256x64 |
| 1280x512x7168 | bfloat16 | 10 | one-tile reduce-scatter, n = 12 | 260.6 | 72.3 | torch | 365.2 | 362.5 | 256x80x64 |
| 1280x1024x7168 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 | 394.4 | 146.3 | torch | 536.3 | 549.2 | 256x80x128 |
| 1280x2048x7168 | bfloat16 | 40 | one-tile owner, n = 4 | 612.7 | 293.0 | torch | 747.1 | 737.7 | 128x160x128 |
| 1536x512x7168 | bfloat16 | 12 | one-tile reduce-scatter, n = 10 | 291.8 | 87.3 | torch | 404.0 | 407.4 | 256x96x64 |
| 1536x1024x7168 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 | 442.3 | 176.3 | torch | 647.5 | 652.2 | 128x96x128 |
| 1536x2048x7168 | bfloat16 | 48 | one-tile owner, n = 3 | 652.7 | 355.9 | torch | 842.4 | 836.9 | 128x192x64 |
| 1792x512x7168 | bfloat16 | 14 | one-tile reduce-scatter, n = 9 | 317.2 | 102.4 | torch | 442.0 | 441.6 | 256x112x64 |
| 1792x1024x7168 | bfloat16 | 28 | one-tile reduce-scatter, n = 4 | 468.7 | 207.4 | flydsl | 600.9 | 610.9 | 256x128x64 |
| 1792x2048x7168 | bfloat16 | 56 | one-tile owner, n = 3 | 742.4 | 414.3 | flydsl | 777.7 | 685.6 | 128x128x256 |
| 2048x512x7168 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 | 380.5 | 116.2 | torch | 507.0 | 510.3 | 128x128x64 |
| 2048x1024x7168 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 523.8 | 237.8 | torch | 707.5 | 712.1 | 128x128x128 |
| 2048x2048x7168 | bfloat16 | 64 | one-tile owner, n = 3 | 828.4 | 473.4 | flydsl | 874.9 | 772.9 | 128x128x256 |
| 2048x3584x7168 | bfloat16 | 112 | one-tile owner, n = 2-3 | 1086.6 | 823.2 | opus | 1004.5 | 1023.8 | 224x256x64 |
| 2048x8192x7168 | bfloat16 | 256 | data-parallel | 1512.3 | 1519.7 | asm | 1478.5 | 1448.5 | 256x256x64 |
| 2048x20480x7168 | bfloat16 | 640 | one-tile owner, n = 2 | 1484.7 | 1390.7 | flydsl | 832.4 | 1284.8 | 224x256x64 |
| 4096x512x7168 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 518.5 | 233.9 | flydsl | 626.4 | 603.8 | 128x128x256 |
| 4096x1024x7168 | bfloat16 | 64 | one-tile owner, n = 3 | 827.0 | 469.7 | torch | 913.8 | 920.4 | 256x128x64 |
| 4096x2048x7168 | bfloat16 | 128 | one-tile owner, n = 2 | 1219.7 | 911.8 | torch | 1181.6 | 1176.6 | 256x128x64 |
| 4096x3584x7168 | bfloat16 | 224 | data-parallel | 1408.6 | 1407.2 | flydsl | 1334.4 | 1333.6 | 224x256x64 |
| 4096x8192x7168 | bfloat16 | 512 | data-parallel | 1552.4 | 1539.3 | flydsl | 1496.7 | 1489.7 | 256x256x64 |
| 4096x20480x7168 | bfloat16 | 1280 | data-parallel | 1576.1 | 1565.9 | flydsl | 1516.6 | 1462.6 | 256x256x64 |
| 6144x512x7168 | bfloat16 | 48 | one-tile owner, n = 3 | 632.2 | 347.2 | flydsl | 723.9 | 657.7 | 128x192x128 |
| 6144x1024x7168 | bfloat16 | 96 | one-tile owner, n = 2-3 | 906.5 | 690.4 | opus | 966.7 | 906.0 | 256x192x64 |
| 6144x2048x7168 | bfloat16 | 192 | data-parallel | 1267.6 | 1277.7 | opus | 1366.8 | 1303.6 | 256x192x64 |
| 8192x512x7168 | bfloat16 | 64 | one-tile owner, n = 3 | 808.7 | 462.0 | flydsl | 845.4 | 756.9 | 128x128x256 |
| 8192x1024x7168 | bfloat16 | 128 | one-tile owner, n = 2 | 1163.0 | 900.3 | torch | 1097.8 | 1106.5 | 256x128x64 |
| 8192x2048x7168 | bfloat16 | 256 | data-parallel | 1497.4 | 1497.1 | torch | 1419.7 | 1421.4 | 256x256x64 |
| 8192x3584x7168 | bfloat16 | 448 | data-parallel | 1438.5 | 1425.4 | flydsl | 1387.6 | 1353.5 | 224x256x64 |
| 8192x8192x7168 | bfloat16 | 1024 | data-parallel | 1570.6 | 1555.5 | flydsl | 1511.9 | 1514.5 | 256x256x64 |
| 8192x20480x7168 | bfloat16 | 2560 | data-parallel | 1608.4 | 1584.5 | asm | 1535.5 | 1478.9 | 256x256x64 |
| 10240x512x7168 | bfloat16 | 80 | one-tile owner, n = 3-4 | 909.0 | 566.7 | torch | 825.9 | 823.1 | 256x160x64 |
| 10240x1024x7168 | bfloat16 | 160 | data-parallel | 1090.8 | 1114.7 | torch | 1211.5 | 1216.1 | 256x160x64 |
| 10240x2048x7168 | bfloat16 | 320 | one-tile owner, n = 3 | 1347.2 | 1146.8 | torch | 1256.9 | 1264.2 | 256x160x64 |
| 10240x8448x7168 | bfloat16 | 1320 | one-tile owner, n = 4 | 1498.3 | 1397.3 | asm | 1401.2 | 1365.0 | 256x224x64 |
| 10240x20480x7168 | bfloat16 | 3200 | one-tile owner, n = 2 | 1525.9 | 1475.8 | asm | 1508.7 | 1468.8 | 256x256x64 |
| 12288x512x7168 | bfloat16 | 96 | one-tile owner, n = 2-3 | 876.4 | 671.7 | opus | 930.9 | 866.4 | 256x192x64 |
| 12288x1024x7168 | bfloat16 | 192 | data-parallel | 1235.1 | 1237.8 | opus | 1344.1 | 1284.4 | 256x192x64 |
| 12288x2048x7168 | bfloat16 | 384 | one-tile owner, n = 2 | 1425.1 | 1303.4 | opus | 1387.8 | 1329.3 | 256x192x64 |
| 14336x512x7168 | bfloat16 | 112 | one-tile owner, n = 2-3 | 1031.8 | 779.5 | opus | 1001.3 | 867.8 | 256x224x64 |
| 14336x1024x7168 | bfloat16 | 224 | data-parallel | 1359.4 | 1364.6 | torch | 1325.4 | 1331.2 | 256x224x64 |
| 14336x2048x7168 | bfloat16 | 448 | data-parallel | 1402.8 | 1403.4 | flydsl | 1371.1 | 1351.5 | 256x224x64 |
| 16384x512x7168 | bfloat16 | 128 | one-tile owner, n = 2 | 1139.9 | 887.6 | opus | 1080.5 | 1038.4 | 256x256x64 |
| 16384x1024x7168 | bfloat16 | 256 | data-parallel | 1473.0 | 1479.0 | torch | 1419.7 | 1427.7 | 256x256x64 |
| 16384x2048x7168 | bfloat16 | 512 | data-parallel | 1529.8 | 1513.2 | torch | 1479.3 | 1484.5 | 256x256x64 |
| 16384x3584x7168 | bfloat16 | 896 | one-tile owner, n = 2 | 1507.9 | 1423.5 | flydsl | 1385.6 | 1385.1 | 224x256x64 |
| 16384x8192x7168 | bfloat16 | 2048 | data-parallel | 1587.3 | 1565.7 | torch | 1510.7 | 1532.3 | 256x256x64 |
| 16384x20480x7168 | bfloat16 | 5120 | data-parallel | 1613.8 | 1585.5 | flydsl | 1541.1 | 1508.1 | 256x256x64 |
| 32768x512x7168 | bfloat16 | 256 | data-parallel | 1382.0 | 1386.1 | torch | 1392.2 | 1393.9 | 256x256x64 |
| 32768x1024x7168 | bfloat16 | 512 | data-parallel | 1496.7 | 1492.7 | torch | 1430.0 | 1435.1 | 256x256x64 |
| 32768x2048x7168 | bfloat16 | 1024 | data-parallel | 1546.9 | 1528.4 | torch | 1483.5 | 1484.8 | 256x256x64 |
| 32768x3584x7168 | bfloat16 | 1792 | data-parallel | 1554.5 | 1535.8 | torch | 1500.2 | 1511.1 | 256x256x64 |
| 32768x8192x7168 | bfloat16 | 4096 | data-parallel | 1600.7 | 1572.4 | torch | 1555.5 | 1558.0 | 256x256x64 |
| 32768x20480x7168 | bfloat16 | 10240 | data-parallel | 1617.9 | 1586.8 | asm | 1557.0 | 1480.7 | 256x256x64 |
| 512x1280x8192 | bfloat16 | 10 | one-tile reduce-scatter, n = 12 | 287.6 | 78.5 | torch | 421.8 | 423.2 | 160x64x128 |
| 512x2560x8192 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 | 440.9 | 156.2 | torch | 558.6 | 564.0 | 256x128x64 |
| 512x4096x8192 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 568.0 | 249.4 | flydsl | 675.4 | 647.6 | 128x128x256 |
| 512x4608x8192 | bfloat16 | 36 | one-tile owner, n = 4 | 605.1 | 281.0 | torch | 721.1 | 721.4 | 160x128x128 |
| 512x7168x8192 | bfloat16 | 56 | one-tile owner, n = 3 | 807.8 | 428.9 | flydsl | 795.3 | 746.9 | 256x176x64 |
| 512x8192x8192 | bfloat16 | 64 | one-tile owner, n = 3 | 887.0 | 484.5 | flydsl | 679.7 | 770.9 | 128x128x256 |
| 512x10240x8192 | bfloat16 | 80 | one-tile owner, n = 3-4 | 982.7 | 602.9 | flydsl | 624.9 | 571.6 | 256x256x64 |
| 512x14336x8192 | bfloat16 | 112 | one-tile owner, n = 2-3 | 1077.7 | 835.5 | torch | 1035.6 | 1044.8 | 224x256x64 |
| 512x16384x8192 | bfloat16 | 128 | one-tile owner, n = 2 | 1164.1 | 903.3 | flydsl | 655.9 | 925.0 | 256x256x64 |
| 512x57344x8192 | bfloat16 | 448 | data-parallel | 1346.6 | 1333.2 | flydsl | 1294.9 | 1270.2 | 224x256x64 |
| 768x4096x8192 | bfloat16 | 48 | one-tile owner, n = 4 | 777.8 | 375.3 | flydsl | 744.2 | 728.6 | 128x192x128 |
| 1024x512x8192 | bfloat16 | 8 | one-tile reduce-scatter, n = 16 | 278.7 | 62.8 | torch | 333.7 | 333.0 | 256x64x64 |
| 1024x1280x8192 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 | 446.4 | 157.6 | torch | 582.7 | 581.4 | 256x128x64 |
| 1024x2560x8192 | bfloat16 | 40 | one-tile owner, n = 4 | 668.9 | 315.6 | torch | 788.1 | 796.3 | 160x128x128 |
| 1024x4096x8192 | bfloat16 | 64 | one-tile owner, n = 3 | 898.1 | 494.8 | flydsl | 896.7 | 802.9 | 128x128x256 |
| 1024x4608x8192 | bfloat16 | 72 | one-tile owner, n = 3-4 | 893.4 | 554.8 | torch | 941.4 | 940.0 | 160x256x64 |
| 1024x7168x8192 | bfloat16 | 112 | one-tile owner, n = 2-3 | 1115.4 | 832.5 | flydsl | 737.3 | 1063.7 | 224x256x64 |
| 1024x8192x8192 | bfloat16 | 128 | one-tile owner, n = 2 | 1210.5 | 933.8 | flydsl | 916.5 | 996.3 | 256x256x64 |
| 1024x10240x8192 | bfloat16 | 160 | data-parallel | 1136.6 | 1161.3 | flydsl | 828.4 | 1218.8 | 160x256x64 |
| 1024x14336x8192 | bfloat16 | 224 | data-parallel | 1379.1 | 1399.9 | flydsl | 1357.8 | 1308.5 | 224x256x64 |
| 1024x16384x8192 | bfloat16 | 256 | data-parallel | 1473.3 | 1480.6 | torch | 1447.0 | 1451.0 | 256x256x64 |
| 1024x57344x8192 | bfloat16 | 896 | one-tile owner, n = 2 | 1499.1 | 1420.8 | asm | 1382.8 | 1328.6 | 224x256x64 |
| 2048x512x8192 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 | 424.8 | 126.3 | torch | 507.3 | 506.3 | 256x128x64 |
| 2048x1280x8192 | bfloat16 | 40 | one-tile owner, n = 4 | 667.5 | 317.9 | torch | 699.6 | 703.6 | 320x192x64 |
| 2048x2560x8192 | bfloat16 | 80 | one-tile owner, n = 3-4 | 1021.3 | 616.3 | torch | 988.3 | 978.3 | 160x256x64 |
| 2048x4096x8192 | bfloat16 | 128 | one-tile owner, n = 2 | 1245.3 | 936.1 | opus | 1139.5 | 1019.7 | 256x256x64 |
| 2048x4608x8192 | bfloat16 | 144 | data-parallel | 1060.1 | 1048.4 | torch | 1161.1 | 1163.1 | 144x256x64 |
| 2048x7168x8192 | bfloat16 | 224 | data-parallel | 1413.1 | 1423.2 | asm | 1377.3 | 1337.4 | 224x256x64 |
| 2048x8192x8192 | bfloat16 | 256 | data-parallel | 1505.6 | 1519.7 | torch | 1466.5 | 1465.7 | 256x256x64 |
| 2048x10240x8192 | bfloat16 | 320 | one-tile owner, n = 3 | 1389.3 | 1169.6 | flydsl | 933.0 | 1254.6 | 160x256x64 |
| 2048x14336x8192 | bfloat16 | 448 | data-parallel | 1450.9 | 1438.6 | flydsl | 1404.5 | 1350.4 | 224x256x64 |
| 2048x16384x8192 | bfloat16 | 512 | data-parallel | 1550.1 | 1530.0 | asm | 1498.5 | 1501.1 | 256x256x64 |
| 2048x57344x8192 | bfloat16 | 1792 | data-parallel | 1592.4 | 1574.0 | flydsl | 1524.9 | 1491.2 | 256x256x64 |
| 4096x512x8192 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 564.8 | 249.4 | flydsl | 643.1 | 659.2 | 128x128x256 |
| 4096x1280x8192 | bfloat16 | 80 | one-tile owner, n = 3-4 | 1010.0 | 612.8 | torch | 985.8 | 983.5 | 160x256x64 |
| 4096x2560x8192 | bfloat16 | 160 | data-parallel | 1145.2 | 1155.6 | torch | 1242.6 | 1250.4 | 160x256x64 |
| 4096x4096x8192 | bfloat16 | 256 | data-parallel | 1521.0 | 1539.3 | torch | 1472.9 | 1480.5 | 256x256x64 |
| 4096x4608x8192 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 | 1362.3 | 1076.1 | torch | 1179.6 | 1186.3 | 144x256x64 |
| 4096x7168x8192 | bfloat16 | 448 | data-parallel | 1464.4 | 1450.7 | flydsl | 1412.4 | 1369.9 | 224x256x64 |
| 4096x8192x8192 | bfloat16 | 512 | data-parallel | 1555.4 | 1546.9 | asm | 1526.7 | 1503.9 | 256x256x64 |
| 4096x10240x8192 | bfloat16 | 640 | one-tile owner, n = 2 | 1518.5 | 1416.7 | flydsl | 1389.3 | 1337.9 | 224x256x64 |
| 4096x14336x8192 | bfloat16 | 896 | one-tile owner, n = 2 | 1544.0 | 1459.7 | flydsl | 1420.4 | 1360.2 | 224x256x64 |
| 4096x16384x8192 | bfloat16 | 1024 | data-parallel | 1584.4 | 1571.4 | flydsl | 1519.8 | 1520.4 | 256x256x64 |
| 4096x57344x8192 | bfloat16 | 3584 | data-parallel | 1609.8 | 1594.1 | flydsl | 1546.8 | 1507.8 | 256x256x64 |
| 8192x512x8192 | bfloat16 | 64 | one-tile owner, n = 3 | 874.9 | 479.5 | flydsl | 863.2 | 793.4 | 128x128x256 |
| 8192x1280x8192 | bfloat16 | 160 | data-parallel | 1126.5 | 1129.7 | torch | 1227.8 | 1237.0 | 160x256x64 |
| 8192x2560x8192 | bfloat16 | 320 | one-tile owner, n = 3 | 1377.7 | 1156.1 | torch | 1259.5 | 1265.6 | 160x256x64 |
| 8192x4096x8192 | bfloat16 | 512 | data-parallel | 1564.5 | 1559.0 | torch | 1541.3 | 1552.0 | 256x256x64 |
| 8192x4608x8192 | bfloat16 | 576 | one-tile owner, n = 3 | 1481.6 | 1336.4 | torch | 1347.4 | 1355.4 | 192x256x64 |
| 8192x7168x8192 | bfloat16 | 896 | one-tile owner, n = 2 | 1548.2 | 1458.5 | asm | 1427.3 | 1402.7 | 224x256x64 |
| 8192x8192x8192 | bfloat16 | 1024 | data-parallel | 1586.5 | 1579.2 | asm | 1562.3 | 1517.4 | 256x256x64 |
| 8192x10240x8192 | bfloat16 | 1280 | data-parallel | 1606.4 | 1586.4 | flydsl | 1538.4 | 1564.6 | 256x256x64 |
| 8192x14336x8192 | bfloat16 | 1792 | data-parallel | 1610.3 | 1593.0 | flydsl | 1549.8 | 1539.0 | 256x256x64 |
| 8192x16384x8192 | bfloat16 | 2048 | data-parallel | 1605.9 | 1584.1 | asm | 1553.9 | 1538.8 | 256x256x64 |
| 8192x57344x8192 | bfloat16 | 7168 | data-parallel | 1613.1 | 1605.7 | asm | 1544.8 | 1500.4 | 256x256x64 |
| 16384x512x8192 | bfloat16 | 128 | one-tile owner, n = 2 | 1137.3 | 899.2 | torch | 1065.0 | 1072.0 | 256x256x64 |
| 16384x1280x8192 | bfloat16 | 320 | one-tile owner, n = 3 | 1348.7 | 1152.6 | torch | 1251.6 | 1257.1 | 160x256x64 |
| 16384x2560x8192 | bfloat16 | 640 | one-tile owner, n = 2 | 1486.3 | 1382.4 | flydsl | 1345.1 | 1310.3 | 256x224x64 |
| 16384x4608x8192 | bfloat16 | 1152 | one-tile owner, n = 2 | 1538.0 | 1466.6 | flydsl | 1431.9 | 1206.6 | 256x240x64 |
| 16384x7168x8192 | bfloat16 | 1792 | data-parallel | 1587.5 | 1576.6 | torch | 1525.4 | 1541.2 | 256x256x64 |
| 16384x8192x8192 | bfloat16 | 2048 | data-parallel | 1602.8 | 1585.5 | torch | 1549.1 | 1571.9 | 256x256x64 |
| 16384x10240x8192 | bfloat16 | 2560 | data-parallel | 1613.2 | 1590.1 | torch | 1575.2 | 1578.5 | 256x256x64 |
| 16384x14336x8192 | bfloat16 | 3584 | data-parallel | 1616.1 | 1594.3 | asm | 1562.6 | 1551.4 | 256x256x64 |
| 16384x16384x8192 | bfloat16 | 4096 | data-parallel | 1616.2 | 1598.7 | asm | 1566.6 | 1538.0 | 256x256x64 |
| 16384x57344x8192 | bfloat16 | 14336 | data-parallel | 1509.7 | 1596.8 | asm | 1514.6 | 1406.8 | 256x256x64 |
| 32768x1280x8192 | bfloat16 | 640 | one-tile owner, n = 2 | 1450.0 | 1348.6 | flydsl | 1314.9 | 1336.6 | 256x224x64 |
| 32768x2560x8192 | bfloat16 | 1280 | data-parallel | 1549.1 | 1533.6 | torch | 1499.0 | 1500.3 | 256x256x64 |
| 32768x7168x8192 | bfloat16 | 3584 | data-parallel | 1610.1 | 1581.9 | torch | 1571.8 | 1586.5 | 256x256x64 |
| 32768x8192x8192 | bfloat16 | 4096 | data-parallel | 1619.6 | 1596.6 | flydsl | 1565.8 | 1595.7 | 256x256x64 |
| 32768x10240x8192 | bfloat16 | 5120 | data-parallel | 1619.8 | 1594.6 | torch | 1586.0 | 1593.5 | 256x256x64 |
| 32768x14336x8192 | bfloat16 | 7168 | data-parallel | 1614.2 | 1597.7 | asm | 1567.4 | 1537.3 | 256x256x64 |
| 32768x16384x8192 | bfloat16 | 8192 | data-parallel | 1608.5 | 1600.7 | asm | 1569.6 | 1528.1 | 256x256x64 |
| 32768x57344x8192 | bfloat16 | 28672 | data-parallel | 1540.3 | 1584.9 | asm | 1567.9 | 1430.9 | 256x256x64 |
| 512x6144x12288 | bfloat16 | 48 | one-tile owner, n = 4 | 855.0 | 352.1 | torch | 780.7 | 789.7 | 192x128x128 |
| 1024x6144x12288 | bfloat16 | 96 | one-tile owner, n = 2-3 | 1044.9 | 701.8 | torch | 1093.5 | 1088.7 | 192x256x64 |
| 2048x6144x12288 | bfloat16 | 192 | data-parallel | 1307.6 | 1315.6 | torch | 1323.3 | 1330.5 | 192x256x64 |
| 4096x6144x12288 | bfloat16 | 384 | one-tile owner, n = 2 | 1507.2 | 1349.5 | torch | 1335.8 | 1341.8 | 192x256x64 |
| 8192x6144x12288 | bfloat16 | 768 | data-parallel | 1568.8 | 1555.1 | torch | 1515.1 | 1527.1 | 256x256x64 |
| 16384x6144x12288 | bfloat16 | 1536 | data-parallel | 1581.0 | 1566.4 | torch | 1533.3 | 1530.2 | 256x256x64 |
| 32768x6144x12288 | bfloat16 | 3072 | data-parallel | 1592.3 | 1571.7 | asm | 1553.0 | 1548.8 | 256x256x64 |
| 512x16384x13312 | bfloat16 | 128 | one-tile owner, n = 2 | 1270.3 | 926.6 | flydsl | 704.3 | 1100.0 | 256x256x64 |
| 1024x16384x13312 | bfloat16 | 256 | data-parallel | 1526.8 | 1529.6 | asm | 1488.2 | 1495.8 | 256x256x64 |
| 2048x16384x13312 | bfloat16 | 512 | data-parallel | 1582.3 | 1563.8 | flydsl | 1012.0 | 1511.4 | 256x256x64 |
| 4096x16384x13312 | bfloat16 | 1024 | data-parallel | 1620.2 | 1603.2 | flydsl | 1552.1 | 1522.8 | 256x256x64 |
| 8192x16384x13312 | bfloat16 | 2048 | data-parallel | 1606.1 | 1584.0 | flydsl | 1532.8 | 1525.5 | 256x256x64 |
| 16384x16384x13312 | bfloat16 | 4096 | data-parallel | 1609.5 | 1586.2 | asm | 1543.1 | 1514.9 | 256x256x64 |
| 32768x16384x13312 | bfloat16 | 8192 | data-parallel | 1608.0 | 1581.3 | flydsl | 1528.9 | 1449.7 | 256x256x64 |
| 512x2304x16384 | bfloat16 | 18 | one-tile reduce-scatter, n = 7 | 545.4 | 139.2 | torch | 704.6 | 705.9 | 256x128x64 |
| 512x4608x16384 | bfloat16 | 36 | one-tile owner, n = 6 | 828.5 | 279.2 | flydsl | 504.1 | 797.0 | 160x128x128 |
| 512x9216x16384 | bfloat16 | 72 | one-tile owner, n = 3-4 | 1047.3 | 549.1 | flydsl | 638.1 | 760.6 | 256x256x64 |
| 512x13312x16384 | bfloat16 | 104 | one-tile owner, n = 2-3 | 1135.7 | 758.4 | flydsl | 545.8 | 1083.0 | 208x256x64 |
| 512x26624x16384 | bfloat16 | 208 | data-parallel | 1289.2 | 1304.6 | flydsl | 768.6 | 1188.6 | 208x256x64 |
| 512x53248x16384 | bfloat16 | 416 | data-parallel | 1311.6 | 1302.3 | flydsl | 646.9 | 1207.8 | 208x256x64 |
| 1024x2304x16384 | bfloat16 | 36 | one-tile owner, n = 6 | 843.4 | 277.3 | torch | 872.9 | 877.0 | 256x256x64 |
| 1024x4608x16384 | bfloat16 | 72 | one-tile owner, n = 3-4 | 1095.9 | 556.7 | flydsl | 650.2 | 1031.8 | 144x256x64 |
| 1024x9216x16384 | bfloat16 | 144 | data-parallel | 1018.2 | 1031.7 | flydsl | 677.8 | 1154.4 | 144x256x64 |
| 1024x13312x16384 | bfloat16 | 208 | data-parallel | 1345.1 | 1351.9 | asm | 1333.6 | 1243.9 | 208x256x64 |
| 1024x26624x16384 | bfloat16 | 416 | data-parallel | 1384.8 | 1376.7 | flydsl | 948.8 | 1256.7 | 208x256x64 |
| 1024x53248x16384 | bfloat16 | 832 | one-tile owner, n = 4 | 1535.5 | 1373.2 | triton | 990.1 | 1260.1 | 208x256x64 |
| 2048x2304x16384 | bfloat16 | 72 | one-tile owner, n = 3-4 | 1111.9 | 552.8 | torch | 876.0 | 856.5 | 256x256x64 |
| 2048x4608x16384 | bfloat16 | 144 | data-parallel | 1062.0 | 1063.6 | flydsl | 717.6 | 1190.3 | 144x256x64 |
| 2048x9216x16384 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 | 1418.5 | 1083.6 | flydsl | 913.3 | 1164.6 | 144x256x64 |
| 2048x13312x16384 | bfloat16 | 416 | data-parallel | 1419.6 | 1403.9 | flydsl | 1365.3 | 1244.8 | 208x256x64 |
| 2048x26624x16384 | bfloat16 | 832 | one-tile owner, n = 4 | 1568.5 | 1412.7 | flydsl | 910.1 | 1248.5 | 208x256x64 |
| 2048x53248x16384 | bfloat16 | 1664 | one-tile owner, n = 2 | 1602.2 | 1519.7 | flydsl | 801.2 | 1294.0 | 240x256x64 |
| 4096x2304x16384 | bfloat16 | 144 | data-parallel | 1044.0 | 1033.7 | torch | 1187.8 | 1194.2 | 144x256x64 |
| 4096x4608x16384 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 | 1426.7 | 1068.7 | torch | 1180.7 | 1184.6 | 144x256x64 |
| 4096x9216x16384 | bfloat16 | 576 | one-tile owner, n = 4 | 1548.2 | 1355.7 | torch | 1329.9 | 1339.2 | 192x256x64 |
| 4096x13312x16384 | bfloat16 | 832 | one-tile owner, n = 4 | 1575.0 | 1409.3 | flydsl | 1386.1 | 1269.3 | 208x256x64 |
| 4096x26624x16384 | bfloat16 | 1664 | one-tile owner, n = 2 | 1606.9 | 1532.4 | flydsl | 1502.0 | 1324.0 | 240x256x64 |
| 4096x53248x16384 | bfloat16 | 3328 | data-parallel | 1619.8 | 1610.9 | flydsl | 1553.6 | 1480.3 | 256x256x64 |
| 8192x2304x16384 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 | 1382.4 | 1012.1 | torch | 1182.2 | 1186.3 | 144x256x64 |
| 8192x4608x16384 | bfloat16 | 576 | one-tile owner, n = 4 | 1455.1 | 1311.6 | torch | 1356.1 | 1361.3 | 192x256x64 |
| 8192x9216x16384 | bfloat16 | 1152 | one-tile owner, n = 2 | 1528.5 | 1471.2 | flydsl | 1409.2 | 1148.8 | 256x240x64 |
| 8192x13312x16384 | bfloat16 | 1664 | one-tile owner, n = 2 | 1573.5 | 1499.3 | flydsl | 1466.8 | 1311.4 | 240x256x64 |
| 8192x26624x16384 | bfloat16 | 3328 | data-parallel | 1578.4 | 1574.5 | flydsl | 1509.1 | 1439.1 | 256x256x64 |
| 8192x53248x16384 | bfloat16 | 6656 | data-parallel | 1542.6 | 1572.5 | flydsl | 1507.3 | 1444.6 | 256x256x64 |
| 16384x2304x16384 | bfloat16 | 576 | one-tile owner, n = 4 | 1495.9 | 1300.8 | torch | 1347.8 | 1350.3 | 192x256x64 |
| 16384x4608x16384 | bfloat16 | 1152 | one-tile owner, n = 2 | 1553.9 | 1465.2 | flydsl | 1436.6 | 1175.5 | 256x240x64 |
| 16384x9216x16384 | bfloat16 | 2304 | data-parallel | 1588.2 | 1576.2 | flydsl | 1536.9 | 1518.9 | 256x256x64 |
| 16384x13312x16384 | bfloat16 | 3328 | data-parallel | 1597.6 | 1573.8 | asm | 1530.6 | 1515.4 | 256x256x64 |
| 16384x26624x16384 | bfloat16 | 6656 | data-parallel | 1586.0 | 1582.3 | flydsl | 1542.5 | 1425.4 | 256x256x64 |
| 16384x53248x16384 | bfloat16 | 13312 | data-parallel | 1585.4 | 1579.1 | asm | 1535.4 | 1430.5 | 256x256x64 |
| 32768x2304x16384 | bfloat16 | 1152 | one-tile owner, n = 2 | 1528.3 | 1450.2 | flydsl | 1413.5 | 1182.8 | 256x240x64 |
| 32768x4608x16384 | bfloat16 | 2304 | data-parallel | 1576.9 | 1563.4 | torch | 1518.8 | 1522.9 | 256x256x64 |
| 32768x9216x16384 | bfloat16 | 4608 | data-parallel | 1580.8 | 1572.5 | asm | 1514.1 | 1442.7 | 256x256x64 |
| 32768x13312x16384 | bfloat16 | 6656 | data-parallel | 1578.1 | 1571.4 | asm | 1533.3 | 1436.8 | 256x256x64 |
| 32768x26624x16384 | bfloat16 | 13312 | data-parallel | 1556.1 | 1569.4 | flydsl | 1532.0 | 1459.4 | 256x256x64 |
| 32768x53248x16384 | bfloat16 | 26624 | data-parallel | 1552.2 | 1574.0 | flydsl | 1511.0 | 1471.1 | 256x256x64 |
| 512x6144x18432 | bfloat16 | 48 | one-tile owner, n = 5-6 | 978.8 | 354.4 | flydsl | 787.4 | 836.1 | 192x128x128 |
| 1024x6144x18432 | bfloat16 | 96 | one-tile owner, n = 2-3 | 1131.8 | 709.8 | torch | 1167.8 | 1181.4 | 192x256x64 |
| 2048x6144x18432 | bfloat16 | 192 | data-parallel | 1345.1 | 1336.5 | torch | 1343.9 | 1351.1 | 192x256x64 |
| 4096x6144x18432 | bfloat16 | 384 | one-tile owner, n = 2 | 1552.2 | 1355.7 | torch | 1339.2 | 1348.3 | 192x256x64 |
| 8192x6144x18432 | bfloat16 | 768 | data-parallel | 1599.4 | 1583.5 | torch | 1499.6 | 1517.0 | 256x256x64 |
| 16384x6144x18432 | bfloat16 | 1536 | data-parallel | 1609.5 | 1591.1 | asm | 1555.2 | 1507.8 | 256x256x64 |
| 32768x6144x18432 | bfloat16 | 3072 | data-parallel | 1616.7 | 1595.5 | asm | 1529.4 | 1480.5 | 256x256x64 |
| 512x5120x25600 | bfloat16 | 40 | one-tile owner, n = 6-7 | 967.7 | 296.1 | flydsl | 429.7 | 875.8 | 160x128x128 |
| 1024x5120x25600 | bfloat16 | 80 | one-tile owner, n = 3-4 | 1276.6 | 596.3 | torch | 1144.5 | 1151.7 | 160x256x64 |
| 2048x5120x25600 | bfloat16 | 160 | data-parallel | 1185.1 | 1180.4 | torch | 1289.6 | 1295.4 | 160x256x64 |
| 4096x5120x25600 | bfloat16 | 320 | one-tile owner, n = 4 | 1543.8 | 1160.2 | torch | 1274.7 | 1281.5 | 160x256x64 |
| 8192x5120x25600 | bfloat16 | 640 | one-tile owner, n = 2 | 1578.3 | 1416.8 | asm | 1400.4 | 1351.0 | 256x224x64 |
| 16384x5120x25600 | bfloat16 | 1280 | data-parallel | 1608.8 | 1590.0 | torch | 1507.2 | 1512.5 | 256x256x64 |
| 32768x5120x25600 | bfloat16 | 2560 | data-parallel | 1607.4 | 1591.4 | torch | 1498.6 | 1497.0 | 256x256x64 |
| 512x16384x26624 | bfloat16 | 128 | one-tile owner, n = 2 | 1372.5 | 946.8 | flydsl | 654.2 | 1193.4 | 256x256x64 |
| 1024x16384x26624 | bfloat16 | 256 | data-parallel | 1559.7 | 1567.4 | flydsl | 755.6 | 1528.4 | 256x256x64 |
| 2048x16384x26624 | bfloat16 | 512 | data-parallel | 1615.4 | 1602.0 | flydsl | 826.8 | 1532.5 | 256x256x64 |
| 4096x16384x26624 | bfloat16 | 1024 | data-parallel | 1634.2 | 1617.5 | flydsl | 1562.5 | 1538.6 | 256x256x64 |
| 8192x16384x26624 | bfloat16 | 2048 | data-parallel | 1617.3 | 1601.6 | asm | 239.5 | 1455.1 | 256x256x64 |
| 16384x16384x26624 | bfloat16 | 4096 | data-parallel | 1604.9 | 1604.7 | asm | 1523.8 | 1433.6 | 256x256x64 |
| 32768x16384x26624 | bfloat16 | 8192 | data-parallel | 1632.0 | 1608.4 | asm | 1568.7 | 1456.3 | 256x256x64 |
| 512x8192x28672 | bfloat16 | 64 | one-tile owner, n = 4 | 1218.1 | 475.4 | flydsl | 788.1 | 917.2 | 240x256x64 |
| 1024x8192x28672 | bfloat16 | 128 | one-tile owner, n = 2 | 1444.2 | 942.2 | triton | 694.3 | 1347.3 | 256x256x64 |
| 2048x8192x28672 | bfloat16 | 256 | data-parallel | 1596.1 | 1596.9 | flydsl | 1031.1 | 1548.3 | 256x256x64 |
| 4096x8192x28672 | bfloat16 | 512 | data-parallel | 1626.0 | 1611.6 | flydsl | 1565.2 | 1562.8 | 256x256x64 |
| 8192x8192x28672 | bfloat16 | 1024 | data-parallel | 1620.7 | 1604.9 | asm | 1574.1 | 1563.7 | 256x256x64 |
| 16384x8192x28672 | bfloat16 | 2048 | data-parallel | 1621.5 | 1605.1 | asm | 1520.2 | 1538.4 | 256x256x64 |
| 32768x8192x28672 | bfloat16 | 4096 | data-parallel | 1631.3 | 1609.4 | asm | 1554.1 | 1436.7 | 256x256x64 |
| 512x7168x35840 | bfloat16 | 56 | one-tile owner, n = 4-5 | 1162.3 | 438.8 | flydsl | 820.1 | 965.9 | 256x176x64 |
| 1024x7168x35840 | bfloat16 | 112 | one-tile owner, n = 2-3 | 1354.0 | 870.7 | flydsl | 575.0 | 1295.0 | 224x256x64 |
| 2048x7168x35840 | bfloat16 | 224 | spread | 1499.2 | 1486.1 | asm | 1451.5 | 1410.8 | 224x256x64 |
| 4096x7168x35840 | bfloat16 | 448 | spread | 1574.4 | 1501.0 | flydsl | 1465.0 | 1405.6 | 224x256x64 |
| 8192x7168x35840 | bfloat16 | 896 | one-tile owner, n = 2 | 1605.1 | 1472.8 | asm | 1430.3 | 1406.8 | 224x256x64 |
| 16384x7168x35840 | bfloat16 | 1792 | data-parallel | 1561.8 | 1579.6 | torch | 1449.8 | 1473.0 | 256x256x64 |
| 32768x7168x35840 | bfloat16 | 3584 | data-parallel | 1583.9 | 1577.6 | flydsl | 1538.9 | 1469.7 | 256x256x64 |
