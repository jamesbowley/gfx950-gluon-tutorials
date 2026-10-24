# streamk_wip (v27, capped) vs AITER, v9 and hipBLASLt on the 630 shapes (2048 MB rotating buffer)

Shapes: `supported_status_scoped.csv` (630 shapes, no bias; 598 bf16 and 32 fp32 outputs). Inputs from `torch.manual_seed(0)`, every output checked against an fp32 `F.linear` reference, `do_bench_cudagraph` with 3 interleaved rounds and the median taken.

- Stack: torch 2.10.0+rocm7.2.4.git3d3aa833, hipBLASLt 100202 (dabb6df2b9), Triton 3.8.0 @ /home/jbowley/repos/triton_gfx950-tutorial-v2.2/python/triton.
- GPU: HIP device 3, PCI bus 21, gfx950:sramecc+:xnack-.
- AITER: e2c7aa0aa [Triton/Gluon] gfx950 a16w16: skip the Gluon kernel on Triton without cd_regclass, on top of a1455f1c4 2026-10-02 [CK] [FlyDSL] add mxfp8 a8w8 blockscale for gfx950 (#4951).
- streamk_wip: tutorial `v27_streamk_tile`, snapshot sha256 6c5ab21df9acab91 (newest WORKLOG entry: "15. REFIT results, a fitted rule, and the held-out test (2026-10-15)"), `STREAMK_POLICY=capped` (the kernel's default), v9 tile order. The configuration capped ran on each shape is in the appendix and in `v20_label`.
- hipBLASLt: the call AITER's `torch` libtype makes, `F.linear(x, w)` for bf16 and `torch.mm(x, w.t(), out_dtype=torch.float32)` for fp32, timed the same way. Its tile is the MT of the first `Cijk_` kernel it launches.
- Cache state: each callable rotates over max(2, ceil(2048 MB / set)) copies of x, w and C inside the captured graph, so no launch reads the inputs of the previous one. Outputs that a callable allocates itself (AITER's pick, hipBLASLt) are kept alive in a ring so they rotate too.
- "Faster" and "slower" mean a gap of more than 1%; "within 1%" holds the rest.

Copies per shape: median 15, minimum 2; 70 shapes at the minimum of 2.
`do_bench_cudagraph` sizes its graph from a host-side estimate, so on 11 small shapes a timed graph touched fewer copies than it had. What matters is the data a graph cycles through: copies touched x set size is at least 1131 MB on every shape (median 2107 MB), against a 256 MB last-level cache; 0 shapes are under 512 MB.

**Against AITER's pick, by the configuration capped chose**

| | Shapes | v9: faster | v9: within 1% | v9: slower | v9: median | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All shapes | 630 | 306 | 18 | 306 | +0.2% | 479 | 30 | 121 | +4.6% |
| Data-parallel | 439 | 227 | 15 | 197 | +1.5% | 329 | 20 | 90 | +4.1% |
| One-tile reduce-scatter, K >= 2048 | 12 | 5 | 0 | 7 | -6.4% | 12 | 0 | 0 | +17.1% |
| One-tile owner, K >= 2048 | 137 | 58 | 2 | 77 | -6.5% | 108 | 7 | 22 | +9.3% |
| Spread | 42 | 16 | 1 | 25 | -7.9% | 30 | 3 | 9 | +7.9% |

Each pair of columns compares that kernel with AITER's pick; positive medians mean it is faster.

**Against AITER's pick, by the tile AITER uses**

| | Shapes | v9: faster | v9: within 1% | v9: slower | v9: median | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All shapes | 630 | 306 | 18 | 306 | +0.2% | 479 | 30 | 121 | +4.6% |
| AITER uses a 256x256 tile | 248 | 222 | 13 | 13 | +3.7% | 237 | 9 | 2 | +4.5% |
| AITER uses another tile | 382 | 84 | 5 | 293 | -22.1% | 242 | 21 | 119 | +5.6% |

Each pair of columns compares that kernel with AITER's pick; positive medians mean it is faster.

**Against AITER's pick, by v9 grid size (number of 256x256 output tiles)**

| | Shapes | v9: faster | v9: within 1% | v9: slower | v9: median | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All shapes | 630 | 306 | 18 | 306 | +0.2% | 479 | 30 | 121 | +4.6% |
| Under 256 (less than one wave) | 321 | 49 | 5 | 267 | -34.1% | 183 | 23 | 115 | +2.8% |
| 256 to 1023 | 152 | 118 | 4 | 30 | +3.8% | 143 | 4 | 5 | +7.8% |
| 1024 or more | 157 | 139 | 9 | 9 | +3.8% | 153 | 3 | 1 | +4.5% |

Each pair of columns compares that kernel with AITER's pick; positive medians mean it is faster.

**Against hipBLASLt (torch), by v9 grid size**

| | Shapes | AITER's pick: faster | AITER's pick: within 1% | AITER's pick: slower | AITER's pick: median | v9: faster | v9: within 1% | v9: slower | v9: median | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All shapes | 630 | 190 | 237 | 203 | -0.4% | 263 | 42 | 325 | -2.0% | 453 | 37 | 140 | +3.9% |
| Under 256 (less than one wave) | 321 | 112 | 109 | 100 | -0.3% | 39 | 11 | 271 | -34.5% | 168 | 23 | 130 | +1.4% |
| 256 to 1023 | 152 | 36 | 61 | 55 | -0.5% | 103 | 11 | 38 | +3.0% | 137 | 7 | 8 | +6.8% |
| 1024 or more | 157 | 42 | 67 | 48 | -0.5% | 121 | 20 | 16 | +3.4% | 148 | 7 | 2 | +4.0% |

Each pair of columns compares that kernel with hipBLASLt; positive medians mean it is faster.

**hipBLASLt's tile choice**

| hipBLASLt tile (MxNxK) | Shapes | hipBLASLt median TFLOPS | streamk_wip vs hipBLASLt: faster | within 1% | slower | median |
|---|---:|---:|---:|---:|---:|---:|
| 256x256x64 | 225 | 1404 | 210 | 12 | 3 | +3.8% |
| 128x128x256 | 40 | 659 | 37 | 1 | 2 | +20.7% |
| 192x256x64 | 34 | 1180 | 17 | 4 | 13 | +1.0% |
| 224x256x64 | 32 | 1256 | 31 | 0 | 1 | +9.1% |
| 256x128x64 | 27 | 730 | 7 | 2 | 18 | -11.7% |
| 160x256x64 | 27 | 1164 | 13 | 1 | 13 | +0.8% |
| 256x224x64 | 20 | 1262 | 18 | 0 | 2 | +9.6% |
| 256x240x64 | 18 | 1137 | 18 | 0 | 0 | +22.6% |
| 144x256x64 | 15 | 1147 | 13 | 0 | 2 | +5.3% |
| 128x128x128 | 14 | 561 | 7 | 0 | 7 | -1.1% |
| 160x128x128 | 13 | 785 | 1 | 0 | 12 | -12.1% |
| 96x128x128 | 12 | 557 | 7 | 2 | 3 | +2.9% |
| 192x128x128 | 12 | 765 | 5 | 3 | 4 | +0.7% |
| 256x208x64 | 12 | 1072 | 12 | 0 | 0 | +20.2% |
| 256x192x64 | 11 | 1198 | 6 | 1 | 4 | +1.9% |
| 208x256x64 | 11 | 1236 | 10 | 0 | 1 | +11.3% |
| 256x160x64 | 10 | 1125 | 3 | 0 | 7 | -4.9% |
| 64x64x256 | 9 | 404 | 2 | 3 | 4 | +0.1% |
| 128x192x128 | 7 | 877 | 4 | 2 | 1 | +2.9% |
| 128x160x128 | 7 | 858 | 0 | 1 | 6 | -11.3% |
| 240x256x64 | 7 | 1194 | 7 | 0 | 0 | +18.7% |
| 128x80x128 | 6 | 604 | 3 | 2 | 1 | +0.5% |
| 128x96x128 | 6 | 659 | 5 | 0 | 1 | +7.3% |
| 224x128x64 | 5 | 783 | 4 | 1 | 0 | +3.6% |
| 256x176x64 | 4 | 697 | 1 | 1 | 2 | -1.8% |
| 128x64x128 | 4 | 454 | 0 | 0 | 4 | -9.8% |
| 160x64x128 | 4 | 423 | 1 | 0 | 3 | -10.2% |
| 64x80x256 | 4 | 464 | 0 | 0 | 4 | -16.1% |
| 64x96x256 | 4 | 502 | 0 | 0 | 4 | -4.2% |
| 64x32x256 | 3 | 264 | 2 | 1 | 0 | +1.4% |
| 64x48x256 | 3 | 368 | 0 | 0 | 3 | -8.9% |
| 160x192x128 | 3 | 881 | 3 | 0 | 0 | +14.7% |
| 192x128x64 | 2 | 813 | 0 | 0 | 2 | -9.7% |
| 128x192x64 | 2 | 847 | 0 | 0 | 2 | -10.2% |
| 256x112x64 | 2 | 442 | 1 | 0 | 1 | +2.4% |
| 192x64x128 | 2 | 398 | 0 | 0 | 2 | -11.9% |
| 128x128x64 | 2 | 445 | 0 | 0 | 2 | -10.1% |
| 256x96x64 | 2 | 404 | 0 | 0 | 2 | -13.6% |
| 256x64x64 | 2 | 312 | 2 | 0 | 0 | +6.4% |
| 96x64x128 | 1 | 391 | 1 | 0 | 0 | +3.6% |
| 224x160x64 | 1 | 1044 | 1 | 0 | 0 | +15.4% |
| 192x192x64 | 1 | 988 | 0 | 0 | 1 | -6.3% |
| 64x96x128 | 1 | 512 | 0 | 0 | 1 | -12.7% |
| 256x80x64 | 1 | 364 | 0 | 0 | 1 | -19.3% |
| 256x80x128 | 1 | 543 | 0 | 0 | 1 | -22.8% |
| 320x192x64 | 1 | 692 | 1 | 0 | 0 | +1.5% |

Solution-name tokens (stream-K `SK`, global split-U `GSU`, kernels launched): SK3 GSU0 x1: 604, SK3 GSU0 x2: 24, none GSU- x1: 2. `SK3` marks a kernel built with stream-K support, not that this launch split K.

**streamk_wip against v9, by configuration**

| | Shapes | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|
| All shapes | 630 | 431 | 176 | 23 | +6.0% |
| Data-parallel | 439 | 245 | 172 | 22 | +1.6% |
| One-tile reduce-scatter, K >= 2048 | 12 | 11 | 1 | 0 | +27.7% |
| One-tile owner, K >= 2048 | 137 | 135 | 2 | 0 | +17.7% |
| Spread | 42 | 40 | 1 | 1 | +20.5% |

Each pair of columns compares that kernel with v9; positive medians mean it is faster.

**Diagnostic: capped's split against forcing data-parallel**

| Configuration | Shapes | split faster | within 1% | split slower | median |
|---|---:|---:|---:|---:|---:|
| One-tile owner, K >= 2048 | 137 | 134 | 2 | 1 | +16.3% |
| One-tile reduce-scatter, K >= 2048 | 12 | 11 | 1 | 0 | +26.9% |
| Spread | 42 | 40 | 1 | 1 | +13.2% |

**Selected shapes (TFLOPS)**

| Shape (MxNxK) | Output tiles | AITER pick | AITER | hipBLASLt | hipBLASLt tile | v9 | streamk_wip | streamk_wip config | vs v9 | vs AITER | vs hipBLASLt |
|---|---:|---|---:|---:|---|---:|---:|---|---:|---:|---:|
| 512x2304x16384 | 18 | torch | 710 | 708 | 256x128x64 | 140 | 461 | spread | +228% | -35% | -35% |
| 512x5120x25600 | 40 | flydsl | 431 | 879 | 160x128x128 | 296 | 717 | spread | +142% | +66% | -18% |
| 512x7168x35840 | 56 | flydsl | 822 | 965 | 256x176x64 | 437 | 909 | data-parallel | +108% | +11% | -6% |
| 8192x2304x16384 | 288 | torch | 1185 | 1194 | 144x256x64 | 1006 | 1386 | one-tile reduce-scatter, n = 4 | +38% | +17% | +16% |
| 4096x5120x25600 | 320 | torch | 1243 | 1246 | 160x256x64 | 1110 | 1493 | one-tile owner, n = 4 | +35% | +20% | +20% |
| 4096x4608x8192 | 288 | torch | 1160 | 1166 | 144x256x64 | 1016 | 1345 | one-tile reduce-scatter, n = 4 | +32% | +16% | +15% |
| 1024x3072x512 | 48 | flydsl | 368 | 322 | 96x128x128 | 203 | 371 | data-parallel | +83% | +1% | +15% |
| 2048x2048x512 | 64 | flydsl | 431 | 394 | 128x128x128 | 261 | 376 | data-parallel | +44% | -13% | -5% |
| 4096x5120x640 | 320 | flydsl | 614 | 650 | 256x176x64 | 637 | 653 | data-parallel | +3% | +7% | +1% |
| 8192x2048x512 | 256 | torch | 754 | 758 | 256x256x64 | 772 | 768 | data-parallel | -1% | +2% | +1% |

**Against `results/details_v26_rot2048.csv`** (this run / that run - 1, median; for v20 the baseline may use a different policy or kernel snapshot)

| | Shapes | AITER's pick | v9 | v20 | hipBLASLt |
|---|---:|---:|---:|---:|---:|
| All shapes | 630 | +0.1% | +0.3% | +0.4% | +0.0% |
| Under 256 (less than one wave) | 321 | +0.3% | +0.6% | +3.2% | +0.2% |
| 256 to 1023 | 152 | -0.1% | -0.0% | +0.2% | -0.3% |
| 1024 or more | 157 | -0.1% | -0.1% | +0.2% | -0.0% |
| K < 2048 | 91 | +0.2% | +0.4% | +0.7% | +0.2% |
| K >= 2048 | 539 | +0.0% | +0.3% | +0.4% | -0.0% |

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
| 1280x512x4096 | bfloat16 | False (2) | True | True | True (0.5) | True | 0 |  |
| 512x6400x5120 | bfloat16 | False (2) | True | True | True (1) | True | 0 |  |
| 512x12800x5120 | bfloat16 | False (2) | True | True | True (1) | True | 0 |  |
| 1024x6400x5120 | bfloat16 | False (2) | True | True | True (1) | True | 0 |  |
| 512x1280x6144 | bfloat16 | False (2) | True | True | True (1) | True | 0 |  |
| 512x1536x6144 | bfloat16 | False (2) | True | True | True (1) | True | 0 |  |
| 512x2304x6144 | bfloat16 | False (2) | True | True | True (1) | True | 0 |  |
| 512x3072x6144 | bfloat16 | False (2) | True | True | True (1) | True | 0 |  |
| 512x5120x6400 | bfloat16 | False (2) | True | True | True (1) | True | 0 |  |
| 768x2048x7168 | bfloat16 | False (2) | True | True | True (2) | True | 0 |  |
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
| 512x7168x35840 | bfloat16 | False (4) | True | True | True (2) | True | 0 |  |

29 of these are AITER's pick alone outside the tolerance; every other kernel passed.

## Appendix: every shape

| Shape | out | tiles | streamk_wip config | streamk_wip | v9 | AITER pick | AITER | hipBLASLt | hipBLASLt tile |
|---|---|---:|---|---:|---:|---|---:|---:|---|
| 1024x3072x512 | bfloat16 | 48 | data-parallel | 371.3 | 202.9 | flydsl | 368.4 | 321.8 | 96x128x128 |
| 1024x4096x512 | bfloat16 | 64 | data-parallel | 370.8 | 259.2 | flydsl | 439.0 | 392.4 | 128x128x128 |
| 1024x7168x512 | bfloat16 | 112 | data-parallel | 555.6 | 418.4 | flydsl | 515.3 | 501.1 | 256x128x64 |
| 2048x2048x512 | bfloat16 | 64 | data-parallel | 376.1 | 260.8 | flydsl | 431.4 | 394.3 | 128x128x128 |
| 2048x4096x512 | bfloat16 | 128 | data-parallel | 465.4 | 463.6 | torch | 558.7 | 560.3 | 256x128x64 |
| 4096x2048x512 | bfloat16 | 128 | data-parallel | 463.4 | 458.3 | torch | 643.2 | 648.9 | 256x128x64 |
| 4096x4096x512 | bfloat16 | 256 | data-parallel | 765.3 | 770.1 | torch | 762.0 | 766.6 | 256x256x64 |
| 4096x7168x512 | bfloat16 | 448 | data-parallel | 782.6 | 763.8 | flydsl | 744.4 | 720.3 | 224x256x64 |
| 8192x2048x512 | bfloat16 | 256 | data-parallel | 767.5 | 772.1 | torch | 753.5 | 758.0 | 256x256x64 |
| 8192x4096x512 | bfloat16 | 512 | data-parallel | 840.9 | 814.8 | flydsl | 801.4 | 835.7 | 256x256x64 |
| 16384x2048x512 | bfloat16 | 512 | data-parallel | 853.1 | 819.8 | torch | 830.6 | 828.8 | 256x256x64 |
| 16384x4096x512 | bfloat16 | 1024 | data-parallel | 896.1 | 838.5 | torch | 881.8 | 891.4 | 256x256x64 |
| 16384x7168x512 | bfloat16 | 1792 | data-parallel | 964.8 | 900.2 | torch | 972.5 | 981.9 | 256x256x64 |
| 32768x2048x512 | bfloat16 | 1024 | data-parallel | 906.0 | 844.6 | torch | 872.9 | 872.1 | 256x256x64 |
| 1024x5120x640 | bfloat16 | 80 | data-parallel | 462.0 | 325.7 | torch | 444.2 | 443.9 | 160x128x128 |
| 2048x5120x640 | bfloat16 | 160 | data-parallel | 598.6 | 603.6 | flydsl | 594.2 | 502.6 | 192x256x64 |
| 4096x5120x640 | bfloat16 | 320 | data-parallel | 653.4 | 636.7 | flydsl | 613.5 | 649.8 | 256x176x64 |
| 8192x5120x640 | bfloat16 | 640 | data-parallel | 827.2 | 796.9 | torch | 845.8 | 871.1 | 256x224x64 |
| 16384x5120x640 | bfloat16 | 1280 | data-parallel | 1004.8 | 953.2 | torch | 998.1 | 1001.5 | 256x256x64 |
| 32768x5120x640 | bfloat16 | 2560 | data-parallel | 1051.6 | 989.7 | torch | 1038.6 | 1051.0 | 256x256x64 |
| 1024x7168x768 | bfloat16 | 112 | data-parallel | 637.7 | 487.3 | flydsl | 597.6 | 584.7 | 224x128x64 |
| 10240x7168x768 | bfloat16 | 1120 | data-parallel | 1039.3 | 1002.1 | opus | 388.7 | 850.9 | 256x240x64 |
| 512x4096x1024 | bfloat16 | 32 | data-parallel | 370.1 | 181.7 | torch | 391.5 | 392.9 | 128x64x128 |
| 512x6144x1024 | bfloat16 | 48 | data-parallel | 521.7 | 269.5 | flydsl | 498.2 | 474.8 | 96x128x128 |
| 512x7168x1024 | bfloat16 | 56 | data-parallel | 578.5 | 310.2 | flydsl | 540.1 | 493.9 | 128x128x128 |
| 512x8192x1024 | bfloat16 | 64 | data-parallel | 474.0 | 346.9 | flydsl | 575.8 | 548.9 | 128x128x128 |
| 1024x4096x1024 | bfloat16 | 64 | data-parallel | 484.3 | 346.7 | torch | 559.1 | 562.0 | 128x128x128 |
| 1024x7168x1024 | bfloat16 | 112 | data-parallel | 741.9 | 570.8 | flydsl | 667.6 | 692.8 | 224x128x64 |
| 1024x8192x1024 | bfloat16 | 128 | data-parallel | 625.4 | 627.5 | torch | 756.2 | 760.6 | 256x128x64 |
| 2048x4096x1024 | bfloat16 | 128 | data-parallel | 632.0 | 631.4 | torch | 766.1 | 772.9 | 256x128x64 |
| 2048x7168x1024 | bfloat16 | 224 | data-parallel | 979.3 | 997.1 | flydsl | 949.6 | 957.6 | 256x256x64 |
| 2048x8192x1024 | bfloat16 | 256 | data-parallel | 1051.5 | 1055.5 | flydsl | 1009.2 | 1026.1 | 256x256x64 |
| 3072x4096x1024 | bfloat16 | 192 | data-parallel | 874.5 | 886.4 | torch | 907.2 | 916.0 | 256x192x64 |
| 4096x4096x1024 | bfloat16 | 256 | data-parallel | 1065.8 | 1063.6 | torch | 1013.1 | 1023.7 | 256x256x64 |
| 4096x7168x1024 | bfloat16 | 448 | data-parallel | 1045.6 | 1029.0 | flydsl | 984.5 | 972.9 | 224x256x64 |
| 4096x8192x1024 | bfloat16 | 512 | data-parallel | 1115.9 | 1096.3 | asm | 1105.4 | 1101.7 | 256x256x64 |
| 5120x4096x1024 | bfloat16 | 320 | data-parallel | 864.3 | 853.6 | torch | 972.1 | 977.9 | 256x160x64 |
| 7680x4096x1024 | bfloat16 | 480 | data-parallel | 1100.7 | 1084.5 | torch | 932.8 | 937.7 | 256x240x64 |
| 8192x7168x1024 | bfloat16 | 896 | data-parallel | 1097.7 | 1045.4 | asm | 1036.6 | 1018.8 | 224x256x64 |
| 8192x8192x1024 | bfloat16 | 1024 | data-parallel | 1173.2 | 1127.6 | asm | 1136.6 | 1152.1 | 256x256x64 |
| 16384x4096x1024 | bfloat16 | 1024 | data-parallel | 1175.6 | 1137.3 | torch | 1147.3 | 1157.0 | 256x256x64 |
| 16384x7168x1024 | bfloat16 | 1792 | data-parallel | 1221.0 | 1176.2 | asm | 1131.1 | 1199.5 | 256x256x64 |
| 16384x8192x1024 | bfloat16 | 2048 | data-parallel | 1225.1 | 1168.7 | torch | 1183.2 | 1198.4 | 256x256x64 |
| 32768x7168x1024 | bfloat16 | 3584 | data-parallel | 1246.8 | 1202.3 | torch | 1216.0 | 1226.8 | 256x256x64 |
| 32768x8192x1024 | bfloat16 | 4096 | data-parallel | 1249.6 | 1197.3 | torch | 1204.7 | 1214.3 | 256x256x64 |
| 512x5120x1280 | bfloat16 | 40 | data-parallel | 429.3 | 204.0 | flydsl | 450.2 | 424.5 | 160x64x128 |
| 1024x5120x1280 | bfloat16 | 80 | data-parallel | 543.0 | 396.7 | torch | 602.6 | 604.8 | 160x128x128 |
| 2048x5120x1280 | bfloat16 | 160 | data-parallel | 754.2 | 759.2 | torch | 843.8 | 847.0 | 160x256x64 |
| 4096x5120x1280 | bfloat16 | 320 | data-parallel | 801.0 | 793.6 | torch | 877.3 | 878.2 | 160x256x64 |
| 8192x5120x1280 | bfloat16 | 640 | data-parallel | 1041.3 | 1010.9 | flydsl | 989.2 | 1074.3 | 256x224x64 |
| 16384x5120x1280 | bfloat16 | 1280 | data-parallel | 1234.4 | 1180.5 | flydsl | 1162.3 | 1230.7 | 256x256x64 |
| 32768x5120x1280 | bfloat16 | 2560 | data-parallel | 1263.8 | 1222.8 | torch | 1243.8 | 1250.6 | 256x256x64 |
| 512x1536x1536 | bfloat16 | 12 | data-parallel | 272.8 | 70.2 | flydsl | 267.0 | 236.7 | 64x64x256 |
| 512x2048x1536 | bfloat16 | 16 | data-parallel | 272.1 | 93.0 | torch | 312.3 | 311.8 | 64x64x256 |
| 512x2304x1536 | bfloat16 | 18 | data-parallel | 305.8 | 104.7 | flydsl | 321.6 | 313.8 | 64x80x256 |
| 512x4096x1536 | bfloat16 | 32 | data-parallel | 404.6 | 187.2 | torch | 445.7 | 448.9 | 128x64x128 |
| 512x6144x1536 | bfloat16 | 48 | data-parallel | 568.3 | 278.5 | flydsl | 544.9 | 543.2 | 96x128x128 |
| 512x7168x1536 | bfloat16 | 56 | data-parallel | 640.0 | 321.0 | flydsl | 581.7 | 491.8 | 128x128x256 |
| 1024x1536x1536 | bfloat16 | 24 | data-parallel | 405.1 | 141.0 | flydsl | 402.9 | 391.2 | 96x64x128 |
| 1024x2048x1536 | bfloat16 | 32 | data-parallel | 414.0 | 185.7 | torch | 456.9 | 458.9 | 128x64x128 |
| 1024x2304x1536 | bfloat16 | 36 | data-parallel | 457.4 | 208.6 | torch | 457.5 | 459.6 | 128x80x128 |
| 1024x3072x1536 | bfloat16 | 48 | data-parallel | 586.4 | 277.7 | flydsl | 552.7 | 556.3 | 96x128x128 |
| 1024x4096x1536 | bfloat16 | 64 | data-parallel | 501.1 | 365.6 | torch | 556.2 | 558.0 | 128x128x256 |
| 1024x7168x1536 | bfloat16 | 112 | data-parallel | 797.0 | 615.1 | torch | 778.3 | 783.2 | 224x128x64 |
| 2048x1536x1536 | bfloat16 | 48 | data-parallel | 584.1 | 282.8 | torch | 551.8 | 556.0 | 96x128x128 |
| 2048x2048x1536 | bfloat16 | 64 | data-parallel | 498.9 | 370.3 | torch | 559.6 | 561.1 | 128x128x256 |
| 2048x4096x1536 | bfloat16 | 128 | data-parallel | 695.2 | 695.4 | torch | 845.6 | 850.2 | 256x128x64 |
| 3072x2048x1536 | bfloat16 | 96 | data-parallel | 712.5 | 539.4 | opus | 689.6 | 763.1 | 128x192x128 |
| 3072x4096x1536 | bfloat16 | 192 | data-parallel | 974.4 | 985.6 | opus | 994.1 | 951.8 | 256x192x64 |
| 4096x1536x1536 | bfloat16 | 96 | data-parallel | 713.1 | 538.1 | torch | 743.9 | 745.8 | 192x128x128 |
| 4096x2048x1536 | bfloat16 | 128 | data-parallel | 690.7 | 692.4 | opus | 857.5 | 846.5 | 256x128x64 |
| 4096x3072x1536 | bfloat16 | 192 | data-parallel | 980.7 | 989.5 | flydsl | 942.0 | 1000.9 | 192x256x64 |
| 4096x4096x1536 | bfloat16 | 256 | data-parallel | 1186.0 | 1186.4 | torch | 1154.4 | 1163.9 | 256x256x64 |
| 5120x2048x1536 | bfloat16 | 160 | data-parallel | 836.3 | 843.9 | torch | 926.3 | 949.5 | 256x160x64 |
| 5120x4096x1536 | bfloat16 | 320 | data-parallel | 940.0 | 935.1 | torch | 997.4 | 998.3 | 256x160x64 |
| 7680x2048x1536 | bfloat16 | 240 | data-parallel | 1119.6 | 1139.4 | torch | 946.8 | 949.9 | 256x240x64 |
| 7680x4096x1536 | bfloat16 | 480 | data-parallel | 1192.2 | 1188.3 | torch | 1019.4 | 1022.2 | 256x240x64 |
| 8192x1536x1536 | bfloat16 | 192 | data-parallel | 965.6 | 975.7 | torch | 986.4 | 985.3 | 192x256x64 |
| 10240x7168x1536 | bfloat16 | 1120 | data-parallel | 1224.0 | 1204.1 | asm | 1143.5 | 1120.0 | 224x256x64 |
| 16384x1536x1536 | bfloat16 | 384 | data-parallel | 1058.6 | 1052.0 | torch | 1033.6 | 1045.9 | 192x256x64 |
| 16384x2048x1536 | bfloat16 | 512 | data-parallel | 1206.8 | 1209.0 | torch | 1206.2 | 1208.5 | 256x256x64 |
| 16384x3072x1536 | bfloat16 | 768 | data-parallel | 1263.1 | 1242.2 | torch | 1236.6 | 1240.0 | 256x256x64 |
| 16384x4096x1536 | bfloat16 | 1024 | data-parallel | 1262.6 | 1236.8 | torch | 1256.1 | 1260.8 | 256x256x64 |
| 32768x1536x1536 | bfloat16 | 768 | data-parallel | 1251.3 | 1237.7 | flydsl | 1199.9 | 1217.3 | 256x256x64 |
| 512x7168x1792 | bfloat16 | 56 | data-parallel | 660.0 | 345.6 | flydsl | 617.9 | 501.4 | 128x128x256 |
| 1024x7168x1792 | bfloat16 | 112 | data-parallel | 856.3 | 663.1 | flydsl | 780.1 | 826.5 | 224x128x64 |
| 2048x7168x1792 | bfloat16 | 224 | data-parallel | 1143.3 | 1168.8 | flydsl | 1099.0 | 1049.6 | 224x256x64 |
| 4096x7168x1792 | bfloat16 | 448 | data-parallel | 1210.7 | 1204.1 | flydsl | 1150.5 | 1106.5 | 224x256x64 |
| 8192x7168x1792 | bfloat16 | 896 | data-parallel | 1246.0 | 1218.7 | asm | 1187.5 | 1142.4 | 224x256x64 |
| 16384x7168x1792 | bfloat16 | 1792 | data-parallel | 1355.3 | 1322.4 | torch | 1320.5 | 1329.9 | 256x256x64 |
| 32768x7168x1792 | bfloat16 | 3584 | data-parallel | 1382.7 | 1346.2 | torch | 1343.2 | 1350.3 | 256x256x64 |
| 512x4096x2048 | bfloat16 | 32 | data-parallel | 440.9 | 215.2 | torch | 489.2 | 489.9 | 128x64x128 |
| 512x6144x2048 | float32 | 48 | data-parallel | 558.9 | 289.9 | asm | 333.7 | 557.9 | 96x128x128 |
| 512x6144x2048 | bfloat16 | 48 | data-parallel | 620.2 | 322.4 | flydsl | 602.0 | 592.9 | 96x128x128 |
| 512x8192x2048 | bfloat16 | 64 | one-tile owner, n = 2 | 599.5 | 413.0 | flydsl | 693.6 | 572.8 | 128x128x256 |
| 512x16384x2048 | bfloat16 | 128 | one-tile owner, n = 2 | 767.4 | 751.2 | flydsl | 713.8 | 878.6 | 256x128x64 |
| 1024x4096x2048 | bfloat16 | 64 | one-tile owner, n = 2 | 615.8 | 417.7 | torch | 610.5 | 612.3 | 128x128x256 |
| 1024x6144x2048 | float32 | 96 | data-parallel | 710.6 | 550.5 | opus | 556.7 | 780.7 | 192x128x64 |
| 1024x6144x2048 | bfloat16 | 96 | data-parallel | 799.9 | 608.8 | flydsl | 728.3 | 783.3 | 192x128x128 |
| 1024x8192x2048 | bfloat16 | 128 | one-tile owner, n = 2 | 781.4 | 771.2 | flydsl | 878.1 | 926.5 | 256x128x64 |
| 1024x16384x2048 | bfloat16 | 256 | data-parallel | 1249.7 | 1257.6 | flydsl | 746.0 | 1201.5 | 256x256x64 |
| 2048x4096x2048 | bfloat16 | 128 | one-tile owner, n = 2 | 786.3 | 776.8 | torch | 935.9 | 942.1 | 256x128x64 |
| 2048x6144x2048 | float32 | 192 | data-parallel | 951.9 | 962.1 | asm | 915.7 | 1001.9 | 192x256x64 |
| 2048x6144x2048 | bfloat16 | 192 | data-parallel | 1072.5 | 1094.0 | flydsl | 1034.5 | 1083.9 | 192x256x64 |
| 2048x8192x2048 | bfloat16 | 256 | data-parallel | 1253.9 | 1260.2 | flydsl | 1193.0 | 1224.1 | 256x256x64 |
| 2048x16384x2048 | bfloat16 | 512 | data-parallel | 1311.9 | 1302.5 | flydsl | 1024.9 | 1263.5 | 256x256x64 |
| 3072x4096x2048 | bfloat16 | 192 | data-parallel | 1067.0 | 1085.7 | torch | 1080.9 | 1090.9 | 256x192x64 |
| 4096x4096x2048 | bfloat16 | 256 | data-parallel | 1270.4 | 1272.6 | torch | 1239.1 | 1239.6 | 256x256x64 |
| 4096x6144x2048 | float32 | 384 | data-parallel | 982.5 | 990.4 | asm | 947.9 | 1053.3 | 192x256x64 |
| 4096x6144x2048 | bfloat16 | 384 | data-parallel | 1141.9 | 1136.1 | flydsl | 895.1 | 1140.3 | 192x256x64 |
| 4096x8192x2048 | bfloat16 | 512 | data-parallel | 1318.6 | 1306.6 | flydsl | 1256.6 | 1278.8 | 256x256x64 |
| 4096x16384x2048 | bfloat16 | 1024 | data-parallel | 1350.2 | 1331.6 | flydsl | 859.5 | 1312.5 | 256x256x64 |
| 5120x4096x2048 | bfloat16 | 320 | one-tile owner, n = 2 | 1036.8 | 996.6 | torch | 1114.3 | 1115.3 | 256x160x64 |
| 7680x4096x2048 | bfloat16 | 480 | data-parallel | 1288.2 | 1283.9 | torch | 1093.0 | 1099.5 | 256x240x64 |
| 8192x6144x2048 | float32 | 768 | data-parallel | 1184.7 | 1174.0 | asm | 1131.0 | 1091.8 | 256x256x64 |
| 8192x6144x2048 | bfloat16 | 768 | data-parallel | 1355.8 | 1335.1 | flydsl | 836.0 | 1318.4 | 256x256x64 |
| 8192x8192x2048 | bfloat16 | 1024 | data-parallel | 1366.6 | 1340.4 | asm | 1314.8 | 1330.2 | 256x256x64 |
| 8192x16384x2048 | bfloat16 | 2048 | data-parallel | 1377.3 | 1341.0 | triton | 1020.2 | 1353.1 | 256x256x64 |
| 16384x4096x2048 | bfloat16 | 1024 | data-parallel | 1358.1 | 1333.7 | torch | 1308.5 | 1318.4 | 256x256x64 |
| 16384x6144x2048 | float32 | 1536 | data-parallel | 1206.9 | 1188.2 | asm | 1162.2 | 1125.9 | 256x256x64 |
| 16384x6144x2048 | bfloat16 | 1536 | data-parallel | 1380.0 | 1351.3 | torch | 1344.3 | 1352.9 | 256x256x64 |
| 16384x8192x2048 | bfloat16 | 2048 | data-parallel | 1391.9 | 1357.4 | torch | 1353.0 | 1362.9 | 256x256x64 |
| 16384x16384x2048 | bfloat16 | 4096 | data-parallel | 1398.7 | 1360.1 | flydsl | 959.1 | 1383.6 | 256x256x64 |
| 32768x6144x2048 | bfloat16 | 3072 | data-parallel | 1411.0 | 1376.7 | flydsl | 996.7 | 1378.1 | 256x256x64 |
| 32768x8192x2048 | bfloat16 | 4096 | data-parallel | 1419.1 | 1380.1 | torch | 1376.1 | 1381.1 | 256x256x64 |
| 32768x16384x2048 | bfloat16 | 8192 | data-parallel | 1398.2 | 1356.4 | flydsl | 828.8 | 1399.0 | 256x256x64 |
| 512x6144x3072 | float32 | 48 | data-parallel | 612.3 | 295.2 | opus | 420.6 | 617.7 | 96x128x128 |
| 512x6144x3072 | bfloat16 | 48 | data-parallel | 667.4 | 316.6 | flydsl | 642.9 | 659.8 | 96x128x128 |
| 1024x6144x3072 | float32 | 96 | data-parallel | 757.0 | 580.4 | opus | 625.2 | 845.4 | 192x128x64 |
| 1024x6144x3072 | bfloat16 | 96 | data-parallel | 810.3 | 620.7 | flydsl | 783.1 | 831.8 | 192x128x128 |
| 2048x6144x3072 | float32 | 192 | data-parallel | 1041.6 | 1048.8 | asm | 1018.5 | 1075.1 | 192x256x64 |
| 2048x6144x3072 | bfloat16 | 192 | data-parallel | 1147.5 | 1159.9 | flydsl | 1115.3 | 1150.3 | 192x256x64 |
| 4096x6144x3072 | float32 | 384 | data-parallel | 1072.1 | 1085.1 | asm | 1040.9 | 1118.9 | 192x256x64 |
| 4096x6144x3072 | bfloat16 | 384 | data-parallel | 1201.1 | 1199.0 | flydsl | 931.4 | 1190.8 | 192x256x64 |
| 8192x6144x3072 | float32 | 768 | data-parallel | 1270.8 | 1265.9 | asm | 1221.2 | 1172.6 | 256x256x64 |
| 8192x6144x3072 | bfloat16 | 768 | data-parallel | 1416.8 | 1401.4 | flydsl | 783.7 | 1367.9 | 256x256x64 |
| 16384x6144x3072 | float32 | 1536 | data-parallel | 1297.3 | 1282.7 | asm | 1257.6 | 1210.5 | 256x256x64 |
| 16384x6144x3072 | bfloat16 | 1536 | data-parallel | 1443.9 | 1420.0 | torch | 1391.6 | 1398.8 | 256x256x64 |
| 32768x6144x3072 | bfloat16 | 3072 | data-parallel | 1460.8 | 1438.3 | flydsl | 865.8 | 1413.8 | 256x256x64 |
| 512x5120x3200 | bfloat16 | 40 | data-parallel | 553.5 | 270.3 | flydsl | 578.6 | 565.7 | 160x64x128 |
| 1024x5120x3200 | bfloat16 | 80 | data-parallel | 733.6 | 533.8 | torch | 819.3 | 823.0 | 160x128x128 |
| 2048x5120x3200 | bfloat16 | 160 | data-parallel | 1038.1 | 1051.2 | torch | 1075.8 | 1081.8 | 160x256x64 |
| 4096x5120x3200 | bfloat16 | 320 | one-tile owner, n = 2 | 1159.1 | 1037.0 | torch | 1108.7 | 1114.7 | 160x256x64 |
| 8192x5120x3200 | bfloat16 | 640 | data-parallel | 1297.7 | 1287.3 | asm | 1242.4 | 1212.3 | 256x224x64 |
| 16384x5120x3200 | bfloat16 | 1280 | data-parallel | 1425.7 | 1410.0 | torch | 1380.2 | 1387.1 | 256x256x64 |
| 32768x5120x3200 | bfloat16 | 2560 | data-parallel | 1452.1 | 1438.4 | torch | 1404.7 | 1411.3 | 256x256x64 |
| 512x7168x3584 | bfloat16 | 56 | data-parallel | 766.9 | 379.0 | flydsl | 707.7 | 585.6 | 128x128x256 |
| 512x8192x3584 | bfloat16 | 64 | one-tile owner, n = 2 | 746.7 | 431.2 | flydsl | 672.4 | 650.1 | 128x128x256 |
| 1024x7168x3584 | bfloat16 | 112 | data-parallel | 957.2 | 740.1 | torch | 958.7 | 963.7 | 224x128x64 |
| 1024x8192x3584 | bfloat16 | 128 | one-tile owner, n = 2 | 966.1 | 831.8 | flydsl | 790.5 | 1019.2 | 256x128x64 |
| 2048x8192x3584 | bfloat16 | 256 | data-parallel | 1357.3 | 1358.9 | flydsl | 1253.7 | 1311.3 | 256x256x64 |
| 4096x8192x3584 | bfloat16 | 512 | data-parallel | 1400.8 | 1394.9 | flydsl | 987.7 | 1333.8 | 256x256x64 |
| 8192x8192x3584 | bfloat16 | 1024 | data-parallel | 1430.1 | 1420.6 | torch | 1372.0 | 1380.4 | 256x256x64 |
| 10240x7168x3584 | bfloat16 | 1120 | one-tile owner, n = 2-3 | 1377.5 | 1326.0 | flydsl | 785.1 | 1245.0 | 224x256x64 |
| 16384x8192x3584 | bfloat16 | 2048 | data-parallel | 1453.1 | 1445.2 | flydsl | 810.2 | 1404.0 | 256x256x64 |
| 32768x8192x3584 | bfloat16 | 4096 | data-parallel | 1470.4 | 1463.5 | flydsl | 860.7 | 1418.6 | 256x256x64 |
| 512x1024x4096 | bfloat16 | 8 | one-tile owner, n = 2 | 267.8 | 60.3 | flydsl | 284.2 | 264.2 | 64x32x256 |
| 512x2048x4096 | bfloat16 | 16 | one-tile owner, n = 2 | 398.5 | 120.6 | flydsl | 438.3 | 422.2 | 64x64x256 |
| 512x4096x4096 | bfloat16 | 32 | one-tile owner, n = 2 | 588.9 | 240.8 | flydsl | 596.3 | 614.3 | 128x128x128 |
| 512x8192x4096 | bfloat16 | 64 | one-tile owner, n = 2 | 790.9 | 467.3 | flydsl | 799.6 | 672.8 | 128x128x256 |
| 512x8704x4096 | bfloat16 | 68 | data-parallel | 655.6 | 498.6 | torch | 745.0 | 745.9 | 160x128x128 |
| 512x10240x4096 | bfloat16 | 80 | data-parallel | 761.1 | 577.2 | torch | 859.7 | 862.4 | 128x192x64 |
| 512x16384x4096 | bfloat16 | 128 | one-tile owner, n = 2 | 999.7 | 848.6 | flydsl | 635.4 | 679.1 | 256x256x64 |
| 512x129280x4096 | bfloat16 | 1010 | data-parallel | 1349.1 | 1358.1 | asm | 1281.2 | 1148.1 | 256x256x64 |
| 768x1024x4096 | bfloat16 | 12 | data-parallel | 341.7 | 91.3 | torch | 368.3 | 372.4 | 64x48x256 |
| 768x2048x4096 | bfloat16 | 24 | data-parallel | 492.5 | 182.2 | flydsl | 524.9 | 499.4 | 64x96x256 |
| 768x4096x4096 | bfloat16 | 48 | data-parallel | 719.1 | 361.4 | flydsl | 692.6 | 653.9 | 128x96x128 |
| 768x8192x4096 | bfloat16 | 96 | data-parallel | 908.7 | 683.8 | torch | 878.1 | 883.1 | 128x192x128 |
| 768x8704x4096 | bfloat16 | 102 | data-parallel | 950.1 | 725.9 | flydsl | 864.4 | 843.7 | 160x192x128 |
| 768x10240x4096 | bfloat16 | 120 | data-parallel | 1058.6 | 831.0 | flydsl | 959.1 | 923.0 | 160x192x128 |
| 768x129280x4096 | bfloat16 | 1515 | data-parallel | 1301.5 | 1408.9 | torch | 1349.9 | 1354.4 | 256x256x64 |
| 1024x512x4096 | bfloat16 | 8 | one-tile owner, n = 2 | 265.4 | 60.3 | flydsl | 282.4 | 261.4 | 64x32x256 |
| 1024x1024x4096 | bfloat16 | 16 | one-tile owner, n = 2 | 399.7 | 122.1 | flydsl | 440.6 | 440.8 | 64x64x256 |
| 1024x2048x4096 | bfloat16 | 32 | one-tile owner, n = 2 | 608.9 | 241.9 | flydsl | 599.7 | 492.4 | 128x128x256 |
| 1024x4096x4096 | bfloat16 | 64 | one-tile owner, n = 2 | 834.9 | 471.2 | flydsl | 811.9 | 718.4 | 128x128x256 |
| 1024x8192x4096 | bfloat16 | 128 | one-tile owner, n = 2 | 1000.6 | 875.7 | flydsl | 1013.9 | 679.8 | 256x256x64 |
| 1024x8704x4096 | bfloat16 | 136 | data-parallel | 922.2 | 927.4 | torch | 987.2 | 989.4 | 144x256x64 |
| 1024x10240x4096 | bfloat16 | 160 | data-parallel | 1053.2 | 1061.5 | torch | 1104.0 | 1105.6 | 160x256x64 |
| 1024x16384x4096 | bfloat16 | 256 | data-parallel | 1382.0 | 1379.5 | flydsl | 826.3 | 1304.3 | 256x256x64 |
| 1024x129280x4096 | bfloat16 | 2020 | data-parallel | 1458.0 | 1432.5 | torch | 1373.5 | 1364.3 | 256x256x64 |
| 1280x512x4096 | bfloat16 | 10 | data-parallel | 283.3 | 76.1 | flydsl | 319.5 | 315.2 | 64x48x256 |
| 1280x1024x4096 | bfloat16 | 20 | data-parallel | 408.8 | 152.1 | torch | 482.0 | 485.4 | 64x80x256 |
| 1280x2048x4096 | bfloat16 | 40 | data-parallel | 622.0 | 304.2 | flydsl | 614.0 | 611.2 | 128x80x128 |
| 1280x129280x4096 | bfloat16 | 2525 | data-parallel | 1423.7 | 1407.2 | torch | 1379.8 | 1390.7 | 256x256x64 |
| 1536x512x4096 | bfloat16 | 12 | data-parallel | 334.7 | 90.9 | torch | 364.6 | 367.6 | 64x48x256 |
| 1536x1024x4096 | bfloat16 | 24 | data-parallel | 488.7 | 182.8 | flydsl | 517.9 | 504.6 | 64x96x256 |
| 1536x2048x4096 | bfloat16 | 48 | data-parallel | 733.7 | 364.2 | flydsl | 689.8 | 685.0 | 128x96x128 |
| 1536x129280x4096 | bfloat16 | 3030 | data-parallel | 1444.4 | 1428.1 | torch | 1386.3 | 1395.8 | 256x256x64 |
| 1792x512x4096 | bfloat16 | 14 | data-parallel | 383.2 | 106.0 | flydsl | 389.9 | 382.9 | 64x64x256 |
| 1792x1024x4096 | bfloat16 | 28 | data-parallel | 555.0 | 213.8 | flydsl | 566.4 | 463.5 | 256x128x64 |
| 1792x2048x4096 | bfloat16 | 56 | data-parallel | 830.2 | 422.7 | flydsl | 749.2 | 662.5 | 128x128x256 |
| 1792x129280x4096 | bfloat16 | 3535 | data-parallel | 1457.7 | 1441.4 | asm | 1394.7 | 1360.4 | 256x256x64 |
| 2048x512x4096 | bfloat16 | 16 | one-tile owner, n = 2 | 391.4 | 121.6 | flydsl | 425.0 | 427.8 | 64x64x256 |
| 2048x1024x4096 | bfloat16 | 32 | one-tile owner, n = 2 | 607.8 | 244.2 | flydsl | 602.6 | 496.0 | 128x128x256 |
| 2048x2048x4096 | bfloat16 | 64 | one-tile owner, n = 2 | 830.8 | 476.2 | flydsl | 831.3 | 731.7 | 128x128x256 |
| 2048x4096x4096 | bfloat16 | 128 | one-tile owner, n = 2 | 1012.1 | 884.3 | opus | 1036.9 | 679.2 | 256x256x64 |
| 2048x8192x4096 | bfloat16 | 256 | data-parallel | 1385.3 | 1384.7 | flydsl | 1282.1 | 1337.7 | 256x256x64 |
| 2048x8704x4096 | bfloat16 | 272 | one-tile reduce-scatter, n = 8 | 1205.3 | 920.2 | torch | 1038.8 | 1044.5 | 224x160x64 |
| 2048x10240x4096 | bfloat16 | 320 | one-tile owner, n = 2 | 1199.5 | 1070.4 | torch | 1117.8 | 1125.6 | 160x256x64 |
| 2048x16384x4096 | bfloat16 | 512 | data-parallel | 1422.0 | 1416.0 | flydsl | 1345.8 | 1361.7 | 256x256x64 |
| 2048x129280x4096 | bfloat16 | 4040 | data-parallel | 1497.6 | 1465.3 | flydsl | 1402.3 | 1387.8 | 256x256x64 |
| 2304x512x4096 | bfloat16 | 18 | data-parallel | 364.6 | 136.1 | flydsl | 353.1 | 443.3 | 64x80x256 |
| 2304x1024x4096 | bfloat16 | 36 | data-parallel | 564.6 | 273.2 | flydsl | 512.1 | 651.4 | 128x80x128 |
| 2304x2048x4096 | bfloat16 | 72 | data-parallel | 719.3 | 531.0 | opus | 685.4 | 885.1 | 128x160x128 |
| 2560x512x4096 | bfloat16 | 20 | data-parallel | 405.4 | 151.4 | flydsl | 381.8 | 485.5 | 64x80x256 |
| 2560x1024x4096 | bfloat16 | 40 | data-parallel | 618.2 | 304.4 | flydsl | 557.2 | 608.2 | 128x80x128 |
| 2560x2048x4096 | bfloat16 | 80 | data-parallel | 790.9 | 583.5 | opus | 766.6 | 885.5 | 128x160x128 |
| 2816x512x4096 | bfloat16 | 22 | data-parallel | 437.7 | 165.9 | flydsl | 424.0 | 468.1 | 64x96x256 |
| 2816x1024x4096 | bfloat16 | 44 | data-parallel | 674.7 | 334.0 | flydsl | 602.8 | 643.8 | 128x96x128 |
| 2816x2048x4096 | bfloat16 | 88 | data-parallel | 864.6 | 638.8 | opus | 818.4 | 836.8 | 192x128x128 |
| 3072x512x4096 | bfloat16 | 24 | data-parallel | 479.6 | 181.8 | flydsl | 452.5 | 505.8 | 64x96x256 |
| 3072x1024x4096 | bfloat16 | 48 | data-parallel | 725.9 | 363.9 | flydsl | 643.6 | 675.8 | 128x96x128 |
| 3072x2048x4096 | bfloat16 | 96 | data-parallel | 932.7 | 689.6 | opus | 925.6 | 903.5 | 128x192x128 |
| 3328x512x4096 | bfloat16 | 26 | data-parallel | 507.9 | 196.5 | flydsl | 477.1 | 444.0 | 256x112x64 |
| 3328x1024x4096 | bfloat16 | 52 | data-parallel | 773.3 | 392.2 | flydsl | 680.2 | 625.9 | 128x128x256 |
| 3328x2048x4096 | bfloat16 | 104 | data-parallel | 986.2 | 739.0 | opus | 940.7 | 775.4 | 256x208x64 |
| 3584x512x4096 | bfloat16 | 28 | data-parallel | 543.9 | 211.0 | flydsl | 509.1 | 449.2 | 256x128x64 |
| 3584x1024x4096 | bfloat16 | 56 | data-parallel | 809.7 | 418.0 | flydsl | 708.3 | 662.8 | 128x128x256 |
| 3584x2048x4096 | bfloat16 | 112 | data-parallel | 1048.2 | 789.5 | opus | 1007.1 | 881.1 | 160x192x128 |
| 3840x512x4096 | bfloat16 | 30 | data-parallel | 581.7 | 226.6 | flydsl | 534.0 | 480.8 | 128x128x256 |
| 3840x1024x4096 | bfloat16 | 60 | data-parallel | 851.1 | 447.7 | flydsl | 738.5 | 700.9 | 128x128x256 |
| 3840x2048x4096 | bfloat16 | 120 | data-parallel | 1108.3 | 838.1 | opus | 1045.9 | 731.3 | 256x240x64 |
| 4096x512x4096 | bfloat16 | 32 | one-tile owner, n = 2 | 590.2 | 242.1 | flydsl | 589.8 | 494.1 | 128x128x256 |
| 4096x1024x4096 | bfloat16 | 64 | one-tile owner, n = 2 | 823.2 | 472.3 | flydsl | 824.3 | 732.2 | 128x128x256 |
| 4096x2048x4096 | bfloat16 | 128 | one-tile owner, n = 2 | 1008.0 | 878.0 | opus | 1086.8 | 698.7 | 256x256x64 |
| 4096x4096x4096 | bfloat16 | 256 | data-parallel | 1383.2 | 1384.1 | flydsl | 1309.7 | 1322.3 | 256x256x64 |
| 4096x8192x4096 | bfloat16 | 512 | data-parallel | 1423.8 | 1418.0 | asm | 1353.9 | 1362.8 | 256x256x64 |
| 4096x8704x4096 | bfloat16 | 544 | one-tile reduce-scatter, n = 4 | 1322.2 | 1142.4 | torch | 1184.0 | 1169.8 | 192x256x64 |
| 4096x10240x4096 | bfloat16 | 640 | one-tile owner, n = 2 | 1365.1 | 1324.8 | flydsl | 1274.7 | 1217.1 | 224x256x64 |
| 4096x16384x4096 | bfloat16 | 1024 | data-parallel | 1447.2 | 1442.9 | flydsl | 1359.9 | 1390.6 | 256x256x64 |
| 4096x129280x4096 | bfloat16 | 8080 | data-parallel | 1478.0 | 1466.4 | asm | 1396.2 | 1394.4 | 256x256x64 |
| 4352x512x4096 | bfloat16 | 34 | data-parallel | 523.9 | 253.7 | flydsl | 469.7 | 525.4 | 128x80x128 |
| 4352x1024x4096 | bfloat16 | 68 | data-parallel | 677.1 | 496.3 | opus | 640.7 | 804.2 | 128x160x128 |
| 4352x2048x4096 | bfloat16 | 136 | data-parallel | 925.2 | 933.7 | torch | 987.1 | 987.7 | 192x192x64 |
| 5120x512x4096 | bfloat16 | 40 | data-parallel | 607.6 | 299.5 | flydsl | 536.1 | 600.4 | 128x80x128 |
| 5120x1024x4096 | bfloat16 | 80 | data-parallel | 782.8 | 580.4 | opus | 768.9 | 882.7 | 128x160x128 |
| 5120x2048x4096 | bfloat16 | 160 | data-parallel | 1056.3 | 1063.7 | torch | 1144.0 | 1142.8 | 256x160x64 |
| 6144x512x4096 | bfloat16 | 48 | data-parallel | 714.7 | 359.4 | flydsl | 681.0 | 664.6 | 128x96x128 |
| 6144x1024x4096 | bfloat16 | 96 | data-parallel | 916.4 | 683.7 | opus | 927.3 | 917.5 | 128x192x128 |
| 6144x2048x4096 | bfloat16 | 192 | data-parallel | 1220.8 | 1238.1 | opus | 1220.1 | 1197.6 | 256x192x64 |
| 6144x129280x4096 | bfloat16 | 12120 | one-tile owner, n = 2-3 | 1471.8 | 1463.2 | asm | 1403.8 | 1405.9 | 256x256x64 |
| 6400x512x4096 | bfloat16 | 50 | data-parallel | 732.3 | 374.8 | flydsl | 512.2 | 608.6 | 128x128x256 |
| 6400x1024x4096 | bfloat16 | 100 | data-parallel | 934.4 | 713.4 | opus | 888.0 | 763.6 | 256x208x64 |
| 6400x2048x4096 | bfloat16 | 200 | data-parallel | 1245.2 | 1253.1 | torch | 1106.6 | 1109.6 | 256x208x64 |
| 6912x512x4096 | bfloat16 | 54 | data-parallel | 772.5 | 400.6 | flydsl | 625.3 | 645.4 | 128x128x256 |
| 6912x1024x4096 | bfloat16 | 108 | data-parallel | 999.9 | 760.9 | opus | 935.3 | 688.5 | 256x224x64 |
| 6912x2048x4096 | bfloat16 | 216 | data-parallel | 1283.0 | 1295.1 | torch | 1193.1 | 1201.0 | 256x224x64 |
| 7168x512x4096 | bfloat16 | 56 | data-parallel | 792.5 | 412.8 | flydsl | 621.1 | 662.6 | 128x128x256 |
| 7168x1024x4096 | bfloat16 | 112 | data-parallel | 1029.3 | 781.0 | opus | 1006.1 | 1054.9 | 256x128x64 |
| 7168x2048x4096 | bfloat16 | 224 | data-parallel | 1310.9 | 1317.8 | torch | 1221.4 | 1226.8 | 256x224x64 |
| 7424x512x4096 | bfloat16 | 58 | data-parallel | 818.4 | 427.1 | flydsl | 654.9 | 654.9 | 128x128x256 |
| 7424x1024x4096 | bfloat16 | 116 | data-parallel | 1057.2 | 806.3 | opus | 983.8 | 738.6 | 256x240x64 |
| 7424x2048x4096 | bfloat16 | 232 | data-parallel | 1325.7 | 1338.1 | torch | 1124.4 | 1128.6 | 256x240x64 |
| 7680x512x4096 | bfloat16 | 60 | data-parallel | 836.5 | 441.4 | flydsl | 623.4 | 672.4 | 128x128x256 |
| 7680x1024x4096 | bfloat16 | 120 | data-parallel | 1087.0 | 828.4 | opus | 1035.8 | 749.8 | 256x240x64 |
| 7680x2048x4096 | bfloat16 | 240 | data-parallel | 1344.7 | 1360.5 | torch | 1144.0 | 1150.8 | 256x240x64 |
| 7936x512x4096 | bfloat16 | 62 | data-parallel | 853.6 | 455.8 | flydsl | 660.5 | 690.4 | 128x128x256 |
| 7936x1024x4096 | bfloat16 | 124 | data-parallel | 1117.6 | 852.1 | opus | 1019.9 | 707.1 | 256x256x64 |
| 7936x2048x4096 | bfloat16 | 248 | data-parallel | 1352.2 | 1359.6 | torch | 1328.8 | 1335.7 | 256x256x64 |
| 8192x512x4096 | bfloat16 | 64 | one-tile owner, n = 2 | 792.2 | 467.6 | flydsl | 785.6 | 706.5 | 128x128x256 |
| 8192x1024x4096 | bfloat16 | 128 | one-tile owner, n = 2 | 998.1 | 869.1 | opus | 1082.4 | 675.4 | 256x256x64 |
| 8192x2048x4096 | bfloat16 | 256 | data-parallel | 1379.7 | 1374.2 | torch | 1350.0 | 1353.9 | 256x256x64 |
| 8192x4096x4096 | bfloat16 | 512 | data-parallel | 1426.7 | 1428.3 | flydsl | 1348.3 | 1394.8 | 256x256x64 |
| 8192x8192x4096 | bfloat16 | 1024 | data-parallel | 1465.9 | 1453.6 | asm | 1402.8 | 1415.1 | 256x256x64 |
| 8192x8704x4096 | bfloat16 | 1088 | one-tile owner, n = 2 | 1386.9 | 1317.1 | asm | 1273.4 | 1265.8 | 256x224x64 |
| 8192x10240x4096 | bfloat16 | 1280 | data-parallel | 1469.4 | 1466.9 | torch | 1413.5 | 1422.2 | 256x256x64 |
| 8192x16384x4096 | bfloat16 | 2048 | data-parallel | 1482.7 | 1471.4 | torch | 1418.8 | 1423.5 | 256x256x64 |
| 8192x129280x4096 | bfloat16 | 16160 | one-tile reduce-scatter, n = 4 | 1478.3 | 1477.7 | flydsl | 1409.6 | 1400.3 | 256x256x64 |
| 10240x512x4096 | bfloat16 | 80 | data-parallel | 767.9 | 570.0 | torch | 860.0 | 857.6 | 128x160x128 |
| 10240x1024x4096 | bfloat16 | 160 | data-parallel | 1093.4 | 1095.7 | torch | 1140.0 | 1134.9 | 256x160x64 |
| 10240x2048x4096 | bfloat16 | 320 | one-tile owner, n = 2 | 1202.1 | 1075.5 | torch | 1172.8 | 1182.1 | 256x160x64 |
| 10240x129280x4096 | bfloat16 | 20200 | data-parallel | 1465.8 | 1464.7 | asm | 1426.2 | 1387.9 | 256x256x64 |
| 12288x512x4096 | bfloat16 | 96 | data-parallel | 873.9 | 673.0 | opus | 895.0 | 877.0 | 128x192x128 |
| 12288x1024x4096 | bfloat16 | 192 | data-parallel | 1191.6 | 1204.9 | opus | 1248.0 | 1204.8 | 256x192x64 |
| 12288x2048x4096 | bfloat16 | 384 | one-tile owner, n = 2 | 1309.3 | 1254.9 | opus | 1272.3 | 1243.1 | 256x192x64 |
| 12288x129280x4096 | bfloat16 | 24240 | data-parallel | 1498.8 | 1489.6 | asm | 1432.3 | 1382.3 | 256x256x64 |
| 14336x512x4096 | bfloat16 | 112 | data-parallel | 957.0 | 765.3 | opus | 965.8 | 668.3 | 256x224x64 |
| 14336x1024x4096 | bfloat16 | 224 | data-parallel | 1290.9 | 1296.7 | torch | 1238.9 | 1240.0 | 256x224x64 |
| 14336x2048x4096 | bfloat16 | 448 | data-parallel | 1344.6 | 1338.9 | flydsl | 1295.3 | 1280.9 | 256x224x64 |
| 14336x129280x4096 | bfloat16 | 28280 | one-tile owner, n = 2-3 | 1480.2 | 1468.9 | asm | 1415.7 | 1388.3 | 256x256x64 |
| 16384x512x4096 | bfloat16 | 128 | one-tile owner, n = 2 | 999.0 | 851.9 | opus | 1034.8 | 789.4 | 256x256x64 |
| 16384x1024x4096 | bfloat16 | 256 | data-parallel | 1390.9 | 1384.6 | flydsl | 1294.4 | 1301.0 | 256x256x64 |
| 16384x2048x4096 | bfloat16 | 512 | data-parallel | 1428.2 | 1419.0 | torch | 1373.5 | 1379.2 | 256x256x64 |
| 16384x16384x4096 | bfloat16 | 4096 | data-parallel | 1497.7 | 1490.1 | torch | 1431.6 | 1437.4 | 256x256x64 |
| 16384x129280x4096 | bfloat16 | 32320 | one-tile owner, n = 2 | 1503.7 | 1478.2 | asm | 1407.8 | 1400.8 | 256x256x64 |
| 32768x16384x4096 | bfloat16 | 8192 | data-parallel | 1484.7 | 1475.1 | flydsl | 1400.2 | 1432.4 | 256x256x64 |
| 512x7168x4224 | bfloat16 | 56 | data-parallel | 770.1 | 379.0 | flydsl | 729.6 | 725.5 | 128x128x128 |
| 1024x7168x4224 | bfloat16 | 112 | data-parallel | 933.6 | 745.2 | flydsl | 934.6 | 789.7 | 240x256x64 |
| 10240x7168x4224 | bfloat16 | 1120 | one-tile owner, n = 2-3 | 1407.6 | 1334.5 | opus | 909.2 | 1256.7 | 224x256x64 |
| 512x6144x4608 | bfloat16 | 48 | data-parallel | 707.4 | 339.7 | flydsl | 682.4 | 612.3 | 192x128x128 |
| 1024x6144x4608 | bfloat16 | 96 | data-parallel | 867.9 | 666.8 | torch | 868.6 | 872.3 | 192x128x128 |
| 2048x6144x4608 | bfloat16 | 192 | data-parallel | 1227.0 | 1233.2 | flydsl | 1180.1 | 1200.0 | 192x256x64 |
| 4096x6144x4608 | bfloat16 | 384 | one-tile owner, n = 2 | 1316.6 | 1264.6 | torch | 1213.7 | 1221.9 | 192x256x64 |
| 8192x6144x4608 | bfloat16 | 768 | data-parallel | 1446.0 | 1438.0 | torch | 1385.9 | 1400.7 | 256x256x64 |
| 16384x6144x4608 | bfloat16 | 1536 | data-parallel | 1469.9 | 1459.3 | torch | 1418.9 | 1428.9 | 256x256x64 |
| 32768x6144x4608 | bfloat16 | 3072 | data-parallel | 1492.2 | 1478.1 | torch | 1441.7 | 1442.4 | 256x256x64 |
| 512x1024x5120 | float32 | 8 | one-tile owner, n = 2 | 273.7 | 51.9 | asm | 249.0 | 275.6 | 64x32x256 |
| 512x5120x5120 | bfloat16 | 40 | data-parallel | 565.4 | 270.5 | torch | 650.5 | 653.3 | 96x128x128 |
| 512x6400x5120 | bfloat16 | 50 | data-parallel | 693.1 | 339.9 | flydsl | 609.4 | 563.0 | 128x128x256 |
| 512x12800x5120 | bfloat16 | 100 | data-parallel | 847.5 | 659.3 | flydsl | 662.5 | 809.9 | 208x256x64 |
| 512x37888x5120 | bfloat16 | 296 | one-tile owner, n = 3 | 1152.1 | 935.6 | torch | 1048.4 | 1053.8 | 160x256x64 |
| 512x51200x5120 | bfloat16 | 400 | data-parallel | 1193.0 | 1199.4 | flydsl | 1134.3 | 1087.0 | 208x256x64 |
| 768x1024x5120 | float32 | 12 | data-parallel | 316.0 | 78.5 | asm | 312.6 | 314.3 | 64x64x256 |
| 768x37888x5120 | bfloat16 | 444 | data-parallel | 1268.8 | 1303.0 | torch | 1208.8 | 1215.8 | 224x256x64 |
| 1024x1024x5120 | float32 | 16 | one-tile owner, n = 2 | 406.4 | 105.6 | asm | 334.2 | 404.1 | 64x64x256 |
| 1024x5120x5120 | bfloat16 | 80 | data-parallel | 731.9 | 537.9 | torch | 844.4 | 846.9 | 160x128x128 |
| 1024x6400x5120 | bfloat16 | 100 | data-parallel | 876.0 | 670.5 | flydsl | 820.3 | 801.4 | 256x208x64 |
| 1024x12800x5120 | bfloat16 | 200 | data-parallel | 1229.8 | 1229.7 | flydsl | 727.7 | 1059.0 | 256x208x64 |
| 1024x37888x5120 | bfloat16 | 592 | one-tile owner, n = 2 | 1313.0 | 1195.6 | torch | 1060.0 | 1064.3 | 256x208x64 |
| 1024x51200x5120 | bfloat16 | 800 | one-tile reduce-scatter, n = 4 | 1357.5 | 1195.8 | asm | 1141.4 | 1071.9 | 256x208x64 |
| 1536x1024x5120 | float32 | 24 | data-parallel | 447.0 | 158.7 | asm | 438.9 | 511.8 | 64x96x128 |
| 1536x37888x5120 | bfloat16 | 888 | one-tile owner, n = 2-3 | 1400.7 | 1316.3 | torch | 1228.9 | 1232.0 | 224x256x64 |
| 2048x1024x5120 | float32 | 32 | one-tile owner, n = 2 | 598.3 | 211.3 | opus | 432.1 | 559.2 | 128x128x128 |
| 2048x5120x5120 | bfloat16 | 160 | data-parallel | 1052.9 | 1049.4 | torch | 1129.2 | 1141.8 | 160x256x64 |
| 2048x6400x5120 | bfloat16 | 200 | data-parallel | 1249.1 | 1253.2 | flydsl | 1130.3 | 1079.5 | 256x208x64 |
| 2048x12800x5120 | bfloat16 | 400 | data-parallel | 1290.2 | 1289.1 | flydsl | 812.5 | 1071.8 | 256x208x64 |
| 2048x37888x5120 | bfloat16 | 1184 | data-parallel | 1436.2 | 1424.5 | torch | 1188.3 | 1193.6 | 240x256x64 |
| 2048x51200x5120 | bfloat16 | 1600 | one-tile owner, n = 2 | 1436.2 | 1372.7 | asm | 212.0 | 1171.0 | 240x256x64 |
| 4096x1024x5120 | float32 | 64 | one-tile owner, n = 2 | 789.3 | 416.6 | opus | 713.7 | 768.0 | 128x128x128 |
| 4096x5120x5120 | bfloat16 | 320 | one-tile owner, n = 2 | 1245.3 | 1076.5 | torch | 1155.6 | 1164.5 | 160x256x64 |
| 4096x6400x5120 | bfloat16 | 400 | data-parallel | 1307.1 | 1299.9 | flydsl | 1245.3 | 1103.2 | 256x208x64 |
| 4096x12800x5120 | bfloat16 | 800 | one-tile reduce-scatter, n = 4 | 1408.0 | 1242.9 | asm | 1197.4 | 1155.9 | 256x208x64 |
| 4096x51200x5120 | bfloat16 | 3200 | one-tile owner, n = 2 | 1484.8 | 1460.6 | asm | 1384.4 | 1381.2 | 256x256x64 |
| 8192x1024x5120 | float32 | 128 | one-tile owner, n = 2 | 1008.1 | 798.4 | opus | 1023.6 | 985.4 | 256x128x64 |
| 8192x5120x5120 | bfloat16 | 640 | one-tile owner, n = 2 | 1402.5 | 1336.5 | torch | 1252.2 | 1257.4 | 256x224x64 |
| 8192x6400x5120 | bfloat16 | 800 | one-tile reduce-scatter, n = 4 | 1400.8 | 1238.0 | flydsl | 1195.0 | 1167.3 | 256x208x64 |
| 8192x12800x5120 | bfloat16 | 1600 | one-tile owner, n = 2 | 1439.2 | 1379.0 | asm | 1329.4 | 1180.0 | 256x240x64 |
| 8192x51200x5120 | bfloat16 | 6400 | data-parallel | 1520.7 | 1514.5 | flydsl | 1438.4 | 1417.3 | 256x256x64 |
| 16384x1024x5120 | float32 | 256 | data-parallel | 1322.5 | 1298.0 | opus | 1146.1 | 1195.5 | 256x256x64 |
| 16384x5120x5120 | bfloat16 | 1280 | data-parallel | 1480.8 | 1464.1 | torch | 1432.9 | 1450.4 | 256x256x64 |
| 16384x6400x5120 | bfloat16 | 1600 | one-tile owner, n = 2 | 1448.7 | 1374.0 | flydsl | 1319.0 | 1197.2 | 256x240x64 |
| 16384x12800x5120 | bfloat16 | 3200 | one-tile owner, n = 2 | 1516.5 | 1474.4 | torch | 1454.0 | 1463.1 | 256x256x64 |
| 16384x51200x5120 | bfloat16 | 12800 | data-parallel | 1527.7 | 1516.1 | flydsl | 1440.6 | 1418.3 | 256x256x64 |
| 32768x5120x5120 | bfloat16 | 2560 | data-parallel | 1515.9 | 1504.3 | torch | 1467.0 | 1478.2 | 256x256x64 |
| 32768x6400x5120 | bfloat16 | 3200 | one-tile owner, n = 2 | 1510.7 | 1467.4 | torch | 1461.7 | 1474.3 | 256x256x64 |
| 32768x12800x5120 | bfloat16 | 6400 | data-parallel | 1546.5 | 1513.4 | torch | 1481.3 | 1502.1 | 256x256x64 |
| 32768x51200x5120 | bfloat16 | 25600 | data-parallel | 1524.1 | 1502.8 | flydsl | 1421.1 | 1407.6 | 256x256x64 |
| 512x1280x6144 | bfloat16 | 10 | data-parallel | 289.2 | 67.7 | flydsl | 341.5 | 354.1 | 160x64x128 |
| 512x1536x6144 | bfloat16 | 12 | data-parallel | 343.6 | 81.7 | flydsl | 384.0 | 379.3 | 192x64x128 |
| 512x2048x6144 | float32 | 16 | one-tile owner, n = 2 | 451.1 | 105.6 | asm | 343.8 | 423.9 | 64x64x256 |
| 512x2304x6144 | bfloat16 | 18 | data-parallel | 374.0 | 122.2 | flydsl | 439.8 | 371.9 | 192x128x128 |
| 512x2560x6144 | bfloat16 | 20 | data-parallel | 416.0 | 136.3 | torch | 473.9 | 473.0 | 256x128x64 |
| 512x3072x6144 | float32 | 24 | data-parallel | 463.3 | 158.3 | opus | 402.9 | 554.0 | 96x128x128 |
| 512x3072x6144 | bfloat16 | 24 | data-parallel | 468.6 | 163.2 | flydsl | 549.1 | 464.7 | 192x128x128 |
| 512x6144x6144 | bfloat16 | 48 | data-parallel | 727.2 | 328.8 | flydsl | 690.2 | 684.2 | 192x128x128 |
| 512x9216x6144 | bfloat16 | 72 | spread | 795.7 | 482.4 | flydsl | 601.3 | 589.9 | 256x256x64 |
| 1024x2048x6144 | float32 | 32 | one-tile owner, n = 2 | 655.9 | 212.4 | opus | 408.7 | 634.9 | 128x128x128 |
| 1024x2304x6144 | bfloat16 | 36 | data-parallel | 571.7 | 246.0 | torch | 597.3 | 596.9 | 256x176x64 |
| 1024x2560x6144 | bfloat16 | 40 | data-parallel | 634.7 | 273.1 | torch | 685.6 | 682.6 | 160x128x128 |
| 1024x3072x6144 | float32 | 48 | data-parallel | 700.1 | 317.1 | opus | 500.8 | 721.7 | 96x128x128 |
| 1024x6144x6144 | bfloat16 | 96 | data-parallel | 848.6 | 652.0 | torch | 972.5 | 984.2 | 192x256x64 |
| 1024x9216x6144 | bfloat16 | 144 | spread | 1082.8 | 950.1 | torch | 1058.4 | 1060.6 | 144x256x64 |
| 2048x2048x6144 | float32 | 64 | one-tile owner, n = 2 | 841.0 | 428.5 | opus | 674.9 | 812.0 | 128x128x128 |
| 2048x2304x6144 | bfloat16 | 72 | spread | 837.2 | 502.0 | torch | 841.3 | 843.0 | 128x160x128 |
| 2048x2560x6144 | bfloat16 | 80 | data-parallel | 736.9 | 556.4 | torch | 912.6 | 912.0 | 160x128x128 |
| 2048x3072x6144 | float32 | 96 | data-parallel | 861.9 | 642.3 | opus | 758.0 | 887.9 | 192x256x64 |
| 2048x6144x6144 | bfloat16 | 192 | data-parallel | 1235.8 | 1237.3 | torch | 1220.3 | 1233.4 | 192x256x64 |
| 2048x9216x6144 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 | 1278.5 | 980.5 | torch | 1081.5 | 1087.0 | 144x256x64 |
| 4096x2048x6144 | float32 | 128 | one-tile owner, n = 2 | 1057.1 | 825.5 | opus | 1028.9 | 790.1 | 256x256x64 |
| 4096x2304x6144 | bfloat16 | 144 | spread | 1098.3 | 972.3 | flydsl | 683.2 | 1083.9 | 144x256x64 |
| 4096x2560x6144 | bfloat16 | 160 | data-parallel | 1071.2 | 1072.6 | flydsl | 637.2 | 1171.7 | 160x256x64 |
| 4096x3072x6144 | float32 | 192 | data-parallel | 1174.5 | 1188.0 | asm | 1146.9 | 1196.3 | 192x256x64 |
| 4096x6144x6144 | bfloat16 | 384 | one-tile owner, n = 2 | 1363.2 | 1283.4 | torch | 1241.5 | 1246.1 | 192x256x64 |
| 4096x9216x6144 | bfloat16 | 576 | one-tile owner, n = 3 | 1380.5 | 1231.6 | torch | 1247.8 | 1248.4 | 192x256x64 |
| 8192x2048x6144 | float32 | 256 | data-parallel | 1318.9 | 1316.6 | opus | 1212.7 | 1208.1 | 256x256x64 |
| 8192x2304x6144 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 | 1217.2 | 974.1 | torch | 1087.3 | 1095.5 | 144x256x64 |
| 8192x2560x6144 | bfloat16 | 320 | one-tile owner, n = 3 | 1274.2 | 1083.6 | asm | 408.6 | 1180.8 | 160x256x64 |
| 8192x3072x6144 | float32 | 384 | one-tile owner, n = 2 | 1287.2 | 1205.0 | asm | 1175.0 | 1207.4 | 192x256x64 |
| 8192x6144x6144 | bfloat16 | 768 | data-parallel | 1478.4 | 1471.4 | flydsl | 1408.4 | 1437.8 | 256x256x64 |
| 8192x9216x6144 | bfloat16 | 1152 | one-tile owner, n = 2 | 1469.1 | 1416.6 | asm | 1375.9 | 1188.7 | 256x240x64 |
| 16384x2048x6144 | float32 | 512 | data-parallel | 1349.9 | 1349.3 | opus | 1216.2 | 1258.3 | 256x256x64 |
| 16384x2304x6144 | bfloat16 | 576 | one-tile owner, n = 3 | 1368.8 | 1225.5 | flydsl | 973.5 | 1287.0 | 192x256x64 |
| 16384x2560x6144 | bfloat16 | 640 | one-tile owner, n = 2 | 1429.7 | 1348.0 | flydsl | 1293.8 | 1287.3 | 256x224x64 |
| 16384x3072x6144 | float32 | 768 | data-parallel | 1401.9 | 1394.2 | asm | 1361.5 | 1305.7 | 256x256x64 |
| 16384x6144x6144 | bfloat16 | 1536 | data-parallel | 1520.3 | 1502.8 | asm | 1448.4 | 1471.9 | 256x256x64 |
| 16384x9216x6144 | bfloat16 | 2304 | data-parallel | 1527.6 | 1513.4 | torch | 1462.9 | 1476.0 | 256x256x64 |
| 32768x2304x6144 | bfloat16 | 1152 | one-tile owner, n = 2 | 1466.7 | 1402.3 | flydsl | 649.3 | 1191.5 | 256x240x64 |
| 32768x2560x6144 | bfloat16 | 1280 | data-parallel | 1500.8 | 1493.2 | flydsl | 1433.1 | 1477.4 | 256x256x64 |
| 32768x6144x6144 | bfloat16 | 3072 | data-parallel | 1543.0 | 1529.3 | torch | 1482.2 | 1507.8 | 256x256x64 |
| 32768x9216x6144 | bfloat16 | 4608 | data-parallel | 1551.8 | 1537.9 | torch | 1484.2 | 1514.7 | 256x256x64 |
| 512x5120x6400 | bfloat16 | 40 | data-parallel | 603.5 | 281.0 | flydsl | 544.8 | 676.3 | 160x128x128 |
| 1024x5120x6400 | bfloat16 | 80 | data-parallel | 768.4 | 562.2 | flydsl | 594.0 | 891.7 | 160x128x128 |
| 2048x5120x6400 | bfloat16 | 160 | data-parallel | 1086.8 | 1092.9 | flydsl | 730.0 | 1172.3 | 160x256x64 |
| 4096x5120x6400 | bfloat16 | 320 | one-tile owner, n = 3 | 1300.5 | 1104.3 | torch | 1183.2 | 1189.2 | 160x256x64 |
| 8192x5120x6400 | bfloat16 | 640 | one-tile owner, n = 2 | 1428.9 | 1360.2 | flydsl | 1311.1 | 1270.9 | 256x224x64 |
| 16384x5120x6400 | bfloat16 | 1280 | data-parallel | 1497.0 | 1488.7 | torch | 1446.5 | 1454.6 | 256x256x64 |
| 32768x5120x6400 | bfloat16 | 2560 | data-parallel | 1518.4 | 1508.9 | torch | 1464.2 | 1461.1 | 256x256x64 |
| 512x16384x6656 | bfloat16 | 128 | one-tile owner, n = 2 | 1150.7 | 878.9 | flydsl | 845.2 | 925.2 | 256x256x64 |
| 1024x16384x6656 | bfloat16 | 256 | data-parallel | 1438.3 | 1450.2 | asm | 364.4 | 1388.5 | 256x256x64 |
| 2048x16384x6656 | bfloat16 | 512 | data-parallel | 1506.0 | 1474.1 | asm | 1386.1 | 1424.2 | 256x256x64 |
| 4096x16384x6656 | bfloat16 | 1024 | data-parallel | 1494.0 | 1488.5 | torch | 1432.3 | 1433.2 | 256x256x64 |
| 8192x16384x6656 | bfloat16 | 2048 | data-parallel | 1509.7 | 1507.3 | asm | 1422.7 | 1452.0 | 256x256x64 |
| 16384x16384x6656 | bfloat16 | 4096 | data-parallel | 1541.4 | 1524.0 | asm | 1462.6 | 1492.3 | 256x256x64 |
| 32768x16384x6656 | bfloat16 | 8192 | data-parallel | 1553.0 | 1532.9 | asm | 291.4 | 1485.7 | 256x256x64 |
| 512x1024x7168 | bfloat16 | 8 | one-tile owner, n = 2 | 328.8 | 57.2 | torch | 383.3 | 384.5 | 128x128x64 |
| 512x1536x7168 | bfloat16 | 12 | data-parallel | 355.9 | 86.1 | torch | 411.8 | 415.9 | 192x64x128 |
| 512x2048x7168 | bfloat16 | 16 | one-tile owner, n = 2 | 482.5 | 114.6 | torch | 540.3 | 541.8 | 128x128x128 |
| 512x3584x7168 | bfloat16 | 28 | data-parallel | 556.8 | 202.9 | flydsl | 560.1 | 540.8 | 128x128x256 |
| 512x6144x7168 | bfloat16 | 48 | data-parallel | 741.3 | 350.2 | flydsl | 721.0 | 710.0 | 192x128x128 |
| 512x8192x7168 | bfloat16 | 64 | one-tile owner, n = 2 | 890.0 | 461.7 | flydsl | 712.3 | 722.0 | 128x128x256 |
| 512x8448x7168 | bfloat16 | 66 | spread | 834.7 | 475.8 | flydsl | 671.1 | 478.7 | 256x256x64 |
| 512x20480x7168 | bfloat16 | 160 | spread | 1108.3 | 1066.0 | flydsl | 1029.0 | 1147.2 | 160x256x64 |
| 768x1024x7168 | bfloat16 | 12 | data-parallel | 353.3 | 86.6 | torch | 404.2 | 404.9 | 256x96x64 |
| 768x2048x7168 | bfloat16 | 24 | data-parallel | 505.0 | 174.2 | flydsl | 553.7 | 572.1 | 256x128x64 |
| 1024x512x7168 | bfloat16 | 8 | one-tile owner, n = 2 | 325.3 | 57.6 | flydsl | 331.5 | 291.0 | 256x64x64 |
| 1024x1024x7168 | bfloat16 | 16 | one-tile owner, n = 2 | 480.6 | 116.1 | torch | 550.7 | 548.6 | 128x128x128 |
| 1024x1536x7168 | bfloat16 | 24 | data-parallel | 511.8 | 174.4 | flydsl | 561.2 | 571.8 | 256x128x64 |
| 1024x2048x7168 | bfloat16 | 32 | one-tile owner, n = 2 | 731.4 | 232.3 | torch | 726.7 | 729.6 | 256x128x64 |
| 1024x3584x7168 | bfloat16 | 56 | data-parallel | 867.9 | 407.7 | flydsl | 776.8 | 681.3 | 128x128x256 |
| 1024x6144x7168 | bfloat16 | 96 | data-parallel | 904.0 | 692.3 | torch | 969.1 | 943.1 | 192x256x64 |
| 1024x8192x7168 | bfloat16 | 128 | one-tile owner, n = 2 | 1159.6 | 901.4 | flydsl | 1010.0 | 903.1 | 256x256x64 |
| 1024x8448x7168 | bfloat16 | 132 | spread | 1099.1 | 931.0 | torch | 1031.3 | 1038.2 | 144x256x64 |
| 1024x20480x7168 | bfloat16 | 320 | one-tile owner, n = 3 | 1294.2 | 1066.6 | flydsl | 750.8 | 1177.1 | 160x256x64 |
| 1280x512x7168 | bfloat16 | 10 | data-parallel | 293.6 | 72.0 | torch | 364.0 | 363.7 | 256x80x64 |
| 1280x1024x7168 | bfloat16 | 20 | data-parallel | 419.0 | 145.9 | torch | 529.7 | 543.0 | 256x80x128 |
| 1280x2048x7168 | bfloat16 | 40 | data-parallel | 639.0 | 292.6 | torch | 737.7 | 735.5 | 128x160x128 |
| 1536x512x7168 | bfloat16 | 12 | data-parallel | 345.6 | 87.0 | torch | 403.6 | 403.7 | 256x96x64 |
| 1536x1024x7168 | bfloat16 | 24 | data-parallel | 503.4 | 175.7 | torch | 645.1 | 644.8 | 128x96x128 |
| 1536x2048x7168 | bfloat16 | 48 | data-parallel | 761.2 | 355.7 | torch | 830.3 | 832.5 | 128x192x64 |
| 1792x512x7168 | bfloat16 | 14 | data-parallel | 397.0 | 100.9 | torch | 438.7 | 439.2 | 256x112x64 |
| 1792x1024x7168 | bfloat16 | 28 | data-parallel | 573.6 | 207.9 | flydsl | 607.2 | 603.6 | 256x128x64 |
| 1792x2048x7168 | bfloat16 | 56 | data-parallel | 860.9 | 412.6 | flydsl | 782.0 | 687.0 | 128x128x256 |
| 2048x512x7168 | bfloat16 | 16 | one-tile owner, n = 2 | 476.5 | 117.1 | torch | 504.8 | 505.0 | 128x128x64 |
| 2048x1024x7168 | bfloat16 | 32 | one-tile owner, n = 2 | 728.5 | 236.7 | torch | 710.5 | 714.0 | 128x128x128 |
| 2048x2048x7168 | bfloat16 | 64 | one-tile owner, n = 2 | 934.2 | 473.3 | flydsl | 873.6 | 766.1 | 128x128x256 |
| 2048x3584x7168 | bfloat16 | 112 | data-parallel | 1066.5 | 807.8 | opus | 1002.4 | 993.2 | 224x256x64 |
| 2048x8192x7168 | bfloat16 | 256 | data-parallel | 1453.8 | 1450.9 | asm | 1374.9 | 1403.3 | 256x256x64 |
| 2048x20480x7168 | bfloat16 | 640 | one-tile owner, n = 2 | 1454.5 | 1375.3 | flydsl | 833.7 | 1254.1 | 224x256x64 |
| 4096x512x7168 | bfloat16 | 32 | one-tile owner, n = 2 | 708.6 | 234.8 | flydsl | 629.4 | 612.6 | 128x128x256 |
| 4096x1024x7168 | bfloat16 | 64 | one-tile owner, n = 2 | 928.2 | 468.1 | torch | 904.3 | 911.3 | 256x128x64 |
| 4096x2048x7168 | bfloat16 | 128 | one-tile owner, n = 2 | 1187.6 | 909.5 | torch | 1167.0 | 1185.7 | 256x128x64 |
| 4096x3584x7168 | bfloat16 | 224 | data-parallel | 1383.5 | 1389.5 | flydsl | 1324.4 | 1276.7 | 224x256x64 |
| 4096x8192x7168 | bfloat16 | 512 | data-parallel | 1488.6 | 1480.7 | flydsl | 1407.7 | 1432.6 | 256x256x64 |
| 4096x20480x7168 | bfloat16 | 1280 | data-parallel | 1512.3 | 1509.6 | flydsl | 1437.4 | 1451.3 | 256x256x64 |
| 6144x512x7168 | bfloat16 | 48 | data-parallel | 732.2 | 347.3 | flydsl | 726.1 | 646.4 | 128x192x128 |
| 6144x1024x7168 | bfloat16 | 96 | data-parallel | 906.6 | 687.9 | opus | 971.3 | 844.4 | 256x192x64 |
| 6144x2048x7168 | bfloat16 | 192 | data-parallel | 1257.5 | 1262.2 | opus | 1282.3 | 1257.2 | 256x192x64 |
| 8192x512x7168 | bfloat16 | 64 | one-tile owner, n = 2 | 902.9 | 461.9 | flydsl | 842.7 | 755.3 | 128x128x256 |
| 8192x1024x7168 | bfloat16 | 128 | one-tile owner, n = 2 | 1169.4 | 897.5 | torch | 1091.5 | 1094.7 | 256x128x64 |
| 8192x2048x7168 | bfloat16 | 256 | data-parallel | 1462.5 | 1466.5 | torch | 1379.5 | 1392.1 | 256x256x64 |
| 8192x3584x7168 | bfloat16 | 448 | data-parallel | 1413.8 | 1402.5 | flydsl | 1357.3 | 1309.4 | 224x256x64 |
| 8192x8192x7168 | bfloat16 | 1024 | data-parallel | 1508.8 | 1502.0 | flydsl | 1428.8 | 1464.5 | 256x256x64 |
| 8192x20480x7168 | bfloat16 | 2560 | data-parallel | 1543.2 | 1529.3 | asm | 1468.4 | 1471.6 | 256x256x64 |
| 10240x512x7168 | bfloat16 | 80 | spread | 913.0 | 566.5 | torch | 859.3 | 807.4 | 256x160x64 |
| 10240x1024x7168 | bfloat16 | 160 | spread | 1164.9 | 1114.4 | torch | 1200.0 | 1212.8 | 256x160x64 |
| 10240x2048x7168 | bfloat16 | 320 | one-tile owner, n = 3 | 1331.7 | 1095.0 | torch | 1233.8 | 1248.3 | 256x160x64 |
| 10240x8448x7168 | bfloat16 | 1320 | one-tile owner, n = 4 | 1464.3 | 1342.5 | asm | 1291.3 | 1313.2 | 256x224x64 |
| 10240x20480x7168 | bfloat16 | 3200 | one-tile owner, n = 2 | 1511.9 | 1463.5 | asm | 1432.6 | 1437.0 | 256x256x64 |
| 12288x512x7168 | bfloat16 | 96 | data-parallel | 875.4 | 671.0 | opus | 934.9 | 849.4 | 256x192x64 |
| 12288x1024x7168 | bfloat16 | 192 | data-parallel | 1235.3 | 1238.4 | opus | 1334.3 | 1264.2 | 256x192x64 |
| 12288x2048x7168 | bfloat16 | 384 | one-tile owner, n = 2 | 1420.2 | 1291.0 | opus | 1333.3 | 1285.1 | 256x192x64 |
| 14336x512x7168 | bfloat16 | 112 | data-parallel | 1005.3 | 776.9 | opus | 999.2 | 898.2 | 256x224x64 |
| 14336x1024x7168 | bfloat16 | 224 | data-parallel | 1332.1 | 1340.7 | torch | 1307.8 | 1307.2 | 256x224x64 |
| 14336x2048x7168 | bfloat16 | 448 | data-parallel | 1394.0 | 1381.6 | flydsl | 1347.6 | 1342.8 | 256x224x64 |
| 16384x512x7168 | bfloat16 | 128 | one-tile owner, n = 2 | 1152.6 | 879.1 | opus | 1080.0 | 968.5 | 256x256x64 |
| 16384x1024x7168 | bfloat16 | 256 | data-parallel | 1469.3 | 1467.1 | torch | 1411.6 | 1415.7 | 256x256x64 |
| 16384x2048x7168 | bfloat16 | 512 | data-parallel | 1514.7 | 1500.4 | torch | 1464.5 | 1469.9 | 256x256x64 |
| 16384x3584x7168 | bfloat16 | 896 | one-tile owner, n = 2 | 1499.4 | 1405.1 | flydsl | 1365.0 | 1337.2 | 224x256x64 |
| 16384x8192x7168 | bfloat16 | 2048 | data-parallel | 1554.7 | 1536.9 | torch | 1484.8 | 1507.1 | 256x256x64 |
| 16384x20480x7168 | bfloat16 | 5120 | data-parallel | 1584.2 | 1554.8 | flydsl | 1476.8 | 1479.2 | 256x256x64 |
| 32768x512x7168 | bfloat16 | 256 | data-parallel | 1366.9 | 1373.9 | torch | 1355.3 | 1362.5 | 256x256x64 |
| 32768x1024x7168 | bfloat16 | 512 | data-parallel | 1484.3 | 1475.8 | torch | 1424.2 | 1433.9 | 256x256x64 |
| 32768x2048x7168 | bfloat16 | 1024 | data-parallel | 1536.3 | 1510.4 | torch | 1479.6 | 1489.6 | 256x256x64 |
| 32768x3584x7168 | bfloat16 | 1792 | data-parallel | 1541.2 | 1516.6 | torch | 1496.2 | 1508.6 | 256x256x64 |
| 32768x8192x7168 | bfloat16 | 4096 | data-parallel | 1576.1 | 1546.0 | torch | 1502.3 | 1537.1 | 256x256x64 |
| 32768x20480x7168 | bfloat16 | 10240 | data-parallel | 1556.6 | 1538.2 | asm | 1487.3 | 1439.2 | 256x256x64 |
| 512x1280x8192 | bfloat16 | 10 | data-parallel | 342.9 | 78.2 | torch | 419.7 | 420.8 | 160x64x128 |
| 512x2560x8192 | bfloat16 | 20 | data-parallel | 476.1 | 156.5 | torch | 555.5 | 555.4 | 256x128x64 |
| 512x4096x8192 | bfloat16 | 32 | one-tile owner, n = 2 | 752.6 | 248.6 | flydsl | 674.7 | 649.8 | 128x128x256 |
| 512x4608x8192 | bfloat16 | 36 | data-parallel | 611.1 | 282.2 | torch | 722.7 | 725.1 | 160x128x128 |
| 512x7168x8192 | bfloat16 | 56 | data-parallel | 883.6 | 433.0 | flydsl | 800.3 | 743.8 | 256x176x64 |
| 512x8192x8192 | bfloat16 | 64 | one-tile owner, n = 2 | 927.8 | 484.7 | flydsl | 687.0 | 773.6 | 128x128x256 |
| 512x10240x8192 | bfloat16 | 80 | spread | 874.8 | 604.0 | flydsl | 608.5 | 576.4 | 256x256x64 |
| 512x14336x8192 | bfloat16 | 112 | data-parallel | 1042.4 | 823.6 | torch | 991.4 | 1022.7 | 224x256x64 |
| 512x16384x8192 | bfloat16 | 128 | one-tile owner, n = 2 | 1187.2 | 893.1 | flydsl | 656.2 | 898.8 | 256x256x64 |
| 512x57344x8192 | bfloat16 | 448 | data-parallel | 1337.3 | 1336.9 | flydsl | 1281.1 | 1256.2 | 224x256x64 |
| 768x4096x8192 | bfloat16 | 48 | data-parallel | 816.4 | 374.3 | flydsl | 746.3 | 705.3 | 128x192x128 |
| 1024x512x8192 | bfloat16 | 8 | one-tile owner, n = 2 | 337.1 | 63.0 | torch | 331.0 | 333.7 | 256x64x64 |
| 1024x1280x8192 | bfloat16 | 20 | data-parallel | 463.9 | 157.7 | torch | 578.2 | 577.6 | 256x128x64 |
| 1024x2560x8192 | bfloat16 | 40 | data-parallel | 717.2 | 313.6 | torch | 776.7 | 785.2 | 160x128x128 |
| 1024x4096x8192 | bfloat16 | 64 | one-tile owner, n = 2 | 1022.1 | 492.9 | flydsl | 904.4 | 784.0 | 128x128x256 |
| 1024x4608x8192 | bfloat16 | 72 | spread | 868.8 | 552.5 | torch | 933.2 | 935.2 | 160x256x64 |
| 1024x7168x8192 | bfloat16 | 112 | data-parallel | 1095.4 | 834.8 | flydsl | 733.8 | 1039.5 | 224x256x64 |
| 1024x8192x8192 | bfloat16 | 128 | one-tile owner, n = 2 | 1210.7 | 927.5 | flydsl | 924.0 | 968.6 | 256x256x64 |
| 1024x10240x8192 | bfloat16 | 160 | spread | 1193.4 | 1127.0 | flydsl | 818.7 | 1184.5 | 160x256x64 |
| 1024x14336x8192 | bfloat16 | 224 | data-parallel | 1363.9 | 1372.3 | flydsl | 1308.3 | 1266.6 | 224x256x64 |
| 1024x16384x8192 | bfloat16 | 256 | data-parallel | 1477.9 | 1482.4 | torch | 1427.9 | 1448.3 | 256x256x64 |
| 1024x57344x8192 | bfloat16 | 896 | one-tile owner, n = 2 | 1487.7 | 1400.4 | asm | 1338.5 | 1307.0 | 224x256x64 |
| 2048x512x8192 | bfloat16 | 16 | one-tile owner, n = 2 | 521.9 | 126.2 | torch | 504.7 | 504.7 | 256x128x64 |
| 2048x1280x8192 | bfloat16 | 40 | data-parallel | 702.3 | 317.2 | torch | 685.2 | 691.6 | 320x192x64 |
| 2048x2560x8192 | bfloat16 | 80 | spread | 927.0 | 614.3 | torch | 943.7 | 949.0 | 160x256x64 |
| 2048x4096x8192 | bfloat16 | 128 | one-tile owner, n = 2 | 1232.2 | 940.1 | opus | 1126.1 | 984.6 | 256x256x64 |
| 2048x4608x8192 | bfloat16 | 144 | spread | 1183.0 | 1042.0 | torch | 1137.3 | 1146.9 | 144x256x64 |
| 2048x7168x8192 | bfloat16 | 224 | data-parallel | 1399.7 | 1400.3 | asm | 1344.5 | 1291.8 | 224x256x64 |
| 2048x8192x8192 | bfloat16 | 256 | data-parallel | 1510.2 | 1501.7 | torch | 1438.5 | 1446.3 | 256x256x64 |
| 2048x10240x8192 | bfloat16 | 320 | one-tile owner, n = 3 | 1366.7 | 1118.6 | flydsl | 929.0 | 1208.3 | 160x256x64 |
| 2048x14336x8192 | bfloat16 | 448 | data-parallel | 1420.9 | 1419.9 | flydsl | 1367.2 | 1304.0 | 224x256x64 |
| 2048x16384x8192 | bfloat16 | 512 | data-parallel | 1524.7 | 1516.8 | asm | 1428.5 | 1483.7 | 256x256x64 |
| 2048x57344x8192 | bfloat16 | 1792 | data-parallel | 1561.9 | 1554.0 | flydsl | 1493.2 | 1467.0 | 256x256x64 |
| 4096x512x8192 | bfloat16 | 32 | one-tile owner, n = 2 | 755.3 | 249.5 | flydsl | 648.4 | 664.6 | 128x128x256 |
| 4096x1280x8192 | bfloat16 | 80 | spread | 928.8 | 609.8 | torch | 963.9 | 945.9 | 160x256x64 |
| 4096x2560x8192 | bfloat16 | 160 | spread | 1222.1 | 1133.4 | torch | 1199.5 | 1208.8 | 160x256x64 |
| 4096x4096x8192 | bfloat16 | 256 | data-parallel | 1504.9 | 1496.6 | torch | 1441.6 | 1450.3 | 256x256x64 |
| 4096x4608x8192 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 | 1344.8 | 1015.8 | torch | 1159.7 | 1166.0 | 144x256x64 |
| 4096x7168x8192 | bfloat16 | 448 | data-parallel | 1429.9 | 1428.5 | flydsl | 1371.1 | 1310.9 | 224x256x64 |
| 4096x8192x8192 | bfloat16 | 512 | data-parallel | 1525.8 | 1514.1 | asm | 1450.4 | 1465.7 | 256x256x64 |
| 4096x10240x8192 | bfloat16 | 640 | one-tile owner, n = 2 | 1483.3 | 1394.3 | flydsl | 1349.0 | 1294.8 | 224x256x64 |
| 4096x14336x8192 | bfloat16 | 896 | one-tile owner, n = 2 | 1510.9 | 1433.0 | flydsl | 1386.4 | 1336.4 | 224x256x64 |
| 4096x16384x8192 | bfloat16 | 1024 | data-parallel | 1539.1 | 1531.4 | flydsl | 1459.3 | 1475.1 | 256x256x64 |
| 4096x57344x8192 | bfloat16 | 3584 | data-parallel | 1545.2 | 1555.3 | flydsl | 1477.8 | 1477.8 | 256x256x64 |
| 8192x512x8192 | bfloat16 | 64 | one-tile owner, n = 2 | 970.2 | 481.9 | flydsl | 872.8 | 799.4 | 128x128x256 |
| 8192x1280x8192 | bfloat16 | 160 | spread | 1198.0 | 1109.9 | torch | 1199.2 | 1210.7 | 160x256x64 |
| 8192x2560x8192 | bfloat16 | 320 | one-tile owner, n = 3 | 1367.8 | 1120.1 | torch | 1229.6 | 1235.7 | 160x256x64 |
| 8192x4096x8192 | bfloat16 | 512 | data-parallel | 1518.9 | 1527.1 | torch | 1485.8 | 1474.4 | 256x256x64 |
| 8192x4608x8192 | bfloat16 | 576 | one-tile owner, n = 3 | 1451.6 | 1268.6 | torch | 1294.3 | 1308.7 | 192x256x64 |
| 8192x7168x8192 | bfloat16 | 896 | one-tile owner, n = 2 | 1515.0 | 1433.9 | asm | 1390.9 | 1349.5 | 224x256x64 |
| 8192x8192x8192 | bfloat16 | 1024 | data-parallel | 1544.6 | 1534.6 | asm | 1478.7 | 1494.4 | 256x256x64 |
| 8192x10240x8192 | bfloat16 | 1280 | data-parallel | 1550.7 | 1539.4 | flydsl | 1469.2 | 1506.5 | 256x256x64 |
| 8192x14336x8192 | bfloat16 | 1792 | data-parallel | 1561.7 | 1553.1 | flydsl | 1504.5 | 1509.3 | 256x256x64 |
| 8192x16384x8192 | bfloat16 | 2048 | data-parallel | 1561.5 | 1548.8 | asm | 1453.8 | 1501.1 | 256x256x64 |
| 8192x57344x8192 | bfloat16 | 7168 | data-parallel | 1547.8 | 1553.2 | asm | 1497.4 | 1445.4 | 256x256x64 |
| 16384x512x8192 | bfloat16 | 128 | one-tile owner, n = 2 | 1181.0 | 899.3 | torch | 1012.1 | 1053.4 | 256x256x64 |
| 16384x1280x8192 | bfloat16 | 320 | one-tile owner, n = 3 | 1344.0 | 1114.4 | torch | 1229.5 | 1236.1 | 160x256x64 |
| 16384x2560x8192 | bfloat16 | 640 | one-tile owner, n = 2 | 1477.2 | 1368.9 | flydsl | 1334.9 | 1291.1 | 256x224x64 |
| 16384x4608x8192 | bfloat16 | 1152 | one-tile owner, n = 2 | 1525.8 | 1452.8 | flydsl | 1401.6 | 1203.6 | 256x240x64 |
| 16384x7168x8192 | bfloat16 | 1792 | data-parallel | 1575.3 | 1555.9 | torch | 1507.9 | 1523.4 | 256x256x64 |
| 16384x8192x8192 | bfloat16 | 2048 | data-parallel | 1583.5 | 1554.3 | torch | 1507.6 | 1543.4 | 256x256x64 |
| 16384x10240x8192 | bfloat16 | 2560 | data-parallel | 1568.7 | 1555.0 | torch | 1512.3 | 1535.8 | 256x256x64 |
| 16384x14336x8192 | bfloat16 | 3584 | data-parallel | 1571.5 | 1563.4 | asm | 1503.0 | 1512.3 | 256x256x64 |
| 16384x16384x8192 | bfloat16 | 4096 | data-parallel | 1559.8 | 1563.6 | asm | 1498.7 | 1513.4 | 256x256x64 |
| 16384x57344x8192 | bfloat16 | 14336 | data-parallel | 1537.9 | 1571.0 | asm | 1497.2 | 1426.7 | 256x256x64 |
| 32768x1280x8192 | bfloat16 | 640 | one-tile owner, n = 2 | 1444.2 | 1344.5 | flydsl | 1310.1 | 1317.5 | 256x224x64 |
| 32768x2560x8192 | bfloat16 | 1280 | data-parallel | 1536.3 | 1516.4 | torch | 1493.7 | 1502.6 | 256x256x64 |
| 32768x7168x8192 | bfloat16 | 3584 | data-parallel | 1581.6 | 1564.4 | torch | 1545.9 | 1539.2 | 256x256x64 |
| 32768x8192x8192 | bfloat16 | 4096 | data-parallel | 1581.2 | 1564.2 | flydsl | 1521.3 | 1551.1 | 256x256x64 |
| 32768x10240x8192 | bfloat16 | 5120 | data-parallel | 1564.7 | 1560.4 | torch | 1513.8 | 1545.0 | 256x256x64 |
| 32768x14336x8192 | bfloat16 | 7168 | data-parallel | 1570.8 | 1560.5 | asm | 1503.8 | 1507.4 | 256x256x64 |
| 32768x16384x8192 | bfloat16 | 8192 | data-parallel | 1545.7 | 1558.3 | asm | 1510.1 | 1475.3 | 256x256x64 |
| 32768x57344x8192 | bfloat16 | 28672 | data-parallel | 1530.9 | 1560.2 | asm | 1484.9 | 1418.1 | 256x256x64 |
| 512x6144x12288 | bfloat16 | 48 | data-parallel | 742.7 | 353.0 | torch | 792.0 | 789.7 | 192x128x128 |
| 1024x6144x12288 | bfloat16 | 96 | spread | 1060.7 | 699.1 | torch | 1076.0 | 1080.1 | 192x256x64 |
| 2048x6144x12288 | bfloat16 | 192 | spread | 1377.3 | 1301.8 | torch | 1273.3 | 1287.0 | 192x256x64 |
| 4096x6144x12288 | bfloat16 | 384 | one-tile owner, n = 2 | 1462.0 | 1334.5 | torch | 1278.3 | 1280.4 | 192x256x64 |
| 8192x6144x12288 | bfloat16 | 768 | data-parallel | 1557.3 | 1538.6 | torch | 1465.4 | 1467.5 | 256x256x64 |
| 16384x6144x12288 | bfloat16 | 1536 | data-parallel | 1572.3 | 1544.1 | torch | 1500.4 | 1481.6 | 256x256x64 |
| 32768x6144x12288 | bfloat16 | 3072 | data-parallel | 1577.7 | 1546.9 | asm | 1520.0 | 1516.9 | 256x256x64 |
| 512x16384x13312 | bfloat16 | 128 | one-tile owner, n = 2 | 1281.1 | 928.0 | flydsl | 716.0 | 1091.4 | 256x256x64 |
| 1024x16384x13312 | bfloat16 | 256 | data-parallel | 1518.3 | 1517.3 | asm | 1466.6 | 1476.2 | 256x256x64 |
| 2048x16384x13312 | bfloat16 | 512 | data-parallel | 1565.8 | 1557.3 | flydsl | 1014.6 | 1484.8 | 256x256x64 |
| 4096x16384x13312 | bfloat16 | 1024 | data-parallel | 1538.7 | 1534.7 | flydsl | 1473.8 | 1479.9 | 256x256x64 |
| 8192x16384x13312 | bfloat16 | 2048 | data-parallel | 1584.4 | 1550.3 | flydsl | 1506.2 | 1485.9 | 256x256x64 |
| 16384x16384x13312 | bfloat16 | 4096 | data-parallel | 1584.7 | 1568.4 | asm | 1503.7 | 1484.1 | 256x256x64 |
| 32768x16384x13312 | bfloat16 | 8192 | data-parallel | 1569.4 | 1568.6 | flydsl | 1500.6 | 1468.9 | 256x256x64 |
| 512x2304x16384 | bfloat16 | 18 | spread | 460.7 | 140.3 | torch | 709.8 | 708.2 | 256x128x64 |
| 512x4608x16384 | bfloat16 | 36 | spread | 698.8 | 280.1 | flydsl | 508.1 | 798.5 | 160x128x128 |
| 512x9216x16384 | bfloat16 | 72 | spread | 813.9 | 549.3 | flydsl | 637.1 | 763.1 | 256x256x64 |
| 512x13312x16384 | bfloat16 | 104 | spread | 1021.6 | 753.8 | flydsl | 544.7 | 1076.7 | 208x256x64 |
| 512x26624x16384 | bfloat16 | 208 | spread | 1267.2 | 1295.3 | flydsl | 774.0 | 1195.5 | 208x256x64 |
| 512x53248x16384 | bfloat16 | 416 | spread | 1337.7 | 1302.4 | flydsl | 650.7 | 1202.0 | 208x256x64 |
| 1024x2304x16384 | bfloat16 | 36 | spread | 705.8 | 279.2 | torch | 869.0 | 872.8 | 256x256x64 |
| 1024x4608x16384 | bfloat16 | 72 | spread | 867.6 | 553.7 | flydsl | 649.4 | 1018.8 | 144x256x64 |
| 1024x9216x16384 | bfloat16 | 144 | spread | 1218.2 | 1052.7 | flydsl | 670.1 | 1160.3 | 144x256x64 |
| 1024x13312x16384 | bfloat16 | 208 | spread | 1346.6 | 1355.8 | asm | 1323.8 | 1237.8 | 208x256x64 |
| 1024x26624x16384 | bfloat16 | 416 | spread | 1441.7 | 1369.8 | flydsl | 924.8 | 1249.4 | 208x256x64 |
| 1024x53248x16384 | bfloat16 | 832 | one-tile owner, n = 4 | 1499.2 | 1333.3 | triton | 996.2 | 1251.4 | 208x256x64 |
| 2048x2304x16384 | bfloat16 | 72 | spread | 873.2 | 554.5 | torch | 872.1 | 867.9 | 256x256x64 |
| 2048x4608x16384 | bfloat16 | 144 | spread | 1250.7 | 1053.2 | flydsl | 687.8 | 1187.7 | 144x256x64 |
| 2048x9216x16384 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 | 1415.2 | 1031.9 | flydsl | 911.7 | 1164.9 | 144x256x64 |
| 2048x13312x16384 | bfloat16 | 416 | spread | 1477.1 | 1395.3 | flydsl | 1359.9 | 1235.6 | 208x256x64 |
| 2048x26624x16384 | bfloat16 | 832 | one-tile owner, n = 4 | 1521.6 | 1364.9 | flydsl | 905.8 | 1237.1 | 208x256x64 |
| 2048x53248x16384 | bfloat16 | 1664 | one-tile owner, n = 2 | 1534.7 | 1482.3 | flydsl | 793.4 | 1274.6 | 240x256x64 |
| 4096x2304x16384 | bfloat16 | 144 | spread | 1251.6 | 1029.1 | torch | 1189.3 | 1194.6 | 144x256x64 |
| 4096x4608x16384 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 | 1404.5 | 1036.0 | torch | 1177.7 | 1178.4 | 144x256x64 |
| 4096x9216x16384 | bfloat16 | 576 | one-tile owner, n = 4 | 1518.4 | 1288.8 | torch | 1314.2 | 1310.5 | 192x256x64 |
| 4096x13312x16384 | bfloat16 | 832 | one-tile owner, n = 4 | 1535.8 | 1359.9 | flydsl | 1313.7 | 1253.4 | 208x256x64 |
| 4096x26624x16384 | bfloat16 | 1664 | one-tile owner, n = 2 | 1543.7 | 1506.8 | flydsl | 1447.6 | 1328.6 | 240x256x64 |
| 4096x53248x16384 | bfloat16 | 3328 | data-parallel | 1574.3 | 1581.8 | flydsl | 1523.8 | 1474.4 | 256x256x64 |
| 8192x2304x16384 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 | 1385.8 | 1006.4 | torch | 1185.2 | 1194.0 | 144x256x64 |
| 8192x4608x16384 | bfloat16 | 576 | one-tile owner, n = 4 | 1497.0 | 1271.9 | torch | 1337.8 | 1328.7 | 192x256x64 |
| 8192x9216x16384 | bfloat16 | 1152 | one-tile owner, n = 2 | 1527.7 | 1448.5 | flydsl | 1395.1 | 1144.8 | 256x240x64 |
| 8192x13312x16384 | bfloat16 | 1664 | one-tile owner, n = 2 | 1527.8 | 1477.2 | flydsl | 1435.1 | 1287.4 | 240x256x64 |
| 8192x26624x16384 | bfloat16 | 3328 | data-parallel | 1555.9 | 1554.6 | flydsl | 1502.3 | 1466.4 | 256x256x64 |
| 8192x53248x16384 | bfloat16 | 6656 | data-parallel | 1538.1 | 1548.9 | flydsl | 1502.3 | 1463.5 | 256x256x64 |
| 16384x2304x16384 | bfloat16 | 576 | one-tile owner, n = 4 | 1489.7 | 1274.1 | torch | 1335.7 | 1321.7 | 192x256x64 |
| 16384x4608x16384 | bfloat16 | 1152 | one-tile owner, n = 2 | 1542.7 | 1447.9 | flydsl | 1409.2 | 1170.3 | 256x240x64 |
| 16384x9216x16384 | bfloat16 | 2304 | data-parallel | 1561.2 | 1559.1 | flydsl | 1499.0 | 1484.4 | 256x256x64 |
| 16384x13312x16384 | bfloat16 | 3328 | data-parallel | 1542.8 | 1546.6 | asm | 1487.2 | 1467.1 | 256x256x64 |
| 16384x26624x16384 | bfloat16 | 6656 | data-parallel | 1532.3 | 1537.0 | flydsl | 1493.1 | 1454.9 | 256x256x64 |
| 16384x53248x16384 | bfloat16 | 13312 | data-parallel | 1510.9 | 1538.9 | asm | 1459.1 | 1433.1 | 256x256x64 |
| 32768x2304x16384 | bfloat16 | 1152 | one-tile owner, n = 2 | 1498.2 | 1436.6 | flydsl | 1399.9 | 1181.0 | 256x240x64 |
| 32768x4608x16384 | bfloat16 | 2304 | data-parallel | 1540.1 | 1539.9 | torch | 1523.5 | 1533.8 | 256x256x64 |
| 32768x9216x16384 | bfloat16 | 4608 | data-parallel | 1526.7 | 1532.1 | asm | 1456.7 | 1457.0 | 256x256x64 |
| 32768x13312x16384 | bfloat16 | 6656 | data-parallel | 1529.4 | 1534.2 | asm | 1464.1 | 1444.0 | 256x256x64 |
| 32768x26624x16384 | bfloat16 | 13312 | data-parallel | 1511.3 | 1535.1 | flydsl | 1473.3 | 1445.0 | 256x256x64 |
| 32768x53248x16384 | bfloat16 | 26624 | data-parallel | 1502.6 | 1501.8 | flydsl | 1439.5 | 1454.9 | 256x256x64 |
| 512x6144x18432 | bfloat16 | 48 | data-parallel | 775.1 | 357.5 | flydsl | 792.6 | 843.5 | 192x128x128 |
| 1024x6144x18432 | bfloat16 | 96 | spread | 1110.1 | 712.4 | torch | 1155.0 | 1155.5 | 192x256x64 |
| 2048x6144x18432 | bfloat16 | 192 | spread | 1429.9 | 1332.6 | torch | 1296.4 | 1307.1 | 192x256x64 |
| 4096x6144x18432 | bfloat16 | 384 | one-tile owner, n = 2 | 1501.8 | 1341.4 | torch | 1285.3 | 1302.9 | 192x256x64 |
| 8192x6144x18432 | bfloat16 | 768 | data-parallel | 1581.6 | 1572.9 | torch | 1519.0 | 1513.5 | 256x256x64 |
| 16384x6144x18432 | bfloat16 | 1536 | data-parallel | 1589.2 | 1580.4 | asm | 1518.9 | 1528.0 | 256x256x64 |
| 32768x6144x18432 | bfloat16 | 3072 | data-parallel | 1595.0 | 1580.3 | asm | 1518.2 | 1509.2 | 256x256x64 |
| 512x5120x25600 | bfloat16 | 40 | spread | 716.7 | 296.4 | flydsl | 430.8 | 879.1 | 160x128x128 |
| 1024x5120x25600 | bfloat16 | 80 | spread | 921.9 | 595.4 | torch | 1131.5 | 1140.9 | 160x256x64 |
| 2048x5120x25600 | bfloat16 | 160 | spread | 1393.5 | 1167.1 | torch | 1249.1 | 1257.3 | 160x256x64 |
| 4096x5120x25600 | bfloat16 | 320 | one-tile owner, n = 4 | 1493.1 | 1109.8 | torch | 1243.4 | 1246.3 | 160x256x64 |
| 8192x5120x25600 | bfloat16 | 640 | one-tile owner, n = 2 | 1540.8 | 1387.3 | asm | 1367.7 | 1335.8 | 256x224x64 |
| 16384x5120x25600 | bfloat16 | 1280 | data-parallel | 1587.7 | 1546.8 | torch | 1482.7 | 1538.2 | 256x256x64 |
| 32768x5120x25600 | bfloat16 | 2560 | data-parallel | 1582.5 | 1579.8 | torch | 1511.7 | 1523.6 | 256x256x64 |
| 512x16384x26624 | bfloat16 | 128 | one-tile owner, n = 2 | 1360.2 | 949.1 | flydsl | 655.9 | 1187.2 | 256x256x64 |
| 1024x16384x26624 | bfloat16 | 256 | data-parallel | 1551.6 | 1549.7 | flydsl | 760.4 | 1506.7 | 256x256x64 |
| 2048x16384x26624 | bfloat16 | 512 | data-parallel | 1545.0 | 1541.8 | flydsl | 826.8 | 1500.4 | 256x256x64 |
| 4096x16384x26624 | bfloat16 | 1024 | data-parallel | 1561.9 | 1555.2 | flydsl | 1505.1 | 1497.4 | 256x256x64 |
| 8192x16384x26624 | bfloat16 | 2048 | data-parallel | 1575.9 | 1562.0 | asm | 242.3 | 1491.7 | 256x256x64 |
| 16384x16384x26624 | bfloat16 | 4096 | data-parallel | 1590.2 | 1586.1 | asm | 1530.5 | 1483.4 | 256x256x64 |
| 32768x16384x26624 | bfloat16 | 8192 | data-parallel | 1591.1 | 1585.0 | asm | 1505.2 | 1472.7 | 256x256x64 |
| 512x8192x28672 | bfloat16 | 64 | one-tile owner, n = 2 | 1041.5 | 479.8 | flydsl | 789.4 | 926.3 | 240x256x64 |
| 1024x8192x28672 | bfloat16 | 128 | one-tile owner, n = 2 | 1441.6 | 945.5 | triton | 695.9 | 1305.8 | 256x256x64 |
| 2048x8192x28672 | bfloat16 | 256 | data-parallel | 1533.5 | 1531.9 | flydsl | 1038.7 | 1499.2 | 256x256x64 |
| 4096x8192x28672 | bfloat16 | 512 | data-parallel | 1541.0 | 1537.7 | flydsl | 1487.0 | 1503.8 | 256x256x64 |
| 8192x8192x28672 | bfloat16 | 1024 | data-parallel | 1600.4 | 1589.8 | asm | 1545.0 | 1529.0 | 256x256x64 |
| 16384x8192x28672 | bfloat16 | 2048 | data-parallel | 1576.8 | 1588.9 | asm | 1529.9 | 1521.0 | 256x256x64 |
| 32768x8192x28672 | bfloat16 | 4096 | data-parallel | 1594.8 | 1590.3 | asm | 1529.0 | 1486.4 | 256x256x64 |
| 512x7168x35840 | bfloat16 | 56 | data-parallel | 909.4 | 436.6 | flydsl | 821.7 | 965.0 | 256x176x64 |
| 1024x7168x35840 | bfloat16 | 112 | spread | 1214.3 | 863.5 | flydsl | 566.5 | 1275.3 | 224x256x64 |
| 2048x7168x35840 | bfloat16 | 224 | spread | 1494.0 | 1460.4 | asm | 1436.9 | 1355.1 | 224x256x64 |
| 4096x7168x35840 | bfloat16 | 448 | spread | 1533.7 | 1477.4 | flydsl | 1411.4 | 1352.7 | 224x256x64 |
| 8192x7168x35840 | bfloat16 | 896 | one-tile owner, n = 2 | 1563.8 | 1422.3 | asm | 1366.9 | 1360.9 | 224x256x64 |
| 16384x7168x35840 | bfloat16 | 1792 | data-parallel | 1565.5 | 1561.6 | torch | 1505.5 | 1506.6 | 256x256x64 |
| 32768x7168x35840 | bfloat16 | 3584 | data-parallel | 1585.5 | 1582.4 | flydsl | 1509.0 | 1486.2 | 256x256x64 |
