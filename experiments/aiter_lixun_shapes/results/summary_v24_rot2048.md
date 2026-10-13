# streamk_wip (v24, capped) vs AITER, v9 and hipBLASLt on the 630 shapes (2048 MB rotating buffer)

Shapes: `supported_status_scoped.csv` (630 shapes, no bias; 598 bf16 and 32 fp32 outputs). Inputs from `torch.manual_seed(0)`, every output checked against an fp32 `F.linear` reference, `do_bench_cudagraph` with 3 interleaved rounds and the median taken.

- Stack: torch 2.10.0+rocm7.2.4.git3d3aa833, hipBLASLt 100202 (dabb6df2b9), Triton 3.8.0 @ /home/jbowley/repos/triton_gfx950-tutorial-v2.2/python/triton.
- GPU: HIP device 3, PCI bus 21, gfx950:sramecc+:xnack-.
- AITER: e2c7aa0aa [Triton/Gluon] gfx950 a16w16: skip the Gluon kernel on Triton without cd_regclass, on top of a1455f1c4 2026-10-02 [CK] [FlyDSL] add mxfp8 a8w8 blockscale for gfx950 (#4951).
- streamk_wip: tutorial `v24_streamk_combined`, snapshot sha256 07108e9456bf44c4 (newest WORKLOG entry: "1. Every ending, placement and partial type in one kernel (2026-10-08)"), `STREAMK_POLICY=capped` (the kernel's default), v9 tile order. The configuration capped ran on each shape is in the appendix and in `v20_label`.
- hipBLASLt: the call AITER's `torch` libtype makes, `F.linear(x, w)` for bf16 and `torch.mm(x, w.t(), out_dtype=torch.float32)` for fp32, timed the same way. Its tile is the MT of the first `Cijk_` kernel it launches.
- Cache state: each callable rotates over max(2, ceil(2048 MB / set)) copies of x, w and C inside the captured graph, so no launch reads the inputs of the previous one. Outputs that a callable allocates itself (AITER's pick, hipBLASLt) are kept alive in a ring so they rotate too.
- "Faster" and "slower" mean a gap of more than 1%; "within 1%" holds the rest.

Copies per shape: median 15, minimum 2; 70 shapes at the minimum of 2.
`do_bench_cudagraph` sizes its graph from a host-side estimate, so on 4 small shapes a timed graph touched fewer copies than it had. What matters is the data a graph cycles through: copies touched x set size is at least 1290 MB on every shape (median 2107 MB), against a 256 MB last-level cache; 0 shapes are under 512 MB.

**Against AITER's pick, by the configuration capped chose**

| | Shapes | v9: faster | v9: within 1% | v9: slower | v9: median | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All shapes | 630 | 298 | 25 | 307 | -0.1% | 388 | 23 | 219 | +3.2% |
| Data-parallel | 324 | 233 | 20 | 71 | +3.3% | 251 | 12 | 61 | +3.6% |
| One-tile reduce-scatter, K >= 2048 | 77 | 5 | 0 | 72 | -68.7% | 16 | 1 | 60 | -18.6% |
| One-tile owner, K >= 2048 | 219 | 58 | 5 | 156 | -20.2% | 119 | 10 | 90 | +3.2% |
| One-tile reduce-scatter, K < 2048 | 6 | 0 | 0 | 6 | -66.3% | 0 | 0 | 6 | -57.1% |
| One-tile owner, K < 2048 | 2 | 0 | 0 | 2 | -49.8% | 0 | 0 | 2 | -47.1% |
| Spread | 2 | 2 | 0 | 0 | +2.2% | 2 | 0 | 0 | +5.9% |

Each pair of columns compares that kernel with AITER's pick; positive medians mean it is faster.

**Against AITER's pick, by the tile AITER uses**

| | Shapes | v9: faster | v9: within 1% | v9: slower | v9: median | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All shapes | 630 | 298 | 25 | 307 | -0.1% | 388 | 23 | 219 | +3.2% |
| AITER uses a 256x256 tile | 248 | 219 | 16 | 13 | +3.5% | 241 | 6 | 1 | +4.0% |
| AITER uses another tile | 382 | 79 | 9 | 294 | -22.3% | 147 | 17 | 218 | -4.5% |

Each pair of columns compares that kernel with AITER's pick; positive medians mean it is faster.

**Against AITER's pick, by v9 grid size (number of 256x256 output tiles)**

| | Shapes | v9: faster | v9: within 1% | v9: slower | v9: median | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All shapes | 630 | 298 | 25 | 307 | -0.1% | 388 | 23 | 219 | +3.2% |
| Under 256 (less than one wave) | 321 | 45 | 7 | 269 | -34.5% | 91 | 16 | 214 | -8.9% |
| 256 to 1023 | 152 | 115 | 8 | 29 | +3.5% | 143 | 4 | 5 | +6.9% |
| 1024 or more | 157 | 138 | 10 | 9 | +3.5% | 154 | 3 | 0 | +3.9% |

Each pair of columns compares that kernel with AITER's pick; positive medians mean it is faster.

**Against hipBLASLt (torch), by v9 grid size**

| | Shapes | AITER's pick: faster | AITER's pick: within 1% | AITER's pick: slower | AITER's pick: median | v9: faster | v9: within 1% | v9: slower | v9: median | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All shapes | 630 | 197 | 241 | 192 | -0.4% | 245 | 50 | 335 | -2.3% | 367 | 30 | 233 | +2.2% |
| Under 256 (less than one wave) | 321 | 112 | 117 | 92 | -0.2% | 34 | 9 | 278 | -34.8% | 89 | 9 | 223 | -10.6% |
| 256 to 1023 | 152 | 33 | 67 | 52 | -0.6% | 101 | 11 | 40 | +3.0% | 136 | 7 | 9 | +6.2% |
| 1024 or more | 157 | 52 | 57 | 48 | -0.6% | 110 | 30 | 17 | +3.2% | 142 | 14 | 1 | +3.9% |

Each pair of columns compares that kernel with hipBLASLt; positive medians mean it is faster.

**hipBLASLt's tile choice**

| hipBLASLt tile (MxNxK) | Shapes | hipBLASLt median TFLOPS | streamk_wip vs hipBLASLt: faster | within 1% | slower | median |
|---|---:|---:|---:|---:|---:|---:|
| 256x256x64 | 225 | 1415 | 205 | 18 | 2 | +3.6% |
| 128x128x256 | 40 | 658 | 9 | 0 | 31 | -11.7% |
| 192x256x64 | 34 | 1194 | 13 | 6 | 15 | -0.5% |
| 224x256x64 | 32 | 1271 | 32 | 0 | 0 | +8.5% |
| 256x128x64 | 27 | 724 | 2 | 1 | 24 | -15.5% |
| 160x256x64 | 27 | 1171 | 14 | 1 | 12 | +3.0% |
| 256x224x64 | 20 | 1287 | 18 | 0 | 2 | +8.7% |
| 256x240x64 | 18 | 1129 | 18 | 0 | 0 | +22.6% |
| 144x256x64 | 15 | 1151 | 7 | 0 | 8 | -7.2% |
| 128x128x128 | 14 | 571 | 0 | 0 | 14 | -24.1% |
| 160x128x128 | 13 | 783 | 2 | 0 | 11 | -13.4% |
| 96x128x128 | 12 | 557 | 0 | 0 | 12 | -40.3% |
| 192x128x128 | 12 | 763 | 2 | 0 | 10 | -9.3% |
| 256x208x64 | 12 | 1072 | 12 | 0 | 0 | +16.6% |
| 256x192x64 | 11 | 1208 | 6 | 1 | 4 | +1.0% |
| 208x256x64 | 11 | 1230 | 11 | 0 | 0 | +9.7% |
| 256x160x64 | 10 | 1126 | 3 | 0 | 7 | -6.7% |
| 64x64x256 | 9 | 402 | 0 | 0 | 9 | -27.5% |
| 128x192x128 | 7 | 884 | 1 | 0 | 6 | -14.2% |
| 128x160x128 | 7 | 853 | 1 | 0 | 6 | -14.5% |
| 240x256x64 | 7 | 1197 | 7 | 0 | 0 | +20.6% |
| 128x80x128 | 6 | 602 | 0 | 0 | 6 | -23.8% |
| 128x96x128 | 6 | 660 | 0 | 0 | 6 | -19.1% |
| 224x128x64 | 5 | 778 | 0 | 0 | 5 | -17.4% |
| 256x176x64 | 4 | 694 | 2 | 1 | 1 | +5.1% |
| 128x64x128 | 4 | 452 | 0 | 0 | 4 | -47.2% |
| 160x64x128 | 4 | 421 | 0 | 0 | 4 | -30.0% |
| 64x80x256 | 4 | 465 | 0 | 0 | 4 | -30.6% |
| 64x96x256 | 4 | 500 | 0 | 0 | 4 | -25.5% |
| 64x32x256 | 3 | 264 | 0 | 0 | 3 | -32.6% |
| 64x48x256 | 3 | 368 | 0 | 0 | 3 | -32.0% |
| 160x192x128 | 3 | 879 | 1 | 2 | 0 | +1.0% |
| 192x128x64 | 2 | 811 | 0 | 0 | 2 | -28.5% |
| 128x192x64 | 2 | 844 | 0 | 0 | 2 | -16.7% |
| 256x112x64 | 2 | 440 | 0 | 0 | 2 | -16.3% |
| 192x64x128 | 2 | 397 | 0 | 0 | 2 | -24.8% |
| 128x128x64 | 2 | 445 | 0 | 0 | 2 | -25.6% |
| 256x96x64 | 2 | 403 | 0 | 0 | 2 | -20.5% |
| 256x64x64 | 2 | 310 | 0 | 0 | 2 | -9.5% |
| 96x64x128 | 1 | 391 | 0 | 0 | 1 | -55.7% |
| 224x160x64 | 1 | 1062 | 1 | 0 | 0 | +13.5% |
| 192x192x64 | 1 | 1021 | 0 | 0 | 1 | -9.1% |
| 64x96x128 | 1 | 512 | 0 | 0 | 1 | -22.0% |
| 256x80x64 | 1 | 362 | 0 | 0 | 1 | -20.6% |
| 256x80x128 | 1 | 542 | 0 | 0 | 1 | -21.7% |
| 320x192x64 | 1 | 690 | 0 | 0 | 1 | -1.8% |

Solution-name tokens (stream-K `SK`, global split-U `GSU`, kernels launched): SK3 GSU0 x1: 604, SK3 GSU0 x2: 24, none GSU- x1: 2. `SK3` marks a kernel built with stream-K support, not that this launch split K.

**streamk_wip against v9, by configuration**

| | Shapes | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|
| All shapes | 630 | 373 | 241 | 16 | +2.9% |
| Data-parallel | 324 | 70 | 240 | 14 | +0.2% |
| One-tile reduce-scatter, K >= 2048 | 77 | 77 | 0 | 0 | +154.2% |
| One-tile owner, K >= 2048 | 219 | 216 | 1 | 2 | +29.8% |
| One-tile reduce-scatter, K < 2048 | 6 | 6 | 0 | 0 | +30.1% |
| One-tile owner, K < 2048 | 2 | 2 | 0 | 0 | +5.3% |
| Spread | 2 | 2 | 0 | 0 | +3.5% |

Each pair of columns compares that kernel with v9; positive medians mean it is faster.

**Diagnostic: capped's split against forcing data-parallel**

| Configuration | Shapes | split faster | within 1% | split slower | median |
|---|---:|---:|---:|---:|---:|
| One-tile owner, K < 2048 | 2 | 2 | 0 | 0 | +8.5% |
| One-tile owner, K >= 2048 | 219 | 214 | 3 | 2 | +31.5% |
| One-tile reduce-scatter, K < 2048 | 6 | 6 | 0 | 0 | +35.5% |
| One-tile reduce-scatter, K >= 2048 | 77 | 76 | 1 | 0 | +155.4% |
| Spread | 2 | 2 | 0 | 0 | +3.3% |

**Selected shapes (TFLOPS)**

| Shape (MxNxK) | Output tiles | AITER pick | AITER | hipBLASLt | hipBLASLt tile | v9 | streamk_wip | streamk_wip config | vs v9 | vs AITER | vs hipBLASLt |
|---|---:|---|---:|---:|---|---:|---:|---|---:|---:|---:|
| 512x2304x16384 | 18 | torch | 705 | 701 | 256x128x64 | 139 | 569 | one-tile reduce-scatter, n = 7 | +309% | -19% | -19% |
| 512x5120x25600 | 40 | flydsl | 428 | 878 | 160x128x128 | 294 | 1014 | one-tile owner, n = 6-7 | +244% | +137% | +15% |
| 512x7168x35840 | 56 | flydsl | 813 | 963 | 256x176x64 | 435 | 1183 | one-tile owner, n = 4-5 | +172% | +45% | +23% |
| 8192x2304x16384 | 288 | torch | 1178 | 1190 | 144x256x64 | 996 | 1377 | one-tile reduce-scatter, n = 4 | +38% | +17% | +16% |
| 4096x5120x25600 | 320 | torch | 1262 | 1262 | 160x256x64 | 1137 | 1513 | one-tile owner, n = 4 | +33% | +20% | +20% |
| 4096x4608x8192 | 288 | torch | 1167 | 1171 | 144x256x64 | 1020 | 1337 | one-tile reduce-scatter, n = 4 | +31% | +15% | +14% |
| 1024x3072x512 | 48 | flydsl | 364 | 321 | 96x128x128 | 202 | 202 | data-parallel | +0% | -45% | -37% |
| 2048x2048x512 | 64 | flydsl | 433 | 392 | 128x128x128 | 260 | 260 | data-parallel | -0% | -40% | -34% |
| 4096x5120x640 | 320 | flydsl | 611 | 646 | 256x176x64 | 632 | 650 | data-parallel | +3% | +6% | +1% |
| 8192x2048x512 | 256 | torch | 749 | 755 | 256x256x64 | 771 | 773 | data-parallel | +0% | +3% | +2% |

**Against `results/details_aisched_rot2048.csv`** (this run / that run - 1, median; for v20 the baseline may use a different policy or kernel snapshot)

| | Shapes | AITER's pick | v9 | v20 | hipBLASLt |
|---|---:|---:|---:|---:|---:|
| All shapes | 630 | -0.6% | -0.9% | -0.6% | -1.0% |
| Under 256 (less than one wave) | 321 | -0.2% | -0.4% | +1.1% | -0.8% |
| 256 to 1023 | 152 | -1.1% | -1.3% | -0.9% | -1.6% |
| 1024 or more | 157 | -1.1% | -1.3% | -1.0% | -0.9% |
| K < 2048 | 91 | -0.6% | -0.6% | -0.5% | -0.8% |
| K >= 2048 | 539 | -0.6% | -0.9% | -0.6% | -1.0% |

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
| 1024x3072x512 | bfloat16 | 48 | data-parallel | 201.9 | 201.8 | flydsl | 364.4 | 320.8 | 96x128x128 |
| 1024x4096x512 | bfloat16 | 64 | data-parallel | 258.5 | 258.1 | flydsl | 435.2 | 390.2 | 128x128x128 |
| 1024x7168x512 | bfloat16 | 112 | data-parallel | 415.6 | 415.6 | flydsl | 508.2 | 502.7 | 256x128x64 |
| 2048x2048x512 | bfloat16 | 64 | data-parallel | 260.1 | 260.3 | flydsl | 432.6 | 391.8 | 128x128x128 |
| 2048x4096x512 | bfloat16 | 128 | data-parallel | 463.1 | 464.4 | torch | 557.6 | 559.1 | 256x128x64 |
| 4096x2048x512 | bfloat16 | 128 | data-parallel | 455.7 | 461.6 | torch | 637.9 | 643.5 | 256x128x64 |
| 4096x4096x512 | bfloat16 | 256 | data-parallel | 767.7 | 766.6 | torch | 757.5 | 764.3 | 256x256x64 |
| 4096x7168x512 | bfloat16 | 448 | data-parallel | 782.7 | 757.8 | flydsl | 743.1 | 717.1 | 224x256x64 |
| 8192x2048x512 | bfloat16 | 256 | data-parallel | 772.6 | 771.4 | torch | 749.2 | 755.2 | 256x256x64 |
| 8192x4096x512 | bfloat16 | 512 | data-parallel | 833.2 | 813.1 | flydsl | 795.7 | 834.3 | 256x256x64 |
| 16384x2048x512 | bfloat16 | 512 | data-parallel | 840.8 | 816.7 | torch | 827.2 | 830.3 | 256x256x64 |
| 16384x4096x512 | bfloat16 | 1024 | data-parallel | 892.4 | 834.6 | torch | 880.3 | 887.5 | 256x256x64 |
| 16384x7168x512 | bfloat16 | 1792 | data-parallel | 965.1 | 897.0 | torch | 972.0 | 981.2 | 256x256x64 |
| 32768x2048x512 | bfloat16 | 1024 | data-parallel | 902.3 | 842.3 | torch | 870.4 | 874.1 | 256x256x64 |
| 1024x5120x640 | bfloat16 | 80 | data-parallel | 324.2 | 324.0 | torch | 440.9 | 443.8 | 160x128x128 |
| 2048x5120x640 | bfloat16 | 160 | data-parallel | 600.4 | 600.8 | flydsl | 591.4 | 505.4 | 192x256x64 |
| 4096x5120x640 | bfloat16 | 320 | data-parallel | 649.8 | 632.2 | flydsl | 611.0 | 646.2 | 256x176x64 |
| 8192x5120x640 | bfloat16 | 640 | data-parallel | 822.9 | 793.2 | torch | 858.2 | 864.6 | 256x224x64 |
| 16384x5120x640 | bfloat16 | 1280 | data-parallel | 1001.1 | 951.4 | torch | 996.4 | 1000.4 | 256x256x64 |
| 32768x5120x640 | bfloat16 | 2560 | data-parallel | 1046.1 | 985.7 | torch | 1021.2 | 1038.1 | 256x256x64 |
| 1024x7168x768 | bfloat16 | 112 | data-parallel | 483.9 | 484.8 | flydsl | 593.0 | 581.6 | 224x128x64 |
| 10240x7168x768 | bfloat16 | 1120 | data-parallel | 1037.9 | 1004.4 | opus | 375.7 | 847.5 | 256x240x64 |
| 512x4096x1024 | bfloat16 | 32 | data-parallel | 180.5 | 181.0 | torch | 388.4 | 390.7 | 128x64x128 |
| 512x6144x1024 | bfloat16 | 48 | data-parallel | 266.9 | 268.1 | flydsl | 496.2 | 475.6 | 96x128x128 |
| 512x7168x1024 | bfloat16 | 56 | data-parallel | 308.1 | 308.4 | flydsl | 537.2 | 492.1 | 128x128x128 |
| 512x8192x1024 | bfloat16 | 64 | data-parallel | 344.7 | 345.0 | flydsl | 576.6 | 546.4 | 128x128x128 |
| 1024x4096x1024 | bfloat16 | 64 | data-parallel | 344.4 | 344.8 | torch | 557.7 | 560.2 | 128x128x128 |
| 1024x7168x1024 | bfloat16 | 112 | data-parallel | 567.8 | 567.7 | flydsl | 665.4 | 687.7 | 224x128x64 |
| 1024x8192x1024 | bfloat16 | 128 | data-parallel | 624.5 | 624.0 | torch | 753.8 | 757.8 | 256x128x64 |
| 2048x4096x1024 | bfloat16 | 128 | data-parallel | 629.7 | 629.5 | torch | 763.5 | 769.1 | 256x128x64 |
| 2048x7168x1024 | bfloat16 | 224 | data-parallel | 991.8 | 993.8 | flydsl | 945.3 | 951.0 | 256x256x64 |
| 2048x8192x1024 | bfloat16 | 256 | data-parallel | 1060.9 | 1061.1 | flydsl | 999.5 | 1010.7 | 256x256x64 |
| 3072x4096x1024 | bfloat16 | 192 | data-parallel | 882.8 | 884.5 | torch | 903.6 | 911.3 | 256x192x64 |
| 4096x4096x1024 | bfloat16 | 256 | data-parallel | 1071.2 | 1058.9 | torch | 1011.2 | 1014.8 | 256x256x64 |
| 4096x7168x1024 | bfloat16 | 448 | data-parallel | 1040.2 | 1024.8 | flydsl | 979.0 | 970.1 | 224x256x64 |
| 4096x8192x1024 | bfloat16 | 512 | data-parallel | 1112.4 | 1095.2 | asm | 1101.6 | 1099.2 | 256x256x64 |
| 5120x4096x1024 | bfloat16 | 320 | data-parallel | 854.7 | 852.7 | torch | 971.7 | 974.9 | 256x160x64 |
| 7680x4096x1024 | bfloat16 | 480 | data-parallel | 1086.7 | 1080.5 | torch | 930.3 | 935.9 | 256x240x64 |
| 8192x7168x1024 | bfloat16 | 896 | data-parallel | 1089.3 | 1041.7 | asm | 1035.4 | 1016.1 | 224x256x64 |
| 8192x8192x1024 | bfloat16 | 1024 | data-parallel | 1166.3 | 1124.0 | asm | 1134.9 | 1150.7 | 256x256x64 |
| 16384x4096x1024 | bfloat16 | 1024 | data-parallel | 1174.5 | 1134.0 | torch | 1145.0 | 1155.5 | 256x256x64 |
| 16384x7168x1024 | bfloat16 | 1792 | data-parallel | 1217.1 | 1174.7 | asm | 1129.1 | 1201.0 | 256x256x64 |
| 16384x8192x1024 | bfloat16 | 2048 | data-parallel | 1219.4 | 1167.9 | torch | 1184.7 | 1193.1 | 256x256x64 |
| 32768x7168x1024 | bfloat16 | 3584 | data-parallel | 1241.1 | 1201.9 | torch | 1214.5 | 1221.4 | 256x256x64 |
| 32768x8192x1024 | bfloat16 | 4096 | data-parallel | 1244.5 | 1195.8 | torch | 1203.3 | 1223.1 | 256x256x64 |
| 512x5120x1280 | bfloat16 | 40 | data-parallel | 202.3 | 202.5 | flydsl | 443.7 | 423.2 | 160x64x128 |
| 1024x5120x1280 | bfloat16 | 80 | data-parallel | 393.3 | 393.3 | torch | 601.4 | 602.9 | 160x128x128 |
| 2048x5120x1280 | bfloat16 | 160 | data-parallel | 754.1 | 755.2 | torch | 839.7 | 842.3 | 160x256x64 |
| 4096x5120x1280 | bfloat16 | 320 | data-parallel | 797.4 | 788.6 | torch | 870.7 | 875.9 | 160x256x64 |
| 8192x5120x1280 | bfloat16 | 640 | data-parallel | 1035.2 | 1006.2 | flydsl | 983.8 | 1071.1 | 256x224x64 |
| 16384x5120x1280 | bfloat16 | 1280 | data-parallel | 1226.0 | 1183.6 | flydsl | 1160.9 | 1219.1 | 256x256x64 |
| 32768x5120x1280 | bfloat16 | 2560 | data-parallel | 1258.8 | 1215.6 | torch | 1233.8 | 1246.9 | 256x256x64 |
| 512x1536x1536 | bfloat16 | 12 | one-tile reduce-scatter, n = 8 | 99.6 | 69.7 | flydsl | 265.3 | 237.2 | 64x64x256 |
| 512x2048x1536 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 | 129.1 | 92.5 | torch | 310.5 | 311.4 | 64x64x256 |
| 512x2304x1536 | bfloat16 | 18 | one-tile reduce-scatter, n = 7 | 136.0 | 103.8 | flydsl | 318.8 | 314.1 | 64x80x256 |
| 512x4096x1536 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 238.4 | 185.7 | torch | 445.4 | 446.4 | 128x64x128 |
| 512x6144x1536 | bfloat16 | 48 | data-parallel | 276.2 | 276.6 | flydsl | 541.9 | 542.7 | 96x128x128 |
| 512x7168x1536 | bfloat16 | 56 | data-parallel | 318.3 | 318.8 | flydsl | 583.8 | 489.6 | 128x128x256 |
| 1024x1536x1536 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 | 173.1 | 140.0 | flydsl | 400.7 | 391.0 | 96x64x128 |
| 1024x2048x1536 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 238.7 | 184.9 | torch | 455.4 | 457.8 | 128x64x128 |
| 1024x2304x1536 | bfloat16 | 36 | one-tile owner, n = 2 | 215.3 | 206.9 | torch | 457.5 | 457.1 | 128x80x128 |
| 1024x3072x1536 | bfloat16 | 48 | data-parallel | 275.8 | 276.4 | flydsl | 553.0 | 557.0 | 96x128x128 |
| 1024x4096x1536 | bfloat16 | 64 | data-parallel | 363.0 | 363.3 | torch | 554.8 | 556.6 | 128x128x256 |
| 1024x7168x1536 | bfloat16 | 112 | data-parallel | 610.6 | 609.8 | torch | 776.1 | 778.3 | 224x128x64 |
| 2048x1536x1536 | bfloat16 | 48 | data-parallel | 279.1 | 281.6 | torch | 552.1 | 553.1 | 96x128x128 |
| 2048x2048x1536 | bfloat16 | 64 | data-parallel | 368.6 | 367.8 | torch | 558.2 | 558.5 | 128x128x256 |
| 2048x4096x1536 | bfloat16 | 128 | data-parallel | 686.8 | 691.1 | torch | 843.5 | 845.5 | 256x128x64 |
| 3072x2048x1536 | bfloat16 | 96 | data-parallel | 535.0 | 535.8 | opus | 685.5 | 758.6 | 128x192x128 |
| 3072x4096x1536 | bfloat16 | 192 | data-parallel | 978.0 | 979.7 | opus | 993.8 | 953.6 | 256x192x64 |
| 4096x1536x1536 | bfloat16 | 96 | data-parallel | 532.2 | 537.3 | torch | 734.8 | 739.1 | 192x128x128 |
| 4096x2048x1536 | bfloat16 | 128 | data-parallel | 682.5 | 684.4 | opus | 852.4 | 843.2 | 256x128x64 |
| 4096x3072x1536 | bfloat16 | 192 | data-parallel | 981.5 | 983.9 | flydsl | 936.8 | 995.5 | 192x256x64 |
| 4096x4096x1536 | bfloat16 | 256 | data-parallel | 1181.0 | 1184.8 | torch | 1158.7 | 1163.5 | 256x256x64 |
| 5120x2048x1536 | bfloat16 | 160 | data-parallel | 836.7 | 839.7 | torch | 945.8 | 949.1 | 256x160x64 |
| 5120x4096x1536 | bfloat16 | 320 | data-parallel | 937.9 | 936.5 | torch | 996.9 | 1005.5 | 256x160x64 |
| 7680x2048x1536 | bfloat16 | 240 | data-parallel | 1131.5 | 1138.6 | torch | 947.7 | 950.4 | 256x240x64 |
| 7680x4096x1536 | bfloat16 | 480 | data-parallel | 1189.9 | 1188.3 | torch | 1016.4 | 1024.5 | 256x240x64 |
| 8192x1536x1536 | bfloat16 | 192 | data-parallel | 968.8 | 970.4 | torch | 987.5 | 986.1 | 192x256x64 |
| 10240x7168x1536 | bfloat16 | 1120 | data-parallel | 1228.8 | 1208.7 | asm | 1144.3 | 1124.1 | 224x256x64 |
| 16384x1536x1536 | bfloat16 | 384 | data-parallel | 1044.4 | 1049.8 | torch | 1040.6 | 1063.5 | 192x256x64 |
| 16384x2048x1536 | bfloat16 | 512 | data-parallel | 1204.7 | 1205.9 | torch | 1205.9 | 1206.3 | 256x256x64 |
| 16384x3072x1536 | bfloat16 | 768 | data-parallel | 1256.6 | 1245.2 | torch | 1233.4 | 1243.8 | 256x256x64 |
| 16384x4096x1536 | bfloat16 | 1024 | data-parallel | 1261.9 | 1238.3 | torch | 1253.9 | 1260.8 | 256x256x64 |
| 32768x1536x1536 | bfloat16 | 768 | data-parallel | 1249.8 | 1235.6 | flydsl | 1199.6 | 1221.5 | 256x256x64 |
| 512x7168x1792 | bfloat16 | 56 | one-tile owner, n = 2 | 364.2 | 341.6 | flydsl | 619.1 | 500.1 | 128x128x256 |
| 1024x7168x1792 | bfloat16 | 112 | data-parallel | 657.4 | 657.5 | flydsl | 779.5 | 827.8 | 224x128x64 |
| 2048x7168x1792 | bfloat16 | 224 | data-parallel | 1160.5 | 1164.3 | flydsl | 1102.5 | 1061.4 | 224x256x64 |
| 4096x7168x1792 | bfloat16 | 448 | data-parallel | 1205.9 | 1198.6 | flydsl | 1149.4 | 1121.2 | 224x256x64 |
| 8192x7168x1792 | bfloat16 | 896 | data-parallel | 1238.5 | 1219.1 | asm | 1188.2 | 1154.7 | 224x256x64 |
| 16384x7168x1792 | bfloat16 | 1792 | data-parallel | 1351.9 | 1330.1 | torch | 1326.8 | 1339.3 | 256x256x64 |
| 32768x7168x1792 | bfloat16 | 3584 | data-parallel | 1376.7 | 1348.5 | torch | 1345.1 | 1359.5 | 256x256x64 |
| 512x4096x2048 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 305.4 | 213.5 | torch | 489.7 | 490.8 | 128x64x128 |
| 512x6144x2048 | float32 | 48 | one-tile owner, n = 2 | 320.4 | 287.5 | asm | 331.7 | 557.7 | 96x128x128 |
| 512x6144x2048 | bfloat16 | 48 | one-tile owner, n = 2 | 352.7 | 319.4 | flydsl | 604.1 | 595.7 | 96x128x128 |
| 512x8192x2048 | bfloat16 | 64 | one-tile owner, n = 2 | 451.2 | 410.7 | flydsl | 703.1 | 571.8 | 128x128x256 |
| 512x16384x2048 | bfloat16 | 128 | data-parallel | 745.7 | 750.1 | flydsl | 715.9 | 878.3 | 256x128x64 |
| 1024x4096x2048 | bfloat16 | 64 | one-tile owner, n = 2 | 454.7 | 415.1 | torch | 608.7 | 609.5 | 128x128x256 |
| 1024x6144x2048 | float32 | 96 | data-parallel | 541.4 | 544.4 | opus | 555.4 | 780.2 | 192x128x64 |
| 1024x6144x2048 | bfloat16 | 96 | data-parallel | 602.8 | 605.3 | flydsl | 734.9 | 786.3 | 192x128x128 |
| 1024x8192x2048 | bfloat16 | 128 | data-parallel | 763.7 | 764.7 | flydsl | 879.0 | 926.4 | 256x128x64 |
| 1024x16384x2048 | bfloat16 | 256 | data-parallel | 1248.6 | 1256.6 | flydsl | 739.1 | 1208.7 | 256x256x64 |
| 2048x4096x2048 | bfloat16 | 128 | data-parallel | 772.0 | 773.7 | torch | 938.5 | 943.8 | 256x128x64 |
| 2048x6144x2048 | float32 | 192 | data-parallel | 953.6 | 957.7 | asm | 908.9 | 1003.2 | 192x256x64 |
| 2048x6144x2048 | bfloat16 | 192 | data-parallel | 1079.9 | 1088.5 | flydsl | 1035.7 | 1091.1 | 192x256x64 |
| 2048x8192x2048 | bfloat16 | 256 | data-parallel | 1268.5 | 1269.1 | flydsl | 1207.5 | 1226.3 | 256x256x64 |
| 2048x16384x2048 | bfloat16 | 512 | data-parallel | 1314.7 | 1303.1 | flydsl | 1032.3 | 1258.9 | 256x256x64 |
| 3072x4096x2048 | bfloat16 | 192 | data-parallel | 1079.5 | 1084.0 | torch | 1088.2 | 1097.7 | 256x192x64 |
| 4096x4096x2048 | bfloat16 | 256 | data-parallel | 1273.8 | 1279.2 | torch | 1235.9 | 1245.5 | 256x256x64 |
| 4096x6144x2048 | float32 | 384 | data-parallel | 984.6 | 988.3 | asm | 942.9 | 1053.2 | 192x256x64 |
| 4096x6144x2048 | bfloat16 | 384 | data-parallel | 1136.2 | 1134.7 | flydsl | 888.0 | 1142.2 | 192x256x64 |
| 4096x8192x2048 | bfloat16 | 512 | data-parallel | 1312.8 | 1308.4 | flydsl | 1261.1 | 1286.9 | 256x256x64 |
| 4096x16384x2048 | bfloat16 | 1024 | data-parallel | 1346.1 | 1332.3 | flydsl | 856.0 | 1334.4 | 256x256x64 |
| 5120x4096x2048 | bfloat16 | 320 | one-tile owner, n = 2 | 1042.8 | 1000.2 | torch | 1116.3 | 1117.4 | 256x160x64 |
| 7680x4096x2048 | bfloat16 | 480 | data-parallel | 1287.2 | 1282.6 | torch | 1089.9 | 1099.4 | 256x240x64 |
| 8192x6144x2048 | float32 | 768 | data-parallel | 1184.0 | 1172.8 | asm | 1138.5 | 1098.1 | 256x256x64 |
| 8192x6144x2048 | bfloat16 | 768 | data-parallel | 1358.3 | 1337.2 | flydsl | 832.6 | 1327.0 | 256x256x64 |
| 8192x8192x2048 | bfloat16 | 1024 | data-parallel | 1364.2 | 1339.6 | asm | 1315.7 | 1337.6 | 256x256x64 |
| 8192x16384x2048 | bfloat16 | 2048 | data-parallel | 1370.9 | 1337.6 | triton | 1021.0 | 1369.2 | 256x256x64 |
| 16384x4096x2048 | bfloat16 | 1024 | data-parallel | 1353.6 | 1334.4 | torch | 1312.3 | 1325.4 | 256x256x64 |
| 16384x6144x2048 | float32 | 1536 | data-parallel | 1204.7 | 1185.1 | asm | 1159.6 | 1126.7 | 256x256x64 |
| 16384x6144x2048 | bfloat16 | 1536 | data-parallel | 1376.2 | 1353.1 | torch | 1347.1 | 1360.4 | 256x256x64 |
| 16384x8192x2048 | bfloat16 | 2048 | data-parallel | 1385.7 | 1358.5 | torch | 1354.5 | 1365.2 | 256x256x64 |
| 16384x16384x2048 | bfloat16 | 4096 | data-parallel | 1391.5 | 1357.9 | flydsl | 954.1 | 1392.8 | 256x256x64 |
| 32768x6144x2048 | bfloat16 | 3072 | data-parallel | 1401.5 | 1373.5 | flydsl | 990.5 | 1375.7 | 256x256x64 |
| 32768x8192x2048 | bfloat16 | 4096 | data-parallel | 1410.0 | 1379.7 | torch | 1371.9 | 1384.6 | 256x256x64 |
| 32768x16384x2048 | bfloat16 | 8192 | data-parallel | 1394.3 | 1351.7 | flydsl | 826.1 | 1405.4 | 256x256x64 |
| 512x6144x3072 | float32 | 48 | one-tile owner, n = 2 | 371.9 | 291.8 | opus | 412.7 | 617.5 | 96x128x128 |
| 512x6144x3072 | bfloat16 | 48 | one-tile owner, n = 2 | 402.8 | 313.8 | flydsl | 635.7 | 658.8 | 96x128x128 |
| 1024x6144x3072 | float32 | 96 | one-tile owner, n = 2-3 | 619.2 | 577.2 | opus | 636.7 | 841.8 | 192x128x64 |
| 1024x6144x3072 | bfloat16 | 96 | one-tile owner, n = 2-3 | 651.7 | 614.9 | flydsl | 781.4 | 832.3 | 192x128x128 |
| 2048x6144x3072 | float32 | 192 | data-parallel | 1043.3 | 1043.4 | asm | 1014.3 | 1087.8 | 192x256x64 |
| 2048x6144x3072 | bfloat16 | 192 | data-parallel | 1154.8 | 1154.1 | flydsl | 1112.7 | 1168.5 | 192x256x64 |
| 4096x6144x3072 | float32 | 384 | data-parallel | 1069.4 | 1082.5 | asm | 1039.9 | 1132.7 | 192x256x64 |
| 4096x6144x3072 | bfloat16 | 384 | data-parallel | 1200.3 | 1198.4 | flydsl | 923.8 | 1206.8 | 192x256x64 |
| 8192x6144x3072 | float32 | 768 | data-parallel | 1274.5 | 1273.4 | asm | 1237.9 | 1184.8 | 256x256x64 |
| 8192x6144x3072 | bfloat16 | 768 | data-parallel | 1413.1 | 1402.2 | flydsl | 778.2 | 1393.8 | 256x256x64 |
| 16384x6144x3072 | float32 | 1536 | data-parallel | 1303.4 | 1289.9 | asm | 1264.7 | 1217.3 | 256x256x64 |
| 16384x6144x3072 | bfloat16 | 1536 | data-parallel | 1444.5 | 1426.4 | torch | 1406.6 | 1419.3 | 256x256x64 |
| 32768x6144x3072 | bfloat16 | 3072 | data-parallel | 1468.3 | 1444.2 | flydsl | 858.5 | 1439.2 | 256x256x64 |
| 512x5120x3200 | bfloat16 | 40 | one-tile owner, n = 2 | 364.7 | 268.5 | flydsl | 576.0 | 563.6 | 160x64x128 |
| 1024x5120x3200 | bfloat16 | 80 | one-tile owner, n = 2 | 692.2 | 528.6 | torch | 819.5 | 822.4 | 160x128x128 |
| 2048x5120x3200 | bfloat16 | 160 | data-parallel | 1022.2 | 1027.2 | torch | 1081.6 | 1093.7 | 160x256x64 |
| 4096x5120x3200 | bfloat16 | 320 | one-tile owner, n = 2 | 1160.9 | 1036.1 | torch | 1121.6 | 1126.6 | 160x256x64 |
| 8192x5120x3200 | bfloat16 | 640 | data-parallel | 1309.5 | 1298.2 | asm | 1254.4 | 1243.3 | 256x224x64 |
| 16384x5120x3200 | bfloat16 | 1280 | data-parallel | 1444.5 | 1427.2 | torch | 1410.9 | 1423.4 | 256x256x64 |
| 32768x5120x3200 | bfloat16 | 2560 | data-parallel | 1476.0 | 1451.6 | torch | 1429.8 | 1441.3 | 256x256x64 |
| 512x7168x3584 | bfloat16 | 56 | one-tile owner, n = 2 | 509.8 | 375.2 | flydsl | 708.5 | 582.7 | 128x128x256 |
| 512x8192x3584 | bfloat16 | 64 | one-tile owner, n = 2 | 571.8 | 427.5 | flydsl | 669.9 | 647.5 | 128x128x256 |
| 1024x7168x3584 | bfloat16 | 112 | one-tile owner, n = 2-3 | 851.7 | 734.0 | torch | 955.3 | 960.4 | 224x128x64 |
| 1024x8192x3584 | bfloat16 | 128 | data-parallel | 824.8 | 826.4 | flydsl | 789.8 | 1019.9 | 256x128x64 |
| 2048x8192x3584 | bfloat16 | 256 | data-parallel | 1386.4 | 1384.0 | flydsl | 1279.1 | 1340.3 | 256x256x64 |
| 4096x8192x3584 | bfloat16 | 512 | data-parallel | 1427.1 | 1420.4 | flydsl | 978.3 | 1370.2 | 256x256x64 |
| 8192x8192x3584 | bfloat16 | 1024 | data-parallel | 1464.5 | 1443.8 | torch | 1403.9 | 1413.6 | 256x256x64 |
| 10240x7168x3584 | bfloat16 | 1120 | one-tile owner, n = 2-3 | 1374.1 | 1344.4 | flydsl | 781.9 | 1268.4 | 224x256x64 |
| 16384x8192x3584 | bfloat16 | 2048 | data-parallel | 1480.4 | 1466.8 | flydsl | 807.9 | 1443.1 | 256x256x64 |
| 32768x8192x3584 | bfloat16 | 4096 | data-parallel | 1498.1 | 1480.2 | flydsl | 854.5 | 1467.9 | 256x256x64 |
| 512x1024x4096 | bfloat16 | 8 | one-tile reduce-scatter, n = 8 | 176.5 | 59.7 | flydsl | 282.9 | 264.2 | 64x32x256 |
| 512x2048x4096 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 | 306.8 | 119.6 | flydsl | 440.2 | 422.7 | 64x64x256 |
| 512x4096x4096 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 463.7 | 239.4 | flydsl | 600.8 | 609.6 | 128x128x128 |
| 512x8192x4096 | bfloat16 | 64 | one-tile owner, n = 2 | 619.6 | 464.4 | flydsl | 801.2 | 672.0 | 128x128x256 |
| 512x8704x4096 | bfloat16 | 68 | one-tile owner, n = 2 | 649.2 | 492.7 | torch | 739.2 | 741.0 | 160x128x128 |
| 512x10240x4096 | bfloat16 | 80 | one-tile owner, n = 2 | 745.2 | 574.0 | torch | 855.7 | 861.1 | 128x192x64 |
| 512x16384x4096 | bfloat16 | 128 | one-tile owner, n = 2 | 994.2 | 847.9 | flydsl | 631.1 | 698.1 | 256x256x64 |
| 512x129280x4096 | bfloat16 | 1010 | data-parallel | 1347.5 | 1357.7 | asm | 1285.8 | 1143.5 | 256x256x64 |
| 768x1024x4096 | bfloat16 | 12 | one-tile reduce-scatter, n = 8 | 251.5 | 90.9 | torch | 370.6 | 372.0 | 64x48x256 |
| 768x2048x4096 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 | 373.9 | 180.7 | flydsl | 526.2 | 498.2 | 64x96x256 |
| 768x4096x4096 | bfloat16 | 48 | one-tile owner, n = 3 | 549.1 | 359.6 | flydsl | 694.9 | 654.7 | 128x96x128 |
| 768x8192x4096 | bfloat16 | 96 | one-tile owner, n = 2-3 | 774.8 | 679.8 | torch | 879.7 | 884.4 | 128x192x128 |
| 768x8704x4096 | bfloat16 | 102 | one-tile owner, n = 2-3 | 836.7 | 716.3 | flydsl | 864.8 | 844.7 | 160x192x128 |
| 768x10240x4096 | bfloat16 | 120 | one-tile owner, n = 2-3 | 975.8 | 826.3 | flydsl | 967.4 | 933.0 | 160x192x128 |
| 768x129280x4096 | bfloat16 | 1515 | data-parallel | 1422.2 | 1407.5 | torch | 1346.5 | 1354.2 | 256x256x64 |
| 1024x512x4096 | bfloat16 | 8 | one-tile reduce-scatter, n = 8 | 175.9 | 60.3 | flydsl | 281.8 | 261.0 | 64x32x256 |
| 1024x1024x4096 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 | 308.0 | 121.1 | flydsl | 442.1 | 441.3 | 64x64x256 |
| 1024x2048x4096 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 463.7 | 240.3 | flydsl | 597.8 | 496.6 | 128x128x256 |
| 1024x4096x4096 | bfloat16 | 64 | one-tile owner, n = 2 | 625.3 | 468.6 | flydsl | 829.6 | 708.8 | 128x128x256 |
| 1024x8192x4096 | bfloat16 | 128 | one-tile owner, n = 2 | 1008.5 | 869.9 | flydsl | 1014.3 | 671.8 | 256x256x64 |
| 1024x8704x4096 | bfloat16 | 136 | data-parallel | 920.8 | 922.8 | torch | 993.6 | 992.2 | 144x256x64 |
| 1024x10240x4096 | bfloat16 | 160 | data-parallel | 1052.1 | 1053.4 | torch | 1111.2 | 1119.0 | 160x256x64 |
| 1024x16384x4096 | bfloat16 | 256 | data-parallel | 1387.4 | 1399.0 | flydsl | 825.8 | 1321.4 | 256x256x64 |
| 1024x129280x4096 | bfloat16 | 2020 | data-parallel | 1450.2 | 1425.7 | torch | 1370.0 | 1379.0 | 256x256x64 |
| 1280x512x4096 | bfloat16 | 10 | one-tile reduce-scatter, n = 8 | 214.8 | 75.1 | flydsl | 320.8 | 310.2 | 64x48x256 |
| 1280x1024x4096 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 | 344.2 | 150.9 | torch | 483.9 | 487.9 | 64x80x256 |
| 1280x2048x4096 | bfloat16 | 40 | one-tile owner, n = 3 | 466.7 | 302.2 | flydsl | 624.1 | 611.5 | 128x80x128 |
| 1280x129280x4096 | bfloat16 | 2525 | data-parallel | 1408.1 | 1403.4 | torch | 1387.0 | 1397.6 | 256x256x64 |
| 1536x512x4096 | bfloat16 | 12 | one-tile reduce-scatter, n = 8 | 249.8 | 90.2 | torch | 366.2 | 367.5 | 64x48x256 |
| 1536x1024x4096 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 | 378.6 | 182.5 | flydsl | 521.4 | 504.7 | 64x96x256 |
| 1536x2048x4096 | bfloat16 | 48 | one-tile owner, n = 3 | 551.8 | 362.7 | flydsl | 700.2 | 685.1 | 128x96x128 |
| 1536x129280x4096 | bfloat16 | 3030 | data-parallel | 1435.7 | 1412.6 | torch | 1391.9 | 1402.5 | 256x256x64 |
| 1792x512x4096 | bfloat16 | 14 | one-tile reduce-scatter, n = 8 | 278.8 | 105.7 | flydsl | 394.0 | 384.5 | 64x64x256 |
| 1792x1024x4096 | bfloat16 | 28 | one-tile reduce-scatter, n = 4 | 419.1 | 212.7 | flydsl | 578.5 | 463.3 | 256x128x64 |
| 1792x2048x4096 | bfloat16 | 56 | one-tile owner, n = 2 | 555.9 | 418.3 | flydsl | 753.4 | 669.0 | 128x128x256 |
| 1792x129280x4096 | bfloat16 | 3535 | data-parallel | 1452.9 | 1436.4 | asm | 1418.2 | 1356.1 | 256x256x64 |
| 2048x512x4096 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 | 306.0 | 121.5 | flydsl | 428.3 | 430.4 | 64x64x256 |
| 2048x1024x4096 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 464.0 | 243.0 | flydsl | 599.6 | 494.9 | 128x128x256 |
| 2048x2048x4096 | bfloat16 | 64 | one-tile owner, n = 2 | 630.6 | 473.8 | flydsl | 841.0 | 736.9 | 128x128x256 |
| 2048x4096x4096 | bfloat16 | 128 | one-tile owner, n = 2 | 1029.9 | 880.2 | opus | 1034.0 | 704.1 | 256x256x64 |
| 2048x8192x4096 | bfloat16 | 256 | data-parallel | 1390.3 | 1392.6 | flydsl | 1280.5 | 1352.2 | 256x256x64 |
| 2048x8704x4096 | bfloat16 | 272 | one-tile reduce-scatter, n = 8 | 1205.2 | 926.7 | torch | 1055.6 | 1061.7 | 224x160x64 |
| 2048x10240x4096 | bfloat16 | 320 | one-tile owner, n = 2 | 1193.6 | 1069.8 | torch | 1129.1 | 1136.6 | 160x256x64 |
| 2048x16384x4096 | bfloat16 | 512 | data-parallel | 1441.6 | 1438.2 | flydsl | 1361.3 | 1370.5 | 256x256x64 |
| 2048x129280x4096 | bfloat16 | 4040 | data-parallel | 1484.4 | 1472.1 | flydsl | 1424.5 | 1379.1 | 256x256x64 |
| 2304x512x4096 | bfloat16 | 18 | one-tile reduce-scatter, n = 7 | 304.9 | 135.6 | flydsl | 351.7 | 444.5 | 64x80x256 |
| 2304x1024x4096 | bfloat16 | 36 | one-tile owner, n = 3 | 422.0 | 272.7 | flydsl | 510.1 | 648.7 | 128x80x128 |
| 2304x2048x4096 | bfloat16 | 72 | one-tile owner, n = 2 | 694.5 | 527.5 | opus | 680.4 | 879.5 | 128x160x128 |
| 2560x512x4096 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 | 341.2 | 150.2 | flydsl | 385.7 | 486.3 | 64x80x256 |
| 2560x1024x4096 | bfloat16 | 40 | one-tile owner, n = 3 | 467.4 | 301.3 | flydsl | 554.6 | 606.3 | 128x80x128 |
| 2560x2048x4096 | bfloat16 | 80 | one-tile owner, n = 2 | 758.8 | 581.9 | opus | 761.5 | 885.4 | 128x160x128 |
| 2816x512x4096 | bfloat16 | 22 | one-tile reduce-scatter, n = 5 | 344.4 | 164.7 | flydsl | 418.6 | 465.9 | 64x96x256 |
| 2816x1024x4096 | bfloat16 | 44 | one-tile owner, n = 3 | 508.1 | 331.5 | flydsl | 601.8 | 641.8 | 128x96x128 |
| 2816x2048x4096 | bfloat16 | 88 | one-tile owner, n = 2-3 | 803.2 | 635.4 | opus | 814.0 | 851.5 | 192x128x128 |
| 3072x512x4096 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 | 370.4 | 180.9 | flydsl | 448.0 | 502.2 | 64x96x256 |
| 3072x1024x4096 | bfloat16 | 48 | one-tile owner, n = 3 | 549.5 | 362.0 | flydsl | 644.5 | 675.0 | 128x96x128 |
| 3072x2048x4096 | bfloat16 | 96 | one-tile owner, n = 2-3 | 766.0 | 687.1 | opus | 921.0 | 908.1 | 128x192x128 |
| 3328x512x4096 | bfloat16 | 26 | one-tile reduce-scatter, n = 4 | 394.3 | 196.4 | flydsl | 484.2 | 443.5 | 256x112x64 |
| 3328x1024x4096 | bfloat16 | 52 | one-tile owner, n = 2 | 526.2 | 389.5 | flydsl | 679.1 | 625.7 | 128x128x256 |
| 3328x2048x4096 | bfloat16 | 104 | one-tile owner, n = 2-3 | 843.4 | 735.2 | opus | 935.7 | 771.2 | 256x208x64 |
| 3584x512x4096 | bfloat16 | 28 | one-tile reduce-scatter, n = 4 | 417.3 | 211.0 | flydsl | 509.5 | 449.0 | 256x128x64 |
| 3584x1024x4096 | bfloat16 | 56 | one-tile owner, n = 2 | 552.0 | 414.6 | flydsl | 710.3 | 662.1 | 128x128x256 |
| 3584x2048x4096 | bfloat16 | 112 | one-tile owner, n = 2-3 | 887.0 | 784.8 | opus | 1003.2 | 878.6 | 160x192x128 |
| 3840x512x4096 | bfloat16 | 30 | one-tile reduce-scatter, n = 4 | 438.8 | 226.1 | flydsl | 534.2 | 485.7 | 128x128x256 |
| 3840x1024x4096 | bfloat16 | 60 | one-tile owner, n = 2 | 586.8 | 441.7 | flydsl | 737.5 | 699.7 | 128x128x256 |
| 3840x2048x4096 | bfloat16 | 120 | one-tile owner, n = 2-3 | 999.4 | 832.6 | opus | 1038.1 | 737.1 | 256x240x64 |
| 4096x512x4096 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 461.2 | 238.9 | flydsl | 590.5 | 500.9 | 128x128x256 |
| 4096x1024x4096 | bfloat16 | 64 | one-tile owner, n = 2 | 626.5 | 471.1 | flydsl | 829.3 | 735.1 | 128x128x256 |
| 4096x2048x4096 | bfloat16 | 128 | one-tile owner, n = 2 | 1022.5 | 875.8 | opus | 1083.3 | 678.8 | 256x256x64 |
| 4096x4096x4096 | bfloat16 | 256 | data-parallel | 1390.2 | 1391.3 | flydsl | 1314.0 | 1331.2 | 256x256x64 |
| 4096x8192x4096 | bfloat16 | 512 | data-parallel | 1433.8 | 1425.7 | asm | 1362.8 | 1372.9 | 256x256x64 |
| 4096x8704x4096 | bfloat16 | 544 | one-tile reduce-scatter, n = 4 | 1328.8 | 1155.4 | torch | 1193.6 | 1181.4 | 192x256x64 |
| 4096x10240x4096 | bfloat16 | 640 | one-tile owner, n = 2 | 1377.9 | 1332.3 | flydsl | 1285.9 | 1221.9 | 224x256x64 |
| 4096x16384x4096 | bfloat16 | 1024 | data-parallel | 1464.1 | 1459.3 | flydsl | 1387.9 | 1414.3 | 256x256x64 |
| 4096x129280x4096 | bfloat16 | 8080 | data-parallel | 1494.5 | 1486.0 | asm | 1428.5 | 1388.1 | 256x256x64 |
| 4352x512x4096 | bfloat16 | 34 | one-tile owner, n = 3 | 397.0 | 251.7 | flydsl | 465.6 | 522.2 | 128x80x128 |
| 4352x1024x4096 | bfloat16 | 68 | one-tile owner, n = 2 | 656.5 | 493.4 | opus | 638.6 | 805.4 | 128x160x128 |
| 4352x2048x4096 | bfloat16 | 136 | data-parallel | 927.8 | 930.0 | torch | 1016.2 | 1020.7 | 192x192x64 |
| 5120x512x4096 | bfloat16 | 40 | one-tile owner, n = 3 | 461.3 | 298.3 | flydsl | 528.6 | 597.1 | 128x80x128 |
| 5120x1024x4096 | bfloat16 | 80 | one-tile owner, n = 2 | 751.9 | 576.3 | opus | 762.9 | 879.3 | 128x160x128 |
| 5120x2048x4096 | bfloat16 | 160 | data-parallel | 1054.4 | 1055.2 | torch | 1148.6 | 1145.8 | 256x160x64 |
| 6144x512x4096 | bfloat16 | 48 | one-tile owner, n = 3 | 540.9 | 355.9 | flydsl | 689.2 | 665.5 | 128x96x128 |
| 6144x1024x4096 | bfloat16 | 96 | one-tile owner, n = 2-3 | 776.4 | 680.0 | opus | 923.7 | 907.6 | 128x192x128 |
| 6144x2048x4096 | bfloat16 | 192 | data-parallel | 1221.0 | 1217.2 | opus | 1227.3 | 1208.5 | 256x192x64 |
| 6144x129280x4096 | bfloat16 | 12120 | one-tile owner, n = 2-3 | 1463.3 | 1463.7 | asm | 1412.1 | 1397.0 | 256x256x64 |
| 6400x512x4096 | bfloat16 | 50 | one-tile owner, n = 3 | 556.4 | 368.5 | flydsl | 597.1 | 607.0 | 128x128x256 |
| 6400x1024x4096 | bfloat16 | 100 | one-tile owner, n = 2-3 | 831.7 | 703.1 | opus | 880.0 | 764.0 | 256x208x64 |
| 6400x2048x4096 | bfloat16 | 200 | data-parallel | 1245.7 | 1246.6 | torch | 1103.4 | 1110.7 | 256x208x64 |
| 6912x512x4096 | bfloat16 | 54 | one-tile owner, n = 2 | 531.9 | 396.3 | flydsl | 611.1 | 643.8 | 128x128x256 |
| 6912x1024x4096 | bfloat16 | 108 | one-tile owner, n = 2-3 | 892.0 | 749.8 | opus | 929.2 | 690.2 | 256x224x64 |
| 6912x2048x4096 | bfloat16 | 216 | data-parallel | 1293.4 | 1290.7 | torch | 1217.4 | 1221.7 | 256x224x64 |
| 7168x512x4096 | bfloat16 | 56 | one-tile owner, n = 2 | 547.3 | 410.1 | flydsl | 642.9 | 664.7 | 128x128x256 |
| 7168x1024x4096 | bfloat16 | 112 | one-tile owner, n = 2-3 | 940.0 | 779.1 | opus | 999.6 | 1051.7 | 256x128x64 |
| 7168x2048x4096 | bfloat16 | 224 | data-parallel | 1321.4 | 1315.6 | torch | 1236.8 | 1245.9 | 256x224x64 |
| 7424x512x4096 | bfloat16 | 58 | one-tile owner, n = 2 | 568.2 | 423.1 | flydsl | 622.7 | 655.1 | 128x128x256 |
| 7424x1024x4096 | bfloat16 | 116 | one-tile owner, n = 2-3 | 970.8 | 799.4 | opus | 973.8 | 741.1 | 256x240x64 |
| 7424x2048x4096 | bfloat16 | 232 | data-parallel | 1336.6 | 1340.8 | torch | 1121.2 | 1125.1 | 256x240x64 |
| 7680x512x4096 | bfloat16 | 60 | one-tile owner, n = 2 | 583.5 | 437.7 | flydsl | 662.0 | 671.7 | 128x128x256 |
| 7680x1024x4096 | bfloat16 | 120 | one-tile owner, n = 2-3 | 979.3 | 826.4 | opus | 1029.7 | 746.5 | 256x240x64 |
| 7680x2048x4096 | bfloat16 | 240 | data-parallel | 1364.4 | 1365.9 | torch | 1148.4 | 1154.7 | 256x240x64 |
| 7936x512x4096 | bfloat16 | 62 | one-tile owner, n = 2 | 602.9 | 452.8 | flydsl | 712.1 | 689.0 | 128x128x256 |
| 7936x1024x4096 | bfloat16 | 124 | one-tile owner, n = 2-3 | 1018.3 | 843.8 | opus | 1010.8 | 702.4 | 256x256x64 |
| 7936x2048x4096 | bfloat16 | 248 | data-parallel | 1383.4 | 1392.3 | torch | 1333.1 | 1356.3 | 256x256x64 |
| 8192x512x4096 | bfloat16 | 64 | one-tile owner, n = 2 | 614.8 | 464.4 | flydsl | 784.9 | 704.6 | 128x128x256 |
| 8192x1024x4096 | bfloat16 | 128 | one-tile owner, n = 2 | 1012.2 | 862.0 | opus | 1074.7 | 683.5 | 256x256x64 |
| 8192x2048x4096 | bfloat16 | 256 | data-parallel | 1387.2 | 1389.5 | torch | 1352.7 | 1364.9 | 256x256x64 |
| 8192x4096x4096 | bfloat16 | 512 | data-parallel | 1440.6 | 1438.4 | flydsl | 1356.8 | 1403.2 | 256x256x64 |
| 8192x8192x4096 | bfloat16 | 1024 | data-parallel | 1466.0 | 1459.7 | asm | 1406.2 | 1414.9 | 256x256x64 |
| 8192x8704x4096 | bfloat16 | 1088 | one-tile owner, n = 2 | 1391.2 | 1322.4 | asm | 1278.3 | 1283.5 | 256x224x64 |
| 8192x10240x4096 | bfloat16 | 1280 | data-parallel | 1476.8 | 1474.0 | torch | 1421.3 | 1425.6 | 256x256x64 |
| 8192x16384x4096 | bfloat16 | 2048 | data-parallel | 1494.1 | 1486.4 | torch | 1437.6 | 1447.5 | 256x256x64 |
| 8192x129280x4096 | bfloat16 | 16160 | one-tile reduce-scatter, n = 4 | 1508.2 | 1490.3 | flydsl | 1423.3 | 1384.2 | 256x256x64 |
| 10240x512x4096 | bfloat16 | 80 | one-tile owner, n = 2 | 736.6 | 567.7 | torch | 854.4 | 853.4 | 128x160x128 |
| 10240x1024x4096 | bfloat16 | 160 | data-parallel | 1097.9 | 1094.9 | torch | 1129.9 | 1134.9 | 256x160x64 |
| 10240x2048x4096 | bfloat16 | 320 | one-tile owner, n = 2 | 1203.5 | 1077.1 | torch | 1173.8 | 1185.4 | 256x160x64 |
| 10240x129280x4096 | bfloat16 | 20200 | data-parallel | 1471.6 | 1461.8 | asm | 1439.9 | 1384.0 | 256x256x64 |
| 12288x512x4096 | bfloat16 | 96 | one-tile owner, n = 2-3 | 763.1 | 668.0 | opus | 887.6 | 889.9 | 128x192x128 |
| 12288x1024x4096 | bfloat16 | 192 | data-parallel | 1188.7 | 1190.5 | opus | 1246.3 | 1211.4 | 256x192x64 |
| 12288x2048x4096 | bfloat16 | 384 | one-tile owner, n = 2 | 1305.3 | 1254.5 | opus | 1282.5 | 1248.4 | 256x192x64 |
| 12288x129280x4096 | bfloat16 | 24240 | data-parallel | 1498.1 | 1493.4 | asm | 1439.1 | 1387.3 | 256x256x64 |
| 14336x512x4096 | bfloat16 | 112 | one-tile owner, n = 2-3 | 914.6 | 763.3 | opus | 956.3 | 699.3 | 256x224x64 |
| 14336x1024x4096 | bfloat16 | 224 | data-parallel | 1287.5 | 1287.4 | torch | 1245.3 | 1253.8 | 256x224x64 |
| 14336x2048x4096 | bfloat16 | 448 | data-parallel | 1339.2 | 1337.5 | flydsl | 1297.1 | 1290.5 | 256x224x64 |
| 14336x129280x4096 | bfloat16 | 28280 | one-tile owner, n = 2-3 | 1448.5 | 1484.6 | asm | 1429.1 | 1377.7 | 256x256x64 |
| 16384x512x4096 | bfloat16 | 128 | one-tile owner, n = 2 | 1000.6 | 848.5 | opus | 1029.8 | 781.1 | 256x256x64 |
| 16384x1024x4096 | bfloat16 | 256 | data-parallel | 1392.3 | 1392.4 | flydsl | 1310.3 | 1321.9 | 256x256x64 |
| 16384x2048x4096 | bfloat16 | 512 | data-parallel | 1425.2 | 1421.4 | torch | 1377.3 | 1382.7 | 256x256x64 |
| 16384x16384x4096 | bfloat16 | 4096 | data-parallel | 1518.6 | 1503.0 | torch | 1453.7 | 1477.3 | 256x256x64 |
| 16384x129280x4096 | bfloat16 | 32320 | one-tile owner, n = 2 | 1457.5 | 1491.8 | asm | 1428.8 | 1392.1 | 256x256x64 |
| 32768x16384x4096 | bfloat16 | 8192 | data-parallel | 1497.4 | 1489.0 | flydsl | 1402.0 | 1435.3 | 256x256x64 |
| 512x7168x4224 | bfloat16 | 56 | one-tile owner, n = 2 | 540.1 | 374.7 | flydsl | 724.8 | 713.9 | 128x128x128 |
| 1024x7168x4224 | bfloat16 | 112 | one-tile owner, n = 2-3 | 934.0 | 738.1 | flydsl | 932.1 | 793.6 | 240x256x64 |
| 10240x7168x4224 | bfloat16 | 1120 | one-tile owner, n = 2-3 | 1411.3 | 1353.8 | opus | 909.0 | 1274.4 | 224x256x64 |
| 512x6144x4608 | bfloat16 | 48 | one-tile owner, n = 3 | 555.0 | 336.0 | flydsl | 686.5 | 614.4 | 192x128x128 |
| 1024x6144x4608 | bfloat16 | 96 | one-tile owner, n = 2-3 | 793.4 | 660.8 | torch | 871.3 | 872.6 | 192x128x128 |
| 2048x6144x4608 | bfloat16 | 192 | data-parallel | 1228.7 | 1227.7 | flydsl | 1184.0 | 1221.3 | 192x256x64 |
| 4096x6144x4608 | bfloat16 | 384 | one-tile owner, n = 2 | 1328.1 | 1263.8 | torch | 1236.0 | 1246.6 | 192x256x64 |
| 8192x6144x4608 | bfloat16 | 768 | data-parallel | 1456.4 | 1454.8 | torch | 1427.4 | 1434.5 | 256x256x64 |
| 16384x6144x4608 | bfloat16 | 1536 | data-parallel | 1490.2 | 1477.2 | torch | 1443.4 | 1453.2 | 256x256x64 |
| 32768x6144x4608 | bfloat16 | 3072 | data-parallel | 1516.3 | 1500.5 | torch | 1460.8 | 1480.0 | 256x256x64 |
| 512x1024x5120 | float32 | 8 | one-tile reduce-scatter, n = 10 | 191.7 | 51.4 | asm | 247.8 | 274.7 | 64x32x256 |
| 512x5120x5120 | bfloat16 | 40 | one-tile owner, n = 3 | 486.7 | 268.2 | torch | 653.7 | 655.3 | 96x128x128 |
| 512x6400x5120 | bfloat16 | 50 | one-tile owner, n = 3 | 597.4 | 336.8 | flydsl | 603.9 | 556.4 | 128x128x256 |
| 512x12800x5120 | bfloat16 | 100 | one-tile owner, n = 2-3 | 845.0 | 651.3 | flydsl | 657.1 | 808.2 | 208x256x64 |
| 512x37888x5120 | bfloat16 | 296 | one-tile owner, n = 3 | 1149.8 | 931.3 | torch | 1043.9 | 1051.8 | 160x256x64 |
| 512x51200x5120 | bfloat16 | 400 | data-parallel | 1195.5 | 1189.1 | flydsl | 1129.5 | 1078.4 | 208x256x64 |
| 768x1024x5120 | float32 | 12 | one-tile reduce-scatter, n = 10 | 271.6 | 77.9 | asm | 311.6 | 312.0 | 64x64x256 |
| 768x37888x5120 | bfloat16 | 444 | data-parallel | 1301.5 | 1301.2 | torch | 1210.0 | 1213.0 | 224x256x64 |
| 1024x1024x5120 | float32 | 16 | one-tile reduce-scatter, n = 8 | 329.6 | 104.2 | asm | 332.4 | 401.6 | 64x64x256 |
| 1024x5120x5120 | bfloat16 | 80 | one-tile owner, n = 2 | 766.7 | 533.8 | torch | 840.6 | 842.1 | 160x128x128 |
| 1024x6400x5120 | bfloat16 | 100 | one-tile owner, n = 2-3 | 852.5 | 665.8 | flydsl | 823.1 | 798.8 | 256x208x64 |
| 1024x12800x5120 | bfloat16 | 200 | data-parallel | 1220.6 | 1226.1 | flydsl | 724.6 | 1054.3 | 256x208x64 |
| 1024x37888x5120 | bfloat16 | 592 | one-tile owner, n = 2 | 1314.3 | 1206.5 | torch | 1054.5 | 1060.1 | 256x208x64 |
| 1024x51200x5120 | bfloat16 | 800 | one-tile reduce-scatter, n = 4 | 1359.8 | 1203.9 | asm | 1147.5 | 1067.2 | 256x208x64 |
| 1536x1024x5120 | float32 | 24 | one-tile reduce-scatter, n = 5 | 399.4 | 157.1 | asm | 436.2 | 512.0 | 64x96x128 |
| 1536x37888x5120 | bfloat16 | 888 | one-tile owner, n = 2-3 | 1382.3 | 1326.2 | torch | 1242.7 | 1253.2 | 224x256x64 |
| 2048x1024x5120 | float32 | 32 | one-tile reduce-scatter, n = 4 | 471.8 | 210.6 | opus | 428.4 | 581.2 | 128x128x128 |
| 2048x5120x5120 | bfloat16 | 160 | data-parallel | 1045.7 | 1048.6 | torch | 1140.3 | 1149.0 | 160x256x64 |
| 2048x6400x5120 | bfloat16 | 200 | data-parallel | 1245.6 | 1247.1 | flydsl | 1128.4 | 1088.7 | 256x208x64 |
| 2048x12800x5120 | bfloat16 | 400 | data-parallel | 1289.0 | 1286.3 | flydsl | 809.3 | 1076.8 | 256x208x64 |
| 2048x37888x5120 | bfloat16 | 1184 | data-parallel | 1429.9 | 1417.1 | torch | 1187.9 | 1197.1 | 240x256x64 |
| 2048x51200x5120 | bfloat16 | 1600 | one-tile owner, n = 2 | 1441.2 | 1383.4 | asm | 209.5 | 1173.3 | 240x256x64 |
| 4096x1024x5120 | float32 | 64 | one-tile owner, n = 2 | 590.1 | 410.0 | opus | 709.3 | 765.4 | 128x128x128 |
| 4096x5120x5120 | bfloat16 | 320 | one-tile owner, n = 2 | 1240.1 | 1072.7 | torch | 1162.2 | 1170.7 | 160x256x64 |
| 4096x6400x5120 | bfloat16 | 400 | data-parallel | 1301.2 | 1296.2 | flydsl | 1240.1 | 1108.4 | 256x208x64 |
| 4096x12800x5120 | bfloat16 | 800 | one-tile reduce-scatter, n = 4 | 1404.6 | 1252.6 | asm | 1203.4 | 1152.6 | 256x208x64 |
| 4096x51200x5120 | bfloat16 | 3200 | one-tile owner, n = 2 | 1513.6 | 1477.3 | asm | 1398.3 | 1380.0 | 256x256x64 |
| 8192x1024x5120 | float32 | 128 | one-tile owner, n = 2 | 1024.2 | 793.7 | opus | 1019.2 | 981.0 | 256x128x64 |
| 8192x5120x5120 | bfloat16 | 640 | one-tile owner, n = 2 | 1413.3 | 1342.4 | torch | 1276.2 | 1290.2 | 256x224x64 |
| 8192x6400x5120 | bfloat16 | 800 | one-tile reduce-scatter, n = 4 | 1410.9 | 1247.9 | flydsl | 1204.5 | 1177.9 | 256x208x64 |
| 8192x12800x5120 | bfloat16 | 1600 | one-tile owner, n = 2 | 1446.5 | 1385.3 | asm | 1334.1 | 1182.4 | 256x240x64 |
| 8192x51200x5120 | bfloat16 | 6400 | data-parallel | 1539.2 | 1529.3 | flydsl | 1458.4 | 1414.9 | 256x256x64 |
| 16384x1024x5120 | float32 | 256 | data-parallel | 1321.9 | 1318.6 | opus | 1165.2 | 1215.9 | 256x256x64 |
| 16384x5120x5120 | bfloat16 | 1280 | data-parallel | 1482.4 | 1474.5 | torch | 1455.2 | 1465.3 | 256x256x64 |
| 16384x6400x5120 | bfloat16 | 1600 | one-tile owner, n = 2 | 1452.0 | 1381.0 | flydsl | 1325.5 | 1196.3 | 256x240x64 |
| 16384x12800x5120 | bfloat16 | 3200 | one-tile owner, n = 2 | 1515.1 | 1478.6 | torch | 1454.2 | 1472.8 | 256x256x64 |
| 16384x51200x5120 | bfloat16 | 12800 | data-parallel | 1549.5 | 1532.6 | flydsl | 1481.6 | 1411.3 | 256x256x64 |
| 32768x5120x5120 | bfloat16 | 2560 | data-parallel | 1521.7 | 1503.1 | torch | 1474.6 | 1487.2 | 256x256x64 |
| 32768x6400x5120 | bfloat16 | 3200 | one-tile owner, n = 2 | 1510.7 | 1469.3 | torch | 1464.1 | 1475.1 | 256x256x64 |
| 32768x12800x5120 | bfloat16 | 6400 | data-parallel | 1535.8 | 1514.7 | torch | 1489.5 | 1515.5 | 256x256x64 |
| 32768x51200x5120 | bfloat16 | 25600 | data-parallel | 1541.9 | 1529.0 | flydsl | 1433.8 | 1395.1 | 256x256x64 |
| 512x1280x6144 | bfloat16 | 10 | one-tile reduce-scatter, n = 12 | 268.3 | 67.2 | flydsl | 340.6 | 355.9 | 160x64x128 |
| 512x1536x6144 | bfloat16 | 12 | one-tile reduce-scatter, n = 10 | 284.1 | 80.9 | flydsl | 382.7 | 381.1 | 192x64x128 |
| 512x2048x6144 | float32 | 16 | one-tile reduce-scatter, n = 8 | 358.0 | 104.5 | asm | 340.8 | 421.2 | 64x64x256 |
| 512x2304x6144 | bfloat16 | 18 | one-tile reduce-scatter, n = 7 | 350.4 | 121.1 | flydsl | 432.4 | 372.1 | 192x128x128 |
| 512x2560x6144 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 | 401.9 | 135.2 | torch | 471.0 | 473.1 | 256x128x64 |
| 512x3072x6144 | float32 | 24 | one-tile reduce-scatter, n = 5 | 406.3 | 157.3 | opus | 401.3 | 549.7 | 96x128x128 |
| 512x3072x6144 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 | 421.8 | 161.4 | flydsl | 542.5 | 460.0 | 192x128x128 |
| 512x6144x6144 | bfloat16 | 48 | one-tile owner, n = 3 | 613.5 | 326.0 | flydsl | 687.9 | 682.0 | 192x128x128 |
| 512x9216x6144 | bfloat16 | 72 | one-tile owner, n = 3-4 | 791.4 | 480.2 | flydsl | 596.7 | 589.4 | 256x256x64 |
| 1024x2048x6144 | float32 | 32 | one-tile reduce-scatter, n = 4 | 498.1 | 211.1 | opus | 405.4 | 646.6 | 128x128x128 |
| 1024x2304x6144 | bfloat16 | 36 | one-tile owner, n = 4 | 515.4 | 244.1 | torch | 594.5 | 594.1 | 256x176x64 |
| 1024x2560x6144 | bfloat16 | 40 | one-tile owner, n = 3 | 524.2 | 271.1 | torch | 699.8 | 688.9 | 160x128x128 |
| 1024x3072x6144 | float32 | 48 | one-tile owner, n = 3 | 583.3 | 314.6 | opus | 447.7 | 717.0 | 96x128x128 |
| 1024x6144x6144 | bfloat16 | 96 | one-tile owner, n = 2-3 | 865.4 | 647.9 | torch | 982.1 | 990.7 | 192x256x64 |
| 1024x9216x6144 | bfloat16 | 144 | data-parallel | 947.1 | 942.4 | torch | 1057.1 | 1058.2 | 144x256x64 |
| 2048x2048x6144 | float32 | 64 | one-tile owner, n = 3 | 755.5 | 425.6 | opus | 670.5 | 806.2 | 128x128x128 |
| 2048x2304x6144 | bfloat16 | 72 | one-tile owner, n = 3-4 | 866.6 | 497.0 | torch | 838.6 | 839.0 | 128x160x128 |
| 2048x2560x6144 | bfloat16 | 80 | one-tile owner, n = 2 | 838.3 | 551.9 | torch | 908.9 | 911.8 | 160x128x128 |
| 2048x3072x6144 | float32 | 96 | one-tile owner, n = 2-3 | 889.8 | 637.1 | opus | 684.9 | 913.4 | 192x256x64 |
| 2048x6144x6144 | bfloat16 | 192 | data-parallel | 1233.2 | 1231.5 | torch | 1233.1 | 1242.1 | 192x256x64 |
| 2048x9216x6144 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 | 1274.1 | 975.0 | torch | 1083.7 | 1088.8 | 144x256x64 |
| 4096x2048x6144 | float32 | 128 | one-tile owner, n = 2 | 1090.4 | 813.2 | opus | 1020.2 | 819.5 | 256x256x64 |
| 4096x2304x6144 | bfloat16 | 144 | data-parallel | 967.1 | 962.0 | flydsl | 679.8 | 1087.2 | 144x256x64 |
| 4096x2560x6144 | bfloat16 | 160 | data-parallel | 1067.6 | 1072.6 | flydsl | 637.2 | 1173.2 | 160x256x64 |
| 4096x3072x6144 | float32 | 192 | data-parallel | 1174.4 | 1181.1 | asm | 1139.5 | 1211.3 | 192x256x64 |
| 4096x6144x6144 | bfloat16 | 384 | one-tile owner, n = 2 | 1378.9 | 1279.9 | torch | 1257.2 | 1266.5 | 192x256x64 |
| 4096x9216x6144 | bfloat16 | 576 | one-tile owner, n = 3 | 1395.4 | 1244.6 | torch | 1265.3 | 1270.6 | 192x256x64 |
| 8192x2048x6144 | float32 | 256 | data-parallel | 1341.5 | 1345.5 | opus | 1228.3 | 1221.5 | 256x256x64 |
| 8192x2304x6144 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 | 1216.7 | 972.4 | torch | 1090.2 | 1094.9 | 144x256x64 |
| 8192x2560x6144 | bfloat16 | 320 | one-tile owner, n = 3 | 1272.5 | 1075.4 | asm | 405.5 | 1182.6 | 160x256x64 |
| 8192x3072x6144 | float32 | 384 | one-tile owner, n = 2 | 1286.6 | 1197.0 | asm | 1170.3 | 1216.9 | 192x256x64 |
| 8192x6144x6144 | bfloat16 | 768 | data-parallel | 1486.1 | 1483.6 | flydsl | 1420.2 | 1459.4 | 256x256x64 |
| 8192x9216x6144 | bfloat16 | 1152 | one-tile owner, n = 2 | 1479.1 | 1423.3 | asm | 1381.3 | 1195.9 | 256x240x64 |
| 16384x2048x6144 | float32 | 512 | data-parallel | 1352.3 | 1358.3 | opus | 1220.5 | 1263.8 | 256x256x64 |
| 16384x2304x6144 | bfloat16 | 576 | one-tile owner, n = 3 | 1358.1 | 1225.0 | flydsl | 968.8 | 1287.7 | 192x256x64 |
| 16384x2560x6144 | bfloat16 | 640 | one-tile owner, n = 2 | 1417.3 | 1341.3 | flydsl | 1292.9 | 1299.6 | 256x224x64 |
| 16384x3072x6144 | float32 | 768 | data-parallel | 1393.7 | 1389.4 | asm | 1356.4 | 1309.7 | 256x256x64 |
| 16384x6144x6144 | bfloat16 | 1536 | data-parallel | 1525.5 | 1508.1 | asm | 1478.7 | 1485.9 | 256x256x64 |
| 16384x9216x6144 | bfloat16 | 2304 | data-parallel | 1530.9 | 1522.2 | torch | 1478.9 | 1504.1 | 256x256x64 |
| 32768x2304x6144 | bfloat16 | 1152 | one-tile owner, n = 2 | 1459.1 | 1403.6 | flydsl | 649.3 | 1188.9 | 256x240x64 |
| 32768x2560x6144 | bfloat16 | 1280 | data-parallel | 1496.0 | 1487.3 | flydsl | 1435.1 | 1481.7 | 256x256x64 |
| 32768x6144x6144 | bfloat16 | 3072 | data-parallel | 1545.6 | 1533.5 | torch | 1492.1 | 1517.8 | 256x256x64 |
| 32768x9216x6144 | bfloat16 | 4608 | data-parallel | 1542.4 | 1535.6 | torch | 1499.7 | 1521.4 | 256x256x64 |
| 512x5120x6400 | bfloat16 | 40 | one-tile owner, n = 4 | 567.5 | 279.6 | flydsl | 541.6 | 666.5 | 160x128x128 |
| 1024x5120x6400 | bfloat16 | 80 | one-tile owner, n = 2 | 825.6 | 557.1 | flydsl | 590.8 | 889.7 | 160x128x128 |
| 2048x5120x6400 | bfloat16 | 160 | data-parallel | 1085.2 | 1084.5 | flydsl | 728.8 | 1186.3 | 160x256x64 |
| 4096x5120x6400 | bfloat16 | 320 | one-tile owner, n = 3 | 1317.6 | 1104.8 | torch | 1206.0 | 1211.0 | 160x256x64 |
| 8192x5120x6400 | bfloat16 | 640 | one-tile owner, n = 2 | 1441.1 | 1364.6 | flydsl | 1323.9 | 1303.5 | 256x224x64 |
| 16384x5120x6400 | bfloat16 | 1280 | data-parallel | 1516.1 | 1510.5 | torch | 1473.7 | 1483.4 | 256x256x64 |
| 32768x5120x6400 | bfloat16 | 2560 | data-parallel | 1536.8 | 1527.5 | torch | 1490.0 | 1514.1 | 256x256x64 |
| 512x16384x6656 | bfloat16 | 128 | one-tile owner, n = 2 | 1144.7 | 871.4 | flydsl | 840.3 | 899.8 | 256x256x64 |
| 1024x16384x6656 | bfloat16 | 256 | data-parallel | 1425.1 | 1441.6 | asm | 361.0 | 1393.4 | 256x256x64 |
| 2048x16384x6656 | bfloat16 | 512 | data-parallel | 1502.7 | 1493.7 | asm | 1432.6 | 1441.2 | 256x256x64 |
| 4096x16384x6656 | bfloat16 | 1024 | data-parallel | 1531.7 | 1521.3 | torch | 1460.7 | 1465.3 | 256x256x64 |
| 8192x16384x6656 | bfloat16 | 2048 | data-parallel | 1540.8 | 1528.0 | asm | 1489.2 | 1489.7 | 256x256x64 |
| 16384x16384x6656 | bfloat16 | 4096 | data-parallel | 1555.2 | 1549.8 | asm | 1508.8 | 1516.8 | 256x256x64 |
| 32768x16384x6656 | bfloat16 | 8192 | data-parallel | 1544.3 | 1544.9 | asm | 290.9 | 1507.7 | 256x256x64 |
| 512x1024x7168 | bfloat16 | 8 | one-tile reduce-scatter, n = 14 | 255.9 | 57.0 | torch | 384.8 | 385.3 | 128x128x64 |
| 512x1536x7168 | bfloat16 | 12 | one-tile reduce-scatter, n = 10 | 313.0 | 85.6 | torch | 412.2 | 412.2 | 192x64x128 |
| 512x2048x7168 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 | 411.0 | 113.7 | torch | 536.9 | 539.1 | 128x128x128 |
| 512x3584x7168 | bfloat16 | 28 | one-tile reduce-scatter, n = 4 | 506.4 | 201.7 | flydsl | 560.5 | 535.0 | 128x128x256 |
| 512x6144x7168 | bfloat16 | 48 | one-tile owner, n = 3 | 652.2 | 347.0 | flydsl | 718.9 | 720.2 | 192x128x128 |
| 512x8192x7168 | bfloat16 | 64 | one-tile owner, n = 3 | 818.3 | 458.7 | flydsl | 703.6 | 714.1 | 128x128x256 |
| 512x8448x7168 | bfloat16 | 66 | one-tile owner, n = 3-4 | 787.5 | 472.5 | flydsl | 668.5 | 469.1 | 256x256x64 |
| 512x20480x7168 | bfloat16 | 160 | data-parallel | 1056.6 | 1057.1 | flydsl | 1020.2 | 1143.6 | 160x256x64 |
| 768x1024x7168 | bfloat16 | 12 | one-tile reduce-scatter, n = 10 | 320.9 | 86.1 | torch | 403.4 | 403.4 | 256x96x64 |
| 768x2048x7168 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 | 479.0 | 172.5 | flydsl | 558.2 | 569.3 | 256x128x64 |
| 1024x512x7168 | bfloat16 | 8 | one-tile reduce-scatter, n = 14 | 258.7 | 56.6 | flydsl | 333.4 | 289.6 | 256x64x64 |
| 1024x1024x7168 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 | 414.8 | 116.1 | torch | 546.1 | 548.5 | 128x128x128 |
| 1024x1536x7168 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 | 481.4 | 172.9 | flydsl | 564.1 | 569.4 | 256x128x64 |
| 1024x2048x7168 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 567.0 | 230.4 | torch | 724.7 | 723.9 | 256x128x64 |
| 1024x3584x7168 | bfloat16 | 56 | one-tile owner, n = 3 | 755.9 | 404.6 | flydsl | 775.7 | 678.9 | 128x128x256 |
| 1024x6144x7168 | bfloat16 | 96 | one-tile owner, n = 2-3 | 876.1 | 686.4 | torch | 953.0 | 968.7 | 192x256x64 |
| 1024x8192x7168 | bfloat16 | 128 | one-tile owner, n = 2 | 1157.9 | 892.6 | flydsl | 1006.7 | 927.7 | 256x256x64 |
| 1024x8448x7168 | bfloat16 | 132 | data-parallel | 917.9 | 917.2 | torch | 1029.3 | 1035.3 | 144x256x64 |
| 1024x20480x7168 | bfloat16 | 320 | one-tile owner, n = 3 | 1291.5 | 1065.5 | flydsl | 744.2 | 1183.8 | 160x256x64 |
| 1280x512x7168 | bfloat16 | 10 | one-tile reduce-scatter, n = 12 | 287.2 | 71.6 | torch | 360.1 | 361.7 | 256x80x64 |
| 1280x1024x7168 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 | 424.3 | 144.7 | torch | 541.8 | 542.2 | 256x80x128 |
| 1280x2048x7168 | bfloat16 | 40 | one-tile owner, n = 4 | 618.0 | 290.0 | torch | 736.3 | 735.4 | 128x160x128 |
| 1536x512x7168 | bfloat16 | 12 | one-tile reduce-scatter, n = 10 | 319.6 | 86.9 | torch | 401.9 | 402.7 | 256x96x64 |
| 1536x1024x7168 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 | 481.4 | 174.8 | torch | 640.8 | 642.4 | 128x96x128 |
| 1536x2048x7168 | bfloat16 | 48 | one-tile owner, n = 3 | 663.1 | 351.3 | torch | 834.3 | 827.5 | 128x192x64 |
| 1792x512x7168 | bfloat16 | 14 | one-tile reduce-scatter, n = 9 | 343.2 | 100.8 | torch | 437.3 | 437.1 | 256x112x64 |
| 1792x1024x7168 | bfloat16 | 28 | one-tile reduce-scatter, n = 4 | 510.8 | 205.6 | flydsl | 603.0 | 603.0 | 256x128x64 |
| 1792x2048x7168 | bfloat16 | 56 | one-tile owner, n = 3 | 762.3 | 409.3 | flydsl | 778.6 | 686.8 | 128x128x256 |
| 2048x512x7168 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 | 415.2 | 116.4 | torch | 502.7 | 503.9 | 128x128x64 |
| 2048x1024x7168 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 569.7 | 236.0 | torch | 715.9 | 715.0 | 128x128x128 |
| 2048x2048x7168 | bfloat16 | 64 | one-tile owner, n = 3 | 848.3 | 469.4 | flydsl | 872.1 | 768.7 | 128x128x256 |
| 2048x3584x7168 | bfloat16 | 112 | one-tile owner, n = 2-3 | 1055.4 | 804.3 | opus | 999.6 | 990.1 | 224x256x64 |
| 2048x8192x7168 | bfloat16 | 256 | data-parallel | 1450.0 | 1447.2 | asm | 1371.4 | 1410.9 | 256x256x64 |
| 2048x20480x7168 | bfloat16 | 640 | one-tile owner, n = 2 | 1445.8 | 1372.2 | flydsl | 827.8 | 1254.4 | 224x256x64 |
| 4096x512x7168 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 563.6 | 231.6 | flydsl | 628.5 | 595.4 | 128x128x256 |
| 4096x1024x7168 | bfloat16 | 64 | one-tile owner, n = 3 | 838.4 | 467.5 | torch | 905.2 | 908.5 | 256x128x64 |
| 4096x2048x7168 | bfloat16 | 128 | one-tile owner, n = 2 | 1165.8 | 902.0 | torch | 1164.4 | 1172.8 | 256x128x64 |
| 4096x3584x7168 | bfloat16 | 224 | data-parallel | 1381.8 | 1390.1 | flydsl | 1324.4 | 1277.1 | 224x256x64 |
| 4096x8192x7168 | bfloat16 | 512 | data-parallel | 1487.1 | 1487.6 | flydsl | 1411.8 | 1443.2 | 256x256x64 |
| 4096x20480x7168 | bfloat16 | 1280 | data-parallel | 1530.3 | 1513.2 | flydsl | 1443.1 | 1451.3 | 256x256x64 |
| 6144x512x7168 | bfloat16 | 48 | one-tile owner, n = 3 | 642.7 | 344.2 | flydsl | 722.4 | 655.4 | 128x192x128 |
| 6144x1024x7168 | bfloat16 | 96 | one-tile owner, n = 2-3 | 929.7 | 684.7 | opus | 967.1 | 865.1 | 256x192x64 |
| 6144x2048x7168 | bfloat16 | 192 | data-parallel | 1252.9 | 1267.6 | opus | 1300.8 | 1260.9 | 256x192x64 |
| 8192x512x7168 | bfloat16 | 64 | one-tile owner, n = 3 | 817.5 | 458.9 | flydsl | 839.9 | 748.6 | 128x128x256 |
| 8192x1024x7168 | bfloat16 | 128 | one-tile owner, n = 2 | 1160.0 | 893.3 | torch | 1087.6 | 1093.6 | 256x128x64 |
| 8192x2048x7168 | bfloat16 | 256 | data-parallel | 1452.5 | 1451.5 | torch | 1383.0 | 1387.7 | 256x256x64 |
| 8192x3584x7168 | bfloat16 | 448 | data-parallel | 1403.6 | 1397.8 | flydsl | 1348.6 | 1306.7 | 224x256x64 |
| 8192x8192x7168 | bfloat16 | 1024 | data-parallel | 1514.4 | 1500.8 | flydsl | 1430.8 | 1457.2 | 256x256x64 |
| 8192x20480x7168 | bfloat16 | 2560 | data-parallel | 1537.8 | 1527.1 | asm | 1464.7 | 1469.5 | 256x256x64 |
| 10240x512x7168 | bfloat16 | 80 | one-tile owner, n = 3-4 | 941.0 | 563.1 | torch | 849.4 | 838.4 | 256x160x64 |
| 10240x1024x7168 | bfloat16 | 160 | data-parallel | 1106.8 | 1102.4 | torch | 1205.0 | 1213.5 | 256x160x64 |
| 10240x2048x7168 | bfloat16 | 320 | one-tile owner, n = 3 | 1328.1 | 1097.5 | torch | 1231.1 | 1245.3 | 256x160x64 |
| 10240x8448x7168 | bfloat16 | 1320 | one-tile owner, n = 4 | 1461.3 | 1350.4 | asm | 1301.8 | 1334.1 | 256x224x64 |
| 10240x20480x7168 | bfloat16 | 3200 | one-tile owner, n = 2 | 1496.8 | 1457.7 | asm | 1443.2 | 1443.0 | 256x256x64 |
| 12288x512x7168 | bfloat16 | 96 | one-tile owner, n = 2-3 | 907.7 | 665.7 | opus | 928.2 | 866.6 | 256x192x64 |
| 12288x1024x7168 | bfloat16 | 192 | data-parallel | 1230.2 | 1230.2 | opus | 1311.2 | 1250.6 | 256x192x64 |
| 12288x2048x7168 | bfloat16 | 384 | one-tile owner, n = 2 | 1402.4 | 1288.2 | opus | 1332.0 | 1292.0 | 256x192x64 |
| 14336x512x7168 | bfloat16 | 112 | one-tile owner, n = 2-3 | 1053.0 | 769.0 | opus | 994.8 | 869.3 | 256x224x64 |
| 14336x1024x7168 | bfloat16 | 224 | data-parallel | 1332.6 | 1333.5 | torch | 1298.1 | 1308.9 | 256x224x64 |
| 14336x2048x7168 | bfloat16 | 448 | data-parallel | 1384.9 | 1374.8 | flydsl | 1341.9 | 1336.0 | 256x224x64 |
| 16384x512x7168 | bfloat16 | 128 | one-tile owner, n = 2 | 1149.4 | 870.8 | opus | 1072.1 | 1014.4 | 256x256x64 |
| 16384x1024x7168 | bfloat16 | 256 | data-parallel | 1460.6 | 1465.3 | torch | 1411.1 | 1421.6 | 256x256x64 |
| 16384x2048x7168 | bfloat16 | 512 | data-parallel | 1490.5 | 1488.8 | torch | 1458.2 | 1462.6 | 256x256x64 |
| 16384x3584x7168 | bfloat16 | 896 | one-tile owner, n = 2 | 1486.1 | 1402.1 | flydsl | 1365.2 | 1334.4 | 224x256x64 |
| 16384x8192x7168 | bfloat16 | 2048 | data-parallel | 1544.3 | 1538.7 | torch | 1486.4 | 1517.3 | 256x256x64 |
| 16384x20480x7168 | bfloat16 | 5120 | data-parallel | 1559.3 | 1558.1 | flydsl | 1504.7 | 1474.1 | 256x256x64 |
| 32768x512x7168 | bfloat16 | 256 | data-parallel | 1361.7 | 1361.9 | torch | 1363.1 | 1368.1 | 256x256x64 |
| 32768x1024x7168 | bfloat16 | 512 | data-parallel | 1477.8 | 1471.5 | torch | 1425.5 | 1432.6 | 256x256x64 |
| 32768x2048x7168 | bfloat16 | 1024 | data-parallel | 1513.3 | 1509.1 | torch | 1474.7 | 1486.2 | 256x256x64 |
| 32768x3584x7168 | bfloat16 | 1792 | data-parallel | 1528.1 | 1505.8 | torch | 1496.3 | 1500.1 | 256x256x64 |
| 32768x8192x7168 | bfloat16 | 4096 | data-parallel | 1561.4 | 1544.0 | torch | 1502.7 | 1537.6 | 256x256x64 |
| 32768x20480x7168 | bfloat16 | 10240 | data-parallel | 1570.4 | 1560.1 | asm | 1493.0 | 1446.2 | 256x256x64 |
| 512x1280x8192 | bfloat16 | 10 | one-tile reduce-scatter, n = 12 | 318.1 | 77.4 | torch | 418.4 | 418.0 | 160x64x128 |
| 512x2560x8192 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 | 483.2 | 155.1 | torch | 553.0 | 552.8 | 256x128x64 |
| 512x4096x8192 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 618.7 | 246.5 | flydsl | 666.9 | 643.2 | 128x128x256 |
| 512x4608x8192 | bfloat16 | 36 | one-tile owner, n = 4 | 614.5 | 278.7 | torch | 721.8 | 721.4 | 160x128x128 |
| 512x7168x8192 | bfloat16 | 56 | one-tile owner, n = 3 | 813.0 | 429.8 | flydsl | 799.5 | 742.0 | 256x176x64 |
| 512x8192x8192 | bfloat16 | 64 | one-tile owner, n = 3 | 899.8 | 478.4 | flydsl | 679.5 | 771.1 | 128x128x256 |
| 512x10240x8192 | bfloat16 | 80 | one-tile owner, n = 3-4 | 1002.0 | 601.9 | flydsl | 608.0 | 571.5 | 256x256x64 |
| 512x14336x8192 | bfloat16 | 112 | one-tile owner, n = 2-3 | 1105.5 | 816.9 | torch | 1016.3 | 1023.1 | 224x256x64 |
| 512x16384x8192 | bfloat16 | 128 | one-tile owner, n = 2 | 1173.4 | 889.5 | flydsl | 651.9 | 921.5 | 256x256x64 |
| 512x57344x8192 | bfloat16 | 448 | data-parallel | 1328.6 | 1316.2 | flydsl | 1270.9 | 1238.7 | 224x256x64 |
| 768x4096x8192 | bfloat16 | 48 | one-tile owner, n = 4 | 787.6 | 371.6 | flydsl | 742.0 | 708.9 | 128x192x128 |
| 1024x512x8192 | bfloat16 | 8 | one-tile reduce-scatter, n = 16 | 303.7 | 62.2 | torch | 329.4 | 331.1 | 256x64x64 |
| 1024x1280x8192 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 | 493.3 | 156.5 | torch | 571.0 | 573.5 | 256x128x64 |
| 1024x2560x8192 | bfloat16 | 40 | one-tile owner, n = 4 | 677.7 | 312.2 | torch | 785.9 | 783.0 | 160x128x128 |
| 1024x4096x8192 | bfloat16 | 64 | one-tile owner, n = 3 | 912.7 | 490.0 | flydsl | 902.2 | 799.8 | 128x128x256 |
| 1024x4608x8192 | bfloat16 | 72 | one-tile owner, n = 3-4 | 929.6 | 548.0 | torch | 927.5 | 929.0 | 160x256x64 |
| 1024x7168x8192 | bfloat16 | 112 | one-tile owner, n = 2-3 | 1130.0 | 826.6 | flydsl | 735.6 | 1010.0 | 224x256x64 |
| 1024x8192x8192 | bfloat16 | 128 | one-tile owner, n = 2 | 1205.3 | 920.7 | flydsl | 921.9 | 973.1 | 256x256x64 |
| 1024x10240x8192 | bfloat16 | 160 | data-parallel | 1111.9 | 1119.7 | flydsl | 818.4 | 1180.9 | 160x256x64 |
| 1024x14336x8192 | bfloat16 | 224 | data-parallel | 1363.9 | 1364.3 | flydsl | 1316.8 | 1276.3 | 224x256x64 |
| 1024x16384x8192 | bfloat16 | 256 | data-parallel | 1466.6 | 1468.4 | torch | 1412.2 | 1439.7 | 256x256x64 |
| 1024x57344x8192 | bfloat16 | 896 | one-tile owner, n = 2 | 1481.6 | 1396.9 | asm | 1356.4 | 1309.6 | 224x256x64 |
| 2048x512x8192 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 | 460.6 | 125.5 | torch | 496.9 | 496.7 | 256x128x64 |
| 2048x1280x8192 | bfloat16 | 40 | one-tile owner, n = 4 | 677.7 | 315.3 | torch | 694.8 | 689.8 | 320x192x64 |
| 2048x2560x8192 | bfloat16 | 80 | one-tile owner, n = 3-4 | 1055.0 | 611.9 | torch | 954.8 | 957.3 | 160x256x64 |
| 2048x4096x8192 | bfloat16 | 128 | one-tile owner, n = 2 | 1233.2 | 930.8 | opus | 1125.5 | 972.8 | 256x256x64 |
| 2048x4608x8192 | bfloat16 | 144 | data-parallel | 1033.9 | 1034.6 | torch | 1143.2 | 1151.4 | 144x256x64 |
| 2048x7168x8192 | bfloat16 | 224 | data-parallel | 1401.7 | 1397.4 | asm | 1341.5 | 1292.4 | 224x256x64 |
| 2048x8192x8192 | bfloat16 | 256 | data-parallel | 1486.6 | 1490.0 | torch | 1435.7 | 1445.7 | 256x256x64 |
| 2048x10240x8192 | bfloat16 | 320 | one-tile owner, n = 3 | 1358.5 | 1125.5 | flydsl | 926.5 | 1212.8 | 160x256x64 |
| 2048x14336x8192 | bfloat16 | 448 | data-parallel | 1414.8 | 1417.6 | flydsl | 1366.8 | 1304.3 | 224x256x64 |
| 2048x16384x8192 | bfloat16 | 512 | data-parallel | 1515.3 | 1506.1 | asm | 1430.9 | 1452.7 | 256x256x64 |
| 2048x57344x8192 | bfloat16 | 1792 | data-parallel | 1556.5 | 1551.9 | flydsl | 1500.9 | 1465.0 | 256x256x64 |
| 4096x512x8192 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 616.0 | 248.1 | flydsl | 644.0 | 660.5 | 128x128x256 |
| 4096x1280x8192 | bfloat16 | 80 | one-tile owner, n = 3-4 | 1008.1 | 604.6 | torch | 956.3 | 958.9 | 160x256x64 |
| 4096x2560x8192 | bfloat16 | 160 | data-parallel | 1129.8 | 1127.7 | torch | 1202.3 | 1206.9 | 160x256x64 |
| 4096x4096x8192 | bfloat16 | 256 | data-parallel | 1491.8 | 1493.1 | torch | 1437.3 | 1447.1 | 256x256x64 |
| 4096x4608x8192 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 | 1337.3 | 1019.6 | torch | 1167.1 | 1171.1 | 144x256x64 |
| 4096x7168x8192 | bfloat16 | 448 | data-parallel | 1425.2 | 1427.0 | flydsl | 1369.9 | 1313.6 | 224x256x64 |
| 4096x8192x8192 | bfloat16 | 512 | data-parallel | 1507.6 | 1506.3 | asm | 1447.5 | 1462.3 | 256x256x64 |
| 4096x10240x8192 | bfloat16 | 640 | one-tile owner, n = 2 | 1477.2 | 1389.9 | flydsl | 1350.8 | 1296.0 | 224x256x64 |
| 4096x14336x8192 | bfloat16 | 896 | one-tile owner, n = 2 | 1505.5 | 1433.8 | flydsl | 1381.9 | 1333.9 | 224x256x64 |
| 4096x16384x8192 | bfloat16 | 1024 | data-parallel | 1534.1 | 1534.2 | flydsl | 1457.7 | 1482.1 | 256x256x64 |
| 4096x57344x8192 | bfloat16 | 3584 | data-parallel | 1575.9 | 1560.5 | flydsl | 1484.9 | 1485.3 | 256x256x64 |
| 8192x512x8192 | bfloat16 | 64 | one-tile owner, n = 3 | 897.0 | 477.6 | flydsl | 868.9 | 797.7 | 128x128x256 |
| 8192x1280x8192 | bfloat16 | 160 | data-parallel | 1108.2 | 1106.8 | torch | 1200.1 | 1210.2 | 160x256x64 |
| 8192x2560x8192 | bfloat16 | 320 | one-tile owner, n = 3 | 1358.8 | 1117.6 | torch | 1223.9 | 1232.6 | 160x256x64 |
| 8192x4096x8192 | bfloat16 | 512 | data-parallel | 1505.7 | 1506.9 | torch | 1464.0 | 1461.1 | 256x256x64 |
| 8192x4608x8192 | bfloat16 | 576 | one-tile owner, n = 3 | 1444.6 | 1266.1 | torch | 1300.1 | 1304.4 | 192x256x64 |
| 8192x7168x8192 | bfloat16 | 896 | one-tile owner, n = 2 | 1501.9 | 1429.1 | asm | 1387.6 | 1345.0 | 224x256x64 |
| 8192x8192x8192 | bfloat16 | 1024 | data-parallel | 1547.8 | 1532.4 | asm | 1479.1 | 1489.8 | 256x256x64 |
| 8192x10240x8192 | bfloat16 | 1280 | data-parallel | 1541.1 | 1540.5 | flydsl | 1462.0 | 1508.4 | 256x256x64 |
| 8192x14336x8192 | bfloat16 | 1792 | data-parallel | 1552.5 | 1545.9 | flydsl | 1468.8 | 1508.2 | 256x256x64 |
| 8192x16384x8192 | bfloat16 | 2048 | data-parallel | 1551.6 | 1552.5 | asm | 1488.5 | 1498.2 | 256x256x64 |
| 8192x57344x8192 | bfloat16 | 7168 | data-parallel | 1548.0 | 1558.4 | asm | 1512.6 | 1447.8 | 256x256x64 |
| 16384x512x8192 | bfloat16 | 128 | one-tile owner, n = 2 | 1174.1 | 887.5 | torch | 1000.8 | 1038.6 | 256x256x64 |
| 16384x1280x8192 | bfloat16 | 320 | one-tile owner, n = 3 | 1339.4 | 1111.7 | torch | 1228.0 | 1235.4 | 160x256x64 |
| 16384x2560x8192 | bfloat16 | 640 | one-tile owner, n = 2 | 1469.6 | 1366.8 | flydsl | 1331.5 | 1292.2 | 256x224x64 |
| 16384x4608x8192 | bfloat16 | 1152 | one-tile owner, n = 2 | 1518.8 | 1448.0 | flydsl | 1396.0 | 1199.1 | 256x240x64 |
| 16384x7168x8192 | bfloat16 | 1792 | data-parallel | 1543.7 | 1552.5 | torch | 1499.5 | 1519.1 | 256x256x64 |
| 16384x8192x8192 | bfloat16 | 2048 | data-parallel | 1568.5 | 1555.9 | torch | 1504.1 | 1540.5 | 256x256x64 |
| 16384x10240x8192 | bfloat16 | 2560 | data-parallel | 1573.3 | 1560.2 | torch | 1509.8 | 1544.8 | 256x256x64 |
| 16384x14336x8192 | bfloat16 | 3584 | data-parallel | 1571.8 | 1556.3 | asm | 1520.2 | 1512.7 | 256x256x64 |
| 16384x16384x8192 | bfloat16 | 4096 | data-parallel | 1573.5 | 1563.4 | asm | 1528.9 | 1501.6 | 256x256x64 |
| 16384x57344x8192 | bfloat16 | 14336 | data-parallel | 1560.1 | 1574.8 | asm | 1499.0 | 1425.9 | 256x256x64 |
| 32768x1280x8192 | bfloat16 | 640 | one-tile owner, n = 2 | 1436.5 | 1339.6 | flydsl | 1306.2 | 1314.0 | 256x224x64 |
| 32768x2560x8192 | bfloat16 | 1280 | data-parallel | 1523.6 | 1512.1 | torch | 1492.9 | 1504.3 | 256x256x64 |
| 32768x7168x8192 | bfloat16 | 3584 | data-parallel | 1569.2 | 1551.3 | torch | 1520.6 | 1541.9 | 256x256x64 |
| 32768x8192x8192 | bfloat16 | 4096 | data-parallel | 1576.9 | 1567.3 | flydsl | 1489.1 | 1550.8 | 256x256x64 |
| 32768x10240x8192 | bfloat16 | 5120 | data-parallel | 1580.6 | 1564.2 | torch | 1538.1 | 1547.5 | 256x256x64 |
| 32768x14336x8192 | bfloat16 | 7168 | data-parallel | 1574.6 | 1557.1 | asm | 1518.2 | 1496.0 | 256x256x64 |
| 32768x16384x8192 | bfloat16 | 8192 | data-parallel | 1547.5 | 1573.4 | asm | 1512.1 | 1470.7 | 256x256x64 |
| 32768x57344x8192 | bfloat16 | 28672 | data-parallel | 1539.3 | 1556.0 | asm | 1491.5 | 1420.9 | 256x256x64 |
| 512x6144x12288 | bfloat16 | 48 | one-tile owner, n = 4 | 869.4 | 348.0 | torch | 784.3 | 789.2 | 192x128x128 |
| 1024x6144x12288 | bfloat16 | 96 | one-tile owner, n = 2-3 | 1053.2 | 694.9 | torch | 1077.4 | 1088.0 | 192x256x64 |
| 2048x6144x12288 | bfloat16 | 192 | data-parallel | 1295.9 | 1297.6 | torch | 1289.3 | 1298.2 | 192x256x64 |
| 4096x6144x12288 | bfloat16 | 384 | one-tile owner, n = 2 | 1479.2 | 1336.6 | torch | 1294.9 | 1303.0 | 192x256x64 |
| 8192x6144x12288 | bfloat16 | 768 | data-parallel | 1544.8 | 1540.5 | torch | 1499.6 | 1515.4 | 256x256x64 |
| 16384x6144x12288 | bfloat16 | 1536 | data-parallel | 1556.4 | 1549.9 | torch | 1530.7 | 1532.4 | 256x256x64 |
| 32768x6144x12288 | bfloat16 | 3072 | data-parallel | 1563.9 | 1559.4 | asm | 1528.9 | 1546.6 | 256x256x64 |
| 512x16384x13312 | bfloat16 | 128 | one-tile owner, n = 2 | 1270.4 | 920.2 | flydsl | 706.1 | 1086.5 | 256x256x64 |
| 1024x16384x13312 | bfloat16 | 256 | data-parallel | 1512.3 | 1510.6 | asm | 1468.8 | 1477.5 | 256x256x64 |
| 2048x16384x13312 | bfloat16 | 512 | data-parallel | 1557.8 | 1552.4 | flydsl | 1002.5 | 1487.0 | 256x256x64 |
| 4096x16384x13312 | bfloat16 | 1024 | data-parallel | 1561.7 | 1582.4 | flydsl | 1525.2 | 1496.2 | 256x256x64 |
| 8192x16384x13312 | bfloat16 | 2048 | data-parallel | 1570.2 | 1566.0 | flydsl | 1505.2 | 1486.2 | 256x256x64 |
| 16384x16384x13312 | bfloat16 | 4096 | data-parallel | 1566.3 | 1564.6 | asm | 1502.2 | 1472.2 | 256x256x64 |
| 32768x16384x13312 | bfloat16 | 8192 | data-parallel | 1550.1 | 1557.7 | flydsl | 1475.0 | 1440.3 | 256x256x64 |
| 512x2304x16384 | bfloat16 | 18 | one-tile reduce-scatter, n = 7 | 569.1 | 139.0 | torch | 705.3 | 701.0 | 256x128x64 |
| 512x4608x16384 | bfloat16 | 36 | one-tile owner, n = 6 | 846.6 | 276.3 | flydsl | 507.2 | 797.4 | 160x128x128 |
| 512x9216x16384 | bfloat16 | 72 | one-tile owner, n = 3-4 | 1075.7 | 542.6 | flydsl | 630.3 | 757.1 | 256x256x64 |
| 512x13312x16384 | bfloat16 | 104 | one-tile owner, n = 2-3 | 1150.2 | 748.7 | flydsl | 540.5 | 1083.7 | 208x256x64 |
| 512x26624x16384 | bfloat16 | 208 | data-parallel | 1284.5 | 1257.4 | flydsl | 767.8 | 1189.7 | 208x256x64 |
| 512x53248x16384 | bfloat16 | 416 | data-parallel | 1290.2 | 1295.9 | flydsl | 649.9 | 1186.8 | 208x256x64 |
| 1024x2304x16384 | bfloat16 | 36 | one-tile owner, n = 6 | 859.0 | 276.3 | torch | 869.9 | 869.1 | 256x256x64 |
| 1024x4608x16384 | bfloat16 | 72 | one-tile owner, n = 3-4 | 1065.2 | 544.1 | flydsl | 645.9 | 1007.7 | 144x256x64 |
| 1024x9216x16384 | bfloat16 | 144 | data-parallel | 1020.8 | 1024.7 | flydsl | 664.9 | 1161.8 | 144x256x64 |
| 1024x13312x16384 | bfloat16 | 208 | data-parallel | 1344.9 | 1343.0 | asm | 1310.1 | 1231.3 | 208x256x64 |
| 1024x26624x16384 | bfloat16 | 416 | data-parallel | 1360.1 | 1359.3 | flydsl | 928.3 | 1239.7 | 208x256x64 |
| 1024x53248x16384 | bfloat16 | 832 | one-tile owner, n = 4 | 1504.9 | 1342.5 | triton | 983.8 | 1242.1 | 208x256x64 |
| 2048x2304x16384 | bfloat16 | 72 | one-tile owner, n = 3-4 | 1157.2 | 545.4 | torch | 849.2 | 869.3 | 256x256x64 |
| 2048x4608x16384 | bfloat16 | 144 | data-parallel | 1046.1 | 1026.4 | flydsl | 688.5 | 1186.0 | 144x256x64 |
| 2048x9216x16384 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 | 1411.6 | 1034.0 | flydsl | 905.6 | 1162.1 | 144x256x64 |
| 2048x13312x16384 | bfloat16 | 416 | data-parallel | 1394.2 | 1390.9 | flydsl | 1358.7 | 1229.9 | 208x256x64 |
| 2048x26624x16384 | bfloat16 | 832 | one-tile owner, n = 4 | 1543.9 | 1363.8 | flydsl | 901.2 | 1230.2 | 208x256x64 |
| 2048x53248x16384 | bfloat16 | 1664 | one-tile owner, n = 2 | 1565.9 | 1496.1 | flydsl | 779.4 | 1270.7 | 240x256x64 |
| 4096x2304x16384 | bfloat16 | 144 | data-parallel | 1027.4 | 1019.5 | torch | 1174.3 | 1184.5 | 144x256x64 |
| 4096x4608x16384 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 | 1397.8 | 1034.0 | torch | 1170.9 | 1179.6 | 144x256x64 |
| 4096x9216x16384 | bfloat16 | 576 | one-tile owner, n = 4 | 1516.7 | 1292.8 | torch | 1294.2 | 1307.7 | 192x256x64 |
| 4096x13312x16384 | bfloat16 | 832 | one-tile owner, n = 4 | 1544.9 | 1363.9 | flydsl | 1314.0 | 1248.9 | 208x256x64 |
| 4096x26624x16384 | bfloat16 | 1664 | one-tile owner, n = 2 | 1574.3 | 1503.3 | flydsl | 1452.0 | 1314.8 | 240x256x64 |
| 4096x53248x16384 | bfloat16 | 3328 | data-parallel | 1583.8 | 1587.5 | flydsl | 1526.8 | 1473.3 | 256x256x64 |
| 8192x2304x16384 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 | 1377.2 | 996.2 | torch | 1177.7 | 1190.0 | 144x256x64 |
| 8192x4608x16384 | bfloat16 | 576 | one-tile owner, n = 4 | 1483.9 | 1269.1 | torch | 1340.2 | 1330.5 | 192x256x64 |
| 8192x9216x16384 | bfloat16 | 1152 | one-tile owner, n = 2 | 1537.9 | 1448.9 | flydsl | 1396.1 | 1132.8 | 256x240x64 |
| 8192x13312x16384 | bfloat16 | 1664 | one-tile owner, n = 2 | 1542.5 | 1476.5 | flydsl | 1436.2 | 1279.1 | 240x256x64 |
| 8192x26624x16384 | bfloat16 | 3328 | data-parallel | 1548.2 | 1543.4 | flydsl | 1494.1 | 1460.9 | 256x256x64 |
| 8192x53248x16384 | bfloat16 | 6656 | data-parallel | 1520.8 | 1544.1 | flydsl | 1480.0 | 1432.2 | 256x256x64 |
| 16384x2304x16384 | bfloat16 | 576 | one-tile owner, n = 4 | 1473.7 | 1268.2 | torch | 1325.7 | 1318.8 | 192x256x64 |
| 16384x4608x16384 | bfloat16 | 1152 | one-tile owner, n = 2 | 1523.0 | 1439.1 | flydsl | 1406.2 | 1169.4 | 256x240x64 |
| 16384x9216x16384 | bfloat16 | 2304 | data-parallel | 1547.1 | 1550.2 | flydsl | 1489.9 | 1481.3 | 256x256x64 |
| 16384x13312x16384 | bfloat16 | 3328 | data-parallel | 1538.4 | 1545.7 | asm | 1483.8 | 1460.2 | 256x256x64 |
| 16384x26624x16384 | bfloat16 | 6656 | data-parallel | 1520.8 | 1540.4 | flydsl | 1488.2 | 1436.4 | 256x256x64 |
| 16384x53248x16384 | bfloat16 | 13312 | data-parallel | 1499.6 | 1527.7 | asm | 1454.9 | 1421.5 | 256x256x64 |
| 32768x2304x16384 | bfloat16 | 1152 | one-tile owner, n = 2 | 1491.9 | 1431.9 | flydsl | 1394.8 | 1175.6 | 256x240x64 |
| 32768x4608x16384 | bfloat16 | 2304 | data-parallel | 1533.8 | 1540.5 | torch | 1514.0 | 1522.4 | 256x256x64 |
| 32768x9216x16384 | bfloat16 | 4608 | data-parallel | 1516.6 | 1537.8 | asm | 1445.6 | 1450.5 | 256x256x64 |
| 32768x13312x16384 | bfloat16 | 6656 | data-parallel | 1515.6 | 1535.2 | asm | 1466.4 | 1445.2 | 256x256x64 |
| 32768x26624x16384 | bfloat16 | 13312 | data-parallel | 1495.7 | 1525.3 | flydsl | 1460.2 | 1434.1 | 256x256x64 |
| 32768x53248x16384 | bfloat16 | 26624 | data-parallel | 1497.4 | 1536.7 | flydsl | 1431.9 | 1441.6 | 256x256x64 |
| 512x6144x18432 | bfloat16 | 48 | one-tile owner, n = 5-6 | 1011.7 | 352.9 | flydsl | 786.6 | 838.4 | 192x128x128 |
| 1024x6144x18432 | bfloat16 | 96 | one-tile owner, n = 2-3 | 1138.5 | 705.6 | torch | 1163.3 | 1168.1 | 192x256x64 |
| 2048x6144x18432 | bfloat16 | 192 | data-parallel | 1319.9 | 1321.0 | torch | 1314.3 | 1319.5 | 192x256x64 |
| 4096x6144x18432 | bfloat16 | 384 | one-tile owner, n = 2 | 1535.5 | 1345.4 | torch | 1317.5 | 1323.5 | 192x256x64 |
| 8192x6144x18432 | bfloat16 | 768 | data-parallel | 1567.1 | 1566.5 | torch | 1516.0 | 1519.4 | 256x256x64 |
| 16384x6144x18432 | bfloat16 | 1536 | data-parallel | 1575.1 | 1571.3 | asm | 1525.2 | 1515.8 | 256x256x64 |
| 32768x6144x18432 | bfloat16 | 3072 | data-parallel | 1574.8 | 1570.7 | asm | 1514.0 | 1499.3 | 256x256x64 |
| 512x5120x25600 | bfloat16 | 40 | one-tile owner, n = 6-7 | 1013.6 | 294.5 | flydsl | 427.9 | 878.3 | 160x128x128 |
| 1024x5120x25600 | bfloat16 | 80 | one-tile owner, n = 3-4 | 1292.4 | 590.9 | torch | 1141.2 | 1137.6 | 160x256x64 |
| 2048x5120x25600 | bfloat16 | 160 | data-parallel | 1159.9 | 1155.3 | torch | 1274.4 | 1264.8 | 160x256x64 |
| 4096x5120x25600 | bfloat16 | 320 | one-tile owner, n = 4 | 1513.0 | 1136.6 | torch | 1262.5 | 1261.7 | 160x256x64 |
| 8192x5120x25600 | bfloat16 | 640 | one-tile owner, n = 2 | 1554.8 | 1394.4 | asm | 1387.8 | 1322.1 | 256x224x64 |
| 16384x5120x25600 | bfloat16 | 1280 | data-parallel | 1582.2 | 1574.7 | torch | 1530.4 | 1527.5 | 256x256x64 |
| 32768x5120x25600 | bfloat16 | 2560 | data-parallel | 1570.5 | 1570.8 | torch | 1504.3 | 1510.4 | 256x256x64 |
| 512x16384x26624 | bfloat16 | 128 | one-tile owner, n = 2 | 1366.7 | 932.1 | flydsl | 653.1 | 1169.6 | 256x256x64 |
| 1024x16384x26624 | bfloat16 | 256 | data-parallel | 1542.1 | 1538.2 | flydsl | 755.4 | 1491.0 | 256x256x64 |
| 2048x16384x26624 | bfloat16 | 512 | data-parallel | 1564.1 | 1559.8 | flydsl | 821.9 | 1494.1 | 256x256x64 |
| 4096x16384x26624 | bfloat16 | 1024 | data-parallel | 1600.2 | 1598.0 | flydsl | 1536.3 | 1498.6 | 256x256x64 |
| 8192x16384x26624 | bfloat16 | 2048 | data-parallel | 1587.8 | 1566.5 | asm | 251.2 | 1485.8 | 256x256x64 |
| 16384x16384x26624 | bfloat16 | 4096 | data-parallel | 1577.7 | 1584.7 | asm | 1527.4 | 1471.4 | 256x256x64 |
| 32768x16384x26624 | bfloat16 | 8192 | data-parallel | 1565.7 | 1574.1 | asm | 1484.8 | 1442.6 | 256x256x64 |
| 512x8192x28672 | bfloat16 | 64 | one-tile owner, n = 4 | 1231.4 | 470.5 | flydsl | 777.2 | 923.2 | 240x256x64 |
| 1024x8192x28672 | bfloat16 | 128 | one-tile owner, n = 2 | 1433.7 | 937.3 | triton | 690.9 | 1321.7 | 256x256x64 |
| 2048x8192x28672 | bfloat16 | 256 | data-parallel | 1580.4 | 1574.5 | flydsl | 1037.2 | 1543.2 | 256x256x64 |
| 4096x8192x28672 | bfloat16 | 512 | data-parallel | 1598.1 | 1588.6 | flydsl | 1537.7 | 1536.1 | 256x256x64 |
| 8192x8192x28672 | bfloat16 | 1024 | data-parallel | 1587.2 | 1585.4 | asm | 1538.8 | 1525.4 | 256x256x64 |
| 16384x8192x28672 | bfloat16 | 2048 | data-parallel | 1586.0 | 1581.4 | asm | 1531.7 | 1500.5 | 256x256x64 |
| 32768x8192x28672 | bfloat16 | 4096 | data-parallel | 1579.6 | 1581.4 | asm | 1510.3 | 1478.5 | 256x256x64 |
| 512x7168x35840 | bfloat16 | 56 | one-tile owner, n = 4-5 | 1182.9 | 434.6 | flydsl | 813.0 | 963.2 | 256x176x64 |
| 1024x7168x35840 | bfloat16 | 112 | one-tile owner, n = 2-3 | 1287.2 | 858.4 | flydsl | 565.9 | 1274.1 | 224x256x64 |
| 2048x7168x35840 | bfloat16 | 224 | spread | 1496.5 | 1466.5 | asm | 1437.1 | 1385.8 | 224x256x64 |
| 4096x7168x35840 | bfloat16 | 448 | spread | 1556.6 | 1482.0 | flydsl | 1446.5 | 1381.3 | 224x256x64 |
| 8192x7168x35840 | bfloat16 | 896 | one-tile owner, n = 2 | 1580.0 | 1441.2 | asm | 1396.9 | 1364.7 | 224x256x64 |
| 16384x7168x35840 | bfloat16 | 1792 | data-parallel | 1578.3 | 1575.7 | torch | 1488.5 | 1497.6 | 256x256x64 |
| 32768x7168x35840 | bfloat16 | 3584 | data-parallel | 1556.0 | 1563.7 | flydsl | 1480.6 | 1463.6 | 256x256x64 |
