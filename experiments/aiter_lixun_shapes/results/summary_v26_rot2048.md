# streamk_wip (v26, capped) vs AITER, v9 and hipBLASLt on the 630 shapes (2048 MB rotating buffer)

Shapes: `supported_status_scoped.csv` (630 shapes, no bias; 598 bf16 and 32 fp32 outputs). Inputs from `torch.manual_seed(0)`, every output checked against an fp32 `F.linear` reference, `do_bench_cudagraph` with 3 interleaved rounds and the median taken.

- Stack: torch 2.10.0+rocm7.2.4.git3d3aa833, hipBLASLt 100202 (dabb6df2b9), Triton 3.8.0 @ /home/jbowley/repos/triton_gfx950-tutorial-v2.2/python/triton.
- GPU: HIP device 3, PCI bus 21, gfx950:sramecc+:xnack-.
- AITER: e2c7aa0aa [Triton/Gluon] gfx950 a16w16: skip the Gluon kernel on Triton without cd_regclass, on top of a1455f1c4 2026-10-02 [CK] [FlyDSL] add mxfp8 a8w8 blockscale for gfx950 (#4951).
- streamk_wip: tutorial `v26_streamk_ragged_k`, snapshot sha256 96f09f5c62943d6b (newest WORKLOG entry: "5. The host rule from sweeps G and H (2026-10-12)"), `STREAMK_POLICY=capped` (the kernel's default), v9 tile order. The configuration capped ran on each shape is in the appendix and in `v20_label`.
- hipBLASLt: the call AITER's `torch` libtype makes, `F.linear(x, w)` for bf16 and `torch.mm(x, w.t(), out_dtype=torch.float32)` for fp32, timed the same way. Its tile is the MT of the first `Cijk_` kernel it launches.
- Cache state: each callable rotates over max(2, ceil(2048 MB / set)) copies of x, w and C inside the captured graph, so no launch reads the inputs of the previous one. Outputs that a callable allocates itself (AITER's pick, hipBLASLt) are kept alive in a ring so they rotate too.
- "Faster" and "slower" mean a gap of more than 1%; "within 1%" holds the rest.

Copies per shape: median 15, minimum 2; 70 shapes at the minimum of 2.
`do_bench_cudagraph` sizes its graph from a host-side estimate, so on 5 small shapes a timed graph touched fewer copies than it had. What matters is the data a graph cycles through: copies touched x set size is at least 1280 MB on every shape (median 2107 MB), against a 256 MB last-level cache; 0 shapes are under 512 MB.

**Against AITER's pick, by the configuration capped chose**

| | Shapes | v9: faster | v9: within 1% | v9: slower | v9: median | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All shapes | 630 | 299 | 26 | 305 | +0.0% | 405 | 27 | 198 | +3.2% |
| Data-parallel | 287 | 222 | 19 | 46 | +3.3% | 237 | 11 | 39 | +3.6% |
| One-tile reduce-scatter, K >= 2048 | 147 | 5 | 0 | 142 | -54.7% | 39 | 5 | 103 | -8.6% |
| One-tile owner, K >= 2048 | 155 | 60 | 5 | 90 | -7.9% | 109 | 10 | 36 | +7.8% |
| One-tile reduce-scatter, K < 2048 | 16 | 0 | 0 | 16 | -54.2% | 0 | 0 | 16 | -50.5% |
| One-tile owner, K < 2048 | 3 | 0 | 0 | 3 | -21.8% | 0 | 0 | 3 | -33.6% |
| Spread | 22 | 12 | 2 | 8 | +2.2% | 20 | 1 | 1 | +7.2% |

Each pair of columns compares that kernel with AITER's pick; positive medians mean it is faster.

**Against AITER's pick, by the tile AITER uses**

| | Shapes | v9: faster | v9: within 1% | v9: slower | v9: median | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All shapes | 630 | 299 | 26 | 305 | +0.0% | 405 | 27 | 198 | +3.2% |
| AITER uses a 256x256 tile | 248 | 218 | 18 | 12 | +3.3% | 240 | 8 | 0 | +3.9% |
| AITER uses another tile | 382 | 81 | 8 | 293 | -22.0% | 165 | 19 | 198 | -1.4% |

Each pair of columns compares that kernel with AITER's pick; positive medians mean it is faster.

**Against AITER's pick, by v9 grid size (number of 256x256 output tiles)**

| | Shapes | v9: faster | v9: within 1% | v9: slower | v9: median | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All shapes | 630 | 299 | 26 | 305 | +0.0% | 405 | 27 | 198 | +3.2% |
| Under 256 (less than one wave) | 321 | 46 | 8 | 267 | -34.5% | 110 | 18 | 193 | -4.4% |
| 256 to 1023 | 152 | 117 | 5 | 30 | +3.4% | 142 | 5 | 5 | +6.7% |
| 1024 or more | 157 | 136 | 13 | 8 | +3.4% | 153 | 4 | 0 | +3.8% |

Each pair of columns compares that kernel with AITER's pick; positive medians mean it is faster.

**Against hipBLASLt (torch), by v9 grid size**

| | Shapes | AITER's pick: faster | AITER's pick: within 1% | AITER's pick: slower | AITER's pick: median | v9: faster | v9: within 1% | v9: slower | v9: median | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All shapes | 630 | 203 | 231 | 196 | -0.4% | 243 | 58 | 329 | -2.2% | 386 | 39 | 205 | +2.4% |
| Under 256 (less than one wave) | 321 | 116 | 111 | 94 | -0.3% | 38 | 10 | 273 | -34.8% | 113 | 13 | 195 | -5.3% |
| 256 to 1023 | 152 | 34 | 70 | 48 | -0.6% | 101 | 12 | 39 | +3.0% | 134 | 9 | 9 | +6.2% |
| 1024 or more | 157 | 53 | 50 | 54 | -0.6% | 104 | 36 | 17 | +3.2% | 139 | 17 | 1 | +3.9% |

Each pair of columns compares that kernel with hipBLASLt; positive medians mean it is faster.

**hipBLASLt's tile choice**

| hipBLASLt tile (MxNxK) | Shapes | hipBLASLt median TFLOPS | streamk_wip vs hipBLASLt: faster | within 1% | slower | median |
|---|---:|---:|---:|---:|---:|---:|
| 256x256x64 | 225 | 1414 | 201 | 23 | 1 | +3.4% |
| 128x128x256 | 40 | 659 | 14 | 0 | 26 | -4.3% |
| 192x256x64 | 34 | 1193 | 15 | 5 | 14 | -0.8% |
| 224x256x64 | 32 | 1272 | 32 | 0 | 0 | +8.5% |
| 256x128x64 | 27 | 725 | 2 | 1 | 24 | -10.3% |
| 160x256x64 | 27 | 1172 | 16 | 2 | 9 | +5.0% |
| 256x224x64 | 20 | 1286 | 18 | 0 | 2 | +8.7% |
| 256x240x64 | 18 | 1137 | 18 | 0 | 0 | +22.9% |
| 144x256x64 | 15 | 1153 | 14 | 0 | 1 | +6.1% |
| 128x128x128 | 14 | 556 | 0 | 0 | 14 | -25.5% |
| 160x128x128 | 13 | 775 | 4 | 2 | 7 | -5.1% |
| 96x128x128 | 12 | 558 | 0 | 0 | 12 | -35.6% |
| 192x128x128 | 12 | 762 | 6 | 1 | 5 | +0.9% |
| 256x208x64 | 12 | 1071 | 12 | 0 | 0 | +16.2% |
| 256x192x64 | 11 | 1206 | 5 | 2 | 4 | +0.8% |
| 208x256x64 | 11 | 1228 | 11 | 0 | 0 | +12.4% |
| 256x160x64 | 10 | 1127 | 3 | 0 | 7 | -5.7% |
| 64x64x256 | 9 | 402 | 0 | 0 | 9 | -33.3% |
| 128x192x128 | 7 | 883 | 2 | 0 | 5 | -14.4% |
| 128x160x128 | 7 | 855 | 1 | 0 | 6 | -12.6% |
| 240x256x64 | 7 | 1196 | 7 | 0 | 0 | +20.4% |
| 128x80x128 | 6 | 603 | 0 | 0 | 6 | -11.6% |
| 128x96x128 | 6 | 660 | 0 | 0 | 6 | -11.7% |
| 224x128x64 | 5 | 780 | 0 | 0 | 5 | -17.2% |
| 256x176x64 | 4 | 694 | 2 | 1 | 1 | +4.4% |
| 128x64x128 | 4 | 452 | 0 | 0 | 4 | -52.2% |
| 160x64x128 | 4 | 421 | 0 | 0 | 4 | -25.8% |
| 64x80x256 | 4 | 466 | 0 | 0 | 4 | -31.1% |
| 64x96x256 | 4 | 500 | 0 | 0 | 4 | -22.7% |
| 64x32x256 | 3 | 264 | 0 | 0 | 3 | -29.5% |
| 64x48x256 | 3 | 367 | 0 | 0 | 3 | -39.0% |
| 160x192x128 | 3 | 879 | 1 | 2 | 0 | +0.6% |
| 192x128x64 | 2 | 812 | 0 | 0 | 2 | -31.0% |
| 128x192x64 | 2 | 847 | 0 | 0 | 2 | -9.8% |
| 256x112x64 | 2 | 440 | 0 | 0 | 2 | -15.1% |
| 192x64x128 | 2 | 391 | 0 | 0 | 2 | -20.4% |
| 128x128x64 | 2 | 444 | 0 | 0 | 2 | -30.3% |
| 256x96x64 | 2 | 403 | 0 | 0 | 2 | -18.6% |
| 256x64x64 | 2 | 309 | 0 | 0 | 2 | -17.8% |
| 96x64x128 | 1 | 391 | 0 | 0 | 1 | -52.6% |
| 224x160x64 | 1 | 1062 | 1 | 0 | 0 | +13.5% |
| 192x192x64 | 1 | 1025 | 0 | 0 | 1 | -9.3% |
| 64x96x128 | 1 | 512 | 0 | 0 | 1 | -17.7% |
| 256x80x64 | 1 | 362 | 0 | 0 | 1 | -18.6% |
| 256x80x128 | 1 | 536 | 0 | 0 | 1 | -16.4% |
| 320x192x64 | 1 | 695 | 1 | 0 | 0 | +6.1% |

Solution-name tokens (stream-K `SK`, global split-U `GSU`, kernels launched): SK3 GSU0 x1: 604, SK3 GSU0 x2: 24, none GSU- x1: 2. `SK3` marks a kernel built with stream-K support, not that this launch split K.

**streamk_wip against v9, by configuration**

| | Shapes | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|
| All shapes | 630 | 392 | 209 | 29 | +4.5% |
| Data-parallel | 287 | 62 | 205 | 20 | +0.2% |
| One-tile reduce-scatter, K >= 2048 | 147 | 146 | 1 | 0 | +112.1% |
| One-tile owner, K >= 2048 | 155 | 151 | 0 | 4 | +17.2% |
| One-tile reduce-scatter, K < 2048 | 16 | 13 | 2 | 1 | +9.0% |
| One-tile owner, K < 2048 | 3 | 0 | 0 | 3 | -12.9% |
| Spread | 22 | 20 | 1 | 1 | +7.0% |

Each pair of columns compares that kernel with v9; positive medians mean it is faster.

**Diagnostic: capped's split against forcing data-parallel**

| Configuration | Shapes | split faster | within 1% | split slower | median |
|---|---:|---:|---:|---:|---:|
| One-tile owner, K < 2048 | 3 | 0 | 0 | 3 | -9.1% |
| One-tile owner, K >= 2048 | 155 | 148 | 3 | 4 | +17.2% |
| One-tile reduce-scatter, K < 2048 | 16 | 16 | 0 | 0 | +14.3% |
| One-tile reduce-scatter, K >= 2048 | 147 | 146 | 1 | 0 | +114.4% |
| Spread | 22 | 20 | 2 | 0 | +6.6% |

**Selected shapes (TFLOPS)**

| Shape (MxNxK) | Output tiles | AITER pick | AITER | hipBLASLt | hipBLASLt tile | v9 | streamk_wip | streamk_wip config | vs v9 | vs AITER | vs hipBLASLt |
|---|---:|---|---:|---:|---|---:|---:|---|---:|---:|---:|
| 512x2304x16384 | 18 | torch | 705 | 706 | 256x128x64 | 139 | 682 | one-tile reduce-scatter, n = 10 | +390% | -3% | -3% |
| 512x5120x25600 | 40 | flydsl | 428 | 874 | 160x128x128 | 295 | 893 | one-tile reduce-scatter, n = 4 | +203% | +109% | +2% |
| 512x7168x35840 | 56 | flydsl | 815 | 961 | 256x176x64 | 433 | 1056 | one-tile reduce-scatter, n = 3 | +144% | +30% | +10% |
| 8192x2304x16384 | 288 | torch | 1181 | 1186 | 144x256x64 | 1000 | 1362 | one-tile reduce-scatter, n = 4 | +36% | +15% | +15% |
| 4096x5120x25600 | 320 | torch | 1263 | 1257 | 160x256x64 | 1138 | 1521 | one-tile owner, n = 4 | +34% | +20% | +21% |
| 4096x4608x8192 | 288 | torch | 1166 | 1172 | 144x256x64 | 1020 | 1339 | one-tile reduce-scatter, n = 4 | +31% | +15% | +14% |
| 1024x3072x512 | 48 | flydsl | 366 | 321 | 96x128x128 | 202 | 202 | data-parallel | +0% | -45% | -37% |
| 2048x2048x512 | 64 | flydsl | 432 | 391 | 128x128x128 | 260 | 260 | data-parallel | +0% | -40% | -33% |
| 4096x5120x640 | 320 | flydsl | 611 | 647 | 256x176x64 | 633 | 649 | data-parallel | +3% | +6% | +0% |
| 8192x2048x512 | 256 | torch | 751 | 756 | 256x256x64 | 772 | 768 | data-parallel | -1% | +2% | +2% |

**Against `results/details_v24_rot2048.csv`** (this run / that run - 1, median; for v20 the baseline may use a different policy or kernel snapshot)

| | Shapes | AITER's pick | v9 | v20 | hipBLASLt |
|---|---:|---:|---:|---:|---:|
| All shapes | 630 | +0.1% | +0.1% | +0.1% | +0.0% |
| Under 256 (less than one wave) | 321 | +0.1% | +0.0% | +0.2% | +0.0% |
| 256 to 1023 | 152 | +0.1% | +0.1% | +0.0% | +0.0% |
| 1024 or more | 157 | +0.1% | +0.1% | +0.0% | +0.1% |
| K < 2048 | 91 | +0.1% | +0.0% | +0.0% | +0.0% |
| K >= 2048 | 539 | +0.1% | +0.1% | +0.1% | +0.0% |

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
| 1024x3072x512 | bfloat16 | 48 | data-parallel | 201.8 | 201.8 | flydsl | 365.5 | 321.4 | 96x128x128 |
| 1024x4096x512 | bfloat16 | 64 | data-parallel | 258.7 | 258.5 | flydsl | 435.3 | 390.4 | 128x128x128 |
| 1024x7168x512 | bfloat16 | 112 | data-parallel | 416.0 | 415.9 | flydsl | 508.2 | 501.5 | 256x128x64 |
| 2048x2048x512 | bfloat16 | 64 | data-parallel | 260.5 | 260.1 | flydsl | 432.0 | 391.0 | 128x128x128 |
| 2048x4096x512 | bfloat16 | 128 | data-parallel | 463.4 | 463.6 | torch | 557.4 | 558.4 | 256x128x64 |
| 4096x2048x512 | bfloat16 | 128 | data-parallel | 455.7 | 462.1 | torch | 637.5 | 644.9 | 256x128x64 |
| 4096x4096x512 | bfloat16 | 256 | data-parallel | 761.2 | 768.2 | torch | 757.9 | 764.7 | 256x256x64 |
| 4096x7168x512 | bfloat16 | 448 | data-parallel | 780.7 | 759.0 | flydsl | 744.2 | 717.4 | 224x256x64 |
| 8192x2048x512 | bfloat16 | 256 | data-parallel | 767.5 | 772.5 | torch | 750.6 | 755.5 | 256x256x64 |
| 8192x4096x512 | bfloat16 | 512 | data-parallel | 834.9 | 811.5 | flydsl | 796.2 | 835.9 | 256x256x64 |
| 16384x2048x512 | bfloat16 | 512 | data-parallel | 842.8 | 815.8 | torch | 827.7 | 829.5 | 256x256x64 |
| 16384x4096x512 | bfloat16 | 1024 | data-parallel | 893.2 | 836.2 | torch | 890.2 | 892.5 | 256x256x64 |
| 16384x7168x512 | bfloat16 | 1792 | data-parallel | 963.7 | 897.3 | torch | 973.0 | 980.6 | 256x256x64 |
| 32768x2048x512 | bfloat16 | 1024 | data-parallel | 900.8 | 842.0 | torch | 871.3 | 874.5 | 256x256x64 |
| 1024x5120x640 | bfloat16 | 80 | data-parallel | 323.7 | 324.4 | torch | 441.9 | 443.6 | 160x128x128 |
| 2048x5120x640 | bfloat16 | 160 | data-parallel | 601.9 | 601.6 | flydsl | 591.5 | 505.6 | 192x256x64 |
| 4096x5120x640 | bfloat16 | 320 | data-parallel | 648.6 | 632.6 | flydsl | 611.0 | 646.6 | 256x176x64 |
| 8192x5120x640 | bfloat16 | 640 | data-parallel | 826.1 | 793.3 | torch | 860.9 | 865.8 | 256x224x64 |
| 16384x5120x640 | bfloat16 | 1280 | data-parallel | 1001.5 | 951.1 | torch | 995.1 | 998.8 | 256x256x64 |
| 32768x5120x640 | bfloat16 | 2560 | data-parallel | 1046.6 | 988.5 | torch | 1030.9 | 1035.7 | 256x256x64 |
| 1024x7168x768 | bfloat16 | 112 | data-parallel | 484.4 | 484.5 | flydsl | 595.7 | 581.8 | 224x128x64 |
| 10240x7168x768 | bfloat16 | 1120 | data-parallel | 1038.6 | 1005.4 | opus | 376.4 | 847.9 | 256x240x64 |
| 512x4096x1024 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 | 180.3 | 180.3 | torch | 389.5 | 392.0 | 128x64x128 |
| 512x6144x1024 | bfloat16 | 48 | data-parallel | 267.8 | 267.8 | flydsl | 494.9 | 473.5 | 96x128x128 |
| 512x7168x1024 | bfloat16 | 56 | data-parallel | 308.2 | 308.9 | flydsl | 539.3 | 491.6 | 128x128x128 |
| 512x8192x1024 | bfloat16 | 64 | data-parallel | 345.3 | 345.3 | flydsl | 581.1 | 547.2 | 128x128x128 |
| 1024x4096x1024 | bfloat16 | 64 | data-parallel | 344.1 | 345.4 | torch | 558.9 | 561.1 | 128x128x128 |
| 1024x7168x1024 | bfloat16 | 112 | data-parallel | 569.9 | 569.4 | flydsl | 667.4 | 688.4 | 224x128x64 |
| 1024x8192x1024 | bfloat16 | 128 | data-parallel | 625.7 | 623.7 | torch | 753.8 | 757.6 | 256x128x64 |
| 2048x4096x1024 | bfloat16 | 128 | data-parallel | 630.0 | 628.0 | torch | 763.2 | 769.4 | 256x128x64 |
| 2048x7168x1024 | bfloat16 | 224 | data-parallel | 987.1 | 992.6 | flydsl | 949.2 | 950.9 | 256x256x64 |
| 2048x8192x1024 | bfloat16 | 256 | data-parallel | 1045.8 | 1058.7 | flydsl | 1000.6 | 1011.8 | 256x256x64 |
| 3072x4096x1024 | bfloat16 | 192 | data-parallel | 879.3 | 885.2 | torch | 907.8 | 912.3 | 256x192x64 |
| 4096x4096x1024 | bfloat16 | 256 | data-parallel | 1059.0 | 1063.1 | torch | 1014.9 | 1017.7 | 256x256x64 |
| 4096x7168x1024 | bfloat16 | 448 | data-parallel | 1033.8 | 1025.1 | flydsl | 981.3 | 970.2 | 224x256x64 |
| 4096x8192x1024 | bfloat16 | 512 | data-parallel | 1106.9 | 1094.4 | asm | 1103.5 | 1098.2 | 256x256x64 |
| 5120x4096x1024 | bfloat16 | 320 | data-parallel | 855.3 | 853.2 | torch | 970.9 | 976.4 | 256x160x64 |
| 7680x4096x1024 | bfloat16 | 480 | data-parallel | 1089.4 | 1080.1 | torch | 929.9 | 937.8 | 256x240x64 |
| 8192x7168x1024 | bfloat16 | 896 | data-parallel | 1088.4 | 1043.6 | asm | 1037.6 | 1014.9 | 224x256x64 |
| 8192x8192x1024 | bfloat16 | 1024 | data-parallel | 1168.4 | 1124.5 | asm | 1134.9 | 1153.6 | 256x256x64 |
| 16384x4096x1024 | bfloat16 | 1024 | data-parallel | 1176.7 | 1135.0 | torch | 1145.7 | 1158.2 | 256x256x64 |
| 16384x7168x1024 | bfloat16 | 1792 | data-parallel | 1216.8 | 1173.1 | asm | 1131.1 | 1199.4 | 256x256x64 |
| 16384x8192x1024 | bfloat16 | 2048 | data-parallel | 1220.4 | 1167.9 | torch | 1183.3 | 1192.6 | 256x256x64 |
| 32768x7168x1024 | bfloat16 | 3584 | data-parallel | 1239.0 | 1199.9 | torch | 1213.7 | 1221.8 | 256x256x64 |
| 32768x8192x1024 | bfloat16 | 4096 | data-parallel | 1242.3 | 1194.2 | torch | 1204.6 | 1222.1 | 256x256x64 |
| 512x5120x1280 | bfloat16 | 40 | one-tile reduce-scatter, n = 4 | 216.2 | 202.8 | flydsl | 447.8 | 424.1 | 160x64x128 |
| 1024x5120x1280 | bfloat16 | 80 | data-parallel | 394.2 | 394.1 | torch | 600.3 | 603.2 | 160x128x128 |
| 2048x5120x1280 | bfloat16 | 160 | data-parallel | 754.4 | 754.8 | torch | 839.5 | 841.9 | 160x256x64 |
| 4096x5120x1280 | bfloat16 | 320 | data-parallel | 798.8 | 788.7 | torch | 872.1 | 874.7 | 160x256x64 |
| 8192x5120x1280 | bfloat16 | 640 | data-parallel | 1033.6 | 1005.4 | flydsl | 984.8 | 1069.5 | 256x224x64 |
| 16384x5120x1280 | bfloat16 | 1280 | data-parallel | 1228.2 | 1180.1 | flydsl | 1161.5 | 1221.1 | 256x256x64 |
| 32768x5120x1280 | bfloat16 | 2560 | data-parallel | 1257.4 | 1221.4 | torch | 1238.1 | 1246.6 | 256x256x64 |
| 512x1536x1536 | bfloat16 | 12 | one-tile reduce-scatter, n = 6 | 101.9 | 69.7 | flydsl | 265.0 | 237.4 | 64x64x256 |
| 512x2048x1536 | bfloat16 | 16 | one-tile reduce-scatter, n = 6 | 136.2 | 92.5 | torch | 310.5 | 311.4 | 64x64x256 |
| 512x2304x1536 | bfloat16 | 18 | one-tile reduce-scatter, n = 6 | 146.8 | 104.0 | flydsl | 319.5 | 313.6 | 64x80x256 |
| 512x4096x1536 | bfloat16 | 32 | one-tile reduce-scatter, n = 6 | 216.2 | 185.7 | torch | 445.5 | 447.0 | 128x64x128 |
| 512x6144x1536 | bfloat16 | 48 | one-tile reduce-scatter, n = 4 | 301.0 | 276.3 | flydsl | 541.5 | 541.1 | 96x128x128 |
| 512x7168x1536 | bfloat16 | 56 | one-tile reduce-scatter, n = 3 | 328.8 | 319.3 | flydsl | 582.5 | 490.4 | 128x128x256 |
| 1024x1536x1536 | bfloat16 | 24 | one-tile reduce-scatter, n = 6 | 185.1 | 140.1 | flydsl | 401.5 | 390.6 | 96x64x128 |
| 1024x2048x1536 | bfloat16 | 32 | one-tile reduce-scatter, n = 6 | 216.3 | 184.7 | torch | 455.1 | 457.7 | 128x64x128 |
| 1024x2304x1536 | bfloat16 | 36 | one-tile reduce-scatter, n = 5 | 231.6 | 207.3 | torch | 458.0 | 459.1 | 128x80x128 |
| 1024x3072x1536 | bfloat16 | 48 | one-tile reduce-scatter, n = 4 | 301.5 | 276.5 | flydsl | 551.0 | 555.2 | 96x128x128 |
| 1024x4096x1536 | bfloat16 | 64 | one-tile reduce-scatter, n = 3 | 360.1 | 363.3 | torch | 554.7 | 556.3 | 128x128x256 |
| 1024x7168x1536 | bfloat16 | 112 | data-parallel | 608.8 | 610.8 | torch | 778.1 | 780.5 | 224x128x64 |
| 2048x1536x1536 | bfloat16 | 48 | one-tile reduce-scatter, n = 4 | 301.9 | 281.5 | torch | 552.9 | 555.8 | 96x128x128 |
| 2048x2048x1536 | bfloat16 | 64 | one-tile reduce-scatter, n = 3 | 364.9 | 368.8 | torch | 558.9 | 559.4 | 128x128x256 |
| 2048x4096x1536 | bfloat16 | 128 | data-parallel | 687.9 | 691.6 | torch | 844.6 | 847.7 | 256x128x64 |
| 3072x2048x1536 | bfloat16 | 96 | one-tile owner, n = 2-3 | 455.5 | 536.6 | opus | 686.0 | 763.1 | 128x192x128 |
| 3072x4096x1536 | bfloat16 | 192 | data-parallel | 979.8 | 983.7 | opus | 993.3 | 952.6 | 256x192x64 |
| 4096x1536x1536 | bfloat16 | 96 | one-tile owner, n = 2-3 | 467.8 | 537.2 | torch | 733.6 | 736.8 | 192x128x128 |
| 4096x2048x1536 | bfloat16 | 128 | data-parallel | 683.7 | 684.7 | opus | 854.3 | 843.7 | 256x128x64 |
| 4096x3072x1536 | bfloat16 | 192 | data-parallel | 977.2 | 985.3 | flydsl | 941.6 | 1004.7 | 192x256x64 |
| 4096x4096x1536 | bfloat16 | 256 | data-parallel | 1177.5 | 1185.5 | torch | 1155.0 | 1167.2 | 256x256x64 |
| 5120x2048x1536 | bfloat16 | 160 | data-parallel | 834.7 | 839.3 | torch | 945.3 | 952.5 | 256x160x64 |
| 5120x4096x1536 | bfloat16 | 320 | data-parallel | 936.1 | 935.8 | torch | 996.9 | 1004.8 | 256x160x64 |
| 7680x2048x1536 | bfloat16 | 240 | data-parallel | 1123.1 | 1138.8 | torch | 946.4 | 950.8 | 256x240x64 |
| 7680x4096x1536 | bfloat16 | 480 | data-parallel | 1190.7 | 1190.3 | torch | 1015.3 | 1026.1 | 256x240x64 |
| 8192x1536x1536 | bfloat16 | 192 | data-parallel | 970.7 | 973.5 | torch | 987.5 | 998.2 | 192x256x64 |
| 10240x7168x1536 | bfloat16 | 1120 | data-parallel | 1228.8 | 1208.7 | asm | 1143.7 | 1124.5 | 224x256x64 |
| 16384x1536x1536 | bfloat16 | 384 | data-parallel | 1042.9 | 1051.0 | torch | 1036.7 | 1059.4 | 192x256x64 |
| 16384x2048x1536 | bfloat16 | 512 | data-parallel | 1204.8 | 1207.4 | torch | 1207.1 | 1214.1 | 256x256x64 |
| 16384x3072x1536 | bfloat16 | 768 | data-parallel | 1258.5 | 1245.0 | torch | 1232.1 | 1244.0 | 256x256x64 |
| 16384x4096x1536 | bfloat16 | 1024 | data-parallel | 1265.0 | 1240.0 | torch | 1250.9 | 1263.6 | 256x256x64 |
| 32768x1536x1536 | bfloat16 | 768 | data-parallel | 1246.5 | 1236.1 | flydsl | 1198.6 | 1221.5 | 256x256x64 |
| 512x7168x1792 | bfloat16 | 56 | one-tile reduce-scatter, n = 3 | 371.8 | 343.4 | flydsl | 620.2 | 501.2 | 128x128x256 |
| 1024x7168x1792 | bfloat16 | 112 | one-tile owner, n = 2-3 | 617.2 | 656.8 | flydsl | 783.1 | 829.4 | 224x128x64 |
| 2048x7168x1792 | bfloat16 | 224 | data-parallel | 1146.8 | 1166.9 | flydsl | 1104.9 | 1060.9 | 224x256x64 |
| 4096x7168x1792 | bfloat16 | 448 | data-parallel | 1203.1 | 1199.5 | flydsl | 1151.2 | 1119.4 | 224x256x64 |
| 8192x7168x1792 | bfloat16 | 896 | data-parallel | 1238.8 | 1216.4 | asm | 1189.6 | 1154.6 | 224x256x64 |
| 16384x7168x1792 | bfloat16 | 1792 | data-parallel | 1354.6 | 1329.2 | torch | 1326.1 | 1340.9 | 256x256x64 |
| 32768x7168x1792 | bfloat16 | 3584 | data-parallel | 1374.2 | 1349.0 | torch | 1344.2 | 1359.6 | 256x256x64 |
| 512x4096x2048 | bfloat16 | 32 | one-tile reduce-scatter, n = 6 | 289.8 | 213.9 | torch | 490.4 | 491.8 | 128x64x128 |
| 512x6144x2048 | float32 | 48 | one-tile reduce-scatter, n = 4 | 358.5 | 289.2 | asm | 332.2 | 559.3 | 96x128x128 |
| 512x6144x2048 | bfloat16 | 48 | one-tile reduce-scatter, n = 4 | 384.4 | 319.8 | flydsl | 602.7 | 594.0 | 96x128x128 |
| 512x8192x2048 | bfloat16 | 64 | one-tile reduce-scatter, n = 3 | 449.0 | 411.2 | flydsl | 708.2 | 574.4 | 128x128x256 |
| 512x16384x2048 | bfloat16 | 128 | one-tile owner, n = 2 | 759.9 | 748.9 | flydsl | 717.9 | 878.4 | 256x128x64 |
| 1024x4096x2048 | bfloat16 | 64 | one-tile reduce-scatter, n = 3 | 455.0 | 416.2 | torch | 610.6 | 609.1 | 128x128x256 |
| 1024x6144x2048 | float32 | 96 | one-tile owner, n = 2-3 | 505.9 | 545.7 | opus | 554.6 | 780.3 | 192x128x64 |
| 1024x6144x2048 | bfloat16 | 96 | one-tile owner, n = 2-3 | 539.8 | 604.5 | flydsl | 735.8 | 788.3 | 192x128x128 |
| 1024x8192x2048 | bfloat16 | 128 | one-tile owner, n = 2 | 775.7 | 764.3 | flydsl | 881.2 | 926.1 | 256x128x64 |
| 1024x16384x2048 | bfloat16 | 256 | data-parallel | 1245.2 | 1259.1 | flydsl | 740.7 | 1213.1 | 256x256x64 |
| 2048x4096x2048 | bfloat16 | 128 | one-tile owner, n = 2 | 784.2 | 773.1 | torch | 939.1 | 945.4 | 256x128x64 |
| 2048x6144x2048 | float32 | 192 | data-parallel | 955.4 | 959.9 | asm | 911.6 | 1004.6 | 192x256x64 |
| 2048x6144x2048 | bfloat16 | 192 | data-parallel | 1084.2 | 1090.7 | flydsl | 1034.8 | 1093.3 | 192x256x64 |
| 2048x8192x2048 | bfloat16 | 256 | data-parallel | 1254.9 | 1270.4 | flydsl | 1209.0 | 1228.6 | 256x256x64 |
| 2048x16384x2048 | bfloat16 | 512 | data-parallel | 1311.8 | 1309.3 | flydsl | 1031.4 | 1264.8 | 256x256x64 |
| 3072x4096x2048 | bfloat16 | 192 | data-parallel | 1078.0 | 1086.2 | torch | 1088.1 | 1096.4 | 256x192x64 |
| 4096x4096x2048 | bfloat16 | 256 | data-parallel | 1270.1 | 1279.0 | torch | 1238.8 | 1245.3 | 256x256x64 |
| 4096x6144x2048 | float32 | 384 | data-parallel | 984.0 | 987.3 | asm | 944.4 | 1054.5 | 192x256x64 |
| 4096x6144x2048 | bfloat16 | 384 | data-parallel | 1137.2 | 1135.5 | flydsl | 889.5 | 1148.0 | 192x256x64 |
| 4096x8192x2048 | bfloat16 | 512 | data-parallel | 1320.1 | 1310.0 | flydsl | 1266.3 | 1288.8 | 256x256x64 |
| 4096x16384x2048 | bfloat16 | 1024 | data-parallel | 1348.2 | 1331.8 | flydsl | 854.6 | 1335.8 | 256x256x64 |
| 5120x4096x2048 | bfloat16 | 320 | one-tile owner, n = 2 | 1045.8 | 999.7 | torch | 1123.2 | 1117.9 | 256x160x64 |
| 7680x4096x2048 | bfloat16 | 480 | data-parallel | 1292.6 | 1284.1 | torch | 1091.1 | 1099.1 | 256x240x64 |
| 8192x6144x2048 | float32 | 768 | data-parallel | 1182.6 | 1171.3 | asm | 1138.6 | 1098.7 | 256x256x64 |
| 8192x6144x2048 | bfloat16 | 768 | data-parallel | 1357.3 | 1339.8 | flydsl | 834.3 | 1328.9 | 256x256x64 |
| 8192x8192x2048 | bfloat16 | 1024 | data-parallel | 1364.6 | 1341.4 | asm | 1315.3 | 1339.7 | 256x256x64 |
| 8192x16384x2048 | bfloat16 | 2048 | data-parallel | 1371.4 | 1338.2 | triton | 1015.3 | 1371.2 | 256x256x64 |
| 16384x4096x2048 | bfloat16 | 1024 | data-parallel | 1353.4 | 1333.1 | torch | 1310.8 | 1325.7 | 256x256x64 |
| 16384x6144x2048 | float32 | 1536 | data-parallel | 1205.6 | 1186.1 | asm | 1160.7 | 1127.7 | 256x256x64 |
| 16384x6144x2048 | bfloat16 | 1536 | data-parallel | 1377.6 | 1350.3 | torch | 1346.1 | 1361.1 | 256x256x64 |
| 16384x8192x2048 | bfloat16 | 2048 | data-parallel | 1383.8 | 1359.2 | torch | 1354.0 | 1370.6 | 256x256x64 |
| 16384x16384x2048 | bfloat16 | 4096 | data-parallel | 1392.8 | 1357.7 | flydsl | 953.7 | 1392.9 | 256x256x64 |
| 32768x6144x2048 | bfloat16 | 3072 | data-parallel | 1405.6 | 1377.3 | flydsl | 992.4 | 1377.0 | 256x256x64 |
| 32768x8192x2048 | bfloat16 | 4096 | data-parallel | 1413.1 | 1378.8 | torch | 1373.7 | 1388.2 | 256x256x64 |
| 32768x16384x2048 | bfloat16 | 8192 | data-parallel | 1393.5 | 1351.5 | flydsl | 824.7 | 1405.3 | 256x256x64 |
| 512x6144x3072 | float32 | 48 | one-tile reduce-scatter, n = 4 | 455.6 | 292.6 | opus | 371.0 | 619.2 | 96x128x128 |
| 512x6144x3072 | bfloat16 | 48 | one-tile reduce-scatter, n = 4 | 483.6 | 314.4 | flydsl | 643.8 | 660.2 | 96x128x128 |
| 1024x6144x3072 | float32 | 96 | one-tile owner, n = 2-3 | 617.1 | 576.2 | opus | 568.6 | 843.0 | 192x128x64 |
| 1024x6144x3072 | bfloat16 | 96 | one-tile owner, n = 2-3 | 650.9 | 616.1 | flydsl | 785.7 | 829.9 | 192x128x128 |
| 2048x6144x3072 | float32 | 192 | data-parallel | 1038.9 | 1044.4 | asm | 1018.3 | 1085.1 | 192x256x64 |
| 2048x6144x3072 | bfloat16 | 192 | data-parallel | 1146.0 | 1156.1 | flydsl | 1112.9 | 1166.1 | 192x256x64 |
| 4096x6144x3072 | float32 | 384 | data-parallel | 1068.7 | 1081.4 | asm | 1038.7 | 1134.0 | 192x256x64 |
| 4096x6144x3072 | bfloat16 | 384 | data-parallel | 1197.2 | 1200.3 | flydsl | 926.1 | 1205.8 | 192x256x64 |
| 8192x6144x3072 | float32 | 768 | data-parallel | 1274.5 | 1271.5 | asm | 1233.8 | 1182.7 | 256x256x64 |
| 8192x6144x3072 | bfloat16 | 768 | data-parallel | 1416.5 | 1406.3 | flydsl | 780.2 | 1397.1 | 256x256x64 |
| 16384x6144x3072 | float32 | 1536 | data-parallel | 1299.6 | 1290.4 | asm | 1263.2 | 1218.3 | 256x256x64 |
| 16384x6144x3072 | bfloat16 | 1536 | data-parallel | 1441.8 | 1429.4 | torch | 1407.3 | 1422.0 | 256x256x64 |
| 32768x6144x3072 | bfloat16 | 3072 | data-parallel | 1466.6 | 1444.6 | flydsl | 859.5 | 1439.1 | 256x256x64 |
| 512x5120x3200 | bfloat16 | 40 | one-tile reduce-scatter, n = 4 | 453.0 | 268.4 | flydsl | 577.7 | 564.4 | 160x64x128 |
| 1024x5120x3200 | bfloat16 | 80 | one-tile owner, n = 3-4 | 684.6 | 529.7 | torch | 821.4 | 821.3 | 160x128x128 |
| 2048x5120x3200 | bfloat16 | 160 | data-parallel | 1032.1 | 1041.7 | torch | 1087.9 | 1095.9 | 160x256x64 |
| 4096x5120x3200 | bfloat16 | 320 | one-tile owner, n = 2 | 1160.1 | 1034.6 | torch | 1123.0 | 1129.6 | 160x256x64 |
| 8192x5120x3200 | bfloat16 | 640 | data-parallel | 1308.3 | 1298.3 | asm | 1251.5 | 1241.4 | 256x224x64 |
| 16384x5120x3200 | bfloat16 | 1280 | data-parallel | 1446.1 | 1430.3 | torch | 1410.5 | 1421.5 | 256x256x64 |
| 32768x5120x3200 | bfloat16 | 2560 | data-parallel | 1476.3 | 1452.0 | torch | 1427.0 | 1441.5 | 256x256x64 |
| 512x7168x3584 | bfloat16 | 56 | one-tile reduce-scatter, n = 3 | 562.8 | 374.6 | flydsl | 705.7 | 582.9 | 128x128x256 |
| 512x8192x3584 | bfloat16 | 64 | one-tile reduce-scatter, n = 3 | 616.2 | 427.8 | flydsl | 671.1 | 647.8 | 128x128x256 |
| 1024x7168x3584 | bfloat16 | 112 | one-tile owner, n = 2-3 | 852.0 | 732.9 | torch | 955.8 | 960.9 | 224x128x64 |
| 1024x8192x3584 | bfloat16 | 128 | one-tile owner, n = 2 | 986.5 | 825.8 | flydsl | 787.7 | 1020.1 | 256x128x64 |
| 2048x8192x3584 | bfloat16 | 256 | data-parallel | 1384.6 | 1385.4 | flydsl | 1277.3 | 1337.3 | 256x256x64 |
| 4096x8192x3584 | bfloat16 | 512 | data-parallel | 1426.3 | 1418.9 | flydsl | 981.3 | 1370.0 | 256x256x64 |
| 8192x8192x3584 | bfloat16 | 1024 | data-parallel | 1456.9 | 1443.9 | torch | 1399.3 | 1410.8 | 256x256x64 |
| 10240x7168x3584 | bfloat16 | 1120 | one-tile owner, n = 2-3 | 1372.9 | 1346.0 | flydsl | 782.4 | 1270.6 | 224x256x64 |
| 16384x8192x3584 | bfloat16 | 2048 | data-parallel | 1477.8 | 1464.8 | flydsl | 808.8 | 1437.0 | 256x256x64 |
| 32768x8192x3584 | bfloat16 | 4096 | data-parallel | 1497.0 | 1478.3 | flydsl | 854.6 | 1466.3 | 256x256x64 |
| 512x1024x4096 | bfloat16 | 8 | one-tile reduce-scatter, n = 16 | 183.7 | 59.7 | flydsl | 281.4 | 263.7 | 64x32x256 |
| 512x2048x4096 | bfloat16 | 16 | one-tile reduce-scatter, n = 12 | 286.8 | 119.9 | flydsl | 441.0 | 421.7 | 64x64x256 |
| 512x4096x4096 | bfloat16 | 32 | one-tile reduce-scatter, n = 6 | 460.0 | 239.6 | flydsl | 598.7 | 615.9 | 128x128x128 |
| 512x8192x4096 | bfloat16 | 64 | one-tile reduce-scatter, n = 3 | 686.4 | 465.1 | flydsl | 804.4 | 671.3 | 128x128x256 |
| 512x8704x4096 | bfloat16 | 68 | one-tile owner, n = 3-4 | 647.2 | 491.1 | torch | 740.0 | 742.1 | 160x128x128 |
| 512x10240x4096 | bfloat16 | 80 | one-tile owner, n = 3-4 | 758.8 | 572.6 | torch | 854.1 | 855.4 | 128x192x64 |
| 512x16384x4096 | bfloat16 | 128 | one-tile owner, n = 2 | 998.5 | 847.8 | flydsl | 632.9 | 695.6 | 256x256x64 |
| 512x129280x4096 | bfloat16 | 1010 | data-parallel | 1347.3 | 1342.3 | asm | 1276.2 | 1145.7 | 256x256x64 |
| 768x1024x4096 | bfloat16 | 12 | one-tile reduce-scatter, n = 16 | 226.4 | 90.9 | torch | 370.9 | 372.1 | 64x48x256 |
| 768x2048x4096 | bfloat16 | 24 | one-tile reduce-scatter, n = 8 | 387.7 | 180.7 | flydsl | 528.4 | 497.4 | 64x96x256 |
| 768x4096x4096 | bfloat16 | 48 | one-tile reduce-scatter, n = 4 | 598.3 | 359.6 | flydsl | 696.3 | 654.3 | 128x96x128 |
| 768x8192x4096 | bfloat16 | 96 | one-tile owner, n = 2-3 | 782.7 | 679.9 | torch | 876.6 | 883.4 | 128x192x128 |
| 768x8704x4096 | bfloat16 | 102 | one-tile owner, n = 2-3 | 838.0 | 717.5 | flydsl | 869.4 | 846.4 | 160x192x128 |
| 768x10240x4096 | bfloat16 | 120 | one-tile owner, n = 2-3 | 969.6 | 826.9 | flydsl | 965.9 | 937.9 | 160x192x128 |
| 768x129280x4096 | bfloat16 | 1515 | data-parallel | 1420.1 | 1410.2 | torch | 1341.4 | 1350.6 | 256x256x64 |
| 1024x512x4096 | bfloat16 | 8 | one-tile reduce-scatter, n = 16 | 184.4 | 60.3 | flydsl | 280.8 | 261.1 | 64x32x256 |
| 1024x1024x4096 | bfloat16 | 16 | one-tile reduce-scatter, n = 12 | 285.9 | 121.0 | flydsl | 444.6 | 437.9 | 64x64x256 |
| 1024x2048x4096 | bfloat16 | 32 | one-tile reduce-scatter, n = 6 | 456.6 | 240.4 | flydsl | 602.9 | 494.5 | 128x128x256 |
| 1024x4096x4096 | bfloat16 | 64 | one-tile reduce-scatter, n = 3 | 695.2 | 468.0 | flydsl | 832.2 | 708.6 | 128x128x256 |
| 1024x8192x4096 | bfloat16 | 128 | one-tile owner, n = 2 | 1005.9 | 871.0 | flydsl | 1014.6 | 695.0 | 256x256x64 |
| 1024x8704x4096 | bfloat16 | 136 | data-parallel | 924.7 | 924.0 | torch | 996.1 | 999.2 | 144x256x64 |
| 1024x10240x4096 | bfloat16 | 160 | data-parallel | 1053.3 | 1054.6 | torch | 1112.6 | 1118.0 | 160x256x64 |
| 1024x16384x4096 | bfloat16 | 256 | data-parallel | 1386.6 | 1391.4 | flydsl | 824.4 | 1322.1 | 256x256x64 |
| 1024x129280x4096 | bfloat16 | 2020 | data-parallel | 1437.3 | 1429.6 | torch | 1370.2 | 1379.1 | 256x256x64 |
| 1280x512x4096 | bfloat16 | 10 | one-tile reduce-scatter, n = 16 | 206.7 | 75.3 | flydsl | 321.4 | 310.0 | 64x48x256 |
| 1280x1024x4096 | bfloat16 | 20 | one-tile reduce-scatter, n = 9 | 340.1 | 151.3 | torch | 483.3 | 487.8 | 64x80x256 |
| 1280x2048x4096 | bfloat16 | 40 | one-tile reduce-scatter, n = 4 | 540.2 | 302.5 | flydsl | 620.2 | 610.6 | 128x80x128 |
| 1280x129280x4096 | bfloat16 | 2525 | data-parallel | 1412.1 | 1407.5 | torch | 1385.4 | 1396.6 | 256x256x64 |
| 1536x512x4096 | bfloat16 | 12 | one-tile reduce-scatter, n = 16 | 223.8 | 90.4 | torch | 366.7 | 367.0 | 64x48x256 |
| 1536x1024x4096 | bfloat16 | 24 | one-tile reduce-scatter, n = 8 | 387.0 | 182.7 | flydsl | 523.3 | 505.0 | 64x96x256 |
| 1536x2048x4096 | bfloat16 | 48 | one-tile reduce-scatter, n = 4 | 598.3 | 362.2 | flydsl | 699.6 | 681.9 | 128x96x128 |
| 1536x129280x4096 | bfloat16 | 3030 | data-parallel | 1421.4 | 1421.6 | torch | 1389.0 | 1396.8 | 256x256x64 |
| 1792x512x4096 | bfloat16 | 14 | one-tile reduce-scatter, n = 13 | 250.2 | 105.7 | flydsl | 395.6 | 384.3 | 64x64x256 |
| 1792x1024x4096 | bfloat16 | 28 | one-tile reduce-scatter, n = 6 | 434.2 | 212.8 | flydsl | 574.7 | 462.8 | 256x128x64 |
| 1792x2048x4096 | bfloat16 | 56 | one-tile reduce-scatter, n = 3 | 637.6 | 418.9 | flydsl | 752.3 | 667.4 | 128x128x256 |
| 1792x129280x4096 | bfloat16 | 3535 | data-parallel | 1446.5 | 1432.1 | asm | 1410.2 | 1354.1 | 256x256x64 |
| 2048x512x4096 | bfloat16 | 16 | one-tile reduce-scatter, n = 12 | 286.8 | 121.4 | flydsl | 426.0 | 430.1 | 64x64x256 |
| 2048x1024x4096 | bfloat16 | 32 | one-tile reduce-scatter, n = 6 | 461.8 | 243.5 | flydsl | 599.0 | 492.7 | 128x128x256 |
| 2048x2048x4096 | bfloat16 | 64 | one-tile reduce-scatter, n = 3 | 704.3 | 474.5 | flydsl | 836.4 | 738.4 | 128x128x256 |
| 2048x4096x4096 | bfloat16 | 128 | one-tile owner, n = 2 | 1028.5 | 877.4 | opus | 1033.3 | 684.7 | 256x256x64 |
| 2048x8192x4096 | bfloat16 | 256 | data-parallel | 1391.0 | 1391.4 | flydsl | 1286.6 | 1345.3 | 256x256x64 |
| 2048x8704x4096 | bfloat16 | 272 | one-tile reduce-scatter, n = 8 | 1205.2 | 925.1 | torch | 1055.2 | 1062.3 | 224x160x64 |
| 2048x10240x4096 | bfloat16 | 320 | one-tile owner, n = 2 | 1195.4 | 1070.4 | torch | 1130.0 | 1138.0 | 160x256x64 |
| 2048x16384x4096 | bfloat16 | 512 | data-parallel | 1437.8 | 1434.3 | flydsl | 1363.2 | 1368.1 | 256x256x64 |
| 2048x129280x4096 | bfloat16 | 4040 | data-parallel | 1487.6 | 1470.6 | flydsl | 1421.9 | 1382.0 | 256x256x64 |
| 2304x512x4096 | bfloat16 | 18 | one-tile reduce-scatter, n = 10 | 307.8 | 135.7 | flydsl | 351.5 | 445.2 | 64x80x256 |
| 2304x1024x4096 | bfloat16 | 36 | one-tile reduce-scatter, n = 5 | 495.3 | 272.0 | flydsl | 509.9 | 649.1 | 128x80x128 |
| 2304x2048x4096 | bfloat16 | 72 | one-tile owner, n = 3-4 | 716.0 | 528.3 | opus | 680.4 | 878.6 | 128x160x128 |
| 2560x512x4096 | bfloat16 | 20 | one-tile reduce-scatter, n = 9 | 333.7 | 150.5 | flydsl | 386.8 | 486.2 | 64x80x256 |
| 2560x1024x4096 | bfloat16 | 40 | one-tile reduce-scatter, n = 4 | 537.4 | 302.0 | flydsl | 556.7 | 606.2 | 128x80x128 |
| 2560x2048x4096 | bfloat16 | 80 | one-tile owner, n = 3-4 | 774.1 | 582.1 | opus | 760.6 | 885.5 | 128x160x128 |
| 2816x512x4096 | bfloat16 | 22 | one-tile reduce-scatter, n = 8 | 367.1 | 165.0 | flydsl | 419.1 | 466.1 | 64x96x256 |
| 2816x1024x4096 | bfloat16 | 44 | one-tile reduce-scatter, n = 4 | 567.4 | 331.9 | flydsl | 602.1 | 643.0 | 128x96x128 |
| 2816x2048x4096 | bfloat16 | 88 | one-tile owner, n = 2-3 | 801.6 | 634.3 | opus | 812.5 | 853.2 | 192x128x128 |
| 3072x512x4096 | bfloat16 | 24 | one-tile reduce-scatter, n = 8 | 381.2 | 181.0 | flydsl | 453.2 | 503.3 | 64x96x256 |
| 3072x1024x4096 | bfloat16 | 48 | one-tile reduce-scatter, n = 4 | 597.0 | 361.6 | flydsl | 646.8 | 676.4 | 128x96x128 |
| 3072x2048x4096 | bfloat16 | 96 | one-tile owner, n = 2-3 | 764.5 | 686.5 | opus | 921.6 | 909.6 | 128x192x128 |
| 3328x512x4096 | bfloat16 | 26 | one-tile reduce-scatter, n = 7 | 397.6 | 195.9 | flydsl | 480.6 | 442.9 | 256x112x64 |
| 3328x1024x4096 | bfloat16 | 52 | one-tile reduce-scatter, n = 3 | 599.6 | 388.9 | flydsl | 680.1 | 626.8 | 128x128x256 |
| 3328x2048x4096 | bfloat16 | 104 | one-tile owner, n = 2-3 | 845.5 | 736.3 | opus | 934.4 | 774.6 | 256x208x64 |
| 3584x512x4096 | bfloat16 | 28 | one-tile reduce-scatter, n = 6 | 425.4 | 210.6 | flydsl | 512.2 | 448.7 | 256x128x64 |
| 3584x1024x4096 | bfloat16 | 56 | one-tile reduce-scatter, n = 3 | 630.9 | 415.8 | flydsl | 712.8 | 662.5 | 128x128x256 |
| 3584x2048x4096 | bfloat16 | 112 | one-tile owner, n = 2-3 | 884.1 | 785.5 | opus | 1002.5 | 879.1 | 160x192x128 |
| 3840x512x4096 | bfloat16 | 30 | one-tile reduce-scatter, n = 6 | 443.0 | 226.5 | flydsl | 537.4 | 481.9 | 128x128x256 |
| 3840x1024x4096 | bfloat16 | 60 | one-tile reduce-scatter, n = 3 | 657.1 | 441.6 | flydsl | 733.2 | 699.1 | 128x128x256 |
| 3840x2048x4096 | bfloat16 | 120 | one-tile owner, n = 2-3 | 988.5 | 832.9 | opus | 1034.2 | 752.1 | 256x240x64 |
| 4096x512x4096 | bfloat16 | 32 | one-tile reduce-scatter, n = 6 | 454.3 | 239.6 | flydsl | 591.9 | 497.8 | 128x128x256 |
| 4096x1024x4096 | bfloat16 | 64 | one-tile reduce-scatter, n = 3 | 692.7 | 471.0 | flydsl | 830.7 | 736.8 | 128x128x256 |
| 4096x2048x4096 | bfloat16 | 128 | one-tile owner, n = 2 | 1011.0 | 879.5 | opus | 1084.1 | 704.5 | 256x256x64 |
| 4096x4096x4096 | bfloat16 | 256 | data-parallel | 1388.9 | 1388.5 | flydsl | 1310.5 | 1330.7 | 256x256x64 |
| 4096x8192x4096 | bfloat16 | 512 | data-parallel | 1427.6 | 1426.4 | asm | 1357.6 | 1370.4 | 256x256x64 |
| 4096x8704x4096 | bfloat16 | 544 | one-tile reduce-scatter, n = 4 | 1328.9 | 1152.7 | torch | 1192.5 | 1179.6 | 192x256x64 |
| 4096x10240x4096 | bfloat16 | 640 | one-tile owner, n = 2 | 1374.2 | 1334.6 | flydsl | 1284.0 | 1223.5 | 224x256x64 |
| 4096x16384x4096 | bfloat16 | 1024 | data-parallel | 1463.0 | 1460.3 | flydsl | 1383.5 | 1411.7 | 256x256x64 |
| 4096x129280x4096 | bfloat16 | 8080 | data-parallel | 1491.7 | 1487.9 | asm | 1428.5 | 1391.5 | 256x256x64 |
| 4352x512x4096 | bfloat16 | 34 | one-tile reduce-scatter, n = 5 | 475.9 | 251.4 | flydsl | 466.5 | 523.0 | 128x80x128 |
| 4352x1024x4096 | bfloat16 | 68 | one-tile owner, n = 3-4 | 671.2 | 492.3 | opus | 637.8 | 806.2 | 128x160x128 |
| 4352x2048x4096 | bfloat16 | 136 | data-parallel | 929.8 | 931.2 | torch | 1018.4 | 1025.3 | 192x192x64 |
| 5120x512x4096 | bfloat16 | 40 | one-tile reduce-scatter, n = 4 | 529.8 | 298.4 | flydsl | 537.0 | 599.2 | 128x80x128 |
| 5120x1024x4096 | bfloat16 | 80 | one-tile owner, n = 3-4 | 767.6 | 575.4 | opus | 762.8 | 880.6 | 128x160x128 |
| 5120x2048x4096 | bfloat16 | 160 | data-parallel | 1061.5 | 1058.5 | torch | 1146.4 | 1144.4 | 256x160x64 |
| 6144x512x4096 | bfloat16 | 48 | one-tile reduce-scatter, n = 4 | 591.2 | 356.6 | flydsl | 687.7 | 665.0 | 128x96x128 |
| 6144x1024x4096 | bfloat16 | 96 | one-tile owner, n = 2-3 | 772.2 | 680.6 | opus | 927.7 | 907.6 | 128x192x128 |
| 6144x2048x4096 | bfloat16 | 192 | data-parallel | 1214.8 | 1235.7 | opus | 1227.9 | 1205.7 | 256x192x64 |
| 6144x129280x4096 | bfloat16 | 12120 | one-tile owner, n = 2-3 | 1485.2 | 1465.0 | asm | 1413.4 | 1400.7 | 256x256x64 |
| 6400x512x4096 | bfloat16 | 50 | one-tile reduce-scatter, n = 3 | 571.0 | 368.6 | flydsl | 594.1 | 606.1 | 128x128x256 |
| 6400x1024x4096 | bfloat16 | 100 | one-tile owner, n = 2-3 | 828.3 | 703.4 | opus | 881.9 | 756.9 | 256x208x64 |
| 6400x2048x4096 | bfloat16 | 200 | data-parallel | 1244.6 | 1251.2 | torch | 1105.1 | 1113.9 | 256x208x64 |
| 6912x512x4096 | bfloat16 | 54 | one-tile reduce-scatter, n = 3 | 608.7 | 396.3 | flydsl | 609.7 | 646.4 | 128x128x256 |
| 6912x1024x4096 | bfloat16 | 108 | one-tile owner, n = 2-3 | 891.0 | 752.7 | opus | 932.8 | 680.2 | 256x224x64 |
| 6912x2048x4096 | bfloat16 | 216 | data-parallel | 1292.2 | 1294.2 | torch | 1209.0 | 1217.3 | 256x224x64 |
| 7168x512x4096 | bfloat16 | 56 | one-tile reduce-scatter, n = 3 | 624.9 | 410.3 | flydsl | 643.2 | 665.3 | 128x128x256 |
| 7168x1024x4096 | bfloat16 | 112 | one-tile owner, n = 2-3 | 932.0 | 777.3 | opus | 997.5 | 1051.5 | 256x128x64 |
| 7168x2048x4096 | bfloat16 | 224 | data-parallel | 1321.1 | 1316.5 | torch | 1237.3 | 1244.6 | 256x224x64 |
| 7424x512x4096 | bfloat16 | 58 | one-tile reduce-scatter, n = 3 | 636.6 | 424.0 | flydsl | 620.4 | 655.5 | 128x128x256 |
| 7424x1024x4096 | bfloat16 | 116 | one-tile owner, n = 2-3 | 962.9 | 798.8 | opus | 977.3 | 735.5 | 256x240x64 |
| 7424x2048x4096 | bfloat16 | 232 | data-parallel | 1339.6 | 1339.8 | torch | 1121.6 | 1127.9 | 256x240x64 |
| 7680x512x4096 | bfloat16 | 60 | one-tile reduce-scatter, n = 3 | 653.8 | 438.1 | flydsl | 662.0 | 672.1 | 128x128x256 |
| 7680x1024x4096 | bfloat16 | 120 | one-tile owner, n = 2-3 | 977.4 | 824.4 | opus | 1030.4 | 715.5 | 256x240x64 |
| 7680x2048x4096 | bfloat16 | 240 | data-parallel | 1366.6 | 1374.0 | torch | 1148.2 | 1155.5 | 256x240x64 |
| 7936x512x4096 | bfloat16 | 62 | one-tile reduce-scatter, n = 3 | 673.4 | 451.8 | flydsl | 711.9 | 689.1 | 128x128x256 |
| 7936x1024x4096 | bfloat16 | 124 | one-tile owner, n = 2-3 | 1004.6 | 844.3 | opus | 1013.4 | 696.8 | 256x256x64 |
| 7936x2048x4096 | bfloat16 | 248 | data-parallel | 1377.0 | 1388.9 | torch | 1345.7 | 1356.5 | 256x256x64 |
| 8192x512x4096 | bfloat16 | 64 | one-tile reduce-scatter, n = 3 | 675.7 | 465.4 | flydsl | 789.4 | 706.0 | 128x128x256 |
| 8192x1024x4096 | bfloat16 | 128 | one-tile owner, n = 2 | 1005.7 | 867.6 | opus | 1082.4 | 683.9 | 256x256x64 |
| 8192x2048x4096 | bfloat16 | 256 | data-parallel | 1387.3 | 1389.7 | torch | 1359.0 | 1363.3 | 256x256x64 |
| 8192x4096x4096 | bfloat16 | 512 | data-parallel | 1438.9 | 1438.7 | flydsl | 1355.0 | 1401.2 | 256x256x64 |
| 8192x8192x4096 | bfloat16 | 1024 | data-parallel | 1463.2 | 1460.3 | asm | 1405.5 | 1415.0 | 256x256x64 |
| 8192x8704x4096 | bfloat16 | 1088 | one-tile owner, n = 2 | 1388.4 | 1323.2 | asm | 1278.6 | 1282.6 | 256x224x64 |
| 8192x10240x4096 | bfloat16 | 1280 | data-parallel | 1476.6 | 1472.6 | torch | 1422.5 | 1429.3 | 256x256x64 |
| 8192x16384x4096 | bfloat16 | 2048 | data-parallel | 1502.0 | 1488.9 | torch | 1438.0 | 1448.7 | 256x256x64 |
| 8192x129280x4096 | bfloat16 | 16160 | one-tile reduce-scatter, n = 4 | 1504.8 | 1496.6 | flydsl | 1426.4 | 1387.5 | 256x256x64 |
| 10240x512x4096 | bfloat16 | 80 | one-tile owner, n = 3-4 | 762.1 | 568.2 | torch | 853.7 | 855.1 | 128x160x128 |
| 10240x1024x4096 | bfloat16 | 160 | data-parallel | 1094.8 | 1095.6 | torch | 1131.3 | 1135.4 | 256x160x64 |
| 10240x2048x4096 | bfloat16 | 320 | one-tile owner, n = 2 | 1199.3 | 1077.1 | torch | 1175.0 | 1182.2 | 256x160x64 |
| 10240x129280x4096 | bfloat16 | 20200 | data-parallel | 1469.9 | 1455.5 | asm | 1443.2 | 1377.3 | 256x256x64 |
| 12288x512x4096 | bfloat16 | 96 | one-tile owner, n = 2-3 | 763.0 | 668.7 | opus | 884.0 | 891.1 | 128x192x128 |
| 12288x1024x4096 | bfloat16 | 192 | data-parallel | 1185.6 | 1197.8 | opus | 1259.8 | 1209.4 | 256x192x64 |
| 12288x2048x4096 | bfloat16 | 384 | one-tile owner, n = 2 | 1307.6 | 1253.3 | opus | 1284.8 | 1247.2 | 256x192x64 |
| 12288x129280x4096 | bfloat16 | 24240 | data-parallel | 1496.9 | 1490.2 | asm | 1443.4 | 1390.6 | 256x256x64 |
| 14336x512x4096 | bfloat16 | 112 | one-tile owner, n = 2-3 | 924.8 | 763.4 | opus | 954.5 | 692.1 | 256x224x64 |
| 14336x1024x4096 | bfloat16 | 224 | data-parallel | 1291.7 | 1291.4 | torch | 1249.4 | 1258.9 | 256x224x64 |
| 14336x2048x4096 | bfloat16 | 448 | data-parallel | 1343.1 | 1340.0 | flydsl | 1299.4 | 1290.2 | 256x224x64 |
| 14336x129280x4096 | bfloat16 | 28280 | one-tile owner, n = 2-3 | 1453.8 | 1473.1 | asm | 1433.8 | 1383.4 | 256x256x64 |
| 16384x512x4096 | bfloat16 | 128 | one-tile owner, n = 2 | 985.3 | 850.1 | opus | 1032.0 | 780.7 | 256x256x64 |
| 16384x1024x4096 | bfloat16 | 256 | data-parallel | 1388.5 | 1392.6 | flydsl | 1306.9 | 1321.1 | 256x256x64 |
| 16384x2048x4096 | bfloat16 | 512 | data-parallel | 1422.9 | 1421.4 | torch | 1380.7 | 1386.5 | 256x256x64 |
| 16384x16384x4096 | bfloat16 | 4096 | data-parallel | 1515.5 | 1504.6 | torch | 1454.9 | 1484.9 | 256x256x64 |
| 16384x129280x4096 | bfloat16 | 32320 | one-tile owner, n = 2 | 1447.5 | 1489.7 | asm | 1431.0 | 1387.5 | 256x256x64 |
| 32768x16384x4096 | bfloat16 | 8192 | data-parallel | 1502.6 | 1493.0 | flydsl | 1400.2 | 1446.0 | 256x256x64 |
| 512x7168x4224 | bfloat16 | 56 | one-tile reduce-scatter, n = 3 | 609.4 | 375.0 | flydsl | 724.3 | 711.9 | 128x128x128 |
| 1024x7168x4224 | bfloat16 | 112 | one-tile owner, n = 2-3 | 932.1 | 736.5 | flydsl | 928.8 | 765.9 | 240x256x64 |
| 10240x7168x4224 | bfloat16 | 1120 | one-tile owner, n = 2-3 | 1416.4 | 1352.5 | opus | 902.7 | 1276.6 | 224x256x64 |
| 512x6144x4608 | bfloat16 | 48 | one-tile reduce-scatter, n = 4 | 622.9 | 336.3 | flydsl | 687.6 | 613.1 | 192x128x128 |
| 1024x6144x4608 | bfloat16 | 96 | one-tile owner, n = 2-3 | 780.7 | 661.5 | torch | 872.5 | 875.3 | 192x128x128 |
| 2048x6144x4608 | bfloat16 | 192 | data-parallel | 1227.7 | 1235.4 | flydsl | 1184.0 | 1222.9 | 192x256x64 |
| 4096x6144x4608 | bfloat16 | 384 | one-tile owner, n = 2 | 1330.2 | 1264.2 | torch | 1241.2 | 1246.5 | 192x256x64 |
| 8192x6144x4608 | bfloat16 | 768 | data-parallel | 1464.5 | 1454.8 | torch | 1426.6 | 1433.4 | 256x256x64 |
| 16384x6144x4608 | bfloat16 | 1536 | data-parallel | 1491.8 | 1478.4 | torch | 1449.0 | 1455.6 | 256x256x64 |
| 32768x6144x4608 | bfloat16 | 3072 | data-parallel | 1516.5 | 1499.7 | torch | 1466.4 | 1490.4 | 256x256x64 |
| 512x1024x5120 | float32 | 8 | one-tile reduce-scatter, n = 20 | 193.7 | 51.2 | asm | 248.2 | 274.7 | 64x32x256 |
| 512x5120x5120 | bfloat16 | 40 | one-tile reduce-scatter, n = 4 | 569.8 | 268.2 | torch | 653.9 | 654.7 | 96x128x128 |
| 512x6400x5120 | bfloat16 | 50 | one-tile reduce-scatter, n = 3 | 601.1 | 336.7 | flydsl | 605.3 | 554.5 | 128x128x256 |
| 512x12800x5120 | bfloat16 | 100 | one-tile owner, n = 2-3 | 844.0 | 653.9 | flydsl | 657.6 | 804.2 | 208x256x64 |
| 512x37888x5120 | bfloat16 | 296 | one-tile owner, n = 3 | 1148.0 | 933.1 | torch | 1046.8 | 1051.3 | 160x256x64 |
| 512x51200x5120 | bfloat16 | 400 | data-parallel | 1191.7 | 1192.1 | flydsl | 1131.6 | 1081.2 | 208x256x64 |
| 768x1024x5120 | float32 | 12 | one-tile reduce-scatter, n = 16 | 254.9 | 78.3 | asm | 312.9 | 312.0 | 64x64x256 |
| 768x37888x5120 | bfloat16 | 444 | data-parallel | 1300.6 | 1299.9 | torch | 1212.0 | 1215.6 | 224x256x64 |
| 1024x1024x5120 | float32 | 16 | one-tile reduce-scatter, n = 12 | 307.5 | 104.2 | asm | 332.7 | 402.2 | 64x64x256 |
| 1024x5120x5120 | bfloat16 | 80 | one-tile owner, n = 3-4 | 853.3 | 535.0 | torch | 841.2 | 843.0 | 160x128x128 |
| 1024x6400x5120 | bfloat16 | 100 | one-tile owner, n = 2-3 | 858.3 | 664.2 | flydsl | 820.9 | 806.3 | 256x208x64 |
| 1024x12800x5120 | bfloat16 | 200 | data-parallel | 1220.4 | 1223.7 | flydsl | 725.5 | 1054.4 | 256x208x64 |
| 1024x37888x5120 | bfloat16 | 592 | one-tile owner, n = 2 | 1313.2 | 1207.5 | torch | 1052.4 | 1056.5 | 256x208x64 |
| 1024x51200x5120 | bfloat16 | 800 | one-tile reduce-scatter, n = 4 | 1373.6 | 1203.5 | asm | 1150.2 | 1066.7 | 256x208x64 |
| 1536x1024x5120 | float32 | 24 | one-tile reduce-scatter, n = 8 | 421.6 | 157.2 | asm | 436.8 | 512.2 | 64x96x128 |
| 1536x37888x5120 | bfloat16 | 888 | one-tile owner, n = 2-3 | 1383.8 | 1323.4 | torch | 1244.4 | 1251.8 | 224x256x64 |
| 2048x1024x5120 | float32 | 32 | one-tile reduce-scatter, n = 6 | 498.4 | 210.8 | opus | 428.4 | 550.0 | 128x128x128 |
| 2048x5120x5120 | bfloat16 | 160 | data-parallel | 1046.0 | 1046.9 | torch | 1141.2 | 1148.8 | 160x256x64 |
| 2048x6400x5120 | bfloat16 | 200 | data-parallel | 1243.2 | 1244.5 | flydsl | 1124.3 | 1085.6 | 256x208x64 |
| 2048x12800x5120 | bfloat16 | 400 | data-parallel | 1282.9 | 1287.9 | flydsl | 808.3 | 1075.4 | 256x208x64 |
| 2048x37888x5120 | bfloat16 | 1184 | data-parallel | 1429.9 | 1417.9 | torch | 1189.0 | 1196.4 | 240x256x64 |
| 2048x51200x5120 | bfloat16 | 1600 | one-tile owner, n = 2 | 1441.4 | 1383.8 | asm | 209.9 | 1170.3 | 240x256x64 |
| 4096x1024x5120 | float32 | 64 | one-tile reduce-scatter, n = 3 | 698.2 | 411.1 | opus | 708.7 | 765.4 | 128x128x128 |
| 4096x5120x5120 | bfloat16 | 320 | one-tile owner, n = 2 | 1240.1 | 1072.9 | torch | 1162.7 | 1172.6 | 160x256x64 |
| 4096x6400x5120 | bfloat16 | 400 | data-parallel | 1297.0 | 1298.4 | flydsl | 1241.6 | 1111.5 | 256x208x64 |
| 4096x12800x5120 | bfloat16 | 800 | one-tile reduce-scatter, n = 4 | 1408.5 | 1253.5 | asm | 1207.1 | 1153.5 | 256x208x64 |
| 4096x51200x5120 | bfloat16 | 3200 | one-tile owner, n = 2 | 1513.8 | 1479.1 | asm | 1404.2 | 1378.9 | 256x256x64 |
| 8192x1024x5120 | float32 | 128 | one-tile owner, n = 2 | 1021.4 | 795.4 | opus | 1019.9 | 978.4 | 256x128x64 |
| 8192x5120x5120 | bfloat16 | 640 | one-tile owner, n = 2 | 1415.3 | 1343.1 | torch | 1281.1 | 1294.0 | 256x224x64 |
| 8192x6400x5120 | bfloat16 | 800 | one-tile reduce-scatter, n = 4 | 1408.0 | 1248.5 | flydsl | 1204.4 | 1178.1 | 256x208x64 |
| 8192x12800x5120 | bfloat16 | 1600 | one-tile owner, n = 2 | 1449.6 | 1386.2 | asm | 1338.3 | 1179.9 | 256x240x64 |
| 8192x51200x5120 | bfloat16 | 6400 | data-parallel | 1536.7 | 1535.3 | flydsl | 1462.4 | 1414.9 | 256x256x64 |
| 16384x1024x5120 | float32 | 256 | data-parallel | 1320.3 | 1320.7 | opus | 1165.4 | 1218.3 | 256x256x64 |
| 16384x5120x5120 | bfloat16 | 1280 | data-parallel | 1487.0 | 1476.8 | torch | 1457.9 | 1464.7 | 256x256x64 |
| 16384x6400x5120 | bfloat16 | 1600 | one-tile owner, n = 2 | 1450.8 | 1383.1 | flydsl | 1327.5 | 1197.8 | 256x240x64 |
| 16384x12800x5120 | bfloat16 | 3200 | one-tile owner, n = 2 | 1519.6 | 1482.8 | torch | 1459.2 | 1472.9 | 256x256x64 |
| 16384x51200x5120 | bfloat16 | 12800 | data-parallel | 1555.7 | 1528.9 | flydsl | 1484.2 | 1417.0 | 256x256x64 |
| 32768x5120x5120 | bfloat16 | 2560 | data-parallel | 1520.7 | 1504.4 | torch | 1477.0 | 1490.5 | 256x256x64 |
| 32768x6400x5120 | bfloat16 | 3200 | one-tile owner, n = 2 | 1500.4 | 1469.5 | torch | 1464.2 | 1476.8 | 256x256x64 |
| 32768x12800x5120 | bfloat16 | 6400 | data-parallel | 1534.8 | 1515.3 | torch | 1492.5 | 1517.1 | 256x256x64 |
| 32768x51200x5120 | bfloat16 | 25600 | data-parallel | 1529.4 | 1527.0 | flydsl | 1442.9 | 1384.0 | 256x256x64 |
| 512x1280x6144 | bfloat16 | 10 | one-tile reduce-scatter, n = 19 | 246.0 | 67.0 | flydsl | 339.8 | 336.8 | 160x64x128 |
| 512x1536x6144 | bfloat16 | 12 | one-tile reduce-scatter, n = 16 | 297.7 | 80.8 | flydsl | 381.3 | 371.0 | 192x64x128 |
| 512x2048x6144 | float32 | 16 | one-tile reduce-scatter, n = 12 | 355.6 | 104.7 | asm | 341.4 | 420.2 | 64x64x256 |
| 512x2304x6144 | bfloat16 | 18 | one-tile reduce-scatter, n = 10 | 377.3 | 121.1 | flydsl | 436.0 | 370.8 | 192x128x128 |
| 512x2560x6144 | bfloat16 | 20 | one-tile reduce-scatter, n = 9 | 404.4 | 135.3 | torch | 473.2 | 472.5 | 256x128x64 |
| 512x3072x6144 | float32 | 24 | one-tile reduce-scatter, n = 8 | 469.7 | 157.1 | opus | 401.7 | 538.8 | 96x128x128 |
| 512x3072x6144 | bfloat16 | 24 | one-tile reduce-scatter, n = 8 | 479.3 | 161.9 | flydsl | 543.2 | 452.7 | 192x128x128 |
| 512x6144x6144 | bfloat16 | 48 | one-tile reduce-scatter, n = 4 | 688.9 | 325.7 | flydsl | 686.1 | 687.7 | 192x128x128 |
| 512x9216x6144 | bfloat16 | 72 | one-tile owner, n = 3-4 | 793.7 | 479.4 | flydsl | 592.1 | 582.5 | 256x256x64 |
| 1024x2048x6144 | float32 | 32 | one-tile reduce-scatter, n = 6 | 546.2 | 211.1 | opus | 406.2 | 644.7 | 128x128x128 |
| 1024x2304x6144 | bfloat16 | 36 | one-tile reduce-scatter, n = 5 | 578.8 | 244.1 | torch | 593.0 | 592.6 | 256x176x64 |
| 1024x2560x6144 | bfloat16 | 40 | one-tile reduce-scatter, n = 4 | 613.4 | 271.1 | torch | 684.3 | 698.9 | 160x128x128 |
| 1024x3072x6144 | float32 | 48 | one-tile reduce-scatter, n = 4 | 668.2 | 315.0 | opus | 497.6 | 716.3 | 96x128x128 |
| 1024x6144x6144 | bfloat16 | 96 | one-tile owner, n = 2-3 | 872.0 | 646.1 | torch | 986.9 | 991.3 | 192x256x64 |
| 1024x9216x6144 | bfloat16 | 144 | spread | 1088.2 | 939.7 | torch | 1054.3 | 1057.1 | 144x256x64 |
| 2048x2048x6144 | float32 | 64 | one-tile reduce-scatter, n = 3 | 758.6 | 425.0 | opus | 671.0 | 807.0 | 128x128x128 |
| 2048x2304x6144 | bfloat16 | 72 | one-tile owner, n = 3-4 | 865.9 | 496.7 | torch | 837.8 | 838.5 | 128x160x128 |
| 2048x2560x6144 | bfloat16 | 80 | one-tile owner, n = 3-4 | 911.3 | 552.7 | torch | 908.4 | 911.2 | 160x128x128 |
| 2048x3072x6144 | float32 | 96 | one-tile owner, n = 2-3 | 890.9 | 638.8 | opus | 755.8 | 908.7 | 192x256x64 |
| 2048x6144x6144 | bfloat16 | 192 | data-parallel | 1230.9 | 1231.8 | torch | 1232.8 | 1242.4 | 192x256x64 |
| 2048x9216x6144 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 | 1276.0 | 974.2 | torch | 1084.2 | 1088.2 | 144x256x64 |
| 4096x2048x6144 | float32 | 128 | one-tile owner, n = 2 | 1087.1 | 821.2 | opus | 1021.3 | 809.2 | 256x256x64 |
| 4096x2304x6144 | bfloat16 | 144 | spread | 1104.7 | 954.6 | flydsl | 680.4 | 1088.1 | 144x256x64 |
| 4096x2560x6144 | bfloat16 | 160 | data-parallel | 1068.5 | 1069.5 | flydsl | 637.2 | 1172.4 | 160x256x64 |
| 4096x3072x6144 | float32 | 192 | data-parallel | 1170.0 | 1180.2 | asm | 1138.5 | 1210.8 | 192x256x64 |
| 4096x6144x6144 | bfloat16 | 384 | one-tile owner, n = 2 | 1380.5 | 1282.0 | torch | 1261.4 | 1266.6 | 192x256x64 |
| 4096x9216x6144 | bfloat16 | 576 | one-tile owner, n = 3 | 1398.5 | 1246.4 | torch | 1262.7 | 1269.5 | 192x256x64 |
| 8192x2048x6144 | float32 | 256 | data-parallel | 1340.4 | 1344.2 | opus | 1225.9 | 1225.2 | 256x256x64 |
| 8192x2304x6144 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 | 1218.0 | 973.9 | torch | 1089.2 | 1096.0 | 144x256x64 |
| 8192x2560x6144 | bfloat16 | 320 | one-tile owner, n = 3 | 1276.7 | 1076.8 | asm | 405.5 | 1182.4 | 160x256x64 |
| 8192x3072x6144 | float32 | 384 | one-tile owner, n = 2 | 1276.3 | 1198.7 | asm | 1171.4 | 1216.6 | 192x256x64 |
| 8192x6144x6144 | bfloat16 | 768 | data-parallel | 1493.0 | 1488.7 | flydsl | 1422.2 | 1460.6 | 256x256x64 |
| 8192x9216x6144 | bfloat16 | 1152 | one-tile owner, n = 2 | 1481.4 | 1426.9 | asm | 1382.4 | 1201.5 | 256x240x64 |
| 16384x2048x6144 | float32 | 512 | data-parallel | 1354.0 | 1361.0 | opus | 1269.3 | 1264.7 | 256x256x64 |
| 16384x2304x6144 | bfloat16 | 576 | one-tile owner, n = 3 | 1359.8 | 1226.6 | flydsl | 970.2 | 1284.9 | 192x256x64 |
| 16384x2560x6144 | bfloat16 | 640 | one-tile owner, n = 2 | 1413.9 | 1342.6 | flydsl | 1296.0 | 1296.4 | 256x224x64 |
| 16384x3072x6144 | float32 | 768 | data-parallel | 1397.4 | 1391.2 | asm | 1353.6 | 1310.6 | 256x256x64 |
| 16384x6144x6144 | bfloat16 | 1536 | data-parallel | 1526.8 | 1513.0 | asm | 1475.6 | 1491.4 | 256x256x64 |
| 16384x9216x6144 | bfloat16 | 2304 | data-parallel | 1535.9 | 1524.7 | torch | 1484.3 | 1509.8 | 256x256x64 |
| 32768x2304x6144 | bfloat16 | 1152 | one-tile owner, n = 2 | 1459.9 | 1399.6 | flydsl | 649.9 | 1188.1 | 256x240x64 |
| 32768x2560x6144 | bfloat16 | 1280 | data-parallel | 1494.8 | 1487.6 | flydsl | 1434.6 | 1482.7 | 256x256x64 |
| 32768x6144x6144 | bfloat16 | 3072 | data-parallel | 1547.5 | 1537.1 | torch | 1495.0 | 1524.1 | 256x256x64 |
| 32768x9216x6144 | bfloat16 | 4608 | data-parallel | 1549.8 | 1537.3 | torch | 1504.3 | 1531.6 | 256x256x64 |
| 512x5120x6400 | bfloat16 | 40 | one-tile reduce-scatter, n = 4 | 624.2 | 279.3 | flydsl | 542.8 | 664.7 | 160x128x128 |
| 1024x5120x6400 | bfloat16 | 80 | one-tile owner, n = 3-4 | 911.5 | 557.6 | flydsl | 591.3 | 890.0 | 160x128x128 |
| 2048x5120x6400 | bfloat16 | 160 | data-parallel | 1085.4 | 1087.1 | flydsl | 729.6 | 1189.2 | 160x256x64 |
| 4096x5120x6400 | bfloat16 | 320 | one-tile owner, n = 3 | 1319.2 | 1106.1 | torch | 1207.2 | 1212.9 | 160x256x64 |
| 8192x5120x6400 | bfloat16 | 640 | one-tile owner, n = 2 | 1445.7 | 1366.2 | flydsl | 1327.6 | 1307.8 | 256x224x64 |
| 16384x5120x6400 | bfloat16 | 1280 | data-parallel | 1524.6 | 1509.3 | torch | 1478.6 | 1489.5 | 256x256x64 |
| 32768x5120x6400 | bfloat16 | 2560 | data-parallel | 1541.9 | 1527.2 | torch | 1494.2 | 1520.1 | 256x256x64 |
| 512x16384x6656 | bfloat16 | 128 | one-tile owner, n = 2 | 1138.9 | 870.5 | flydsl | 839.7 | 900.8 | 256x256x64 |
| 1024x16384x6656 | bfloat16 | 256 | data-parallel | 1435.8 | 1440.3 | asm | 361.7 | 1392.4 | 256x256x64 |
| 2048x16384x6656 | bfloat16 | 512 | data-parallel | 1500.0 | 1490.8 | asm | 1433.0 | 1440.4 | 256x256x64 |
| 4096x16384x6656 | bfloat16 | 1024 | data-parallel | 1530.7 | 1524.2 | torch | 1458.9 | 1467.2 | 256x256x64 |
| 8192x16384x6656 | bfloat16 | 2048 | data-parallel | 1555.4 | 1534.4 | asm | 1496.0 | 1497.5 | 256x256x64 |
| 16384x16384x6656 | bfloat16 | 4096 | data-parallel | 1555.9 | 1545.2 | asm | 1509.8 | 1519.4 | 256x256x64 |
| 32768x16384x6656 | bfloat16 | 8192 | data-parallel | 1547.2 | 1545.4 | asm | 290.7 | 1512.6 | 256x256x64 |
| 512x1024x7168 | bfloat16 | 8 | one-tile reduce-scatter, n = 24 | 228.5 | 56.7 | torch | 382.3 | 383.3 | 128x128x64 |
| 512x1536x7168 | bfloat16 | 12 | one-tile reduce-scatter, n = 16 | 324.0 | 85.7 | torch | 408.3 | 410.1 | 192x64x128 |
| 512x2048x7168 | bfloat16 | 16 | one-tile reduce-scatter, n = 12 | 399.6 | 113.5 | torch | 537.0 | 537.6 | 128x128x128 |
| 512x3584x7168 | bfloat16 | 28 | one-tile reduce-scatter, n = 6 | 551.1 | 201.7 | flydsl | 557.6 | 529.3 | 128x128x256 |
| 512x6144x7168 | bfloat16 | 48 | one-tile reduce-scatter, n = 4 | 757.9 | 346.5 | flydsl | 718.5 | 706.5 | 192x128x128 |
| 512x8192x7168 | bfloat16 | 64 | one-tile reduce-scatter, n = 3 | 821.1 | 458.2 | flydsl | 703.3 | 714.1 | 128x128x256 |
| 512x8448x7168 | bfloat16 | 66 | one-tile owner, n = 3-4 | 784.9 | 471.6 | flydsl | 666.6 | 468.8 | 256x256x64 |
| 512x20480x7168 | bfloat16 | 160 | spread | 1101.4 | 1054.5 | flydsl | 1021.7 | 1144.5 | 160x256x64 |
| 768x1024x7168 | bfloat16 | 12 | one-tile reduce-scatter, n = 16 | 329.0 | 86.0 | torch | 403.1 | 403.9 | 256x96x64 |
| 768x2048x7168 | bfloat16 | 24 | one-tile reduce-scatter, n = 8 | 539.4 | 172.5 | flydsl | 556.6 | 569.6 | 256x128x64 |
| 1024x512x7168 | bfloat16 | 8 | one-tile reduce-scatter, n = 24 | 230.2 | 56.9 | flydsl | 336.5 | 288.8 | 256x64x64 |
| 1024x1024x7168 | bfloat16 | 16 | one-tile reduce-scatter, n = 12 | 401.0 | 115.8 | torch | 544.3 | 547.3 | 128x128x128 |
| 1024x1536x7168 | bfloat16 | 24 | one-tile reduce-scatter, n = 8 | 536.7 | 172.6 | flydsl | 561.6 | 571.6 | 256x128x64 |
| 1024x2048x7168 | bfloat16 | 32 | one-tile reduce-scatter, n = 6 | 615.9 | 230.1 | torch | 723.6 | 724.7 | 256x128x64 |
| 1024x3584x7168 | bfloat16 | 56 | one-tile reduce-scatter, n = 3 | 754.5 | 404.0 | flydsl | 773.6 | 678.6 | 128x128x256 |
| 1024x6144x7168 | bfloat16 | 96 | one-tile owner, n = 2-3 | 876.2 | 686.6 | torch | 928.6 | 972.0 | 192x256x64 |
| 1024x8192x7168 | bfloat16 | 128 | one-tile owner, n = 2 | 1157.4 | 894.3 | flydsl | 1007.9 | 884.9 | 256x256x64 |
| 1024x8448x7168 | bfloat16 | 132 | spread | 1101.6 | 919.3 | torch | 1032.2 | 1038.5 | 144x256x64 |
| 1024x20480x7168 | bfloat16 | 320 | one-tile owner, n = 3 | 1291.4 | 1066.7 | flydsl | 745.3 | 1185.1 | 160x256x64 |
| 1280x512x7168 | bfloat16 | 10 | one-tile reduce-scatter, n = 19 | 294.9 | 71.7 | torch | 362.0 | 362.2 | 256x80x64 |
| 1280x1024x7168 | bfloat16 | 20 | one-tile reduce-scatter, n = 9 | 448.3 | 144.8 | torch | 534.5 | 536.5 | 256x80x128 |
| 1280x2048x7168 | bfloat16 | 40 | one-tile reduce-scatter, n = 4 | 670.0 | 290.0 | torch | 741.5 | 726.5 | 128x160x128 |
| 1536x512x7168 | bfloat16 | 12 | one-tile reduce-scatter, n = 16 | 327.0 | 86.8 | torch | 401.9 | 402.4 | 256x96x64 |
| 1536x1024x7168 | bfloat16 | 24 | one-tile reduce-scatter, n = 8 | 538.5 | 174.3 | torch | 642.4 | 643.8 | 128x96x128 |
| 1536x2048x7168 | bfloat16 | 48 | one-tile reduce-scatter, n = 4 | 769.8 | 350.7 | torch | 838.7 | 839.5 | 128x192x64 |
| 1792x512x7168 | bfloat16 | 14 | one-tile reduce-scatter, n = 13 | 350.0 | 100.9 | torch | 437.4 | 437.0 | 256x112x64 |
| 1792x1024x7168 | bfloat16 | 28 | one-tile reduce-scatter, n = 6 | 564.0 | 205.6 | flydsl | 603.5 | 603.9 | 256x128x64 |
| 1792x2048x7168 | bfloat16 | 56 | one-tile reduce-scatter, n = 3 | 765.0 | 409.3 | flydsl | 778.4 | 687.4 | 128x128x256 |
| 2048x512x7168 | bfloat16 | 16 | one-tile reduce-scatter, n = 12 | 402.3 | 116.0 | torch | 504.7 | 504.5 | 128x128x64 |
| 2048x1024x7168 | bfloat16 | 32 | one-tile reduce-scatter, n = 6 | 621.8 | 235.6 | torch | 717.7 | 718.8 | 128x128x128 |
| 2048x2048x7168 | bfloat16 | 64 | one-tile reduce-scatter, n = 3 | 855.5 | 469.9 | flydsl | 873.1 | 767.0 | 128x128x256 |
| 2048x3584x7168 | bfloat16 | 112 | one-tile owner, n = 2-3 | 1055.1 | 804.3 | opus | 1002.2 | 987.8 | 224x256x64 |
| 2048x8192x7168 | bfloat16 | 256 | data-parallel | 1455.1 | 1452.3 | asm | 1371.4 | 1410.5 | 256x256x64 |
| 2048x20480x7168 | bfloat16 | 640 | one-tile owner, n = 2 | 1447.2 | 1375.8 | flydsl | 827.3 | 1255.8 | 224x256x64 |
| 4096x512x7168 | bfloat16 | 32 | one-tile reduce-scatter, n = 6 | 612.5 | 231.3 | flydsl | 625.7 | 591.3 | 128x128x256 |
| 4096x1024x7168 | bfloat16 | 64 | one-tile reduce-scatter, n = 3 | 847.4 | 466.9 | torch | 902.4 | 905.3 | 256x128x64 |
| 4096x2048x7168 | bfloat16 | 128 | one-tile owner, n = 2 | 1168.4 | 904.1 | torch | 1163.9 | 1170.0 | 256x128x64 |
| 4096x3584x7168 | bfloat16 | 224 | data-parallel | 1382.4 | 1388.0 | flydsl | 1319.0 | 1277.7 | 224x256x64 |
| 4096x8192x7168 | bfloat16 | 512 | data-parallel | 1488.8 | 1486.8 | flydsl | 1414.7 | 1445.2 | 256x256x64 |
| 4096x20480x7168 | bfloat16 | 1280 | data-parallel | 1543.8 | 1513.7 | flydsl | 1445.4 | 1451.4 | 256x256x64 |
| 6144x512x7168 | bfloat16 | 48 | one-tile reduce-scatter, n = 4 | 745.2 | 345.0 | flydsl | 723.7 | 654.6 | 128x192x128 |
| 6144x1024x7168 | bfloat16 | 96 | one-tile owner, n = 2-3 | 922.8 | 683.9 | opus | 969.6 | 835.1 | 256x192x64 |
| 6144x2048x7168 | bfloat16 | 192 | data-parallel | 1258.4 | 1268.9 | opus | 1302.5 | 1267.9 | 256x192x64 |
| 8192x512x7168 | bfloat16 | 64 | one-tile reduce-scatter, n = 3 | 822.9 | 457.5 | flydsl | 838.7 | 751.1 | 128x128x256 |
| 8192x1024x7168 | bfloat16 | 128 | one-tile owner, n = 2 | 1162.0 | 890.6 | torch | 1090.9 | 1096.0 | 256x128x64 |
| 8192x2048x7168 | bfloat16 | 256 | data-parallel | 1450.1 | 1455.3 | torch | 1381.4 | 1390.9 | 256x256x64 |
| 8192x3584x7168 | bfloat16 | 448 | data-parallel | 1405.2 | 1400.7 | flydsl | 1351.1 | 1307.9 | 224x256x64 |
| 8192x8192x7168 | bfloat16 | 1024 | data-parallel | 1511.1 | 1503.9 | flydsl | 1431.6 | 1460.5 | 256x256x64 |
| 8192x20480x7168 | bfloat16 | 2560 | data-parallel | 1545.2 | 1535.2 | asm | 1503.8 | 1467.1 | 256x256x64 |
| 10240x512x7168 | bfloat16 | 80 | one-tile owner, n = 3-4 | 936.5 | 561.5 | torch | 828.2 | 812.1 | 256x160x64 |
| 10240x1024x7168 | bfloat16 | 160 | spread | 1153.5 | 1100.7 | torch | 1204.9 | 1214.2 | 256x160x64 |
| 10240x2048x7168 | bfloat16 | 320 | one-tile owner, n = 3 | 1323.2 | 1098.1 | torch | 1239.5 | 1240.6 | 256x160x64 |
| 10240x8448x7168 | bfloat16 | 1320 | one-tile owner, n = 4 | 1465.2 | 1348.6 | asm | 1301.9 | 1336.7 | 256x224x64 |
| 10240x20480x7168 | bfloat16 | 3200 | one-tile owner, n = 2 | 1497.3 | 1457.7 | asm | 1446.7 | 1439.7 | 256x256x64 |
| 12288x512x7168 | bfloat16 | 96 | one-tile owner, n = 2-3 | 904.9 | 667.4 | opus | 929.3 | 873.4 | 256x192x64 |
| 12288x1024x7168 | bfloat16 | 192 | data-parallel | 1230.9 | 1231.5 | opus | 1316.7 | 1256.8 | 256x192x64 |
| 12288x2048x7168 | bfloat16 | 384 | one-tile owner, n = 2 | 1404.0 | 1285.6 | opus | 1336.2 | 1292.6 | 256x192x64 |
| 14336x512x7168 | bfloat16 | 112 | one-tile owner, n = 2-3 | 1058.3 | 769.7 | opus | 994.2 | 894.1 | 256x224x64 |
| 14336x1024x7168 | bfloat16 | 224 | data-parallel | 1330.0 | 1336.2 | torch | 1298.9 | 1307.2 | 256x224x64 |
| 14336x2048x7168 | bfloat16 | 448 | data-parallel | 1383.6 | 1383.4 | flydsl | 1345.0 | 1340.1 | 256x224x64 |
| 16384x512x7168 | bfloat16 | 128 | one-tile owner, n = 2 | 1146.6 | 873.8 | opus | 1073.5 | 1011.4 | 256x256x64 |
| 16384x1024x7168 | bfloat16 | 256 | data-parallel | 1460.6 | 1457.3 | torch | 1412.7 | 1419.4 | 256x256x64 |
| 16384x2048x7168 | bfloat16 | 512 | data-parallel | 1497.8 | 1494.1 | torch | 1461.2 | 1471.0 | 256x256x64 |
| 16384x3584x7168 | bfloat16 | 896 | one-tile owner, n = 2 | 1487.2 | 1402.8 | flydsl | 1361.5 | 1335.2 | 224x256x64 |
| 16384x8192x7168 | bfloat16 | 2048 | data-parallel | 1557.9 | 1543.5 | torch | 1487.3 | 1529.0 | 256x256x64 |
| 16384x20480x7168 | bfloat16 | 5120 | data-parallel | 1579.3 | 1560.3 | flydsl | 1513.8 | 1475.1 | 256x256x64 |
| 32768x512x7168 | bfloat16 | 256 | data-parallel | 1363.2 | 1363.9 | torch | 1363.1 | 1373.1 | 256x256x64 |
| 32768x1024x7168 | bfloat16 | 512 | data-parallel | 1478.6 | 1474.4 | torch | 1426.9 | 1432.0 | 256x256x64 |
| 32768x2048x7168 | bfloat16 | 1024 | data-parallel | 1521.5 | 1504.3 | torch | 1473.9 | 1489.1 | 256x256x64 |
| 32768x3584x7168 | bfloat16 | 1792 | data-parallel | 1528.7 | 1505.3 | torch | 1494.8 | 1509.1 | 256x256x64 |
| 32768x8192x7168 | bfloat16 | 4096 | data-parallel | 1555.1 | 1545.6 | torch | 1515.2 | 1541.4 | 256x256x64 |
| 32768x20480x7168 | bfloat16 | 10240 | data-parallel | 1543.7 | 1546.4 | asm | 1506.5 | 1431.0 | 256x256x64 |
| 512x1280x8192 | bfloat16 | 10 | one-tile reduce-scatter, n = 19 | 315.7 | 77.5 | torch | 418.9 | 418.8 | 160x64x128 |
| 512x2560x8192 | bfloat16 | 20 | one-tile reduce-scatter, n = 9 | 524.7 | 155.2 | torch | 553.7 | 550.8 | 256x128x64 |
| 512x4096x8192 | bfloat16 | 32 | one-tile reduce-scatter, n = 6 | 690.7 | 246.2 | flydsl | 673.3 | 648.6 | 128x128x256 |
| 512x4608x8192 | bfloat16 | 36 | one-tile reduce-scatter, n = 5 | 719.6 | 278.1 | torch | 721.2 | 723.0 | 160x128x128 |
| 512x7168x8192 | bfloat16 | 56 | one-tile reduce-scatter, n = 3 | 805.1 | 430.7 | flydsl | 800.3 | 742.4 | 256x176x64 |
| 512x8192x8192 | bfloat16 | 64 | one-tile reduce-scatter, n = 3 | 880.5 | 476.5 | flydsl | 682.8 | 767.9 | 128x128x256 |
| 512x10240x8192 | bfloat16 | 80 | one-tile owner, n = 3-4 | 1005.4 | 599.1 | flydsl | 609.4 | 572.3 | 256x256x64 |
| 512x14336x8192 | bfloat16 | 112 | one-tile owner, n = 2-3 | 1098.0 | 821.6 | torch | 995.7 | 1009.2 | 224x256x64 |
| 512x16384x8192 | bfloat16 | 128 | one-tile owner, n = 2 | 1177.0 | 890.7 | flydsl | 650.9 | 920.3 | 256x256x64 |
| 512x57344x8192 | bfloat16 | 448 | data-parallel | 1334.8 | 1327.7 | flydsl | 1271.2 | 1252.0 | 224x256x64 |
| 768x4096x8192 | bfloat16 | 48 | one-tile reduce-scatter, n = 4 | 823.5 | 371.1 | flydsl | 739.7 | 709.4 | 128x192x128 |
| 1024x512x8192 | bfloat16 | 8 | one-tile reduce-scatter, n = 24 | 279.0 | 62.2 | torch | 330.9 | 329.8 | 256x64x64 |
| 1024x1280x8192 | bfloat16 | 20 | one-tile reduce-scatter, n = 9 | 522.8 | 156.4 | torch | 572.2 | 572.6 | 256x128x64 |
| 1024x2560x8192 | bfloat16 | 40 | one-tile reduce-scatter, n = 4 | 735.1 | 311.5 | torch | 781.0 | 774.6 | 160x128x128 |
| 1024x4096x8192 | bfloat16 | 64 | one-tile reduce-scatter, n = 3 | 889.5 | 489.8 | flydsl | 900.2 | 798.6 | 128x128x256 |
| 1024x4608x8192 | bfloat16 | 72 | one-tile owner, n = 3-4 | 926.5 | 548.2 | torch | 924.6 | 927.3 | 160x256x64 |
| 1024x7168x8192 | bfloat16 | 112 | one-tile owner, n = 2-3 | 1135.9 | 825.2 | flydsl | 732.9 | 1025.8 | 224x256x64 |
| 1024x8192x8192 | bfloat16 | 128 | one-tile owner, n = 2 | 1205.7 | 918.3 | flydsl | 909.7 | 920.8 | 256x256x64 |
| 1024x10240x8192 | bfloat16 | 160 | spread | 1185.5 | 1117.2 | flydsl | 819.4 | 1184.5 | 160x256x64 |
| 1024x14336x8192 | bfloat16 | 224 | data-parallel | 1367.3 | 1370.9 | flydsl | 1317.4 | 1281.2 | 224x256x64 |
| 1024x16384x8192 | bfloat16 | 256 | data-parallel | 1467.1 | 1472.8 | torch | 1426.9 | 1442.9 | 256x256x64 |
| 1024x57344x8192 | bfloat16 | 896 | one-tile owner, n = 2 | 1478.5 | 1400.7 | asm | 1357.4 | 1310.6 | 224x256x64 |
| 2048x512x8192 | bfloat16 | 16 | one-tile reduce-scatter, n = 12 | 446.9 | 125.6 | torch | 499.0 | 498.4 | 256x128x64 |
| 2048x1280x8192 | bfloat16 | 40 | one-tile reduce-scatter, n = 4 | 737.3 | 315.0 | torch | 694.6 | 694.6 | 320x192x64 |
| 2048x2560x8192 | bfloat16 | 80 | one-tile owner, n = 3-4 | 1054.5 | 612.3 | torch | 952.1 | 948.7 | 160x256x64 |
| 2048x4096x8192 | bfloat16 | 128 | one-tile owner, n = 2 | 1233.8 | 935.3 | opus | 1121.6 | 946.9 | 256x256x64 |
| 2048x4608x8192 | bfloat16 | 144 | spread | 1182.9 | 1034.2 | torch | 1143.8 | 1152.8 | 144x256x64 |
| 2048x7168x8192 | bfloat16 | 224 | data-parallel | 1402.6 | 1402.3 | asm | 1347.3 | 1295.1 | 224x256x64 |
| 2048x8192x8192 | bfloat16 | 256 | data-parallel | 1494.4 | 1505.4 | torch | 1438.8 | 1449.8 | 256x256x64 |
| 2048x10240x8192 | bfloat16 | 320 | one-tile owner, n = 3 | 1361.8 | 1124.8 | flydsl | 925.0 | 1214.1 | 160x256x64 |
| 2048x14336x8192 | bfloat16 | 448 | data-parallel | 1418.1 | 1419.2 | flydsl | 1370.3 | 1309.9 | 224x256x64 |
| 2048x16384x8192 | bfloat16 | 512 | data-parallel | 1520.0 | 1513.2 | asm | 1436.8 | 1452.5 | 256x256x64 |
| 2048x57344x8192 | bfloat16 | 1792 | data-parallel | 1558.6 | 1552.3 | flydsl | 1498.7 | 1456.5 | 256x256x64 |
| 4096x512x8192 | bfloat16 | 32 | one-tile reduce-scatter, n = 6 | 680.0 | 248.3 | flydsl | 647.6 | 666.4 | 128x128x256 |
| 4096x1280x8192 | bfloat16 | 80 | one-tile owner, n = 3-4 | 1006.3 | 603.5 | torch | 938.8 | 946.7 | 160x256x64 |
| 4096x2560x8192 | bfloat16 | 160 | spread | 1219.5 | 1132.8 | torch | 1203.0 | 1206.6 | 160x256x64 |
| 4096x4096x8192 | bfloat16 | 256 | data-parallel | 1493.2 | 1497.2 | torch | 1440.2 | 1447.2 | 256x256x64 |
| 4096x4608x8192 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 | 1339.1 | 1020.4 | torch | 1166.2 | 1171.8 | 144x256x64 |
| 4096x7168x8192 | bfloat16 | 448 | data-parallel | 1431.0 | 1431.6 | flydsl | 1373.5 | 1316.2 | 224x256x64 |
| 4096x8192x8192 | bfloat16 | 512 | data-parallel | 1509.8 | 1513.0 | asm | 1449.1 | 1462.0 | 256x256x64 |
| 4096x10240x8192 | bfloat16 | 640 | one-tile owner, n = 2 | 1479.2 | 1394.4 | flydsl | 1350.6 | 1299.3 | 224x256x64 |
| 4096x14336x8192 | bfloat16 | 896 | one-tile owner, n = 2 | 1513.8 | 1433.8 | flydsl | 1384.5 | 1336.2 | 224x256x64 |
| 4096x16384x8192 | bfloat16 | 1024 | data-parallel | 1547.9 | 1536.9 | flydsl | 1463.0 | 1484.1 | 256x256x64 |
| 4096x57344x8192 | bfloat16 | 3584 | data-parallel | 1576.3 | 1573.7 | flydsl | 1514.6 | 1470.8 | 256x256x64 |
| 8192x512x8192 | bfloat16 | 64 | one-tile reduce-scatter, n = 3 | 881.0 | 477.2 | flydsl | 865.6 | 799.9 | 128x128x256 |
| 8192x1280x8192 | bfloat16 | 160 | spread | 1192.8 | 1108.5 | torch | 1198.1 | 1208.2 | 160x256x64 |
| 8192x2560x8192 | bfloat16 | 320 | one-tile owner, n = 3 | 1359.6 | 1118.0 | torch | 1222.1 | 1230.7 | 160x256x64 |
| 8192x4096x8192 | bfloat16 | 512 | data-parallel | 1505.0 | 1507.6 | torch | 1463.1 | 1469.7 | 256x256x64 |
| 8192x4608x8192 | bfloat16 | 576 | one-tile owner, n = 3 | 1442.4 | 1269.7 | torch | 1304.9 | 1304.1 | 192x256x64 |
| 8192x7168x8192 | bfloat16 | 896 | one-tile owner, n = 2 | 1506.3 | 1430.7 | asm | 1392.1 | 1349.2 | 224x256x64 |
| 8192x8192x8192 | bfloat16 | 1024 | data-parallel | 1547.9 | 1537.9 | asm | 1485.5 | 1494.1 | 256x256x64 |
| 8192x10240x8192 | bfloat16 | 1280 | data-parallel | 1563.5 | 1540.5 | flydsl | 1465.9 | 1519.5 | 256x256x64 |
| 8192x14336x8192 | bfloat16 | 1792 | data-parallel | 1573.2 | 1551.6 | flydsl | 1502.2 | 1509.7 | 256x256x64 |
| 8192x16384x8192 | bfloat16 | 2048 | data-parallel | 1555.0 | 1560.4 | asm | 1513.5 | 1496.4 | 256x256x64 |
| 8192x57344x8192 | bfloat16 | 7168 | data-parallel | 1542.6 | 1561.6 | asm | 1501.4 | 1441.0 | 256x256x64 |
| 16384x512x8192 | bfloat16 | 128 | one-tile owner, n = 2 | 1172.5 | 900.5 | torch | 1044.4 | 1049.0 | 256x256x64 |
| 16384x1280x8192 | bfloat16 | 320 | one-tile owner, n = 3 | 1339.5 | 1113.2 | torch | 1229.6 | 1237.7 | 160x256x64 |
| 16384x2560x8192 | bfloat16 | 640 | one-tile owner, n = 2 | 1465.0 | 1365.3 | flydsl | 1330.3 | 1289.9 | 256x224x64 |
| 16384x4608x8192 | bfloat16 | 1152 | one-tile owner, n = 2 | 1509.5 | 1444.4 | flydsl | 1396.8 | 1195.7 | 256x240x64 |
| 16384x7168x8192 | bfloat16 | 1792 | data-parallel | 1558.9 | 1542.7 | torch | 1503.9 | 1533.2 | 256x256x64 |
| 16384x8192x8192 | bfloat16 | 2048 | data-parallel | 1549.3 | 1554.6 | torch | 1500.3 | 1542.0 | 256x256x64 |
| 16384x10240x8192 | bfloat16 | 2560 | data-parallel | 1575.4 | 1567.4 | torch | 1548.3 | 1535.5 | 256x256x64 |
| 16384x14336x8192 | bfloat16 | 3584 | data-parallel | 1576.5 | 1574.5 | asm | 1527.5 | 1531.9 | 256x256x64 |
| 16384x16384x8192 | bfloat16 | 4096 | data-parallel | 1581.8 | 1567.4 | asm | 1531.2 | 1507.3 | 256x256x64 |
| 16384x57344x8192 | bfloat16 | 14336 | data-parallel | 1559.8 | 1576.5 | asm | 1495.2 | 1413.7 | 256x256x64 |
| 32768x1280x8192 | bfloat16 | 640 | one-tile owner, n = 2 | 1443.9 | 1339.4 | flydsl | 1308.6 | 1316.2 | 256x224x64 |
| 32768x2560x8192 | bfloat16 | 1280 | data-parallel | 1521.0 | 1516.7 | torch | 1497.4 | 1509.3 | 256x256x64 |
| 32768x7168x8192 | bfloat16 | 3584 | data-parallel | 1545.4 | 1547.6 | torch | 1523.4 | 1545.4 | 256x256x64 |
| 32768x8192x8192 | bfloat16 | 4096 | data-parallel | 1573.9 | 1575.1 | flydsl | 1524.4 | 1539.2 | 256x256x64 |
| 32768x10240x8192 | bfloat16 | 5120 | data-parallel | 1552.1 | 1559.9 | torch | 1516.8 | 1549.9 | 256x256x64 |
| 32768x14336x8192 | bfloat16 | 7168 | data-parallel | 1560.9 | 1559.3 | asm | 1499.6 | 1495.5 | 256x256x64 |
| 32768x16384x8192 | bfloat16 | 8192 | data-parallel | 1554.7 | 1553.0 | asm | 1511.6 | 1470.3 | 256x256x64 |
| 32768x57344x8192 | bfloat16 | 28672 | data-parallel | 1516.0 | 1553.1 | asm | 1486.7 | 1407.6 | 256x256x64 |
| 512x6144x12288 | bfloat16 | 48 | one-tile reduce-scatter, n = 4 | 899.2 | 348.7 | torch | 785.1 | 787.2 | 192x128x128 |
| 1024x6144x12288 | bfloat16 | 96 | one-tile owner, n = 2-3 | 1057.0 | 695.9 | torch | 1076.4 | 1098.1 | 192x256x64 |
| 2048x6144x12288 | bfloat16 | 192 | spread | 1378.1 | 1296.4 | torch | 1291.3 | 1300.9 | 192x256x64 |
| 4096x6144x12288 | bfloat16 | 384 | one-tile owner, n = 2 | 1492.9 | 1337.8 | torch | 1300.7 | 1307.1 | 192x256x64 |
| 8192x6144x12288 | bfloat16 | 768 | data-parallel | 1543.4 | 1532.1 | torch | 1501.0 | 1515.8 | 256x256x64 |
| 16384x6144x12288 | bfloat16 | 1536 | data-parallel | 1552.4 | 1549.2 | torch | 1521.6 | 1532.4 | 256x256x64 |
| 32768x6144x12288 | bfloat16 | 3072 | data-parallel | 1559.6 | 1557.4 | asm | 1520.2 | 1541.2 | 256x256x64 |
| 512x16384x13312 | bfloat16 | 128 | one-tile owner, n = 2 | 1270.3 | 919.1 | flydsl | 703.5 | 1083.1 | 256x256x64 |
| 1024x16384x13312 | bfloat16 | 256 | data-parallel | 1509.7 | 1508.8 | asm | 1461.9 | 1472.7 | 256x256x64 |
| 2048x16384x13312 | bfloat16 | 512 | data-parallel | 1557.3 | 1556.0 | flydsl | 1003.4 | 1485.5 | 256x256x64 |
| 4096x16384x13312 | bfloat16 | 1024 | data-parallel | 1584.8 | 1582.1 | flydsl | 1523.9 | 1500.3 | 256x256x64 |
| 8192x16384x13312 | bfloat16 | 2048 | data-parallel | 1568.4 | 1564.9 | flydsl | 1498.7 | 1486.2 | 256x256x64 |
| 16384x16384x13312 | bfloat16 | 4096 | data-parallel | 1568.3 | 1568.6 | asm | 1503.9 | 1469.1 | 256x256x64 |
| 32768x16384x13312 | bfloat16 | 8192 | data-parallel | 1553.3 | 1566.4 | flydsl | 1485.9 | 1446.7 | 256x256x64 |
| 512x2304x16384 | bfloat16 | 18 | one-tile reduce-scatter, n = 10 | 682.0 | 139.2 | torch | 705.0 | 706.2 | 256x128x64 |
| 512x4608x16384 | bfloat16 | 36 | one-tile reduce-scatter, n = 5 | 863.3 | 275.2 | flydsl | 507.6 | 794.2 | 160x128x128 |
| 512x9216x16384 | bfloat16 | 72 | one-tile owner, n = 3-4 | 1081.0 | 540.0 | flydsl | 618.3 | 758.1 | 256x256x64 |
| 512x13312x16384 | bfloat16 | 104 | one-tile owner, n = 2-3 | 1140.1 | 709.6 | flydsl | 540.5 | 1075.3 | 208x256x64 |
| 512x26624x16384 | bfloat16 | 208 | spread | 1263.1 | 1283.1 | flydsl | 766.2 | 1166.2 | 208x256x64 |
| 512x53248x16384 | bfloat16 | 416 | spread | 1331.0 | 1294.3 | flydsl | 655.1 | 1183.9 | 208x256x64 |
| 1024x2304x16384 | bfloat16 | 36 | one-tile reduce-scatter, n = 5 | 890.6 | 278.1 | torch | 868.4 | 871.6 | 256x256x64 |
| 1024x4608x16384 | bfloat16 | 72 | one-tile owner, n = 3-4 | 1130.7 | 548.0 | flydsl | 643.9 | 1018.3 | 144x256x64 |
| 1024x9216x16384 | bfloat16 | 144 | spread | 1209.1 | 1018.6 | flydsl | 664.4 | 1162.6 | 144x256x64 |
| 1024x13312x16384 | bfloat16 | 208 | spread | 1347.6 | 1346.9 | asm | 1313.2 | 1233.0 | 208x256x64 |
| 1024x26624x16384 | bfloat16 | 416 | spread | 1436.8 | 1361.2 | flydsl | 931.7 | 1239.8 | 208x256x64 |
| 1024x53248x16384 | bfloat16 | 832 | one-tile owner, n = 4 | 1501.1 | 1338.2 | triton | 990.4 | 1243.9 | 208x256x64 |
| 2048x2304x16384 | bfloat16 | 72 | one-tile owner, n = 3-4 | 1142.0 | 547.0 | torch | 853.0 | 869.2 | 256x256x64 |
| 2048x4608x16384 | bfloat16 | 144 | spread | 1247.4 | 1045.4 | flydsl | 688.0 | 1185.5 | 144x256x64 |
| 2048x9216x16384 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 | 1394.3 | 1037.4 | flydsl | 908.2 | 1159.1 | 144x256x64 |
| 2048x13312x16384 | bfloat16 | 416 | spread | 1463.2 | 1390.2 | flydsl | 1358.0 | 1227.6 | 208x256x64 |
| 2048x26624x16384 | bfloat16 | 832 | one-tile owner, n = 4 | 1541.0 | 1370.1 | flydsl | 906.3 | 1230.1 | 208x256x64 |
| 2048x53248x16384 | bfloat16 | 1664 | one-tile owner, n = 2 | 1530.8 | 1497.1 | flydsl | 788.3 | 1266.9 | 240x256x64 |
| 4096x2304x16384 | bfloat16 | 144 | spread | 1248.1 | 1028.0 | torch | 1179.2 | 1187.2 | 144x256x64 |
| 4096x4608x16384 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 | 1415.8 | 1034.6 | torch | 1171.4 | 1179.2 | 144x256x64 |
| 4096x9216x16384 | bfloat16 | 576 | one-tile owner, n = 4 | 1525.8 | 1297.0 | torch | 1316.3 | 1307.9 | 192x256x64 |
| 4096x13312x16384 | bfloat16 | 832 | one-tile owner, n = 4 | 1548.7 | 1365.1 | flydsl | 1315.2 | 1247.9 | 208x256x64 |
| 4096x26624x16384 | bfloat16 | 1664 | one-tile owner, n = 2 | 1569.1 | 1504.3 | flydsl | 1451.3 | 1317.6 | 240x256x64 |
| 4096x53248x16384 | bfloat16 | 3328 | data-parallel | 1564.4 | 1581.6 | flydsl | 1523.4 | 1465.0 | 256x256x64 |
| 8192x2304x16384 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 | 1361.9 | 1000.5 | torch | 1181.1 | 1185.5 | 144x256x64 |
| 8192x4608x16384 | bfloat16 | 576 | one-tile owner, n = 4 | 1494.3 | 1270.1 | torch | 1320.5 | 1327.3 | 192x256x64 |
| 8192x9216x16384 | bfloat16 | 1152 | one-tile owner, n = 2 | 1535.0 | 1446.3 | flydsl | 1392.0 | 1147.0 | 256x240x64 |
| 8192x13312x16384 | bfloat16 | 1664 | one-tile owner, n = 2 | 1544.3 | 1477.4 | flydsl | 1437.2 | 1282.7 | 240x256x64 |
| 8192x26624x16384 | bfloat16 | 3328 | data-parallel | 1541.3 | 1554.6 | flydsl | 1499.6 | 1463.5 | 256x256x64 |
| 8192x53248x16384 | bfloat16 | 6656 | data-parallel | 1526.1 | 1553.2 | flydsl | 1486.3 | 1433.2 | 256x256x64 |
| 16384x2304x16384 | bfloat16 | 576 | one-tile owner, n = 4 | 1477.7 | 1270.5 | torch | 1332.3 | 1320.2 | 192x256x64 |
| 16384x4608x16384 | bfloat16 | 1152 | one-tile owner, n = 2 | 1523.3 | 1444.2 | flydsl | 1407.4 | 1168.3 | 256x240x64 |
| 16384x9216x16384 | bfloat16 | 2304 | data-parallel | 1544.8 | 1550.5 | flydsl | 1489.8 | 1479.7 | 256x256x64 |
| 16384x13312x16384 | bfloat16 | 3328 | data-parallel | 1529.0 | 1543.3 | asm | 1476.2 | 1458.0 | 256x256x64 |
| 16384x26624x16384 | bfloat16 | 6656 | data-parallel | 1521.5 | 1532.9 | flydsl | 1482.9 | 1441.0 | 256x256x64 |
| 16384x53248x16384 | bfloat16 | 13312 | data-parallel | 1494.4 | 1533.0 | asm | 1462.6 | 1436.1 | 256x256x64 |
| 32768x2304x16384 | bfloat16 | 1152 | one-tile owner, n = 2 | 1495.5 | 1429.8 | flydsl | 1391.7 | 1175.4 | 256x240x64 |
| 32768x4608x16384 | bfloat16 | 2304 | data-parallel | 1534.8 | 1537.3 | torch | 1520.4 | 1526.0 | 256x256x64 |
| 32768x9216x16384 | bfloat16 | 4608 | data-parallel | 1517.6 | 1537.0 | asm | 1454.7 | 1455.2 | 256x256x64 |
| 32768x13312x16384 | bfloat16 | 6656 | data-parallel | 1507.5 | 1538.0 | asm | 1468.6 | 1435.6 | 256x256x64 |
| 32768x26624x16384 | bfloat16 | 13312 | data-parallel | 1488.3 | 1534.4 | flydsl | 1452.6 | 1420.7 | 256x256x64 |
| 32768x53248x16384 | bfloat16 | 26624 | data-parallel | 1480.6 | 1505.3 | flydsl | 1429.5 | 1442.6 | 256x256x64 |
| 512x6144x18432 | bfloat16 | 48 | one-tile reduce-scatter, n = 4 | 991.4 | 353.1 | flydsl | 785.1 | 835.9 | 192x128x128 |
| 1024x6144x18432 | bfloat16 | 96 | one-tile owner, n = 2-3 | 1131.0 | 707.6 | torch | 1153.3 | 1152.6 | 192x256x64 |
| 2048x6144x18432 | bfloat16 | 192 | spread | 1432.1 | 1322.7 | torch | 1310.8 | 1322.0 | 192x256x64 |
| 4096x6144x18432 | bfloat16 | 384 | one-tile owner, n = 2 | 1537.4 | 1347.5 | torch | 1326.5 | 1319.0 | 192x256x64 |
| 8192x6144x18432 | bfloat16 | 768 | data-parallel | 1569.2 | 1567.9 | torch | 1516.9 | 1522.5 | 256x256x64 |
| 16384x6144x18432 | bfloat16 | 1536 | data-parallel | 1579.0 | 1575.0 | asm | 1516.8 | 1526.3 | 256x256x64 |
| 32768x6144x18432 | bfloat16 | 3072 | data-parallel | 1581.6 | 1575.1 | asm | 1514.2 | 1495.7 | 256x256x64 |
| 512x5120x25600 | bfloat16 | 40 | one-tile reduce-scatter, n = 4 | 893.1 | 294.8 | flydsl | 428.2 | 873.6 | 160x128x128 |
| 1024x5120x25600 | bfloat16 | 80 | one-tile owner, n = 3-4 | 1266.8 | 590.5 | torch | 1128.9 | 1137.5 | 160x256x64 |
| 2048x5120x25600 | bfloat16 | 160 | spread | 1386.0 | 1160.9 | torch | 1262.5 | 1269.5 | 160x256x64 |
| 4096x5120x25600 | bfloat16 | 320 | one-tile owner, n = 4 | 1521.2 | 1137.8 | torch | 1262.6 | 1257.3 | 160x256x64 |
| 8192x5120x25600 | bfloat16 | 640 | one-tile owner, n = 2 | 1550.5 | 1397.4 | asm | 1384.8 | 1322.3 | 256x224x64 |
| 16384x5120x25600 | bfloat16 | 1280 | data-parallel | 1574.0 | 1568.6 | torch | 1524.4 | 1530.1 | 256x256x64 |
| 32768x5120x25600 | bfloat16 | 2560 | data-parallel | 1570.6 | 1565.0 | torch | 1503.0 | 1514.6 | 256x256x64 |
| 512x16384x26624 | bfloat16 | 128 | one-tile owner, n = 2 | 1365.6 | 936.3 | flydsl | 650.0 | 1173.6 | 256x256x64 |
| 1024x16384x26624 | bfloat16 | 256 | data-parallel | 1538.7 | 1542.5 | flydsl | 756.0 | 1493.1 | 256x256x64 |
| 2048x16384x26624 | bfloat16 | 512 | data-parallel | 1589.3 | 1569.5 | flydsl | 823.7 | 1500.8 | 256x256x64 |
| 4096x16384x26624 | bfloat16 | 1024 | data-parallel | 1605.7 | 1602.3 | flydsl | 1539.2 | 1503.6 | 256x256x64 |
| 8192x16384x26624 | bfloat16 | 2048 | data-parallel | 1588.4 | 1567.4 | asm | 240.7 | 1480.8 | 256x256x64 |
| 16384x16384x26624 | bfloat16 | 4096 | data-parallel | 1581.1 | 1583.5 | asm | 1515.9 | 1466.7 | 256x256x64 |
| 32768x16384x26624 | bfloat16 | 8192 | data-parallel | 1551.1 | 1576.0 | asm | 1485.0 | 1442.4 | 256x256x64 |
| 512x8192x28672 | bfloat16 | 64 | one-tile reduce-scatter, n = 3 | 1105.2 | 473.5 | flydsl | 780.1 | 921.4 | 240x256x64 |
| 1024x8192x28672 | bfloat16 | 128 | one-tile owner, n = 2 | 1441.0 | 940.2 | triton | 691.5 | 1304.1 | 256x256x64 |
| 2048x8192x28672 | bfloat16 | 256 | data-parallel | 1576.6 | 1572.0 | flydsl | 1035.5 | 1543.0 | 256x256x64 |
| 4096x8192x28672 | bfloat16 | 512 | data-parallel | 1596.1 | 1592.0 | flydsl | 1537.6 | 1536.1 | 256x256x64 |
| 8192x8192x28672 | bfloat16 | 1024 | data-parallel | 1589.9 | 1587.3 | asm | 1536.7 | 1524.1 | 256x256x64 |
| 16384x8192x28672 | bfloat16 | 2048 | data-parallel | 1586.5 | 1585.6 | asm | 1538.1 | 1496.8 | 256x256x64 |
| 32768x8192x28672 | bfloat16 | 4096 | data-parallel | 1586.0 | 1586.8 | asm | 1522.1 | 1462.7 | 256x256x64 |
| 512x7168x35840 | bfloat16 | 56 | one-tile reduce-scatter, n = 3 | 1056.1 | 432.7 | flydsl | 815.2 | 960.8 | 256x176x64 |
| 1024x7168x35840 | bfloat16 | 112 | one-tile owner, n = 2-3 | 1360.8 | 857.2 | flydsl | 565.5 | 1272.6 | 224x256x64 |
| 2048x7168x35840 | bfloat16 | 224 | spread | 1489.1 | 1465.6 | asm | 1435.4 | 1386.3 | 224x256x64 |
| 4096x7168x35840 | bfloat16 | 448 | spread | 1542.7 | 1479.8 | flydsl | 1446.5 | 1379.2 | 224x256x64 |
| 8192x7168x35840 | bfloat16 | 896 | one-tile owner, n = 2 | 1555.4 | 1451.7 | asm | 1399.1 | 1366.5 | 224x256x64 |
| 16384x7168x35840 | bfloat16 | 1792 | data-parallel | 1572.6 | 1572.5 | torch | 1496.7 | 1499.9 | 256x256x64 |
| 32768x7168x35840 | bfloat16 | 3584 | data-parallel | 1563.1 | 1571.0 | flydsl | 1504.6 | 1465.6 | 256x256x64 |
