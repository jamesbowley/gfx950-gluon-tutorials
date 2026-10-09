# v20 stream-K (auto) vs AITER's pick on the 630 shapes

Shapes: `supported_status_scoped.csv` (630 shapes, no bias; 598 bf16 and 32 fp32 outputs). Method as in `hip_version_repro/sweep.py` (Sep 29): inputs from `torch.manual_seed(0)`, every output checked against an fp32 `F.linear` reference, `do_bench_cudagraph` with 2 interleaved rounds and the median taken.

- Stack: torch 2.10.0+rocm7.2.4.git3d3aa833, hipBLASLt 100202 (dabb6df2b9), Triton 3.8.0 @ /home/jbowley/repos/triton_gfx950-tutorial-v2.2/python/triton.
- GPU: HIP device 0, PCI bus 117 (the Sep 29 GPU), gfx950:sramecc+:xnack-.
- AITER: `/home/jbowley/repos/aiter_llirsched_wt/aiter` at e2c7aa0aa [Triton/Gluon] gfx950 a16w16: skip the Gluon kernel on Triton without cd_regclass, on top of a1455f1c4 2026-10-02 [CK] [FlyDSL] add mxfp8 a8w8 blockscale for gfx950 (#4951); its llirSched plugin `/root/.aiter/llir_sched/ccf9a789cf3f11ab/libLlirSched.so`.
- v20: the tutorial's `v20_streamk_reduce_scatter` (snapshot sha256 93847ce6674f120e), auto policy and fixup, all 256 programs, v9 tile order, with the tutorial's llirSched plugin and amdgcnas hook. It writes C in C's dtype, so the fp32-output shapes are like for like.
- Gaps are v20 against the other kernel: positive means v20 is faster. "Within 1%" counts shapes inside run-to-run noise.

Measured 630 shapes; 630 have both v20 and AITER timings.

## v20 against AITER's pick

| | shapes | v20 faster | within 1% | median | p10 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| all shapes | 630 | 351 | 38 | +1.4% | -44.5% | +25.5% |

### By the policy auto chose

| | shapes | v20 faster | within 1% | median | p10 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| data-parallel | 274 | 224 | 26 | +3.2% | -3.5% | +43.0% |
| two-tile (rs) | 9 | 9 | 0 | +8.5% | +6.5% | +11.4% |
| one-tile reduce-scatter | 75 | 3 | 0 | -44.6% | -67.4% | -20.5% |
| one-tile owner | 272 | 115 | 12 | -5.2% | -41.8% | +36.6% |

### By AITER's pick

| | shapes | v20 faster | within 1% | median | p10 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| asm | 87 | 75 | 7 | +3.4% | -1.0% | +8.2% |
| flydsl | 257 | 144 | 8 | +3.1% | -48.7% | +65.3% |
| opus | 54 | 10 | 5 | -12.9% | -24.6% | +6.4% |
| torch | 229 | 119 | 18 | +0.6% | -42.4% | +10.2% |
| triton | 3 | 3 | 0 | +53.3% | +36.0% | +106.5% |

### By scope (as in the CSV)

| | shapes | v20 faster | within 1% | median | p10 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| in: 256x256 | 331 | 300 | 28 | +3.9% | +0.2% | +59.8% |
| in: non-pow2 tile | 117 | 26 | 5 | -9.0% | -44.6% | +8.5% |
| in: other listed tile | 144 | 22 | 5 | -20.1% | -51.0% | +10.2% |
| out: smaller tile | 23 | 1 | 0 | -55.8% | -75.4% | -27.5% |
| out: split-K | 15 | 2 | 0 | -40.2% | -53.4% | +9.7% |

### By output tiles at 256x256

| | shapes | v20 faster | within 1% | median | p10 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| under 1 wave | 321 | 71 | 19 | -16.1% | -55.8% | +13.9% |
| 1 to 2 waves | 84 | 63 | 8 | +3.8% | -2.7% | +54.8% |
| 2 to 4 waves | 68 | 63 | 2 | +7.3% | +1.4% | +45.9% |
| 4 to 8 waves | 67 | 65 | 3 | +4.0% | +1.6% | +20.2% |
| 8 waves or more | 90 | 89 | 6 | +3.6% | +1.2% | +10.7% |

### By K

| | shapes | v20 faster | within 1% | median | p10 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| K < 2048 | 91 | 30 | 6 | -10.9% | -70.8% | +4.3% |
| K >= 2048 | 539 | 321 | 32 | +2.2% | -34.7% | +41.3% |

### By leftover tiles S

| | shapes | v20 faster | within 1% | median | p10 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| S = 0 | 181 | 167 | 13 | +3.6% | +0.5% | +38.6% |
| 1 <= S <= 128 | 356 | 127 | 12 | -13.4% | -53.4% | +20.7% |
| S > 128 | 93 | 57 | 13 | +2.0% | -8.2% | +43.9% |

## v20 against v9 (stream-K and persistence gain)

| | shapes | v20 faster than v9 | within 1% | median | p10 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| all shapes | 630 | 451 | 101 | +1.8% | -5.7% | +74.6% |
| data-parallel | 274 | 157 | 100 | +0.5% | -4.0% | +1.8% |
| two-tile (rs) | 9 | 9 | 0 | +11.7% | +5.1% | +25.1% |
| one-tile reduce-scatter | 75 | 67 | 0 | +84.5% | -8.8% | +145.4% |
| one-tile owner | 272 | 218 | 1 | +11.1% | -16.0% | +69.5% |

## Diagnostic: auto's stream-K choice against forcing data-parallel

On the 356 shapes where auto picks a stream-K path. Positive means auto's choice is faster than v20 `dp`; this column never counts as v20's result.

| | shapes | auto faster | within 1% | median | p10 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| all | 356 | 292 | 7 | +19.7% | -13.9% | +99.8% |
| one-tile owner, K < 2048 | 42 | 0 | 0 | -22.4% | -41.9% | -8.2% |
| one-tile owner, K >= 2048 | 230 | 216 | 6 | +17.7% | +2.2% | +73.6% |
| one-tile reduce-scatter, K < 2048 | 7 | 0 | 0 | -24.0% | -25.8% | -22.0% |
| one-tile reduce-scatter, K >= 2048 | 68 | 67 | 0 | +97.9% | +42.7% | +141.6% |
| two-tile (rs), K >= 2048 | 9 | 9 | 1 | +11.3% | +4.6% | +24.6% |

## Largest gaps (TFLOPS)

### AITER furthest ahead

| shape | out | tiles | S | k-steps | AITER pick | AITER | v9 | v20 | v20 policy | v20 dp | v20 vs AITER |
|---|---|---:|---:|---:|---|---:|---:|---:|---|---:|---:|
| 512x1536x1536 | bfloat16 | 12 | 12 | 24 | flydsl | 344 | 86 | 67 | one-tile reduce-scatter | 84 | -80.6% |
| 512x2048x1536 | bfloat16 | 16 | 16 | 24 | torch | 398 | 113 | 88 | one-tile reduce-scatter | 112 | -78.0% |
| 512x4096x1024 | bfloat16 | 32 | 32 | 16 | torch | 491 | 200 | 110 | one-tile reduce-scatter | 197 | -77.6% |
| 1024x3072x512 | bfloat16 | 48 | 48 | 8 | flydsl | 462 | 226 | 105 | one-tile owner | 212 | -77.2% |
| 1024x1536x1536 | bfloat16 | 24 | 24 | 24 | flydsl | 498 | 169 | 122 | one-tile reduce-scatter | 165 | -75.4% |
| 512x2304x1536 | bfloat16 | 18 | 18 | 24 | flydsl | 392 | 127 | 97 | one-tile reduce-scatter | 126 | -75.2% |
| 2048x2048x512 | bfloat16 | 64 | 64 | 8 | flydsl | 531 | 286 | 135 | one-tile owner | 274 | -74.6% |
| 1024x4096x512 | bfloat16 | 64 | 64 | 8 | flydsl | 532 | 286 | 135 | one-tile owner | 273 | -74.6% |
| 512x6144x1024 | bfloat16 | 48 | 48 | 16 | flydsl | 580 | 298 | 168 | one-tile owner | 289 | -71.0% |
| 1024x2048x1536 | bfloat16 | 32 | 32 | 24 | torch | 568 | 221 | 166 | one-tile reduce-scatter | 218 | -70.8% |
| 512x4096x1536 | bfloat16 | 32 | 32 | 24 | torch | 569 | 225 | 167 | one-tile reduce-scatter | 222 | -70.6% |
| 512x5120x1280 | bfloat16 | 40 | 40 | 20 | flydsl | 537 | 264 | 160 | one-tile owner | 257 | -70.3% |
| 1024x2304x1536 | bfloat16 | 36 | 36 | 24 | torch | 532 | 249 | 164 | one-tile owner | 245 | -69.3% |
| 512x7168x1024 | bfloat16 | 56 | 56 | 16 | flydsl | 646 | 336 | 207 | one-tile owner | 325 | -67.9% |
| 512x1024x4096 | bfloat16 | 8 | 8 | 64 | flydsl | 332 | 67 | 108 | one-tile reduce-scatter | 67 | -67.4% |
| 1024x512x4096 | bfloat16 | 8 | 8 | 64 | flydsl | 325 | 67 | 108 | one-tile reduce-scatter | 67 | -66.7% |
| 2048x1536x1536 | bfloat16 | 48 | 48 | 24 | torch | 664 | 330 | 237 | one-tile owner | 323 | -64.3% |
| 512x4096x2048 | bfloat16 | 32 | 32 | 32 | torch | 606 | 238 | 217 | one-tile reduce-scatter | 237 | -64.2% |
| 512x8192x1024 | bfloat16 | 64 | 64 | 16 | flydsl | 696 | 375 | 249 | one-tile owner | 367 | -64.2% |
| 1536x512x4096 | bfloat16 | 12 | 12 | 64 | torch | 434 | 100 | 162 | one-tile reduce-scatter | 101 | -62.8% |

### v20 furthest ahead

| shape | out | tiles | S | k-steps | AITER pick | AITER | v9 | v20 | v20 policy | v20 dp | v20 vs AITER |
|---|---|---:|---:|---:|---|---:|---:|---:|---|---:|---:|
| 2048x51200x5120 | bfloat16 | 1600 | 64 | 80 | asm | 213 | 1416 | 1476 | one-tile owner | 1440 | +592.4% |
| 8192x16384x26624 | bfloat16 | 2048 | 0 | 416 | asm | 238 | 1605 | 1625 | data-parallel | - | +583.1% |
| 32768x16384x6656 | bfloat16 | 8192 | 0 | 104 | asm | 292 | 1581 | 1611 | data-parallel | - | +451.6% |
| 1024x16384x6656 | bfloat16 | 256 | 0 | 104 | asm | 452 | 1573 | 1524 | data-parallel | - | +236.9% |
| 8192x2560x6144 | bfloat16 | 320 | 64 | 96 | asm | 521 | 1175 | 1383 | one-tile owner | 1179 | +165.5% |
| 10240x7168x768 | bfloat16 | 1120 | 96 | 12 | opus | 408 | 1066 | 986 | one-tile owner | 1104 | +141.3% |
| 1024x7168x35840 | bfloat16 | 112 | 112 | 560 | flydsl | 575 | 869 | 1354 | one-tile owner | 863 | +135.3% |
| 512x5120x25600 | bfloat16 | 40 | 40 | 400 | flydsl | 438 | 326 | 992 | one-tile owner | 332 | +126.7% |
| 32768x2304x6144 | bfloat16 | 1152 | 128 | 96 | flydsl | 656 | 1422 | 1474 | one-tile owner | 1430 | +124.7% |
| 512x16384x26624 | bfloat16 | 128 | 128 | 416 | flydsl | 652 | 894 | 1376 | one-tile owner | 888 | +111.1% |
| 512x13312x16384 | bfloat16 | 104 | 104 | 256 | flydsl | 548 | 773 | 1150 | one-tile owner | 757 | +109.8% |
| 1024x8192x28672 | bfloat16 | 128 | 128 | 448 | triton | 698 | 955 | 1442 | one-tile owner | 941 | +106.5% |
| 512x53248x16384 | bfloat16 | 416 | 160 | 256 | flydsl | 656 | 1310 | 1319 | data-parallel | - | +101.2% |
| 2048x53248x16384 | bfloat16 | 1664 | 128 | 256 | flydsl | 802 | 1521 | 1606 | one-tile owner | 1545 | +100.2% |
| 1024x16384x26624 | bfloat16 | 256 | 0 | 416 | flydsl | 784 | 1571 | 1564 | data-parallel | - | +99.4% |
| 2048x16384x26624 | bfloat16 | 512 | 0 | 416 | flydsl | 826 | 1602 | 1621 | data-parallel | - | +96.2% |
| 16384x8192x3584 | bfloat16 | 2048 | 0 | 56 | flydsl | 847 | 1526 | 1542 | data-parallel | - | +82.2% |
| 512x16384x13312 | bfloat16 | 128 | 128 | 208 | flydsl | 708 | 922 | 1279 | one-tile owner | 926 | +80.6% |
| 2048x20480x7168 | bfloat16 | 640 | 128 | 112 | flydsl | 826 | 1399 | 1489 | one-tile owner | 1407 | +80.4% |
| 8192x6144x3072 | bfloat16 | 768 | 0 | 48 | flydsl | 853 | 1514 | 1513 | data-parallel | - | +77.5% |

## Against `results/details.csv`

- AITER's library and kernel are the same on 630 of 630 shapes and differ on 0.
- AITER's pick now against then, same kernel: median -0.1%, p10 -0.9%, p90 +0.6%.
- v9 now against then: median -0.1%, p10 -0.6%, p90 +0.2%.
- v20 now against then: median -0.0%, p10 -0.8%, p90 +1.3%.
- v20 faster than AITER's pick: 351 shapes now (median +1.4%), 353 then (median +1.5%).

## Against the Sep 29 sweep (`hip_version_repro/details_rocm724.csv`)

- AITER's library and kernel are the same on 630 of 630 shapes and differ on 0.
- AITER's pick now against then, same kernel: median -0.1%, p10 -1.3%, p90 +0.6%.
- v9 now against then: median -0.0%, p10 -0.5%, p90 +0.5%.

## Wrong outputs, failures and notes

| shape | out | pick ok | v9 ok | v20 ok | v20 after timing | v20 flags left | note |
|---|---|---|---|---|---|---|---|
| 1024x6144x2048 | bfloat16 | False (1) | True | True (1) | True | 0 |  |
| 8192x6144x2048 | bfloat16 | False (1) | True | True (1) | True | 0 |  |
| 1024x6144x3072 | bfloat16 | False (2) | True | True (1) | True | 0 |  |
| 32768x6144x3072 | bfloat16 | False (2) | True | True (1) | True | 0 |  |
| 512x1024x4096 | bfloat16 | False (2) | True | True (1) | True | 0 |  |
| 768x8704x4096 | bfloat16 | False (2) | True | True (1) | True | 0 |  |
| 1280x512x4096 | bfloat16 | False (2) | True | True (1) | True | 0 |  |
| 512x6400x5120 | bfloat16 | False (2) | True | True (1) | True | 0 |  |
| 512x12800x5120 | bfloat16 | False (2) | True | True (1) | True | 0 |  |
| 1024x6400x5120 | bfloat16 | False (2) | True | True (1) | True | 0 |  |
| 512x1280x6144 | bfloat16 | False (2) | True | True (1) | True | 0 |  |
| 512x1536x6144 | bfloat16 | False (2) | True | True (1) | True | 0 |  |
| 512x2304x6144 | bfloat16 | False (2) | True | True (1) | True | 0 |  |
| 512x3072x6144 | bfloat16 | False (2) | True | True (1) | True | 0 |  |
| 512x5120x6400 | bfloat16 | False (2) | True | True (1) | True | 0 |  |
| 768x2048x7168 | bfloat16 | False (2) | True | True (1) | True | 0 |  |
| 1024x512x7168 | bfloat16 | False (2) | True | True (1) | True | 0 |  |
| 1024x1536x7168 | bfloat16 | False (2) | True | True (2) | True | 0 |  |
| 4096x512x7168 | bfloat16 | False (2) | True | True (2) | True | 0 |  |
| 512x4096x8192 | bfloat16 | False (2) | True | True (1) | True | 0 |  |
| 512x16384x8192 | bfloat16 | False (2) | True | True (2) | True | 0 |  |
| 1024x10240x8192 | bfloat16 | False (2) | True | True (2) | True | 0 |  |
| 512x13312x16384 | bfloat16 | False (4) | True | True (2) | True | 0 |  |
| 512x6144x18432 | bfloat16 | False (4) | True | True (2) | True | 0 |  |
| 512x5120x25600 | bfloat16 | False (4) | True | True (2) | True | 0 |  |
| 512x16384x26624 | bfloat16 | False (4) | True | True (2) | True | 0 |  |
| 2048x16384x26624 | bfloat16 | False (4) | True | True (4) | True | 0 |  |
| 512x8192x28672 | bfloat16 | False (4) | True | True (4) | True | 0 |  |
| 512x7168x35840 | bfloat16 | False (4) | True | True (2) | True | 0 |  |
