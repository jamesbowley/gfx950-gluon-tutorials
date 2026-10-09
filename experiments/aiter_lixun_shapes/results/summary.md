# v20 stream-K (auto) vs AITER's pick on the 630 shapes

Shapes: `supported_status_scoped.csv` (630 shapes, no bias; 598 bf16 and 32 fp32 outputs). Method as in `hip_version_repro/sweep.py` (Sep 29): inputs from `torch.manual_seed(0)`, every output checked against an fp32 `F.linear` reference, `do_bench_cudagraph` with 2 interleaved rounds and the median taken.

- Stack: torch 2.10.0+rocm7.2.4.git3d3aa833, hipBLASLt 100202 (dabb6df2b9), Triton 3.8.0 @ /home/jbowley/repos/triton_gfx950-tutorial-v2.2/python/triton.
- GPU: HIP device 0, PCI bus 117 (the Sep 29 GPU), gfx950:sramecc+:xnack-.
- AITER: `/home/jbowley/repos/aiter_llirsched_wt/aiter` at 2a1c2087c (llirSched_integration), on top of 8253efc40 (main, Sep 28); its llirSched plugin `/root/.aiter/llir_sched/ccf9a789cf3f11ab/libLlirSched.so`.
- v20: the tutorial's `v20_streamk_reduce_scatter` (snapshot sha256 93847ce6674f120e), auto policy and fixup, all 256 programs, v9 tile order, with the tutorial's llirSched plugin and amdgcnas hook. It writes C in C's dtype, so the fp32-output shapes are like for like.
- Gaps are v20 against the other kernel: positive means v20 is faster. "Within 1%" counts shapes inside run-to-run noise.

Measured 630 shapes; 630 have both v20 and AITER timings.

## v20 against AITER's pick

| | shapes | v20 faster | within 1% | median | p10 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| all shapes | 630 | 353 | 36 | +1.5% | -44.5% | +25.6% |

### By the policy auto chose

| | shapes | v20 faster | within 1% | median | p10 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| data-parallel | 274 | 225 | 27 | +3.3% | -4.1% | +42.3% |
| two-tile (rs) | 9 | 9 | 0 | +9.6% | +6.6% | +12.1% |
| one-tile reduce-scatter | 75 | 3 | 0 | -45.0% | -68.2% | -21.0% |
| one-tile owner | 272 | 116 | 9 | -5.5% | -42.3% | +33.4% |

### By AITER's pick

| | shapes | v20 faster | within 1% | median | p10 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| asm | 87 | 75 | 2 | +3.2% | -2.2% | +9.6% |
| flydsl | 257 | 143 | 9 | +2.7% | -49.5% | +65.3% |
| opus | 54 | 11 | 4 | -12.9% | -23.9% | +6.3% |
| torch | 229 | 121 | 21 | +0.6% | -43.0% | +10.2% |
| triton | 3 | 3 | 0 | +54.9% | +36.6% | +107.2% |

### By scope (as in the CSV)

| | shapes | v20 faster | within 1% | median | p10 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| in: 256x256 | 331 | 301 | 26 | +4.0% | +0.2% | +59.4% |
| in: non-pow2 tile | 117 | 28 | 5 | -9.2% | -45.0% | +8.8% |
| in: other listed tile | 144 | 21 | 5 | -21.3% | -51.5% | +10.1% |
| out: smaller tile | 23 | 1 | 0 | -56.4% | -75.8% | -28.0% |
| out: split-K | 15 | 2 | 0 | -40.7% | -53.8% | +11.1% |

### By output tiles at 256x256

| | shapes | v20 faster | within 1% | median | p10 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| under 1 wave | 321 | 72 | 13 | -15.8% | -56.4% | +14.0% |
| 1 to 2 waves | 84 | 64 | 9 | +3.6% | -3.7% | +56.8% |
| 2 to 4 waves | 68 | 63 | 3 | +7.1% | +1.3% | +45.2% |
| 4 to 8 waves | 67 | 66 | 5 | +4.1% | +1.6% | +21.1% |
| 8 waves or more | 90 | 88 | 6 | +3.4% | +1.2% | +10.0% |

### By K

| | shapes | v20 faster | within 1% | median | p10 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| K < 2048 | 91 | 29 | 8 | -11.4% | -71.3% | +4.4% |
| K >= 2048 | 539 | 324 | 28 | +2.3% | -35.5% | +41.9% |

### By leftover tiles S

| | shapes | v20 faster | within 1% | median | p10 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| S = 0 | 181 | 167 | 17 | +3.5% | +0.3% | +37.3% |
| 1 <= S <= 128 | 356 | 128 | 9 | -12.9% | -54.5% | +20.9% |
| S > 128 | 93 | 58 | 10 | +1.9% | -8.0% | +45.0% |

## v20 against v9 (stream-K and persistence gain)

| | shapes | v20 faster than v9 | within 1% | median | p10 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| all shapes | 630 | 454 | 95 | +1.9% | -6.7% | +72.8% |
| data-parallel | 274 | 162 | 95 | +0.5% | -3.7% | +1.8% |
| two-tile (rs) | 9 | 9 | 0 | +12.9% | +6.4% | +26.7% |
| one-tile reduce-scatter | 75 | 67 | 0 | +86.9% | -9.3% | +144.2% |
| one-tile owner | 272 | 216 | 0 | +11.6% | -17.5% | +67.0% |

## Diagnostic: auto's stream-K choice against forcing data-parallel

On the 356 shapes where auto picks a stream-K path. Positive means auto's choice is faster than v20 `dp`; this column never counts as v20's result.

| | shapes | auto faster | within 1% | median | p10 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| all | 356 | 290 | 6 | +19.4% | -15.2% | +98.8% |
| one-tile owner, K < 2048 | 42 | 0 | 0 | -23.0% | -42.7% | -8.4% |
| one-tile owner, K >= 2048 | 230 | 214 | 5 | +17.3% | +1.7% | +71.1% |
| one-tile reduce-scatter, K < 2048 | 7 | 0 | 0 | -25.1% | -26.5% | -24.3% |
| one-tile reduce-scatter, K >= 2048 | 68 | 67 | 0 | +95.1% | +42.3% | +141.5% |
| two-tile (rs), K >= 2048 | 9 | 9 | 1 | +12.2% | +5.2% | +25.6% |

## Largest gaps (TFLOPS)

### AITER furthest ahead

| shape | out | tiles | S | k-steps | AITER pick | AITER | v9 | v20 | v20 policy | v20 dp | v20 vs AITER |
|---|---|---:|---:|---:|---|---:|---:|---:|---|---:|---:|
| 512x1536x1536 | bfloat16 | 12 | 12 | 24 | flydsl | 346 | 86 | 65 | one-tile reduce-scatter | 84 | -81.3% |
| 512x2048x1536 | bfloat16 | 16 | 16 | 24 | torch | 394 | 114 | 84 | one-tile reduce-scatter | 112 | -78.6% |
| 1024x3072x512 | bfloat16 | 48 | 48 | 8 | flydsl | 461 | 226 | 103 | one-tile owner | 212 | -77.7% |
| 512x4096x1024 | bfloat16 | 32 | 32 | 16 | torch | 483 | 201 | 110 | one-tile reduce-scatter | 197 | -77.3% |
| 1024x1536x1536 | bfloat16 | 24 | 24 | 24 | flydsl | 502 | 169 | 121 | one-tile reduce-scatter | 165 | -75.8% |
| 512x2304x1536 | bfloat16 | 18 | 18 | 24 | flydsl | 390 | 127 | 95 | one-tile reduce-scatter | 126 | -75.7% |
| 2048x2048x512 | bfloat16 | 64 | 64 | 8 | flydsl | 527 | 286 | 134 | one-tile owner | 273 | -74.6% |
| 1024x4096x512 | bfloat16 | 64 | 64 | 8 | flydsl | 524 | 286 | 134 | one-tile owner | 273 | -74.4% |
| 512x6144x1024 | bfloat16 | 48 | 48 | 16 | flydsl | 588 | 299 | 166 | one-tile owner | 290 | -71.8% |
| 1024x2048x1536 | bfloat16 | 32 | 32 | 24 | torch | 574 | 222 | 165 | one-tile reduce-scatter | 218 | -71.3% |
| 512x5120x1280 | bfloat16 | 40 | 40 | 20 | flydsl | 537 | 264 | 157 | one-tile owner | 257 | -70.8% |
| 512x4096x1536 | bfloat16 | 32 | 32 | 24 | torch | 566 | 225 | 166 | one-tile reduce-scatter | 222 | -70.6% |
| 1024x2304x1536 | bfloat16 | 36 | 36 | 24 | torch | 533 | 249 | 164 | one-tile owner | 244 | -69.3% |
| 512x7168x1024 | bfloat16 | 56 | 56 | 16 | flydsl | 629 | 337 | 198 | one-tile owner | 324 | -68.5% |
| 512x1024x4096 | bfloat16 | 8 | 8 | 64 | flydsl | 340 | 67 | 108 | one-tile reduce-scatter | 67 | -68.2% |
| 1024x512x4096 | bfloat16 | 8 | 8 | 64 | flydsl | 327 | 67 | 108 | one-tile reduce-scatter | 67 | -67.1% |
| 512x4096x2048 | bfloat16 | 32 | 32 | 32 | torch | 611 | 238 | 216 | one-tile reduce-scatter | 237 | -64.6% |
| 512x8192x1024 | bfloat16 | 64 | 64 | 16 | flydsl | 691 | 376 | 246 | one-tile owner | 367 | -64.4% |
| 2048x1536x1536 | bfloat16 | 48 | 48 | 24 | torch | 659 | 330 | 235 | one-tile owner | 323 | -64.3% |
| 1024x3072x1536 | bfloat16 | 48 | 48 | 24 | flydsl | 647 | 330 | 234 | one-tile owner | 324 | -63.8% |

### v20 furthest ahead

| shape | out | tiles | S | k-steps | AITER pick | AITER | v9 | v20 | v20 policy | v20 dp | v20 vs AITER |
|---|---|---:|---:|---:|---|---:|---:|---:|---|---:|---:|
| 2048x51200x5120 | bfloat16 | 1600 | 64 | 80 | asm | 214 | 1420 | 1479 | one-tile owner | 1443 | +592.3% |
| 8192x16384x26624 | bfloat16 | 2048 | 0 | 416 | asm | 238 | 1609 | 1629 | data-parallel | - | +584.6% |
| 32768x16384x6656 | bfloat16 | 8192 | 0 | 104 | asm | 292 | 1586 | 1612 | data-parallel | - | +452.1% |
| 1024x16384x6656 | bfloat16 | 256 | 0 | 104 | asm | 452 | 1578 | 1540 | data-parallel | - | +240.9% |
| 8192x2560x6144 | bfloat16 | 320 | 64 | 96 | asm | 520 | 1182 | 1378 | one-tile owner | 1189 | +164.7% |
| 1024x7168x35840 | bfloat16 | 112 | 112 | 560 | flydsl | 575 | 872 | 1362 | one-tile owner | 867 | +136.9% |
| 10240x7168x768 | bfloat16 | 1120 | 96 | 12 | opus | 420 | 1066 | 980 | one-tile owner | 1103 | +133.2% |
| 512x5120x25600 | bfloat16 | 40 | 40 | 400 | flydsl | 437 | 327 | 990 | one-tile owner | 333 | +126.4% |
| 32768x2304x6144 | bfloat16 | 1152 | 128 | 96 | flydsl | 657 | 1429 | 1480 | one-tile owner | 1432 | +125.3% |
| 512x13312x16384 | bfloat16 | 104 | 104 | 256 | flydsl | 542 | 779 | 1164 | one-tile owner | 792 | +114.7% |
| 512x16384x26624 | bfloat16 | 128 | 128 | 416 | flydsl | 652 | 914 | 1379 | one-tile owner | 895 | +111.5% |
| 1024x8192x28672 | bfloat16 | 128 | 128 | 448 | triton | 699 | 953 | 1448 | one-tile owner | 938 | +107.2% |
| 512x53248x16384 | bfloat16 | 416 | 160 | 256 | flydsl | 652 | 1301 | 1317 | data-parallel | - | +101.8% |
| 1024x16384x26624 | bfloat16 | 256 | 0 | 416 | flydsl | 788 | 1574 | 1566 | data-parallel | - | +98.7% |
| 2048x53248x16384 | bfloat16 | 1664 | 128 | 256 | flydsl | 816 | 1526 | 1605 | one-tile owner | 1547 | +96.6% |
| 2048x16384x26624 | bfloat16 | 512 | 0 | 416 | flydsl | 830 | 1607 | 1625 | data-parallel | - | +95.9% |
| 512x16384x13312 | bfloat16 | 128 | 128 | 208 | flydsl | 701 | 919 | 1278 | one-tile owner | 930 | +82.2% |
| 16384x8192x3584 | bfloat16 | 2048 | 0 | 56 | flydsl | 854 | 1530 | 1541 | data-parallel | - | +80.6% |
| 2048x20480x7168 | bfloat16 | 640 | 128 | 112 | flydsl | 828 | 1402 | 1493 | one-tile owner | 1410 | +80.4% |
| 32768x8192x3584 | bfloat16 | 4096 | 0 | 56 | flydsl | 879 | 1520 | 1547 | data-parallel | - | +76.0% |

## Against the Sep 29 sweep (`hip_version_repro/details_rocm724.csv`)

- AITER's library and kernel are the same on 630 of 630 shapes and differ on 0.
- AITER's pick now against then, same kernel: median -0.0%, p10 -0.9%, p90 +0.7%.
- v9 now against then: median +0.1%, p10 -0.3%, p90 +0.8%.

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
