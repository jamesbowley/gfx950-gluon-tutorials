## Results table

| Run | Caches | faster | within 1% | slower | median | under one wave: median | same-tile behind | v9: median |
| base | cold | 421 | 47 | 162 | +3.2% | -0.3% | 3 of 262 | -3.9% |
| this | cold | 446 | 44 | 140 | +3.4% | +1.6% | 3 of 262 | -3.9% |

## Three-way against the best library

| Caches | Kernel | faster | within 1% | slower | median | same-tile behind |
| cold | v27b (capped) | 446 | 44 | 140 | +3.4% | 3 of 262 |
| cold | AITER v9 | 248 | 38 | 344 | -3.9% | 27 of 262 |
| cold | v27b data-parallel (forced) | 348 | 47 | 235 | +1.6% | 15 of 262 |

## Per path against the baseline (this / base - 1, median)

| Path | Caches | Shapes | v27b vs base | AITER v9 (card) | v27b corrected by v9 | dp | hipBLASLt | AITER pick |
| All shapes | cold | 630 | -0.2% | +0.0% | -0.2% (86 up, 83 down) | -0.0% | +0.0% | +0.0% |
| Handed to v9 | cold | 267 | -0.1% | +0.0% | -0.1% (52 up, 29 down) | +0.2% | +0.0% | +0.0% |
| Stream-K split | cold | 166 | -0.3% | +0.0% | -0.3% (17 up, 24 down) | +0.0% | +0.0% | +0.0% |
| Persistent data-parallel | cold | 197 | -0.3% | +0.0% | -0.3% (17 up, 30 down) | -0.3% | +0.0% | +0.0% |

cold: behind->ahead 26, ahead->behind 0; v27b vs v9: 449 faster, 36 slower, median +6.9%

## Same-tile behind

Cold: 3 behind by more than 1%, 6 behind by less (ties)

| # | Shape (MxNxK) | Output tiles | Full waves | S | k-steps | Best library | v27b config | v27 tile | Best TFLOPS | v9 TFLOPS | v27b TFLOPS | v27b vs best | v27b vs v9 |
|---:|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | 1024x2304x16384 | 36 | 0 | 36 | 256 | hipBLASLt | spread | 128x128x128 | 873 | 279 | 705 | -19.2% | +152.4% |
| 2 | 768x129280x4096 | 1515 | 5 | 235 | 64 | hipBLASLt | data-parallel | 256x256x64 | 1354 | 1409 | 1309 | -3.4% | -7.1% |
| 3 | 16384x7168x512 | 1792 | 7 | 0 | 8 | hipBLASLt | data-parallel | 256x256x64 | 982 | 900 | 964 | -1.9% | +7.0% |

## Hand-off

| Caches | Shapes | Against the persistent loop | Against AITER's v9 |
| Cold | 267 all | +1.8% (199 faster, 5 slower) | +34.5% (47 within 1%, 203 faster, 17 slower) |
| Cold | 69 at 32 k-steps or fewer | +3.2% (67 faster, 1 slower) | +30.9% (13 within 1%, 48 faster, 8 slower) |
| Cold | 198 above 32 k-steps | +1.4% (132 faster, 4 slower) | +36.2% (34 within 1%, 155 faster, 9 slower) |

## Spread and correctness

cold: v27b median spread 0.63%, 43 shapes over 2%; v9 0.48%, 45 over 2% | base: cold: v27b median spread 0.49%, 51 shapes over 2%; v9 0.48%, 45 over 2%
cold: 0 shapes with a v27b/v9/hipBLASLt/dp failure or flag left; AITER pick outside tolerance on 29; notes on 0

## v27b (auto tile) against v26, timed in the same run (this / v26 - 1)

Tiles: v27 auto against v27 at 256x256x64, both on Triton 3.9. Compiler: v27 at 256x256x64 (v26's code) against v26 on Triton v2.2 + llirSched. On each shape the two ratios multiply to v27 against v26; their medians need not.

| Group | Caches | Shapes | v27b vs v26: faster | within 1% | slower | median | tiles: median | compiler: median | v27 at 256 vs v26: faster | slower |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All shapes | cold | 630 | 231 | 319 | 80 | +0.3% | -0.0% | +0.5% | 215 | 21 |
| v27 keeps 256x256x64 | cold | 385 | 46 | 308 | 31 | -0.0% | -0.3% | +0.3% | 30 | 17 |
| v27 picks a smaller tile | cold | 245 | 185 | 11 | 49 | +15.6% | +12.5% | +2.8% | 185 | 4 |
| tile 256x256x64 | cold | 385 | 46 | 308 | 31 | -0.0% | -0.3% | +0.3% | 30 | 17 |
| tile 128x256x64 | cold | 97 | 68 | 6 | 23 | +8.5% | +7.0% | +1.4% | 56 | 4 |
| tile 128x128x128 | cold | 83 | 66 | 2 | 15 | +22.9% | +18.0% | +2.6% | 67 | 0 |
| tile 64x128x128 | cold | 41 | 29 | 2 | 10 | +20.9% | +15.9% | +4.3% | 39 | 0 |
| tile 64x64x128 | cold | 24 | 22 | 1 | 1 | +41.0% | +32.6% | +4.1% | 23 | 0 |
| Handed to v9 | cold | 267 | 162 | 62 | 43 | +9.2% | +5.9% | +1.5% | 150 | 5 |
| Persistent data-parallel | cold | 197 | 24 | 156 | 17 | -0.0% | -0.4% | +0.3% | 15 | 7 |
| Stream-K split: reduce-scatter | cold | 12 | 1 | 11 | 0 | +0.5% | +0.1% | +0.4% | 0 | 1 |
| Stream-K split: owner | cold | 112 | 40 | 69 | 3 | +0.5% | +0.2% | +0.6% | 35 | 8 |
| Stream-K split: spread | cold | 42 | 4 | 21 | 17 | -0.7% | -1.2% | +0.5% | 15 | 0 |
| under one wave | cold | 321 | 187 | 74 | 60 | +8.0% | +5.8% | +1.7% | 191 | 6 |
| one wave or more | cold | 309 | 44 | 245 | 20 | +0.1% | -0.3% | +0.4% | 24 | 15 |

## Every kernel against the best library

| Caches | Kernel | faster | within 1% | slower | median | under one wave: median | same-tile behind |
|---|---|---:|---:|---:|---:|---:|---:|
| cold | v27b auto (capped) | 446 | 44 | 140 | +3.4% | +1.6% | 3 of 262 |
| cold | v27b auto, data-parallel (forced) | 348 | 47 | 235 | +1.6% | -5.2% | 15 of 262 |
| cold | v27b at 256x256x64 (capped) | 374 | 33 | 223 | +2.7% | -6.4% | 4 of 262 |
| cold | v26 (capped) | 361 | 33 | 236 | +2.2% | -7.5% | 1 of 262 |
| cold | AITER v9, existing | 248 | 38 | 344 | -3.9% | -36.5% | 27 of 262 |
| cold | AITER v9, new | 334 | 38 | 258 | +1.6% | -4.8% | 25 of 262 |

## v27b auto against the best library, by its tile

| Group | Caches | Shapes | faster | within 1% | slower | median | v26 vs best: median |
|---|---|---:|---:|---:|---:|---:|---:|
| All shapes | cold | 630 | 446 | 44 | 140 | +3.4% | +2.2% |
| v27 keeps 256x256x64 | cold | 385 | 322 | 27 | 36 | +3.8% | +3.7% |
| v27 picks a smaller tile | cold | 245 | 124 | 17 | 104 | +1.2% | -13.0% |
| tile 256x256x64 | cold | 385 | 322 | 27 | 36 | +3.8% | +3.7% |
| tile 128x256x64 | cold | 97 | 55 | 5 | 37 | +2.6% | -6.2% |
| tile 128x128x128 | cold | 83 | 55 | 6 | 22 | +4.4% | -11.5% |
| tile 64x128x128 | cold | 41 | 12 | 2 | 27 | -5.8% | -21.7% |
| tile 64x64x128 | cold | 24 | 2 | 4 | 18 | -7.8% | -32.0% |
| Handed to v9 | cold | 267 | 144 | 18 | 105 | +1.7% | -9.2% |
| Persistent data-parallel | cold | 197 | 170 | 18 | 9 | +3.1% | +3.1% |
| Stream-K split: reduce-scatter | cold | 12 | 12 | 0 | 0 | +16.2% | +16.0% |
| Stream-K split: owner | cold | 112 | 97 | 6 | 9 | +8.7% | +8.3% |
| Stream-K split: spread | cold | 42 | 23 | 2 | 17 | +1.8% | +4.7% |
| under one wave | cold | 321 | 167 | 24 | 130 | +1.6% | -7.5% |
| one wave or more | cold | 309 | 279 | 20 | 10 | +3.9% | +3.7% |

## The new AITER v9 against the existing one (this run)

| Group | Caches | Shapes | new faster | within 1% | new slower | median | new vs best: median | existing vs best: median |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| All shapes | cold | 630 | 286 | 303 | 41 | +0.5% | +1.6% | -3.9% |
| new v9 tile 256x256 | cold | 334 | 17 | 300 | 17 | +0.0% | +3.0% | +2.9% |
| new v9 tile 256x128 | cold | 138 | 111 | 3 | 24 | +28.9% | -3.5% | -19.0% |
| new v9 tile 128x128 | cold | 86 | 86 | 0 | 0 | +86.2% | -1.2% | -47.2% |
| new v9 tile 128x64 | cold | 42 | 42 | 0 | 0 | +138.5% | -17.0% | -65.1% |
| new v9 tile 64x64 | cold | 30 | 30 | 0 | 0 | +218.7% | -30.1% | -77.7% |
| under one wave | cold | 321 | 260 | 56 | 5 | +36.1% | -4.8% | -36.5% |
| one wave or more | cold | 309 | 26 | 247 | 36 | +0.1% | +3.0% | +2.9% |

## v27b auto against the new AITER v9

| Group | Caches | Shapes | v27 faster | within 1% | v27 slower | median |
|---|---|---:|---:|---:|---:|---:|
| All shapes | cold | 630 | 383 | 203 | 44 | +2.1% |
| v27 keeps 256x256x64 | cold | 385 | 198 | 159 | 28 | +1.1% |
| v27 picks a smaller tile | cold | 245 | 185 | 44 | 16 | +7.8% |
| tile 256x256x64 | cold | 385 | 198 | 159 | 28 | +1.1% |
| tile 128x256x64 | cold | 97 | 44 | 42 | 11 | +0.8% |
| tile 128x128x128 | cold | 83 | 76 | 2 | 5 | +7.9% |
| tile 64x128x128 | cold | 41 | 41 | 0 | 0 | +13.3% |
| tile 64x64x128 | cold | 24 | 24 | 0 | 0 | +26.4% |
| Handed to v9 | cold | 267 | 142 | 103 | 22 | +1.5% |
| Persistent data-parallel | cold | 197 | 84 | 93 | 20 | +0.6% |
| Stream-K split: reduce-scatter | cold | 12 | 12 | 0 | 0 | +26.0% |
| Stream-K split: owner | cold | 112 | 106 | 5 | 1 | +8.9% |
| Stream-K split: spread | cold | 42 | 39 | 2 | 1 | +13.3% |
| under one wave | cold | 321 | 213 | 86 | 22 | +6.1% |
| one wave or more | cold | 309 | 170 | 117 | 22 | +1.3% |

## Checks of the extra columns

- cold: 0 shapes with a new v9 / v26 / v27-at-256 failure or v26 flag left; builds on Triton 3.9 434, without sched_barriers: new v9 0, v27 and v9_tile 3; notes on 0.
