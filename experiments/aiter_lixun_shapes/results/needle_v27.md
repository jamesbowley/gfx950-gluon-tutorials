## Results table

| Run | Caches | faster | within 1% | slower | median | under one wave: median | same-tile behind | v9: median |
| base | cold | 354 | 43 | 233 | +1.8% | -8.2% | 1 of 260 | -4.1% |
| this | cold | 421 | 47 | 162 | +3.2% | -0.3% | 3 of 262 | -3.9% |

## Three-way against the best library

| Caches | Kernel | faster | within 1% | slower | median | same-tile behind |
| cold | v27 (capped) | 421 | 47 | 162 | +3.2% | 3 of 262 |
| cold | AITER v9 | 248 | 38 | 344 | -3.9% | 27 of 262 |
| cold | v27 data-parallel (forced) | 322 | 37 | 271 | +1.2% | 13 of 262 |

## Per path against the baseline (this / base - 1, median)

| Path | Caches | Shapes | v27 vs base | AITER v9 (card) | v27 corrected by v9 | dp | hipBLASLt | AITER pick |
| All shapes | cold | 630 | +0.4% | +0.3% | +0.4% (204 up, 79 down) | +0.6% | +0.0% | +0.1% |
| Handed to v9 | cold | 242 | +1.8% | +0.6% | +1.6% (124 up, 36 down) | +34.6% | +0.1% | +0.2% |
| Stream-K split | cold | 191 | +0.3% | +0.4% | +0.2% (55 up, 35 down) | +0.4% | +0.1% | +0.1% |
| Persistent data-parallel | cold | 197 | +0.3% | -0.0% | +0.3% (25 up, 8 down) | +0.3% | -0.2% | -0.1% |

cold: behind->ahead 70, ahead->behind 15; v27 vs v9: 431 faster, 23 slower, median +6.0%

## Same-tile behind

Cold: 3 behind by more than 1%, 3 behind by less (ties)

| # | Shape (MxNxK) | Output tiles | Full waves | S | k-steps | Best library | v27 config | v27 tile | Best TFLOPS | v9 TFLOPS | v27 TFLOPS | v27 vs best | v27 vs v9 |
|---:|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | 1024x2304x16384 | 36 | 0 | 36 | 256 | hipBLASLt | spread | 128x128x128 | 873 | 279 | 706 | -19.1% | +152.8% |
| 2 | 768x129280x4096 | 1515 | 5 | 235 | 64 | hipBLASLt | data-parallel | 256x256x64 | 1354 | 1409 | 1302 | -3.9% | -7.6% |
| 3 | 16384x7168x512 | 1792 | 7 | 0 | 8 | hipBLASLt | data-parallel | 256x256x64 | 982 | 900 | 965 | -1.7% | +7.2% |

## Hand-off

| Caches | Shapes | Against the persistent loop | Against AITER's v9 |
| Cold | 242 all | +1.9% (183 faster, 3 slower) | +34.7% (60 within 1%, 167 faster, 15 slower) |
| Cold | 64 at 32 k-steps or fewer | +3.4% (61 faster, 0 slower) | +31.1% (17 within 1%, 38 faster, 9 slower) |
| Cold | 178 above 32 k-steps | +1.6% (122 faster, 3 slower) | +35.5% (43 within 1%, 129 faster, 6 slower) |

## Spread and correctness

cold: v27 median spread 0.49%, 51 shapes over 2%; v9 0.48%, 45 over 2% | base: cold: v27 median spread 0.52%, 46 shapes over 2%; v9 0.31%, 8 over 2%
cold: 0 shapes with a v27/v9/hipBLASLt/dp failure or flag left; AITER pick outside tolerance on 29; notes on 0

## v27 (auto tile) against v26, timed in the same run (this / v26 - 1)

Tiles: v27 auto against v27 at 256x256x64, both on Triton 3.9. Compiler: v27 at 256x256x64 (v26's code) against v26 on Triton v2.2 + llirSched. On each shape the two ratios multiply to v27 against v26; their medians need not.

| Group | Caches | Shapes | v27 vs v26: faster | within 1% | slower | median | tiles: median | compiler: median | v27 at 256 vs v26: faster | slower |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All shapes | cold | 630 | 197 | 363 | 70 | +0.4% | -0.1% | +0.5% | 215 | 21 |
| v27 keeps 256x256x64 | cold | 401 | 23 | 354 | 24 | +0.2% | -0.1% | +0.3% | 34 | 18 |
| v27 picks a smaller tile | cold | 229 | 174 | 9 | 46 | +15.3% | +12.1% | +3.0% | 181 | 3 |
| tile 256x256x64 | cold | 401 | 23 | 354 | 24 | +0.2% | -0.1% | +0.3% | 34 | 18 |
| tile 128x256x64 | cold | 95 | 70 | 5 | 20 | +9.6% | +7.7% | +1.6% | 59 | 3 |
| tile 128x128x128 | cold | 77 | 60 | 1 | 16 | +20.5% | +16.2% | +3.1% | 68 | 0 |
| tile 64x128x128 | cold | 37 | 25 | 2 | 10 | +19.9% | +14.9% | +4.0% | 35 | 0 |
| tile 64x64x128 | cold | 20 | 19 | 1 | 0 | +30.8% | +26.4% | +3.8% | 19 | 0 |
| Handed to v9 | cold | 242 | 128 | 77 | 37 | +2.8% | +0.8% | +1.5% | 134 | 4 |
| Persistent data-parallel | cold | 197 | 16 | 174 | 7 | +0.3% | -0.1% | +0.3% | 15 | 7 |
| Stream-K split: reduce-scatter | cold | 12 | 0 | 11 | 1 | +0.4% | -0.1% | +0.4% | 0 | 1 |
| Stream-K split: owner | cold | 137 | 49 | 78 | 10 | +0.6% | -0.0% | +0.7% | 51 | 9 |
| Stream-K split: spread | cold | 42 | 4 | 23 | 15 | -0.3% | -0.5% | +0.5% | 15 | 0 |
| under one wave | cold | 321 | 175 | 89 | 57 | +4.3% | +2.5% | +1.7% | 191 | 6 |
| one wave or more | cold | 309 | 22 | 274 | 13 | +0.3% | -0.1% | +0.4% | 24 | 15 |

## Every kernel against the best library

| Caches | Kernel | faster | within 1% | slower | median | under one wave: median | same-tile behind |
|---|---|---:|---:|---:|---:|---:|---:|
| cold | v27 auto (capped) | 421 | 47 | 162 | +3.2% | -0.3% | 3 of 262 |
| cold | v27 auto, data-parallel (forced) | 322 | 37 | 271 | +1.2% | -8.4% | 13 of 262 |
| cold | v27 at 256x256x64 (capped) | 374 | 33 | 223 | +2.7% | -6.4% | 4 of 262 |
| cold | v26 (capped) | 361 | 33 | 236 | +2.2% | -7.5% | 1 of 262 |
| cold | AITER v9, existing | 248 | 38 | 344 | -3.9% | -36.5% | 27 of 262 |
| cold | AITER v9, new | 334 | 38 | 258 | +1.6% | -4.8% | 25 of 262 |

## v27 auto against the best library, by its tile

| Group | Caches | Shapes | faster | within 1% | slower | median | v26 vs best: median |
|---|---|---:|---:|---:|---:|---:|---:|
| All shapes | cold | 630 | 421 | 47 | 162 | +3.2% | +2.2% |
| v27 keeps 256x256x64 | cold | 401 | 328 | 21 | 52 | +3.8% | +3.6% |
| v27 picks a smaller tile | cold | 229 | 93 | 26 | 110 | -0.7% | -13.0% |
| tile 256x256x64 | cold | 401 | 328 | 21 | 52 | +3.8% | +3.6% |
| tile 128x256x64 | cold | 95 | 41 | 12 | 42 | -0.1% | -8.3% |
| tile 128x128x128 | cold | 77 | 45 | 9 | 23 | +2.0% | -11.3% |
| tile 64x128x128 | cold | 37 | 5 | 3 | 29 | -9.3% | -17.8% |
| tile 64x64x128 | cold | 20 | 2 | 2 | 16 | -9.3% | -30.5% |
| Handed to v9 | cold | 242 | 108 | 21 | 113 | -0.2% | -8.4% |
| Persistent data-parallel | cold | 197 | 175 | 13 | 9 | +3.4% | +3.1% |
| Stream-K split: reduce-scatter | cold | 12 | 12 | 0 | 0 | +16.6% | +16.0% |
| Stream-K split: owner | cold | 137 | 103 | 10 | 24 | +7.1% | +6.0% |
| Stream-K split: spread | cold | 42 | 23 | 3 | 16 | +1.5% | +4.7% |
| under one wave | cold | 321 | 137 | 32 | 152 | -0.3% | -7.5% |
| one wave or more | cold | 309 | 284 | 15 | 10 | +4.0% | +3.7% |

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

## v27 auto against the new AITER v9

| Group | Caches | Shapes | v27 faster | within 1% | v27 slower | median |
|---|---|---:|---:|---:|---:|---:|
| All shapes | cold | 630 | 358 | 209 | 63 | +1.8% |
| v27 keeps 256x256x64 | cold | 401 | 180 | 184 | 37 | +0.7% |
| v27 picks a smaller tile | cold | 229 | 178 | 25 | 26 | +8.4% |
| tile 256x256x64 | cold | 401 | 180 | 184 | 37 | +0.7% |
| tile 128x256x64 | cold | 95 | 50 | 25 | 20 | +1.1% |
| tile 128x128x128 | cold | 77 | 72 | 0 | 5 | +8.9% |
| tile 64x128x128 | cold | 37 | 36 | 0 | 1 | +13.9% |
| tile 64x64x128 | cold | 20 | 20 | 0 | 0 | +27.8% |
| Handed to v9 | cold | 242 | 124 | 87 | 31 | +1.1% |
| Persistent data-parallel | cold | 197 | 69 | 117 | 11 | +0.6% |
| Stream-K split: reduce-scatter | cold | 12 | 11 | 1 | 0 | +26.4% |
| Stream-K split: owner | cold | 137 | 114 | 3 | 20 | +6.6% |
| Stream-K split: spread | cold | 42 | 40 | 1 | 1 | +13.4% |
| under one wave | cold | 321 | 205 | 65 | 51 | +5.3% |
| one wave or more | cold | 309 | 153 | 144 | 12 | +1.0% |

## Checks of the extra columns

- cold: 0 shapes with a new v9 / v26 / v27-at-256 failure or v26 flag left; builds on Triton 3.9 1045, without sched_barriers: new v9 0, v27 and v9_tile 2; notes on 0.
