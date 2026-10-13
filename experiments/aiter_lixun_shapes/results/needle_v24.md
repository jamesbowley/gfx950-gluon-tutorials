## v24 against v20 on the same card (this / control - 1)

| Path | Caches | Shapes | v24 vs v20: median | faster | within 1% | slower | data-parallel (forced) vs v20's |
| All shapes | warm | 630 | +0.3% | 215 | 393 | 22 | +0.0% |
| Handed to v9 | warm | 124 | -0.2% | 2 | 119 | 3 | -0.1% |
| Stream-K split | warm | 306 | +1.8% | 194 | 104 | 8 | +0.0% |
| Split: reduce-scatter | warm | 83 | +13.5% | 81 | 2 | 0 | +0.0% |
| Split: owner | warm | 221 | +1.1% | 113 | 101 | 7 | -0.0% |
| Split: spread | warm | 2 | -0.6% | 0 | 1 | 1 | -0.1% |
| Persistent data-parallel | warm | 200 | +0.1% | 19 | 170 | 11 | +0.1% |
| All shapes | cold | 630 | +0.8% | 281 | 343 | 6 | -0.0% |
| Handed to v9 | cold | 124 | -0.2% | 3 | 121 | 0 | -0.1% |
| Stream-K split | cold | 306 | +1.8% | 240 | 66 | 0 | -0.1% |
| Split: reduce-scatter | cold | 83 | +10.2% | 83 | 0 | 0 | -0.2% |
| Split: owner | cold | 221 | +1.4% | 157 | 64 | 0 | -0.0% |
| Split: spread | cold | 2 | +0.5% | 0 | 2 | 0 | -0.0% |
| Persistent data-parallel | cold | 200 | +0.0% | 38 | 156 | 6 | +0.0% |

## Results table

| Run | Caches | faster | within 1% | slower | median | under one wave: median | same-tile behind | v9: median |
| base | warm | 315 | 50 | 265 | +1.0% | -16.7% | 7 of 260 | -2.6% |
| this | warm | 359 | 38 | 233 | +2.0% | -12.3% | 2 of 262 | -0.2% |
| base | cold | 323 | 46 | 261 | +1.2% | -15.3% | 2 of 258 | -4.8% |
| this | cold | 341 | 36 | 253 | +1.6% | -12.6% | 2 of 260 | -4.4% |

## Three-way against the best library

| Caches | Kernel | faster | within 1% | slower | median | same-tile behind |
| warm | v24 (capped) | 359 | 38 | 233 | +2.0% | 2 of 262 |
| warm | AITER v9 | 281 | 44 | 305 | -0.2% | 14 of 262 |
| warm | v24 data-parallel (forced) | 279 | 41 | 310 | -0.6% | 12 of 262 |
| cold | v24 (capped) | 341 | 36 | 253 | +1.6% | 2 of 260 |
| cold | AITER v9 | 230 | 47 | 353 | -4.4% | 29 of 260 |
| cold | v24 data-parallel (forced) | 265 | 28 | 337 | -4.5% | 13 of 260 |

## Per path against the baseline (this / base - 1, median)

| Path | Caches | Shapes | v24 vs v20 | AITER v9 (card) | v24 corrected by v9 | dp | hipBLASLt | AITER pick |
| All shapes | warm | 630 | -1.5% | -1.5% | +0.1% (181 up, 78 down) | -1.4% | -2.4% | -2.3% |
| Handed to v9 | warm | 124 | -1.9% | -1.8% | -0.0% (1 up, 3 down) | -2.3% | -4.2% | -2.9% |
| Stream-K split | warm | 306 | -0.6% | -1.0% | +1.3% (174 up, 61 down) | -0.9% | -2.2% | -1.8% |
| Persistent data-parallel | warm | 200 | -1.8% | -1.7% | -0.0% (6 up, 14 down) | -1.8% | -2.2% | -2.1% |
| All shapes | cold | 630 | -0.6% | -0.9% | +0.6% (232 up, 21 down) | -0.8% | -1.0% | -0.6% |
| Handed to v9 | cold | 124 | -0.8% | -0.9% | +0.2% (8 up, 2 down) | -0.9% | -0.9% | -0.7% |
| Stream-K split | cold | 306 | +1.2% | -0.5% | +2.1% (214 up, 13 down) | -0.6% | -0.9% | -0.3% |
| Persistent data-parallel | cold | 200 | -0.9% | -1.2% | +0.2% (10 up, 6 down) | -0.9% | -1.1% | -1.1% |

warm: behind->ahead 20, ahead->behind 2; v24 vs v9: 350 faster, 18 slower, median +1.7%

cold: behind->ahead 1, ahead->behind 0; v24 vs v9: 373 faster, 16 slower, median +2.9%

## Same-tile behind

Warm: 2 behind by more than 1%, 1 behind by less (ties)

| # | Shape (MxNxK) | Output tiles | Full waves | S | k-steps | Best library | v24 config | Best TFLOPS | v9 TFLOPS | v24 TFLOPS | v24 vs best | v24 vs v9 | Cold: vs best |
|---:|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 32768x16384x2048 | 8192 | 32 | 0 | 32 | hipBLASLt | data-parallel | 1411 | 1348 | 1389 | -1.6% | +3.0% | -0.8% |
| 2 | 16384x16384x2048 | 4096 | 16 | 0 | 32 | hipBLASLt | data-parallel | 1404 | 1352 | 1385 | -1.4% | +2.4% | -0.1% |

Cold: 2 behind by more than 1%, 5 behind by less (ties)

| # | Shape (MxNxK) | Output tiles | Full waves | S | k-steps | Best library | v24 config | Best TFLOPS | v9 TFLOPS | v24 TFLOPS | v24 vs best | v24 vs v9 | Warm: vs best |
|---:|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 16384x7168x512 | 1792 | 7 | 0 | 8 | hipBLASLt | data-parallel | 981 | 897 | 965 | -1.6% | +7.6% | +7.1% |
| 2 | 1024x2304x16384 | 36 | 0 | 36 | 256 | AITER torch | one-tile owner, n = 6 | 870 | 276 | 859 | -1.3% | +210.9% | +2.1% |

## Hand-off

| Caches | Shapes | Against the persistent loop | Against AITER's v9 |
| Warm | 124 all | +1.2% (75 faster, 1 slower) | -0.2% (121 within 1%, 0 faster, 3 slower) |
| Warm | 56 at 32 k-steps or fewer | +3.0% (54 faster, 0 slower) | -0.1% (55 within 1%, 0 faster, 1 slower) |
| Warm | 68 above 32 k-steps | +0.5% (21 faster, 1 slower) | -0.2% (66 within 1%, 0 faster, 2 slower) |
| Cold | 124 all | +1.1% (64 faster, 1 slower) | -0.1% (118 within 1%, 3 faster, 3 slower) |
| Cold | 56 at 32 k-steps or fewer | +2.8% (54 faster, 0 slower) | -0.2% (54 within 1%, 1 faster, 1 slower) |
| Cold | 68 above 32 k-steps | +0.4% (10 faster, 1 slower) | -0.1% (64 within 1%, 2 faster, 2 slower) |

## Spread and correctness

warm: v24 median spread 0.54%, 76 shapes over 2%; v9 0.38%, 31 over 2% | base: warm: v24 median spread 0.40%, 24 shapes over 2%; v9 0.27%, 19 over 2%
warm: 0 shapes with a v24/v9/hipBLASLt/dp failure or flag left; AITER pick outside tolerance on 29; notes on 0
cold: v24 median spread 0.58%, 52 shapes over 2%; v9 0.42%, 54 over 2% | base: cold: v24 median spread 0.66%, 17 shapes over 2%; v9 0.29%, 11 over 2%
cold: 0 shapes with a v24/v9/hipBLASLt/dp failure or flag left; AITER pick outside tolerance on 29; notes on 0
