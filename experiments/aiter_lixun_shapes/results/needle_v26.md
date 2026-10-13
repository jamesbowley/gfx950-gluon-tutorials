## v26 against v24 on the same card (this / control - 1)

| Path | Caches | Shapes | v26 vs v24: median | faster | within 1% | slower | data-parallel (forced) vs v24's |
| All shapes | warm | 630 | +0.0% | 129 | 398 | 103 | -0.0% |
| Handed to v9 | warm | 90 | -0.2% | 0 | 88 | 2 | -0.0% |
| Stream-K split | warm | 343 | +0.1% | 112 | 141 | 90 | -0.1% |
| Split: reduce-scatter | warm | 163 | +0.1% | 71 | 30 | 62 | -0.1% |
| Split: owner | warm | 158 | +0.1% | 29 | 105 | 24 | -0.1% |
| Split: spread | warm | 22 | +1.3% | 12 | 6 | 4 | -0.1% |
| Persistent data-parallel | warm | 197 | +0.0% | 17 | 169 | 11 | +0.0% |
| Same configuration: data-parallel | warm | 197 | +0.0% | 17 | 169 | 11 | +0.0% |
| Same configuration: handed to v9 | warm | 90 | -0.2% | 0 | 88 | 2 | -0.0% |
| Same configuration: owner | warm | 138 | +0.1% | 19 | 102 | 17 | -0.1% |
| Same configuration: reduce-scatter | warm | 12 | -0.0% | 1 | 11 | 0 | -0.3% |
| Same configuration: spread | warm | 2 | -0.1% | 0 | 2 | 0 | +0.2% |
| Changed: data-parallel -> spread | warm | 3 | +5.3% | 3 | 0 | 0 | +0.8% |
| Changed: handed to v9 -> owner | warm | 9 | +0.2% | 2 | 3 | 4 | -0.3% |
| Changed: handed to v9 -> reduce-scatter | warm | 8 | -2.7% | 0 | 3 | 5 | +0.4% |
| Changed: handed to v9 -> spread | warm | 17 | +1.0% | 9 | 4 | 4 | -0.5% |
| Changed: owner -> reduce-scatter | warm | 72 | +4.8% | 53 | 6 | 13 | -0.0% |
| Changed: owner, another split count | warm | 11 | +1.7% | 8 | 0 | 3 | +0.3% |
| Changed: reduce-scatter, another split count | warm | 71 | -2.3% | 17 | 10 | 44 | -0.1% |
| All shapes | cold | 630 | +0.2% | 185 | 383 | 62 | +0.0% |
| Handed to v9 | cold | 90 | -0.2% | 3 | 83 | 4 | +0.0% |
| Stream-K split | cold | 343 | +0.5% | 146 | 146 | 51 | +0.0% |
| Split: reduce-scatter | cold | 163 | +3.2% | 103 | 27 | 33 | -0.1% |
| Split: owner | cold | 158 | +0.1% | 24 | 117 | 17 | +0.2% |
| Split: spread | cold | 22 | +7.2% | 19 | 2 | 1 | +0.0% |
| Persistent data-parallel | cold | 197 | +0.0% | 36 | 154 | 7 | +0.0% |
| Same configuration: data-parallel | cold | 197 | +0.0% | 36 | 154 | 7 | +0.0% |
| Same configuration: handed to v9 | cold | 90 | -0.2% | 3 | 83 | 4 | +0.0% |
| Same configuration: owner | cold | 138 | +0.1% | 12 | 115 | 11 | +0.2% |
| Same configuration: reduce-scatter | cold | 12 | +0.4% | 3 | 9 | 0 | +0.3% |
| Same configuration: spread | cold | 2 | -0.2% | 0 | 2 | 0 | -0.2% |
| Changed: data-parallel -> spread | cold | 3 | +4.8% | 3 | 0 | 0 | -0.1% |
| Changed: handed to v9 -> owner | cold | 9 | -6.3% | 3 | 1 | 5 | +0.1% |
| Changed: handed to v9 -> reduce-scatter | cold | 8 | +4.8% | 5 | 2 | 1 | -0.2% |
| Changed: handed to v9 -> spread | cold | 17 | +8.4% | 16 | 0 | 1 | +0.2% |
| Changed: owner -> reduce-scatter | cold | 72 | +9.7% | 57 | 7 | 8 | -0.1% |
| Changed: owner, another split count | cold | 11 | +2.2% | 9 | 1 | 1 | -0.3% |
| Changed: reduce-scatter, another split count | cold | 71 | +1.9% | 38 | 9 | 24 | -0.1% |

## Results table

| Run | Caches | faster | within 1% | slower | median | under one wave: median | same-tile behind | v9: median |
| base | warm | 359 | 38 | 233 | +2.0% | -12.3% | 2 of 262 | -0.2% |
| this | warm | 365 | 38 | 227 | +2.2% | -9.1% | 2 of 261 | -0.1% |
| base | cold | 341 | 36 | 253 | +1.6% | -12.6% | 2 of 260 | -4.4% |
| this | cold | 354 | 43 | 233 | +1.8% | -8.2% | 1 of 260 | -4.1% |

## Three-way against the best library

| Caches | Kernel | faster | within 1% | slower | median | same-tile behind |
| warm | v26 (capped) | 365 | 38 | 227 | +2.2% | 2 of 261 |
| warm | AITER v9 | 283 | 41 | 306 | -0.1% | 14 of 261 |
| warm | v26 data-parallel (forced) | 280 | 42 | 308 | -0.3% | 12 of 261 |
| cold | v26 (capped) | 354 | 43 | 233 | +1.8% | 1 of 260 |
| cold | AITER v9 | 223 | 58 | 349 | -4.1% | 28 of 260 |
| cold | v26 data-parallel (forced) | 260 | 32 | 338 | -4.0% | 13 of 260 |

## Per path against the baseline (this / base - 1, median)

| Path | Caches | Shapes | v26 vs base | AITER v9 (card) | v26 corrected by v9 | dp | hipBLASLt | AITER pick |
| All shapes | warm | 630 | +0.0% | +0.0% | +0.1% (130 up, 95 down) | -0.0% | +0.0% | -0.0% |
| Handed to v9 | warm | 90 | -0.1% | -0.0% | -0.0% (1 up, 1 down) | +0.0% | -0.0% | -0.0% |
| Stream-K split | warm | 343 | +0.3% | +0.0% | +0.4% (123 up, 85 down) | +0.0% | +0.0% | +0.0% |
| Persistent data-parallel | warm | 197 | -0.0% | -0.0% | -0.0% (6 up, 9 down) | -0.0% | -0.0% | -0.0% |
| All shapes | cold | 630 | +0.1% | +0.1% | +0.0% (147 up, 71 down) | +0.0% | +0.0% | +0.1% |
| Handed to v9 | cold | 90 | +0.0% | +0.1% | -0.1% (0 up, 6 down) | +0.1% | +0.0% | +0.1% |
| Stream-K split | cold | 343 | +0.3% | +0.0% | +0.3% (141 up, 55 down) | +0.0% | +0.0% | +0.1% |
| Persistent data-parallel | cold | 197 | +0.0% | +0.1% | -0.0% (6 up, 10 down) | +0.0% | +0.1% | +0.1% |

warm: behind->ahead 6, ahead->behind 2; v26 vs v9: 360 faster, 30 slower, median +2.4%

cold: behind->ahead 17, ahead->behind 1; v26 vs v9: 392 faster, 29 slower, median +4.5%

## Same-tile behind

Warm: 2 behind by more than 1%, 1 behind by less (ties)

| # | Shape (MxNxK) | Output tiles | Full waves | S | k-steps | Best library | v26 config | Best TFLOPS | v9 TFLOPS | v26 TFLOPS | v26 vs best | v26 vs v9 | Cold: vs best |
|---:|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 32768x16384x2048 | 8192 | 32 | 0 | 32 | hipBLASLt | data-parallel | 1411 | 1350 | 1390 | -1.5% | +3.0% | -0.8% |
| 2 | 16384x16384x2048 | 4096 | 16 | 0 | 32 | hipBLASLt | data-parallel | 1399 | 1352 | 1384 | -1.1% | +2.3% | -0.0% |

Cold: 1 behind by more than 1%, 6 behind by less (ties)

| # | Shape (MxNxK) | Output tiles | Full waves | S | k-steps | Best library | v26 config | Best TFLOPS | v9 TFLOPS | v26 TFLOPS | v26 vs best | v26 vs v9 | Warm: vs best |
|---:|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 16384x7168x512 | 1792 | 7 | 0 | 8 | hipBLASLt | data-parallel | 981 | 897 | 964 | -1.7% | +7.4% | +7.1% |

## Hand-off

| Caches | Shapes | Against the persistent loop | Against AITER's v9 |
| Warm | 90 all | +1.3% (50 faster, 1 slower) | -0.2% (88 within 1%, 1 faster, 1 slower) |
| Warm | 40 at 32 k-steps or fewer | +2.8% (38 faster, 0 slower) | -0.1% (40 within 1%, 0 faster, 0 slower) |
| Warm | 50 above 32 k-steps | +0.6% (12 faster, 1 slower) | -0.2% (48 within 1%, 1 faster, 1 slower) |
| Cold | 90 all | +0.7% (39 faster, 1 slower) | -0.3% (82 within 1%, 0 faster, 8 slower) |
| Cold | 40 at 32 k-steps or fewer | +2.5% (35 faster, 0 slower) | -0.4% (34 within 1%, 0 faster, 6 slower) |
| Cold | 50 above 32 k-steps | +0.3% (4 faster, 1 slower) | -0.2% (48 within 1%, 0 faster, 2 slower) |

## Spread and correctness

warm: v26 median spread 0.48%, 58 shapes over 2%; v9 0.29%, 22 over 2% | base: warm: v26 median spread 0.54%, 76 shapes over 2%; v9 0.38%, 31 over 2%
warm: 0 shapes with a v26/v9/hipBLASLt/dp failure or flag left; AITER pick outside tolerance on 29; notes on 0
cold: v26 median spread 0.52%, 46 shapes over 2%; v9 0.31%, 8 over 2% | base: cold: v26 median spread 0.58%, 52 shapes over 2%; v9 0.42%, 54 over 2%
cold: 0 shapes with a v26/v9/hipBLASLt/dp failure or flag left; AITER pick outside tolerance on 29; notes on 0
