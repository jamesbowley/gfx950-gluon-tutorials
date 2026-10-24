# Phase 0: the two-stage plan checked on REFIT (CPU only)

REFIT (results/refit.csv): 350 shapes, every tile and method, 2-3 rounds, other GPUs and days. Used to steer the new sweep only. Ratios are within-block, so GPU offsets cancel. Regret columns: median / 90th / worst / geomean / shapes above 1 + sigma (sigma = 1.89%) and above 1.15.

## 1. Tile/method interaction

**compute-bound** (198 shapes)

| selection | regret against the joint best |
|---|---|
| tile chosen under cap, then best method at it | 1.000 / 1.020 / 1.179 / 1.006 / 21 > sigma, 1 > 1.15 |
| cap only (tile chosen under cap) | 1.006 / 1.045 / 1.187 / 1.016 / 47 > sigma, 1 > 1.15 |
| tile chosen under dp, then best method at it | 1.000 / 1.229 / 1.799 / 1.046 / 41 > sigma, 26 > 1.15 |
| dp only (tile chosen under dp) | 1.057 / 1.428 / 2.867 / 1.131 / 138 > sigma, 45 > 1.15 |
| tile chosen under pdp, then best method at it | 1.000 / 1.180 / 1.864 / 1.043 / 42 > sigma, 23 > 1.15 |
| pdp only (tile chosen under pdp) | 1.065 / 1.432 / 2.891 / 1.143 / 150 > sigma, 49 > 1.15 |
| each tile at its predicted method (kNN in S, k; shape left out), best tile among those | 1.001 / 1.015 / 1.056 / 1.005 / 17 > sigma, 0 > 1.15 |
| the same, kNN also in log M/N | 1.002 / 1.014 / 1.260 / 1.006 / 15 > sigma, 1 > 1.15 |

Shapes whose best tile under that method differs from the joint best tile: cap 33/198, dp 45/198, pdp 51/198

**memory-bound** (152 shapes)

| selection | regret against the joint best |
|---|---|
| tile chosen under cap, then best method at it | 1.000 / 1.079 / 1.202 / 1.018 / 33 > sigma, 2 > 1.15 |
| cap only (tile chosen under cap) | 1.013 / 1.145 / 1.264 / 1.045 / 70 > sigma, 12 > 1.15 |
| tile chosen under dp, then best method at it | 1.000 / 1.158 / 1.777 / 1.040 / 35 > sigma, 17 > 1.15 |
| dp only (tile chosen under dp) | 1.008 / 1.606 / 3.015 / 1.153 / 70 > sigma, 47 > 1.15 |
| tile chosen under pdp, then best method at it | 1.000 / 1.156 / 1.777 / 1.041 / 40 > sigma, 17 > 1.15 |
| pdp only (tile chosen under pdp) | 1.078 / 1.655 / 3.082 / 1.195 / 126 > sigma, 55 > 1.15 |
| each tile at its predicted method (kNN in S, k; shape left out), best tile among those | 1.002 / 1.038 / 1.120 / 1.012 / 37 > sigma, 0 > 1.15 |
| the same, kNN also in log M/N | 1.004 / 1.040 / 1.158 / 1.015 / 37 > sigma, 1 > 1.15 |

Shapes whose best tile under that method differs from the joint best tile: cap 53/152, dp 55/152, pdp 57/152

## 2. Does the best method depend only on (tile, waves, S, k)?

cfg regret of the method predicted from the 3 nearest other shapes at the same tile and wave class, on tiles within 1.10 of the shape's best.

| tile class | waves | pairs | kNN in S, k | kNN in S, k, M/N |
|---|---|---|---|---|
| quadrant | 0 | 375 | 1.002 / 1.065 / 1.316 / 1.019 / 107 > sigma, 3 > 1.15 | 1.003 / 1.073 / 1.316 / 1.021 / 114 > sigma, 8 > 1.15 |
| quadrant | 1-3 | 130 | 1.000 / 1.021 / 1.181 / 1.008 / 16 > sigma, 1 > 1.15 | 1.000 / 1.022 / 1.151 / 1.008 / 15 > sigma, 1 > 1.15 |
| quadrant | 4+ | 132 | 1.001 / 1.013 / 1.055 / 1.005 / 5 > sigma, 0 > 1.15 | 1.001 / 1.013 / 1.044 / 1.005 / 5 > sigma, 0 > 1.15 |
| small | 0 | 478 | 1.005 / 1.102 / 1.298 / 1.030 / 183 > sigma, 16 > 1.15 | 1.007 / 1.118 / 1.298 / 1.037 / 198 > sigma, 25 > 1.15 |
| small | 1-3 | 21 | 1.004 / 1.040 / 1.065 / 1.013 / 6 > sigma, 0 > 1.15 | 1.006 / 1.055 / 1.128 / 1.019 / 6 > sigma, 0 > 1.15 |

## 3. Tile pruning (greedy covers)

- **compute-bound, within 1.05:** 256x256x64 (151), 128x128x64 (25), 64x64x128 (11), 64x128x128 (6), 256x128x64 (4), 128x128x128 (1)
- **compute-bound, within 1.1:** 256x256x64 (158), 128x128x64 (25), 64x64x128 (11), 64x128x64 (4)
- **memory-bound, within 1.05:** 16x16x256 (95), 32x32x128 (20), 128x128x128 (11), 64x128x128 (8), 64x64x128 (5), 128x256x64 (4), 16x32x128 (3), 128x64x64 (2), 32x256x64 (1), 32x128x128 (1), 256x256x64 (1), 16x16x128 (1)
- **memory-bound, within 1.1:** 16x16x256 (103), 64x64x128 (19), 128x128x64 (15), 32x32x128 (8), 64x128x128 (2), 32x128x128 (2), 256x256x64 (2), 128x128x128 (1)

## 4. Method usage (tiles within 1.10 of the shape's best)

**quadrant** (637 (shape, tile) pairs)

| kind | timed | within sigma | wins |
|---|---|---|---|
| cap | 637 | 353 | 140 |
| dp | 637 | 219 | 118 |
| rsall | 438 | 184 | 75 |
| pdp | 637 | 113 | 38 |
| rs4 | 271 | 90 | 38 |
| rs5 | 231 | 76 | 25 |
| rs6 | 184 | 63 | 14 |
| owall | 438 | 61 | 13 |
| 2t | 253 | 59 | 19 |
| rs8 | 124 | 58 | 21 |
| ow2 | 432 | 56 | 23 |
| rs3 | 362 | 52 | 16 |
| rs7 | 167 | 52 | 10 |
| rs2 | 432 | 50 | 6 |
| sp | 561 | 47 | 24 |
| ow3 | 362 | 43 | 11 |
| rs10 | 98 | 43 | 8 |
| rs16 | 46 | 41 | 13 |
| rs12 | 75 | 35 | 8 |
| pdp512 | 216 | 27 | 7 |
| ow4 | 271 | 24 | 1 |
| rs24 | 22 | 16 | 3 |
| ow5 | 231 | 15 | 1 |
| ow6 | 184 | 14 | 0 |
| ow7 | 167 | 13 | 0 |
| ow8 | 124 | 13 | 0 |
| ow10 | 98 | 10 | 0 |
| ow12 | 75 | 9 | 0 |
| rs32 | 8 | 6 | 3 |
| ow16 | 46 | 5 | 0 |
| ow24 | 22 | 2 | 0 |
| pdp1024 | 14 | 2 | 1 |
| pdp768 | 14 | 2 | 1 |
| ow32 | 8 | 0 | 0 |

**small** (499 (shape, tile) pairs)

| kind | timed | within sigma | wins |
|---|---|---|---|
| cap | 499 | 193 | 88 |
| dp | 499 | 189 | 108 |
| rsall | 435 | 113 | 59 |
| pdp | 499 | 84 | 19 |
| rs8 | 127 | 51 | 40 |
| rs3 | 354 | 46 | 27 |
| rs6 | 203 | 38 | 26 |
| rs5 | 232 | 36 | 14 |
| rs7 | 181 | 35 | 23 |
| rs4 | 250 | 34 | 26 |
| rs2 | 413 | 32 | 8 |
| ow2 | 413 | 25 | 17 |
| ow3 | 354 | 23 | 11 |
| rs16 | 32 | 18 | 15 |
| owall | 435 | 16 | 3 |
| rs12 | 90 | 11 | 6 |
| rs10 | 111 | 10 | 5 |
| sp | 375 | 7 | 2 |
| ow5 | 232 | 7 | 0 |
| ow4 | 250 | 5 | 2 |
| 2t | 20 | 3 | 0 |
| ow6 | 203 | 1 | 0 |
| ow7 | 181 | 0 | 0 |
| ow8 | 127 | 0 | 0 |
| ow10 | 111 | 0 | 0 |
| ow12 | 90 | 0 | 0 |
| ow16 | 32 | 0 | 0 |
| rs24 | 21 | 0 | 0 |
| ow24 | 21 | 0 | 0 |
| rs32 | 2 | 0 | 0 |
| ow32 | 2 | 0 | 0 |

**Where 2t, sp and owner endings are within sigma of the best** (cells: tile class, waves, S floor pow2, k-steps nearest pow2: count)

- **2t:** within sigma on 62 of 273 pairs; S/P min/median/max 0.016, 0.078, 0.672; top cells quadrant w1-3 S~8 k~512: 3; quadrant w1-3 S~32 k~32: 3; quadrant w1-3 S~16 k~128: 3; quadrant w4+ S~32 k~32: 3; quadrant w4+ S~8 k~512: 3; quadrant w4+ S~32 k~128: 3; quadrant w4+ S~16 k~64: 3; quadrant w1-3 S~8 k~256: 2; quadrant w1-3 S~32 k~64: 2; quadrant w4+ S~8 k~256: 2; quadrant w4+ S~4 k~128: 2; quadrant w4+ S~16 k~8: 2
- **sp:** within sigma on 54 of 936 pairs; S/P min/median/max 0.082, 0.602, 0.980; top cells quadrant w0 S~128 k~128: 9; quadrant w0 S~128 k~64: 7; quadrant w0 S~128 k~256: 3; quadrant w0 S~128 k~512: 3; quadrant w1-3 S~128 k~256: 3; quadrant w0 S~128 k~32: 2; quadrant w1-3 S~128 k~512: 2; quadrant w4+ S~128 k~512: 2; quadrant w4+ S~128 k~64: 2; quadrant w4+ S~128 k~128: 2; small w0 S~256 k~128: 2; quadrant w0 S~64 k~32: 1
- **ow:** within sigma on 175 of 873 pairs; S/P min/median/max 0.016, 0.223, 0.492; top cells quadrant w0 S~64 k~128: 29; small w0 S~64 k~64: 16; quadrant w0 S~64 k~256: 15; quadrant w4+ S~8 k~512: 14; quadrant w4+ S~8 k~256: 14; small w0 S~64 k~128: 14; quadrant w0 S~64 k~64: 13; quadrant w1-3 S~32 k~256: 12; quadrant w4+ S~16 k~512: 12; quadrant w4+ S~4 k~128: 11; quadrant w4+ S~32 k~128: 11; quadrant w4+ S~16 k~256: 11

## 5. Coarse-then-refine n

Coarse n (2, 4, 8, 16, 32) and all, then the explicit n between the coarse best's neighbours.

| ending | pairs | regret against the family's best n | share of n timings saved |
|---|---|---|---|
| rs | 521 | 1.000 / 1.000 / 1.145 / 1.002 / 19 > sigma, 0 > 1.15 | 18% |
| ow | 521 | 1.000 / 1.000 / 1.164 / 1.005 / 32 > sigma, 2 > 1.15 | 20% |

## 6. BK transfer (same shape, the other BK's best method)

| tiles | pairs | cfg regret |
|---|---|---|
| 128x128x128 <-> 128x128x64 | 655 | 1.016 / 1.134 / 1.301 / 1.043 / 315 > sigma, 39 > 1.15 |
| 64x128x128 <-> 64x128x64 | 678 | 1.019 / 1.172 / 1.298 / 1.054 / 336 > sigma, 98 > 1.15 |
| 64x64x128 <-> 64x64x64 | 667 | 1.000 / 1.092 / 1.741 / 1.028 / 174 > sigma, 35 > 1.15 |
| 128x64x128 <-> 128x64x64 | 678 | 1.021 / 1.152 / 1.276 / 1.050 / 345 > sigma, 72 > 1.15 |
| 16x16x256 <-> 16x16x128 | 287 | 1.005 / 1.110 / 1.454 / 1.033 / 93 > sigma, 19 > 1.15 |
| 32x256x128 <-> 32x256x64 | 285 | 1.000 / 1.094 / 1.253 / 1.024 / 79 > sigma, 13 > 1.15 |
| 16x128x128 <-> 16x128x64 | 291 | 1.002 / 1.102 / 1.411 / 1.031 / 101 > sigma, 15 > 1.15 |

## 7. Where the method matters (tiles within 1.10 of the shape's best)

Geomean / 90th percentile of the method's time over the tile's best method.

| tile class | waves | cap | dp | pdp |
|---|---|---|---|---|
| quadrant | 0 | 1.049 / 1.151 (375) | 1.847 / 4.611 (375) | 1.890 / 4.600 (375) |
| quadrant | 1-3 | 1.052 / 1.165 (130) | 1.093 / 1.339 (130) | 1.131 / 1.356 (130) |
| quadrant | 4+ | 1.039 / 1.148 (132) | 1.044 / 1.097 (132) | 1.060 / 1.151 (132) |
| small | 0 | 1.140 / 1.446 (478) | 1.438 / 2.766 (478) | 1.482 / 2.790 (478) |
| small | 1-3 | 1.024 / 1.058 (21) | 1.051 / 1.125 (21) | 1.018 / 1.033 (21) |

## 8. Noise model (|rel round 1 / rel round 2 - 1|, within-block)

| tile class | kernel time (us) | points | median | 90th | 99th |
|---|---|---|---|---|---|
| quadrant | <15 | 10839 | 0.68% | 2.05% | 6.69% |
| quadrant | <30 | 14765 | 0.57% | 1.81% | 4.84% |
| quadrant | <60 | 9991 | 0.53% | 1.99% | 6.21% |
| quadrant | <150 | 8818 | 0.62% | 2.19% | 5.45% |
| quadrant | <500 | 6983 | 0.57% | 1.75% | 3.83% |
| quadrant | >=500 | 10034 | 0.58% | 1.88% | 3.73% |
| small | <15 | 14111 | 0.81% | 2.72% | 9.35% |
| small | <30 | 7439 | 0.63% | 2.18% | 6.66% |
| small | <60 | 3325 | 0.55% | 2.48% | 8.95% |
| small | <150 | 1933 | 0.50% | 2.19% | 7.54% |
| small | <500 | 698 | 0.41% | 1.07% | 2.73% |
| small | >=500 | 97 | 0.28% | 0.65% | 1.46% |

## 9. Racing replay on REFIT's rounds

Per (shape, tile): round 1 for every method, survivors within leader x (1 + z sigma90 + delta) get round 2, close survivors round 3. Regret against the merged best on tiles within 1.10 of the shape's best; points against timing everything twice.

| z | delta | points used | geomean regret | 90th | worst | > sigma |
|---|---|---|---|---|---|---|
| 1.0 | 0.00 | 62% | 1.0004 | 1.0000 | 1.0947 | 6 / 1136 |
| 1.0 | 0.01 | 64% | 1.0002 | 1.0000 | 1.0947 | 3 / 1136 |
| 1.0 | 0.02 | 65% | 1.0002 | 1.0000 | 1.0420 | 2 / 1136 |
| 1.0 | 0.03 | 67% | 1.0001 | 1.0000 | 1.0420 | 2 / 1136 |
| 1.5 | 0.00 | 64% | 1.0002 | 1.0000 | 1.0947 | 3 / 1136 |
| 1.5 | 0.01 | 65% | 1.0002 | 1.0000 | 1.0420 | 2 / 1136 |
| 1.5 | 0.02 | 67% | 1.0001 | 1.0000 | 1.0420 | 2 / 1136 |
| 1.5 | 0.03 | 68% | 1.0001 | 1.0000 | 1.0420 | 2 / 1136 |
| 2.0 | 0.00 | 66% | 1.0002 | 1.0000 | 1.0420 | 2 / 1136 |
| 2.0 | 0.01 | 67% | 1.0001 | 1.0000 | 1.0420 | 2 / 1136 |
| 2.0 | 0.02 | 68% | 1.0001 | 1.0000 | 1.0420 | 2 / 1136 |
| 2.0 | 0.03 | 69% | 1.0001 | 1.0000 | 1.0420 | 2 / 1136 |
| 3.0 | 0.00 | 68% | 1.0001 | 1.0000 | 1.0420 | 2 / 1136 |
| 3.0 | 0.01 | 69% | 1.0001 | 1.0000 | 1.0420 | 2 / 1136 |
| 3.0 | 0.02 | 70% | 1.0001 | 1.0000 | 1.0420 | 2 / 1136 |
| 3.0 | 0.03 | 71% | 1.0001 | 1.0000 | 1.0420 | 2 / 1136 |

