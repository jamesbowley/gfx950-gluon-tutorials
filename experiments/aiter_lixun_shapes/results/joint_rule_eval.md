# Joint tile + split rule, CPU-only evaluation

Regret: the pick's time over the best point timed for the shape (median / 90th / worst / geomean, shapes above 1.05).

## Fitted parameters (geomean regret on REFIT's 198 compute-bound shapes)

- Form A: `{'form': 'A', 'k_min': 2048, 'fill': 0.625, 'owner2': True}`, geomean 1.028
- Form B: `{'form': 'B', 'k_min': 1536, 'split_us': 4.0, 'delta': 0.1, 'owner2': True}`, geomean 1.039
- Form C: `{'form': 'C', 'k_min': 1536, 'split_us': 8.0, 'per_split_us': 1.0, 'delta': 0.1, 'owner2': True}`, geomean 1.039
- Top 5 overall: 1.028 {'form': 'A', 'k_min': 2048, 'fill': 0.625, 'owner2': True}; 1.029 {'form': 'A', 'k_min': 2048, 'fill': 0.5, 'owner2': True}; 1.029 {'form': 'A', 'k_min': 2048, 'fill': 0.625, 'owner2': False}; 1.030 {'form': 'A', 'k_min': 2048, 'fill': 0.5, 'owner2': False}; 1.030 {'form': 'A', 'k_min': 3072, 'fill': 0.5, 'owner2': True}

## REFIT compute-bound (fitting set): 198 shapes

| picker | shapes | regret |
|---|---:|---|
| auto_tile + cap (current) | 198 | 1.007 / 1.204 / 1.864 / 1.053, 36 > 1.05 |
| oracle tile + cap | 198 | 1.006 / 1.045 / 1.185 / 1.016, 18 > 1.05 |
| joint rule, form A (0 picks at a nearby n) | 198 | 1.008 / 1.089 / 1.282 / 1.028, 33 > 1.05 |
|   form A: its tile, best configuration there (tile error alone) | 198 | 1.000 / 1.044 / 1.282 / 1.013, 18 > 1.05 |
|   form A: the best tile, its split (split error alone) | 198 | 1.006 / 1.091 / 2.078 / 1.031, 31 > 1.05 |
| joint rule, form B (0 picks at a nearby n) | 198 | 1.010 / 1.142 / 1.437 / 1.039, 44 > 1.05 |
|   form B: its tile, best configuration there (tile error alone) | 198 | 1.000 / 1.083 / 1.312 / 1.021, 28 > 1.05 |
|   form B: the best tile, its split (split error alone) | 198 | 1.006 / 1.091 / 1.497 / 1.027, 28 > 1.05 |
| joint rule, form C (0 picks at a nearby n) | 198 | 1.010 / 1.142 / 1.437 / 1.039, 45 > 1.05 |
|   form C: its tile, best configuration there (tile error alone) | 198 | 1.000 / 1.071 / 1.312 / 1.020, 26 > 1.05 |
|   form C: the best tile, its split (split error alone) | 198 | 1.006 / 1.091 / 1.497 / 1.027, 28 > 1.05 |

## REFIT, under one 256x256 wave: 66 shapes

| picker | shapes | regret |
|---|---:|---|
| auto_tile + cap (current) | 66 | 1.007 / 1.451 / 1.864 / 1.127, 26 > 1.05 |
| oracle tile + cap | 66 | 1.006 / 1.072 / 1.126 / 1.021, 10 > 1.05 |
| joint rule, form A (0 picks at a nearby n) | 66 | 1.026 / 1.142 / 1.282 / 1.049, 23 > 1.05 |
|   form A: its tile, best configuration there (tile error alone) | 66 | 1.000 / 1.079 / 1.282 / 1.025, 12 > 1.05 |
|   form A: the best tile, its split (split error alone) | 66 | 1.005 / 1.149 / 2.078 / 1.056, 19 > 1.05 |
| joint rule, form B (0 picks at a nearby n) | 66 | 1.053 / 1.189 / 1.437 / 1.082, 34 > 1.05 |
|   form B: its tile, best configuration there (tile error alone) | 66 | 1.019 / 1.139 / 1.312 / 1.049, 22 > 1.05 |
|   form B: the best tile, its split (split error alone) | 66 | 1.005 / 1.149 / 1.497 / 1.044, 16 > 1.05 |
| joint rule, form C (0 picks at a nearby n) | 66 | 1.054 / 1.189 / 1.437 / 1.081, 35 > 1.05 |
|   form C: its tile, best configuration there (tile error alone) | 66 | 1.011 / 1.139 / 1.312 / 1.046, 20 > 1.05 |
|   form C: the best tile, its split (split error alone) | 66 | 1.005 / 1.149 / 1.497 / 1.044, 16 > 1.05 |

## WORST10 (held out): 60 shapes

| picker | shapes | regret |
|---|---:|---|
| auto_tile + cap (current) | 60 | 1.155 / 1.377 / 1.597 / 1.179, 47 > 1.05 |
| oracle tile + cap | 60 | 1.034 / 1.102 / 1.145 / 1.044, 22 > 1.05 |
| joint rule, form A (0 picks at a nearby n) | 60 | 1.075 / 1.174 / 1.217 / 1.083, 36 > 1.05 |
|   form A: its tile, best configuration there (tile error alone) | 60 | 1.009 / 1.081 / 1.217 / 1.030, 15 > 1.05 |
|   form A: the best tile, its split (split error alone) | 60 | 1.065 / 1.276 / 1.608 / 1.108, 32 > 1.05 |
| joint rule, form B (0 picks at a nearby n) | 60 | 1.071 / 1.230 / 1.309 / 1.095, 39 > 1.05 |
|   form B: its tile, best configuration there (tile error alone) | 60 | 1.010 / 1.089 / 1.262 / 1.036, 18 > 1.05 |
|   form B: the best tile, its split (split error alone) | 60 | 1.037 / 1.152 / 1.329 / 1.060, 28 > 1.05 |
| joint rule, form C (0 picks at a nearby n) | 60 | 1.118 / 1.309 / 1.532 / 1.140, 41 > 1.05 |
|   form C: its tile, best configuration there (tile error alone) | 60 | 1.050 / 1.186 / 1.305 / 1.070, 30 > 1.05 |
|   form C: the best tile, its split (split error alone) | 60 | 1.037 / 1.152 / 1.329 / 1.060, 28 > 1.05 |

## WORST10, under one wave: 59 shapes

| picker | shapes | regret |
|---|---:|---|
| auto_tile + cap (current) | 59 | 1.156 / 1.377 / 1.597 / 1.180, 46 > 1.05 |
| oracle tile + cap | 59 | 1.034 / 1.102 / 1.145 / 1.044, 22 > 1.05 |
| joint rule, form A (0 picks at a nearby n) | 59 | 1.074 / 1.174 / 1.217 / 1.083, 35 > 1.05 |
|   form A: its tile, best configuration there (tile error alone) | 59 | 1.007 / 1.074 / 1.217 / 1.029, 14 > 1.05 |
|   form A: the best tile, its split (split error alone) | 59 | 1.064 / 1.276 / 1.608 / 1.108, 31 > 1.05 |
| joint rule, form B (0 picks at a nearby n) | 59 | 1.067 / 1.230 / 1.309 / 1.095, 38 > 1.05 |
|   form B: its tile, best configuration there (tile error alone) | 59 | 1.010 / 1.086 / 1.262 / 1.035, 17 > 1.05 |
|   form B: the best tile, its split (split error alone) | 59 | 1.036 / 1.152 / 1.329 / 1.059, 27 > 1.05 |
| joint rule, form C (0 picks at a nearby n) | 59 | 1.123 / 1.309 / 1.532 / 1.141, 40 > 1.05 |
|   form C: its tile, best configuration there (tile error alone) | 59 | 1.048 / 1.186 / 1.305 / 1.070, 29 > 1.05 |
|   form C: the best tile, its split (split error alone) | 59 | 1.036 / 1.152 / 1.329 / 1.059, 27 > 1.05 |

## Sensitivity (held-out WORST10 geomean; REFIT geomean in brackets)

- {'form': 'A', 'k_min': 2048, 'fill': 0.625, 'owner2': True}: WORST10 1.083 (REFIT 1.028)
- {'form': 'A', 'k_min': 2048, 'fill': 0.5, 'owner2': True}: WORST10 1.127 (REFIT 1.029)
- {'form': 'A', 'k_min': 2048, 'fill': 0.625, 'owner2': False}: WORST10 1.093 (REFIT 1.029)
- {'form': 'A', 'k_min': 2048, 'fill': 0.5, 'owner2': False}: WORST10 1.133 (REFIT 1.030)
- {'form': 'A', 'k_min': 3072, 'fill': 0.5, 'owner2': True}: WORST10 1.145 (REFIT 1.030)
- {'form': 'A', 'k_min': 1536, 'fill': 0.5, 'owner2': True}: WORST10 1.121 (REFIT 1.030)
- {'form': 'A', 'k_min': 3072, 'fill': 0.5, 'owner2': False}: WORST10 1.151 (REFIT 1.031)
- {'form': 'A', 'k_min': 3072, 'fill': 0.625, 'owner2': True}: WORST10 1.107 (REFIT 1.031)
- {'form': 'B', 'k_min': 1536, 'split_us': 4.0, 'delta': 0.1, 'owner2': True}: WORST10 1.095 (REFIT 1.039)
- {'form': 'B', 'k_min': 1536, 'split_us': 8.0, 'delta': 0.1, 'owner2': True}: WORST10 1.102 (REFIT 1.039)
- {'form': 'B', 'k_min': 1536, 'split_us': 4.0, 'delta': 0.1, 'owner2': False}: WORST10 1.101 (REFIT 1.040)
- {'form': 'B', 'k_min': 1536, 'split_us': 8.0, 'delta': 0.1, 'owner2': False}: WORST10 1.104 (REFIT 1.040)
- {'form': 'B', 'k_min': 2048, 'split_us': 0.0, 'delta': 0.1, 'owner2': True}: WORST10 1.130 (REFIT 1.040)
- {'form': 'B', 'k_min': 1536, 'split_us': 0.0, 'delta': 0.1, 'owner2': True}: WORST10 1.092 (REFIT 1.040)
- {'form': 'B', 'k_min': 1536, 'split_us': 16.0, 'delta': 0.1, 'owner2': True}: WORST10 1.207 (REFIT 1.040)
- {'form': 'B', 'k_min': 1536, 'split_us': 16.0, 'delta': 0.1, 'owner2': False}: WORST10 1.207 (REFIT 1.040)
- {'form': 'C', 'k_min': 1536, 'split_us': 8.0, 'per_split_us': 1.0, 'delta': 0.1, 'owner2': True}: WORST10 1.140 (REFIT 1.039)
- {'form': 'C', 'k_min': 1536, 'split_us': 2.0, 'per_split_us': 1.0, 'delta': 0.1, 'owner2': True}: WORST10 1.092 (REFIT 1.039)
- {'form': 'C', 'k_min': 1536, 'split_us': 4.0, 'per_split_us': 1.0, 'delta': 0.1, 'owner2': True}: WORST10 1.094 (REFIT 1.039)
- {'form': 'C', 'k_min': 1536, 'split_us': 8.0, 'per_split_us': 1.0, 'delta': 0.1, 'owner2': False}: WORST10 1.141 (REFIT 1.039)
- {'form': 'C', 'k_min': 1536, 'split_us': 2.0, 'per_split_us': 1.0, 'delta': 0.1, 'owner2': False}: WORST10 1.098 (REFIT 1.039)
- {'form': 'C', 'k_min': 1536, 'split_us': 4.0, 'per_split_us': 1.0, 'delta': 0.1, 'owner2': False}: WORST10 1.098 (REFIT 1.039)
- {'form': 'C', 'k_min': 1536, 'split_us': 4.0, 'per_split_us': 0.0, 'delta': 0.1, 'owner2': True}: WORST10 1.095 (REFIT 1.039)
- {'form': 'C', 'k_min': 1536, 'split_us': 8.0, 'per_split_us': 0.0, 'delta': 0.1, 'owner2': True}: WORST10 1.102 (REFIT 1.039)

## Form A split variants (geomean / shapes above 1.05)

| variant | REFIT | REFIT under one wave | WORST10 | GE512: vs AITER median, behind > 4% (covered) |
|---|---|---|---|---|
| form A as fitted (the largest split) | 1.028 / 33 | 1.049 / 23 | 1.083 / 36 | -7.9%, 77 of 107 |
| form A's tile, cap (capped's split) | 1.030 / 33 | 1.052 / 23 | 1.077 / 35 | -8.0%, 80 of 107 |
| form A's tile, the largest split up to 3 | 1.078 / 42 | 1.209 / 32 | 1.123 / 40 | -7.9%, 77 of 107 |
| form A's tile, the largest split up to 4 | 1.058 / 40 | 1.143 / 30 | 1.096 / 40 | -7.9%, 77 of 107 |
| form A's tile, the largest split up to 6 | 1.041 / 38 | 1.086 / 28 | 1.083 / 36 | -7.9%, 77 of 107 |
| form A's tile, the largest split up to 8 | 1.032 / 35 | 1.061 / 25 | 1.083 / 36 | -7.9%, 77 of 107 |
| form A with splits up to 4, also in the fill test | 1.034 / 37 | 1.064 / 27 | 1.093 / 40 | -7.9%, 77 of 107 |
| form A with splits up to 6, also in the fill test | 1.030 / 36 | 1.054 / 26 | 1.083 / 36 | -7.9%, 77 of 107 |
| form A with splits up to 8, also in the fill test | 1.027 / 32 | 1.045 / 22 | 1.083 / 36 | -7.9%, 77 of 107 |

## The split tables (WORKLOG §17, bracket lookup) as the split, at each tile rule (geomean / shapes > 1.05; a table kind not timed falls back to cap)

| tile rule | REFIT | REFIT under one wave | WORST10 |
|---|---|---|---|
| auto_tile + table split | 1.055 / 39 | 1.137 / 30 | 1.134 / 43 |
| form A's tile + table split | 1.031 / 36 | 1.062 / 27 | 1.055 / 26 |
| best tile (oracle) + table split | 1.020 / 17 | 1.041 / 11 | 1.037 / 19 |

## GE512 (167 both-sides->=512 losers), form A

- The rule's tile was timed on 107 of 167 shapes (63 at exactly its split). Untimed tiles: 256x64x64 19, 128x64x128 18, 64x128x128 11, 128x256x64 9, 64x256x64 3.
- auto_tile + cap (as run) against AITER's pick, on those 107: median -10.7%, behind by > 4% on 102, by > 10% on 57
- joint rule against AITER's pick, on those 107: median -7.9%, behind by > 4% on 77, by > 10% on 34
- best timed point against AITER's pick, on those 107: median -2.2%, behind by > 4% on 48, by > 10% on 23
- joint rule regret against the best timed point: 1.011 / 1.169 / 1.268 / 1.053, 42 > 1.05
- tiles chosen: 256x256x64 35, 256x64x64 25, 64x64x128 21, 128x64x128 19, 256x128x64 18, 128x128x128 17, 128x256x64 15, 64x128x128 14, 64x256x64 3

## GE512 (167 both-sides->=512 losers), form B

- The rule's tile was timed on 107 of 167 shapes (69 at exactly its split). Untimed tiles: 128x128x64 44, 128x64x64 11, 64x128x64 3, 64x64x128 1, 128x256x64 1.
- auto_tile + cap (as run) against AITER's pick, on those 107: median -11.6%, behind by > 4% on 102, by > 10% on 64
- joint rule against AITER's pick, on those 107: median -8.1%, behind by > 4% on 75, by > 10% on 45
- best timed point against AITER's pick, on those 107: median -0.9%, behind by > 4% on 45, by > 10% on 23
- joint rule regret against the best timed point: 1.039 / 1.174 / 1.252 / 1.063, 49 > 1.05
- tiles chosen: 128x128x64 55, 64x64x128 42, 256x256x64 30, 256x128x64 25, 128x64x64 11, 64x128x64 3, 128x256x64 1

## GE512 (167 both-sides->=512 losers), form C

- The rule's tile was timed on 101 of 167 shapes (55 at exactly its split). Untimed tiles: 128x64x64 27, 128x128x64 19, 64x64x128 13, 64x128x64 6, 128x256x64 1.
- auto_tile + cap (as run) against AITER's pick, on those 101: median -10.8%, behind by > 4% on 96, by > 10% on 55
- joint rule against AITER's pick, on those 101: median -10.2%, behind by > 4% on 86, by > 10% on 51
- best timed point against AITER's pick, on those 101: median -1.4%, behind by > 4% on 45, by > 10% on 23
- joint rule regret against the best timed point: 1.060 / 1.270 / 1.429 / 1.094, 51 > 1.05
- tiles chosen: 64x64x128 69, 256x256x64 28, 128x64x64 27, 128x128x64 24, 256x128x64 12, 64x128x64 6, 128x256x64 1
