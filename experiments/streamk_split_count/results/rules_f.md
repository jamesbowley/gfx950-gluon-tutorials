# Host rules against sweep F

Regret = kernel time of the rule's choice / kernel time of the fastest measured choice in the cell among the configurations the rule can pick: data-parallel, v24 reduce-scatter and owner with fp32 partials (`one rounding`), or data-parallel, reduce-scatter and owner with bf16-pair partials and the atomic ending (`any rounding`). S <= 128, 90 cells of 4096x8192xK. Approximated: cells where the rule's n was not measured and the nearest measured n stood in. The leave-one-out rows fit without that S (or K) and score the held-out cells only.

| rule | cells | median regret | 90th percentile | worst | cells more than 2% off | approximated |
|---|---|---|---|---|---|---|
| capped (v24 host rule), one rounding | 90 | 1.000 | 1.044 | 1.111 | 21 | 16 |
| fitted, one rounding | 90 | 1.001 | 1.039 | 1.086 | 19 | 10 |
| fitted, one rounding, leave one S out | 90 | 1.001 | 1.058 | 1.098 | 23 | 10 |
| fitted, one rounding, leave one K out | 90 | 1.000 | 1.052 | 1.111 | 20 | 10 |
| fitted, any rounding | 90 | 1.005 | 1.047 | 1.094 | 24 | 7 |
| fitted, any rounding, leave one S out | 90 | 1.005 | 1.048 | 1.177 | 24 | 7 |
| fitted, any rounding, leave one K out | 90 | 1.007 | 1.057 | 1.094 | 29 | 7 |

Fitted on every cell. One rounding: T = 12, E = 6, m = 4, P = 128, c0 = 3.0, c1 = inf. Any rounding: T = 12, E = 3, m = 4, P = 192, c0 = 1.5, c1 = inf, A = inf. T: idle k-steps below which data-parallel runs; E: programs per tile from which reduce-scatter runs; m: k-steps per reduce-scatter program; P: programs the reduce-scatter split may use (n <= P // S); c0, c1: the owner's c = max(c0, S / c1); A: S from which the atomic ending replaces bf16 reduce-scatter.


## What the configurations no rule picks would add

Kernel time of the fastest choice among the rule's configurations / the fastest among every configuration of the same rounding class (chunk-major, l2 with the XCD-local split, v20's slice, two-tile).

| rounding | median | 90th percentile | worst | cells more than 2% |
|---|---|---|---|---|
| one | 1.003 | 1.029 | 1.048 | 13 |
| any | 1.000 | 1.007 | 1.025 | 1 |

## capped (v24 host rule), one rounding: the worst cells

| S | K | regret | choice |
|---|---|---|---|
| 1 | 1024 | 1.111 | v13 n = - |
| 2 | 1024 | 1.106 | v13 n = - |
| 3 | 1024 | 1.098 | v13 n = - |
| 4 | 1024 | 1.097 | v13 n = - |
| 6 | 1024 | 1.088 | v13 n = - |
| 8 | 1024 | 1.074 | v13 n = - |
| 12 | 1024 | 1.064 | v13 n = - |
| 1 | 4096 | 1.056 | v24 rs n = 8 |

## fitted, one rounding: the worst cells

| S | K | regret | choice |
|---|---|---|---|
| 38 | 1024 | 1.086 | v24 rs n = 3 |
| 64 | 1024 | 1.082 | v24 owner n = 2 |
| 48 | 1024 | 1.076 | v24 owner n = 2 |
| 32 | 1024 | 1.071 | v24 rs n = 4 |
| 48 | 4096 | 1.058 | v24 owner n = default |
| 1 | 1024 | 1.052 | v24 rs n = 4 |
| 38 | 16384 | 1.043 | v24 rs n = 3 |
| 2 | 1024 | 1.041 | v24 rs n = 4 |

## fitted, any rounding: the worst cells

| S | K | regret | choice |
|---|---|---|---|
| 1 | 1024 | 1.094 | v24 rs bf16_pairs n = 4 |
| 1 | 2048 | 1.076 | v24 rs bf16_pairs n = 8 |
| 2 | 1024 | 1.070 | v24 rs bf16_pairs n = 4 |
| 3 | 1024 | 1.057 | v24 rs bf16_pairs n = 4 |
| 32 | 2048 | 1.057 | v24 rs bf16_pairs n = 6 |
| 2 | 512 | 1.049 | v13 n = - |
| 48 | 1024 | 1.048 | v24 rs bf16_pairs n = 4 |
| 6 | 1024 | 1.047 | v24 rs bf16_pairs n = 4 |
