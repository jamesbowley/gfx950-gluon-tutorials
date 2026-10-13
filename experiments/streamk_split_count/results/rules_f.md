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

# Sweep G: less than one wave

Sweep G's cells (4096x8192xK, GEMM_TILE_LIMIT = S, no full wave), S <= 128: regret against the fastest of data-parallel (v9_ragged), v25 reduce-scatter and owner (fp32) at any measured n. The capped rule is replayed with S tiles in all; sweep F's rule uses version 1's constants as frozen; `fitted on sweep G` is the same family fitted on these 115 cells only (wider grid); `n <= Q / sqrt(S)` adds one cap to its reduce-scatter split, n <= Q / sqrt(S) (below one wave the best n falls as about 32 / sqrt(S), which the family's P // S and k / m cannot follow). Leave-one-out rows as above. Both are frozen in version 2, before the AITER test is re-scored.

| rule | cells | median regret | 90th percentile | worst | cells more than 2% off | approximated |
|---|---|---|---|---|---|---|
| capped (v24 / v25 host rule) | 115 | 1.002 | 1.150 | 1.404 | 44 | 20 |
| sweep F's rule, unchanged (version 1) | 115 | 1.003 | 1.148 | 1.231 | 47 | 9 |
| fitted on sweep G | 115 | 1.000 | 1.103 | 1.484 | 42 | 10 |
| fitted on sweep G, leave one S out | 115 | 1.000 | 1.139 | 1.484 | 45 | 6 |
| fitted on sweep G, leave one K out | 115 | 1.000 | 1.155 | 1.484 | 46 | 10 |
| fitted on sweep G, n <= Q / sqrt(S) | 115 | 1.000 | 1.094 | 1.168 | 40 | 27 |
| fitted on sweep G, n <= Q / sqrt(S), leave one S out | 115 | 1.000 | 1.102 | 1.276 | 43 | 23 |
| fitted on sweep G, n <= Q / sqrt(S), leave one K out | 115 | 1.000 | 1.121 | 1.404 | 43 | 27 |

Fitted on every cell of sweep G (S <= 128): T = 14, E = 4, m = 4, P = 192, c0 = 1.5, c1 = inf. With n <= Q / sqrt(S): T = 14, E = 5, m = 4, P = 256, Q = 32, c0 = 3.0, c1 = inf. Sweep F's (version 1): T = 12, E = 6, m = 4, P = 128, c0 = 3.0, c1 = inf.


## Sweep G, capped (v24 / v25 host rule): the worst cells

| S | K | regret | choice |
|---|---|---|---|
| 1 | 1024 | 1.404 | v13 n = - |
| 2 | 1024 | 1.253 | v13 n = - |
| 8 | 1024 | 1.219 | v13 n = - |
| 4 | 1024 | 1.208 | v13 n = - |
| 36 | 8192 | 1.202 | v24 owner n = 4 |
| 36 | 2048 | 1.201 | v24 owner n = 2 |
| 6 | 1024 | 1.199 | v13 n = - |
| 2 | 4096 | 1.177 | v24 rs n = 8 |

## Sweep G, sweep F's rule, unchanged (version 1): the worst cells

| S | K | regret | choice |
|---|---|---|---|
| 36 | 16384 | 1.231 | v24 rs n = 3 |
| 48 | 1024 | 1.216 | v24 owner n = 2 |
| 48 | 4096 | 1.206 | v24 owner n = default |
| 36 | 8192 | 1.202 | v24 rs n = 3 |
| 64 | 1024 | 1.196 | v24 owner n = 2 |
| 40 | 16384 | 1.191 | v24 rs n = 3 |
| 1 | 1024 | 1.168 | v24 rs n = 4 |
| 96 | 4096 | 1.163 | v24 owner n = default |

## Sweep G, fitted on sweep G: the worst cells

| S | K | regret | choice |
|---|---|---|---|
| 4 | 16384 | 1.484 | v24 rs n = 64 (nearest measured) |
| 8 | 8192 | 1.413 | v24 rs n = 32 (nearest measured) |
| 8 | 16384 | 1.307 | v24 rs n = 32 (nearest measured) |
| 6 | 8192 | 1.211 | v24 rs n = 32 |
| 1 | 1024 | 1.168 | v24 rs n = 4 |
| 96 | 4096 | 1.163 | v24 owner n = default |
| 12 | 4096 | 1.155 | v24 rs n = 16 |
| 1 | 16384 | 1.147 | v24 rs n = 64 |

## Sweep G, fitted on sweep G, n <= Q / sqrt(S): the worst cells

| S | K | regret | choice |
|---|---|---|---|
| 1 | 1024 | 1.168 | v24 rs n = 4 |
| 96 | 4096 | 1.163 | v24 owner n = default |
| 96 | 2048 | 1.138 | v24 owner n = default |
| 6 | 8192 | 1.132 | v24 rs n = 12 (nearest measured) |
| 24 | 2048 | 1.127 | v24 rs n = 8 (nearest measured) |
| 48 | 2048 | 1.121 | v24 rs n = default |
| 56 | 4096 | 1.121 | v24 owner n = default |
| 40 | 2048 | 1.120 | v24 rs n = 6 (nearest measured) |

## Sweep G, the rules' choices per cell (fitted on sweep G / capped, regret in brackets; dp = data-parallel, d = default split)

| S | K = 512 | K = 1024 | K = 2048 | K = 4096 | K = 8192 | K = 16384 | K = 32768 |
|---|---|---|---|---|---|---|---|
| 1 | dp (1.000) / dp (1.000) | rs 4 (1.168) / dp (1.404) | rs 8 (1.045) / rs 8 (1.045) | rs 16 (1.000) / rs 8 (1.162) | rs 32 (1.000) / rs 16 (1.030) | rs 64 (1.147) / rs 32 (1.000) | - |
| 2 | dp (1.000) / dp (1.000) | rs 4 (1.094) / dp (1.253) | rs 8 (1.064) / rs 8 (1.064) | rs 16 (1.000) / rs 8 (1.177) | rs 32 (1.000) / rs 16 (1.007) | rs 64 (1.081) / rs 32 (1.000) | - |
| 4 | dp (1.000) / dp (1.000) | rs 4 (1.056) / dp (1.208) | rs 8 (1.000) / rs 8 (1.000) | rs 16 (1.000) / rs 8 (1.138) | rs 32 (1.035) / rs 16 (1.000) | rs 64 (1.484) / rs 32 (1.000) | - |
| 6 | dp (1.000) / dp (1.000) | rs 4 (1.053) / dp (1.199) | rs 8 (1.013) / rs 8 (1.013) | rs 16 (1.000) / rs 8 (1.127) | rs 32 (1.211) / rs 16 (1.000) | rs 32 (1.139) / rs 16 (1.000) | - |
| 8 | dp (1.000) / dp (1.000) | rs 4 (1.044) / dp (1.219) | rs 8 (1.000) / rs 8 (1.000) | rs 16 (1.000) / rs 8 (1.011) | rs 32 (1.413) / rs 16 (1.000) | rs 32 (1.307) / rs 16 (1.000) | - |
| 12 | dp (1.000) / dp (1.000) | rs 4 (1.015) / dp (1.113) | rs 8 (1.000) / rs 8 (1.000) | rs 16 (1.155) / rs 8 (1.000) | rs 16 (1.025) / rs 12 (1.008) | rs 16 (1.053) / rs 12 (1.000) | - |
| 16 | dp (1.000) / dp (1.000) | rs 4 (1.000) / dp (1.096) | rs 8 (1.000) / rs 8 (1.000) | rs 12 (1.073) / rs 8 (1.000) | rs 12 (1.043) / rs 8 (1.000) | rs 12 (1.000) / rs 8 (1.057) | - |
| 20 | dp (1.000) / dp (1.000) | rs 4 (1.000) / dp (1.099) | rs 8 (1.037) / rs 6 (1.002) | rs 8 (1.000) / rs 6 (1.018) | rs 8 (1.000) / rs 6 (1.052) | rs 8 (1.012) / rs 6 (1.000) | - |
| 24 | dp (1.000) / dp (1.000) | rs 4 (1.000) / dp (1.050) | rs 8 (1.127) / rs 6 (1.025) | rs 8 (1.023) / rs 6 (1.000) | rs 8 (1.000) / rs 6 (1.009) | rs 8 (1.040) / rs 6 (1.000) | - |
| 28 | dp (1.000) / dp (1.000) | rs 4 (1.006) / dp (1.000) | rs 6 (1.038) / rs 4 (1.000) | rs 6 (1.000) / rs 4 (1.011) | rs 6 (1.000) / rs 4 (1.148) | rs 6 (1.000) / rs 4 (1.143) | - |
| 32 | dp (1.000) / dp (1.000) | rs 4 (1.032) / dp (1.000) | rs 6 (1.052) / rs 4 (1.000) | rs 6 (1.016) / rs 4 (1.000) | rs 6 (1.000) / rs 4 (1.110) | rs 6 (1.000) / rs 4 (1.150) | - |
| 36 | dp (1.000) / dp (1.000) | dp (1.000) / dp (1.000) | rs 6 (1.062) / owner 2 (1.201) | rs 6 (1.047) / owner 3 (1.170) | rs 6 (1.000) / owner 4 (1.202) | rs 6 (1.000) / owner 6 (1.107) | - |
| 40 | dp (1.000) / dp (1.000) | dp (1.000) / dp (1.000) | rs 4 (1.000) / owner 2 (1.147) | rs 4 (1.000) / owner 3 (1.117) | rs 4 (1.034) / owner 4 (1.137) | rs 4 (1.077) / owner d (1.120) | - |
| 48 | dp (1.000) / dp (1.000) | dp (1.000) / dp (1.000) | rs 4 (1.000) / owner 2 (1.064) | rs 4 (1.000) / owner 3 (1.074) | rs 4 (1.007) / owner 4 (1.069) | rs 4 (1.000) / owner d (1.076) | - |
| 56 | dp (1.000) / dp (1.000) | dp (1.000) / dp (1.000) | rs 3 (1.008) / owner 2 (1.006) | rs 3 (1.016) / owner 2 (1.099) | rs 3 (1.103) / owner 3 (1.050) | rs 3 (1.079) / owner d (1.048) | - |
| 64 | dp (1.000) / dp (1.000) | dp (1.000) / dp (1.000) | rs 3 (1.029) / owner 2 (1.012) | rs 3 (1.015) / owner 2 (1.073) | rs 3 (1.058) / owner 3 (1.000) | rs 3 (1.027) / owner d (1.008) | - |
| 80 | dp (1.000) / dp (1.000) | dp (1.000) / dp (1.000) | owner d (1.062) / dp (1.088) | owner d (1.011) / owner 2 (1.002) | owner d (1.002) / owner d (1.002) | owner d (1.000) / owner d (1.000) | - |
| 96 | dp (1.000) / dp (1.000) | dp (1.000) / dp (1.000) | owner d (1.138) / dp (1.038) | owner d (1.163) / owner d (1.163) | owner d (1.071) / owner d (1.071) | owner d (1.094) / owner d (1.094) | - |
| 128 | dp (1.000) / dp (1.000) | dp (1.000) / dp (1.000) | owner d (1.052) / dp (1.000) | owner d (1.010) / owner d (1.010) | owner d (1.000) / owner d (1.000) | owner d (1.000) / owner d (1.000) | owner d (1.000) / owner d (1.000) |

## Sweep G, the rules' choices per cell (fitted on sweep G, n <= Q / sqrt(S) / capped, regret in brackets; dp = data-parallel, d = default split)

| S | K = 512 | K = 1024 | K = 2048 | K = 4096 | K = 8192 | K = 16384 | K = 32768 |
|---|---|---|---|---|---|---|---|
| 1 | dp (1.000) / dp (1.000) | rs 4 (1.168) / dp (1.404) | rs 8 (1.045) / rs 8 (1.045) | rs 16 (1.000) / rs 8 (1.162) | rs 32 (1.000) / rs 16 (1.030) | rs 32 (1.000) / rs 32 (1.000) | - |
| 2 | dp (1.000) / dp (1.000) | rs 4 (1.094) / dp (1.253) | rs 8 (1.064) / rs 8 (1.064) | rs 16 (1.000) / rs 8 (1.177) | rs 32 (1.000) / rs 16 (1.007) | rs 32 (1.000) / rs 32 (1.000) | - |
| 4 | dp (1.000) / dp (1.000) | rs 4 (1.056) / dp (1.208) | rs 8 (1.000) / rs 8 (1.000) | rs 16 (1.000) / rs 8 (1.138) | rs 16 (1.000) / rs 16 (1.000) | rs 16 (1.023) / rs 32 (1.000) | - |
| 6 | dp (1.000) / dp (1.000) | rs 4 (1.053) / dp (1.199) | rs 8 (1.013) / rs 8 (1.013) | rs 12 (1.098) / rs 8 (1.127) | rs 12 (1.132) / rs 16 (1.000) | rs 12 (1.092) / rs 16 (1.000) | - |
| 8 | dp (1.000) / dp (1.000) | rs 4 (1.044) / dp (1.219) | rs 8 (1.000) / rs 8 (1.000) | rs 12 (1.021) / rs 8 (1.011) | rs 12 (1.094) / rs 16 (1.000) | rs 12 (1.010) / rs 16 (1.000) | - |
| 12 | dp (1.000) / dp (1.000) | rs 4 (1.015) / dp (1.113) | rs 8 (1.000) / rs 8 (1.000) | rs 8 (1.000) / rs 8 (1.000) | rs 8 (1.000) / rs 12 (1.008) | rs 8 (1.109) / rs 12 (1.000) | - |
| 16 | dp (1.000) / dp (1.000) | rs 4 (1.000) / dp (1.096) | rs 8 (1.000) / rs 8 (1.000) | rs 8 (1.000) / rs 8 (1.000) | rs 8 (1.000) / rs 8 (1.000) | rs 8 (1.057) / rs 8 (1.057) | - |
| 20 | dp (1.000) / dp (1.000) | rs 4 (1.000) / dp (1.099) | rs 8 (1.037) / rs 6 (1.002) | rs 8 (1.000) / rs 6 (1.018) | rs 8 (1.000) / rs 6 (1.052) | rs 8 (1.012) / rs 6 (1.000) | - |
| 24 | dp (1.000) / dp (1.000) | rs 4 (1.000) / dp (1.050) | rs 8 (1.127) / rs 6 (1.025) | rs 8 (1.023) / rs 6 (1.000) | rs 8 (1.000) / rs 6 (1.009) | rs 8 (1.040) / rs 6 (1.000) | - |
| 28 | dp (1.000) / dp (1.000) | rs 4 (1.006) / dp (1.000) | rs 6 (1.038) / rs 4 (1.000) | rs 6 (1.000) / rs 4 (1.011) | rs 6 (1.000) / rs 4 (1.148) | rs 6 (1.000) / rs 4 (1.143) | - |
| 32 | dp (1.000) / dp (1.000) | rs 4 (1.032) / dp (1.000) | rs 6 (1.052) / rs 4 (1.000) | rs 6 (1.016) / rs 4 (1.000) | rs 6 (1.000) / rs 4 (1.110) | rs 6 (1.000) / rs 4 (1.150) | - |
| 36 | dp (1.000) / dp (1.000) | dp (1.000) / dp (1.000) | rs 6 (1.062) / owner 2 (1.201) | rs 6 (1.047) / owner 3 (1.170) | rs 6 (1.000) / owner 4 (1.202) | rs 6 (1.000) / owner 6 (1.107) | - |
| 40 | dp (1.000) / dp (1.000) | dp (1.000) / dp (1.000) | rs 6 (1.120) / owner 2 (1.147) | rs 6 (1.056) / owner 3 (1.117) | rs 6 (1.000) / owner 4 (1.137) | rs 6 (1.000) / owner d (1.120) | - |
| 48 | dp (1.000) / dp (1.000) | dp (1.000) / dp (1.000) | rs d (1.121) / owner 2 (1.064) | rs d (1.054) / owner 3 (1.074) | rs d (1.000) / owner 4 (1.069) | rs d (1.065) / owner d (1.076) | - |
| 56 | dp (1.000) / dp (1.000) | dp (1.000) / dp (1.000) | owner 3 (1.000) / owner 2 (1.006) | owner d (1.121) / owner 2 (1.099) | owner d (1.067) / owner 3 (1.050) | owner d (1.048) / owner d (1.048) | - |
| 64 | dp (1.000) / dp (1.000) | dp (1.000) / dp (1.000) | owner 3 (1.003) / owner 2 (1.012) | owner d (1.075) / owner 2 (1.073) | owner d (1.026) / owner 3 (1.000) | owner d (1.008) / owner d (1.008) | - |
| 80 | dp (1.000) / dp (1.000) | dp (1.000) / dp (1.000) | owner d (1.062) / dp (1.088) | owner d (1.011) / owner 2 (1.002) | owner d (1.002) / owner d (1.002) | owner d (1.000) / owner d (1.000) | - |
| 96 | dp (1.000) / dp (1.000) | dp (1.000) / dp (1.000) | owner d (1.138) / dp (1.038) | owner d (1.163) / owner d (1.163) | owner d (1.071) / owner d (1.071) | owner d (1.094) / owner d (1.094) | - |
| 128 | dp (1.000) / dp (1.000) | dp (1.000) / dp (1.000) | owner d (1.052) / dp (1.000) | owner d (1.010) / owner d (1.010) | owner d (1.000) / owner d (1.000) | owner d (1.000) / owner d (1.000) | owner d (1.000) / owner d (1.000) |

# S > 128: spread or data-parallel

A tile-aligned split has one program per tile above 128 (S n <= 256), which is data-parallel plus overhead, so the rule's only split there is spread (end-to-end reversed cuts over all programs): spread if k >= k_s, else data-parallel. Regret against the faster of the two. Below one wave (sweep G, T = S) data-parallel is v9_ragged and spread is v25's. With one wave or more (sweep F's 15 cells, T = 256 + S: v13 and v24 spread) data-parallel is the faster of v9_ragged and v13 by sweep G's full-wave cells: v9_ragged / v13 median 1.002 (range 0.998-1.006, 15 cells), so `v13`. capped's data-parallel kernel is v9_ragged below one wave (its fallback) and v13 above.


## one wave or more (15 cells)

| rule | cells | median regret | 90th percentile | worst | cells more than 2% off | approximated |
|---|---|---|---|---|---|---|
| capped (spread if k >= 512; its data-parallel kernel) | 15 | 1.000 | 1.013 | 1.027 | 1 | 0 |
| fitted, k_s = 512 | 15 | 1.000 | 1.013 | 1.027 | 1 | 0 |
| fitted, leave one S out | 15 | 1.007 | 1.037 | 1.056 | 4 | 0 |
| fitted, leave one K out | 15 | 1.000 | 1.013 | 1.027 | 1 | 0 |

| S | K | data-parallel | spread | spread / data-parallel |
|---|---|---|---|---|
| 160 | 8192 | 308.1 | 311.8 | 1.012 |
| 160 | 16384 | 657.8 | 651.7 | 0.991 |
| 160 | 32768 | 1251.9 | 1207.3 | 0.964 |
| 192 | 8192 | 316.5 | 321.6 | 1.016 |
| 192 | 16384 | 667.2 | 649.5 | 0.973 |
| 192 | 32768 | 1282.5 | 1237.7 | 0.965 |
| 224 | 8192 | 329.9 | 342.4 | 1.038 |
| 224 | 16384 | 681.4 | 681.9 | 1.001 |
| 224 | 32768 | 1317.3 | 1306.8 | 0.992 |
| 240 | 8192 | 333.1 | 350.5 | 1.052 |
| 240 | 16384 | 696.9 | 698.8 | 1.003 |
| 240 | 32768 | 1330.9 | 1343.8 | 1.010 |
| 255 | 8192 | 342.3 | 349.7 | 1.022 |
| 255 | 16384 | 685.9 | 724.3 | 1.056 |
| 255 | 32768 | 1342.3 | 1360.0 | 1.013 |

## less than one wave (35 cells)

| rule | cells | median regret | 90th percentile | worst | cells more than 2% off | approximated |
|---|---|---|---|---|---|---|
| capped (spread if k >= 512; its data-parallel kernel) | 35 | 1.000 | 1.000 | 1.066 | 2 | 0 |
| fitted, k_s = 512 | 35 | 1.000 | 1.000 | 1.066 | 2 | 0 |
| fitted, leave one S out | 35 | 1.000 | 1.000 | 1.066 | 2 | 0 |
| fitted, leave one K out | 35 | 1.000 | 1.019 | 1.117 | 3 | 0 |

| S | K | data-parallel | spread | spread / data-parallel |
|---|---|---|---|---|
| 160 | 512 | 17.3 | 33.9 | 1.961 |
| 160 | 1024 | 25.9 | 45.4 | 1.756 |
| 160 | 2048 | 41.5 | 58.1 | 1.399 |
| 160 | 4096 | 75.4 | 86.0 | 1.141 |
| 160 | 8192 | 134.2 | 144.0 | 1.073 |
| 160 | 16384 | 304.4 | 308.4 | 1.013 |
| 160 | 32768 | 588.6 | 527.1 | 0.895 |
| 192 | 512 | 19.2 | 35.2 | 1.833 |
| 192 | 1024 | 27.2 | 41.6 | 1.529 |
| 192 | 2048 | 45.4 | 58.8 | 1.295 |
| 192 | 4096 | 81.0 | 91.8 | 1.134 |
| 192 | 8192 | 145.8 | 152.7 | 1.047 |
| 192 | 16384 | 317.1 | 297.4 | 0.938 |
| 192 | 32768 | 607.4 | 557.2 | 0.917 |
| 224 | 512 | 20.0 | 33.8 | 1.685 |
| 224 | 1024 | 28.6 | 49.5 | 1.730 |
| 224 | 2048 | 48.6 | 67.3 | 1.386 |
| 224 | 4096 | 89.2 | 103.2 | 1.157 |
| 224 | 8192 | 160.3 | 173.1 | 1.080 |
| 224 | 16384 | 332.1 | 334.4 | 1.007 |
| 224 | 32768 | 644.1 | 632.4 | 0.982 |
| 240 | 512 | 19.8 | 32.8 | 1.655 |
| 240 | 1024 | 29.6 | 44.0 | 1.488 |
| 240 | 2048 | 50.2 | 72.5 | 1.443 |
| 240 | 4096 | 91.6 | 107.2 | 1.169 |
| 240 | 8192 | 164.7 | 179.2 | 1.088 |
| 240 | 16384 | 344.5 | 351.3 | 1.020 |
| 240 | 32768 | 661.6 | 668.8 | 1.011 |
| 255 | 512 | 20.5 | 32.2 | 1.574 |
| 255 | 1024 | 30.2 | 39.8 | 1.316 |
| 255 | 2048 | 52.0 | 60.8 | 1.169 |
| 255 | 4096 | 94.4 | 102.7 | 1.088 |
| 255 | 8192 | 169.9 | 178.0 | 1.048 |
| 255 | 16384 | 350.6 | 368.7 | 1.052 |
| 255 | 32768 | 675.0 | 698.3 | 1.034 |

# S > 128 on sweep H: which spread test

Sweep H (4096x8192xK, S = 132-255, K = 2048-32768, below one wave and with one, two rounds, means): spread (v26) or data-parallel (v9_ragged below one wave, v13 with one). Regret against the faster of the two. `capped` (and version 2) spread from k >= 512; the fitted tests threshold k alone or the data-parallel last wave's idle k-steps, (256 - S) k, fitted on sweep H only (the cut between measured values that scores best, placed at the geometric mean of the values on either side), with leave-one-S-out and leave-one-K-out rows.


## one wave or more (117 cells)

| rule | cells | median regret | 90th percentile | worst | cells more than 2% off | approximated |
|---|---|---|---|---|---|---|
| capped / version 2 (k >= 512) | 117 | 1.000 | 1.034 | 1.066 | 18 | 0 |
| fitted, k >= 157 | 117 | 1.000 | 1.009 | 1.054 | 4 | 0 |
| fitted, k, leave one S out | 117 | 1.000 | 1.009 | 1.054 | 4 | 0 |
| fitted, k, leave one K out | 117 | 1.000 | 1.009 | 1.054 | 4 | 0 |
| fitted, (256 - S) k >= 10493 | 117 | 1.000 | 1.000 | 1.019 | 0 | 0 |
| fitted, (256 - S) k, leave one S out | 117 | 1.000 | 1.000 | 1.022 | 1 | 0 |
| fitted, (256 - S) k, leave one K out | 117 | 1.000 | 1.000 | 1.022 | 1 | 0 |

## less than one wave (117 cells)

| rule | cells | median regret | 90th percentile | worst | cells more than 2% off | approximated |
|---|---|---|---|---|---|---|
| capped / version 2 (k >= 512) | 117 | 1.000 | 1.063 | 1.164 | 23 | 0 |
| fitted, k >= 222 | 117 | 1.000 | 1.021 | 1.078 | 12 | 0 |
| fitted, k, leave one S out | 117 | 1.000 | 1.021 | 1.078 | 12 | 0 |
| fitted, k, leave one K out | 117 | 1.000 | 1.031 | 1.126 | 20 | 0 |
| fitted, (256 - S) k >= 10493 | 117 | 1.000 | 1.000 | 1.039 | 3 | 0 |
| fitted, (256 - S) k, leave one S out | 117 | 1.000 | 1.000 | 1.039 | 3 | 0 |
| fitted, (256 - S) k, leave one K out | 117 | 1.000 | 1.000 | 1.039 | 3 | 0 |
