# Spread or data-parallel on ragged shapes (S > 128, v26)

Sweeps G and H, and every AITER shape, have M and N multiples of 256 and K of 128, so v26's spread test
((256 - S) k >= 10493) was never fitted or tested with its edge builds, which spill more (EDGE_M / EDGE_N,
and more with a bias). Real shapes (no GEMM_TILE_LIMIT) of S tiles (less than one wave) or 256 + S (one full
wave first), S = 136-248, K = 4096-32768, in four variants of the same tile grid: aligned, M and N ragged
(M = 256 tm - 72, N = 256 tn - 48; v26 sets both edge flags when either is ragged), K ragged (K - 112, EDGE_K),
and all three; each with and without a bias. Per shape: v26 spread and data-parallel as deployed (v9_ragged
below one wave, v26's persistent loop with a full wave). rocprof as the sweeps (1000 launches over rotating
inputs, median, B and C rows padded as bench.py pads them), one round.

Cells: spread / data-parallel (below 1: spread faster). Bold: v26's rule ((256 - S) k >= 10493) picks spread.


## Less than one wave, without bias

| variant | S | K = 4096 | K = 8192 | K = 16384 | K = 32768 |
|---|---|---|---|---|---|
| aligned | 136 | 0.993 | **0.858** | **0.850** | **0.867** |
| aligned | 160 | 1.060 | **0.964** | **0.938** | **0.930** |
| aligned | 192 | 1.074 | 0.981 | **0.880** | **0.906** |
| aligned | 224 | 1.118 | 1.045 | 1.046 | **0.979** |
| aligned | 248 | 1.172 | 1.077 | 1.032 | 1.014 |
| MN ragged | 136 | 1.058 | **0.949** | **0.955** | **1.034** |
| MN ragged | 160 | 1.132 | **1.001** | **1.083** | **1.062** |
| MN ragged | 192 | 1.115 | 1.010 | **0.976** | **0.938** |
| MN ragged | 224 | 1.153 | 1.068 | 1.072 | **0.996** |
| MN ragged | 248 | 1.195 | 1.103 | 1.063 | 1.033 |
| K ragged | 136 | 0.739 | **0.645** | **0.673** | **0.657** |
| K ragged | 160 | 0.830 | **0.747** | **0.776** | **0.801** |
| K ragged | 192 | 0.931 | 0.857 | **0.811** | **0.815** |
| K ragged | 224 | 0.982 | 0.871 | 0.955 | **0.889** |
| K ragged | 248 | 1.171 | 1.064 | 0.934 | 0.915 |
| MNK ragged | 136 | 0.828 | **0.718** | **0.742** | **0.721** |
| MNK ragged | 160 | 0.902 | **0.813** | **0.830** | **0.908** |
| MNK ragged | 192 | 1.002 | 0.903 | **0.870** | **0.846** |
| MNK ragged | 224 | 1.045 | 0.948 | 0.992 | **0.946** |
| MNK ragged | 248 | 1.204 | 1.098 | 0.994 | 0.979 |

## Less than one wave, with bias

| variant | S | K = 4096 | K = 8192 | K = 16384 | K = 32768 |
|---|---|---|---|---|---|
| aligned | 136 | 1.021 | **0.872** | **0.853** | **0.874** |
| aligned | 160 | 1.110 | **0.988** | **0.958** | **0.941** |
| aligned | 192 | 1.133 | 1.002 | **0.884** | **0.916** |
| aligned | 224 | 1.153 | 1.054 | 1.052 | **0.982** |
| aligned | 248 | 1.198 | 1.086 | 1.047 | 1.024 |
| MN ragged | 136 | 1.127 | **0.982** | **0.971** | **1.037** |
| MN ragged | 160 | 1.184 | **1.022** | **1.110** | **1.026** |
| MN ragged | 192 | 1.158 | 1.032 | **0.989** | **0.954** |
| MN ragged | 224 | 1.203 | 1.086 | 1.071 | **0.989** |
| MN ragged | 248 | 1.241 | 1.122 | 1.069 | 1.030 |
| K ragged | 136 | 0.760 | **0.651** | **0.676** | **0.661** |
| K ragged | 160 | 0.839 | **0.756** | **0.793** | **0.804** |
| K ragged | 192 | 0.947 | 0.860 | **0.829** | **0.814** |
| K ragged | 224 | 1.012 | 0.886 | 0.981 | **0.888** |
| K ragged | 248 | 1.199 | 1.096 | 0.950 | 0.914 |
| MNK ragged | 136 | 0.882 | **0.743** | **0.757** | **0.721** |
| MNK ragged | 160 | 0.943 | **0.841** | **0.859** | **0.874** |
| MNK ragged | 192 | 1.050 | 0.930 | **0.885** | **0.852** |
| MNK ragged | 224 | 1.102 | 0.967 | 0.985 | **0.958** |
| MNK ragged | 248 | 1.251 | 1.127 | 1.009 | 0.986 |

## One full wave, then S, without bias

| variant | S | K = 4096 | K = 8192 | K = 16384 | K = 32768 |
|---|---|---|---|---|---|
| aligned | 136 | 1.020 | **0.970** | **0.953** | **0.981** |
| aligned | 160 | 1.012 | **0.975** | **1.016** | **0.949** |
| aligned | 192 | 1.021 | 0.986 | **0.980** | **0.952** |
| aligned | 224 | 1.059 | 1.026 | 0.990 | **0.994** |
| aligned | 248 | 1.073 | 1.044 | 0.996 | 1.011 |
| MN ragged | 136 | 1.062 | **0.985** | **0.975** | **0.984** |
| MN ragged | 160 | 1.045 | **0.993** | **1.014** | **0.958** |
| MN ragged | 192 | 1.043 | 1.001 | **0.978** | **0.959** |
| MN ragged | 224 | 1.079 | 1.051 | 1.009 | **0.997** |
| MN ragged | 248 | 1.093 | 1.065 | 1.028 | 1.014 |
| K ragged | 136 | 1.027 | **0.986** | **0.879** | **0.884** |
| K ragged | 160 | 0.961 | **0.936** | **0.951** | **0.898** |
| K ragged | 192 | 0.993 | 0.961 | **0.973** | **0.910** |
| K ragged | 224 | 1.062 | 1.047 | 0.942 | **0.945** |
| K ragged | 248 | 1.086 | 1.055 | 0.973 | 0.948 |
| MNK ragged | 136 | 1.045 | **0.996** | **0.926** | **0.940** |
| MNK ragged | 160 | 0.989 | **0.951** | **0.984** | **0.946** |
| MNK ragged | 192 | 1.022 | 0.974 | **0.971** | **0.924** |
| MNK ragged | 224 | 1.077 | 1.067 | 0.977 | **0.977** |
| MNK ragged | 248 | 1.075 | 1.065 | 1.003 | 0.977 |

## One full wave, then S, with bias

| variant | S | K = 4096 | K = 8192 | K = 16384 | K = 32768 |
|---|---|---|---|---|---|
| aligned | 136 | 1.036 | **0.986** | **0.956** | **0.982** |
| aligned | 160 | 1.032 | **0.988** | **1.012** | **0.955** |
| aligned | 192 | 1.049 | 0.997 | **0.991** | **0.956** |
| aligned | 224 | 1.080 | 1.046 | 0.999 | **0.994** |
| aligned | 248 | 1.088 | 1.052 | 1.002 | 1.013 |
| MN ragged | 136 | 1.079 | **1.006** | **0.989** | **1.013** |
| MN ragged | 160 | 1.083 | **1.011** | **1.015** | **0.965** |
| MN ragged | 192 | 1.075 | 1.015 | **0.994** | **0.970** |
| MN ragged | 224 | 1.120 | 1.062 | 1.019 | **1.003** |
| MN ragged | 248 | 1.116 | 1.080 | 1.037 | 1.020 |
| K ragged | 136 | 1.034 | **0.997** | **0.876** | **0.884** |
| K ragged | 160 | 0.978 | **0.942** | **0.954** | **0.901** |
| K ragged | 192 | 1.006 | 0.965 | **0.981** | **0.912** |
| K ragged | 224 | 1.078 | 1.055 | 0.945 | **0.951** |
| K ragged | 248 | 1.094 | 1.057 | 0.980 | 0.951 |
| MNK ragged | 136 | 1.075 | **1.011** | **0.933** | **0.941** |
| MNK ragged | 160 | 1.019 | **0.969** | **0.999** | **0.951** |
| MNK ragged | 192 | 1.046 | 0.989 | **0.977** | **0.931** |
| MNK ragged | 224 | 1.106 | 1.086 | 0.982 | **0.975** |
| MNK ragged | 248 | 1.107 | 1.087 | 1.015 | 0.979 |

## Rules on these cells (regret against the faster of spread and data-parallel)

`v3 rule`: spread if (256 - S) k >= 10493 for every shape. `no spread when ragged`: the same, but data-parallel whenever the variant has an edge flag set (M, N or K ragged). `no spread when M / N ragged`: data-parallel when EDGE_M / EDGE_N are on (the spilling edge build), K ragged allowed. `old (k >= 512)`: the rule before v26 entry 5.

| variant | bias | cells | rule | median | 90th percentile | worst | cells more than 2% off |
|---|---|---|---|---|---|---|---|
| aligned | False | 40 | v3 rule | 1.000 | 1.010 | 1.019 | 0 |
| aligned | False | 40 | old (k >= 512) | 1.000 | 1.066 | 1.177 | 9 |
| aligned | True | 40 | v3 rule | 1.000 | 1.000 | 1.012 | 0 |
| aligned | True | 40 | old (k >= 512) | 1.000 | 1.046 | 1.173 | 6 |
| MN ragged | False | 40 | v3 rule | 1.000 | 1.014 | 1.083 | 3 |
| MN ragged | False | 40 | no spread when ragged | 1.000 | 1.044 | 1.066 | 8 |
| MN ragged | False | 40 | no spread when M / N ragged | 1.000 | 1.044 | 1.066 | 8 |
| MN ragged | False | 40 | old (k >= 512) | 1.000 | 1.034 | 1.062 | 8 |
| MN ragged | True | 40 | v3 rule | 1.000 | 1.022 | 1.110 | 4 |
| MN ragged | True | 40 | no spread when ragged | 1.000 | 1.030 | 1.048 | 4 |
| MN ragged | True | 40 | no spread when M / N ragged | 1.000 | 1.030 | 1.048 | 4 |
| MN ragged | True | 40 | old (k >= 512) | 1.000 | 1.026 | 1.037 | 5 |
| K ragged | False | 40 | v3 rule | 1.000 | 1.148 | 1.354 | 13 |
| K ragged | False | 40 | no spread when ragged | 1.069 | 1.354 | 1.551 | 30 |
| K ragged | False | 40 | no spread when M / N ragged | 1.000 | 1.148 | 1.354 | 13 |
| K ragged | False | 40 | old (k >= 512) | 1.023 | 1.339 | 1.551 | 20 |
| K ragged | True | 40 | v3 rule | 1.000 | 1.129 | 1.316 | 12 |
| K ragged | True | 40 | no spread when ragged | 1.057 | 1.323 | 1.536 | 28 |
| K ragged | True | 40 | no spread when M / N ragged | 1.000 | 1.129 | 1.316 | 12 |
| K ragged | True | 40 | old (k >= 512) | 1.011 | 1.316 | 1.536 | 18 |
| MNK ragged | False | 40 | v3 rule | 1.000 | 1.055 | 1.208 | 8 |
| MNK ragged | False | 40 | no spread when ragged | 1.025 | 1.229 | 1.392 | 24 |
| MNK ragged | False | 40 | no spread when M / N ragged | 1.025 | 1.229 | 1.392 | 24 |
| MNK ragged | False | 40 | old (k >= 512) | 1.000 | 1.208 | 1.392 | 14 |
| MNK ragged | True | 40 | v3 rule | 1.000 | 1.034 | 1.134 | 5 |
| MNK ragged | True | 40 | no spread when ragged | 1.023 | 1.189 | 1.387 | 21 |
| MNK ragged | True | 40 | no spread when M / N ragged | 1.023 | 1.189 | 1.387 | 21 |
| MNK ragged | True | 40 | old (k >= 512) | 1.000 | 1.165 | 1.347 | 12 |
