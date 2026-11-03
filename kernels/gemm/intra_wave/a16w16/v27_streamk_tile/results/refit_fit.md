# REFIT rule fit

From results/refit.csv (merged rounds; noise sigma 1.89%). Each shape is left out in turn and the rule picks from the others. Regret: the pick's time over the shape's best timed (tile, configuration). Columns: median | 90th percentile | worst | geometric mean | shapes above 1 + sigma.


## compute-bound (198 shapes)

| rule | median | 90th | worst | geomean | > sigma |
|---|---|---|---|---|---|
| v26 deployed (cap at 256x256x64) | 1.010 | 1.604 | 3.545 | 1.136 | 79 / 198 |
| fixed: best single (tile, configuration) | 1.010 | 1.604 | 3.545 | 1.136 | 79 / 198 |
| cap at the kNN tile (log MNK, k = 5) | 1.008 | 1.146 | 1.465 | 1.038 | 68 / 198 |
| kNN regret: log MNK, k=1 | 1.010 | 1.171 | 1.559 | 1.048 | 87 / 198 |
| kNN regret: log MNK, k=3 | 1.007 | 1.098 | 1.277 | 1.026 | 58 / 198 |
| kNN regret: log MNK, k=5 | 1.007 | 1.071 | 1.238 | 1.023 | 58 / 198 |
| kNN regret: log MNK, k=8 | 1.007 | 1.072 | 1.348 | 1.025 | 62 / 198 |
| kNN regret: tile coordinates, k=3 | 1.006 | 1.098 | 1.277 | 1.025 | 57 / 198 |
| kNN regret: tile coordinates, k=5 | 1.005 | 1.072 | 1.246 | 1.023 | 56 / 198 |
| kNN regret: tile coordinates, k=8 | 1.007 | 1.071 | 1.316 | 1.023 | 59 / 198 |

Chosen (smallest geometric-mean regret): **tile coordinates, k=5**.

Best tile per shape: 256x256x64 138, 128x128x64 23, 64x64x128 11, 128x256x64 9, 64x128x128 6, 128x128x128 5, 256x128x64 4, 128x64x128 2.
Best configuration per shape: cap 42, dp 29, pdp 18, 2t 16, rsall 11, rs8 11, ow2 10, rs5 9, rs4 9, sp 7, ow3 5, rs12 5.

## memory-bound (152 shapes)

| rule | median | 90th | worst | geomean | > sigma |
|---|---|---|---|---|---|
| v26 deployed (cap at 256x256x64) | 2.459 | 3.846 | 4.646 | 2.383 | 151 / 152 |
| fixed: best single (tile, configuration) | 1.250 | 1.509 | 1.944 | 1.257 | 141 / 152 |
| cap at the kNN tile (log MNK, k = 5) | 1.059 | 1.216 | 1.462 | 1.086 | 98 / 152 |
| kNN regret: log MNK, k=1 | 1.042 | 1.224 | 1.792 | 1.078 | 93 / 152 |
| kNN regret: log MNK, k=3 | 1.028 | 1.175 | 1.383 | 1.061 | 83 / 152 |
| kNN regret: log MNK, k=5 | 1.025 | 1.158 | 1.383 | 1.053 | 79 / 152 |
| kNN regret: log MNK, k=8 | 1.029 | 1.155 | 1.353 | 1.053 | 87 / 152 |
| kNN regret: tile coordinates, k=3 | 1.027 | 1.202 | 1.383 | 1.063 | 82 / 152 |
| kNN regret: tile coordinates, k=5 | 1.022 | 1.152 | 1.383 | 1.051 | 78 / 152 |
| kNN regret: tile coordinates, k=8 | 1.026 | 1.155 | 1.235 | 1.051 | 83 / 152 |
| kNN regret: log MNK, k=5, its K bin left out | 1.039 | 1.169 | 1.403 | 1.063 | 94 / 152 |
| kNN regret: tile coordinates, k=5, its K bin left out | 1.043 | 1.173 | 1.403 | 1.067 | 96 / 152 |

Chosen (smallest geometric-mean regret): **tile coordinates, k=8**.

Best tile per shape: 16x16x256 53, 1x16x256 17, 16x16x128 15, 16x32x128 12, 32x32x128 11, 64x64x128 9, 64x128x64 5, 32x128x128 4, 128x128x128 4, 16x128x128 3, 64x128x128 3, 128x128x64 3, 256x128x64 3, 16x64x128 2, 128x256x64 2, 128x64x128 2, 256x256x64 2, 32x256x128 1, 128x64x64 1.
Best configuration per shape: cap 43, dp 35, rsall 16, rs7 8, rs8 7, pdp 7, rs6 6, rs4 6, rs3 5, rs5 3, rs16 3, owall 3.
