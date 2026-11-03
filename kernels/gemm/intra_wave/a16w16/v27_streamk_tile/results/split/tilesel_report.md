# Tile selection (tilesel.py)

Time over the best point timed for the shape (same block): median / 90th / worst / geomean, and shapes above 1.05.

| group | selection | shapes | regret |
|---|---|---|---|
| aiter | v26 (cap at 256x256x64) | 150 | 1.005 / 1.321 / 1.964 / 1.094, 48 |
| aiter | auto_tile + cap | 150 | 1.003 / 1.088 / 1.428 / 1.025, 22 |
| aiter | oracle tile at its table pick | 150 | 1.002 / 1.025 / 1.332 / 1.011, 6 |
| aiter | 256x256x64 at its table pick | 150 | 1.014 / 1.365 / 1.923 / 1.112, 53 |
| large | v26 (cap at 256x256x64) | 30 | 1.000 / 1.004 / 1.010 / 1.001, 0 |
| large | auto_tile + cap | 30 | 1.000 / 1.004 / 1.010 / 1.001, 0 |
| large | oracle tile at its table pick | 30 | 1.000 / 1.007 / 1.010 / 1.001, 0 |
| large | 256x256x64 at its table pick | 30 | 1.000 / 1.007 / 1.010 / 1.001, 0 |
| refit-cb | v26 (cap at 256x256x64) | 70 | 1.002 / 2.000 / 3.597 / 1.143, 20 |
| refit-cb | auto_tile + cap | 70 | 1.000 / 1.427 / 1.624 / 1.064, 15 |
| refit-cb | oracle tile at its table pick | 70 | 1.000 / 1.034 / 1.318 / 1.016, 4 |
| refit-cb | 256x256x64 at its table pick | 70 | 1.008 / 2.000 / 3.597 / 1.145, 21 |
| refit-mb | v26 (cap at 256x256x64) | 50 | 2.427 / 4.042 / 4.700 / 2.388, 49 |
| refit-mb | auto_tile + cap | 50 | 2.427 / 4.042 / 4.700 / 2.388, 49 |
| refit-mb | oracle tile at its table pick | 50 | 1.000 / 1.052 / 1.177 / 1.013, 5 |
| refit-mb | 256x256x64 at its table pick | 23 | 2.284 / 3.912 / 4.396 / 2.194, 22 |
| small-M | v26 (cap at 256x256x64) | 100 | 1.679 / 3.899 / 4.311 / 1.876, 91 |
| small-M | auto_tile + cap | 100 | 1.679 / 3.899 / 4.311 / 1.876, 91 |
| small-M | oracle tile at its table pick | 100 | 1.000 / 1.050 / 1.367 / 1.016, 10 |
| small-M | 256x256x64 at its table pick | 42 | 1.302 / 2.857 / 3.727 / 1.536, 32 |
| small-N | v26 (cap at 256x256x64) | 60 | 1.330 / 2.324 / 4.128 / 1.466, 57 |
| small-N | auto_tile + cap | 43 | 1.158 / 1.459 / 1.603 / 1.185, 37 |
| small-N | oracle tile at its table pick | 60 | 1.000 / 1.035 / 1.165 / 1.009, 3 |
| small-N | 256x256x64 at its table pick | 24 | 1.195 / 1.922 / 3.107 / 1.352, 22 |
| tall-skinny | v26 (cap at 256x256x64) | 60 | 1.003 / 1.301 / 1.873 / 1.059, 15 |
| tall-skinny | auto_tile + cap | 60 | 1.003 / 1.117 / 1.278 / 1.026, 10 |
| tall-skinny | oracle tile at its table pick | 60 | 1.001 / 1.062 / 1.121 / 1.012, 6 |
| tall-skinny | 256x256x64 at its table pick | 60 | 1.003 / 1.387 / 1.873 / 1.070, 18 |
| wide-short | v26 (cap at 256x256x64) | 50 | 1.000 / 1.009 / 1.065 / 1.005, 1 |
| wide-short | auto_tile + cap | 50 | 1.000 / 1.009 / 1.065 / 1.005, 1 |
| wide-short | oracle tile at its table pick | 50 | 1.000 / 1.013 / 1.068 / 1.005, 1 |
| wide-short | 256x256x64 at its table pick | 50 | 1.000 / 1.033 / 1.313 / 1.012, 3 |
