# Joint rule on REFIT's 152 memory-bound shapes (M = 1-256)

Regret: median / 90th / worst / geomean over the best point timed per shape.

| picker | regret |
|---|---|
| 256x256x64 + cap (auto_tile's memory-bound branch, v26) | 2.462 / 3.838 / 4.664 / 2.382, 149 > 1.05, 146 > 1.15 |
| joint rule + table split, fitted on all ((2048, 0.25), in sample) | 1.543 / 2.267 / 3.193 / 1.541, 129 > 1.05, 110 > 1.15 |
| joint rule + table split, leave one shape out | 1.654 / 2.512 / 5.156 / 1.659, 136 > 1.05, 118 > 1.15 |
| joint rule's tile + cap ((2048, 0.25)) | 1.637 / 2.267 / 2.958 / 1.570, 131 > 1.05, 117 > 1.15 |
| best tile + table split | 1.005 / 1.125 / 1.594 / 1.038, 35 > 1.05, 11 > 1.15 |
| best tile + cap | 1.020 / 1.333 / 2.126 / 1.103, 64 > 1.05, 35 > 1.15 |

For comparison (WORKLOG §15/§16): the REFIT nearest-neighbour rule left one shape out 1.051 geomean, worst 1.235; the depth-5 tree of results/split/tile_rule.md on small shapes 1.07-1.09.

Parameters chosen in the leave-one-out folds: (2048, 0.25) 144, (2048, 0.375) 4, (1024, 0.5) 4