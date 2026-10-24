# Memory-bound tile + split by least traffic

Regret: median / 90th / worst / geomean over the best point timed per shape, shapes above 1.05 / 1.15.

## Fit on REFIT memory-bound (152 shapes)

- (1024, 0.75, 'cu', 'table'): 1.162
- (512, 1.0, 'cu', 'table'): 1.179
- (1024, 0.5, 'cu', 'table'): 1.180
- (512, 0.75, 'cu', 'table'): 1.193
- (2048, 0.5, 'cu', 'table'): 1.199
- (1024, 0.75, 'cu', 'own'): 1.200
- (256, 1.0, 'cu', 'table'): 1.210
- (2048, 0.75, 'cu', 'table'): 1.219

Chosen: K_MIN 1024, FILL 0.75 (cu), split table.

## Results

| set | picker | shapes | regret | picks falling back to cap |
|---|---|---:|---|---:|
| REFIT MB | 256x256x64 + cap (current) | 152 | 2.462 / 3.838 / 4.664 / 2.382, 149 > 1.05, 146 > 1.15 | |
| REFIT MB | traffic rule, in sample | 152 | 1.113 / 1.418 / 2.414 / 1.162, 101 > 1.05, 62 > 1.15 | 53 |
| REFIT MB | traffic rule, leave one shape out | 152 | 1.113 / 1.418 / 2.414 / 1.162, 101 > 1.05, 62 > 1.15 | |
| REFIT MB | best tile + table split | 152 | 1.005 / 1.125 / 1.594 / 1.038, 35 > 1.05, 11 > 1.15 | 25 |
| tilesel small-M | 256x256x64 + cap (current) | 100 | 1.663 / 3.814 / 4.311 / 1.873, 91 > 1.05, 86 > 1.15 | |
| tilesel small-M | traffic rule (REFIT-fitted, held out) | 78 | 1.087 / 1.345 / 3.189 / 1.150, 50 > 1.05, 28 > 1.15 | 19 |
| tilesel small-M | best tile + table split | 100 | 1.000 / 1.074 / 1.516 / 1.023, 13 > 1.05, 3 > 1.15 | 23 |
| tilesel small-N | 256x256x64 + cap (current) | 60 | 1.314 / 2.131 / 4.128 / 1.460, 57 > 1.05, 43 > 1.15 | |
| tilesel small-N | traffic rule (REFIT-fitted, held out) | 32 | 1.097 / 1.247 / 1.541 / 1.136, 27 > 1.05, 12 > 1.15 | 9 |
| tilesel small-N | best tile + table split | 60 | 1.000 / 1.058 / 1.228 / 1.020, 7 > 1.05, 3 > 1.15 | 14 |

For comparison: REFIT nearest neighbours left one shape out 1.051 (WORKLOG §15); results/split/tile_rule.md's tree fitted to small shapes 1.101 cross-validated; the fill rule (eval_joint_rule_mb.py) 1.66.

Tiles the rule picks on REFIT MB: 32x32x128 split 14, 1x16x256 dp 12, 16x16x256 split 10, 32x128x128 split 10, 16x16x256 dp 9, 1x16x256 split 9, 16x64x128 split 9, 16x128x128 split 9, 256x64x64 dp 8, 16x32x128 split 6, 64x128x128 split 6, 32x32x128 dp 6