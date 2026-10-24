# Memory-bound tile + split by estimated time (traffic + padded MFMA work)

Regret: median / 90th / worst / geomean over the best point timed per shape, shapes above 1.05 / 1.15.

Top fits on REFIT memory-bound: 1.070 (K_MIN 2048, RATE 2e+15, SPLIT_US 4); 1.070 (K_MIN 2048, RATE 2e+15, SPLIT_US 1); 1.070 (K_MIN 2048, RATE 4e+15, SPLIT_US 4); 1.070 (K_MIN 2048, RATE 2e+15, SPLIT_US 0); 1.071 (K_MIN 2048, RATE 1e+15, SPLIT_US 4)

| set | picker | shapes | regret | table kind not timed (cap) |
|---|---|---:|---|---:|
| REFIT MB (152) | 256x256x64 + cap (current) | 152 | 2.462 / 3.838 / 4.664 / 2.382, 149 > 1.05, 146 > 1.15 | |
| REFIT MB | rule, in sample | 152 | 1.040 / 1.196 / 1.594 / 1.070, 67 > 1.05, 24 > 1.15 | 28 |
| REFIT MB | rule, leave one shape out | 152 | 1.043 / 1.197 / 1.594 / 1.075, 71 > 1.05, 26 > 1.15 | |
| REFIT MB | best tile + table split | 152 | 1.005 / 1.125 / 1.594 / 1.038, 35 > 1.05, 11 > 1.15 | |
| tilesel small-M (100) | 256x256x64 + cap (current) | 100 | 1.663 / 3.814 / 4.311 / 1.873, 91 > 1.05, 86 > 1.15 | |
| tilesel small-M | rule (REFIT-fitted, held out) | 74 | 1.025 / 1.160 / 1.593 / 1.060, 26 > 1.05, 10 > 1.15 | 19 |
| tilesel small-M | best tile + table split | 100 | 1.000 / 1.074 / 1.516 / 1.023, 13 > 1.05, 3 > 1.15 | |
| tilesel small-N (60) | 256x256x64 + cap (current) | 60 | 1.314 / 2.131 / 4.128 / 1.460, 57 > 1.05, 43 > 1.15 | |
| tilesel small-N | rule (REFIT-fitted, held out) | 49 | 1.037 / 1.173 / 1.492 / 1.068, 21 > 1.05, 8 > 1.15 | 24 |
| tilesel small-N | best tile + table split | 60 | 1.000 / 1.058 / 1.228 / 1.020, 7 > 1.05, 3 > 1.15 | |

For comparison: REFIT nearest neighbours left one shape out 1.051 (WORKLOG §15); §17's tree fitted to small shapes 1.101 cross-validated; the fill rule 1.66 (eval_joint_rule_mb.py); least traffic with a fill floor 1.162 (eval_mb_traffic.py).

Tilesel shapes are scored only where the rule's tile was timed there.

Tiles the rule picks on REFIT MB (tile, split): 1x16x256 dp 25, 16x16x256 dp 25, 16x16x256 split 14, 16x32x128 dp 12, 32x32x128 dp 9, 16x64x128 dp 8, 32x32x128 split 7, 128x256x64 split 7, 16x32x128 split 6, 64x64x128 dp 6, 16x64x128 split 5, 64x64x128 split 5, 16x128x128 split 3, 64x128x128 split 3