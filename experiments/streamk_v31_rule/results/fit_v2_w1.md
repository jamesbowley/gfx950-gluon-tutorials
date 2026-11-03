# v31 rule fit on results/grid_r1.csv, results/grid_r2.csv, results/tilesweep_r1.csv, results/tilesweep_r2.csv

2285 (shape, tile) blocks; 0 failed or incorrect points.

## Fitted constants

```json
{
 "below": {
  "k_min": 16,
  "owner_max_per_tile": 3,
  "owner_n_max": 3,
  "owner_k_min": 24,
  "rs_b": 6
 },
 "after": {
  "k_min": 16,
  "owner_max_per_tile": 4,
  "owner_n_max": 3,
  "owner_k_min": 20,
  "rs_b": 8
 }
}
```

## Regret (time / block's best): geomean / median / 90th percentile / worst

| set | rule | dp | best owner | best rs |
|---|---|---|---|---|
| all (2285) | 1.0062 / 1.000 / 1.018 / 1.241 | 1.2286 / 1.116 / 1.810 / 5.602 | 1.0469 / 1.027 / 1.126 / 1.605 | 1.0302 / 1.000 / 1.104 / 1.698 |
| below one wave (1067) | 1.0039 / 1.000 / 1.000 / 1.240 | 1.3307 / 1.000 / 2.444 / 5.602 | 1.0555 / 1.000 / 1.213 / 1.605 | 1.0574 / 1.017 / 1.182 / 1.698 |
| one wave or more (1218) | 1.0082 / 1.000 / 1.025 / 1.241 | 1.1457 / 1.132 / 1.355 / 1.645 | 1.0428 / 1.034 / 1.105 / 1.218 | 1.0170 / 1.000 / 1.060 / 1.408 |

Rule slower than dp beyond 1.01x on 39 blocks; rule's choice not timed on 14.

## Cross-validation (refit without the held-out bin)

- leave one k out: 1.0068 / 1.000 / 1.022 / 1.240
- leave one S out: 1.0067 / 1.000 / 1.021 / 1.276

Best family per block: {'dp': 790, 'rs': 958, 'ow': 497, '2t': 23, 'l2rs': 17}

