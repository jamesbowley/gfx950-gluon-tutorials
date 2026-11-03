# v31 rule fit on results/grid_r1.csv, results/grid_r2.csv

1089 (shape, tile) blocks; 0 failed or incorrect points.

## Fitted constants

```json
{
 "below": {
  "k_min": 12,
  "rs_min_per_tile": 5,
  "rs_ppp": 2,
  "rs_div": 1.5,
  "rs_n_max": 64,
  "owner_k_min": 16,
  "owner_n_max": 3
 },
 "after": {
  "k_min": 12,
  "rs_min_per_tile": 8,
  "rs_ppp": 1,
  "rs_div": 2,
  "rs_n_max": 32,
  "owner_k_min": 24,
  "owner_n_max": 3
 }
}
```

## Regret (time / block's best): geomean / median / 90th percentile / worst

| set | rule | dp | best owner | best rs |
|---|---|---|---|---|
| all (1089) | 1.0119 / 1.000 / 1.031 / 1.241 | 1.2164 / 1.165 / 1.475 / 5.602 | 1.0517 / 1.038 / 1.120 / 1.605 | 1.0225 / 1.000 / 1.081 / 1.698 |
| below one wave (153) | 1.0135 / 1.000 / 1.070 / 1.176 | 1.5895 / 1.534 / 2.857 / 5.602 | 1.0829 / 1.000 / 1.287 / 1.605 | 1.0652 / 1.033 / 1.175 / 1.698 |
| one wave or more (936) | 1.0116 / 1.001 / 1.030 / 1.241 | 1.1644 / 1.150 / 1.376 / 1.645 | 1.0467 / 1.039 / 1.110 / 1.218 | 1.0157 / 1.000 / 1.051 / 1.408 |

Rule slower than dp beyond 1.01x on 31 blocks; rule's choice not timed on 0.

## Cross-validation (refit without the held-out bin)

- leave one k out: 1.0152 / 1.002 / 1.041 / 1.280
- leave one S out: 1.0181 / 1.002 / 1.058 / 1.613

Best family per block: {'dp': 123, 'rs': 675, 'ow': 251, '2t': 23, 'l2rs': 17}

