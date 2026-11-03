# v31 rule fit on results/grid_r1.csv

1089 (shape, tile) blocks; 0 failed or incorrect points.

## Fitted constants

```json
{
 "below": {
  "k_min": 12,
  "rs_min_per_tile": 5,
  "rs_div": 1.5,
  "rs_n_max": 16,
  "owner_n_max": 3
 },
 "after": {
  "k_min": 12,
  "rs_min_per_tile": 6,
  "rs_div": 2,
  "rs_n_max": 32,
  "owner_n_max": 3
 }
}
```

## Regret (time / block's best): geomean / median / 90th percentile / worst

| set | rule | dp | best owner | best rs |
|---|---|---|---|---|
| all (1089) | 1.0129 / 1.000 / 1.035 / 1.239 | 1.2169 / 1.165 / 1.472 / 5.665 | 1.0520 / 1.039 / 1.122 / 1.607 | 1.0227 / 1.000 / 1.080 / 1.699 |
| below one wave (153) | 1.0198 / 1.000 / 1.095 / 1.239 | 1.5895 / 1.527 / 2.844 / 5.665 | 1.0831 / 1.000 / 1.283 / 1.607 | 1.0655 / 1.031 / 1.174 / 1.699 |
| one wave or more (936) | 1.0117 / 1.001 / 1.032 / 1.194 | 1.1649 / 1.151 / 1.373 / 1.664 | 1.0470 / 1.040 / 1.109 / 1.225 | 1.0158 / 1.000 / 1.050 / 1.414 |

Rule slower than dp beyond 1.01x on 46 blocks; rule's choice not timed on 0.

## Cross-validation (refit without the held-out bin)

- leave one k out: 1.0194 / 1.002 / 1.055 / 1.855
- leave one S out: 1.0196 / 1.000 / 1.055 / 1.678

Best family per block: {'rs': 672, 'ow': 253, 'dp': 123, '2t': 24, 'l2rs': 17}

