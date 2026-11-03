# v31 rule fit on experiments/streamk_v31_rule/results/grid_r1.csv

1089 (shape, tile) blocks; 0 failed or incorrect points.

## Fitted constants

```json
{
 "below": {
  "min_idle": 12,
  "rs_min_per_tile": 6,
  "rs_kpp": 8,
  "rs_min_n": 4,
  "owner_c": 3.0
 },
 "after": {
  "min_idle": 20,
  "rs_min_per_tile": 6,
  "rs_kpp": 4,
  "rs_min_n": 2,
  "owner_c": 4.5
 }
}
```

## Regret (time / block's best): geomean / median / 90th percentile / worst

| set | rule | dp | best owner | best rs |
|---|---|---|---|---|
| all (1089) | 1.0219 / 1.004 / 1.074 / 1.278 | 1.2169 / 1.165 / 1.472 / 5.665 | 1.0520 / 1.039 / 1.122 / 1.607 | 1.0227 / 1.000 / 1.080 / 1.699 |
| below one wave (153) | 1.0151 / 1.000 / 1.066 / 1.243 | 1.5895 / 1.527 / 2.844 / 5.665 | 1.0831 / 1.000 / 1.283 / 1.607 | 1.0655 / 1.031 / 1.174 / 1.699 |
| one wave or more (936) | 1.0230 / 1.007 / 1.079 / 1.278 | 1.1649 / 1.151 / 1.373 / 1.664 | 1.0470 / 1.040 / 1.109 / 1.225 | 1.0158 / 1.000 / 1.050 / 1.414 |

Rule slower than dp beyond 1.01x on 14 blocks; rule's choice not timed on 0.

## Cross-validation (refit without the held-out bin)

- leave one k out: 1.0302 / 1.007 / 1.110 / 1.332
- leave one S out: 1.0234 / 1.005 / 1.079 / 1.278

Best family per block: {'rs': 672, 'ow': 253, 'dp': 123, '2t': 24, 'l2rs': 17}

