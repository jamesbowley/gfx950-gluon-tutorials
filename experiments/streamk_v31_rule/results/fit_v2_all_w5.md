# v31 rule fit on results/grid_r1.csv, results/grid_r2.csv, results/tilesweep_r1.csv, results/tilesweep_r2.csv, results/gap_r1.csv, results/gap_r2.csv

3292 (shape, tile) blocks; 0 failed or incorrect points.

## Fitted constants

```json
{
 "below": {
  "k_min": 16,
  "owner_max_per_tile": 3,
  "owner_n_max": 3,
  "owner_k_min": 20,
  "rs_b": 6
 },
 "after": {
  "k_min": 16,
  "owner_max_per_tile": 4,
  "owner_n_max": 3,
  "owner_k_min": 32,
  "rs_b": 8
 }
}
```

## Regret (time / block's best): geomean / median / 90th percentile / worst

| set | rule | dp | best owner | best rs |
|---|---|---|---|---|
| all (3292) | 1.0093 / 1.000 / 1.029 / 1.306 | 1.2264 / 1.125 / 1.738 / 5.602 | 1.0479 / 1.028 / 1.126 / 1.605 | 1.0274 / 1.000 / 1.096 / 1.698 |
| below one wave (1295) | 1.0056 / 1.000 / 1.009 / 1.240 | 1.3715 / 1.123 / 2.509 / 5.602 | 1.0692 / 1.021 / 1.227 / 1.605 | 1.0567 / 1.005 / 1.192 / 1.698 |
| one wave or more (1997) | 1.0117 / 1.000 / 1.034 / 1.306 | 1.1407 / 1.125 / 1.340 / 1.842 | 1.0395 / 1.029 / 1.100 / 1.218 | 1.0156 / 1.000 / 1.056 / 1.408 |

Rule slower than dp beyond 1.01x on 52 blocks; rule's choice not timed on 14.


Best family per block: {'dp': 929, 'rs': 1611, 'ow': 712, '2t': 23, 'l2rs': 17}

