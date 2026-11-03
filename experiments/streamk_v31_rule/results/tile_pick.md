# Stream-K-aware tile pick (rule results/rule_safe.json)

Fitted on results/grid_r1.csv, results/grid_r2.csv, results/tilesweep_r1.csv, results/tilesweep_r2.csv, results/gap_r1.csv, results/gap_r2.csv (shapes timed at 2+ tiles): {"below": 6, "after": 6, "below_tail": 0.7} (below / after: k-steps per split tail; below_tail: a dp launch's partial wave below one wave).

Default pipeline (tile pick, then the rule) time over the shape's best among its timed tiles (geomean / median / 90th / worst):

| set | pick_gemm_tile + rule | Stream-K-aware pick + rule |
|---|---|---|
| sweep (fit) (543) | 1.0964 / 1.029 / 1.331 / 1.880 | 1.0248 / 1.000 / 1.087 / 1.312 |
| held-out AITER shapes (597) | 1.0265 / 1.000 / 1.119 / 1.730 | 1.0065 / 1.000 / 1.017 / 1.176 |
