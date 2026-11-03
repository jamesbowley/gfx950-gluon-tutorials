# Tile choice with the formula's plan

573 held-out shapes timed at all 13 specs; aware pick fitted on 300 training shapes with 2+ specs: F = 10 K-steps, tail = 0.7, below one wave 0.8.

Time over the shape's best (every spec, method and n), geomean / median / 90th / worst:

| choice | result |
|---|---|
| pick_gemm_tile, dp | 1.0775 / 1.010 / 1.256 / 1.943 |
| pick_gemm_tile, formula | 1.0664 / 1.000 / 1.224 / 1.943 |
| aware pick, formula | 1.0152 / 1.000 / 1.055 / 1.275 |
| best spec, formula | 1.0016 / 1.000 / 1.000 / 1.071 |
| best spec, dp | 1.0602 / 1.000 / 1.217 / 1.784 |
