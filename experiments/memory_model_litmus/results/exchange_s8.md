HIP_VISIBLE_DEVICES=1, AMD Instinct MI355X, triton 3.8.0, 2026-10-06 18:59


## s8

S = 8 tiles, each split n ways (W = 8 n writers), 256 KiB partials, `wt` synchronisation, each group on one XCD: the exchange costs that split_model.py combines with the GEMM's MAC time. Phases as in the split table.

| pattern | n | W | measured us | publish us | sync us | read us | model us | wrong | timeouts |
|---|---|---|---|---|---|---|---|---|---|
| reduce-scatter | 2 | 16 | 8.23 | 3.22 | 0.60 | 4.08 | 7.32 | 0 | 0 |
| reduce-scatter | 4 | 32 | 7.75 | 3.38 | 0.64 | 3.48 | 7.32 | 0 | 0 |
| reduce-scatter | 8 | 64 | 8.50 | 3.77 | 0.68 | 3.68 | 7.32 | 0 | 0 |
| reduce-scatter | 16 | 128 | 12.09 | 4.99 | 0.84 | 5.80 | 9.61 | 0 | 0 |
| reduce-scatter | 32 | 256 | 20.98 | 9.11 | 1.72 | 8.64 | 18.62 | 0 | 0 |
| owner | 2 | 16 | 7.66 | 3.10 | 0.48 | 3.68 | 7.32 | 0 | 0 |
| owner | 4 | 32 | 16.20 | 3.28 | 0.44 | 11.68 | 14.61 | 0 | 0 |
| owner | 8 | 64 | 33.44 | 3.71 | 0.52 | 27.88 | 29.17 | 0 | 0 |
| owner | 16 | 128 | 66.22 | 4.77 | 0.36 | 59.26 | 59.46 | 0 | 0 |
| owner | 32 | 256 | 138.39 | 8.84 | 0.24 | 124.14 | 122.25 | 0 | 0 |
| last arriver | 2 | 16 | 7.66 | 3.20 | 0.40 | 3.76 | 7.32 | 0 | 0 |
| last arriver | 4 | 32 | 15.18 | 3.38 | 0.40 | 10.90 | 14.61 | 0 | 0 |
| last arriver | 8 | 64 | 30.30 | 3.78 | 0.40 | 25.14 | 29.17 | 0 | 0 |
| last arriver | 16 | 128 | 60.83 | 5.02 | 0.44 | 53.72 | 59.75 | 0 | 0 |
| last arriver | 32 | 256 | 123.50 | 9.12 | 0.52 | 110.92 | 122.54 | 0 | 0 |
