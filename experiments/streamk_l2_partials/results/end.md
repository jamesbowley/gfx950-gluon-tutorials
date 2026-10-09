HIP_VISIBLE_DEVICES=6 launches/point=20

## end

| T | n | MiB per XCD | variant | launch (events) | in-kernel span | launch - span |
|---|---|---|---|---|---|---|
| 1 | 8 | 2 | wt_relaxed | 38.7 | 10.9 | 27.7 |
| 1 | 8 | 2 | l2_flags | 35.9 | 7.1 | 28.9 |
| 2 | 8 | 4 | wt_relaxed | 45.6 | 15.1 | 30.5 |
| 2 | 8 | 4 | l2_flags | 45.5 | 12.9 | 32.6 |
| 1 | 16 | 4 | wt_relaxed | 46.2 | 16.2 | 30.0 |
| 1 | 16 | 4 | l2_flags | 45.1 | 13.7 | 31.4 |
| 2 | 16 | 8 | wt_relaxed | 65.9 | 27.6 | 38.3 |
| 2 | 16 | 8 | l2_flags | 60.2 | 21.5 | 38.7 |
| 4 | 8 | 8 | wt_relaxed | 61.0 | 24.9 | 36.1 |
| 4 | 8 | 8 | l2_flags | 58.1 | 20.9 | 37.2 |
