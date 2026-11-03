# Best tile against GPU fill (tmp/tile_fill.py)

tilesel.py's 570 shapes; each tile at its best timed point. Compute-bound: arithmetic intensity >= 312 flop/byte (2.5 PF / 8 TB/s). Fill = tiles / 256 CUs.

## compute-bound (353 shapes)

- Best tile's fill (tiles / 256): 10th 0.50, 25th 0.88, median 2.03, 75th 4.88, 90th 14.57; below 0.5: 31, 0.5-1: 79, 1-2: 62, 2 or more: 181.
- Best tile's BM x BN: 256x256 275, 128x128 46, 128x256 24, 256x128 5, 64x64 2, 64x128 1
- Best tile's BK: 64 320, 128 33

Did a larger tile with at least as many tiles as the GPU has CUs lose? Shapes where the best tile is not the largest-area candidate with tiles >= 256, and how far that one is behind:

| rule | regret: median / 90th / worst / geomean | > 1.05 |
|---|---|
| largest tile with tiles >= 0.25 x 256 | 1.000 / 1.112 / 1.912 / 1.036 | 58 |
| largest tile with tiles >= 0.5 x 256 | 1.000 / 1.067 / 1.622 / 1.021 | 42 |
| largest tile with tiles >= 0.75 x 256 | 1.000 / 1.103 / 1.622 / 1.026 | 59 |
| largest tile with tiles >= 1.0 x 256 | 1.000 / 1.243 / 1.758 / 1.067 | 114 |
| largest tile with tiles >= 1.5 x 256 | 1.000 / 1.349 / 1.875 / 1.106 | 157 |
| largest tile with tiles >= 2.0 x 256 | 1.000 / 1.476 / 2.356 / 1.144 | 169 |
| largest tile with tiles >= 4.0 x 256 | 1.218 / 1.736 / 2.859 / 1.261 | 231 |
| largest tile with tiles >= 0.5 x 256, BK 128 if K <= 4096 | 1.000 / 1.067 / 1.622 / 1.021 | 42 |
| largest tile with tiles >= 1.0 x 256, BK 128 if K <= 4096 | 1.000 / 1.243 / 1.758 / 1.067 | 114 |

Best of these: **largest tile with tiles >= 0.5 x 256**.

Best tile's fill by shape group (median): aiter 1.00 (150), large 22.27 (30), refit-cb 3.10 (61), small-M 0.97 (9), small-N 0.41 (3), tall-skinny 1.16 (51), wide-short 3.49 (49)

## memory-bound (217 shapes)

- Best tile's fill (tiles / 256): 10th 0.12, 25th 0.38, median 0.81, 75th 2.25, 90th 4.81; below 0.5: 64, 0.5-1: 60, 1-2: 28, 2 or more: 65.
- Best tile's BM x BN: 16x16 78, 64x64 27, 64x128 18, 16x32 15, 32x128 14, 128x128 14, 256x256 14, 32x32 12, 256x128 12, 128x256 8, 1x16 5
- Best tile's BK: 64 37, 128 121, 256 59

Did a larger tile with at least as many tiles as the GPU has CUs lose? Shapes where the best tile is not the largest-area candidate with tiles >= 256, and how far that one is behind:

| rule | regret: median / 90th / worst / geomean | > 1.05 |
|---|---|
| largest tile with tiles >= 0.25 x 256 | 1.204 / 1.683 / 3.436 / 1.258 | 166 |
| largest tile with tiles >= 0.5 x 256 | 1.141 / 1.461 / 4.170 / 1.195 | 149 |
| largest tile with tiles >= 0.75 x 256 | 1.143 / 1.541 / 4.609 / 1.230 | 157 |
| largest tile with tiles >= 1.0 x 256 | 1.191 / 1.804 / 8.678 / 1.318 | 168 |
| largest tile with tiles >= 1.5 x 256 | 1.160 / 2.575 / 8.678 / 1.368 | 154 |
| largest tile with tiles >= 2.0 x 256 | 1.197 / 3.436 / 8.916 / 1.478 | 162 |
| largest tile with tiles >= 4.0 x 256 | 1.515 / 5.852 / 12.039 / 1.921 | 180 |
| largest tile with tiles >= 0.5 x 256, BK 128 if K <= 4096 | 1.148 / 1.461 / 4.170 / 1.199 | 152 |
| largest tile with tiles >= 1.0 x 256, BK 128 if K <= 4096 | 1.191 / 1.804 / 8.678 / 1.323 | 174 |

Best of these: **largest tile with tiles >= 0.5 x 256**.

Best tile's fill by shape group (median): refit-cb 0.16 (9), refit-mb 0.66 (50), small-M 0.77 (91), small-N 2.12 (57), tall-skinny 0.73 (9), wide-short 2.93 (1)

