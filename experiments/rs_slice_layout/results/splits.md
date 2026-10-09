HIP_VISIBLE_DEVICES=1, AMD Instinct MI355X, 2026-10-08 10:33

## splits

Split counts that do not divide 64, and 32 / 64. N programs per tile publish fp32 partials; the first R read: reader j sums blocks [64 j / R, 64 (j + 1) / R) of all N partials in groups of G blocks, then stores C. 'plain' stores from v20's L3 layout (8 B per lane, 4 rows per 4-lane group); 'block' converts each group to 8 bf16 per lane along a block row; 'band' (power-of-two R only) reshapes the group to its 2-D band and converts to a row layout of 16 B per lane (8 B for a 32 x 32 block). N <= 32 on one XCD per tile, N = 64 spread. 'slice' is the critical reader's read + C store. Medians over 30 cold launches and over groups; every C element checked.

| setup | N | W | R | G | store | total us | publish us | sync us | slice us | wrong |
|---|---|---|---|---|---|---|---|---|---|---|
| tiles 16 | 3 | 48 | 3 | 16 | plain | 16.10 | 7.05 | 0.52 | 8.12 | 0 |
| tiles 16 | 3 | 48 | 3 | 16 | block | 15.04 | 7.02 | 0.52 | 6.80 | 0 |
| tiles 16 | 3 | 48 | 2 | 16 | plain | 17.25 | 6.92 | 0.52 | 9.12 | 0 |
| tiles 16 | 3 | 48 | 2 | 16 | band | 14.93 | 6.65 | 0.52 | 6.92 | 0 |
| tiles 16 | 5 | 80 | 5 | 16 | plain | 16.62 | 7.63 | 0.56 | 6.28 | 0 |
| tiles 16 | 5 | 80 | 5 | 16 | block | 16.17 | 7.45 | 0.56 | 5.88 | 0 |
| tiles 16 | 5 | 80 | 4 | 16 | plain | 15.45 | 7.26 | 0.52 | 5.76 | 0 |
| tiles 16 | 5 | 80 | 4 | 16 | band | 14.39 | 7.47 | 0.52 | 4.72 | 0 |
| tiles 16 | 6 | 96 | 6 | 16 | plain | 17.98 | 8.09 | 0.56 | 7.08 | 0 |
| tiles 16 | 6 | 96 | 6 | 16 | block | 17.56 | 8.50 | 0.56 | 7.00 | 0 |
| tiles 16 | 6 | 96 | 4 | 16 | plain | 15.97 | 7.53 | 0.52 | 6.60 | 0 |
| tiles 16 | 6 | 96 | 4 | 16 | band | 15.33 | 7.65 | 0.52 | 5.84 | 0 |
| tiles 16 | 10 | 160 | 10 | 8 | plain | 19.60 | 9.18 | 0.60 | 7.68 | 0 |
| tiles 16 | 10 | 160 | 10 | 8 | block | 19.85 | 8.66 | 0.60 | 6.76 | 0 |
| tiles 16 | 10 | 160 | 8 | 8 | plain | 19.34 | 9.02 | 0.80 | 6.92 | 0 |
| tiles 16 | 10 | 160 | 8 | 8 | band | 18.73 | 9.50 | 0.60 | 6.28 | 0 |
| tiles 16 | 12 | 192 | 12 | 8 | plain | 20.47 | 10.20 | 0.88 | 8.44 | 0 |
| tiles 16 | 12 | 192 | 12 | 8 | block | 22.11 | 10.73 | 0.80 | 8.24 | 0 |
| tiles 16 | 12 | 192 | 8 | 8 | plain | 20.15 | 10.64 | 0.72 | 7.96 | 0 |
| tiles 16 | 12 | 192 | 8 | 8 | band | 18.52 | 9.81 | 0.68 | 7.40 | 0 |
| tiles 16 | 16 | 256 | 16 | 4 | plain | 24.37 | 10.89 | 0.96 | 10.76 | 0 |
| tiles 16 | 16 | 256 | 16 | 4 | band | 23.08 | 10.25 | 0.80 | 8.98 | 0 |
| tiles 16 | 64 | 256 | 64 | 1 | plain | 24.02 | 13.40 | 1.06 | 9.32 | 0 |
| tiles 16 | 64 | 256 | 64 | 1 | band | 24.51 | 13.34 | 1.47 | 9.20 | 0 |
| CUs 256 | 3 | 240 | 3 | 16 | plain | 24.71 | 11.77 | 0.52 | 11.36 | 0 |
| CUs 256 | 3 | 240 | 3 | 16 | block | 24.72 | 12.22 | 0.52 | 11.32 | 0 |
| CUs 256 | 3 | 240 | 2 | 16 | plain | 25.40 | 11.62 | 0.48 | 12.56 | 0 |
| CUs 256 | 3 | 240 | 2 | 16 | band | 22.46 | 10.24 | 0.48 | 11.12 | 0 |
| CUs 256 | 5 | 240 | 5 | 16 | plain | 25.11 | 12.85 | 0.60 | 11.00 | 0 |
| CUs 256 | 5 | 240 | 5 | 16 | block | 24.24 | 11.78 | 0.60 | 10.56 | 0 |
| CUs 256 | 5 | 240 | 4 | 16 | plain | 25.71 | 11.24 | 0.68 | 9.72 | 0 |
| CUs 256 | 5 | 240 | 4 | 16 | band | 23.60 | 11.31 | 0.60 | 8.88 | 0 |
| CUs 256 | 6 | 240 | 6 | 16 | plain | 25.78 | 12.64 | 0.72 | 11.00 | 0 |
| CUs 256 | 6 | 240 | 6 | 16 | block | 26.49 | 12.80 | 0.68 | 10.80 | 0 |
| CUs 256 | 6 | 240 | 4 | 16 | plain | 22.82 | 11.38 | 0.56 | 10.08 | 0 |
| CUs 256 | 6 | 240 | 4 | 16 | band | 21.75 | 10.82 | 0.56 | 9.32 | 0 |
| CUs 256 | 10 | 240 | 10 | 8 | plain | 23.45 | 12.00 | 0.72 | 10.00 | 0 |
| CUs 256 | 10 | 240 | 10 | 8 | block | 21.42 | 11.00 | 0.72 | 9.24 | 0 |
| CUs 256 | 10 | 240 | 8 | 8 | plain | 25.18 | 11.46 | 0.84 | 9.24 | 0 |
| CUs 256 | 10 | 240 | 8 | 8 | band | 22.09 | 11.59 | 0.76 | 8.76 | 0 |
| CUs 256 | 12 | 192 | 12 | 8 | plain | 20.44 | 10.25 | 0.88 | 8.48 | 0 |
| CUs 256 | 12 | 192 | 12 | 8 | block | 22.06 | 10.89 | 0.80 | 8.24 | 0 |
| CUs 256 | 12 | 192 | 8 | 8 | plain | 20.10 | 10.68 | 0.68 | 8.00 | 0 |
| CUs 256 | 12 | 192 | 8 | 8 | band | 18.59 | 9.81 | 0.68 | 7.36 | 0 |
| CUs 256 | 16 | 256 | 16 | 4 | plain | 24.41 | 10.91 | 0.96 | 10.76 | 0 |
| CUs 256 | 16 | 256 | 16 | 4 | band | 23.18 | 10.23 | 0.88 | 9.14 | 0 |
| CUs 256 | 32 | 256 | 32 | 2 | plain | 24.63 | 12.80 | 1.24 | 9.32 | 0 |
| CUs 256 | 32 | 256 | 32 | 2 | band | 24.67 | 13.11 | 1.20 | 9.44 | 0 |
| CUs 256 | 64 | 256 | 64 | 1 | plain | 24.02 | 13.45 | 1.06 | 9.42 | 0 |
| CUs 256 | 64 | 256 | 64 | 1 | band | 24.44 | 13.70 | 1.41 | 9.02 | 0 |
