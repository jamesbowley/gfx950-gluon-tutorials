# Sweep F: every ending, placement and partial type (v24) on one matrix

rocprof kernel times (1000 launches over rotating inputs, median) on 4096x8192xK with GEMM_TILE_LIMIT = 256 + S. Tail = kernel time minus v13's with 256 tiles at the same K (one control for every configuration). Each split configuration at its best n (STREAMK_NUM_PROGRAMS = S n; d: the default, all 256 programs), n in brackets. `chunk`: v18's chunk-major placement with the rotating role 0; `l2`: v21's XCD-local split with the partials and flags in the XCD's L2; `bf16_pairs`: v23's bf16 partials (about 1.6 roundings); `atomic_iz`: v22's bf16 pk_add into C, zeroed in the GEMM (0.5 + 0.75 sqrt(n) roundings). Raw rows: results/sweep_f.jsonl.

## Full wave alone (v13, T = 256)

| K | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 |
|---|---|---|---|---|---|---|---|
| us | 20.4 | 30.7 | 52.0 | 94.4 | 170.4 | 352.2 | 675.1 |

## Tails at each configuration's best n (us)

| S | K | v13 | v24 two_tile | v20 rs | v24 owner | v24 owner chunk | v24 owner bf16_pairs | v24 rs | v24 rs chunk | v24 rs l2 | v24 rs bf16_pairs | v24 atomic_iz | v24 atomic_iz chunk | fastest | fastest, one rounding |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 512 | 8.0 | 15.3 | 9.7 (1) | 11.6 (1) | 10.9 (1) | 12.0 (1) | 10.2 (d) | 9.8 (d) | 7.0 (d) | 9.3 (1) | 7.7 (d) | 7.4 (d) | v24 rs l2 (1.14x) | v24 rs l2 (1.14x) |
| 1 | 1024 | 13.6 | 13.6 | 9.0 (d) | 16.2 (1) | 15.0 (1) | 15.9 (2) | 9.2 (d) | 9.0 (d) | 8.8 (d) | 7.6 (d) | 8.4 (d) | 8.2 (d) | v24 rs bf16_pairs (1.78x) | v24 rs l2 (1.55x) |
| 1 | 2048 | 25.4 | 14.1 | 10.0 (16) | 21.8 (3) | 21.0 (3) | 19.6 (3) | 9.4 (d) | 9.5 (d) | 8.8 (8) | 6.3 (d) | 6.8 (d) | 6.4 (d) | v24 rs bf16_pairs (4.04x) | v24 rs l2 (2.90x) |
| 1 | 4096 | 51.7 | 14.6 | 8.8 (32) | 29.4 (4) | 28.1 (4) | 27.0 (4) | 8.9 (32) | 9.9 (32) | 9.2 (16) | 7.2 (32) | 6.6 (d) | 6.2 (d) | v24 atomic_iz chunk (8.28x) | v20 rs (5.85x) |
| 1 | 8192 | 111.3 | 20.6 | 16.4 (64) | 47.0 (8) | 44.8 (6) | 43.4 (8) | 16.7 (64) | 16.0 (64) | 14.4 (16) | 14.1 (64) | 16.1 (32) | 16.2 (32) | v24 rs bf16_pairs (7.88x) | v24 rs l2 (7.73x) |
| 1 | 16384 | 231.7 | 21.2 | 8.2 (64) | 54.4 (12) | 53.2 (12) | 46.4 (12) | 9.2 (64) | 8.7 (64) | 12.0 (32) | 6.8 (64) | 7.6 (64) | 7.4 (64) | v24 rs bf16_pairs (33.88x) | v20 rs (28.26x) |
| 2 | 512 | 8.8 | 15.0 | 10.5 (d) | 11.7 (1) | 11.5 (1) | 12.1 (1) | 10.2 (d) | 9.6 (d) | 7.8 (4) | 7.5 (d) | 7.4 (d) | 7.2 (d) | v24 atomic_iz chunk (1.21x) | v24 rs l2 (1.12x) |
| 2 | 1024 | 14.5 | 15.0 | 10.6 (d) | 17.3 (1) | 17.1 (1) | 15.7 (2) | 10.2 (d) | 10.2 (8) | 9.0 (d) | 8.8 (d) | 8.4 (d) | 8.3 (d) | v24 atomic_iz chunk (1.74x) | v24 rs l2 (1.61x) |
| 2 | 2048 | 27.5 | 16.8 | 9.7 (16) | 21.4 (3) | 21.8 (4) | 19.2 (3) | 9.9 (16) | 9.9 (16) | 10.6 (8) | 8.0 (16) | 7.9 (16) | 7.9 (16) | v24 atomic_iz chunk (3.47x) | v20 rs (2.84x) |
| 2 | 4096 | 55.3 | 16.6 | 9.8 (32) | 28.8 (6) | 28.2 (4) | 25.7 (6) | 9.4 (32) | 9.6 (32) | 14.2 (16) | 7.0 (32) | 8.9 (16) | 8.8 (16) | v24 rs bf16_pairs (7.95x) | v24 rs (5.91x) |
| 2 | 8192 | 119.3 | 18.9 | 16.9 (32) | 46.1 (8) | 46.7 (6) | 42.7 (8) | 17.1 (32) | 16.8 (32) | 23.9 (16) | 14.8 (64) | 15.0 (32) | 17.7 (32) | v24 rs bf16_pairs (8.06x) | v24 rs chunk (7.10x) |
| 2 | 16384 | 240.4 | 30.0 | 11.7 (32) | 51.2 (8) | 54.5 (12) | 44.6 (12) | 11.0 (32) | 11.5 (32) | 16.2 (32) | 7.7 (64) | 7.8 (32) | 8.9 (32) | v24 rs bf16_pairs (31.14x) | v24 rs (21.77x) |
| 3 | 512 | 8.8 | 15.0 | 11.0 (1) | 11.7 (1) | 11.8 (1) | 12.0 (1) | 11.2 (d) | 10.8 (d) | 7.9 (4) | 7.9 (d) | 7.5 (d) | 7.6 (d) | v24 atomic_iz (1.18x) | v24 rs l2 (1.12x) |
| 3 | 1024 | 14.8 | 14.8 | 10.4 (8) | 17.2 (1) | 17.4 (1) | 16.3 (2) | 10.7 (8) | 10.3 (8) | 9.1 (4) | 9.4 (8) | 8.6 (d) | 8.8 (8) | v24 atomic_iz (1.72x) | v24 rs l2 (1.63x) |
| 3 | 2048 | 27.7 | 17.0 | 10.6 (16) | 21.0 (3) | 22.8 (4) | 18.9 (3) | 9.9 (16) | 10.4 (16) | 10.9 (8) | 8.5 (16) | 8.3 (16) | 8.2 (16) | v24 atomic_iz chunk (3.38x) | v24 rs (2.80x) |
| 3 | 4096 | 55.6 | 17.2 | 11.4 (16) | 28.6 (6) | 29.4 (4) | 25.2 (6) | 10.8 (16) | 10.8 (32) | 13.7 (16) | 8.8 (32) | 8.6 (16) | 9.0 (16) | v24 atomic_iz (6.44x) | v24 rs (5.17x) |
| 3 | 8192 | 119.0 | 21.6 | 16.9 (32) | 46.5 (6) | 46.4 (8) | 42.6 (8) | 18.3 (32) | 17.9 (32) | 23.9 (16) | 15.3 (32) | 15.4 (32) | 18.1 (16) | v24 rs bf16_pairs (7.79x) | v20 rs (7.05x) |
| 3 | 16384 | 240.6 | 32.8 | 14.3 (32) | 49.6 (12) | 54.2 (12) | 41.8 (12) | 13.4 (32) | 14.9 (32) | 17.2 (d) | 10.8 (32) | 10.0 (32) | 10.8 (32) | v24 atomic_iz (24.16x) | v24 rs (17.90x) |
| 4 | 512 | 8.8 | 15.4 | 11.0 (1) | 11.8 (1) | 11.8 (1) | 11.7 (1) | 11.0 (1) | 11.0 (d) | 8.0 (4) | 8.9 (d) | 8.0 (d) | 7.7 (d) | v24 atomic_iz chunk (1.15x) | v24 rs l2 (1.10x) |
| 4 | 1024 | 14.9 | 16.7 | 10.6 (8) | 17.4 (1) | 17.6 (1) | 15.7 (3) | 10.9 (8) | 11.0 (8) | 9.2 (4) | 9.1 (8) | 9.1 (d) | 9.0 (8) | v24 atomic_iz chunk (1.65x) | v24 rs l2 (1.62x) |
| 4 | 2048 | 27.7 | 18.8 | 10.3 (16) | 21.1 (3) | 21.9 (4) | 19.5 (3) | 10.2 (16) | 10.6 (16) | 11.0 (8) | 8.6 (16) | 9.2 (12) | 8.9 (12) | v24 rs bf16_pairs (3.21x) | v24 rs (2.72x) |
| 4 | 4096 | 55.6 | 18.8 | 11.6 (16) | 28.6 (6) | 28.5 (4) | 26.0 (6) | 10.4 (16) | 10.8 (16) | 14.2 (16) | 8.8 (16) | 9.1 (16) | 9.8 (16) | v24 rs bf16_pairs (6.29x) | v24 rs (5.34x) |
| 4 | 8192 | 119.2 | 26.7 | 21.0 (32) | 46.2 (8) | 46.8 (6) | 41.8 (8) | 20.5 (16) | 19.3 (32) | 24.0 (16) | 15.0 (32) | 16.5 (32) | 21.0 (16) | v24 rs bf16_pairs (7.97x) | v24 rs chunk (6.18x) |
| 4 | 16384 | 241.2 | 33.5 | 16.6 (32) | 47.3 (12) | 55.3 (12) | 40.4 (12) | 16.6 (32) | 18.0 (32) | 20.7 (32) | 12.6 (32) | 14.0 (32) | 15.1 (32) | v24 rs bf16_pairs (19.21x) | v24 rs (14.57x) |
| 6 | 512 | 9.0 | 16.3 | 11.2 (1) | 11.8 (1) | 12.0 (1) | 12.3 (1) | 11.2 (1) | 11.3 (1) | 8.7 (4) | 10.7 (4) | 8.7 (4) | 8.8 (d) | v24 atomic_iz (1.04x) | v24 rs l2 (1.03x) |
| 6 | 1024 | 15.1 | 16.6 | 11.4 (8) | 17.4 (1) | 17.6 (1) | 15.7 (2) | 11.4 (8) | 11.2 (8) | 9.4 (4) | 9.0 (8) | 9.3 (8) | 8.8 (8) | v24 atomic_iz chunk (1.71x) | v24 rs l2 (1.60x) |
| 6 | 2048 | 27.8 | 19.4 | 12.3 (8) | 20.5 (3) | 21.8 (4) | 18.7 (3) | 10.8 (8) | 11.6 (8) | 12.0 (8) | 9.3 (8) | 9.5 (8) | 9.4 (8) | v24 rs bf16_pairs (3.00x) | v24 rs (2.56x) |
| 6 | 4096 | 55.8 | 20.7 | 14.2 (16) | 28.7 (4) | 28.7 (4) | 26.4 (4) | 13.3 (16) | 13.2 (16) | 15.9 (8) | 10.8 (16) | 11.0 (12) | 12.2 (12) | v24 rs bf16_pairs (5.16x) | v24 rs chunk (4.22x) |
| 6 | 8192 | 119.4 | 29.4 | 22.4 (16) | 45.6 (6) | 46.8 (8) | 41.3 (8) | 20.4 (16) | 20.9 (16) | 23.4 (16) | 17.2 (32) | 18.1 (16) | 22.8 (12) | v24 rs bf16_pairs (6.92x) | v24 rs (5.86x) |
| 6 | 16384 | 244.0 | 35.6 | 24.0 (16) | 51.6 (12) | 55.0 (12) | 44.8 (12) | 23.6 (16) | 24.8 (16) | 23.1 (16) | 17.7 (32) | 18.8 (16) | 19.7 (16) | v24 rs bf16_pairs (13.80x) | v24 rs l2 (10.57x) |
| 8 | 512 | 8.8 | 17.4 | 11.1 (1) | 11.8 (1) | 11.8 (1) | 12.0 (1) | 11.2 (4) | 11.2 (4) | 8.8 (4) | 10.7 (4) | 8.6 (4) | 8.3 (4) | v24 atomic_iz chunk (1.06x) | v13 (1.00x) |
| 8 | 1024 | 14.7 | 17.8 | 12.2 (8) | 17.2 (2) | 17.4 (2) | 15.5 (2) | 11.6 (4) | 12.0 (4) | 10.0 (4) | 9.8 (8) | 9.8 (4) | 9.6 (6) | v24 atomic_iz chunk (1.53x) | v24 rs l2 (1.47x) |
| 8 | 2048 | 27.6 | 20.6 | 13.0 (8) | 20.4 (3) | 21.8 (4) | 18.5 (3) | 11.4 (8) | 12.3 (8) | 12.6 (4) | 10.3 (8) | 10.9 (8) | 10.5 (8) | v24 rs bf16_pairs (2.68x) | v24 rs (2.42x) |
| 8 | 4096 | 55.6 | 23.9 | 15.9 (8) | 28.1 (4) | 28.4 (4) | 25.8 (4) | 14.4 (8) | 15.2 (8) | 17.6 (8) | 12.0 (8) | 12.7 (8) | 12.7 (8) | v24 rs bf16_pairs (4.65x) | v24 rs (3.85x) |
| 8 | 8192 | 119.6 | 31.8 | 23.5 (16) | 44.8 (8) | 47.1 (6) | 42.4 (6) | 22.6 (16) | 22.6 (16) | 23.8 (16) | 18.9 (16) | 19.7 (12) | 24.3 (8) | v24 rs bf16_pairs (6.32x) | v24 rs chunk (5.29x) |
| 8 | 16384 | 247.4 | 41.7 | 28.7 (16) | 52.7 (12) | 56.8 (8) | 45.9 (12) | 27.7 (16) | 28.4 (16) | 26.2 (16) | 22.0 (16) | 21.6 (16) | 23.3 (16) | v24 atomic_iz (11.48x) | v24 rs l2 (9.44x) |
| 12 | 512 | 9.2 | 18.2 | 11.2 (1) | 11.6 (1) | 11.6 (1) | 12.0 (1) | 11.5 (1) | 11.3 (1) | 10.5 (2) | 11.6 (4) | 10.4 (4) | 10.8 (4) | v13 (1.00x) | v13 (1.00x) |
| 12 | 1024 | 15.3 | 19.7 | 13.9 (8) | 17.3 (1) | 17.2 (1) | 15.9 (2) | 12.6 (4) | 12.4 (4) | 11.9 (4) | 11.4 (8) | 10.5 (4) | 10.4 (4) | v24 atomic_iz chunk (1.47x) | v24 rs l2 (1.29x) |
| 12 | 2048 | 27.6 | 23.2 | 15.7 (8) | 21.4 (3) | 21.7 (4) | 19.4 (3) | 14.9 (4) | 14.2 (4) | 15.0 (4) | 11.3 (8) | 12.5 (4) | 12.5 (4) | v24 rs bf16_pairs (2.45x) | v24 rs chunk (1.94x) |
| 12 | 4096 | 55.7 | 24.8 | 18.5 (8) | 29.7 (4) | 28.8 (4) | 27.4 (4) | 18.2 (8) | 17.8 (8) | 19.1 (8) | 15.9 (16) | 16.2 (8) | 15.1 (8) | v24 atomic_iz chunk (3.69x) | v24 rs chunk (3.13x) |
| 12 | 8192 | 119.8 | 36.4 | 27.8 (12) | 46.6 (6) | 47.3 (8) | 43.1 (8) | 26.0 (12) | 26.3 (16) | 29.2 (d) | 21.9 (16) | 23.8 (12) | 25.2 (8) | v24 rs bf16_pairs (5.48x) | v24 rs (4.61x) |
| 12 | 16384 | 246.2 | 48.2 | 36.5 (16) | 51.9 (8) | 57.4 (8) | 45.8 (12) | 35.1 (12) | 35.2 (16) | 35.0 (d) | 28.2 (16) | 29.8 (12) | 29.1 (12) | v24 rs bf16_pairs (8.73x) | v24 rs l2 (7.04x) |
| 16 | 512 | 9.0 | 18.7 | 10.8 (1) | 11.5 (1) | 11.9 (1) | 11.9 (1) | 11.1 (1) | 11.4 (1) | 10.2 (1) | 11.3 (4) | 11.2 (3) | 11.0 (3) | v13 (1.00x) | v13 (1.00x) |
| 16 | 1024 | 14.7 | 20.4 | 15.4 (4) | 17.4 (1) | 18.2 (1) | 16.2 (2) | 13.0 (4) | 13.0 (4) | 13.1 (4) | 11.7 (4) | 12.0 (3) | 11.4 (4) | v24 atomic_iz chunk (1.29x) | v24 rs (1.14x) |
| 16 | 2048 | 27.5 | 26.3 | 17.0 (8) | 20.6 (3) | 22.5 (4) | 19.2 (3) | 15.0 (4) | 14.4 (4) | 16.8 (4) | 12.8 (8) | 13.4 (4) | 13.4 (4) | v24 rs bf16_pairs (2.15x) | v24 rs chunk (1.90x) |
| 16 | 4096 | 55.7 | 26.0 | 21.4 (8) | 28.8 (4) | 29.2 (4) | 26.3 (4) | 20.0 (6) | 20.2 (8) | 21.2 (8) | 16.3 (8) | 18.4 (6) | 16.6 (6) | v24 rs bf16_pairs (3.42x) | v24 rs (2.79x) |
| 16 | 8192 | 119.6 | 36.5 | 30.3 (8) | 46.3 (6) | 47.2 (6) | 42.8 (6) | 28.6 (8) | 29.1 (8) | 30.8 (8) | 24.2 (12) | 25.2 (8) | 24.3 (8) | v24 rs bf16_pairs (4.94x) | v24 rs (4.19x) |
| 16 | 16384 | 247.8 | 59.7 | 43.3 (12) | 54.5 (8) | 57.7 (8) | 48.2 (8) | 42.0 (12) | 40.7 (12) | 41.0 (12) | 32.7 (d) | 33.8 (12) | 32.3 (12) | v24 atomic_iz chunk (7.68x) | v24 rs chunk (6.09x) |
| 24 | 512 | 9.4 | 20.1 | 11.4 (1) | 12.2 (1) | 11.9 (1) | 12.6 (1) | 11.5 (1) | 11.6 (1) | 10.7 (1) | 11.1 (1) | 12.5 (2) | 11.7 (2) | v13 (1.00x) | v13 (1.00x) |
| 24 | 1024 | 15.7 | 23.7 | 17.7 (1) | 18.2 (1) | 18.1 (1) | 16.5 (2) | 16.1 (4) | 15.0 (4) | 15.9 (2) | 13.9 (4) | 14.0 (3) | 14.1 (3) | v24 rs bf16_pairs (1.13x) | v24 rs chunk (1.05x) |
| 24 | 2048 | 27.7 | 27.9 | 20.8 (4) | 22.5 (3) | 23.8 (4) | 20.2 (3) | 19.2 (4) | 16.8 (4) | 20.4 (4) | 16.2 (4) | 16.9 (4) | 17.0 (4) | v24 rs bf16_pairs (1.72x) | v24 rs chunk (1.65x) |
| 24 | 4096 | 55.9 | 31.9 | 26.2 (4) | 30.6 (4) | 30.2 (4) | 28.4 (4) | 23.8 (4) | 23.6 (4) | 26.0 (4) | 21.6 (8) | 21.6 (4) | 21.6 (4) | v24 rs bf16_pairs (2.59x) | v24 rs chunk (2.37x) |
| 24 | 8192 | 119.6 | 46.4 | 40.4 (6) | 48.9 (6) | 48.6 (6) | 45.9 (6) | 37.2 (6) | 34.0 (8) | 38.4 (8) | 31.0 (8) | 33.2 (6) | 30.3 (8) | v24 atomic_iz chunk (3.95x) | v24 rs chunk (3.51x) |
| 24 | 16384 | 239.6 | 72.8 | 59.8 (8) | 62.7 (6) | 65.5 (6) | 60.5 (6) | 58.8 (8) | 54.8 (d) | 54.8 (d) | 49.2 (d) | 51.0 (d) | 46.1 (d) | v24 atomic_iz chunk (5.20x) | v24 rs chunk (4.37x) |
| 32 | 512 | 9.4 | 22.0 | 11.4 (1) | 12.0 (1) | 11.9 (1) | 12.0 (1) | 11.6 (1) | 11.6 (1) | 11.5 (1) | 12.4 (1) | 13.0 (2) | 13.8 (2) | v13 (1.00x) | v13 (1.00x) |
| 32 | 1024 | 15.8 | 26.9 | 18.1 (1) | 18.5 (1) | 18.7 (1) | 17.2 (2) | 17.8 (2) | 16.3 (4) | 17.5 (1) | 14.8 (4) | 15.6 (2) | 16.3 (2) | v24 rs bf16_pairs (1.07x) | v13 (1.00x) |
| 32 | 2048 | 28.1 | 28.0 | 24.0 (4) | 23.2 (3) | 24.1 (3) | 21.7 (2) | 20.8 (4) | 18.7 (4) | 23.1 (4) | 17.6 (4) | 18.6 (3) | 18.2 (3) | v24 rs bf16_pairs (1.60x) | v24 rs chunk (1.50x) |
| 32 | 4096 | 55.7 | 31.6 | 30.1 (4) | 32.1 (3) | 32.8 (4) | 29.9 (4) | 27.8 (4) | 27.5 (4) | 29.7 (4) | 24.4 (4) | 26.0 (3) | 25.2 (3) | v24 rs bf16_pairs (2.29x) | v24 rs chunk (2.03x) |
| 32 | 8192 | 119.9 | 47.0 | 45.9 (6) | 51.3 (6) | 50.9 (4) | 47.8 (6) | 41.6 (6) | 39.8 (6) | 42.8 (6) | 35.9 (8) | 37.0 (6) | 33.6 (6) | v24 atomic_iz chunk (3.56x) | v24 rs chunk (3.02x) |
| 32 | 16384 | 237.8 | 128.0 | 68.0 (8) | 73.1 (6) | 75.6 (d) | 66.7 (6) | 65.9 (d) | 63.0 (8) | 63.5 (8) | 56.0 (d) | 58.2 (8) | 54.8 (d) | v24 atomic_iz chunk (4.34x) | v24 rs chunk (3.78x) |
| 38 | 512 | 10.0 | 23.1 | 11.6 (1) | 12.4 (1) | 12.6 (1) | 12.3 (1) | 12.0 (1) | 11.9 (1) | 10.7 (1) | 12.0 (1) | 14.3 (2) | 14.9 (2) | v13 (1.00x) | v13 (1.00x) |
| 38 | 1024 | 16.5 | 29.2 | 18.3 (1) | 19.2 (1) | 19.3 (1) | 17.8 (2) | 18.6 (1) | 18.1 (2) | 17.7 (1) | 17.2 (2) | 17.0 (2) | 16.8 (3) | v13 (1.00x) | v13 (1.00x) |
| 38 | 2048 | 28.2 | 30.4 | 26.6 (4) | 25.0 (2) | 24.3 (2) | 23.1 (2) | 23.0 (2) | 21.4 (4) | 26.4 (4) | 20.0 (4) | 21.7 (2) | 20.0 (3) | v24 rs bf16_pairs (1.41x) | v24 rs chunk (1.31x) |
| 38 | 4096 | 56.0 | 36.9 | 34.5 (4) | 34.7 (3) | 33.9 (3) | 32.2 (3) | 32.8 (3) | 30.8 (4) | 32.5 (4) | 27.8 (4) | 29.7 (3) | 27.5 (3) | v24 atomic_iz chunk (2.04x) | v24 rs chunk (1.82x) |
| 38 | 8192 | 119.8 | 56.2 | 51.5 (4) | 54.6 (4) | 52.1 (4) | 51.3 (4) | 49.7 (4) | 45.8 (4) | 49.2 (4) | 42.9 (6) | 43.4 (4) | 39.5 (6) | v24 atomic_iz chunk (3.03x) | v24 rs chunk (2.61x) |
| 38 | 16384 | 235.0 | 141.6 | 79.6 (6) | 81.2 (4) | 79.0 (4) | 75.0 (6) | 76.2 (6) | 73.9 (6) | 72.0 (6) | 68.4 (6) | 67.8 (6) | 62.5 (6) | v24 atomic_iz chunk (3.76x) | v24 rs l2 (3.27x) |
| 48 | 512 | 10.0 | 25.2 | 11.9 (1) | 12.6 (1) | 12.7 (1) | 12.4 (1) | 11.9 (1) | 12.0 (1) | 11.2 (1) | 11.8 (1) | 16.1 (2) | 16.0 (1) | v13 (1.00x) | v13 (1.00x) |
| 48 | 1024 | 17.0 | 30.1 | 18.8 (1) | 19.5 (1) | 19.7 (1) | 18.4 (2) | 19.0 (1) | 18.8 (2) | 18.0 (1) | 18.3 (2) | 18.5 (2) | 19.4 (2) | v13 (1.00x) | v13 (1.00x) |
| 48 | 2048 | 28.2 | 31.8 | 27.8 (2) | 24.9 (2) | 24.7 (2) | 23.3 (2) | 24.2 (2) | 23.3 (2) | 25.8 (2) | 21.9 (2) | 23.2 (2) | 22.8 (3) | v24 rs bf16_pairs (1.29x) | v24 rs chunk (1.21x) |
| 48 | 4096 | 55.5 | 40.6 | 38.2 (4) | 35.4 (3) | 35.5 (3) | 31.6 (3) | 36.4 (3) | 33.3 (4) | 36.6 (3) | 30.4 (4) | 32.0 (3) | 31.8 (3) | v24 rs bf16_pairs (1.83x) | v24 rs chunk (1.67x) |
| 48 | 8192 | 119.9 | 63.4 | 55.8 (4) | 55.8 (4) | 55.3 (4) | 52.7 (4) | 52.6 (4) | 51.4 (4) | 51.1 (d) | 47.0 (4) | 45.8 (4) | 43.3 (d) | v24 atomic_iz chunk (2.77x) | v24 rs l2 (2.35x) |
| 48 | 16384 | 240.0 | 181.4 | 94.5 (d) | 93.2 (d) | 90.4 (d) | 84.1 (d) | 92.8 (d) | 88.8 (d) | 81.0 (d) | 79.0 (d) | 74.7 (d) | 70.8 (d) | v24 atomic_iz chunk (3.39x) | v24 rs l2 (2.96x) |
| 64 | 512 | 10.7 | 26.6 | 12.4 (1) | 13.0 (1) | 13.2 (1) | 13.1 (1) | 12.7 (1) | 12.7 (1) | 11.7 (1) | 12.6 (1) | 17.6 (1) | 17.7 (1) | v13 (1.00x) | v13 (1.00x) |
| 64 | 1024 | 17.6 | 28.5 | 19.4 (1) | 19.8 (1) | 19.9 (1) | 19.7 (2) | 19.7 (1) | 19.6 (1) | 18.8 (1) | 18.1 (2) | 22.4 (2) | 22.7 (2) | v13 (1.00x) | v13 (1.00x) |
| 64 | 2048 | 28.7 | 35.0 | 30.3 (1) | 26.1 (2) | 25.8 (2) | 24.2 (2) | 28.0 (2) | 26.3 (2) | 28.7 (2) | 25.0 (2) | 26.2 (2) | 26.6 (2) | v24 owner bf16_pairs (1.19x) | v24 owner chunk (1.11x) |
| 64 | 4096 | 56.3 | 47.8 | 44.3 (2) | 39.5 (3) | 40.3 (3) | 36.8 (3) | 42.2 (3) | 38.8 (4) | 43.0 (4) | 35.0 (4) | 37.8 (3) | 37.4 (3) | v24 rs bf16_pairs (1.61x) | v24 rs chunk (1.45x) |
| 64 | 8192 | 120.3 | 67.6 | 66.4 (d) | 62.4 (3) | 62.7 (4) | 59.4 (4) | 65.3 (3) | 63.9 (3) | 60.0 (d) | 56.8 (d) | 53.8 (d) | 52.4 (4) | v24 atomic_iz chunk (2.30x) | v24 rs l2 (2.00x) |
| 64 | 16384 | 241.8 | 205.5 | 106.5 (4) | 102.9 (4) | 103.6 (d) | 95.1 (4) | 103.5 (d) | 104.2 (4) | 98.3 (4) | 93.3 (4) | 95.2 (4) | 93.0 (4) | v24 atomic_iz chunk (2.60x) | v24 rs l2 (2.46x) |
| 96 | 512 | 13.4 | 27.8 | 14.8 (1) | 15.2 (1) | 15.4 (1) | 15.2 (1) | 15.0 (1) | 15.3 (1) | 14.4 (1) | 14.8 (1) | 21.0 (1) | 21.1 (1) | v13 (1.00x) | v13 (1.00x) |
| 96 | 1024 | 20.4 | 32.8 | 21.8 (1) | 22.4 (1) | 22.4 (1) | 22.4 (1) | 22.0 (1) | 22.0 (1) | 21.4 (1) | 21.9 (1) | 26.4 (1) | 26.9 (1) | v13 (1.00x) | v13 (1.00x) |
| 96 | 2048 | 35.2 | 44.4 | 36.0 (1) | 33.3 (2) | 34.7 (2) | 29.5 (2) | 36.5 (1) | 36.0 (1) | 35.4 (1) | 31.0 (2) | 35.3 (2) | 35.0 (2) | v24 owner bf16_pairs (1.19x) | v24 owner (1.06x) |
| 96 | 4096 | 60.3 | 69.9 | 57.4 (2) | 47.7 (d) | 49.3 (2) | 44.9 (d) | 53.8 (2) | 53.1 (2) | 54.2 (d) | 47.3 (d) | 49.6 (d) | 49.5 (d) | v24 owner bf16_pairs (1.34x) | v24 owner (1.27x) |
| 96 | 8192 | 120.7 | 114.3 | 92.7 (2) | 81.9 (d) | 84.7 (2) | 75.5 (d) | 87.4 (d) | 88.2 (2) | 87.0 (d) | 77.0 (d) | 73.6 (d) | 77.2 (d) | v24 atomic_iz (1.64x) | v24 owner (1.47x) |
| 96 | 16384 | 235.5 | 325.4 | 157.0 (2) | 154.3 (2) | 149.9 (2) | 149.7 (d) | 161.5 (2) | 156.1 (2) | 155.5 (2) | 152.0 (2) | 150.5 (d) | 146.4 (2) | v24 atomic_iz chunk (1.61x) | v24 owner chunk (1.57x) |
| 128 | 512 | 12.8 | 28.4 | 15.1 (1) | 15.5 (1) | 15.6 (1) | 15.7 (1) | 15.3 (1) | 14.8 (1) | 14.2 (1) | 14.8 (1) | 24.2 (1) | 24.2 (1) | v13 (1.00x) | v13 (1.00x) |
| 128 | 1024 | 20.2 | 33.9 | 22.0 (1) | 22.4 (1) | 23.0 (1) | 22.9 (1) | 22.1 (1) | 22.1 (1) | 21.8 (1) | 22.4 (1) | 28.9 (1) | 29.2 (1) | v13 (1.00x) | v13 (1.00x) |
| 128 | 2048 | 33.0 | 44.5 | 36.4 (1) | 35.7 (d) | 34.7 (2) | 32.6 (d) | 36.6 (1) | 37.2 (1) | 35.4 (1) | 34.8 (d) | 41.5 (1) | 41.8 (1) | v24 owner bf16_pairs (1.01x) | v13 (1.00x) |
| 128 | 4096 | 62.6 | 65.4 | 62.2 (1) | 55.7 (2) | 54.7 (d) | 53.7 (d) | 61.2 (d) | 59.8 (d) | 62.3 (d) | 54.8 (d) | 58.5 (2) | 58.3 (2) | v24 owner bf16_pairs (1.17x) | v24 owner chunk (1.15x) |
| 128 | 8192 | 123.6 | 108.3 | 111.6 (2) | 97.4 (d) | 96.3 (d) | 92.3 (2) | 106.6 (2) | 104.7 (2) | 98.8 (2) | 94.2 (2) | 92.2 (2) | 92.5 (2) | v24 atomic_iz (1.34x) | v24 owner chunk (1.28x) |
| 128 | 16384 | 282.9 | 265.4 | 201.6 (d) | 184.4 (2) | 186.4 (d) | 179.3 (d) | 197.7 (d) | 195.2 (d) | 192.3 (d) | 183.3 (d) | 186.3 (d) | 185.2 (2) | v24 owner bf16_pairs (1.58x) | v24 owner (1.53x) |

## Comparisons (tail ratio at each one's best n: median (10th-90th percentile), and how often the first is shorter), S <= 128

| comparison | what it isolates | K | ratio |
|---|---|---|---|
| v24 atomic_iz / v24 rs bf16_pairs | atomics against bf16 reduce-scatter, like for like | all | 1.038 (0.946-1.193), shorter in 25 of 90 |
| v24 atomic_iz / v24 rs bf16_pairs | atomics against bf16 reduce-scatter, like for like | 512 | 0.989 (0.810-1.416), shorter in 8 of 15 |
| v24 atomic_iz / v24 rs bf16_pairs | atomics against bf16 reduce-scatter, like for like | 1024 | 1.013 (0.923-1.236), shorter in 4 of 15 |
| v24 atomic_iz / v24 rs bf16_pairs | atomics against bf16 reduce-scatter, like for like | 2048 | 1.055 (0.990-1.137), shorter in 2 of 15 |
| v24 atomic_iz / v24 rs bf16_pairs | atomics against bf16 reduce-scatter, like for like | 4096 | 1.054 (0.986-1.128), shorter in 2 of 15 |
| v24 atomic_iz / v24 rs bf16_pairs | atomics against bf16 reduce-scatter, like for like | 8192 | 1.030 (0.956-1.104), shorter in 4 of 15 |
| v24 atomic_iz / v24 rs bf16_pairs | atomics against bf16 reduce-scatter, like for like | 16384 | 1.021 (0.946-1.111), shorter in 5 of 15 |
| v24 atomic_iz / v24 rs | atomics against fp32 reduce-scatter (one rounding) | all | 0.882 (0.772-1.134), shorter in 78 of 90 |
| v24 atomic_iz / v24 rs | atomics against fp32 reduce-scatter (one rounding) | 512 | 1.011 (0.723-1.397), shorter in 7 of 15 |
| v24 atomic_iz / v24 rs | atomics against fp32 reduce-scatter (one rounding) | 1024 | 0.872 (0.817-1.198), shorter in 12 of 15 |
| v24 atomic_iz / v24 rs | atomics against fp32 reduce-scatter (one rounding) | 2048 | 0.894 (0.798-0.967), shorter in 14 of 15 |
| v24 atomic_iz / v24 rs | atomics against fp32 reduce-scatter (one rounding) | 4096 | 0.895 (0.803-0.953), shorter in 15 of 15 |
| v24 atomic_iz / v24 rs | atomics against fp32 reduce-scatter (one rounding) | 8192 | 0.874 (0.824-0.914), shorter in 15 of 15 |
| v24 atomic_iz / v24 rs | atomics against fp32 reduce-scatter (one rounding) | 16384 | 0.843 (0.741-0.932), shorter in 15 of 15 |
| v24 rs bf16_pairs / v24 rs | bf16 partials against fp32, reduce-scatter | all | 0.856 (0.758-0.987), shorter in 85 of 90 |
| v24 rs bf16_pairs / v24 rs | bf16 partials against fp32, reduce-scatter | 512 | 0.969 (0.731-1.022), shorter in 11 of 15 |
| v24 rs bf16_pairs / v24 rs | bf16 partials against fp32, reduce-scatter | 1024 | 0.881 (0.832-0.996), shorter in 14 of 15 |
| v24 rs bf16_pairs / v24 rs | bf16 partials against fp32, reduce-scatter | 2048 | 0.851 (0.758-0.906), shorter in 15 of 15 |
| v24 rs bf16_pairs / v24 rs | bf16 partials against fp32, reduce-scatter | 4096 | 0.833 (0.811-0.895), shorter in 15 of 15 |
| v24 rs bf16_pairs / v24 rs | bf16 partials against fp32, reduce-scatter | 8192 | 0.849 (0.834-0.884), shorter in 15 of 15 |
| v24 rs bf16_pairs / v24 rs | bf16 partials against fp32, reduce-scatter | 16384 | 0.803 (0.743-0.927), shorter in 15 of 15 |
| v24 owner bf16_pairs / v24 owner | bf16 partials against fp32, owner | all | 0.926 (0.886-1.020), shorter in 79 of 90 |
| v24 owner bf16_pairs / v24 owner | bf16 partials against fp32, owner | 512 | 1.020 (0.990-1.038), shorter in 5 of 15 |
| v24 owner bf16_pairs / v24 owner | bf16 partials against fp32, owner | 1024 | 0.927 (0.903-0.998), shorter in 14 of 15 |
| v24 owner bf16_pairs / v24 owner | bf16 partials against fp32, owner | 2048 | 0.914 (0.897-0.934), shorter in 15 of 15 |
| v24 owner bf16_pairs / v24 owner | bf16 partials against fp32, owner | 4096 | 0.919 (0.893-0.941), shorter in 15 of 15 |
| v24 owner bf16_pairs / v24 owner | bf16 partials against fp32, owner | 8192 | 0.926 (0.906-0.947), shorter in 15 of 15 |
| v24 owner bf16_pairs / v24 owner | bf16 partials against fp32, owner | 16384 | 0.884 (0.854-0.970), shorter in 15 of 15 |
| v24 owner chunk / v24 owner | chunk-major against tile-major, owner | all | 1.009 (0.971-1.065), shorter in 34 of 90 |
| v24 owner chunk / v24 owner | chunk-major against tile-major, owner | 512 | 1.003 (0.977-1.024), shorter in 5 of 15 |
| v24 owner chunk / v24 owner | chunk-major against tile-major, owner | 1024 | 1.007 (0.991-1.027), shorter in 5 of 15 |
| v24 owner chunk / v24 owner | chunk-major against tile-major, owner | 2048 | 1.038 (0.971-1.090), shorter in 5 of 15 |
| v24 owner chunk / v24 owner | chunk-major against tile-major, owner | 4096 | 0.999 (0.969-1.025), shorter in 8 of 15 |
| v24 owner chunk / v24 owner | chunk-major against tile-major, owner | 8192 | 1.005 (0.954-1.034), shorter in 7 of 15 |
| v24 owner chunk / v24 owner | chunk-major against tile-major, owner | 16384 | 1.044 (0.971-1.106), shorter in 4 of 15 |
| v24 rs chunk / v24 rs | chunk-major against tile-major, reduce-scatter | all | 0.992 (0.934-1.040), shorter in 52 of 90 |
| v24 rs chunk / v24 rs | chunk-major against tile-major, reduce-scatter | 512 | 1.000 (0.957-1.019), shorter in 6 of 15 |
| v24 rs chunk / v24 rs | chunk-major against tile-major, reduce-scatter | 1024 | 0.992 (0.933-1.011), shorter in 9 of 15 |
| v24 rs chunk / v24 rs | chunk-major against tile-major, reduce-scatter | 2048 | 0.987 (0.896-1.074), shorter in 9 of 15 |
| v24 rs chunk / v24 rs | chunk-major against tile-major, reduce-scatter | 4096 | 0.992 (0.919-1.055), shorter in 9 of 15 |
| v24 rs chunk / v24 rs | chunk-major against tile-major, reduce-scatter | 8192 | 0.978 (0.923-1.018), shorter in 11 of 15 |
| v24 rs chunk / v24 rs | chunk-major against tile-major, reduce-scatter | 16384 | 0.987 (0.948-1.085), shorter in 8 of 15 |
| v24 atomic_iz chunk / v24 atomic_iz | chunk-major against tile-major, atomics | all | 0.994 (0.939-1.083), shorter in 53 of 90 |
| v24 atomic_iz chunk / v24 atomic_iz | chunk-major against tile-major, atomics | 512 | 1.000 (0.963-1.039), shorter in 7 of 15 |
| v24 atomic_iz chunk / v24 atomic_iz | chunk-major against tile-major, atomics | 1024 | 0.992 (0.957-1.046), shorter in 8 of 15 |
| v24 atomic_iz chunk / v24 atomic_iz | chunk-major against tile-major, atomics | 2048 | 0.987 (0.941-1.008), shorter in 11 of 15 |
| v24 atomic_iz chunk / v24 atomic_iz | chunk-major against tile-major, atomics | 4096 | 0.995 (0.926-1.079), shorter in 11 of 15 |
| v24 atomic_iz chunk / v24 atomic_iz | chunk-major against tile-major, atomics | 8192 | 1.003 (0.910-1.263), shorter in 6 of 15 |
| v24 atomic_iz chunk / v24 atomic_iz | chunk-major against tile-major, atomics | 16384 | 0.977 (0.922-1.088), shorter in 10 of 15 |
| v24 rs l2 / v24 rs | partials through one XCD's L2 against write-through | all | 0.989 (0.861-1.249), shorter in 50 of 90 |
| v24 rs l2 / v24 rs | partials through one XCD's L2 against write-through | 512 | 0.916 (0.710-0.957), shorter in 15 of 15 |
| v24 rs l2 / v24 rs | partials through one XCD's L2 against write-through | 1024 | 0.948 (0.846-0.988), shorter in 14 of 15 |
| v24 rs l2 / v24 rs | partials through one XCD's L2 against write-through | 2048 | 1.072 (0.966-1.114), shorter in 3 of 15 |
| v24 rs l2 / v24 rs | partials through one XCD's L2 against write-through | 4096 | 1.060 (1.005-1.365), shorter in 1 of 15 |
| v24 rs l2 / v24 rs | partials through one XCD's L2 against write-through | 8192 | 1.032 (0.919-1.306), shorter in 6 of 15 |
| v24 rs l2 / v24 rs | partials through one XCD's L2 against write-through | 16384 | 0.972 (0.933-1.309), shorter in 11 of 15 |
| v24 rs / v20 rs | v23's band store and power-of-two readers, against v20's slice | all | 0.972 (0.907-1.025), shorter in 60 of 90 |
| v24 rs / v20 rs | v23's band store and power-of-two readers, against v20's slice | 512 | 1.013 (0.997-1.029), shorter in 2 of 15 |
| v24 rs / v20 rs | v23's band store and power-of-two readers, against v20's slice | 1024 | 1.005 (0.902-1.030), shorter in 7 of 15 |
| v24 rs / v20 rs | v23's band store and power-of-two readers, against v20's slice | 2048 | 0.923 (0.867-1.014), shorter in 12 of 15 |
| v24 rs / v20 rs | v23's band store and power-of-two readers, against v20's slice | 4096 | 0.941 (0.909-0.985), shorter in 14 of 15 |
| v24 rs / v20 rs | v23's band store and power-of-two readers, against v20's slice | 8192 | 0.955 (0.911-1.020), shorter in 12 of 15 |
| v24 rs / v20 rs | v23's band store and power-of-two readers, against v20's slice | 16384 | 0.972 (0.942-1.029), shorter in 13 of 15 |
| v24 rs / v24 owner | reduce-scatter against owner (both fp32) | all | 0.877 (0.376-1.047), shorter in 77 of 90 |
| v24 rs / v24 owner | reduce-scatter against owner (both fp32) | 512 | 0.956 (0.880-0.986), shorter in 15 of 15 |
| v24 rs / v24 owner | reduce-scatter against owner (both fp32) | 1024 | 0.743 (0.588-0.986), shorter in 15 of 15 |
| v24 rs / v24 owner | reduce-scatter against owner (both fp32) | 2048 | 0.731 (0.464-1.072), shorter in 12 of 15 |
| v24 rs / v24 owner | reduce-scatter against owner (both fp32) | 4096 | 0.692 (0.325-1.099), shorter in 11 of 15 |
| v24 rs / v24 owner | reduce-scatter against owner (both fp32) | 8192 | 0.617 (0.371-1.067), shorter in 12 of 15 |
| v24 rs / v24 owner | reduce-scatter against owner (both fp32) | 16384 | 0.770 (0.216-1.047), shorter in 12 of 15 |
| v24 two_tile / v24 rs | two-tile (v19) against one-tile reduce-scatter | all | 1.491 (1.137-1.955), shorter in 0 of 90 |
| v24 two_tile / v24 rs | two-tile (v19) against one-tile reduce-scatter | 512 | 1.686 (1.400-2.098), shorter in 0 of 15 |
| v24 two_tile / v24 rs | two-tile (v19) against one-tile reduce-scatter | 1024 | 1.507 (1.444-1.574), shorter in 0 of 15 |
| v24 two_tile / v24 rs | two-tile (v19) against one-tile reduce-scatter | 2048 | 1.491 (1.218-1.807), shorter in 0 of 15 |
| v24 two_tile / v24 rs | two-tile (v19) against one-tile reduce-scatter | 4096 | 1.339 (1.115-1.769), shorter in 0 of 15 |
| v24 two_tile / v24 rs | two-tile (v19) against one-tile reduce-scatter | 8192 | 1.232 (1.035-1.405), shorter in 0 of 15 |
| v24 two_tile / v24 rs | two-tile (v19) against one-tile reduce-scatter | 16384 | 1.941 (1.342-2.438), shorter in 0 of 15 |

## Fastest configuration per cell, against data-parallel (v13)

| S | K = 512 | K = 1024 | K = 2048 | K = 4096 | K = 8192 | K = 16384 |
|---|---|---|---|---|---|---|
| 1 | rs l2 7.0 (d) | rs bf16_pairs 7.6 (d); one rounding: rs l2 8.8 (d) | rs bf16_pairs 6.3 (d); one rounding: rs l2 8.8 (8) | atomic_iz chunk 6.2 (d); one rounding: v20 rs 8.8 (32) | rs bf16_pairs 14.1 (64); one rounding: rs l2 14.4 (16) | rs bf16_pairs 6.8 (64); one rounding: v20 rs 8.2 (64) |
| 2 | atomic_iz chunk 7.2 (d); one rounding: rs l2 7.8 (4) | atomic_iz chunk 8.3 (d); one rounding: rs l2 9.0 (d) | atomic_iz chunk 7.9 (16); one rounding: v20 rs 9.7 (16) | rs bf16_pairs 7.0 (32); one rounding: rs 9.4 (32) | rs bf16_pairs 14.8 (64); one rounding: rs chunk 16.8 (32) | rs bf16_pairs 7.7 (64); one rounding: rs 11.0 (32) |
| 3 | atomic_iz 7.5 (d); one rounding: rs l2 7.9 (4) | atomic_iz 8.6 (d); one rounding: rs l2 9.1 (4) | atomic_iz chunk 8.2 (16); one rounding: rs 9.9 (16) | atomic_iz 8.6 (16); one rounding: rs 10.8 (16) | rs bf16_pairs 15.3 (32); one rounding: v20 rs 16.9 (32) | atomic_iz 10.0 (32); one rounding: rs 13.4 (32) |
| 4 | atomic_iz chunk 7.7 (d); one rounding: rs l2 8.0 (4) | atomic_iz chunk 9.0 (8); one rounding: rs l2 9.2 (4) | rs bf16_pairs 8.6 (16); one rounding: rs 10.2 (16) | rs bf16_pairs 8.8 (16); one rounding: rs 10.4 (16) | rs bf16_pairs 15.0 (32); one rounding: rs chunk 19.3 (32) | rs bf16_pairs 12.6 (32); one rounding: rs 16.6 (32) |
| 6 | atomic_iz 8.7 (4); one rounding: rs l2 8.7 (4) | atomic_iz chunk 8.8 (8); one rounding: rs l2 9.4 (4) | rs bf16_pairs 9.3 (8); one rounding: rs 10.8 (8) | rs bf16_pairs 10.8 (16); one rounding: rs chunk 13.2 (16) | rs bf16_pairs 17.2 (32); one rounding: rs 20.4 (16) | rs bf16_pairs 17.7 (32); one rounding: rs l2 23.1 (16) |
| 8 | atomic_iz chunk 8.3 (4); one rounding: v13 8.8 | atomic_iz chunk 9.6 (6); one rounding: rs l2 10.0 (4) | rs bf16_pairs 10.3 (8); one rounding: rs 11.4 (8) | rs bf16_pairs 12.0 (8); one rounding: rs 14.4 (8) | rs bf16_pairs 18.9 (16); one rounding: rs chunk 22.6 (16) | atomic_iz 21.6 (16); one rounding: rs l2 26.2 (16) |
| 12 | v13 9.2 | atomic_iz chunk 10.4 (4); one rounding: rs l2 11.9 (4) | rs bf16_pairs 11.3 (8); one rounding: rs chunk 14.2 (4) | atomic_iz chunk 15.1 (8); one rounding: rs chunk 17.8 (8) | rs bf16_pairs 21.9 (16); one rounding: rs 26.0 (12) | rs bf16_pairs 28.2 (16); one rounding: rs l2 35.0 (d) |
| 16 | v13 9.0 | atomic_iz chunk 11.4 (4); one rounding: rs 13.0 (4) | rs bf16_pairs 12.8 (8); one rounding: rs chunk 14.4 (4) | rs bf16_pairs 16.3 (8); one rounding: rs 20.0 (6) | rs bf16_pairs 24.2 (12); one rounding: rs 28.6 (8) | atomic_iz chunk 32.3 (12); one rounding: rs chunk 40.7 (12) |
| 24 | v13 9.4 | rs bf16_pairs 13.9 (4); one rounding: rs chunk 15.0 (4) | rs bf16_pairs 16.2 (4); one rounding: rs chunk 16.8 (4) | rs bf16_pairs 21.6 (8); one rounding: rs chunk 23.6 (4) | atomic_iz chunk 30.3 (8); one rounding: rs chunk 34.0 (8) | atomic_iz chunk 46.1 (d); one rounding: rs chunk 54.8 (d) |
| 32 | v13 9.4 | rs bf16_pairs 14.8 (4); one rounding: v13 15.8 | rs bf16_pairs 17.6 (4); one rounding: rs chunk 18.7 (4) | rs bf16_pairs 24.4 (4); one rounding: rs chunk 27.5 (4) | atomic_iz chunk 33.6 (6); one rounding: rs chunk 39.8 (6) | atomic_iz chunk 54.8 (d); one rounding: rs chunk 63.0 (8) |
| 38 | v13 10.0 | v13 16.5 | rs bf16_pairs 20.0 (4); one rounding: rs chunk 21.4 (4) | atomic_iz chunk 27.5 (3); one rounding: rs chunk 30.8 (4) | atomic_iz chunk 39.5 (6); one rounding: rs chunk 45.8 (4) | atomic_iz chunk 62.5 (6); one rounding: rs l2 72.0 (6) |
| 48 | v13 10.0 | v13 17.0 | rs bf16_pairs 21.9 (2); one rounding: rs chunk 23.3 (2) | rs bf16_pairs 30.4 (4); one rounding: rs chunk 33.3 (4) | atomic_iz chunk 43.3 (d); one rounding: rs l2 51.1 (d) | atomic_iz chunk 70.8 (d); one rounding: rs l2 81.0 (d) |
| 64 | v13 10.7 | v13 17.6 | owner bf16_pairs 24.2 (2); one rounding: owner chunk 25.8 (2) | rs bf16_pairs 35.0 (4); one rounding: rs chunk 38.8 (4) | atomic_iz chunk 52.4 (4); one rounding: rs l2 60.0 (d) | atomic_iz chunk 93.0 (4); one rounding: rs l2 98.3 (4) |
| 96 | v13 13.4 | v13 20.4 | owner bf16_pairs 29.5 (2); one rounding: owner 33.3 (2) | owner bf16_pairs 44.9 (d); one rounding: owner 47.7 (d) | atomic_iz 73.6 (d); one rounding: owner 81.9 (d) | atomic_iz chunk 146.4 (2); one rounding: owner chunk 149.9 (2) |
| 128 | v13 12.8 | v13 20.2 | owner bf16_pairs 32.6 (d); one rounding: v13 33.0 | owner bf16_pairs 53.7 (d); one rounding: owner chunk 54.7 (d) | atomic_iz 92.2 (2); one rounding: owner chunk 96.3 (d) | owner bf16_pairs 179.3 (d); one rounding: owner 184.4 (2) |

## S > 128 (tails, us)

| S | K | v13 | v24 two_tile | v24 spread |
|---|---|---|---|---|
| 160 | 8192 | 137.7 | 205.2 | 141.4 |
| 160 | 16384 | 305.6 | 507.8 | 299.5 |
| 160 | 32768 | 576.8 | 927.0 | 532.2 |
| 192 | 8192 | 146.1 | 199.6 | 151.2 |
| 192 | 16384 | 315.0 | 475.3 | 297.3 |
| 192 | 32768 | 607.4 | 955.4 | 562.6 |
| 224 | 8192 | 159.5 | 271.9 | 172.0 |
| 224 | 16384 | 329.2 | 645.6 | 329.7 |
| 224 | 32768 | 642.2 | 1205.6 | 631.7 |
| 240 | 8192 | 162.7 | 289.8 | 180.1 |
| 240 | 16384 | 344.7 | 637.5 | 346.6 |
| 240 | 32768 | 655.8 | 1280.3 | 668.7 |
| 255 | 8192 | 171.9 | 195.0 | 179.3 |
| 255 | 16384 | 333.7 | 417.8 | 372.1 |
| 255 | 32768 | 667.2 | 832.8 | 684.9 |
