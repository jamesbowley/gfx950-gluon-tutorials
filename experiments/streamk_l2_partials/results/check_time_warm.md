HIP_VISIBLE_DEVICES=5 launches/point=20

## check

| T | n | MiB per XCD | variant | XCD-local: bad launches | spread: bad launches | expected |
|---|---|---|---|---|---|---|
| 1 | 4 | 1 | mem | 0/20 | 0/20 | both ok: yes |
| 1 | 4 | 1 | wt_relaxed | 0/20 | 0/20 | both ok: yes |
| 1 | 4 | 1 | l2_cg | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 1 | 4 | 1 | l2_def | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 1 | 4 | 1 | l2_cv | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 1 | 4 | 1 | l2_cg_noinv | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 1 | 4 | 1 | l2_nts | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 1 | 4 | 1 | l2_pre | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 1 | 4 | 1 | l2_flags | 0/20 | not run (would hang) | local ok: yes |
| 1 | 4 | 1 | l2_flags_pre | 0/20 | not run (would hang) | local ok: yes |
| 1 | 4 | | placement | every XCD-local group on one XCC_ID: True | | |
| 1 | 8 | 2 | mem | 0/20 | 0/20 | both ok: yes |
| 1 | 8 | 2 | wt_relaxed | 0/20 | 0/20 | both ok: yes |
| 1 | 8 | 2 | l2_cg | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 1 | 8 | 2 | l2_def | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 1 | 8 | 2 | l2_cv | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 1 | 8 | 2 | l2_cg_noinv | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 1 | 8 | 2 | l2_nts | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 1 | 8 | 2 | l2_pre | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 1 | 8 | 2 | l2_flags | 0/20 | not run (would hang) | local ok: yes |
| 1 | 8 | 2 | l2_flags_pre | 0/20 | not run (would hang) | local ok: yes |
| 1 | 8 | | placement | every XCD-local group on one XCC_ID: True | | |
| 1 | 16 | 4 | mem | 0/20 | 0/20 | both ok: yes |
| 1 | 16 | 4 | wt_relaxed | 0/20 | 0/20 | both ok: yes |
| 1 | 16 | 4 | l2_cg | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 1 | 16 | 4 | l2_def | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 1 | 16 | 4 | l2_cv | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 1 | 16 | 4 | l2_cg_noinv | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 1 | 16 | 4 | l2_nts | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 1 | 16 | 4 | l2_pre | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 1 | 16 | 4 | l2_flags | 0/20 | not run (would hang) | local ok: yes |
| 1 | 16 | 4 | l2_flags_pre | 0/20 | not run (would hang) | local ok: yes |
| 1 | 16 | | placement | every XCD-local group on one XCC_ID: True | | |
| 1 | 32 | 8 | mem | 0/20 | 0/20 | both ok: yes |
| 1 | 32 | 8 | wt_relaxed | 0/20 | 0/20 | both ok: yes |
| 1 | 32 | 8 | l2_cg | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 1 | 32 | 8 | l2_def | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 1 | 32 | 8 | l2_cv | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 1 | 32 | 8 | l2_cg_noinv | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 1 | 32 | 8 | l2_nts | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 1 | 32 | 8 | l2_pre | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 1 | 32 | 8 | l2_flags | 0/20 | not run (would hang) | local ok: yes |
| 1 | 32 | 8 | l2_flags_pre | 0/20 | not run (would hang) | local ok: yes |
| 1 | 32 | | placement | every XCD-local group on one XCC_ID: True | | |
| 2 | 4 | 2 | mem | 0/20 | 0/20 | both ok: yes |
| 2 | 4 | 2 | wt_relaxed | 0/20 | 0/20 | both ok: yes |
| 2 | 4 | 2 | l2_cg | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 2 | 4 | 2 | l2_def | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 2 | 4 | 2 | l2_cv | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 2 | 4 | 2 | l2_cg_noinv | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 2 | 4 | 2 | l2_nts | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 2 | 4 | 2 | l2_pre | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 2 | 4 | 2 | l2_flags | 0/20 | not run (would hang) | local ok: yes |
| 2 | 4 | 2 | l2_flags_pre | 0/20 | not run (would hang) | local ok: yes |
| 2 | 4 | | placement | every XCD-local group on one XCC_ID: True | | |
| 2 | 8 | 4 | mem | 0/20 | 0/20 | both ok: yes |
| 2 | 8 | 4 | wt_relaxed | 0/20 | 0/20 | both ok: yes |
| 2 | 8 | 4 | l2_cg | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 2 | 8 | 4 | l2_def | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 2 | 8 | 4 | l2_cv | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 2 | 8 | 4 | l2_cg_noinv | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 2 | 8 | 4 | l2_nts | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 2 | 8 | 4 | l2_pre | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 2 | 8 | 4 | l2_flags | 0/20 | not run (would hang) | local ok: yes |
| 2 | 8 | 4 | l2_flags_pre | 0/20 | not run (would hang) | local ok: yes |
| 2 | 8 | | placement | every XCD-local group on one XCC_ID: True | | |
| 2 | 16 | 8 | mem | 0/20 | 0/20 | both ok: yes |
| 2 | 16 | 8 | wt_relaxed | 0/20 | 0/20 | both ok: yes |
| 2 | 16 | 8 | l2_cg | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 2 | 16 | 8 | l2_def | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 2 | 16 | 8 | l2_cv | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 2 | 16 | 8 | l2_cg_noinv | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 2 | 16 | 8 | l2_nts | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 2 | 16 | 8 | l2_pre | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 2 | 16 | 8 | l2_flags | 0/20 | not run (would hang) | local ok: yes |
| 2 | 16 | 8 | l2_flags_pre | 0/20 | not run (would hang) | local ok: yes |
| 2 | 16 | | placement | every XCD-local group on one XCC_ID: True | | |
| 4 | 4 | 4 | mem | 0/20 | 0/20 | both ok: yes |
| 4 | 4 | 4 | wt_relaxed | 0/20 | 0/20 | both ok: yes |
| 4 | 4 | 4 | l2_cg | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 4 | 4 | 4 | l2_def | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 4 | 4 | 4 | l2_cv | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 4 | 4 | 4 | l2_cg_noinv | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 4 | 4 | 4 | l2_nts | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 4 | 4 | 4 | l2_pre | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 4 | 4 | 4 | l2_flags | 0/20 | not run (would hang) | local ok: yes |
| 4 | 4 | 4 | l2_flags_pre | 0/20 | not run (would hang) | local ok: yes |
| 4 | 4 | | placement | every XCD-local group on one XCC_ID: True | | |
| 4 | 8 | 8 | mem | 0/20 | 0/20 | both ok: yes |
| 4 | 8 | 8 | wt_relaxed | 0/20 | 0/20 | both ok: yes |
| 4 | 8 | 8 | l2_cg | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 4 | 8 | 8 | l2_def | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 4 | 8 | 8 | l2_cv | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 4 | 8 | 8 | l2_cg_noinv | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 4 | 8 | 8 | l2_nts | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 4 | 8 | 8 | l2_pre | 0/20 | 20/20 | local ok, spread FAIL: yes |
| 4 | 8 | 8 | l2_flags | 0/20 | not run (would hang) | local ok: yes |
| 4 | 8 | 8 | l2_flags_pre | 0/20 | not run (would hang) | local ok: yes |
| 4 | 8 | | placement | every XCD-local group on one XCC_ID: True | | |

## asm

| variant | VGPRs | instructions with cache bits (count) |
|---|---|---|
| mem | 320 | `buffer_load_dwordx4` x64, `buffer_store_dwordx4 sc0 sc1` x64, `buffer_wbl2 sc1` x2, `buffer_inv sc1` x2, `buffer_store_dwordx4` x8, `buffer_load_dwordx4 sc0 sc1` x24 |
| wt_relaxed | 320 | `buffer_load_dwordx4` x64, `buffer_store_dwordx4 sc0 sc1` x64, `buffer_store_dwordx4` x8, `buffer_load_dwordx4 sc0 sc1` x24 |
| l2_cg | 320 | `buffer_load_dwordx4` x64, `buffer_store_dwordx4` x72, `buffer_inv sc0` x1, `buffer_load_dwordx4 sc0 nt` x24 |
| l2_def | 320 | `buffer_load_dwordx4` x88, `buffer_store_dwordx4` x72, `buffer_inv sc0` x1 |
| l2_cv | 320 | `buffer_load_dwordx4` x64, `buffer_store_dwordx4` x72, `buffer_inv sc0` x1, `buffer_load_dwordx4 sc0 sc1` x24 |
| l2_cg_noinv | 320 | `buffer_load_dwordx4` x64, `buffer_store_dwordx4` x72, `buffer_load_dwordx4 sc0 nt` x24 |
| l2_nts | 320 | `buffer_load_dwordx4` x64, `buffer_store_dwordx4 sc0 nt` x64, `buffer_inv sc0` x1, `buffer_store_dwordx4` x8, `buffer_load_dwordx4 sc0 nt` x24 |
| l2_pre | 358 | `buffer_load_dwordx4` x128, `buffer_store_dwordx4` x72, `buffer_inv sc0` x1, `buffer_load_dwordx4 sc0 nt` x24 |
| l2_flags | 316 | `buffer_load_dwordx4` x64, `buffer_store_dwordx4` x72, `buffer_inv sc0` x1, `buffer_load_dwordx4 sc0 nt` x24 |
| l2_flags_pre | 360 | `buffer_load_dwordx4` x128, `buffer_store_dwordx4` x72, `buffer_inv sc0` x1, `buffer_load_dwordx4 sc0 nt` x24 |

## time

Medians over programs and launches, us (k-steps of 1.21 us); warm.

| T | n | MiB per XCD | variant | store | sync | slice | ending | span |
|---|---|---|---|---|---|---|---|---|
| 1 | 4 | 1 | mem | 3.12 (2.58) | 3.56 (2.94) | 3.64 (3.01) | 10.24 (8.46) | 10.91 (9.02) |
| 1 | 4 | 1 | wt_relaxed | 3.12 (2.58) | 2.64 (2.18) | 3.64 (3.01) | 9.40 (7.77) | 10.33 (8.54) |
| 1 | 4 | 1 | l2_cg | 2.12 (1.75) | 2.68 (2.21) | 2.80 (2.31) | 7.60 (6.28) | 8.38 (6.93) |
| 1 | 4 | 1 | l2_def | 2.12 (1.75) | 2.68 (2.21) | 2.80 (2.31) | 7.56 (6.25) | 8.34 (6.89) |
| 1 | 4 | 1 | l2_cv | 2.12 (1.75) | 2.64 (2.18) | 2.80 (2.31) | 7.56 (6.25) | 8.25 (6.82) |
| 1 | 4 | 1 | l2_pre | 2.08 (1.72) | 2.16 (1.79) | 2.80 (2.31) | 7.04 (5.82) | 7.73 (6.39) |
| 1 | 4 | 1 | l2_flags | 2.12 (1.75) | 1.04 (0.86) | 2.80 (2.31) | 5.96 (4.93) | 6.70 (5.53) |
| 1 | 4 | 1 | l2_flags_pre | 2.00 (1.65) | 1.08 (0.89) | 2.80 (2.31) | 5.84 (4.83) | 6.59 (5.45) |
| 1 | 8 | 2 | mem | 3.32 (2.74) | 4.24 (3.50) | 3.48 (2.88) | 11.04 (9.12) | 11.82 (9.76) |
| 1 | 8 | 2 | wt_relaxed | 3.32 (2.74) | 3.28 (2.71) | 3.48 (2.88) | 10.08 (8.33) | 10.96 (9.05) |
| 1 | 8 | 2 | l2_cg | 2.44 (2.02) | 3.32 (2.74) | 2.64 (2.18) | 8.40 (6.94) | 9.34 (7.72) |
| 1 | 8 | 2 | l2_def | 2.44 (2.02) | 3.28 (2.71) | 2.60 (2.15) | 8.36 (6.91) | 9.50 (7.85) |
| 1 | 8 | 2 | l2_cv | 2.44 (2.02) | 3.20 (2.64) | 2.56 (2.12) | 8.28 (6.84) | 9.32 (7.70) |
| 1 | 8 | 2 | l2_pre | 2.40 (1.98) | 3.00 (2.48) | 2.40 (1.98) | 7.76 (6.41) | 8.68 (7.17) |
| 1 | 8 | 2 | l2_flags | 2.44 (2.02) | 0.92 (0.76) | 2.64 (2.18) | 6.04 (4.99) | 6.84 (5.65) |
| 1 | 8 | 2 | l2_flags_pre | 2.36 (1.95) | 1.00 (0.83) | 2.40 (1.98) | 5.72 (4.73) | 6.53 (5.39) |
| 1 | 16 | 4 | mem | 4.48 (3.70) | 7.28 (6.02) | 5.28 (4.36) | 17.32 (14.31) | 19.47 (16.09) |
| 1 | 16 | 4 | wt_relaxed | 4.48 (3.70) | 4.84 (4.00) | 5.28 (4.36) | 14.80 (12.23) | 15.92 (13.15) |
| 1 | 16 | 4 | l2_cg | 3.32 (2.74) | 6.20 (5.12) | 6.08 (5.02) | 15.88 (13.12) | 19.14 (15.82) |
| 1 | 16 | 4 | l2_def | 3.32 (2.74) | 6.48 (5.36) | 6.68 (5.52) | 16.64 (13.75) | 18.05 (14.92) |
| 1 | 16 | 4 | l2_cv | 3.32 (2.74) | 6.48 (5.36) | 6.60 (5.45) | 16.64 (13.75) | 18.19 (15.03) |
| 1 | 16 | 4 | l2_pre | 3.28 (2.71) | 6.00 (4.96) | 4.44 (3.67) | 13.88 (11.47) | 15.59 (12.88) |
| 1 | 16 | 4 | l2_flags | 3.32 (2.74) | 1.64 (1.36) | 6.20 (5.12) | 11.20 (9.26) | 14.00 (11.57) |
| 1 | 16 | 4 | l2_flags_pre | 3.28 (2.71) | 1.72 (1.42) | 4.68 (3.87) | 9.60 (7.93) | 11.16 (9.22) |
| 1 | 32 | 8 | mem | 8.72 (7.21) | 15.12 (12.50) | 9.68 (8.00) | 33.52 (27.70) | 36.54 (30.19) |
| 1 | 32 | 8 | wt_relaxed | 8.76 (7.24) | 8.40 (6.94) | 9.44 (7.80) | 26.52 (21.92) | 28.32 (23.40) |
| 1 | 32 | 8 | l2_cg | 7.24 (5.98) | 10.16 (8.40) | 10.30 (8.51) | 30.20 (24.96) | 36.95 (30.54) |
| 1 | 32 | 8 | l2_def | 7.20 (5.95) | 10.24 (8.46) | 11.04 (9.12) | 29.64 (24.50) | 32.42 (26.79) |
| 1 | 32 | 8 | l2_cv | 7.24 (5.98) | 10.84 (8.96) | 11.28 (9.32) | 30.08 (24.86) | 33.13 (27.38) |
| 1 | 32 | 8 | l2_pre | 7.24 (5.98) | 13.12 (10.84) | 9.64 (7.97) | 31.56 (26.08) | 33.91 (28.03) |
| 1 | 32 | 8 | l2_flags | 6.84 (5.65) | 2.04 (1.69) | 11.70 (9.67) | 20.62 (17.04) | 26.48 (21.88) |
| 1 | 32 | 8 | l2_flags_pre | 6.64 (5.49) | 2.24 (1.85) | 9.72 (8.03) | 18.80 (15.54) | 19.79 (16.35) |
| 2 | 4 | 2 | mem | 3.32 (2.74) | 4.12 (3.40) | 3.88 (3.21) | 11.32 (9.36) | 12.40 (10.25) |
| 2 | 4 | 2 | wt_relaxed | 3.32 (2.74) | 2.52 (2.08) | 3.84 (3.17) | 9.68 (8.00) | 10.77 (8.90) |
| 2 | 4 | 2 | l2_cg | 2.44 (2.02) | 2.60 (2.15) | 3.00 (2.48) | 8.08 (6.68) | 9.25 (7.65) |
| 2 | 4 | 2 | l2_def | 2.44 (2.02) | 2.60 (2.15) | 2.96 (2.45) | 8.08 (6.68) | 9.59 (7.93) |
| 2 | 4 | 2 | l2_cv | 2.44 (2.02) | 2.56 (2.12) | 2.96 (2.45) | 8.08 (6.68) | 9.54 (7.88) |
| 2 | 4 | 2 | l2_pre | 2.36 (1.95) | 2.60 (2.15) | 2.92 (2.41) | 7.92 (6.55) | 8.96 (7.40) |
| 2 | 4 | 2 | l2_flags | 2.44 (2.02) | 0.92 (0.76) | 2.92 (2.41) | 6.36 (5.26) | 7.39 (6.11) |
| 2 | 4 | 2 | l2_flags_pre | 2.36 (1.95) | 0.96 (0.79) | 2.84 (2.35) | 6.12 (5.06) | 6.86 (5.67) |
| 2 | 8 | 4 | mem | 4.52 (3.74) | 7.68 (6.35) | 4.40 (3.64) | 16.86 (13.93) | 18.95 (15.66) |
| 2 | 8 | 4 | wt_relaxed | 4.52 (3.74) | 4.64 (3.83) | 4.40 (3.64) | 13.80 (11.40) | 15.52 (12.83) |
| 2 | 8 | 4 | l2_cg | 3.36 (2.78) | 5.40 (4.46) | 5.16 (4.26) | 14.24 (11.77) | 17.39 (14.37) |
| 2 | 8 | 4 | l2_def | 3.36 (2.78) | 5.40 (4.46) | 6.32 (5.22) | 15.36 (12.69) | 16.99 (14.04) |
| 2 | 8 | 4 | l2_cv | 3.36 (2.78) | 5.68 (4.69) | 6.24 (5.16) | 15.64 (12.93) | 17.65 (14.59) |
| 2 | 8 | 4 | l2_pre | 3.28 (2.71) | 5.68 (4.69) | 2.88 (2.38) | 12.04 (9.95) | 14.04 (11.61) |
| 2 | 8 | 4 | l2_flags | 3.36 (2.78) | 1.44 (1.19) | 5.16 (4.26) | 10.16 (8.40) | 12.66 (10.47) |
| 2 | 8 | 4 | l2_flags_pre | 3.28 (2.71) | 1.56 (1.29) | 3.00 (2.48) | 7.84 (6.48) | 9.82 (8.12) |
| 2 | 16 | 8 | mem | 8.72 (7.21) | 16.12 (13.32) | 8.12 (6.71) | 32.96 (27.24) | 36.29 (29.99) |
| 2 | 16 | 8 | wt_relaxed | 8.68 (7.17) | 7.60 (6.28) | 7.80 (6.45) | 24.20 (20.00) | 26.96 (22.29) |
| 2 | 16 | 8 | l2_cg | 7.20 (5.95) | 9.76 (8.07) | 8.72 (7.21) | 26.32 (21.75) | 31.30 (25.86) |
| 2 | 16 | 8 | l2_def | 7.20 (5.95) | 9.92 (8.20) | 9.88 (8.17) | 27.96 (23.11) | 32.17 (26.59) |
| 2 | 16 | 8 | l2_cv | 7.16 (5.92) | 10.44 (8.63) | 9.92 (8.20) | 28.32 (23.40) | 32.63 (26.97) |
| 2 | 16 | 8 | l2_pre | 7.16 (5.92) | 11.56 (9.55) | 8.44 (6.98) | 28.80 (23.80) | 31.71 (26.20) |
| 2 | 16 | 8 | l2_flags | 6.84 (5.65) | 1.96 (1.62) | 9.08 (7.50) | 17.84 (14.74) | 21.48 (17.75) |
| 2 | 16 | 8 | l2_flags_pre | 6.64 (5.49) | 2.20 (1.82) | 9.16 (7.57) | 18.16 (15.01) | 19.95 (16.48) |
| 4 | 4 | 4 | mem | 4.52 (3.74) | 5.68 (4.69) | 4.76 (3.93) | 15.12 (12.50) | 18.09 (14.95) |
| 4 | 4 | 4 | wt_relaxed | 4.52 (3.74) | 2.80 (2.31) | 4.80 (3.97) | 12.32 (10.18) | 14.32 (11.83) |
| 4 | 4 | 4 | l2_cg | 3.36 (2.78) | 3.44 (2.84) | 5.76 (4.76) | 13.20 (10.91) | 16.52 (13.65) |
| 4 | 4 | 4 | l2_def | 3.36 (2.78) | 3.44 (2.84) | 6.68 (5.52) | 13.68 (11.31) | 15.79 (13.05) |
| 4 | 4 | 4 | l2_cv | 3.36 (2.78) | 3.48 (2.88) | 6.60 (5.45) | 13.68 (11.31) | 15.55 (12.86) |
| 4 | 4 | 4 | l2_pre | 3.28 (2.71) | 3.44 (2.84) | 3.24 (2.68) | 10.16 (8.40) | 12.52 (10.34) |
| 4 | 4 | 4 | l2_flags | 3.36 (2.78) | 1.28 (1.06) | 5.72 (4.73) | 10.80 (8.93) | 13.70 (11.32) |
| 4 | 4 | 4 | l2_flags_pre | 3.24 (2.68) | 1.40 (1.16) | 3.44 (2.84) | 8.04 (6.64) | 10.06 (8.31) |
| 4 | 8 | 8 | mem | 8.72 (7.21) | 11.24 (9.29) | 8.60 (7.11) | 28.40 (23.47) | 33.48 (27.67) |
| 4 | 8 | 8 | wt_relaxed | 8.72 (7.21) | 4.68 (3.87) | 8.16 (6.74) | 21.72 (17.95) | 24.98 (20.64) |
| 4 | 8 | 8 | l2_cg | 7.16 (5.92) | 5.60 (4.63) | 9.08 (7.50) | 22.32 (18.45) | 27.51 (22.74) |
| 4 | 8 | 8 | l2_def | 7.20 (5.95) | 5.72 (4.73) | 10.60 (8.76) | 23.28 (19.24) | 28.55 (23.60) |
| 4 | 8 | 8 | l2_cv | 7.12 (5.88) | 5.56 (4.60) | 10.72 (8.86) | 23.48 (19.40) | 28.28 (23.37) |
| 4 | 8 | 8 | l2_pre | 7.12 (5.88) | 6.44 (5.32) | 8.96 (7.40) | 22.72 (18.78) | 27.45 (22.69) |
| 4 | 8 | 8 | l2_flags | 6.84 (5.65) | 2.04 (1.69) | 8.92 (7.37) | 18.08 (14.94) | 20.88 (17.26) |
| 4 | 8 | 8 | l2_flags_pre | 6.64 (5.49) | 2.56 (2.12) | 8.32 (6.88) | 17.84 (14.74) | 20.69 (17.10) |
