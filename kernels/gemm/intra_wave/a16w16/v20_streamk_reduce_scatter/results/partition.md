# check_partition.py (pure Python)

all partition checks passed: 2040 cases (S = 1-255, two-tile and spread, K = 8192-65536)
compaction checks passed for SNP = 1-256; partition checks passed for 3636 cases with SNP = S * n (n = 1-8, K = 512-8192)

Spread: distinct k-steps an XCD's programs read at once (lockstep model)

| S | K | q | pairs per program | distinct k, reversed | distinct k, forward (v16) |
|---|---|---|---|---|---|
| 128 | 8192 | 2 | 32 | 2.0 | 2.0 |
| 128 | 32768 | 2 | 128 | 2.0 | 2.0 |
| 160 | 8192 | 8 | 40 | 4.0 | 8.0 |
| 160 | 32768 | 8 | 160 | 4.0 | 8.0 |
| 192 | 8192 | 4 | 48 | 2.0 | 4.0 |
| 192 | 32768 | 4 | 192 | 2.0 | 4.0 |
| 200 | 8192 | 32 | 50 | 8.0 | 32.0 |
| 200 | 32768 | 32 | 200 | 8.0 | 32.0 |
| 224 | 8192 | 8 | 56 | 2.0 | 8.0 |
| 224 | 32768 | 8 | 224 | 2.0 | 8.0 |
| 240 | 8192 | 16 | 60 | 2.0 | 16.0 |
| 240 | 32768 | 16 | 240 | 2.0 | 16.0 |
| 255 | 8192 | 256 | 63+ | 1.1 | 8.7 |
| 255 | 32768 | 256 | 255 | 1.1 | 32.0 |
