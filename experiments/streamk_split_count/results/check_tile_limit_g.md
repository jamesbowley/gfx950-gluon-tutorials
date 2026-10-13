
# GEMM_TILE_LIMIT check (sweep G cases)

2003 of 2003 cases ran, 6 launches each over 3 rotating inputs, on 4096x8192xK with B and C padded as bench.py pads them.

| config | cases | launches with wrong tiles | partially written tiles | tiles written that should not be, or missing | flags or counts left up | errors |
|---|---|---|---|---|---|---|
| v13 | 165 | 0 | 0 | 0 | 0 | 0 |
| v25 capped | 150 | 0 | 0 | 0 | 0 | 0 |
| v25 owner | 744 | 0 | 0 | 0 | 0 | 0 |
| v25 rs | 744 | 0 | 0 | 0 | 0 | 0 |
| v25 spread | 35 | 0 | 0 | 0 | 0 | 0 |
| v9r dp | 165 | 0 | 0 | 0 | 0 | 0 |

PASS
