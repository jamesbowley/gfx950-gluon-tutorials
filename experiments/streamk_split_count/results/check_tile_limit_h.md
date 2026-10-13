
# GEMM_TILE_LIMIT check (sweep H cases)

508 of 508 cases ran, 6 launches each over 3 rotating inputs, on 4096x8192xK with B and C padded as bench.py pads them.

| config | cases | launches with wrong tiles | partially written tiles | tiles written that should not be, or missing | flags or counts left up | errors |
|---|---|---|---|---|---|---|
| v13 | 117 | 0 | 0 | 0 | 0 | 0 |
| v25 spread | 40 | 0 | 0 | 0 | 0 | 0 |
| v26 spread | 234 | 0 | 0 | 0 | 0 | 0 |
| v9r dp | 117 | 0 | 0 | 0 | 0 | 0 |

PASS
