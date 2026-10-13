
# GEMM_TILE_LIMIT check (sweep C cases)

1699 of 1699 cases ran, 6 launches each over 3 rotating inputs, on 4096x8192xK with B and C padded as bench.py pads them.

| config | cases | launches with wrong tiles | partially written tiles | tiles written that should not be, or missing | flags or counts left up | errors |
|---|---|---|---|---|---|---|
| v13 | 75 | 0 | 0 | 0 | 0 | 0 |
| v13 base | 10 | 0 | 0 | 0 | 0 | 0 |
| v16 | 473 | 0 | 0 | 0 | 0 | 0 |
| v16 two_tile | 75 | 0 | 0 | 0 | 0 | 0 |
| v19 | 75 | 0 | 0 | 0 | 0 | 0 |
| v20 owner | 458 | 0 | 0 | 0 | 0 | 0 |
| v20 rs | 458 | 0 | 0 | 0 | 0 | 0 |
| v20 spread | 75 | 0 | 0 | 0 | 0 | 0 |

PASS
