
# GEMM_TILE_LIMIT check (sweep F2T cases)

105 of 105 cases ran, 6 launches each over 3 rotating inputs, on 4096x8192xK with B and C padded as bench.py pads them.

| config | cases | launches with wrong tiles | partially written tiles | tiles written that should not be, or missing | flags or counts left up | errors |
|---|---|---|---|---|---|---|
| v26 two_tile | 105 | 0 | 0 | 0 | 0 | 0 |

PASS
