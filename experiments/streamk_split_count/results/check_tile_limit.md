
# GEMM_TILE_LIMIT check (sweep C cases)

4281 of 4281 cases ran, 6 launches each over 3 rotating inputs, on 4096x8192xK with B and C padded as bench.py pads them.

| config | cases | launches with wrong tiles | partially written tiles | tiles written that should not be, or missing | flags or counts left up | errors |
|---|---|---|---|---|---|---|
| v13 | 105 | 0 | 0 | 0 | 0 | 0 |
| v13 base | 14 | 0 | 0 | 0 | 0 | 0 |
| v16 | 473 | 0 | 0 | 0 | 0 | 0 |
| v16 two_tile | 75 | 0 | 0 | 0 | 0 | 0 |
| v19 | 75 | 0 | 0 | 0 | 0 | 0 |
| v20 owner | 621 | 0 | 0 | 0 | 0 | 0 |
| v20 rs | 621 | 0 | 0 | 0 | 0 | 0 |
| v20 rs 1xcd | 194 | 0 | 0 | 0 | 0 | 0 |
| v20 spread | 473 | 0 | 0 | 0 | 0 | 0 |
| v22 atomic | 621 | 0 | 0 | 0 | 0 | 0 |
| v22 atomic 1xcd | 194 | 0 | 0 | 0 | 0 | 0 |
| v22 atomic_iz | 621 | 0 | 0 | 0 | 0 | 0 |
| v22 atomic_iz 1xcd | 194 | 0 | 0 | 0 | 0 | 0 |

PASS
