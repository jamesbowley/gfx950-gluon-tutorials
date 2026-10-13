
# GEMM_TILE_LIMIT check (sweep E1 cases)

163 of 163 cases ran, 6 launches each over 3 rotating inputs, on 4096x8192xK with B and C padded as bench.py pads them.

| config | cases | launches with wrong tiles | partially written tiles | tiles written that should not be, or missing | flags or counts left up | errors |
|---|---|---|---|---|---|---|
| v13 base | 3 | 0 | 0 | 0 | 0 | 0 |
| v20 owner | 19 | 0 | 0 | 0 | 0 | 0 |
| v20 rs | 19 | 0 | 0 | 0 | 0 | 0 |
| v22 atomic | 19 | 0 | 0 | 0 | 0 | 0 |
| v22 atomic_iz | 19 | 0 | 0 | 0 | 0 | 0 |
| v23 owner bf16_pairs | 19 | 0 | 0 | 0 | 0 | 0 |
| v23 rs | 19 | 0 | 0 | 0 | 0 | 0 |
| v23 rs all | 8 | 0 | 0 | 0 | 0 | 0 |
| v23 rs bf16 | 19 | 0 | 0 | 0 | 0 | 0 |
| v23 rs bf16_pairs | 19 | 0 | 0 | 0 | 0 | 0 |

PASS
