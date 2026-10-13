
# GEMM_TILE_LIMIT check (sweep F cases)

5701 of 5701 cases ran, 6 launches each over 3 rotating inputs, on 4096x8192xK with B and C padded as bench.py pads them.

| config | cases | launches with wrong tiles | partially written tiles | tiles written that should not be, or missing | flags or counts left up | errors |
|---|---|---|---|---|---|---|
| v24 atomic_iz | 621 | 0 | 0 | 0 | 0 | 0 |
| v24 atomic_iz chunk | 621 | 0 | 0 | 0 | 0 | 0 |
| v24 owner | 621 | 0 | 0 | 0 | 0 | 0 |
| v24 owner bf16_pairs | 621 | 0 | 0 | 0 | 0 | 0 |
| v24 owner chunk | 621 | 0 | 0 | 0 | 0 | 0 |
| v24 rs | 621 | 0 | 0 | 0 | 0 | 0 |
| v24 rs bf16_pairs | 621 | 0 | 0 | 0 | 0 | 0 |
| v24 rs chunk | 621 | 0 | 0 | 0 | 0 | 0 |
| v24 rs l2 | 613 | 0 | 0 | 0 | 0 | 0 |
| v24 spread | 15 | 0 | 0 | 0 | 0 | 0 |
| v24 two_tile | 105 | 0 | 0 | 0 | 0 | 0 |

PASS
