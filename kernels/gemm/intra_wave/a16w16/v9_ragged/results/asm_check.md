# v9_ragged asm check

| shape | bias | v9 VGPRs / spills | v9_ragged VGPRs / spills | instruction lines changed | result |
|---|---|---|---|---|---|
| 4096x4096x8192 | False | 448 / 0 | 448 / 0 | 0 | identical |
| 4096x4096x8192 | True | 448 / 0 | 448 / 0 | 0 | identical |
| 2048x4096x4096 | False | 448 / 0 | 448 / 0 | 0 | identical |
| 256x8192x8192 | True | 448 / 0 | 448 / 0 | 0 | identical |
| 4000x4000x8192 (ragged) | False | - | 460 / 0 | - | edge build |
| 4000x4000x8192 (ragged) | True | - | 460 / 0 | - | edge build |
| 128x8192x8192 (ragged) | False | - | 460 / 0 | - | edge build |
| 4096x4096x8176 (ragged) | False | - | 448 / 0 | - | edge build |
| 4096x4096x8080 (ragged) | False | - | 448 / 0 | - | edge build |
| 4000x4000x8080 (ragged) | True | - | 468 / 0 | - | edge build |
| 4096x8192x8192, GEMM_TILE_LIMIT = 40 | False | v9_ragged unlimited: 448 / 0 | 448 / 0 | 8 | hook build |
| 4096x8192x8192, GEMM_TILE_LIMIT = 400 | False | v9_ragged unlimited: 448 / 0 | 448 / 0 | 8 | hook build |

PASS
