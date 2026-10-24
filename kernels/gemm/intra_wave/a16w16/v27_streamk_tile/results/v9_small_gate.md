v9_tile_small, 4096x4096x8192 bf16, `before` against `after` (instruction lines, comments and directives dropped).

| tile | VGPRs / AGPRs / spills, before | after | instructions | lines changed | result |
|---|---|---|---|---|---|
| 16x256x128 | 220 / 20 / 0 | 220 / 20 / 0 | 715 | 0 | identical |
| 16x256x64w2 | 224 / 40 / 0 | 224 / 40 / 0 | 830 | 0 | identical |
| 16x128x128 | 220 / 20 / 0 | 220 / 20 / 0 | 767 | 0 | identical |
| 16x64x128 | 220 / 24 / 0 | 220 / 24 / 0 | 718 | 0 | identical |
| 16x64x64 | 124 / 20 / 0 | 124 / 20 / 0 | 511 | 0 | identical |
| 16x64x256 | 396 / 140 / 0 | 396 / 140 / 0 | 1138 | 0 | identical |
| 16x32x128 | 136 / 12 / 0 | 136 / 12 / 0 | 491 | 0 | identical |
| 16x16x128 | 120 / 8 / 0 | 120 / 8 / 0 | 464 | 0 | identical |
| 16x16x256 | 160 / 4 / 0 | 160 / 4 / 0 | 521 | 0 | identical |
| 32x32x128 | 180 / 24 / 0 | 180 / 24 / 0 | 635 | 0 | identical |
| 32x64x128 | 276 / 48 / 0 | 276 / 48 / 0 | 939 | 0 | identical |
| 32x128x128 | 280 / 44 / 0 | 280 / 44 / 0 | 975 | 0 | identical |
| 32x256x64 | 164 / 44 / 0 | 164 / 44 / 0 | 689 | 0 | identical |
| 8x256x128 | 220 / 20 / 0 | 220 / 20 / 0 | 722 | 0 | identical |
| 1x256x128 | 220 / 20 / 0 | 220 / 20 / 0 | 713 | 0 | identical |
| 1x64x256 | 392 / 136 / 0 | 392 / 136 / 0 | 1111 | 0 | identical |
| 256x16x128 | 220 / 20 / 0 | 220 / 20 / 0 | 752 | 0 | identical |
| 16x16x128p | 120 / 8 / 0 | 120 / 8 / 0 | 471 | 0 | identical |
| 16x64x64p | 124 / 20 / 0 | 124 / 20 / 0 | 515 | 0 | identical |
| 16x256x128p | 220 / 20 / 0 | 220 / 20 / 0 | 715 | 0 | identical |

PASS
