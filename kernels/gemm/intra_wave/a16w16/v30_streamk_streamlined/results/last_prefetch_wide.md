# v30 last-trip prefetch: 9 rounds of 5 x 100 launches, median us per launch, GPU 1

trips: persistent trips per program (ceil); S: Stream-K tiles; n: programs per Stream-K tile.

| point | tile | shape | plan | S | n | trips | clamp us | streamk us | streamk / clamp | spread (max/min - 1) |
|---|---|---|---|---|---|---|---|---|---|---|
| 256x256x64 capped S=16 | 256x256x64 | 4352x4096x4096 | one_tile/rs | 16 | 8 | 1 | 117.22 | 114.13 | 0.974 | 2.2%/1.0% |
| 256x256x64 rs S=16 ragged bias | 256x256x64 | 4320x4064x4080 | one_tile/rs | 16 | 4 | 1 | 146.20 | 142.83 | 0.977 | 0.3%/0.5% |
| 256x256x64 owner S=16 3 waves | 256x256x64 | 12544x4096x4096 | one_tile/owner | 16 | 4 | 3 | 299.07 | 296.92 | 0.993 | 0.6%/0.6% |
| 256x256x64 l2 S=16 K=8192 | 256x256x64 | 4352x4096x8192 | one_tile/rs | 16 | 4 | 1 | 215.76 | 213.87 | 0.991 | 0.7%/0.9% |
| 256x256x64 two_tile | 256x256x64 | 8448x4096x4096 | two_tile/owner | 272 | 1 | 1 | 206.04 | 205.95 | 1.000 | 1.1%/0.7% |
| 256x256x64 two_tile bias | 256x256x64 | 8448x4096x4096 | two_tile/owner | 272 | 1 | 1 | 212.05 | 210.22 | 0.991 | 0.7%/0.6% |
| 256x256x64 two_tile ragged | 256x256x64 | 8416x4064x3984 | two_tile/owner | 272 | 1 | 1 | 254.80 | 255.63 | 1.003 | 0.5%/0.7% |
| 256x256x64 two_tile 2 trips K=8192 | 256x256x64 | 12544x4096x8192 | two_tile/owner | 272 | 1 | 2 | 559.82 | 558.96 | 0.998 | 0.6%/2.1% |
| 256x256x64 spread | 256x256x64 | 4096x7168x4096 | spread/owner | 192 | 1 | 1 | 176.12 | 174.95 | 0.993 | 1.3%/0.5% |
| 256x256x64 spread ragged bias | 256x256x64 | 4064x7136x4080 | spread/owner | 192 | 1 | 1 | 226.27 | 225.50 | 0.997 | 0.4%/0.3% |
| 256x256x64 spread K=16384 | 256x256x64 | 4096x7168x16384 | spread/owner | 192 | 1 | 1 | 626.32 | 625.95 | 0.999 | 1.4%/1.5% |
| 256x128x64 capped S=16 | 256x128x64 | 4352x2048x4096 | one_tile/rs | 16 | 8 | 1 | 71.77 | 70.58 | 0.983 | 1.2%/1.6% |
| 256x128x64 rs S=16 ragged bias | 256x128x64 | 4320x2032x4080 | one_tile/rs | 16 | 4 | 1 | 103.89 | 102.49 | 0.987 | 0.3%/0.5% |
| 256x128x64 owner S=16 3 waves | 256x128x64 | 12544x2048x4096 | one_tile/owner | 16 | 4 | 3 | 185.59 | 184.94 | 0.997 | 1.5%/1.5% |
| 256x128x64 l2 S=16 K=8192 | 256x128x64 | 4352x2048x8192 | one_tile/rs | 16 | 4 | 1 | 135.85 | 135.19 | 0.995 | 0.9%/1.2% |
| 256x128x64 two_tile | 256x128x64 | 8448x2048x4096 | two_tile/owner | 272 | 1 | 1 | 132.28 | 131.32 | 0.993 | 1.9%/1.7% |
| 256x128x64 two_tile bias | 256x128x64 | 8448x2048x4096 | two_tile/owner | 272 | 1 | 1 | 133.64 | 133.12 | 0.996 | 1.5%/1.7% |
| 256x128x64 two_tile ragged | 256x128x64 | 8416x2032x3984 | two_tile/owner | 272 | 1 | 1 | 182.56 | 182.96 | 1.002 | 0.5%/0.4% |
| 256x128x64 two_tile 2 trips K=8192 | 256x128x64 | 12544x2048x8192 | two_tile/owner | 272 | 1 | 2 | 345.21 | 343.49 | 0.995 | 0.7%/1.4% |
| 256x128x64 spread | 256x128x64 | 4096x3584x4096 | spread/owner | 192 | 1 | 1 | 109.30 | 108.62 | 0.994 | 1.8%/2.4% |
| 256x128x64 spread ragged bias | 256x128x64 | 4064x3568x4080 | spread/owner | 192 | 1 | 1 | 145.87 | 145.19 | 0.995 | 0.3%/0.5% |
| 256x128x64 spread K=16384 | 256x128x64 | 4096x3584x16384 | spread/owner | 192 | 1 | 1 | 388.83 | 387.50 | 0.997 | 0.6%/0.6% |
| 128x256x64 capped S=16 | 128x256x64 | 2176x4096x4096 | one_tile/rs | 16 | 8 | 1 | 70.99 | 69.64 | 0.981 | 1.6%/1.4% |
| 128x256x64 rs S=16 ragged bias | 128x256x64 | 2160x4064x4080 | one_tile/rs | 16 | 4 | 1 | 101.54 | 100.15 | 0.986 | 0.5%/0.3% |
| 128x256x64 owner S=16 3 waves | 128x256x64 | 6272x4096x4096 | one_tile/owner | 16 | 4 | 3 | 175.61 | 174.86 | 0.996 | 0.8%/1.3% |
| 128x256x64 l2 S=16 K=8192 | 128x256x64 | 2176x4096x8192 | one_tile/rs | 16 | 4 | 1 | 133.85 | 133.07 | 0.994 | 1.0%/1.0% |
| 128x256x64 two_tile | 128x256x64 | 4224x4096x4096 | two_tile/owner | 272 | 1 | 1 | 130.52 | 128.99 | 0.988 | 1.3%/1.3% |
| 128x256x64 two_tile bias | 128x256x64 | 4224x4096x4096 | two_tile/owner | 272 | 1 | 1 | 133.02 | 131.78 | 0.991 | 0.9%/1.1% |
| 128x256x64 two_tile ragged | 128x256x64 | 4208x4064x3984 | two_tile/owner | 272 | 1 | 1 | 176.53 | 176.40 | 0.999 | 0.2%/0.2% |
| 128x256x64 two_tile 2 trips K=8192 | 128x256x64 | 6272x4096x8192 | two_tile/owner | 272 | 1 | 2 | 326.58 | 326.05 | 0.998 | 0.7%/0.6% |
| 128x256x64 spread | 128x256x64 | 2048x7168x4096 | spread/owner | 192 | 1 | 1 | 109.00 | 108.53 | 0.996 | 1.3%/1.6% |
| 128x256x64 spread ragged bias | 128x256x64 | 2032x7136x4080 | spread/owner | 192 | 1 | 1 | 149.45 | 149.38 | 1.000 | 0.4%/0.4% |
| 128x256x64 spread K=16384 | 128x256x64 | 2048x7168x16384 | spread/owner | 192 | 1 | 1 | 392.23 | 390.42 | 0.995 | 0.5%/0.3% |
| 128x128x128 capped S=16 | 128x128x128 | 2176x2048x4096 | one_tile/rs | 16 | 8 | 1 | 45.75 | 44.87 | 0.981 | 1.8%/1.3% |
| 128x128x128 rs S=16 ragged bias | 128x128x128 | 2160x2032x4080 | one_tile/rs | 16 | 4 | 1 | 60.92 | 59.58 | 0.978 | 0.6%/0.7% |
| 128x128x128 owner S=16 3 waves | 128x128x128 | 6272x2048x4096 | one_tile/owner | 16 | 4 | 3 | 110.58 | 110.25 | 0.997 | 0.6%/1.5% |
| 128x128x128 l2 S=16 K=8192 | 128x128x128 | 2176x2048x8192 | one_tile/rs | 16 | 4 | 1 | 81.38 | 80.87 | 0.994 | 1.2%/0.9% |
| 128x128x128 two_tile | 128x128x128 | 4224x2048x4096 | two_tile/owner | 272 | 1 | 1 | 80.35 | 79.71 | 0.992 | 1.1%/0.9% |
| 128x128x128 two_tile bias | 128x128x128 | 4224x2048x4096 | two_tile/owner | 272 | 1 | 1 | 82.23 | 81.17 | 0.987 | 0.9%/0.9% |
| 128x128x128 two_tile ragged | 128x128x128 | 4208x2032x3984 | two_tile/owner | 272 | 1 | 1 | 97.55 | 96.82 | 0.993 | 0.3%/0.7% |
| 128x128x128 two_tile 2 trips K=8192 | 128x128x128 | 6272x2048x8192 | two_tile/owner | 272 | 1 | 2 | 202.82 | 201.72 | 0.995 | 0.5%/0.4% |
| 128x128x128 spread | 128x128x128 | 2048x3584x4096 | spread/owner | 192 | 1 | 1 | 68.11 | 67.38 | 0.989 | 1.3%/0.6% |
| 128x128x128 spread ragged bias | 128x128x128 | 2032x3568x4080 | spread/owner | 192 | 1 | 1 | 85.74 | 85.61 | 0.998 | 0.5%/0.6% |
| 128x128x128 spread K=16384 | 128x128x128 | 2048x3584x16384 | spread/owner | 192 | 1 | 1 | 242.47 | 241.88 | 0.998 | 0.7%/0.5% |
| 64x64x64 capped S=16 | 64x64x64 | 1088x1024x4096 | one_tile/rs | 16 | 8 | 1 | 30.40 | 30.32 | 0.997 | 0.6%/0.8% |
| 64x64x64 rs S=16 ragged bias | 64x64x64 | 1080x1016x4080 | one_tile/rs | 16 | 4 | 1 | 36.65 | 36.44 | 0.994 | 0.4%/0.5% |
| 64x64x64 owner S=16 3 waves | 64x64x64 | 3136x1024x4096 | one_tile/owner | 16 | 4 | 3 | 70.88 | 70.72 | 0.998 | 0.5%/0.6% |
| 64x64x64 l2 S=16 K=8192 | 64x64x64 | 1088x1024x8192 | one_tile/rs | 16 | 4 | 1 | 54.56 | 54.33 | 0.996 | 0.8%/0.7% |
| 64x64x64 two_tile | 64x64x64 | 2112x1024x4096 | two_tile/owner | 272 | 1 | 1 | 51.78 | 51.56 | 0.996 | 0.7%/0.6% |
| 64x64x64 two_tile bias | 64x64x64 | 2112x1024x4096 | two_tile/owner | 272 | 1 | 1 | 53.46 | 53.14 | 0.994 | 0.6%/0.6% |
| 64x64x64 two_tile ragged | 64x64x64 | 2104x1016x3984 | two_tile/owner | 272 | 1 | 1 | 59.21 | 59.01 | 0.997 | 0.5%/0.3% |
| 64x64x64 two_tile 2 trips K=8192 | 64x64x64 | 3136x1024x8192 | two_tile/owner | 272 | 1 | 2 | 122.04 | 121.10 | 0.992 | 0.5%/0.9% |
| 64x64x64 spread | 64x64x64 | 1024x1792x4096 | spread/owner | 192 | 1 | 1 | 42.73 | 42.71 | 1.000 | 1.3%/1.3% |
| 64x64x64 spread ragged bias | 64x64x64 | 1016x1784x4080 | spread/owner | 192 | 1 | 1 | 48.89 | 48.39 | 0.990 | 0.6%/0.6% |
| 64x64x64 spread K=16384 | 64x64x64 | 1024x1792x16384 | spread/owner | 192 | 1 | 1 | 146.13 | 145.73 | 0.997 | 0.2%/0.4% |

## Registers

| point | mode | VGPRs | AGPRs | SGPRs | VGPR spills | SGPR spills | instructions |
|---|---|---|---|---|---|---|---|
| 256x256x64 capped S=16 | clamp | 472 | 256 | 106 | 0 | 1 | 5479 |
| 256x256x64 capped S=16 | streamk | 488 | 256 | 106 | 0 | 28 | 5083 |
| 256x256x64 rs S=16 ragged bias | clamp | 512 | 256 | 106 | 23 | 75 | 6193 |
| 256x256x64 rs S=16 ragged bias | streamk | 512 | 256 | 106 | 2 | 100 | 6136 |
| 256x256x64 owner S=16 3 waves | clamp | 480 | 256 | 106 | 0 | 2 | 6674 |
| 256x256x64 owner S=16 3 waves | streamk | 492 | 256 | 106 | 0 | 29 | 6240 |
| 256x256x64 l2 S=16 K=8192 | clamp | 472 | 256 | 106 | 0 | 1 | 5453 |
| 256x256x64 l2 S=16 K=8192 | streamk | 488 | 256 | 106 | 0 | 29 | 5068 |
| 256x256x64 two_tile | clamp | 512 | 256 | 106 | 7 | 23 | 7999 |
| 256x256x64 two_tile | streamk | 512 | 256 | 106 | 17 | 46 | 12071 |
| 256x256x64 two_tile bias | clamp | 512 | 256 | 106 | 11 | 25 | 8594 |
| 256x256x64 two_tile bias | streamk | 512 | 256 | 106 | 20 | 46 | 13002 |
| 256x256x64 two_tile ragged | clamp | 512 | 256 | 106 | 36 | 91 | 8139 |
| 256x256x64 two_tile ragged | streamk | 512 | 256 | 106 | 29 | 114 | 12933 |
| 256x256x64 two_tile 2 trips K=8192 | clamp | 512 | 256 | 106 | 7 | 23 | 8000 |
| 256x256x64 two_tile 2 trips K=8192 | streamk | 512 | 256 | 106 | 17 | 46 | 12073 |
| 256x256x64 spread | clamp | 512 | 256 | 106 | 7 | 23 | 7999 |
| 256x256x64 spread | streamk | 512 | 256 | 106 | 17 | 46 | 12071 |
| 256x256x64 spread ragged bias | clamp | 512 | 256 | 106 | 53 | 96 | 8943 |
| 256x256x64 spread ragged bias | streamk | 512 | 256 | 106 | 47 | 114 | 14105 |
| 256x256x64 spread K=16384 | clamp | 512 | 256 | 106 | 7 | 23 | 8000 |
| 256x256x64 spread K=16384 | streamk | 512 | 256 | 106 | 17 | 46 | 12073 |
| 256x128x64 capped S=16 | clamp | 336 | 188 | 102 | 0 | 0 | 3416 |
| 256x128x64 capped S=16 | streamk | 296 | 148 | 106 | 0 | 21 | 3472 |
| 256x128x64 rs S=16 ragged bias | clamp | 344 | 160 | 106 | 0 | 42 | 3872 |
| 256x128x64 rs S=16 ragged bias | streamk | 332 | 148 | 106 | 0 | 68 | 3962 |
| 256x128x64 owner S=16 3 waves | clamp | 404 | 188 | 102 | 0 | 0 | 3993 |
| 256x128x64 owner S=16 3 waves | streamk | 364 | 148 | 106 | 0 | 19 | 4051 |
| 256x128x64 l2 S=16 K=8192 | clamp | 340 | 188 | 106 | 0 | 0 | 3420 |
| 256x128x64 l2 S=16 K=8192 | streamk | 296 | 148 | 106 | 0 | 24 | 3486 |
| 256x128x64 two_tile | clamp | 444 | 188 | 106 | 0 | 15 | 4840 |
| 256x128x64 two_tile | streamk | 404 | 148 | 106 | 0 | 31 | 7640 |
| 256x128x64 two_tile bias | clamp | 444 | 188 | 106 | 0 | 17 | 5052 |
| 256x128x64 two_tile bias | streamk | 404 | 148 | 106 | 0 | 36 | 8013 |
| 256x128x64 two_tile ragged | clamp | 416 | 160 | 106 | 0 | 43 | 4867 |
| 256x128x64 two_tile ragged | streamk | 400 | 144 | 106 | 0 | 67 | 7670 |
| 256x128x64 two_tile 2 trips K=8192 | clamp | 444 | 188 | 106 | 0 | 15 | 4841 |
| 256x128x64 two_tile 2 trips K=8192 | streamk | 404 | 148 | 106 | 0 | 31 | 7642 |
| 256x128x64 spread | clamp | 444 | 188 | 106 | 0 | 15 | 4840 |
| 256x128x64 spread | streamk | 404 | 148 | 106 | 0 | 31 | 7640 |
| 256x128x64 spread ragged bias | clamp | 416 | 160 | 106 | 0 | 49 | 5344 |
| 256x128x64 spread ragged bias | streamk | 428 | 172 | 106 | 0 | 73 | 8354 |
| 256x128x64 spread K=16384 | clamp | 444 | 188 | 106 | 0 | 15 | 4841 |
| 256x128x64 spread K=16384 | streamk | 404 | 148 | 106 | 0 | 31 | 7642 |
| 128x256x64 capped S=16 | clamp | 340 | 192 | 102 | 0 | 0 | 3396 |
| 128x256x64 capped S=16 | streamk | 308 | 148 | 106 | 0 | 20 | 3450 |
| 128x256x64 rs S=16 ragged bias | clamp | 360 | 160 | 106 | 0 | 41 | 3978 |
| 128x256x64 rs S=16 ragged bias | streamk | 348 | 148 | 106 | 0 | 67 | 4076 |
| 128x256x64 owner S=16 3 waves | clamp | 408 | 192 | 102 | 0 | 0 | 3968 |
| 128x256x64 owner S=16 3 waves | streamk | 364 | 148 | 106 | 0 | 20 | 4022 |
| 128x256x64 l2 S=16 K=8192 | clamp | 340 | 192 | 102 | 0 | 0 | 3397 |
| 128x256x64 l2 S=16 K=8192 | streamk | 308 | 148 | 106 | 0 | 22 | 3452 |
| 128x256x64 two_tile | clamp | 448 | 192 | 106 | 0 | 15 | 4787 |
| 128x256x64 two_tile | streamk | 404 | 148 | 106 | 0 | 31 | 7598 |
| 128x256x64 two_tile bias | clamp | 448 | 192 | 106 | 0 | 17 | 5127 |
| 128x256x64 two_tile bias | streamk | 404 | 148 | 106 | 0 | 36 | 8138 |
| 128x256x64 two_tile ragged | clamp | 416 | 160 | 106 | 0 | 44 | 4862 |
| 128x256x64 two_tile ragged | streamk | 400 | 144 | 106 | 0 | 67 | 7640 |
| 128x256x64 two_tile 2 trips K=8192 | clamp | 448 | 192 | 106 | 0 | 15 | 4788 |
| 128x256x64 two_tile 2 trips K=8192 | streamk | 404 | 148 | 106 | 0 | 31 | 7600 |
| 128x256x64 spread | clamp | 448 | 192 | 106 | 0 | 15 | 4787 |
| 128x256x64 spread | streamk | 404 | 148 | 106 | 0 | 31 | 7598 |
| 128x256x64 spread ragged bias | clamp | 417 | 161 | 106 | 0 | 56 | 5526 |
| 128x256x64 spread ragged bias | streamk | 424 | 168 | 106 | 0 | 67 | 8518 |
| 128x256x64 spread K=16384 | clamp | 448 | 192 | 106 | 0 | 15 | 4788 |
| 128x256x64 spread K=16384 | streamk | 404 | 148 | 106 | 0 | 31 | 7600 |
| 128x128x128 capped S=16 | clamp | 260 | 80 | 106 | 0 | 4 | 3250 |
| 128x128x128 capped S=16 | streamk | 256 | 72 | 106 | 0 | 30 | 3268 |
| 128x128x128 rs S=16 ragged bias | clamp | 292 | 80 | 106 | 0 | 30 | 3612 |
| 128x128x128 rs S=16 ragged bias | streamk | 284 | 72 | 106 | 0 | 58 | 3663 |
| 128x128x128 owner S=16 3 waves | clamp | 284 | 80 | 106 | 0 | 2 | 3550 |
| 128x128x128 owner S=16 3 waves | streamk | 276 | 72 | 106 | 0 | 27 | 3570 |
| 128x128x128 l2 S=16 K=8192 | clamp | 260 | 80 | 106 | 0 | 2 | 3200 |
| 128x128x128 l2 S=16 K=8192 | streamk | 256 | 72 | 106 | 0 | 32 | 3235 |
| 128x128x128 two_tile | clamp | 300 | 80 | 106 | 0 | 26 | 4028 |
| 128x128x128 two_tile | streamk | 292 | 72 | 106 | 0 | 47 | 5973 |
| 128x128x128 two_tile bias | clamp | 304 | 80 | 106 | 0 | 27 | 4207 |
| 128x128x128 two_tile bias | streamk | 296 | 72 | 106 | 0 | 48 | 6330 |
| 128x128x128 two_tile ragged | clamp | 304 | 80 | 106 | 0 | 45 | 4220 |
| 128x128x128 two_tile ragged | streamk | 320 | 96 | 106 | 0 | 64 | 6278 |
| 128x128x128 two_tile 2 trips K=8192 | clamp | 300 | 80 | 106 | 0 | 26 | 4028 |
| 128x128x128 two_tile 2 trips K=8192 | streamk | 292 | 72 | 106 | 0 | 47 | 5973 |
| 128x128x128 spread | clamp | 300 | 80 | 106 | 0 | 26 | 4028 |
| 128x128x128 spread | streamk | 292 | 72 | 106 | 0 | 47 | 5973 |
| 128x128x128 spread ragged bias | clamp | 308 | 80 | 106 | 0 | 55 | 4473 |
| 128x128x128 spread ragged bias | streamk | 328 | 96 | 106 | 0 | 78 | 6672 |
| 128x128x128 spread K=16384 | clamp | 300 | 80 | 106 | 0 | 26 | 4028 |
| 128x128x128 spread K=16384 | streamk | 292 | 72 | 106 | 0 | 47 | 5973 |
| 64x64x64 capped S=16 | clamp | 80 | 20 | 87 | 0 | 0 | 1638 |
| 64x64x64 capped S=16 | streamk | 88 | 24 | 106 | 0 | 8 | 1740 |
| 64x64x64 rs S=16 ragged bias | clamp | 96 | 24 | 102 | 0 | 0 | 1651 |
| 64x64x64 rs S=16 ragged bias | streamk | 100 | 24 | 106 | 0 | 24 | 1800 |
| 64x64x64 owner S=16 3 waves | clamp | 88 | 24 | 86 | 0 | 0 | 1734 |
| 64x64x64 owner S=16 3 waves | streamk | 88 | 24 | 106 | 0 | 5 | 1825 |
| 64x64x64 l2 S=16 K=8192 | clamp | 80 | 20 | 87 | 0 | 0 | 1539 |
| 64x64x64 l2 S=16 K=8192 | streamk | 88 | 24 | 106 | 0 | 7 | 1635 |
| 64x64x64 two_tile | clamp | 88 | 20 | 106 | 0 | 1 | 2051 |
| 64x64x64 two_tile | streamk | 88 | 20 | 106 | 0 | 8 | 3240 |
| 64x64x64 two_tile bias | clamp | 92 | 20 | 106 | 0 | 4 | 2150 |
| 64x64x64 two_tile bias | streamk | 92 | 20 | 106 | 0 | 10 | 3357 |
| 64x64x64 two_tile ragged | clamp | 88 | 16 | 106 | 0 | 2 | 2052 |
| 64x64x64 two_tile ragged | streamk | 92 | 16 | 106 | 0 | 19 | 3274 |
| 64x64x64 two_tile 2 trips K=8192 | clamp | 116 | 20 | 106 | 0 | 20 | 2114 |
| 64x64x64 two_tile 2 trips K=8192 | streamk | 88 | 20 | 106 | 0 | 7 | 3219 |
| 64x64x64 spread | clamp | 88 | 20 | 106 | 0 | 1 | 2051 |
| 64x64x64 spread | streamk | 88 | 20 | 106 | 0 | 8 | 3240 |
| 64x64x64 spread ragged bias | clamp | 100 | 24 | 106 | 0 | 8 | 2196 |
| 64x64x64 spread ragged bias | streamk | 100 | 24 | 106 | 0 | 26 | 3474 |
| 64x64x64 spread K=16384 | clamp | 116 | 20 | 106 | 0 | 20 | 2114 |
| 64x64x64 spread K=16384 | streamk | 88 | 20 | 106 | 0 | 7 | 3219 |
