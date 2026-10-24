# v27 against AITER on all gfx950 tuned bf16 shapes (details_alltuned_rot2048.csv)

2328 rows. Ours: the fastest correct of v27 auto and the REFIT short list (both sides >= 512: v27 auto alone). gap = ours / reference - 1.

- AITER dispatched an untuned config on 0 rows. Its pick failed the harness's check (allclose atol 0.1, rtol 1%) on 1094 rows, 1030 of them FlyDSL; those timings still count, as in earlier runs.

**Against AITER's pick, by group**

| group | shapes | unsupported | ahead > 1% | within 1% | behind > 1% | behind > 4% | behind > 10% | median gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M < 512 | 1682 | 0 | 678 | 102 | 902 | 707 | 349 | -1.9% |
| N < 512 <= M | 225 | 0 | 84 | 21 | 120 | 92 | 42 | -2.1% |
| both >= 512 | 421 | 9 | 188 | 31 | 193 | 167 | 95 | +0.0% |
| **all** | 2328 | 9 | 950 | 154 | 1215 | 966 | 486 | -1.6% |

**Against AITER's pick, by min(M, N)**

| min(M, N) | shapes | unsupported | ahead > 1% | within 1% | behind > 1% | behind > 4% | behind > 10% | median gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1-15 | 403 | 0 | 220 | 11 | 172 | 126 | 73 | +2.4% |
| 16-63 | 360 | 0 | 86 | 25 | 249 | 203 | 102 | -5.7% |
| 64-255 | 771 | 0 | 259 | 56 | 456 | 372 | 186 | -3.6% |
| 256-511 | 373 | 0 | 197 | 31 | 145 | 98 | 30 | +1.3% |
| >=512 | 421 | 9 | 188 | 31 | 193 | 167 | 95 | +0.0% |
| **all** | 2328 | 9 | 950 | 154 | 1215 | 966 | 486 | -1.6% |

**Against the best library (AITER's pick or hipBLASLt), by group**

| group | shapes | unsupported | ahead > 1% | within 1% | behind > 1% | behind > 4% | behind > 10% | median gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M < 512 | 1682 | 0 | 589 | 103 | 990 | 786 | 386 | -3.3% |
| N < 512 <= M | 225 | 0 | 72 | 23 | 130 | 104 | 48 | -2.7% |
| both >= 512 | 421 | 9 | 186 | 26 | 200 | 170 | 101 | -0.4% |
| **all** | 2328 | 9 | 847 | 152 | 1320 | 1060 | 535 | -3.0% |

**Against the best library (AITER's pick or hipBLASLt), by min(M, N)**

| min(M, N) | shapes | unsupported | ahead > 1% | within 1% | behind > 1% | behind > 4% | behind > 10% | median gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1-15 | 403 | 0 | 209 | 13 | 181 | 137 | 76 | +1.4% |
| 16-63 | 360 | 0 | 71 | 22 | 267 | 219 | 109 | -7.0% |
| 64-255 | 771 | 0 | 204 | 54 | 513 | 420 | 210 | -5.2% |
| 256-511 | 373 | 0 | 177 | 37 | 159 | 114 | 39 | +0.0% |
| >=512 | 421 | 9 | 186 | 26 | 200 | 170 | 101 | -0.4% |
| **all** | 2328 | 9 | 847 | 152 | 1320 | 1060 | 535 | -3.0% |

**Against AITER's pick, by AITER's library**

| AITER pick | shapes | unsupported | ahead > 1% | within 1% | behind > 1% | behind > 4% | behind > 10% | median gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| asm | 218 | 0 | 154 | 23 | 41 | 24 | 8 | +6.0% |
| flydsl | 1565 | 9 | 574 | 113 | 869 | 688 | 336 | -2.5% |
| hipblaslt | 7 | 0 | 2 | 1 | 4 | 4 | 1 | -5.5% |
| opus | 28 | 0 | 27 | 0 | 1 | 0 | 0 | +14.0% |
| skinny | 16 | 0 | 1 | 0 | 15 | 14 | 12 | -22.0% |
| torch | 380 | 0 | 130 | 12 | 238 | 199 | 100 | -5.0% |
| triton | 114 | 0 | 62 | 5 | 47 | 37 | 29 | +2.6% |
| **all** | 2328 | 9 | 950 | 154 | 1215 | 966 | 486 | -1.6% |

**Which of our columns was fastest:** best 882, auto 548, tile2 519, tile3 325, x5 45

**The short list over the rule's single pick (pk_best):** median +1.1%, 90th percentile +16.2%, 845 of 1906 shapes more than 2% faster than the single pick.

**New AITER v9 against AITER's pick:** median -37.6% (2277 rows correct).

**Worst 40 against the best library** (of 1320 more than 1% behind; full list in `losers_all_tuned.csv`):

| Shape (MxNxK) | out | group | best library | library TFLOPS | ours TFLOPS | ours | gap vs best | gap vs AITER |
|---|---|---|---|---:|---:|---|---:|---:|
| 576x2048x7168 | bfloat16 | both >= 512 | hipBLASLt | 585.2 | 380.0 | auto 64x128x128 data-parallel | -35.1% | -34.6% |
| 576x1024x7168 | bfloat16 | both >= 512 | AITER torch | 404.6 | 266.7 | auto 64x64x128 data-parallel | -34.1% | -34.1% |
| 1x256x4096 | bfloat16 | M < 512 | AITER skinny | 0.7 | 0.4 | best: rsall 16x16x128 | -34.0% | -34.0% |
| 1x128x4096 | bfloat16 | M < 512 | AITER skinny | 0.3 | 0.2 | best: rsall 16x16x128 | -32.6% | -32.6% |
| 1344x2048x7168 | bfloat16 | both >= 512 | hipBLASLt | 726.7 | 493.9 | auto 64x256x64 data-parallel | -32.0% | -31.9% |
| 1x64x4096 | bfloat16 | M < 512 | AITER skinny | 0.2 | 0.1 | best: rsall 16x16x128 | -31.8% | -31.8% |
| 2x256x4096 | bfloat16 | M < 512 | AITER skinny | 1.2 | 0.8 | best: rsall 16x16x128 | -28.0% | -28.0% |
| 2x64x4096 | bfloat16 | M < 512 | AITER skinny | 0.3 | 0.2 | best: rsall 16x16x128 | -27.1% | -27.1% |
| 1088x2048x4096 | bfloat16 | both >= 512 | AITER flydsl | 570.4 | 416.7 | auto 64x256x64 data-parallel | -26.9% | -26.9% |
| 1x200x5120 | bfloat16 | M < 512 | AITER skinny | 0.6 | 0.4 | tile3: rsall 1x16x256 | -26.0% | -26.0% |
| 128x201088x2880 | bfloat16 | M < 512 | hipBLASLt | 763.8 | 566.3 | best: cap 128x128x128 | -25.9% | -25.6% |
| 1216x1024x7168 | bfloat16 | both >= 512 | hipBLASLt | 531.3 | 397.6 | auto 64x128x128 data-parallel | -25.2% | -24.5% |
| 8x128x6144 | bfloat16 | M < 512 | AITER flydsl | 3.1 | 2.4 | best: rsall 16x16x256 | -24.5% | -24.5% |
| 2x128x4096 | bfloat16 | M < 512 | AITER skinny | 0.6 | 0.5 | best: rsall 16x16x128 | -23.9% | -23.9% |
| 1088x1024x7168 | bfloat16 | both >= 512 | hipBLASLt | 471.1 | 358.9 | auto 64x128x128 data-parallel | -23.8% | -23.6% |
| 96x64x4096 | bfloat16 | M < 512 | AITER flydsl | 12.0 | 9.2 | best: rsall 16x16x128 | -23.6% | -23.6% |
| 192x75968x4096 | bfloat16 | M < 512 | hipBLASLt | 919.8 | 702.9 | auto 256x256x64 one-tile owner, n = 3 | -23.6% | -21.3% |
| 1152x1024x7168 | bfloat16 | both >= 512 | hipBLASLt | 493.0 | 377.5 | auto 64x128x128 data-parallel | -23.4% | -23.0% |
| 1088x512x7168 | bfloat16 | both >= 512 | AITER flydsl | 326.0 | 249.6 | auto 64x64x128 data-parallel | -23.4% | -23.4% |
| 8x64x4096 | bfloat16 | M < 512 | AITER flydsl | 1.2 | 0.9 | best: rsall 16x16x256 | -23.4% | -23.4% |
| 32x64x4096 | bfloat16 | M < 512 | AITER flydsl | 4.7 | 3.6 | best: rsall 16x16x256 | -23.4% | -23.4% |
| 640x2048x7168 | bfloat16 | both >= 512 | hipBLASLt | 547.8 | 420.4 | auto 64x128x128 data-parallel | -23.3% | -21.9% |
| 2x32x6144 | bfloat16 | M < 512 | AITER flydsl | 0.2 | 0.2 | tile2: rsall 16x16x256 | -22.9% | -22.9% |
| 16x64x7168 | bfloat16 | M < 512 | AITER flydsl | 3.6 | 2.8 | best: rsall 16x16x256 | -22.6% | -22.6% |
| 1216x2048x4096 | bfloat16 | both >= 512 | AITER flydsl | 600.0 | 464.7 | auto 64x256x64 data-parallel | -22.6% | -22.6% |
| 544x1024x7168 | bfloat16 | both >= 512 | AITER flydsl | 331.4 | 256.8 | auto 64x64x128 data-parallel | -22.5% | -22.5% |
| 1x100x5120 | bfloat16 | M < 512 | AITER flydsl | 0.3 | 0.2 | tile3: rsall 1x16x256 | -22.4% | -22.4% |
| 1x64x7168 | bfloat16 | M < 512 | AITER flydsl | 0.2 | 0.2 | best: rsall 16x16x128 | -22.0% | -22.0% |
| 1x160x6144 | bfloat16 | M < 512 | AITER skinny | 0.5 | 0.4 | tile2: rsall 16x16x256 | -22.0% | -22.0% |
| 1x256x6144 | bfloat16 | M < 512 | AITER skinny | 0.8 | 0.6 | tile3: rsall 1x16x256 | -22.0% | -22.0% |
| 48x64x4096 | bfloat16 | M < 512 | AITER flydsl | 6.5 | 5.0 | best: rsall 16x16x256 | -22.0% | -22.0% |
| 64x13312x16384 | bfloat16 | M < 512 | hipBLASLt | 345.1 | 269.7 | tile3: cap 64x128x128 | -21.8% | +12.9% |
| 16x128x6144 | bfloat16 | M < 512 | AITER flydsl | 5.9 | 4.6 | best: rsall 16x16x256 | -21.8% | -21.8% |
| 4x896x7168 | bfloat16 | M < 512 | AITER flydsl | 8.4 | 6.6 | tile2: rsall 16x16x256 | -21.8% | -21.8% |
| 160x64x4096 | bfloat16 | M < 512 | AITER flydsl | 17.8 | 14.0 | best: rsall 16x16x128 | -21.7% | -21.7% |
| 48x32x6144 | bfloat16 | M < 512 | AITER flydsl | 4.7 | 3.7 | best: rsall 16x16x256 | -21.6% | -21.6% |
| 112x64x4096 | bfloat16 | M < 512 | AITER flydsl | 13.3 | 10.5 | tile2: rsall 16x16x256 | -21.4% | -21.4% |
| 128x57344x8192 | bfloat16 | M < 512 | hipBLASLt | 788.6 | 620.5 | tile2: cap 128x256x64 | -21.3% | -18.6% |
| 1x32x6144 | bfloat16 | M < 512 | AITER skinny | 0.1 | 0.1 | best: rsall 16x16x128 | -21.3% | -21.3% |
| 4x64x4096 | bfloat16 | M < 512 | AITER flydsl | 0.6 | 0.4 | tile2: rsall 16x16x128 | -21.3% | -21.3% |

**Unsupported** (no correct v27 column): 9: 512x1024x128, 512x8192x256, 1030x1024x128, 1338x1024x128, 2310x1024x128, 2484x1024x128, 7066x1024x128, 7680x1024x128, 7681x1024x128
