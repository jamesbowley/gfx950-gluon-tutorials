# v28 at origami's tile and split: 2,692 shapes

From `details_origami_rot2048.csv`, `details_origami_redo.csv`; 38 timed shapes have no origami pick (no tile with two pairs of k-steps) and are left out of every table.

## Against AITER's pick

Ours: the fastest origami slot (fp32 partials).

| group | rows | ahead > 1% | within 1% | behind > 1% | behind > 4% | behind > 10% | median |
|---|---:|---:|---:|---:|---:|---:|---:|
| M < 512 | 1,557 | 406 | 81 | 1,070 | 948 | 716 | -8.3% |
| N < 512 <= M | 218 | 75 | 22 | 121 | 93 | 62 | -2.0% |
| both >= 512 | 917 | 697 | 44 | 176 | 130 | 57 | +5.3% |
| **all** | 2,692 | 1,178 | 147 | 1,367 | 1,171 | 835 | -1.2% |

| min(M, N) | rows | ahead > 1% | within 1% | behind > 1% | behind > 4% | behind > 10% | median |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1-15 | 367 | 104 | 17 | 246 | 231 | 206 | -13.4% |
| 16-63 | 341 | 25 | 17 | 299 | 277 | 212 | -13.2% |
| 64-255 | 716 | 169 | 34 | 513 | 443 | 319 | -8.2% |
| 256-511 | 351 | 183 | 35 | 133 | 90 | 41 | +1.5% |
| >= 512 | 917 | 697 | 44 | 176 | 130 | 57 | +5.3% |
| **all** | 2,692 | 1,178 | 147 | 1,367 | 1,171 | 835 | -1.2% |

## Against hipBLASLt (torch.mm)

| group | rows | ahead > 1% | within 1% | behind > 1% | behind > 4% | behind > 10% | median |
|---|---:|---:|---:|---:|---:|---:|---:|
| M < 512 | 1,557 | 1,052 | 46 | 459 | 360 | 194 | +8.2% |
| N < 512 <= M | 218 | 158 | 6 | 54 | 41 | 22 | +6.4% |
| both >= 512 | 917 | 728 | 41 | 148 | 111 | 52 | +7.1% |
| **all** | 2,692 | 1,938 | 93 | 661 | 512 | 268 | +7.5% |

| min(M, N) | rows | ahead > 1% | within 1% | behind > 1% | behind > 4% | behind > 10% | median |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1-15 | 367 | 250 | 6 | 111 | 95 | 61 | +10.8% |
| 16-63 | 341 | 204 | 10 | 127 | 100 | 49 | +6.3% |
| 64-255 | 716 | 481 | 29 | 206 | 156 | 80 | +6.8% |
| 256-511 | 351 | 275 | 7 | 69 | 50 | 26 | +9.7% |
| >= 512 | 917 | 728 | 41 | 148 | 111 | 52 | +7.1% |
| **all** | 2,692 | 1,938 | 93 | 661 | 512 | 268 | +7.5% |

## Against the best library (AITER's pick or hipBLASLt)

| group | rows | ahead > 1% | within 1% | behind > 1% | behind > 4% | behind > 10% | median |
|---|---:|---:|---:|---:|---:|---:|---:|
| M < 512 | 1,557 | 336 | 74 | 1,147 | 1,023 | 761 | -9.6% |
| N < 512 <= M | 218 | 63 | 22 | 133 | 102 | 69 | -3.2% |
| both >= 512 | 917 | 671 | 52 | 194 | 141 | 62 | +4.6% |
| **all** | 2,692 | 1,070 | 148 | 1,474 | 1,266 | 892 | -2.8% |

## Against AITER's pick, by AITER's library

| AITER pick | rows | ahead > 1% | within 1% | behind > 1% | behind > 4% | behind > 10% | median |
|---|---:|---:|---:|---:|---:|---:|---:|
| flydsl | 1,765 | 696 | 93 | 976 | 842 | 618 | -3.2% |
| torch | 562 | 275 | 30 | 257 | 222 | 131 | +0.4% |
| asm | 213 | 158 | 12 | 43 | 29 | 18 | +4.4% |
| triton | 109 | 33 | 9 | 67 | 60 | 53 | -9.7% |
| opus | 27 | 15 | 3 | 9 | 3 | 0 | +1.6% |
| skinny | 16 | 1 | 0 | 15 | 15 | 15 | -41.3% |
| **all** | 2,692 | 1,178 | 147 | 1,367 | 1,171 | 835 | -1.2% |

## Where AITER's pick is hipBLASLt (562 shapes)

AITER's pick there is hipBLASLt itself (torch libtype, or hipblaslt with a tuned solution index).

| group | rows | ahead > 1% | within 1% | behind > 1% | behind > 4% | behind > 10% | median |
|---|---:|---:|---:|---:|---:|---:|---:|
| M < 512 | 175 | 38 | 3 | 134 | 119 | 79 | -8.5% |
| N < 512 <= M | 39 | 16 | 2 | 21 | 17 | 10 | -2.1% |
| both >= 512 | 348 | 221 | 25 | 102 | 86 | 42 | +3.6% |
| **all** | 562 | 275 | 30 | 257 | 222 | 131 | +0.4% |

## AITER picks that reduce K in bf16 (1122 shapes)

fp32: ours as above; bf16: with rs / ow on bf16 (bf16-pair) partials, where their mean_ratio is within 1 + sqrt(n). mean_ratio is the largest 32-row region's mean error over one bf16 rounding's (1.0 for an fp32 reduction).

fp32 partials:

| class | rows | ahead > 1% | within 1% | behind > 1% | behind > 4% | behind > 10% | median |
|---|---:|---:|---:|---:|---:|---:|---:|
| asm_splitk | 64 | 25 | 5 | 34 | 25 | 17 | -1.4% |
| flydsl_kwaves | 198 | 85 | 11 | 102 | 79 | 52 | -1.4% |
| flydsl_splitk | 860 | 188 | 48 | 624 | 551 | 412 | -9.0% |
| **all** | 1,122 | 298 | 64 | 760 | 655 | 481 | -7.2% |

with bf16 partials:

| class | rows | ahead > 1% | within 1% | behind > 1% | behind > 4% | behind > 10% | median |
|---|---:|---:|---:|---:|---:|---:|---:|
| asm_splitk | 64 | 39 | 3 | 22 | 13 | 8 | +3.9% |
| flydsl_kwaves | 198 | 107 | 14 | 77 | 64 | 44 | +2.3% |
| flydsl_splitk | 860 | 284 | 43 | 533 | 464 | 351 | -6.2% |
| **all** | 1,122 | 430 | 60 | 632 | 541 | 403 | -3.4% |

| class | AITER pick mean_ratio (median, max) | ours bf16 slots mean_ratio (median, max) | bf16 slot fastest |
|---|---:|---:|---:|
| asm_splitk | 2.29, 3.72 | 1.67, 1.82 | 49 of 64 |
| flydsl_kwaves | 1.67, 1.90 | 1.64, 2.10 | 88 of 198 |
| flydsl_splitk | 2.31, 3.19 | 1.67, 1.99 | 545 of 860 |

## Which ending wins at origami's tile

| slot | shapes timed | fastest on | median loss to the best slot | loss > 2% on |
|---|---:|---:|---:|---:|
| v9 | 2,692 | 940 | +4.5% | 1,458 |
| pdp | 2,692 | 320 | +10.6% | 1,985 |
| rs | 1,452 | 1,136 | +0.0% | 177 |
| ow | 1,452 | 215 | +6.1% | 1,083 |
| 2t | 410 | 43 | +4.7% | 271 |
| sp | 402 | 38 | +2.5% | 226 |

Origami's reduction hint as the ending (parallel: reduce-scatter, tree: owner), on the 1,452 shapes with both: the faster of the two on 1,059; following it costs a median +0.0% against the faster, more than 2% on 296.

## Selectors against AITER's pick

| selector | rows | ahead > 1% | behind > 4% | behind > 10% | median |
|---|---:|---:|---:|---:|---:|
| origami (fp32) | 2,692 | 1,178 | 1,171 | 835 | -1.2% |
| origami (+ bf16 on bf16-reducing picks) | 2,692 | 1,310 | 1,057 | 757 | +0.4% |
| cap at origami's tile | 2,692 | 1,012 | 1,354 | 996 | -4.2% |
| REFIT rule's single pick | 2,651 | 1,090 | 1,200 | 758 | -2.3% |
| v28 auto + capped | 2,691 | 716 | 1,812 | 1,654 | -22.2% |
| best of all of these | 2,692 | 1,507 | 786 | 419 | +2.5% |

## Against 2026-10-20 (v27, REFIT short list of up to 4 picks, or auto)

On the 2,177 shapes in both runs, each against its own run's AITER pick (AITER's pick moved a median +0.0% between the runs):

| | median | behind > 4% | behind > 10% |
|---|---:|---:|---:|
| 2026-10-20 best of short list | -2.1% | 937 | 473 |
| origami (fp32) | -4.8% | 1,123 | 819 |

## Origami's choices

| tile | shapes | median gap against AITER's pick |
|---|---:|---:|
| 256x256x64 | 664 | +5.5% |
| 1x16x256 | 242 | -20.3% |
| 16x16x256 | 203 | -20.5% |
| 128x256x64 | 154 | -3.7% |
| 64x64x128 | 141 | -0.4% |
| 256x128x64 | 133 | +2.4% |
| 128x128x64 | 133 | +4.3% |
| 64x128x64 | 128 | -1.0% |
| 64x64x64 | 97 | -8.3% |
| 64x256x64 | 88 | -11.0% |
| 128x128x128 | 83 | +6.7% |
| 16x256x128 | 71 | -11.1% |
| 64x128x128 | 60 | -1.7% |
| 16x64x128 | 60 | -16.8% |
| 32x128x128 | 53 | -13.8% |

## The 40 worst shapes against the best library

| M x N x K | group | models | AITER pick | ours (slot, tile) | gap vs pick | vs hipBLASLt | best of all vs lib |
|---|---|---|---|---|---:|---:|---:|
| 8x384x7168 | M < 512 | kimik2 | flydsl | dp 1x16x256 | -54.6% | -20.2% | -12.9% |
| 4x16160x7168 | M < 512 | dsv3 | torch | pdp 1x16x256 | -52.9% | -53.3% | -5.5% |
| 1x128x4096 | M < 512 | q3vl | skinny | rs2 1x16x256 | -52.3% | +10.8% | -31.1% |
| 1x256x4096 | M < 512 | qwen3_5_397b | skinny | rs2 1x16x256 | -52.3% | +18.5% | -34.0% |
| 4x6016x7168 | M < 512 | kimik3 | flydsl | pdp 1x16x256 | -51.8% | -38.8% | -2.4% |
| 8x512x7168 | M < 512 | dsv4 | flydsl | dp 1x16x256 | -51.3% | -17.9% | -9.5% |
| 4x20480x7168 | M < 512 | kimik3 | flydsl | pdp 1x16x256 | -51.1% | -47.4% | -7.9% |
| 4x151936x4096 | M < 512 | qwen3_8b | triton | 2t 1x16x256 | -38.6% | -51.1% | -3.9% |
| 24x160x6144 | M < 512 | glm5 | flydsl | dp 1x16x256 | -51.0% | +1.0% | -10.7% |
| 4x6144x4608 | M < 512 | minimax_m3_eagle | flydsl | pdp 1x16x256 | -50.6% | -29.4% | -2.6% |
| 4x24576x4096 | M < 512 | qwen3_8b | triton | pdp 1x16x256 | -38.0% | -50.4% | -7.7% |
| 4x6144x7168 | M < 512 | kimik3 | flydsl | pdp 1x16x256 | -50.3% | -38.5% | +0.0% |
| 32x64x7168 | M < 512 | dsv4 | flydsl | rs2 16x1x256 | -50.0% | +19.6% | -14.8% |
| 4x6144x3072 | M < 512 | glm5 | flydsl | pdp 1x16x256 | -49.1% | -40.1% | +0.0% |
| 48x129280x4096 | M < 512 | dsv4 | torch | sp 16x256x128 | -48.8% | -49.0% | -1.8% |
| 64x32x6144 | M < 512 | glm5 | flydsl | rs2 16x1x256 | -48.8% | +22.6% | -18.9% |
| 1x200x5120 | M < 512 | qwen32B | skinny | rs2 1x16x256 | -48.6% | +9.5% | -25.5% |
| 4x3584x7168 | M < 512 | kimik3 | flydsl | dp 1x16x256 | -48.6% | -27.3% | -21.8% |
| 4x7168x1792 | M < 512 | kimik3 | flydsl | pdp 1x16x256 | -48.4% | -34.4% | +1.5% |
| 4x896x7168 | M < 512 | kimik3 | flydsl | dp 1x16x256 | -48.3% | +20.0% | -21.5% |
| 16x128x6144 | M < 512 | glm5 | flydsl | rs2 16x1x256 | -48.2% | +21.2% | -18.5% |
| 4x75968x4096 | M < 512 | q3vl | torch | pdp 1x16x256 | -47.7% | -48.1% | -1.9% |
| 4x9216x6144 | M < 512 | minimax_m3_eagle | flydsl | rsall 1x16x256 | -47.9% | -42.2% | -7.0% |
| 2x256x4096 | M < 512 | qwen3_5_397b | skinny | rs2 1x16x256 | -47.8% | +13.0% | -28.0% |
| 48x151936x4096 | M < 512 | q3vl | torch | pdp 16x256x128 | -47.1% | -47.7% | -4.2% |
| 4x7168x3584 | M < 512 | kimik3 | flydsl | pdp 1x16x256 | -47.5% | -0.5% | +3.9% |
| 4x4480x7168 | M < 512 | kimik3 | flydsl | rsall 1x16x256 | -47.2% | -35.4% | -11.7% |
| 48x75968x4096 | M < 512 | q3vl | torch | sp 16x256x128 | -47.0% | -47.2% | -11.5% |
| 4x6144x6144 | M < 512 | glm5 | flydsl | pdp 1x16x256 | -47.0% | -34.8% | +0.0% |
| 4x19360x6144 | M < 512 | glm5 | flydsl | pdp 1x16x256 | -46.8% | -36.4% | -15.1% |
| 8x256x7168 | M < 512 | dsv3 | flydsl | rs2 1x16x256 | -46.8% | -9.6% | -10.7% |
| 48x32x6144 | M < 512 | glm5 | flydsl | rs2 16x1x256 | -46.7% | +36.0% | -24.5% |
| 4x8192x4096 | M < 512 | qwen3_5_397b | flydsl | dp 1x16x256 | -46.6% | -36.8% | +4.7% |
| 2x128x4096 | M < 512 | q3vl | skinny | rs2 1x16x256 | -46.2% | +15.4% | -25.5% |
| 4x1024x7168 | M < 512 | dsv4 | flydsl | dp 1x16x256 | -45.8% | +20.8% | -13.3% |
| 96x151936x4096 | M < 512 | q3vl | torch | dp 32x256x128 | -45.6% | -45.8% | +0.3% |
| 1x100x5120 | M < 512 | qwen32B | flydsl | rs2 1x16x256 | -45.7% | +10.0% | -22.4% |
| 8x256x6144 | M < 512 | glm5 | flydsl | rs2 1x16x256 | -45.3% | -2.3% | -13.0% |
| 16x64x7168 | M < 512 | dsv4 | flydsl | rs3 16x1x256 | -45.2% | +53.4% | -23.1% |
| 64x64x4096 | M < 512 | dsv4 | flydsl | dp 16x1x256 | -45.2% | +5.5% | -21.6% |
