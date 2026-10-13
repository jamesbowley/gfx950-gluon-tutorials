# streamk_wip (v20, capped, v9 for one-wave dp) vs AITER, v9 and hipBLASLt, contiguous and 512-byte-padded strides (2048 MB rotating buffer)

- Stack: torch 2.10.0+rocm7.2.4.git3d3aa833, hipBLASLt 100202 (dabb6df2b9), Triton 3.8.0 @ /home/jbowley/repos/triton_gfx950-tutorial-v2.2/python/triton; GPU: PCI bus 117, gfx950:sramecc+:xnack-; AITER e2c7aa0aa [Triton/Gluon] gfx950 a16w16: skip the Gluon kernel on Triton without cd_regclass on a1455f1c4 2026-10-02 [CK] [FlyDSL] add mxfp8 a8w8 blockscale for gfx950 (#4951).
- streamk_wip: snapshot sha256 56e1e2af1bb77dfa (newest WORKLOG entry: "11. Padded row strides accepted; fp32 C stays off v9 (2026-10-06)"); the v9 it delegates to: live sha256 505b5e31f5d2099e.
- Layouts: contiguous, and padded: every row stride (x, w, C) that is a multiple of 512 bytes gets 256 more (x and w: K + 128 when K % 256 == 0; C: N + 128 for bf16, N + 64 for fp32). v9, streamk_wip and hipBLASLt (`torch.mm(x, w.t(), out=y)`) get padded x, w and C; AITER's pick allocates its own C, so it gets padded x and w only.
- Timing: 3 rounds, each with fresh inputs (new buffers) when `realloc` is on (here: True); median of the rounds. Spread = (slowest - fastest round) / median.
- Best library: the fastest valid time of AITER's pick and hipBLASLt; for the padded comparison over both layouts. A padded call that launches a copy kernel (FlyDSL makes its inputs contiguous) is not valid.
- "Faster" and "slower" mean a gap of more than 1%.

Padded calls that copied their inputs (left out of the best library): pick (flydsl) on 252 shapes.

The best library's time comes from the contiguous layout on 402 shapes, padded layout on 228 shapes.

## v9 and streamk_wip padded, libraries contiguous (as AITER calls them)

| Shapes | Count | v9: faster | v9: within 1% | v9: slower | v9: median | data-parallel only: faster | data-parallel only: within 1% | data-parallel only: slower | data-parallel only: median | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All shapes | 630 | 239 | 48 | 343 | -4.4% | 250 | 37 | 343 | -4.5% | 323 | 37 | 270 | +1.5% |
| Under 256 tiles (less than one wave) | 321 | 22 | 10 | 289 | -39.8% | 18 | 4 | 299 | -40.2% | 56 | 12 | 253 | -17.7% |
| 256 to 1023 tiles | 152 | 104 | 9 | 39 | +3.0% | 100 | 15 | 37 | +3.1% | 130 | 9 | 13 | +4.7% |
| 1024 or more tiles | 157 | 113 | 29 | 15 | +2.7% | 132 | 18 | 7 | +4.0% | 137 | 16 | 4 | +4.1% |
| Best library uses 256x256 | 259 | 196 | 34 | 29 | +3.0% | 209 | 27 | 23 | +3.7% | 228 | 20 | 11 | +4.1% |
| Best library uses another tile | 371 | 43 | 14 | 314 | -31.2% | 41 | 10 | 320 | -31.5% | 95 | 17 | 259 | -11.2% |

## Everything contiguous (comparable with Oct 5)

| Shapes | Count | v9: faster | v9: within 1% | v9: slower | v9: median | data-parallel only: faster | data-parallel only: within 1% | data-parallel only: slower | data-parallel only: median | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All shapes | 630 | 231 | 44 | 355 | -4.7% | 244 | 36 | 350 | -5.4% | 303 | 41 | 286 | +0.6% |
| Under 256 tiles (less than one wave) | 321 | 21 | 6 | 294 | -37.5% | 19 | 2 | 300 | -38.1% | 55 | 6 | 260 | -16.5% |
| 256 to 1023 tiles | 152 | 97 | 12 | 43 | +2.3% | 91 | 15 | 46 | +2.3% | 114 | 16 | 22 | +3.6% |
| 1024 or more tiles | 157 | 113 | 26 | 18 | +2.5% | 134 | 19 | 4 | +3.2% | 134 | 19 | 4 | +3.2% |
| Best library uses 256x256 | 259 | 194 | 35 | 30 | +2.6% | 205 | 30 | 24 | +2.9% | 209 | 34 | 16 | +3.0% |
| Best library uses another tile | 371 | 37 | 9 | 325 | -29.8% | 39 | 6 | 326 | -31.9% | 94 | 7 | 270 | -11.4% |

## v9 and streamk_wip padded, against the best library over both layouts (strictest)

| Shapes | Count | v9: faster | v9: within 1% | v9: slower | v9: median | data-parallel only: faster | data-parallel only: within 1% | data-parallel only: slower | data-parallel only: median | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All shapes | 630 | 227 | 52 | 351 | -5.1% | 242 | 42 | 346 | -5.5% | 321 | 34 | 275 | +1.2% |
| Under 256 tiles (less than one wave) | 321 | 21 | 6 | 294 | -39.9% | 18 | 2 | 301 | -40.3% | 56 | 8 | 257 | -18.2% |
| 256 to 1023 tiles | 152 | 98 | 14 | 40 | +2.7% | 95 | 19 | 38 | +2.8% | 130 | 9 | 13 | +4.5% |
| 1024 or more tiles | 157 | 108 | 32 | 17 | +2.3% | 129 | 21 | 7 | +3.4% | 135 | 17 | 5 | +3.6% |
| Best library uses 256x256 | 260 | 190 | 38 | 32 | +2.6% | 205 | 30 | 25 | +3.3% | 227 | 21 | 12 | +3.7% |
| Best library uses another tile | 370 | 37 | 14 | 319 | -31.6% | 37 | 12 | 321 | -32.1% | 94 | 13 | 263 | -11.6% |

"Data-parallel only" is `STREAMK_POLICY=dp`, the persistent loop without stream-K and without the v9 hand-off; where capped already runs data-parallel on the persistent loop it is streamk_wip's time.

## v9 delegation (118 shapes: capped picks data-parallel, at most 256 tiles, bf16 output)

streamk_wip (now v9) against the persistent loop it replaced and against AITER's v9, same sweep. Positive means streamk_wip is faster.

| Layout | Shapes | Against | faster | within 1% | slower | median | p10 | p90 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| contiguous | all (118) | persistent loop (dp) | 38 | 64 | 16 | +0.1% | -1.1% | +3.6% |
| contiguous | all (118) | AITER's v9 | 1 | 54 | 63 | -1.0% | -4.0% | -0.2% |
| contiguous | k-steps <= 32 (54) | persistent loop (dp) | 37 | 13 | 4 | +2.0% | -0.7% | +4.3% |
| contiguous | k-steps <= 32 (54) | AITER's v9 | 0 | 31 | 23 | -0.9% | -5.0% | -0.2% |
| contiguous | k-steps > 32 (64) | persistent loop (dp) | 1 | 51 | 12 | -0.5% | -1.2% | +0.4% |
| contiguous | k-steps > 32 (64) | AITER's v9 | 1 | 23 | 40 | -1.2% | -2.7% | +0.0% |
| padded | all (118) | persistent loop (dp) | 60 | 57 | 1 | +1.0% | +0.4% | +2.4% |
| padded | all (118) | AITER's v9 | 1 | 84 | 33 | -0.6% | -2.0% | +0.1% |
| padded | k-steps <= 32 (54) | persistent loop (dp) | 48 | 6 | 0 | +1.8% | +0.9% | +3.0% |
| padded | k-steps <= 32 (54) | AITER's v9 | 1 | 28 | 25 | -0.7% | -2.5% | -0.1% |
| padded | k-steps > 32 (64) | persistent loop (dp) | 12 | 51 | 1 | +0.8% | +0.1% | +1.2% |
| padded | k-steps > 32 (64) | AITER's v9 | 0 | 56 | 8 | -0.5% | -1.1% | +0.2% |

### Against `results/details_capped_hbl_rot2048.csv` (contiguous, this run / that run - 1)

| Shapes | Count | streamk_wip | v9 (AITER) | AITER's pick | hipBLASLt |
|---|---:|---:|---:|---:|---:|
| Delegated to v9 now | 118 | +0.9% | -0.1% | -0.2% | -0.0% |
| Not delegated | 512 | -0.7% | -0.2% | -0.3% | -0.0% |

### The 17 same-tile shapes where streamk_wip was behind on Oct 5

| Shape | Tiles | k-steps | Delegated now | Oct 5: vs best | Now contiguous: vs best | Now padded: vs best | Now padded: vs v9 padded |
|---|---:|---:|---|---:|---:|---:|---:|
| 4096x4096x1536 | 256 | 24 | yes | -6.1% | -4.3% | -0.1% | -2.6% |
| 4096x4096x512 | 256 | 8 | yes | -5.2% | -3.1% | -2.0% | -1.5% |
| 4096x4096x1024 | 256 | 16 | yes | -5.0% | -1.6% | +1.5% | -3.1% |
| 1024x16384x2048 | 256 | 32 | yes | -4.2% | -0.9% | +1.4% | -2.3% |
| 2048x7168x1024 | 224 | 16 | yes | -4.1% | -0.9% | +2.3% | -2.9% |
| 1024x2304x16384 | 36 | 256 | no | -3.8% | -5.7% | -21.4% | +182.4% |
| 4096x4096x2048 | 256 | 32 | yes | -3.8% | -2.9% | +1.0% | -1.7% |
| 16384x2048x1536 | 512 | 24 | no | -2.8% | -2.6% | +1.4% | +0.5% |
| 8192x2048x512 | 256 | 8 | yes | -2.8% | -2.7% | -2.3% | -2.4% |
| 2048x7168x1792 | 224 | 28 | yes | -2.1% | +1.2% | +3.3% | -1.7% |
| 4096x8192x1024 | 512 | 16 | no | -2.0% | -1.2% | +0.7% | -0.7% |
| 32768x57344x8192 | 28672 | 128 | no | -1.8% | -0.2% | -4.6% | -5.8% |
| 2048x8192x1024 | 256 | 16 | yes | -1.7% | -0.9% | +1.1% | -4.1% |
| 16384x7168x512 | 1792 | 8 | no | -1.6% | -3.4% | -5.3% | +6.6% |
| 2048x5120x640 | 160 | 10 | yes | -1.4% | +0.5% | +1.0% | -0.4% |
| 16384x5120x640 | 1280 | 10 | no | -1.2% | +0.2% | +0.0% | +5.1% |
| 2048x8192x2048 | 256 | 32 | yes | -1.1% | -1.1% | +1.8% | -2.0% |

## Padding: does it remove the noise?

Spread between the 3 rounds, per kernel and layout. With `realloc` every round has new buffers, so a timing that depends on where the buffers landed shows up here. Control: the 16 shapes with K % 256 == 128, whose x and w strides are not padded (only C is).

| Kernel | Shapes | contiguous: median spread | contiguous: > 2% | contiguous: > 5% | padded: median spread | padded: > 2% | padded: > 5% |
|---|---|---:|---:|---:|---:|---:|---:|
| v9 | All, strides padded (614) | 0.31% | 11 | 2 | 0.29% | 5 | 1 |
| v9 | Under 256 (less than one wave), strides padded (314) | 0.35% | 5 | 1 | 0.32% | 1 | 0 |
| v9 | 256 to 1023, strides padded (148) | 0.29% | 3 | 0 | 0.28% | 3 | 0 |
| v9 | 1024 or more, strides padded (152) | 0.27% | 3 | 1 | 0.24% | 1 | 1 |
| v9 | Control (K % 256 == 128) (16) | 0.28% | 0 | 0 | 0.28% | 0 | 0 |
| streamk_wip | All, strides padded (614) | 0.66% | 45 | 3 | 0.37% | 13 | 3 |
| streamk_wip | Under 256 (less than one wave), strides padded (314) | 0.77% | 24 | 1 | 0.41% | 6 | 0 |
| streamk_wip | 256 to 1023, strides padded (148) | 0.67% | 11 | 1 | 0.34% | 2 | 0 |
| streamk_wip | 1024 or more, strides padded (152) | 0.48% | 10 | 1 | 0.30% | 5 | 3 |
| streamk_wip | Control (K % 256 == 128) (16) | 0.61% | 0 | 0 | 0.41% | 0 | 0 |
| AITER's pick | All, strides padded (362) | 0.46% | 9 | 1 | 0.39% | 18 | 0 |
| AITER's pick | Under 256 (less than one wave), strides padded (168) | 0.44% | 2 | 0 | 0.47% | 13 | 0 |
| AITER's pick | 256 to 1023, strides padded (89) | 0.38% | 1 | 0 | 0.36% | 1 | 0 |
| AITER's pick | 1024 or more, strides padded (105) | 0.57% | 6 | 1 | 0.33% | 4 | 0 |
| AITER's pick | Control (K % 256 == 128) (16) | 0.39% | 1 | 0 | 0.55% | 0 | 0 |
| hipBLASLt | All, strides padded (614) | 0.41% | 36 | 10 | 0.41% | 62 | 15 |
| hipBLASLt | Under 256 (less than one wave), strides padded (314) | 0.42% | 26 | 8 | 0.45% | 44 | 10 |
| hipBLASLt | 256 to 1023, strides padded (148) | 0.30% | 0 | 0 | 0.30% | 0 | 0 |
| hipBLASLt | 1024 or more, strides padded (152) | 0.48% | 10 | 2 | 0.47% | 18 | 5 |
| hipBLASLt | Control (K % 256 == 128) (16) | 0.38% | 0 | 0 | 0.34% | 0 | 0 |

AITER's pick leaves out (on both layouts) the shapes where its padded call copied its inputs.

### Padded against contiguous, median time over the rounds

Padded TFLOPS / contiguous TFLOPS - 1 (positive: padding is faster).

| Kernel | Shapes | Count | faster | within 1% | slower | median | p10 | p90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| v9 | All, strides padded | 614 | 210 | 177 | 227 | +0.3% | -7.3% | +3.0% |
| v9 | Under 256 (less than one wave) | 314 | 98 | 30 | 186 | -2.9% | -9.9% | +4.9% |
| v9 | 256 to 1023 | 148 | 72 | 55 | 21 | +0.9% | -1.5% | +2.4% |
| v9 | 1024 or more | 152 | 40 | 92 | 20 | +0.5% | -1.5% | +1.4% |
| v9 | Control (K % 256 == 128) | 16 | 1 | 15 | 0 | +0.2% | -0.1% | +0.9% |
| streamk_wip | All, strides padded | 614 | 287 | 166 | 161 | +0.9% | -3.8% | +3.4% |
| streamk_wip | Under 256 (less than one wave) | 314 | 131 | 53 | 130 | +0.2% | -5.1% | +4.2% |
| streamk_wip | 256 to 1023 | 148 | 89 | 47 | 12 | +1.2% | -0.2% | +3.1% |
| streamk_wip | 1024 or more | 152 | 67 | 66 | 19 | +0.9% | -1.7% | +2.0% |
| streamk_wip | Control (K % 256 == 128) | 16 | 7 | 9 | 0 | +0.7% | +0.4% | +1.9% |
| AITER's pick | All, strides padded | 362 | 114 | 162 | 86 | +0.4% | -2.4% | +2.5% |
| AITER's pick | Under 256 (less than one wave) | 168 | 52 | 44 | 72 | -0.4% | -3.6% | +3.0% |
| AITER's pick | 256 to 1023 | 89 | 31 | 46 | 12 | +0.6% | -1.2% | +2.3% |
| AITER's pick | 1024 or more | 105 | 31 | 72 | 2 | +0.5% | -0.2% | +2.0% |
| AITER's pick | Control (K % 256 == 128) | 16 | 1 | 15 | 0 | +0.3% | +0.0% | +0.9% |
| hipBLASLt | All, strides padded | 614 | 103 | 308 | 203 | -0.3% | -3.0% | +1.7% |
| hipBLASLt | Under 256 (less than one wave) | 314 | 53 | 108 | 153 | -0.9% | -3.7% | +2.1% |
| hipBLASLt | 256 to 1023 | 148 | 24 | 98 | 26 | +0.1% | -1.4% | +1.6% |
| hipBLASLt | 1024 or more | 152 | 26 | 102 | 24 | -0.2% | -1.5% | +1.7% |
| hipBLASLt | Control (K % 256 == 128) | 16 | 1 | 13 | 2 | -0.3% | -0.7% | +0.3% |

## Correctness

Checks: hbl_ok, pick_ok, v9_ok, v20_ok, v20_dp_ok, v20_capped_ok, v20_two_tile_ok, v20_post_ok, v20_capped_post_ok, pick_pad_ok, v9_pad_ok, hbl_pad_ok, v20_pad_ok, v20_dp_pad_ok, v20_pad_post_ok; and streamk_wip's flags and counts are back at 0 after timing.

- Shapes with any failed check or note: 50; of those, 26 are AITER's pick alone (contiguous and/or padded) outside the tolerance.
  - 1792x1024x4096 bfloat16: failed ['hbl_pad_ok'], flags 0, note 
  - 3328x512x4096 bfloat16: failed ['hbl_pad_ok'], flags 0, note 
  - 3584x512x4096 bfloat16: failed ['hbl_pad_ok'], flags 0, note 
  - 512x2560x6144 bfloat16: failed ['hbl_pad_ok'], flags 0, note 
  - 1024x2304x6144 bfloat16: failed ['hbl_pad_ok'], flags 0, note 
  - 512x1024x7168 bfloat16: failed ['hbl_pad_ok'], flags 0, note 
  - 768x1024x7168 bfloat16: failed ['hbl_pad_ok'], flags 0, note 
  - 768x2048x7168 bfloat16: failed ['pick_ok', 'pick_pad_ok', 'hbl_pad_ok'], flags 0, note 
  - 1024x512x7168 bfloat16: failed ['pick_ok', 'pick_pad_ok', 'hbl_pad_ok'], flags 0, note 
  - 1024x1536x7168 bfloat16: failed ['pick_ok', 'pick_pad_ok', 'hbl_pad_ok'], flags 0, note 
  - 1024x2048x7168 bfloat16: failed ['hbl_pad_ok'], flags 0, note 
  - 1280x512x7168 bfloat16: failed ['hbl_pad_ok'], flags 0, note 
  - 1536x512x7168 bfloat16: failed ['hbl_pad_ok'], flags 0, note 
  - 1792x512x7168 bfloat16: failed ['hbl_pad_ok'], flags 0, note 
  - 1792x1024x7168 bfloat16: failed ['hbl_pad_ok'], flags 0, note 
  - 2048x512x7168 bfloat16: failed ['hbl_pad_ok'], flags 0, note 
  - 512x1280x8192 bfloat16: failed ['hbl_pad_ok'], flags 0, note 
  - 512x2560x8192 bfloat16: failed ['hbl_pad_ok'], flags 0, note 
  - 1024x512x8192 bfloat16: failed ['hbl_pad_ok'], flags 0, note 
  - 1024x1280x8192 bfloat16: failed ['hbl_pad_ok'], flags 0, note 
  - 2048x512x8192 bfloat16: failed ['hbl_pad_ok'], flags 0, note 
  - 2048x1280x8192 bfloat16: failed ['hbl_pad_ok'], flags 0, note 
  - 512x2304x16384 bfloat16: failed ['hbl_pad_ok'], flags 0, note 
  - 1024x2304x16384 bfloat16: failed ['hbl_pad_ok'], flags 0, note 

## Appendix: every shape (TFLOPS)

| Shape | out | tiles | streamk_wip config | v9 hand-off | streamk_wip | padded | dp only | padded | v9 | padded | AITER pick | AITER | padded | hipBLASLt | padded | best (padded comparison) |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| 1024x3072x512 | bfloat16 | 48 | data-parallel | yes | 201.3 | 185.5 | 193.4 | 179.6 | 202.3 | 186.6 | flydsl | 374.1 | 183.8 (copy) | 322.5 | 322.9 | AITER flydsl contiguous 374.1 |
| 1024x4096x512 | bfloat16 | 64 | data-parallel | yes | 258.8 | 240.4 | 250.8 | 234.8 | 259.2 | 240.6 | flydsl | 435.6 | 230.9 (copy) | 393.6 | 387.1 | AITER flydsl contiguous 435.6 |
| 1024x7168x512 | bfloat16 | 112 | data-parallel | yes | 418.0 | 393.9 | 403.2 | 382.3 | 420.1 | 395.1 | flydsl | 513.7 | 311.6 (copy) | 510.4 | 478.4 | AITER flydsl contiguous 513.7 |
| 2048x2048x512 | bfloat16 | 64 | data-parallel | yes | 260.0 | 242.7 | 252.5 | 237.3 | 260.7 | 244.0 | flydsl | 433.2 | 241.2 (copy) | 394.1 | 388.8 | AITER flydsl contiguous 433.2 |
| 2048x4096x512 | bfloat16 | 128 | data-parallel | yes | 463.1 | 445.7 | 451.4 | 437.0 | 465.6 | 447.7 | torch | 560.3 | 533.0 | 559.2 | 543.2 | AITER torch contiguous 560.3 |
| 4096x2048x512 | bfloat16 | 128 | data-parallel | yes | 459.0 | 447.0 | 446.1 | 437.4 | 463.5 | 442.6 | torch | 647.2 | 609.1 | 653.4 | 608.0 | hipBLASLt contiguous 653.4 |
| 4096x4096x512 | bfloat16 | 256 | data-parallel | yes | 753.4 | 762.1 | 743.7 | 748.6 | 785.8 | 773.6 | torch | 769.9 | 755.4 | 777.9 | 755.3 | hipBLASLt contiguous 777.9 |
| 4096x7168x512 | bfloat16 | 448 | data-parallel |  | 766.2 | 758.5 | 766.2 | 758.5 | 767.5 | 740.9 | flydsl | 749.1 | 624.6 (copy) | 727.4 | 698.1 | AITER flydsl contiguous 749.1 |
| 8192x2048x512 | bfloat16 | 256 | data-parallel | yes | 748.9 | 752.0 | 741.5 | 734.7 | 788.6 | 770.3 | torch | 761.3 | 746.3 | 769.6 | 746.9 | hipBLASLt contiguous 769.6 |
| 8192x4096x512 | bfloat16 | 512 | data-parallel |  | 825.4 | 825.4 | 825.4 | 825.4 | 816.2 | 806.7 | flydsl | 804.9 | 653.1 (copy) | 845.4 | 837.0 | hipBLASLt contiguous 845.4 |
| 16384x2048x512 | bfloat16 | 512 | data-parallel |  | 820.0 | 822.7 | 820.0 | 822.7 | 827.9 | 795.8 | torch | 837.1 | 825.6 | 834.7 | 825.3 | AITER torch contiguous 837.1 |
| 16384x4096x512 | bfloat16 | 1024 | data-parallel |  | 879.8 | 896.2 | 879.8 | 896.2 | 841.3 | 845.2 | torch | 887.7 | 876.7 | 891.0 | 883.9 | hipBLASLt contiguous 891.0 |
| 16384x7168x512 | bfloat16 | 1792 | data-parallel |  | 958.4 | 939.6 | 958.4 | 939.6 | 905.6 | 881.2 | torch | 979.5 | 960.4 | 992.0 | 967.7 | hipBLASLt contiguous 992.0 |
| 32768x2048x512 | bfloat16 | 1024 | data-parallel |  | 892.0 | 872.7 | 892.0 | 872.7 | 855.3 | 809.9 | torch | 877.4 | 873.9 | 876.9 | 870.9 | AITER torch contiguous 877.4 |
| 1024x5120x640 | bfloat16 | 80 | data-parallel | yes | 322.8 | 328.9 | 316.4 | 317.2 | 325.1 | 325.5 | torch | 445.1 | 446.1 | 447.1 | 448.2 | hipBLASLt padded 448.2 |
| 2048x5120x640 | bfloat16 | 160 | data-parallel | yes | 599.2 | 602.5 | 586.4 | 588.4 | 606.0 | 605.2 | flydsl | 596.5 | 596.5 | 508.4 | 507.1 | AITER flydsl contiguous 596.5 |
| 4096x5120x640 | bfloat16 | 320 | data-parallel |  | 650.0 | 654.5 | 650.0 | 654.5 | 640.0 | 638.0 | flydsl | 613.4 | 614.7 | 654.1 | 652.1 | hipBLASLt contiguous 654.1 |
| 8192x5120x640 | bfloat16 | 640 | data-parallel |  | 821.9 | 825.1 | 821.9 | 825.1 | 800.2 | 798.9 | torch | 870.1 | 867.6 | 867.2 | 861.7 | AITER torch contiguous 870.1 |
| 16384x5120x640 | bfloat16 | 1280 | data-parallel |  | 1003.3 | 1013.4 | 1003.3 | 1013.4 | 962.2 | 964.6 | torch | 993.5 | 1009.8 | 1001.7 | 1012.9 | hipBLASLt padded 1012.9 |
| 32768x5120x640 | bfloat16 | 2560 | data-parallel |  | 1045.9 | 1054.3 | 1045.9 | 1054.3 | 1000.4 | 1005.5 | torch | 1049.2 | 1039.2 | 1062.0 | 1042.4 | hipBLASLt contiguous 1062.0 |
| 1024x7168x768 | bfloat16 | 112 | data-parallel | yes | 483.2 | 496.9 | 466.8 | 482.5 | 486.6 | 498.4 | flydsl | 596.6 | 384.7 (copy) | 590.0 | 603.8 | hipBLASLt padded 603.8 |
| 10240x7168x768 | bfloat16 | 1120 | data-parallel |  | 1026.5 | 1056.9 | 1026.5 | 1056.9 | 1014.2 | 1022.5 | opus | 391.5 | 401.7 | 862.9 | 861.7 | hipBLASLt contiguous 862.9 |
| 512x4096x1024 | bfloat16 | 32 | data-parallel | yes | 180.3 | 174.4 | 177.0 | 170.0 | 181.4 | 175.5 | torch | 389.0 | 379.7 | 393.0 | 382.5 | hipBLASLt contiguous 393.0 |
| 512x6144x1024 | bfloat16 | 48 | data-parallel | yes | 268.0 | 259.2 | 261.8 | 250.8 | 269.8 | 261.0 | flydsl | 496.6 | 265.4 (copy) | 475.0 | 464.7 | AITER flydsl contiguous 496.6 |
| 512x7168x1024 | bfloat16 | 56 | data-parallel | yes | 308.9 | 299.8 | 301.7 | 289.3 | 311.1 | 301.5 | flydsl | 542.0 | 295.8 (copy) | 500.9 | 497.3 | AITER flydsl contiguous 542.0 |
| 512x8192x1024 | bfloat16 | 64 | data-parallel | yes | 346.0 | 337.5 | 339.4 | 328.3 | 346.8 | 339.6 | flydsl | 576.0 | 325.0 (copy) | 551.5 | 553.9 | AITER flydsl contiguous 576.0 |
| 1024x4096x1024 | bfloat16 | 64 | data-parallel | yes | 345.5 | 337.4 | 342.5 | 329.5 | 346.6 | 338.7 | torch | 561.0 | 554.7 | 563.2 | 553.8 | hipBLASLt contiguous 563.2 |
| 1024x7168x1024 | bfloat16 | 112 | data-parallel | yes | 571.1 | 563.3 | 559.5 | 544.1 | 572.5 | 563.8 | flydsl | 671.7 | 454.0 (copy) | 695.9 | 690.8 | hipBLASLt contiguous 695.9 |
| 1024x8192x1024 | bfloat16 | 128 | data-parallel | yes | 632.9 | 633.3 | 623.7 | 619.4 | 631.5 | 642.0 | torch | 761.9 | 746.5 | 768.5 | 766.8 | hipBLASLt contiguous 768.5 |
| 2048x4096x1024 | bfloat16 | 128 | data-parallel | yes | 633.6 | 632.3 | 624.6 | 616.4 | 633.6 | 633.6 | torch | 770.8 | 756.2 | 776.9 | 771.9 | hipBLASLt contiguous 776.9 |
| 2048x7168x1024 | bfloat16 | 224 | data-parallel | yes | 955.0 | 987.9 | 960.5 | 963.5 | 1011.4 | 1017.8 | flydsl | 953.6 | 680.1 (copy) | 963.2 | 965.8 | hipBLASLt padded 965.8 |
| 2048x8192x1024 | bfloat16 | 256 | data-parallel | yes | 1009.3 | 1052.5 | 1013.3 | 1023.6 | 1071.6 | 1097.1 | flydsl | 1018.6 | 728.3 (copy) | 1017.2 | 1041.0 | hipBLASLt padded 1041.0 |
| 3072x4096x1024 | bfloat16 | 192 | data-parallel | yes | 871.7 | 877.4 | 862.5 | 860.7 | 899.0 | 899.9 | torch | 921.2 | 921.7 | 928.4 | 929.3 | hipBLASLt padded 929.3 |
| 4096x4096x1024 | bfloat16 | 256 | data-parallel | yes | 1030.7 | 1063.0 | 1025.3 | 1040.4 | 1084.8 | 1096.5 | torch | 1041.1 | 1027.1 | 1047.2 | 1024.8 | hipBLASLt contiguous 1047.2 |
| 4096x7168x1024 | bfloat16 | 448 | data-parallel |  | 1025.9 | 1038.7 | 1025.9 | 1038.7 | 1045.8 | 1046.9 | flydsl | 995.3 | 835.1 (copy) | 984.7 | 976.8 | AITER flydsl contiguous 995.3 |
| 4096x8192x1024 | bfloat16 | 512 | data-parallel |  | 1102.9 | 1124.6 | 1102.9 | 1124.6 | 1106.8 | 1132.6 | asm | 1111.3 | 1105.1 | 1115.8 | 1116.4 | hipBLASLt padded 1116.4 |
| 5120x4096x1024 | bfloat16 | 320 | data-parallel |  | 848.8 | 865.6 | 848.8 | 865.6 | 862.8 | 862.9 | torch | 974.1 | 979.0 | 977.8 | 977.5 | AITER torch padded 979.0 |
| 7680x4096x1024 | bfloat16 | 480 | data-parallel |  | 1064.2 | 1085.0 | 1064.2 | 1085.0 | 1095.2 | 1097.4 | torch | 945.3 | 940.6 | 950.4 | 937.4 | hipBLASLt contiguous 950.4 |
| 8192x7168x1024 | bfloat16 | 896 | data-parallel |  | 1081.8 | 1082.4 | 1081.8 | 1082.4 | 1053.5 | 1051.5 | asm | 1043.4 | 1042.1 | 1030.1 | 1015.3 | AITER asm contiguous 1043.4 |
| 8192x8192x1024 | bfloat16 | 1024 | data-parallel |  | 1162.8 | 1180.6 | 1162.8 | 1180.6 | 1139.8 | 1151.3 | asm | 1139.2 | 1141.5 | 1168.0 | 1157.7 | hipBLASLt contiguous 1168.0 |
| 16384x4096x1024 | bfloat16 | 1024 | data-parallel |  | 1174.0 | 1184.6 | 1174.0 | 1184.6 | 1148.1 | 1158.3 | torch | 1150.8 | 1155.0 | 1158.8 | 1161.0 | hipBLASLt padded 1161.0 |
| 16384x7168x1024 | bfloat16 | 1792 | data-parallel |  | 1221.5 | 1221.0 | 1221.5 | 1221.0 | 1189.7 | 1191.8 | asm | 1137.8 | 1142.6 | 1212.3 | 1198.3 | hipBLASLt contiguous 1212.3 |
| 16384x8192x1024 | bfloat16 | 2048 | data-parallel |  | 1226.6 | 1229.7 | 1226.6 | 1229.7 | 1182.0 | 1189.2 | torch | 1189.8 | 1193.3 | 1201.2 | 1197.5 | hipBLASLt contiguous 1201.2 |
| 32768x7168x1024 | bfloat16 | 3584 | data-parallel |  | 1246.4 | 1246.7 | 1246.4 | 1246.7 | 1212.7 | 1213.6 | torch | 1227.6 | 1218.8 | 1240.3 | 1222.0 | hipBLASLt contiguous 1240.3 |
| 32768x8192x1024 | bfloat16 | 4096 | data-parallel |  | 1250.6 | 1253.9 | 1250.6 | 1253.9 | 1205.7 | 1210.8 | torch | 1214.1 | 1212.7 | 1223.8 | 1220.3 | hipBLASLt contiguous 1223.8 |
| 512x5120x1280 | bfloat16 | 40 | data-parallel | yes | 202.2 | 228.2 | 197.4 | 224.3 | 203.8 | 229.8 | flydsl | 446.5 | 257.2 (copy) | 424.5 | 437.1 | AITER flydsl contiguous 446.5 |
| 1024x5120x1280 | bfloat16 | 80 | data-parallel | yes | 391.6 | 438.7 | 378.1 | 428.6 | 395.3 | 440.9 | torch | 604.7 | 628.5 | 609.1 | 631.0 | hipBLASLt padded 631.0 |
| 2048x5120x1280 | bfloat16 | 160 | data-parallel | yes | 759.0 | 823.5 | 723.5 | 808.9 | 760.2 | 832.9 | torch | 846.0 | 890.4 | 850.9 | 890.5 | hipBLASLt padded 890.5 |
| 4096x5120x1280 | bfloat16 | 320 | data-parallel |  | 799.3 | 863.8 | 799.3 | 863.8 | 798.2 | 860.0 | torch | 876.5 | 925.0 | 882.3 | 929.0 | hipBLASLt padded 929.0 |
| 8192x5120x1280 | bfloat16 | 640 | data-parallel |  | 1035.6 | 1104.4 | 1035.6 | 1104.4 | 1018.5 | 1084.3 | flydsl | 996.4 | 925.3 (copy) | 1077.6 | 1079.3 | hipBLASLt padded 1079.3 |
| 16384x5120x1280 | bfloat16 | 1280 | data-parallel |  | 1231.0 | 1279.3 | 1231.0 | 1279.3 | 1200.1 | 1248.7 | flydsl | 1176.7 | 1087.9 (copy) | 1228.8 | 1239.8 | hipBLASLt padded 1239.8 |
| 32768x5120x1280 | bfloat16 | 2560 | data-parallel |  | 1263.2 | 1298.8 | 1263.2 | 1298.8 | 1237.9 | 1273.4 | torch | 1252.6 | 1266.0 | 1262.7 | 1267.0 | hipBLASLt padded 1267.0 |
| 512x1536x1536 | bfloat16 | 12 | one-tile reduce-scatter, n = 8 |  | 89.5 | 94.0 | 66.6 | 70.7 | 69.9 | 72.0 | flydsl | 265.8 | 140.3 (copy) | 239.9 | 232.6 | AITER flydsl contiguous 265.8 |
| 512x2048x1536 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 |  | 113.8 | 117.4 | 88.8 | 94.1 | 92.8 | 95.3 | torch | 309.6 | 309.3 | 312.8 | 308.5 | hipBLASLt contiguous 312.8 |
| 512x2304x1536 | bfloat16 | 18 | one-tile reduce-scatter, n = 7 |  | 118.8 | 122.1 | 99.9 | 106.4 | 104.5 | 107.2 | flydsl | 318.5 | 187.6 (copy) | 313.6 | 313.2 | AITER flydsl contiguous 318.5 |
| 512x4096x1536 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 195.5 | 202.4 | 179.1 | 190.7 | 186.4 | 191.3 | torch | 445.2 | 437.2 | 449.5 | 436.4 | hipBLASLt contiguous 449.5 |
| 512x6144x1536 | bfloat16 | 48 | data-parallel | yes | 275.9 | 284.2 | 264.4 | 279.9 | 278.5 | 285.0 | flydsl | 543.0 | 331.6 (copy) | 546.6 | 541.8 | hipBLASLt contiguous 546.6 |
| 512x7168x1536 | bfloat16 | 56 | data-parallel | yes | 317.9 | 327.3 | 306.2 | 322.7 | 320.3 | 329.2 | flydsl | 583.9 | 356.5 (copy) | 496.5 | 487.9 | AITER flydsl contiguous 583.9 |
| 1024x1536x1536 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 |  | 150.8 | 158.5 | 133.0 | 141.5 | 140.3 | 143.8 | flydsl | 404.6 | 250.9 (copy) | 391.6 | 391.4 | AITER flydsl contiguous 404.6 |
| 1024x2048x1536 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 196.2 | 201.7 | 177.0 | 187.7 | 185.8 | 190.1 | torch | 458.3 | 451.7 | 461.5 | 453.3 | hipBLASLt contiguous 461.5 |
| 1024x2304x1536 | bfloat16 | 36 | one-tile owner, n = 2 |  | 210.3 | 213.8 | 198.3 | 209.9 | 207.8 | 213.3 | torch | 457.8 | 458.4 | 460.3 | 460.2 | hipBLASLt contiguous 460.3 |
| 1024x3072x1536 | bfloat16 | 48 | data-parallel | yes | 275.0 | 282.6 | 266.0 | 280.1 | 277.1 | 283.6 | flydsl | 547.8 | 350.8 (copy) | 557.3 | 552.9 | hipBLASLt contiguous 557.3 |
| 1024x4096x1536 | bfloat16 | 64 | data-parallel | yes | 361.6 | 372.9 | 348.2 | 370.2 | 364.4 | 374.6 | torch | 559.9 | 566.1 | 561.2 | 567.6 | hipBLASLt padded 567.6 |
| 1024x7168x1536 | bfloat16 | 112 | data-parallel | yes | 612.0 | 629.7 | 590.1 | 621.5 | 616.3 | 634.6 | torch | 781.3 | 792.2 | 785.7 | 796.3 | hipBLASLt padded 796.3 |
| 2048x1536x1536 | bfloat16 | 48 | data-parallel | yes | 279.5 | 288.3 | 265.9 | 284.7 | 282.2 | 288.6 | torch | 552.3 | 548.4 | 553.7 | 550.0 | hipBLASLt contiguous 553.7 |
| 2048x2048x1536 | bfloat16 | 64 | data-parallel | yes | 367.6 | 374.8 | 351.8 | 372.4 | 370.3 | 378.7 | torch | 563.8 | 571.9 | 565.9 | 558.4 | AITER torch padded 571.9 |
| 2048x4096x1536 | bfloat16 | 128 | data-parallel | yes | 694.3 | 707.6 | 666.4 | 701.1 | 696.2 | 715.2 | torch | 847.9 | 868.2 | 853.4 | 877.3 | hipBLASLt padded 877.3 |
| 3072x2048x1536 | bfloat16 | 96 | data-parallel | yes | 536.8 | 548.0 | 512.8 | 542.5 | 539.1 | 550.5 | opus | 689.3 | 710.0 | 777.2 | 756.3 | hipBLASLt contiguous 777.2 |
| 3072x4096x1536 | bfloat16 | 192 | data-parallel | yes | 962.6 | 986.7 | 943.6 | 971.2 | 989.9 | 1001.9 | opus | 994.2 | 1040.5 | 961.6 | 964.1 | AITER opus padded 1040.5 |
| 4096x1536x1536 | bfloat16 | 96 | data-parallel | yes | 533.8 | 547.6 | 511.9 | 540.5 | 539.2 | 547.6 | torch | 762.2 | 747.0 | 768.5 | 753.6 | hipBLASLt contiguous 768.5 |
| 4096x2048x1536 | bfloat16 | 128 | data-parallel | yes | 684.2 | 707.5 | 660.4 | 701.1 | 692.9 | 708.6 | opus | 854.9 | 875.2 | 849.1 | 871.9 | AITER opus padded 875.2 |
| 4096x3072x1536 | bfloat16 | 192 | data-parallel | yes | 959.4 | 989.1 | 948.0 | 974.0 | 992.5 | 1003.6 | flydsl | 942.7 | 715.9 (copy) | 1011.6 | 1025.9 | hipBLASLt padded 1025.9 |
| 4096x4096x1536 | bfloat16 | 256 | data-parallel | yes | 1128.9 | 1184.2 | 1146.5 | 1175.8 | 1200.8 | 1215.6 | torch | 1173.8 | 1183.8 | 1180.1 | 1185.5 | hipBLASLt padded 1185.5 |
| 5120x2048x1536 | bfloat16 | 160 | data-parallel | yes | 824.9 | 855.9 | 805.9 | 844.6 | 845.4 | 876.1 | torch | 950.9 | 974.8 | 960.5 | 983.8 | hipBLASLt padded 983.8 |
| 5120x4096x1536 | bfloat16 | 320 | data-parallel |  | 935.1 | 977.1 | 935.1 | 977.1 | 946.8 | 964.8 | torch | 1004.1 | 1022.1 | 1013.4 | 1025.8 | hipBLASLt padded 1025.8 |
| 7680x2048x1536 | bfloat16 | 240 | data-parallel | yes | 1076.8 | 1133.4 | 1084.8 | 1112.4 | 1148.0 | 1164.0 | torch | 958.8 | 973.8 | 963.4 | 971.6 | AITER torch padded 973.8 |
| 7680x4096x1536 | bfloat16 | 480 | data-parallel |  | 1170.0 | 1213.0 | 1170.0 | 1213.0 | 1195.5 | 1213.1 | torch | 1030.6 | 1039.6 | 1039.8 | 1032.8 | hipBLASLt contiguous 1039.8 |
| 8192x1536x1536 | bfloat16 | 192 | data-parallel | yes | 951.8 | 977.0 | 936.1 | 966.5 | 982.4 | 995.4 | torch | 998.8 | 1040.1 | 1001.4 | 1045.9 | hipBLASLt padded 1045.9 |
| 10240x7168x1536 | bfloat16 | 1120 | data-parallel |  | 1232.2 | 1255.1 | 1232.2 | 1255.1 | 1218.3 | 1227.9 | asm | 1148.7 | 1154.0 | 1136.6 | 1134.2 | AITER asm padded 1154.0 |
| 16384x1536x1536 | bfloat16 | 384 | data-parallel |  | 1024.1 | 1073.5 | 1024.1 | 1073.5 | 1056.0 | 1079.0 | torch | 1054.4 | 1081.7 | 1067.7 | 1081.5 | AITER torch padded 1081.7 |
| 16384x2048x1536 | bfloat16 | 512 | data-parallel |  | 1187.5 | 1244.5 | 1187.5 | 1244.5 | 1215.9 | 1238.3 | torch | 1213.7 | 1223.7 | 1219.4 | 1227.8 | hipBLASLt padded 1227.8 |
| 16384x3072x1536 | bfloat16 | 768 | data-parallel |  | 1244.4 | 1280.1 | 1244.4 | 1280.1 | 1252.6 | 1268.2 | torch | 1243.3 | 1251.0 | 1251.5 | 1252.0 | hipBLASLt padded 1252.0 |
| 16384x4096x1536 | bfloat16 | 1024 | data-parallel |  | 1260.6 | 1309.4 | 1260.6 | 1309.4 | 1248.0 | 1278.0 | torch | 1258.6 | 1274.8 | 1269.2 | 1277.3 | hipBLASLt padded 1277.3 |
| 32768x1536x1536 | bfloat16 | 768 | data-parallel |  | 1234.6 | 1271.2 | 1234.6 | 1271.2 | 1237.1 | 1252.4 | flydsl | 1203.7 | 939.4 (copy) | 1227.7 | 1216.3 | hipBLASLt contiguous 1227.7 |
| 512x7168x1792 | bfloat16 | 56 | one-tile owner, n = 2 |  | 356.7 | 349.7 | 337.7 | 326.0 | 345.9 | 332.3 | flydsl | 620.0 | 380.9 (copy) | 506.0 | 504.6 | AITER flydsl contiguous 620.0 |
| 1024x7168x1792 | bfloat16 | 112 | data-parallel | yes | 658.3 | 644.4 | 654.0 | 630.8 | 668.4 | 647.5 | flydsl | 783.9 | 548.3 (copy) | 833.4 | 827.8 | hipBLASLt contiguous 833.4 |
| 2048x7168x1792 | bfloat16 | 224 | data-parallel | yes | 1124.0 | 1147.6 | 1130.2 | 1123.0 | 1176.2 | 1167.5 | flydsl | 1110.8 | 831.3 (copy) | 1082.4 | 1072.9 | AITER flydsl contiguous 1110.8 |
| 4096x7168x1792 | bfloat16 | 448 | data-parallel |  | 1186.5 | 1201.9 | 1186.5 | 1201.9 | 1210.3 | 1202.4 | flydsl | 1158.6 | 983.7 (copy) | 1129.6 | 1128.8 | AITER flydsl contiguous 1158.6 |
| 8192x7168x1792 | bfloat16 | 896 | data-parallel |  | 1244.6 | 1241.8 | 1244.6 | 1241.8 | 1229.7 | 1214.5 | asm | 1193.9 | 1193.1 | 1166.0 | 1152.7 | AITER asm contiguous 1193.9 |
| 16384x7168x1792 | bfloat16 | 1792 | data-parallel |  | 1363.3 | 1363.6 | 1363.3 | 1363.6 | 1341.9 | 1335.4 | torch | 1335.8 | 1340.6 | 1352.9 | 1342.2 | hipBLASLt contiguous 1352.9 |
| 32768x7168x1792 | bfloat16 | 3584 | data-parallel |  | 1381.8 | 1386.4 | 1381.8 | 1386.4 | 1357.3 | 1353.6 | torch | 1353.5 | 1356.5 | 1361.8 | 1357.9 | hipBLASLt contiguous 1361.8 |
| 512x4096x2048 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 254.5 | 247.6 | 211.9 | 197.8 | 215.2 | 200.7 | torch | 490.4 | 480.7 | 492.0 | 482.0 | hipBLASLt contiguous 492.0 |
| 512x6144x2048 | float32 | 48 | one-tile owner, n = 2 |  | 311.7 | 299.1 | 281.3 | 266.4 | 289.2 | 273.6 | asm | 332.6 | 331.3 | 563.4 | 563.0 | hipBLASLt contiguous 563.4 |
| 512x6144x2048 | bfloat16 | 48 | one-tile owner, n = 2 |  | 342.1 | 330.9 | 315.3 | 295.3 | 322.2 | 300.2 | flydsl | 604.9 | 368.0 (copy) | 596.8 | 589.6 | AITER flydsl contiguous 604.9 |
| 512x8192x2048 | bfloat16 | 64 | one-tile owner, n = 2 |  | 439.2 | 428.6 | 406.7 | 388.2 | 412.1 | 393.2 | flydsl | 705.7 | 412.1 (copy) | 578.5 | 572.6 | AITER flydsl contiguous 705.7 |
| 512x16384x2048 | bfloat16 | 128 | data-parallel | yes | 747.8 | 739.8 | 742.5 | 731.6 | 757.5 | 748.6 | flydsl | 717.3 | 485.3 (copy) | 879.3 | 862.1 | hipBLASLt contiguous 879.3 |
| 1024x4096x2048 | bfloat16 | 64 | one-tile owner, n = 2 |  | 447.5 | 432.2 | 413.4 | 392.0 | 419.2 | 397.5 | torch | 617.8 | 605.0 | 619.8 | 604.5 | hipBLASLt contiguous 619.8 |
| 1024x6144x2048 | float32 | 96 | data-parallel |  | 533.0 | 514.2 | 533.0 | 514.2 | 550.4 | 529.8 | opus | 559.9 | 537.2 | 789.6 | 780.3 | hipBLASLt contiguous 789.6 |
| 1024x6144x2048 | bfloat16 | 96 | data-parallel | yes | 608.0 | 590.2 | 599.9 | 579.9 | 610.0 | 591.8 | flydsl | 739.6 | 524.0 (copy) | 806.3 | 784.0 | hipBLASLt contiguous 806.3 |
| 1024x8192x2048 | bfloat16 | 128 | data-parallel | yes | 764.5 | 764.1 | 759.1 | 752.3 | 772.1 | 773.4 | flydsl | 880.4 | 659.5 (copy) | 926.0 | 911.0 | hipBLASLt contiguous 926.0 |
| 1024x16384x2048 | bfloat16 | 256 | data-parallel | yes | 1218.7 | 1252.8 | 1227.0 | 1237.0 | 1265.2 | 1281.7 | flydsl | 749.1 | 671.6 (copy) | 1230.1 | 1235.7 | hipBLASLt padded 1235.7 |
| 2048x4096x2048 | bfloat16 | 128 | data-parallel | yes | 776.1 | 772.0 | 771.6 | 759.6 | 783.3 | 780.2 | torch | 940.4 | 927.8 | 949.4 | 933.5 | hipBLASLt contiguous 949.4 |
| 2048x6144x2048 | float32 | 192 | data-parallel |  | 920.2 | 926.6 | 920.2 | 926.6 | 972.7 | 961.8 | asm | 920.6 | 926.3 | 1022.8 | 1021.7 | hipBLASLt contiguous 1022.8 |
| 2048x6144x2048 | bfloat16 | 192 | data-parallel | yes | 1054.3 | 1080.1 | 1066.1 | 1061.9 | 1103.8 | 1102.5 | flydsl | 1045.3 | 794.3 (copy) | 1103.5 | 1096.9 | hipBLASLt contiguous 1103.5 |
| 2048x8192x2048 | bfloat16 | 256 | data-parallel | yes | 1224.3 | 1277.3 | 1250.7 | 1256.9 | 1285.9 | 1303.8 | flydsl | 1222.0 | 923.1 (copy) | 1237.3 | 1255.1 | hipBLASLt padded 1255.1 |
| 2048x16384x2048 | bfloat16 | 512 | data-parallel |  | 1293.3 | 1328.2 | 1293.3 | 1328.2 | 1312.7 | 1330.3 | flydsl | 1047.3 | 878.0 (copy) | 1287.8 | 1289.1 | hipBLASLt padded 1289.1 |
| 3072x4096x2048 | bfloat16 | 192 | data-parallel | yes | 1064.1 | 1075.8 | 1061.5 | 1054.5 | 1098.0 | 1099.4 | torch | 1099.5 | 1104.4 | 1107.4 | 1103.4 | hipBLASLt contiguous 1107.4 |
| 4096x4096x2048 | bfloat16 | 256 | data-parallel | yes | 1233.4 | 1282.5 | 1258.1 | 1264.4 | 1296.6 | 1304.6 | torch | 1263.1 | 1267.6 | 1269.9 | 1267.0 | hipBLASLt contiguous 1269.9 |
| 4096x6144x2048 | float32 | 384 | data-parallel |  | 988.4 | 980.9 | 988.4 | 980.9 | 1002.9 | 991.3 | asm | 955.4 | 944.5 | 1066.9 | 1056.0 | hipBLASLt contiguous 1066.9 |
| 4096x6144x2048 | bfloat16 | 384 | data-parallel |  | 1127.6 | 1135.2 | 1127.6 | 1135.2 | 1132.5 | 1129.6 | flydsl | 892.9 | 862.5 (copy) | 1147.5 | 1138.4 | hipBLASLt contiguous 1147.5 |
| 4096x8192x2048 | bfloat16 | 512 | data-parallel |  | 1299.0 | 1334.1 | 1299.0 | 1334.1 | 1324.7 | 1334.2 | flydsl | 1274.9 | 1079.5 (copy) | 1309.1 | 1313.9 | hipBLASLt padded 1313.9 |
| 4096x16384x2048 | bfloat16 | 1024 | data-parallel |  | 1357.7 | 1372.6 | 1357.7 | 1372.6 | 1342.0 | 1352.6 | flydsl | 857.3 | 859.1 (copy) | 1343.7 | 1328.1 | hipBLASLt contiguous 1343.7 |
| 5120x4096x2048 | bfloat16 | 320 | one-tile owner, n = 2 |  | 1026.6 | 1043.7 | 1036.5 | 1029.7 | 1036.3 | 1029.1 | torch | 1133.3 | 1098.4 | 1142.7 | 1089.8 | hipBLASLt contiguous 1142.7 |
| 7680x4096x2048 | bfloat16 | 480 | data-parallel |  | 1264.7 | 1298.5 | 1264.7 | 1298.5 | 1291.2 | 1293.2 | torch | 1100.3 | 1094.4 | 1111.7 | 1093.0 | hipBLASLt contiguous 1111.7 |
| 8192x6144x2048 | float32 | 768 | data-parallel |  | 1182.8 | 1191.5 | 1182.8 | 1191.5 | 1188.4 | 1187.5 | asm | 1153.0 | 1154.1 | 1112.8 | 1109.6 | AITER asm padded 1154.1 |
| 8192x6144x2048 | bfloat16 | 768 | data-parallel |  | 1351.2 | 1371.0 | 1351.2 | 1371.0 | 1347.3 | 1348.1 | flydsl | 832.5 | 775.0 (copy) | 1340.9 | 1334.1 | hipBLASLt contiguous 1340.9 |
| 8192x8192x2048 | bfloat16 | 1024 | data-parallel |  | 1372.0 | 1382.3 | 1372.0 | 1382.3 | 1350.2 | 1357.7 | asm | 1318.5 | 1318.5 | 1351.4 | 1347.6 | hipBLASLt contiguous 1351.4 |
| 8192x16384x2048 | bfloat16 | 2048 | data-parallel |  | 1384.4 | 1387.6 | 1384.4 | 1387.6 | 1354.5 | 1360.2 | triton | 1006.9 | 1007.4 | 1376.3 | 1360.9 | hipBLASLt contiguous 1376.3 |
| 16384x4096x2048 | bfloat16 | 1024 | data-parallel |  | 1363.1 | 1378.2 | 1363.1 | 1378.2 | 1345.8 | 1354.1 | torch | 1320.9 | 1322.6 | 1332.7 | 1324.4 | hipBLASLt contiguous 1332.7 |
| 16384x6144x2048 | float32 | 1536 | data-parallel |  | 1212.3 | 1216.6 | 1212.3 | 1216.6 | 1200.4 | 1204.8 | asm | 1169.9 | 1169.5 | 1142.4 | 1141.4 | AITER asm contiguous 1169.9 |
| 16384x6144x2048 | bfloat16 | 1536 | data-parallel |  | 1388.0 | 1395.3 | 1388.0 | 1395.3 | 1363.6 | 1369.2 | torch | 1357.2 | 1363.8 | 1372.6 | 1366.6 | hipBLASLt contiguous 1372.6 |
| 16384x8192x2048 | bfloat16 | 2048 | data-parallel |  | 1395.4 | 1410.1 | 1395.4 | 1410.1 | 1369.7 | 1378.0 | torch | 1370.0 | 1373.6 | 1386.0 | 1372.9 | hipBLASLt contiguous 1386.0 |
| 16384x16384x2048 | bfloat16 | 4096 | data-parallel |  | 1405.4 | 1400.0 | 1405.4 | 1400.0 | 1365.9 | 1363.0 | flydsl | 960.6 | 918.9 (copy) | 1397.0 | 1387.0 | hipBLASLt contiguous 1397.0 |
| 32768x6144x2048 | bfloat16 | 3072 | data-parallel |  | 1406.7 | 1419.7 | 1406.7 | 1419.7 | 1391.8 | 1385.0 | flydsl | 991.0 | 938.2 (copy) | 1385.2 | 1377.4 | hipBLASLt contiguous 1385.2 |
| 32768x8192x2048 | bfloat16 | 4096 | data-parallel |  | 1421.8 | 1426.6 | 1421.8 | 1426.6 | 1393.0 | 1383.2 | torch | 1379.0 | 1381.5 | 1389.7 | 1387.0 | hipBLASLt contiguous 1389.7 |
| 32768x16384x2048 | bfloat16 | 8192 | data-parallel |  | 1404.7 | 1407.6 | 1404.7 | 1407.6 | 1360.4 | 1367.0 | flydsl | 828.0 | 814.7 (copy) | 1408.4 | 1398.8 | hipBLASLt contiguous 1408.4 |
| 512x6144x3072 | float32 | 48 | one-tile owner, n = 2 |  | 364.3 | 379.4 | 282.7 | 300.5 | 294.3 | 300.7 | opus | 372.2 | 378.5 | 625.0 | 632.5 | hipBLASLt padded 632.5 |
| 512x6144x3072 | bfloat16 | 48 | one-tile owner, n = 2 |  | 394.2 | 409.9 | 304.5 | 323.1 | 316.0 | 322.4 | flydsl | 641.6 | 421.1 (copy) | 657.4 | 654.3 | hipBLASLt contiguous 657.4 |
| 1024x6144x3072 | float32 | 96 | one-tile owner, n = 2-3 |  | 601.3 | 620.0 | 554.9 | 584.8 | 577.4 | 589.6 | opus | 566.8 | 582.7 | 847.1 | 855.5 | hipBLASLt padded 855.5 |
| 1024x6144x3072 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 640.1 | 669.3 | 597.7 | 636.0 | 619.0 | 636.3 | flydsl | 784.6 | 599.9 (copy) | 846.2 | 830.7 | hipBLASLt contiguous 846.2 |
| 2048x6144x3072 | float32 | 192 | data-parallel |  | 1000.3 | 1048.3 | 1000.3 | 1048.3 | 1049.7 | 1066.9 | asm | 1016.2 | 1019.8 | 1119.5 | 1129.6 | hipBLASLt padded 1129.6 |
| 2048x6144x3072 | bfloat16 | 192 | data-parallel | yes | 1128.1 | 1186.2 | 1129.0 | 1171.8 | 1178.3 | 1197.5 | flydsl | 1118.1 | 871.7 (copy) | 1179.0 | 1184.8 | hipBLASLt padded 1184.8 |
| 4096x6144x3072 | float32 | 384 | data-parallel |  | 1062.1 | 1109.8 | 1062.1 | 1109.8 | 1086.7 | 1116.9 | asm | 1048.1 | 1061.8 | 1158.4 | 1160.9 | hipBLASLt padded 1160.9 |
| 4096x6144x3072 | bfloat16 | 384 | data-parallel |  | 1190.9 | 1228.6 | 1190.9 | 1228.6 | 1204.2 | 1219.3 | flydsl | 925.7 | 913.8 (copy) | 1222.7 | 1223.6 | hipBLASLt padded 1223.6 |
| 8192x6144x3072 | float32 | 768 | data-parallel |  | 1281.8 | 1306.1 | 1281.8 | 1306.1 | 1290.3 | 1303.7 | asm | 1257.4 | 1267.9 | 1210.5 | 1210.6 | AITER asm padded 1267.9 |
| 8192x6144x3072 | bfloat16 | 768 | data-parallel |  | 1430.4 | 1472.9 | 1430.4 | 1472.9 | 1427.3 | 1461.0 | flydsl | 780.8 | 780.3 (copy) | 1406.9 | 1408.4 | hipBLASLt padded 1408.4 |
| 16384x6144x3072 | float32 | 1536 | data-parallel |  | 1311.4 | 1328.9 | 1311.4 | 1328.9 | 1304.1 | 1319.4 | asm | 1273.5 | 1279.3 | 1235.0 | 1231.6 | AITER asm padded 1279.3 |
| 16384x6144x3072 | bfloat16 | 1536 | data-parallel |  | 1474.3 | 1497.3 | 1474.3 | 1497.3 | 1457.3 | 1479.6 | torch | 1421.4 | 1428.7 | 1437.2 | 1431.1 | hipBLASLt contiguous 1437.2 |
| 32768x6144x3072 | bfloat16 | 3072 | data-parallel |  | 1492.9 | 1520.4 | 1492.9 | 1520.4 | 1468.7 | 1489.7 | flydsl | 863.7 | 804.9 (copy) | 1447.2 | 1444.6 | hipBLASLt contiguous 1447.2 |
| 512x5120x3200 | bfloat16 | 40 | one-tile owner, n = 2 |  | 357.9 | 357.8 | 270.2 | 270.5 | 269.3 | 269.9 | flydsl | 575.5 | 580.9 | 563.6 | 565.4 | AITER flydsl padded 580.9 |
| 1024x5120x3200 | bfloat16 | 80 | one-tile owner, n = 2 |  | 669.1 | 681.9 | 535.6 | 536.5 | 532.9 | 533.4 | torch | 823.7 | 825.5 | 825.7 | 826.3 | hipBLASLt padded 826.3 |
| 2048x5120x3200 | bfloat16 | 160 | data-parallel | yes | 1017.1 | 1040.7 | 1027.7 | 1038.9 | 1050.6 | 1055.0 | torch | 1093.7 | 1096.5 | 1102.4 | 1095.3 | hipBLASLt contiguous 1102.4 |
| 4096x5120x3200 | bfloat16 | 320 | one-tile owner, n = 2 |  | 1145.0 | 1163.4 | 1075.8 | 1073.9 | 1063.4 | 1065.0 | torch | 1128.0 | 1129.6 | 1136.4 | 1131.3 | hipBLASLt contiguous 1136.4 |
| 8192x5120x3200 | bfloat16 | 640 | data-parallel |  | 1325.8 | 1334.9 | 1325.8 | 1334.9 | 1316.5 | 1320.8 | asm | 1261.9 | 1265.9 | 1281.5 | 1264.4 | hipBLASLt contiguous 1281.5 |
| 16384x5120x3200 | bfloat16 | 1280 | data-parallel |  | 1477.1 | 1480.6 | 1477.1 | 1480.6 | 1457.2 | 1469.7 | torch | 1423.1 | 1428.7 | 1434.4 | 1428.1 | hipBLASLt contiguous 1434.4 |
| 32768x5120x3200 | bfloat16 | 2560 | data-parallel |  | 1493.3 | 1510.6 | 1493.3 | 1510.6 | 1474.0 | 1481.3 | torch | 1435.6 | 1441.0 | 1449.8 | 1445.2 | hipBLASLt contiguous 1449.8 |
| 512x7168x3584 | bfloat16 | 56 | one-tile owner, n = 2 |  | 498.4 | 505.0 | 373.4 | 384.6 | 376.9 | 388.1 | flydsl | 711.6 | 476.4 (copy) | 601.3 | 604.5 | AITER flydsl contiguous 711.6 |
| 512x8192x3584 | bfloat16 | 64 | one-tile owner, n = 2 |  | 563.7 | 565.9 | 427.5 | 436.6 | 430.0 | 440.2 | flydsl | 674.8 | 465.9 (copy) | 656.3 | 665.9 | AITER flydsl contiguous 674.8 |
| 1024x7168x3584 | bfloat16 | 112 | one-tile owner, n = 2-3 |  | 833.4 | 843.9 | 735.3 | 759.0 | 741.9 | 771.4 | torch | 956.6 | 968.2 | 962.1 | 967.9 | AITER torch padded 968.2 |
| 1024x8192x3584 | bfloat16 | 128 | data-parallel | yes | 823.9 | 871.4 | 822.5 | 861.2 | 832.5 | 875.9 | flydsl | 788.4 | 631.4 (copy) | 1021.5 | 1030.1 | hipBLASLt padded 1030.1 |
| 2048x8192x3584 | bfloat16 | 256 | data-parallel | yes | 1358.5 | 1415.6 | 1386.2 | 1396.4 | 1393.8 | 1422.6 | flydsl | 1357.4 | 1019.3 (copy) | 1360.4 | 1374.5 | hipBLASLt padded 1374.5 |
| 4096x8192x3584 | bfloat16 | 512 | data-parallel |  | 1430.4 | 1464.1 | 1430.4 | 1464.1 | 1434.5 | 1460.9 | flydsl | 981.0 | 894.8 (copy) | 1416.9 | 1420.8 | hipBLASLt padded 1420.8 |
| 8192x8192x3584 | bfloat16 | 1024 | data-parallel |  | 1481.3 | 1499.5 | 1481.3 | 1499.5 | 1456.2 | 1481.5 | torch | 1434.6 | 1443.3 | 1454.1 | 1450.2 | hipBLASLt contiguous 1454.1 |
| 10240x7168x3584 | bfloat16 | 1120 | one-tile owner, n = 2-3 |  | 1415.7 | 1423.5 | 1401.3 | 1405.3 | 1385.6 | 1392.8 | flydsl | 782.0 | 736.2 (copy) | 1308.3 | 1302.7 | hipBLASLt contiguous 1308.3 |
| 16384x8192x3584 | bfloat16 | 2048 | data-parallel |  | 1492.4 | 1521.0 | 1492.4 | 1521.0 | 1477.3 | 1498.1 | flydsl | 804.3 | 798.4 (copy) | 1468.0 | 1464.9 | hipBLASLt contiguous 1468.0 |
| 32768x8192x3584 | bfloat16 | 4096 | data-parallel |  | 1520.4 | 1545.7 | 1520.4 | 1545.7 | 1493.9 | 1510.5 | flydsl | 856.9 | 828.8 (copy) | 1470.6 | 1471.7 | hipBLASLt padded 1471.7 |
| 512x1024x4096 | bfloat16 | 8 | one-tile reduce-scatter, n = 8 |  | 158.1 | 151.2 | 59.3 | 53.1 | 60.1 | 53.2 | flydsl | 284.2 | 186.6 (copy) | 264.9 | 259.4 | AITER flydsl contiguous 284.2 |
| 512x2048x4096 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 |  | 271.7 | 259.6 | 119.6 | 105.8 | 120.2 | 106.6 | flydsl | 441.1 | 283.3 (copy) | 425.7 | 419.0 | AITER flydsl contiguous 441.1 |
| 512x4096x4096 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 404.8 | 384.7 | 239.4 | 214.7 | 240.2 | 216.3 | flydsl | 600.3 | 397.9 (copy) | 620.7 | 616.3 | hipBLASLt contiguous 620.7 |
| 512x8192x4096 | bfloat16 | 64 | one-tile owner, n = 2 |  | 604.7 | 582.1 | 467.5 | 426.9 | 468.6 | 429.8 | flydsl | 804.4 | 510.3 (copy) | 683.4 | 684.3 | AITER flydsl contiguous 804.4 |
| 512x8704x4096 | bfloat16 | 68 | one-tile owner, n = 2 |  | 629.8 | 607.1 | 498.0 | 452.7 | 495.4 | 455.8 | torch | 749.7 | 758.4 | 751.7 | 757.1 | AITER torch padded 758.4 |
| 512x10240x4096 | bfloat16 | 80 | one-tile owner, n = 2 |  | 724.6 | 697.2 | 573.5 | 528.7 | 576.3 | 531.9 | torch | 853.4 | 861.7 | 860.0 | 866.5 | hipBLASLt padded 866.5 |
| 512x16384x4096 | bfloat16 | 128 | one-tile owner, n = 2 |  | 937.8 | 965.8 | 844.4 | 810.4 | 853.5 | 817.7 | flydsl | 632.2 | 416.3 (copy) | 690.2 | 719.6 | hipBLASLt padded 719.6 |
| 512x129280x4096 | bfloat16 | 1010 | data-parallel |  | 1376.0 | 1377.1 | 1376.0 | 1377.1 | 1376.8 | 1372.2 | asm | 1302.6 | 1316.6 | 1149.7 | 1185.8 | AITER asm padded 1316.6 |
| 768x1024x4096 | bfloat16 | 12 | one-tile reduce-scatter, n = 8 |  | 222.0 | 212.9 | 90.0 | 80.5 | 91.2 | 81.4 | torch | 371.4 | 366.3 | 372.1 | 366.9 | hipBLASLt contiguous 372.1 |
| 768x2048x4096 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 |  | 332.7 | 311.9 | 180.4 | 160.3 | 181.3 | 162.7 | flydsl | 528.7 | 334.5 (copy) | 498.6 | 480.8 | AITER flydsl contiguous 528.7 |
| 768x4096x4096 | bfloat16 | 48 | one-tile owner, n = 3 |  | 525.0 | 497.1 | 359.6 | 324.5 | 361.8 | 327.2 | flydsl | 697.0 | 480.3 (copy) | 660.7 | 663.1 | AITER flydsl contiguous 697.0 |
| 768x8192x4096 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 749.0 | 725.2 | 685.0 | 640.0 | 687.3 | 647.4 | torch | 874.5 | 858.6 | 881.7 | 856.8 | hipBLASLt contiguous 881.7 |
| 768x8704x4096 | bfloat16 | 102 | one-tile owner, n = 2-3 |  | 801.5 | 782.5 | 718.8 | 674.8 | 728.3 | 678.8 | flydsl | 871.3 | 618.8 (copy) | 845.9 | 819.2 | AITER flydsl contiguous 871.3 |
| 768x10240x4096 | bfloat16 | 120 | one-tile owner, n = 2-3 |  | 909.0 | 936.2 | 834.2 | 781.7 | 834.4 | 789.2 | flydsl | 970.4 | 721.0 (copy) | 936.9 | 892.8 | AITER flydsl contiguous 970.4 |
| 768x129280x4096 | bfloat16 | 1515 | data-parallel |  | 1438.4 | 1443.5 | 1438.4 | 1443.5 | 1419.9 | 1422.4 | torch | 1339.1 | 1342.2 | 1350.4 | 1345.3 | hipBLASLt contiguous 1350.4 |
| 1024x512x4096 | bfloat16 | 8 | one-tile reduce-scatter, n = 8 |  | 158.1 | 152.8 | 60.1 | 53.5 | 60.5 | 53.7 | flydsl | 282.4 | 182.0 (copy) | 262.8 | 257.4 | AITER flydsl contiguous 282.4 |
| 1024x1024x4096 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 |  | 274.6 | 265.0 | 121.6 | 110.0 | 121.7 | 108.5 | flydsl | 443.7 | 299.5 (copy) | 443.0 | 430.0 | AITER flydsl contiguous 443.7 |
| 1024x2048x4096 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 403.5 | 386.7 | 240.1 | 215.1 | 241.1 | 216.0 | flydsl | 603.6 | 424.2 (copy) | 495.6 | 509.2 | AITER flydsl contiguous 603.6 |
| 1024x4096x4096 | bfloat16 | 64 | one-tile owner, n = 2 |  | 616.3 | 593.3 | 470.5 | 429.8 | 471.3 | 432.6 | flydsl | 828.2 | 592.9 (copy) | 733.7 | 721.0 | AITER flydsl contiguous 828.2 |
| 1024x8192x4096 | bfloat16 | 128 | one-tile owner, n = 2 |  | 972.4 | 1001.8 | 882.6 | 847.8 | 880.0 | 859.0 | flydsl | 1015.9 | 785.7 (copy) | 690.0 | 695.2 | AITER flydsl contiguous 1015.9 |
| 1024x8704x4096 | bfloat16 | 136 | data-parallel | yes | 937.0 | 893.7 | 936.6 | 889.0 | 941.8 | 894.7 | torch | 998.2 | 984.4 | 1000.6 | 972.7 | hipBLASLt contiguous 1000.6 |
| 1024x10240x4096 | bfloat16 | 160 | data-parallel | yes | 1058.6 | 1038.2 | 1063.4 | 1029.9 | 1077.4 | 1049.8 | torch | 1119.8 | 1109.4 | 1128.6 | 1109.9 | hipBLASLt contiguous 1128.6 |
| 1024x16384x4096 | bfloat16 | 256 | data-parallel | yes | 1376.9 | 1418.5 | 1383.0 | 1405.5 | 1404.1 | 1426.8 | flydsl | 821.1 | 638.0 (copy) | 1335.3 | 1347.0 | hipBLASLt padded 1347.0 |
| 1024x129280x4096 | bfloat16 | 2020 | data-parallel |  | 1466.5 | 1471.2 | 1466.5 | 1471.2 | 1446.8 | 1446.5 | torch | 1366.4 | 1366.6 | 1374.5 | 1366.5 | hipBLASLt contiguous 1374.5 |
| 1280x512x4096 | bfloat16 | 10 | one-tile reduce-scatter, n = 8 |  | 190.8 | 182.1 | 75.2 | 67.3 | 76.2 | 67.4 | flydsl | 319.4 | 190.2 (copy) | 310.2 | 308.1 | AITER flydsl contiguous 319.4 |
| 1280x1024x4096 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 |  | 295.4 | 281.0 | 150.3 | 135.1 | 151.7 | 135.8 | torch | 486.3 | 473.9 | 489.2 | 474.1 | hipBLASLt contiguous 489.2 |
| 1280x2048x4096 | bfloat16 | 40 | one-tile owner, n = 3 |  | 451.6 | 421.1 | 301.2 | 269.8 | 303.5 | 273.4 | flydsl | 620.5 | 435.2 (copy) | 614.0 | 604.9 | AITER flydsl contiguous 620.5 |
| 1280x129280x4096 | bfloat16 | 2525 | data-parallel |  | 1435.8 | 1437.9 | 1435.8 | 1437.9 | 1421.6 | 1406.4 | torch | 1385.6 | 1384.7 | 1391.3 | 1387.6 | hipBLASLt contiguous 1391.3 |
| 1536x512x4096 | bfloat16 | 12 | one-tile reduce-scatter, n = 8 |  | 218.5 | 212.4 | 91.6 | 81.9 | 91.4 | 81.5 | torch | 367.9 | 360.2 | 369.1 | 360.5 | hipBLASLt contiguous 369.1 |
| 1536x1024x4096 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 |  | 327.1 | 315.6 | 181.9 | 163.2 | 183.4 | 163.7 | flydsl | 522.4 | 339.2 (copy) | 506.1 | 491.1 | AITER flydsl contiguous 522.4 |
| 1536x2048x4096 | bfloat16 | 48 | one-tile owner, n = 3 |  | 528.8 | 504.2 | 364.0 | 330.3 | 365.4 | 330.0 | flydsl | 699.6 | 500.3 (copy) | 683.7 | 684.6 | AITER flydsl contiguous 699.6 |
| 1536x129280x4096 | bfloat16 | 3030 | data-parallel |  | 1453.6 | 1460.1 | 1453.6 | 1460.1 | 1436.0 | 1436.2 | torch | 1391.2 | 1391.5 | 1400.3 | 1389.8 | hipBLASLt contiguous 1400.3 |
| 1792x512x4096 | bfloat16 | 14 | one-tile reduce-scatter, n = 8 |  | 248.9 | 235.9 | 104.7 | 94.9 | 106.6 | 95.3 | flydsl | 393.4 | 239.5 (copy) | 385.7 | 366.9 | AITER flydsl contiguous 393.4 |
| 1792x1024x4096 | bfloat16 | 28 | one-tile reduce-scatter, n = 4 |  | 368.3 | 349.4 | 212.0 | 191.8 | 214.5 | 192.6 | flydsl | 575.6 | 376.9 (copy) | 469.2 | 456.9 | AITER flydsl contiguous 575.6 |
| 1792x2048x4096 | bfloat16 | 56 | one-tile owner, n = 2 |  | 560.6 | 526.3 | 419.6 | 380.7 | 424.1 | 384.9 | flydsl | 751.9 | 557.8 (copy) | 668.6 | 652.2 | AITER flydsl contiguous 751.9 |
| 1792x129280x4096 | bfloat16 | 3535 | data-parallel |  | 1475.5 | 1476.5 | 1475.5 | 1476.5 | 1449.9 | 1449.4 | asm | 1436.3 | 1438.2 | 1357.5 | 1348.3 | AITER asm padded 1438.2 |
| 2048x512x4096 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 |  | 271.1 | 262.0 | 121.6 | 107.9 | 121.9 | 109.3 | flydsl | 428.5 | 272.8 (copy) | 432.9 | 414.3 | hipBLASLt contiguous 432.9 |
| 2048x1024x4096 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 403.3 | 391.6 | 243.4 | 219.1 | 244.2 | 220.7 | flydsl | 606.6 | 431.9 (copy) | 505.3 | 499.6 | AITER flydsl contiguous 606.6 |
| 2048x2048x4096 | bfloat16 | 64 | one-tile owner, n = 2 |  | 626.1 | 602.6 | 476.8 | 439.1 | 475.9 | 443.3 | flydsl | 840.1 | 607.6 (copy) | 744.5 | 732.2 | AITER flydsl contiguous 840.1 |
| 2048x4096x4096 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1006.7 | 1013.5 | 886.8 | 859.9 | 888.6 | 869.9 | opus | 1034.6 | 1039.1 | 720.0 | 715.6 | AITER opus padded 1039.1 |
| 2048x8192x4096 | bfloat16 | 256 | data-parallel | yes | 1407.0 | 1447.3 | 1423.3 | 1425.3 | 1431.3 | 1454.5 | flydsl | 1391.1 | 1055.8 (copy) | 1389.5 | 1399.2 | hipBLASLt padded 1399.2 |
| 2048x8704x4096 | bfloat16 | 272 | one-tile reduce-scatter, n = 8 |  | 1206.9 | 1222.4 | 990.1 | 989.6 | 980.6 | 984.8 | torch | 1082.3 | 1071.6 | 1084.1 | 1073.2 | hipBLASLt contiguous 1084.1 |
| 2048x10240x4096 | bfloat16 | 320 | one-tile owner, n = 2 |  | 1199.6 | 1214.5 | 1112.3 | 1095.5 | 1115.2 | 1090.8 | torch | 1156.1 | 1141.7 | 1164.2 | 1140.4 | hipBLASLt contiguous 1164.2 |
| 2048x16384x4096 | bfloat16 | 512 | data-parallel |  | 1447.1 | 1478.8 | 1447.1 | 1478.8 | 1454.1 | 1474.0 | flydsl | 1402.0 | 1118.6 (copy) | 1400.5 | 1398.1 | AITER flydsl contiguous 1402.0 |
| 2048x129280x4096 | bfloat16 | 4040 | data-parallel |  | 1508.5 | 1517.6 | 1508.5 | 1517.6 | 1483.8 | 1483.8 | flydsl | 1430.3 | 1155.1 (copy) | 1375.3 | 1368.7 | AITER flydsl contiguous 1430.3 |
| 2304x512x4096 | bfloat16 | 18 | one-tile reduce-scatter, n = 7 |  | 263.4 | 259.0 | 134.7 | 121.0 | 136.2 | 122.1 | flydsl | 353.6 | 261.0 (copy) | 442.8 | 431.5 | hipBLASLt contiguous 442.8 |
| 2304x1024x4096 | bfloat16 | 36 | one-tile owner, n = 3 |  | 410.4 | 384.2 | 272.0 | 243.8 | 274.3 | 246.1 | flydsl | 509.7 | 419.8 (copy) | 647.1 | 650.6 | hipBLASLt padded 650.6 |
| 2304x2048x4096 | bfloat16 | 72 | one-tile owner, n = 2 |  | 678.6 | 656.5 | 529.3 | 487.1 | 530.3 | 490.3 | opus | 684.9 | 648.0 | 900.9 | 876.1 | hipBLASLt contiguous 900.9 |
| 2560x512x4096 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 |  | 291.1 | 279.7 | 149.9 | 134.3 | 151.0 | 135.0 | flydsl | 386.7 | 283.7 (copy) | 486.2 | 470.8 | hipBLASLt contiguous 486.2 |
| 2560x1024x4096 | bfloat16 | 40 | one-tile owner, n = 3 |  | 454.9 | 421.7 | 301.7 | 271.4 | 303.5 | 273.1 | flydsl | 557.8 | 452.2 (copy) | 608.2 | 598.7 | hipBLASLt contiguous 608.2 |
| 2560x2048x4096 | bfloat16 | 80 | one-tile owner, n = 2 |  | 738.8 | 721.6 | 583.1 | 538.3 | 586.3 | 543.6 | opus | 766.7 | 720.3 | 886.8 | 892.7 | hipBLASLt padded 892.7 |
| 2816x512x4096 | bfloat16 | 22 | one-tile reduce-scatter, n = 5 |  | 302.0 | 290.0 | 164.3 | 147.3 | 165.6 | 148.2 | flydsl | 421.1 | 300.8 (copy) | 469.6 | 450.5 | hipBLASLt contiguous 469.6 |
| 2816x1024x4096 | bfloat16 | 44 | one-tile owner, n = 3 |  | 492.7 | 461.8 | 332.4 | 297.9 | 333.5 | 301.1 | flydsl | 601.6 | 476.7 (copy) | 644.5 | 632.4 | hipBLASLt contiguous 644.5 |
| 2816x2048x4096 | bfloat16 | 88 | one-tile owner, n = 2-3 |  | 775.2 | 771.0 | 634.2 | 588.2 | 638.4 | 594.1 | opus | 816.9 | 771.2 | 858.7 | 826.4 | hipBLASLt contiguous 858.7 |
| 3072x512x4096 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 |  | 323.7 | 311.5 | 179.7 | 161.0 | 181.5 | 161.5 | flydsl | 453.8 | 319.6 (copy) | 502.2 | 491.2 | hipBLASLt contiguous 502.2 |
| 3072x1024x4096 | bfloat16 | 48 | one-tile owner, n = 3 |  | 532.7 | 499.2 | 361.9 | 325.5 | 364.9 | 328.0 | flydsl | 646.4 | 495.6 (copy) | 674.8 | 673.9 | hipBLASLt contiguous 674.8 |
| 3072x2048x4096 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 758.9 | 753.0 | 696.1 | 647.0 | 692.5 | 655.8 | opus | 917.3 | 931.1 | 927.2 | 890.4 | AITER opus padded 931.1 |
| 3328x512x4096 | bfloat16 | 26 | one-tile reduce-scatter, n = 4 |  | 345.6 | 330.3 | 194.8 | 175.3 | 196.7 | 175.9 | flydsl | 483.4 | 337.0 (copy) | 447.0 | 438.6 | AITER flydsl contiguous 483.4 |
| 3328x1024x4096 | bfloat16 | 52 | one-tile owner, n = 2 |  | 515.6 | 484.5 | 387.8 | 350.4 | 393.0 | 354.8 | flydsl | 680.6 | 510.1 (copy) | 629.3 | 615.9 | AITER flydsl contiguous 680.6 |
| 3328x2048x4096 | bfloat16 | 104 | one-tile owner, n = 2-3 |  | 821.3 | 819.5 | 740.2 | 695.9 | 742.7 | 702.9 | opus | 939.6 | 891.1 | 824.0 | 790.8 | AITER opus contiguous 939.6 |
| 3584x512x4096 | bfloat16 | 28 | one-tile reduce-scatter, n = 4 |  | 364.0 | 348.0 | 209.9 | 188.7 | 212.7 | 189.4 | flydsl | 509.6 | 355.1 (copy) | 452.5 | 444.5 | AITER flydsl contiguous 509.6 |
| 3584x1024x4096 | bfloat16 | 56 | one-tile owner, n = 2 |  | 549.4 | 520.0 | 416.6 | 375.8 | 420.2 | 379.7 | flydsl | 711.7 | 523.3 (copy) | 667.6 | 652.1 | AITER flydsl contiguous 711.7 |
| 3584x2048x4096 | bfloat16 | 112 | one-tile owner, n = 2-3 |  | 907.8 | 902.0 | 787.5 | 745.8 | 792.5 | 757.0 | opus | 1005.0 | 966.3 | 911.1 | 858.1 | AITER opus contiguous 1005.0 |
| 3840x512x4096 | bfloat16 | 30 | one-tile reduce-scatter, n = 4 |  | 380.9 | 366.5 | 225.4 | 202.8 | 228.2 | 204.0 | flydsl | 537.3 | 369.2 (copy) | 488.5 | 482.0 | AITER flydsl contiguous 537.3 |
| 3840x1024x4096 | bfloat16 | 60 | one-tile owner, n = 2 |  | 582.7 | 556.8 | 444.2 | 408.9 | 448.0 | 409.2 | flydsl | 737.8 | 537.1 (copy) | 701.6 | 687.6 | AITER flydsl contiguous 737.8 |
| 3840x2048x4096 | bfloat16 | 120 | one-tile owner, n = 2-3 |  | 966.9 | 962.0 | 832.4 | 801.6 | 845.0 | 810.2 | opus | 1038.2 | 1000.8 | 806.2 | 754.5 | AITER opus contiguous 1038.2 |
| 4096x512x4096 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 402.6 | 388.1 | 238.5 | 216.8 | 241.1 | 217.4 | flydsl | 590.5 | 392.2 (copy) | 506.6 | 500.2 | AITER flydsl contiguous 590.5 |
| 4096x1024x4096 | bfloat16 | 64 | one-tile owner, n = 2 |  | 617.7 | 595.2 | 470.0 | 436.9 | 472.6 | 434.8 | flydsl | 830.6 | 587.3 (copy) | 738.6 | 731.0 | AITER flydsl contiguous 830.6 |
| 4096x2048x4096 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1000.5 | 1017.6 | 882.0 | 858.3 | 887.1 | 861.4 | opus | 1091.5 | 1072.0 | 724.4 | 714.8 | AITER opus contiguous 1091.5 |
| 4096x4096x4096 | bfloat16 | 256 | data-parallel | yes | 1405.5 | 1443.3 | 1416.8 | 1437.3 | 1435.6 | 1447.3 | flydsl | 1383.7 | 1107.0 (copy) | 1392.6 | 1385.8 | hipBLASLt contiguous 1392.6 |
| 4096x8192x4096 | bfloat16 | 512 | data-parallel |  | 1469.0 | 1488.9 | 1469.0 | 1488.9 | 1472.8 | 1486.1 | asm | 1443.1 | 1454.5 | 1429.8 | 1433.0 | AITER asm padded 1454.5 |
| 4096x8704x4096 | bfloat16 | 544 | one-tile reduce-scatter, n = 4 |  | 1349.2 | 1346.0 | 1244.8 | 1217.8 | 1228.7 | 1211.1 | torch | 1210.2 | 1214.2 | 1216.0 | 1209.9 | hipBLASLt contiguous 1216.0 |
| 4096x10240x4096 | bfloat16 | 640 | one-tile owner, n = 2 |  | 1396.9 | 1400.5 | 1376.3 | 1362.7 | 1354.1 | 1354.3 | flydsl | 1314.3 | 1144.0 (copy) | 1276.1 | 1250.8 | AITER flydsl contiguous 1314.3 |
| 4096x16384x4096 | bfloat16 | 1024 | data-parallel |  | 1497.8 | 1510.1 | 1497.8 | 1510.1 | 1484.6 | 1498.0 | flydsl | 1456.0 | 1255.3 (copy) | 1454.8 | 1451.3 | AITER flydsl contiguous 1456.0 |
| 4096x129280x4096 | bfloat16 | 8080 | data-parallel |  | 1532.1 | 1539.4 | 1532.1 | 1539.4 | 1502.3 | 1501.0 | asm | 1464.8 | 1468.9 | 1396.4 | 1378.5 | AITER asm padded 1468.9 |
| 4352x512x4096 | bfloat16 | 34 | one-tile owner, n = 3 |  | 386.8 | 356.4 | 251.8 | 227.4 | 254.0 | 227.0 | flydsl | 468.8 | 339.9 (copy) | 523.9 | 514.1 | hipBLASLt contiguous 523.9 |
| 4352x1024x4096 | bfloat16 | 68 | one-tile owner, n = 2 |  | 640.3 | 618.8 | 496.5 | 458.4 | 499.5 | 457.9 | opus | 642.9 | 605.5 | 814.5 | 799.9 | hipBLASLt contiguous 814.5 |
| 4352x2048x4096 | bfloat16 | 136 | data-parallel | yes | 945.2 | 908.9 | 958.7 | 902.6 | 979.3 | 906.7 | torch | 1040.2 | 1043.1 | 1049.4 | 1045.9 | hipBLASLt contiguous 1049.4 |
| 5120x512x4096 | bfloat16 | 40 | one-tile owner, n = 3 |  | 445.0 | 413.9 | 297.5 | 267.4 | 299.5 | 268.1 | flydsl | 536.6 | 368.6 (copy) | 599.3 | 587.9 | hipBLASLt contiguous 599.3 |
| 5120x1024x4096 | bfloat16 | 80 | one-tile owner, n = 2 |  | 732.3 | 707.8 | 575.3 | 531.3 | 581.1 | 534.5 | opus | 767.6 | 727.4 | 881.1 | 882.2 | hipBLASLt padded 882.2 |
| 5120x2048x4096 | bfloat16 | 160 | data-parallel | yes | 1059.7 | 1048.0 | 1087.9 | 1038.0 | 1104.3 | 1055.1 | torch | 1170.7 | 1147.3 | 1177.8 | 1153.9 | hipBLASLt contiguous 1177.8 |
| 6144x512x4096 | bfloat16 | 48 | one-tile owner, n = 3 |  | 517.7 | 489.2 | 357.6 | 320.7 | 360.9 | 322.6 | flydsl | 689.4 | 438.4 (copy) | 664.7 | 661.9 | AITER flydsl contiguous 689.4 |
| 6144x1024x4096 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 750.5 | 742.1 | 683.8 | 634.6 | 687.6 | 637.6 | opus | 926.8 | 936.3 | 921.8 | 887.8 | AITER opus padded 936.3 |
| 6144x2048x4096 | bfloat16 | 192 | data-parallel | yes | 1200.2 | 1204.1 | 1217.8 | 1194.8 | 1228.8 | 1212.9 | opus | 1295.3 | 1266.8 | 1233.8 | 1233.2 | AITER opus contiguous 1295.3 |
| 6144x129280x4096 | bfloat16 | 12120 | one-tile owner, n = 2-3 |  | 1498.4 | 1517.4 | 1514.9 | 1514.1 | 1482.3 | 1477.2 | asm | 1473.2 | 1477.3 | 1389.0 | 1376.2 | AITER asm padded 1477.3 |
| 6400x512x4096 | bfloat16 | 50 | one-tile owner, n = 3 |  | 532.8 | 505.4 | 369.2 | 334.0 | 373.7 | 336.2 | flydsl | 609.7 | 400.5 (copy) | 608.2 | 589.4 | AITER flydsl contiguous 609.7 |
| 6400x1024x4096 | bfloat16 | 100 | one-tile owner, n = 2-3 |  | 778.2 | 768.0 | 705.1 | 657.7 | 712.0 | 664.0 | opus | 886.7 | 846.3 | 787.6 | 781.8 | AITER opus contiguous 886.7 |
| 6400x2048x4096 | bfloat16 | 200 | data-parallel | yes | 1219.0 | 1233.3 | 1240.5 | 1220.8 | 1252.2 | 1241.0 | torch | 1108.7 | 1095.9 | 1117.2 | 1096.3 | hipBLASLt contiguous 1117.2 |
| 6912x512x4096 | bfloat16 | 54 | one-tile owner, n = 2 |  | 525.8 | 493.5 | 396.2 | 359.2 | 400.3 | 362.4 | flydsl | 632.8 | 365.2 (copy) | 645.7 | 631.3 | hipBLASLt contiguous 645.7 |
| 6912x1024x4096 | bfloat16 | 108 | one-tile owner, n = 2-3 |  | 857.7 | 831.7 | 754.6 | 710.5 | 764.4 | 714.6 | opus | 937.1 | 899.6 | 713.7 | 711.4 | AITER opus contiguous 937.1 |
| 6912x2048x4096 | bfloat16 | 216 | data-parallel | yes | 1279.0 | 1295.5 | 1288.9 | 1283.2 | 1305.0 | 1303.1 | torch | 1241.0 | 1240.1 | 1256.2 | 1242.5 | hipBLASLt contiguous 1256.2 |
| 7168x512x4096 | bfloat16 | 56 | one-tile owner, n = 2 |  | 543.0 | 510.4 | 410.4 | 371.0 | 416.0 | 374.3 | flydsl | 638.1 | 422.9 (copy) | 665.6 | 649.0 | hipBLASLt contiguous 665.6 |
| 7168x1024x4096 | bfloat16 | 112 | one-tile owner, n = 2-3 |  | 885.4 | 869.6 | 778.6 | 733.0 | 789.4 | 737.0 | opus | 1004.0 | 968.6 | 1063.3 | 1052.7 | hipBLASLt contiguous 1063.3 |
| 7168x2048x4096 | bfloat16 | 224 | data-parallel | yes | 1307.7 | 1324.1 | 1316.1 | 1311.6 | 1330.9 | 1330.6 | torch | 1272.4 | 1275.7 | 1282.3 | 1280.5 | hipBLASLt contiguous 1282.3 |
| 7424x512x4096 | bfloat16 | 58 | one-tile owner, n = 2 |  | 558.3 | 526.1 | 425.2 | 387.9 | 426.8 | 388.9 | flydsl | 640.9 | 430.5 (copy) | 661.6 | 658.8 | hipBLASLt contiguous 661.6 |
| 7424x1024x4096 | bfloat16 | 116 | one-tile owner, n = 2-3 |  | 900.6 | 899.7 | 805.4 | 760.3 | 811.9 | 773.1 | opus | 983.8 | 942.7 | 766.2 | 756.8 | AITER opus contiguous 983.8 |
| 7424x2048x4096 | bfloat16 | 232 | data-parallel | yes | 1337.6 | 1346.9 | 1345.0 | 1339.4 | 1362.6 | 1358.8 | torch | 1132.4 | 1118.7 | 1137.5 | 1112.0 | hipBLASLt contiguous 1137.5 |
| 7680x512x4096 | bfloat16 | 60 | one-tile owner, n = 2 |  | 575.3 | 542.8 | 439.9 | 403.0 | 441.8 | 402.9 | flydsl | 665.8 | 464.0 (copy) | 680.0 | 674.0 | hipBLASLt contiguous 680.0 |
| 7680x1024x4096 | bfloat16 | 120 | one-tile owner, n = 2-3 |  | 943.9 | 941.8 | 833.7 | 784.3 | 839.0 | 797.0 | opus | 1036.1 | 1003.8 | 780.7 | 750.4 | AITER opus contiguous 1036.1 |
| 7680x2048x4096 | bfloat16 | 240 | data-parallel | yes | 1356.3 | 1377.9 | 1370.4 | 1366.2 | 1388.1 | 1385.8 | torch | 1160.0 | 1149.8 | 1167.5 | 1142.5 | hipBLASLt contiguous 1167.5 |
| 7936x512x4096 | bfloat16 | 62 | one-tile owner, n = 2 |  | 590.3 | 565.0 | 454.6 | 414.2 | 457.0 | 417.9 | flydsl | 662.2 | 411.4 (copy) | 695.2 | 677.6 | hipBLASLt contiguous 695.2 |
| 7936x1024x4096 | bfloat16 | 124 | one-tile owner, n = 2-3 |  | 962.4 | 970.6 | 856.5 | 809.1 | 861.2 | 821.5 | opus | 1024.6 | 992.7 | 727.3 | 718.9 | AITER opus contiguous 1024.6 |
| 7936x2048x4096 | bfloat16 | 248 | data-parallel | yes | 1381.4 | 1407.6 | 1392.9 | 1394.2 | 1414.4 | 1418.5 | torch | 1372.4 | 1374.4 | 1375.9 | 1371.7 | hipBLASLt contiguous 1375.9 |
| 8192x512x4096 | bfloat16 | 64 | one-tile owner, n = 2 |  | 604.6 | 583.5 | 468.0 | 427.7 | 467.1 | 429.7 | flydsl | 784.0 | 507.1 (copy) | 713.0 | 708.1 | AITER flydsl contiguous 784.0 |
| 8192x1024x4096 | bfloat16 | 128 | one-tile owner, n = 2 |  | 967.9 | 1002.4 | 873.3 | 844.9 | 878.7 | 855.0 | opus | 1077.8 | 1067.2 | 716.9 | 700.6 | AITER opus contiguous 1077.8 |
| 8192x2048x4096 | bfloat16 | 256 | data-parallel | yes | 1409.4 | 1440.2 | 1422.6 | 1425.5 | 1438.9 | 1448.5 | torch | 1396.3 | 1396.1 | 1404.8 | 1396.0 | hipBLASLt contiguous 1404.8 |
| 8192x4096x4096 | bfloat16 | 512 | data-parallel |  | 1465.3 | 1488.7 | 1465.3 | 1488.7 | 1471.7 | 1484.6 | flydsl | 1421.9 | 1211.2 (copy) | 1450.0 | 1442.2 | hipBLASLt contiguous 1450.0 |
| 8192x8192x4096 | bfloat16 | 1024 | data-parallel |  | 1515.8 | 1512.1 | 1515.8 | 1512.1 | 1500.7 | 1501.7 | asm | 1468.7 | 1475.9 | 1472.3 | 1467.6 | AITER asm padded 1475.9 |
| 8192x8704x4096 | bfloat16 | 1088 | one-tile owner, n = 2 |  | 1438.5 | 1423.9 | 1410.5 | 1402.8 | 1377.3 | 1387.6 | asm | 1335.2 | 1336.0 | 1331.3 | 1326.1 | AITER asm padded 1336.0 |
| 8192x10240x4096 | bfloat16 | 1280 | data-parallel |  | 1525.2 | 1526.4 | 1525.2 | 1526.4 | 1508.1 | 1508.0 | torch | 1458.2 | 1458.3 | 1469.5 | 1461.3 | hipBLASLt contiguous 1469.5 |
| 8192x16384x4096 | bfloat16 | 2048 | data-parallel |  | 1519.2 | 1532.7 | 1519.2 | 1532.7 | 1497.5 | 1512.3 | torch | 1458.0 | 1462.2 | 1471.0 | 1460.6 | hipBLASLt contiguous 1471.0 |
| 8192x129280x4096 | bfloat16 | 16160 | one-tile reduce-scatter, n = 4 |  | 1555.1 | 1563.6 | 1547.8 | 1551.8 | 1516.6 | 1516.0 | flydsl | 1469.7 | 1379.7 (copy) | 1407.1 | 1374.9 | AITER flydsl contiguous 1469.7 |
| 10240x512x4096 | bfloat16 | 80 | one-tile owner, n = 2 |  | 714.9 | 692.3 | 572.2 | 523.2 | 575.4 | 526.4 | torch | 854.3 | 861.6 | 857.9 | 862.7 | hipBLASLt padded 862.7 |
| 10240x1024x4096 | bfloat16 | 160 | data-parallel | yes | 1064.2 | 1026.8 | 1076.7 | 1018.5 | 1100.8 | 1038.7 | torch | 1143.8 | 1133.9 | 1155.9 | 1134.9 | hipBLASLt contiguous 1155.9 |
| 10240x2048x4096 | bfloat16 | 320 | one-tile owner, n = 2 |  | 1207.3 | 1218.5 | 1126.5 | 1128.5 | 1116.7 | 1126.4 | torch | 1204.8 | 1199.8 | 1213.2 | 1198.1 | hipBLASLt contiguous 1213.2 |
| 10240x129280x4096 | bfloat16 | 20200 | data-parallel |  | 1488.2 | 1530.9 | 1488.2 | 1530.9 | 1498.0 | 1486.0 | asm | 1486.9 | 1490.1 | 1430.2 | 1380.5 | AITER asm padded 1490.1 |
| 12288x512x4096 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 728.8 | 718.5 | 666.1 | 620.4 | 676.5 | 623.9 | opus | 895.5 | 907.1 | 891.2 | 851.7 | AITER opus padded 907.1 |
| 12288x1024x4096 | bfloat16 | 192 | data-parallel | yes | 1174.5 | 1183.3 | 1184.7 | 1172.0 | 1205.1 | 1191.2 | opus | 1274.4 | 1249.8 | 1214.9 | 1223.1 | AITER opus contiguous 1274.4 |
| 12288x2048x4096 | bfloat16 | 384 | one-tile owner, n = 2 |  | 1303.7 | 1311.4 | 1271.8 | 1271.6 | 1266.2 | 1264.8 | opus | 1308.1 | 1301.2 | 1253.4 | 1266.1 | AITER opus contiguous 1308.1 |
| 12288x129280x4096 | bfloat16 | 24240 | data-parallel |  | 1505.8 | 1552.9 | 1505.8 | 1552.9 | 1510.5 | 1506.7 | asm | 1489.8 | 1486.8 | 1430.5 | 1363.5 | AITER asm contiguous 1489.8 |
| 14336x512x4096 | bfloat16 | 112 | one-tile owner, n = 2-3 |  | 870.3 | 845.9 | 764.3 | 712.6 | 770.9 | 719.7 | opus | 965.0 | 943.4 | 723.5 | 654.0 | AITER opus contiguous 965.0 |
| 14336x1024x4096 | bfloat16 | 224 | data-parallel | yes | 1281.1 | 1303.8 | 1289.8 | 1293.4 | 1303.3 | 1311.3 | torch | 1237.3 | 1241.7 | 1244.6 | 1261.7 | hipBLASLt padded 1261.7 |
| 14336x2048x4096 | bfloat16 | 448 | data-parallel |  | 1351.0 | 1362.9 | 1351.0 | 1362.9 | 1354.9 | 1355.9 | flydsl | 1318.4 | 1056.0 (copy) | 1292.5 | 1300.0 | AITER flydsl contiguous 1318.4 |
| 14336x129280x4096 | bfloat16 | 28280 | one-tile owner, n = 2-3 |  | 1477.7 | 1472.9 | 1477.5 | 1447.0 | 1490.6 | 1487.7 | asm | 1481.2 | 1493.8 | 1431.4 | 1405.9 | AITER asm padded 1493.8 |
| 16384x512x4096 | bfloat16 | 128 | one-tile owner, n = 2 |  | 932.5 | 966.2 | 848.8 | 805.8 | 854.3 | 818.1 | opus | 1020.8 | 1020.3 | 821.6 | 790.2 | AITER opus contiguous 1020.8 |
| 16384x1024x4096 | bfloat16 | 256 | data-parallel | yes | 1386.5 | 1400.7 | 1383.0 | 1401.5 | 1401.1 | 1408.1 | flydsl | 1344.0 | 901.3 (copy) | 1326.7 | 1309.3 | AITER flydsl contiguous 1344.0 |
| 16384x2048x4096 | bfloat16 | 512 | data-parallel |  | 1445.4 | 1474.6 | 1445.4 | 1474.6 | 1444.9 | 1467.2 | torch | 1401.7 | 1408.3 | 1415.3 | 1396.0 | hipBLASLt contiguous 1415.3 |
| 16384x16384x4096 | bfloat16 | 4096 | data-parallel |  | 1540.0 | 1551.2 | 1540.0 | 1551.2 | 1509.5 | 1516.7 | torch | 1484.6 | 1482.4 | 1499.9 | 1481.4 | hipBLASLt contiguous 1499.9 |
| 16384x129280x4096 | bfloat16 | 32320 | one-tile owner, n = 2 |  | 1492.2 | 1567.2 | 1512.1 | 1546.0 | 1526.0 | 1528.0 | asm | 1481.6 | 1487.2 | 1419.8 | 1389.1 | AITER asm padded 1487.2 |
| 32768x16384x4096 | bfloat16 | 8192 | data-parallel |  | 1553.8 | 1553.5 | 1553.8 | 1553.5 | 1519.3 | 1518.4 | flydsl | 1484.5 | 1406.4 (copy) | 1522.3 | 1499.3 | hipBLASLt contiguous 1522.3 |
| 512x7168x4224 | bfloat16 | 56 | one-tile owner, n = 2 |  | 530.7 | 533.1 | 374.1 | 374.1 | 376.3 | 376.3 | flydsl | 724.7 | 726.5 | 722.8 | 723.4 | AITER flydsl padded 726.5 |
| 1024x7168x4224 | bfloat16 | 112 | one-tile owner, n = 2-3 |  | 888.7 | 902.7 | 745.0 | 744.3 | 742.9 | 751.3 | flydsl | 932.1 | 935.5 | 816.6 | 815.8 | AITER flydsl padded 935.5 |
| 10240x7168x4224 | bfloat16 | 1120 | one-tile owner, n = 2-3 |  | 1449.5 | 1459.5 | 1427.8 | 1428.7 | 1406.8 | 1413.6 | opus | 910.6 | 917.8 | 1338.0 | 1328.8 | hipBLASLt contiguous 1338.0 |
| 512x6144x4608 | bfloat16 | 48 | one-tile owner, n = 3 |  | 538.3 | 556.3 | 335.4 | 342.7 | 338.4 | 345.9 | flydsl | 686.2 | 453.8 (copy) | 615.9 | 613.9 | AITER flydsl contiguous 686.2 |
| 1024x6144x4608 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 768.4 | 775.9 | 657.5 | 680.0 | 665.0 | 681.5 | torch | 898.9 | 879.4 | 903.3 | 881.0 | hipBLASLt contiguous 903.3 |
| 2048x6144x4608 | bfloat16 | 192 | data-parallel | yes | 1218.8 | 1248.9 | 1223.2 | 1226.8 | 1247.8 | 1256.1 | flydsl | 1190.0 | 948.4 (copy) | 1240.3 | 1250.0 | hipBLASLt padded 1250.0 |
| 4096x6144x4608 | bfloat16 | 384 | one-tile owner, n = 2 |  | 1343.9 | 1360.7 | 1287.2 | 1291.2 | 1273.8 | 1288.0 | torch | 1265.1 | 1280.6 | 1281.5 | 1286.3 | hipBLASLt padded 1286.3 |
| 8192x6144x4608 | bfloat16 | 768 | data-parallel |  | 1517.1 | 1532.6 | 1517.1 | 1532.6 | 1505.3 | 1528.0 | torch | 1453.8 | 1468.3 | 1473.1 | 1469.0 | hipBLASLt contiguous 1473.1 |
| 16384x6144x4608 | bfloat16 | 1536 | data-parallel |  | 1535.8 | 1553.8 | 1535.8 | 1553.8 | 1520.2 | 1537.7 | torch | 1468.6 | 1482.5 | 1487.8 | 1485.2 | hipBLASLt contiguous 1487.8 |
| 32768x6144x4608 | bfloat16 | 3072 | data-parallel |  | 1550.3 | 1570.4 | 1550.3 | 1570.4 | 1530.5 | 1542.6 | torch | 1479.5 | 1489.3 | 1493.0 | 1490.3 | hipBLASLt contiguous 1493.0 |
| 512x1024x5120 | float32 | 8 | one-tile reduce-scatter, n = 10 |  | 174.8 | 183.4 | 51.9 | 54.4 | 51.9 | 54.2 | asm | 248.9 | 253.7 | 275.3 | 278.6 | hipBLASLt padded 278.6 |
| 512x5120x5120 | bfloat16 | 40 | one-tile owner, n = 3 |  | 477.5 | 499.9 | 270.1 | 282.3 | 269.0 | 283.5 | torch | 642.0 | 668.5 | 644.8 | 668.5 | AITER torch padded 668.5 |
| 512x6400x5120 | bfloat16 | 50 | one-tile owner, n = 3 |  | 582.4 | 602.1 | 338.9 | 354.6 | 339.6 | 355.2 | flydsl | 612.5 | 433.0 (copy) | 565.9 | 577.0 | AITER flydsl contiguous 612.5 |
| 512x12800x5120 | bfloat16 | 100 | one-tile owner, n = 2-3 |  | 783.8 | 806.2 | 658.6 | 682.4 | 657.7 | 683.2 | flydsl | 659.7 | 428.9 (copy) | 831.2 | 836.4 | hipBLASLt padded 836.4 |
| 512x37888x5120 | bfloat16 | 296 | one-tile owner, n = 3 |  | 1140.7 | 1164.7 | 965.2 | 981.5 | 954.3 | 967.7 | torch | 1053.4 | 1079.9 | 1059.1 | 1080.0 | hipBLASLt padded 1080.0 |
| 512x51200x5120 | bfloat16 | 400 | data-parallel |  | 1200.7 | 1224.7 | 1200.7 | 1224.7 | 1197.9 | 1225.0 | flydsl | 1138.4 | 636.1 (copy) | 1100.3 | 1122.1 | AITER flydsl contiguous 1138.4 |
| 768x1024x5120 | float32 | 12 | one-tile reduce-scatter, n = 10 |  | 239.1 | 251.5 | 77.9 | 81.6 | 78.2 | 82.5 | asm | 313.1 | 323.2 | 312.6 | 315.8 | AITER asm padded 323.2 |
| 768x37888x5120 | bfloat16 | 444 | data-parallel |  | 1320.4 | 1341.2 | 1320.4 | 1341.2 | 1319.4 | 1339.5 | torch | 1223.9 | 1237.6 | 1234.5 | 1241.4 | hipBLASLt padded 1241.4 |
| 1024x1024x5120 | float32 | 16 | one-tile reduce-scatter, n = 8 |  | 295.3 | 313.2 | 105.1 | 109.9 | 105.6 | 109.9 | asm | 334.0 | 338.2 | 402.2 | 405.9 | hipBLASLt padded 405.9 |
| 1024x5120x5120 | bfloat16 | 80 | one-tile owner, n = 2 |  | 748.8 | 787.1 | 537.5 | 561.1 | 536.5 | 562.4 | torch | 846.2 | 890.9 | 854.8 | 891.5 | hipBLASLt padded 891.5 |
| 1024x6400x5120 | bfloat16 | 100 | one-tile owner, n = 2-3 |  | 809.8 | 854.1 | 670.2 | 697.0 | 669.1 | 698.6 | flydsl | 827.4 | 659.6 (copy) | 822.3 | 839.8 | hipBLASLt padded 839.8 |
| 1024x12800x5120 | bfloat16 | 200 | data-parallel | yes | 1216.1 | 1268.2 | 1224.9 | 1256.1 | 1232.8 | 1268.2 | flydsl | 722.6 | 555.6 (copy) | 1053.4 | 1077.3 | hipBLASLt padded 1077.3 |
| 1024x37888x5120 | bfloat16 | 592 | one-tile owner, n = 2 |  | 1324.1 | 1356.6 | 1263.9 | 1278.4 | 1248.9 | 1268.5 | torch | 1050.3 | 1079.8 | 1057.6 | 1081.2 | hipBLASLt padded 1081.2 |
| 1024x51200x5120 | bfloat16 | 800 | one-tile reduce-scatter, n = 4 |  | 1377.0 | 1392.6 | 1280.6 | 1301.1 | 1262.8 | 1287.4 | asm | 1217.4 | 1244.0 | 1065.9 | 1093.5 | AITER asm padded 1244.0 |
| 1536x1024x5120 | float32 | 24 | one-tile reduce-scatter, n = 5 |  | 371.5 | 374.9 | 158.0 | 164.9 | 157.6 | 165.7 | asm | 437.0 | 455.9 | 512.2 | 523.0 | hipBLASLt padded 523.0 |
| 1536x37888x5120 | bfloat16 | 888 | one-tile owner, n = 2-3 |  | 1389.4 | 1416.5 | 1353.6 | 1373.6 | 1340.9 | 1365.2 | torch | 1261.2 | 1286.5 | 1276.7 | 1288.9 | hipBLASLt padded 1288.9 |
| 2048x1024x5120 | float32 | 32 | one-tile reduce-scatter, n = 4 |  | 429.8 | 442.6 | 211.7 | 220.2 | 211.4 | 221.5 | opus | 428.9 | 429.1 | 549.1 | 610.7 | hipBLASLt padded 610.7 |
| 2048x5120x5120 | bfloat16 | 160 | data-parallel | yes | 1041.7 | 1114.8 | 1053.6 | 1103.2 | 1071.8 | 1119.9 | torch | 1148.0 | 1171.6 | 1146.6 | 1172.6 | hipBLASLt padded 1172.6 |
| 2048x6400x5120 | bfloat16 | 200 | data-parallel | yes | 1239.8 | 1286.3 | 1255.4 | 1274.0 | 1265.0 | 1286.7 | flydsl | 1132.9 | 923.3 (copy) | 1087.8 | 1100.7 | AITER flydsl contiguous 1132.9 |
| 2048x12800x5120 | bfloat16 | 400 | data-parallel |  | 1294.5 | 1313.3 | 1294.5 | 1313.3 | 1295.9 | 1308.9 | flydsl | 810.7 | 675.6 (copy) | 1077.9 | 1098.2 | hipBLASLt padded 1098.2 |
| 2048x37888x5120 | bfloat16 | 1184 | data-parallel |  | 1452.0 | 1472.2 | 1452.0 | 1472.2 | 1437.5 | 1451.2 | torch | 1199.7 | 1236.2 | 1215.5 | 1238.1 | hipBLASLt padded 1238.1 |
| 2048x51200x5120 | bfloat16 | 1600 | one-tile owner, n = 2 |  | 1457.7 | 1481.1 | 1433.4 | 1448.6 | 1410.1 | 1432.0 | asm | 210.4 | 215.8 | 1191.2 | 1211.7 | hipBLASLt padded 1211.7 |
| 4096x1024x5120 | float32 | 64 | one-tile owner, n = 2 |  | 583.3 | 622.0 | 410.8 | 435.9 | 414.6 | 435.2 | opus | 714.8 | 727.1 | 769.3 | 802.7 | hipBLASLt padded 802.7 |
| 4096x5120x5120 | bfloat16 | 320 | one-tile owner, n = 2 |  | 1249.3 | 1277.7 | 1106.0 | 1135.9 | 1100.9 | 1129.4 | torch | 1172.6 | 1201.0 | 1183.5 | 1203.3 | hipBLASLt padded 1203.3 |
| 4096x6400x5120 | bfloat16 | 400 | data-parallel |  | 1304.2 | 1327.2 | 1304.2 | 1327.2 | 1305.7 | 1325.6 | flydsl | 1244.7 | 1089.6 (copy) | 1118.9 | 1147.8 | AITER flydsl contiguous 1244.7 |
| 4096x12800x5120 | bfloat16 | 800 | one-tile reduce-scatter, n = 4 |  | 1416.2 | 1438.4 | 1334.2 | 1349.9 | 1318.6 | 1338.4 | asm | 1282.3 | 1294.6 | 1174.0 | 1192.2 | AITER asm padded 1294.6 |
| 4096x51200x5120 | bfloat16 | 3200 | one-tile owner, n = 2 |  | 1521.9 | 1550.9 | 1525.7 | 1537.4 | 1487.9 | 1504.5 | asm | 1445.7 | 1465.4 | 1378.2 | 1397.7 | AITER asm padded 1465.4 |
| 8192x1024x5120 | float32 | 128 | one-tile owner, n = 2 |  | 995.9 | 1023.5 | 795.6 | 833.8 | 801.5 | 850.3 | opus | 1024.1 | 1057.9 | 985.5 | 1014.8 | AITER opus padded 1057.9 |
| 8192x5120x5120 | bfloat16 | 640 | one-tile owner, n = 2 |  | 1428.9 | 1451.4 | 1381.4 | 1389.8 | 1365.4 | 1381.7 | torch | 1293.1 | 1314.8 | 1308.9 | 1313.5 | AITER torch padded 1314.8 |
| 8192x6400x5120 | bfloat16 | 800 | one-tile reduce-scatter, n = 4 |  | 1419.6 | 1442.6 | 1334.6 | 1351.7 | 1316.9 | 1338.3 | flydsl | 1252.7 | 1144.9 (copy) | 1220.2 | 1230.1 | AITER flydsl contiguous 1252.7 |
| 8192x12800x5120 | bfloat16 | 1600 | one-tile owner, n = 2 |  | 1477.1 | 1504.3 | 1454.1 | 1468.6 | 1433.5 | 1458.6 | asm | 1373.1 | 1400.9 | 1189.8 | 1187.9 | AITER asm padded 1400.9 |
| 8192x51200x5120 | bfloat16 | 6400 | data-parallel |  | 1567.1 | 1589.8 | 1567.1 | 1589.8 | 1539.1 | 1557.0 | flydsl | 1483.1 | 1389.3 (copy) | 1413.4 | 1440.2 | AITER flydsl contiguous 1483.1 |
| 16384x1024x5120 | float32 | 256 | data-parallel |  | 1305.5 | 1331.6 | 1305.5 | 1331.6 | 1330.1 | 1343.9 | opus | 1217.1 | 1211.2 | 1239.3 | 1231.1 | hipBLASLt contiguous 1239.3 |
| 16384x5120x5120 | bfloat16 | 1280 | data-parallel |  | 1524.7 | 1550.3 | 1524.7 | 1550.3 | 1510.0 | 1534.7 | torch | 1468.6 | 1483.1 | 1486.4 | 1483.6 | hipBLASLt contiguous 1486.4 |
| 16384x6400x5120 | bfloat16 | 1600 | one-tile owner, n = 2 |  | 1464.1 | 1497.8 | 1446.3 | 1461.9 | 1416.2 | 1442.6 | flydsl | 1362.3 | 1253.2 (copy) | 1205.5 | 1206.7 | AITER flydsl contiguous 1362.3 |
| 16384x12800x5120 | bfloat16 | 3200 | one-tile owner, n = 2 |  | 1528.1 | 1557.0 | 1527.3 | 1541.3 | 1494.3 | 1512.5 | torch | 1457.2 | 1475.6 | 1472.7 | 1481.2 | hipBLASLt padded 1481.2 |
| 16384x51200x5120 | bfloat16 | 12800 | data-parallel |  | 1569.4 | 1555.6 | 1569.4 | 1555.6 | 1535.9 | 1554.4 | flydsl | 1485.3 | 1434.3 (copy) | 1411.8 | 1416.2 | AITER flydsl contiguous 1485.3 |
| 32768x5120x5120 | bfloat16 | 2560 | data-parallel |  | 1537.5 | 1567.9 | 1537.5 | 1567.9 | 1518.4 | 1537.4 | torch | 1483.0 | 1491.0 | 1488.6 | 1493.7 | hipBLASLt padded 1493.7 |
| 32768x6400x5120 | bfloat16 | 3200 | one-tile owner, n = 2 |  | 1521.0 | 1550.9 | 1520.8 | 1535.5 | 1482.2 | 1503.0 | torch | 1464.2 | 1473.3 | 1473.5 | 1473.7 | hipBLASLt padded 1473.7 |
| 32768x12800x5120 | bfloat16 | 6400 | data-parallel |  | 1556.0 | 1578.9 | 1556.0 | 1578.9 | 1531.3 | 1548.5 | torch | 1487.3 | 1528.2 | 1505.0 | 1531.0 | hipBLASLt padded 1531.0 |
| 32768x51200x5120 | bfloat16 | 25600 | data-parallel |  | 1563.2 | 1535.4 | 1563.2 | 1535.4 | 1549.0 | 1565.2 | flydsl | 1483.1 | 1450.4 (copy) | 1400.8 | 1454.8 | AITER flydsl contiguous 1483.1 |
| 512x1280x6144 | bfloat16 | 10 | one-tile reduce-scatter, n = 12 |  | 237.1 | 247.0 | 66.4 | 71.1 | 67.3 | 71.7 | flydsl | 340.5 | 235.1 (copy) | 350.8 | 352.4 | hipBLASLt padded 352.4 |
| 512x1536x6144 | bfloat16 | 12 | one-tile reduce-scatter, n = 10 |  | 251.5 | 276.3 | 79.8 | 85.5 | 81.0 | 85.9 | flydsl | 382.6 | 256.1 (copy) | 377.1 | 377.3 | AITER flydsl contiguous 382.6 |
| 512x2048x6144 | float32 | 16 | one-tile reduce-scatter, n = 8 |  | 329.6 | 341.8 | 103.8 | 111.4 | 105.4 | 112.0 | asm | 342.8 | 350.8 | 423.7 | 422.9 | hipBLASLt contiguous 423.7 |
| 512x2304x6144 | bfloat16 | 18 | one-tile reduce-scatter, n = 7 |  | 323.0 | 349.5 | 121.3 | 129.7 | 121.8 | 130.1 | flydsl | 442.1 | 311.1 (copy) | 375.6 | 370.9 | AITER flydsl contiguous 442.1 |
| 512x2560x6144 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 |  | 356.8 | 369.6 | 134.1 | 143.3 | 135.9 | 143.9 | torch | 475.2 | 479.9 | 476.0 | 480.6 | hipBLASLt padded 480.6 |
| 512x3072x6144 | float32 | 24 | one-tile reduce-scatter, n = 5 |  | 372.8 | 394.6 | 155.2 | 166.4 | 157.9 | 168.5 | opus | 404.6 | 432.1 | 547.6 | 540.9 | hipBLASLt contiguous 547.6 |
| 512x3072x6144 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 |  | 376.8 | 400.7 | 160.4 | 171.9 | 162.5 | 173.2 | flydsl | 544.6 | 363.3 (copy) | 462.0 | 469.0 | AITER flydsl contiguous 544.6 |
| 512x6144x6144 | bfloat16 | 48 | one-tile owner, n = 3 |  | 601.6 | 618.9 | 325.5 | 348.7 | 326.8 | 350.5 | flydsl | 687.1 | 481.6 (copy) | 684.9 | 674.0 | AITER flydsl contiguous 687.1 |
| 512x9216x6144 | bfloat16 | 72 | one-tile owner, n = 3-4 |  | 744.0 | 753.2 | 482.2 | 517.5 | 485.2 | 519.0 | flydsl | 615.5 | 407.4 (copy) | 603.5 | 625.0 | hipBLASLt padded 625.0 |
| 1024x2048x6144 | float32 | 32 | one-tile reduce-scatter, n = 4 |  | 465.3 | 479.5 | 208.7 | 223.1 | 212.4 | 225.4 | opus | 408.7 | 429.1 | 642.5 | 640.4 | hipBLASLt contiguous 642.5 |
| 1024x2304x6144 | bfloat16 | 36 | one-tile owner, n = 4 |  | 499.8 | 523.0 | 242.2 | 260.5 | 245.7 | 261.3 | torch | 604.4 | 614.6 | 606.0 | 618.8 | hipBLASLt padded 618.8 |
| 1024x2560x6144 | bfloat16 | 40 | one-tile owner, n = 3 |  | 516.4 | 528.4 | 269.2 | 288.1 | 272.8 | 290.7 | torch | 693.0 | 717.8 | 695.2 | 708.2 | AITER torch padded 717.8 |
| 1024x3072x6144 | float32 | 48 | one-tile owner, n = 3 |  | 572.8 | 588.1 | 311.8 | 334.1 | 316.1 | 337.5 | opus | 456.8 | 474.4 | 725.1 | 731.7 | hipBLASLt padded 731.7 |
| 1024x6144x6144 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 849.5 | 865.3 | 647.3 | 689.1 | 650.3 | 694.3 | torch | 1013.0 | 1034.2 | 1025.6 | 1034.6 | hipBLASLt padded 1034.6 |
| 1024x9216x6144 | bfloat16 | 144 | data-parallel | yes | 951.3 | 1023.6 | 945.2 | 1017.1 | 955.9 | 1029.6 | torch | 1065.8 | 1096.5 | 1077.7 | 1100.1 | hipBLASLt padded 1100.1 |
| 2048x2048x6144 | float32 | 64 | one-tile owner, n = 3 |  | 733.7 | 757.3 | 421.9 | 454.0 | 426.4 | 459.9 | opus | 675.3 | 690.6 | 816.9 | 837.1 | hipBLASLt padded 837.1 |
| 2048x2304x6144 | bfloat16 | 72 | one-tile owner, n = 3-4 |  | 830.1 | 843.2 | 493.7 | 528.5 | 500.3 | 532.7 | torch | 845.4 | 875.5 | 848.1 | 874.3 | AITER torch padded 875.5 |
| 2048x2560x6144 | bfloat16 | 80 | one-tile owner, n = 2 |  | 822.7 | 847.8 | 549.3 | 587.2 | 556.5 | 591.3 | torch | 910.6 | 933.6 | 916.0 | 937.4 | hipBLASLt padded 937.4 |
| 2048x3072x6144 | float32 | 96 | one-tile owner, n = 2-3 |  | 882.7 | 901.7 | 630.7 | 676.5 | 642.1 | 681.4 | opus | 690.7 | 718.2 | 944.0 | 952.9 | hipBLASLt padded 952.9 |
| 2048x6144x6144 | bfloat16 | 192 | data-parallel | yes | 1224.6 | 1288.0 | 1235.3 | 1274.1 | 1245.0 | 1288.1 | torch | 1262.3 | 1282.3 | 1276.6 | 1287.8 | hipBLASLt padded 1287.8 |
| 2048x9216x6144 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 |  | 1264.5 | 1288.4 | 1013.7 | 1048.0 | 1005.6 | 1048.0 | torch | 1099.9 | 1124.8 | 1106.5 | 1123.7 | AITER torch padded 1124.8 |
| 4096x2048x6144 | float32 | 128 | one-tile owner, n = 2 |  | 1087.5 | 1118.5 | 817.7 | 879.9 | 835.3 | 888.7 | opus | 1030.6 | 1078.2 | 832.9 | 812.0 | AITER opus padded 1078.2 |
| 4096x2304x6144 | bfloat16 | 144 | data-parallel | yes | 975.1 | 1047.8 | 965.9 | 1039.0 | 987.6 | 1047.5 | flydsl | 685.8 | 639.9 (copy) | 1093.8 | 1121.5 | hipBLASLt padded 1121.5 |
| 4096x2560x6144 | bfloat16 | 160 | data-parallel | yes | 1064.1 | 1145.9 | 1064.7 | 1136.7 | 1091.0 | 1147.7 | flydsl | 639.7 | 594.0 (copy) | 1177.9 | 1203.1 | hipBLASLt padded 1203.1 |
| 4096x3072x6144 | float32 | 192 | data-parallel |  | 1147.3 | 1217.1 | 1147.3 | 1217.1 | 1186.6 | 1234.4 | asm | 1144.6 | 1169.2 | 1241.7 | 1253.2 | hipBLASLt padded 1253.2 |
| 4096x6144x6144 | bfloat16 | 384 | one-tile owner, n = 2 |  | 1398.5 | 1420.5 | 1307.5 | 1326.4 | 1295.8 | 1323.6 | torch | 1293.4 | 1310.7 | 1309.5 | 1313.8 | hipBLASLt padded 1313.8 |
| 4096x9216x6144 | bfloat16 | 576 | one-tile owner, n = 3 |  | 1415.8 | 1439.3 | 1311.6 | 1330.9 | 1297.7 | 1332.3 | torch | 1298.5 | 1323.7 | 1313.2 | 1324.4 | hipBLASLt padded 1324.4 |
| 8192x2048x6144 | float32 | 256 | data-parallel |  | 1355.3 | 1406.7 | 1355.3 | 1406.7 | 1392.2 | 1430.8 | opus | 1275.7 | 1261.8 | 1294.0 | 1305.6 | hipBLASLt padded 1305.6 |
| 8192x2304x6144 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 |  | 1242.0 | 1273.9 | 1007.5 | 1045.2 | 1003.4 | 1042.0 | torch | 1100.2 | 1127.2 | 1108.9 | 1127.6 | hipBLASLt padded 1127.6 |
| 8192x2560x6144 | bfloat16 | 320 | one-tile owner, n = 3 |  | 1305.9 | 1319.4 | 1115.7 | 1142.7 | 1106.5 | 1140.6 | asm | 406.3 | 416.1 | 1202.1 | 1228.0 | hipBLASLt padded 1228.0 |
| 8192x3072x6144 | float32 | 384 | one-tile owner, n = 2 |  | 1303.4 | 1335.1 | 1203.5 | 1251.0 | 1214.8 | 1258.5 | asm | 1170.1 | 1189.7 | 1235.1 | 1233.3 | hipBLASLt contiguous 1235.1 |
| 8192x6144x6144 | bfloat16 | 768 | data-parallel |  | 1538.0 | 1562.4 | 1538.0 | 1562.4 | 1531.7 | 1558.2 | flydsl | 1481.0 | 1309.6 (copy) | 1501.0 | 1498.4 | hipBLASLt contiguous 1501.0 |
| 8192x9216x6144 | bfloat16 | 1152 | one-tile owner, n = 2 |  | 1510.8 | 1532.1 | 1468.3 | 1487.9 | 1447.1 | 1475.2 | asm | 1415.3 | 1438.7 | 1207.8 | 1207.0 | AITER asm padded 1438.7 |
| 16384x2048x6144 | float32 | 512 | data-parallel |  | 1396.8 | 1440.2 | 1396.8 | 1440.2 | 1411.0 | 1450.5 | opus | 1249.4 | 1251.5 | 1319.6 | 1326.9 | hipBLASLt padded 1326.9 |
| 16384x2304x6144 | bfloat16 | 576 | one-tile owner, n = 3 |  | 1385.2 | 1413.3 | 1287.2 | 1303.9 | 1280.3 | 1298.4 | flydsl | 970.6 | 832.8 (copy) | 1311.5 | 1313.2 | hipBLASLt padded 1313.2 |
| 16384x2560x6144 | bfloat16 | 640 | one-tile owner, n = 2 |  | 1434.7 | 1459.1 | 1363.6 | 1381.1 | 1353.1 | 1370.5 | flydsl | 1323.5 | 1088.6 (copy) | 1312.5 | 1314.0 | AITER flydsl contiguous 1323.5 |
| 16384x3072x6144 | float32 | 768 | data-parallel |  | 1412.2 | 1453.7 | 1412.2 | 1453.7 | 1419.1 | 1451.7 | asm | 1368.8 | 1390.0 | 1335.2 | 1335.7 | AITER asm padded 1390.0 |
| 16384x6144x6144 | bfloat16 | 1536 | data-parallel |  | 1548.2 | 1581.1 | 1548.2 | 1581.1 | 1543.8 | 1565.7 | asm | 1498.8 | 1522.7 | 1508.8 | 1512.7 | AITER asm padded 1522.7 |
| 16384x9216x6144 | bfloat16 | 2304 | data-parallel |  | 1553.8 | 1589.2 | 1553.8 | 1589.2 | 1546.2 | 1572.2 | torch | 1504.0 | 1510.0 | 1514.1 | 1509.7 | hipBLASLt contiguous 1514.1 |
| 32768x2304x6144 | bfloat16 | 1152 | one-tile owner, n = 2 |  | 1465.3 | 1489.5 | 1426.2 | 1449.9 | 1412.0 | 1434.4 | flydsl | 649.7 | 590.8 (copy) | 1199.7 | 1198.2 | hipBLASLt contiguous 1199.7 |
| 32768x2560x6144 | bfloat16 | 1280 | data-parallel |  | 1506.8 | 1540.9 | 1506.8 | 1540.9 | 1503.1 | 1526.8 | flydsl | 1446.9 | 1193.3 (copy) | 1482.3 | 1480.1 | hipBLASLt contiguous 1482.3 |
| 32768x6144x6144 | bfloat16 | 3072 | data-parallel |  | 1560.5 | 1594.0 | 1560.5 | 1594.0 | 1550.3 | 1568.4 | torch | 1501.5 | 1516.2 | 1508.6 | 1514.4 | AITER torch padded 1516.2 |
| 32768x9216x6144 | bfloat16 | 4608 | data-parallel |  | 1570.1 | 1596.3 | 1570.1 | 1596.3 | 1550.0 | 1571.0 | torch | 1523.8 | 1540.3 | 1534.1 | 1542.7 | hipBLASLt padded 1542.7 |
| 512x5120x6400 | bfloat16 | 40 | one-tile owner, n = 4 |  | 553.8 | 560.5 | 277.6 | 287.7 | 280.6 | 290.3 | flydsl | 542.8 | 429.8 (copy) | 692.2 | 696.7 | hipBLASLt padded 696.7 |
| 1024x5120x6400 | bfloat16 | 80 | one-tile owner, n = 2 |  | 811.0 | 839.3 | 554.6 | 574.4 | 559.6 | 577.8 | flydsl | 599.1 | 519.1 (copy) | 895.4 | 906.3 | hipBLASLt padded 906.3 |
| 2048x5120x6400 | bfloat16 | 160 | data-parallel | yes | 1097.8 | 1146.7 | 1099.4 | 1132.7 | 1111.9 | 1151.3 | flydsl | 741.6 | 644.7 (copy) | 1206.2 | 1199.6 | hipBLASLt contiguous 1206.2 |
| 4096x5120x6400 | bfloat16 | 320 | one-tile owner, n = 3 |  | 1345.2 | 1357.3 | 1132.2 | 1157.3 | 1138.6 | 1160.1 | torch | 1221.5 | 1230.2 | 1234.0 | 1226.4 | hipBLASLt contiguous 1234.0 |
| 8192x5120x6400 | bfloat16 | 640 | one-tile owner, n = 2 |  | 1475.1 | 1486.5 | 1406.9 | 1404.9 | 1390.2 | 1399.0 | flydsl | 1348.4 | 1184.7 (copy) | 1350.2 | 1343.9 | hipBLASLt contiguous 1350.2 |
| 16384x5120x6400 | bfloat16 | 1280 | data-parallel |  | 1549.0 | 1567.8 | 1549.0 | 1567.8 | 1541.8 | 1555.9 | torch | 1496.6 | 1505.6 | 1514.4 | 1509.7 | hipBLASLt contiguous 1514.4 |
| 32768x5120x6400 | bfloat16 | 2560 | data-parallel |  | 1563.9 | 1588.6 | 1563.9 | 1588.6 | 1552.6 | 1563.6 | torch | 1503.5 | 1513.4 | 1512.8 | 1517.2 | hipBLASLt padded 1517.2 |
| 512x16384x6656 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1097.8 | 1126.8 | 873.1 | 876.9 | 876.2 | 885.9 | flydsl | 839.0 | 513.2 (copy) | 917.9 | 884.0 | hipBLASLt contiguous 917.9 |
| 1024x16384x6656 | bfloat16 | 256 | data-parallel | yes | 1429.8 | 1480.6 | 1434.4 | 1471.9 | 1444.6 | 1479.4 | asm | 361.6 | 369.3 | 1392.0 | 1387.1 | hipBLASLt contiguous 1392.0 |
| 2048x16384x6656 | bfloat16 | 512 | data-parallel |  | 1520.8 | 1530.2 | 1520.8 | 1530.2 | 1511.7 | 1528.1 | asm | 1460.1 | 1463.0 | 1446.4 | 1447.9 | AITER asm padded 1463.0 |
| 4096x16384x6656 | bfloat16 | 1024 | data-parallel |  | 1551.2 | 1565.2 | 1551.2 | 1565.2 | 1538.7 | 1558.4 | torch | 1472.9 | 1484.0 | 1486.1 | 1481.8 | hipBLASLt contiguous 1486.1 |
| 8192x16384x6656 | bfloat16 | 2048 | data-parallel |  | 1576.3 | 1593.7 | 1576.3 | 1593.7 | 1563.3 | 1579.8 | asm | 1513.5 | 1521.9 | 1509.7 | 1512.8 | AITER asm padded 1521.9 |
| 16384x16384x6656 | bfloat16 | 4096 | data-parallel |  | 1582.3 | 1609.4 | 1582.3 | 1609.4 | 1567.6 | 1581.4 | asm | 1528.0 | 1534.8 | 1548.0 | 1542.6 | hipBLASLt contiguous 1548.0 |
| 32768x16384x6656 | bfloat16 | 8192 | data-parallel |  | 1597.0 | 1612.0 | 1597.0 | 1612.0 | 1577.9 | 1585.6 | asm | 291.4 | 292.3 | 1563.3 | 1556.6 | hipBLASLt contiguous 1563.3 |
| 512x1024x7168 | bfloat16 | 8 | one-tile reduce-scatter, n = 14 |  | 229.1 | 221.0 | 55.7 | 54.6 | 57.0 | 55.0 | torch | 386.5 | 387.6 | 386.2 | 386.6 | AITER torch padded 387.6 |
| 512x1536x7168 | bfloat16 | 12 | one-tile reduce-scatter, n = 10 |  | 285.7 | 289.7 | 83.8 | 82.0 | 85.9 | 83.1 | torch | 410.3 | 403.9 | 411.0 | 404.0 | hipBLASLt contiguous 411.0 |
| 512x2048x7168 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 |  | 372.5 | 364.5 | 112.3 | 109.7 | 114.1 | 110.4 | torch | 547.4 | 538.4 | 549.2 | 538.1 | hipBLASLt contiguous 549.2 |
| 512x3584x7168 | bfloat16 | 28 | one-tile reduce-scatter, n = 4 |  | 453.8 | 437.5 | 198.8 | 194.1 | 202.1 | 195.6 | flydsl | 560.7 | 385.1 (copy) | 538.5 | 532.6 | AITER flydsl contiguous 560.7 |
| 512x6144x7168 | bfloat16 | 48 | one-tile owner, n = 3 |  | 634.0 | 631.3 | 342.6 | 333.8 | 348.7 | 336.7 | flydsl | 718.1 | 486.6 (copy) | 716.1 | 689.7 | AITER flydsl contiguous 718.1 |
| 512x8192x7168 | bfloat16 | 64 | one-tile owner, n = 3 |  | 801.5 | 795.0 | 453.5 | 441.8 | 459.6 | 445.1 | flydsl | 705.3 | 435.8 (copy) | 739.2 | 729.8 | hipBLASLt contiguous 739.2 |
| 512x8448x7168 | bfloat16 | 66 | one-tile owner, n = 3-4 |  | 734.7 | 710.4 | 468.9 | 455.6 | 474.5 | 459.2 | flydsl | 670.6 | 429.4 (copy) | 469.8 | 464.1 | AITER flydsl contiguous 670.6 |
| 512x20480x7168 | bfloat16 | 160 | data-parallel | yes | 1054.3 | 1033.0 | 1045.3 | 1024.0 | 1067.0 | 1038.7 | flydsl | 1030.2 | 581.7 (copy) | 1156.7 | 1149.4 | hipBLASLt contiguous 1156.7 |
| 768x1024x7168 | bfloat16 | 12 | one-tile reduce-scatter, n = 10 |  | 287.3 | 292.9 | 84.8 | 82.6 | 86.2 | 83.7 | torch | 411.1 | 404.8 | 407.3 | 411.0 | AITER torch contiguous 411.1 |
| 768x2048x7168 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 |  | 435.2 | 431.3 | 170.0 | 166.6 | 172.9 | 167.5 | flydsl | 558.6 | 388.5 (copy) | 566.5 | 557.4 | hipBLASLt contiguous 566.5 |
| 1024x512x7168 | bfloat16 | 8 | one-tile reduce-scatter, n = 14 |  | 230.6 | 221.8 | 55.9 | 55.2 | 57.4 | 55.6 | flydsl | 336.0 | 217.4 (copy) | 291.4 | 305.3 | AITER flydsl contiguous 336.0 |
| 1024x1024x7168 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 |  | 373.8 | 370.5 | 113.8 | 111.8 | 116.5 | 112.0 | torch | 561.2 | 547.7 | 563.0 | 547.5 | hipBLASLt contiguous 563.0 |
| 1024x1536x7168 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 |  | 436.3 | 431.1 | 169.5 | 166.2 | 173.2 | 167.7 | flydsl | 563.2 | 407.9 (copy) | 570.9 | 559.2 | hipBLASLt contiguous 570.9 |
| 1024x2048x7168 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 514.2 | 502.0 | 228.5 | 222.5 | 231.0 | 224.2 | torch | 731.2 | 710.6 | 732.9 | 710.4 | hipBLASLt contiguous 732.9 |
| 1024x3584x7168 | bfloat16 | 56 | one-tile owner, n = 3 |  | 727.7 | 728.6 | 399.3 | 388.5 | 406.2 | 393.6 | flydsl | 773.4 | 593.1 (copy) | 682.7 | 678.3 | AITER flydsl contiguous 773.4 |
| 1024x6144x7168 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 907.0 | 893.0 | 680.8 | 664.9 | 691.8 | 671.4 | torch | 980.2 | 968.6 | 982.4 | 975.8 | hipBLASLt contiguous 982.4 |
| 1024x8192x7168 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1165.5 | 1190.8 | 888.7 | 877.7 | 901.2 | 884.3 | flydsl | 1017.8 | 733.4 (copy) | 945.6 | 904.6 | AITER flydsl contiguous 1017.8 |
| 1024x8448x7168 | bfloat16 | 132 | data-parallel | yes | 927.1 | 904.7 | 914.2 | 897.6 | 925.8 | 904.9 | torch | 1032.6 | 1025.0 | 1042.5 | 1024.9 | hipBLASLt contiguous 1042.5 |
| 1024x20480x7168 | bfloat16 | 320 | one-tile owner, n = 3 |  | 1318.9 | 1324.2 | 1127.3 | 1105.7 | 1118.0 | 1110.1 | flydsl | 737.2 | 601.2 (copy) | 1204.9 | 1201.3 | hipBLASLt contiguous 1204.9 |
| 1280x512x7168 | bfloat16 | 10 | one-tile reduce-scatter, n = 12 |  | 256.4 | 260.8 | 70.5 | 69.1 | 72.0 | 69.4 | torch | 364.4 | 359.8 | 364.9 | 367.5 | hipBLASLt padded 367.5 |
| 1280x1024x7168 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 |  | 389.9 | 393.6 | 142.7 | 139.9 | 145.6 | 140.7 | torch | 547.2 | 531.6 | 546.4 | 535.3 | AITER torch contiguous 547.2 |
| 1280x2048x7168 | bfloat16 | 40 | one-tile owner, n = 4 |  | 604.7 | 599.6 | 286.5 | 279.3 | 292.0 | 281.5 | torch | 750.5 | 735.9 | 749.8 | 751.1 | hipBLASLt padded 751.1 |
| 1536x512x7168 | bfloat16 | 12 | one-tile reduce-scatter, n = 10 |  | 287.1 | 292.2 | 85.0 | 83.4 | 86.8 | 84.2 | torch | 403.5 | 403.6 | 404.9 | 409.2 | hipBLASLt padded 409.2 |
| 1536x1024x7168 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 |  | 440.9 | 434.2 | 171.1 | 167.6 | 174.8 | 169.0 | torch | 650.2 | 640.2 | 650.5 | 640.4 | hipBLASLt contiguous 650.5 |
| 1536x2048x7168 | bfloat16 | 48 | one-tile owner, n = 3 |  | 653.0 | 652.4 | 346.7 | 338.7 | 354.2 | 343.7 | torch | 838.3 | 837.8 | 840.0 | 837.7 | hipBLASLt contiguous 840.0 |
| 1792x512x7168 | bfloat16 | 14 | one-tile reduce-scatter, n = 9 |  | 314.3 | 319.2 | 99.1 | 98.1 | 100.9 | 97.2 | torch | 440.7 | 442.1 | 441.3 | 445.9 | hipBLASLt padded 445.9 |
| 1792x1024x7168 | bfloat16 | 28 | one-tile reduce-scatter, n = 4 |  | 465.4 | 450.0 | 201.6 | 198.6 | 206.0 | 200.8 | flydsl | 604.9 | 420.1 (copy) | 610.8 | 593.5 | hipBLASLt contiguous 610.8 |
| 1792x2048x7168 | bfloat16 | 56 | one-tile owner, n = 3 |  | 741.6 | 737.6 | 403.0 | 394.9 | 412.3 | 398.9 | flydsl | 778.0 | 645.9 (copy) | 686.5 | 683.2 | AITER flydsl contiguous 778.0 |
| 2048x512x7168 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 |  | 374.7 | 367.9 | 113.8 | 112.1 | 116.7 | 112.8 | torch | 506.9 | 503.2 | 507.1 | 503.2 | hipBLASLt contiguous 507.1 |
| 2048x1024x7168 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 520.4 | 507.8 | 231.3 | 226.9 | 235.4 | 228.6 | torch | 711.0 | 710.5 | 713.7 | 707.0 | hipBLASLt contiguous 713.7 |
| 2048x2048x7168 | bfloat16 | 64 | one-tile owner, n = 3 |  | 835.7 | 831.6 | 462.7 | 456.5 | 472.8 | 458.6 | flydsl | 873.4 | 698.7 (copy) | 773.1 | 764.5 | AITER flydsl contiguous 873.4 |
| 2048x3584x7168 | bfloat16 | 112 | one-tile owner, n = 2-3 |  | 1076.1 | 1081.6 | 802.6 | 792.2 | 812.0 | 798.7 | opus | 1000.5 | 995.6 | 1050.9 | 1018.6 | hipBLASLt contiguous 1050.9 |
| 2048x8192x7168 | bfloat16 | 256 | data-parallel | yes | 1501.5 | 1528.6 | 1514.1 | 1516.4 | 1512.4 | 1524.7 | asm | 1474.4 | 1480.2 | 1453.0 | 1460.3 | AITER asm padded 1480.2 |
| 2048x20480x7168 | bfloat16 | 640 | one-tile owner, n = 2 |  | 1480.1 | 1482.8 | 1403.2 | 1400.1 | 1382.5 | 1391.9 | flydsl | 827.3 | 712.1 (copy) | 1279.8 | 1287.8 | hipBLASLt padded 1287.8 |
| 4096x512x7168 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 508.0 | 498.9 | 228.7 | 224.2 | 233.2 | 224.2 | flydsl | 626.1 | 427.7 (copy) | 603.1 | 589.3 | AITER flydsl contiguous 626.1 |
| 4096x1024x7168 | bfloat16 | 64 | one-tile owner, n = 3 |  | 812.0 | 823.7 | 458.9 | 450.3 | 468.0 | 452.0 | torch | 900.7 | 879.0 | 916.3 | 877.5 | hipBLASLt contiguous 916.3 |
| 4096x2048x7168 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1195.0 | 1213.1 | 899.5 | 899.7 | 912.1 | 905.8 | torch | 1173.8 | 1156.2 | 1181.6 | 1150.2 | hipBLASLt contiguous 1181.6 |
| 4096x3584x7168 | bfloat16 | 224 | data-parallel | yes | 1399.4 | 1418.7 | 1406.2 | 1406.8 | 1409.6 | 1415.1 | flydsl | 1330.5 | 1074.5 (copy) | 1332.7 | 1332.0 | hipBLASLt contiguous 1332.7 |
| 4096x8192x7168 | bfloat16 | 512 | data-parallel |  | 1546.0 | 1556.7 | 1546.0 | 1556.7 | 1541.2 | 1554.5 | flydsl | 1492.1 | 1260.8 (copy) | 1479.9 | 1483.4 | AITER flydsl contiguous 1492.1 |
| 4096x20480x7168 | bfloat16 | 1280 | data-parallel |  | 1568.2 | 1585.1 | 1568.2 | 1585.1 | 1561.8 | 1570.0 | flydsl | 1507.6 | 1313.8 (copy) | 1462.7 | 1457.7 | AITER flydsl contiguous 1507.6 |
| 6144x512x7168 | bfloat16 | 48 | one-tile owner, n = 3 |  | 628.8 | 625.0 | 339.8 | 330.6 | 346.6 | 334.3 | flydsl | 722.3 | 477.1 (copy) | 667.9 | 640.3 | AITER flydsl contiguous 722.3 |
| 6144x1024x7168 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 889.6 | 892.4 | 673.9 | 662.8 | 687.8 | 664.7 | opus | 966.0 | 974.5 | 883.7 | 844.5 | AITER opus padded 974.5 |
| 6144x2048x7168 | bfloat16 | 192 | data-parallel | yes | 1261.3 | 1266.4 | 1258.2 | 1260.0 | 1276.1 | 1272.4 | opus | 1357.5 | 1340.9 | 1305.0 | 1308.1 | AITER opus contiguous 1357.5 |
| 8192x512x7168 | bfloat16 | 64 | one-tile owner, n = 3 |  | 796.1 | 802.2 | 454.8 | 443.9 | 461.4 | 447.7 | flydsl | 842.9 | 491.9 (copy) | 756.4 | 750.2 | AITER flydsl contiguous 842.9 |
| 8192x1024x7168 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1159.0 | 1188.8 | 887.4 | 872.4 | 901.9 | 878.7 | torch | 1090.4 | 1070.2 | 1096.8 | 1070.8 | hipBLASLt contiguous 1096.8 |
| 8192x2048x7168 | bfloat16 | 256 | data-parallel | yes | 1482.5 | 1503.6 | 1494.4 | 1512.4 | 1498.1 | 1518.8 | torch | 1407.1 | 1416.3 | 1417.7 | 1413.7 | hipBLASLt contiguous 1417.7 |
| 8192x3584x7168 | bfloat16 | 448 | data-parallel |  | 1423.3 | 1435.2 | 1423.3 | 1435.2 | 1414.8 | 1426.3 | flydsl | 1381.5 | 1170.1 (copy) | 1350.3 | 1348.5 | AITER flydsl contiguous 1381.5 |
| 8192x8192x7168 | bfloat16 | 1024 | data-parallel |  | 1559.8 | 1581.4 | 1559.8 | 1581.4 | 1549.5 | 1576.4 | flydsl | 1498.1 | 1339.1 (copy) | 1520.1 | 1515.6 | hipBLASLt contiguous 1520.1 |
| 8192x20480x7168 | bfloat16 | 2560 | data-parallel |  | 1590.1 | 1609.9 | 1590.1 | 1609.9 | 1584.0 | 1588.9 | asm | 1521.2 | 1540.4 | 1488.3 | 1475.7 | AITER asm padded 1540.4 |
| 10240x512x7168 | bfloat16 | 80 | one-tile owner, n = 3-4 |  | 889.9 | 902.8 | 556.6 | 543.2 | 566.4 | 547.9 | torch | 821.9 | 851.7 | 822.5 | 844.9 | AITER torch padded 851.7 |
| 10240x1024x7168 | bfloat16 | 160 | data-parallel | yes | 1093.2 | 1080.2 | 1088.8 | 1069.7 | 1106.8 | 1084.9 | torch | 1208.7 | 1211.3 | 1219.3 | 1213.2 | hipBLASLt contiguous 1219.3 |
| 10240x2048x7168 | bfloat16 | 320 | one-tile owner, n = 3 |  | 1334.1 | 1346.3 | 1151.6 | 1144.3 | 1141.9 | 1140.2 | torch | 1248.3 | 1254.1 | 1263.3 | 1253.3 | hipBLASLt contiguous 1263.3 |
| 10240x8448x7168 | bfloat16 | 1320 | one-tile owner, n = 4 |  | 1487.4 | 1502.7 | 1419.5 | 1424.5 | 1394.7 | 1410.4 | asm | 1397.3 | 1400.6 | 1363.2 | 1367.9 | AITER asm padded 1400.6 |
| 10240x20480x7168 | bfloat16 | 3200 | one-tile owner, n = 2 |  | 1512.8 | 1535.5 | 1510.2 | 1520.1 | 1477.4 | 1492.5 | asm | 1497.2 | 1513.3 | 1468.4 | 1461.5 | AITER asm padded 1513.3 |
| 12288x512x7168 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 862.4 | 860.3 | 659.3 | 644.7 | 670.6 | 650.0 | opus | 928.5 | 933.3 | 853.4 | 908.3 | AITER opus padded 933.3 |
| 12288x1024x7168 | bfloat16 | 192 | data-parallel | yes | 1230.1 | 1245.2 | 1234.1 | 1234.4 | 1242.5 | 1247.7 | opus | 1334.9 | 1312.3 | 1282.7 | 1285.0 | AITER opus contiguous 1334.9 |
| 12288x2048x7168 | bfloat16 | 384 | one-tile owner, n = 2 |  | 1415.9 | 1428.2 | 1308.5 | 1308.7 | 1296.5 | 1303.8 | opus | 1374.4 | 1371.1 | 1326.8 | 1327.5 | AITER opus contiguous 1374.4 |
| 14336x512x7168 | bfloat16 | 112 | one-tile owner, n = 2-3 |  | 1009.1 | 1011.6 | 763.8 | 750.2 | 776.7 | 760.0 | opus | 993.9 | 968.0 | 860.8 | 909.0 | AITER opus contiguous 993.9 |
| 14336x1024x7168 | bfloat16 | 224 | data-parallel | yes | 1351.7 | 1376.5 | 1358.9 | 1362.1 | 1368.3 | 1376.6 | torch | 1306.5 | 1323.5 | 1329.2 | 1332.5 | hipBLASLt padded 1332.5 |
| 14336x2048x7168 | bfloat16 | 448 | data-parallel |  | 1410.1 | 1420.1 | 1410.1 | 1420.1 | 1403.6 | 1414.1 | flydsl | 1364.4 | 1074.2 (copy) | 1344.5 | 1352.1 | AITER flydsl contiguous 1364.4 |
| 16384x512x7168 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1113.9 | 1135.1 | 867.2 | 853.7 | 879.0 | 855.9 | opus | 1074.9 | 1057.8 | 1048.5 | 1010.5 | AITER opus contiguous 1074.9 |
| 16384x1024x7168 | bfloat16 | 256 | data-parallel | yes | 1472.5 | 1494.7 | 1475.5 | 1481.7 | 1478.2 | 1497.9 | torch | 1409.4 | 1413.7 | 1418.3 | 1418.5 | hipBLASLt padded 1418.5 |
| 16384x2048x7168 | bfloat16 | 512 | data-parallel |  | 1516.0 | 1533.8 | 1516.0 | 1533.8 | 1512.8 | 1532.9 | torch | 1468.6 | 1475.1 | 1474.7 | 1476.9 | hipBLASLt padded 1476.9 |
| 16384x3584x7168 | bfloat16 | 896 | one-tile owner, n = 2 |  | 1497.9 | 1508.0 | 1441.2 | 1442.8 | 1419.7 | 1434.5 | flydsl | 1378.8 | 1192.1 (copy) | 1383.1 | 1375.0 | hipBLASLt contiguous 1383.1 |
| 16384x8192x7168 | bfloat16 | 2048 | data-parallel |  | 1574.9 | 1597.9 | 1574.9 | 1597.9 | 1562.9 | 1575.2 | torch | 1512.4 | 1521.7 | 1519.5 | 1521.4 | AITER torch padded 1521.7 |
| 16384x20480x7168 | bfloat16 | 5120 | data-parallel |  | 1603.0 | 1615.6 | 1603.0 | 1615.6 | 1583.5 | 1589.7 | flydsl | 1531.5 | 1452.6 (copy) | 1508.3 | 1488.8 | AITER flydsl contiguous 1531.5 |
| 32768x512x7168 | bfloat16 | 256 | data-parallel | yes | 1371.9 | 1366.7 | 1379.7 | 1391.3 | 1389.8 | 1371.8 | torch | 1379.3 | 1383.4 | 1395.1 | 1369.9 | hipBLASLt contiguous 1395.1 |
| 32768x1024x7168 | bfloat16 | 512 | data-parallel |  | 1490.2 | 1508.0 | 1490.2 | 1508.0 | 1493.1 | 1506.0 | torch | 1424.2 | 1430.6 | 1427.9 | 1428.9 | AITER torch padded 1430.6 |
| 32768x2048x7168 | bfloat16 | 1024 | data-parallel |  | 1537.5 | 1552.9 | 1537.5 | 1552.9 | 1526.1 | 1540.9 | torch | 1474.3 | 1483.9 | 1485.9 | 1485.8 | hipBLASLt contiguous 1485.9 |
| 32768x3584x7168 | bfloat16 | 1792 | data-parallel |  | 1546.5 | 1565.9 | 1546.5 | 1565.9 | 1531.3 | 1544.9 | torch | 1504.5 | 1506.3 | 1510.1 | 1510.1 | hipBLASLt contiguous 1510.1 |
| 32768x8192x7168 | bfloat16 | 4096 | data-parallel |  | 1591.6 | 1604.8 | 1591.6 | 1604.8 | 1570.1 | 1572.3 | torch | 1531.5 | 1555.3 | 1541.4 | 1554.8 | AITER torch padded 1555.3 |
| 32768x20480x7168 | bfloat16 | 10240 | data-parallel |  | 1600.4 | 1612.9 | 1600.4 | 1612.9 | 1583.9 | 1587.9 | asm | 1555.4 | 1557.4 | 1505.9 | 1475.5 | AITER asm padded 1557.4 |
| 512x1280x8192 | bfloat16 | 10 | one-tile reduce-scatter, n = 12 |  | 284.2 | 280.0 | 77.8 | 72.4 | 78.4 | 72.8 | torch | 422.0 | 410.8 | 422.4 | 409.0 | hipBLASLt contiguous 422.4 |
| 512x2560x8192 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 |  | 435.2 | 424.8 | 155.4 | 146.8 | 156.3 | 146.4 | torch | 564.9 | 544.6 | 562.4 | 545.5 | AITER torch contiguous 564.9 |
| 512x4096x8192 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 558.7 | 550.2 | 247.3 | 237.3 | 247.0 | 238.0 | flydsl | 681.9 | 442.8 (copy) | 647.2 | 619.9 | AITER flydsl contiguous 681.9 |
| 512x4608x8192 | bfloat16 | 36 | one-tile owner, n = 4 |  | 601.4 | 594.1 | 281.8 | 266.4 | 280.0 | 266.8 | torch | 718.8 | 709.9 | 724.1 | 713.4 | hipBLASLt contiguous 724.1 |
| 512x7168x8192 | bfloat16 | 56 | one-tile owner, n = 3 |  | 792.0 | 787.7 | 434.4 | 413.0 | 434.4 | 415.1 | flydsl | 803.3 | 478.4 (copy) | 744.7 | 741.4 | AITER flydsl contiguous 803.3 |
| 512x8192x8192 | bfloat16 | 64 | one-tile owner, n = 3 |  | 869.1 | 876.9 | 477.2 | 471.4 | 476.9 | 472.3 | flydsl | 686.3 | 448.0 (copy) | 770.5 | 750.8 | hipBLASLt contiguous 770.5 |
| 512x10240x8192 | bfloat16 | 80 | one-tile owner, n = 3-4 |  | 963.5 | 981.5 | 604.3 | 582.8 | 604.2 | 587.8 | flydsl | 624.6 | 415.5 (copy) | 570.1 | 557.9 | AITER flydsl contiguous 624.6 |
| 512x14336x8192 | bfloat16 | 112 | one-tile owner, n = 2-3 |  | 1063.2 | 1068.1 | 824.3 | 796.0 | 848.3 | 808.8 | torch | 1038.0 | 1033.5 | 1052.2 | 1031.7 | hipBLASLt contiguous 1052.2 |
| 512x16384x8192 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1137.6 | 1176.4 | 893.1 | 908.9 | 897.4 | 908.6 | flydsl | 651.6 | 440.7 (copy) | 921.7 | 981.3 | hipBLASLt padded 981.3 |
| 512x57344x8192 | bfloat16 | 448 | data-parallel |  | 1332.2 | 1353.7 | 1332.2 | 1353.7 | 1325.8 | 1357.0 | flydsl | 1287.4 | 683.5 (copy) | 1268.8 | 1284.1 | AITER flydsl contiguous 1287.4 |
| 768x4096x8192 | bfloat16 | 48 | one-tile owner, n = 4 |  | 768.2 | 760.1 | 375.1 | 357.3 | 376.5 | 360.1 | flydsl | 746.2 | 548.8 (copy) | 741.5 | 676.3 | AITER flydsl contiguous 746.2 |
| 1024x512x8192 | bfloat16 | 8 | one-tile reduce-scatter, n = 16 |  | 273.1 | 267.9 | 62.4 | 58.3 | 62.7 | 58.4 | torch | 335.3 | 327.8 | 335.6 | 338.7 | hipBLASLt padded 338.7 |
| 1024x1280x8192 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 |  | 444.2 | 429.9 | 157.7 | 147.0 | 158.2 | 148.1 | torch | 577.2 | 555.8 | 579.9 | 558.5 | hipBLASLt contiguous 579.9 |
| 1024x2560x8192 | bfloat16 | 40 | one-tile owner, n = 4 |  | 661.1 | 653.7 | 313.2 | 294.7 | 315.4 | 296.7 | torch | 787.9 | 762.0 | 791.4 | 760.9 | hipBLASLt contiguous 791.4 |
| 1024x4096x8192 | bfloat16 | 64 | one-tile owner, n = 3 |  | 892.2 | 900.3 | 494.3 | 475.2 | 492.8 | 477.7 | flydsl | 902.9 | 657.9 (copy) | 810.1 | 769.2 | AITER flydsl contiguous 902.9 |
| 1024x4608x8192 | bfloat16 | 72 | one-tile owner, n = 3-4 |  | 881.2 | 887.3 | 554.9 | 529.8 | 554.6 | 532.0 | torch | 945.3 | 925.5 | 952.6 | 926.6 | hipBLASLt contiguous 952.6 |
| 1024x7168x8192 | bfloat16 | 112 | one-tile owner, n = 2-3 |  | 1100.2 | 1097.1 | 833.3 | 815.4 | 837.4 | 820.7 | flydsl | 744.2 | 576.3 (copy) | 1047.8 | 1044.0 | hipBLASLt contiguous 1047.8 |
| 1024x8192x8192 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1201.1 | 1228.1 | 923.4 | 929.8 | 926.9 | 936.5 | flydsl | 917.4 | 651.5 (copy) | 995.8 | 985.2 | hipBLASLt contiguous 995.8 |
| 1024x10240x8192 | bfloat16 | 160 | data-parallel | yes | 1142.4 | 1145.0 | 1144.1 | 1143.4 | 1143.8 | 1145.8 | flydsl | 828.6 | 628.6 (copy) | 1218.7 | 1202.2 | hipBLASLt contiguous 1218.7 |
| 1024x14336x8192 | bfloat16 | 224 | data-parallel | yes | 1390.5 | 1414.7 | 1396.5 | 1402.3 | 1399.3 | 1411.8 | flydsl | 1357.8 | 890.4 (copy) | 1311.4 | 1310.5 | AITER flydsl contiguous 1357.8 |
| 1024x16384x8192 | bfloat16 | 256 | data-parallel | yes | 1482.2 | 1499.6 | 1486.6 | 1490.7 | 1483.5 | 1506.0 | torch | 1438.9 | 1430.0 | 1455.4 | 1423.8 | hipBLASLt contiguous 1455.4 |
| 1024x57344x8192 | bfloat16 | 896 | one-tile owner, n = 2 |  | 1488.9 | 1494.4 | 1437.2 | 1437.7 | 1416.5 | 1422.1 | asm | 1373.3 | 1388.5 | 1327.1 | 1333.9 | AITER asm padded 1388.5 |
| 2048x512x8192 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 |  | 418.3 | 409.2 | 126.4 | 118.2 | 127.3 | 119.1 | torch | 503.8 | 499.6 | 505.9 | 505.5 | hipBLASLt contiguous 505.9 |
| 2048x1280x8192 | bfloat16 | 40 | one-tile owner, n = 4 |  | 663.1 | 661.5 | 317.2 | 300.2 | 318.3 | 300.8 | torch | 692.1 | 698.7 | 695.7 | 703.3 | hipBLASLt padded 703.3 |
| 2048x2560x8192 | bfloat16 | 80 | one-tile owner, n = 3-4 |  | 1010.9 | 1021.8 | 620.9 | 601.5 | 618.1 | 603.6 | torch | 997.3 | 984.6 | 992.0 | 981.2 | AITER torch contiguous 997.3 |
| 2048x4096x8192 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1228.3 | 1258.2 | 947.5 | 943.4 | 944.4 | 943.2 | opus | 1113.9 | 1132.6 | 1015.5 | 985.2 | AITER opus padded 1132.6 |
| 2048x4608x8192 | bfloat16 | 144 | data-parallel | yes | 1056.7 | 1068.0 | 1050.5 | 1064.4 | 1050.6 | 1074.1 | torch | 1160.0 | 1144.4 | 1170.4 | 1147.2 | hipBLASLt contiguous 1170.4 |
| 2048x7168x8192 | bfloat16 | 224 | data-parallel | yes | 1417.4 | 1431.3 | 1425.6 | 1421.9 | 1429.2 | 1432.5 | asm | 1379.9 | 1387.2 | 1340.5 | 1338.4 | AITER asm padded 1387.2 |
| 2048x8192x8192 | bfloat16 | 256 | data-parallel | yes | 1521.0 | 1538.7 | 1519.9 | 1528.8 | 1520.9 | 1535.3 | torch | 1454.6 | 1464.3 | 1472.0 | 1468.4 | hipBLASLt contiguous 1472.0 |
| 2048x10240x8192 | bfloat16 | 320 | one-tile owner, n = 3 |  | 1383.9 | 1395.5 | 1180.1 | 1181.1 | 1173.0 | 1180.0 | flydsl | 932.6 | 766.3 (copy) | 1254.2 | 1237.2 | hipBLASLt contiguous 1254.2 |
| 2048x14336x8192 | bfloat16 | 448 | data-parallel |  | 1438.0 | 1447.1 | 1438.0 | 1447.1 | 1437.1 | 1447.5 | flydsl | 1400.4 | 1095.7 (copy) | 1346.5 | 1348.8 | AITER flydsl contiguous 1400.4 |
| 2048x16384x8192 | bfloat16 | 512 | data-parallel |  | 1541.1 | 1555.1 | 1541.1 | 1555.1 | 1534.2 | 1551.1 | asm | 1486.0 | 1491.3 | 1490.5 | 1482.5 | AITER asm padded 1491.3 |
| 2048x57344x8192 | bfloat16 | 1792 | data-parallel |  | 1576.8 | 1586.3 | 1576.8 | 1586.3 | 1566.9 | 1569.8 | flydsl | 1508.6 | 1203.5 (copy) | 1490.8 | 1483.0 | AITER flydsl contiguous 1508.6 |
| 4096x512x8192 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 557.8 | 552.0 | 247.2 | 239.0 | 250.3 | 238.5 | flydsl | 647.2 | 446.5 (copy) | 666.5 | 623.1 | hipBLASLt contiguous 666.5 |
| 4096x1280x8192 | bfloat16 | 80 | one-tile owner, n = 3-4 |  | 1002.0 | 1018.0 | 610.8 | 596.7 | 612.8 | 594.7 | torch | 1003.3 | 972.9 | 1007.6 | 973.5 | hipBLASLt contiguous 1007.6 |
| 4096x2560x8192 | bfloat16 | 160 | data-parallel | yes | 1149.3 | 1164.6 | 1150.7 | 1158.4 | 1157.3 | 1173.6 | torch | 1241.0 | 1232.9 | 1252.7 | 1235.8 | hipBLASLt contiguous 1252.7 |
| 4096x4096x8192 | bfloat16 | 256 | data-parallel | yes | 1506.4 | 1549.8 | 1522.0 | 1540.2 | 1526.2 | 1546.0 | torch | 1468.5 | 1469.5 | 1480.5 | 1464.9 | hipBLASLt contiguous 1480.5 |
| 4096x4608x8192 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 |  | 1347.1 | 1361.8 | 1081.7 | 1078.7 | 1077.2 | 1077.6 | torch | 1174.1 | 1155.0 | 1186.7 | 1156.3 | hipBLASLt contiguous 1186.7 |
| 4096x7168x8192 | bfloat16 | 448 | data-parallel |  | 1458.7 | 1459.2 | 1458.7 | 1459.2 | 1450.5 | 1450.7 | flydsl | 1409.8 | 1195.2 (copy) | 1372.2 | 1365.3 | AITER flydsl contiguous 1409.8 |
| 4096x8192x8192 | bfloat16 | 512 | data-parallel |  | 1547.0 | 1572.2 | 1547.0 | 1572.2 | 1546.7 | 1566.7 | asm | 1521.7 | 1522.9 | 1507.9 | 1502.3 | AITER asm padded 1522.9 |
| 4096x10240x8192 | bfloat16 | 640 | one-tile owner, n = 2 |  | 1512.9 | 1522.9 | 1425.5 | 1431.7 | 1413.4 | 1424.9 | flydsl | 1369.8 | 1179.2 (copy) | 1339.1 | 1332.2 | AITER flydsl contiguous 1369.8 |
| 4096x14336x8192 | bfloat16 | 896 | one-tile owner, n = 2 |  | 1536.0 | 1547.5 | 1476.7 | 1477.1 | 1460.3 | 1466.4 | flydsl | 1411.2 | 1226.5 (copy) | 1359.7 | 1359.1 | AITER flydsl contiguous 1411.2 |
| 4096x16384x8192 | bfloat16 | 1024 | data-parallel |  | 1575.9 | 1590.8 | 1575.9 | 1590.8 | 1568.8 | 1580.6 | flydsl | 1508.9 | 1309.3 (copy) | 1516.0 | 1502.1 | hipBLASLt contiguous 1516.0 |
| 4096x57344x8192 | bfloat16 | 3584 | data-parallel |  | 1602.5 | 1606.4 | 1602.5 | 1606.4 | 1586.7 | 1592.7 | flydsl | 1532.0 | 1354.8 (copy) | 1510.3 | 1511.7 | AITER flydsl contiguous 1532.0 |
| 8192x512x8192 | bfloat16 | 64 | one-tile owner, n = 3 |  | 862.8 | 875.9 | 476.7 | 469.4 | 477.5 | 471.4 | flydsl | 870.9 | 505.5 (copy) | 798.1 | 764.3 | AITER flydsl contiguous 870.9 |
| 8192x1280x8192 | bfloat16 | 160 | data-parallel | yes | 1126.4 | 1143.4 | 1119.5 | 1138.7 | 1130.7 | 1144.3 | torch | 1223.9 | 1219.7 | 1237.7 | 1222.1 | hipBLASLt contiguous 1237.7 |
| 8192x2560x8192 | bfloat16 | 320 | one-tile owner, n = 3 |  | 1376.4 | 1382.7 | 1168.9 | 1157.6 | 1155.5 | 1154.7 | torch | 1251.2 | 1251.5 | 1267.1 | 1255.3 | hipBLASLt contiguous 1267.1 |
| 8192x4096x8192 | bfloat16 | 512 | data-parallel |  | 1553.4 | 1575.2 | 1553.4 | 1575.2 | 1550.6 | 1572.8 | torch | 1527.7 | 1538.9 | 1553.3 | 1542.6 | hipBLASLt contiguous 1553.3 |
| 8192x4608x8192 | bfloat16 | 576 | one-tile owner, n = 3 |  | 1477.5 | 1484.4 | 1349.4 | 1343.9 | 1329.6 | 1337.2 | torch | 1333.6 | 1348.1 | 1354.2 | 1351.3 | hipBLASLt contiguous 1354.2 |
| 8192x7168x8192 | bfloat16 | 896 | one-tile owner, n = 2 |  | 1539.2 | 1550.9 | 1479.7 | 1480.3 | 1462.0 | 1465.6 | asm | 1418.1 | 1426.7 | 1399.8 | 1391.3 | AITER asm padded 1426.7 |
| 8192x8192x8192 | bfloat16 | 1024 | data-parallel |  | 1575.6 | 1599.3 | 1575.6 | 1599.3 | 1570.9 | 1591.0 | asm | 1543.1 | 1551.5 | 1518.1 | 1518.0 | AITER asm padded 1551.5 |
| 8192x10240x8192 | bfloat16 | 1280 | data-parallel |  | 1589.6 | 1608.0 | 1589.6 | 1608.0 | 1585.4 | 1594.0 | flydsl | 1523.9 | 1382.0 (copy) | 1536.7 | 1527.7 | hipBLASLt contiguous 1536.7 |
| 8192x14336x8192 | bfloat16 | 1792 | data-parallel |  | 1596.8 | 1614.1 | 1596.8 | 1614.1 | 1592.7 | 1599.9 | flydsl | 1533.9 | 1404.8 (copy) | 1539.4 | 1516.5 | hipBLASLt contiguous 1539.4 |
| 8192x16384x8192 | bfloat16 | 2048 | data-parallel |  | 1586.8 | 1613.5 | 1586.8 | 1613.5 | 1583.7 | 1597.2 | asm | 1544.0 | 1538.4 | 1533.1 | 1517.0 | AITER asm contiguous 1544.0 |
| 8192x57344x8192 | bfloat16 | 7168 | data-parallel |  | 1598.7 | 1599.3 | 1598.7 | 1599.3 | 1602.4 | 1605.1 | asm | 1545.1 | 1547.8 | 1468.5 | 1463.0 | AITER asm padded 1547.8 |
| 16384x512x8192 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1131.7 | 1177.3 | 893.4 | 897.4 | 897.4 | 904.5 | torch | 1069.7 | 1065.4 | 1092.0 | 1066.6 | hipBLASLt contiguous 1092.0 |
| 16384x1280x8192 | bfloat16 | 320 | one-tile owner, n = 3 |  | 1344.2 | 1364.1 | 1160.2 | 1153.9 | 1152.3 | 1151.2 | torch | 1245.6 | 1252.1 | 1255.0 | 1252.3 | hipBLASLt contiguous 1255.0 |
| 16384x2560x8192 | bfloat16 | 640 | one-tile owner, n = 2 |  | 1477.6 | 1493.6 | 1398.4 | 1402.0 | 1378.9 | 1397.3 | flydsl | 1339.8 | 1111.0 (copy) | 1310.8 | 1306.6 | AITER flydsl contiguous 1339.8 |
| 16384x4608x8192 | bfloat16 | 1152 | one-tile owner, n = 2 |  | 1528.4 | 1549.7 | 1489.9 | 1492.5 | 1467.1 | 1478.2 | flydsl | 1419.5 | 1261.8 (copy) | 1210.6 | 1190.0 | AITER flydsl contiguous 1419.5 |
| 16384x7168x8192 | bfloat16 | 1792 | data-parallel |  | 1580.2 | 1601.9 | 1580.2 | 1601.9 | 1576.0 | 1582.7 | torch | 1514.1 | 1530.2 | 1536.9 | 1536.4 | hipBLASLt contiguous 1536.9 |
| 16384x8192x8192 | bfloat16 | 2048 | data-parallel |  | 1588.8 | 1607.9 | 1588.8 | 1607.9 | 1579.9 | 1593.2 | torch | 1538.1 | 1554.5 | 1563.3 | 1552.4 | hipBLASLt contiguous 1563.3 |
| 16384x10240x8192 | bfloat16 | 2560 | data-parallel |  | 1598.2 | 1613.7 | 1598.2 | 1613.7 | 1587.2 | 1595.1 | torch | 1554.1 | 1566.9 | 1578.0 | 1566.1 | hipBLASLt contiguous 1578.0 |
| 16384x14336x8192 | bfloat16 | 3584 | data-parallel |  | 1606.3 | 1617.1 | 1606.3 | 1617.1 | 1597.4 | 1599.3 | asm | 1546.2 | 1560.0 | 1556.3 | 1557.8 | AITER asm padded 1560.0 |
| 16384x16384x8192 | bfloat16 | 4096 | data-parallel |  | 1606.0 | 1622.3 | 1606.0 | 1622.3 | 1600.5 | 1600.9 | asm | 1557.1 | 1546.9 | 1541.9 | 1538.6 | AITER asm contiguous 1557.1 |
| 16384x57344x8192 | bfloat16 | 14336 | data-parallel |  | 1517.1 | 1560.9 | 1517.1 | 1560.9 | 1604.8 | 1602.4 | asm | 1561.7 | 1559.2 | 1440.3 | 1393.9 | AITER asm contiguous 1561.7 |
| 32768x1280x8192 | bfloat16 | 640 | one-tile owner, n = 2 |  | 1443.1 | 1458.7 | 1366.0 | 1377.6 | 1356.8 | 1367.0 | flydsl | 1311.1 | 963.2 (copy) | 1335.8 | 1332.9 | hipBLASLt contiguous 1335.8 |
| 32768x2560x8192 | bfloat16 | 1280 | data-parallel |  | 1540.4 | 1554.0 | 1540.4 | 1554.0 | 1530.5 | 1539.6 | torch | 1489.1 | 1512.3 | 1498.4 | 1509.4 | AITER torch padded 1512.3 |
| 32768x7168x8192 | bfloat16 | 3584 | data-parallel |  | 1590.2 | 1596.6 | 1590.2 | 1596.6 | 1580.0 | 1575.4 | torch | 1549.6 | 1573.9 | 1578.9 | 1574.5 | hipBLASLt contiguous 1578.9 |
| 32768x8192x8192 | bfloat16 | 4096 | data-parallel |  | 1606.8 | 1609.5 | 1606.8 | 1609.5 | 1593.7 | 1588.7 | flydsl | 1547.4 | 1447.8 (copy) | 1582.6 | 1584.4 | hipBLASLt padded 1584.4 |
| 32768x10240x8192 | bfloat16 | 5120 | data-parallel |  | 1606.7 | 1607.0 | 1606.7 | 1607.0 | 1596.1 | 1589.1 | torch | 1571.1 | 1583.8 | 1592.0 | 1589.1 | hipBLASLt contiguous 1592.0 |
| 32768x14336x8192 | bfloat16 | 7168 | data-parallel |  | 1604.9 | 1598.3 | 1604.9 | 1598.3 | 1598.5 | 1588.0 | asm | 1555.3 | 1560.8 | 1541.6 | 1551.6 | AITER asm padded 1560.8 |
| 32768x16384x8192 | bfloat16 | 8192 | data-parallel |  | 1588.4 | 1595.1 | 1588.4 | 1595.1 | 1599.2 | 1581.5 | asm | 1556.4 | 1552.2 | 1509.6 | 1507.4 | AITER asm contiguous 1556.4 |
| 32768x57344x8192 | bfloat16 | 28672 | data-parallel |  | 1548.5 | 1479.1 | 1548.5 | 1479.1 | 1564.0 | 1570.2 | asm | 1551.2 | 1544.5 | 1445.3 | 1468.5 | AITER asm contiguous 1551.2 |
| 512x6144x12288 | bfloat16 | 48 | one-tile owner, n = 4 |  | 845.9 | 873.2 | 347.4 | 362.0 | 347.4 | 361.4 | torch | 784.1 | 782.9 | 784.0 | 782.9 | AITER torch contiguous 784.1 |
| 1024x6144x12288 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 1041.9 | 1065.4 | 694.6 | 719.7 | 697.2 | 720.1 | torch | 1103.0 | 1124.2 | 1111.1 | 1126.0 | hipBLASLt padded 1126.0 |
| 2048x6144x12288 | bfloat16 | 192 | data-parallel | yes | 1301.6 | 1337.1 | 1310.0 | 1330.8 | 1308.0 | 1328.9 | torch | 1311.6 | 1339.4 | 1326.2 | 1339.9 | hipBLASLt padded 1339.9 |
| 4096x6144x12288 | bfloat16 | 384 | one-tile owner, n = 2 |  | 1505.5 | 1522.0 | 1358.8 | 1362.7 | 1350.1 | 1358.4 | torch | 1325.5 | 1351.0 | 1344.0 | 1351.1 | hipBLASLt padded 1351.1 |
| 8192x6144x12288 | bfloat16 | 768 | data-parallel |  | 1561.0 | 1590.6 | 1561.0 | 1590.6 | 1559.4 | 1580.4 | torch | 1510.2 | 1558.8 | 1528.3 | 1557.6 | AITER torch padded 1558.8 |
| 16384x6144x12288 | bfloat16 | 1536 | data-parallel |  | 1578.5 | 1604.6 | 1578.5 | 1604.6 | 1567.0 | 1587.8 | torch | 1513.6 | 1574.6 | 1532.7 | 1577.2 | hipBLASLt padded 1577.2 |
| 32768x6144x12288 | bfloat16 | 3072 | data-parallel |  | 1587.1 | 1614.0 | 1587.1 | 1614.0 | 1573.5 | 1591.6 | asm | 1549.8 | 1557.0 | 1574.4 | 1586.7 | hipBLASLt padded 1586.7 |
| 512x16384x13312 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1259.2 | 1270.2 | 920.6 | 913.4 | 928.6 | 918.2 | flydsl | 706.2 | 465.4 (copy) | 1103.9 | 1094.8 | hipBLASLt contiguous 1103.9 |
| 1024x16384x13312 | bfloat16 | 256 | data-parallel | yes | 1530.4 | 1538.5 | 1530.1 | 1527.7 | 1530.7 | 1534.5 | asm | 1482.9 | 1492.0 | 1495.9 | 1498.1 | hipBLASLt padded 1498.1 |
| 2048x16384x13312 | bfloat16 | 512 | data-parallel |  | 1578.0 | 1587.4 | 1578.0 | 1587.4 | 1559.7 | 1580.8 | flydsl | 1005.9 | 840.9 (copy) | 1513.3 | 1519.7 | hipBLASLt padded 1519.7 |
| 4096x16384x13312 | bfloat16 | 1024 | data-parallel |  | 1603.2 | 1618.2 | 1603.2 | 1618.2 | 1596.8 | 1606.2 | flydsl | 1545.2 | 1335.5 (copy) | 1519.6 | 1531.5 | AITER flydsl contiguous 1545.2 |
| 8192x16384x13312 | bfloat16 | 2048 | data-parallel |  | 1583.2 | 1608.7 | 1583.2 | 1608.7 | 1575.2 | 1592.8 | flydsl | 1517.3 | 1395.6 (copy) | 1518.7 | 1537.8 | hipBLASLt padded 1537.8 |
| 16384x16384x13312 | bfloat16 | 4096 | data-parallel |  | 1599.2 | 1609.5 | 1599.2 | 1609.5 | 1583.4 | 1589.4 | asm | 1515.1 | 1535.6 | 1518.9 | 1527.5 | AITER asm padded 1535.6 |
| 32768x16384x13312 | bfloat16 | 8192 | data-parallel |  | 1563.7 | 1607.0 | 1563.7 | 1607.0 | 1578.1 | 1590.1 | flydsl | 1494.4 | 1458.3 (copy) | 1520.7 | 1449.6 | hipBLASLt contiguous 1520.7 |
| 512x2304x16384 | bfloat16 | 18 | one-tile reduce-scatter, n = 7 |  | 544.8 | 511.0 | 139.8 | 129.3 | 139.1 | 131.9 | torch | 706.1 | 623.5 | 707.8 | 625.7 | hipBLASLt contiguous 707.8 |
| 512x4608x16384 | bfloat16 | 36 | one-tile owner, n = 6 |  | 815.1 | 753.9 | 279.2 | 260.2 | 272.8 | 266.7 | flydsl | 506.8 | 347.4 (copy) | 797.0 | 768.7 | hipBLASLt contiguous 797.0 |
| 512x9216x16384 | bfloat16 | 72 | one-tile owner, n = 3-4 |  | 1028.8 | 994.7 | 537.5 | 505.7 | 506.0 | 512.2 | flydsl | 585.1 | 425.7 (copy) | 760.2 | 732.2 | hipBLASLt contiguous 760.2 |
| 512x13312x16384 | bfloat16 | 104 | one-tile owner, n = 2-3 |  | 1112.0 | 1092.0 | 740.9 | 709.0 | 738.8 | 723.1 | flydsl | 542.0 | 384.1 (copy) | 1087.8 | 1094.9 | hipBLASLt padded 1094.9 |
| 512x26624x16384 | bfloat16 | 208 | data-parallel | yes | 1290.2 | 1232.6 | 1288.3 | 1226.8 | 1282.4 | 1242.2 | flydsl | 765.8 | 493.1 (copy) | 1191.2 | 1206.5 | hipBLASLt padded 1206.5 |
| 512x53248x16384 | bfloat16 | 416 | data-parallel |  | 1304.2 | 1261.9 | 1304.2 | 1261.9 | 1306.5 | 1246.8 | flydsl | 631.8 | 437.8 (copy) | 1213.9 | 1217.8 | hipBLASLt padded 1217.8 |
| 1024x2304x16384 | bfloat16 | 36 | one-tile owner, n = 6 |  | 827.0 | 689.6 | 279.3 | 238.1 | 275.5 | 244.2 | torch | 871.9 | 757.6 | 877.3 | 761.0 | hipBLASLt contiguous 877.3 |
| 1024x4608x16384 | bfloat16 | 72 | one-tile owner, n = 3-4 |  | 1097.7 | 1031.0 | 550.2 | 480.3 | 550.3 | 497.0 | flydsl | 647.5 | 509.5 (copy) | 1032.1 | 983.5 | hipBLASLt contiguous 1032.1 |
| 1024x9216x16384 | bfloat16 | 144 | data-parallel | yes | 1008.2 | 946.2 | 1012.0 | 941.2 | 1030.0 | 960.5 | flydsl | 677.9 | 551.0 (copy) | 1162.9 | 1118.3 | hipBLASLt contiguous 1162.9 |
| 1024x13312x16384 | bfloat16 | 208 | data-parallel | yes | 1347.4 | 1301.0 | 1353.9 | 1304.7 | 1359.4 | 1320.7 | asm | 1329.7 | 1314.2 | 1245.7 | 1253.8 | AITER asm contiguous 1329.7 |
| 1024x26624x16384 | bfloat16 | 416 | data-parallel |  | 1383.1 | 1336.1 | 1383.1 | 1336.1 | 1375.7 | 1343.7 | flydsl | 948.0 | 708.2 (copy) | 1256.2 | 1261.3 | hipBLASLt padded 1261.3 |
| 1024x53248x16384 | bfloat16 | 832 | one-tile owner, n = 4 |  | 1525.9 | 1496.9 | 1388.3 | 1344.2 | 1369.7 | 1348.7 | triton | 992.0 | 961.4 | 1261.2 | 1268.2 | hipBLASLt padded 1268.2 |
| 2048x2304x16384 | bfloat16 | 72 | one-tile owner, n = 3-4 |  | 1117.2 | 994.5 | 548.1 | 472.7 | 553.4 | 503.1 | torch | 873.7 | 829.4 | 879.4 | 830.0 | hipBLASLt contiguous 879.4 |
| 2048x4608x16384 | bfloat16 | 144 | data-parallel | yes | 1046.7 | 961.0 | 1050.6 | 949.6 | 1055.7 | 987.6 | flydsl | 714.7 | 614.6 (copy) | 1195.3 | 1146.4 | hipBLASLt contiguous 1195.3 |
| 2048x9216x16384 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 |  | 1416.4 | 1351.0 | 1073.7 | 987.4 | 1044.3 | 999.3 | flydsl | 903.9 | 749.8 (copy) | 1161.9 | 1129.6 | hipBLASLt contiguous 1161.9 |
| 2048x13312x16384 | bfloat16 | 416 | data-parallel |  | 1413.4 | 1361.6 | 1413.4 | 1361.6 | 1402.0 | 1372.7 | flydsl | 1355.0 | 1072.0 (copy) | 1242.4 | 1248.3 | AITER flydsl contiguous 1355.0 |
| 2048x26624x16384 | bfloat16 | 832 | one-tile owner, n = 4 |  | 1562.9 | 1514.4 | 1421.2 | 1356.9 | 1401.2 | 1362.1 | flydsl | 903.8 | 778.6 (copy) | 1243.4 | 1242.2 | hipBLASLt contiguous 1243.4 |
| 2048x53248x16384 | bfloat16 | 1664 | one-tile owner, n = 2 |  | 1594.4 | 1558.0 | 1541.4 | 1490.7 | 1511.5 | 1488.6 | flydsl | 782.3 | 663.2 (copy) | 1290.7 | 1295.3 | hipBLASLt padded 1295.3 |
| 4096x2304x16384 | bfloat16 | 144 | data-parallel | yes | 1029.3 | 935.9 | 1036.9 | 932.9 | 1039.7 | 965.1 | torch | 1182.0 | 1145.5 | 1193.6 | 1151.2 | hipBLASLt contiguous 1193.6 |
| 4096x4608x16384 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 |  | 1401.9 | 1351.5 | 1064.1 | 997.8 | 1078.4 | 1013.9 | torch | 1176.5 | 1135.8 | 1187.9 | 1136.4 | hipBLASLt contiguous 1187.9 |
| 4096x9216x16384 | bfloat16 | 576 | one-tile owner, n = 4 |  | 1538.8 | 1522.7 | 1372.9 | 1311.9 | 1358.8 | 1323.7 | torch | 1323.3 | 1341.6 | 1339.0 | 1344.1 | hipBLASLt padded 1344.1 |
| 4096x13312x16384 | bfloat16 | 832 | one-tile owner, n = 4 |  | 1561.6 | 1560.0 | 1427.8 | 1414.3 | 1418.0 | 1406.1 | flydsl | 1375.0 | 1187.2 (copy) | 1264.9 | 1275.5 | AITER flydsl contiguous 1375.0 |
| 4096x26624x16384 | bfloat16 | 1664 | one-tile owner, n = 2 |  | 1602.3 | 1565.7 | 1553.5 | 1500.0 | 1529.4 | 1497.6 | flydsl | 1488.7 | 1299.9 (copy) | 1319.8 | 1321.1 | AITER flydsl contiguous 1488.7 |
| 4096x53248x16384 | bfloat16 | 3328 | data-parallel |  | 1605.8 | 1624.7 | 1605.8 | 1624.7 | 1601.6 | 1607.5 | flydsl | 1540.4 | 1356.2 (copy) | 1494.2 | 1509.2 | AITER flydsl contiguous 1540.4 |
| 8192x2304x16384 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 |  | 1346.3 | 1283.5 | 958.8 | 923.1 | 1002.1 | 921.8 | torch | 1177.2 | 1138.4 | 1187.3 | 1137.3 | hipBLASLt contiguous 1187.3 |
| 8192x4608x16384 | bfloat16 | 576 | one-tile owner, n = 4 |  | 1464.6 | 1429.2 | 1248.7 | 1210.6 | 1295.5 | 1210.7 | torch | 1342.3 | 1355.8 | 1360.4 | 1357.7 | hipBLASLt contiguous 1360.4 |
| 8192x9216x16384 | bfloat16 | 1152 | one-tile owner, n = 2 |  | 1548.8 | 1493.0 | 1476.1 | 1401.2 | 1463.6 | 1402.1 | flydsl | 1405.0 | 1267.6 (copy) | 1145.2 | 1143.7 | AITER flydsl contiguous 1405.0 |
| 8192x13312x16384 | bfloat16 | 1664 | one-tile owner, n = 2 |  | 1557.0 | 1503.1 | 1512.3 | 1431.1 | 1492.6 | 1433.6 | flydsl | 1450.5 | 1320.2 (copy) | 1310.9 | 1314.7 | AITER flydsl contiguous 1450.5 |
| 8192x26624x16384 | bfloat16 | 3328 | data-parallel |  | 1572.4 | 1517.0 | 1572.4 | 1517.0 | 1565.0 | 1508.6 | flydsl | 1516.8 | 1397.5 (copy) | 1480.7 | 1484.4 | AITER flydsl contiguous 1516.8 |
| 8192x53248x16384 | bfloat16 | 6656 | data-parallel |  | 1557.6 | 1553.9 | 1557.6 | 1553.9 | 1570.5 | 1541.4 | flydsl | 1524.7 | 1406.0 (copy) | 1443.4 | 1504.5 | AITER flydsl contiguous 1524.7 |
| 16384x2304x16384 | bfloat16 | 576 | one-tile owner, n = 4 |  | 1486.5 | 1416.0 | 1314.8 | 1211.6 | 1297.5 | 1213.1 | torch | 1336.5 | 1334.5 | 1350.3 | 1336.0 | hipBLASLt contiguous 1350.3 |
| 16384x4608x16384 | bfloat16 | 1152 | one-tile owner, n = 2 |  | 1542.7 | 1496.6 | 1483.4 | 1405.7 | 1464.1 | 1421.7 | flydsl | 1423.2 | 1248.5 (copy) | 1179.1 | 1161.8 | AITER flydsl contiguous 1423.2 |
| 16384x9216x16384 | bfloat16 | 2304 | data-parallel |  | 1576.4 | 1518.1 | 1576.4 | 1518.1 | 1568.9 | 1509.9 | flydsl | 1508.4 | 1399.9 (copy) | 1531.3 | 1485.3 | hipBLASLt contiguous 1531.3 |
| 16384x13312x16384 | bfloat16 | 3328 | data-parallel |  | 1572.2 | 1518.6 | 1572.2 | 1518.6 | 1567.8 | 1510.6 | asm | 1519.4 | 1533.8 | 1462.5 | 1478.1 | AITER asm padded 1533.8 |
| 16384x26624x16384 | bfloat16 | 6656 | data-parallel |  | 1564.9 | 1515.2 | 1564.9 | 1515.2 | 1570.9 | 1517.0 | flydsl | 1518.7 | 1453.6 (copy) | 1453.8 | 1504.5 | AITER flydsl contiguous 1518.7 |
| 16384x53248x16384 | bfloat16 | 13312 | data-parallel |  | 1553.6 | 1556.0 | 1553.6 | 1556.0 | 1569.7 | 1548.3 | asm | 1522.2 | 1536.8 | 1432.0 | 1464.2 | AITER asm padded 1536.8 |
| 32768x2304x16384 | bfloat16 | 1152 | one-tile owner, n = 2 |  | 1524.3 | 1516.8 | 1466.7 | 1422.5 | 1449.5 | 1409.6 | flydsl | 1410.2 | 1152.4 (copy) | 1183.2 | 1163.9 | AITER flydsl contiguous 1410.2 |
| 32768x4608x16384 | bfloat16 | 2304 | data-parallel |  | 1566.1 | 1542.1 | 1566.1 | 1542.1 | 1562.8 | 1535.9 | torch | 1514.1 | 1516.9 | 1525.8 | 1517.3 | hipBLASLt contiguous 1525.8 |
| 32768x9216x16384 | bfloat16 | 4608 | data-parallel |  | 1563.5 | 1545.3 | 1563.5 | 1545.3 | 1569.3 | 1539.7 | asm | 1485.1 | 1501.3 | 1454.4 | 1474.5 | AITER asm padded 1501.3 |
| 32768x13312x16384 | bfloat16 | 6656 | data-parallel |  | 1548.7 | 1540.2 | 1548.7 | 1540.2 | 1565.4 | 1541.3 | asm | 1495.0 | 1538.8 | 1444.2 | 1511.5 | AITER asm padded 1538.8 |
| 32768x26624x16384 | bfloat16 | 13312 | data-parallel |  | 1560.8 | 1562.5 | 1560.8 | 1562.5 | 1572.3 | 1553.9 | flydsl | 1527.2 | 1476.3 (copy) | 1432.1 | 1486.0 | AITER flydsl contiguous 1527.2 |
| 32768x53248x16384 | bfloat16 | 26624 | data-parallel |  | 1571.1 | 1568.8 | 1571.1 | 1568.8 | 1568.9 | 1559.7 | flydsl | 1522.3 | 1492.3 (copy) | 1452.2 | 1487.9 | AITER flydsl contiguous 1522.3 |
| 512x6144x18432 | bfloat16 | 48 | one-tile owner, n = 5-6 |  | 966.7 | 980.0 | 354.1 | 361.6 | 355.5 | 362.9 | flydsl | 787.8 | 487.0 (copy) | 838.4 | 819.6 | hipBLASLt contiguous 838.4 |
| 1024x6144x18432 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 1125.8 | 1136.8 | 711.3 | 703.0 | 720.3 | 703.5 | torch | 1180.0 | 1190.1 | 1186.3 | 1190.1 | AITER torch padded 1190.1 |
| 2048x6144x18432 | bfloat16 | 192 | data-parallel | yes | 1342.9 | 1341.7 | 1341.4 | 1336.5 | 1339.8 | 1344.9 | torch | 1333.7 | 1357.2 | 1350.6 | 1361.0 | hipBLASLt padded 1361.0 |
| 4096x6144x18432 | bfloat16 | 384 | one-tile owner, n = 2 |  | 1548.7 | 1555.5 | 1363.7 | 1369.0 | 1356.3 | 1365.3 | torch | 1328.0 | 1359.0 | 1347.5 | 1359.3 | hipBLASLt padded 1359.3 |
| 8192x6144x18432 | bfloat16 | 768 | data-parallel |  | 1584.4 | 1610.2 | 1584.4 | 1610.2 | 1581.1 | 1598.9 | torch | 1499.8 | 1543.7 | 1508.3 | 1550.1 | hipBLASLt padded 1550.1 |
| 16384x6144x18432 | bfloat16 | 1536 | data-parallel |  | 1595.6 | 1613.7 | 1595.6 | 1613.7 | 1587.5 | 1600.9 | asm | 1536.1 | 1555.9 | 1527.9 | 1561.9 | hipBLASLt padded 1561.9 |
| 32768x6144x18432 | bfloat16 | 3072 | data-parallel |  | 1592.5 | 1586.0 | 1592.5 | 1586.0 | 1589.9 | 1595.7 | asm | 1543.4 | 1542.4 | 1480.5 | 1495.2 | AITER asm contiguous 1543.4 |
| 512x5120x25600 | bfloat16 | 40 | one-tile owner, n = 6-7 |  | 951.2 | 985.3 | 294.0 | 308.0 | 295.2 | 309.1 | flydsl | 428.3 | 313.7 (copy) | 876.6 | 887.7 | hipBLASLt padded 887.7 |
| 1024x5120x25600 | bfloat16 | 80 | one-tile owner, n = 3-4 |  | 1271.2 | 1300.0 | 590.1 | 620.1 | 594.9 | 620.8 | torch | 1132.9 | 1166.0 | 1146.3 | 1166.5 | hipBLASLt padded 1166.5 |
| 2048x5120x25600 | bfloat16 | 160 | data-parallel | yes | 1170.6 | 1223.1 | 1179.8 | 1223.6 | 1177.4 | 1234.6 | torch | 1280.3 | 1302.3 | 1293.4 | 1302.9 | hipBLASLt padded 1302.9 |
| 4096x5120x25600 | bfloat16 | 320 | one-tile owner, n = 4 |  | 1532.7 | 1546.6 | 1157.6 | 1203.5 | 1160.5 | 1198.7 | torch | 1266.1 | 1291.1 | 1281.9 | 1293.0 | hipBLASLt padded 1293.0 |
| 8192x5120x25600 | bfloat16 | 640 | one-tile owner, n = 2 |  | 1568.1 | 1587.4 | 1438.5 | 1452.8 | 1419.6 | 1436.6 | asm | 1390.2 | 1406.9 | 1349.1 | 1365.7 | AITER asm padded 1406.9 |
| 16384x5120x25600 | bfloat16 | 1280 | data-parallel |  | 1601.6 | 1605.0 | 1601.6 | 1605.0 | 1588.8 | 1595.9 | torch | 1498.7 | 1570.3 | 1515.5 | 1570.3 | AITER torch padded 1570.3 |
| 32768x5120x25600 | bfloat16 | 2560 | data-parallel |  | 1579.8 | 1585.4 | 1579.8 | 1585.4 | 1591.1 | 1595.8 | torch | 1435.3 | 1432.6 | 1431.5 | 1429.1 | AITER torch contiguous 1435.3 |
| 512x16384x26624 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1366.3 | 1376.8 | 942.1 | 941.5 | 937.3 | 933.3 | flydsl | 650.8 | 437.6 (copy) | 1200.4 | 1196.8 | hipBLASLt contiguous 1200.4 |
| 1024x16384x26624 | bfloat16 | 256 | data-parallel | yes | 1564.0 | 1573.4 | 1561.9 | 1565.9 | 1557.5 | 1571.0 | flydsl | 753.1 | 583.5 (copy) | 1524.8 | 1537.1 | hipBLASLt padded 1537.1 |
| 2048x16384x26624 | bfloat16 | 512 | data-parallel |  | 1608.4 | 1611.6 | 1608.4 | 1611.6 | 1591.9 | 1606.2 | flydsl | 808.3 | 703.8 (copy) | 1532.4 | 1545.6 | hipBLASLt padded 1545.6 |
| 4096x16384x26624 | bfloat16 | 1024 | data-parallel |  | 1628.4 | 1623.8 | 1628.4 | 1623.8 | 1610.7 | 1617.9 | flydsl | 1557.7 | 1347.6 (copy) | 1537.0 | 1554.6 | AITER flydsl contiguous 1557.7 |
| 8192x16384x26624 | bfloat16 | 2048 | data-parallel |  | 1614.2 | 1620.0 | 1614.2 | 1620.0 | 1604.6 | 1605.5 | asm | 230.4 | 242.0 | 1527.6 | 1544.5 | hipBLASLt padded 1544.5 |
| 16384x16384x26624 | bfloat16 | 4096 | data-parallel |  | 1579.4 | 1594.3 | 1579.4 | 1594.3 | 1605.8 | 1610.9 | asm | 1531.0 | 1563.9 | 1466.9 | 1417.3 | AITER asm padded 1563.9 |
| 32768x16384x26624 | bfloat16 | 8192 | data-parallel |  | 1614.1 | 1494.0 | 1614.1 | 1494.0 | 1606.0 | 1597.9 | asm | 1546.1 | 1561.8 | 1441.8 | 1463.9 | AITER asm padded 1561.8 |
| 512x8192x28672 | bfloat16 | 64 | one-tile owner, n = 4 |  | 1215.6 | 1235.9 | 477.1 | 485.7 | 473.1 | 481.1 | flydsl | 785.8 | 483.1 (copy) | 919.8 | 917.8 | hipBLASLt contiguous 919.8 |
| 1024x8192x28672 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1435.7 | 1455.0 | 938.2 | 962.2 | 945.3 | 965.4 | triton | 694.3 | 695.4 | 1343.5 | 1348.4 | hipBLASLt padded 1348.4 |
| 2048x8192x28672 | bfloat16 | 256 | data-parallel | yes | 1603.6 | 1612.3 | 1607.4 | 1610.2 | 1575.8 | 1606.1 | flydsl | 1033.2 | 846.3 (copy) | 1569.2 | 1573.0 | hipBLASLt padded 1573.0 |
| 4096x8192x28672 | bfloat16 | 512 | data-parallel |  | 1619.2 | 1631.9 | 1619.2 | 1631.9 | 1609.5 | 1623.1 | flydsl | 1548.3 | 1297.0 (copy) | 1563.2 | 1582.2 | hipBLASLt padded 1582.2 |
| 8192x8192x28672 | bfloat16 | 1024 | data-parallel |  | 1614.9 | 1623.0 | 1614.9 | 1623.0 | 1601.8 | 1614.0 | asm | 1562.9 | 1578.7 | 1558.7 | 1579.3 | hipBLASLt padded 1579.3 |
| 16384x8192x28672 | bfloat16 | 2048 | data-parallel |  | 1609.5 | 1607.5 | 1609.5 | 1607.5 | 1604.2 | 1608.3 | asm | 1541.4 | 1572.8 | 1533.7 | 1534.8 | AITER asm padded 1572.8 |
| 32768x8192x28672 | bfloat16 | 4096 | data-parallel |  | 1607.1 | 1539.2 | 1607.1 | 1539.2 | 1606.5 | 1604.9 | asm | 1568.1 | 1567.6 | 1486.3 | 1474.4 | AITER asm contiguous 1568.1 |
| 512x7168x35840 | bfloat16 | 56 | one-tile owner, n = 4-5 |  | 1153.2 | 1137.0 | 438.7 | 409.4 | 438.2 | 406.9 | flydsl | 815.6 | 497.6 (copy) | 964.6 | 959.2 | hipBLASLt contiguous 964.6 |
| 1024x7168x35840 | bfloat16 | 112 | one-tile owner, n = 2-3 |  | 1349.7 | 1349.7 | 869.7 | 818.0 | 867.1 | 814.3 | flydsl | 574.1 | 463.0 (copy) | 1291.3 | 1302.9 | hipBLASLt padded 1302.9 |
| 2048x7168x35840 | bfloat16 | 224 | spread |  | 1500.4 | 1507.0 | 1499.6 | 1501.8 | 1490.7 | 1492.7 | asm | 1442.8 | 1460.4 | 1410.5 | 1412.1 | AITER asm padded 1460.4 |
| 4096x7168x35840 | bfloat16 | 448 | spread |  | 1572.0 | 1581.2 | 1515.3 | 1518.3 | 1499.7 | 1505.3 | flydsl | 1458.6 | 1220.5 (copy) | 1404.9 | 1406.2 | AITER flydsl contiguous 1458.6 |
| 8192x7168x35840 | bfloat16 | 896 | one-tile owner, n = 2 |  | 1596.2 | 1606.6 | 1490.8 | 1499.5 | 1472.4 | 1482.8 | asm | 1423.1 | 1431.9 | 1406.2 | 1413.7 | AITER asm padded 1431.9 |
| 16384x7168x35840 | bfloat16 | 1792 | data-parallel |  | 1565.4 | 1574.9 | 1565.4 | 1574.9 | 1528.0 | 1579.3 | torch | 1471.8 | 1492.4 | 1457.8 | 1491.5 | AITER torch padded 1492.4 |
| 32768x7168x35840 | bfloat16 | 3584 | data-parallel |  | 1612.9 | 1585.6 | 1612.9 | 1585.6 | 1599.3 | 1606.3 | flydsl | 1490.9 | 1354.0 (copy) | 1462.0 | 1503.4 | hipBLASLt padded 1503.4 |
