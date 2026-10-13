# streamk_wip (v20, capped, v9 for one-wave dp) vs AITER, v9 and hipBLASLt, contiguous and 512-byte-padded strides (warm)

- Stack: torch 2.10.0+rocm7.2.4.git3d3aa833, hipBLASLt 100202 (dabb6df2b9), Triton 3.8.0 @ /home/jbowley/repos/triton_gfx950-tutorial-v2.2/python/triton; GPU: PCI bus 117, gfx950:sramecc+:xnack-; AITER e2c7aa0aa [Triton/Gluon] gfx950 a16w16: skip the Gluon kernel on Triton without cd_regclass on a1455f1c4 2026-10-02 [CK] [FlyDSL] add mxfp8 a8w8 blockscale for gfx950 (#4951).
- streamk_wip: snapshot sha256 56e1e2af1bb77dfa (newest WORKLOG entry: "11. Padded row strides accepted; fp32 C stays off v9 (2026-10-06)"); the v9 it delegates to: live sha256 505b5e31f5d2099e.
- Layouts: contiguous, and padded: every row stride (x, w, C) that is a multiple of 512 bytes gets 256 more (x and w: K + 128 when K % 256 == 0; C: N + 128 for bf16, N + 64 for fp32). v9, streamk_wip and hipBLASLt (`torch.mm(x, w.t(), out=y)`) get padded x, w and C; AITER's pick allocates its own C, so it gets padded x and w only.
- Timing: 3 rounds, each with fresh inputs (new buffers) when `realloc` is on (here: True); median of the rounds. Spread = (slowest - fastest round) / median.
- Best library: the fastest valid time of AITER's pick and hipBLASLt; for the padded comparison over both layouts. A padded call that launches a copy kernel (FlyDSL makes its inputs contiguous) is not valid.
- "Faster" and "slower" mean a gap of more than 1%.

Padded calls that copied their inputs (left out of the best library): pick (flydsl) on 252 shapes.

The best library's time comes from the contiguous layout on 345 shapes, padded layout on 285 shapes.

## v9 and streamk_wip padded, libraries contiguous (as AITER calls them)

| Shapes | Count | v9: faster | v9: within 1% | v9: slower | v9: median | data-parallel only: faster | data-parallel only: within 1% | data-parallel only: slower | data-parallel only: median | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All shapes | 630 | 230 | 53 | 347 | -2.6% | 231 | 55 | 344 | -4.1% | 304 | 50 | 276 | +0.5% |
| Under 256 tiles (less than one wave) | 321 | 23 | 8 | 290 | -37.7% | 17 | 8 | 296 | -37.1% | 51 | 12 | 258 | -18.5% |
| 256 to 1023 tiles | 152 | 101 | 9 | 42 | +2.7% | 92 | 16 | 44 | +2.4% | 128 | 9 | 15 | +4.4% |
| 1024 or more tiles | 157 | 106 | 36 | 15 | +2.0% | 122 | 31 | 4 | +3.2% | 125 | 29 | 3 | +3.5% |
| Best library uses 256x256 | 260 | 191 | 43 | 26 | +2.4% | 195 | 46 | 19 | +2.9% | 215 | 37 | 8 | +3.4% |
| Best library uses another tile | 370 | 39 | 10 | 321 | -31.2% | 36 | 9 | 325 | -31.6% | 89 | 13 | 268 | -13.8% |

## Everything contiguous (comparable with Oct 5)

| Shapes | Count | v9: faster | v9: within 1% | v9: slower | v9: median | data-parallel only: faster | data-parallel only: within 1% | data-parallel only: slower | data-parallel only: median | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All shapes | 630 | 234 | 54 | 342 | -2.6% | 235 | 47 | 348 | -4.0% | 296 | 43 | 291 | +0.1% |
| Under 256 tiles (less than one wave) | 321 | 24 | 10 | 287 | -37.4% | 19 | 7 | 295 | -37.2% | 49 | 6 | 266 | -18.5% |
| 256 to 1023 tiles | 152 | 101 | 11 | 40 | +2.6% | 89 | 14 | 49 | +2.1% | 119 | 12 | 21 | +3.6% |
| 1024 or more tiles | 157 | 109 | 33 | 15 | +2.3% | 127 | 26 | 4 | +2.8% | 128 | 25 | 4 | +3.0% |
| Best library uses 256x256 | 260 | 194 | 42 | 24 | +2.6% | 198 | 39 | 23 | +2.6% | 208 | 38 | 14 | +2.8% |
| Best library uses another tile | 370 | 40 | 12 | 318 | -30.5% | 37 | 8 | 325 | -31.3% | 88 | 5 | 277 | -14.1% |

## v9 and streamk_wip padded, against the best library over both layouts (strictest)

| Shapes | Count | v9: faster | v9: within 1% | v9: slower | v9: median | data-parallel only: faster | data-parallel only: within 1% | data-parallel only: slower | data-parallel only: median | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All shapes | 630 | 217 | 59 | 354 | -2.7% | 225 | 58 | 347 | -4.1% | 299 | 52 | 279 | +0.4% |
| Under 256 tiles (less than one wave) | 321 | 22 | 7 | 292 | -37.7% | 16 | 7 | 298 | -37.2% | 51 | 10 | 260 | -19.0% |
| 256 to 1023 tiles | 152 | 101 | 8 | 43 | +2.2% | 91 | 17 | 44 | +1.8% | 126 | 11 | 15 | +4.0% |
| 1024 or more tiles | 157 | 94 | 44 | 19 | +1.6% | 118 | 34 | 5 | +2.5% | 122 | 31 | 4 | +3.0% |
| Best library uses 256x256 | 259 | 177 | 51 | 31 | +2.0% | 188 | 50 | 21 | +2.4% | 209 | 41 | 9 | +3.0% |
| Best library uses another tile | 371 | 40 | 8 | 323 | -31.2% | 37 | 8 | 326 | -31.7% | 90 | 11 | 270 | -14.2% |

"Data-parallel only" is `STREAMK_POLICY=dp`, the persistent loop without stream-K and without the v9 hand-off; where capped already runs data-parallel on the persistent loop it is streamk_wip's time.

## v9 delegation (118 shapes: capped picks data-parallel, at most 256 tiles, bf16 output)

streamk_wip (now v9) against the persistent loop it replaced and against AITER's v9, same sweep. Positive means streamk_wip is faster.

| Layout | Shapes | Against | faster | within 1% | slower | median | p10 | p90 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| contiguous | all (118) | persistent loop (dp) | 45 | 60 | 13 | +0.2% | -1.0% | +3.8% |
| contiguous | all (118) | AITER's v9 | 13 | 50 | 55 | -0.7% | -4.0% | +1.1% |
| contiguous | k-steps <= 32 (54) | persistent loop (dp) | 44 | 9 | 1 | +2.6% | +0.6% | +4.7% |
| contiguous | k-steps <= 32 (54) | AITER's v9 | 12 | 23 | 19 | +0.6% | -4.7% | +1.2% |
| contiguous | k-steps > 32 (64) | persistent loop (dp) | 1 | 51 | 12 | -0.1% | -1.2% | +0.4% |
| contiguous | k-steps > 32 (64) | AITER's v9 | 1 | 27 | 36 | -1.4% | -3.2% | +0.0% |
| padded | all (118) | persistent loop (dp) | 70 | 47 | 1 | +1.2% | +0.3% | +3.8% |
| padded | all (118) | AITER's v9 | 14 | 76 | 28 | -0.0% | -2.0% | +1.1% |
| padded | k-steps <= 32 (54) | persistent loop (dp) | 51 | 3 | 0 | +3.1% | +1.4% | +5.6% |
| padded | k-steps <= 32 (54) | AITER's v9 | 13 | 24 | 17 | +0.6% | -2.8% | +1.2% |
| padded | k-steps > 32 (64) | persistent loop (dp) | 19 | 44 | 1 | +0.8% | +0.2% | +1.5% |
| padded | k-steps > 32 (64) | AITER's v9 | 1 | 52 | 11 | -0.2% | -1.4% | +0.5% |

### Against `results/details_capped_hbl.csv` (contiguous, this run / that run - 1)

| Shapes | Count | streamk_wip | v9 (AITER) | AITER's pick | hipBLASLt |
|---|---:|---:|---:|---:|---:|
| Delegated to v9 now | 118 | +1.7% | -0.0% | +0.1% | +0.0% |
| Not delegated | 512 | -0.2% | +0.0% | -0.2% | +0.0% |

### The 23 same-tile shapes where streamk_wip was behind on Oct 5

| Shape | Tiles | k-steps | Delegated now | Oct 5: vs best | Now contiguous: vs best | Now padded: vs best | Now padded: vs v9 padded |
|---|---:|---:|---|---:|---:|---:|---:|
| 4096x4096x1024 | 256 | 16 | yes | -5.5% | -3.0% | -1.2% | -2.8% |
| 2048x7168x1024 | 224 | 16 | yes | -4.5% | -1.5% | +1.3% | -2.5% |
| 2048x5120x640 | 160 | 10 | yes | -4.5% | +0.2% | -0.6% | -0.4% |
| 4096x4096x1536 | 256 | 24 | yes | -3.7% | -1.6% | +0.6% | -1.2% |
| 4096x4096x2048 | 256 | 32 | yes | -3.6% | -0.6% | +2.2% | -1.0% |
| 2048x8192x1024 | 256 | 16 | yes | -3.4% | +0.5% | +2.2% | -2.9% |
| 4096x4096x512 | 256 | 8 | yes | -3.0% | +1.2% | +3.8% | -3.1% |
| 4096x8192x1024 | 512 | 16 | no | -3.0% | -2.6% | +0.5% | -1.2% |
| 1024x16384x2048 | 256 | 32 | yes | -2.9% | +1.3% | +2.4% | -1.1% |
| 32768x512x7168 | 256 | 112 | yes | -2.5% | -1.5% | -2.2% | -1.2% |
| 2048x8192x2048 | 256 | 32 | yes | -2.4% | +1.2% | +1.2% | -2.5% |
| 8192x2048x512 | 256 | 8 | yes | -2.4% | +1.8% | +4.4% | -2.1% |
| 2048x7168x1792 | 224 | 28 | yes | -2.3% | -1.2% | +1.4% | -2.9% |
| 16384x1024x7168 | 256 | 112 | yes | -2.2% | -1.2% | -0.4% | -0.1% |
| 16384x2048x7168 | 512 | 112 | no | -1.7% | -1.8% | -1.1% | +0.1% |
| 2048x16384x6656 | 512 | 104 | no | -1.6% | -1.7% | -1.5% | -0.3% |
| 16384x57344x8192 | 14336 | 128 | no | -1.6% | -1.9% | -1.4% | -4.0% |
| 4096x7168x1024 | 448 | 16 | no | -1.4% | -1.4% | +0.5% | -1.3% |
| 1024x14336x8192 | 224 | 128 | yes | -1.2% | +0.6% | +1.3% | -0.3% |
| 16384x16384x2048 | 4096 | 32 | no | -1.1% | -1.1% | -0.8% | +2.3% |
| 4096x7168x1792 | 448 | 28 | no | -1.1% | -0.6% | +1.3% | -0.2% |
| 4096x16384x6656 | 1024 | 104 | no | -1.1% | -0.9% | -0.3% | +0.5% |
| 1024x16384x8192 | 256 | 128 | yes | -1.0% | -0.3% | +0.8% | +0.1% |

## Padding: does it remove the noise?

Spread between the 3 rounds, per kernel and layout. With `realloc` every round has new buffers, so a timing that depends on where the buffers landed shows up here. Control: the 16 shapes with K % 256 == 128, whose x and w strides are not padded (only C is).

| Kernel | Shapes | contiguous: median spread | contiguous: > 2% | contiguous: > 5% | padded: median spread | padded: > 2% | padded: > 5% |
|---|---|---:|---:|---:|---:|---:|---:|
| v9 | All, strides padded (614) | 0.26% | 14 | 3 | 0.26% | 23 | 1 |
| v9 | Under 256 (less than one wave), strides padded (314) | 0.25% | 9 | 2 | 0.27% | 12 | 1 |
| v9 | 256 to 1023, strides padded (148) | 0.32% | 3 | 1 | 0.29% | 6 | 0 |
| v9 | 1024 or more, strides padded (152) | 0.23% | 2 | 0 | 0.23% | 5 | 0 |
| v9 | Control (K % 256 == 128) (16) | 0.45% | 0 | 0 | 0.36% | 1 | 0 |
| streamk_wip | All, strides padded (614) | 0.61% | 51 | 3 | 0.45% | 36 | 3 |
| streamk_wip | Under 256 (less than one wave), strides padded (314) | 0.79% | 37 | 1 | 0.59% | 25 | 2 |
| streamk_wip | 256 to 1023, strides padded (148) | 0.52% | 5 | 0 | 0.37% | 5 | 1 |
| streamk_wip | 1024 or more, strides padded (152) | 0.38% | 9 | 2 | 0.28% | 6 | 0 |
| streamk_wip | Control (K % 256 == 128) (16) | 0.75% | 1 | 0 | 0.66% | 2 | 0 |
| AITER's pick | All, strides padded (362) | 0.47% | 22 | 3 | 0.34% | 19 | 3 |
| AITER's pick | Under 256 (less than one wave), strides padded (168) | 0.63% | 14 | 1 | 0.42% | 11 | 2 |
| AITER's pick | 256 to 1023, strides padded (89) | 0.38% | 0 | 0 | 0.33% | 3 | 1 |
| AITER's pick | 1024 or more, strides padded (105) | 0.41% | 8 | 2 | 0.27% | 5 | 0 |
| AITER's pick | Control (K % 256 == 128) (16) | 0.62% | 1 | 0 | 0.26% | 0 | 0 |
| hipBLASLt | All, strides padded (614) | 0.43% | 58 | 16 | 0.36% | 64 | 12 |
| hipBLASLt | Under 256 (less than one wave), strides padded (314) | 0.48% | 39 | 10 | 0.43% | 47 | 9 |
| hipBLASLt | 256 to 1023, strides padded (148) | 0.31% | 0 | 0 | 0.27% | 2 | 0 |
| hipBLASLt | 1024 or more, strides padded (152) | 0.44% | 19 | 6 | 0.33% | 15 | 3 |
| hipBLASLt | Control (K % 256 == 128) (16) | 0.35% | 0 | 0 | 0.26% | 0 | 0 |

AITER's pick leaves out (on both layouts) the shapes where its padded call copied its inputs.

### Padded against contiguous, median time over the rounds

Padded TFLOPS / contiguous TFLOPS - 1 (positive: padding is faster).

| Kernel | Shapes | Count | faster | within 1% | slower | median | p10 | p90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| v9 | All, strides padded | 614 | 43 | 516 | 55 | -0.1% | -0.9% | +0.8% |
| v9 | Under 256 (less than one wave) | 314 | 17 | 269 | 28 | -0.3% | -0.9% | +0.6% |
| v9 | 256 to 1023 | 148 | 13 | 126 | 9 | -0.0% | -0.7% | +0.8% |
| v9 | 1024 or more | 152 | 13 | 121 | 18 | +0.0% | -1.2% | +1.0% |
| v9 | Control (K % 256 == 128) | 16 | 1 | 15 | 0 | -0.1% | -0.4% | +0.4% |
| streamk_wip | All, strides padded | 614 | 197 | 339 | 78 | +0.5% | -1.4% | +2.0% |
| streamk_wip | Under 256 (less than one wave) | 314 | 100 | 159 | 55 | +0.3% | -1.9% | +2.4% |
| streamk_wip | 256 to 1023 | 148 | 56 | 86 | 6 | +0.6% | -0.1% | +2.0% |
| streamk_wip | 1024 or more | 152 | 41 | 94 | 17 | +0.4% | -1.4% | +1.4% |
| streamk_wip | Control (K % 256 == 128) | 16 | 6 | 10 | 0 | +0.6% | -0.2% | +2.8% |
| AITER's pick | All, strides padded | 362 | 64 | 273 | 25 | +0.4% | -0.6% | +1.5% |
| AITER's pick | Under 256 (less than one wave) | 168 | 22 | 128 | 18 | +0.2% | -1.0% | +1.2% |
| AITER's pick | 256 to 1023 | 89 | 14 | 70 | 5 | +0.4% | -0.2% | +1.7% |
| AITER's pick | 1024 or more | 105 | 28 | 75 | 2 | +0.5% | -0.2% | +2.1% |
| AITER's pick | Control (K % 256 == 128) | 16 | 1 | 15 | 0 | +0.4% | +0.1% | +0.7% |
| hipBLASLt | All, strides padded | 614 | 103 | 414 | 97 | -0.0% | -1.5% | +1.4% |
| hipBLASLt | Under 256 (less than one wave) | 314 | 54 | 185 | 75 | -0.2% | -1.7% | +1.5% |
| hipBLASLt | 256 to 1023 | 148 | 21 | 120 | 7 | +0.1% | -0.4% | +1.2% |
| hipBLASLt | 1024 or more | 152 | 28 | 109 | 15 | +0.0% | -0.9% | +1.6% |
| hipBLASLt | Control (K % 256 == 128) | 16 | 0 | 15 | 1 | -0.0% | -0.5% | +0.3% |

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
| 1024x3072x512 | bfloat16 | 48 | data-parallel | yes | 227.3 | 226.9 | 213.5 | 212.2 | 226.3 | 226.3 | flydsl | 464.5 | 211.2 (copy) | 393.3 | 400.7 | AITER flydsl contiguous 464.5 |
| 1024x4096x512 | bfloat16 | 64 | data-parallel | yes | 287.1 | 286.3 | 272.0 | 271.8 | 285.8 | 284.9 | flydsl | 528.8 | 258.5 (copy) | 471.0 | 469.6 | AITER flydsl contiguous 528.8 |
| 1024x7168x512 | bfloat16 | 112 | data-parallel | yes | 464.9 | 455.5 | 438.5 | 431.8 | 460.9 | 452.3 | flydsl | 616.2 | 351.2 (copy) | 584.9 | 577.5 | AITER flydsl contiguous 616.2 |
| 2048x2048x512 | bfloat16 | 64 | data-parallel | yes | 286.9 | 286.2 | 272.0 | 270.1 | 285.1 | 284.8 | flydsl | 528.0 | 256.2 (copy) | 474.1 | 467.5 | AITER flydsl contiguous 528.0 |
| 2048x4096x512 | bfloat16 | 128 | data-parallel | yes | 501.3 | 502.9 | 480.3 | 485.0 | 497.5 | 497.6 | torch | 620.4 | 620.1 | 622.4 | 628.9 | hipBLASLt padded 628.9 |
| 4096x2048x512 | bfloat16 | 128 | data-parallel | yes | 499.2 | 503.6 | 479.3 | 486.9 | 495.8 | 499.2 | torch | 670.8 | 669.4 | 670.1 | 662.5 | AITER torch contiguous 670.8 |
| 4096x4096x512 | bfloat16 | 256 | data-parallel | yes | 821.5 | 843.0 | 797.2 | 798.1 | 862.3 | 870.1 | torch | 811.9 | 810.3 | 811.2 | 808.5 | AITER torch contiguous 811.9 |
| 4096x7168x512 | bfloat16 | 448 | data-parallel |  | 856.4 | 866.6 | 856.4 | 866.6 | 882.5 | 872.6 | flydsl | 861.7 | 683.9 (copy) | 771.9 | 764.7 | AITER flydsl contiguous 861.7 |
| 8192x2048x512 | bfloat16 | 256 | data-parallel | yes | 822.4 | 843.3 | 799.8 | 792.6 | 866.5 | 861.3 | torch | 808.0 | 800.9 | 803.7 | 804.5 | AITER torch contiguous 808.0 |
| 8192x4096x512 | bfloat16 | 512 | data-parallel |  | 899.6 | 923.5 | 899.6 | 923.5 | 930.6 | 934.3 | flydsl | 862.2 | 702.7 (copy) | 863.9 | 861.5 | hipBLASLt contiguous 863.9 |
| 16384x2048x512 | bfloat16 | 512 | data-parallel |  | 897.0 | 913.9 | 897.0 | 913.9 | 933.9 | 930.0 | torch | 860.8 | 859.0 | 862.0 | 857.8 | hipBLASLt contiguous 862.0 |
| 16384x4096x512 | bfloat16 | 1024 | data-parallel |  | 983.4 | 1009.7 | 983.4 | 1009.7 | 980.1 | 979.6 | torch | 943.1 | 942.8 | 948.5 | 946.9 | hipBLASLt contiguous 948.5 |
| 16384x7168x512 | bfloat16 | 1792 | data-parallel |  | 1023.6 | 1045.5 | 1023.6 | 1045.5 | 1013.8 | 1006.0 | torch | 987.7 | 982.9 | 993.6 | 987.7 | hipBLASLt contiguous 993.6 |
| 32768x2048x512 | bfloat16 | 1024 | data-parallel |  | 982.3 | 1004.4 | 982.3 | 1004.4 | 979.5 | 978.3 | torch | 937.0 | 933.3 | 944.7 | 941.6 | hipBLASLt contiguous 944.7 |
| 1024x5120x640 | bfloat16 | 80 | data-parallel | yes | 382.8 | 383.0 | 363.5 | 360.6 | 378.2 | 378.3 | torch | 532.2 | 533.2 | 531.3 | 532.0 | AITER torch padded 533.2 |
| 2048x5120x640 | bfloat16 | 160 | data-parallel | yes | 677.3 | 674.8 | 650.4 | 646.5 | 680.9 | 677.4 | flydsl | 676.2 | 679.1 | 562.3 | 556.2 | AITER flydsl padded 679.1 |
| 4096x5120x640 | bfloat16 | 320 | data-parallel |  | 762.3 | 784.0 | 762.3 | 784.0 | 754.8 | 753.5 | flydsl | 816.5 | 813.7 | 776.2 | 776.0 | AITER flydsl contiguous 816.5 |
| 8192x5120x640 | bfloat16 | 640 | data-parallel |  | 940.5 | 966.1 | 940.5 | 966.1 | 937.9 | 937.0 | torch | 899.2 | 902.4 | 908.2 | 902.2 | hipBLASLt contiguous 908.2 |
| 16384x5120x640 | bfloat16 | 1280 | data-parallel |  | 1091.6 | 1108.8 | 1091.6 | 1108.8 | 1081.1 | 1079.6 | torch | 1049.8 | 1053.8 | 1056.7 | 1057.2 | hipBLASLt padded 1057.2 |
| 32768x5120x640 | bfloat16 | 2560 | data-parallel |  | 1113.9 | 1121.0 | 1113.9 | 1121.0 | 1081.8 | 1076.9 | torch | 1088.7 | 1092.9 | 1095.0 | 1098.4 | hipBLASLt padded 1098.4 |
| 1024x7168x768 | bfloat16 | 112 | data-parallel | yes | 546.7 | 547.8 | 522.2 | 523.5 | 541.0 | 541.8 | flydsl | 714.5 | 429.5 (copy) | 678.1 | 678.7 | AITER flydsl contiguous 714.5 |
| 10240x7168x768 | bfloat16 | 1120 | data-parallel |  | 1095.7 | 1097.9 | 1095.7 | 1097.9 | 1068.1 | 1069.4 | opus | 403.6 | 409.7 | 864.4 | 863.1 | hipBLASLt contiguous 864.4 |
| 512x4096x1024 | bfloat16 | 32 | data-parallel | yes | 202.4 | 202.5 | 196.2 | 195.7 | 200.2 | 200.4 | torch | 490.9 | 487.3 | 492.2 | 486.0 | hipBLASLt contiguous 492.2 |
| 512x6144x1024 | bfloat16 | 48 | data-parallel | yes | 298.8 | 298.9 | 288.7 | 283.1 | 297.2 | 296.1 | flydsl | 592.6 | 298.1 (copy) | 570.8 | 580.2 | AITER flydsl contiguous 592.6 |
| 512x7168x1024 | bfloat16 | 56 | data-parallel | yes | 339.2 | 337.3 | 326.2 | 324.5 | 335.9 | 334.1 | flydsl | 641.5 | 331.6 (copy) | 606.9 | 606.5 | AITER flydsl contiguous 641.5 |
| 512x8192x1024 | bfloat16 | 64 | data-parallel | yes | 378.5 | 376.7 | 364.9 | 363.8 | 374.9 | 374.0 | flydsl | 699.9 | 363.2 (copy) | 665.7 | 660.6 | AITER flydsl contiguous 699.9 |
| 1024x4096x1024 | bfloat16 | 64 | data-parallel | yes | 378.7 | 376.3 | 367.0 | 364.6 | 374.0 | 372.3 | torch | 661.7 | 661.3 | 665.4 | 662.2 | hipBLASLt contiguous 665.4 |
| 1024x7168x1024 | bfloat16 | 112 | data-parallel | yes | 617.2 | 614.5 | 594.3 | 592.8 | 611.4 | 608.0 | flydsl | 781.8 | 498.4 (copy) | 774.9 | 769.8 | AITER flydsl contiguous 781.8 |
| 1024x8192x1024 | bfloat16 | 128 | data-parallel | yes | 677.8 | 687.6 | 660.1 | 665.7 | 671.6 | 683.9 | torch | 844.1 | 843.8 | 849.5 | 871.4 | hipBLASLt padded 871.4 |
| 2048x4096x1024 | bfloat16 | 128 | data-parallel | yes | 677.3 | 681.3 | 660.8 | 659.5 | 671.6 | 677.1 | torch | 846.3 | 847.6 | 850.3 | 857.9 | hipBLASLt padded 857.9 |
| 2048x7168x1024 | bfloat16 | 224 | data-parallel | yes | 1017.2 | 1046.4 | 1009.0 | 1009.9 | 1071.4 | 1073.3 | flydsl | 1032.7 | 725.3 (copy) | 1016.0 | 1014.1 | AITER flydsl contiguous 1032.7 |
| 2048x8192x1024 | bfloat16 | 256 | data-parallel | yes | 1104.3 | 1122.7 | 1079.8 | 1093.4 | 1132.2 | 1156.5 | flydsl | 1098.4 | 796.0 (copy) | 1061.8 | 1084.6 | AITER flydsl contiguous 1098.4 |
| 3072x4096x1024 | bfloat16 | 192 | data-parallel | yes | 923.2 | 937.1 | 910.3 | 903.1 | 956.1 | 953.1 | torch | 975.7 | 974.8 | 982.2 | 977.9 | hipBLASLt contiguous 982.2 |
| 4096x4096x1024 | bfloat16 | 256 | data-parallel | yes | 1101.9 | 1122.0 | 1084.7 | 1089.6 | 1149.3 | 1154.7 | torch | 1128.2 | 1127.3 | 1136.0 | 1133.2 | hipBLASLt contiguous 1136.0 |
| 4096x7168x1024 | bfloat16 | 448 | data-parallel |  | 1089.1 | 1109.8 | 1089.1 | 1109.8 | 1129.0 | 1124.5 | flydsl | 1104.2 | 896.6 (copy) | 1036.0 | 1034.5 | AITER flydsl contiguous 1104.2 |
| 4096x8192x1024 | bfloat16 | 512 | data-parallel |  | 1156.1 | 1192.0 | 1156.1 | 1192.0 | 1186.2 | 1205.9 | asm | 1186.6 | 1184.5 | 1145.4 | 1162.3 | AITER asm contiguous 1186.6 |
| 5120x4096x1024 | bfloat16 | 320 | data-parallel |  | 879.1 | 900.4 | 879.1 | 900.4 | 893.4 | 892.3 | torch | 1019.9 | 1021.0 | 1023.8 | 1021.3 | hipBLASLt contiguous 1023.8 |
| 7680x4096x1024 | bfloat16 | 480 | data-parallel |  | 1115.5 | 1150.4 | 1115.5 | 1150.4 | 1159.7 | 1162.6 | torch | 983.0 | 986.1 | 984.4 | 981.2 | AITER torch padded 986.1 |
| 8192x7168x1024 | bfloat16 | 896 | data-parallel |  | 1158.6 | 1176.8 | 1158.6 | 1176.8 | 1163.3 | 1156.4 | asm | 1103.0 | 1103.7 | 1059.0 | 1054.9 | AITER asm padded 1103.7 |
| 8192x8192x1024 | bfloat16 | 1024 | data-parallel |  | 1231.8 | 1262.6 | 1231.8 | 1262.6 | 1222.7 | 1232.3 | asm | 1202.7 | 1201.7 | 1205.0 | 1207.4 | hipBLASLt padded 1207.4 |
| 16384x4096x1024 | bfloat16 | 1024 | data-parallel |  | 1238.6 | 1252.0 | 1238.6 | 1252.0 | 1243.3 | 1238.7 | torch | 1205.4 | 1208.2 | 1214.7 | 1210.9 | hipBLASLt contiguous 1214.7 |
| 16384x7168x1024 | bfloat16 | 1792 | data-parallel |  | 1277.4 | 1292.9 | 1277.4 | 1292.9 | 1260.2 | 1255.2 | asm | 1199.2 | 1202.7 | 1243.7 | 1243.9 | hipBLASLt padded 1243.9 |
| 16384x8192x1024 | bfloat16 | 2048 | data-parallel |  | 1265.1 | 1270.8 | 1265.1 | 1270.8 | 1241.9 | 1245.7 | torch | 1233.5 | 1238.1 | 1243.3 | 1245.6 | hipBLASLt padded 1245.6 |
| 32768x7168x1024 | bfloat16 | 3584 | data-parallel |  | 1260.3 | 1257.1 | 1260.3 | 1257.1 | 1239.7 | 1231.0 | torch | 1249.3 | 1257.9 | 1261.0 | 1260.0 | hipBLASLt contiguous 1261.0 |
| 32768x8192x1024 | bfloat16 | 4096 | data-parallel |  | 1263.9 | 1262.2 | 1263.9 | 1262.2 | 1228.3 | 1226.0 | torch | 1249.9 | 1258.4 | 1260.6 | 1261.6 | hipBLASLt padded 1261.6 |
| 512x5120x1280 | bfloat16 | 40 | data-parallel | yes | 265.5 | 267.8 | 255.9 | 258.5 | 263.6 | 265.2 | flydsl | 543.1 | 287.2 (copy) | 570.6 | 563.9 | hipBLASLt contiguous 570.6 |
| 1024x5120x1280 | bfloat16 | 80 | data-parallel | yes | 492.5 | 493.0 | 472.9 | 477.9 | 487.2 | 489.1 | torch | 720.5 | 730.1 | 720.5 | 730.5 | hipBLASLt padded 730.5 |
| 2048x5120x1280 | bfloat16 | 160 | data-parallel | yes | 872.7 | 881.9 | 863.0 | 854.2 | 882.9 | 883.3 | torch | 945.5 | 946.1 | 946.6 | 942.0 | hipBLASLt contiguous 946.6 |
| 4096x5120x1280 | bfloat16 | 320 | data-parallel |  | 944.2 | 966.4 | 944.2 | 966.4 | 950.4 | 949.0 | torch | 995.1 | 993.9 | 997.6 | 999.1 | hipBLASLt padded 999.1 |
| 8192x5120x1280 | bfloat16 | 640 | data-parallel |  | 1172.6 | 1194.3 | 1172.6 | 1194.3 | 1177.3 | 1183.1 | flydsl | 1151.6 | 973.3 (copy) | 1135.6 | 1133.4 | AITER flydsl contiguous 1151.6 |
| 16384x5120x1280 | bfloat16 | 1280 | data-parallel |  | 1330.0 | 1345.0 | 1330.0 | 1345.0 | 1317.6 | 1320.4 | flydsl | 1258.4 | 1097.7 (copy) | 1289.5 | 1291.5 | hipBLASLt padded 1291.5 |
| 32768x5120x1280 | bfloat16 | 2560 | data-parallel |  | 1307.1 | 1310.7 | 1307.1 | 1310.7 | 1287.7 | 1292.6 | torch | 1298.6 | 1308.5 | 1307.0 | 1309.0 | hipBLASLt padded 1309.0 |
| 512x1536x1536 | bfloat16 | 12 | one-tile reduce-scatter, n = 8 |  | 103.1 | 105.8 | 84.4 | 84.7 | 85.7 | 85.7 | flydsl | 348.7 | 155.9 (copy) | 306.3 | 310.4 | AITER flydsl contiguous 348.7 |
| 512x2048x1536 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 |  | 128.6 | 131.1 | 112.3 | 112.4 | 113.4 | 113.2 | torch | 402.5 | 400.3 | 403.2 | 403.6 | hipBLASLt padded 403.6 |
| 512x2304x1536 | bfloat16 | 18 | one-tile reduce-scatter, n = 7 |  | 132.3 | 134.1 | 125.6 | 125.7 | 127.0 | 126.6 | flydsl | 393.1 | 205.6 (copy) | 383.8 | 383.6 | AITER flydsl contiguous 393.1 |
| 512x4096x1536 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 212.7 | 215.7 | 222.2 | 220.9 | 224.8 | 223.3 | torch | 574.7 | 577.3 | 577.2 | 573.1 | AITER torch padded 577.3 |
| 512x6144x1536 | bfloat16 | 48 | data-parallel | yes | 331.6 | 332.1 | 323.3 | 325.2 | 328.0 | 330.1 | flydsl | 647.2 | 367.8 (copy) | 668.6 | 662.9 | hipBLASLt contiguous 668.6 |
| 512x7168x1536 | bfloat16 | 56 | data-parallel | yes | 379.8 | 378.9 | 366.4 | 367.7 | 375.1 | 374.4 | flydsl | 683.8 | 388.0 (copy) | 591.9 | 583.7 | AITER flydsl contiguous 683.8 |
| 1024x1536x1536 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 |  | 173.7 | 174.1 | 165.1 | 166.1 | 168.7 | 169.4 | flydsl | 501.2 | 276.0 (copy) | 480.3 | 480.5 | AITER flydsl contiguous 501.2 |
| 1024x2048x1536 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 211.9 | 215.0 | 217.5 | 220.3 | 221.2 | 222.8 | torch | 574.6 | 574.6 | 574.7 | 563.7 | hipBLASLt contiguous 574.7 |
| 1024x2304x1536 | bfloat16 | 36 | one-tile owner, n = 2 |  | 239.5 | 241.6 | 244.4 | 246.3 | 248.6 | 248.8 | torch | 533.6 | 532.6 | 539.6 | 529.7 | hipBLASLt contiguous 539.6 |
| 1024x3072x1536 | bfloat16 | 48 | data-parallel | yes | 330.3 | 330.9 | 320.0 | 322.0 | 328.6 | 328.2 | flydsl | 646.0 | 372.9 (copy) | 670.2 | 663.2 | hipBLASLt contiguous 670.2 |
| 1024x4096x1536 | bfloat16 | 64 | data-parallel | yes | 426.7 | 425.4 | 414.4 | 416.6 | 420.7 | 419.3 | torch | 636.4 | 633.8 | 638.2 | 633.7 | hipBLASLt contiguous 638.2 |
| 1024x7168x1536 | bfloat16 | 112 | data-parallel | yes | 698.1 | 695.3 | 681.8 | 685.5 | 690.1 | 691.7 | torch | 888.3 | 895.2 | 894.7 | 900.2 | hipBLASLt padded 900.2 |
| 2048x1536x1536 | bfloat16 | 48 | data-parallel | yes | 331.4 | 330.3 | 319.7 | 322.6 | 328.6 | 327.4 | torch | 669.1 | 655.2 | 672.5 | 662.4 | hipBLASLt contiguous 672.5 |
| 2048x2048x1536 | bfloat16 | 64 | data-parallel | yes | 426.5 | 425.6 | 415.1 | 416.4 | 420.9 | 419.7 | torch | 638.1 | 635.6 | 639.4 | 634.4 | hipBLASLt contiguous 639.4 |
| 2048x4096x1536 | bfloat16 | 128 | data-parallel | yes | 771.3 | 776.6 | 752.9 | 758.1 | 764.7 | 767.8 | torch | 953.6 | 957.9 | 958.1 | 967.5 | hipBLASLt padded 967.5 |
| 3072x2048x1536 | bfloat16 | 96 | data-parallel | yes | 609.5 | 609.7 | 595.4 | 594.0 | 603.5 | 602.9 | opus | 879.5 | 886.1 | 817.0 | 823.4 | AITER opus padded 886.1 |
| 3072x4096x1536 | bfloat16 | 192 | data-parallel | yes | 1037.7 | 1057.8 | 1022.2 | 1023.3 | 1090.7 | 1073.7 | opus | 1138.3 | 1137.4 | 1099.3 | 1107.7 | AITER opus contiguous 1138.3 |
| 4096x1536x1536 | bfloat16 | 96 | data-parallel | yes | 610.5 | 608.5 | 595.1 | 593.8 | 603.6 | 603.3 | torch | 820.8 | 818.9 | 821.1 | 819.1 | hipBLASLt contiguous 821.1 |
| 4096x2048x1536 | bfloat16 | 128 | data-parallel | yes | 769.9 | 775.7 | 753.4 | 755.8 | 764.4 | 769.6 | opus | 1023.8 | 1028.0 | 959.7 | 969.5 | AITER opus padded 1028.0 |
| 4096x3072x1536 | bfloat16 | 192 | data-parallel | yes | 1023.6 | 1058.2 | 1025.2 | 1028.1 | 1078.7 | 1074.3 | flydsl | 1046.1 | 773.8 (copy) | 1101.1 | 1101.5 | hipBLASLt padded 1101.5 |
| 4096x4096x1536 | bfloat16 | 256 | data-parallel | yes | 1211.0 | 1246.0 | 1215.4 | 1226.7 | 1270.0 | 1260.8 | torch | 1228.5 | 1231.5 | 1230.3 | 1238.1 | hipBLASLt padded 1238.1 |
| 5120x2048x1536 | bfloat16 | 160 | data-parallel | yes | 902.0 | 935.7 | 896.5 | 901.9 | 927.7 | 927.6 | torch | 1036.7 | 1036.6 | 1035.8 | 1047.7 | hipBLASLt padded 1047.7 |
| 5120x4096x1536 | bfloat16 | 320 | data-parallel |  | 983.5 | 1001.0 | 983.5 | 1001.0 | 989.0 | 988.9 | torch | 1095.6 | 1095.9 | 1097.5 | 1100.7 | hipBLASLt padded 1100.7 |
| 7680x2048x1536 | bfloat16 | 240 | data-parallel | yes | 1172.3 | 1186.0 | 1170.3 | 1165.9 | 1217.3 | 1209.8 | torch | 1040.4 | 1048.7 | 1042.3 | 1043.8 | AITER torch padded 1048.7 |
| 7680x4096x1536 | bfloat16 | 480 | data-parallel |  | 1239.1 | 1263.0 | 1239.1 | 1263.0 | 1258.2 | 1269.1 | torch | 1085.6 | 1092.5 | 1091.6 | 1090.7 | AITER torch padded 1092.5 |
| 8192x1536x1536 | bfloat16 | 192 | data-parallel | yes | 1055.2 | 1057.6 | 1024.3 | 1023.2 | 1075.5 | 1071.9 | torch | 1097.0 | 1097.5 | 1097.8 | 1101.7 | hipBLASLt padded 1101.7 |
| 10240x7168x1536 | bfloat16 | 1120 | data-parallel |  | 1275.6 | 1290.5 | 1275.6 | 1290.5 | 1255.9 | 1256.1 | asm | 1218.9 | 1221.2 | 1173.6 | 1171.8 | AITER asm padded 1221.2 |
| 16384x1536x1536 | bfloat16 | 384 | data-parallel |  | 1108.4 | 1130.7 | 1108.4 | 1130.7 | 1136.5 | 1133.8 | torch | 1159.9 | 1162.3 | 1154.4 | 1157.7 | AITER torch padded 1162.3 |
| 16384x2048x1536 | bfloat16 | 512 | data-parallel |  | 1278.8 | 1313.5 | 1278.8 | 1313.5 | 1314.5 | 1316.2 | torch | 1281.0 | 1285.9 | 1280.8 | 1287.6 | hipBLASLt padded 1287.6 |
| 16384x3072x1536 | bfloat16 | 768 | data-parallel |  | 1338.4 | 1350.5 | 1338.4 | 1350.5 | 1323.7 | 1329.8 | torch | 1303.6 | 1307.3 | 1303.0 | 1308.8 | hipBLASLt padded 1308.8 |
| 16384x4096x1536 | bfloat16 | 1024 | data-parallel |  | 1349.6 | 1370.4 | 1349.6 | 1370.4 | 1349.3 | 1352.1 | torch | 1318.7 | 1324.0 | 1321.4 | 1328.5 | hipBLASLt padded 1328.5 |
| 32768x1536x1536 | bfloat16 | 768 | data-parallel |  | 1324.9 | 1343.5 | 1324.9 | 1343.5 | 1327.9 | 1331.2 | flydsl | 1278.6 | 946.7 (copy) | 1305.7 | 1309.6 | hipBLASLt padded 1309.6 |
| 512x7168x1792 | bfloat16 | 56 | one-tile owner, n = 2 |  | 396.5 | 400.4 | 385.5 | 379.9 | 391.1 | 387.5 | flydsl | 718.5 | 418.4 (copy) | 604.1 | 597.9 | AITER flydsl contiguous 718.5 |
| 1024x7168x1792 | bfloat16 | 112 | data-parallel | yes | 722.8 | 723.8 | 703.4 | 714.1 | 721.5 | 713.7 | flydsl | 857.0 | 587.6 (copy) | 936.5 | 936.4 | hipBLASLt contiguous 936.5 |
| 2048x7168x1792 | bfloat16 | 224 | data-parallel | yes | 1181.7 | 1213.1 | 1200.1 | 1191.4 | 1249.3 | 1248.8 | flydsl | 1196.2 | 895.2 (copy) | 1146.8 | 1147.6 | AITER flydsl contiguous 1196.2 |
| 4096x7168x1792 | bfloat16 | 448 | data-parallel |  | 1254.5 | 1277.9 | 1254.5 | 1277.9 | 1285.6 | 1281.0 | flydsl | 1261.8 | 1037.7 (copy) | 1191.6 | 1192.4 | AITER flydsl contiguous 1261.8 |
| 8192x7168x1792 | bfloat16 | 896 | data-parallel |  | 1308.0 | 1321.5 | 1308.0 | 1321.5 | 1312.6 | 1306.5 | asm | 1256.5 | 1263.8 | 1209.0 | 1205.5 | AITER asm padded 1263.8 |
| 16384x7168x1792 | bfloat16 | 1792 | data-parallel |  | 1397.8 | 1403.2 | 1397.8 | 1403.2 | 1387.4 | 1388.9 | torch | 1373.5 | 1378.0 | 1383.5 | 1381.2 | hipBLASLt contiguous 1383.5 |
| 32768x7168x1792 | bfloat16 | 3584 | data-parallel |  | 1393.4 | 1400.0 | 1393.4 | 1400.0 | 1376.0 | 1373.6 | torch | 1380.9 | 1391.0 | 1395.0 | 1392.5 | hipBLASLt contiguous 1395.0 |
| 512x4096x2048 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 279.2 | 265.0 | 237.2 | 235.9 | 238.1 | 237.3 | torch | 610.3 | 603.7 | 616.8 | 606.2 | hipBLASLt contiguous 616.8 |
| 512x6144x2048 | float32 | 48 | one-tile owner, n = 2 |  | 341.6 | 337.7 | 307.5 | 306.3 | 315.8 | 313.4 | asm | 420.5 | 420.6 | 657.2 | 643.3 | hipBLASLt contiguous 657.2 |
| 512x6144x2048 | bfloat16 | 48 | one-tile owner, n = 2 |  | 380.1 | 373.0 | 348.6 | 346.9 | 353.8 | 349.5 | flydsl | 690.9 | 402.5 (copy) | 713.8 | 702.5 | hipBLASLt contiguous 713.8 |
| 512x8192x2048 | bfloat16 | 64 | one-tile owner, n = 2 |  | 485.1 | 480.9 | 450.1 | 453.2 | 452.9 | 449.0 | flydsl | 791.1 | 462.8 (copy) | 684.0 | 682.2 | AITER flydsl contiguous 791.1 |
| 512x16384x2048 | bfloat16 | 128 | data-parallel | yes | 828.3 | 834.5 | 821.0 | 839.5 | 825.1 | 867.0 | flydsl | 813.6 | 522.1 (copy) | 1032.3 | 1048.0 | hipBLASLt padded 1048.0 |
| 1024x4096x2048 | bfloat16 | 64 | one-tile owner, n = 2 |  | 485.8 | 481.3 | 449.2 | 448.0 | 452.4 | 447.9 | torch | 685.1 | 681.7 | 685.0 | 678.9 | AITER torch contiguous 685.1 |
| 1024x6144x2048 | float32 | 96 | data-parallel |  | 578.3 | 576.4 | 578.3 | 576.4 | 608.0 | 594.6 | opus | 668.7 | 651.2 | 900.2 | 903.8 | hipBLASLt padded 903.8 |
| 1024x6144x2048 | bfloat16 | 96 | data-parallel | yes | 659.7 | 654.1 | 649.7 | 642.9 | 650.4 | 645.4 | flydsl | 811.3 | 558.7 (copy) | 891.4 | 880.1 | hipBLASLt contiguous 891.4 |
| 1024x8192x2048 | bfloat16 | 128 | data-parallel | yes | 831.2 | 830.6 | 820.6 | 830.8 | 825.1 | 825.8 | flydsl | 1064.7 | 711.4 (copy) | 1039.6 | 1051.7 | AITER flydsl contiguous 1064.7 |
| 1024x16384x2048 | bfloat16 | 256 | data-parallel | yes | 1343.3 | 1357.6 | 1329.5 | 1332.1 | 1382.0 | 1372.7 | flydsl | 935.4 | 695.9 (copy) | 1325.6 | 1321.0 | hipBLASLt contiguous 1325.6 |
| 2048x4096x2048 | bfloat16 | 128 | data-parallel | yes | 829.8 | 826.4 | 821.0 | 820.5 | 827.3 | 823.2 | torch | 1034.6 | 1037.8 | 1039.1 | 1043.3 | hipBLASLt padded 1043.3 |
| 2048x6144x2048 | float32 | 192 | data-parallel |  | 1010.1 | 1010.4 | 1010.1 | 1010.4 | 1073.7 | 1072.8 | asm | 1032.3 | 1029.0 | 1083.7 | 1080.0 | hipBLASLt contiguous 1083.7 |
| 2048x6144x2048 | bfloat16 | 192 | data-parallel | yes | 1142.2 | 1144.2 | 1123.2 | 1119.1 | 1168.3 | 1184.5 | flydsl | 1142.6 | 844.7 (copy) | 1175.5 | 1177.9 | hipBLASLt padded 1177.9 |
| 2048x8192x2048 | bfloat16 | 256 | data-parallel | yes | 1330.8 | 1338.0 | 1317.9 | 1322.8 | 1361.1 | 1371.7 | flydsl | 1314.6 | 980.8 (copy) | 1306.6 | 1321.9 | hipBLASLt padded 1321.9 |
| 2048x16384x2048 | bfloat16 | 512 | data-parallel |  | 1398.5 | 1406.4 | 1398.5 | 1406.4 | 1400.3 | 1407.5 | flydsl | 1102.8 | 884.7 (copy) | 1361.7 | 1358.2 | hipBLASLt contiguous 1361.7 |
| 3072x4096x2048 | bfloat16 | 192 | data-parallel | yes | 1140.3 | 1138.0 | 1133.4 | 1114.1 | 1188.1 | 1151.6 | torch | 1177.5 | 1178.6 | 1178.7 | 1180.9 | hipBLASLt padded 1180.9 |
| 4096x4096x2048 | bfloat16 | 256 | data-parallel | yes | 1313.6 | 1354.7 | 1323.6 | 1326.3 | 1374.5 | 1368.3 | torch | 1317.8 | 1321.0 | 1321.8 | 1325.8 | hipBLASLt padded 1325.8 |
| 4096x6144x2048 | float32 | 384 | data-parallel |  | 1040.7 | 1049.1 | 1040.7 | 1049.1 | 1067.3 | 1063.8 | asm | 1020.2 | 1021.8 | 1119.2 | 1118.9 | hipBLASLt contiguous 1119.2 |
| 4096x6144x2048 | bfloat16 | 384 | data-parallel |  | 1185.7 | 1205.2 | 1185.7 | 1205.2 | 1210.6 | 1202.2 | flydsl | 1056.3 | 889.9 (copy) | 1213.8 | 1209.3 | hipBLASLt contiguous 1213.8 |
| 4096x8192x2048 | bfloat16 | 512 | data-parallel |  | 1376.4 | 1402.7 | 1376.4 | 1402.7 | 1391.0 | 1396.6 | flydsl | 1343.3 | 1122.4 (copy) | 1355.9 | 1366.8 | hipBLASLt padded 1366.8 |
| 4096x16384x2048 | bfloat16 | 1024 | data-parallel |  | 1438.0 | 1447.1 | 1438.0 | 1447.1 | 1427.5 | 1429.3 | flydsl | 968.2 | 859.0 (copy) | 1395.2 | 1393.3 | hipBLASLt contiguous 1395.2 |
| 5120x4096x2048 | bfloat16 | 320 | one-tile owner, n = 2 |  | 1058.6 | 1073.4 | 1048.0 | 1050.6 | 1048.8 | 1041.9 | torch | 1236.1 | 1232.4 | 1228.0 | 1227.4 | AITER torch contiguous 1236.1 |
| 7680x4096x2048 | bfloat16 | 480 | data-parallel |  | 1331.3 | 1358.2 | 1331.3 | 1358.2 | 1357.5 | 1356.5 | torch | 1143.5 | 1145.9 | 1148.5 | 1145.4 | hipBLASLt contiguous 1148.5 |
| 8192x6144x2048 | float32 | 768 | data-parallel |  | 1246.1 | 1260.9 | 1246.1 | 1260.9 | 1258.3 | 1260.8 | asm | 1221.1 | 1219.6 | 1165.1 | 1172.5 | AITER asm contiguous 1221.1 |
| 8192x6144x2048 | bfloat16 | 768 | data-parallel |  | 1415.8 | 1432.8 | 1415.8 | 1432.8 | 1419.0 | 1417.1 | flydsl | 855.7 | 781.9 (copy) | 1375.0 | 1376.3 | hipBLASLt padded 1376.3 |
| 8192x8192x2048 | bfloat16 | 1024 | data-parallel |  | 1421.9 | 1442.4 | 1421.9 | 1442.4 | 1420.3 | 1421.0 | asm | 1392.1 | 1396.5 | 1384.8 | 1395.9 | AITER asm padded 1396.5 |
| 8192x16384x2048 | bfloat16 | 2048 | data-parallel |  | 1400.2 | 1403.3 | 1400.2 | 1403.3 | 1382.0 | 1381.7 | triton | 1039.5 | 1030.7 | 1407.6 | 1406.6 | hipBLASLt contiguous 1407.6 |
| 16384x4096x2048 | bfloat16 | 1024 | data-parallel |  | 1436.7 | 1442.4 | 1436.7 | 1442.4 | 1428.2 | 1425.9 | torch | 1361.7 | 1369.1 | 1369.9 | 1373.6 | hipBLASLt padded 1373.6 |
| 16384x6144x2048 | float32 | 1536 | data-parallel |  | 1234.5 | 1239.7 | 1234.5 | 1239.7 | 1229.8 | 1227.4 | asm | 1182.6 | 1182.5 | 1142.5 | 1150.1 | AITER asm contiguous 1182.6 |
| 16384x6144x2048 | bfloat16 | 1536 | data-parallel |  | 1430.6 | 1448.7 | 1430.6 | 1448.7 | 1420.2 | 1434.4 | torch | 1394.4 | 1401.7 | 1403.7 | 1404.3 | hipBLASLt padded 1404.3 |
| 16384x8192x2048 | bfloat16 | 2048 | data-parallel |  | 1424.7 | 1429.3 | 1424.7 | 1429.3 | 1413.6 | 1413.0 | torch | 1400.4 | 1405.5 | 1406.6 | 1410.5 | hipBLASLt padded 1410.5 |
| 16384x16384x2048 | bfloat16 | 4096 | data-parallel |  | 1398.9 | 1403.4 | 1398.9 | 1403.4 | 1365.6 | 1371.3 | flydsl | 966.2 | 923.3 (copy) | 1414.9 | 1414.2 | hipBLASLt contiguous 1414.9 |
| 32768x6144x2048 | bfloat16 | 3072 | data-parallel |  | 1418.9 | 1436.7 | 1418.9 | 1436.7 | 1403.9 | 1391.2 | flydsl | 995.6 | 937.1 (copy) | 1410.7 | 1412.5 | hipBLASLt padded 1412.5 |
| 32768x8192x2048 | bfloat16 | 4096 | data-parallel |  | 1432.3 | 1437.4 | 1432.3 | 1437.4 | 1414.1 | 1406.2 | torch | 1402.1 | 1411.1 | 1413.7 | 1416.0 | hipBLASLt padded 1416.0 |
| 32768x16384x2048 | bfloat16 | 8192 | data-parallel |  | 1405.0 | 1413.9 | 1405.0 | 1413.9 | 1357.1 | 1371.5 | flydsl | 831.5 | 813.8 (copy) | 1420.8 | 1418.8 | hipBLASLt contiguous 1420.8 |
| 512x6144x3072 | float32 | 48 | one-tile owner, n = 2 |  | 422.5 | 428.0 | 340.9 | 342.4 | 345.9 | 345.7 | opus | 394.7 | 397.6 | 737.2 | 734.3 | hipBLASLt contiguous 737.2 |
| 512x6144x3072 | bfloat16 | 48 | one-tile owner, n = 2 |  | 461.3 | 466.4 | 379.0 | 378.5 | 379.9 | 378.4 | flydsl | 744.6 | 450.0 (copy) | 782.6 | 779.6 | hipBLASLt contiguous 782.6 |
| 1024x6144x3072 | float32 | 96 | one-tile owner, n = 2-3 |  | 675.3 | 701.5 | 661.8 | 663.2 | 687.6 | 670.1 | opus | 631.2 | 632.6 | 962.6 | 973.5 | hipBLASLt padded 973.5 |
| 1024x6144x3072 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 739.1 | 772.7 | 703.7 | 714.6 | 715.7 | 705.3 | flydsl | 890.3 | 628.7 (copy) | 952.5 | 955.7 | hipBLASLt padded 955.7 |
| 2048x6144x3072 | float32 | 192 | data-parallel |  | 1093.5 | 1150.2 | 1093.5 | 1150.2 | 1195.7 | 1194.3 | asm | 1147.5 | 1149.4 | 1190.1 | 1187.2 | hipBLASLt contiguous 1190.1 |
| 2048x6144x3072 | bfloat16 | 192 | data-parallel | yes | 1213.3 | 1253.3 | 1237.3 | 1243.4 | 1279.2 | 1288.3 | flydsl | 1217.2 | 939.8 (copy) | 1256.6 | 1254.9 | hipBLASLt contiguous 1256.6 |
| 4096x6144x3072 | float32 | 384 | data-parallel |  | 1144.2 | 1146.1 | 1144.2 | 1146.1 | 1165.6 | 1163.1 | asm | 1123.9 | 1126.9 | 1236.3 | 1234.1 | hipBLASLt contiguous 1236.3 |
| 4096x6144x3072 | bfloat16 | 384 | data-parallel |  | 1248.3 | 1265.2 | 1248.3 | 1265.2 | 1270.0 | 1271.7 | flydsl | 1108.4 | 937.8 (copy) | 1284.2 | 1282.3 | hipBLASLt contiguous 1284.2 |
| 8192x6144x3072 | float32 | 768 | data-parallel |  | 1364.1 | 1378.4 | 1364.1 | 1378.4 | 1366.9 | 1368.1 | asm | 1319.4 | 1320.9 | 1259.6 | 1264.5 | AITER asm padded 1320.9 |
| 8192x6144x3072 | bfloat16 | 768 | data-parallel |  | 1506.9 | 1516.3 | 1506.9 | 1516.3 | 1511.2 | 1515.5 | flydsl | 842.1 | 771.4 (copy) | 1454.6 | 1452.2 | hipBLASLt contiguous 1454.6 |
| 16384x6144x3072 | float32 | 1536 | data-parallel |  | 1330.5 | 1340.9 | 1330.5 | 1340.9 | 1322.3 | 1326.2 | asm | 1288.2 | 1295.9 | 1238.7 | 1242.1 | AITER asm padded 1295.9 |
| 16384x6144x3072 | bfloat16 | 1536 | data-parallel |  | 1494.7 | 1503.3 | 1494.7 | 1503.3 | 1489.9 | 1490.1 | torch | 1468.3 | 1475.1 | 1475.5 | 1474.6 | hipBLASLt contiguous 1475.5 |
| 32768x6144x3072 | bfloat16 | 3072 | data-parallel |  | 1507.5 | 1529.1 | 1507.5 | 1529.1 | 1486.1 | 1500.6 | flydsl | 862.3 | 806.2 (copy) | 1478.1 | 1476.9 | hipBLASLt contiguous 1478.1 |
| 512x5120x3200 | bfloat16 | 40 | one-tile owner, n = 2 |  | 408.4 | 409.7 | 320.1 | 320.0 | 319.7 | 319.3 | flydsl | 641.7 | 643.9 | 697.8 | 698.4 | hipBLASLt padded 698.4 |
| 1024x5120x3200 | bfloat16 | 80 | one-tile owner, n = 2 |  | 739.4 | 752.6 | 602.7 | 610.5 | 598.0 | 597.7 | torch | 954.1 | 957.2 | 955.5 | 958.1 | hipBLASLt padded 958.1 |
| 2048x5120x3200 | bfloat16 | 160 | data-parallel | yes | 1097.9 | 1131.1 | 1111.5 | 1086.4 | 1156.1 | 1158.6 | torch | 1173.4 | 1174.8 | 1172.7 | 1173.2 | AITER torch padded 1174.8 |
| 4096x5120x3200 | bfloat16 | 320 | one-tile owner, n = 2 |  | 1185.4 | 1199.3 | 1105.8 | 1111.4 | 1110.9 | 1106.1 | torch | 1198.1 | 1198.9 | 1203.9 | 1200.5 | hipBLASLt contiguous 1203.9 |
| 8192x5120x3200 | bfloat16 | 640 | data-parallel |  | 1367.6 | 1377.5 | 1367.6 | 1377.5 | 1378.9 | 1373.1 | asm | 1318.9 | 1324.0 | 1310.2 | 1308.5 | AITER asm padded 1324.0 |
| 16384x5120x3200 | bfloat16 | 1280 | data-parallel |  | 1509.6 | 1506.6 | 1509.6 | 1506.6 | 1498.1 | 1504.4 | torch | 1466.5 | 1473.5 | 1471.5 | 1472.3 | AITER torch padded 1473.5 |
| 32768x5120x3200 | bfloat16 | 2560 | data-parallel |  | 1513.4 | 1521.5 | 1513.4 | 1521.5 | 1487.5 | 1493.0 | torch | 1471.6 | 1480.6 | 1478.3 | 1478.3 | AITER torch padded 1480.6 |
| 512x7168x3584 | bfloat16 | 56 | one-tile owner, n = 2 |  | 570.2 | 578.6 | 447.2 | 447.4 | 446.5 | 445.7 | flydsl | 841.8 | 506.0 (copy) | 712.5 | 709.1 | AITER flydsl contiguous 841.8 |
| 512x8192x3584 | bfloat16 | 64 | one-tile owner, n = 2 |  | 641.8 | 651.3 | 502.7 | 508.1 | 502.1 | 501.7 | flydsl | 788.6 | 493.0 (copy) | 790.5 | 795.0 | hipBLASLt padded 795.0 |
| 1024x7168x3584 | bfloat16 | 112 | one-tile owner, n = 2-3 |  | 909.7 | 911.7 | 854.0 | 859.6 | 856.5 | 855.2 | torch | 1086.6 | 1091.4 | 1089.3 | 1095.4 | hipBLASLt padded 1095.4 |
| 1024x8192x3584 | bfloat16 | 128 | data-parallel | yes | 931.6 | 971.3 | 943.1 | 961.8 | 964.6 | 987.3 | flydsl | 886.2 | 658.6 (copy) | 1153.7 | 1158.6 | hipBLASLt padded 1158.6 |
| 2048x8192x3584 | bfloat16 | 256 | data-parallel | yes | 1459.1 | 1488.9 | 1466.8 | 1464.8 | 1488.4 | 1491.0 | flydsl | 1436.2 | 1069.2 (copy) | 1430.9 | 1449.2 | hipBLASLt padded 1449.2 |
| 4096x8192x3584 | bfloat16 | 512 | data-parallel |  | 1503.4 | 1521.7 | 1503.4 | 1521.7 | 1509.9 | 1521.0 | flydsl | 1034.5 | 895.7 (copy) | 1468.9 | 1476.5 | hipBLASLt padded 1476.5 |
| 8192x8192x3584 | bfloat16 | 1024 | data-parallel |  | 1541.0 | 1543.5 | 1541.0 | 1543.5 | 1531.9 | 1535.3 | torch | 1476.1 | 1482.8 | 1483.3 | 1485.6 | hipBLASLt padded 1485.6 |
| 10240x7168x3584 | bfloat16 | 1120 | one-tile owner, n = 2-3 |  | 1456.5 | 1456.2 | 1436.1 | 1433.9 | 1424.6 | 1425.1 | flydsl | 833.6 | 740.0 (copy) | 1341.5 | 1341.0 | hipBLASLt contiguous 1341.5 |
| 16384x8192x3584 | bfloat16 | 2048 | data-parallel |  | 1534.8 | 1540.9 | 1534.8 | 1540.9 | 1516.4 | 1517.5 | flydsl | 841.7 | 801.1 (copy) | 1493.7 | 1490.3 | hipBLASLt contiguous 1493.7 |
| 32768x8192x3584 | bfloat16 | 4096 | data-parallel |  | 1536.2 | 1552.6 | 1536.2 | 1552.6 | 1513.2 | 1519.6 | flydsl | 869.3 | 833.3 (copy) | 1484.7 | 1489.4 | hipBLASLt padded 1489.4 |
| 512x1024x4096 | bfloat16 | 8 | one-tile reduce-scatter, n = 8 |  | 179.5 | 173.3 | 66.9 | 67.3 | 66.7 | 66.6 | flydsl | 344.6 | 195.3 (copy) | 330.1 | 330.0 | AITER flydsl contiguous 344.6 |
| 512x2048x4096 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 |  | 293.7 | 287.9 | 134.1 | 133.5 | 132.9 | 132.2 | flydsl | 517.1 | 303.9 (copy) | 514.1 | 517.5 | hipBLASLt padded 517.5 |
| 512x4096x4096 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 442.4 | 425.2 | 266.3 | 265.6 | 263.8 | 262.5 | flydsl | 704.0 | 417.2 (copy) | 632.6 | 624.8 | AITER flydsl contiguous 704.0 |
| 512x8192x4096 | bfloat16 | 64 | one-tile owner, n = 2 |  | 684.6 | 686.7 | 518.6 | 516.8 | 515.8 | 515.2 | flydsl | 909.1 | 541.3 (copy) | 809.1 | 803.7 | AITER flydsl contiguous 909.1 |
| 512x8704x4096 | bfloat16 | 68 | one-tile owner, n = 2 |  | 705.1 | 719.0 | 549.2 | 550.0 | 550.6 | 546.0 | torch | 905.8 | 903.0 | 907.0 | 900.7 | hipBLASLt contiguous 907.0 |
| 512x10240x4096 | bfloat16 | 80 | one-tile owner, n = 2 |  | 797.1 | 793.8 | 644.8 | 632.8 | 639.3 | 630.6 | torch | 875.6 | 872.5 | 873.0 | 879.2 | hipBLASLt padded 879.2 |
| 512x16384x4096 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1064.4 | 1078.3 | 977.4 | 982.2 | 996.4 | 1010.8 | flydsl | 840.5 | 418.6 (copy) | 695.6 | 723.5 | AITER flydsl contiguous 840.5 |
| 512x129280x4096 | bfloat16 | 1010 | data-parallel |  | 1384.6 | 1382.0 | 1384.6 | 1382.0 | 1381.0 | 1374.3 | asm | 1303.6 | 1327.3 | 1143.9 | 1188.6 | AITER asm padded 1327.3 |
| 768x1024x4096 | bfloat16 | 12 | one-tile reduce-scatter, n = 8 |  | 246.2 | 235.5 | 100.6 | 100.3 | 100.3 | 99.7 | torch | 436.1 | 437.2 | 437.8 | 436.9 | hipBLASLt contiguous 437.8 |
| 768x2048x4096 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 |  | 369.6 | 358.4 | 200.1 | 199.8 | 199.5 | 198.1 | flydsl | 551.1 | 359.3 (copy) | 568.3 | 563.9 | hipBLASLt contiguous 568.3 |
| 768x4096x4096 | bfloat16 | 48 | one-tile owner, n = 3 |  | 581.0 | 586.6 | 397.2 | 395.9 | 395.9 | 392.6 | flydsl | 760.5 | 509.4 (copy) | 809.8 | 810.7 | hipBLASLt padded 810.7 |
| 768x8192x4096 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 841.9 | 872.9 | 766.5 | 765.9 | 760.1 | 762.1 | torch | 1009.6 | 1011.8 | 1008.6 | 1014.9 | hipBLASLt padded 1014.9 |
| 768x8704x4096 | bfloat16 | 102 | one-tile owner, n = 2-3 |  | 887.4 | 877.3 | 808.2 | 806.9 | 805.2 | 804.7 | flydsl | 956.4 | 648.4 (copy) | 986.8 | 981.6 | hipBLASLt contiguous 986.8 |
| 768x10240x4096 | bfloat16 | 120 | one-tile owner, n = 2-3 |  | 1005.6 | 1018.9 | 933.4 | 941.0 | 946.2 | 945.5 | flydsl | 1206.0 | 678.1 (copy) | 1080.6 | 1077.5 | AITER flydsl contiguous 1206.0 |
| 768x129280x4096 | bfloat16 | 1515 | data-parallel |  | 1439.4 | 1443.1 | 1439.4 | 1443.1 | 1423.8 | 1422.6 | torch | 1349.3 | 1344.6 | 1353.7 | 1349.7 | hipBLASLt contiguous 1353.7 |
| 1024x512x4096 | bfloat16 | 8 | one-tile reduce-scatter, n = 8 |  | 178.6 | 172.9 | 67.1 | 67.2 | 66.6 | 66.6 | flydsl | 327.1 | 190.3 (copy) | 329.8 | 329.7 | hipBLASLt contiguous 329.8 |
| 1024x1024x4096 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 |  | 293.6 | 287.9 | 134.2 | 133.9 | 133.1 | 132.3 | flydsl | 506.2 | 309.4 (copy) | 527.7 | 534.4 | hipBLASLt padded 534.4 |
| 1024x2048x4096 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 439.3 | 426.5 | 267.1 | 265.9 | 265.1 | 263.6 | flydsl | 700.0 | 452.9 (copy) | 547.6 | 559.7 | AITER flydsl contiguous 700.0 |
| 1024x4096x4096 | bfloat16 | 64 | one-tile owner, n = 2 |  | 683.7 | 690.9 | 521.1 | 518.8 | 515.3 | 513.6 | flydsl | 925.0 | 625.0 (copy) | 807.6 | 802.5 | AITER flydsl contiguous 925.0 |
| 1024x8192x4096 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1065.2 | 1082.2 | 985.1 | 993.1 | 965.2 | 1009.3 | flydsl | 1225.3 | 824.1 (copy) | 693.4 | 725.2 | AITER flydsl contiguous 1225.3 |
| 1024x8704x4096 | bfloat16 | 136 | data-parallel | yes | 1023.0 | 1047.9 | 1044.2 | 1032.6 | 1063.3 | 1064.3 | torch | 1137.3 | 1136.5 | 1130.2 | 1129.4 | AITER torch contiguous 1137.3 |
| 1024x10240x4096 | bfloat16 | 160 | data-parallel | yes | 1150.6 | 1170.2 | 1156.5 | 1159.3 | 1186.4 | 1178.4 | torch | 1214.1 | 1218.7 | 1220.4 | 1221.6 | hipBLASLt padded 1221.6 |
| 1024x16384x4096 | bfloat16 | 256 | data-parallel | yes | 1488.1 | 1512.9 | 1502.2 | 1501.6 | 1514.7 | 1514.9 | flydsl | 893.8 | 635.8 (copy) | 1481.9 | 1477.5 | hipBLASLt contiguous 1481.9 |
| 1024x129280x4096 | bfloat16 | 2020 | data-parallel |  | 1469.5 | 1472.6 | 1469.5 | 1472.6 | 1451.2 | 1449.1 | torch | 1374.0 | 1375.9 | 1382.6 | 1377.6 | hipBLASLt contiguous 1382.6 |
| 1280x512x4096 | bfloat16 | 10 | one-tile reduce-scatter, n = 8 |  | 213.1 | 204.0 | 83.7 | 83.9 | 83.5 | 83.4 | flydsl | 330.4 | 200.2 (copy) | 376.6 | 378.5 | hipBLASLt padded 378.5 |
| 1280x1024x4096 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 |  | 322.1 | 319.4 | 167.3 | 166.9 | 166.2 | 165.4 | torch | 561.2 | 565.7 | 566.7 | 568.0 | hipBLASLt padded 568.0 |
| 1280x2048x4096 | bfloat16 | 40 | one-tile owner, n = 3 |  | 498.8 | 498.8 | 332.2 | 329.1 | 330.2 | 328.6 | flydsl | 655.5 | 452.7 (copy) | 720.0 | 709.5 | hipBLASLt contiguous 720.0 |
| 1280x129280x4096 | bfloat16 | 2525 | data-parallel |  | 1439.0 | 1440.5 | 1439.0 | 1440.5 | 1424.8 | 1417.1 | torch | 1390.9 | 1388.4 | 1392.0 | 1390.2 | hipBLASLt contiguous 1392.0 |
| 1536x512x4096 | bfloat16 | 12 | one-tile reduce-scatter, n = 8 |  | 245.4 | 237.9 | 100.5 | 100.3 | 100.2 | 99.7 | torch | 434.8 | 435.4 | 435.5 | 435.5 | hipBLASLt contiguous 435.5 |
| 1536x1024x4096 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 |  | 370.7 | 364.3 | 200.0 | 199.5 | 199.4 | 197.8 | flydsl | 543.9 | 354.7 (copy) | 569.2 | 563.6 | hipBLASLt contiguous 569.2 |
| 1536x2048x4096 | bfloat16 | 48 | one-tile owner, n = 3 |  | 583.8 | 587.0 | 396.8 | 394.5 | 395.3 | 393.1 | flydsl | 760.1 | 533.7 (copy) | 808.7 | 808.6 | hipBLASLt contiguous 808.7 |
| 1536x129280x4096 | bfloat16 | 3030 | data-parallel |  | 1462.1 | 1466.3 | 1462.1 | 1466.3 | 1443.8 | 1436.9 | torch | 1398.0 | 1395.1 | 1406.0 | 1393.2 | hipBLASLt contiguous 1406.0 |
| 1792x512x4096 | bfloat16 | 14 | one-tile reduce-scatter, n = 8 |  | 271.1 | 262.9 | 117.2 | 117.1 | 117.1 | 116.5 | flydsl | 411.5 | 258.4 (copy) | 478.2 | 480.6 | hipBLASLt padded 480.6 |
| 1792x1024x4096 | bfloat16 | 28 | one-tile reduce-scatter, n = 4 |  | 398.1 | 388.8 | 233.8 | 232.9 | 232.9 | 231.6 | flydsl | 606.5 | 398.3 (copy) | 505.4 | 496.6 | AITER flydsl contiguous 606.5 |
| 1792x2048x4096 | bfloat16 | 56 | one-tile owner, n = 2 |  | 609.0 | 607.6 | 455.8 | 457.6 | 457.8 | 454.3 | flydsl | 828.7 | 593.0 (copy) | 745.0 | 745.7 | AITER flydsl contiguous 828.7 |
| 1792x129280x4096 | bfloat16 | 3535 | data-parallel |  | 1474.5 | 1476.0 | 1474.5 | 1476.0 | 1452.9 | 1449.2 | asm | 1437.0 | 1443.7 | 1359.8 | 1348.9 | AITER asm padded 1443.7 |
| 2048x512x4096 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 |  | 293.7 | 288.0 | 134.1 | 133.9 | 133.0 | 132.4 | flydsl | 481.3 | 292.5 (copy) | 526.4 | 524.4 | hipBLASLt contiguous 526.4 |
| 2048x1024x4096 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 440.7 | 427.2 | 266.4 | 266.1 | 264.6 | 262.6 | flydsl | 699.4 | 452.7 (copy) | 561.1 | 551.2 | AITER flydsl contiguous 699.4 |
| 2048x2048x4096 | bfloat16 | 64 | one-tile owner, n = 2 |  | 678.0 | 685.4 | 520.8 | 513.4 | 516.6 | 516.2 | flydsl | 909.3 | 646.6 (copy) | 807.4 | 802.8 | AITER flydsl contiguous 909.3 |
| 2048x4096x4096 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1060.6 | 1077.5 | 951.6 | 978.9 | 993.8 | 997.0 | opus | 1210.0 | 1222.5 | 766.5 | 780.6 | AITER opus padded 1222.5 |
| 2048x8192x4096 | bfloat16 | 256 | data-parallel | yes | 1491.2 | 1511.1 | 1496.4 | 1497.9 | 1517.0 | 1514.8 | flydsl | 1461.2 | 1083.7 (copy) | 1457.3 | 1469.9 | hipBLASLt padded 1469.9 |
| 2048x8704x4096 | bfloat16 | 272 | one-tile reduce-scatter, n = 8 |  | 1249.2 | 1262.3 | 992.6 | 995.6 | 992.6 | 987.9 | torch | 1144.0 | 1144.0 | 1154.1 | 1153.8 | hipBLASLt contiguous 1154.1 |
| 2048x10240x4096 | bfloat16 | 320 | one-tile owner, n = 2 |  | 1253.3 | 1259.1 | 1135.6 | 1136.4 | 1143.2 | 1130.3 | torch | 1240.4 | 1244.1 | 1244.1 | 1242.6 | hipBLASLt contiguous 1244.1 |
| 2048x16384x4096 | bfloat16 | 512 | data-parallel |  | 1535.9 | 1547.4 | 1535.9 | 1547.4 | 1535.7 | 1542.3 | flydsl | 1494.0 | 1120.7 (copy) | 1491.2 | 1487.6 | AITER flydsl contiguous 1494.0 |
| 2048x129280x4096 | bfloat16 | 4040 | data-parallel |  | 1508.6 | 1517.4 | 1508.6 | 1517.4 | 1486.0 | 1483.3 | flydsl | 1430.3 | 1153.8 (copy) | 1380.0 | 1372.5 | AITER flydsl contiguous 1430.3 |
| 2304x512x4096 | bfloat16 | 18 | one-tile reduce-scatter, n = 7 |  | 301.6 | 299.2 | 151.2 | 150.6 | 149.8 | 148.9 | flydsl | 440.1 | 283.1 (copy) | 525.5 | 531.1 | hipBLASLt padded 531.1 |
| 2304x1024x4096 | bfloat16 | 36 | one-tile owner, n = 3 |  | 456.3 | 454.3 | 299.7 | 298.1 | 296.9 | 295.8 | flydsl | 639.2 | 432.6 (copy) | 727.6 | 730.3 | hipBLASLt padded 730.3 |
| 2304x2048x4096 | bfloat16 | 72 | one-tile owner, n = 2 |  | 749.5 | 749.0 | 572.4 | 579.0 | 573.0 | 569.9 | opus | 881.3 | 884.9 | 951.5 | 953.6 | hipBLASLt padded 953.6 |
| 2560x512x4096 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 |  | 322.7 | 320.9 | 166.7 | 166.4 | 165.8 | 164.8 | flydsl | 480.4 | 303.5 (copy) | 566.3 | 569.6 | hipBLASLt padded 569.6 |
| 2560x1024x4096 | bfloat16 | 40 | one-tile owner, n = 3 |  | 494.3 | 497.8 | 331.8 | 329.5 | 330.0 | 327.9 | flydsl | 685.5 | 464.3 (copy) | 720.2 | 708.2 | hipBLASLt contiguous 720.2 |
| 2560x2048x4096 | bfloat16 | 80 | one-tile owner, n = 2 |  | 791.9 | 804.6 | 636.0 | 640.0 | 634.3 | 630.0 | opus | 958.8 | 966.0 | 1017.7 | 1015.2 | hipBLASLt contiguous 1017.7 |
| 2816x512x4096 | bfloat16 | 22 | one-tile reduce-scatter, n = 5 |  | 345.6 | 341.3 | 184.3 | 183.7 | 182.7 | 181.7 | flydsl | 515.7 | 330.4 (copy) | 533.5 | 524.2 | hipBLASLt contiguous 533.5 |
| 2816x1024x4096 | bfloat16 | 44 | one-tile owner, n = 3 |  | 542.5 | 543.8 | 365.5 | 362.9 | 363.2 | 361.5 | flydsl | 722.5 | 495.1 (copy) | 756.5 | 758.7 | hipBLASLt padded 758.7 |
| 2816x2048x4096 | bfloat16 | 88 | one-tile owner, n = 2-3 |  | 821.9 | 817.0 | 703.4 | 704.6 | 699.1 | 695.1 | opus | 1007.8 | 1009.5 | 930.6 | 929.7 | AITER opus padded 1009.5 |
| 3072x512x4096 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 |  | 366.4 | 361.9 | 200.1 | 199.3 | 199.6 | 198.0 | flydsl | 547.8 | 344.4 (copy) | 573.7 | 565.2 | hipBLASLt contiguous 573.7 |
| 3072x1024x4096 | bfloat16 | 48 | one-tile owner, n = 3 |  | 590.4 | 583.8 | 396.1 | 395.0 | 394.2 | 392.1 | flydsl | 752.2 | 520.2 (copy) | 808.4 | 809.8 | hipBLASLt padded 809.8 |
| 3072x2048x4096 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 843.0 | 871.8 | 761.3 | 760.1 | 762.3 | 760.0 | opus | 1125.2 | 1124.4 | 1006.9 | 1013.5 | AITER opus contiguous 1125.2 |
| 3328x512x4096 | bfloat16 | 26 | one-tile reduce-scatter, n = 4 |  | 380.1 | 368.3 | 217.4 | 216.5 | 216.5 | 215.3 | flydsl | 575.1 | 356.3 (copy) | 476.6 | 467.2 | AITER flydsl contiguous 575.1 |
| 3328x1024x4096 | bfloat16 | 52 | one-tile owner, n = 2 |  | 567.9 | 564.1 | 427.8 | 426.3 | 427.6 | 425.7 | flydsl | 760.4 | 541.1 (copy) | 692.8 | 698.0 | AITER flydsl contiguous 760.4 |
| 3328x2048x4096 | bfloat16 | 104 | one-tile owner, n = 2-3 |  | 897.9 | 906.0 | 825.5 | 814.2 | 820.3 | 810.8 | opus | 1102.0 | 1105.8 | 824.8 | 861.0 | AITER opus padded 1105.8 |
| 3584x512x4096 | bfloat16 | 28 | one-tile reduce-scatter, n = 4 |  | 397.8 | 388.2 | 233.9 | 233.2 | 233.2 | 231.3 | flydsl | 601.3 | 371.4 (copy) | 474.0 | 475.2 | AITER flydsl contiguous 601.3 |
| 3584x1024x4096 | bfloat16 | 56 | one-tile owner, n = 2 |  | 611.2 | 608.9 | 458.0 | 456.4 | 457.8 | 454.5 | flydsl | 785.7 | 554.0 (copy) | 747.8 | 743.5 | AITER flydsl contiguous 785.7 |
| 3584x2048x4096 | bfloat16 | 112 | one-tile owner, n = 2-3 |  | 967.2 | 961.0 | 877.2 | 878.7 | 874.3 | 882.4 | opus | 1155.0 | 1156.3 | 993.0 | 979.0 | AITER opus padded 1156.3 |
| 3840x512x4096 | bfloat16 | 30 | one-tile reduce-scatter, n = 4 |  | 419.3 | 411.5 | 250.1 | 249.9 | 249.6 | 248.0 | flydsl | 620.3 | 386.6 (copy) | 548.8 | 539.0 | AITER flydsl contiguous 620.3 |
| 3840x1024x4096 | bfloat16 | 60 | one-tile owner, n = 2 |  | 647.3 | 649.4 | 488.6 | 487.7 | 490.5 | 488.0 | flydsl | 805.6 | 568.5 (copy) | 786.1 | 780.9 | AITER flydsl contiguous 805.6 |
| 3840x2048x4096 | bfloat16 | 120 | one-tile owner, n = 2-3 |  | 1014.0 | 1016.6 | 926.3 | 929.5 | 933.0 | 946.0 | opus | 1183.4 | 1186.3 | 801.5 | 820.8 | AITER opus padded 1186.3 |
| 4096x512x4096 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 441.1 | 429.6 | 265.9 | 265.9 | 264.8 | 263.2 | flydsl | 683.5 | 409.5 (copy) | 566.4 | 562.3 | AITER flydsl contiguous 683.5 |
| 4096x1024x4096 | bfloat16 | 64 | one-tile owner, n = 2 |  | 684.4 | 686.6 | 512.8 | 516.6 | 513.6 | 514.2 | flydsl | 910.3 | 619.7 (copy) | 812.5 | 808.2 | AITER flydsl contiguous 910.3 |
| 4096x2048x4096 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1068.7 | 1086.7 | 969.6 | 980.0 | 989.7 | 1007.1 | opus | 1235.1 | 1238.2 | 696.4 | 730.1 | AITER opus padded 1238.2 |
| 4096x4096x4096 | bfloat16 | 256 | data-parallel | yes | 1495.6 | 1511.9 | 1496.4 | 1494.7 | 1519.6 | 1514.5 | flydsl | 1469.1 | 1147.7 (copy) | 1405.4 | 1402.0 | AITER flydsl contiguous 1469.1 |
| 4096x8192x4096 | bfloat16 | 512 | data-parallel |  | 1534.8 | 1545.9 | 1534.8 | 1545.9 | 1539.0 | 1542.8 | asm | 1511.6 | 1522.3 | 1490.1 | 1492.7 | AITER asm padded 1522.3 |
| 4096x8704x4096 | bfloat16 | 544 | one-tile reduce-scatter, n = 4 |  | 1389.2 | 1396.1 | 1261.8 | 1256.5 | 1258.7 | 1252.9 | torch | 1260.6 | 1265.1 | 1267.3 | 1265.1 | hipBLASLt contiguous 1267.3 |
| 4096x10240x4096 | bfloat16 | 640 | one-tile owner, n = 2 |  | 1451.9 | 1452.7 | 1414.2 | 1410.8 | 1412.2 | 1410.8 | flydsl | 1363.4 | 1148.0 (copy) | 1314.1 | 1302.8 | AITER flydsl contiguous 1363.4 |
| 4096x16384x4096 | bfloat16 | 1024 | data-parallel |  | 1550.7 | 1543.2 | 1550.7 | 1543.2 | 1536.3 | 1533.5 | flydsl | 1489.8 | 1258.0 (copy) | 1509.3 | 1504.2 | hipBLASLt contiguous 1509.3 |
| 4096x129280x4096 | bfloat16 | 8080 | data-parallel |  | 1531.9 | 1540.2 | 1531.9 | 1540.2 | 1510.2 | 1502.2 | asm | 1469.7 | 1471.0 | 1395.7 | 1377.4 | AITER asm padded 1471.0 |
| 4352x512x4096 | bfloat16 | 34 | one-tile owner, n = 3 |  | 428.8 | 430.4 | 281.8 | 281.7 | 281.1 | 280.1 | flydsl | 539.4 | 363.2 (copy) | 633.3 | 624.1 | hipBLASLt contiguous 633.3 |
| 4352x1024x4096 | bfloat16 | 68 | one-tile owner, n = 2 |  | 708.6 | 714.2 | 545.3 | 550.1 | 545.5 | 540.9 | opus | 843.7 | 855.0 | 905.4 | 901.9 | hipBLASLt contiguous 905.4 |
| 4352x2048x4096 | bfloat16 | 136 | data-parallel | yes | 999.1 | 1044.8 | 992.1 | 1027.7 | 1042.8 | 1055.0 | torch | 1096.3 | 1088.3 | 1098.9 | 1090.9 | hipBLASLt contiguous 1098.9 |
| 5120x512x4096 | bfloat16 | 40 | one-tile owner, n = 3 |  | 500.3 | 497.9 | 331.3 | 330.8 | 331.3 | 329.6 | flydsl | 617.1 | 404.0 (copy) | 717.3 | 706.2 | hipBLASLt contiguous 717.3 |
| 5120x1024x4096 | bfloat16 | 80 | one-tile owner, n = 2 |  | 791.0 | 792.0 | 639.1 | 642.2 | 634.9 | 632.9 | opus | 989.6 | 992.0 | 1011.1 | 1010.2 | hipBLASLt contiguous 1011.1 |
| 5120x2048x4096 | bfloat16 | 160 | data-parallel | yes | 1141.6 | 1162.1 | 1149.9 | 1149.4 | 1179.1 | 1172.6 | torch | 1240.1 | 1242.2 | 1248.8 | 1248.8 | hipBLASLt contiguous 1248.8 |
| 6144x512x4096 | bfloat16 | 48 | one-tile owner, n = 3 |  | 583.9 | 584.5 | 396.8 | 395.4 | 394.9 | 392.2 | flydsl | 760.5 | 475.3 (copy) | 806.3 | 806.8 | hipBLASLt padded 806.8 |
| 6144x1024x4096 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 831.6 | 859.7 | 744.7 | 759.9 | 758.0 | 751.6 | opus | 1118.9 | 1117.8 | 1017.1 | 1011.9 | AITER opus contiguous 1118.9 |
| 6144x2048x4096 | bfloat16 | 192 | data-parallel | yes | 1271.4 | 1288.7 | 1269.5 | 1276.7 | 1306.6 | 1307.3 | opus | 1374.6 | 1380.0 | 1301.5 | 1302.4 | AITER opus padded 1380.0 |
| 6144x129280x4096 | bfloat16 | 12120 | one-tile owner, n = 2-3 |  | 1498.0 | 1516.7 | 1520.8 | 1520.4 | 1486.7 | 1481.8 | asm | 1480.9 | 1484.4 | 1381.9 | 1371.1 | AITER asm padded 1484.4 |
| 6400x512x4096 | bfloat16 | 50 | one-tile owner, n = 3 |  | 597.6 | 603.1 | 412.6 | 410.3 | 411.5 | 409.5 | flydsl | 699.6 | 446.4 (copy) | 691.7 | 684.0 | AITER flydsl contiguous 699.6 |
| 6400x1024x4096 | bfloat16 | 100 | one-tile owner, n = 2-3 |  | 861.7 | 863.3 | 782.8 | 789.6 | 786.1 | 782.5 | opus | 1075.9 | 1077.1 | 844.7 | 833.8 | AITER opus padded 1077.1 |
| 6400x2048x4096 | bfloat16 | 200 | data-parallel | yes | 1294.6 | 1310.3 | 1300.1 | 1296.3 | 1325.2 | 1305.6 | torch | 1233.8 | 1231.1 | 1240.9 | 1232.1 | hipBLASLt contiguous 1240.9 |
| 6912x512x4096 | bfloat16 | 54 | one-tile owner, n = 2 |  | 587.9 | 586.1 | 442.8 | 440.8 | 443.4 | 439.9 | flydsl | 614.6 | 425.8 (copy) | 728.7 | 709.1 | hipBLASLt contiguous 728.7 |
| 6912x1024x4096 | bfloat16 | 108 | one-tile owner, n = 2-3 |  | 915.8 | 921.6 | 851.4 | 841.5 | 851.3 | 847.1 | opus | 1110.1 | 1115.6 | 733.3 | 723.8 | AITER opus padded 1115.6 |
| 6912x2048x4096 | bfloat16 | 216 | data-parallel | yes | 1353.2 | 1364.3 | 1345.3 | 1352.4 | 1372.3 | 1368.7 | torch | 1321.1 | 1323.0 | 1326.9 | 1324.9 | hipBLASLt contiguous 1326.9 |
| 7168x512x4096 | bfloat16 | 56 | one-tile owner, n = 2 |  | 612.9 | 607.9 | 456.9 | 456.4 | 457.9 | 454.5 | flydsl | 641.7 | 448.9 (copy) | 750.9 | 749.2 | hipBLASLt contiguous 750.9 |
| 7168x1024x4096 | bfloat16 | 112 | one-tile owner, n = 2-3 |  | 960.8 | 944.2 | 864.8 | 871.7 | 885.4 | 875.0 | opus | 1179.5 | 1176.2 | 1061.3 | 1054.1 | AITER opus contiguous 1179.5 |
| 7168x2048x4096 | bfloat16 | 224 | data-parallel | yes | 1369.4 | 1385.7 | 1373.5 | 1373.5 | 1395.5 | 1392.0 | torch | 1347.6 | 1350.1 | 1353.5 | 1355.5 | hipBLASLt padded 1355.5 |
| 7424x512x4096 | bfloat16 | 58 | one-tile owner, n = 2 |  | 627.5 | 626.6 | 469.9 | 470.2 | 474.8 | 471.2 | flydsl | 769.0 | 463.3 (copy) | 758.5 | 746.3 | AITER flydsl contiguous 769.0 |
| 7424x1024x4096 | bfloat16 | 116 | one-tile owner, n = 2-3 |  | 978.5 | 975.6 | 914.6 | 887.6 | 920.8 | 908.3 | opus | 1135.3 | 1142.5 | 818.9 | 811.7 | AITER opus padded 1142.5 |
| 7424x2048x4096 | bfloat16 | 232 | data-parallel | yes | 1399.0 | 1423.2 | 1403.7 | 1407.9 | 1426.9 | 1423.6 | torch | 1211.7 | 1214.1 | 1218.6 | 1212.5 | hipBLASLt contiguous 1218.6 |
| 7680x512x4096 | bfloat16 | 60 | one-tile owner, n = 2 |  | 643.8 | 647.6 | 489.9 | 488.4 | 488.5 | 485.8 | flydsl | 694.4 | 497.0 (copy) | 779.6 | 768.8 | hipBLASLt contiguous 779.6 |
| 7680x1024x4096 | bfloat16 | 120 | one-tile owner, n = 2-3 |  | 1007.0 | 1008.1 | 930.5 | 931.6 | 952.6 | 939.6 | opus | 1187.9 | 1199.4 | 804.4 | 800.6 | AITER opus padded 1199.4 |
| 7680x2048x4096 | bfloat16 | 240 | data-parallel | yes | 1428.7 | 1446.7 | 1432.2 | 1430.3 | 1454.8 | 1452.5 | torch | 1237.7 | 1237.5 | 1243.4 | 1239.7 | hipBLASLt contiguous 1243.4 |
| 7936x512x4096 | bfloat16 | 62 | one-tile owner, n = 2 |  | 664.2 | 665.1 | 505.4 | 503.5 | 505.0 | 502.0 | flydsl | 784.6 | 429.4 (copy) | 798.7 | 786.0 | hipBLASLt contiguous 798.7 |
| 7936x1024x4096 | bfloat16 | 124 | one-tile owner, n = 2-3 |  | 1029.9 | 1045.6 | 949.6 | 953.3 | 970.2 | 956.2 | opus | 1171.8 | 1180.9 | 741.6 | 737.4 | AITER opus padded 1180.9 |
| 7936x2048x4096 | bfloat16 | 248 | data-parallel | yes | 1464.3 | 1476.4 | 1465.8 | 1465.5 | 1488.7 | 1483.7 | torch | 1438.5 | 1443.2 | 1447.2 | 1443.6 | hipBLASLt contiguous 1447.2 |
| 8192x512x4096 | bfloat16 | 64 | one-tile owner, n = 2 |  | 688.0 | 685.3 | 520.4 | 516.5 | 516.0 | 509.4 | flydsl | 871.7 | 533.3 (copy) | 818.8 | 806.6 | AITER flydsl contiguous 871.7 |
| 8192x1024x4096 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1076.3 | 1081.3 | 984.9 | 978.5 | 975.9 | 996.3 | opus | 1249.2 | 1256.9 | 749.6 | 746.7 | AITER opus padded 1256.9 |
| 8192x2048x4096 | bfloat16 | 256 | data-parallel | yes | 1484.8 | 1506.7 | 1500.3 | 1497.9 | 1518.3 | 1511.3 | torch | 1461.7 | 1464.1 | 1469.3 | 1465.3 | hipBLASLt contiguous 1469.3 |
| 8192x4096x4096 | bfloat16 | 512 | data-parallel |  | 1531.0 | 1546.2 | 1531.0 | 1546.2 | 1542.3 | 1536.6 | flydsl | 1494.9 | 1209.6 (copy) | 1491.7 | 1494.2 | AITER flydsl contiguous 1494.9 |
| 8192x8192x4096 | bfloat16 | 1024 | data-parallel |  | 1559.2 | 1563.0 | 1559.2 | 1563.0 | 1555.7 | 1553.0 | asm | 1525.7 | 1534.1 | 1499.6 | 1498.9 | AITER asm padded 1534.1 |
| 8192x8704x4096 | bfloat16 | 1088 | one-tile owner, n = 2 |  | 1472.9 | 1476.0 | 1440.7 | 1434.7 | 1427.6 | 1424.4 | asm | 1398.5 | 1402.1 | 1361.2 | 1358.4 | AITER asm padded 1402.1 |
| 8192x10240x4096 | bfloat16 | 1280 | data-parallel |  | 1539.4 | 1545.1 | 1539.4 | 1545.1 | 1544.6 | 1531.9 | torch | 1494.2 | 1504.9 | 1505.5 | 1502.6 | hipBLASLt contiguous 1505.5 |
| 8192x16384x4096 | bfloat16 | 2048 | data-parallel |  | 1540.0 | 1547.3 | 1540.0 | 1547.3 | 1520.8 | 1521.7 | torch | 1510.0 | 1514.8 | 1518.5 | 1512.7 | hipBLASLt contiguous 1518.5 |
| 8192x129280x4096 | bfloat16 | 16160 | one-tile reduce-scatter, n = 4 |  | 1557.3 | 1563.0 | 1558.6 | 1552.2 | 1516.2 | 1513.6 | flydsl | 1467.4 | 1377.2 (copy) | 1386.7 | 1372.2 | AITER flydsl contiguous 1467.4 |
| 10240x512x4096 | bfloat16 | 80 | one-tile owner, n = 2 |  | 793.3 | 793.4 | 632.5 | 634.1 | 631.5 | 627.0 | torch | 1003.1 | 1001.1 | 1004.2 | 1005.7 | hipBLASLt padded 1005.7 |
| 10240x1024x4096 | bfloat16 | 160 | data-parallel | yes | 1133.1 | 1159.6 | 1145.8 | 1150.8 | 1167.9 | 1159.8 | torch | 1248.2 | 1248.4 | 1253.0 | 1251.0 | hipBLASLt contiguous 1253.0 |
| 10240x2048x4096 | bfloat16 | 320 | one-tile owner, n = 2 |  | 1252.3 | 1259.3 | 1140.8 | 1138.8 | 1140.8 | 1137.1 | torch | 1272.8 | 1277.5 | 1278.0 | 1277.6 | hipBLASLt contiguous 1278.0 |
| 10240x129280x4096 | bfloat16 | 20200 | data-parallel |  | 1505.7 | 1529.4 | 1505.7 | 1529.4 | 1498.4 | 1487.3 | asm | 1490.9 | 1491.3 | 1426.9 | 1387.8 | AITER asm padded 1491.3 |
| 12288x512x4096 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 835.0 | 869.5 | 745.5 | 755.0 | 749.4 | 740.5 | opus | 1102.4 | 1099.7 | 1007.3 | 1002.3 | AITER opus contiguous 1102.4 |
| 12288x1024x4096 | bfloat16 | 192 | data-parallel | yes | 1275.3 | 1282.4 | 1288.3 | 1280.4 | 1310.5 | 1309.1 | opus | 1376.6 | 1380.5 | 1303.4 | 1300.9 | AITER opus padded 1380.5 |
| 12288x2048x4096 | bfloat16 | 384 | one-tile owner, n = 2 |  | 1361.1 | 1369.9 | 1302.5 | 1304.0 | 1308.2 | 1303.8 | opus | 1386.3 | 1394.8 | 1325.5 | 1325.7 | AITER opus padded 1394.8 |
| 12288x129280x4096 | bfloat16 | 24240 | data-parallel |  | 1530.0 | 1550.4 | 1530.0 | 1550.4 | 1511.8 | 1507.9 | asm | 1492.3 | 1487.8 | 1360.6 | 1359.2 | AITER asm contiguous 1492.3 |
| 14336x512x4096 | bfloat16 | 112 | one-tile owner, n = 2-3 |  | 946.8 | 947.5 | 867.1 | 866.9 | 875.7 | 872.1 | opus | 1174.8 | 1183.6 | 705.1 | 669.1 | AITER opus padded 1183.6 |
| 14336x1024x4096 | bfloat16 | 224 | data-parallel | yes | 1383.2 | 1391.3 | 1376.5 | 1374.4 | 1399.1 | 1391.4 | torch | 1346.9 | 1351.6 | 1352.9 | 1358.4 | hipBLASLt padded 1358.4 |
| 14336x2048x4096 | bfloat16 | 448 | data-parallel |  | 1417.0 | 1425.9 | 1417.0 | 1425.9 | 1421.7 | 1419.4 | flydsl | 1392.2 | 1057.3 (copy) | 1379.9 | 1381.4 | AITER flydsl contiguous 1392.2 |
| 14336x129280x4096 | bfloat16 | 28280 | one-tile owner, n = 2-3 |  | 1510.3 | 1516.7 | 1496.5 | 1493.8 | 1497.6 | 1488.5 | asm | 1484.6 | 1492.6 | 1409.7 | 1408.1 | AITER asm padded 1492.6 |
| 16384x512x4096 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1052.5 | 1065.7 | 980.6 | 974.0 | 984.5 | 994.1 | opus | 1245.2 | 1254.0 | 806.8 | 794.0 | AITER opus padded 1254.0 |
| 16384x1024x4096 | bfloat16 | 256 | data-parallel | yes | 1497.8 | 1514.4 | 1497.6 | 1496.3 | 1516.3 | 1511.0 | flydsl | 1470.1 | 904.6 (copy) | 1332.4 | 1310.4 | AITER flydsl contiguous 1470.1 |
| 16384x2048x4096 | bfloat16 | 512 | data-parallel |  | 1532.1 | 1545.3 | 1532.1 | 1545.3 | 1536.7 | 1536.6 | torch | 1413.3 | 1414.8 | 1418.6 | 1406.0 | hipBLASLt contiguous 1418.6 |
| 16384x16384x4096 | bfloat16 | 4096 | data-parallel |  | 1541.9 | 1548.3 | 1541.9 | 1548.3 | 1516.3 | 1514.3 | torch | 1517.9 | 1527.0 | 1534.4 | 1526.7 | hipBLASLt contiguous 1534.4 |
| 16384x129280x4096 | bfloat16 | 32320 | one-tile owner, n = 2 |  | 1494.2 | 1562.7 | 1509.7 | 1531.5 | 1535.1 | 1530.7 | asm | 1482.3 | 1494.5 | 1401.0 | 1388.6 | AITER asm padded 1494.5 |
| 32768x16384x4096 | bfloat16 | 8192 | data-parallel |  | 1548.1 | 1549.2 | 1548.1 | 1549.2 | 1514.1 | 1512.9 | flydsl | 1485.3 | 1406.8 (copy) | 1523.0 | 1511.2 | hipBLASLt contiguous 1523.0 |
| 512x7168x4224 | bfloat16 | 56 | one-tile owner, n = 2 |  | 627.7 | 631.8 | 460.6 | 459.8 | 456.7 | 456.7 | flydsl | 839.0 | 848.5 | 882.3 | 879.8 | hipBLASLt contiguous 882.3 |
| 1024x7168x4224 | bfloat16 | 112 | one-tile owner, n = 2-3 |  | 977.0 | 971.3 | 884.5 | 874.2 | 869.1 | 882.1 | flydsl | 1148.9 | 1151.6 | 851.8 | 847.7 | AITER flydsl padded 1151.6 |
| 10240x7168x4224 | bfloat16 | 1120 | one-tile owner, n = 2-3 |  | 1481.2 | 1477.6 | 1457.5 | 1461.6 | 1446.6 | 1445.9 | opus | 953.5 | 959.8 | 1364.8 | 1363.5 | hipBLASLt contiguous 1364.8 |
| 512x6144x4608 | bfloat16 | 48 | one-tile owner, n = 3 |  | 611.9 | 610.7 | 401.5 | 401.0 | 398.6 | 397.6 | flydsl | 784.0 | 483.6 (copy) | 695.4 | 693.2 | AITER flydsl contiguous 784.0 |
| 1024x6144x4608 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 864.8 | 902.1 | 767.5 | 777.5 | 775.7 | 780.2 | torch | 1007.8 | 1012.7 | 1011.2 | 1012.2 | AITER torch padded 1012.7 |
| 2048x6144x4608 | bfloat16 | 192 | data-parallel | yes | 1294.9 | 1332.1 | 1324.6 | 1315.2 | 1344.6 | 1342.8 | flydsl | 1275.4 | 987.4 (copy) | 1317.1 | 1317.8 | hipBLASLt padded 1317.8 |
| 4096x6144x4608 | bfloat16 | 384 | one-tile owner, n = 2 |  | 1401.5 | 1415.6 | 1322.4 | 1321.6 | 1326.6 | 1322.9 | torch | 1329.5 | 1337.2 | 1335.7 | 1338.5 | hipBLASLt padded 1338.5 |
| 8192x6144x4608 | bfloat16 | 768 | data-parallel |  | 1567.5 | 1570.5 | 1567.5 | 1570.5 | 1563.7 | 1568.2 | torch | 1506.6 | 1507.3 | 1510.2 | 1508.2 | hipBLASLt contiguous 1510.2 |
| 16384x6144x4608 | bfloat16 | 1536 | data-parallel |  | 1560.9 | 1570.0 | 1560.9 | 1570.0 | 1547.6 | 1554.8 | torch | 1512.7 | 1519.9 | 1521.9 | 1515.7 | hipBLASLt contiguous 1521.9 |
| 32768x6144x4608 | bfloat16 | 3072 | data-parallel |  | 1558.4 | 1578.4 | 1558.4 | 1578.4 | 1543.5 | 1550.3 | torch | 1493.4 | 1506.7 | 1502.5 | 1510.5 | hipBLASLt padded 1510.5 |
| 512x1024x5120 | float32 | 8 | one-tile reduce-scatter, n = 10 |  | 205.0 | 203.4 | 65.7 | 66.0 | 65.4 | 65.8 | asm | 334.6 | 338.8 | 338.9 | 336.0 | hipBLASLt contiguous 338.9 |
| 512x5120x5120 | bfloat16 | 40 | one-tile owner, n = 3 |  | 560.9 | 572.3 | 339.0 | 339.2 | 337.2 | 336.9 | torch | 794.8 | 788.8 | 794.2 | 793.0 | AITER torch contiguous 794.8 |
| 512x6400x5120 | bfloat16 | 50 | one-tile owner, n = 3 |  | 666.7 | 664.3 | 420.0 | 423.1 | 419.4 | 418.5 | flydsl | 701.5 | 456.3 (copy) | 707.0 | 726.2 | hipBLASLt padded 726.2 |
| 512x12800x5120 | bfloat16 | 100 | one-tile owner, n = 2-3 |  | 944.5 | 952.7 | 822.4 | 820.1 | 821.7 | 820.8 | flydsl | 802.2 | 432.3 (copy) | 834.9 | 839.9 | hipBLASLt padded 839.9 |
| 512x37888x5120 | bfloat16 | 296 | one-tile owner, n = 3 |  | 1151.1 | 1174.7 | 989.7 | 1003.6 | 980.4 | 1003.7 | torch | 1053.3 | 1081.4 | 1057.0 | 1080.2 | AITER torch padded 1081.4 |
| 512x51200x5120 | bfloat16 | 400 | data-parallel |  | 1210.1 | 1225.5 | 1210.1 | 1225.5 | 1202.2 | 1226.7 | flydsl | 1143.3 | 634.4 (copy) | 1094.6 | 1124.5 | AITER flydsl contiguous 1143.3 |
| 768x1024x5120 | float32 | 12 | one-tile reduce-scatter, n = 10 |  | 270.5 | 271.0 | 98.4 | 98.9 | 97.9 | 98.6 | asm | 412.7 | 421.7 | 381.6 | 385.7 | AITER asm padded 421.7 |
| 768x37888x5120 | bfloat16 | 444 | data-parallel |  | 1336.3 | 1357.4 | 1336.3 | 1357.4 | 1337.7 | 1353.7 | torch | 1240.5 | 1247.8 | 1244.6 | 1252.4 | hipBLASLt padded 1252.4 |
| 1024x1024x5120 | float32 | 16 | one-tile reduce-scatter, n = 8 |  | 332.0 | 334.9 | 131.1 | 131.6 | 130.4 | 130.5 | asm | 457.9 | 461.8 | 485.7 | 490.5 | hipBLASLt padded 490.5 |
| 1024x5120x5120 | bfloat16 | 80 | one-tile owner, n = 2 |  | 865.9 | 892.0 | 658.9 | 663.5 | 656.9 | 656.7 | torch | 1009.5 | 1036.5 | 1009.1 | 1039.3 | hipBLASLt padded 1039.3 |
| 1024x6400x5120 | bfloat16 | 100 | one-tile owner, n = 2-3 |  | 942.8 | 956.7 | 810.4 | 820.0 | 811.4 | 815.8 | flydsl | 952.4 | 676.4 (copy) | 888.1 | 889.2 | AITER flydsl contiguous 952.4 |
| 1024x12800x5120 | bfloat16 | 200 | data-parallel | yes | 1333.5 | 1363.1 | 1341.6 | 1341.3 | 1359.6 | 1358.7 | flydsl | 797.1 | 557.5 (copy) | 1235.6 | 1248.8 | hipBLASLt padded 1248.8 |
| 1024x37888x5120 | bfloat16 | 592 | one-tile owner, n = 2 |  | 1345.0 | 1362.5 | 1271.0 | 1288.8 | 1259.5 | 1280.5 | torch | 1063.9 | 1092.4 | 1065.7 | 1093.8 | hipBLASLt padded 1093.8 |
| 1024x51200x5120 | bfloat16 | 800 | one-tile reduce-scatter, n = 4 |  | 1383.2 | 1397.1 | 1284.5 | 1307.1 | 1271.4 | 1290.8 | asm | 1224.5 | 1253.2 | 1071.0 | 1101.4 | AITER asm padded 1253.2 |
| 1536x1024x5120 | float32 | 24 | one-tile reduce-scatter, n = 5 |  | 417.5 | 412.3 | 194.8 | 196.0 | 194.7 | 194.9 | asm | 577.0 | 585.7 | 612.5 | 621.4 | hipBLASLt padded 621.4 |
| 1536x37888x5120 | bfloat16 | 888 | one-tile owner, n = 2-3 |  | 1402.2 | 1421.5 | 1351.2 | 1385.2 | 1361.2 | 1370.3 | torch | 1272.5 | 1295.8 | 1281.2 | 1296.5 | hipBLASLt padded 1296.5 |
| 2048x1024x5120 | float32 | 32 | one-tile reduce-scatter, n = 4 |  | 485.7 | 479.6 | 257.6 | 256.9 | 258.2 | 256.5 | opus | 632.9 | 633.3 | 650.4 | 671.5 | hipBLASLt padded 671.5 |
| 2048x5120x5120 | bfloat16 | 160 | data-parallel | yes | 1205.0 | 1213.9 | 1202.7 | 1194.8 | 1229.0 | 1226.8 | torch | 1243.2 | 1254.5 | 1251.1 | 1255.2 | hipBLASLt padded 1255.2 |
| 2048x6400x5120 | bfloat16 | 200 | data-parallel | yes | 1355.4 | 1354.4 | 1337.0 | 1347.6 | 1354.6 | 1354.4 | flydsl | 1248.8 | 946.4 (copy) | 1247.4 | 1260.4 | hipBLASLt padded 1260.4 |
| 2048x12800x5120 | bfloat16 | 400 | data-parallel |  | 1363.5 | 1375.5 | 1363.5 | 1375.5 | 1384.1 | 1380.3 | flydsl | 805.0 | 676.5 (copy) | 1259.9 | 1277.0 | hipBLASLt padded 1277.0 |
| 2048x37888x5120 | bfloat16 | 1184 | data-parallel |  | 1458.4 | 1469.3 | 1458.4 | 1469.3 | 1444.4 | 1459.1 | torch | 1202.6 | 1239.8 | 1216.0 | 1243.0 | hipBLASLt padded 1243.0 |
| 2048x51200x5120 | bfloat16 | 1600 | one-tile owner, n = 2 |  | 1462.6 | 1483.1 | 1438.8 | 1448.4 | 1414.5 | 1431.8 | asm | 212.0 | 217.0 | 1182.5 | 1211.5 | hipBLASLt padded 1211.5 |
| 4096x1024x5120 | float32 | 64 | one-tile owner, n = 2 |  | 688.2 | 705.9 | 495.7 | 500.2 | 488.3 | 500.0 | opus | 966.5 | 975.5 | 903.1 | 924.1 | AITER opus padded 975.5 |
| 4096x5120x5120 | bfloat16 | 320 | one-tile owner, n = 2 |  | 1308.1 | 1313.0 | 1156.8 | 1162.1 | 1161.4 | 1157.6 | torch | 1265.1 | 1272.1 | 1270.5 | 1273.3 | hipBLASLt padded 1273.3 |
| 4096x6400x5120 | bfloat16 | 400 | data-parallel |  | 1370.4 | 1374.9 | 1370.4 | 1374.9 | 1379.2 | 1375.1 | flydsl | 1327.7 | 1091.0 (copy) | 1265.8 | 1276.1 | AITER flydsl contiguous 1327.7 |
| 4096x12800x5120 | bfloat16 | 800 | one-tile reduce-scatter, n = 4 |  | 1446.5 | 1459.6 | 1372.1 | 1376.4 | 1362.1 | 1361.1 | asm | 1321.9 | 1324.0 | 1273.2 | 1284.7 | AITER asm padded 1324.0 |
| 4096x51200x5120 | bfloat16 | 3200 | one-tile owner, n = 2 |  | 1529.8 | 1554.6 | 1526.9 | 1537.6 | 1500.4 | 1510.2 | asm | 1455.2 | 1477.6 | 1382.0 | 1407.9 | AITER asm padded 1477.6 |
| 8192x1024x5120 | float32 | 128 | one-tile owner, n = 2 |  | 1076.8 | 1092.6 | 962.4 | 973.6 | 985.0 | 989.2 | opus | 1246.7 | 1253.0 | 1141.3 | 1145.4 | AITER opus padded 1253.0 |
| 8192x5120x5120 | bfloat16 | 640 | one-tile owner, n = 2 |  | 1486.8 | 1494.2 | 1436.3 | 1430.0 | 1424.1 | 1425.1 | torch | 1352.3 | 1361.6 | 1357.3 | 1354.8 | AITER torch padded 1361.6 |
| 8192x6400x5120 | bfloat16 | 800 | one-tile reduce-scatter, n = 4 |  | 1481.0 | 1481.9 | 1372.8 | 1377.3 | 1363.5 | 1362.3 | flydsl | 1320.4 | 1146.7 (copy) | 1279.2 | 1288.1 | AITER flydsl contiguous 1320.4 |
| 8192x12800x5120 | bfloat16 | 1600 | one-tile owner, n = 2 |  | 1497.9 | 1514.6 | 1468.2 | 1476.5 | 1456.5 | 1468.4 | asm | 1412.2 | 1423.8 | 1236.7 | 1242.6 | AITER asm padded 1423.8 |
| 8192x51200x5120 | bfloat16 | 6400 | data-parallel |  | 1571.5 | 1589.5 | 1571.5 | 1589.5 | 1546.4 | 1560.4 | flydsl | 1491.0 | 1388.8 (copy) | 1413.4 | 1455.4 | AITER flydsl contiguous 1491.0 |
| 16384x1024x5120 | float32 | 256 | data-parallel |  | 1412.0 | 1438.3 | 1412.0 | 1438.3 | 1451.5 | 1445.5 | opus | 1337.3 | 1339.5 | 1335.3 | 1335.1 | AITER opus padded 1339.5 |
| 16384x5120x5120 | bfloat16 | 1280 | data-parallel |  | 1558.2 | 1566.3 | 1558.2 | 1566.3 | 1546.9 | 1557.4 | torch | 1519.7 | 1525.4 | 1529.6 | 1529.2 | hipBLASLt contiguous 1529.6 |
| 16384x6400x5120 | bfloat16 | 1600 | one-tile owner, n = 2 |  | 1489.8 | 1507.2 | 1461.3 | 1469.5 | 1439.5 | 1454.9 | flydsl | 1377.6 | 1257.9 (copy) | 1241.4 | 1245.4 | AITER flydsl contiguous 1377.6 |
| 16384x12800x5120 | bfloat16 | 3200 | one-tile owner, n = 2 |  | 1525.9 | 1555.8 | 1526.7 | 1539.2 | 1497.3 | 1515.7 | torch | 1480.0 | 1507.7 | 1485.5 | 1510.7 | hipBLASLt padded 1510.7 |
| 16384x51200x5120 | bfloat16 | 12800 | data-parallel |  | 1571.3 | 1580.5 | 1571.3 | 1580.5 | 1539.5 | 1556.6 | flydsl | 1488.5 | 1432.7 (copy) | 1415.5 | 1458.6 | AITER flydsl contiguous 1488.5 |
| 32768x5120x5120 | bfloat16 | 2560 | data-parallel |  | 1547.2 | 1568.6 | 1547.2 | 1568.6 | 1532.6 | 1542.9 | torch | 1494.9 | 1512.4 | 1500.8 | 1508.3 | AITER torch padded 1512.4 |
| 32768x6400x5120 | bfloat16 | 3200 | one-tile owner, n = 2 |  | 1524.5 | 1547.1 | 1521.9 | 1538.4 | 1492.5 | 1511.3 | torch | 1473.6 | 1492.4 | 1483.0 | 1494.0 | hipBLASLt padded 1494.0 |
| 32768x12800x5120 | bfloat16 | 6400 | data-parallel |  | 1555.9 | 1578.8 | 1555.9 | 1578.8 | 1535.5 | 1549.1 | torch | 1509.5 | 1541.2 | 1520.8 | 1545.7 | hipBLASLt padded 1545.7 |
| 32768x51200x5120 | bfloat16 | 25600 | data-parallel |  | 1567.6 | 1545.1 | 1567.6 | 1545.1 | 1543.7 | 1558.7 | flydsl | 1488.3 | 1461.2 (copy) | 1423.1 | 1424.4 | AITER flydsl contiguous 1488.3 |
| 512x1280x6144 | bfloat16 | 10 | one-tile reduce-scatter, n = 12 |  | 269.6 | 271.1 | 86.5 | 87.2 | 86.3 | 86.3 | flydsl | 404.8 | 252.0 (copy) | 385.8 | 393.4 | AITER flydsl contiguous 404.8 |
| 512x1536x6144 | bfloat16 | 12 | one-tile reduce-scatter, n = 10 |  | 306.7 | 306.9 | 104.0 | 104.3 | 103.6 | 103.4 | flydsl | 434.3 | 274.4 (copy) | 411.5 | 415.6 | AITER flydsl contiguous 434.3 |
| 512x2048x6144 | float32 | 16 | one-tile reduce-scatter, n = 8 |  | 377.9 | 377.0 | 134.3 | 134.7 | 133.7 | 133.8 | asm | 487.0 | 493.0 | 510.3 | 509.4 | hipBLASLt contiguous 510.3 |
| 512x2304x6144 | bfloat16 | 18 | one-tile reduce-scatter, n = 7 |  | 385.1 | 388.0 | 156.3 | 156.9 | 154.5 | 154.7 | flydsl | 519.9 | 326.0 (copy) | 410.9 | 410.2 | AITER flydsl contiguous 519.9 |
| 512x2560x6144 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 |  | 418.0 | 416.1 | 173.9 | 173.7 | 171.6 | 171.6 | torch | 512.9 | 508.9 | 513.5 | 513.3 | hipBLASLt contiguous 513.5 |
| 512x3072x6144 | float32 | 24 | one-tile reduce-scatter, n = 5 |  | 461.9 | 456.2 | 200.7 | 200.6 | 199.6 | 199.4 | opus | 503.6 | 510.2 | 632.2 | 634.1 | hipBLASLt padded 634.1 |
| 512x3072x6144 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 |  | 464.6 | 470.1 | 207.7 | 207.9 | 205.5 | 205.6 | flydsl | 603.2 | 386.7 (copy) | 490.3 | 507.9 | AITER flydsl contiguous 603.2 |
| 512x6144x6144 | bfloat16 | 48 | one-tile owner, n = 3 |  | 703.4 | 685.8 | 414.4 | 410.9 | 408.1 | 407.6 | flydsl | 808.5 | 498.1 (copy) | 782.3 | 763.8 | AITER flydsl contiguous 808.5 |
| 512x9216x6144 | bfloat16 | 72 | one-tile owner, n = 3-4 |  | 876.5 | 888.7 | 611.2 | 610.7 | 607.2 | 608.1 | flydsl | 895.7 | 408.8 (copy) | 589.1 | 620.7 | AITER flydsl contiguous 895.7 |
| 1024x2048x6144 | float32 | 32 | one-tile reduce-scatter, n = 4 |  | 533.6 | 528.3 | 265.0 | 264.7 | 264.6 | 264.9 | opus | 537.1 | 554.1 | 734.2 | 744.6 | hipBLASLt padded 744.6 |
| 1024x2304x6144 | bfloat16 | 36 | one-tile owner, n = 4 |  | 570.9 | 570.5 | 310.6 | 311.2 | 306.3 | 306.3 | torch | 674.5 | 676.6 | 677.7 | 683.0 | hipBLASLt padded 683.0 |
| 1024x2560x6144 | bfloat16 | 40 | one-tile owner, n = 3 |  | 603.4 | 605.3 | 344.0 | 345.1 | 340.5 | 341.2 | torch | 781.3 | 785.7 | 787.1 | 776.0 | hipBLASLt contiguous 787.1 |
| 1024x3072x6144 | float32 | 48 | one-tile owner, n = 3 |  | 641.5 | 646.6 | 389.7 | 393.9 | 391.7 | 393.7 | opus | 565.1 | 577.8 | 851.9 | 848.7 | hipBLASLt contiguous 851.9 |
| 1024x6144x6144 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 977.3 | 1002.1 | 806.0 | 802.7 | 805.4 | 805.7 | torch | 1037.1 | 1046.3 | 1031.9 | 1049.8 | hipBLASLt padded 1049.8 |
| 1024x9216x6144 | bfloat16 | 144 | data-parallel | yes | 1140.4 | 1156.6 | 1152.7 | 1153.2 | 1165.4 | 1168.5 | torch | 1225.7 | 1235.8 | 1235.0 | 1239.1 | hipBLASLt padded 1239.1 |
| 2048x2048x6144 | float32 | 64 | one-tile owner, n = 3 |  | 800.6 | 807.5 | 514.0 | 513.2 | 511.7 | 517.1 | opus | 887.6 | 892.9 | 947.5 | 966.0 | hipBLASLt padded 966.0 |
| 2048x2304x6144 | bfloat16 | 72 | one-tile owner, n = 3-4 |  | 880.0 | 881.4 | 610.6 | 611.3 | 604.6 | 606.7 | torch | 988.1 | 995.8 | 990.9 | 998.4 | hipBLASLt padded 998.4 |
| 2048x2560x6144 | bfloat16 | 80 | one-tile owner, n = 2 |  | 919.2 | 949.0 | 671.1 | 674.1 | 669.7 | 673.1 | torch | 1052.4 | 1063.0 | 1052.6 | 1064.5 | hipBLASLt padded 1064.5 |
| 2048x3072x6144 | float32 | 96 | one-tile owner, n = 2-3 |  | 916.8 | 942.2 | 774.3 | 770.9 | 777.4 | 774.9 | opus | 817.5 | 819.8 | 986.9 | 973.3 | hipBLASLt contiguous 986.9 |
| 2048x6144x6144 | bfloat16 | 192 | data-parallel | yes | 1337.4 | 1350.6 | 1337.4 | 1344.1 | 1366.6 | 1353.3 | torch | 1342.6 | 1347.9 | 1349.2 | 1350.9 | hipBLASLt padded 1350.9 |
| 2048x9216x6144 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 |  | 1333.9 | 1349.4 | 1074.2 | 1072.6 | 1073.0 | 1070.5 | torch | 1237.9 | 1249.3 | 1244.6 | 1251.2 | hipBLASLt padded 1251.2 |
| 4096x2048x6144 | float32 | 128 | one-tile owner, n = 2 |  | 1143.8 | 1168.4 | 992.6 | 996.2 | 1005.8 | 1011.9 | opus | 1156.6 | 1156.3 | 893.9 | 814.6 | AITER opus contiguous 1156.6 |
| 4096x2304x6144 | bfloat16 | 144 | data-parallel | yes | 1128.9 | 1155.8 | 1146.6 | 1149.2 | 1156.6 | 1165.6 | flydsl | 779.7 | 654.5 (copy) | 1233.4 | 1236.3 | hipBLASLt padded 1236.3 |
| 4096x2560x6144 | bfloat16 | 160 | data-parallel | yes | 1213.1 | 1240.3 | 1230.9 | 1227.7 | 1244.8 | 1246.9 | flydsl | 708.6 | 601.9 (copy) | 1280.6 | 1276.6 | hipBLASLt contiguous 1280.6 |
| 4096x3072x6144 | float32 | 192 | data-parallel |  | 1276.0 | 1293.3 | 1276.0 | 1293.3 | 1320.4 | 1321.7 | asm | 1228.8 | 1233.2 | 1314.0 | 1316.0 | hipBLASLt padded 1316.0 |
| 4096x6144x6144 | bfloat16 | 384 | one-tile owner, n = 2 |  | 1469.0 | 1470.1 | 1350.5 | 1347.6 | 1353.4 | 1352.6 | torch | 1355.9 | 1363.8 | 1366.3 | 1364.3 | hipBLASLt contiguous 1366.3 |
| 4096x9216x6144 | bfloat16 | 576 | one-tile owner, n = 3 |  | 1467.4 | 1472.6 | 1343.4 | 1345.5 | 1343.2 | 1339.9 | torch | 1366.1 | 1370.3 | 1372.8 | 1365.5 | hipBLASLt contiguous 1372.8 |
| 8192x2048x6144 | float32 | 256 | data-parallel |  | 1459.2 | 1484.5 | 1459.2 | 1484.5 | 1498.5 | 1493.9 | opus | 1394.8 | 1397.0 | 1366.1 | 1374.7 | AITER opus padded 1397.0 |
| 8192x2304x6144 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 |  | 1337.0 | 1344.8 | 1077.2 | 1077.0 | 1073.3 | 1073.9 | torch | 1234.5 | 1245.5 | 1240.3 | 1249.0 | hipBLASLt padded 1249.0 |
| 8192x2560x6144 | bfloat16 | 320 | one-tile owner, n = 3 |  | 1379.1 | 1389.9 | 1169.4 | 1173.6 | 1174.8 | 1173.3 | asm | 519.4 | 528.4 | 1285.0 | 1290.1 | hipBLASLt padded 1290.1 |
| 8192x3072x6144 | float32 | 384 | one-tile owner, n = 2 |  | 1377.2 | 1386.7 | 1275.5 | 1285.7 | 1292.8 | 1290.6 | asm | 1242.5 | 1244.6 | 1320.8 | 1324.4 | hipBLASLt padded 1324.4 |
| 8192x6144x6144 | bfloat16 | 768 | data-parallel |  | 1594.2 | 1599.5 | 1594.2 | 1599.5 | 1590.9 | 1594.4 | flydsl | 1545.7 | 1314.7 (copy) | 1538.0 | 1531.5 | AITER flydsl contiguous 1545.7 |
| 8192x9216x6144 | bfloat16 | 1152 | one-tile owner, n = 2 |  | 1533.4 | 1543.7 | 1492.3 | 1495.5 | 1474.1 | 1487.0 | asm | 1454.2 | 1468.3 | 1253.8 | 1258.2 | AITER asm padded 1468.3 |
| 16384x2048x6144 | float32 | 512 | data-parallel |  | 1407.5 | 1445.9 | 1407.5 | 1445.9 | 1430.7 | 1455.4 | opus | 1345.0 | 1348.0 | 1333.4 | 1335.6 | AITER opus padded 1348.0 |
| 16384x2304x6144 | bfloat16 | 576 | one-tile owner, n = 3 |  | 1421.5 | 1438.6 | 1331.6 | 1337.9 | 1320.6 | 1322.1 | flydsl | 1044.1 | 833.9 (copy) | 1340.5 | 1337.8 | hipBLASLt contiguous 1340.5 |
| 16384x2560x6144 | bfloat16 | 640 | one-tile owner, n = 2 |  | 1473.0 | 1484.0 | 1417.1 | 1420.1 | 1407.6 | 1414.8 | flydsl | 1374.0 | 1090.7 (copy) | 1362.9 | 1361.5 | AITER flydsl contiguous 1374.0 |
| 16384x3072x6144 | float32 | 768 | data-parallel |  | 1432.3 | 1460.5 | 1432.3 | 1460.5 | 1435.5 | 1460.3 | asm | 1389.4 | 1404.7 | 1352.2 | 1352.7 | AITER asm padded 1404.7 |
| 16384x6144x6144 | bfloat16 | 1536 | data-parallel |  | 1565.6 | 1585.1 | 1565.6 | 1585.1 | 1560.1 | 1568.8 | asm | 1518.6 | 1536.4 | 1564.5 | 1563.7 | hipBLASLt contiguous 1564.5 |
| 16384x9216x6144 | bfloat16 | 2304 | data-parallel |  | 1560.4 | 1585.0 | 1560.4 | 1585.0 | 1550.7 | 1568.3 | torch | 1530.1 | 1550.7 | 1543.2 | 1556.9 | hipBLASLt padded 1556.9 |
| 32768x2304x6144 | bfloat16 | 1152 | one-tile owner, n = 2 |  | 1471.6 | 1491.6 | 1427.0 | 1448.8 | 1416.8 | 1432.7 | flydsl | 652.7 | 592.1 (copy) | 1210.6 | 1215.5 | hipBLASLt padded 1215.5 |
| 32768x2560x6144 | bfloat16 | 1280 | data-parallel |  | 1512.8 | 1547.1 | 1512.8 | 1547.1 | 1513.0 | 1528.9 | flydsl | 1457.2 | 1199.0 (copy) | 1490.6 | 1521.8 | hipBLASLt padded 1521.8 |
| 32768x6144x6144 | bfloat16 | 3072 | data-parallel |  | 1580.8 | 1598.5 | 1580.8 | 1598.5 | 1565.7 | 1573.7 | torch | 1514.3 | 1558.8 | 1524.3 | 1555.2 | AITER torch padded 1558.8 |
| 32768x9216x6144 | bfloat16 | 4608 | data-parallel |  | 1573.7 | 1595.3 | 1573.7 | 1595.3 | 1556.8 | 1572.6 | torch | 1550.3 | 1574.5 | 1572.7 | 1576.0 | hipBLASLt padded 1576.0 |
| 512x5120x6400 | bfloat16 | 40 | one-tile owner, n = 4 |  | 638.6 | 636.2 | 346.0 | 346.5 | 341.8 | 340.5 | flydsl | 716.2 | 460.8 (copy) | 791.4 | 795.9 | hipBLASLt padded 795.9 |
| 1024x5120x6400 | bfloat16 | 80 | one-tile owner, n = 2 |  | 921.0 | 946.5 | 675.4 | 675.5 | 674.3 | 672.5 | flydsl | 702.9 | 537.0 (copy) | 1061.9 | 1068.9 | hipBLASLt padded 1068.9 |
| 2048x5120x6400 | bfloat16 | 160 | data-parallel | yes | 1200.5 | 1222.2 | 1213.3 | 1202.3 | 1238.9 | 1233.6 | flydsl | 785.4 | 646.7 (copy) | 1281.8 | 1282.7 | hipBLASLt padded 1282.7 |
| 4096x5120x6400 | bfloat16 | 320 | one-tile owner, n = 3 |  | 1390.4 | 1398.9 | 1175.9 | 1175.5 | 1179.8 | 1175.2 | torch | 1285.6 | 1296.3 | 1298.1 | 1299.9 | hipBLASLt padded 1299.9 |
| 8192x5120x6400 | bfloat16 | 640 | one-tile owner, n = 2 |  | 1516.5 | 1520.1 | 1451.5 | 1451.1 | 1447.0 | 1446.6 | flydsl | 1411.9 | 1192.0 (copy) | 1374.3 | 1371.3 | AITER flydsl contiguous 1411.9 |
| 16384x5120x6400 | bfloat16 | 1280 | data-parallel |  | 1570.5 | 1578.3 | 1570.5 | 1578.3 | 1561.1 | 1565.2 | torch | 1544.6 | 1562.2 | 1572.8 | 1566.0 | hipBLASLt contiguous 1572.8 |
| 32768x5120x6400 | bfloat16 | 2560 | data-parallel |  | 1573.6 | 1588.2 | 1573.6 | 1588.2 | 1563.6 | 1564.6 | torch | 1525.7 | 1563.0 | 1539.0 | 1557.9 | AITER torch padded 1563.0 |
| 512x16384x6656 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1147.9 | 1162.8 | 1056.5 | 1057.7 | 1049.4 | 1057.3 | flydsl | 951.3 | 515.2 (copy) | 927.7 | 900.2 | AITER flydsl contiguous 951.3 |
| 1024x16384x6656 | bfloat16 | 256 | data-parallel | yes | 1563.4 | 1569.6 | 1557.9 | 1550.3 | 1563.7 | 1555.4 | asm | 452.1 | 433.8 | 1553.3 | 1550.1 | hipBLASLt contiguous 1553.3 |
| 2048x16384x6656 | bfloat16 | 512 | data-parallel |  | 1535.8 | 1538.8 | 1535.8 | 1538.8 | 1531.7 | 1542.9 | asm | 1471.7 | 1480.3 | 1561.8 | 1556.5 | hipBLASLt contiguous 1561.8 |
| 4096x16384x6656 | bfloat16 | 1024 | data-parallel |  | 1564.0 | 1573.9 | 1564.0 | 1573.9 | 1560.2 | 1565.3 | torch | 1566.3 | 1577.6 | 1579.0 | 1570.2 | hipBLASLt contiguous 1579.0 |
| 8192x16384x6656 | bfloat16 | 2048 | data-parallel |  | 1595.8 | 1607.3 | 1595.8 | 1607.3 | 1577.7 | 1589.9 | asm | 1522.0 | 1529.1 | 1560.5 | 1575.4 | hipBLASLt padded 1575.4 |
| 16384x16384x6656 | bfloat16 | 4096 | data-parallel |  | 1594.0 | 1606.4 | 1594.0 | 1606.4 | 1576.2 | 1580.9 | asm | 1535.4 | 1543.8 | 1572.4 | 1570.0 | hipBLASLt contiguous 1572.4 |
| 32768x16384x6656 | bfloat16 | 8192 | data-parallel |  | 1598.2 | 1614.3 | 1598.2 | 1614.3 | 1580.9 | 1587.2 | asm | 291.3 | 292.8 | 1562.3 | 1562.2 | hipBLASLt contiguous 1562.3 |
| 512x1024x7168 | bfloat16 | 8 | one-tile reduce-scatter, n = 14 |  | 258.2 | 254.8 | 70.6 | 70.4 | 70.0 | 69.8 | torch | 435.5 | 434.8 | 438.3 | 433.8 | hipBLASLt contiguous 438.3 |
| 512x1536x7168 | bfloat16 | 12 | one-tile reduce-scatter, n = 10 |  | 336.9 | 344.0 | 105.5 | 105.8 | 104.8 | 104.3 | torch | 453.4 | 456.4 | 454.3 | 456.2 | AITER torch padded 456.4 |
| 512x2048x7168 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 |  | 410.6 | 416.2 | 140.3 | 140.6 | 138.8 | 138.3 | torch | 578.7 | 597.1 | 579.3 | 598.3 | hipBLASLt padded 598.3 |
| 512x3584x7168 | bfloat16 | 28 | one-tile reduce-scatter, n = 4 |  | 522.9 | 517.9 | 245.8 | 246.0 | 242.3 | 241.7 | flydsl | 613.9 | 408.6 (copy) | 633.2 | 625.1 | hipBLASLt contiguous 633.2 |
| 512x6144x7168 | bfloat16 | 48 | one-tile owner, n = 3 |  | 755.8 | 762.9 | 414.5 | 419.5 | 413.7 | 411.5 | flydsl | 823.2 | 473.2 (copy) | 817.4 | 822.6 | AITER flydsl contiguous 823.2 |
| 512x8192x7168 | bfloat16 | 64 | one-tile owner, n = 3 |  | 913.0 | 934.7 | 555.9 | 552.9 | 548.1 | 545.4 | flydsl | 879.4 | 457.0 (copy) | 898.2 | 895.1 | hipBLASLt contiguous 898.2 |
| 512x8448x7168 | bfloat16 | 66 | one-tile owner, n = 3-4 |  | 887.8 | 897.2 | 570.4 | 566.5 | 565.0 | 563.8 | flydsl | 749.9 | 430.4 (copy) | 471.6 | 464.4 | AITER flydsl contiguous 749.9 |
| 512x20480x7168 | bfloat16 | 160 | data-parallel | yes | 1091.8 | 1085.9 | 1091.5 | 1074.4 | 1110.4 | 1095.8 | flydsl | 1037.7 | 582.7 (copy) | 1156.4 | 1152.9 | hipBLASLt contiguous 1156.4 |
| 768x1024x7168 | bfloat16 | 12 | one-tile reduce-scatter, n = 10 |  | 341.1 | 348.6 | 105.4 | 105.7 | 104.8 | 104.3 | torch | 445.1 | 447.9 | 445.3 | 454.1 | hipBLASLt padded 454.1 |
| 768x2048x7168 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 |  | 498.6 | 498.6 | 211.2 | 211.2 | 208.5 | 208.1 | flydsl | 608.9 | 402.0 (copy) | 642.4 | 637.0 | hipBLASLt contiguous 642.4 |
| 1024x512x7168 | bfloat16 | 8 | one-tile reduce-scatter, n = 14 |  | 256.9 | 253.4 | 70.4 | 70.7 | 70.0 | 69.9 | flydsl | 381.0 | 231.6 (copy) | 325.1 | 337.7 | AITER flydsl contiguous 381.0 |
| 1024x1024x7168 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 |  | 413.4 | 417.0 | 141.0 | 141.0 | 139.3 | 138.8 | torch | 588.1 | 608.4 | 589.6 | 605.3 | AITER torch padded 608.4 |
| 1024x1536x7168 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 |  | 499.9 | 501.2 | 209.9 | 210.0 | 208.7 | 207.9 | flydsl | 632.6 | 433.5 (copy) | 636.1 | 632.1 | hipBLASLt contiguous 636.1 |
| 1024x2048x7168 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 575.4 | 570.6 | 281.2 | 280.7 | 275.5 | 275.5 | torch | 728.3 | 732.8 | 727.4 | 728.6 | AITER torch padded 732.8 |
| 1024x3584x7168 | bfloat16 | 56 | one-tile owner, n = 3 |  | 836.9 | 858.1 | 484.2 | 483.7 | 480.3 | 477.5 | flydsl | 870.8 | 622.6 (copy) | 828.4 | 828.2 | AITER flydsl contiguous 870.8 |
| 1024x6144x7168 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 1026.7 | 1045.4 | 824.6 | 814.3 | 810.7 | 812.3 | torch | 1061.5 | 1028.4 | 1071.6 | 1027.2 | hipBLASLt contiguous 1071.6 |
| 1024x8192x7168 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1257.6 | 1281.6 | 1059.4 | 1068.5 | 1062.7 | 1069.6 | flydsl | 1098.6 | 735.6 (copy) | 984.0 | 972.6 | AITER flydsl contiguous 1098.6 |
| 1024x8448x7168 | bfloat16 | 132 | data-parallel | yes | 1077.3 | 1094.2 | 1085.9 | 1085.7 | 1097.8 | 1099.9 | torch | 1188.9 | 1176.5 | 1196.7 | 1181.1 | hipBLASLt contiguous 1196.7 |
| 1024x20480x7168 | bfloat16 | 320 | one-tile owner, n = 3 |  | 1313.6 | 1314.0 | 1127.3 | 1111.9 | 1119.2 | 1110.8 | flydsl | 779.1 | 602.1 (copy) | 1209.2 | 1207.8 | hipBLASLt contiguous 1209.2 |
| 1280x512x7168 | bfloat16 | 10 | one-tile reduce-scatter, n = 12 |  | 299.0 | 300.0 | 88.1 | 88.2 | 87.4 | 87.1 | torch | 413.6 | 410.4 | 414.6 | 418.5 | hipBLASLt padded 418.5 |
| 1280x1024x7168 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 |  | 448.9 | 450.0 | 176.0 | 175.6 | 173.5 | 173.2 | torch | 599.3 | 595.6 | 602.6 | 595.8 | hipBLASLt contiguous 602.6 |
| 1280x2048x7168 | bfloat16 | 40 | one-tile owner, n = 4 |  | 663.5 | 669.4 | 349.8 | 349.9 | 346.2 | 344.8 | torch | 855.9 | 840.4 | 859.1 | 825.1 | hipBLASLt contiguous 859.1 |
| 1536x512x7168 | bfloat16 | 12 | one-tile reduce-scatter, n = 10 |  | 340.7 | 349.3 | 105.9 | 105.5 | 105.0 | 104.5 | torch | 447.3 | 448.8 | 450.3 | 457.1 | hipBLASLt padded 457.1 |
| 1536x1024x7168 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 |  | 503.1 | 500.4 | 210.8 | 210.5 | 208.4 | 207.6 | torch | 754.3 | 745.4 | 753.7 | 745.2 | AITER torch contiguous 754.3 |
| 1536x2048x7168 | bfloat16 | 48 | one-tile owner, n = 3 |  | 747.4 | 758.5 | 417.8 | 415.5 | 412.1 | 410.4 | torch | 966.3 | 967.4 | 967.2 | 965.3 | AITER torch padded 967.4 |
| 1792x512x7168 | bfloat16 | 14 | one-tile reduce-scatter, n = 9 |  | 379.5 | 378.3 | 123.6 | 123.1 | 122.1 | 121.7 | torch | 493.0 | 494.2 | 494.0 | 494.8 | hipBLASLt padded 494.8 |
| 1792x1024x7168 | bfloat16 | 28 | one-tile reduce-scatter, n = 4 |  | 525.6 | 518.6 | 247.0 | 245.6 | 243.3 | 242.8 | flydsl | 634.6 | 443.7 (copy) | 661.9 | 650.0 | hipBLASLt contiguous 661.9 |
| 1792x2048x7168 | bfloat16 | 56 | one-tile owner, n = 3 |  | 834.3 | 858.2 | 485.2 | 483.4 | 481.3 | 478.7 | flydsl | 906.0 | 662.5 (copy) | 814.6 | 813.9 | AITER flydsl contiguous 906.0 |
| 2048x512x7168 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 |  | 414.2 | 416.7 | 141.5 | 141.4 | 139.4 | 139.0 | torch | 582.9 | 589.0 | 583.8 | 582.9 | AITER torch padded 589.0 |
| 2048x1024x7168 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 579.2 | 572.8 | 281.1 | 280.0 | 277.5 | 276.6 | torch | 805.9 | 802.0 | 810.0 | 799.6 | hipBLASLt contiguous 810.0 |
| 2048x2048x7168 | bfloat16 | 64 | one-tile owner, n = 3 |  | 940.7 | 943.1 | 551.8 | 551.1 | 547.9 | 544.3 | flydsl | 989.9 | 728.4 (copy) | 896.4 | 895.8 | AITER flydsl contiguous 989.9 |
| 2048x3584x7168 | bfloat16 | 112 | one-tile owner, n = 2-3 |  | 1131.9 | 1139.5 | 941.6 | 934.8 | 944.0 | 942.2 | opus | 1220.0 | 1231.3 | 1051.2 | 1038.1 | AITER opus padded 1231.3 |
| 2048x8192x7168 | bfloat16 | 256 | data-parallel | yes | 1587.9 | 1593.5 | 1581.5 | 1581.4 | 1583.9 | 1585.7 | asm | 1554.2 | 1558.1 | 1552.0 | 1528.9 | AITER asm padded 1558.1 |
| 2048x20480x7168 | bfloat16 | 640 | one-tile owner, n = 2 |  | 1485.1 | 1486.1 | 1409.5 | 1409.8 | 1394.9 | 1395.8 | flydsl | 825.8 | 714.9 (copy) | 1289.5 | 1304.9 | hipBLASLt padded 1304.9 |
| 4096x512x7168 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 579.9 | 572.7 | 281.3 | 281.3 | 277.4 | 276.7 | flydsl | 709.0 | 455.6 (copy) | 692.6 | 684.4 | AITER flydsl contiguous 709.0 |
| 4096x1024x7168 | bfloat16 | 64 | one-tile owner, n = 3 |  | 907.9 | 941.0 | 553.3 | 555.1 | 546.3 | 545.5 | torch | 976.8 | 964.2 | 977.1 | 961.5 | hipBLASLt contiguous 977.1 |
| 4096x2048x7168 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1258.2 | 1282.6 | 1056.5 | 1060.5 | 1066.0 | 1068.5 | torch | 1299.2 | 1297.6 | 1305.6 | 1301.6 | hipBLASLt contiguous 1305.6 |
| 4096x3584x7168 | bfloat16 | 224 | data-parallel | yes | 1463.5 | 1470.5 | 1466.9 | 1458.8 | 1464.0 | 1464.3 | flydsl | 1400.6 | 1075.6 (copy) | 1395.2 | 1393.8 | AITER flydsl contiguous 1400.6 |
| 4096x8192x7168 | bfloat16 | 512 | data-parallel |  | 1602.8 | 1603.1 | 1602.8 | 1603.1 | 1599.9 | 1598.9 | flydsl | 1547.6 | 1264.2 (copy) | 1566.0 | 1566.2 | hipBLASLt padded 1566.2 |
| 4096x20480x7168 | bfloat16 | 1280 | data-parallel |  | 1581.3 | 1592.4 | 1581.3 | 1592.4 | 1572.7 | 1576.2 | flydsl | 1521.4 | 1316.6 (copy) | 1493.8 | 1490.3 | AITER flydsl contiguous 1521.4 |
| 6144x512x7168 | bfloat16 | 48 | one-tile owner, n = 3 |  | 747.9 | 756.9 | 418.3 | 419.3 | 414.5 | 412.9 | flydsl | 800.1 | 468.6 (copy) | 757.8 | 738.3 | AITER flydsl contiguous 800.1 |
| 6144x1024x7168 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 1030.6 | 1025.9 | 818.0 | 809.5 | 814.6 | 810.9 | opus | 1191.6 | 1195.7 | 889.3 | 886.9 | AITER opus padded 1195.7 |
| 6144x2048x7168 | bfloat16 | 192 | data-parallel | yes | 1350.0 | 1353.9 | 1341.5 | 1342.9 | 1356.4 | 1354.4 | opus | 1469.3 | 1481.5 | 1368.8 | 1367.0 | AITER opus padded 1481.5 |
| 8192x512x7168 | bfloat16 | 64 | one-tile owner, n = 3 |  | 917.1 | 934.3 | 550.4 | 549.2 | 546.1 | 543.9 | flydsl | 969.3 | 494.1 (copy) | 898.4 | 898.1 | AITER flydsl contiguous 969.3 |
| 8192x1024x7168 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1268.2 | 1277.8 | 1057.5 | 1060.3 | 1066.6 | 1065.5 | torch | 1153.0 | 1142.5 | 1159.5 | 1148.1 | hipBLASLt contiguous 1159.5 |
| 8192x2048x7168 | bfloat16 | 256 | data-parallel | yes | 1580.0 | 1589.6 | 1578.4 | 1580.3 | 1583.0 | 1582.4 | torch | 1446.4 | 1460.6 | 1453.5 | 1460.9 | hipBLASLt padded 1460.9 |
| 8192x3584x7168 | bfloat16 | 448 | data-parallel |  | 1497.5 | 1494.7 | 1497.5 | 1494.7 | 1491.6 | 1492.5 | flydsl | 1451.8 | 1179.3 (copy) | 1397.5 | 1397.7 | AITER flydsl contiguous 1451.8 |
| 8192x8192x7168 | bfloat16 | 1024 | data-parallel |  | 1590.2 | 1594.5 | 1590.2 | 1594.5 | 1584.9 | 1588.2 | flydsl | 1532.6 | 1347.1 (copy) | 1586.1 | 1579.9 | hipBLASLt contiguous 1586.1 |
| 8192x20480x7168 | bfloat16 | 2560 | data-parallel |  | 1608.5 | 1613.3 | 1608.5 | 1613.3 | 1594.3 | 1594.0 | asm | 1531.4 | 1551.6 | 1504.9 | 1507.1 | AITER asm padded 1551.6 |
| 10240x512x7168 | bfloat16 | 80 | one-tile owner, n = 3-4 |  | 1014.6 | 1004.8 | 677.0 | 687.2 | 678.6 | 679.7 | torch | 826.9 | 864.7 | 829.2 | 865.1 | hipBLASLt padded 865.1 |
| 10240x1024x7168 | bfloat16 | 160 | data-parallel | yes | 1224.2 | 1240.8 | 1234.0 | 1233.0 | 1241.2 | 1239.4 | torch | 1310.5 | 1312.4 | 1316.6 | 1315.3 | hipBLASLt contiguous 1316.6 |
| 10240x2048x7168 | bfloat16 | 320 | one-tile owner, n = 3 |  | 1412.7 | 1413.6 | 1185.2 | 1176.2 | 1185.7 | 1180.7 | torch | 1325.6 | 1330.5 | 1328.5 | 1333.6 | hipBLASLt padded 1333.6 |
| 10240x8448x7168 | bfloat16 | 1320 | one-tile owner, n = 4 |  | 1504.0 | 1518.7 | 1436.2 | 1440.2 | 1420.2 | 1426.5 | asm | 1413.3 | 1418.4 | 1370.0 | 1382.1 | AITER asm padded 1418.4 |
| 10240x20480x7168 | bfloat16 | 3200 | one-tile owner, n = 2 |  | 1522.0 | 1537.8 | 1510.8 | 1521.6 | 1485.4 | 1493.8 | asm | 1505.5 | 1519.5 | 1469.7 | 1481.5 | AITER asm padded 1519.5 |
| 12288x512x7168 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 1024.2 | 984.8 | 812.0 | 814.3 | 812.7 | 811.5 | opus | 1192.1 | 1188.7 | 880.5 | 877.6 | AITER opus contiguous 1192.1 |
| 12288x1024x7168 | bfloat16 | 192 | data-parallel | yes | 1347.8 | 1358.5 | 1348.2 | 1349.7 | 1357.8 | 1358.7 | opus | 1473.8 | 1482.1 | 1368.9 | 1367.3 | AITER opus padded 1482.1 |
| 12288x2048x7168 | bfloat16 | 384 | one-tile owner, n = 2 |  | 1470.2 | 1471.7 | 1362.5 | 1362.5 | 1355.4 | 1353.8 | opus | 1473.5 | 1480.1 | 1382.2 | 1377.7 | AITER opus padded 1480.1 |
| 14336x512x7168 | bfloat16 | 112 | one-tile owner, n = 2-3 |  | 1084.3 | 1079.3 | 934.4 | 942.3 | 943.7 | 946.3 | opus | 1253.2 | 1260.0 | 880.6 | 955.0 | AITER opus padded 1260.0 |
| 14336x1024x7168 | bfloat16 | 224 | data-parallel | yes | 1451.2 | 1463.7 | 1454.5 | 1449.8 | 1454.9 | 1452.7 | torch | 1409.7 | 1417.6 | 1415.3 | 1419.2 | hipBLASLt padded 1419.2 |
| 14336x2048x7168 | bfloat16 | 448 | data-parallel |  | 1465.2 | 1463.1 | 1465.2 | 1463.1 | 1445.8 | 1462.0 | flydsl | 1418.7 | 1079.6 (copy) | 1392.0 | 1391.4 | AITER flydsl contiguous 1418.7 |
| 16384x512x7168 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1164.8 | 1169.8 | 1053.8 | 1008.0 | 1063.5 | 1038.2 | opus | 1327.1 | 1205.1 | 1052.6 | 1016.7 | AITER opus contiguous 1327.1 |
| 16384x1024x7168 | bfloat16 | 256 | data-parallel | yes | 1512.9 | 1525.2 | 1511.2 | 1519.9 | 1520.6 | 1526.2 | torch | 1521.4 | 1526.8 | 1531.3 | 1527.1 | hipBLASLt contiguous 1531.3 |
| 16384x2048x7168 | bfloat16 | 512 | data-parallel |  | 1533.5 | 1543.9 | 1533.5 | 1543.9 | 1533.4 | 1542.1 | torch | 1545.4 | 1544.1 | 1561.8 | 1541.2 | hipBLASLt contiguous 1561.8 |
| 16384x3584x7168 | bfloat16 | 896 | one-tile owner, n = 2 |  | 1511.6 | 1519.0 | 1456.1 | 1459.8 | 1442.2 | 1449.2 | flydsl | 1393.1 | 1200.6 (copy) | 1394.5 | 1393.6 | hipBLASLt contiguous 1394.5 |
| 16384x8192x7168 | bfloat16 | 2048 | data-parallel |  | 1583.3 | 1597.4 | 1583.3 | 1597.4 | 1573.3 | 1576.2 | torch | 1548.6 | 1586.2 | 1559.5 | 1577.2 | AITER torch padded 1586.2 |
| 16384x20480x7168 | bfloat16 | 5120 | data-parallel |  | 1603.3 | 1615.2 | 1603.3 | 1615.2 | 1584.0 | 1587.9 | flydsl | 1534.9 | 1456.2 (copy) | 1513.8 | 1515.6 | AITER flydsl contiguous 1534.9 |
| 32768x512x7168 | bfloat16 | 256 | data-parallel | yes | 1375.9 | 1365.6 | 1384.2 | 1392.7 | 1395.1 | 1382.5 | torch | 1382.6 | 1388.6 | 1396.6 | 1384.3 | hipBLASLt contiguous 1396.6 |
| 32768x1024x7168 | bfloat16 | 512 | data-parallel |  | 1497.9 | 1508.0 | 1497.9 | 1508.0 | 1499.7 | 1507.2 | torch | 1446.3 | 1470.2 | 1460.2 | 1472.0 | hipBLASLt padded 1472.0 |
| 32768x2048x7168 | bfloat16 | 1024 | data-parallel |  | 1546.3 | 1559.7 | 1546.3 | 1559.7 | 1537.7 | 1545.4 | torch | 1489.7 | 1533.0 | 1515.7 | 1532.0 | AITER torch padded 1533.0 |
| 32768x3584x7168 | bfloat16 | 1792 | data-parallel |  | 1558.4 | 1570.0 | 1558.4 | 1570.0 | 1544.0 | 1548.5 | torch | 1525.5 | 1552.3 | 1537.7 | 1552.1 | AITER torch padded 1552.3 |
| 32768x8192x7168 | bfloat16 | 4096 | data-parallel |  | 1596.1 | 1605.6 | 1596.1 | 1605.6 | 1575.3 | 1574.8 | torch | 1555.1 | 1589.5 | 1568.7 | 1586.4 | AITER torch padded 1589.5 |
| 32768x20480x7168 | bfloat16 | 10240 | data-parallel |  | 1611.6 | 1609.5 | 1611.6 | 1609.5 | 1589.0 | 1591.7 | asm | 1555.3 | 1559.4 | 1502.2 | 1489.7 | AITER asm padded 1559.4 |
| 512x1280x8192 | bfloat16 | 10 | one-tile reduce-scatter, n = 12 |  | 320.0 | 324.5 | 89.2 | 89.1 | 88.3 | 87.9 | torch | 472.2 | 463.9 | 472.0 | 459.6 | AITER torch contiguous 472.2 |
| 512x2560x8192 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 |  | 493.8 | 484.4 | 178.1 | 177.3 | 175.0 | 174.7 | torch | 606.6 | 600.2 | 606.8 | 602.8 | hipBLASLt contiguous 606.8 |
| 512x4096x8192 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 623.4 | 612.1 | 282.9 | 282.6 | 279.2 | 278.7 | flydsl | 759.7 | 474.8 (copy) | 775.7 | 710.2 | hipBLASLt contiguous 775.7 |
| 512x4608x8192 | bfloat16 | 36 | one-tile owner, n = 4 |  | 662.6 | 657.4 | 319.2 | 318.5 | 313.1 | 312.6 | torch | 849.3 | 825.5 | 850.8 | 826.1 | hipBLASLt contiguous 850.8 |
| 512x7168x8192 | bfloat16 | 56 | one-tile owner, n = 3 |  | 883.1 | 895.8 | 490.1 | 486.5 | 485.0 | 483.3 | flydsl | 921.9 | 481.3 (copy) | 874.5 | 869.5 | AITER flydsl contiguous 921.9 |
| 512x8192x8192 | bfloat16 | 64 | one-tile owner, n = 3 |  | 961.3 | 961.0 | 557.8 | 555.1 | 552.0 | 549.7 | flydsl | 875.8 | 446.0 (copy) | 906.0 | 899.1 | hipBLASLt contiguous 906.0 |
| 512x10240x8192 | bfloat16 | 80 | one-tile owner, n = 3-4 |  | 1065.7 | 1075.6 | 695.9 | 696.4 | 689.6 | 686.9 | flydsl | 689.9 | 414.7 (copy) | 571.5 | 559.1 | AITER flydsl contiguous 689.9 |
| 512x14336x8192 | bfloat16 | 112 | one-tile owner, n = 2-3 |  | 1106.0 | 1103.8 | 948.3 | 944.5 | 958.6 | 951.8 | torch | 1043.5 | 1036.6 | 1038.6 | 1041.9 | AITER torch contiguous 1043.5 |
| 512x16384x8192 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1173.6 | 1199.2 | 990.1 | 950.7 | 997.7 | 962.9 | flydsl | 680.6 | 438.1 (copy) | 921.5 | 983.6 | hipBLASLt padded 983.6 |
| 512x57344x8192 | bfloat16 | 448 | data-parallel |  | 1344.6 | 1359.8 | 1344.6 | 1359.8 | 1346.7 | 1357.9 | flydsl | 1298.9 | 688.2 (copy) | 1271.4 | 1286.8 | AITER flydsl contiguous 1298.9 |
| 768x4096x8192 | bfloat16 | 48 | one-tile owner, n = 4 |  | 818.5 | 816.8 | 423.5 | 422.0 | 416.0 | 414.9 | flydsl | 840.7 | 569.2 (copy) | 788.9 | 766.3 | AITER flydsl contiguous 840.7 |
| 1024x512x8192 | bfloat16 | 8 | one-tile reduce-scatter, n = 16 |  | 297.4 | 300.0 | 71.6 | 71.5 | 70.6 | 70.4 | torch | 365.2 | 370.6 | 365.2 | 385.8 | hipBLASLt padded 385.8 |
| 1024x1280x8192 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 |  | 493.2 | 487.9 | 178.1 | 178.0 | 175.5 | 175.1 | torch | 620.9 | 608.0 | 619.3 | 619.3 | AITER torch contiguous 620.9 |
| 1024x2560x8192 | bfloat16 | 40 | one-tile owner, n = 4 |  | 716.0 | 711.9 | 353.5 | 353.9 | 349.0 | 347.9 | torch | 878.7 | 865.6 | 882.7 | 858.0 | hipBLASLt contiguous 882.7 |
| 1024x4096x8192 | bfloat16 | 64 | one-tile owner, n = 3 |  | 961.2 | 985.2 | 553.0 | 555.1 | 550.7 | 550.1 | flydsl | 993.9 | 688.5 (copy) | 911.7 | 895.2 | AITER flydsl contiguous 993.9 |
| 1024x4608x8192 | bfloat16 | 72 | one-tile owner, n = 3-4 |  | 987.4 | 997.1 | 622.9 | 624.5 | 620.2 | 619.3 | torch | 991.9 | 995.0 | 993.4 | 988.5 | AITER torch padded 995.0 |
| 1024x7168x8192 | bfloat16 | 112 | one-tile owner, n = 2-3 |  | 1205.9 | 1174.8 | 950.9 | 946.5 | 953.8 | 952.4 | flydsl | 836.4 | 578.4 (copy) | 1096.7 | 1109.4 | hipBLASLt padded 1109.4 |
| 1024x8192x8192 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1307.8 | 1325.8 | 1081.7 | 1075.2 | 1072.6 | 1079.3 | flydsl | 1159.1 | 660.3 (copy) | 1005.8 | 1002.5 | AITER flydsl contiguous 1159.1 |
| 1024x10240x8192 | bfloat16 | 160 | data-parallel | yes | 1257.2 | 1265.1 | 1262.8 | 1263.7 | 1273.5 | 1272.2 | flydsl | 872.0 | 629.1 (copy) | 1309.3 | 1306.1 | hipBLASLt contiguous 1309.3 |
| 1024x14336x8192 | bfloat16 | 224 | data-parallel | yes | 1425.1 | 1435.3 | 1427.4 | 1419.8 | 1427.5 | 1439.3 | flydsl | 1417.0 | 892.1 (copy) | 1359.5 | 1350.9 | AITER flydsl contiguous 1417.0 |
| 1024x16384x8192 | bfloat16 | 256 | data-parallel | yes | 1496.4 | 1513.0 | 1498.0 | 1502.8 | 1503.5 | 1511.0 | torch | 1489.1 | 1488.8 | 1501.3 | 1487.5 | hipBLASLt contiguous 1501.3 |
| 1024x57344x8192 | bfloat16 | 896 | one-tile owner, n = 2 |  | 1496.8 | 1498.9 | 1432.2 | 1436.2 | 1423.6 | 1422.7 | asm | 1381.0 | 1393.6 | 1330.0 | 1338.4 | AITER asm padded 1393.6 |
| 2048x512x8192 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 |  | 450.2 | 452.1 | 143.0 | 142.7 | 140.4 | 140.4 | torch | 531.4 | 539.1 | 534.1 | 543.6 | hipBLASLt padded 543.6 |
| 2048x1280x8192 | bfloat16 | 40 | one-tile owner, n = 4 |  | 715.1 | 709.6 | 353.0 | 354.6 | 349.5 | 349.0 | torch | 727.7 | 726.7 | 728.8 | 732.1 | hipBLASLt padded 732.1 |
| 2048x2560x8192 | bfloat16 | 80 | one-tile owner, n = 3-4 |  | 1077.5 | 1080.3 | 697.8 | 693.1 | 687.8 | 686.9 | torch | 1022.7 | 1021.9 | 1022.2 | 1025.8 | hipBLASLt padded 1025.8 |
| 2048x4096x8192 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1312.8 | 1324.7 | 1079.0 | 1078.1 | 1079.7 | 1076.6 | opus | 1299.5 | 1306.3 | 1005.2 | 1016.7 | AITER opus padded 1306.3 |
| 2048x4608x8192 | bfloat16 | 144 | data-parallel | yes | 1181.9 | 1184.5 | 1178.9 | 1182.3 | 1189.5 | 1187.7 | torch | 1259.0 | 1266.1 | 1265.3 | 1268.4 | hipBLASLt padded 1268.4 |
| 2048x7168x8192 | bfloat16 | 224 | data-parallel | yes | 1480.4 | 1484.7 | 1479.9 | 1478.2 | 1485.3 | 1477.0 | asm | 1454.6 | 1455.7 | 1408.5 | 1405.6 | AITER asm padded 1455.7 |
| 2048x8192x8192 | bfloat16 | 256 | data-parallel | yes | 1596.6 | 1605.7 | 1603.8 | 1596.5 | 1603.9 | 1597.5 | torch | 1552.5 | 1565.5 | 1568.7 | 1545.0 | hipBLASLt contiguous 1568.7 |
| 2048x10240x8192 | bfloat16 | 320 | one-tile owner, n = 3 |  | 1423.3 | 1422.3 | 1191.6 | 1190.8 | 1193.0 | 1193.5 | flydsl | 990.1 | 764.6 (copy) | 1318.1 | 1318.1 | hipBLASLt contiguous 1318.1 |
| 2048x14336x8192 | bfloat16 | 448 | data-parallel |  | 1454.2 | 1451.6 | 1454.2 | 1451.6 | 1448.8 | 1452.4 | flydsl | 1410.4 | 1100.2 (copy) | 1352.5 | 1360.2 | AITER flydsl contiguous 1410.4 |
| 2048x16384x8192 | bfloat16 | 512 | data-parallel |  | 1557.2 | 1558.2 | 1557.2 | 1558.2 | 1557.9 | 1559.2 | asm | 1499.8 | 1502.7 | 1555.8 | 1554.9 | hipBLASLt contiguous 1555.8 |
| 2048x57344x8192 | bfloat16 | 1792 | data-parallel |  | 1585.6 | 1587.3 | 1585.6 | 1587.3 | 1569.1 | 1572.6 | flydsl | 1516.6 | 1207.6 (copy) | 1487.2 | 1481.3 | AITER flydsl contiguous 1516.6 |
| 4096x512x8192 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 617.7 | 611.3 | 284.1 | 284.4 | 279.1 | 278.2 | flydsl | 745.4 | 468.7 (copy) | 771.3 | 713.4 | hipBLASLt contiguous 771.3 |
| 4096x1280x8192 | bfloat16 | 80 | one-tile owner, n = 3-4 |  | 1060.1 | 1074.3 | 692.6 | 691.6 | 687.7 | 687.4 | torch | 1034.6 | 1027.9 | 1031.0 | 1037.7 | hipBLASLt padded 1037.7 |
| 4096x2560x8192 | bfloat16 | 160 | data-parallel | yes | 1262.7 | 1265.8 | 1260.9 | 1263.5 | 1272.3 | 1267.0 | torch | 1302.5 | 1308.3 | 1309.5 | 1309.5 | hipBLASLt contiguous 1309.5 |
| 4096x4096x8192 | bfloat16 | 256 | data-parallel | yes | 1599.6 | 1607.7 | 1600.9 | 1597.4 | 1608.2 | 1600.3 | torch | 1537.5 | 1547.3 | 1556.0 | 1544.8 | hipBLASLt contiguous 1556.0 |
| 4096x4608x8192 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 |  | 1399.5 | 1401.6 | 1097.7 | 1096.5 | 1086.4 | 1086.9 | torch | 1264.3 | 1276.7 | 1273.9 | 1278.5 | hipBLASLt padded 1278.5 |
| 4096x7168x8192 | bfloat16 | 448 | data-parallel |  | 1497.9 | 1498.1 | 1497.9 | 1498.1 | 1495.8 | 1494.5 | flydsl | 1463.0 | 1196.3 (copy) | 1418.1 | 1416.2 | AITER flydsl contiguous 1463.0 |
| 4096x8192x8192 | bfloat16 | 512 | data-parallel |  | 1614.4 | 1611.0 | 1614.4 | 1611.0 | 1608.0 | 1604.4 | asm | 1592.0 | 1574.6 | 1586.5 | 1579.8 | AITER asm contiguous 1592.0 |
| 4096x10240x8192 | bfloat16 | 640 | one-tile owner, n = 2 |  | 1526.8 | 1529.9 | 1445.6 | 1441.1 | 1431.9 | 1432.7 | flydsl | 1394.3 | 1190.7 (copy) | 1363.8 | 1362.0 | AITER flydsl contiguous 1394.3 |
| 4096x14336x8192 | bfloat16 | 896 | one-tile owner, n = 2 |  | 1549.9 | 1548.9 | 1481.6 | 1482.3 | 1468.6 | 1470.9 | flydsl | 1422.6 | 1230.7 (copy) | 1373.9 | 1379.2 | AITER flydsl contiguous 1422.6 |
| 4096x16384x8192 | bfloat16 | 1024 | data-parallel |  | 1587.9 | 1596.0 | 1587.9 | 1596.0 | 1579.1 | 1586.3 | flydsl | 1523.7 | 1315.4 (copy) | 1549.5 | 1544.9 | hipBLASLt contiguous 1549.5 |
| 4096x57344x8192 | bfloat16 | 3584 | data-parallel |  | 1605.1 | 1607.8 | 1605.1 | 1607.8 | 1593.0 | 1593.4 | flydsl | 1535.1 | 1355.2 (copy) | 1513.4 | 1518.6 | AITER flydsl contiguous 1535.1 |
| 8192x512x8192 | bfloat16 | 64 | one-tile owner, n = 3 |  | 951.8 | 978.1 | 553.4 | 555.4 | 550.3 | 549.5 | flydsl | 978.3 | 509.8 (copy) | 913.4 | 897.1 | AITER flydsl contiguous 978.3 |
| 8192x1280x8192 | bfloat16 | 160 | data-parallel | yes | 1256.4 | 1264.4 | 1253.4 | 1260.3 | 1268.0 | 1265.1 | torch | 1300.0 | 1304.7 | 1308.5 | 1308.0 | hipBLASLt contiguous 1308.5 |
| 8192x2560x8192 | bfloat16 | 320 | one-tile owner, n = 3 |  | 1445.1 | 1446.6 | 1199.6 | 1196.5 | 1189.0 | 1194.0 | torch | 1304.0 | 1313.0 | 1312.2 | 1314.6 | hipBLASLt padded 1314.6 |
| 8192x4096x8192 | bfloat16 | 512 | data-parallel |  | 1617.4 | 1612.6 | 1617.4 | 1612.6 | 1609.2 | 1608.4 | torch | 1587.7 | 1578.4 | 1582.0 | 1578.5 | AITER torch contiguous 1587.7 |
| 8192x4608x8192 | bfloat16 | 576 | one-tile owner, n = 3 |  | 1507.2 | 1505.1 | 1361.7 | 1364.8 | 1356.9 | 1357.4 | torch | 1374.0 | 1381.5 | 1382.1 | 1384.3 | hipBLASLt padded 1384.3 |
| 8192x7168x8192 | bfloat16 | 896 | one-tile owner, n = 2 |  | 1563.5 | 1564.3 | 1494.8 | 1489.6 | 1485.5 | 1482.4 | asm | 1437.1 | 1443.0 | 1419.9 | 1414.8 | AITER asm padded 1443.0 |
| 8192x8192x8192 | bfloat16 | 1024 | data-parallel |  | 1604.4 | 1610.5 | 1604.4 | 1610.5 | 1602.1 | 1600.2 | asm | 1574.3 | 1578.7 | 1574.8 | 1570.7 | AITER asm padded 1578.7 |
| 8192x10240x8192 | bfloat16 | 1280 | data-parallel |  | 1604.6 | 1611.1 | 1604.6 | 1611.1 | 1601.5 | 1602.3 | flydsl | 1548.7 | 1383.0 (copy) | 1587.2 | 1584.4 | hipBLASLt contiguous 1587.2 |
| 8192x14336x8192 | bfloat16 | 1792 | data-parallel |  | 1615.4 | 1621.6 | 1615.4 | 1621.6 | 1602.8 | 1606.7 | flydsl | 1548.9 | 1408.1 (copy) | 1566.4 | 1574.7 | hipBLASLt padded 1574.7 |
| 8192x16384x8192 | bfloat16 | 2048 | data-parallel |  | 1607.1 | 1623.5 | 1607.1 | 1623.5 | 1601.3 | 1606.9 | asm | 1549.0 | 1547.7 | 1541.9 | 1546.8 | AITER asm contiguous 1549.0 |
| 8192x57344x8192 | bfloat16 | 7168 | data-parallel |  | 1593.6 | 1586.4 | 1593.6 | 1586.4 | 1613.2 | 1607.6 | asm | 1521.3 | 1561.0 | 1498.2 | 1506.4 | AITER asm padded 1561.0 |
| 16384x512x8192 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1167.4 | 1195.9 | 991.8 | 955.6 | 994.0 | 964.4 | torch | 1075.9 | 1070.5 | 1094.8 | 1071.8 | hipBLASLt contiguous 1094.8 |
| 16384x1280x8192 | bfloat16 | 320 | one-tile owner, n = 3 |  | 1385.2 | 1389.7 | 1178.4 | 1172.8 | 1185.6 | 1175.8 | torch | 1261.4 | 1273.5 | 1268.1 | 1270.6 | AITER torch padded 1273.5 |
| 16384x2560x8192 | bfloat16 | 640 | one-tile owner, n = 2 |  | 1489.5 | 1492.9 | 1398.9 | 1400.6 | 1388.2 | 1394.7 | flydsl | 1351.4 | 1112.4 (copy) | 1325.3 | 1326.5 | AITER flydsl contiguous 1351.4 |
| 16384x4608x8192 | bfloat16 | 1152 | one-tile owner, n = 2 |  | 1537.7 | 1551.5 | 1492.8 | 1492.7 | 1473.0 | 1479.9 | flydsl | 1428.9 | 1264.8 (copy) | 1230.5 | 1209.4 | AITER flydsl contiguous 1428.9 |
| 16384x7168x8192 | bfloat16 | 1792 | data-parallel |  | 1591.2 | 1605.2 | 1591.2 | 1605.2 | 1579.6 | 1585.7 | torch | 1570.3 | 1589.6 | 1588.2 | 1591.1 | hipBLASLt padded 1591.1 |
| 16384x8192x8192 | bfloat16 | 2048 | data-parallel |  | 1594.6 | 1611.7 | 1594.6 | 1611.7 | 1588.0 | 1593.3 | torch | 1575.2 | 1597.0 | 1598.4 | 1590.6 | hipBLASLt contiguous 1598.4 |
| 16384x10240x8192 | bfloat16 | 2560 | data-parallel |  | 1607.0 | 1613.9 | 1607.0 | 1613.9 | 1597.4 | 1594.1 | torch | 1580.4 | 1600.2 | 1602.3 | 1590.2 | hipBLASLt contiguous 1602.3 |
| 16384x14336x8192 | bfloat16 | 3584 | data-parallel |  | 1611.4 | 1615.5 | 1611.4 | 1615.5 | 1597.6 | 1597.9 | asm | 1554.7 | 1565.8 | 1560.3 | 1571.7 | hipBLASLt padded 1571.7 |
| 16384x16384x8192 | bfloat16 | 4096 | data-parallel |  | 1609.4 | 1616.5 | 1609.4 | 1616.5 | 1599.5 | 1595.0 | asm | 1553.0 | 1552.2 | 1544.7 | 1543.7 | AITER asm contiguous 1553.0 |
| 16384x57344x8192 | bfloat16 | 14336 | data-parallel |  | 1534.7 | 1542.2 | 1534.7 | 1542.2 | 1605.7 | 1606.8 | asm | 1564.7 | 1561.8 | 1494.8 | 1432.7 | AITER asm contiguous 1564.7 |
| 32768x1280x8192 | bfloat16 | 640 | one-tile owner, n = 2 |  | 1449.7 | 1458.5 | 1371.2 | 1377.2 | 1362.2 | 1368.8 | flydsl | 1322.7 | 963.5 (copy) | 1343.7 | 1344.6 | hipBLASLt padded 1344.6 |
| 32768x2560x8192 | bfloat16 | 1280 | data-parallel |  | 1552.0 | 1556.4 | 1552.0 | 1556.4 | 1541.7 | 1543.2 | torch | 1543.6 | 1557.8 | 1560.9 | 1557.3 | hipBLASLt contiguous 1560.9 |
| 32768x7168x8192 | bfloat16 | 3584 | data-parallel |  | 1597.5 | 1595.9 | 1597.5 | 1595.9 | 1586.0 | 1579.1 | torch | 1578.5 | 1593.6 | 1598.2 | 1588.9 | hipBLASLt contiguous 1598.2 |
| 32768x8192x8192 | bfloat16 | 4096 | data-parallel |  | 1608.5 | 1611.1 | 1608.5 | 1611.1 | 1592.0 | 1582.8 | flydsl | 1557.5 | 1445.0 (copy) | 1598.3 | 1587.2 | hipBLASLt contiguous 1598.3 |
| 32768x10240x8192 | bfloat16 | 5120 | data-parallel |  | 1610.0 | 1607.0 | 1610.0 | 1607.0 | 1598.0 | 1587.2 | torch | 1590.6 | 1596.9 | 1601.8 | 1586.7 | hipBLASLt contiguous 1601.8 |
| 32768x14336x8192 | bfloat16 | 7168 | data-parallel |  | 1612.1 | 1603.6 | 1612.1 | 1603.6 | 1604.2 | 1588.3 | asm | 1564.2 | 1564.8 | 1541.2 | 1556.0 | AITER asm padded 1564.8 |
| 32768x16384x8192 | bfloat16 | 8192 | data-parallel |  | 1598.9 | 1596.6 | 1598.9 | 1596.6 | 1605.4 | 1585.6 | asm | 1561.5 | 1555.3 | 1532.0 | 1534.6 | AITER asm contiguous 1561.5 |
| 32768x57344x8192 | bfloat16 | 28672 | data-parallel |  | 1534.7 | 1491.0 | 1534.7 | 1491.0 | 1583.3 | 1574.4 | asm | 1546.8 | 1546.1 | 1446.3 | 1452.4 | AITER asm contiguous 1546.8 |
| 512x6144x12288 | bfloat16 | 48 | one-tile owner, n = 4 |  | 949.1 | 959.8 | 431.3 | 429.5 | 423.3 | 422.9 | torch | 934.0 | 934.3 | 935.4 | 935.2 | hipBLASLt contiguous 935.4 |
| 1024x6144x12288 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 1154.1 | 1178.9 | 844.2 | 842.7 | 836.5 | 839.1 | torch | 1198.6 | 1188.0 | 1200.4 | 1195.0 | hipBLASLt contiguous 1200.4 |
| 2048x6144x12288 | bfloat16 | 192 | data-parallel | yes | 1410.2 | 1408.4 | 1404.4 | 1395.8 | 1399.7 | 1399.6 | torch | 1393.8 | 1392.6 | 1395.5 | 1391.3 | hipBLASLt contiguous 1395.5 |
| 4096x6144x12288 | bfloat16 | 384 | one-tile owner, n = 2 |  | 1531.6 | 1537.0 | 1373.6 | 1375.2 | 1379.0 | 1371.5 | torch | 1383.0 | 1394.1 | 1396.0 | 1394.0 | hipBLASLt contiguous 1396.0 |
| 8192x6144x12288 | bfloat16 | 768 | data-parallel |  | 1591.4 | 1609.0 | 1591.4 | 1609.0 | 1587.0 | 1597.3 | torch | 1568.5 | 1610.4 | 1586.8 | 1604.6 | AITER torch padded 1610.4 |
| 16384x6144x12288 | bfloat16 | 1536 | data-parallel |  | 1579.4 | 1605.3 | 1579.4 | 1605.3 | 1571.0 | 1588.2 | torch | 1571.0 | 1610.5 | 1591.5 | 1607.2 | AITER torch padded 1610.5 |
| 32768x6144x12288 | bfloat16 | 3072 | data-parallel |  | 1594.0 | 1612.3 | 1594.0 | 1612.3 | 1575.9 | 1592.7 | asm | 1567.6 | 1573.3 | 1598.2 | 1609.9 | hipBLASLt padded 1609.9 |
| 512x16384x13312 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1271.9 | 1275.6 | 924.5 | 915.0 | 928.8 | 919.6 | flydsl | 707.0 | 467.2 (copy) | 1105.9 | 1098.0 | hipBLASLt contiguous 1105.9 |
| 1024x16384x13312 | bfloat16 | 256 | data-parallel | yes | 1533.2 | 1539.2 | 1532.1 | 1528.7 | 1538.0 | 1538.6 | asm | 1493.1 | 1502.5 | 1498.0 | 1508.0 | hipBLASLt padded 1508.0 |
| 2048x16384x13312 | bfloat16 | 512 | data-parallel |  | 1587.2 | 1587.0 | 1587.2 | 1587.0 | 1574.0 | 1583.8 | flydsl | 1008.7 | 841.8 (copy) | 1515.2 | 1527.4 | hipBLASLt padded 1527.4 |
| 4096x16384x13312 | bfloat16 | 1024 | data-parallel |  | 1621.6 | 1624.5 | 1621.6 | 1624.5 | 1610.0 | 1613.5 | flydsl | 1554.5 | 1336.9 (copy) | 1523.6 | 1542.0 | AITER flydsl contiguous 1554.5 |
| 8192x16384x13312 | bfloat16 | 2048 | data-parallel |  | 1605.6 | 1618.1 | 1605.6 | 1618.1 | 1587.3 | 1597.2 | flydsl | 1530.1 | 1401.7 (copy) | 1524.9 | 1544.0 | hipBLASLt padded 1544.0 |
| 16384x16384x13312 | bfloat16 | 4096 | data-parallel |  | 1602.9 | 1616.0 | 1602.9 | 1616.0 | 1590.4 | 1590.9 | asm | 1540.8 | 1544.0 | 1518.0 | 1533.6 | AITER asm padded 1544.0 |
| 32768x16384x13312 | bfloat16 | 8192 | data-parallel |  | 1602.1 | 1592.5 | 1602.1 | 1592.5 | 1580.7 | 1590.8 | flydsl | 1495.1 | 1459.5 (copy) | 1426.1 | 1459.1 | AITER flydsl contiguous 1495.1 |
| 512x2304x16384 | bfloat16 | 18 | one-tile reduce-scatter, n = 7 |  | 635.2 | 614.2 | 165.1 | 165.3 | 162.0 | 161.0 | torch | 737.4 | 640.1 | 736.0 | 640.2 | AITER torch contiguous 737.4 |
| 512x4608x16384 | bfloat16 | 36 | one-tile owner, n = 6 |  | 915.5 | 893.6 | 329.5 | 328.1 | 321.7 | 319.8 | flydsl | 580.8 | 345.9 (copy) | 955.5 | 869.4 | hipBLASLt contiguous 955.5 |
| 512x9216x16384 | bfloat16 | 72 | one-tile owner, n = 3-4 |  | 1071.1 | 1019.7 | 590.6 | 552.3 | 591.7 | 554.3 | flydsl | 646.7 | 412.2 (copy) | 760.2 | 725.6 | hipBLASLt contiguous 760.2 |
| 512x13312x16384 | bfloat16 | 104 | one-tile owner, n = 2-3 |  | 1141.4 | 1089.2 | 782.0 | 712.4 | 781.8 | 719.5 | flydsl | 545.1 | 385.2 (copy) | 1088.8 | 1104.3 | hipBLASLt padded 1104.3 |
| 512x26624x16384 | bfloat16 | 208 | data-parallel | yes | 1287.7 | 1225.5 | 1286.6 | 1216.3 | 1286.7 | 1242.6 | flydsl | 775.2 | 495.3 (copy) | 1204.3 | 1199.6 | hipBLASLt contiguous 1204.3 |
| 512x53248x16384 | bfloat16 | 416 | data-parallel |  | 1315.1 | 1260.8 | 1315.1 | 1260.8 | 1310.3 | 1256.8 | flydsl | 632.6 | 442.6 (copy) | 1211.8 | 1219.4 | hipBLASLt padded 1219.4 |
| 1024x2304x16384 | bfloat16 | 36 | one-tile owner, n = 6 |  | 896.0 | 793.5 | 327.7 | 327.8 | 321.8 | 320.3 | torch | 896.3 | 779.5 | 902.9 | 782.4 | hipBLASLt contiguous 902.9 |
| 1024x4608x16384 | bfloat16 | 72 | one-tile owner, n = 3-4 |  | 1190.2 | 1174.3 | 647.6 | 649.8 | 641.7 | 638.7 | flydsl | 838.8 | 516.5 (copy) | 1149.8 | 1148.6 | hipBLASLt contiguous 1149.8 |
| 1024x9216x16384 | bfloat16 | 144 | data-parallel | yes | 1130.4 | 1089.7 | 1129.1 | 1089.8 | 1132.2 | 1095.9 | flydsl | 696.9 | 547.4 (copy) | 1177.4 | 1162.6 | hipBLASLt contiguous 1177.4 |
| 1024x13312x16384 | bfloat16 | 208 | data-parallel | yes | 1370.3 | 1332.5 | 1367.1 | 1321.1 | 1369.8 | 1315.1 | asm | 1340.4 | 1294.2 | 1250.1 | 1250.3 | AITER asm contiguous 1340.4 |
| 1024x26624x16384 | bfloat16 | 416 | data-parallel |  | 1380.2 | 1359.4 | 1380.2 | 1359.4 | 1373.8 | 1331.9 | flydsl | 954.0 | 711.2 (copy) | 1254.8 | 1263.5 | hipBLASLt padded 1263.5 |
| 1024x53248x16384 | bfloat16 | 832 | one-tile owner, n = 4 |  | 1529.7 | 1530.4 | 1389.6 | 1380.1 | 1377.3 | 1342.2 | triton | 989.4 | 965.1 | 1258.8 | 1270.1 | hipBLASLt padded 1270.1 |
| 2048x2304x16384 | bfloat16 | 72 | one-tile owner, n = 3-4 |  | 1203.8 | 1179.0 | 647.2 | 651.7 | 639.7 | 637.4 | torch | 938.2 | 879.7 | 944.4 | 878.3 | hipBLASLt contiguous 944.4 |
| 2048x4608x16384 | bfloat16 | 144 | data-parallel | yes | 1215.1 | 1224.0 | 1216.6 | 1215.6 | 1221.1 | 1221.1 | flydsl | 736.9 | 614.7 (copy) | 1310.5 | 1298.2 | hipBLASLt contiguous 1310.5 |
| 2048x9216x16384 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 |  | 1428.5 | 1420.4 | 1100.2 | 1066.5 | 1093.9 | 1066.3 | flydsl | 916.1 | 750.8 (copy) | 1164.9 | 1134.2 | hipBLASLt contiguous 1164.9 |
| 2048x13312x16384 | bfloat16 | 416 | data-parallel |  | 1419.4 | 1393.7 | 1419.4 | 1393.7 | 1409.5 | 1406.2 | flydsl | 1372.6 | 1077.3 (copy) | 1245.0 | 1264.5 | AITER flydsl contiguous 1372.6 |
| 2048x26624x16384 | bfloat16 | 832 | one-tile owner, n = 4 |  | 1568.0 | 1557.5 | 1416.8 | 1406.4 | 1408.0 | 1400.8 | flydsl | 907.4 | 779.4 (copy) | 1245.6 | 1252.3 | hipBLASLt padded 1252.3 |
| 2048x53248x16384 | bfloat16 | 1664 | one-tile owner, n = 2 |  | 1601.1 | 1561.4 | 1547.5 | 1493.0 | 1524.1 | 1492.4 | flydsl | 799.9 | 677.7 (copy) | 1294.8 | 1303.3 | hipBLASLt padded 1303.3 |
| 4096x2304x16384 | bfloat16 | 144 | data-parallel | yes | 1216.2 | 1221.7 | 1215.8 | 1216.1 | 1209.8 | 1219.9 | torch | 1308.7 | 1302.4 | 1312.5 | 1297.2 | hipBLASLt contiguous 1312.5 |
| 4096x4608x16384 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 |  | 1456.1 | 1454.6 | 1114.8 | 1089.6 | 1104.0 | 1094.4 | torch | 1254.4 | 1185.8 | 1260.3 | 1191.0 | hipBLASLt contiguous 1260.3 |
| 4096x9216x16384 | bfloat16 | 576 | one-tile owner, n = 4 |  | 1563.4 | 1570.1 | 1376.3 | 1371.5 | 1370.0 | 1364.3 | torch | 1328.5 | 1347.5 | 1335.8 | 1346.8 | AITER torch padded 1347.5 |
| 4096x13312x16384 | bfloat16 | 832 | one-tile owner, n = 4 |  | 1579.3 | 1580.9 | 1431.2 | 1429.7 | 1418.3 | 1423.7 | flydsl | 1375.3 | 1195.6 (copy) | 1267.6 | 1279.1 | AITER flydsl contiguous 1375.3 |
| 4096x26624x16384 | bfloat16 | 1664 | one-tile owner, n = 2 |  | 1611.6 | 1600.8 | 1557.5 | 1533.0 | 1537.5 | 1524.0 | flydsl | 1494.5 | 1300.1 (copy) | 1320.6 | 1323.9 | AITER flydsl contiguous 1494.5 |
| 4096x53248x16384 | bfloat16 | 3328 | data-parallel |  | 1625.8 | 1632.2 | 1625.8 | 1632.2 | 1610.2 | 1611.5 | flydsl | 1542.6 | 1361.8 (copy) | 1513.6 | 1495.8 | AITER flydsl contiguous 1542.6 |
| 8192x2304x16384 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 |  | 1396.3 | 1329.7 | 1021.0 | 940.8 | 1006.0 | 936.3 | torch | 1199.1 | 1140.1 | 1205.5 | 1143.9 | hipBLASLt contiguous 1205.5 |
| 8192x4608x16384 | bfloat16 | 576 | one-tile owner, n = 4 |  | 1516.0 | 1460.4 | 1322.1 | 1224.3 | 1316.6 | 1238.7 | torch | 1357.2 | 1379.4 | 1368.3 | 1378.4 | AITER torch padded 1379.4 |
| 8192x9216x16384 | bfloat16 | 1152 | one-tile owner, n = 2 |  | 1555.4 | 1524.0 | 1474.8 | 1420.8 | 1465.6 | 1412.6 | flydsl | 1406.6 | 1265.4 (copy) | 1144.0 | 1143.9 | AITER flydsl contiguous 1406.6 |
| 8192x13312x16384 | bfloat16 | 1664 | one-tile owner, n = 2 |  | 1570.6 | 1531.7 | 1518.1 | 1457.2 | 1498.3 | 1453.2 | flydsl | 1453.8 | 1321.5 (copy) | 1308.2 | 1318.0 | AITER flydsl contiguous 1453.8 |
| 8192x26624x16384 | bfloat16 | 3328 | data-parallel |  | 1565.2 | 1514.4 | 1565.2 | 1514.4 | 1576.1 | 1509.1 | flydsl | 1504.0 | 1407.5 (copy) | 1516.3 | 1485.8 | hipBLASLt contiguous 1516.3 |
| 8192x53248x16384 | bfloat16 | 6656 | data-parallel |  | 1550.8 | 1555.8 | 1550.8 | 1555.8 | 1575.7 | 1554.2 | flydsl | 1401.6 | 1360.6 (copy) | 1450.5 | 1465.7 | hipBLASLt padded 1465.7 |
| 16384x2304x16384 | bfloat16 | 576 | one-tile owner, n = 4 |  | 1493.7 | 1450.2 | 1313.4 | 1258.3 | 1304.8 | 1264.5 | torch | 1349.2 | 1360.6 | 1361.8 | 1358.7 | hipBLASLt contiguous 1361.8 |
| 16384x4608x16384 | bfloat16 | 1152 | one-tile owner, n = 2 |  | 1547.2 | 1494.8 | 1482.7 | 1405.8 | 1465.8 | 1409.4 | flydsl | 1423.3 | 1253.4 (copy) | 1177.2 | 1162.7 | AITER flydsl contiguous 1423.3 |
| 16384x9216x16384 | bfloat16 | 2304 | data-parallel |  | 1576.2 | 1510.5 | 1576.2 | 1510.5 | 1569.7 | 1496.0 | flydsl | 1520.6 | 1401.8 (copy) | 1525.6 | 1499.3 | hipBLASLt contiguous 1525.6 |
| 16384x13312x16384 | bfloat16 | 3328 | data-parallel |  | 1586.5 | 1511.2 | 1586.5 | 1511.2 | 1576.9 | 1504.0 | asm | 1509.4 | 1537.9 | 1514.7 | 1479.1 | AITER asm padded 1537.9 |
| 16384x26624x16384 | bfloat16 | 6656 | data-parallel |  | 1572.4 | 1534.7 | 1572.4 | 1534.7 | 1578.3 | 1519.8 | flydsl | 1537.7 | 1460.3 (copy) | 1471.8 | 1433.8 | AITER flydsl contiguous 1537.7 |
| 16384x53248x16384 | bfloat16 | 13312 | data-parallel |  | 1551.5 | 1540.9 | 1551.5 | 1540.9 | 1577.3 | 1549.0 | asm | 1528.2 | 1539.0 | 1433.7 | 1473.6 | AITER asm padded 1539.0 |
| 32768x2304x16384 | bfloat16 | 1152 | one-tile owner, n = 2 |  | 1520.7 | 1521.8 | 1463.6 | 1433.5 | 1448.1 | 1430.8 | flydsl | 1411.1 | 1150.5 (copy) | 1188.6 | 1170.8 | AITER flydsl contiguous 1411.1 |
| 32768x4608x16384 | bfloat16 | 2304 | data-parallel |  | 1567.0 | 1538.9 | 1567.0 | 1538.9 | 1565.3 | 1529.8 | torch | 1538.9 | 1531.8 | 1552.8 | 1534.2 | hipBLASLt contiguous 1552.8 |
| 32768x9216x16384 | bfloat16 | 4608 | data-parallel |  | 1581.0 | 1551.4 | 1581.0 | 1551.4 | 1575.5 | 1543.9 | asm | 1501.4 | 1503.4 | 1499.8 | 1477.9 | AITER asm padded 1503.4 |
| 32768x13312x16384 | bfloat16 | 6656 | data-parallel |  | 1570.1 | 1549.3 | 1570.1 | 1549.3 | 1577.7 | 1547.0 | asm | 1526.7 | 1541.1 | 1426.7 | 1420.1 | AITER asm padded 1541.1 |
| 32768x26624x16384 | bfloat16 | 13312 | data-parallel |  | 1563.9 | 1561.8 | 1563.9 | 1561.8 | 1574.9 | 1553.4 | flydsl | 1507.2 | 1484.2 (copy) | 1440.8 | 1467.6 | AITER flydsl contiguous 1507.2 |
| 32768x53248x16384 | bfloat16 | 26624 | data-parallel |  | 1572.4 | 1568.6 | 1572.4 | 1568.6 | 1572.0 | 1562.8 | flydsl | 1523.1 | 1494.1 (copy) | 1461.3 | 1511.8 | AITER flydsl contiguous 1523.1 |
| 512x6144x18432 | bfloat16 | 48 | one-tile owner, n = 5-6 |  | 1014.7 | 1021.9 | 439.2 | 434.7 | 428.8 | 428.3 | flydsl | 916.5 | 488.7 (copy) | 951.7 | 961.6 | hipBLASLt padded 961.6 |
| 1024x6144x18432 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 1169.3 | 1146.1 | 800.3 | 817.6 | 800.5 | 807.8 | torch | 1196.3 | 1210.9 | 1206.0 | 1215.5 | hipBLASLt padded 1215.5 |
| 2048x6144x18432 | bfloat16 | 192 | data-parallel | yes | 1346.2 | 1349.6 | 1349.6 | 1342.7 | 1347.1 | 1344.9 | torch | 1348.3 | 1366.4 | 1354.8 | 1369.8 | hipBLASLt padded 1369.8 |
| 4096x6144x18432 | bfloat16 | 384 | one-tile owner, n = 2 |  | 1561.7 | 1566.0 | 1385.8 | 1384.9 | 1383.0 | 1382.4 | torch | 1339.1 | 1365.1 | 1347.5 | 1366.6 | hipBLASLt padded 1366.6 |
| 8192x6144x18432 | bfloat16 | 768 | data-parallel |  | 1599.5 | 1614.0 | 1599.5 | 1614.0 | 1591.1 | 1601.2 | torch | 1520.5 | 1565.7 | 1530.3 | 1571.9 | hipBLASLt padded 1571.9 |
| 16384x6144x18432 | bfloat16 | 1536 | data-parallel |  | 1606.3 | 1616.5 | 1606.3 | 1616.5 | 1593.8 | 1602.2 | asm | 1548.6 | 1563.9 | 1520.6 | 1576.7 | hipBLASLt padded 1576.7 |
| 32768x6144x18432 | bfloat16 | 3072 | data-parallel |  | 1607.6 | 1593.1 | 1607.6 | 1593.1 | 1596.7 | 1597.2 | asm | 1533.0 | 1557.1 | 1513.4 | 1518.2 | AITER asm padded 1557.1 |
| 512x5120x25600 | bfloat16 | 40 | one-tile owner, n = 6-7 |  | 992.4 | 1014.3 | 323.9 | 328.1 | 323.9 | 332.4 | flydsl | 437.3 | 311.2 (copy) | 888.5 | 905.0 | hipBLASLt padded 905.0 |
| 1024x5120x25600 | bfloat16 | 80 | one-tile owner, n = 3-4 |  | 1294.4 | 1310.1 | 647.2 | 651.6 | 653.9 | 661.8 | torch | 1165.4 | 1179.0 | 1168.1 | 1181.5 | hipBLASLt padded 1181.5 |
| 2048x5120x25600 | bfloat16 | 160 | data-parallel | yes | 1176.5 | 1226.9 | 1181.3 | 1232.7 | 1184.5 | 1243.3 | torch | 1290.7 | 1310.6 | 1296.3 | 1312.6 | hipBLASLt padded 1312.6 |
| 4096x5120x25600 | bfloat16 | 320 | one-tile owner, n = 4 |  | 1547.3 | 1554.3 | 1160.1 | 1204.8 | 1163.4 | 1198.8 | torch | 1275.5 | 1298.4 | 1281.9 | 1299.3 | hipBLASLt padded 1299.3 |
| 8192x5120x25600 | bfloat16 | 640 | one-tile owner, n = 2 |  | 1578.0 | 1587.9 | 1441.9 | 1461.9 | 1425.2 | 1442.9 | asm | 1405.0 | 1416.3 | 1354.9 | 1373.2 | AITER asm padded 1416.3 |
| 16384x5120x25600 | bfloat16 | 1280 | data-parallel |  | 1609.8 | 1604.4 | 1609.8 | 1604.4 | 1596.3 | 1598.8 | torch | 1515.2 | 1584.1 | 1516.7 | 1585.7 | hipBLASLt padded 1585.7 |
| 32768x5120x25600 | bfloat16 | 2560 | data-parallel |  | 1570.0 | 1584.0 | 1570.0 | 1584.0 | 1598.2 | 1594.2 | torch | 1431.0 | 1431.3 | 1432.8 | 1429.9 | hipBLASLt contiguous 1432.8 |
| 512x16384x26624 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1373.2 | 1381.9 | 935.8 | 943.1 | 930.5 | 944.0 | flydsl | 651.3 | 440.4 (copy) | 1205.2 | 1200.6 | hipBLASLt contiguous 1205.2 |
| 1024x16384x26624 | bfloat16 | 256 | data-parallel | yes | 1567.8 | 1573.3 | 1563.0 | 1568.2 | 1567.6 | 1570.2 | flydsl | 784.6 | 588.4 (copy) | 1528.6 | 1543.9 | hipBLASLt padded 1543.9 |
| 2048x16384x26624 | bfloat16 | 512 | data-parallel |  | 1611.1 | 1614.2 | 1611.1 | 1614.2 | 1597.3 | 1608.2 | flydsl | 821.9 | 713.2 (copy) | 1533.9 | 1548.3 | hipBLASLt padded 1548.3 |
| 4096x16384x26624 | bfloat16 | 1024 | data-parallel |  | 1637.3 | 1629.2 | 1637.3 | 1629.2 | 1615.3 | 1623.2 | flydsl | 1566.1 | 1351.6 (copy) | 1537.2 | 1559.4 | AITER flydsl contiguous 1566.1 |
| 8192x16384x26624 | bfloat16 | 2048 | data-parallel |  | 1624.1 | 1614.3 | 1624.1 | 1614.3 | 1607.0 | 1606.6 | asm | 244.3 | 240.9 | 1526.8 | 1551.3 | hipBLASLt padded 1551.3 |
| 16384x16384x26624 | bfloat16 | 4096 | data-parallel |  | 1612.9 | 1574.5 | 1612.9 | 1574.5 | 1611.0 | 1607.4 | asm | 1569.0 | 1572.5 | 1420.4 | 1409.9 | AITER asm padded 1572.5 |
| 32768x16384x26624 | bfloat16 | 8192 | data-parallel |  | 1616.2 | 1522.5 | 1616.2 | 1522.5 | 1608.4 | 1591.5 | asm | 1565.1 | 1566.3 | 1449.1 | 1486.4 | AITER asm padded 1566.3 |
| 512x8192x28672 | bfloat16 | 64 | one-tile owner, n = 4 |  | 1229.5 | 1232.7 | 474.1 | 485.2 | 477.0 | 481.6 | flydsl | 794.5 | 489.3 (copy) | 921.5 | 921.6 | hipBLASLt padded 921.6 |
| 1024x8192x28672 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1434.5 | 1456.0 | 934.3 | 974.2 | 945.6 | 971.8 | triton | 697.1 | 697.2 | 1350.0 | 1360.2 | hipBLASLt padded 1360.2 |
| 2048x8192x28672 | bfloat16 | 256 | data-parallel | yes | 1608.5 | 1607.9 | 1604.6 | 1600.1 | 1589.0 | 1605.7 | flydsl | 1032.6 | 845.4 (copy) | 1566.7 | 1582.0 | hipBLASLt padded 1582.0 |
| 4096x8192x28672 | bfloat16 | 512 | data-parallel |  | 1619.8 | 1628.1 | 1619.8 | 1628.1 | 1613.1 | 1623.2 | flydsl | 1562.3 | 1300.3 (copy) | 1562.6 | 1585.3 | hipBLASLt padded 1585.3 |
| 8192x8192x28672 | bfloat16 | 1024 | data-parallel |  | 1616.1 | 1621.6 | 1616.1 | 1621.6 | 1603.4 | 1611.1 | asm | 1569.2 | 1579.4 | 1554.8 | 1579.8 | hipBLASLt padded 1579.8 |
| 16384x8192x28672 | bfloat16 | 2048 | data-parallel |  | 1540.1 | 1560.5 | 1540.1 | 1560.5 | 1606.6 | 1586.2 | asm | 1557.2 | 1530.0 | 1475.7 | 1484.0 | AITER asm contiguous 1557.2 |
| 32768x8192x28672 | bfloat16 | 4096 | data-parallel |  | 1627.3 | 1583.5 | 1627.3 | 1583.5 | 1612.6 | 1608.8 | asm | 1575.3 | 1583.3 | 1511.9 | 1422.7 | AITER asm padded 1583.3 |
| 512x7168x35840 | bfloat16 | 56 | one-tile owner, n = 4-5 |  | 1172.5 | 1141.2 | 439.6 | 410.1 | 440.3 | 407.2 | flydsl | 820.3 | 502.7 (copy) | 967.1 | 965.2 | hipBLASLt contiguous 967.1 |
| 1024x7168x35840 | bfloat16 | 112 | one-tile owner, n = 2-3 |  | 1356.4 | 1356.4 | 868.8 | 821.4 | 867.8 | 816.7 | flydsl | 574.3 | 465.1 (copy) | 1293.2 | 1310.0 | hipBLASLt padded 1310.0 |
| 2048x7168x35840 | bfloat16 | 224 | spread |  | 1500.5 | 1506.0 | 1502.3 | 1508.5 | 1495.3 | 1496.8 | asm | 1458.3 | 1466.2 | 1412.4 | 1418.9 | AITER asm padded 1466.2 |
| 4096x7168x35840 | bfloat16 | 448 | spread |  | 1581.4 | 1589.8 | 1518.7 | 1523.0 | 1504.7 | 1508.7 | flydsl | 1470.9 | 1226.5 (copy) | 1410.5 | 1413.8 | AITER flydsl contiguous 1470.9 |
| 8192x7168x35840 | bfloat16 | 896 | one-tile owner, n = 2 |  | 1605.8 | 1612.9 | 1490.8 | 1501.6 | 1476.8 | 1484.9 | asm | 1427.6 | 1433.7 | 1408.0 | 1420.5 | AITER asm padded 1433.7 |
| 16384x7168x35840 | bfloat16 | 1792 | data-parallel |  | 1606.5 | 1599.3 | 1606.5 | 1599.3 | 1597.9 | 1605.0 | torch | 1480.4 | 1510.5 | 1514.4 | 1536.3 | hipBLASLt padded 1536.3 |
| 32768x7168x35840 | bfloat16 | 3584 | data-parallel |  | 1607.2 | 1536.9 | 1607.2 | 1536.9 | 1591.9 | 1592.6 | flydsl | 1525.5 | 1375.9 (copy) | 1492.9 | 1486.7 | AITER flydsl contiguous 1525.5 |
