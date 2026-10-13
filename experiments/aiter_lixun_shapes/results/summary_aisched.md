# streamk_wip (v20, capped, v9 for one-wave dp) vs AITER, v9 and hipBLASLt, contiguous and 512-byte-padded strides (warm)

- Stack: torch 2.10.0+rocm7.2.4.git3d3aa833, hipBLASLt 100202 (dabb6df2b9), Triton 3.8.0 @ /home/jbowley/repos/triton_gfx950-tutorial-v2.2/python/triton; GPU: PCI bus 117, gfx950:sramecc+:xnack-; AITER e2c7aa0aa [Triton/Gluon] gfx950 a16w16: skip the Gluon kernel on Triton without cd_regclass on a1455f1c4 2026-10-02 [CK] [FlyDSL] add mxfp8 a8w8 blockscale for gfx950 (#4951).
- streamk_wip: snapshot sha256 03957ee906733825 (newest WORKLOG entry: "13. v9 writes C in C's dtype; fp32 C goes to v9 too (2026-10-06)"); the v9 it delegates to: live sha256 7b4e246a4877a02c.
- Layouts: contiguous, and padded: every row stride (x, w, C) that is a multiple of 512 bytes gets 256 more (x and w: K + 128 when K % 256 == 0; C: N + 128 for bf16, N + 64 for fp32). v9, streamk_wip and hipBLASLt (`torch.mm(x, w.t(), out=y)`) get padded x, w and C; AITER's pick allocates its own C, so it gets padded x and w only.
- Timing: 3 rounds, each with fresh inputs (new buffers) when `realloc` is on (here: True); median of the rounds. Spread = (slowest - fastest round) / median.
- Best library: the fastest valid time of AITER's pick and hipBLASLt; for the padded comparison over both layouts. A padded call that launches a copy kernel (FlyDSL makes its inputs contiguous) is not valid.
- "Faster" and "slower" mean a gap of more than 1%.

Padded calls that copied their inputs (left out of the best library): pick (flydsl) on 252 shapes.

The best library's time comes from the contiguous layout on 326 shapes, padded layout on 304 shapes.

## v9 and streamk_wip padded, libraries contiguous (as AITER calls them)

| Shapes | Count | v9: faster | v9: within 1% | v9: slower | v9: median | data-parallel only: faster | data-parallel only: within 1% | data-parallel only: slower | data-parallel only: median | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All shapes | 630 | 236 | 52 | 342 | -2.7% | 247 | 47 | 336 | -3.6% | 314 | 50 | 266 | +1.0% |
| Under 256 tiles (less than one wave) | 321 | 22 | 10 | 289 | -37.8% | 18 | 10 | 293 | -37.4% | 54 | 16 | 251 | -17.9% |
| 256 to 1023 tiles | 152 | 103 | 9 | 40 | +2.7% | 102 | 9 | 41 | +2.5% | 131 | 8 | 13 | +4.8% |
| 1024 or more tiles | 157 | 111 | 33 | 13 | +2.1% | 127 | 28 | 2 | +2.9% | 129 | 26 | 2 | +3.1% |
| Best library uses 256x256 | 260 | 197 | 40 | 23 | +2.6% | 211 | 36 | 13 | +2.8% | 221 | 33 | 6 | +3.3% |
| Best library uses another tile | 370 | 39 | 12 | 319 | -31.0% | 36 | 11 | 323 | -32.0% | 93 | 17 | 260 | -12.8% |

## Everything contiguous (comparable with Oct 5)

| Shapes | Count | v9: faster | v9: within 1% | v9: slower | v9: median | data-parallel only: faster | data-parallel only: within 1% | data-parallel only: slower | data-parallel only: median | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All shapes | 630 | 238 | 49 | 343 | -2.6% | 247 | 43 | 340 | -3.3% | 315 | 50 | 265 | +1.0% |
| Under 256 tiles (less than one wave) | 321 | 25 | 8 | 288 | -37.5% | 20 | 8 | 293 | -38.2% | 60 | 12 | 249 | -16.7% |
| 256 to 1023 tiles | 152 | 101 | 11 | 40 | +2.6% | 100 | 9 | 43 | +2.3% | 128 | 12 | 12 | +4.1% |
| 1024 or more tiles | 157 | 112 | 30 | 15 | +2.3% | 127 | 26 | 4 | +2.6% | 127 | 26 | 4 | +2.8% |
| Best library uses 256x256 | 260 | 198 | 38 | 24 | +2.6% | 210 | 34 | 16 | +2.6% | 219 | 34 | 7 | +3.0% |
| Best library uses another tile | 370 | 40 | 11 | 319 | -30.3% | 37 | 9 | 324 | -31.9% | 96 | 16 | 258 | -11.8% |

## v9 and streamk_wip padded, against the best library over both layouts (strictest)

| Shapes | Count | v9: faster | v9: within 1% | v9: slower | v9: median | data-parallel only: faster | data-parallel only: within 1% | data-parallel only: slower | data-parallel only: median | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All shapes | 630 | 222 | 55 | 353 | -2.8% | 238 | 50 | 342 | -3.8% | 306 | 52 | 272 | +0.5% |
| Under 256 tiles (less than one wave) | 321 | 21 | 9 | 291 | -37.8% | 17 | 8 | 296 | -37.4% | 53 | 14 | 254 | -18.3% |
| 256 to 1023 tiles | 152 | 102 | 8 | 42 | +2.2% | 100 | 10 | 42 | +2.0% | 130 | 8 | 14 | +4.2% |
| 1024 or more tiles | 157 | 99 | 38 | 20 | +1.5% | 121 | 32 | 4 | +2.3% | 123 | 30 | 4 | +2.7% |
| Best library uses 256x256 | 259 | 182 | 46 | 31 | +2.0% | 201 | 42 | 16 | +2.3% | 213 | 38 | 8 | +2.9% |
| Best library uses another tile | 371 | 40 | 9 | 322 | -31.6% | 37 | 8 | 326 | -32.6% | 93 | 14 | 264 | -13.5% |

"Data-parallel only" is `STREAMK_POLICY=dp`, the persistent loop without stream-K and without the v9 hand-off; where capped already runs data-parallel on the persistent loop it is streamk_wip's time.

## v9 delegation (124 shapes: capped picks data-parallel, at most 256 tiles, bf16 output)

streamk_wip (now v9) against the persistent loop it replaced and against AITER's v9, same sweep. Positive means streamk_wip is faster.

| Layout | Shapes | Against | faster | within 1% | slower | median | p10 | p90 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| contiguous | all (124) | persistent loop (dp) | 71 | 52 | 1 | +1.1% | +0.2% | +3.6% |
| contiguous | all (124) | AITER's v9 | 0 | 121 | 3 | -0.2% | -0.5% | +0.2% |
| contiguous | k-steps <= 32 (56) | persistent loop (dp) | 51 | 5 | 0 | +2.7% | +1.3% | +4.1% |
| contiguous | k-steps <= 32 (56) | AITER's v9 | 0 | 56 | 0 | -0.1% | -0.5% | +0.2% |
| contiguous | k-steps > 32 (68) | persistent loop (dp) | 20 | 47 | 1 | +0.7% | +0.1% | +1.3% |
| contiguous | k-steps > 32 (68) | AITER's v9 | 0 | 65 | 3 | -0.2% | -0.5% | +0.1% |
| padded | all (124) | persistent loop (dp) | 76 | 45 | 3 | +1.3% | +0.3% | +3.8% |
| padded | all (124) | AITER's v9 | 3 | 119 | 2 | -0.0% | -0.3% | +0.2% |
| padded | k-steps <= 32 (56) | persistent loop (dp) | 53 | 2 | 1 | +2.6% | +1.3% | +5.3% |
| padded | k-steps <= 32 (56) | AITER's v9 | 2 | 54 | 0 | -0.0% | -0.3% | +0.2% |
| padded | k-steps > 32 (68) | persistent loop (dp) | 23 | 43 | 2 | +0.8% | +0.2% | +1.4% |
| padded | k-steps > 32 (68) | AITER's v9 | 1 | 65 | 2 | -0.1% | -0.3% | +0.2% |

### Against `../../../aiter/streamk_study/2026-10-06/results/details_v9fb_pad.csv` (contiguous, this run / that run - 1)

| Shapes | Count | streamk_wip | v9 (AITER) | AITER's pick | hipBLASLt |
|---|---:|---:|---:|---:|---:|
| Delegated to v9 now | 124 | +0.9% | +0.0% | +0.0% | -0.0% |
| Not delegated | 506 | +0.3% | -0.1% | -0.1% | -0.1% |

### The 14 same-tile shapes where streamk_wip was behind on Oct 5

| Shape | Tiles | k-steps | Delegated now | Oct 5: vs best | Now contiguous: vs best | Now padded: vs best | Now padded: vs v9 padded |
|---|---:|---:|---|---:|---:|---:|---:|
| 4096x4096x1024 | 256 | 16 | yes | -3.0% | +0.9% | +1.9% | +0.1% |
| 4096x8192x1024 | 512 | 16 | no | -2.6% | +0.3% | +2.0% | +0.8% |
| 16384x57344x8192 | 14336 | 128 | no | -1.9% | +2.0% | +2.3% | -0.3% |
| 16384x2048x7168 | 512 | 112 | no | -1.8% | -2.2% | -1.1% | +0.1% |
| 2048x16384x6656 | 512 | 104 | no | -1.7% | -2.2% | -1.6% | +0.1% |
| 4096x4096x1536 | 256 | 24 | yes | -1.6% | +2.5% | +2.7% | +0.3% |
| 2048x7168x1024 | 224 | 16 | yes | -1.5% | +3.5% | +3.9% | +0.3% |
| 32768x512x7168 | 256 | 112 | yes | -1.5% | +0.1% | -2.1% | -0.3% |
| 4096x7168x1024 | 448 | 16 | no | -1.4% | +2.1% | +2.7% | +0.6% |
| 2048x7168x1792 | 224 | 28 | yes | -1.2% | +4.4% | +3.9% | -0.2% |
| 16384x1024x7168 | 256 | 112 | yes | -1.2% | -1.0% | -0.8% | -0.3% |
| 16384x16384x2048 | 4096 | 32 | no | -1.1% | -1.3% | -1.1% | +2.8% |
| 32768x16384x2048 | 8192 | 32 | no | -1.1% | -1.3% | -0.8% | +2.2% |
| 16384x8192x28672 | 2048 | 448 | no | -1.1% | +4.1% | +2.1% | -0.2% |

## Padding: does it remove the noise?

Spread between the 3 rounds, per kernel and layout. With `realloc` every round has new buffers, so a timing that depends on where the buffers landed shows up here. Control: the 16 shapes with K % 256 == 128, whose x and w strides are not padded (only C is).

| Kernel | Shapes | contiguous: median spread | contiguous: > 2% | contiguous: > 5% | padded: median spread | padded: > 2% | padded: > 5% |
|---|---|---:|---:|---:|---:|---:|---:|
| v9 | All, strides padded (614) | 0.27% | 19 | 4 | 0.28% | 21 | 1 |
| v9 | Under 256 (less than one wave), strides padded (314) | 0.27% | 15 | 1 | 0.26% | 11 | 0 |
| v9 | 256 to 1023, strides padded (148) | 0.29% | 4 | 3 | 0.31% | 6 | 1 |
| v9 | 1024 or more, strides padded (152) | 0.24% | 0 | 0 | 0.27% | 4 | 0 |
| v9 | Control (K % 256 == 128) (16) | 0.39% | 0 | 0 | 0.31% | 0 | 0 |
| streamk_wip | All, strides padded (614) | 0.40% | 24 | 3 | 0.34% | 32 | 10 |
| streamk_wip | Under 256 (less than one wave), strides padded (314) | 0.47% | 15 | 1 | 0.44% | 23 | 6 |
| streamk_wip | 256 to 1023, strides padded (148) | 0.35% | 1 | 1 | 0.28% | 4 | 3 |
| streamk_wip | 1024 or more, strides padded (152) | 0.33% | 8 | 1 | 0.24% | 5 | 1 |
| streamk_wip | Control (K % 256 == 128) (16) | 0.34% | 0 | 0 | 0.39% | 0 | 0 |
| AITER's pick | All, strides padded (362) | 0.46% | 13 | 0 | 0.35% | 20 | 3 |
| AITER's pick | Under 256 (less than one wave), strides padded (168) | 0.59% | 8 | 0 | 0.45% | 14 | 2 |
| AITER's pick | 256 to 1023, strides padded (89) | 0.40% | 1 | 0 | 0.32% | 2 | 0 |
| AITER's pick | 1024 or more, strides padded (105) | 0.42% | 4 | 0 | 0.32% | 4 | 1 |
| AITER's pick | Control (K % 256 == 128) (16) | 0.46% | 1 | 0 | 0.27% | 0 | 0 |
| hipBLASLt | All, strides padded (614) | 0.39% | 52 | 18 | 0.38% | 68 | 15 |
| hipBLASLt | Under 256 (less than one wave), strides padded (314) | 0.45% | 38 | 12 | 0.43% | 47 | 13 |
| hipBLASLt | 256 to 1023, strides padded (148) | 0.30% | 0 | 0 | 0.30% | 3 | 0 |
| hipBLASLt | 1024 or more, strides padded (152) | 0.43% | 14 | 6 | 0.36% | 18 | 2 |
| hipBLASLt | Control (K % 256 == 128) (16) | 0.32% | 0 | 0 | 0.30% | 0 | 0 |

AITER's pick leaves out (on both layouts) the shapes where its padded call copied its inputs.

### Padded against contiguous, median time over the rounds

Padded TFLOPS / contiguous TFLOPS - 1 (positive: padding is faster).

| Kernel | Shapes | Count | faster | within 1% | slower | median | p10 | p90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| v9 | All, strides padded | 614 | 48 | 515 | 51 | -0.1% | -0.9% | +0.8% |
| v9 | Under 256 (less than one wave) | 314 | 19 | 267 | 28 | -0.3% | -0.9% | +0.6% |
| v9 | 256 to 1023 | 148 | 14 | 127 | 7 | +0.0% | -0.6% | +0.9% |
| v9 | 1024 or more | 152 | 15 | 121 | 16 | +0.1% | -1.0% | +1.0% |
| v9 | Control (K % 256 == 128) | 16 | 0 | 16 | 0 | -0.1% | -0.2% | +0.2% |
| streamk_wip | All, strides padded | 614 | 81 | 433 | 100 | +0.1% | -1.7% | +1.2% |
| streamk_wip | Under 256 (less than one wave) | 314 | 33 | 198 | 83 | -0.3% | -2.3% | +1.0% |
| streamk_wip | 256 to 1023 | 148 | 18 | 123 | 7 | +0.3% | -0.3% | +1.1% |
| streamk_wip | 1024 or more | 152 | 30 | 112 | 10 | +0.5% | -0.4% | +1.3% |
| streamk_wip | Control (K % 256 == 128) | 16 | 2 | 13 | 1 | +0.2% | -0.2% | +1.2% |
| AITER's pick | All, strides padded | 362 | 88 | 251 | 23 | +0.4% | -0.6% | +1.8% |
| AITER's pick | Under 256 (less than one wave) | 168 | 33 | 119 | 16 | +0.3% | -0.9% | +1.4% |
| AITER's pick | 256 to 1023 | 89 | 21 | 63 | 5 | +0.5% | -0.5% | +1.7% |
| AITER's pick | 1024 or more | 105 | 34 | 69 | 2 | +0.5% | -0.3% | +2.0% |
| AITER's pick | Control (K % 256 == 128) | 16 | 0 | 16 | 0 | +0.4% | +0.2% | +0.8% |
| hipBLASLt | All, strides padded | 614 | 108 | 415 | 91 | +0.0% | -1.3% | +1.4% |
| hipBLASLt | Under 256 (less than one wave) | 314 | 53 | 191 | 70 | -0.1% | -1.6% | +1.4% |
| hipBLASLt | 256 to 1023 | 148 | 22 | 120 | 6 | +0.1% | -0.5% | +1.3% |
| hipBLASLt | 1024 or more | 152 | 33 | 104 | 15 | +0.2% | -0.9% | +1.7% |
| hipBLASLt | Control (K % 256 == 128) | 16 | 0 | 15 | 1 | -0.1% | -0.6% | +0.4% |

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
| 1024x3072x512 | bfloat16 | 48 | data-parallel | yes | 226.5 | 225.9 | 214.4 | 213.1 | 226.2 | 226.0 | flydsl | 466.1 | 211.0 (copy) | 403.7 | 400.7 | AITER flydsl contiguous 466.1 |
| 1024x4096x512 | bfloat16 | 64 | data-parallel | yes | 286.7 | 284.9 | 274.1 | 269.8 | 286.0 | 285.3 | flydsl | 528.6 | 258.1 (copy) | 473.4 | 472.6 | AITER flydsl contiguous 528.6 |
| 1024x7168x512 | bfloat16 | 112 | data-parallel | yes | 460.3 | 453.7 | 440.6 | 431.0 | 460.9 | 453.8 | flydsl | 616.0 | 350.9 (copy) | 582.1 | 578.6 | AITER flydsl contiguous 616.0 |
| 2048x2048x512 | bfloat16 | 64 | data-parallel | yes | 286.2 | 284.9 | 273.0 | 269.2 | 285.7 | 285.0 | flydsl | 531.5 | 257.1 (copy) | 473.5 | 471.3 | AITER flydsl contiguous 531.5 |
| 2048x4096x512 | bfloat16 | 128 | data-parallel | yes | 497.0 | 498.1 | 481.5 | 480.1 | 497.2 | 498.1 | torch | 621.9 | 619.9 | 621.9 | 629.2 | hipBLASLt padded 629.2 |
| 4096x2048x512 | bfloat16 | 128 | data-parallel | yes | 495.5 | 500.4 | 479.2 | 482.9 | 495.8 | 500.0 | torch | 669.8 | 671.8 | 671.4 | 661.2 | AITER torch padded 671.8 |
| 4096x4096x512 | bfloat16 | 256 | data-parallel | yes | 864.2 | 871.1 | 835.4 | 825.9 | 863.3 | 869.7 | torch | 811.7 | 806.7 | 810.0 | 807.7 | AITER torch contiguous 811.7 |
| 4096x7168x512 | bfloat16 | 448 | data-parallel |  | 884.1 | 894.2 | 884.1 | 894.2 | 881.2 | 872.8 | flydsl | 857.5 | 683.3 (copy) | 772.6 | 762.7 | AITER flydsl contiguous 857.5 |
| 8192x2048x512 | bfloat16 | 256 | data-parallel | yes | 861.7 | 864.4 | 831.5 | 820.5 | 866.9 | 862.6 | torch | 805.6 | 800.0 | 805.8 | 805.4 | hipBLASLt contiguous 805.8 |
| 8192x4096x512 | bfloat16 | 512 | data-parallel |  | 940.4 | 952.9 | 940.4 | 952.9 | 929.8 | 931.4 | flydsl | 858.2 | 702.8 (copy) | 862.3 | 860.0 | hipBLASLt contiguous 862.3 |
| 16384x2048x512 | bfloat16 | 512 | data-parallel |  | 943.7 | 946.9 | 943.7 | 946.9 | 933.9 | 930.9 | torch | 861.7 | 856.9 | 859.8 | 858.8 | AITER torch contiguous 861.7 |
| 16384x4096x512 | bfloat16 | 1024 | data-parallel |  | 1018.9 | 1025.7 | 1018.9 | 1025.7 | 980.5 | 981.0 | torch | 941.7 | 938.4 | 948.5 | 944.1 | hipBLASLt contiguous 948.5 |
| 16384x7168x512 | bfloat16 | 1792 | data-parallel |  | 1052.7 | 1056.4 | 1052.7 | 1056.4 | 1015.9 | 1005.4 | torch | 985.7 | 981.1 | 992.1 | 986.4 | hipBLASLt contiguous 992.1 |
| 32768x2048x512 | bfloat16 | 1024 | data-parallel |  | 1017.6 | 1022.7 | 1017.6 | 1022.7 | 980.7 | 976.8 | torch | 940.8 | 930.4 | 949.0 | 937.9 | hipBLASLt contiguous 949.0 |
| 1024x5120x640 | bfloat16 | 80 | data-parallel | yes | 379.3 | 378.6 | 361.5 | 360.3 | 378.8 | 378.9 | torch | 529.5 | 533.5 | 531.2 | 534.6 | hipBLASLt padded 534.6 |
| 2048x5120x640 | bfloat16 | 160 | data-parallel | yes | 680.7 | 677.9 | 654.4 | 650.1 | 679.4 | 678.8 | flydsl | 672.5 | 677.1 | 563.9 | 557.2 | AITER flydsl padded 677.1 |
| 4096x5120x640 | bfloat16 | 320 | data-parallel |  | 776.7 | 800.4 | 776.7 | 800.4 | 754.6 | 754.8 | flydsl | 815.3 | 816.3 | 779.9 | 779.7 | AITER flydsl padded 816.3 |
| 8192x5120x640 | bfloat16 | 640 | data-parallel |  | 970.4 | 978.1 | 970.4 | 978.1 | 940.1 | 938.5 | torch | 899.1 | 901.7 | 904.0 | 898.2 | hipBLASLt contiguous 904.0 |
| 16384x5120x640 | bfloat16 | 1280 | data-parallel |  | 1116.7 | 1119.7 | 1116.7 | 1119.7 | 1081.6 | 1077.9 | torch | 1053.3 | 1056.3 | 1057.2 | 1056.7 | hipBLASLt contiguous 1057.2 |
| 32768x5120x640 | bfloat16 | 2560 | data-parallel |  | 1123.6 | 1121.1 | 1123.6 | 1121.1 | 1081.0 | 1077.7 | torch | 1089.9 | 1096.1 | 1096.3 | 1098.7 | hipBLASLt padded 1098.7 |
| 1024x7168x768 | bfloat16 | 112 | data-parallel | yes | 542.4 | 543.7 | 524.7 | 520.4 | 543.5 | 541.0 | flydsl | 707.8 | 433.2 (copy) | 678.1 | 678.3 | AITER flydsl contiguous 707.8 |
| 10240x7168x768 | bfloat16 | 1120 | data-parallel |  | 1106.9 | 1107.3 | 1106.9 | 1107.3 | 1069.9 | 1071.8 | opus | 401.1 | 405.1 | 863.8 | 863.2 | hipBLASLt contiguous 863.8 |
| 512x4096x1024 | bfloat16 | 32 | data-parallel | yes | 200.2 | 200.5 | 195.8 | 195.1 | 200.4 | 200.5 | torch | 492.1 | 486.4 | 492.1 | 487.8 | AITER torch contiguous 492.1 |
| 512x6144x1024 | bfloat16 | 48 | data-parallel | yes | 296.9 | 293.5 | 286.2 | 284.6 | 297.9 | 294.8 | flydsl | 591.2 | 298.0 (copy) | 573.5 | 574.8 | AITER flydsl contiguous 591.2 |
| 512x7168x1024 | bfloat16 | 56 | data-parallel | yes | 335.4 | 334.0 | 323.6 | 321.3 | 336.1 | 334.3 | flydsl | 645.1 | 327.9 (copy) | 607.8 | 606.5 | AITER flydsl contiguous 645.1 |
| 512x8192x1024 | bfloat16 | 64 | data-parallel | yes | 373.8 | 373.9 | 364.3 | 361.6 | 375.2 | 373.7 | flydsl | 704.1 | 362.3 (copy) | 665.5 | 661.7 | AITER flydsl contiguous 704.1 |
| 1024x4096x1024 | bfloat16 | 64 | data-parallel | yes | 374.4 | 372.7 | 360.9 | 363.1 | 374.6 | 372.9 | torch | 663.1 | 661.7 | 665.9 | 662.4 | hipBLASLt contiguous 665.9 |
| 1024x7168x1024 | bfloat16 | 112 | data-parallel | yes | 612.6 | 609.5 | 598.2 | 589.7 | 612.7 | 610.1 | flydsl | 786.7 | 498.2 (copy) | 774.3 | 775.1 | AITER flydsl contiguous 786.7 |
| 1024x8192x1024 | bfloat16 | 128 | data-parallel | yes | 673.2 | 684.0 | 664.0 | 663.5 | 673.7 | 684.3 | torch | 843.6 | 845.3 | 850.3 | 871.0 | hipBLASLt padded 871.0 |
| 2048x4096x1024 | bfloat16 | 128 | data-parallel | yes | 674.0 | 677.1 | 662.3 | 657.9 | 672.3 | 676.5 | torch | 847.6 | 848.1 | 853.1 | 859.6 | hipBLASLt padded 859.6 |
| 2048x7168x1024 | bfloat16 | 224 | data-parallel | yes | 1071.6 | 1075.5 | 1038.6 | 1035.9 | 1074.4 | 1071.8 | flydsl | 1035.1 | 727.2 (copy) | 1016.9 | 1015.2 | AITER flydsl contiguous 1035.1 |
| 2048x8192x1024 | bfloat16 | 256 | data-parallel | yes | 1133.0 | 1154.4 | 1120.7 | 1123.9 | 1127.5 | 1153.2 | flydsl | 1096.0 | 796.6 (copy) | 1060.0 | 1085.6 | AITER flydsl contiguous 1096.0 |
| 3072x4096x1024 | bfloat16 | 192 | data-parallel | yes | 953.1 | 954.4 | 935.7 | 921.2 | 958.5 | 953.0 | torch | 977.4 | 975.0 | 983.2 | 981.4 | hipBLASLt contiguous 983.2 |
| 4096x4096x1024 | bfloat16 | 256 | data-parallel | yes | 1143.4 | 1155.4 | 1120.6 | 1123.8 | 1150.5 | 1153.9 | torch | 1130.7 | 1128.1 | 1133.4 | 1129.9 | hipBLASLt contiguous 1133.4 |
| 4096x7168x1024 | bfloat16 | 448 | data-parallel |  | 1128.8 | 1135.5 | 1128.8 | 1135.5 | 1130.0 | 1128.2 | flydsl | 1105.4 | 896.7 (copy) | 1038.8 | 1034.7 | AITER flydsl contiguous 1105.4 |
| 4096x8192x1024 | bfloat16 | 512 | data-parallel |  | 1191.4 | 1211.8 | 1191.4 | 1211.8 | 1188.3 | 1201.8 | asm | 1188.3 | 1176.0 | 1147.0 | 1161.0 | AITER asm contiguous 1188.3 |
| 5120x4096x1024 | bfloat16 | 320 | data-parallel |  | 901.3 | 907.0 | 901.3 | 907.0 | 894.2 | 892.4 | torch | 1018.7 | 1020.4 | 1020.8 | 1018.4 | hipBLASLt contiguous 1020.8 |
| 7680x4096x1024 | bfloat16 | 480 | data-parallel |  | 1167.1 | 1171.5 | 1167.1 | 1171.5 | 1165.3 | 1164.6 | torch | 981.5 | 983.7 | 986.2 | 980.0 | hipBLASLt contiguous 986.2 |
| 8192x7168x1024 | bfloat16 | 896 | data-parallel |  | 1186.3 | 1186.2 | 1186.3 | 1186.2 | 1165.5 | 1159.9 | asm | 1107.6 | 1107.9 | 1064.7 | 1054.1 | AITER asm padded 1107.9 |
| 8192x8192x1024 | bfloat16 | 1024 | data-parallel |  | 1263.9 | 1270.5 | 1263.9 | 1270.5 | 1226.3 | 1239.2 | asm | 1204.7 | 1201.4 | 1199.3 | 1209.5 | hipBLASLt padded 1209.5 |
| 16384x4096x1024 | bfloat16 | 1024 | data-parallel |  | 1268.4 | 1266.8 | 1268.4 | 1266.8 | 1241.2 | 1235.9 | torch | 1206.0 | 1207.3 | 1211.2 | 1208.6 | hipBLASLt contiguous 1211.2 |
| 16384x7168x1024 | bfloat16 | 1792 | data-parallel |  | 1287.4 | 1296.0 | 1287.4 | 1296.0 | 1263.0 | 1255.1 | asm | 1191.0 | 1192.7 | 1243.1 | 1240.0 | hipBLASLt contiguous 1243.1 |
| 16384x8192x1024 | bfloat16 | 2048 | data-parallel |  | 1272.2 | 1274.7 | 1272.2 | 1274.7 | 1238.9 | 1246.6 | torch | 1237.0 | 1237.5 | 1243.2 | 1243.4 | hipBLASLt padded 1243.4 |
| 32768x7168x1024 | bfloat16 | 3584 | data-parallel |  | 1262.3 | 1264.1 | 1262.3 | 1264.1 | 1236.8 | 1230.6 | torch | 1249.1 | 1258.7 | 1262.2 | 1257.0 | hipBLASLt contiguous 1262.2 |
| 32768x8192x1024 | bfloat16 | 4096 | data-parallel |  | 1266.1 | 1265.5 | 1266.1 | 1265.5 | 1229.8 | 1227.7 | torch | 1255.0 | 1258.3 | 1262.8 | 1262.6 | hipBLASLt contiguous 1262.8 |
| 512x5120x1280 | bfloat16 | 40 | data-parallel | yes | 264.5 | 265.8 | 255.2 | 257.9 | 264.5 | 266.3 | flydsl | 538.0 | 290.3 (copy) | 571.1 | 562.2 | hipBLASLt contiguous 571.1 |
| 1024x5120x1280 | bfloat16 | 80 | data-parallel | yes | 488.1 | 488.9 | 471.4 | 470.2 | 486.3 | 489.2 | torch | 720.0 | 732.0 | 722.3 | 730.9 | AITER torch padded 732.0 |
| 2048x5120x1280 | bfloat16 | 160 | data-parallel | yes | 880.6 | 884.4 | 864.4 | 861.6 | 884.8 | 885.1 | torch | 944.4 | 947.0 | 948.1 | 942.1 | hipBLASLt contiguous 948.1 |
| 4096x5120x1280 | bfloat16 | 320 | data-parallel |  | 955.4 | 960.6 | 955.4 | 960.6 | 946.9 | 949.1 | torch | 997.3 | 994.4 | 1000.0 | 997.2 | hipBLASLt contiguous 1000.0 |
| 8192x5120x1280 | bfloat16 | 640 | data-parallel |  | 1191.7 | 1201.4 | 1191.7 | 1201.4 | 1178.9 | 1184.3 | flydsl | 1152.9 | 975.3 (copy) | 1130.0 | 1128.5 | AITER flydsl contiguous 1152.9 |
| 16384x5120x1280 | bfloat16 | 1280 | data-parallel |  | 1343.7 | 1344.8 | 1343.7 | 1344.8 | 1315.3 | 1316.9 | flydsl | 1264.9 | 1100.3 (copy) | 1294.3 | 1297.9 | hipBLASLt padded 1297.9 |
| 32768x5120x1280 | bfloat16 | 2560 | data-parallel |  | 1308.2 | 1317.7 | 1308.2 | 1317.7 | 1294.8 | 1294.5 | torch | 1301.4 | 1306.2 | 1307.3 | 1310.3 | hipBLASLt padded 1310.3 |
| 512x1536x1536 | bfloat16 | 12 | one-tile reduce-scatter, n = 8 |  | 105.8 | 108.5 | 83.5 | 83.9 | 85.9 | 85.7 | flydsl | 349.5 | 155.6 (copy) | 306.9 | 310.6 | AITER flydsl contiguous 349.5 |
| 512x2048x1536 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 |  | 130.6 | 132.1 | 111.3 | 111.4 | 113.5 | 113.4 | torch | 402.0 | 402.1 | 404.7 | 403.5 | hipBLASLt contiguous 404.7 |
| 512x2304x1536 | bfloat16 | 18 | one-tile reduce-scatter, n = 7 |  | 134.7 | 134.2 | 124.4 | 124.2 | 126.8 | 126.9 | flydsl | 392.2 | 205.7 (copy) | 383.1 | 384.9 | AITER flydsl contiguous 392.2 |
| 512x4096x1536 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 213.8 | 216.1 | 220.0 | 218.6 | 224.5 | 223.2 | torch | 570.7 | 578.8 | 576.4 | 567.0 | AITER torch padded 578.8 |
| 512x6144x1536 | bfloat16 | 48 | data-parallel | yes | 326.2 | 327.4 | 316.8 | 320.3 | 328.0 | 328.5 | flydsl | 651.9 | 367.3 (copy) | 673.9 | 666.0 | hipBLASLt contiguous 673.9 |
| 512x7168x1536 | bfloat16 | 56 | data-parallel | yes | 375.5 | 374.1 | 363.1 | 365.1 | 375.2 | 374.9 | flydsl | 679.6 | 389.4 (copy) | 592.1 | 584.7 | AITER flydsl contiguous 679.6 |
| 1024x1536x1536 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 |  | 175.5 | 174.9 | 163.8 | 164.9 | 169.1 | 169.5 | flydsl | 507.5 | 276.0 (copy) | 480.2 | 479.6 | AITER flydsl contiguous 507.5 |
| 1024x2048x1536 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 213.0 | 214.1 | 216.3 | 218.1 | 221.3 | 223.1 | torch | 574.0 | 577.2 | 576.7 | 567.5 | AITER torch padded 577.2 |
| 1024x2304x1536 | bfloat16 | 36 | one-tile owner, n = 2 |  | 242.4 | 243.5 | 242.6 | 244.1 | 249.0 | 249.2 | torch | 535.4 | 532.4 | 538.4 | 533.4 | hipBLASLt contiguous 538.4 |
| 1024x3072x1536 | bfloat16 | 48 | data-parallel | yes | 327.6 | 327.9 | 316.8 | 319.6 | 327.5 | 327.7 | flydsl | 640.4 | 373.3 (copy) | 667.6 | 660.9 | hipBLASLt contiguous 667.6 |
| 1024x4096x1536 | bfloat16 | 64 | data-parallel | yes | 420.7 | 419.7 | 409.1 | 413.3 | 421.4 | 419.3 | torch | 639.4 | 635.8 | 639.2 | 635.6 | AITER torch contiguous 639.4 |
| 1024x7168x1536 | bfloat16 | 112 | data-parallel | yes | 689.4 | 688.8 | 676.6 | 677.6 | 688.5 | 691.2 | torch | 889.2 | 896.1 | 893.6 | 900.5 | hipBLASLt padded 900.5 |
| 2048x1536x1536 | bfloat16 | 48 | data-parallel | yes | 327.7 | 328.7 | 317.9 | 320.4 | 329.0 | 328.7 | torch | 668.6 | 656.5 | 671.3 | 664.4 | hipBLASLt contiguous 671.3 |
| 2048x2048x1536 | bfloat16 | 64 | data-parallel | yes | 421.0 | 420.0 | 411.2 | 413.4 | 421.3 | 420.0 | torch | 639.0 | 636.6 | 639.6 | 636.9 | hipBLASLt contiguous 639.6 |
| 2048x4096x1536 | bfloat16 | 128 | data-parallel | yes | 763.3 | 767.1 | 750.4 | 763.1 | 766.0 | 767.3 | torch | 956.4 | 959.3 | 958.9 | 968.7 | hipBLASLt padded 968.7 |
| 3072x2048x1536 | bfloat16 | 96 | data-parallel | yes | 602.0 | 601.8 | 589.5 | 595.0 | 603.9 | 603.9 | opus | 879.0 | 886.3 | 816.8 | 821.8 | AITER opus padded 886.3 |
| 3072x4096x1536 | bfloat16 | 192 | data-parallel | yes | 1073.9 | 1072.2 | 1042.8 | 1052.4 | 1081.3 | 1072.2 | opus | 1138.7 | 1140.6 | 1098.1 | 1104.4 | AITER opus padded 1140.6 |
| 4096x1536x1536 | bfloat16 | 96 | data-parallel | yes | 603.4 | 604.6 | 588.7 | 593.1 | 605.8 | 603.3 | torch | 821.4 | 820.7 | 820.8 | 819.2 | AITER torch contiguous 821.4 |
| 4096x2048x1536 | bfloat16 | 128 | data-parallel | yes | 765.3 | 770.9 | 746.7 | 756.5 | 764.0 | 769.9 | opus | 1024.2 | 1026.2 | 957.3 | 970.4 | AITER opus padded 1026.2 |
| 4096x3072x1536 | bfloat16 | 192 | data-parallel | yes | 1077.5 | 1073.8 | 1038.2 | 1046.2 | 1078.2 | 1075.8 | flydsl | 1040.9 | 774.8 (copy) | 1103.3 | 1097.1 | hipBLASLt contiguous 1103.3 |
| 4096x4096x1536 | bfloat16 | 256 | data-parallel | yes | 1260.9 | 1269.5 | 1243.9 | 1252.9 | 1266.2 | 1265.4 | torch | 1228.1 | 1231.3 | 1230.7 | 1236.5 | hipBLASLt padded 1236.5 |
| 5120x2048x1536 | bfloat16 | 160 | data-parallel | yes | 924.2 | 929.2 | 903.5 | 907.1 | 926.7 | 928.2 | torch | 1034.5 | 1036.8 | 1038.6 | 1048.0 | hipBLASLt padded 1048.0 |
| 5120x4096x1536 | bfloat16 | 320 | data-parallel |  | 992.3 | 1006.3 | 992.3 | 1006.3 | 988.7 | 989.4 | torch | 1101.0 | 1095.3 | 1103.3 | 1102.5 | hipBLASLt contiguous 1103.3 |
| 7680x2048x1536 | bfloat16 | 240 | data-parallel | yes | 1216.1 | 1207.9 | 1187.1 | 1193.5 | 1210.4 | 1213.7 | torch | 1041.6 | 1047.8 | 1045.6 | 1043.0 | AITER torch padded 1047.8 |
| 7680x4096x1536 | bfloat16 | 480 | data-parallel |  | 1267.2 | 1271.1 | 1267.2 | 1271.1 | 1257.9 | 1265.3 | torch | 1083.7 | 1093.0 | 1089.0 | 1089.3 | AITER torch padded 1093.0 |
| 8192x1536x1536 | bfloat16 | 192 | data-parallel | yes | 1075.0 | 1073.2 | 1045.7 | 1051.3 | 1075.4 | 1073.3 | torch | 1097.0 | 1098.2 | 1101.3 | 1097.6 | hipBLASLt contiguous 1101.3 |
| 10240x7168x1536 | bfloat16 | 1120 | data-parallel |  | 1285.5 | 1285.7 | 1285.5 | 1285.7 | 1259.1 | 1258.7 | asm | 1218.7 | 1224.3 | 1170.8 | 1167.7 | AITER asm padded 1224.3 |
| 16384x1536x1536 | bfloat16 | 384 | data-parallel |  | 1132.6 | 1140.0 | 1132.6 | 1140.0 | 1130.7 | 1132.6 | torch | 1155.6 | 1158.3 | 1158.6 | 1158.7 | hipBLASLt padded 1158.7 |
| 16384x2048x1536 | bfloat16 | 512 | data-parallel |  | 1312.9 | 1317.9 | 1312.9 | 1317.9 | 1309.8 | 1310.6 | torch | 1281.6 | 1288.5 | 1286.4 | 1288.7 | hipBLASLt padded 1288.7 |
| 16384x3072x1536 | bfloat16 | 768 | data-parallel |  | 1345.4 | 1351.7 | 1345.4 | 1351.7 | 1325.6 | 1336.2 | torch | 1303.3 | 1308.0 | 1306.7 | 1310.0 | hipBLASLt padded 1310.0 |
| 16384x4096x1536 | bfloat16 | 1024 | data-parallel |  | 1360.4 | 1361.7 | 1360.4 | 1361.7 | 1344.7 | 1343.9 | torch | 1318.1 | 1322.3 | 1323.9 | 1328.9 | hipBLASLt padded 1328.9 |
| 32768x1536x1536 | bfloat16 | 768 | data-parallel |  | 1339.1 | 1343.7 | 1339.1 | 1343.7 | 1321.3 | 1327.0 | flydsl | 1281.0 | 937.3 (copy) | 1297.4 | 1303.2 | hipBLASLt padded 1303.2 |
| 512x7168x1792 | bfloat16 | 56 | one-tile owner, n = 2 |  | 407.6 | 406.5 | 381.0 | 373.8 | 390.9 | 387.1 | flydsl | 723.9 | 417.2 (copy) | 603.0 | 598.2 | AITER flydsl contiguous 723.9 |
| 1024x7168x1792 | bfloat16 | 112 | data-parallel | yes | 719.0 | 713.8 | 707.9 | 705.4 | 720.3 | 713.0 | flydsl | 856.1 | 587.9 (copy) | 933.9 | 938.7 | hipBLASLt padded 938.7 |
| 2048x7168x1792 | bfloat16 | 224 | data-parallel | yes | 1243.0 | 1237.8 | 1225.4 | 1218.4 | 1245.4 | 1239.9 | flydsl | 1190.9 | 889.7 (copy) | 1144.5 | 1146.0 | AITER flydsl contiguous 1190.9 |
| 4096x7168x1792 | bfloat16 | 448 | data-parallel |  | 1285.3 | 1285.4 | 1285.3 | 1285.4 | 1287.7 | 1284.9 | flydsl | 1261.7 | 1039.6 (copy) | 1195.5 | 1193.2 | AITER flydsl contiguous 1261.7 |
| 8192x7168x1792 | bfloat16 | 896 | data-parallel |  | 1312.0 | 1317.6 | 1312.0 | 1317.6 | 1308.3 | 1305.9 | asm | 1254.4 | 1267.1 | 1209.6 | 1206.1 | AITER asm padded 1267.1 |
| 16384x7168x1792 | bfloat16 | 1792 | data-parallel |  | 1400.6 | 1406.0 | 1400.6 | 1406.0 | 1389.0 | 1389.3 | torch | 1373.3 | 1378.9 | 1384.2 | 1382.1 | hipBLASLt contiguous 1384.2 |
| 32768x7168x1792 | bfloat16 | 3584 | data-parallel |  | 1394.9 | 1398.6 | 1394.9 | 1398.6 | 1376.3 | 1382.0 | torch | 1379.6 | 1388.2 | 1392.0 | 1391.6 | hipBLASLt contiguous 1392.0 |
| 512x4096x2048 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 275.7 | 265.3 | 234.7 | 233.6 | 237.5 | 237.1 | torch | 614.2 | 604.4 | 618.4 | 601.8 | hipBLASLt contiguous 618.4 |
| 512x6144x2048 | float32 | 48 | one-tile owner, n = 2 |  | 347.2 | 342.1 | 306.8 | 304.0 | 315.5 | 313.7 | asm | 421.5 | 420.7 | 658.3 | 642.6 | hipBLASLt contiguous 658.3 |
| 512x6144x2048 | bfloat16 | 48 | one-tile owner, n = 2 |  | 387.5 | 380.5 | 347.8 | 345.2 | 354.0 | 349.6 | flydsl | 690.9 | 402.5 (copy) | 716.1 | 701.7 | hipBLASLt contiguous 716.1 |
| 512x8192x2048 | bfloat16 | 64 | one-tile owner, n = 2 |  | 497.8 | 493.7 | 448.1 | 447.2 | 452.7 | 448.8 | flydsl | 794.0 | 465.7 (copy) | 681.4 | 680.1 | AITER flydsl contiguous 794.0 |
| 512x16384x2048 | bfloat16 | 128 | data-parallel | yes | 823.7 | 871.1 | 822.7 | 827.1 | 823.9 | 828.6 | flydsl | 818.1 | 522.2 (copy) | 1030.1 | 1043.0 | hipBLASLt padded 1043.0 |
| 1024x4096x2048 | bfloat16 | 64 | one-tile owner, n = 2 |  | 501.5 | 495.4 | 446.3 | 444.5 | 451.8 | 448.0 | torch | 686.2 | 680.3 | 686.2 | 679.6 | AITER torch contiguous 686.2 |
| 1024x6144x2048 | float32 | 96 | data-parallel | yes | 604.5 | 590.5 | 580.6 | 574.3 | 607.2 | 591.3 | opus | 670.0 | 650.6 | 901.6 | 904.9 | hipBLASLt padded 904.9 |
| 1024x6144x2048 | bfloat16 | 96 | data-parallel | yes | 653.4 | 645.9 | 648.8 | 635.3 | 657.4 | 645.2 | flydsl | 810.1 | 559.3 (copy) | 891.8 | 896.7 | hipBLASLt padded 896.7 |
| 1024x8192x2048 | bfloat16 | 128 | data-parallel | yes | 826.1 | 827.5 | 818.3 | 853.1 | 827.6 | 828.2 | flydsl | 1066.4 | 712.2 (copy) | 1038.1 | 1052.2 | AITER flydsl contiguous 1066.4 |
| 1024x16384x2048 | bfloat16 | 256 | data-parallel | yes | 1366.8 | 1372.1 | 1349.1 | 1350.5 | 1370.6 | 1369.3 | flydsl | 938.6 | 697.4 (copy) | 1328.1 | 1320.8 | hipBLASLt contiguous 1328.1 |
| 2048x4096x2048 | bfloat16 | 128 | data-parallel | yes | 826.5 | 819.2 | 821.9 | 814.8 | 824.1 | 821.5 | torch | 1035.7 | 1036.9 | 1037.2 | 1043.7 | hipBLASLt padded 1043.7 |
| 2048x6144x2048 | float32 | 192 | data-parallel | yes | 1077.6 | 1070.5 | 1045.4 | 1048.9 | 1077.3 | 1074.4 | asm | 1034.0 | 1031.6 | 1081.9 | 1082.6 | hipBLASLt padded 1082.6 |
| 2048x6144x2048 | bfloat16 | 192 | data-parallel | yes | 1160.3 | 1182.3 | 1160.1 | 1145.8 | 1164.4 | 1166.4 | flydsl | 1140.7 | 846.9 (copy) | 1181.8 | 1180.6 | hipBLASLt contiguous 1181.8 |
| 2048x8192x2048 | bfloat16 | 256 | data-parallel | yes | 1363.1 | 1371.3 | 1345.0 | 1346.4 | 1367.7 | 1374.0 | flydsl | 1316.5 | 981.8 (copy) | 1304.7 | 1320.5 | hipBLASLt padded 1320.5 |
| 2048x16384x2048 | bfloat16 | 512 | data-parallel |  | 1409.4 | 1412.7 | 1409.4 | 1412.7 | 1402.2 | 1407.7 | flydsl | 1102.8 | 890.6 (copy) | 1356.8 | 1354.1 | hipBLASLt contiguous 1356.8 |
| 3072x4096x2048 | bfloat16 | 192 | data-parallel | yes | 1175.8 | 1148.4 | 1144.6 | 1122.1 | 1182.1 | 1156.2 | torch | 1175.4 | 1174.7 | 1178.5 | 1174.2 | hipBLASLt contiguous 1178.5 |
| 4096x4096x2048 | bfloat16 | 256 | data-parallel | yes | 1368.5 | 1369.4 | 1346.5 | 1344.2 | 1366.6 | 1372.9 | torch | 1320.7 | 1322.5 | 1324.7 | 1327.5 | hipBLASLt padded 1327.5 |
| 4096x6144x2048 | float32 | 384 | data-parallel |  | 1055.9 | 1063.4 | 1055.9 | 1063.4 | 1068.3 | 1062.0 | asm | 1022.8 | 1023.5 | 1123.9 | 1124.1 | hipBLASLt padded 1124.1 |
| 4096x6144x2048 | bfloat16 | 384 | data-parallel |  | 1210.3 | 1204.5 | 1210.3 | 1204.5 | 1209.8 | 1204.5 | flydsl | 1055.3 | 889.4 (copy) | 1210.0 | 1209.8 | hipBLASLt contiguous 1210.0 |
| 4096x8192x2048 | bfloat16 | 512 | data-parallel |  | 1401.0 | 1401.3 | 1401.0 | 1401.3 | 1391.6 | 1395.3 | flydsl | 1341.3 | 1126.6 (copy) | 1355.4 | 1368.3 | hipBLASLt padded 1368.3 |
| 4096x16384x2048 | bfloat16 | 1024 | data-parallel |  | 1440.9 | 1446.1 | 1440.9 | 1446.1 | 1425.4 | 1427.5 | flydsl | 969.1 | 859.8 (copy) | 1391.4 | 1393.1 | hipBLASLt padded 1393.1 |
| 5120x4096x2048 | bfloat16 | 320 | one-tile owner, n = 2 |  | 1077.5 | 1074.5 | 1054.7 | 1045.3 | 1050.3 | 1043.7 | torch | 1221.0 | 1226.4 | 1228.3 | 1226.6 | hipBLASLt contiguous 1228.3 |
| 7680x4096x2048 | bfloat16 | 480 | data-parallel |  | 1354.8 | 1353.1 | 1354.8 | 1353.1 | 1350.8 | 1350.1 | torch | 1145.1 | 1148.1 | 1152.2 | 1147.0 | hipBLASLt contiguous 1152.2 |
| 8192x6144x2048 | float32 | 768 | data-parallel |  | 1254.0 | 1251.1 | 1254.0 | 1251.1 | 1257.0 | 1260.2 | asm | 1227.9 | 1232.9 | 1159.9 | 1168.4 | AITER asm padded 1232.9 |
| 8192x6144x2048 | bfloat16 | 768 | data-parallel |  | 1430.5 | 1432.9 | 1430.5 | 1432.9 | 1418.9 | 1414.3 | flydsl | 854.3 | 782.3 (copy) | 1378.1 | 1381.8 | hipBLASLt padded 1381.8 |
| 8192x8192x2048 | bfloat16 | 1024 | data-parallel |  | 1427.7 | 1445.0 | 1427.7 | 1445.0 | 1408.1 | 1422.2 | asm | 1387.3 | 1399.4 | 1385.7 | 1394.2 | AITER asm padded 1399.4 |
| 8192x16384x2048 | bfloat16 | 2048 | data-parallel |  | 1397.6 | 1395.6 | 1397.6 | 1395.6 | 1380.1 | 1371.8 | triton | 1028.2 | 1026.6 | 1409.6 | 1406.9 | hipBLASLt contiguous 1409.6 |
| 16384x4096x2048 | bfloat16 | 1024 | data-parallel |  | 1433.7 | 1440.2 | 1433.7 | 1440.2 | 1427.0 | 1422.8 | torch | 1361.5 | 1366.3 | 1365.3 | 1370.2 | hipBLASLt padded 1370.2 |
| 16384x6144x2048 | float32 | 1536 | data-parallel |  | 1235.3 | 1238.0 | 1235.3 | 1238.0 | 1228.2 | 1223.7 | asm | 1180.5 | 1180.7 | 1143.1 | 1148.9 | AITER asm padded 1180.7 |
| 16384x6144x2048 | bfloat16 | 1536 | data-parallel |  | 1435.5 | 1448.0 | 1435.5 | 1448.0 | 1431.0 | 1432.1 | torch | 1396.7 | 1402.8 | 1405.6 | 1406.5 | hipBLASLt padded 1406.5 |
| 16384x8192x2048 | bfloat16 | 2048 | data-parallel |  | 1427.7 | 1433.1 | 1427.7 | 1433.1 | 1416.6 | 1416.6 | torch | 1401.1 | 1408.5 | 1411.9 | 1414.9 | hipBLASLt padded 1414.9 |
| 16384x16384x2048 | bfloat16 | 4096 | data-parallel |  | 1395.9 | 1401.9 | 1395.9 | 1401.9 | 1368.3 | 1364.3 | flydsl | 963.3 | 918.6 (copy) | 1414.5 | 1417.2 | hipBLASLt padded 1417.2 |
| 32768x6144x2048 | bfloat16 | 3072 | data-parallel |  | 1416.0 | 1437.6 | 1416.0 | 1437.6 | 1390.9 | 1384.4 | flydsl | 995.8 | 937.8 (copy) | 1410.1 | 1410.8 | hipBLASLt padded 1410.8 |
| 32768x8192x2048 | bfloat16 | 4096 | data-parallel |  | 1432.9 | 1428.4 | 1432.9 | 1428.4 | 1414.6 | 1411.9 | torch | 1402.1 | 1408.3 | 1411.9 | 1415.3 | hipBLASLt padded 1415.3 |
| 32768x16384x2048 | bfloat16 | 8192 | data-parallel |  | 1397.2 | 1406.6 | 1397.2 | 1406.6 | 1355.9 | 1376.0 | flydsl | 832.4 | 813.7 (copy) | 1415.5 | 1417.6 | hipBLASLt padded 1417.6 |
| 512x6144x3072 | float32 | 48 | one-tile owner, n = 2 |  | 431.0 | 433.0 | 338.0 | 339.1 | 346.6 | 345.2 | opus | 393.1 | 397.7 | 734.4 | 733.0 | hipBLASLt contiguous 734.4 |
| 512x6144x3072 | bfloat16 | 48 | one-tile owner, n = 2 |  | 471.6 | 472.5 | 374.9 | 374.5 | 380.2 | 378.9 | flydsl | 742.7 | 449.1 (copy) | 781.4 | 778.2 | hipBLASLt contiguous 781.4 |
| 1024x6144x3072 | float32 | 96 | one-tile owner, n = 2-3 |  | 712.2 | 704.8 | 659.2 | 646.8 | 683.7 | 673.5 | opus | 631.5 | 634.8 | 968.2 | 974.0 | hipBLASLt padded 974.0 |
| 1024x6144x3072 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 781.9 | 774.5 | 687.6 | 697.7 | 701.3 | 716.6 | flydsl | 888.5 | 631.4 (copy) | 956.7 | 955.4 | hipBLASLt contiguous 956.7 |
| 2048x6144x3072 | float32 | 192 | data-parallel | yes | 1182.0 | 1194.2 | 1169.1 | 1168.3 | 1193.6 | 1191.7 | asm | 1148.1 | 1157.4 | 1189.5 | 1187.8 | hipBLASLt contiguous 1189.5 |
| 2048x6144x3072 | bfloat16 | 192 | data-parallel | yes | 1276.5 | 1282.6 | 1254.1 | 1265.0 | 1281.7 | 1285.3 | flydsl | 1221.2 | 936.9 (copy) | 1254.1 | 1255.6 | hipBLASLt padded 1255.6 |
| 4096x6144x3072 | float32 | 384 | data-parallel |  | 1153.0 | 1149.2 | 1153.0 | 1149.2 | 1162.5 | 1161.9 | asm | 1124.2 | 1126.4 | 1231.3 | 1232.8 | hipBLASLt padded 1232.8 |
| 4096x6144x3072 | bfloat16 | 384 | data-parallel |  | 1263.1 | 1265.5 | 1263.1 | 1265.5 | 1269.4 | 1270.4 | flydsl | 1108.8 | 935.7 (copy) | 1283.4 | 1283.5 | hipBLASLt padded 1283.5 |
| 8192x6144x3072 | float32 | 768 | data-parallel |  | 1363.0 | 1369.6 | 1363.0 | 1369.6 | 1362.8 | 1368.9 | asm | 1318.1 | 1321.0 | 1260.0 | 1263.7 | AITER asm padded 1321.0 |
| 8192x6144x3072 | bfloat16 | 768 | data-parallel |  | 1514.4 | 1520.5 | 1514.4 | 1520.5 | 1511.9 | 1515.2 | flydsl | 846.3 | 773.9 (copy) | 1456.5 | 1455.1 | hipBLASLt contiguous 1456.5 |
| 16384x6144x3072 | float32 | 1536 | data-parallel |  | 1332.0 | 1344.9 | 1332.0 | 1344.9 | 1325.2 | 1331.6 | asm | 1291.0 | 1301.3 | 1238.9 | 1244.1 | AITER asm padded 1301.3 |
| 16384x6144x3072 | bfloat16 | 1536 | data-parallel |  | 1496.4 | 1503.7 | 1496.4 | 1503.7 | 1494.0 | 1490.9 | torch | 1469.8 | 1475.5 | 1478.5 | 1475.3 | hipBLASLt contiguous 1478.5 |
| 32768x6144x3072 | bfloat16 | 3072 | data-parallel |  | 1505.3 | 1524.8 | 1505.3 | 1524.8 | 1489.0 | 1497.5 | flydsl | 862.7 | 807.5 (copy) | 1475.3 | 1478.7 | hipBLASLt padded 1478.7 |
| 512x5120x3200 | bfloat16 | 40 | one-tile owner, n = 2 |  | 413.6 | 413.5 | 315.1 | 315.9 | 319.2 | 318.8 | flydsl | 642.7 | 642.9 | 697.6 | 697.6 | hipBLASLt contiguous 697.6 |
| 1024x5120x3200 | bfloat16 | 80 | one-tile owner, n = 2 |  | 771.6 | 775.4 | 605.0 | 611.9 | 599.0 | 598.4 | torch | 958.8 | 961.0 | 961.6 | 958.5 | hipBLASLt contiguous 961.6 |
| 2048x5120x3200 | bfloat16 | 160 | data-parallel | yes | 1150.5 | 1150.9 | 1142.3 | 1140.6 | 1156.2 | 1158.0 | torch | 1172.4 | 1177.3 | 1177.2 | 1178.8 | hipBLASLt padded 1178.8 |
| 4096x5120x3200 | bfloat16 | 320 | one-tile owner, n = 2 |  | 1206.1 | 1204.3 | 1108.1 | 1105.9 | 1108.1 | 1107.4 | torch | 1197.2 | 1200.2 | 1205.5 | 1197.3 | hipBLASLt contiguous 1205.5 |
| 8192x5120x3200 | bfloat16 | 640 | data-parallel |  | 1378.6 | 1378.2 | 1378.6 | 1378.2 | 1379.6 | 1378.6 | asm | 1323.6 | 1331.5 | 1316.5 | 1313.5 | AITER asm padded 1331.5 |
| 16384x5120x3200 | bfloat16 | 1280 | data-parallel |  | 1497.2 | 1515.3 | 1497.2 | 1515.3 | 1506.2 | 1508.7 | torch | 1465.2 | 1470.8 | 1475.8 | 1472.6 | hipBLASLt contiguous 1475.8 |
| 32768x5120x3200 | bfloat16 | 2560 | data-parallel |  | 1505.4 | 1516.9 | 1505.4 | 1516.9 | 1490.9 | 1493.6 | torch | 1467.5 | 1479.0 | 1481.4 | 1479.3 | hipBLASLt contiguous 1481.4 |
| 512x7168x3584 | bfloat16 | 56 | one-tile owner, n = 2 |  | 585.7 | 585.5 | 441.2 | 441.3 | 445.6 | 446.2 | flydsl | 846.8 | 506.5 (copy) | 710.5 | 707.1 | AITER flydsl contiguous 846.8 |
| 512x8192x3584 | bfloat16 | 64 | one-tile owner, n = 2 |  | 662.3 | 665.0 | 496.7 | 499.9 | 501.3 | 497.9 | flydsl | 784.3 | 492.9 (copy) | 772.3 | 769.0 | AITER flydsl contiguous 784.3 |
| 1024x7168x3584 | bfloat16 | 112 | one-tile owner, n = 2-3 |  | 905.8 | 892.8 | 848.7 | 844.1 | 855.9 | 864.4 | torch | 1085.0 | 1090.5 | 1089.5 | 1089.2 | AITER torch padded 1090.5 |
| 1024x8192x3584 | bfloat16 | 128 | data-parallel | yes | 945.5 | 986.7 | 942.0 | 963.8 | 969.8 | 985.0 | flydsl | 893.9 | 659.1 (copy) | 1140.2 | 1154.7 | hipBLASLt padded 1154.7 |
| 2048x8192x3584 | bfloat16 | 256 | data-parallel | yes | 1482.7 | 1490.1 | 1471.1 | 1473.7 | 1487.8 | 1490.0 | flydsl | 1436.5 | 1067.1 (copy) | 1430.3 | 1449.6 | hipBLASLt padded 1449.6 |
| 4096x8192x3584 | bfloat16 | 512 | data-parallel |  | 1509.3 | 1518.0 | 1509.3 | 1518.0 | 1510.1 | 1518.1 | flydsl | 1036.4 | 886.2 (copy) | 1473.0 | 1479.4 | hipBLASLt padded 1479.4 |
| 8192x8192x3584 | bfloat16 | 1024 | data-parallel |  | 1532.5 | 1544.6 | 1532.5 | 1544.6 | 1525.3 | 1535.2 | torch | 1481.3 | 1484.2 | 1486.1 | 1489.5 | hipBLASLt padded 1489.5 |
| 10240x7168x3584 | bfloat16 | 1120 | one-tile owner, n = 2-3 |  | 1445.6 | 1443.6 | 1435.8 | 1435.8 | 1426.4 | 1426.6 | flydsl | 835.0 | 738.6 (copy) | 1340.4 | 1339.9 | hipBLASLt contiguous 1340.4 |
| 16384x8192x3584 | bfloat16 | 2048 | data-parallel |  | 1531.5 | 1537.8 | 1531.5 | 1537.8 | 1512.2 | 1512.9 | flydsl | 843.5 | 801.7 (copy) | 1494.8 | 1492.6 | hipBLASLt contiguous 1494.8 |
| 32768x8192x3584 | bfloat16 | 4096 | data-parallel |  | 1527.8 | 1542.9 | 1527.8 | 1542.9 | 1512.9 | 1519.4 | flydsl | 866.7 | 833.4 (copy) | 1480.3 | 1490.2 | hipBLASLt padded 1490.2 |
| 512x1024x4096 | bfloat16 | 8 | one-tile reduce-scatter, n = 8 |  | 178.6 | 175.1 | 65.9 | 66.0 | 66.6 | 66.6 | flydsl | 341.3 | 195.5 (copy) | 329.7 | 329.1 | AITER flydsl contiguous 341.3 |
| 512x2048x4096 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 |  | 291.8 | 287.3 | 131.9 | 131.5 | 132.9 | 132.1 | flydsl | 504.7 | 302.3 (copy) | 511.4 | 519.0 | hipBLASLt padded 519.0 |
| 512x4096x4096 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 439.0 | 425.6 | 262.8 | 261.9 | 264.2 | 262.8 | flydsl | 709.6 | 413.5 (copy) | 631.4 | 625.4 | AITER flydsl contiguous 709.6 |
| 512x8192x4096 | bfloat16 | 64 | one-tile owner, n = 2 |  | 705.4 | 701.7 | 512.8 | 512.5 | 512.9 | 513.1 | flydsl | 909.2 | 545.0 (copy) | 812.6 | 803.3 | AITER flydsl contiguous 909.2 |
| 512x8704x4096 | bfloat16 | 68 | one-tile owner, n = 2 |  | 743.4 | 738.0 | 544.1 | 544.1 | 550.6 | 544.8 | torch | 901.0 | 902.9 | 901.8 | 902.2 | AITER torch padded 902.9 |
| 512x10240x4096 | bfloat16 | 80 | one-tile owner, n = 2 |  | 829.7 | 832.1 | 630.9 | 635.8 | 636.8 | 633.2 | torch | 866.5 | 868.6 | 875.3 | 871.2 | hipBLASLt contiguous 875.3 |
| 512x16384x4096 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1085.5 | 1079.3 | 974.9 | 999.4 | 974.1 | 1005.9 | flydsl | 833.4 | 417.9 (copy) | 697.1 | 727.2 | AITER flydsl contiguous 833.4 |
| 512x129280x4096 | bfloat16 | 1010 | data-parallel |  | 1380.4 | 1377.8 | 1380.4 | 1377.8 | 1379.8 | 1370.4 | asm | 1302.3 | 1320.7 | 1147.9 | 1190.9 | AITER asm padded 1320.7 |
| 768x1024x4096 | bfloat16 | 12 | one-tile reduce-scatter, n = 8 |  | 245.3 | 239.4 | 99.0 | 98.9 | 100.4 | 99.9 | torch | 435.3 | 435.3 | 437.7 | 436.9 | hipBLASLt contiguous 437.7 |
| 768x2048x4096 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 |  | 370.0 | 364.6 | 197.6 | 197.0 | 199.8 | 198.7 | flydsl | 550.3 | 361.9 (copy) | 569.7 | 564.4 | hipBLASLt contiguous 569.7 |
| 768x4096x4096 | bfloat16 | 48 | one-tile owner, n = 3 |  | 605.9 | 599.7 | 390.4 | 390.4 | 394.9 | 392.1 | flydsl | 761.4 | 511.0 (copy) | 807.8 | 809.1 | hipBLASLt padded 809.1 |
| 768x8192x4096 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 872.7 | 899.7 | 754.2 | 735.0 | 762.4 | 757.6 | torch | 1004.5 | 1011.8 | 1005.0 | 1015.2 | hipBLASLt padded 1015.2 |
| 768x8704x4096 | bfloat16 | 102 | one-tile owner, n = 2-3 |  | 934.4 | 908.6 | 804.5 | 796.8 | 806.9 | 810.7 | flydsl | 967.2 | 656.1 (copy) | 989.8 | 982.1 | hipBLASLt contiguous 989.8 |
| 768x10240x4096 | bfloat16 | 120 | one-tile owner, n = 2-3 |  | 1033.9 | 1020.4 | 938.5 | 910.8 | 941.5 | 949.3 | flydsl | 1203.0 | 679.1 (copy) | 1079.1 | 1077.7 | AITER flydsl contiguous 1203.0 |
| 768x129280x4096 | bfloat16 | 1515 | data-parallel |  | 1440.7 | 1445.4 | 1440.7 | 1445.4 | 1425.6 | 1426.6 | torch | 1346.2 | 1348.8 | 1349.1 | 1349.4 | hipBLASLt padded 1349.4 |
| 1024x512x4096 | bfloat16 | 8 | one-tile reduce-scatter, n = 8 |  | 179.3 | 174.4 | 66.0 | 66.0 | 66.6 | 66.6 | flydsl | 327.2 | 191.1 (copy) | 330.1 | 330.0 | hipBLASLt contiguous 330.1 |
| 1024x1024x4096 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 |  | 294.0 | 288.3 | 131.8 | 131.0 | 132.9 | 132.1 | flydsl | 505.8 | 310.1 (copy) | 529.3 | 535.0 | hipBLASLt padded 535.0 |
| 1024x2048x4096 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 440.6 | 425.3 | 262.5 | 261.6 | 263.7 | 262.3 | flydsl | 699.3 | 455.2 (copy) | 563.6 | 545.8 | AITER flydsl contiguous 699.3 |
| 1024x4096x4096 | bfloat16 | 64 | one-tile owner, n = 2 |  | 704.3 | 701.8 | 511.2 | 506.9 | 517.3 | 512.2 | flydsl | 928.1 | 628.0 (copy) | 807.8 | 803.9 | AITER flydsl contiguous 928.1 |
| 1024x8192x4096 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1091.4 | 1083.9 | 989.9 | 995.5 | 991.8 | 1006.7 | flydsl | 1232.2 | 827.3 (copy) | 723.2 | 716.1 | AITER flydsl contiguous 1232.2 |
| 1024x8704x4096 | bfloat16 | 136 | data-parallel | yes | 1069.3 | 1057.3 | 1059.2 | 1028.8 | 1068.3 | 1062.9 | torch | 1134.8 | 1135.9 | 1135.8 | 1139.2 | hipBLASLt padded 1139.2 |
| 1024x10240x4096 | bfloat16 | 160 | data-parallel | yes | 1184.3 | 1176.0 | 1174.1 | 1165.7 | 1184.1 | 1177.6 | torch | 1214.9 | 1214.6 | 1219.4 | 1218.9 | hipBLASLt contiguous 1219.4 |
| 1024x16384x4096 | bfloat16 | 256 | data-parallel | yes | 1517.2 | 1513.2 | 1501.5 | 1503.2 | 1517.0 | 1515.2 | flydsl | 891.5 | 634.1 (copy) | 1483.4 | 1478.0 | hipBLASLt contiguous 1483.4 |
| 1024x129280x4096 | bfloat16 | 2020 | data-parallel |  | 1465.5 | 1465.4 | 1465.5 | 1465.4 | 1451.6 | 1444.5 | torch | 1374.6 | 1376.4 | 1377.6 | 1376.6 | hipBLASLt contiguous 1377.6 |
| 1280x512x4096 | bfloat16 | 10 | one-tile reduce-scatter, n = 8 |  | 213.9 | 207.0 | 82.5 | 82.5 | 83.6 | 83.4 | flydsl | 334.0 | 198.5 (copy) | 376.8 | 377.4 | hipBLASLt padded 377.4 |
| 1280x1024x4096 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 |  | 323.5 | 320.8 | 164.7 | 163.7 | 166.1 | 165.3 | torch | 564.1 | 565.7 | 565.2 | 568.7 | hipBLASLt padded 568.7 |
| 1280x2048x4096 | bfloat16 | 40 | one-tile owner, n = 3 |  | 511.1 | 506.1 | 326.8 | 325.7 | 329.8 | 328.5 | flydsl | 655.2 | 454.0 (copy) | 720.5 | 709.6 | hipBLASLt contiguous 720.5 |
| 1280x129280x4096 | bfloat16 | 2525 | data-parallel |  | 1435.5 | 1434.2 | 1435.5 | 1434.2 | 1424.6 | 1415.7 | torch | 1389.4 | 1391.0 | 1393.2 | 1391.1 | hipBLASLt contiguous 1393.2 |
| 1536x512x4096 | bfloat16 | 12 | one-tile reduce-scatter, n = 8 |  | 245.5 | 240.0 | 99.0 | 98.9 | 100.2 | 99.9 | torch | 433.5 | 433.8 | 436.9 | 435.5 | hipBLASLt contiguous 436.9 |
| 1536x1024x4096 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 |  | 369.7 | 362.5 | 197.2 | 196.4 | 199.8 | 198.6 | flydsl | 546.0 | 355.8 (copy) | 570.6 | 564.5 | hipBLASLt contiguous 570.6 |
| 1536x2048x4096 | bfloat16 | 48 | one-tile owner, n = 3 |  | 604.5 | 600.0 | 392.2 | 389.7 | 395.6 | 392.8 | flydsl | 761.6 | 533.7 (copy) | 808.5 | 808.8 | hipBLASLt padded 808.8 |
| 1536x129280x4096 | bfloat16 | 3030 | data-parallel |  | 1456.0 | 1458.9 | 1456.0 | 1458.9 | 1442.5 | 1436.2 | torch | 1396.2 | 1391.4 | 1400.7 | 1391.3 | hipBLASLt contiguous 1400.7 |
| 1792x512x4096 | bfloat16 | 14 | one-tile reduce-scatter, n = 8 |  | 270.3 | 266.7 | 115.4 | 114.7 | 116.9 | 116.3 | flydsl | 418.7 | 258.5 (copy) | 478.1 | 481.8 | hipBLASLt padded 481.8 |
| 1792x1024x4096 | bfloat16 | 28 | one-tile reduce-scatter, n = 4 |  | 401.4 | 391.4 | 229.4 | 228.4 | 232.6 | 231.4 | flydsl | 606.6 | 397.4 (copy) | 504.6 | 496.7 | AITER flydsl contiguous 606.6 |
| 1792x2048x4096 | bfloat16 | 56 | one-tile owner, n = 2 |  | 624.4 | 616.5 | 453.5 | 450.7 | 458.3 | 455.1 | flydsl | 833.2 | 594.0 (copy) | 744.0 | 744.9 | AITER flydsl contiguous 833.2 |
| 1792x129280x4096 | bfloat16 | 3535 | data-parallel |  | 1472.9 | 1473.8 | 1472.9 | 1473.8 | 1451.6 | 1453.0 | asm | 1435.6 | 1440.8 | 1358.8 | 1354.7 | AITER asm padded 1440.8 |
| 2048x512x4096 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 |  | 293.3 | 289.8 | 132.2 | 131.7 | 133.4 | 132.6 | flydsl | 483.7 | 294.0 (copy) | 525.1 | 525.5 | hipBLASLt padded 525.5 |
| 2048x1024x4096 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 437.0 | 427.7 | 263.3 | 261.3 | 264.8 | 263.2 | flydsl | 703.7 | 450.4 (copy) | 559.5 | 565.6 | AITER flydsl contiguous 703.7 |
| 2048x2048x4096 | bfloat16 | 64 | one-tile owner, n = 2 |  | 704.1 | 701.7 | 511.6 | 508.9 | 514.7 | 508.7 | flydsl | 907.8 | 646.8 (copy) | 810.7 | 807.0 | AITER flydsl contiguous 907.8 |
| 2048x4096x4096 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1094.4 | 1083.0 | 979.3 | 994.8 | 999.5 | 995.5 | opus | 1212.0 | 1221.3 | 714.0 | 737.4 | AITER opus padded 1221.3 |
| 2048x8192x4096 | bfloat16 | 256 | data-parallel | yes | 1512.7 | 1511.4 | 1496.5 | 1500.8 | 1516.3 | 1511.5 | flydsl | 1463.4 | 1076.6 (copy) | 1462.1 | 1470.4 | hipBLASLt padded 1470.4 |
| 2048x8704x4096 | bfloat16 | 272 | one-tile reduce-scatter, n = 8 |  | 1264.2 | 1266.6 | 993.2 | 990.7 | 991.6 | 989.7 | torch | 1149.3 | 1152.3 | 1154.5 | 1153.9 | hipBLASLt contiguous 1154.5 |
| 2048x10240x4096 | bfloat16 | 320 | one-tile owner, n = 2 |  | 1268.9 | 1261.9 | 1136.7 | 1137.8 | 1142.8 | 1137.7 | torch | 1240.0 | 1242.4 | 1245.7 | 1242.4 | hipBLASLt contiguous 1245.7 |
| 2048x16384x4096 | bfloat16 | 512 | data-parallel |  | 1536.6 | 1541.6 | 1536.6 | 1541.6 | 1538.0 | 1542.3 | flydsl | 1491.9 | 1117.4 (copy) | 1491.5 | 1488.3 | AITER flydsl contiguous 1491.9 |
| 2048x129280x4096 | bfloat16 | 4040 | data-parallel |  | 1496.5 | 1506.3 | 1496.5 | 1506.3 | 1484.1 | 1483.1 | flydsl | 1426.3 | 1149.7 (copy) | 1368.5 | 1367.4 | AITER flydsl contiguous 1426.3 |
| 2304x512x4096 | bfloat16 | 18 | one-tile reduce-scatter, n = 7 |  | 299.8 | 296.7 | 148.1 | 147.6 | 149.2 | 148.4 | flydsl | 437.0 | 280.2 (copy) | 525.6 | 530.6 | hipBLASLt padded 530.6 |
| 2304x1024x4096 | bfloat16 | 36 | one-tile owner, n = 3 |  | 464.7 | 460.5 | 294.4 | 293.7 | 296.5 | 294.9 | flydsl | 640.8 | 431.4 (copy) | 732.4 | 730.5 | hipBLASLt contiguous 732.4 |
| 2304x2048x4096 | bfloat16 | 72 | one-tile owner, n = 2 |  | 774.4 | 771.2 | 572.5 | 573.3 | 577.0 | 575.2 | opus | 881.0 | 886.5 | 956.2 | 959.8 | hipBLASLt padded 959.8 |
| 2560x512x4096 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 |  | 323.3 | 320.3 | 164.8 | 164.0 | 166.1 | 165.2 | flydsl | 482.3 | 307.3 (copy) | 566.2 | 569.8 | hipBLASLt padded 569.8 |
| 2560x1024x4096 | bfloat16 | 40 | one-tile owner, n = 3 |  | 510.2 | 507.1 | 327.5 | 325.7 | 330.3 | 328.6 | flydsl | 689.6 | 463.0 (copy) | 720.1 | 708.6 | hipBLASLt contiguous 720.1 |
| 2560x2048x4096 | bfloat16 | 80 | one-tile owner, n = 2 |  | 829.5 | 828.6 | 637.0 | 636.6 | 638.5 | 630.9 | opus | 961.6 | 967.4 | 1015.8 | 1015.7 | hipBLASLt contiguous 1015.8 |
| 2816x512x4096 | bfloat16 | 22 | one-tile reduce-scatter, n = 5 |  | 347.1 | 339.7 | 180.7 | 180.1 | 182.3 | 181.3 | flydsl | 513.7 | 324.9 (copy) | 532.6 | 523.7 | hipBLASLt contiguous 532.6 |
| 2816x1024x4096 | bfloat16 | 44 | one-tile owner, n = 3 |  | 557.0 | 554.5 | 360.0 | 358.1 | 363.8 | 361.7 | flydsl | 724.8 | 496.0 (copy) | 756.6 | 758.4 | hipBLASLt padded 758.4 |
| 2816x2048x4096 | bfloat16 | 88 | one-tile owner, n = 2-3 |  | 844.9 | 831.2 | 696.8 | 691.4 | 695.9 | 694.9 | opus | 1003.4 | 1015.7 | 933.9 | 926.4 | AITER opus padded 1015.7 |
| 3072x512x4096 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 |  | 369.9 | 364.9 | 197.6 | 197.0 | 199.9 | 198.5 | flydsl | 548.6 | 346.2 (copy) | 573.5 | 565.5 | hipBLASLt contiguous 573.5 |
| 3072x1024x4096 | bfloat16 | 48 | one-tile owner, n = 3 |  | 602.9 | 597.1 | 390.6 | 388.5 | 394.5 | 392.1 | flydsl | 755.4 | 521.3 (copy) | 807.2 | 809.9 | hipBLASLt padded 809.9 |
| 3072x2048x4096 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 871.1 | 870.3 | 739.7 | 745.3 | 763.9 | 760.6 | opus | 1120.6 | 1118.1 | 1008.0 | 1009.7 | AITER opus contiguous 1120.6 |
| 3328x512x4096 | bfloat16 | 26 | one-tile reduce-scatter, n = 4 |  | 381.1 | 365.6 | 214.1 | 212.6 | 216.2 | 215.4 | flydsl | 577.4 | 359.4 (copy) | 475.9 | 465.5 | AITER flydsl contiguous 577.4 |
| 3328x1024x4096 | bfloat16 | 52 | one-tile owner, n = 2 |  | 581.3 | 575.7 | 422.9 | 422.4 | 427.3 | 425.7 | flydsl | 757.5 | 540.4 (copy) | 691.7 | 698.6 | AITER flydsl contiguous 757.5 |
| 3328x2048x4096 | bfloat16 | 104 | one-tile owner, n = 2-3 |  | 922.6 | 881.5 | 812.2 | 818.3 | 824.3 | 817.8 | opus | 1100.5 | 1107.1 | 834.4 | 825.8 | AITER opus padded 1107.1 |
| 3584x512x4096 | bfloat16 | 28 | one-tile reduce-scatter, n = 4 |  | 399.0 | 391.5 | 229.8 | 229.7 | 232.9 | 230.8 | flydsl | 599.1 | 376.1 (copy) | 473.5 | 473.8 | AITER flydsl contiguous 599.1 |
| 3584x1024x4096 | bfloat16 | 56 | one-tile owner, n = 2 |  | 623.3 | 618.9 | 453.3 | 450.8 | 458.0 | 455.2 | flydsl | 785.7 | 554.0 (copy) | 747.2 | 744.3 | AITER flydsl contiguous 785.7 |
| 3584x2048x4096 | bfloat16 | 112 | one-tile owner, n = 2-3 |  | 961.7 | 945.2 | 878.2 | 874.8 | 882.8 | 885.6 | opus | 1156.5 | 1156.0 | 991.1 | 978.7 | AITER opus contiguous 1156.5 |
| 3840x512x4096 | bfloat16 | 30 | one-tile reduce-scatter, n = 4 |  | 417.9 | 408.8 | 247.2 | 246.0 | 250.0 | 248.6 | flydsl | 621.1 | 388.3 (copy) | 545.5 | 538.8 | AITER flydsl contiguous 621.1 |
| 3840x1024x4096 | bfloat16 | 60 | one-tile owner, n = 2 |  | 667.0 | 658.4 | 479.8 | 482.0 | 489.4 | 487.5 | flydsl | 809.4 | 566.3 (copy) | 788.6 | 783.2 | AITER flydsl contiguous 809.4 |
| 3840x2048x4096 | bfloat16 | 120 | one-tile owner, n = 2-3 |  | 1018.9 | 1016.3 | 924.3 | 924.3 | 939.5 | 942.1 | opus | 1183.4 | 1187.1 | 880.9 | 812.6 | AITER opus padded 1187.1 |
| 4096x512x4096 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 438.5 | 427.6 | 262.2 | 261.3 | 264.0 | 262.5 | flydsl | 682.7 | 410.1 (copy) | 569.0 | 566.1 | AITER flydsl contiguous 682.7 |
| 4096x1024x4096 | bfloat16 | 64 | one-tile owner, n = 2 |  | 703.7 | 700.4 | 511.2 | 512.4 | 514.2 | 513.4 | flydsl | 911.0 | 621.3 (copy) | 813.4 | 806.8 | AITER flydsl contiguous 911.0 |
| 4096x2048x4096 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1089.9 | 1082.0 | 976.4 | 993.7 | 986.6 | 998.7 | opus | 1236.4 | 1243.2 | 735.7 | 691.9 | AITER opus padded 1243.2 |
| 4096x4096x4096 | bfloat16 | 256 | data-parallel | yes | 1516.7 | 1511.8 | 1502.7 | 1499.9 | 1519.4 | 1514.0 | flydsl | 1463.5 | 1151.3 (copy) | 1400.5 | 1398.2 | AITER flydsl contiguous 1463.5 |
| 4096x8192x4096 | bfloat16 | 512 | data-parallel |  | 1534.6 | 1543.1 | 1534.6 | 1543.1 | 1538.0 | 1539.1 | asm | 1508.5 | 1520.1 | 1487.7 | 1493.0 | AITER asm padded 1520.1 |
| 4096x8704x4096 | bfloat16 | 544 | one-tile reduce-scatter, n = 4 |  | 1393.9 | 1394.6 | 1258.3 | 1253.5 | 1255.7 | 1251.8 | torch | 1255.5 | 1259.8 | 1260.4 | 1260.7 | hipBLASLt padded 1260.7 |
| 4096x10240x4096 | bfloat16 | 640 | one-tile owner, n = 2 |  | 1452.2 | 1452.9 | 1405.8 | 1402.4 | 1404.8 | 1403.2 | flydsl | 1364.3 | 1149.3 (copy) | 1308.5 | 1300.5 | AITER flydsl contiguous 1364.3 |
| 4096x16384x4096 | bfloat16 | 1024 | data-parallel |  | 1545.1 | 1543.2 | 1545.1 | 1543.2 | 1535.4 | 1532.0 | flydsl | 1483.9 | 1258.6 (copy) | 1507.9 | 1505.0 | hipBLASLt contiguous 1507.9 |
| 4096x129280x4096 | bfloat16 | 8080 | data-parallel |  | 1520.4 | 1526.9 | 1520.4 | 1526.9 | 1509.3 | 1502.1 | asm | 1466.7 | 1472.3 | 1402.7 | 1386.1 | AITER asm padded 1472.3 |
| 4352x512x4096 | bfloat16 | 34 | one-tile owner, n = 3 |  | 439.2 | 435.3 | 279.5 | 278.7 | 282.6 | 281.2 | flydsl | 541.3 | 362.8 (copy) | 633.1 | 624.8 | hipBLASLt contiguous 633.1 |
| 4352x1024x4096 | bfloat16 | 68 | one-tile owner, n = 2 |  | 739.0 | 733.3 | 544.8 | 539.6 | 545.9 | 539.6 | opus | 848.1 | 857.0 | 906.7 | 903.2 | hipBLASLt contiguous 906.7 |
| 4352x2048x4096 | bfloat16 | 136 | data-parallel | yes | 1040.0 | 1059.3 | 1026.5 | 1044.9 | 1048.7 | 1059.1 | torch | 1097.0 | 1089.4 | 1094.0 | 1085.3 | AITER torch contiguous 1097.0 |
| 5120x512x4096 | bfloat16 | 40 | one-tile owner, n = 3 |  | 510.0 | 506.7 | 326.9 | 325.5 | 330.0 | 328.3 | flydsl | 564.1 | 391.5 (copy) | 717.4 | 707.0 | hipBLASLt contiguous 717.4 |
| 5120x1024x4096 | bfloat16 | 80 | one-tile owner, n = 2 |  | 812.0 | 796.9 | 620.9 | 634.5 | 639.0 | 633.0 | opus | 982.6 | 1006.9 | 1007.6 | 1006.6 | hipBLASLt contiguous 1007.6 |
| 5120x2048x4096 | bfloat16 | 160 | data-parallel | yes | 1169.8 | 1172.4 | 1158.7 | 1159.1 | 1175.7 | 1175.8 | torch | 1243.6 | 1244.9 | 1246.8 | 1251.8 | hipBLASLt padded 1251.8 |
| 6144x512x4096 | bfloat16 | 48 | one-tile owner, n = 3 |  | 603.1 | 600.9 | 391.8 | 389.6 | 395.7 | 393.4 | flydsl | 762.9 | 472.7 (copy) | 804.7 | 807.8 | hipBLASLt padded 807.8 |
| 6144x1024x4096 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 871.9 | 902.4 | 722.0 | 745.7 | 755.5 | 748.9 | opus | 1116.6 | 1110.7 | 1016.2 | 1008.0 | AITER opus contiguous 1116.6 |
| 6144x2048x4096 | bfloat16 | 192 | data-parallel | yes | 1311.7 | 1309.6 | 1303.3 | 1284.4 | 1312.8 | 1296.6 | opus | 1374.8 | 1383.7 | 1300.5 | 1302.0 | AITER opus padded 1383.7 |
| 6144x129280x4096 | bfloat16 | 12120 | one-tile owner, n = 2-3 |  | 1497.9 | 1501.5 | 1504.2 | 1503.8 | 1484.6 | 1478.4 | asm | 1476.9 | 1480.3 | 1396.3 | 1386.1 | AITER asm padded 1480.3 |
| 6400x512x4096 | bfloat16 | 50 | one-tile owner, n = 3 |  | 617.4 | 616.2 | 406.0 | 405.6 | 411.7 | 409.2 | flydsl | 696.2 | 448.0 (copy) | 691.2 | 683.4 | AITER flydsl contiguous 696.2 |
| 6400x1024x4096 | bfloat16 | 100 | one-tile owner, n = 2-3 |  | 913.8 | 914.1 | 760.1 | 792.0 | 789.7 | 788.8 | opus | 1075.1 | 1076.2 | 850.1 | 840.1 | AITER opus padded 1076.2 |
| 6400x2048x4096 | bfloat16 | 200 | data-parallel | yes | 1322.4 | 1313.1 | 1289.3 | 1303.8 | 1318.1 | 1319.2 | torch | 1235.0 | 1232.0 | 1235.2 | 1233.6 | hipBLASLt contiguous 1235.2 |
| 6912x512x4096 | bfloat16 | 54 | one-tile owner, n = 2 |  | 599.4 | 595.8 | 437.5 | 436.5 | 442.6 | 439.7 | flydsl | 701.7 | 395.5 (copy) | 733.2 | 716.3 | hipBLASLt contiguous 733.2 |
| 6912x1024x4096 | bfloat16 | 108 | one-tile owner, n = 2-3 |  | 928.5 | 964.2 | 850.5 | 847.4 | 853.6 | 852.5 | opus | 1114.4 | 1118.5 | 725.7 | 724.9 | AITER opus padded 1118.5 |
| 6912x2048x4096 | bfloat16 | 216 | data-parallel | yes | 1371.1 | 1367.9 | 1347.7 | 1347.7 | 1370.9 | 1368.4 | torch | 1320.2 | 1321.1 | 1326.7 | 1323.5 | hipBLASLt contiguous 1326.7 |
| 7168x512x4096 | bfloat16 | 56 | one-tile owner, n = 2 |  | 623.6 | 617.9 | 451.7 | 449.6 | 457.8 | 453.6 | flydsl | 696.9 | 402.5 (copy) | 748.1 | 746.7 | hipBLASLt contiguous 748.1 |
| 7168x1024x4096 | bfloat16 | 112 | one-tile owner, n = 2-3 |  | 1004.6 | 959.7 | 884.0 | 867.5 | 886.9 | 876.9 | opus | 1180.8 | 1176.3 | 1060.5 | 1053.8 | AITER opus contiguous 1180.8 |
| 7168x2048x4096 | bfloat16 | 224 | data-parallel | yes | 1391.7 | 1393.2 | 1381.4 | 1375.1 | 1395.8 | 1391.8 | torch | 1344.6 | 1350.4 | 1349.0 | 1355.6 | hipBLASLt padded 1355.6 |
| 7424x512x4096 | bfloat16 | 58 | one-tile owner, n = 2 |  | 638.5 | 637.5 | 468.1 | 466.9 | 474.0 | 470.6 | flydsl | 706.7 | 458.5 (copy) | 758.3 | 746.9 | hipBLASLt contiguous 758.3 |
| 7424x1024x4096 | bfloat16 | 116 | one-tile owner, n = 2-3 |  | 1011.0 | 989.1 | 905.1 | 892.4 | 917.7 | 904.9 | opus | 1137.8 | 1144.1 | 818.4 | 816.3 | AITER opus padded 1144.1 |
| 7424x2048x4096 | bfloat16 | 232 | data-parallel | yes | 1424.2 | 1423.8 | 1408.2 | 1408.7 | 1428.1 | 1423.8 | torch | 1213.7 | 1214.7 | 1220.4 | 1213.3 | hipBLASLt contiguous 1220.4 |
| 7680x512x4096 | bfloat16 | 60 | one-tile owner, n = 2 |  | 659.5 | 660.8 | 483.1 | 481.4 | 487.8 | 485.7 | flydsl | 720.7 | 495.4 (copy) | 781.4 | 768.4 | hipBLASLt contiguous 781.4 |
| 7680x1024x4096 | bfloat16 | 120 | one-tile owner, n = 2-3 |  | 1020.0 | 1017.6 | 928.4 | 939.0 | 948.3 | 936.0 | opus | 1193.6 | 1200.0 | 810.5 | 810.6 | AITER opus padded 1200.0 |
| 7680x2048x4096 | bfloat16 | 240 | data-parallel | yes | 1449.1 | 1447.9 | 1433.4 | 1432.0 | 1454.9 | 1451.0 | torch | 1241.7 | 1243.5 | 1245.8 | 1242.1 | hipBLASLt contiguous 1245.8 |
| 7936x512x4096 | bfloat16 | 62 | one-tile owner, n = 2 |  | 683.9 | 679.4 | 495.9 | 498.7 | 503.3 | 500.8 | flydsl | 653.6 | 449.0 (copy) | 796.8 | 786.1 | hipBLASLt contiguous 796.8 |
| 7936x1024x4096 | bfloat16 | 124 | one-tile owner, n = 2-3 |  | 1062.5 | 1045.9 | 961.4 | 958.6 | 971.5 | 962.1 | opus | 1160.6 | 1169.7 | 740.9 | 741.2 | AITER opus padded 1169.7 |
| 7936x2048x4096 | bfloat16 | 248 | data-parallel | yes | 1481.5 | 1481.2 | 1466.2 | 1464.8 | 1484.2 | 1479.1 | torch | 1435.4 | 1436.0 | 1439.9 | 1442.7 | hipBLASLt padded 1442.7 |
| 8192x512x4096 | bfloat16 | 64 | one-tile owner, n = 2 |  | 706.8 | 702.9 | 512.8 | 512.6 | 515.4 | 512.6 | flydsl | 868.3 | 537.5 (copy) | 821.8 | 808.6 | AITER flydsl contiguous 868.3 |
| 8192x1024x4096 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1091.7 | 1082.3 | 986.2 | 987.3 | 989.3 | 991.3 | opus | 1249.0 | 1257.5 | 746.7 | 709.0 | AITER opus padded 1257.5 |
| 8192x2048x4096 | bfloat16 | 256 | data-parallel | yes | 1515.3 | 1508.4 | 1499.5 | 1499.4 | 1514.4 | 1508.6 | torch | 1467.8 | 1466.4 | 1466.9 | 1465.8 | AITER torch contiguous 1467.8 |
| 8192x4096x4096 | bfloat16 | 512 | data-parallel |  | 1532.8 | 1541.7 | 1532.8 | 1541.7 | 1539.7 | 1539.3 | flydsl | 1498.5 | 1207.1 (copy) | 1494.1 | 1492.9 | AITER flydsl contiguous 1498.5 |
| 8192x8192x4096 | bfloat16 | 1024 | data-parallel |  | 1557.5 | 1564.8 | 1557.5 | 1564.8 | 1556.6 | 1554.0 | asm | 1522.6 | 1533.1 | 1500.4 | 1501.6 | AITER asm padded 1533.1 |
| 8192x8704x4096 | bfloat16 | 1088 | one-tile owner, n = 2 |  | 1465.9 | 1466.4 | 1431.3 | 1426.9 | 1424.0 | 1421.9 | asm | 1396.0 | 1400.3 | 1356.8 | 1355.7 | AITER asm padded 1400.3 |
| 8192x10240x4096 | bfloat16 | 1280 | data-parallel |  | 1536.6 | 1546.5 | 1536.6 | 1546.5 | 1545.3 | 1529.3 | torch | 1497.6 | 1503.9 | 1506.2 | 1505.0 | hipBLASLt contiguous 1506.2 |
| 8192x16384x4096 | bfloat16 | 2048 | data-parallel |  | 1532.1 | 1541.1 | 1532.1 | 1541.1 | 1520.7 | 1522.3 | torch | 1502.7 | 1520.6 | 1513.8 | 1522.2 | hipBLASLt padded 1522.2 |
| 8192x129280x4096 | bfloat16 | 16160 | one-tile reduce-scatter, n = 4 |  | 1530.7 | 1548.1 | 1546.2 | 1540.2 | 1514.9 | 1510.9 | flydsl | 1463.0 | 1377.3 (copy) | 1373.6 | 1365.2 | AITER flydsl contiguous 1463.0 |
| 10240x512x4096 | bfloat16 | 80 | one-tile owner, n = 2 |  | 818.2 | 827.5 | 637.0 | 635.2 | 638.2 | 630.4 | torch | 1005.2 | 1003.2 | 1008.2 | 1007.1 | hipBLASLt contiguous 1008.2 |
| 10240x1024x4096 | bfloat16 | 160 | data-parallel | yes | 1174.9 | 1167.6 | 1153.7 | 1155.2 | 1172.9 | 1169.5 | torch | 1247.2 | 1249.9 | 1253.4 | 1252.4 | hipBLASLt contiguous 1253.4 |
| 10240x2048x4096 | bfloat16 | 320 | one-tile owner, n = 2 |  | 1265.3 | 1262.1 | 1134.9 | 1134.2 | 1142.4 | 1135.4 | torch | 1271.4 | 1274.1 | 1277.1 | 1278.1 | hipBLASLt padded 1278.1 |
| 10240x129280x4096 | bfloat16 | 20200 | data-parallel |  | 1506.7 | 1515.9 | 1506.7 | 1515.9 | 1501.3 | 1486.1 | asm | 1485.5 | 1488.0 | 1429.2 | 1390.9 | AITER asm padded 1488.0 |
| 12288x512x4096 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 873.7 | 863.7 | 745.8 | 756.8 | 758.3 | 751.5 | opus | 1110.1 | 1111.6 | 1002.6 | 1004.5 | AITER opus padded 1111.6 |
| 12288x1024x4096 | bfloat16 | 192 | data-parallel | yes | 1311.7 | 1312.3 | 1292.5 | 1302.9 | 1315.1 | 1309.7 | opus | 1381.1 | 1390.1 | 1305.3 | 1300.5 | AITER opus padded 1390.1 |
| 12288x2048x4096 | bfloat16 | 384 | one-tile owner, n = 2 |  | 1372.3 | 1375.2 | 1311.4 | 1301.8 | 1307.8 | 1304.3 | opus | 1379.5 | 1393.4 | 1326.6 | 1325.9 | AITER opus padded 1393.4 |
| 12288x129280x4096 | bfloat16 | 24240 | data-parallel |  | 1524.1 | 1536.1 | 1524.1 | 1536.1 | 1511.1 | 1505.4 | asm | 1489.4 | 1485.0 | 1418.1 | 1380.4 | AITER asm contiguous 1489.4 |
| 14336x512x4096 | bfloat16 | 112 | one-tile owner, n = 2-3 |  | 1003.6 | 960.6 | 867.8 | 865.2 | 862.9 | 874.1 | opus | 1172.7 | 1175.2 | 659.0 | 660.5 | AITER opus padded 1175.2 |
| 14336x1024x4096 | bfloat16 | 224 | data-parallel | yes | 1395.9 | 1385.6 | 1377.8 | 1373.8 | 1395.3 | 1383.7 | torch | 1348.1 | 1350.7 | 1353.2 | 1355.9 | hipBLASLt padded 1355.9 |
| 14336x2048x4096 | bfloat16 | 448 | data-parallel |  | 1422.0 | 1422.0 | 1422.0 | 1422.0 | 1426.5 | 1424.1 | flydsl | 1388.4 | 1059.8 (copy) | 1380.8 | 1378.8 | AITER flydsl contiguous 1388.4 |
| 14336x129280x4096 | bfloat16 | 28280 | one-tile owner, n = 2-3 |  | 1496.3 | 1516.7 | 1521.2 | 1514.1 | 1498.6 | 1487.8 | asm | 1463.3 | 1489.1 | 1398.9 | 1388.6 | AITER asm padded 1489.1 |
| 16384x512x4096 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1083.4 | 1073.2 | 975.7 | 983.2 | 979.2 | 989.9 | opus | 1247.0 | 1251.9 | 816.3 | 797.0 | AITER opus padded 1251.9 |
| 16384x1024x4096 | bfloat16 | 256 | data-parallel | yes | 1511.7 | 1507.0 | 1501.1 | 1496.5 | 1514.3 | 1511.0 | flydsl | 1467.4 | 899.6 (copy) | 1329.1 | 1313.6 | AITER flydsl contiguous 1467.4 |
| 16384x2048x4096 | bfloat16 | 512 | data-parallel |  | 1536.7 | 1537.9 | 1536.7 | 1537.9 | 1538.1 | 1535.9 | torch | 1416.1 | 1414.1 | 1418.3 | 1406.2 | hipBLASLt contiguous 1418.3 |
| 16384x16384x4096 | bfloat16 | 4096 | data-parallel |  | 1536.5 | 1538.3 | 1536.5 | 1538.3 | 1517.3 | 1512.7 | torch | 1514.5 | 1520.5 | 1533.1 | 1524.4 | hipBLASLt contiguous 1533.1 |
| 16384x129280x4096 | bfloat16 | 32320 | one-tile owner, n = 2 |  | 1487.1 | 1553.6 | 1550.3 | 1551.9 | 1533.0 | 1524.8 | asm | 1486.5 | 1488.5 | 1396.0 | 1378.3 | AITER asm padded 1488.5 |
| 32768x16384x4096 | bfloat16 | 8192 | data-parallel |  | 1540.2 | 1536.4 | 1540.2 | 1536.4 | 1514.3 | 1510.9 | flydsl | 1482.4 | 1404.0 (copy) | 1530.2 | 1512.4 | hipBLASLt contiguous 1530.2 |
| 512x7168x4224 | bfloat16 | 56 | one-tile owner, n = 2 |  | 636.8 | 641.4 | 453.2 | 452.8 | 456.5 | 455.5 | flydsl | 836.0 | 841.0 | 873.4 | 876.9 | hipBLASLt padded 876.9 |
| 1024x7168x4224 | bfloat16 | 112 | one-tile owner, n = 2-3 |  | 1029.5 | 986.1 | 879.6 | 883.0 | 877.8 | 881.0 | flydsl | 1144.3 | 1147.8 | 852.0 | 852.5 | AITER flydsl padded 1147.8 |
| 10240x7168x4224 | bfloat16 | 1120 | one-tile owner, n = 2-3 |  | 1478.0 | 1484.6 | 1455.1 | 1456.3 | 1446.8 | 1448.6 | opus | 952.7 | 956.3 | 1365.9 | 1363.4 | hipBLASLt contiguous 1365.9 |
| 512x6144x4608 | bfloat16 | 48 | one-tile owner, n = 3 |  | 608.1 | 607.6 | 393.6 | 395.3 | 399.1 | 398.2 | flydsl | 788.1 | 483.8 (copy) | 688.1 | 687.2 | AITER flydsl contiguous 788.1 |
| 1024x6144x4608 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 911.4 | 945.8 | 768.2 | 771.2 | 766.1 | 773.7 | torch | 1008.9 | 1009.9 | 1014.0 | 1011.8 | hipBLASLt contiguous 1014.0 |
| 2048x6144x4608 | bfloat16 | 192 | data-parallel | yes | 1333.5 | 1329.7 | 1320.4 | 1317.1 | 1337.3 | 1341.6 | flydsl | 1266.8 | 983.2 (copy) | 1316.9 | 1318.3 | hipBLASLt padded 1318.3 |
| 4096x6144x4608 | bfloat16 | 384 | one-tile owner, n = 2 |  | 1414.0 | 1410.6 | 1318.5 | 1324.3 | 1330.3 | 1324.1 | torch | 1330.5 | 1336.6 | 1335.9 | 1338.0 | hipBLASLt padded 1338.0 |
| 8192x6144x4608 | bfloat16 | 768 | data-parallel |  | 1562.3 | 1571.5 | 1562.3 | 1571.5 | 1565.4 | 1570.7 | torch | 1507.2 | 1510.3 | 1511.1 | 1510.1 | hipBLASLt contiguous 1511.1 |
| 16384x6144x4608 | bfloat16 | 1536 | data-parallel |  | 1556.6 | 1570.3 | 1556.6 | 1570.3 | 1550.0 | 1557.8 | torch | 1513.4 | 1517.5 | 1523.7 | 1512.4 | hipBLASLt contiguous 1523.7 |
| 32768x6144x4608 | bfloat16 | 3072 | data-parallel |  | 1554.3 | 1569.4 | 1554.3 | 1569.4 | 1543.4 | 1550.6 | torch | 1495.1 | 1504.4 | 1501.6 | 1506.1 | hipBLASLt padded 1506.1 |
| 512x1024x5120 | float32 | 8 | one-tile reduce-scatter, n = 10 |  | 204.9 | 203.4 | 64.7 | 65.1 | 65.4 | 65.8 | asm | 333.5 | 339.2 | 339.1 | 336.3 | AITER asm padded 339.2 |
| 512x5120x5120 | bfloat16 | 40 | one-tile owner, n = 3 |  | 567.9 | 574.2 | 333.4 | 334.4 | 336.9 | 336.7 | torch | 796.5 | 795.5 | 797.0 | 793.8 | hipBLASLt contiguous 797.0 |
| 512x6400x5120 | bfloat16 | 50 | one-tile owner, n = 3 |  | 671.0 | 669.3 | 415.3 | 415.2 | 419.5 | 418.8 | flydsl | 698.9 | 457.3 (copy) | 705.9 | 727.3 | hipBLASLt padded 727.3 |
| 512x12800x5120 | bfloat16 | 100 | one-tile owner, n = 2-3 |  | 975.8 | 936.8 | 808.9 | 814.3 | 821.0 | 819.9 | flydsl | 800.3 | 429.4 (copy) | 829.4 | 841.2 | hipBLASLt padded 841.2 |
| 512x37888x5120 | bfloat16 | 296 | one-tile owner, n = 3 |  | 1163.5 | 1176.9 | 983.2 | 1002.5 | 978.3 | 1001.9 | torch | 1053.9 | 1081.0 | 1057.5 | 1079.5 | AITER torch padded 1081.0 |
| 512x51200x5120 | bfloat16 | 400 | data-parallel |  | 1208.0 | 1225.0 | 1208.0 | 1225.0 | 1204.4 | 1223.4 | flydsl | 1138.4 | 635.1 (copy) | 1099.3 | 1123.9 | AITER flydsl contiguous 1138.4 |
| 768x1024x5120 | float32 | 12 | one-tile reduce-scatter, n = 10 |  | 271.2 | 274.6 | 97.0 | 97.5 | 98.0 | 98.6 | asm | 410.3 | 422.9 | 381.2 | 385.8 | AITER asm padded 422.9 |
| 768x37888x5120 | bfloat16 | 444 | data-parallel |  | 1335.3 | 1352.7 | 1335.3 | 1352.7 | 1335.7 | 1351.5 | torch | 1235.2 | 1249.1 | 1243.4 | 1252.2 | hipBLASLt padded 1252.2 |
| 1024x1024x5120 | float32 | 16 | one-tile reduce-scatter, n = 8 |  | 329.8 | 334.9 | 129.1 | 129.5 | 130.4 | 130.6 | asm | 458.8 | 462.1 | 485.2 | 490.9 | hipBLASLt padded 490.9 |
| 1024x5120x5120 | bfloat16 | 80 | one-tile owner, n = 2 |  | 912.9 | 914.1 | 650.9 | 650.7 | 654.0 | 656.7 | torch | 1009.9 | 1035.7 | 1011.5 | 1038.3 | hipBLASLt padded 1038.3 |
| 1024x6400x5120 | bfloat16 | 100 | one-tile owner, n = 2-3 |  | 978.2 | 941.7 | 804.7 | 791.4 | 809.9 | 807.8 | flydsl | 941.1 | 668.1 (copy) | 868.5 | 882.8 | AITER flydsl contiguous 941.1 |
| 1024x12800x5120 | bfloat16 | 200 | data-parallel | yes | 1360.5 | 1363.3 | 1346.4 | 1346.3 | 1366.3 | 1364.8 | flydsl | 793.5 | 556.9 (copy) | 1239.3 | 1251.6 | hipBLASLt padded 1251.6 |
| 1024x37888x5120 | bfloat16 | 592 | one-tile owner, n = 2 |  | 1346.6 | 1363.4 | 1268.7 | 1282.7 | 1262.0 | 1278.4 | torch | 1060.9 | 1090.9 | 1065.1 | 1090.4 | AITER torch padded 1090.9 |
| 1024x51200x5120 | bfloat16 | 800 | one-tile reduce-scatter, n = 4 |  | 1380.8 | 1398.8 | 1282.7 | 1302.4 | 1266.9 | 1290.4 | asm | 1223.8 | 1256.6 | 1070.9 | 1099.1 | AITER asm padded 1256.6 |
| 1536x1024x5120 | float32 | 24 | one-tile reduce-scatter, n = 5 |  | 417.5 | 412.9 | 191.9 | 192.7 | 194.4 | 194.9 | asm | 578.5 | 585.8 | 614.0 | 622.0 | hipBLASLt padded 622.0 |
| 1536x37888x5120 | bfloat16 | 888 | one-tile owner, n = 2-3 |  | 1403.5 | 1430.9 | 1362.6 | 1380.1 | 1357.2 | 1369.8 | torch | 1272.8 | 1294.2 | 1279.7 | 1295.8 | hipBLASLt padded 1295.8 |
| 2048x1024x5120 | float32 | 32 | one-tile reduce-scatter, n = 4 |  | 480.8 | 480.3 | 255.4 | 254.8 | 258.0 | 257.2 | opus | 633.4 | 633.8 | 661.9 | 682.1 | hipBLASLt padded 682.1 |
| 2048x5120x5120 | bfloat16 | 160 | data-parallel | yes | 1224.5 | 1221.8 | 1214.1 | 1217.3 | 1227.2 | 1225.3 | torch | 1244.9 | 1252.4 | 1253.1 | 1253.9 | hipBLASLt padded 1253.9 |
| 2048x6400x5120 | bfloat16 | 200 | data-parallel | yes | 1345.5 | 1346.2 | 1342.0 | 1343.9 | 1362.1 | 1352.7 | flydsl | 1247.4 | 944.4 (copy) | 1246.7 | 1260.8 | hipBLASLt padded 1260.8 |
| 2048x12800x5120 | bfloat16 | 400 | data-parallel |  | 1365.0 | 1372.6 | 1365.0 | 1372.6 | 1378.5 | 1377.5 | flydsl | 806.5 | 674.3 (copy) | 1261.5 | 1274.2 | hipBLASLt padded 1274.2 |
| 2048x37888x5120 | bfloat16 | 1184 | data-parallel |  | 1451.0 | 1468.4 | 1451.0 | 1468.4 | 1443.9 | 1455.3 | torch | 1206.6 | 1238.6 | 1213.8 | 1241.3 | hipBLASLt padded 1241.3 |
| 2048x51200x5120 | bfloat16 | 1600 | one-tile owner, n = 2 |  | 1461.4 | 1479.6 | 1427.0 | 1443.3 | 1416.1 | 1432.2 | asm | 211.6 | 216.4 | 1185.4 | 1213.7 | hipBLASLt padded 1213.7 |
| 4096x1024x5120 | float32 | 64 | one-tile owner, n = 2 |  | 706.5 | 715.6 | 488.0 | 492.1 | 492.0 | 501.7 | opus | 965.9 | 975.7 | 904.3 | 924.8 | AITER opus padded 975.7 |
| 4096x5120x5120 | bfloat16 | 320 | one-tile owner, n = 2 |  | 1318.7 | 1314.2 | 1158.6 | 1151.7 | 1159.5 | 1155.0 | torch | 1262.5 | 1269.8 | 1269.3 | 1270.2 | hipBLASLt padded 1270.2 |
| 4096x6400x5120 | bfloat16 | 400 | data-parallel |  | 1366.6 | 1371.5 | 1366.6 | 1371.5 | 1380.9 | 1382.0 | flydsl | 1330.1 | 1090.0 (copy) | 1266.5 | 1276.9 | AITER flydsl contiguous 1330.1 |
| 4096x12800x5120 | bfloat16 | 800 | one-tile reduce-scatter, n = 4 |  | 1446.7 | 1457.0 | 1368.1 | 1369.2 | 1362.1 | 1364.5 | asm | 1325.6 | 1326.1 | 1271.0 | 1284.8 | AITER asm padded 1326.1 |
| 4096x51200x5120 | bfloat16 | 3200 | one-tile owner, n = 2 |  | 1526.0 | 1547.1 | 1513.9 | 1528.5 | 1495.7 | 1508.6 | asm | 1454.3 | 1474.3 | 1381.6 | 1405.3 | AITER asm padded 1474.3 |
| 8192x1024x5120 | float32 | 128 | one-tile owner, n = 2 |  | 1099.7 | 1100.4 | 967.0 | 977.2 | 986.9 | 992.1 | opus | 1246.1 | 1251.0 | 1140.0 | 1145.6 | AITER opus padded 1251.0 |
| 8192x5120x5120 | bfloat16 | 640 | one-tile owner, n = 2 |  | 1491.0 | 1495.0 | 1428.3 | 1426.4 | 1427.0 | 1426.6 | torch | 1348.7 | 1355.4 | 1359.4 | 1355.0 | hipBLASLt contiguous 1359.4 |
| 8192x6400x5120 | bfloat16 | 800 | one-tile reduce-scatter, n = 4 |  | 1475.1 | 1481.3 | 1372.9 | 1370.0 | 1362.4 | 1369.8 | flydsl | 1325.3 | 1146.4 (copy) | 1281.3 | 1289.3 | AITER flydsl contiguous 1325.3 |
| 8192x12800x5120 | bfloat16 | 1600 | one-tile owner, n = 2 |  | 1494.4 | 1512.3 | 1461.0 | 1476.0 | 1454.9 | 1471.7 | asm | 1408.8 | 1423.6 | 1234.3 | 1241.6 | AITER asm padded 1423.6 |
| 8192x51200x5120 | bfloat16 | 6400 | data-parallel |  | 1562.2 | 1582.4 | 1562.2 | 1582.4 | 1542.5 | 1557.2 | flydsl | 1485.4 | 1384.4 (copy) | 1412.9 | 1448.5 | AITER flydsl contiguous 1485.4 |
| 16384x1024x5120 | float32 | 256 | data-parallel | yes | 1446.3 | 1448.7 | 1430.7 | 1436.6 | 1447.0 | 1447.9 | opus | 1335.2 | 1342.7 | 1337.2 | 1333.3 | AITER opus padded 1342.7 |
| 16384x5120x5120 | bfloat16 | 1280 | data-parallel |  | 1552.7 | 1567.5 | 1552.7 | 1567.5 | 1548.5 | 1556.8 | torch | 1519.9 | 1524.7 | 1534.5 | 1525.7 | hipBLASLt contiguous 1534.5 |
| 16384x6400x5120 | bfloat16 | 1600 | one-tile owner, n = 2 |  | 1478.4 | 1503.8 | 1451.1 | 1466.3 | 1440.9 | 1454.5 | flydsl | 1379.3 | 1258.5 (copy) | 1237.5 | 1246.5 | AITER flydsl contiguous 1379.3 |
| 16384x12800x5120 | bfloat16 | 3200 | one-tile owner, n = 2 |  | 1524.6 | 1550.5 | 1517.4 | 1530.4 | 1495.0 | 1510.5 | torch | 1483.1 | 1509.2 | 1491.6 | 1502.9 | AITER torch padded 1509.2 |
| 16384x51200x5120 | bfloat16 | 12800 | data-parallel |  | 1557.4 | 1577.6 | 1557.4 | 1577.6 | 1537.9 | 1555.6 | flydsl | 1482.3 | 1429.5 (copy) | 1413.2 | 1421.2 | AITER flydsl contiguous 1482.3 |
| 32768x5120x5120 | bfloat16 | 2560 | data-parallel |  | 1543.2 | 1563.3 | 1543.2 | 1563.3 | 1532.6 | 1546.1 | torch | 1496.8 | 1510.9 | 1500.9 | 1508.9 | AITER torch padded 1510.9 |
| 32768x6400x5120 | bfloat16 | 3200 | one-tile owner, n = 2 |  | 1523.0 | 1544.5 | 1513.1 | 1529.3 | 1494.3 | 1507.8 | torch | 1471.2 | 1499.7 | 1475.9 | 1489.3 | AITER torch padded 1499.7 |
| 32768x12800x5120 | bfloat16 | 6400 | data-parallel |  | 1549.6 | 1570.6 | 1549.6 | 1570.6 | 1531.9 | 1549.0 | torch | 1507.7 | 1538.3 | 1517.0 | 1546.8 | hipBLASLt padded 1546.8 |
| 32768x51200x5120 | bfloat16 | 25600 | data-parallel |  | 1558.2 | 1576.4 | 1558.2 | 1576.4 | 1537.5 | 1555.5 | flydsl | 1485.2 | 1458.1 (copy) | 1398.0 | 1456.1 | AITER flydsl contiguous 1485.2 |
| 512x1280x6144 | bfloat16 | 10 | one-tile reduce-scatter, n = 12 |  | 270.5 | 272.4 | 85.4 | 85.4 | 86.4 | 86.5 | flydsl | 401.7 | 254.3 (copy) | 386.2 | 386.7 | AITER flydsl contiguous 401.7 |
| 512x1536x6144 | bfloat16 | 12 | one-tile reduce-scatter, n = 10 |  | 307.4 | 310.5 | 102.6 | 102.8 | 103.7 | 103.7 | flydsl | 432.6 | 274.6 (copy) | 415.4 | 412.9 | AITER flydsl contiguous 432.6 |
| 512x2048x6144 | float32 | 16 | one-tile reduce-scatter, n = 8 |  | 377.5 | 375.7 | 132.7 | 132.5 | 133.8 | 133.7 | asm | 487.5 | 493.5 | 509.9 | 510.8 | hipBLASLt padded 510.8 |
| 512x2304x6144 | bfloat16 | 18 | one-tile reduce-scatter, n = 7 |  | 380.4 | 385.5 | 153.4 | 153.7 | 154.3 | 154.1 | flydsl | 520.0 | 328.5 (copy) | 412.2 | 410.3 | AITER flydsl contiguous 520.0 |
| 512x2560x6144 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 |  | 413.9 | 416.4 | 170.4 | 170.3 | 171.1 | 171.1 | torch | 512.0 | 512.3 | 514.6 | 513.9 | hipBLASLt contiguous 514.6 |
| 512x3072x6144 | float32 | 24 | one-tile reduce-scatter, n = 5 |  | 469.3 | 467.0 | 197.7 | 198.2 | 199.8 | 200.1 | opus | 497.8 | 509.8 | 638.1 | 628.6 | hipBLASLt contiguous 638.1 |
| 512x3072x6144 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 |  | 477.7 | 474.3 | 205.5 | 205.3 | 206.4 | 206.4 | flydsl | 602.0 | 387.0 (copy) | 493.1 | 494.6 | AITER flydsl contiguous 602.0 |
| 512x6144x6144 | bfloat16 | 48 | one-tile owner, n = 3 |  | 694.5 | 695.4 | 406.1 | 403.8 | 408.1 | 408.3 | flydsl | 809.2 | 498.6 (copy) | 766.6 | 775.4 | AITER flydsl contiguous 809.2 |
| 512x9216x6144 | bfloat16 | 72 | one-tile owner, n = 3-4 |  | 930.2 | 930.2 | 601.9 | 600.3 | 607.6 | 606.8 | flydsl | 902.3 | 406.0 (copy) | 624.3 | 622.4 | AITER flydsl contiguous 902.3 |
| 1024x2048x6144 | float32 | 32 | one-tile reduce-scatter, n = 4 |  | 532.3 | 526.0 | 261.7 | 261.3 | 264.1 | 264.5 | opus | 536.0 | 553.5 | 746.7 | 737.1 | hipBLASLt contiguous 746.7 |
| 1024x2304x6144 | bfloat16 | 36 | one-tile owner, n = 4 |  | 573.3 | 574.0 | 306.8 | 306.6 | 307.4 | 306.9 | torch | 675.3 | 677.4 | 675.9 | 683.5 | hipBLASLt padded 683.5 |
| 1024x2560x6144 | bfloat16 | 40 | one-tile owner, n = 3 |  | 617.7 | 618.3 | 340.1 | 338.3 | 341.7 | 341.7 | torch | 776.6 | 779.2 | 777.9 | 780.4 | hipBLASLt padded 780.4 |
| 1024x3072x6144 | float32 | 48 | one-tile owner, n = 3 |  | 648.0 | 649.9 | 386.0 | 389.3 | 391.2 | 393.9 | opus | 564.4 | 570.1 | 847.6 | 852.5 | hipBLASLt padded 852.5 |
| 1024x6144x6144 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 1009.2 | 1023.9 | 793.9 | 798.7 | 797.9 | 804.0 | torch | 1015.0 | 1035.5 | 1029.6 | 1040.9 | hipBLASLt padded 1040.9 |
| 1024x9216x6144 | bfloat16 | 144 | data-parallel | yes | 1159.4 | 1165.8 | 1159.3 | 1155.0 | 1165.6 | 1165.0 | torch | 1228.4 | 1233.3 | 1235.1 | 1236.2 | hipBLASLt padded 1236.2 |
| 2048x2048x6144 | float32 | 64 | one-tile owner, n = 3 |  | 812.4 | 816.0 | 508.4 | 510.6 | 506.8 | 516.1 | opus | 886.0 | 892.9 | 949.9 | 966.3 | hipBLASLt padded 966.3 |
| 2048x2304x6144 | bfloat16 | 72 | one-tile owner, n = 3-4 |  | 929.9 | 935.2 | 602.2 | 600.0 | 604.5 | 607.6 | torch | 988.8 | 998.9 | 989.3 | 998.1 | AITER torch padded 998.9 |
| 2048x2560x6144 | bfloat16 | 80 | one-tile owner, n = 2 |  | 968.9 | 973.8 | 665.5 | 667.2 | 670.3 | 672.7 | torch | 1049.6 | 1061.3 | 1051.9 | 1062.3 | hipBLASLt padded 1062.3 |
| 2048x3072x6144 | float32 | 96 | one-tile owner, n = 2-3 |  | 953.1 | 964.7 | 764.4 | 761.5 | 775.9 | 776.0 | opus | 818.6 | 820.9 | 980.7 | 951.7 | hipBLASLt contiguous 980.7 |
| 2048x6144x6144 | bfloat16 | 192 | data-parallel | yes | 1357.1 | 1357.1 | 1342.1 | 1340.6 | 1371.6 | 1358.8 | torch | 1340.6 | 1346.4 | 1350.2 | 1351.7 | hipBLASLt padded 1351.7 |
| 2048x9216x6144 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 |  | 1343.4 | 1344.6 | 1072.1 | 1071.2 | 1073.8 | 1072.9 | torch | 1237.6 | 1248.8 | 1245.0 | 1250.0 | hipBLASLt padded 1250.0 |
| 4096x2048x6144 | float32 | 128 | one-tile owner, n = 2 |  | 1174.2 | 1177.4 | 995.5 | 1003.6 | 1002.7 | 1020.1 | opus | 1155.8 | 1157.5 | 880.6 | 816.0 | AITER opus padded 1157.5 |
| 4096x2304x6144 | bfloat16 | 144 | data-parallel | yes | 1161.3 | 1162.2 | 1154.1 | 1150.4 | 1160.3 | 1163.7 | flydsl | 785.0 | 654.4 (copy) | 1232.4 | 1236.3 | hipBLASLt padded 1236.3 |
| 4096x2560x6144 | bfloat16 | 160 | data-parallel | yes | 1240.7 | 1246.2 | 1231.6 | 1230.7 | 1241.5 | 1241.5 | flydsl | 710.1 | 603.6 (copy) | 1277.0 | 1277.8 | hipBLASLt padded 1277.8 |
| 4096x3072x6144 | float32 | 192 | data-parallel | yes | 1313.4 | 1317.6 | 1302.0 | 1305.3 | 1316.5 | 1321.2 | asm | 1233.2 | 1232.0 | 1314.9 | 1318.3 | hipBLASLt padded 1318.3 |
| 4096x6144x6144 | bfloat16 | 384 | one-tile owner, n = 2 |  | 1469.7 | 1472.0 | 1349.5 | 1350.0 | 1359.2 | 1354.0 | torch | 1353.5 | 1365.8 | 1364.4 | 1365.8 | AITER torch padded 1365.8 |
| 4096x9216x6144 | bfloat16 | 576 | one-tile owner, n = 3 |  | 1471.4 | 1478.3 | 1344.7 | 1345.4 | 1343.7 | 1345.8 | torch | 1361.0 | 1370.0 | 1374.6 | 1366.6 | hipBLASLt contiguous 1374.6 |
| 8192x2048x6144 | float32 | 256 | data-parallel | yes | 1495.3 | 1492.8 | 1477.7 | 1479.8 | 1505.7 | 1496.2 | opus | 1392.9 | 1399.7 | 1367.5 | 1368.9 | AITER opus padded 1399.7 |
| 8192x2304x6144 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 |  | 1342.4 | 1344.9 | 1074.4 | 1075.4 | 1075.0 | 1074.4 | torch | 1235.6 | 1247.5 | 1239.7 | 1248.7 | hipBLASLt padded 1248.7 |
| 8192x2560x6144 | bfloat16 | 320 | one-tile owner, n = 3 |  | 1392.2 | 1391.5 | 1173.9 | 1176.6 | 1176.0 | 1174.2 | asm | 519.5 | 528.1 | 1285.4 | 1290.0 | hipBLASLt padded 1290.0 |
| 8192x3072x6144 | float32 | 384 | one-tile owner, n = 2 |  | 1384.4 | 1387.5 | 1279.7 | 1275.7 | 1289.0 | 1290.5 | asm | 1243.4 | 1243.2 | 1323.0 | 1326.6 | hipBLASLt padded 1326.6 |
| 8192x6144x6144 | bfloat16 | 768 | data-parallel |  | 1588.9 | 1595.3 | 1588.9 | 1595.3 | 1589.3 | 1593.2 | flydsl | 1544.6 | 1315.5 (copy) | 1535.0 | 1546.2 | hipBLASLt padded 1546.2 |
| 8192x9216x6144 | bfloat16 | 1152 | one-tile owner, n = 2 |  | 1529.1 | 1547.4 | 1482.3 | 1490.7 | 1470.3 | 1484.6 | asm | 1457.7 | 1466.6 | 1253.9 | 1258.4 | AITER asm padded 1466.6 |
| 16384x2048x6144 | float32 | 512 | data-parallel |  | 1410.1 | 1444.6 | 1410.1 | 1444.6 | 1428.9 | 1458.2 | opus | 1345.1 | 1349.5 | 1329.2 | 1335.5 | AITER opus padded 1349.5 |
| 16384x2304x6144 | bfloat16 | 576 | one-tile owner, n = 3 |  | 1431.4 | 1445.3 | 1325.6 | 1330.0 | 1321.4 | 1323.5 | flydsl | 1044.9 | 832.9 (copy) | 1336.7 | 1339.5 | hipBLASLt padded 1339.5 |
| 16384x2560x6144 | bfloat16 | 640 | one-tile owner, n = 2 |  | 1475.5 | 1485.9 | 1407.8 | 1420.4 | 1407.1 | 1411.0 | flydsl | 1367.4 | 1080.9 (copy) | 1362.3 | 1361.8 | AITER flydsl contiguous 1367.4 |
| 16384x3072x6144 | float32 | 768 | data-parallel |  | 1433.3 | 1464.6 | 1433.3 | 1464.6 | 1432.9 | 1456.4 | asm | 1388.3 | 1404.1 | 1351.5 | 1352.4 | AITER asm padded 1404.1 |
| 16384x6144x6144 | bfloat16 | 1536 | data-parallel |  | 1559.4 | 1580.0 | 1559.4 | 1580.0 | 1560.3 | 1567.9 | asm | 1515.5 | 1534.9 | 1558.9 | 1560.8 | hipBLASLt padded 1560.8 |
| 16384x9216x6144 | bfloat16 | 2304 | data-parallel |  | 1554.7 | 1576.8 | 1554.7 | 1576.8 | 1550.0 | 1567.4 | torch | 1528.9 | 1545.4 | 1542.9 | 1540.4 | AITER torch padded 1545.4 |
| 32768x2304x6144 | bfloat16 | 1152 | one-tile owner, n = 2 |  | 1468.8 | 1494.9 | 1422.9 | 1443.4 | 1414.8 | 1431.1 | flydsl | 652.9 | 591.8 (copy) | 1212.5 | 1214.3 | hipBLASLt padded 1214.3 |
| 32768x2560x6144 | bfloat16 | 1280 | data-parallel |  | 1512.0 | 1541.9 | 1512.0 | 1541.9 | 1509.7 | 1529.8 | flydsl | 1458.8 | 1194.2 (copy) | 1491.2 | 1517.5 | hipBLASLt padded 1517.5 |
| 32768x6144x6144 | bfloat16 | 3072 | data-parallel |  | 1569.5 | 1588.5 | 1569.5 | 1588.5 | 1562.2 | 1573.9 | torch | 1509.2 | 1548.7 | 1524.1 | 1549.1 | hipBLASLt padded 1549.1 |
| 32768x9216x6144 | bfloat16 | 4608 | data-parallel |  | 1562.7 | 1586.3 | 1562.7 | 1586.3 | 1553.5 | 1570.7 | torch | 1543.6 | 1571.2 | 1566.4 | 1576.2 | hipBLASLt padded 1576.2 |
| 512x5120x6400 | bfloat16 | 40 | one-tile owner, n = 4 |  | 654.9 | 638.9 | 339.2 | 339.9 | 341.9 | 340.8 | flydsl | 717.7 | 463.9 (copy) | 789.1 | 785.7 | hipBLASLt contiguous 789.1 |
| 1024x5120x6400 | bfloat16 | 80 | one-tile owner, n = 2 |  | 967.2 | 980.2 | 666.3 | 670.8 | 674.0 | 671.9 | flydsl | 702.8 | 536.2 (copy) | 1064.4 | 1070.1 | hipBLASLt padded 1070.1 |
| 2048x5120x6400 | bfloat16 | 160 | data-parallel | yes | 1237.4 | 1236.2 | 1228.4 | 1222.7 | 1237.5 | 1239.4 | flydsl | 787.8 | 646.0 (copy) | 1285.9 | 1281.7 | hipBLASLt contiguous 1285.9 |
| 4096x5120x6400 | bfloat16 | 320 | one-tile owner, n = 3 |  | 1395.3 | 1397.6 | 1180.2 | 1173.7 | 1181.1 | 1178.8 | torch | 1285.9 | 1295.9 | 1296.9 | 1298.9 | hipBLASLt padded 1298.9 |
| 8192x5120x6400 | bfloat16 | 640 | one-tile owner, n = 2 |  | 1511.2 | 1522.1 | 1448.8 | 1446.3 | 1444.8 | 1445.9 | flydsl | 1405.8 | 1189.5 (copy) | 1374.5 | 1370.8 | AITER flydsl contiguous 1405.8 |
| 16384x5120x6400 | bfloat16 | 1280 | data-parallel |  | 1558.9 | 1572.4 | 1558.9 | 1572.4 | 1561.3 | 1560.6 | torch | 1545.5 | 1563.0 | 1566.6 | 1569.2 | hipBLASLt padded 1569.2 |
| 32768x5120x6400 | bfloat16 | 2560 | data-parallel |  | 1564.4 | 1575.5 | 1564.4 | 1575.5 | 1560.3 | 1559.7 | torch | 1519.4 | 1549.8 | 1534.3 | 1556.4 | hipBLASLt padded 1556.4 |
| 512x16384x6656 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1174.4 | 1174.8 | 1058.6 | 1051.5 | 1051.7 | 1060.1 | flydsl | 956.1 | 512.2 (copy) | 927.1 | 890.2 | AITER flydsl contiguous 956.1 |
| 1024x16384x6656 | bfloat16 | 256 | data-parallel | yes | 1560.8 | 1555.5 | 1556.0 | 1540.7 | 1563.7 | 1553.2 | asm | 452.7 | 436.4 | 1552.2 | 1547.9 | hipBLASLt contiguous 1552.2 |
| 2048x16384x6656 | bfloat16 | 512 | data-parallel |  | 1528.7 | 1537.8 | 1528.7 | 1537.8 | 1532.4 | 1535.7 | asm | 1467.6 | 1475.5 | 1563.1 | 1557.5 | hipBLASLt contiguous 1563.1 |
| 4096x16384x6656 | bfloat16 | 1024 | data-parallel |  | 1559.9 | 1571.3 | 1559.9 | 1571.3 | 1556.4 | 1561.8 | torch | 1556.6 | 1579.8 | 1572.4 | 1574.9 | AITER torch padded 1579.8 |
| 8192x16384x6656 | bfloat16 | 2048 | data-parallel |  | 1583.9 | 1598.6 | 1583.9 | 1598.6 | 1577.5 | 1583.4 | asm | 1516.5 | 1528.4 | 1554.8 | 1575.7 | hipBLASLt padded 1575.7 |
| 16384x16384x6656 | bfloat16 | 4096 | data-parallel |  | 1576.2 | 1596.7 | 1576.2 | 1596.7 | 1569.7 | 1580.8 | asm | 1529.0 | 1544.3 | 1568.2 | 1573.4 | hipBLASLt padded 1573.4 |
| 32768x16384x6656 | bfloat16 | 8192 | data-parallel |  | 1587.0 | 1597.7 | 1587.0 | 1597.7 | 1577.8 | 1584.9 | asm | 290.9 | 292.0 | 1566.6 | 1560.0 | hipBLASLt contiguous 1566.6 |
| 512x1024x7168 | bfloat16 | 8 | one-tile reduce-scatter, n = 14 |  | 260.5 | 257.4 | 69.4 | 69.3 | 70.0 | 69.7 | torch | 434.6 | 437.6 | 437.1 | 437.9 | hipBLASLt padded 437.9 |
| 512x1536x7168 | bfloat16 | 12 | one-tile reduce-scatter, n = 10 |  | 342.1 | 349.0 | 103.9 | 103.6 | 104.7 | 104.2 | torch | 453.3 | 452.3 | 454.2 | 453.1 | hipBLASLt contiguous 454.2 |
| 512x2048x7168 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 |  | 417.0 | 417.5 | 138.0 | 138.2 | 139.0 | 138.6 | torch | 580.9 | 599.0 | 579.4 | 598.0 | AITER torch padded 599.0 |
| 512x3584x7168 | bfloat16 | 28 | one-tile reduce-scatter, n = 4 |  | 526.2 | 516.6 | 241.6 | 241.5 | 242.9 | 242.5 | flydsl | 615.9 | 407.8 (copy) | 632.1 | 633.2 | hipBLASLt padded 633.2 |
| 512x6144x7168 | bfloat16 | 48 | one-tile owner, n = 3 |  | 782.9 | 778.4 | 411.2 | 412.1 | 413.3 | 411.4 | flydsl | 822.7 | 463.8 (copy) | 821.0 | 813.3 | AITER flydsl contiguous 822.7 |
| 512x8192x7168 | bfloat16 | 64 | one-tile owner, n = 3 |  | 961.5 | 957.2 | 544.2 | 545.7 | 546.4 | 545.0 | flydsl | 875.0 | 441.4 (copy) | 899.6 | 896.7 | hipBLASLt contiguous 899.6 |
| 512x8448x7168 | bfloat16 | 66 | one-tile owner, n = 3-4 |  | 928.6 | 923.1 | 560.5 | 555.0 | 564.7 | 563.0 | flydsl | 748.2 | 431.9 (copy) | 468.2 | 463.2 | AITER flydsl contiguous 748.2 |
| 512x20480x7168 | bfloat16 | 160 | data-parallel | yes | 1103.0 | 1083.2 | 1098.3 | 1061.6 | 1104.7 | 1080.7 | flydsl | 1032.0 | 582.0 (copy) | 1153.6 | 1149.9 | hipBLASLt contiguous 1153.6 |
| 768x1024x7168 | bfloat16 | 12 | one-tile reduce-scatter, n = 10 |  | 346.1 | 354.4 | 103.5 | 103.9 | 104.9 | 104.6 | torch | 445.8 | 448.0 | 443.8 | 458.9 | hipBLASLt padded 458.9 |
| 768x2048x7168 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 |  | 503.3 | 503.4 | 207.9 | 207.2 | 208.8 | 208.1 | flydsl | 612.6 | 399.3 (copy) | 643.3 | 635.3 | hipBLASLt contiguous 643.3 |
| 1024x512x7168 | bfloat16 | 8 | one-tile reduce-scatter, n = 14 |  | 259.5 | 257.8 | 69.3 | 69.2 | 69.9 | 69.8 | flydsl | 381.8 | 233.2 (copy) | 324.1 | 344.8 | AITER flydsl contiguous 381.8 |
| 1024x1024x7168 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 |  | 416.2 | 415.4 | 137.8 | 137.5 | 139.1 | 138.3 | torch | 592.4 | 610.1 | 592.4 | 608.3 | AITER torch padded 610.1 |
| 1024x1536x7168 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 |  | 499.9 | 504.0 | 207.0 | 206.1 | 208.0 | 207.4 | flydsl | 629.9 | 433.1 (copy) | 635.6 | 628.7 | hipBLASLt contiguous 635.6 |
| 1024x2048x7168 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 576.7 | 572.0 | 276.6 | 274.6 | 276.8 | 276.1 | torch | 750.0 | 754.5 | 741.6 | 742.4 | AITER torch padded 754.5 |
| 1024x3584x7168 | bfloat16 | 56 | one-tile owner, n = 3 |  | 878.1 | 881.9 | 480.3 | 474.1 | 481.6 | 478.5 | flydsl | 872.3 | 621.7 (copy) | 830.6 | 829.4 | AITER flydsl contiguous 872.3 |
| 1024x6144x7168 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 1057.6 | 1029.7 | 803.8 | 807.6 | 811.7 | 809.1 | torch | 1024.2 | 1038.6 | 1027.8 | 1040.6 | hipBLASLt padded 1040.6 |
| 1024x8192x7168 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1293.0 | 1289.7 | 1065.2 | 1053.7 | 1059.0 | 1069.6 | flydsl | 1094.3 | 735.4 (copy) | 948.4 | 997.0 | AITER flydsl contiguous 1094.3 |
| 1024x8448x7168 | bfloat16 | 132 | data-parallel | yes | 1092.7 | 1098.5 | 1096.6 | 1094.2 | 1095.0 | 1096.3 | torch | 1185.8 | 1181.3 | 1193.0 | 1183.7 | hipBLASLt contiguous 1193.0 |
| 1024x20480x7168 | bfloat16 | 320 | one-tile owner, n = 3 |  | 1325.2 | 1334.5 | 1126.3 | 1115.3 | 1120.7 | 1111.5 | flydsl | 776.7 | 605.1 (copy) | 1205.5 | 1203.0 | hipBLASLt contiguous 1205.5 |
| 1280x512x7168 | bfloat16 | 10 | one-tile reduce-scatter, n = 12 |  | 308.9 | 302.8 | 86.6 | 86.2 | 87.6 | 87.3 | torch | 410.5 | 407.5 | 411.7 | 418.7 | hipBLASLt padded 418.7 |
| 1280x1024x7168 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 |  | 451.2 | 451.7 | 172.7 | 172.5 | 173.4 | 172.9 | torch | 599.8 | 590.0 | 595.6 | 587.6 | AITER torch contiguous 599.8 |
| 1280x2048x7168 | bfloat16 | 40 | one-tile owner, n = 4 |  | 668.3 | 673.0 | 340.4 | 341.3 | 345.0 | 344.0 | torch | 845.1 | 848.6 | 846.7 | 832.9 | AITER torch padded 848.6 |
| 1536x512x7168 | bfloat16 | 12 | one-tile reduce-scatter, n = 10 |  | 344.8 | 351.2 | 103.7 | 103.6 | 104.8 | 104.5 | torch | 447.2 | 449.6 | 447.0 | 459.1 | hipBLASLt padded 459.1 |
| 1536x1024x7168 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 |  | 503.3 | 503.3 | 208.0 | 207.4 | 208.9 | 208.0 | torch | 752.4 | 750.7 | 755.9 | 758.5 | hipBLASLt padded 758.5 |
| 1536x2048x7168 | bfloat16 | 48 | one-tile owner, n = 3 |  | 779.3 | 775.9 | 410.9 | 411.8 | 414.0 | 411.7 | torch | 963.4 | 969.7 | 968.8 | 966.2 | AITER torch padded 969.7 |
| 1792x512x7168 | bfloat16 | 14 | one-tile reduce-scatter, n = 9 |  | 378.8 | 376.3 | 121.2 | 121.0 | 122.2 | 121.8 | torch | 492.9 | 493.3 | 491.7 | 498.9 | hipBLASLt padded 498.9 |
| 1792x1024x7168 | bfloat16 | 28 | one-tile reduce-scatter, n = 4 |  | 523.2 | 521.6 | 241.6 | 241.2 | 243.0 | 242.1 | flydsl | 635.0 | 443.8 (copy) | 662.2 | 649.4 | hipBLASLt contiguous 662.2 |
| 1792x2048x7168 | bfloat16 | 56 | one-tile owner, n = 3 |  | 878.8 | 871.5 | 479.8 | 475.6 | 480.0 | 476.7 | flydsl | 907.3 | 667.5 (copy) | 814.2 | 816.4 | AITER flydsl contiguous 907.3 |
| 2048x512x7168 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 |  | 415.6 | 415.4 | 138.6 | 138.2 | 139.1 | 138.9 | torch | 587.2 | 592.9 | 588.1 | 593.9 | hipBLASLt padded 593.9 |
| 2048x1024x7168 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 578.6 | 573.4 | 276.1 | 276.5 | 277.7 | 276.6 | torch | 808.9 | 816.8 | 812.4 | 816.4 | AITER torch padded 816.8 |
| 2048x2048x7168 | bfloat16 | 64 | one-tile owner, n = 3 |  | 959.9 | 955.5 | 544.8 | 545.4 | 546.5 | 545.4 | flydsl | 991.2 | 726.3 (copy) | 899.8 | 902.0 | AITER flydsl contiguous 991.2 |
| 2048x3584x7168 | bfloat16 | 112 | one-tile owner, n = 2-3 |  | 1144.9 | 1140.4 | 936.9 | 933.6 | 946.0 | 941.0 | opus | 1217.1 | 1229.7 | 1080.5 | 1059.5 | AITER opus padded 1229.7 |
| 2048x8192x7168 | bfloat16 | 256 | data-parallel | yes | 1584.0 | 1583.0 | 1577.0 | 1574.7 | 1584.6 | 1584.8 | asm | 1555.3 | 1559.4 | 1543.7 | 1546.0 | AITER asm padded 1559.4 |
| 2048x20480x7168 | bfloat16 | 640 | one-tile owner, n = 2 |  | 1484.7 | 1492.4 | 1398.6 | 1402.3 | 1389.6 | 1391.9 | flydsl | 820.5 | 709.7 (copy) | 1279.0 | 1295.2 | hipBLASLt padded 1295.2 |
| 4096x512x7168 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 573.2 | 568.8 | 275.5 | 276.0 | 277.4 | 276.4 | flydsl | 711.6 | 453.7 (copy) | 686.0 | 682.7 | AITER flydsl contiguous 711.6 |
| 4096x1024x7168 | bfloat16 | 64 | one-tile owner, n = 3 |  | 954.8 | 956.8 | 545.3 | 544.5 | 546.2 | 545.5 | torch | 976.4 | 962.6 | 975.7 | 961.3 | AITER torch contiguous 976.4 |
| 4096x2048x7168 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1291.9 | 1293.4 | 1055.0 | 1061.8 | 1065.8 | 1070.7 | torch | 1297.2 | 1299.6 | 1306.2 | 1300.1 | hipBLASLt contiguous 1306.2 |
| 4096x3584x7168 | bfloat16 | 224 | data-parallel | yes | 1460.8 | 1466.2 | 1453.4 | 1457.7 | 1464.7 | 1463.3 | flydsl | 1401.4 | 1071.6 (copy) | 1396.6 | 1393.9 | AITER flydsl contiguous 1401.4 |
| 4096x8192x7168 | bfloat16 | 512 | data-parallel |  | 1593.7 | 1598.7 | 1593.7 | 1598.7 | 1599.9 | 1599.3 | flydsl | 1552.5 | 1262.9 (copy) | 1565.6 | 1569.1 | hipBLASLt padded 1569.1 |
| 4096x20480x7168 | bfloat16 | 1280 | data-parallel |  | 1575.1 | 1585.6 | 1575.1 | 1585.6 | 1565.8 | 1572.4 | flydsl | 1511.8 | 1313.3 (copy) | 1490.6 | 1495.8 | AITER flydsl contiguous 1511.8 |
| 6144x512x7168 | bfloat16 | 48 | one-tile owner, n = 3 |  | 773.2 | 778.9 | 409.4 | 410.6 | 413.9 | 413.0 | flydsl | 801.1 | 468.4 (copy) | 728.9 | 740.1 | AITER flydsl contiguous 801.1 |
| 6144x1024x7168 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 1057.7 | 1028.1 | 798.6 | 807.7 | 814.8 | 809.7 | opus | 1187.7 | 1200.7 | 886.1 | 922.5 | AITER opus padded 1200.7 |
| 6144x2048x7168 | bfloat16 | 192 | data-parallel | yes | 1357.4 | 1355.1 | 1348.8 | 1349.9 | 1346.9 | 1350.3 | opus | 1470.4 | 1480.9 | 1368.7 | 1368.6 | AITER opus padded 1480.9 |
| 8192x512x7168 | bfloat16 | 64 | one-tile owner, n = 3 |  | 963.0 | 957.7 | 545.8 | 540.1 | 546.6 | 544.4 | flydsl | 970.4 | 492.4 (copy) | 898.8 | 898.7 | AITER flydsl contiguous 970.4 |
| 8192x1024x7168 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1295.0 | 1290.6 | 1061.6 | 1059.8 | 1068.7 | 1069.0 | torch | 1150.9 | 1140.5 | 1160.7 | 1147.0 | hipBLASLt contiguous 1160.7 |
| 8192x2048x7168 | bfloat16 | 256 | data-parallel | yes | 1580.6 | 1582.2 | 1574.6 | 1576.6 | 1583.9 | 1582.1 | torch | 1444.4 | 1461.8 | 1452.6 | 1458.0 | AITER torch padded 1461.8 |
| 8192x3584x7168 | bfloat16 | 448 | data-parallel |  | 1482.3 | 1486.0 | 1482.3 | 1486.0 | 1490.6 | 1490.6 | flydsl | 1449.3 | 1173.5 (copy) | 1396.0 | 1396.6 | AITER flydsl contiguous 1449.3 |
| 8192x8192x7168 | bfloat16 | 1024 | data-parallel |  | 1588.1 | 1594.7 | 1588.1 | 1594.7 | 1581.3 | 1585.8 | flydsl | 1529.0 | 1346.3 (copy) | 1586.0 | 1578.7 | hipBLASLt contiguous 1586.0 |
| 8192x20480x7168 | bfloat16 | 2560 | data-parallel |  | 1596.3 | 1602.7 | 1596.3 | 1602.7 | 1591.4 | 1590.5 | asm | 1529.6 | 1546.3 | 1499.8 | 1503.9 | AITER asm padded 1546.3 |
| 10240x512x7168 | bfloat16 | 80 | one-tile owner, n = 3-4 |  | 1023.8 | 998.1 | 676.9 | 674.6 | 677.2 | 677.3 | torch | 832.9 | 848.4 | 834.4 | 850.1 | hipBLASLt padded 850.1 |
| 10240x1024x7168 | bfloat16 | 160 | data-parallel | yes | 1240.2 | 1241.8 | 1227.4 | 1229.1 | 1241.4 | 1238.8 | torch | 1308.6 | 1311.8 | 1315.1 | 1314.6 | hipBLASLt contiguous 1315.1 |
| 10240x2048x7168 | bfloat16 | 320 | one-tile owner, n = 3 |  | 1411.8 | 1412.6 | 1176.3 | 1175.2 | 1181.3 | 1182.7 | torch | 1324.7 | 1329.8 | 1328.3 | 1333.4 | hipBLASLt padded 1333.4 |
| 10240x8448x7168 | bfloat16 | 1320 | one-tile owner, n = 4 |  | 1501.7 | 1512.1 | 1426.2 | 1435.0 | 1418.6 | 1424.7 | asm | 1408.1 | 1415.7 | 1364.0 | 1379.5 | AITER asm padded 1415.7 |
| 10240x20480x7168 | bfloat16 | 3200 | one-tile owner, n = 2 |  | 1513.9 | 1531.4 | 1501.0 | 1509.2 | 1480.7 | 1492.0 | asm | 1498.6 | 1515.3 | 1465.8 | 1474.1 | AITER asm padded 1515.3 |
| 12288x512x7168 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 1055.0 | 1031.6 | 802.0 | 800.9 | 810.9 | 808.1 | opus | 1177.5 | 1193.8 | 869.2 | 878.7 | AITER opus padded 1193.8 |
| 12288x1024x7168 | bfloat16 | 192 | data-parallel | yes | 1354.3 | 1356.0 | 1344.1 | 1350.4 | 1353.3 | 1353.9 | opus | 1475.1 | 1481.3 | 1366.0 | 1367.5 | AITER opus padded 1481.3 |
| 12288x2048x7168 | bfloat16 | 384 | one-tile owner, n = 2 |  | 1477.8 | 1475.7 | 1348.3 | 1350.8 | 1343.0 | 1352.7 | opus | 1474.0 | 1477.6 | 1381.9 | 1379.9 | AITER opus padded 1477.6 |
| 14336x512x7168 | bfloat16 | 112 | one-tile owner, n = 2-3 |  | 1094.0 | 1078.1 | 928.0 | 941.4 | 945.1 | 939.0 | opus | 1254.5 | 1261.4 | 912.5 | 917.1 | AITER opus padded 1261.4 |
| 14336x1024x7168 | bfloat16 | 224 | data-parallel | yes | 1457.2 | 1455.5 | 1446.4 | 1435.9 | 1458.2 | 1458.1 | torch | 1408.2 | 1407.0 | 1415.3 | 1415.1 | hipBLASLt contiguous 1415.3 |
| 14336x2048x7168 | bfloat16 | 448 | data-parallel |  | 1462.8 | 1452.9 | 1462.8 | 1452.9 | 1443.2 | 1460.6 | flydsl | 1409.8 | 1073.9 (copy) | 1395.4 | 1390.1 | AITER flydsl contiguous 1409.8 |
| 16384x512x7168 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1183.7 | 1173.8 | 1052.6 | 1012.8 | 1063.0 | 1030.9 | opus | 1329.4 | 1204.7 | 1042.8 | 1016.4 | AITER opus contiguous 1329.4 |
| 16384x1024x7168 | bfloat16 | 256 | data-parallel | yes | 1514.7 | 1519.1 | 1506.8 | 1514.6 | 1521.8 | 1524.4 | torch | 1525.5 | 1528.1 | 1530.7 | 1525.0 | hipBLASLt contiguous 1530.7 |
| 16384x2048x7168 | bfloat16 | 512 | data-parallel |  | 1526.5 | 1543.3 | 1526.5 | 1543.3 | 1526.7 | 1541.3 | torch | 1538.3 | 1544.8 | 1561.0 | 1559.7 | hipBLASLt contiguous 1561.0 |
| 16384x3584x7168 | bfloat16 | 896 | one-tile owner, n = 2 |  | 1511.4 | 1520.8 | 1448.7 | 1453.1 | 1439.5 | 1447.4 | flydsl | 1389.1 | 1194.9 (copy) | 1392.9 | 1392.6 | hipBLASLt contiguous 1392.9 |
| 16384x8192x7168 | bfloat16 | 2048 | data-parallel |  | 1575.7 | 1588.6 | 1575.7 | 1588.6 | 1570.3 | 1574.3 | torch | 1544.3 | 1579.2 | 1550.3 | 1576.2 | AITER torch padded 1579.2 |
| 16384x20480x7168 | bfloat16 | 5120 | data-parallel |  | 1592.0 | 1599.9 | 1592.0 | 1599.9 | 1581.6 | 1585.1 | flydsl | 1529.6 | 1453.4 (copy) | 1509.2 | 1518.1 | AITER flydsl contiguous 1529.6 |
| 32768x512x7168 | bfloat16 | 256 | data-parallel | yes | 1390.7 | 1362.5 | 1378.4 | 1393.3 | 1380.8 | 1366.4 | torch | 1383.8 | 1391.2 | 1389.1 | 1388.5 | AITER torch padded 1391.2 |
| 32768x1024x7168 | bfloat16 | 512 | data-parallel |  | 1492.2 | 1504.3 | 1492.2 | 1504.3 | 1494.9 | 1504.6 | torch | 1441.0 | 1469.4 | 1461.9 | 1449.7 | AITER torch padded 1469.4 |
| 32768x2048x7168 | bfloat16 | 1024 | data-parallel |  | 1544.7 | 1559.2 | 1544.7 | 1559.2 | 1537.0 | 1543.9 | torch | 1486.8 | 1523.9 | 1507.3 | 1524.8 | hipBLASLt padded 1524.8 |
| 32768x3584x7168 | bfloat16 | 1792 | data-parallel |  | 1551.7 | 1562.1 | 1551.7 | 1562.1 | 1539.1 | 1545.5 | torch | 1519.6 | 1546.8 | 1533.3 | 1553.6 | hipBLASLt padded 1553.6 |
| 32768x8192x7168 | bfloat16 | 4096 | data-parallel |  | 1580.7 | 1593.7 | 1580.7 | 1593.7 | 1574.8 | 1576.8 | torch | 1559.2 | 1588.0 | 1569.3 | 1586.2 | AITER torch padded 1588.0 |
| 32768x20480x7168 | bfloat16 | 10240 | data-parallel |  | 1595.0 | 1597.5 | 1595.0 | 1597.5 | 1588.2 | 1589.3 | asm | 1551.6 | 1556.8 | 1497.5 | 1487.0 | AITER asm padded 1556.8 |
| 512x1280x8192 | bfloat16 | 10 | one-tile reduce-scatter, n = 12 |  | 325.8 | 329.3 | 87.6 | 87.4 | 88.3 | 87.9 | torch | 471.3 | 466.5 | 471.0 | 464.7 | AITER torch contiguous 471.3 |
| 512x2560x8192 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 |  | 502.1 | 492.6 | 175.1 | 174.7 | 175.2 | 174.6 | torch | 607.3 | 603.7 | 606.2 | 606.8 | AITER torch contiguous 607.3 |
| 512x4096x8192 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 619.6 | 608.9 | 278.0 | 277.8 | 279.2 | 278.3 | flydsl | 763.1 | 475.9 (copy) | 772.2 | 710.4 | hipBLASLt contiguous 772.2 |
| 512x4608x8192 | bfloat16 | 36 | one-tile owner, n = 4 |  | 662.4 | 662.4 | 311.5 | 311.6 | 312.6 | 311.9 | torch | 853.5 | 830.0 | 851.3 | 828.1 | AITER torch contiguous 853.5 |
| 512x7168x8192 | bfloat16 | 56 | one-tile owner, n = 3 |  | 887.6 | 884.6 | 478.1 | 479.2 | 485.1 | 482.1 | flydsl | 919.9 | 478.0 (copy) | 875.8 | 870.9 | AITER flydsl contiguous 919.9 |
| 512x8192x8192 | bfloat16 | 64 | one-tile owner, n = 3 |  | 988.9 | 978.8 | 549.2 | 550.4 | 551.5 | 549.9 | flydsl | 879.4 | 447.4 (copy) | 904.2 | 898.1 | hipBLASLt contiguous 904.2 |
| 512x10240x8192 | bfloat16 | 80 | one-tile owner, n = 3-4 |  | 1079.0 | 1069.8 | 683.1 | 682.1 | 689.9 | 689.0 | flydsl | 690.1 | 416.4 (copy) | 569.6 | 557.3 | AITER flydsl contiguous 690.1 |
| 512x14336x8192 | bfloat16 | 112 | one-tile owner, n = 2-3 |  | 1108.1 | 1105.3 | 938.9 | 953.1 | 951.6 | 953.2 | torch | 1045.1 | 1029.0 | 1050.7 | 1023.0 | hipBLASLt contiguous 1050.7 |
| 512x16384x8192 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1193.5 | 1204.0 | 981.7 | 961.7 | 997.0 | 957.1 | flydsl | 681.5 | 441.0 (copy) | 917.4 | 983.7 | hipBLASLt padded 983.7 |
| 512x57344x8192 | bfloat16 | 448 | data-parallel |  | 1337.7 | 1353.6 | 1337.7 | 1353.6 | 1333.5 | 1352.2 | flydsl | 1291.7 | 687.7 (copy) | 1266.9 | 1283.9 | AITER flydsl contiguous 1291.7 |
| 768x4096x8192 | bfloat16 | 48 | one-tile owner, n = 4 |  | 828.9 | 825.6 | 414.5 | 413.2 | 417.0 | 415.7 | flydsl | 842.0 | 569.3 (copy) | 801.8 | 789.7 | AITER flydsl contiguous 842.0 |
| 1024x512x8192 | bfloat16 | 8 | one-tile reduce-scatter, n = 16 |  | 298.2 | 299.4 | 70.2 | 70.0 | 70.8 | 70.5 | torch | 367.4 | 371.1 | 367.4 | 388.1 | hipBLASLt padded 388.1 |
| 1024x1280x8192 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 |  | 500.8 | 491.8 | 174.9 | 174.5 | 175.3 | 174.8 | torch | 622.2 | 613.4 | 621.5 | 617.6 | AITER torch contiguous 622.2 |
| 1024x2560x8192 | bfloat16 | 40 | one-tile owner, n = 4 |  | 719.1 | 719.0 | 344.3 | 345.4 | 348.4 | 347.4 | torch | 865.7 | 882.7 | 868.3 | 877.2 | AITER torch padded 882.7 |
| 1024x4096x8192 | bfloat16 | 64 | one-tile owner, n = 3 |  | 994.2 | 971.4 | 547.1 | 547.2 | 551.0 | 549.4 | flydsl | 995.1 | 684.4 (copy) | 910.9 | 898.3 | AITER flydsl contiguous 995.1 |
| 1024x4608x8192 | bfloat16 | 72 | one-tile owner, n = 3-4 |  | 1038.6 | 1033.5 | 612.7 | 611.1 | 620.6 | 619.9 | torch | 992.1 | 990.5 | 994.2 | 988.5 | hipBLASLt contiguous 994.2 |
| 1024x7168x8192 | bfloat16 | 112 | one-tile owner, n = 2-3 |  | 1230.6 | 1180.3 | 946.9 | 953.2 | 954.9 | 954.5 | flydsl | 839.4 | 579.3 (copy) | 1124.9 | 1110.3 | hipBLASLt contiguous 1124.9 |
| 1024x8192x8192 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1339.0 | 1338.5 | 1075.1 | 1083.6 | 1071.4 | 1082.2 | flydsl | 1160.2 | 664.0 (copy) | 1082.0 | 1078.8 | AITER flydsl contiguous 1160.2 |
| 1024x10240x8192 | bfloat16 | 160 | data-parallel | yes | 1264.1 | 1265.4 | 1262.2 | 1253.7 | 1269.5 | 1266.3 | flydsl | 873.1 | 629.8 (copy) | 1307.7 | 1308.1 | hipBLASLt padded 1308.1 |
| 1024x14336x8192 | bfloat16 | 224 | data-parallel | yes | 1438.2 | 1422.8 | 1415.4 | 1411.0 | 1430.5 | 1420.4 | flydsl | 1419.4 | 889.6 (copy) | 1356.1 | 1351.8 | AITER flydsl contiguous 1419.4 |
| 1024x16384x8192 | bfloat16 | 256 | data-parallel | yes | 1497.5 | 1508.6 | 1492.1 | 1496.2 | 1499.8 | 1511.3 | torch | 1489.7 | 1490.3 | 1498.6 | 1486.1 | hipBLASLt contiguous 1498.6 |
| 1024x57344x8192 | bfloat16 | 896 | one-tile owner, n = 2 |  | 1498.3 | 1497.7 | 1429.0 | 1429.5 | 1421.9 | 1421.4 | asm | 1377.1 | 1391.2 | 1329.3 | 1337.3 | AITER asm padded 1391.2 |
| 2048x512x8192 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 |  | 453.2 | 450.4 | 139.9 | 139.7 | 140.5 | 140.2 | torch | 526.3 | 542.2 | 535.3 | 552.1 | hipBLASLt padded 552.1 |
| 2048x1280x8192 | bfloat16 | 40 | one-tile owner, n = 4 |  | 719.4 | 715.3 | 347.7 | 346.1 | 348.4 | 347.4 | torch | 725.9 | 725.8 | 726.7 | 731.0 | hipBLASLt padded 731.0 |
| 2048x2560x8192 | bfloat16 | 80 | one-tile owner, n = 3-4 |  | 1079.7 | 1068.7 | 682.5 | 682.4 | 686.4 | 685.8 | torch | 1011.6 | 1034.7 | 1023.1 | 1038.1 | hipBLASLt padded 1038.1 |
| 2048x4096x8192 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1335.5 | 1337.6 | 1073.3 | 1071.6 | 1080.1 | 1077.5 | opus | 1298.8 | 1306.0 | 1056.6 | 1051.4 | AITER opus padded 1306.0 |
| 2048x4608x8192 | bfloat16 | 144 | data-parallel | yes | 1189.4 | 1186.5 | 1188.8 | 1181.4 | 1189.9 | 1187.7 | torch | 1258.4 | 1269.2 | 1267.5 | 1269.9 | hipBLASLt padded 1269.9 |
| 2048x7168x8192 | bfloat16 | 224 | data-parallel | yes | 1483.3 | 1478.2 | 1476.5 | 1473.8 | 1487.4 | 1483.2 | asm | 1455.3 | 1456.4 | 1410.9 | 1407.2 | AITER asm padded 1456.4 |
| 2048x8192x8192 | bfloat16 | 256 | data-parallel | yes | 1602.8 | 1598.7 | 1596.6 | 1592.9 | 1607.2 | 1600.5 | torch | 1555.1 | 1570.6 | 1572.6 | 1558.9 | hipBLASLt contiguous 1572.6 |
| 2048x10240x8192 | bfloat16 | 320 | one-tile owner, n = 3 |  | 1420.1 | 1423.8 | 1190.7 | 1189.4 | 1184.9 | 1193.5 | flydsl | 991.0 | 773.0 (copy) | 1317.1 | 1318.6 | hipBLASLt padded 1318.6 |
| 2048x14336x8192 | bfloat16 | 448 | data-parallel |  | 1445.6 | 1450.4 | 1445.6 | 1450.4 | 1444.1 | 1449.2 | flydsl | 1407.7 | 1097.8 (copy) | 1348.1 | 1353.3 | AITER flydsl contiguous 1407.7 |
| 2048x16384x8192 | bfloat16 | 512 | data-parallel |  | 1547.5 | 1559.2 | 1547.5 | 1559.2 | 1553.3 | 1555.2 | asm | 1491.7 | 1500.6 | 1554.8 | 1550.5 | hipBLASLt contiguous 1554.8 |
| 2048x57344x8192 | bfloat16 | 1792 | data-parallel |  | 1577.5 | 1585.4 | 1577.5 | 1585.4 | 1569.5 | 1572.3 | flydsl | 1514.9 | 1206.4 (copy) | 1484.8 | 1485.3 | AITER flydsl contiguous 1514.9 |
| 4096x512x8192 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 617.3 | 611.7 | 278.5 | 278.8 | 279.4 | 279.4 | flydsl | 747.3 | 468.4 (copy) | 761.8 | 717.3 | hipBLASLt contiguous 761.8 |
| 4096x1280x8192 | bfloat16 | 80 | one-tile owner, n = 3-4 |  | 1077.8 | 1069.4 | 675.9 | 678.9 | 687.2 | 686.4 | torch | 1027.8 | 1010.5 | 1034.1 | 1010.6 | hipBLASLt contiguous 1034.1 |
| 4096x2560x8192 | bfloat16 | 160 | data-parallel | yes | 1263.9 | 1265.3 | 1260.6 | 1257.4 | 1268.5 | 1266.3 | torch | 1301.0 | 1306.3 | 1308.5 | 1307.9 | hipBLASLt contiguous 1308.5 |
| 4096x4096x8192 | bfloat16 | 256 | data-parallel | yes | 1600.9 | 1598.1 | 1594.4 | 1593.7 | 1607.6 | 1600.3 | torch | 1527.4 | 1541.8 | 1554.0 | 1531.2 | hipBLASLt contiguous 1554.0 |
| 4096x4608x8192 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 |  | 1395.9 | 1397.3 | 1084.5 | 1092.4 | 1085.9 | 1085.7 | torch | 1265.7 | 1276.3 | 1272.3 | 1279.3 | hipBLASLt padded 1279.3 |
| 4096x7168x8192 | bfloat16 | 448 | data-parallel |  | 1495.4 | 1497.9 | 1495.4 | 1497.9 | 1497.8 | 1501.1 | flydsl | 1456.0 | 1195.1 (copy) | 1419.3 | 1417.3 | AITER flydsl contiguous 1456.0 |
| 4096x8192x8192 | bfloat16 | 512 | data-parallel |  | 1611.0 | 1609.8 | 1611.0 | 1609.8 | 1614.0 | 1609.6 | asm | 1591.3 | 1579.9 | 1586.7 | 1576.6 | AITER asm contiguous 1591.3 |
| 4096x10240x8192 | bfloat16 | 640 | one-tile owner, n = 2 |  | 1529.8 | 1531.7 | 1433.8 | 1431.0 | 1429.7 | 1427.6 | flydsl | 1397.0 | 1186.8 (copy) | 1363.5 | 1358.9 | AITER flydsl contiguous 1397.0 |
| 4096x14336x8192 | bfloat16 | 896 | one-tile owner, n = 2 |  | 1548.2 | 1553.0 | 1474.1 | 1471.9 | 1467.3 | 1469.1 | flydsl | 1421.4 | 1229.1 (copy) | 1368.3 | 1373.6 | AITER flydsl contiguous 1421.4 |
| 4096x16384x8192 | bfloat16 | 1024 | data-parallel |  | 1582.2 | 1589.3 | 1582.2 | 1589.3 | 1575.9 | 1583.4 | flydsl | 1518.8 | 1313.7 (copy) | 1546.0 | 1542.6 | hipBLASLt contiguous 1546.0 |
| 4096x57344x8192 | bfloat16 | 3584 | data-parallel |  | 1599.2 | 1604.4 | 1599.2 | 1604.4 | 1589.0 | 1593.2 | flydsl | 1530.9 | 1354.1 (copy) | 1506.4 | 1512.6 | AITER flydsl contiguous 1530.9 |
| 8192x512x8192 | bfloat16 | 64 | one-tile owner, n = 3 |  | 1009.1 | 1006.0 | 549.5 | 549.7 | 551.9 | 551.5 | flydsl | 979.4 | 507.5 (copy) | 913.0 | 900.1 | AITER flydsl contiguous 979.4 |
| 8192x1280x8192 | bfloat16 | 160 | data-parallel | yes | 1268.0 | 1266.1 | 1258.5 | 1254.5 | 1269.0 | 1266.3 | torch | 1303.5 | 1305.4 | 1309.9 | 1308.6 | hipBLASLt contiguous 1309.9 |
| 8192x2560x8192 | bfloat16 | 320 | one-tile owner, n = 3 |  | 1445.2 | 1447.9 | 1184.6 | 1177.0 | 1187.5 | 1186.7 | torch | 1302.8 | 1310.5 | 1309.5 | 1314.7 | hipBLASLt padded 1314.7 |
| 8192x4096x8192 | bfloat16 | 512 | data-parallel |  | 1611.8 | 1608.9 | 1611.8 | 1608.9 | 1611.3 | 1609.5 | torch | 1588.4 | 1580.5 | 1582.2 | 1577.0 | AITER torch contiguous 1588.4 |
| 8192x4608x8192 | bfloat16 | 576 | one-tile owner, n = 3 |  | 1507.0 | 1507.8 | 1357.1 | 1348.6 | 1357.1 | 1354.3 | torch | 1374.4 | 1381.3 | 1380.9 | 1381.9 | hipBLASLt padded 1381.9 |
| 8192x7168x8192 | bfloat16 | 896 | one-tile owner, n = 2 |  | 1563.4 | 1565.7 | 1489.5 | 1485.9 | 1484.9 | 1480.5 | asm | 1437.8 | 1441.2 | 1419.5 | 1414.6 | AITER asm padded 1441.2 |
| 8192x8192x8192 | bfloat16 | 1024 | data-parallel |  | 1603.4 | 1605.6 | 1603.4 | 1605.6 | 1603.0 | 1603.3 | asm | 1574.8 | 1579.0 | 1581.1 | 1575.6 | hipBLASLt contiguous 1581.1 |
| 8192x10240x8192 | bfloat16 | 1280 | data-parallel |  | 1603.2 | 1608.0 | 1603.2 | 1608.0 | 1601.1 | 1602.8 | flydsl | 1548.5 | 1381.8 (copy) | 1592.3 | 1590.2 | hipBLASLt contiguous 1592.3 |
| 8192x14336x8192 | bfloat16 | 1792 | data-parallel |  | 1601.8 | 1610.2 | 1601.8 | 1610.2 | 1603.1 | 1604.6 | flydsl | 1552.6 | 1405.1 (copy) | 1567.4 | 1572.0 | hipBLASLt padded 1572.0 |
| 8192x16384x8192 | bfloat16 | 2048 | data-parallel |  | 1597.1 | 1610.2 | 1597.1 | 1610.2 | 1600.8 | 1603.4 | asm | 1547.6 | 1544.9 | 1537.9 | 1539.3 | AITER asm contiguous 1547.6 |
| 8192x57344x8192 | bfloat16 | 7168 | data-parallel |  | 1592.2 | 1585.1 | 1592.2 | 1585.1 | 1609.9 | 1603.7 | asm | 1549.7 | 1550.7 | 1412.0 | 1424.0 | AITER asm padded 1550.7 |
| 16384x512x8192 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1192.9 | 1203.7 | 977.5 | 956.9 | 990.9 | 963.0 | torch | 1085.4 | 1065.0 | 1092.5 | 1069.2 | hipBLASLt contiguous 1092.5 |
| 16384x1280x8192 | bfloat16 | 320 | one-tile owner, n = 3 |  | 1387.5 | 1398.0 | 1167.5 | 1172.6 | 1179.7 | 1171.2 | torch | 1258.5 | 1272.0 | 1266.1 | 1273.0 | hipBLASLt padded 1273.0 |
| 16384x2560x8192 | bfloat16 | 640 | one-tile owner, n = 2 |  | 1489.3 | 1498.6 | 1395.6 | 1397.2 | 1388.8 | 1393.4 | flydsl | 1345.7 | 1117.4 (copy) | 1327.5 | 1325.7 | AITER flydsl contiguous 1345.7 |
| 16384x4608x8192 | bfloat16 | 1152 | one-tile owner, n = 2 |  | 1533.3 | 1546.3 | 1481.1 | 1482.0 | 1466.7 | 1472.1 | flydsl | 1427.9 | 1266.2 (copy) | 1218.9 | 1205.4 | AITER flydsl contiguous 1427.9 |
| 16384x7168x8192 | bfloat16 | 1792 | data-parallel |  | 1577.9 | 1589.5 | 1577.9 | 1589.5 | 1576.5 | 1577.0 | torch | 1562.2 | 1584.0 | 1586.0 | 1586.0 | hipBLASLt contiguous 1586.0 |
| 16384x8192x8192 | bfloat16 | 2048 | data-parallel |  | 1588.8 | 1597.3 | 1588.8 | 1597.3 | 1589.0 | 1589.2 | torch | 1569.3 | 1594.0 | 1596.7 | 1590.7 | hipBLASLt contiguous 1596.7 |
| 16384x10240x8192 | bfloat16 | 2560 | data-parallel |  | 1598.9 | 1606.2 | 1598.9 | 1606.2 | 1595.3 | 1595.5 | torch | 1584.7 | 1599.8 | 1604.9 | 1592.9 | hipBLASLt contiguous 1604.9 |
| 16384x14336x8192 | bfloat16 | 3584 | data-parallel |  | 1600.3 | 1609.4 | 1600.3 | 1609.4 | 1600.3 | 1599.9 | asm | 1555.2 | 1565.2 | 1562.7 | 1569.7 | hipBLASLt padded 1569.7 |
| 16384x16384x8192 | bfloat16 | 4096 | data-parallel |  | 1598.3 | 1606.5 | 1598.3 | 1606.5 | 1600.1 | 1597.2 | asm | 1555.1 | 1550.9 | 1545.2 | 1542.1 | AITER asm contiguous 1555.1 |
| 16384x57344x8192 | bfloat16 | 14336 | data-parallel |  | 1587.7 | 1592.8 | 1587.7 | 1592.8 | 1604.2 | 1597.5 | asm | 1557.0 | 1556.2 | 1453.8 | 1433.8 | AITER asm contiguous 1557.0 |
| 32768x1280x8192 | bfloat16 | 640 | one-tile owner, n = 2 |  | 1451.5 | 1461.3 | 1365.7 | 1371.6 | 1355.7 | 1363.3 | flydsl | 1317.5 | 964.3 (copy) | 1338.3 | 1339.8 | hipBLASLt padded 1339.8 |
| 32768x2560x8192 | bfloat16 | 1280 | data-parallel |  | 1538.9 | 1547.2 | 1538.9 | 1547.2 | 1534.6 | 1538.5 | torch | 1531.2 | 1556.4 | 1553.5 | 1555.9 | AITER torch padded 1556.4 |
| 32768x7168x8192 | bfloat16 | 3584 | data-parallel |  | 1587.6 | 1592.6 | 1587.6 | 1592.6 | 1584.6 | 1578.9 | torch | 1582.2 | 1591.9 | 1599.1 | 1595.1 | hipBLASLt contiguous 1599.1 |
| 32768x8192x8192 | bfloat16 | 4096 | data-parallel |  | 1597.5 | 1596.5 | 1597.5 | 1596.5 | 1592.8 | 1585.1 | flydsl | 1553.9 | 1442.3 (copy) | 1598.0 | 1592.8 | hipBLASLt contiguous 1598.0 |
| 32768x10240x8192 | bfloat16 | 5120 | data-parallel |  | 1598.2 | 1595.0 | 1598.2 | 1595.0 | 1597.1 | 1583.9 | torch | 1586.7 | 1595.2 | 1603.7 | 1591.5 | hipBLASLt contiguous 1603.7 |
| 32768x14336x8192 | bfloat16 | 7168 | data-parallel |  | 1593.2 | 1590.1 | 1593.2 | 1590.1 | 1600.8 | 1587.1 | asm | 1553.8 | 1564.2 | 1525.1 | 1539.0 | AITER asm padded 1564.2 |
| 32768x16384x8192 | bfloat16 | 8192 | data-parallel |  | 1584.9 | 1580.4 | 1584.9 | 1580.4 | 1602.0 | 1581.2 | asm | 1561.3 | 1551.9 | 1468.4 | 1471.4 | AITER asm contiguous 1561.3 |
| 32768x57344x8192 | bfloat16 | 28672 | data-parallel |  | 1561.9 | 1529.0 | 1561.9 | 1529.0 | 1587.5 | 1579.5 | asm | 1556.7 | 1546.1 | 1425.2 | 1429.3 | AITER asm contiguous 1556.7 |
| 512x6144x12288 | bfloat16 | 48 | one-tile owner, n = 4 |  | 964.6 | 989.4 | 422.0 | 422.8 | 424.7 | 423.2 | torch | 932.4 | 936.2 | 930.7 | 939.4 | hipBLASLt padded 939.4 |
| 1024x6144x12288 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 1184.4 | 1198.6 | 837.2 | 829.9 | 836.8 | 835.9 | torch | 1189.0 | 1200.7 | 1193.2 | 1201.4 | hipBLASLt padded 1201.4 |
| 2048x6144x12288 | bfloat16 | 192 | data-parallel | yes | 1399.3 | 1400.6 | 1391.2 | 1401.6 | 1399.6 | 1398.7 | torch | 1392.9 | 1392.6 | 1395.0 | 1391.8 | hipBLASLt contiguous 1395.0 |
| 4096x6144x12288 | bfloat16 | 384 | one-tile owner, n = 2 |  | 1537.1 | 1543.4 | 1369.4 | 1365.7 | 1372.8 | 1371.5 | torch | 1387.8 | 1393.2 | 1396.2 | 1396.2 | hipBLASLt contiguous 1396.2 |
| 8192x6144x12288 | bfloat16 | 768 | data-parallel |  | 1587.8 | 1602.2 | 1587.8 | 1602.2 | 1583.6 | 1596.0 | torch | 1566.4 | 1608.4 | 1577.1 | 1608.3 | AITER torch padded 1608.4 |
| 16384x6144x12288 | bfloat16 | 1536 | data-parallel |  | 1574.0 | 1594.9 | 1574.0 | 1594.9 | 1570.1 | 1586.3 | torch | 1579.8 | 1611.0 | 1590.0 | 1609.9 | AITER torch padded 1611.0 |
| 32768x6144x12288 | bfloat16 | 3072 | data-parallel |  | 1577.6 | 1597.7 | 1577.6 | 1597.7 | 1573.3 | 1589.1 | asm | 1567.2 | 1570.2 | 1593.7 | 1605.7 | hipBLASLt padded 1605.7 |
| 512x16384x13312 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1284.2 | 1275.3 | 924.1 | 910.5 | 927.8 | 920.6 | flydsl | 714.4 | 469.3 (copy) | 1105.0 | 1096.1 | hipBLASLt contiguous 1105.0 |
| 1024x16384x13312 | bfloat16 | 256 | data-parallel | yes | 1535.3 | 1535.7 | 1531.7 | 1531.4 | 1536.2 | 1536.1 | asm | 1487.9 | 1496.1 | 1495.2 | 1507.7 | hipBLASLt padded 1507.7 |
| 2048x16384x13312 | bfloat16 | 512 | data-parallel |  | 1583.6 | 1588.5 | 1583.6 | 1588.5 | 1579.3 | 1584.0 | flydsl | 1015.0 | 848.3 (copy) | 1512.4 | 1525.5 | hipBLASLt padded 1525.5 |
| 4096x16384x13312 | bfloat16 | 1024 | data-parallel |  | 1614.6 | 1619.8 | 1614.6 | 1619.8 | 1606.8 | 1613.5 | flydsl | 1552.3 | 1336.4 (copy) | 1522.9 | 1539.2 | AITER flydsl contiguous 1552.3 |
| 8192x16384x13312 | bfloat16 | 2048 | data-parallel |  | 1589.3 | 1600.6 | 1589.3 | 1600.6 | 1582.3 | 1590.8 | flydsl | 1522.4 | 1393.7 (copy) | 1522.1 | 1539.1 | hipBLASLt padded 1539.1 |
| 16384x16384x13312 | bfloat16 | 4096 | data-parallel |  | 1580.5 | 1592.4 | 1580.5 | 1592.4 | 1583.0 | 1584.1 | asm | 1523.6 | 1538.9 | 1491.8 | 1494.0 | AITER asm padded 1538.9 |
| 32768x16384x13312 | bfloat16 | 8192 | data-parallel |  | 1576.7 | 1581.3 | 1576.7 | 1581.3 | 1573.8 | 1583.8 | flydsl | 1507.6 | 1452.3 (copy) | 1461.0 | 1462.6 | AITER flydsl contiguous 1507.6 |
| 512x2304x16384 | bfloat16 | 18 | one-tile reduce-scatter, n = 7 |  | 639.3 | 591.0 | 162.5 | 162.2 | 162.4 | 161.6 | torch | 738.8 | 650.8 | 737.7 | 653.1 | AITER torch contiguous 738.8 |
| 512x4608x16384 | bfloat16 | 36 | one-tile owner, n = 6 |  | 931.7 | 903.7 | 321.8 | 321.9 | 322.3 | 321.1 | flydsl | 582.9 | 346.0 (copy) | 955.4 | 911.7 | hipBLASLt contiguous 955.4 |
| 512x9216x16384 | bfloat16 | 72 | one-tile owner, n = 3-4 |  | 1088.4 | 1032.6 | 586.0 | 552.0 | 587.4 | 552.8 | flydsl | 641.7 | 386.6 (copy) | 757.7 | 735.6 | hipBLASLt contiguous 757.7 |
| 512x13312x16384 | bfloat16 | 104 | one-tile owner, n = 2-3 |  | 1145.1 | 1114.6 | 772.9 | 724.3 | 784.4 | 725.1 | flydsl | 542.7 | 383.9 (copy) | 1096.5 | 1098.5 | hipBLASLt padded 1098.5 |
| 512x26624x16384 | bfloat16 | 208 | data-parallel | yes | 1305.2 | 1197.1 | 1301.1 | 1176.1 | 1303.1 | 1232.6 | flydsl | 760.0 | 502.0 (copy) | 1182.2 | 1202.9 | hipBLASLt padded 1202.9 |
| 512x53248x16384 | bfloat16 | 416 | data-parallel |  | 1313.1 | 1241.9 | 1313.1 | 1241.9 | 1299.2 | 1238.9 | flydsl | 640.3 | 461.4 (copy) | 1209.3 | 1217.8 | hipBLASLt padded 1217.8 |
| 1024x2304x16384 | bfloat16 | 36 | one-tile owner, n = 6 |  | 899.2 | 807.8 | 324.1 | 320.8 | 322.2 | 321.0 | torch | 892.4 | 806.6 | 902.1 | 811.3 | hipBLASLt contiguous 902.1 |
| 1024x4608x16384 | bfloat16 | 72 | one-tile owner, n = 3-4 |  | 1210.4 | 1164.9 | 635.7 | 637.9 | 639.7 | 637.6 | flydsl | 839.1 | 520.4 (copy) | 1161.8 | 1157.7 | hipBLASLt contiguous 1161.8 |
| 1024x9216x16384 | bfloat16 | 144 | data-parallel | yes | 1126.6 | 1093.0 | 1126.0 | 1092.6 | 1128.2 | 1091.8 | flydsl | 697.0 | 552.4 (copy) | 1173.8 | 1157.2 | hipBLASLt contiguous 1173.8 |
| 1024x13312x16384 | bfloat16 | 208 | data-parallel | yes | 1358.3 | 1336.1 | 1355.5 | 1329.3 | 1358.7 | 1338.9 | asm | 1330.9 | 1317.9 | 1244.9 | 1253.0 | AITER asm contiguous 1330.9 |
| 1024x26624x16384 | bfloat16 | 416 | data-parallel |  | 1380.5 | 1366.5 | 1380.5 | 1366.5 | 1375.8 | 1323.8 | flydsl | 949.7 | 711.3 (copy) | 1253.2 | 1263.8 | hipBLASLt padded 1263.8 |
| 1024x53248x16384 | bfloat16 | 832 | one-tile owner, n = 4 |  | 1530.8 | 1530.2 | 1379.8 | 1369.8 | 1371.0 | 1362.9 | triton | 990.8 | 963.0 | 1255.3 | 1270.1 | hipBLASLt padded 1270.1 |
| 2048x2304x16384 | bfloat16 | 72 | one-tile owner, n = 3-4 |  | 1242.1 | 1217.1 | 636.2 | 632.1 | 639.6 | 638.1 | torch | 934.4 | 894.7 | 945.7 | 880.3 | hipBLASLt contiguous 945.7 |
| 2048x4608x16384 | bfloat16 | 144 | data-parallel | yes | 1207.3 | 1221.4 | 1219.1 | 1217.5 | 1217.3 | 1218.1 | flydsl | 737.4 | 613.3 (copy) | 1310.9 | 1301.6 | hipBLASLt contiguous 1310.9 |
| 2048x9216x16384 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 |  | 1428.9 | 1411.2 | 1092.1 | 1072.6 | 1092.7 | 1074.5 | flydsl | 913.4 | 750.5 (copy) | 1160.5 | 1133.2 | hipBLASLt contiguous 1160.5 |
| 2048x13312x16384 | bfloat16 | 416 | data-parallel |  | 1412.5 | 1404.0 | 1412.5 | 1404.0 | 1407.1 | 1406.2 | flydsl | 1369.3 | 1077.6 (copy) | 1240.4 | 1244.6 | AITER flydsl contiguous 1369.3 |
| 2048x26624x16384 | bfloat16 | 832 | one-tile owner, n = 4 |  | 1566.8 | 1554.0 | 1407.4 | 1405.1 | 1404.1 | 1403.6 | flydsl | 907.1 | 776.9 (copy) | 1243.5 | 1248.7 | hipBLASLt padded 1248.7 |
| 2048x53248x16384 | bfloat16 | 1664 | one-tile owner, n = 2 |  | 1591.3 | 1578.6 | 1531.2 | 1505.6 | 1518.4 | 1502.7 | flydsl | 794.5 | 684.3 (copy) | 1287.2 | 1298.7 | hipBLASLt padded 1298.7 |
| 4096x2304x16384 | bfloat16 | 144 | data-parallel | yes | 1212.7 | 1220.6 | 1217.9 | 1220.4 | 1214.3 | 1220.8 | torch | 1309.2 | 1305.7 | 1313.0 | 1300.6 | hipBLASLt contiguous 1313.0 |
| 4096x4608x16384 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 |  | 1449.3 | 1446.1 | 1104.1 | 1093.4 | 1103.3 | 1096.1 | torch | 1251.3 | 1186.1 | 1254.5 | 1186.3 | hipBLASLt contiguous 1254.5 |
| 4096x9216x16384 | bfloat16 | 576 | one-tile owner, n = 4 |  | 1561.8 | 1565.9 | 1369.4 | 1371.9 | 1370.2 | 1368.2 | torch | 1327.4 | 1349.6 | 1333.2 | 1347.5 | AITER torch padded 1349.6 |
| 4096x13312x16384 | bfloat16 | 832 | one-tile owner, n = 4 |  | 1578.6 | 1575.9 | 1425.8 | 1423.4 | 1425.0 | 1424.3 | flydsl | 1385.6 | 1187.3 (copy) | 1264.9 | 1273.6 | AITER flydsl contiguous 1385.6 |
| 4096x26624x16384 | bfloat16 | 1664 | one-tile owner, n = 2 |  | 1599.9 | 1607.4 | 1544.7 | 1536.5 | 1532.6 | 1530.9 | flydsl | 1493.6 | 1296.5 (copy) | 1317.1 | 1322.5 | AITER flydsl contiguous 1493.6 |
| 4096x53248x16384 | bfloat16 | 3328 | data-parallel |  | 1610.7 | 1622.4 | 1610.7 | 1622.4 | 1605.0 | 1608.6 | flydsl | 1555.2 | 1358.3 (copy) | 1470.4 | 1498.1 | AITER flydsl contiguous 1555.2 |
| 8192x2304x16384 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 |  | 1390.6 | 1346.4 | 1018.6 | 946.0 | 1003.2 | 949.8 | torch | 1197.5 | 1147.6 | 1200.1 | 1150.0 | hipBLASLt contiguous 1200.1 |
| 8192x4608x16384 | bfloat16 | 576 | one-tile owner, n = 4 |  | 1517.3 | 1467.1 | 1318.3 | 1226.5 | 1306.8 | 1231.7 | torch | 1355.5 | 1375.0 | 1365.2 | 1377.7 | hipBLASLt padded 1377.7 |
| 8192x9216x16384 | bfloat16 | 1152 | one-tile owner, n = 2 |  | 1546.2 | 1534.9 | 1436.3 | 1431.0 | 1462.0 | 1429.3 | flydsl | 1407.7 | 1261.5 (copy) | 1141.8 | 1143.6 | AITER flydsl contiguous 1407.7 |
| 8192x13312x16384 | bfloat16 | 1664 | one-tile owner, n = 2 |  | 1564.9 | 1538.7 | 1501.4 | 1464.1 | 1492.0 | 1459.0 | flydsl | 1449.1 | 1319.1 (copy) | 1307.5 | 1317.5 | AITER flydsl contiguous 1449.1 |
| 8192x26624x16384 | bfloat16 | 3328 | data-parallel |  | 1567.4 | 1520.7 | 1567.4 | 1520.7 | 1565.8 | 1512.3 | flydsl | 1509.7 | 1401.9 (copy) | 1505.5 | 1489.6 | AITER flydsl contiguous 1509.7 |
| 8192x53248x16384 | bfloat16 | 6656 | data-parallel |  | 1567.8 | 1574.2 | 1567.8 | 1574.2 | 1571.1 | 1562.3 | flydsl | 1521.2 | 1406.3 (copy) | 1456.7 | 1516.1 | AITER flydsl contiguous 1521.2 |
| 16384x2304x16384 | bfloat16 | 576 | one-tile owner, n = 4 |  | 1488.7 | 1444.1 | 1305.7 | 1260.5 | 1302.3 | 1272.9 | torch | 1347.1 | 1355.0 | 1358.1 | 1354.9 | hipBLASLt contiguous 1358.1 |
| 16384x4608x16384 | bfloat16 | 1152 | one-tile owner, n = 2 |  | 1546.1 | 1527.5 | 1470.6 | 1425.0 | 1461.4 | 1399.7 | flydsl | 1419.6 | 1249.1 (copy) | 1173.1 | 1160.9 | AITER flydsl contiguous 1419.6 |
| 16384x9216x16384 | bfloat16 | 2304 | data-parallel |  | 1569.2 | 1505.4 | 1569.2 | 1505.4 | 1565.8 | 1506.9 | flydsl | 1514.0 | 1397.7 (copy) | 1532.7 | 1495.6 | hipBLASLt contiguous 1532.7 |
| 16384x13312x16384 | bfloat16 | 3328 | data-parallel |  | 1562.7 | 1520.7 | 1562.7 | 1520.7 | 1571.4 | 1513.2 | asm | 1492.3 | 1540.1 | 1513.1 | 1483.7 | AITER asm padded 1540.1 |
| 16384x26624x16384 | bfloat16 | 6656 | data-parallel |  | 1566.7 | 1540.0 | 1566.7 | 1540.0 | 1568.4 | 1541.7 | flydsl | 1521.5 | 1453.3 (copy) | 1458.6 | 1495.7 | AITER flydsl contiguous 1521.5 |
| 16384x53248x16384 | bfloat16 | 13312 | data-parallel |  | 1549.3 | 1542.6 | 1549.3 | 1542.6 | 1570.6 | 1545.7 | asm | 1512.4 | 1541.8 | 1448.2 | 1466.2 | AITER asm padded 1541.8 |
| 32768x2304x16384 | bfloat16 | 1152 | one-tile owner, n = 2 |  | 1522.1 | 1537.9 | 1454.3 | 1447.8 | 1446.5 | 1441.1 | flydsl | 1414.2 | 1151.3 (copy) | 1184.1 | 1173.3 | AITER flydsl contiguous 1414.2 |
| 32768x4608x16384 | bfloat16 | 2304 | data-parallel |  | 1563.0 | 1540.2 | 1563.0 | 1540.2 | 1564.0 | 1531.9 | torch | 1536.3 | 1520.7 | 1545.0 | 1526.8 | hipBLASLt contiguous 1545.0 |
| 32768x9216x16384 | bfloat16 | 4608 | data-parallel |  | 1564.0 | 1554.0 | 1564.0 | 1554.0 | 1571.4 | 1546.7 | asm | 1496.7 | 1507.7 | 1473.6 | 1483.5 | AITER asm padded 1507.7 |
| 32768x13312x16384 | bfloat16 | 6656 | data-parallel |  | 1552.3 | 1564.3 | 1552.3 | 1564.3 | 1570.9 | 1548.0 | asm | 1522.5 | 1538.8 | 1437.3 | 1493.5 | AITER asm padded 1538.8 |
| 32768x26624x16384 | bfloat16 | 13312 | data-parallel |  | 1556.1 | 1569.0 | 1556.1 | 1569.0 | 1563.9 | 1554.9 | flydsl | 1504.6 | 1466.4 (copy) | 1438.4 | 1484.1 | AITER flydsl contiguous 1504.6 |
| 32768x53248x16384 | bfloat16 | 26624 | data-parallel |  | 1553.9 | 1568.7 | 1553.9 | 1568.7 | 1569.8 | 1570.6 | flydsl | 1521.7 | 1495.8 (copy) | 1453.6 | 1495.2 | AITER flydsl contiguous 1521.7 |
| 512x6144x18432 | bfloat16 | 48 | one-tile owner, n = 5-6 |  | 1022.5 | 1022.3 | 426.5 | 429.7 | 429.0 | 428.2 | flydsl | 920.8 | 488.0 (copy) | 960.7 | 944.0 | hipBLASLt contiguous 960.7 |
| 1024x6144x18432 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 1182.7 | 1152.1 | 800.4 | 816.6 | 776.6 | 822.5 | torch | 1193.6 | 1216.7 | 1208.6 | 1220.1 | hipBLASLt padded 1220.1 |
| 2048x6144x18432 | bfloat16 | 192 | data-parallel | yes | 1345.0 | 1346.8 | 1339.6 | 1345.8 | 1346.3 | 1348.1 | torch | 1345.9 | 1365.5 | 1349.3 | 1367.9 | hipBLASLt padded 1367.9 |
| 4096x6144x18432 | bfloat16 | 384 | one-tile owner, n = 2 |  | 1569.0 | 1567.8 | 1380.9 | 1376.9 | 1381.0 | 1379.6 | torch | 1335.3 | 1362.9 | 1345.9 | 1363.0 | hipBLASLt padded 1363.0 |
| 8192x6144x18432 | bfloat16 | 768 | data-parallel |  | 1582.0 | 1602.5 | 1582.0 | 1602.5 | 1584.5 | 1598.4 | torch | 1515.9 | 1569.8 | 1515.6 | 1568.2 | AITER torch padded 1569.8 |
| 16384x6144x18432 | bfloat16 | 1536 | data-parallel |  | 1594.3 | 1607.1 | 1594.3 | 1607.1 | 1588.0 | 1600.0 | asm | 1547.2 | 1560.5 | 1517.9 | 1572.0 | hipBLASLt padded 1572.0 |
| 32768x6144x18432 | bfloat16 | 3072 | data-parallel |  | 1594.5 | 1594.0 | 1594.5 | 1594.0 | 1589.7 | 1590.6 | asm | 1543.4 | 1535.9 | 1443.8 | 1468.0 | AITER asm contiguous 1543.4 |
| 512x5120x25600 | bfloat16 | 40 | one-tile owner, n = 6-7 |  | 996.4 | 1007.5 | 323.9 | 334.8 | 323.3 | 333.5 | flydsl | 438.2 | 310.9 (copy) | 889.0 | 904.9 | hipBLASLt padded 904.9 |
| 1024x5120x25600 | bfloat16 | 80 | one-tile owner, n = 3-4 |  | 1306.5 | 1317.8 | 647.4 | 657.4 | 652.4 | 661.0 | torch | 1159.6 | 1175.5 | 1163.4 | 1177.7 | hipBLASLt padded 1177.7 |
| 2048x5120x25600 | bfloat16 | 160 | data-parallel | yes | 1176.7 | 1206.0 | 1191.2 | 1222.1 | 1180.0 | 1227.9 | torch | 1287.0 | 1306.7 | 1293.6 | 1307.7 | hipBLASLt padded 1307.7 |
| 4096x5120x25600 | bfloat16 | 320 | one-tile owner, n = 4 |  | 1549.6 | 1554.5 | 1154.3 | 1196.3 | 1161.7 | 1197.1 | torch | 1276.9 | 1295.6 | 1280.1 | 1296.5 | hipBLASLt padded 1296.5 |
| 8192x5120x25600 | bfloat16 | 640 | one-tile owner, n = 2 |  | 1574.0 | 1589.9 | 1427.5 | 1441.2 | 1420.4 | 1436.6 | asm | 1398.8 | 1413.7 | 1350.4 | 1371.6 | AITER asm padded 1413.7 |
| 16384x5120x25600 | bfloat16 | 1280 | data-parallel |  | 1598.8 | 1601.9 | 1598.8 | 1601.9 | 1592.9 | 1601.4 | torch | 1512.6 | 1584.6 | 1505.2 | 1584.1 | AITER torch padded 1584.6 |
| 32768x5120x25600 | bfloat16 | 2560 | data-parallel |  | 1598.7 | 1593.6 | 1598.7 | 1593.6 | 1595.5 | 1596.1 | torch | 1458.7 | 1449.1 | 1455.9 | 1440.8 | AITER torch contiguous 1458.7 |
| 512x16384x26624 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1379.7 | 1388.5 | 947.7 | 932.9 | 932.1 | 931.4 | flydsl | 650.0 | 440.5 (copy) | 1204.0 | 1201.2 | hipBLASLt contiguous 1204.0 |
| 1024x16384x26624 | bfloat16 | 256 | data-parallel | yes | 1563.6 | 1570.9 | 1556.6 | 1568.9 | 1561.8 | 1570.1 | flydsl | 782.6 | 584.9 (copy) | 1524.9 | 1539.0 | hipBLASLt padded 1539.0 |
| 2048x16384x26624 | bfloat16 | 512 | data-parallel |  | 1612.8 | 1611.8 | 1612.8 | 1611.8 | 1601.5 | 1612.5 | flydsl | 817.3 | 715.2 (copy) | 1534.5 | 1549.7 | hipBLASLt padded 1549.7 |
| 4096x16384x26624 | bfloat16 | 1024 | data-parallel |  | 1621.3 | 1622.8 | 1621.3 | 1622.8 | 1613.3 | 1622.5 | flydsl | 1563.8 | 1346.5 (copy) | 1542.2 | 1560.6 | AITER flydsl contiguous 1563.8 |
| 8192x16384x26624 | bfloat16 | 2048 | data-parallel |  | 1606.5 | 1606.8 | 1606.5 | 1606.8 | 1603.5 | 1608.0 | asm | 238.1 | 241.9 | 1523.2 | 1544.1 | hipBLASLt padded 1544.1 |
| 16384x16384x26624 | bfloat16 | 4096 | data-parallel |  | 1604.6 | 1596.4 | 1604.6 | 1596.4 | 1606.3 | 1606.8 | asm | 1563.4 | 1568.6 | 1458.9 | 1461.4 | AITER asm padded 1568.6 |
| 32768x16384x26624 | bfloat16 | 8192 | data-parallel |  | 1601.3 | 1561.8 | 1601.3 | 1561.8 | 1606.9 | 1595.3 | asm | 1557.9 | 1565.5 | 1449.9 | 1465.3 | AITER asm padded 1565.5 |
| 512x8192x28672 | bfloat16 | 64 | one-tile owner, n = 4 |  | 1234.7 | 1237.1 | 474.5 | 484.9 | 468.9 | 479.0 | flydsl | 775.2 | 484.5 (copy) | 923.1 | 923.3 | hipBLASLt padded 923.3 |
| 1024x8192x28672 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1453.6 | 1468.3 | 936.9 | 977.3 | 941.1 | 964.2 | triton | 692.5 | 696.5 | 1348.7 | 1368.6 | hipBLASLt padded 1368.6 |
| 2048x8192x28672 | bfloat16 | 256 | data-parallel | yes | 1601.8 | 1605.6 | 1601.9 | 1601.7 | 1590.0 | 1604.6 | flydsl | 1035.0 | 846.1 (copy) | 1564.5 | 1577.8 | hipBLASLt padded 1577.8 |
| 4096x8192x28672 | bfloat16 | 512 | data-parallel |  | 1617.3 | 1618.7 | 1617.3 | 1618.7 | 1605.0 | 1622.1 | flydsl | 1556.0 | 1296.6 (copy) | 1560.9 | 1583.6 | hipBLASLt padded 1583.6 |
| 8192x8192x28672 | bfloat16 | 1024 | data-parallel |  | 1604.0 | 1616.4 | 1604.0 | 1616.4 | 1601.0 | 1612.3 | asm | 1569.0 | 1579.3 | 1551.6 | 1575.5 | AITER asm padded 1579.3 |
| 16384x8192x28672 | bfloat16 | 2048 | data-parallel |  | 1608.6 | 1613.1 | 1608.6 | 1613.1 | 1608.3 | 1616.0 | asm | 1540.2 | 1580.6 | 1544.7 | 1569.8 | AITER asm padded 1580.6 |
| 32768x8192x28672 | bfloat16 | 4096 | data-parallel |  | 1610.8 | 1611.3 | 1610.8 | 1611.3 | 1609.4 | 1617.6 | asm | 1569.3 | 1578.1 | 1505.6 | 1440.0 | AITER asm padded 1578.1 |
| 512x7168x35840 | bfloat16 | 56 | one-tile owner, n = 4-5 |  | 1191.9 | 1134.4 | 438.6 | 407.5 | 438.0 | 404.8 | flydsl | 816.4 | 500.7 (copy) | 964.1 | 962.7 | hipBLASLt contiguous 964.1 |
| 1024x7168x35840 | bfloat16 | 112 | one-tile owner, n = 2-3 |  | 1364.8 | 1365.0 | 866.8 | 817.0 | 862.6 | 810.8 | flydsl | 571.8 | 462.6 (copy) | 1291.0 | 1303.7 | hipBLASLt padded 1303.7 |
| 2048x7168x35840 | bfloat16 | 224 | spread |  | 1522.2 | 1526.7 | 1496.6 | 1498.2 | 1495.3 | 1495.5 | asm | 1458.0 | 1466.1 | 1411.4 | 1415.8 | AITER asm padded 1466.1 |
| 4096x7168x35840 | bfloat16 | 448 | spread |  | 1579.0 | 1583.3 | 1504.9 | 1511.3 | 1503.2 | 1507.2 | flydsl | 1465.7 | 1221.4 (copy) | 1404.3 | 1408.9 | AITER flydsl contiguous 1465.7 |
| 8192x7168x35840 | bfloat16 | 896 | one-tile owner, n = 2 |  | 1595.9 | 1606.0 | 1479.8 | 1487.6 | 1475.3 | 1484.9 | asm | 1429.0 | 1435.8 | 1408.9 | 1417.8 | AITER asm padded 1435.8 |
| 16384x7168x35840 | bfloat16 | 1792 | data-parallel |  | 1588.8 | 1588.7 | 1588.8 | 1588.7 | 1596.2 | 1594.9 | torch | 1530.5 | 1530.9 | 1532.1 | 1523.8 | hipBLASLt contiguous 1532.1 |
| 32768x7168x35840 | bfloat16 | 3584 | data-parallel |  | 1592.3 | 1566.5 | 1592.3 | 1566.5 | 1590.5 | 1585.6 | flydsl | 1539.4 | 1368.0 (copy) | 1468.9 | 1474.3 | AITER flydsl contiguous 1539.4 |
