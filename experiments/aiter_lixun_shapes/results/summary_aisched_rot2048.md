# streamk_wip (v20, capped, v9 for one-wave dp) vs AITER, v9 and hipBLASLt, contiguous and 512-byte-padded strides (2048 MB rotating buffer)

- Stack: torch 2.10.0+rocm7.2.4.git3d3aa833, hipBLASLt 100202 (dabb6df2b9), Triton 3.8.0 @ /home/jbowley/repos/triton_gfx950-tutorial-v2.2/python/triton; GPU: PCI bus 117, gfx950:sramecc+:xnack-; AITER e2c7aa0aa [Triton/Gluon] gfx950 a16w16: skip the Gluon kernel on Triton without cd_regclass on a1455f1c4 2026-10-02 [CK] [FlyDSL] add mxfp8 a8w8 blockscale for gfx950 (#4951).
- streamk_wip: snapshot sha256 03957ee906733825 (newest WORKLOG entry: "13. v9 writes C in C's dtype; fp32 C goes to v9 too (2026-10-06)"); the v9 it delegates to: live sha256 7b4e246a4877a02c.
- Layouts: contiguous, and padded: every row stride (x, w, C) that is a multiple of 512 bytes gets 256 more (x and w: K + 128 when K % 256 == 0; C: N + 128 for bf16, N + 64 for fp32). v9, streamk_wip and hipBLASLt (`torch.mm(x, w.t(), out=y)`) get padded x, w and C; AITER's pick allocates its own C, so it gets padded x and w only.
- Timing: 3 rounds, each with fresh inputs (new buffers) when `realloc` is on (here: True); median of the rounds. Spread = (slowest - fastest round) / median.
- Best library: the fastest valid time of AITER's pick and hipBLASLt; for the padded comparison over both layouts. A padded call that launches a copy kernel (FlyDSL makes its inputs contiguous) is not valid.
- "Faster" and "slower" mean a gap of more than 1%.

Padded calls that copied their inputs (left out of the best library): pick (flydsl) on 252 shapes.

The best library's time comes from the contiguous layout on 398 shapes, padded layout on 232 shapes.

## v9 and streamk_wip padded, libraries contiguous (as AITER calls them)

| Shapes | Count | v9: faster | v9: within 1% | v9: slower | v9: median | data-parallel only: faster | data-parallel only: within 1% | data-parallel only: slower | data-parallel only: median | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All shapes | 630 | 240 | 45 | 345 | -4.3% | 263 | 34 | 333 | -3.5% | 340 | 36 | 254 | +2.0% |
| Under 256 tiles (less than one wave) | 321 | 24 | 6 | 291 | -39.6% | 19 | 7 | 295 | -40.1% | 63 | 14 | 244 | -16.7% |
| 256 to 1023 tiles | 152 | 103 | 10 | 39 | +3.1% | 107 | 10 | 35 | +3.3% | 136 | 9 | 7 | +5.0% |
| 1024 or more tiles | 157 | 113 | 29 | 15 | +2.7% | 137 | 17 | 3 | +3.6% | 141 | 13 | 3 | +3.9% |
| Best library uses 256x256 | 258 | 195 | 35 | 28 | +3.0% | 220 | 22 | 16 | +3.6% | 234 | 20 | 4 | +4.1% |
| Best library uses another tile | 372 | 45 | 10 | 317 | -30.9% | 43 | 12 | 317 | -31.6% | 106 | 16 | 250 | -10.6% |

## Everything contiguous (comparable with Oct 5)

| Shapes | Count | v9: faster | v9: within 1% | v9: slower | v9: median | data-parallel only: faster | data-parallel only: within 1% | data-parallel only: slower | data-parallel only: median | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All shapes | 630 | 233 | 46 | 351 | -4.8% | 254 | 39 | 337 | -4.9% | 323 | 46 | 261 | +1.2% |
| Under 256 tiles (less than one wave) | 321 | 21 | 6 | 294 | -37.7% | 20 | 2 | 299 | -38.3% | 61 | 9 | 251 | -15.3% |
| 256 to 1023 tiles | 152 | 96 | 16 | 40 | +2.4% | 100 | 15 | 37 | +2.4% | 130 | 13 | 9 | +4.1% |
| 1024 or more tiles | 157 | 116 | 24 | 17 | +2.4% | 134 | 22 | 1 | +2.6% | 132 | 24 | 1 | +2.6% |
| Best library uses 256x256 | 258 | 197 | 32 | 29 | +2.5% | 215 | 30 | 13 | +2.7% | 224 | 32 | 2 | +2.8% |
| Best library uses another tile | 372 | 36 | 14 | 322 | -30.0% | 39 | 9 | 324 | -31.7% | 99 | 14 | 259 | -10.5% |

## v9 and streamk_wip padded, against the best library over both layouts (strictest)

| Shapes | Count | v9: faster | v9: within 1% | v9: slower | v9: median | data-parallel only: faster | data-parallel only: within 1% | data-parallel only: slower | data-parallel only: median | streamk_wip: faster | streamk_wip: within 1% | streamk_wip: slower | streamk_wip: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All shapes | 630 | 226 | 52 | 352 | -5.4% | 255 | 36 | 339 | -4.6% | 331 | 38 | 261 | +1.7% |
| Under 256 tiles (less than one wave) | 321 | 21 | 5 | 295 | -40.0% | 19 | 3 | 299 | -40.3% | 59 | 12 | 250 | -17.5% |
| 256 to 1023 tiles | 152 | 99 | 13 | 40 | +2.8% | 103 | 13 | 36 | +2.8% | 135 | 10 | 7 | +4.6% |
| 1024 or more tiles | 157 | 106 | 34 | 17 | +2.3% | 133 | 20 | 4 | +3.2% | 137 | 16 | 4 | +3.4% |
| Best library uses 256x256 | 259 | 188 | 40 | 31 | +2.5% | 216 | 25 | 18 | +3.0% | 231 | 23 | 5 | +3.8% |
| Best library uses another tile | 371 | 38 | 12 | 321 | -31.3% | 39 | 11 | 321 | -32.3% | 100 | 15 | 256 | -11.0% |

"Data-parallel only" is `STREAMK_POLICY=dp`, the persistent loop without stream-K and without the v9 hand-off; where capped already runs data-parallel on the persistent loop it is streamk_wip's time.

## v9 delegation (124 shapes: capped picks data-parallel, at most 256 tiles, bf16 output)

streamk_wip (now v9) against the persistent loop it replaced and against AITER's v9, same sweep. Positive means streamk_wip is faster.

| Layout | Shapes | Against | faster | within 1% | slower | median | p10 | p90 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| contiguous | all (124) | persistent loop (dp) | 56 | 65 | 3 | +0.8% | -0.4% | +4.0% |
| contiguous | all (124) | AITER's v9 | 1 | 114 | 9 | -0.3% | -0.9% | +0.1% |
| contiguous | k-steps <= 32 (56) | persistent loop (dp) | 49 | 7 | 0 | +2.9% | +0.9% | +4.5% |
| contiguous | k-steps <= 32 (56) | AITER's v9 | 0 | 53 | 3 | -0.4% | -0.9% | +0.1% |
| contiguous | k-steps > 32 (68) | persistent loop (dp) | 7 | 58 | 3 | +0.0% | -0.7% | +0.8% |
| contiguous | k-steps > 32 (68) | AITER's v9 | 1 | 61 | 6 | -0.3% | -1.0% | +0.2% |
| padded | all (124) | persistent loop (dp) | 70 | 53 | 1 | +1.1% | +0.3% | +2.9% |
| padded | all (124) | AITER's v9 | 1 | 118 | 5 | +0.1% | -0.2% | +0.5% |
| padded | k-steps <= 32 (56) | persistent loop (dp) | 55 | 1 | 0 | +2.0% | +1.3% | +3.3% |
| padded | k-steps <= 32 (56) | AITER's v9 | 1 | 55 | 0 | +0.1% | -0.1% | +0.6% |
| padded | k-steps > 32 (68) | persistent loop (dp) | 15 | 52 | 1 | +0.7% | -0.1% | +1.1% |
| padded | k-steps > 32 (68) | AITER's v9 | 0 | 63 | 5 | +0.0% | -0.5% | +0.4% |

### Against `../../../aiter/streamk_study/2026-10-06/results/details_v9fb_pad_rot2048.csv` (contiguous, this run / that run - 1)

| Shapes | Count | streamk_wip | v9 (AITER) | AITER's pick | hipBLASLt |
|---|---:|---:|---:|---:|---:|
| Delegated to v9 now | 124 | +0.7% | -0.2% | -0.2% | -0.2% |
| Not delegated | 506 | +0.2% | -0.2% | -0.1% | -0.1% |

### The 16 same-tile shapes where streamk_wip was behind on Oct 5

| Shape | Tiles | k-steps | Delegated now | Oct 5: vs best | Now contiguous: vs best | Now padded: vs best | Now padded: vs v9 padded |
|---|---:|---:|---|---:|---:|---:|---:|
| 1024x2304x16384 | 36 | 256 | no | -5.7% | -4.1% | -21.3% | +181.6% |
| 4096x4096x1536 | 256 | 24 | yes | -4.3% | +0.5% | +2.7% | +0.2% |
| 16384x7168x512 | 1792 | 8 | no | -3.4% | -1.9% | -3.9% | +8.1% |
| 4096x4096x512 | 256 | 8 | yes | -3.1% | +0.7% | +0.3% | +0.8% |
| 4096x4096x2048 | 256 | 32 | yes | -2.9% | +1.0% | +2.8% | +0.4% |
| 16384x57344x8192 | 14336 | 128 | no | -2.9% | +2.4% | +3.1% | +0.4% |
| 8192x2048x512 | 256 | 8 | yes | -2.7% | +1.8% | +0.1% | +0.2% |
| 16384x2048x1536 | 512 | 24 | no | -2.6% | -0.5% | +2.2% | +1.1% |
| 8192x4096x512 | 512 | 8 | no | -2.4% | -0.5% | -0.3% | +4.4% |
| 16384x2048x512 | 512 | 8 | no | -2.0% | +1.2% | +0.2% | +5.5% |
| 32768x512x7168 | 256 | 112 | yes | -1.7% | -0.1% | -0.6% | +0.8% |
| 4096x4096x1024 | 256 | 16 | yes | -1.6% | +2.4% | +4.7% | +0.1% |
| 32768x5120x640 | 2560 | 10 | no | -1.5% | +0.2% | +0.8% | +5.4% |
| 16384x4096x512 | 1024 | 8 | no | -1.3% | +0.0% | +1.1% | +7.3% |
| 4096x8192x1024 | 512 | 16 | no | -1.2% | +1.1% | +2.2% | +0.7% |
| 2048x8192x2048 | 256 | 32 | yes | -1.1% | +3.7% | +3.7% | +0.2% |

## Padding: does it remove the noise?

Spread between the 3 rounds, per kernel and layout. With `realloc` every round has new buffers, so a timing that depends on where the buffers landed shows up here. Control: the 16 shapes with K % 256 == 128, whose x and w strides are not padded (only C is).

| Kernel | Shapes | contiguous: median spread | contiguous: > 2% | contiguous: > 5% | padded: median spread | padded: > 2% | padded: > 5% |
|---|---|---:|---:|---:|---:|---:|---:|
| v9 | All, strides padded (614) | 0.29% | 11 | 3 | 0.28% | 7 | 1 |
| v9 | Under 256 (less than one wave), strides padded (314) | 0.32% | 5 | 2 | 0.30% | 1 | 0 |
| v9 | 256 to 1023, strides padded (148) | 0.29% | 3 | 0 | 0.32% | 4 | 0 |
| v9 | 1024 or more, strides padded (152) | 0.22% | 3 | 1 | 0.26% | 2 | 1 |
| v9 | Control (K % 256 == 128) (16) | 0.28% | 0 | 0 | 0.20% | 0 | 0 |
| streamk_wip | All, strides padded (614) | 0.66% | 17 | 0 | 0.29% | 20 | 1 |
| streamk_wip | Under 256 (less than one wave), strides padded (314) | 0.60% | 9 | 0 | 0.34% | 9 | 0 |
| streamk_wip | 256 to 1023, strides padded (148) | 0.74% | 0 | 0 | 0.30% | 4 | 0 |
| streamk_wip | 1024 or more, strides padded (152) | 0.65% | 8 | 0 | 0.24% | 7 | 1 |
| streamk_wip | Control (K % 256 == 128) (16) | 0.54% | 0 | 0 | 0.21% | 0 | 0 |
| AITER's pick | All, strides padded (362) | 0.45% | 15 | 0 | 0.34% | 16 | 4 |
| AITER's pick | Under 256 (less than one wave), strides padded (168) | 0.43% | 4 | 0 | 0.41% | 10 | 2 |
| AITER's pick | 256 to 1023, strides padded (89) | 0.38% | 2 | 0 | 0.29% | 1 | 0 |
| AITER's pick | 1024 or more, strides padded (105) | 0.52% | 9 | 0 | 0.32% | 5 | 2 |
| AITER's pick | Control (K % 256 == 128) (16) | 0.52% | 0 | 0 | 0.31% | 1 | 0 |
| hipBLASLt | All, strides padded (614) | 0.42% | 43 | 10 | 0.41% | 51 | 9 |
| hipBLASLt | Under 256 (less than one wave), strides padded (314) | 0.47% | 27 | 4 | 0.47% | 35 | 4 |
| hipBLASLt | 256 to 1023, strides padded (148) | 0.30% | 1 | 0 | 0.29% | 1 | 0 |
| hipBLASLt | 1024 or more, strides padded (152) | 0.49% | 15 | 6 | 0.49% | 15 | 5 |
| hipBLASLt | Control (K % 256 == 128) (16) | 0.34% | 1 | 0 | 0.40% | 0 | 0 |

AITER's pick leaves out (on both layouts) the shapes where its padded call copied its inputs.

### Padded against contiguous, median time over the rounds

Padded TFLOPS / contiguous TFLOPS - 1 (positive: padding is faster).

| Kernel | Shapes | Count | faster | within 1% | slower | median | p10 | p90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| v9 | All, strides padded | 614 | 206 | 187 | 221 | +0.2% | -7.4% | +2.9% |
| v9 | Under 256 (less than one wave) | 314 | 98 | 33 | 183 | -2.8% | -10.0% | +5.0% |
| v9 | 256 to 1023 | 148 | 71 | 57 | 20 | +0.9% | -1.8% | +2.3% |
| v9 | 1024 or more | 152 | 37 | 97 | 18 | +0.3% | -1.4% | +1.5% |
| v9 | Control (K % 256 == 128) | 16 | 1 | 15 | 0 | +0.1% | -0.0% | +0.6% |
| streamk_wip | All, strides padded | 614 | 265 | 174 | 175 | +0.7% | -4.2% | +2.9% |
| streamk_wip | Under 256 (less than one wave) | 314 | 103 | 62 | 149 | -0.8% | -5.9% | +3.6% |
| streamk_wip | 256 to 1023 | 148 | 90 | 45 | 13 | +1.3% | -0.8% | +2.4% |
| streamk_wip | 1024 or more | 152 | 72 | 67 | 13 | +0.9% | -0.6% | +2.2% |
| streamk_wip | Control (K % 256 == 128) | 16 | 4 | 11 | 1 | +0.5% | +0.0% | +1.2% |
| AITER's pick | All, strides padded | 362 | 106 | 173 | 83 | +0.4% | -2.4% | +2.5% |
| AITER's pick | Under 256 (less than one wave) | 168 | 50 | 48 | 70 | -0.4% | -3.6% | +3.0% |
| AITER's pick | 256 to 1023 | 89 | 32 | 45 | 12 | +0.7% | -1.2% | +2.5% |
| AITER's pick | 1024 or more | 105 | 24 | 80 | 1 | +0.6% | -0.2% | +2.1% |
| AITER's pick | Control (K % 256 == 128) | 16 | 1 | 15 | 0 | +0.3% | -0.2% | +0.7% |
| hipBLASLt | All, strides padded | 614 | 105 | 314 | 195 | -0.2% | -2.9% | +1.9% |
| hipBLASLt | Under 256 (less than one wave) | 314 | 54 | 114 | 146 | -0.9% | -3.4% | +2.0% |
| hipBLASLt | 256 to 1023 | 148 | 26 | 97 | 25 | +0.1% | -1.4% | +1.9% |
| hipBLASLt | 1024 or more | 152 | 25 | 103 | 24 | -0.1% | -1.2% | +1.7% |
| hipBLASLt | Control (K % 256 == 128) | 16 | 0 | 14 | 2 | -0.3% | -0.7% | +0.0% |

## Correctness

Checks: hbl_ok, pick_ok, v9_ok, v20_ok, v20_dp_ok, v20_capped_ok, v20_two_tile_ok, v20_post_ok, v20_capped_post_ok, pick_pad_ok, v9_pad_ok, hbl_pad_ok, v20_pad_ok, v20_dp_pad_ok, v20_pad_post_ok; and streamk_wip's flags and counts are back at 0 after timing.

- Shapes with any failed check or note: 51; of those, 27 are AITER's pick alone (contiguous and/or padded) outside the tolerance.
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
| 1024x3072x512 | bfloat16 | 48 | data-parallel | yes | 201.7 | 186.5 | 193.0 | 179.5 | 201.8 | 186.1 | flydsl | 368.7 | 183.9 (copy) | 322.4 | 322.1 | AITER flydsl contiguous 368.7 |
| 1024x4096x512 | bfloat16 | 64 | data-parallel | yes | 258.6 | 240.6 | 251.0 | 233.9 | 258.5 | 240.2 | flydsl | 437.4 | 231.0 (copy) | 391.5 | 387.3 | AITER flydsl contiguous 437.4 |
| 1024x7168x512 | bfloat16 | 112 | data-parallel | yes | 417.6 | 394.7 | 403.0 | 383.1 | 418.8 | 394.1 | flydsl | 509.6 | 312.0 (copy) | 504.5 | 476.8 | AITER flydsl contiguous 509.6 |
| 2048x2048x512 | bfloat16 | 64 | data-parallel | yes | 259.4 | 243.2 | 251.8 | 237.3 | 259.9 | 243.1 | flydsl | 430.1 | 241.5 (copy) | 393.3 | 388.6 | AITER flydsl contiguous 430.1 |
| 2048x4096x512 | bfloat16 | 128 | data-parallel | yes | 463.9 | 446.9 | 451.9 | 437.3 | 464.6 | 447.2 | torch | 559.1 | 533.2 | 558.1 | 542.6 | AITER torch contiguous 559.1 |
| 4096x2048x512 | bfloat16 | 128 | data-parallel | yes | 459.2 | 446.6 | 446.9 | 437.3 | 463.3 | 442.0 | torch | 644.5 | 607.2 | 649.9 | 607.2 | hipBLASLt contiguous 649.9 |
| 4096x4096x512 | bfloat16 | 256 | data-parallel | yes | 782.2 | 779.1 | 759.6 | 766.6 | 786.4 | 772.6 | torch | 768.7 | 754.6 | 776.5 | 755.1 | hipBLASLt contiguous 776.5 |
| 4096x7168x512 | bfloat16 | 448 | data-parallel |  | 786.4 | 768.5 | 786.4 | 768.5 | 765.4 | 739.5 | flydsl | 746.5 | 625.1 (copy) | 727.1 | 697.5 | AITER flydsl contiguous 746.5 |
| 8192x2048x512 | bfloat16 | 256 | data-parallel | yes | 782.7 | 769.4 | 761.6 | 754.7 | 787.7 | 768.1 | torch | 760.5 | 743.3 | 768.9 | 746.0 | hipBLASLt contiguous 768.9 |
| 8192x4096x512 | bfloat16 | 512 | data-parallel |  | 838.4 | 840.6 | 838.4 | 840.6 | 813.5 | 805.3 | flydsl | 805.5 | 651.4 (copy) | 843.0 | 833.8 | hipBLASLt contiguous 843.0 |
| 16384x2048x512 | bfloat16 | 512 | data-parallel |  | 847.3 | 839.2 | 847.3 | 839.2 | 826.2 | 795.2 | torch | 837.3 | 824.1 | 834.8 | 824.7 | AITER torch contiguous 837.3 |
| 16384x4096x512 | bfloat16 | 1024 | data-parallel |  | 894.8 | 904.8 | 894.8 | 904.8 | 839.7 | 842.9 | torch | 889.5 | 879.2 | 894.8 | 883.8 | hipBLASLt contiguous 894.8 |
| 16384x7168x512 | bfloat16 | 1792 | data-parallel |  | 972.6 | 952.9 | 972.6 | 952.9 | 905.0 | 881.3 | torch | 981.7 | 973.1 | 991.8 | 980.5 | hipBLASLt contiguous 991.8 |
| 32768x2048x512 | bfloat16 | 1024 | data-parallel |  | 906.8 | 880.8 | 906.8 | 880.8 | 854.5 | 808.4 | torch | 876.6 | 872.0 | 875.6 | 870.5 | AITER torch contiguous 876.6 |
| 1024x5120x640 | bfloat16 | 80 | data-parallel | yes | 324.5 | 325.3 | 314.4 | 314.5 | 324.5 | 324.4 | torch | 444.2 | 445.9 | 447.3 | 447.1 | hipBLASLt contiguous 447.3 |
| 2048x5120x640 | bfloat16 | 160 | data-parallel | yes | 605.3 | 605.6 | 588.8 | 589.3 | 604.6 | 604.4 | flydsl | 595.5 | 594.5 | 507.9 | 506.6 | AITER flydsl contiguous 595.5 |
| 4096x5120x640 | bfloat16 | 320 | data-parallel |  | 653.5 | 656.9 | 653.5 | 656.9 | 638.6 | 637.4 | flydsl | 611.5 | 612.4 | 651.5 | 646.9 | hipBLASLt contiguous 651.5 |
| 8192x5120x640 | bfloat16 | 640 | data-parallel |  | 827.6 | 828.6 | 827.6 | 828.6 | 798.4 | 796.9 | torch | 867.6 | 865.2 | 865.4 | 864.3 | AITER torch contiguous 867.6 |
| 16384x5120x640 | bfloat16 | 1280 | data-parallel |  | 1015.0 | 1018.3 | 1015.0 | 1018.3 | 958.0 | 960.9 | torch | 993.2 | 1007.7 | 1003.0 | 1008.8 | hipBLASLt padded 1008.8 |
| 32768x5120x640 | bfloat16 | 2560 | data-parallel |  | 1050.8 | 1056.8 | 1050.8 | 1056.8 | 999.2 | 1002.6 | torch | 1040.1 | 1033.4 | 1048.6 | 1032.3 | hipBLASLt contiguous 1048.6 |
| 1024x7168x768 | bfloat16 | 112 | data-parallel | yes | 485.3 | 498.6 | 466.7 | 482.2 | 486.2 | 496.8 | flydsl | 596.5 | 384.5 (copy) | 589.0 | 601.5 | hipBLASLt padded 601.5 |
| 10240x7168x768 | bfloat16 | 1120 | data-parallel |  | 1050.8 | 1064.6 | 1050.8 | 1064.6 | 1013.5 | 1022.5 | opus | 387.9 | 398.3 | 862.0 | 861.9 | hipBLASLt contiguous 862.0 |
| 512x4096x1024 | bfloat16 | 32 | data-parallel | yes | 180.7 | 174.5 | 176.8 | 169.2 | 180.7 | 174.6 | torch | 390.4 | 379.7 | 392.0 | 381.2 | hipBLASLt contiguous 392.0 |
| 512x6144x1024 | bfloat16 | 48 | data-parallel | yes | 268.0 | 259.9 | 260.1 | 250.0 | 268.3 | 259.8 | flydsl | 494.0 | 265.7 (copy) | 474.9 | 463.9 | AITER flydsl contiguous 494.0 |
| 512x7168x1024 | bfloat16 | 56 | data-parallel | yes | 309.6 | 300.9 | 301.5 | 289.5 | 310.2 | 300.6 | flydsl | 535.7 | 295.5 (copy) | 498.4 | 495.5 | AITER flydsl contiguous 535.7 |
| 512x8192x1024 | bfloat16 | 64 | data-parallel | yes | 344.4 | 338.9 | 339.4 | 328.8 | 345.4 | 338.2 | flydsl | 580.2 | 325.0 (copy) | 550.3 | 553.3 | AITER flydsl contiguous 580.2 |
| 1024x4096x1024 | bfloat16 | 64 | data-parallel | yes | 344.8 | 337.4 | 338.9 | 328.4 | 345.8 | 337.4 | torch | 559.4 | 557.5 | 562.7 | 553.9 | hipBLASLt contiguous 562.7 |
| 1024x7168x1024 | bfloat16 | 112 | data-parallel | yes | 571.1 | 561.7 | 553.5 | 543.6 | 570.3 | 561.7 | flydsl | 670.5 | 454.1 (copy) | 693.5 | 688.7 | hipBLASLt contiguous 693.5 |
| 1024x8192x1024 | bfloat16 | 128 | data-parallel | yes | 629.0 | 640.1 | 616.2 | 624.4 | 631.1 | 640.6 | torch | 758.9 | 744.8 | 767.9 | 764.6 | hipBLASLt contiguous 767.9 |
| 2048x4096x1024 | bfloat16 | 128 | data-parallel | yes | 630.4 | 635.7 | 620.2 | 617.0 | 633.3 | 632.0 | torch | 768.4 | 755.0 | 776.1 | 770.4 | hipBLASLt contiguous 776.1 |
| 2048x7168x1024 | bfloat16 | 224 | data-parallel | yes | 1002.0 | 1013.7 | 987.4 | 987.6 | 1010.3 | 1016.0 | flydsl | 948.1 | 681.6 (copy) | 963.9 | 965.7 | hipBLASLt padded 965.7 |
| 2048x8192x1024 | bfloat16 | 256 | data-parallel | yes | 1068.9 | 1096.1 | 1054.9 | 1065.0 | 1075.1 | 1096.4 | flydsl | 1015.7 | 726.0 (copy) | 1016.2 | 1043.1 | hipBLASLt padded 1043.1 |
| 3072x4096x1024 | bfloat16 | 192 | data-parallel | yes | 889.9 | 898.5 | 873.6 | 872.4 | 898.5 | 898.0 | torch | 918.6 | 920.3 | 927.5 | 928.4 | hipBLASLt padded 928.4 |
| 4096x4096x1024 | bfloat16 | 256 | data-parallel | yes | 1070.9 | 1094.8 | 1061.1 | 1069.0 | 1085.2 | 1094.1 | torch | 1036.7 | 1025.7 | 1045.7 | 1027.0 | hipBLASLt contiguous 1045.7 |
| 4096x7168x1024 | bfloat16 | 448 | data-parallel |  | 1056.7 | 1054.3 | 1056.7 | 1054.3 | 1044.4 | 1045.5 | flydsl | 992.7 | 834.5 (copy) | 983.1 | 974.7 | AITER flydsl contiguous 992.7 |
| 4096x8192x1024 | bfloat16 | 512 | data-parallel |  | 1123.7 | 1142.3 | 1123.7 | 1142.3 | 1104.7 | 1134.4 | asm | 1107.5 | 1102.7 | 1111.7 | 1117.4 | hipBLASLt padded 1117.4 |
| 5120x4096x1024 | bfloat16 | 320 | data-parallel |  | 864.9 | 866.0 | 864.9 | 866.0 | 860.8 | 862.6 | torch | 972.8 | 975.9 | 978.4 | 976.3 | hipBLASLt contiguous 978.4 |
| 7680x4096x1024 | bfloat16 | 480 | data-parallel |  | 1101.4 | 1104.8 | 1101.4 | 1104.8 | 1093.3 | 1095.7 | torch | 944.2 | 938.6 | 950.7 | 936.3 | hipBLASLt contiguous 950.7 |
| 8192x7168x1024 | bfloat16 | 896 | data-parallel |  | 1096.6 | 1087.9 | 1096.6 | 1087.9 | 1052.5 | 1051.2 | asm | 1042.4 | 1040.0 | 1028.8 | 1016.9 | AITER asm contiguous 1042.4 |
| 8192x8192x1024 | bfloat16 | 1024 | data-parallel |  | 1179.0 | 1186.8 | 1179.0 | 1186.8 | 1137.1 | 1148.1 | asm | 1138.3 | 1137.7 | 1169.7 | 1157.6 | hipBLASLt contiguous 1169.7 |
| 16384x4096x1024 | bfloat16 | 1024 | data-parallel |  | 1182.9 | 1191.6 | 1182.9 | 1191.6 | 1148.7 | 1157.1 | torch | 1151.5 | 1155.3 | 1159.8 | 1157.5 | hipBLASLt contiguous 1159.8 |
| 16384x7168x1024 | bfloat16 | 1792 | data-parallel |  | 1227.3 | 1227.6 | 1227.3 | 1227.6 | 1191.5 | 1190.3 | asm | 1136.9 | 1142.8 | 1210.0 | 1197.8 | hipBLASLt contiguous 1210.0 |
| 16384x8192x1024 | bfloat16 | 2048 | data-parallel |  | 1226.9 | 1231.3 | 1226.9 | 1231.3 | 1178.9 | 1188.2 | torch | 1196.4 | 1192.4 | 1201.1 | 1199.3 | hipBLASLt contiguous 1201.1 |
| 32768x7168x1024 | bfloat16 | 3584 | data-parallel |  | 1249.8 | 1251.7 | 1249.8 | 1251.7 | 1212.6 | 1212.2 | torch | 1224.4 | 1221.6 | 1236.9 | 1223.4 | hipBLASLt contiguous 1236.9 |
| 32768x8192x1024 | bfloat16 | 4096 | data-parallel |  | 1250.7 | 1256.8 | 1250.7 | 1256.8 | 1206.1 | 1206.2 | torch | 1216.8 | 1213.2 | 1223.9 | 1221.3 | hipBLASLt contiguous 1223.9 |
| 512x5120x1280 | bfloat16 | 40 | data-parallel | yes | 202.2 | 229.1 | 194.1 | 223.8 | 203.0 | 228.8 | flydsl | 444.7 | 261.5 (copy) | 424.2 | 437.3 | AITER flydsl contiguous 444.7 |
| 1024x5120x1280 | bfloat16 | 80 | data-parallel | yes | 392.6 | 439.6 | 377.6 | 430.1 | 394.6 | 439.6 | torch | 603.5 | 626.7 | 607.6 | 635.3 | hipBLASLt padded 635.3 |
| 2048x5120x1280 | bfloat16 | 160 | data-parallel | yes | 756.8 | 833.3 | 726.5 | 817.7 | 759.6 | 831.6 | torch | 840.4 | 886.7 | 850.8 | 889.7 | hipBLASLt padded 889.7 |
| 4096x5120x1280 | bfloat16 | 320 | data-parallel |  | 800.0 | 867.8 | 800.0 | 867.8 | 795.6 | 859.4 | torch | 876.2 | 926.2 | 880.8 | 926.1 | AITER torch padded 926.2 |
| 8192x5120x1280 | bfloat16 | 640 | data-parallel |  | 1041.1 | 1108.2 | 1041.1 | 1108.2 | 1017.6 | 1081.3 | flydsl | 991.2 | 927.0 (copy) | 1074.3 | 1082.2 | hipBLASLt padded 1082.2 |
| 16384x5120x1280 | bfloat16 | 1280 | data-parallel |  | 1233.0 | 1281.6 | 1233.0 | 1281.6 | 1202.5 | 1250.3 | flydsl | 1176.1 | 1088.7 (copy) | 1227.0 | 1244.4 | hipBLASLt padded 1244.4 |
| 32768x5120x1280 | bfloat16 | 2560 | data-parallel |  | 1262.9 | 1302.8 | 1262.9 | 1302.8 | 1235.8 | 1271.6 | torch | 1253.0 | 1267.9 | 1262.8 | 1267.5 | AITER torch padded 1267.9 |
| 512x1536x1536 | bfloat16 | 12 | one-tile reduce-scatter, n = 8 |  | 89.0 | 94.2 | 66.6 | 71.1 | 69.7 | 72.0 | flydsl | 265.7 | 140.1 (copy) | 239.4 | 237.2 | AITER flydsl contiguous 265.7 |
| 512x2048x1536 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 |  | 114.3 | 118.0 | 88.3 | 93.9 | 92.4 | 94.9 | torch | 310.5 | 308.3 | 312.0 | 308.7 | hipBLASLt contiguous 312.0 |
| 512x2304x1536 | bfloat16 | 18 | one-tile reduce-scatter, n = 7 |  | 119.1 | 122.9 | 99.9 | 105.7 | 103.9 | 106.7 | flydsl | 320.5 | 187.9 (copy) | 312.7 | 312.5 | AITER flydsl contiguous 320.5 |
| 512x4096x1536 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 196.1 | 202.2 | 177.7 | 188.3 | 186.2 | 190.7 | torch | 444.3 | 433.5 | 447.1 | 433.6 | hipBLASLt contiguous 447.1 |
| 512x6144x1536 | bfloat16 | 48 | data-parallel | yes | 276.9 | 284.1 | 264.3 | 279.1 | 276.6 | 283.8 | flydsl | 539.4 | 329.6 (copy) | 543.3 | 539.0 | hipBLASLt contiguous 543.3 |
| 512x7168x1536 | bfloat16 | 56 | data-parallel | yes | 319.2 | 328.1 | 305.1 | 322.5 | 319.8 | 328.3 | flydsl | 586.6 | 354.2 (copy) | 503.0 | 492.2 | AITER flydsl contiguous 586.6 |
| 1024x1536x1536 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 |  | 151.7 | 158.5 | 133.4 | 141.4 | 139.9 | 143.6 | flydsl | 399.7 | 246.6 (copy) | 390.4 | 390.4 | AITER flydsl contiguous 399.7 |
| 1024x2048x1536 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 196.8 | 202.5 | 177.2 | 187.2 | 185.0 | 189.4 | torch | 454.5 | 450.3 | 458.3 | 451.2 | hipBLASLt contiguous 458.3 |
| 1024x2304x1536 | bfloat16 | 36 | one-tile owner, n = 2 |  | 211.1 | 213.9 | 197.8 | 209.8 | 207.3 | 212.3 | torch | 456.4 | 457.6 | 458.1 | 459.7 | hipBLASLt padded 459.7 |
| 1024x3072x1536 | bfloat16 | 48 | data-parallel | yes | 275.4 | 282.6 | 263.6 | 278.1 | 277.0 | 282.8 | flydsl | 554.1 | 353.7 (copy) | 552.9 | 552.0 | AITER flydsl contiguous 554.1 |
| 1024x4096x1536 | bfloat16 | 64 | data-parallel | yes | 363.3 | 373.8 | 348.6 | 369.9 | 363.2 | 373.5 | torch | 560.3 | 570.4 | 561.9 | 568.0 | AITER torch padded 570.4 |
| 1024x7168x1536 | bfloat16 | 112 | data-parallel | yes | 616.4 | 633.2 | 593.1 | 625.0 | 619.9 | 633.4 | torch | 776.0 | 790.0 | 782.8 | 795.9 | hipBLASLt padded 795.9 |
| 2048x1536x1536 | bfloat16 | 48 | data-parallel | yes | 280.7 | 286.3 | 266.0 | 281.5 | 281.6 | 287.5 | torch | 550.1 | 546.4 | 555.1 | 549.6 | hipBLASLt contiguous 555.1 |
| 2048x2048x1536 | bfloat16 | 64 | data-parallel | yes | 367.2 | 376.5 | 350.6 | 370.6 | 368.7 | 376.3 | torch | 561.9 | 568.6 | 563.8 | 562.5 | AITER torch padded 568.6 |
| 2048x4096x1536 | bfloat16 | 128 | data-parallel | yes | 692.1 | 712.3 | 664.9 | 701.3 | 693.9 | 714.0 | torch | 844.6 | 864.3 | 851.0 | 876.5 | hipBLASLt padded 876.5 |
| 3072x2048x1536 | bfloat16 | 96 | data-parallel | yes | 535.5 | 549.4 | 512.0 | 542.8 | 538.3 | 548.4 | opus | 686.4 | 704.6 | 774.5 | 755.1 | hipBLASLt contiguous 774.5 |
| 3072x4096x1536 | bfloat16 | 192 | data-parallel | yes | 983.2 | 1000.2 | 953.5 | 983.5 | 988.5 | 998.4 | opus | 985.2 | 1028.6 | 957.7 | 960.9 | AITER opus padded 1028.6 |
| 4096x1536x1536 | bfloat16 | 96 | data-parallel | yes | 533.2 | 549.5 | 511.0 | 542.3 | 538.1 | 546.4 | torch | 755.2 | 747.2 | 767.3 | 752.1 | hipBLASLt contiguous 767.3 |
| 4096x2048x1536 | bfloat16 | 128 | data-parallel | yes | 684.8 | 711.7 | 662.4 | 704.4 | 690.7 | 706.8 | opus | 854.8 | 874.1 | 846.3 | 870.8 | AITER opus padded 874.1 |
| 4096x3072x1536 | bfloat16 | 192 | data-parallel | yes | 981.5 | 1002.2 | 957.6 | 982.8 | 993.1 | 1000.7 | flydsl | 940.0 | 713.1 (copy) | 1011.0 | 1025.4 | hipBLASLt padded 1025.4 |
| 4096x4096x1536 | bfloat16 | 256 | data-parallel | yes | 1186.0 | 1218.2 | 1173.4 | 1204.7 | 1199.2 | 1216.1 | torch | 1172.5 | 1182.1 | 1180.1 | 1186.1 | hipBLASLt padded 1186.1 |
| 5120x2048x1536 | bfloat16 | 160 | data-parallel | yes | 842.0 | 871.0 | 811.4 | 856.7 | 844.1 | 866.0 | torch | 948.1 | 969.4 | 957.1 | 982.4 | hipBLASLt padded 982.4 |
| 5120x4096x1536 | bfloat16 | 320 | data-parallel |  | 949.5 | 983.0 | 949.5 | 983.0 | 946.2 | 963.7 | torch | 1001.4 | 1021.0 | 1008.4 | 1022.3 | hipBLASLt padded 1022.3 |
| 7680x2048x1536 | bfloat16 | 240 | data-parallel | yes | 1136.8 | 1161.6 | 1110.9 | 1139.1 | 1144.4 | 1162.3 | torch | 953.2 | 967.1 | 957.0 | 965.1 | AITER torch padded 967.1 |
| 7680x4096x1536 | bfloat16 | 480 | data-parallel |  | 1197.9 | 1226.8 | 1197.9 | 1226.8 | 1196.4 | 1213.7 | torch | 1030.5 | 1038.2 | 1039.9 | 1033.9 | hipBLASLt contiguous 1039.9 |
| 8192x1536x1536 | bfloat16 | 192 | data-parallel | yes | 978.0 | 998.9 | 944.7 | 980.1 | 978.2 | 999.6 | torch | 993.1 | 1028.8 | 996.4 | 1030.0 | hipBLASLt padded 1030.0 |
| 10240x7168x1536 | bfloat16 | 1120 | data-parallel |  | 1236.1 | 1257.0 | 1236.1 | 1257.0 | 1215.0 | 1226.6 | asm | 1145.2 | 1152.1 | 1138.8 | 1129.4 | AITER asm padded 1152.1 |
| 16384x1536x1536 | bfloat16 | 384 | data-parallel |  | 1049.1 | 1085.4 | 1049.1 | 1085.4 | 1053.5 | 1081.6 | torch | 1054.5 | 1075.3 | 1064.7 | 1086.2 | hipBLASLt padded 1086.2 |
| 16384x2048x1536 | bfloat16 | 512 | data-parallel |  | 1215.8 | 1254.1 | 1215.8 | 1254.1 | 1216.7 | 1240.8 | torch | 1213.5 | 1223.3 | 1221.4 | 1226.7 | hipBLASLt padded 1226.7 |
| 16384x3072x1536 | bfloat16 | 768 | data-parallel |  | 1262.4 | 1290.4 | 1262.4 | 1290.4 | 1252.5 | 1268.6 | torch | 1242.8 | 1251.6 | 1251.7 | 1252.8 | hipBLASLt padded 1252.8 |
| 16384x4096x1536 | bfloat16 | 1024 | data-parallel |  | 1266.7 | 1313.4 | 1266.7 | 1313.4 | 1244.1 | 1276.1 | torch | 1256.8 | 1275.1 | 1269.7 | 1280.2 | hipBLASLt padded 1280.2 |
| 32768x1536x1536 | bfloat16 | 768 | data-parallel |  | 1250.7 | 1278.6 | 1250.7 | 1278.6 | 1233.8 | 1250.5 | flydsl | 1203.3 | 937.2 (copy) | 1228.4 | 1218.2 | hipBLASLt contiguous 1228.4 |
| 512x7168x1792 | bfloat16 | 56 | one-tile owner, n = 2 |  | 358.7 | 349.6 | 336.5 | 323.4 | 344.2 | 330.9 | flydsl | 619.6 | 380.5 (copy) | 504.9 | 500.6 | AITER flydsl contiguous 619.6 |
| 1024x7168x1792 | bfloat16 | 112 | data-parallel | yes | 669.1 | 647.2 | 655.9 | 632.1 | 667.5 | 646.2 | flydsl | 781.2 | 545.8 (copy) | 829.5 | 824.6 | hipBLASLt contiguous 829.5 |
| 2048x7168x1792 | bfloat16 | 224 | data-parallel | yes | 1168.6 | 1163.9 | 1153.1 | 1144.5 | 1173.9 | 1163.9 | flydsl | 1105.5 | 829.7 (copy) | 1073.5 | 1070.2 | AITER flydsl contiguous 1105.5 |
| 4096x7168x1792 | bfloat16 | 448 | data-parallel |  | 1213.5 | 1210.3 | 1213.5 | 1210.3 | 1206.4 | 1198.5 | flydsl | 1153.8 | 982.8 (copy) | 1135.0 | 1122.4 | AITER flydsl contiguous 1153.8 |
| 8192x7168x1792 | bfloat16 | 896 | data-parallel |  | 1246.3 | 1240.0 | 1246.3 | 1240.0 | 1223.9 | 1210.0 | asm | 1192.9 | 1189.7 | 1173.0 | 1155.2 | AITER asm contiguous 1192.9 |
| 16384x7168x1792 | bfloat16 | 1792 | data-parallel |  | 1359.2 | 1364.9 | 1359.2 | 1364.9 | 1341.8 | 1335.8 | torch | 1337.2 | 1342.9 | 1353.6 | 1343.9 | hipBLASLt contiguous 1353.6 |
| 32768x7168x1792 | bfloat16 | 3584 | data-parallel |  | 1381.7 | 1388.2 | 1381.7 | 1388.2 | 1353.8 | 1351.8 | torch | 1355.2 | 1359.4 | 1363.1 | 1359.6 | hipBLASLt contiguous 1363.1 |
| 512x4096x2048 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 255.1 | 249.0 | 211.1 | 197.4 | 213.9 | 200.2 | torch | 488.9 | 479.5 | 491.1 | 480.7 | hipBLASLt contiguous 491.1 |
| 512x6144x2048 | float32 | 48 | one-tile owner, n = 2 |  | 312.4 | 299.9 | 281.6 | 266.1 | 288.4 | 272.7 | asm | 331.7 | 330.4 | 563.5 | 562.6 | hipBLASLt contiguous 563.5 |
| 512x6144x2048 | bfloat16 | 48 | one-tile owner, n = 2 |  | 344.8 | 331.0 | 314.1 | 294.3 | 320.5 | 298.8 | flydsl | 602.9 | 366.8 (copy) | 595.6 | 588.8 | AITER flydsl contiguous 602.9 |
| 512x8192x2048 | bfloat16 | 64 | one-tile owner, n = 2 |  | 442.6 | 428.6 | 406.6 | 387.2 | 411.4 | 392.3 | flydsl | 703.4 | 414.6 (copy) | 577.1 | 570.6 | AITER flydsl contiguous 703.4 |
| 512x16384x2048 | bfloat16 | 128 | data-parallel | yes | 746.7 | 746.4 | 740.0 | 737.3 | 752.0 | 745.8 | flydsl | 715.7 | 485.7 (copy) | 874.5 | 863.3 | hipBLASLt contiguous 874.5 |
| 1024x4096x2048 | bfloat16 | 64 | one-tile owner, n = 2 |  | 448.1 | 432.9 | 414.9 | 390.7 | 416.2 | 396.1 | torch | 615.3 | 604.6 | 618.2 | 602.2 | hipBLASLt contiguous 618.2 |
| 1024x6144x2048 | float32 | 96 | data-parallel | yes | 549.1 | 528.5 | 533.1 | 515.0 | 548.1 | 528.0 | opus | 556.8 | 536.5 | 783.9 | 775.8 | hipBLASLt contiguous 783.9 |
| 1024x6144x2048 | bfloat16 | 96 | data-parallel | yes | 605.1 | 591.6 | 592.7 | 580.1 | 608.3 | 590.0 | flydsl | 740.2 | 525.4 (copy) | 810.7 | 788.3 | hipBLASLt contiguous 810.7 |
| 1024x8192x2048 | bfloat16 | 128 | data-parallel | yes | 770.0 | 770.5 | 760.4 | 759.1 | 770.0 | 770.1 | flydsl | 875.7 | 658.8 (copy) | 923.4 | 915.4 | hipBLASLt contiguous 923.4 |
| 1024x16384x2048 | bfloat16 | 256 | data-parallel | yes | 1259.5 | 1279.3 | 1252.6 | 1256.9 | 1265.7 | 1278.4 | flydsl | 744.7 | 667.2 (copy) | 1229.1 | 1233.0 | hipBLASLt padded 1233.0 |
| 2048x4096x2048 | bfloat16 | 128 | data-parallel | yes | 779.8 | 776.1 | 773.1 | 769.1 | 782.5 | 777.2 | torch | 937.4 | 924.6 | 947.3 | 933.1 | hipBLASLt contiguous 947.3 |
| 2048x6144x2048 | float32 | 192 | data-parallel | yes | 970.1 | 961.0 | 942.5 | 934.3 | 967.1 | 958.6 | asm | 919.6 | 925.5 | 1018.7 | 1019.0 | hipBLASLt padded 1019.0 |
| 2048x6144x2048 | bfloat16 | 192 | data-parallel | yes | 1097.1 | 1102.8 | 1082.0 | 1082.7 | 1102.2 | 1104.4 | flydsl | 1042.6 | 792.8 (copy) | 1104.4 | 1099.6 | hipBLASLt contiguous 1104.4 |
| 2048x8192x2048 | bfloat16 | 256 | data-parallel | yes | 1280.5 | 1301.5 | 1269.6 | 1274.8 | 1283.0 | 1299.5 | flydsl | 1215.5 | 920.7 (copy) | 1235.4 | 1255.1 | hipBLASLt padded 1255.1 |
| 2048x16384x2048 | bfloat16 | 512 | data-parallel |  | 1324.2 | 1337.8 | 1324.2 | 1337.8 | 1311.9 | 1329.2 | flydsl | 1044.7 | 878.9 (copy) | 1287.4 | 1289.1 | hipBLASLt padded 1289.1 |
| 3072x4096x2048 | bfloat16 | 192 | data-parallel | yes | 1088.1 | 1097.5 | 1083.3 | 1076.8 | 1097.7 | 1090.9 | torch | 1099.1 | 1099.9 | 1104.5 | 1103.1 | hipBLASLt contiguous 1104.5 |
| 4096x4096x2048 | bfloat16 | 256 | data-parallel | yes | 1281.4 | 1304.9 | 1279.4 | 1281.9 | 1293.4 | 1299.7 | torch | 1258.9 | 1264.4 | 1268.8 | 1268.3 | hipBLASLt contiguous 1268.8 |
| 4096x6144x2048 | float32 | 384 | data-parallel |  | 999.1 | 986.2 | 999.1 | 986.2 | 1001.7 | 990.6 | asm | 955.6 | 944.1 | 1067.1 | 1056.9 | hipBLASLt contiguous 1067.1 |
| 4096x6144x2048 | bfloat16 | 384 | data-parallel |  | 1152.2 | 1141.1 | 1152.2 | 1141.1 | 1133.5 | 1129.5 | flydsl | 890.4 | 860.8 (copy) | 1144.8 | 1137.4 | hipBLASLt contiguous 1144.8 |
| 4096x8192x2048 | bfloat16 | 512 | data-parallel |  | 1323.9 | 1339.9 | 1323.9 | 1339.9 | 1319.6 | 1329.4 | flydsl | 1273.6 | 1075.7 (copy) | 1310.7 | 1311.9 | hipBLASLt padded 1311.9 |
| 4096x16384x2048 | bfloat16 | 1024 | data-parallel |  | 1350.9 | 1371.1 | 1350.9 | 1371.1 | 1339.8 | 1353.5 | flydsl | 855.3 | 859.0 (copy) | 1345.1 | 1335.4 | hipBLASLt contiguous 1345.1 |
| 5120x4096x2048 | bfloat16 | 320 | one-tile owner, n = 2 |  | 1053.1 | 1056.9 | 1042.4 | 1034.3 | 1031.5 | 1029.4 | torch | 1129.4 | 1099.3 | 1133.5 | 1090.8 | hipBLASLt contiguous 1133.5 |
| 7680x4096x2048 | bfloat16 | 480 | data-parallel |  | 1296.2 | 1305.4 | 1296.2 | 1305.4 | 1294.4 | 1292.7 | torch | 1103.6 | 1095.8 | 1109.8 | 1096.0 | hipBLASLt contiguous 1109.8 |
| 8192x6144x2048 | float32 | 768 | data-parallel |  | 1192.9 | 1194.9 | 1192.9 | 1194.9 | 1188.2 | 1187.7 | asm | 1149.5 | 1150.5 | 1110.8 | 1111.0 | AITER asm padded 1150.5 |
| 8192x6144x2048 | bfloat16 | 768 | data-parallel |  | 1367.0 | 1364.0 | 1367.0 | 1364.0 | 1355.4 | 1347.6 | flydsl | 834.0 | 773.8 (copy) | 1338.7 | 1334.2 | hipBLASLt contiguous 1338.7 |
| 8192x8192x2048 | bfloat16 | 1024 | data-parallel |  | 1372.0 | 1381.7 | 1372.0 | 1381.7 | 1352.0 | 1355.5 | asm | 1313.9 | 1317.5 | 1356.0 | 1348.1 | hipBLASLt contiguous 1356.0 |
| 8192x16384x2048 | bfloat16 | 2048 | data-parallel |  | 1374.4 | 1387.4 | 1374.4 | 1387.4 | 1345.1 | 1364.5 | triton | 1019.3 | 1028.3 | 1381.4 | 1369.0 | hipBLASLt contiguous 1381.4 |
| 16384x4096x2048 | bfloat16 | 1024 | data-parallel |  | 1366.6 | 1380.4 | 1366.6 | 1380.4 | 1343.4 | 1362.5 | torch | 1324.6 | 1328.6 | 1335.5 | 1325.6 | hipBLASLt contiguous 1335.5 |
| 16384x6144x2048 | float32 | 1536 | data-parallel |  | 1213.5 | 1220.1 | 1213.5 | 1220.1 | 1201.7 | 1205.3 | asm | 1168.4 | 1169.6 | 1140.5 | 1139.8 | AITER asm padded 1169.6 |
| 16384x6144x2048 | bfloat16 | 1536 | data-parallel |  | 1381.6 | 1394.2 | 1381.6 | 1394.2 | 1364.4 | 1367.8 | torch | 1353.7 | 1362.8 | 1369.8 | 1362.9 | hipBLASLt contiguous 1369.8 |
| 16384x8192x2048 | bfloat16 | 2048 | data-parallel |  | 1388.8 | 1416.3 | 1388.8 | 1416.3 | 1372.9 | 1378.0 | torch | 1367.9 | 1370.0 | 1383.8 | 1374.0 | hipBLASLt contiguous 1383.8 |
| 16384x16384x2048 | bfloat16 | 4096 | data-parallel |  | 1398.8 | 1398.1 | 1398.8 | 1398.1 | 1367.3 | 1366.6 | flydsl | 959.0 | 917.5 (copy) | 1399.9 | 1389.1 | hipBLASLt contiguous 1399.9 |
| 32768x6144x2048 | bfloat16 | 3072 | data-parallel |  | 1410.7 | 1417.6 | 1410.7 | 1417.6 | 1388.6 | 1384.5 | flydsl | 990.2 | 935.2 (copy) | 1389.0 | 1381.1 | hipBLASLt contiguous 1389.0 |
| 32768x8192x2048 | bfloat16 | 4096 | data-parallel |  | 1420.2 | 1427.4 | 1420.2 | 1427.4 | 1391.9 | 1387.2 | torch | 1382.8 | 1386.8 | 1394.0 | 1391.7 | hipBLASLt contiguous 1394.0 |
| 32768x16384x2048 | bfloat16 | 8192 | data-parallel |  | 1403.0 | 1401.8 | 1403.0 | 1401.8 | 1363.0 | 1365.1 | flydsl | 828.8 | 816.1 (copy) | 1406.3 | 1400.9 | hipBLASLt contiguous 1406.3 |
| 512x6144x3072 | float32 | 48 | one-tile owner, n = 2 |  | 365.3 | 377.9 | 283.2 | 298.5 | 292.6 | 299.9 | opus | 415.2 | 425.0 | 623.5 | 630.3 | hipBLASLt padded 630.3 |
| 512x6144x3072 | bfloat16 | 48 | one-tile owner, n = 2 |  | 394.7 | 410.3 | 303.1 | 322.8 | 314.6 | 321.4 | flydsl | 640.4 | 420.4 (copy) | 657.2 | 652.0 | hipBLASLt contiguous 657.2 |
| 1024x6144x3072 | float32 | 96 | one-tile owner, n = 2-3 |  | 605.6 | 622.8 | 553.8 | 584.2 | 577.1 | 588.2 | opus | 634.3 | 654.0 | 846.7 | 854.3 | hipBLASLt padded 854.3 |
| 1024x6144x3072 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 651.0 | 671.7 | 599.2 | 639.2 | 617.3 | 637.6 | flydsl | 786.0 | 591.0 (copy) | 846.2 | 829.3 | hipBLASLt contiguous 846.2 |
| 2048x6144x3072 | float32 | 192 | data-parallel | yes | 1044.9 | 1066.2 | 1018.8 | 1050.1 | 1045.3 | 1061.3 | asm | 1012.0 | 1018.5 | 1116.6 | 1128.4 | hipBLASLt padded 1128.4 |
| 2048x6144x3072 | bfloat16 | 192 | data-parallel | yes | 1164.9 | 1197.6 | 1132.6 | 1177.5 | 1162.9 | 1196.1 | flydsl | 1116.8 | 878.9 (copy) | 1178.6 | 1183.8 | hipBLASLt padded 1183.8 |
| 4096x6144x3072 | float32 | 384 | data-parallel |  | 1075.0 | 1108.2 | 1075.0 | 1108.2 | 1086.6 | 1115.0 | asm | 1046.4 | 1060.9 | 1155.7 | 1157.1 | hipBLASLt padded 1157.1 |
| 4096x6144x3072 | bfloat16 | 384 | data-parallel |  | 1208.7 | 1236.4 | 1208.7 | 1236.4 | 1205.9 | 1222.2 | flydsl | 924.7 | 912.7 (copy) | 1216.3 | 1222.8 | hipBLASLt padded 1222.8 |
| 8192x6144x3072 | float32 | 768 | data-parallel |  | 1285.1 | 1308.0 | 1285.1 | 1308.0 | 1288.1 | 1305.4 | asm | 1256.3 | 1265.3 | 1209.0 | 1211.2 | AITER asm padded 1265.3 |
| 8192x6144x3072 | bfloat16 | 768 | data-parallel |  | 1426.1 | 1467.2 | 1426.1 | 1467.2 | 1424.4 | 1448.1 | flydsl | 779.8 | 772.2 (copy) | 1410.1 | 1409.9 | hipBLASLt contiguous 1410.1 |
| 16384x6144x3072 | float32 | 1536 | data-parallel |  | 1312.3 | 1334.2 | 1312.3 | 1334.2 | 1306.1 | 1318.0 | asm | 1278.0 | 1286.3 | 1236.9 | 1233.7 | AITER asm padded 1286.3 |
| 16384x6144x3072 | bfloat16 | 1536 | data-parallel |  | 1465.1 | 1501.6 | 1465.1 | 1501.6 | 1458.9 | 1481.1 | torch | 1417.3 | 1433.2 | 1438.2 | 1433.4 | hipBLASLt contiguous 1438.2 |
| 32768x6144x3072 | bfloat16 | 3072 | data-parallel |  | 1487.8 | 1515.4 | 1487.8 | 1515.4 | 1473.7 | 1486.1 | flydsl | 858.8 | 801.8 (copy) | 1441.6 | 1451.0 | hipBLASLt padded 1451.0 |
| 512x5120x3200 | bfloat16 | 40 | one-tile owner, n = 2 |  | 358.3 | 358.5 | 270.2 | 270.2 | 268.8 | 269.1 | flydsl | 577.6 | 577.7 | 563.3 | 563.3 | AITER flydsl padded 577.7 |
| 1024x5120x3200 | bfloat16 | 80 | one-tile owner, n = 2 |  | 687.7 | 692.3 | 533.6 | 536.2 | 531.5 | 532.1 | torch | 822.2 | 823.8 | 825.9 | 825.1 | hipBLASLt contiguous 825.9 |
| 2048x5120x3200 | bfloat16 | 160 | data-parallel | yes | 1041.5 | 1054.6 | 1051.5 | 1050.2 | 1047.9 | 1048.8 | torch | 1096.1 | 1095.6 | 1099.5 | 1094.0 | hipBLASLt contiguous 1099.5 |
| 4096x5120x3200 | bfloat16 | 320 | one-tile owner, n = 2 |  | 1161.7 | 1172.4 | 1073.4 | 1072.6 | 1058.0 | 1059.3 | torch | 1126.1 | 1130.7 | 1135.9 | 1128.5 | hipBLASLt contiguous 1135.9 |
| 8192x5120x3200 | bfloat16 | 640 | data-parallel |  | 1325.5 | 1334.6 | 1325.5 | 1334.6 | 1315.0 | 1319.7 | asm | 1257.0 | 1261.2 | 1278.7 | 1263.8 | hipBLASLt contiguous 1278.7 |
| 16384x5120x3200 | bfloat16 | 1280 | data-parallel |  | 1468.8 | 1485.5 | 1468.8 | 1485.5 | 1457.7 | 1462.8 | torch | 1419.8 | 1430.2 | 1432.2 | 1428.6 | hipBLASLt contiguous 1432.2 |
| 32768x5120x3200 | bfloat16 | 2560 | data-parallel |  | 1489.5 | 1507.9 | 1489.5 | 1507.9 | 1470.7 | 1479.5 | torch | 1436.6 | 1439.8 | 1444.0 | 1440.7 | hipBLASLt contiguous 1444.0 |
| 512x7168x3584 | bfloat16 | 56 | one-tile owner, n = 2 |  | 501.4 | 506.1 | 372.1 | 382.8 | 376.4 | 386.5 | flydsl | 711.2 | 473.0 (copy) | 591.8 | 597.1 | AITER flydsl contiguous 711.2 |
| 512x8192x3584 | bfloat16 | 64 | one-tile owner, n = 2 |  | 567.5 | 566.5 | 424.1 | 435.3 | 428.0 | 438.4 | flydsl | 669.8 | 463.5 (copy) | 653.7 | 653.6 | AITER flydsl contiguous 669.8 |
| 1024x7168x3584 | bfloat16 | 112 | one-tile owner, n = 2-3 |  | 845.5 | 837.1 | 728.9 | 758.0 | 738.7 | 769.6 | torch | 951.5 | 964.5 | 959.1 | 964.7 | hipBLASLt padded 964.7 |
| 1024x8192x3584 | bfloat16 | 128 | data-parallel | yes | 826.6 | 875.3 | 818.2 | 872.0 | 831.0 | 873.9 | flydsl | 785.3 | 630.7 (copy) | 1021.0 | 1026.9 | hipBLASLt padded 1026.9 |
| 2048x8192x3584 | bfloat16 | 256 | data-parallel | yes | 1400.8 | 1423.3 | 1397.1 | 1411.3 | 1398.4 | 1423.6 | flydsl | 1359.7 | 1020.0 (copy) | 1361.5 | 1373.0 | hipBLASLt padded 1373.0 |
| 4096x8192x3584 | bfloat16 | 512 | data-parallel |  | 1440.1 | 1465.0 | 1440.1 | 1465.0 | 1436.5 | 1460.2 | flydsl | 980.3 | 891.1 (copy) | 1418.3 | 1420.1 | hipBLASLt padded 1420.1 |
| 8192x8192x3584 | bfloat16 | 1024 | data-parallel |  | 1464.4 | 1497.0 | 1464.4 | 1497.0 | 1458.0 | 1480.1 | torch | 1433.4 | 1442.6 | 1448.5 | 1446.8 | hipBLASLt contiguous 1448.5 |
| 10240x7168x3584 | bfloat16 | 1120 | one-tile owner, n = 2-3 |  | 1413.3 | 1426.8 | 1402.4 | 1404.5 | 1384.3 | 1394.6 | flydsl | 781.2 | 734.8 (copy) | 1308.3 | 1302.9 | hipBLASLt contiguous 1308.3 |
| 16384x8192x3584 | bfloat16 | 2048 | data-parallel |  | 1493.7 | 1522.9 | 1493.7 | 1522.9 | 1480.7 | 1496.7 | flydsl | 807.9 | 799.7 (copy) | 1465.6 | 1465.7 | hipBLASLt padded 1465.7 |
| 32768x8192x3584 | bfloat16 | 4096 | data-parallel |  | 1511.6 | 1536.7 | 1511.6 | 1536.7 | 1494.7 | 1506.7 | flydsl | 858.5 | 828.9 (copy) | 1472.9 | 1470.5 | hipBLASLt contiguous 1472.9 |
| 512x1024x4096 | bfloat16 | 8 | one-tile reduce-scatter, n = 8 |  | 158.0 | 151.2 | 59.2 | 53.0 | 59.8 | 53.0 | flydsl | 281.5 | 186.5 (copy) | 263.7 | 258.5 | AITER flydsl contiguous 281.5 |
| 512x2048x4096 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 |  | 271.5 | 260.7 | 118.6 | 105.6 | 119.6 | 106.4 | flydsl | 438.0 | 282.2 (copy) | 424.0 | 418.0 | AITER flydsl contiguous 438.0 |
| 512x4096x4096 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 404.6 | 385.2 | 237.6 | 214.7 | 239.5 | 215.8 | flydsl | 597.4 | 398.2 (copy) | 621.3 | 615.2 | hipBLASLt contiguous 621.3 |
| 512x8192x4096 | bfloat16 | 64 | one-tile owner, n = 2 |  | 613.4 | 583.4 | 463.2 | 425.5 | 466.7 | 428.1 | flydsl | 803.9 | 507.9 (copy) | 680.0 | 676.4 | AITER flydsl contiguous 803.9 |
| 512x8704x4096 | bfloat16 | 68 | one-tile owner, n = 2 |  | 639.7 | 608.7 | 487.7 | 451.4 | 493.8 | 453.4 | torch | 746.3 | 756.1 | 747.7 | 756.7 | hipBLASLt padded 756.7 |
| 512x10240x4096 | bfloat16 | 80 | one-tile owner, n = 2 |  | 732.6 | 699.4 | 571.7 | 525.4 | 577.5 | 529.9 | torch | 850.0 | 855.4 | 854.2 | 863.3 | hipBLASLt padded 863.3 |
| 512x16384x4096 | bfloat16 | 128 | one-tile owner, n = 2 |  | 955.2 | 981.3 | 842.2 | 811.8 | 852.3 | 816.6 | flydsl | 630.1 | 413.4 (copy) | 686.5 | 720.0 | hipBLASLt padded 720.0 |
| 512x129280x4096 | bfloat16 | 1010 | data-parallel |  | 1372.6 | 1377.4 | 1372.6 | 1377.4 | 1365.1 | 1369.9 | asm | 1301.6 | 1320.7 | 1147.8 | 1181.2 | AITER asm padded 1320.7 |
| 768x1024x4096 | bfloat16 | 12 | one-tile reduce-scatter, n = 8 |  | 221.4 | 211.9 | 89.8 | 79.9 | 90.8 | 80.9 | torch | 370.5 | 365.8 | 370.9 | 365.0 | hipBLASLt contiguous 370.9 |
| 768x2048x4096 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 |  | 329.4 | 313.2 | 179.3 | 161.0 | 181.0 | 162.3 | flydsl | 526.8 | 333.6 (copy) | 499.5 | 479.7 | AITER flydsl contiguous 526.8 |
| 768x4096x4096 | bfloat16 | 48 | one-tile owner, n = 3 |  | 531.5 | 499.0 | 356.9 | 323.8 | 361.3 | 326.3 | flydsl | 693.8 | 480.9 (copy) | 658.6 | 662.3 | AITER flydsl contiguous 693.8 |
| 768x8192x4096 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 759.5 | 736.7 | 675.1 | 641.7 | 684.8 | 641.8 | torch | 878.9 | 856.4 | 881.9 | 855.9 | hipBLASLt contiguous 881.9 |
| 768x8704x4096 | bfloat16 | 102 | one-tile owner, n = 2-3 |  | 814.4 | 794.2 | 717.3 | 671.6 | 725.9 | 676.9 | flydsl | 866.1 | 619.2 (copy) | 844.3 | 818.1 | AITER flydsl contiguous 866.1 |
| 768x10240x4096 | bfloat16 | 120 | one-tile owner, n = 2-3 |  | 910.6 | 911.4 | 827.4 | 778.8 | 834.5 | 787.2 | flydsl | 970.9 | 714.0 (copy) | 927.1 | 905.1 | AITER flydsl contiguous 970.9 |
| 768x129280x4096 | bfloat16 | 1515 | data-parallel |  | 1433.9 | 1442.1 | 1433.9 | 1442.1 | 1417.2 | 1421.4 | torch | 1341.6 | 1344.4 | 1352.4 | 1348.5 | hipBLASLt contiguous 1352.4 |
| 1024x512x4096 | bfloat16 | 8 | one-tile reduce-scatter, n = 8 |  | 158.6 | 152.5 | 59.2 | 53.3 | 60.1 | 53.2 | flydsl | 281.4 | 181.7 (copy) | 261.4 | 256.6 | AITER flydsl contiguous 281.4 |
| 1024x1024x4096 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 |  | 273.9 | 264.4 | 119.9 | 108.0 | 121.4 | 107.9 | flydsl | 443.2 | 298.1 (copy) | 439.9 | 429.3 | AITER flydsl contiguous 443.2 |
| 1024x2048x4096 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 404.3 | 386.2 | 238.6 | 214.6 | 240.6 | 215.5 | flydsl | 600.1 | 423.0 (copy) | 505.9 | 505.6 | AITER flydsl contiguous 600.1 |
| 1024x4096x4096 | bfloat16 | 64 | one-tile owner, n = 2 |  | 623.5 | 594.5 | 465.5 | 429.3 | 469.5 | 432.0 | flydsl | 829.8 | 590.6 (copy) | 736.0 | 720.3 | AITER flydsl contiguous 829.8 |
| 1024x8192x4096 | bfloat16 | 128 | one-tile owner, n = 2 |  | 992.3 | 1017.2 | 876.0 | 848.7 | 879.2 | 854.9 | flydsl | 1012.7 | 787.7 (copy) | 697.7 | 697.1 | AITER flydsl contiguous 1012.7 |
| 1024x8704x4096 | bfloat16 | 136 | data-parallel | yes | 932.5 | 893.3 | 934.5 | 894.7 | 940.2 | 891.4 | torch | 996.6 | 976.7 | 1002.3 | 979.6 | hipBLASLt contiguous 1002.3 |
| 1024x10240x4096 | bfloat16 | 160 | data-parallel | yes | 1067.0 | 1047.1 | 1065.0 | 1037.1 | 1071.9 | 1047.7 | torch | 1124.5 | 1109.3 | 1125.9 | 1110.2 | hipBLASLt contiguous 1125.9 |
| 1024x16384x4096 | bfloat16 | 256 | data-parallel | yes | 1398.6 | 1425.0 | 1393.3 | 1406.2 | 1401.9 | 1427.7 | flydsl | 822.4 | 635.5 (copy) | 1343.6 | 1347.6 | hipBLASLt padded 1347.6 |
| 1024x129280x4096 | bfloat16 | 2020 | data-parallel |  | 1461.8 | 1465.8 | 1461.8 | 1465.8 | 1449.2 | 1445.3 | torch | 1365.8 | 1362.3 | 1373.0 | 1367.1 | hipBLASLt contiguous 1373.0 |
| 1280x512x4096 | bfloat16 | 10 | one-tile reduce-scatter, n = 8 |  | 190.3 | 182.5 | 74.7 | 67.0 | 75.8 | 67.1 | flydsl | 320.4 | 191.0 (copy) | 309.0 | 306.8 | AITER flydsl contiguous 320.4 |
| 1280x1024x4096 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 |  | 294.9 | 280.5 | 149.7 | 135.2 | 151.3 | 135.4 | torch | 485.1 | 472.6 | 487.9 | 473.5 | hipBLASLt contiguous 487.9 |
| 1280x2048x4096 | bfloat16 | 40 | one-tile owner, n = 3 |  | 455.0 | 420.6 | 299.2 | 270.4 | 302.4 | 272.2 | flydsl | 621.1 | 436.0 (copy) | 610.8 | 602.1 | AITER flydsl contiguous 621.1 |
| 1280x129280x4096 | bfloat16 | 2525 | data-parallel |  | 1428.4 | 1428.8 | 1428.4 | 1428.8 | 1413.8 | 1401.3 | torch | 1387.8 | 1387.4 | 1394.9 | 1389.1 | hipBLASLt contiguous 1394.9 |
| 1536x512x4096 | bfloat16 | 12 | one-tile reduce-scatter, n = 8 |  | 220.0 | 212.1 | 89.2 | 80.7 | 90.9 | 81.3 | torch | 366.8 | 358.7 | 367.3 | 359.0 | hipBLASLt contiguous 367.3 |
| 1536x1024x4096 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 |  | 334.1 | 316.7 | 179.6 | 161.8 | 182.8 | 162.8 | flydsl | 521.8 | 335.6 (copy) | 503.9 | 488.4 | AITER flydsl contiguous 521.8 |
| 1536x2048x4096 | bfloat16 | 48 | one-tile owner, n = 3 |  | 541.7 | 503.1 | 359.3 | 325.2 | 363.3 | 330.1 | flydsl | 694.7 | 498.1 (copy) | 680.9 | 684.8 | AITER flydsl contiguous 694.7 |
| 1536x129280x4096 | bfloat16 | 3030 | data-parallel |  | 1451.2 | 1458.4 | 1451.2 | 1458.4 | 1437.1 | 1431.7 | torch | 1390.2 | 1388.5 | 1397.1 | 1389.3 | hipBLASLt contiguous 1397.1 |
| 1792x512x4096 | bfloat16 | 14 | one-tile reduce-scatter, n = 8 |  | 247.8 | 236.6 | 104.5 | 93.8 | 106.0 | 94.9 | flydsl | 394.2 | 239.1 (copy) | 384.6 | 366.4 | AITER flydsl contiguous 394.2 |
| 1792x1024x4096 | bfloat16 | 28 | one-tile reduce-scatter, n = 4 |  | 366.3 | 349.5 | 211.0 | 190.9 | 213.7 | 191.7 | flydsl | 572.9 | 375.7 (copy) | 468.0 | 455.9 | AITER flydsl contiguous 572.9 |
| 1792x2048x4096 | bfloat16 | 56 | one-tile owner, n = 2 |  | 560.4 | 526.2 | 419.6 | 380.6 | 423.2 | 383.6 | flydsl | 750.0 | 558.2 (copy) | 667.8 | 652.7 | AITER flydsl contiguous 750.0 |
| 1792x129280x4096 | bfloat16 | 3535 | data-parallel |  | 1473.6 | 1471.6 | 1473.6 | 1471.6 | 1450.4 | 1448.8 | asm | 1433.3 | 1439.2 | 1356.5 | 1346.0 | AITER asm padded 1439.2 |
| 2048x512x4096 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 |  | 272.7 | 262.2 | 120.1 | 108.3 | 121.3 | 108.6 | flydsl | 426.1 | 273.2 (copy) | 430.2 | 414.4 | hipBLASLt contiguous 430.2 |
| 2048x1024x4096 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 406.0 | 392.0 | 240.3 | 218.8 | 243.5 | 220.2 | flydsl | 601.2 | 426.1 (copy) | 520.5 | 513.7 | AITER flydsl contiguous 601.2 |
| 2048x2048x4096 | bfloat16 | 64 | one-tile owner, n = 2 |  | 631.6 | 604.2 | 471.2 | 439.2 | 475.2 | 442.6 | flydsl | 838.6 | 605.8 (copy) | 747.5 | 731.4 | AITER flydsl contiguous 838.6 |
| 2048x4096x4096 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1035.7 | 1030.3 | 880.6 | 865.4 | 888.5 | 866.3 | opus | 1032.8 | 1035.6 | 779.6 | 683.0 | AITER opus padded 1035.6 |
| 2048x8192x4096 | bfloat16 | 256 | data-parallel | yes | 1421.1 | 1451.9 | 1422.1 | 1434.0 | 1427.9 | 1450.5 | flydsl | 1387.7 | 1058.4 (copy) | 1387.6 | 1398.7 | hipBLASLt padded 1398.7 |
| 2048x8704x4096 | bfloat16 | 272 | one-tile reduce-scatter, n = 8 |  | 1220.1 | 1223.1 | 987.9 | 986.6 | 982.7 | 984.9 | torch | 1075.2 | 1063.8 | 1088.0 | 1073.8 | hipBLASLt contiguous 1088.0 |
| 2048x10240x4096 | bfloat16 | 320 | one-tile owner, n = 2 |  | 1223.5 | 1218.8 | 1118.0 | 1092.1 | 1111.6 | 1090.8 | torch | 1155.9 | 1138.1 | 1164.1 | 1142.2 | hipBLASLt contiguous 1164.1 |
| 2048x16384x4096 | bfloat16 | 512 | data-parallel |  | 1453.4 | 1478.0 | 1453.4 | 1478.0 | 1445.1 | 1470.4 | flydsl | 1402.6 | 1117.9 (copy) | 1399.3 | 1400.2 | AITER flydsl contiguous 1402.6 |
| 2048x129280x4096 | bfloat16 | 4040 | data-parallel |  | 1494.8 | 1505.2 | 1494.8 | 1505.2 | 1481.5 | 1484.1 | flydsl | 1427.5 | 1151.8 (copy) | 1375.3 | 1368.0 | AITER flydsl contiguous 1427.5 |
| 2304x512x4096 | bfloat16 | 18 | one-tile reduce-scatter, n = 7 |  | 268.4 | 259.0 | 134.7 | 122.2 | 135.9 | 121.4 | flydsl | 351.1 | 262.3 (copy) | 442.0 | 430.3 | hipBLASLt contiguous 442.0 |
| 2304x1024x4096 | bfloat16 | 36 | one-tile owner, n = 3 |  | 413.3 | 382.2 | 269.5 | 244.9 | 273.2 | 245.3 | flydsl | 509.6 | 419.1 (copy) | 645.9 | 647.8 | hipBLASLt padded 647.8 |
| 2304x2048x4096 | bfloat16 | 72 | one-tile owner, n = 2 |  | 684.4 | 653.1 | 524.7 | 488.7 | 530.1 | 487.8 | opus | 682.3 | 644.1 | 898.8 | 871.5 | hipBLASLt contiguous 898.8 |
| 2560x512x4096 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 |  | 296.4 | 280.4 | 148.8 | 133.8 | 150.0 | 134.6 | flydsl | 385.4 | 282.8 (copy) | 485.7 | 469.5 | hipBLASLt contiguous 485.7 |
| 2560x1024x4096 | bfloat16 | 40 | one-tile owner, n = 3 |  | 457.2 | 419.7 | 298.7 | 270.7 | 302.4 | 271.8 | flydsl | 555.4 | 450.5 (copy) | 607.0 | 596.0 | hipBLASLt contiguous 607.0 |
| 2560x2048x4096 | bfloat16 | 80 | one-tile owner, n = 2 |  | 745.4 | 717.4 | 578.4 | 541.0 | 584.5 | 541.0 | opus | 761.9 | 716.6 | 882.2 | 889.4 | hipBLASLt padded 889.4 |
| 2816x512x4096 | bfloat16 | 22 | one-tile reduce-scatter, n = 5 |  | 307.5 | 289.3 | 163.4 | 147.1 | 165.1 | 147.8 | flydsl | 419.2 | 300.0 (copy) | 469.4 | 449.3 | hipBLASLt contiguous 469.4 |
| 2816x1024x4096 | bfloat16 | 44 | one-tile owner, n = 3 |  | 495.0 | 460.9 | 329.5 | 297.9 | 333.3 | 300.1 | flydsl | 600.1 | 477.1 (copy) | 641.9 | 630.5 | hipBLASLt contiguous 641.9 |
| 2816x2048x4096 | bfloat16 | 88 | one-tile owner, n = 2-3 |  | 794.4 | 779.1 | 632.7 | 589.3 | 638.5 | 593.4 | opus | 816.8 | 768.9 | 856.8 | 824.4 | hipBLASLt contiguous 856.8 |
| 3072x512x4096 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 |  | 329.8 | 311.3 | 178.6 | 160.2 | 180.7 | 161.0 | flydsl | 453.6 | 320.0 (copy) | 500.2 | 488.6 | hipBLASLt contiguous 500.2 |
| 3072x1024x4096 | bfloat16 | 48 | one-tile owner, n = 3 |  | 535.1 | 498.4 | 357.9 | 324.4 | 363.1 | 326.4 | flydsl | 644.0 | 495.2 (copy) | 674.7 | 673.4 | hipBLASLt contiguous 674.7 |
| 3072x2048x4096 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 771.3 | 760.2 | 675.3 | 648.7 | 691.2 | 652.1 | opus | 913.6 | 929.4 | 912.4 | 883.4 | AITER opus padded 929.4 |
| 3328x512x4096 | bfloat16 | 26 | one-tile reduce-scatter, n = 4 |  | 344.2 | 326.2 | 193.1 | 175.0 | 196.2 | 175.4 | flydsl | 482.6 | 337.4 (copy) | 446.1 | 437.3 | AITER flydsl contiguous 482.6 |
| 3328x1024x4096 | bfloat16 | 52 | one-tile owner, n = 2 |  | 519.7 | 484.7 | 384.3 | 352.0 | 393.2 | 354.1 | flydsl | 678.3 | 510.6 (copy) | 631.5 | 614.4 | AITER flydsl contiguous 678.3 |
| 3328x2048x4096 | bfloat16 | 104 | one-tile owner, n = 2-3 |  | 839.3 | 818.4 | 733.3 | 694.2 | 742.5 | 701.5 | opus | 938.4 | 892.1 | 809.0 | 779.5 | AITER opus contiguous 938.4 |
| 3584x512x4096 | bfloat16 | 28 | one-tile reduce-scatter, n = 4 |  | 364.9 | 346.3 | 207.0 | 189.0 | 211.7 | 189.0 | flydsl | 508.6 | 355.3 (copy) | 451.2 | 443.0 | AITER flydsl contiguous 508.6 |
| 3584x1024x4096 | bfloat16 | 56 | one-tile owner, n = 2 |  | 557.0 | 518.7 | 413.4 | 376.7 | 418.9 | 378.7 | flydsl | 708.4 | 524.0 (copy) | 668.3 | 652.8 | AITER flydsl contiguous 708.4 |
| 3584x2048x4096 | bfloat16 | 112 | one-tile owner, n = 2-3 |  | 919.7 | 887.3 | 780.7 | 749.1 | 792.0 | 754.5 | opus | 1001.1 | 962.8 | 910.7 | 858.6 | AITER opus contiguous 1001.1 |
| 3840x512x4096 | bfloat16 | 30 | one-tile reduce-scatter, n = 4 |  | 381.9 | 367.8 | 222.7 | 202.7 | 227.1 | 203.4 | flydsl | 538.1 | 367.9 (copy) | 491.1 | 480.3 | AITER flydsl contiguous 538.1 |
| 3840x1024x4096 | bfloat16 | 60 | one-tile owner, n = 2 |  | 592.4 | 554.6 | 440.2 | 405.3 | 445.3 | 407.5 | flydsl | 736.5 | 537.4 (copy) | 701.0 | 685.5 | AITER flydsl contiguous 736.5 |
| 3840x2048x4096 | bfloat16 | 120 | one-tile owner, n = 2-3 |  | 979.1 | 969.3 | 833.6 | 800.5 | 844.2 | 805.7 | opus | 1029.2 | 999.3 | 776.4 | 783.0 | AITER opus contiguous 1029.2 |
| 4096x512x4096 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 402.4 | 387.8 | 236.9 | 215.7 | 240.9 | 216.7 | flydsl | 585.3 | 390.6 (copy) | 503.9 | 495.5 | AITER flydsl contiguous 585.3 |
| 4096x1024x4096 | bfloat16 | 64 | one-tile owner, n = 2 |  | 622.0 | 595.3 | 463.3 | 433.8 | 472.0 | 433.2 | flydsl | 828.8 | 586.2 (copy) | 738.8 | 729.6 | AITER flydsl contiguous 828.8 |
| 4096x2048x4096 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1034.1 | 1032.0 | 878.5 | 859.6 | 884.8 | 859.4 | opus | 1087.8 | 1069.2 | 702.9 | 745.1 | AITER opus contiguous 1087.8 |
| 4096x4096x4096 | bfloat16 | 256 | data-parallel | yes | 1435.9 | 1447.3 | 1420.8 | 1443.1 | 1436.5 | 1445.5 | flydsl | 1381.5 | 1102.0 (copy) | 1391.0 | 1386.0 | hipBLASLt contiguous 1391.0 |
| 4096x8192x4096 | bfloat16 | 512 | data-parallel |  | 1479.0 | 1487.3 | 1479.0 | 1487.3 | 1475.7 | 1482.6 | asm | 1439.3 | 1452.5 | 1434.0 | 1431.9 | AITER asm padded 1452.5 |
| 4096x8704x4096 | bfloat16 | 544 | one-tile reduce-scatter, n = 4 |  | 1345.2 | 1344.7 | 1245.8 | 1220.6 | 1228.3 | 1209.7 | torch | 1206.7 | 1207.3 | 1223.0 | 1216.3 | hipBLASLt contiguous 1223.0 |
| 4096x10240x4096 | bfloat16 | 640 | one-tile owner, n = 2 |  | 1394.9 | 1401.3 | 1366.9 | 1361.3 | 1356.2 | 1351.2 | flydsl | 1314.5 | 1144.6 (copy) | 1275.4 | 1248.3 | AITER flydsl contiguous 1314.5 |
| 4096x16384x4096 | bfloat16 | 1024 | data-parallel |  | 1485.3 | 1512.0 | 1485.3 | 1512.0 | 1484.3 | 1497.3 | flydsl | 1451.8 | 1249.8 (copy) | 1449.3 | 1441.9 | AITER flydsl contiguous 1451.8 |
| 4096x129280x4096 | bfloat16 | 8080 | data-parallel |  | 1521.2 | 1526.8 | 1521.2 | 1526.8 | 1499.3 | 1499.9 | asm | 1463.8 | 1466.7 | 1402.1 | 1374.6 | AITER asm padded 1466.7 |
| 4352x512x4096 | bfloat16 | 34 | one-tile owner, n = 3 |  | 386.6 | 358.9 | 248.7 | 225.5 | 253.2 | 226.6 | flydsl | 467.4 | 338.6 (copy) | 522.9 | 511.8 | hipBLASLt contiguous 522.9 |
| 4352x1024x4096 | bfloat16 | 68 | one-tile owner, n = 2 |  | 643.3 | 618.2 | 491.7 | 455.4 | 496.6 | 454.9 | opus | 638.0 | 603.2 | 812.4 | 797.5 | hipBLASLt contiguous 812.4 |
| 4352x2048x4096 | bfloat16 | 136 | data-parallel | yes | 952.6 | 912.4 | 963.6 | 902.4 | 979.2 | 903.9 | torch | 1045.6 | 1045.2 | 1056.6 | 1046.8 | hipBLASLt contiguous 1056.6 |
| 5120x512x4096 | bfloat16 | 40 | one-tile owner, n = 3 |  | 448.5 | 413.8 | 294.2 | 266.3 | 298.4 | 267.2 | flydsl | 536.3 | 368.2 (copy) | 598.1 | 586.2 | hipBLASLt contiguous 598.1 |
| 5120x1024x4096 | bfloat16 | 80 | one-tile owner, n = 2 |  | 742.2 | 710.4 | 573.9 | 528.0 | 578.6 | 532.4 | opus | 766.9 | 724.2 | 879.8 | 879.0 | hipBLASLt contiguous 879.8 |
| 5120x2048x4096 | bfloat16 | 160 | data-parallel | yes | 1073.0 | 1053.2 | 1081.4 | 1043.2 | 1084.9 | 1050.6 | torch | 1165.9 | 1148.6 | 1173.9 | 1151.1 | hipBLASLt contiguous 1173.9 |
| 6144x512x4096 | bfloat16 | 48 | one-tile owner, n = 3 |  | 527.7 | 492.3 | 355.8 | 318.5 | 358.1 | 321.4 | flydsl | 683.8 | 437.2 (copy) | 662.4 | 660.5 | AITER flydsl contiguous 683.8 |
| 6144x1024x4096 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 761.6 | 750.9 | 670.6 | 629.8 | 688.0 | 636.8 | opus | 925.9 | 931.3 | 917.8 | 886.9 | AITER opus padded 931.3 |
| 6144x2048x4096 | bfloat16 | 192 | data-parallel | yes | 1223.5 | 1216.2 | 1217.8 | 1202.8 | 1223.7 | 1209.8 | opus | 1290.9 | 1266.5 | 1234.8 | 1231.9 | AITER opus contiguous 1290.9 |
| 6144x129280x4096 | bfloat16 | 12120 | one-tile owner, n = 2-3 |  | 1492.8 | 1507.5 | 1503.5 | 1503.1 | 1482.1 | 1474.7 | asm | 1474.8 | 1479.5 | 1402.4 | 1382.0 | AITER asm padded 1479.5 |
| 6400x512x4096 | bfloat16 | 50 | one-tile owner, n = 3 |  | 540.2 | 507.9 | 368.4 | 333.2 | 372.2 | 335.0 | flydsl | 605.8 | 428.8 (copy) | 606.3 | 588.7 | hipBLASLt contiguous 606.3 |
| 6400x1024x4096 | bfloat16 | 100 | one-tile owner, n = 2-3 |  | 790.3 | 782.5 | 699.1 | 655.3 | 709.5 | 661.1 | opus | 885.7 | 844.6 | 790.2 | 779.4 | AITER opus contiguous 885.7 |
| 6400x2048x4096 | bfloat16 | 200 | data-parallel | yes | 1247.5 | 1238.7 | 1243.5 | 1229.6 | 1251.8 | 1237.3 | torch | 1108.5 | 1094.8 | 1116.8 | 1097.4 | hipBLASLt contiguous 1116.8 |
| 6912x512x4096 | bfloat16 | 54 | one-tile owner, n = 2 |  | 525.5 | 493.3 | 391.6 | 358.4 | 398.3 | 360.0 | flydsl | 626.3 | 372.3 (copy) | 645.5 | 627.8 | hipBLASLt contiguous 645.5 |
| 6912x1024x4096 | bfloat16 | 108 | one-tile owner, n = 2-3 |  | 864.2 | 853.6 | 752.6 | 706.9 | 762.1 | 711.7 | opus | 933.2 | 894.0 | 716.9 | 716.0 | AITER opus contiguous 933.2 |
| 6912x2048x4096 | bfloat16 | 216 | data-parallel | yes | 1300.5 | 1299.0 | 1290.6 | 1289.3 | 1301.3 | 1297.2 | torch | 1243.5 | 1241.4 | 1253.9 | 1242.4 | hipBLASLt contiguous 1253.9 |
| 7168x512x4096 | bfloat16 | 56 | one-tile owner, n = 2 |  | 547.6 | 510.2 | 408.0 | 370.3 | 412.4 | 372.2 | flydsl | 638.3 | 400.4 (copy) | 666.1 | 645.5 | hipBLASLt contiguous 666.1 |
| 7168x1024x4096 | bfloat16 | 112 | one-tile owner, n = 2-3 |  | 921.7 | 891.0 | 771.5 | 731.3 | 785.3 | 736.2 | opus | 1001.3 | 964.9 | 1059.8 | 1031.0 | hipBLASLt contiguous 1059.8 |
| 7168x2048x4096 | bfloat16 | 224 | data-parallel | yes | 1331.0 | 1329.5 | 1323.1 | 1320.0 | 1331.1 | 1330.1 | torch | 1269.4 | 1274.0 | 1282.4 | 1278.8 | hipBLASLt contiguous 1282.4 |
| 7424x512x4096 | bfloat16 | 58 | one-tile owner, n = 2 |  | 562.9 | 527.3 | 420.3 | 383.8 | 425.8 | 387.0 | flydsl | 663.0 | 412.8 (copy) | 663.3 | 657.5 | hipBLASLt contiguous 663.3 |
| 7424x1024x4096 | bfloat16 | 116 | one-tile owner, n = 2-3 |  | 957.0 | 924.3 | 802.5 | 758.5 | 811.2 | 766.8 | opus | 977.9 | 942.1 | 764.5 | 757.6 | AITER opus contiguous 977.9 |
| 7424x2048x4096 | bfloat16 | 232 | data-parallel | yes | 1360.5 | 1358.1 | 1352.1 | 1342.7 | 1358.8 | 1356.6 | torch | 1131.6 | 1116.6 | 1134.1 | 1106.4 | hipBLASLt contiguous 1134.1 |
| 7680x512x4096 | bfloat16 | 60 | one-tile owner, n = 2 |  | 578.3 | 544.4 | 433.3 | 398.8 | 439.8 | 401.9 | flydsl | 657.6 | 464.2 (copy) | 680.0 | 671.4 | hipBLASLt contiguous 680.0 |
| 7680x1024x4096 | bfloat16 | 120 | one-tile owner, n = 2-3 |  | 976.2 | 962.3 | 828.4 | 786.5 | 837.3 | 793.8 | opus | 1034.4 | 1001.2 | 792.0 | 750.4 | AITER opus contiguous 1034.4 |
| 7680x2048x4096 | bfloat16 | 240 | data-parallel | yes | 1379.4 | 1385.2 | 1375.2 | 1372.0 | 1385.6 | 1385.1 | torch | 1156.3 | 1145.0 | 1165.2 | 1142.8 | hipBLASLt contiguous 1165.2 |
| 7936x512x4096 | bfloat16 | 62 | one-tile owner, n = 2 |  | 604.2 | 565.4 | 450.3 | 413.3 | 455.0 | 416.4 | flydsl | 672.3 | 439.6 (copy) | 696.2 | 676.9 | hipBLASLt contiguous 696.2 |
| 7936x1024x4096 | bfloat16 | 124 | one-tile owner, n = 2-3 |  | 1002.6 | 984.0 | 851.7 | 811.6 | 857.2 | 816.5 | opus | 1019.4 | 989.7 | 723.8 | 723.6 | AITER opus contiguous 1019.4 |
| 7936x2048x4096 | bfloat16 | 248 | data-parallel | yes | 1410.8 | 1416.2 | 1404.5 | 1403.8 | 1411.6 | 1416.7 | torch | 1370.3 | 1370.8 | 1375.7 | 1372.4 | hipBLASLt contiguous 1375.7 |
| 8192x512x4096 | bfloat16 | 64 | one-tile owner, n = 2 |  | 616.2 | 584.4 | 462.3 | 427.3 | 467.2 | 428.5 | flydsl | 784.7 | 505.8 (copy) | 712.9 | 706.2 | AITER flydsl contiguous 784.7 |
| 8192x1024x4096 | bfloat16 | 128 | one-tile owner, n = 2 |  | 985.2 | 1012.4 | 865.5 | 846.2 | 875.5 | 853.7 | opus | 1075.7 | 1064.3 | 715.0 | 711.6 | AITER opus contiguous 1075.7 |
| 8192x2048x4096 | bfloat16 | 256 | data-parallel | yes | 1433.0 | 1446.2 | 1429.0 | 1435.5 | 1436.9 | 1446.0 | torch | 1394.1 | 1395.6 | 1404.3 | 1399.7 | hipBLASLt contiguous 1404.3 |
| 8192x4096x4096 | bfloat16 | 512 | data-parallel |  | 1471.3 | 1487.2 | 1471.3 | 1487.2 | 1469.2 | 1482.3 | flydsl | 1419.6 | 1212.4 (copy) | 1444.8 | 1441.0 | hipBLASLt contiguous 1444.8 |
| 8192x8192x4096 | bfloat16 | 1024 | data-parallel |  | 1505.0 | 1514.5 | 1505.0 | 1514.5 | 1499.9 | 1497.9 | asm | 1466.2 | 1473.0 | 1472.9 | 1468.2 | AITER asm padded 1473.0 |
| 8192x8704x4096 | bfloat16 | 1088 | one-tile owner, n = 2 |  | 1436.3 | 1426.9 | 1404.2 | 1397.9 | 1377.2 | 1383.2 | asm | 1337.4 | 1334.7 | 1332.8 | 1326.2 | AITER asm contiguous 1337.4 |
| 8192x10240x4096 | bfloat16 | 1280 | data-parallel |  | 1510.1 | 1527.5 | 1510.1 | 1527.5 | 1508.2 | 1506.9 | torch | 1454.0 | 1463.5 | 1471.4 | 1464.0 | hipBLASLt contiguous 1471.4 |
| 8192x16384x4096 | bfloat16 | 2048 | data-parallel |  | 1510.6 | 1530.1 | 1510.6 | 1530.1 | 1498.1 | 1512.3 | torch | 1456.2 | 1462.7 | 1473.0 | 1459.1 | hipBLASLt contiguous 1473.0 |
| 8192x129280x4096 | bfloat16 | 16160 | one-tile reduce-scatter, n = 4 |  | 1545.0 | 1547.5 | 1547.3 | 1539.5 | 1518.0 | 1518.4 | flydsl | 1467.7 | 1375.4 (copy) | 1393.8 | 1371.8 | AITER flydsl contiguous 1467.7 |
| 10240x512x4096 | bfloat16 | 80 | one-tile owner, n = 2 |  | 724.9 | 697.0 | 571.4 | 520.7 | 573.0 | 523.7 | torch | 855.1 | 855.7 | 858.7 | 859.7 | hipBLASLt padded 859.7 |
| 10240x1024x4096 | bfloat16 | 160 | data-parallel | yes | 1093.3 | 1035.3 | 1089.1 | 1024.9 | 1098.0 | 1036.5 | torch | 1140.7 | 1132.7 | 1151.5 | 1133.8 | hipBLASLt contiguous 1151.5 |
| 10240x2048x4096 | bfloat16 | 320 | one-tile owner, n = 2 |  | 1221.4 | 1226.5 | 1122.5 | 1124.1 | 1115.8 | 1124.8 | torch | 1199.4 | 1199.3 | 1209.4 | 1197.1 | hipBLASLt contiguous 1209.4 |
| 10240x129280x4096 | bfloat16 | 20200 | data-parallel |  | 1496.6 | 1516.1 | 1496.6 | 1516.1 | 1497.5 | 1484.2 | asm | 1487.2 | 1485.8 | 1427.7 | 1374.7 | AITER asm contiguous 1487.2 |
| 12288x512x4096 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 737.5 | 721.1 | 658.9 | 619.8 | 676.8 | 622.7 | opus | 896.9 | 902.9 | 892.0 | 855.2 | AITER opus padded 902.9 |
| 12288x1024x4096 | bfloat16 | 192 | data-parallel | yes | 1201.2 | 1191.4 | 1196.2 | 1181.4 | 1203.3 | 1194.0 | opus | 1274.3 | 1248.4 | 1212.0 | 1218.6 | AITER opus contiguous 1274.3 |
| 12288x2048x4096 | bfloat16 | 384 | one-tile owner, n = 2 |  | 1312.3 | 1313.4 | 1265.5 | 1277.8 | 1272.0 | 1274.4 | opus | 1307.1 | 1295.5 | 1248.6 | 1264.2 | AITER opus contiguous 1307.1 |
| 12288x129280x4096 | bfloat16 | 24240 | data-parallel |  | 1530.7 | 1539.0 | 1530.7 | 1539.0 | 1510.8 | 1507.0 | asm | 1489.0 | 1485.4 | 1430.9 | 1370.4 | AITER asm contiguous 1489.0 |
| 14336x512x4096 | bfloat16 | 112 | one-tile owner, n = 2-3 |  | 891.9 | 879.7 | 753.1 | 711.7 | 772.5 | 717.2 | opus | 961.7 | 938.8 | 717.1 | 706.4 | AITER opus contiguous 961.7 |
| 14336x1024x4096 | bfloat16 | 224 | data-parallel | yes | 1301.7 | 1310.2 | 1296.1 | 1301.8 | 1306.4 | 1309.6 | torch | 1237.1 | 1243.3 | 1247.9 | 1263.7 | hipBLASLt padded 1263.7 |
| 14336x2048x4096 | bfloat16 | 448 | data-parallel |  | 1352.6 | 1359.8 | 1352.6 | 1359.8 | 1348.7 | 1351.0 | flydsl | 1312.8 | 1055.4 (copy) | 1299.3 | 1299.9 | AITER flydsl contiguous 1312.8 |
| 14336x129280x4096 | bfloat16 | 28280 | one-tile owner, n = 2-3 |  | 1472.9 | 1519.0 | 1522.2 | 1516.5 | 1496.9 | 1488.4 | asm | 1481.0 | 1487.3 | 1429.4 | 1370.6 | AITER asm padded 1487.3 |
| 16384x512x4096 | bfloat16 | 128 | one-tile owner, n = 2 |  | 962.5 | 981.6 | 838.9 | 812.4 | 849.8 | 817.3 | opus | 1017.3 | 1016.7 | 810.1 | 789.1 | AITER opus contiguous 1017.3 |
| 16384x1024x4096 | bfloat16 | 256 | data-parallel | yes | 1398.5 | 1412.4 | 1392.2 | 1404.9 | 1399.6 | 1409.3 | flydsl | 1345.3 | 900.6 (copy) | 1329.4 | 1306.7 | AITER flydsl contiguous 1345.3 |
| 16384x2048x4096 | bfloat16 | 512 | data-parallel |  | 1448.6 | 1474.0 | 1448.6 | 1474.0 | 1443.9 | 1463.5 | torch | 1396.0 | 1403.6 | 1414.0 | 1398.6 | hipBLASLt contiguous 1414.0 |
| 16384x16384x4096 | bfloat16 | 4096 | data-parallel |  | 1530.7 | 1540.8 | 1530.7 | 1540.8 | 1513.7 | 1517.3 | torch | 1484.9 | 1489.0 | 1500.7 | 1489.4 | hipBLASLt contiguous 1500.7 |
| 16384x129280x4096 | bfloat16 | 32320 | one-tile owner, n = 2 |  | 1488.6 | 1554.9 | 1553.3 | 1552.5 | 1526.0 | 1519.7 | asm | 1487.4 | 1481.2 | 1346.9 | 1366.1 | AITER asm contiguous 1487.4 |
| 32768x16384x4096 | bfloat16 | 8192 | data-parallel |  | 1543.1 | 1545.8 | 1543.1 | 1545.8 | 1518.4 | 1516.8 | flydsl | 1481.1 | 1408.1 (copy) | 1520.2 | 1494.9 | hipBLASLt contiguous 1520.2 |
| 512x7168x4224 | bfloat16 | 56 | one-tile owner, n = 2 |  | 531.6 | 531.6 | 370.8 | 372.2 | 374.8 | 375.3 | flydsl | 719.1 | 723.6 | 721.6 | 721.9 | AITER flydsl padded 723.6 |
| 1024x7168x4224 | bfloat16 | 112 | one-tile owner, n = 2-3 |  | 926.2 | 911.9 | 736.5 | 744.3 | 741.1 | 752.1 | flydsl | 928.6 | 931.7 | 817.1 | 814.6 | AITER flydsl padded 931.7 |
| 10240x7168x4224 | bfloat16 | 1120 | one-tile owner, n = 2-3 |  | 1447.4 | 1463.7 | 1420.4 | 1422.1 | 1406.4 | 1412.0 | opus | 910.1 | 915.5 | 1336.0 | 1331.4 | hipBLASLt contiguous 1336.0 |
| 512x6144x4608 | bfloat16 | 48 | one-tile owner, n = 3 |  | 540.8 | 556.7 | 334.3 | 342.3 | 337.1 | 345.1 | flydsl | 683.6 | 453.2 (copy) | 621.1 | 612.6 | AITER flydsl contiguous 683.6 |
| 1024x6144x4608 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 794.5 | 800.0 | 656.6 | 675.3 | 662.9 | 680.8 | torch | 891.6 | 874.9 | 906.0 | 879.8 | hipBLASLt contiguous 906.0 |
| 2048x6144x4608 | bfloat16 | 192 | data-parallel | yes | 1231.0 | 1250.3 | 1232.2 | 1237.1 | 1243.4 | 1253.2 | flydsl | 1186.7 | 949.4 (copy) | 1238.5 | 1249.7 | hipBLASLt padded 1249.7 |
| 4096x6144x4608 | bfloat16 | 384 | one-tile owner, n = 2 |  | 1349.4 | 1360.7 | 1280.2 | 1288.0 | 1276.0 | 1289.3 | torch | 1263.9 | 1281.5 | 1278.3 | 1284.1 | hipBLASLt padded 1284.1 |
| 8192x6144x4608 | bfloat16 | 768 | data-parallel |  | 1505.3 | 1530.6 | 1505.3 | 1530.6 | 1505.9 | 1523.1 | torch | 1457.5 | 1470.0 | 1475.1 | 1467.9 | hipBLASLt contiguous 1475.1 |
| 16384x6144x4608 | bfloat16 | 1536 | data-parallel |  | 1521.8 | 1550.7 | 1521.8 | 1550.7 | 1518.8 | 1537.1 | torch | 1475.0 | 1487.0 | 1489.6 | 1489.5 | hipBLASLt contiguous 1489.6 |
| 32768x6144x4608 | bfloat16 | 3072 | data-parallel |  | 1543.3 | 1565.8 | 1543.3 | 1565.8 | 1530.1 | 1545.3 | torch | 1482.7 | 1492.9 | 1493.7 | 1497.5 | hipBLASLt padded 1497.5 |
| 512x1024x5120 | float32 | 8 | one-tile reduce-scatter, n = 10 |  | 176.0 | 183.6 | 51.4 | 54.1 | 51.6 | 53.9 | asm | 248.3 | 253.4 | 274.3 | 278.0 | hipBLASLt padded 278.0 |
| 512x5120x5120 | bfloat16 | 40 | one-tile owner, n = 3 |  | 481.1 | 500.6 | 268.7 | 281.9 | 267.7 | 281.8 | torch | 640.5 | 666.8 | 641.8 | 666.5 | AITER torch padded 666.8 |
| 512x6400x5120 | bfloat16 | 50 | one-tile owner, n = 3 |  | 583.9 | 606.0 | 337.5 | 352.5 | 337.8 | 353.4 | flydsl | 612.8 | 432.7 (copy) | 563.8 | 575.1 | AITER flydsl contiguous 612.8 |
| 512x12800x5120 | bfloat16 | 100 | one-tile owner, n = 2-3 |  | 795.7 | 815.4 | 653.5 | 681.4 | 653.9 | 681.0 | flydsl | 656.5 | 428.2 (copy) | 829.1 | 837.9 | hipBLASLt padded 837.9 |
| 512x37888x5120 | bfloat16 | 296 | one-tile owner, n = 3 |  | 1145.9 | 1169.4 | 964.1 | 984.9 | 943.3 | 966.6 | torch | 1048.6 | 1075.1 | 1053.0 | 1078.3 | hipBLASLt padded 1078.3 |
| 512x51200x5120 | bfloat16 | 400 | data-parallel |  | 1198.4 | 1219.7 | 1198.4 | 1219.7 | 1197.5 | 1225.5 | flydsl | 1135.6 | 636.4 (copy) | 1100.1 | 1120.6 | AITER flydsl contiguous 1135.6 |
| 768x1024x5120 | float32 | 12 | one-tile reduce-scatter, n = 10 |  | 239.6 | 254.1 | 78.1 | 81.4 | 78.0 | 82.2 | asm | 312.0 | 322.7 | 312.0 | 314.7 | AITER asm padded 322.7 |
| 768x37888x5120 | bfloat16 | 444 | data-parallel |  | 1320.4 | 1343.6 | 1320.4 | 1343.6 | 1316.5 | 1337.4 | torch | 1223.4 | 1235.0 | 1236.1 | 1241.3 | hipBLASLt padded 1241.3 |
| 1024x1024x5120 | float32 | 16 | one-tile reduce-scatter, n = 8 |  | 296.9 | 310.8 | 104.7 | 109.6 | 104.9 | 109.4 | asm | 332.1 | 337.0 | 401.5 | 404.5 | hipBLASLt padded 404.5 |
| 1024x5120x5120 | bfloat16 | 80 | one-tile owner, n = 2 |  | 753.1 | 794.0 | 537.3 | 561.2 | 534.2 | 561.3 | torch | 843.3 | 889.5 | 846.4 | 890.6 | hipBLASLt padded 890.6 |
| 1024x6400x5120 | bfloat16 | 100 | one-tile owner, n = 2-3 |  | 824.2 | 851.0 | 663.4 | 698.5 | 666.6 | 697.8 | flydsl | 825.0 | 655.5 (copy) | 820.4 | 825.0 | AITER flydsl contiguous 825.0 |
| 1024x12800x5120 | bfloat16 | 200 | data-parallel | yes | 1228.1 | 1269.3 | 1232.0 | 1258.1 | 1228.2 | 1270.1 | flydsl | 721.2 | 553.7 (copy) | 1052.2 | 1079.5 | hipBLASLt padded 1079.5 |
| 1024x37888x5120 | bfloat16 | 592 | one-tile owner, n = 2 |  | 1335.0 | 1358.7 | 1258.1 | 1276.7 | 1245.6 | 1273.0 | torch | 1052.7 | 1080.1 | 1060.1 | 1080.7 | hipBLASLt padded 1080.7 |
| 1024x51200x5120 | bfloat16 | 800 | one-tile reduce-scatter, n = 4 |  | 1371.9 | 1398.0 | 1275.6 | 1299.3 | 1264.9 | 1283.4 | asm | 1212.0 | 1242.8 | 1066.9 | 1094.2 | AITER asm padded 1242.8 |
| 1536x1024x5120 | float32 | 24 | one-tile reduce-scatter, n = 5 |  | 372.8 | 376.1 | 156.4 | 164.0 | 157.1 | 165.2 | asm | 435.7 | 454.1 | 512.0 | 518.1 | hipBLASLt padded 518.1 |
| 1536x37888x5120 | bfloat16 | 888 | one-tile owner, n = 2-3 |  | 1387.7 | 1423.4 | 1351.8 | 1372.0 | 1341.8 | 1362.3 | torch | 1261.6 | 1286.1 | 1274.7 | 1287.4 | hipBLASLt padded 1287.4 |
| 2048x1024x5120 | float32 | 32 | one-tile reduce-scatter, n = 4 |  | 433.3 | 440.1 | 209.6 | 220.7 | 210.9 | 221.2 | opus | 428.4 | 428.2 | 554.6 | 608.8 | hipBLASLt padded 608.8 |
| 2048x5120x5120 | bfloat16 | 160 | data-parallel | yes | 1062.6 | 1118.0 | 1070.5 | 1117.9 | 1068.2 | 1115.8 | torch | 1147.2 | 1168.7 | 1142.7 | 1168.9 | hipBLASLt padded 1168.9 |
| 2048x6400x5120 | bfloat16 | 200 | data-parallel | yes | 1258.0 | 1286.8 | 1256.2 | 1277.3 | 1265.4 | 1286.4 | flydsl | 1132.7 | 926.6 (copy) | 1087.4 | 1103.5 | AITER flydsl contiguous 1132.7 |
| 2048x12800x5120 | bfloat16 | 400 | data-parallel |  | 1293.7 | 1310.7 | 1293.7 | 1310.7 | 1293.3 | 1312.1 | flydsl | 806.6 | 675.5 (copy) | 1075.2 | 1097.1 | hipBLASLt padded 1097.1 |
| 2048x37888x5120 | bfloat16 | 1184 | data-parallel |  | 1439.7 | 1466.0 | 1439.7 | 1466.0 | 1432.9 | 1456.2 | torch | 1201.1 | 1233.6 | 1211.6 | 1234.3 | hipBLASLt padded 1234.3 |
| 2048x51200x5120 | bfloat16 | 1600 | one-tile owner, n = 2 |  | 1456.0 | 1477.1 | 1423.8 | 1437.8 | 1407.0 | 1428.3 | asm | 210.2 | 215.4 | 1186.3 | 1209.5 | hipBLASLt padded 1209.5 |
| 4096x1024x5120 | float32 | 64 | one-tile owner, n = 2 |  | 589.1 | 625.3 | 410.8 | 434.0 | 413.3 | 434.1 | opus | 714.3 | 725.5 | 769.1 | 800.1 | hipBLASLt padded 800.1 |
| 4096x5120x5120 | bfloat16 | 320 | one-tile owner, n = 2 |  | 1256.0 | 1281.6 | 1108.4 | 1135.6 | 1100.4 | 1129.7 | torch | 1169.9 | 1204.1 | 1180.5 | 1205.2 | hipBLASLt padded 1205.2 |
| 4096x6400x5120 | bfloat16 | 400 | data-parallel |  | 1308.7 | 1324.8 | 1308.7 | 1324.8 | 1308.0 | 1327.3 | flydsl | 1245.6 | 1087.9 (copy) | 1119.0 | 1145.4 | AITER flydsl contiguous 1245.6 |
| 4096x12800x5120 | bfloat16 | 800 | one-tile reduce-scatter, n = 4 |  | 1409.7 | 1438.6 | 1334.8 | 1347.5 | 1315.3 | 1337.9 | asm | 1281.9 | 1295.1 | 1172.9 | 1192.3 | AITER asm padded 1295.1 |
| 4096x51200x5120 | bfloat16 | 3200 | one-tile owner, n = 2 |  | 1516.4 | 1545.2 | 1512.5 | 1524.9 | 1488.6 | 1504.1 | asm | 1444.8 | 1466.2 | 1379.8 | 1402.6 | AITER asm padded 1466.2 |
| 8192x1024x5120 | float32 | 128 | one-tile owner, n = 2 |  | 1018.6 | 1034.5 | 790.0 | 834.4 | 797.6 | 844.0 | opus | 1018.8 | 1055.8 | 980.5 | 1011.9 | AITER opus padded 1055.8 |
| 8192x5120x5120 | bfloat16 | 640 | one-tile owner, n = 2 |  | 1426.0 | 1454.0 | 1371.8 | 1386.6 | 1362.6 | 1380.6 | torch | 1295.8 | 1316.8 | 1311.6 | 1318.8 | hipBLASLt padded 1318.8 |
| 8192x6400x5120 | bfloat16 | 800 | one-tile reduce-scatter, n = 4 |  | 1410.2 | 1441.8 | 1334.6 | 1346.4 | 1317.0 | 1341.8 | flydsl | 1250.3 | 1145.2 (copy) | 1219.9 | 1228.2 | AITER flydsl contiguous 1250.3 |
| 8192x12800x5120 | bfloat16 | 1600 | one-tile owner, n = 2 |  | 1466.4 | 1502.6 | 1449.5 | 1464.2 | 1436.1 | 1459.3 | asm | 1374.1 | 1397.8 | 1189.5 | 1187.7 | AITER asm padded 1397.8 |
| 8192x51200x5120 | bfloat16 | 6400 | data-parallel |  | 1557.1 | 1576.6 | 1557.1 | 1576.6 | 1539.8 | 1553.1 | flydsl | 1482.6 | 1386.8 (copy) | 1415.2 | 1451.9 | AITER flydsl contiguous 1482.6 |
| 16384x1024x5120 | float32 | 256 | data-parallel | yes | 1328.9 | 1345.4 | 1325.1 | 1332.0 | 1329.0 | 1339.4 | opus | 1217.1 | 1211.5 | 1242.0 | 1232.2 | hipBLASLt contiguous 1242.0 |
| 16384x5120x5120 | bfloat16 | 1280 | data-parallel |  | 1514.9 | 1547.8 | 1514.9 | 1547.8 | 1507.3 | 1533.5 | torch | 1471.3 | 1484.6 | 1485.3 | 1486.4 | hipBLASLt padded 1486.4 |
| 16384x6400x5120 | bfloat16 | 1600 | one-tile owner, n = 2 |  | 1464.0 | 1499.1 | 1437.3 | 1454.6 | 1418.3 | 1445.0 | flydsl | 1360.7 | 1257.7 (copy) | 1203.0 | 1206.5 | AITER flydsl contiguous 1360.7 |
| 16384x12800x5120 | bfloat16 | 3200 | one-tile owner, n = 2 |  | 1518.4 | 1551.6 | 1513.3 | 1530.8 | 1491.6 | 1509.7 | torch | 1456.8 | 1470.7 | 1468.1 | 1472.7 | hipBLASLt padded 1472.7 |
| 16384x51200x5120 | bfloat16 | 12800 | data-parallel |  | 1557.8 | 1576.0 | 1557.8 | 1576.0 | 1538.2 | 1554.7 | flydsl | 1488.5 | 1430.0 (copy) | 1405.3 | 1410.1 | AITER flydsl contiguous 1488.5 |
| 32768x5120x5120 | bfloat16 | 2560 | data-parallel |  | 1527.4 | 1557.6 | 1527.4 | 1557.6 | 1518.4 | 1535.5 | torch | 1484.5 | 1493.8 | 1491.8 | 1495.7 | hipBLASLt padded 1495.7 |
| 32768x6400x5120 | bfloat16 | 3200 | one-tile owner, n = 2 |  | 1513.1 | 1544.6 | 1508.6 | 1526.5 | 1483.0 | 1505.5 | torch | 1466.6 | 1475.5 | 1475.1 | 1478.4 | hipBLASLt padded 1478.4 |
| 32768x12800x5120 | bfloat16 | 6400 | data-parallel |  | 1547.8 | 1569.2 | 1547.8 | 1569.2 | 1528.5 | 1547.8 | torch | 1491.0 | 1522.8 | 1500.3 | 1521.0 | AITER torch padded 1522.8 |
| 32768x51200x5120 | bfloat16 | 25600 | data-parallel |  | 1553.4 | 1580.8 | 1553.4 | 1580.8 | 1543.1 | 1561.0 | flydsl | 1485.5 | 1447.0 (copy) | 1372.2 | 1421.4 | AITER flydsl contiguous 1485.5 |
| 512x1280x6144 | bfloat16 | 10 | one-tile reduce-scatter, n = 12 |  | 235.0 | 246.9 | 65.9 | 70.9 | 67.2 | 71.4 | flydsl | 339.9 | 235.5 (copy) | 353.1 | 349.2 | hipBLASLt contiguous 353.1 |
| 512x1536x6144 | bfloat16 | 12 | one-tile reduce-scatter, n = 10 |  | 248.7 | 278.7 | 79.8 | 85.1 | 80.8 | 85.7 | flydsl | 380.0 | 255.9 (copy) | 371.0 | 385.9 | hipBLASLt padded 385.9 |
| 512x2048x6144 | float32 | 16 | one-tile reduce-scatter, n = 8 |  | 328.5 | 341.2 | 103.6 | 110.8 | 104.9 | 111.7 | asm | 341.1 | 349.5 | 422.5 | 420.7 | hipBLASLt contiguous 422.5 |
| 512x2304x6144 | bfloat16 | 18 | one-tile reduce-scatter, n = 7 |  | 317.6 | 346.3 | 119.9 | 129.1 | 121.5 | 129.8 | flydsl | 436.2 | 310.9 (copy) | 373.1 | 370.1 | AITER flydsl contiguous 436.2 |
| 512x2560x6144 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 |  | 352.2 | 368.9 | 133.7 | 142.9 | 135.2 | 143.5 | torch | 474.2 | 479.3 | 475.0 | 479.2 | AITER torch padded 479.3 |
| 512x3072x6144 | float32 | 24 | one-tile reduce-scatter, n = 5 |  | 371.4 | 396.3 | 154.8 | 166.5 | 157.5 | 168.1 | opus | 403.7 | 431.2 | 548.0 | 542.9 | hipBLASLt contiguous 548.0 |
| 512x3072x6144 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 |  | 375.5 | 403.0 | 159.4 | 171.5 | 161.7 | 172.5 | flydsl | 543.4 | 363.3 (copy) | 460.3 | 464.2 | AITER flydsl contiguous 543.4 |
| 512x6144x6144 | bfloat16 | 48 | one-tile owner, n = 3 |  | 605.3 | 623.8 | 323.9 | 347.5 | 325.4 | 349.8 | flydsl | 684.6 | 478.5 (copy) | 695.4 | 681.9 | hipBLASLt contiguous 695.4 |
| 512x9216x6144 | bfloat16 | 72 | one-tile owner, n = 3-4 |  | 769.5 | 775.5 | 481.0 | 514.1 | 482.5 | 517.1 | flydsl | 611.9 | 404.4 (copy) | 599.3 | 624.0 | hipBLASLt padded 624.0 |
| 1024x2048x6144 | float32 | 32 | one-tile reduce-scatter, n = 4 |  | 464.3 | 481.0 | 208.0 | 222.9 | 211.5 | 224.8 | opus | 406.8 | 427.7 | 644.3 | 641.1 | hipBLASLt contiguous 644.3 |
| 1024x2304x6144 | bfloat16 | 36 | one-tile owner, n = 4 |  | 501.2 | 521.7 | 241.7 | 260.1 | 244.6 | 260.5 | torch | 603.2 | 614.0 | 605.1 | 618.7 | hipBLASLt padded 618.7 |
| 1024x2560x6144 | bfloat16 | 40 | one-tile owner, n = 3 |  | 518.6 | 528.9 | 267.5 | 288.6 | 272.1 | 290.5 | torch | 700.7 | 707.6 | 703.1 | 708.5 | hipBLASLt padded 708.5 |
| 1024x3072x6144 | float32 | 48 | one-tile owner, n = 3 |  | 572.5 | 589.1 | 310.7 | 333.0 | 315.4 | 336.6 | opus | 500.2 | 525.8 | 722.6 | 728.6 | hipBLASLt padded 728.6 |
| 1024x6144x6144 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 868.3 | 885.5 | 647.5 | 688.7 | 646.9 | 692.0 | torch | 1006.4 | 1026.8 | 1019.8 | 1020.5 | AITER torch padded 1026.8 |
| 1024x9216x6144 | bfloat16 | 144 | data-parallel | yes | 957.6 | 1016.4 | 951.9 | 1016.8 | 958.9 | 1028.7 | torch | 1063.7 | 1095.4 | 1066.5 | 1098.9 | hipBLASLt padded 1098.9 |
| 2048x2048x6144 | float32 | 64 | one-tile owner, n = 3 |  | 743.9 | 763.1 | 420.3 | 451.8 | 427.1 | 458.1 | opus | 673.8 | 689.6 | 816.2 | 836.5 | hipBLASLt padded 836.5 |
| 2048x2304x6144 | bfloat16 | 72 | one-tile owner, n = 3-4 |  | 873.7 | 882.6 | 492.6 | 527.4 | 499.1 | 531.1 | torch | 843.6 | 871.7 | 846.0 | 872.0 | hipBLASLt padded 872.0 |
| 2048x2560x6144 | bfloat16 | 80 | one-tile owner, n = 2 |  | 830.6 | 850.4 | 546.2 | 584.2 | 554.7 | 589.9 | torch | 906.2 | 932.6 | 911.8 | 934.5 | hipBLASLt padded 934.5 |
| 2048x3072x6144 | float32 | 96 | one-tile owner, n = 2-3 |  | 902.2 | 922.2 | 629.1 | 672.8 | 639.5 | 679.4 | opus | 757.0 | 799.1 | 923.5 | 927.5 | hipBLASLt padded 927.5 |
| 2048x6144x6144 | bfloat16 | 192 | data-parallel | yes | 1236.9 | 1288.1 | 1233.5 | 1282.5 | 1241.8 | 1286.5 | torch | 1262.3 | 1282.6 | 1273.0 | 1286.9 | hipBLASLt padded 1286.9 |
| 2048x9216x6144 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 |  | 1265.1 | 1284.0 | 1013.3 | 1050.3 | 1003.4 | 1045.8 | torch | 1092.9 | 1119.2 | 1101.8 | 1120.8 | hipBLASLt padded 1120.8 |
| 4096x2048x6144 | float32 | 128 | one-tile owner, n = 2 |  | 1102.5 | 1123.9 | 816.4 | 884.5 | 834.5 | 885.3 | opus | 1025.8 | 1075.9 | 868.6 | 826.3 | AITER opus padded 1075.9 |
| 4096x2304x6144 | bfloat16 | 144 | data-parallel | yes | 963.4 | 1045.6 | 966.5 | 1041.8 | 980.8 | 1042.2 | flydsl | 684.4 | 638.5 (copy) | 1092.6 | 1118.6 | hipBLASLt padded 1118.6 |
| 4096x2560x6144 | bfloat16 | 160 | data-parallel | yes | 1074.0 | 1148.8 | 1071.5 | 1139.6 | 1086.5 | 1143.0 | flydsl | 637.7 | 593.3 (copy) | 1176.8 | 1205.3 | hipBLASLt padded 1205.3 |
| 4096x3072x6144 | float32 | 192 | data-parallel | yes | 1172.9 | 1239.5 | 1167.7 | 1222.8 | 1184.5 | 1234.6 | asm | 1143.7 | 1170.6 | 1235.1 | 1253.8 | hipBLASLt padded 1253.8 |
| 4096x6144x6144 | bfloat16 | 384 | one-tile owner, n = 2 |  | 1393.2 | 1419.6 | 1298.7 | 1324.4 | 1289.7 | 1323.9 | torch | 1293.3 | 1314.5 | 1307.2 | 1314.5 | AITER torch padded 1314.5 |
| 4096x9216x6144 | bfloat16 | 576 | one-tile owner, n = 3 |  | 1410.7 | 1442.5 | 1311.7 | 1329.9 | 1298.3 | 1327.8 | torch | 1296.0 | 1322.1 | 1309.4 | 1324.1 | hipBLASLt padded 1324.1 |
| 8192x2048x6144 | float32 | 256 | data-parallel | yes | 1386.2 | 1427.7 | 1376.0 | 1413.0 | 1391.2 | 1423.0 | opus | 1269.4 | 1268.3 | 1290.2 | 1304.5 | hipBLASLt padded 1304.5 |
| 8192x2304x6144 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 |  | 1243.8 | 1271.1 | 1006.9 | 1043.6 | 996.5 | 1039.0 | torch | 1091.6 | 1125.8 | 1101.2 | 1126.6 | hipBLASLt padded 1126.6 |
| 8192x2560x6144 | bfloat16 | 320 | one-tile owner, n = 3 |  | 1301.8 | 1330.6 | 1112.0 | 1144.4 | 1105.6 | 1138.8 | asm | 404.8 | 415.9 | 1199.8 | 1225.1 | hipBLASLt padded 1225.1 |
| 8192x3072x6144 | float32 | 384 | one-tile owner, n = 2 |  | 1302.8 | 1340.4 | 1196.7 | 1253.8 | 1208.2 | 1256.4 | asm | 1170.3 | 1188.9 | 1230.9 | 1229.1 | hipBLASLt contiguous 1230.9 |
| 8192x6144x6144 | bfloat16 | 768 | data-parallel |  | 1527.0 | 1562.9 | 1527.0 | 1562.9 | 1531.5 | 1557.8 | flydsl | 1482.1 | 1308.7 (copy) | 1494.8 | 1491.7 | hipBLASLt contiguous 1494.8 |
| 8192x9216x6144 | bfloat16 | 1152 | one-tile owner, n = 2 |  | 1500.3 | 1536.2 | 1461.9 | 1480.6 | 1442.2 | 1470.9 | asm | 1416.1 | 1435.3 | 1200.7 | 1202.4 | AITER asm padded 1435.3 |
| 16384x2048x6144 | float32 | 512 | data-parallel |  | 1397.2 | 1436.7 | 1397.2 | 1436.7 | 1405.9 | 1445.8 | opus | 1301.2 | 1298.8 | 1318.0 | 1323.0 | hipBLASLt padded 1323.0 |
| 16384x2304x6144 | bfloat16 | 576 | one-tile owner, n = 3 |  | 1366.9 | 1415.9 | 1276.8 | 1295.2 | 1279.1 | 1291.3 | flydsl | 970.4 | 829.2 (copy) | 1306.7 | 1310.5 | hipBLASLt padded 1310.5 |
| 16384x2560x6144 | bfloat16 | 640 | one-tile owner, n = 2 |  | 1426.0 | 1460.7 | 1361.9 | 1376.6 | 1353.6 | 1371.8 | flydsl | 1306.4 | 1080.7 (copy) | 1307.7 | 1318.9 | hipBLASLt padded 1318.9 |
| 16384x3072x6144 | float32 | 768 | data-parallel |  | 1406.3 | 1450.9 | 1406.3 | 1450.9 | 1420.6 | 1445.6 | asm | 1366.6 | 1389.1 | 1336.8 | 1336.8 | AITER asm padded 1389.1 |
| 16384x6144x6144 | bfloat16 | 1536 | data-parallel |  | 1534.1 | 1572.3 | 1534.1 | 1572.3 | 1541.4 | 1562.2 | asm | 1498.1 | 1521.5 | 1505.7 | 1505.2 | AITER asm padded 1521.5 |
| 16384x9216x6144 | bfloat16 | 2304 | data-parallel |  | 1543.8 | 1578.6 | 1543.8 | 1578.6 | 1542.2 | 1565.2 | torch | 1494.8 | 1507.6 | 1509.4 | 1508.3 | hipBLASLt contiguous 1509.4 |
| 32768x2304x6144 | bfloat16 | 1152 | one-tile owner, n = 2 |  | 1460.0 | 1493.3 | 1421.8 | 1440.6 | 1406.8 | 1429.5 | flydsl | 649.6 | 588.2 (copy) | 1192.2 | 1194.2 | hipBLASLt padded 1194.2 |
| 32768x2560x6144 | bfloat16 | 1280 | data-parallel |  | 1499.0 | 1533.1 | 1499.0 | 1533.1 | 1499.4 | 1520.7 | flydsl | 1447.5 | 1184.2 (copy) | 1484.4 | 1484.5 | hipBLASLt padded 1484.5 |
| 32768x6144x6144 | bfloat16 | 3072 | data-parallel |  | 1550.6 | 1586.9 | 1550.6 | 1586.9 | 1548.5 | 1569.5 | torch | 1500.8 | 1512.3 | 1509.8 | 1512.9 | hipBLASLt padded 1512.9 |
| 32768x9216x6144 | bfloat16 | 4608 | data-parallel |  | 1557.5 | 1585.2 | 1557.5 | 1585.2 | 1551.1 | 1569.5 | torch | 1523.4 | 1535.7 | 1537.4 | 1544.9 | hipBLASLt padded 1544.9 |
| 512x5120x6400 | bfloat16 | 40 | one-tile owner, n = 4 |  | 553.8 | 569.6 | 275.8 | 286.4 | 279.7 | 289.3 | flydsl | 541.0 | 431.3 (copy) | 680.7 | 687.9 | hipBLASLt padded 687.9 |
| 1024x5120x6400 | bfloat16 | 80 | one-tile owner, n = 2 |  | 816.4 | 841.6 | 549.9 | 569.7 | 558.4 | 576.2 | flydsl | 596.6 | 516.8 (copy) | 891.0 | 901.8 | hipBLASLt padded 901.8 |
| 2048x5120x6400 | bfloat16 | 160 | data-parallel | yes | 1092.7 | 1150.4 | 1105.3 | 1137.2 | 1099.9 | 1147.0 | flydsl | 739.2 | 646.4 (copy) | 1203.8 | 1197.3 | hipBLASLt contiguous 1203.8 |
| 4096x5120x6400 | bfloat16 | 320 | one-tile owner, n = 3 |  | 1344.3 | 1359.3 | 1134.3 | 1156.2 | 1135.2 | 1157.8 | torch | 1216.6 | 1231.9 | 1229.0 | 1231.0 | AITER torch padded 1231.9 |
| 8192x5120x6400 | bfloat16 | 640 | one-tile owner, n = 2 |  | 1467.8 | 1489.0 | 1400.1 | 1405.8 | 1391.2 | 1396.6 | flydsl | 1349.2 | 1188.7 (copy) | 1351.5 | 1344.0 | hipBLASLt contiguous 1351.5 |
| 16384x5120x6400 | bfloat16 | 1280 | data-parallel |  | 1540.3 | 1564.4 | 1540.3 | 1564.4 | 1541.9 | 1555.9 | torch | 1493.1 | 1503.8 | 1502.3 | 1502.5 | AITER torch padded 1503.8 |
| 32768x5120x6400 | bfloat16 | 2560 | data-parallel |  | 1551.4 | 1577.0 | 1551.4 | 1577.0 | 1553.7 | 1558.1 | torch | 1498.3 | 1505.1 | 1509.2 | 1511.5 | hipBLASLt padded 1511.5 |
| 512x16384x6656 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1126.8 | 1134.5 | 877.9 | 878.3 | 877.0 | 885.9 | flydsl | 840.5 | 511.8 (copy) | 921.8 | 882.0 | hipBLASLt contiguous 921.8 |
| 1024x16384x6656 | bfloat16 | 256 | data-parallel | yes | 1441.0 | 1474.3 | 1444.4 | 1469.4 | 1444.8 | 1475.1 | asm | 360.4 | 367.7 | 1393.9 | 1400.1 | hipBLASLt padded 1400.1 |
| 2048x16384x6656 | bfloat16 | 512 | data-parallel |  | 1508.1 | 1530.1 | 1508.1 | 1530.1 | 1499.4 | 1524.2 | asm | 1460.1 | 1463.8 | 1444.4 | 1449.8 | AITER asm padded 1463.8 |
| 4096x16384x6656 | bfloat16 | 1024 | data-parallel |  | 1536.5 | 1560.8 | 1536.5 | 1560.8 | 1535.1 | 1555.5 | torch | 1463.9 | 1470.4 | 1479.1 | 1477.0 | hipBLASLt contiguous 1479.1 |
| 8192x16384x6656 | bfloat16 | 2048 | data-parallel |  | 1561.0 | 1587.6 | 1561.0 | 1587.6 | 1559.0 | 1570.7 | asm | 1510.3 | 1525.0 | 1500.6 | 1498.0 | AITER asm padded 1525.0 |
| 16384x16384x6656 | bfloat16 | 4096 | data-parallel |  | 1574.8 | 1590.6 | 1574.8 | 1590.6 | 1564.2 | 1578.6 | asm | 1527.4 | 1532.8 | 1535.5 | 1542.0 | hipBLASLt padded 1542.0 |
| 32768x16384x6656 | bfloat16 | 8192 | data-parallel |  | 1579.7 | 1587.0 | 1579.7 | 1587.0 | 1576.3 | 1580.6 | asm | 290.8 | 292.3 | 1553.3 | 1537.1 | hipBLASLt contiguous 1553.3 |
| 512x1024x7168 | bfloat16 | 8 | one-tile reduce-scatter, n = 14 |  | 232.3 | 227.3 | 55.5 | 54.3 | 56.8 | 54.6 | torch | 382.2 | 383.4 | 383.5 | 385.4 | hipBLASLt padded 385.4 |
| 512x1536x7168 | bfloat16 | 12 | one-tile reduce-scatter, n = 10 |  | 291.5 | 293.0 | 84.2 | 82.5 | 85.5 | 83.1 | torch | 411.2 | 408.3 | 411.0 | 408.7 | AITER torch contiguous 411.2 |
| 512x2048x7168 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 |  | 370.4 | 364.7 | 111.8 | 108.9 | 113.6 | 110.0 | torch | 547.4 | 540.2 | 549.0 | 538.8 | hipBLASLt contiguous 549.0 |
| 512x3584x7168 | bfloat16 | 28 | one-tile reduce-scatter, n = 4 |  | 459.0 | 439.7 | 198.5 | 192.8 | 200.9 | 194.5 | flydsl | 558.4 | 386.0 (copy) | 541.5 | 532.4 | AITER flydsl contiguous 558.4 |
| 512x6144x7168 | bfloat16 | 48 | one-tile owner, n = 3 |  | 641.4 | 633.8 | 341.5 | 332.4 | 347.5 | 336.0 | flydsl | 715.5 | 488.2 (copy) | 709.5 | 687.2 | AITER flydsl contiguous 715.5 |
| 512x8192x7168 | bfloat16 | 64 | one-tile owner, n = 3 |  | 808.9 | 800.9 | 451.6 | 440.9 | 458.7 | 445.1 | flydsl | 702.6 | 434.3 (copy) | 736.4 | 730.7 | hipBLASLt contiguous 736.4 |
| 512x8448x7168 | bfloat16 | 66 | one-tile owner, n = 3-4 |  | 744.9 | 721.5 | 467.5 | 453.3 | 472.4 | 458.4 | flydsl | 667.6 | 427.7 (copy) | 469.3 | 474.7 | AITER flydsl contiguous 667.6 |
| 512x20480x7168 | bfloat16 | 160 | data-parallel | yes | 1064.9 | 1034.9 | 1048.0 | 1030.4 | 1064.0 | 1037.0 | flydsl | 1028.0 | 580.8 (copy) | 1153.2 | 1149.1 | hipBLASLt contiguous 1153.2 |
| 768x1024x7168 | bfloat16 | 12 | one-tile reduce-scatter, n = 10 |  | 293.8 | 296.3 | 84.1 | 82.3 | 85.8 | 83.3 | torch | 405.0 | 403.4 | 404.5 | 407.2 | hipBLASLt padded 407.2 |
| 768x2048x7168 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 |  | 437.7 | 431.1 | 169.5 | 165.0 | 172.5 | 166.7 | flydsl | 557.4 | 387.5 (copy) | 569.7 | 559.8 | hipBLASLt contiguous 569.7 |
| 1024x512x7168 | bfloat16 | 8 | one-tile reduce-scatter, n = 14 |  | 233.7 | 228.0 | 55.9 | 54.8 | 57.0 | 55.3 | flydsl | 334.8 | 216.0 (copy) | 290.6 | 304.2 | AITER flydsl contiguous 334.8 |
| 1024x1024x7168 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 |  | 373.5 | 369.8 | 113.2 | 111.2 | 115.6 | 111.3 | torch | 560.2 | 549.3 | 562.9 | 549.2 | hipBLASLt contiguous 562.9 |
| 1024x1536x7168 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 |  | 437.9 | 431.2 | 169.1 | 165.5 | 172.6 | 167.0 | flydsl | 561.5 | 404.9 (copy) | 567.8 | 560.8 | hipBLASLt contiguous 567.8 |
| 1024x2048x7168 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 513.2 | 501.8 | 227.1 | 221.9 | 230.1 | 223.3 | torch | 728.5 | 709.3 | 731.1 | 709.3 | hipBLASLt contiguous 731.1 |
| 1024x3584x7168 | bfloat16 | 56 | one-tile owner, n = 3 |  | 738.6 | 728.2 | 397.8 | 388.3 | 405.6 | 392.6 | flydsl | 771.8 | 592.6 (copy) | 681.5 | 675.8 | AITER flydsl contiguous 771.8 |
| 1024x6144x7168 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 933.4 | 905.2 | 679.2 | 659.7 | 688.9 | 669.9 | torch | 991.0 | 981.8 | 1002.8 | 982.6 | hipBLASLt contiguous 1002.8 |
| 1024x8192x7168 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1180.4 | 1196.8 | 888.6 | 873.5 | 899.5 | 880.4 | flydsl | 1016.8 | 731.7 (copy) | 935.8 | 939.1 | AITER flydsl contiguous 1016.8 |
| 1024x8448x7168 | bfloat16 | 132 | data-parallel | yes | 926.6 | 903.1 | 914.5 | 891.7 | 924.9 | 902.4 | torch | 1033.1 | 1020.5 | 1039.4 | 1021.8 | hipBLASLt contiguous 1039.4 |
| 1024x20480x7168 | bfloat16 | 320 | one-tile owner, n = 3 |  | 1315.2 | 1326.8 | 1129.4 | 1105.7 | 1119.2 | 1109.5 | flydsl | 738.3 | 600.3 (copy) | 1204.6 | 1197.4 | hipBLASLt contiguous 1204.6 |
| 1280x512x7168 | bfloat16 | 10 | one-tile reduce-scatter, n = 12 |  | 261.8 | 262.2 | 70.3 | 68.8 | 71.7 | 69.2 | torch | 363.5 | 359.0 | 363.8 | 366.5 | hipBLASLt padded 366.5 |
| 1280x1024x7168 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 |  | 394.5 | 395.1 | 142.1 | 138.9 | 145.3 | 140.2 | torch | 537.9 | 530.5 | 543.0 | 529.8 | hipBLASLt contiguous 543.0 |
| 1280x2048x7168 | bfloat16 | 40 | one-tile owner, n = 4 |  | 608.9 | 599.4 | 285.9 | 278.0 | 291.3 | 281.0 | torch | 744.2 | 742.0 | 749.2 | 738.7 | hipBLASLt contiguous 749.2 |
| 1536x512x7168 | bfloat16 | 12 | one-tile reduce-scatter, n = 10 |  | 293.1 | 295.9 | 85.0 | 83.3 | 86.6 | 84.1 | torch | 403.3 | 403.5 | 404.4 | 407.7 | hipBLASLt padded 407.7 |
| 1536x1024x7168 | bfloat16 | 24 | one-tile reduce-scatter, n = 5 |  | 442.4 | 435.6 | 170.9 | 167.5 | 174.8 | 168.9 | torch | 648.4 | 636.2 | 649.2 | 637.7 | hipBLASLt contiguous 649.2 |
| 1536x2048x7168 | bfloat16 | 48 | one-tile owner, n = 3 |  | 657.5 | 653.9 | 344.8 | 337.5 | 352.1 | 342.8 | torch | 830.7 | 831.3 | 833.6 | 831.9 | hipBLASLt contiguous 833.6 |
| 1792x512x7168 | bfloat16 | 14 | one-tile reduce-scatter, n = 9 |  | 312.7 | 319.9 | 98.8 | 97.6 | 100.6 | 97.1 | torch | 440.3 | 442.1 | 440.4 | 440.4 | AITER torch padded 442.1 |
| 1792x1024x7168 | bfloat16 | 28 | one-tile reduce-scatter, n = 4 |  | 467.7 | 452.0 | 201.9 | 198.9 | 205.7 | 200.3 | flydsl | 602.0 | 420.5 (copy) | 608.6 | 591.8 | hipBLASLt contiguous 608.6 |
| 1792x2048x7168 | bfloat16 | 56 | one-tile owner, n = 3 |  | 751.4 | 737.2 | 403.0 | 393.7 | 409.8 | 397.2 | flydsl | 776.9 | 644.3 (copy) | 686.9 | 682.9 | AITER flydsl contiguous 776.9 |
| 2048x512x7168 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 |  | 374.8 | 368.8 | 113.7 | 111.8 | 116.0 | 112.5 | torch | 506.3 | 502.6 | 507.2 | 503.0 | hipBLASLt contiguous 507.2 |
| 2048x1024x7168 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 520.8 | 508.6 | 230.6 | 226.7 | 235.2 | 228.2 | torch | 711.0 | 705.8 | 711.7 | 710.0 | hipBLASLt contiguous 711.7 |
| 2048x2048x7168 | bfloat16 | 64 | one-tile owner, n = 3 |  | 843.7 | 837.9 | 461.8 | 456.4 | 470.9 | 457.7 | flydsl | 872.9 | 696.9 (copy) | 771.4 | 763.3 | AITER flydsl contiguous 872.9 |
| 2048x3584x7168 | bfloat16 | 112 | one-tile owner, n = 2-3 |  | 1101.6 | 1085.1 | 809.6 | 790.5 | 808.1 | 795.6 | opus | 1000.6 | 994.7 | 1031.5 | 1018.9 | hipBLASLt contiguous 1031.5 |
| 2048x8192x7168 | bfloat16 | 256 | data-parallel | yes | 1506.5 | 1525.1 | 1510.9 | 1515.4 | 1497.4 | 1523.1 | asm | 1471.2 | 1480.4 | 1451.4 | 1453.8 | AITER asm padded 1480.4 |
| 2048x20480x7168 | bfloat16 | 640 | one-tile owner, n = 2 |  | 1469.6 | 1488.3 | 1395.3 | 1390.7 | 1381.7 | 1388.2 | flydsl | 826.3 | 710.1 (copy) | 1278.5 | 1286.4 | hipBLASLt padded 1286.4 |
| 4096x512x7168 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 508.9 | 500.5 | 227.6 | 223.3 | 231.8 | 223.3 | flydsl | 622.3 | 428.2 (copy) | 601.1 | 596.1 | AITER flydsl contiguous 622.3 |
| 4096x1024x7168 | bfloat16 | 64 | one-tile owner, n = 3 |  | 833.6 | 829.2 | 457.7 | 450.3 | 466.0 | 450.3 | torch | 902.8 | 879.1 | 916.2 | 876.0 | hipBLASLt contiguous 916.2 |
| 4096x2048x7168 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1215.1 | 1217.4 | 894.3 | 900.5 | 905.6 | 901.8 | torch | 1167.8 | 1155.3 | 1182.0 | 1145.6 | hipBLASLt contiguous 1182.0 |
| 4096x3584x7168 | bfloat16 | 224 | data-parallel | yes | 1395.1 | 1409.4 | 1399.0 | 1399.0 | 1403.6 | 1410.3 | flydsl | 1329.5 | 1069.6 (copy) | 1330.0 | 1328.6 | hipBLASLt contiguous 1330.0 |
| 4096x8192x7168 | bfloat16 | 512 | data-parallel |  | 1533.2 | 1553.8 | 1533.2 | 1553.8 | 1535.3 | 1552.2 | flydsl | 1491.3 | 1261.3 (copy) | 1480.7 | 1489.7 | AITER flydsl contiguous 1491.3 |
| 4096x20480x7168 | bfloat16 | 1280 | data-parallel |  | 1557.1 | 1578.5 | 1557.1 | 1578.5 | 1561.3 | 1568.0 | flydsl | 1505.4 | 1312.3 (copy) | 1461.6 | 1464.3 | AITER flydsl contiguous 1505.4 |
| 6144x512x7168 | bfloat16 | 48 | one-tile owner, n = 3 |  | 633.9 | 624.1 | 339.5 | 329.8 | 345.1 | 333.6 | flydsl | 718.3 | 470.7 (copy) | 674.9 | 640.6 | AITER flydsl contiguous 718.3 |
| 6144x1024x7168 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 920.7 | 903.3 | 673.3 | 656.7 | 684.3 | 664.6 | opus | 964.4 | 971.5 | 885.4 | 926.9 | AITER opus padded 971.5 |
| 6144x2048x7168 | bfloat16 | 192 | data-parallel | yes | 1273.5 | 1269.9 | 1263.5 | 1261.3 | 1274.8 | 1272.9 | opus | 1357.0 | 1338.5 | 1303.0 | 1304.7 | AITER opus contiguous 1357.0 |
| 8192x512x7168 | bfloat16 | 64 | one-tile owner, n = 3 |  | 812.4 | 804.6 | 453.5 | 442.1 | 460.7 | 447.2 | flydsl | 841.7 | 490.4 (copy) | 755.8 | 747.7 | AITER flydsl contiguous 841.7 |
| 8192x1024x7168 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1180.5 | 1200.0 | 892.1 | 862.9 | 901.4 | 878.3 | torch | 1088.2 | 1067.8 | 1097.3 | 1068.0 | hipBLASLt contiguous 1097.3 |
| 8192x2048x7168 | bfloat16 | 256 | data-parallel | yes | 1481.0 | 1516.2 | 1492.3 | 1511.7 | 1493.7 | 1519.2 | torch | 1377.3 | 1412.2 | 1383.4 | 1412.5 | hipBLASLt padded 1412.5 |
| 8192x3584x7168 | bfloat16 | 448 | data-parallel |  | 1418.1 | 1435.5 | 1418.1 | 1435.5 | 1415.7 | 1428.9 | flydsl | 1379.1 | 1177.5 (copy) | 1348.1 | 1345.2 | AITER flydsl contiguous 1379.1 |
| 8192x8192x7168 | bfloat16 | 1024 | data-parallel |  | 1544.9 | 1578.5 | 1544.9 | 1578.5 | 1542.3 | 1570.6 | flydsl | 1499.3 | 1340.0 (copy) | 1514.6 | 1513.0 | hipBLASLt contiguous 1514.6 |
| 8192x20480x7168 | bfloat16 | 2560 | data-parallel |  | 1578.7 | 1598.7 | 1578.7 | 1598.7 | 1581.4 | 1586.1 | asm | 1524.0 | 1540.3 | 1480.1 | 1469.1 | AITER asm padded 1540.3 |
| 10240x512x7168 | bfloat16 | 80 | one-tile owner, n = 3-4 |  | 915.8 | 918.6 | 557.7 | 540.6 | 563.8 | 546.0 | torch | 845.9 | 825.3 | 847.9 | 844.1 | hipBLASLt contiguous 847.9 |
| 10240x1024x7168 | bfloat16 | 160 | data-parallel | yes | 1106.0 | 1084.4 | 1093.6 | 1074.7 | 1107.7 | 1082.1 | torch | 1207.8 | 1210.4 | 1218.2 | 1209.7 | hipBLASLt contiguous 1218.2 |
| 10240x2048x7168 | bfloat16 | 320 | one-tile owner, n = 3 |  | 1337.3 | 1348.5 | 1141.3 | 1145.8 | 1141.4 | 1138.7 | torch | 1247.3 | 1255.9 | 1261.3 | 1256.6 | hipBLASLt contiguous 1261.3 |
| 10240x8448x7168 | bfloat16 | 1320 | one-tile owner, n = 4 |  | 1476.4 | 1502.1 | 1409.4 | 1417.9 | 1391.8 | 1405.4 | asm | 1393.8 | 1401.3 | 1358.7 | 1365.8 | AITER asm padded 1401.3 |
| 10240x20480x7168 | bfloat16 | 3200 | one-tile owner, n = 2 |  | 1502.9 | 1528.7 | 1497.5 | 1504.6 | 1471.9 | 1489.3 | asm | 1496.0 | 1510.0 | 1461.1 | 1455.4 | AITER asm padded 1510.0 |
| 12288x512x7168 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 889.0 | 877.6 | 657.4 | 642.0 | 670.2 | 648.5 | opus | 925.5 | 928.4 | 878.2 | 904.4 | AITER opus padded 928.4 |
| 12288x1024x7168 | bfloat16 | 192 | data-parallel | yes | 1229.9 | 1242.0 | 1230.4 | 1233.1 | 1229.1 | 1242.5 | opus | 1334.6 | 1313.3 | 1284.2 | 1283.8 | AITER opus contiguous 1334.6 |
| 12288x2048x7168 | bfloat16 | 384 | one-tile owner, n = 2 |  | 1413.6 | 1433.8 | 1309.5 | 1310.8 | 1294.5 | 1296.9 | opus | 1379.4 | 1375.6 | 1326.6 | 1325.8 | AITER opus contiguous 1379.4 |
| 14336x512x7168 | bfloat16 | 112 | one-tile owner, n = 2-3 |  | 1034.4 | 1027.7 | 767.6 | 749.5 | 777.5 | 761.2 | opus | 994.1 | 974.5 | 888.2 | 893.5 | AITER opus contiguous 994.1 |
| 14336x1024x7168 | bfloat16 | 224 | data-parallel | yes | 1355.3 | 1376.8 | 1355.6 | 1364.4 | 1364.6 | 1373.7 | torch | 1304.6 | 1325.1 | 1323.8 | 1328.7 | hipBLASLt padded 1328.7 |
| 14336x2048x7168 | bfloat16 | 448 | data-parallel |  | 1391.8 | 1417.4 | 1391.8 | 1417.4 | 1394.7 | 1404.0 | flydsl | 1363.0 | 1074.7 (copy) | 1343.3 | 1351.4 | AITER flydsl contiguous 1363.0 |
| 16384x512x7168 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1137.0 | 1146.5 | 868.0 | 858.5 | 877.7 | 861.4 | opus | 1074.2 | 1057.6 | 1044.5 | 1011.0 | AITER opus contiguous 1074.2 |
| 16384x1024x7168 | bfloat16 | 256 | data-parallel | yes | 1470.3 | 1491.1 | 1475.7 | 1482.3 | 1474.9 | 1492.6 | torch | 1409.4 | 1415.3 | 1420.5 | 1420.4 | hipBLASLt contiguous 1420.5 |
| 16384x2048x7168 | bfloat16 | 512 | data-parallel |  | 1510.5 | 1538.7 | 1510.5 | 1538.7 | 1512.5 | 1535.4 | torch | 1466.0 | 1475.7 | 1475.2 | 1473.0 | AITER torch padded 1475.7 |
| 16384x3584x7168 | bfloat16 | 896 | one-tile owner, n = 2 |  | 1492.4 | 1512.9 | 1434.3 | 1436.2 | 1419.9 | 1428.7 | flydsl | 1372.8 | 1190.9 (copy) | 1382.9 | 1379.3 | hipBLASLt contiguous 1382.9 |
| 16384x8192x7168 | bfloat16 | 2048 | data-parallel |  | 1566.9 | 1588.2 | 1566.9 | 1588.2 | 1563.5 | 1575.1 | torch | 1505.3 | 1517.2 | 1513.3 | 1514.4 | AITER torch padded 1517.2 |
| 16384x20480x7168 | bfloat16 | 5120 | data-parallel |  | 1591.0 | 1601.4 | 1591.0 | 1601.4 | 1580.2 | 1585.2 | flydsl | 1529.2 | 1453.6 (copy) | 1504.7 | 1495.3 | AITER flydsl contiguous 1529.2 |
| 32768x512x7168 | bfloat16 | 256 | data-parallel | yes | 1384.9 | 1378.7 | 1383.7 | 1395.4 | 1394.0 | 1367.3 | torch | 1366.7 | 1378.6 | 1386.4 | 1366.0 | hipBLASLt contiguous 1386.4 |
| 32768x1024x7168 | bfloat16 | 512 | data-parallel |  | 1489.3 | 1506.8 | 1489.3 | 1506.8 | 1486.9 | 1504.0 | torch | 1422.1 | 1429.6 | 1430.6 | 1430.3 | hipBLASLt contiguous 1430.6 |
| 32768x2048x7168 | bfloat16 | 1024 | data-parallel |  | 1531.0 | 1552.8 | 1531.0 | 1552.8 | 1523.0 | 1541.9 | torch | 1471.1 | 1482.9 | 1484.2 | 1481.6 | hipBLASLt contiguous 1484.2 |
| 32768x3584x7168 | bfloat16 | 1792 | data-parallel |  | 1539.9 | 1560.1 | 1539.9 | 1560.1 | 1532.0 | 1544.5 | torch | 1493.2 | 1502.5 | 1505.0 | 1503.4 | hipBLASLt contiguous 1505.0 |
| 32768x8192x7168 | bfloat16 | 4096 | data-parallel |  | 1577.5 | 1590.6 | 1577.5 | 1590.6 | 1569.5 | 1573.2 | torch | 1541.1 | 1550.6 | 1541.5 | 1556.1 | hipBLASLt padded 1556.1 |
| 32768x20480x7168 | bfloat16 | 10240 | data-parallel |  | 1589.2 | 1597.6 | 1589.2 | 1597.6 | 1583.7 | 1584.7 | asm | 1550.9 | 1555.0 | 1461.4 | 1477.4 | AITER asm padded 1555.0 |
| 512x1280x8192 | bfloat16 | 10 | one-tile reduce-scatter, n = 12 |  | 287.6 | 284.2 | 77.5 | 72.4 | 78.3 | 72.7 | torch | 421.0 | 410.5 | 422.1 | 407.9 | hipBLASLt contiguous 422.1 |
| 512x2560x8192 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 |  | 443.0 | 424.8 | 154.9 | 146.0 | 155.8 | 146.5 | torch | 556.1 | 539.6 | 561.2 | 543.1 | hipBLASLt contiguous 561.2 |
| 512x4096x8192 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 564.2 | 551.4 | 245.5 | 236.2 | 247.4 | 238.1 | flydsl | 681.2 | 443.1 (copy) | 649.2 | 622.3 | AITER flydsl contiguous 681.2 |
| 512x4608x8192 | bfloat16 | 36 | one-tile owner, n = 4 |  | 609.3 | 592.4 | 278.8 | 265.2 | 279.8 | 266.4 | torch | 718.9 | 710.6 | 721.6 | 710.8 | hipBLASLt contiguous 721.6 |
| 512x7168x8192 | bfloat16 | 56 | one-tile owner, n = 3 |  | 803.4 | 791.5 | 432.8 | 412.2 | 434.3 | 413.4 | flydsl | 801.5 | 481.6 (copy) | 742.7 | 738.7 | AITER flydsl contiguous 801.5 |
| 512x8192x8192 | bfloat16 | 64 | one-tile owner, n = 3 |  | 890.9 | 887.1 | 475.6 | 469.0 | 478.7 | 470.7 | flydsl | 685.3 | 447.7 (copy) | 768.8 | 751.6 | hipBLASLt contiguous 768.8 |
| 512x10240x8192 | bfloat16 | 80 | one-tile owner, n = 3-4 |  | 990.6 | 996.3 | 589.6 | 583.5 | 605.7 | 587.4 | flydsl | 623.4 | 415.4 (copy) | 570.3 | 564.1 | AITER flydsl contiguous 623.4 |
| 512x14336x8192 | bfloat16 | 112 | one-tile owner, n = 2-3 |  | 1088.0 | 1087.1 | 849.7 | 794.9 | 842.1 | 813.3 | torch | 1039.4 | 1032.2 | 1047.8 | 1025.4 | hipBLASLt contiguous 1047.8 |
| 512x16384x8192 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1161.5 | 1181.6 | 904.3 | 909.2 | 895.8 | 912.1 | flydsl | 650.0 | 436.8 (copy) | 922.6 | 975.9 | hipBLASLt padded 975.9 |
| 512x57344x8192 | bfloat16 | 448 | data-parallel |  | 1346.1 | 1357.6 | 1346.1 | 1357.6 | 1332.0 | 1349.2 | flydsl | 1293.0 | 686.0 (copy) | 1267.8 | 1282.2 | AITER flydsl contiguous 1293.0 |
| 768x4096x8192 | bfloat16 | 48 | one-tile owner, n = 4 |  | 773.6 | 767.3 | 373.9 | 356.6 | 373.5 | 359.2 | flydsl | 743.6 | 548.0 (copy) | 728.9 | 676.2 | AITER flydsl contiguous 743.6 |
| 1024x512x8192 | bfloat16 | 8 | one-tile reduce-scatter, n = 16 |  | 273.2 | 267.7 | 62.4 | 58.1 | 62.5 | 58.2 | torch | 335.7 | 327.3 | 336.1 | 338.5 | hipBLASLt padded 338.5 |
| 1024x1280x8192 | bfloat16 | 20 | one-tile reduce-scatter, n = 6 |  | 440.8 | 429.3 | 157.0 | 146.6 | 157.6 | 147.5 | torch | 575.0 | 550.0 | 576.8 | 553.8 | hipBLASLt contiguous 576.8 |
| 1024x2560x8192 | bfloat16 | 40 | one-tile owner, n = 4 |  | 664.6 | 655.9 | 312.4 | 293.8 | 314.8 | 296.4 | torch | 786.2 | 771.0 | 789.6 | 763.7 | hipBLASLt contiguous 789.6 |
| 1024x4096x8192 | bfloat16 | 64 | one-tile owner, n = 3 |  | 898.1 | 902.8 | 492.0 | 475.1 | 491.6 | 476.7 | flydsl | 899.1 | 657.3 (copy) | 809.9 | 771.2 | AITER flydsl contiguous 899.1 |
| 1024x4608x8192 | bfloat16 | 72 | one-tile owner, n = 3-4 |  | 917.7 | 909.0 | 551.6 | 530.4 | 553.5 | 532.0 | torch | 944.0 | 922.3 | 948.7 | 924.6 | hipBLASLt contiguous 948.7 |
| 1024x7168x8192 | bfloat16 | 112 | one-tile owner, n = 2-3 |  | 1138.8 | 1103.6 | 826.3 | 811.7 | 835.9 | 820.3 | flydsl | 741.2 | 577.1 (copy) | 1071.0 | 1054.4 | hipBLASLt contiguous 1071.0 |
| 1024x8192x8192 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1224.4 | 1235.3 | 926.1 | 931.8 | 928.4 | 933.7 | flydsl | 913.4 | 651.4 (copy) | 1012.2 | 990.1 | hipBLASLt contiguous 1012.2 |
| 1024x10240x8192 | bfloat16 | 160 | data-parallel | yes | 1133.4 | 1142.0 | 1136.9 | 1137.1 | 1136.8 | 1143.0 | flydsl | 823.9 | 628.7 (copy) | 1216.1 | 1201.0 | hipBLASLt contiguous 1216.1 |
| 1024x14336x8192 | bfloat16 | 224 | data-parallel | yes | 1392.5 | 1412.6 | 1393.9 | 1405.0 | 1397.2 | 1411.9 | flydsl | 1360.6 | 887.1 (copy) | 1311.1 | 1315.6 | AITER flydsl contiguous 1360.6 |
| 1024x16384x8192 | bfloat16 | 256 | data-parallel | yes | 1487.8 | 1505.2 | 1489.3 | 1495.9 | 1479.2 | 1504.2 | torch | 1426.5 | 1425.7 | 1443.0 | 1422.7 | hipBLASLt contiguous 1443.0 |
| 1024x57344x8192 | bfloat16 | 896 | one-tile owner, n = 2 |  | 1493.0 | 1499.0 | 1427.0 | 1429.6 | 1416.8 | 1419.5 | asm | 1374.7 | 1388.9 | 1327.4 | 1332.0 | AITER asm padded 1388.9 |
| 2048x512x8192 | bfloat16 | 16 | one-tile reduce-scatter, n = 8 |  | 421.6 | 410.9 | 126.0 | 117.7 | 126.7 | 118.5 | torch | 504.2 | 498.6 | 505.4 | 508.3 | hipBLASLt padded 508.3 |
| 2048x1280x8192 | bfloat16 | 40 | one-tile owner, n = 4 |  | 666.6 | 662.6 | 315.8 | 299.3 | 318.1 | 299.7 | torch | 692.9 | 697.4 | 695.1 | 701.2 | hipBLASLt padded 701.2 |
| 2048x2560x8192 | bfloat16 | 80 | one-tile owner, n = 3-4 |  | 1048.1 | 1032.1 | 611.8 | 598.7 | 616.8 | 603.0 | torch | 988.1 | 976.3 | 995.3 | 967.3 | hipBLASLt contiguous 995.3 |
| 2048x4096x8192 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1259.3 | 1270.1 | 940.8 | 934.5 | 941.5 | 943.0 | opus | 1109.5 | 1130.8 | 1046.1 | 1032.0 | AITER opus padded 1130.8 |
| 2048x4608x8192 | bfloat16 | 144 | data-parallel | yes | 1045.5 | 1057.0 | 1047.3 | 1060.7 | 1051.4 | 1058.7 | torch | 1159.8 | 1143.9 | 1169.2 | 1144.1 | hipBLASLt contiguous 1169.2 |
| 2048x7168x8192 | bfloat16 | 224 | data-parallel | yes | 1407.2 | 1417.1 | 1413.5 | 1421.0 | 1423.8 | 1427.7 | asm | 1375.2 | 1387.3 | 1338.8 | 1335.5 | AITER asm padded 1387.3 |
| 2048x8192x8192 | bfloat16 | 256 | data-parallel | yes | 1511.5 | 1534.2 | 1518.7 | 1526.2 | 1519.6 | 1534.8 | torch | 1454.3 | 1466.2 | 1467.2 | 1467.2 | hipBLASLt contiguous 1467.2 |
| 2048x10240x8192 | bfloat16 | 320 | one-tile owner, n = 3 |  | 1381.7 | 1397.5 | 1172.7 | 1179.9 | 1169.5 | 1177.1 | flydsl | 930.9 | 769.6 (copy) | 1253.3 | 1236.7 | hipBLASLt contiguous 1253.3 |
| 2048x14336x8192 | bfloat16 | 448 | data-parallel |  | 1431.4 | 1443.8 | 1431.4 | 1443.8 | 1436.7 | 1444.6 | flydsl | 1397.9 | 1094.4 (copy) | 1344.9 | 1345.4 | AITER flydsl contiguous 1397.9 |
| 2048x16384x8192 | bfloat16 | 512 | data-parallel |  | 1528.9 | 1552.4 | 1528.9 | 1552.4 | 1529.1 | 1550.4 | asm | 1484.2 | 1492.1 | 1485.2 | 1478.0 | AITER asm padded 1492.1 |
| 2048x57344x8192 | bfloat16 | 1792 | data-parallel |  | 1574.0 | 1578.7 | 1574.0 | 1578.7 | 1567.0 | 1569.0 | flydsl | 1508.1 | 1204.1 (copy) | 1482.2 | 1482.7 | AITER flydsl contiguous 1508.1 |
| 4096x512x8192 | bfloat16 | 32 | one-tile reduce-scatter, n = 4 |  | 560.4 | 551.6 | 246.1 | 238.0 | 249.1 | 237.8 | flydsl | 644.1 | 449.7 (copy) | 668.5 | 629.6 | hipBLASLt contiguous 668.5 |
| 4096x1280x8192 | bfloat16 | 80 | one-tile owner, n = 3-4 |  | 1031.1 | 1016.2 | 606.9 | 593.5 | 610.9 | 592.5 | torch | 987.1 | 955.2 | 994.2 | 956.6 | hipBLASLt contiguous 994.2 |
| 4096x2560x8192 | bfloat16 | 160 | data-parallel | yes | 1145.3 | 1162.7 | 1149.9 | 1167.1 | 1149.9 | 1168.0 | torch | 1240.4 | 1233.3 | 1254.4 | 1232.8 | hipBLASLt contiguous 1254.4 |
| 4096x4096x8192 | bfloat16 | 256 | data-parallel | yes | 1513.0 | 1538.5 | 1527.2 | 1540.1 | 1511.4 | 1545.5 | torch | 1466.9 | 1467.8 | 1479.4 | 1469.6 | hipBLASLt contiguous 1479.4 |
| 4096x4608x8192 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 |  | 1353.1 | 1358.1 | 1082.5 | 1078.0 | 1074.5 | 1076.5 | torch | 1171.8 | 1154.0 | 1182.5 | 1155.6 | hipBLASLt contiguous 1182.5 |
| 4096x7168x8192 | bfloat16 | 448 | data-parallel |  | 1441.1 | 1453.8 | 1441.1 | 1453.8 | 1451.5 | 1456.7 | flydsl | 1411.4 | 1194.9 (copy) | 1369.1 | 1364.1 | AITER flydsl contiguous 1411.4 |
| 4096x8192x8192 | bfloat16 | 512 | data-parallel |  | 1539.5 | 1568.4 | 1539.5 | 1568.4 | 1543.2 | 1568.4 | asm | 1516.8 | 1524.5 | 1506.5 | 1501.5 | AITER asm padded 1524.5 |
| 4096x10240x8192 | bfloat16 | 640 | one-tile owner, n = 2 |  | 1508.1 | 1525.9 | 1429.7 | 1424.5 | 1407.4 | 1418.7 | flydsl | 1372.9 | 1180.8 (copy) | 1338.1 | 1334.7 | AITER flydsl contiguous 1372.9 |
| 4096x14336x8192 | bfloat16 | 896 | one-tile owner, n = 2 |  | 1529.7 | 1549.3 | 1471.8 | 1468.8 | 1459.3 | 1463.6 | flydsl | 1416.2 | 1224.9 (copy) | 1359.9 | 1359.9 | AITER flydsl contiguous 1416.2 |
| 4096x16384x8192 | bfloat16 | 1024 | data-parallel |  | 1562.2 | 1587.4 | 1562.2 | 1587.4 | 1564.3 | 1579.5 | flydsl | 1511.7 | 1315.9 (copy) | 1511.4 | 1495.6 | AITER flydsl contiguous 1511.7 |
| 4096x57344x8192 | bfloat16 | 3584 | data-parallel |  | 1591.9 | 1600.1 | 1591.9 | 1600.1 | 1587.5 | 1592.0 | flydsl | 1533.9 | 1356.9 (copy) | 1510.5 | 1513.4 | AITER flydsl contiguous 1533.9 |
| 8192x512x8192 | bfloat16 | 64 | one-tile owner, n = 3 |  | 884.6 | 888.2 | 473.0 | 469.5 | 477.0 | 469.6 | flydsl | 870.1 | 506.1 (copy) | 795.7 | 763.2 | AITER flydsl contiguous 870.1 |
| 8192x1280x8192 | bfloat16 | 160 | data-parallel | yes | 1123.4 | 1140.1 | 1125.6 | 1127.3 | 1122.2 | 1140.1 | torch | 1219.0 | 1217.8 | 1233.8 | 1222.6 | hipBLASLt contiguous 1233.8 |
| 8192x2560x8192 | bfloat16 | 320 | one-tile owner, n = 3 |  | 1378.2 | 1394.7 | 1160.0 | 1150.9 | 1152.7 | 1152.2 | torch | 1253.4 | 1252.6 | 1264.2 | 1252.9 | hipBLASLt contiguous 1264.2 |
| 8192x4096x8192 | bfloat16 | 512 | data-parallel |  | 1548.0 | 1573.3 | 1548.0 | 1573.3 | 1554.6 | 1573.8 | torch | 1531.1 | 1540.0 | 1556.8 | 1542.3 | hipBLASLt contiguous 1556.8 |
| 8192x4608x8192 | bfloat16 | 576 | one-tile owner, n = 3 |  | 1467.7 | 1483.9 | 1339.8 | 1336.1 | 1328.7 | 1333.8 | torch | 1338.6 | 1350.3 | 1353.4 | 1349.7 | hipBLASLt contiguous 1353.4 |
| 8192x7168x8192 | bfloat16 | 896 | one-tile owner, n = 2 |  | 1531.2 | 1550.8 | 1471.2 | 1470.2 | 1458.5 | 1468.4 | asm | 1415.8 | 1427.1 | 1399.6 | 1391.6 | AITER asm padded 1427.1 |
| 8192x8192x8192 | bfloat16 | 1024 | data-parallel |  | 1564.3 | 1594.1 | 1564.3 | 1594.1 | 1574.3 | 1585.9 | asm | 1543.8 | 1552.5 | 1516.0 | 1516.6 | AITER asm padded 1552.5 |
| 8192x10240x8192 | bfloat16 | 1280 | data-parallel |  | 1581.1 | 1599.1 | 1581.1 | 1599.1 | 1587.7 | 1591.1 | flydsl | 1527.2 | 1378.2 (copy) | 1553.2 | 1528.6 | hipBLASLt contiguous 1553.2 |
| 8192x14336x8192 | bfloat16 | 1792 | data-parallel |  | 1578.0 | 1605.4 | 1578.0 | 1605.4 | 1592.8 | 1597.3 | flydsl | 1533.6 | 1402.9 (copy) | 1533.0 | 1526.4 | AITER flydsl contiguous 1533.6 |
| 8192x16384x8192 | bfloat16 | 2048 | data-parallel |  | 1573.8 | 1605.0 | 1573.8 | 1605.0 | 1582.0 | 1593.7 | asm | 1544.4 | 1536.3 | 1531.6 | 1519.4 | AITER asm contiguous 1544.4 |
| 8192x57344x8192 | bfloat16 | 7168 | data-parallel |  | 1602.8 | 1608.8 | 1602.8 | 1608.8 | 1601.0 | 1602.4 | asm | 1545.3 | 1553.8 | 1465.3 | 1466.3 | AITER asm padded 1553.8 |
| 16384x512x8192 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1162.7 | 1184.2 | 904.3 | 889.3 | 899.2 | 904.4 | torch | 1070.8 | 1068.1 | 1088.1 | 1066.3 | hipBLASLt contiguous 1088.1 |
| 16384x1280x8192 | bfloat16 | 320 | one-tile owner, n = 3 |  | 1346.7 | 1366.5 | 1155.9 | 1151.4 | 1145.6 | 1150.1 | torch | 1242.0 | 1252.5 | 1255.3 | 1252.8 | hipBLASLt contiguous 1255.3 |
| 16384x2560x8192 | bfloat16 | 640 | one-tile owner, n = 2 |  | 1477.0 | 1498.5 | 1393.4 | 1398.9 | 1380.6 | 1393.0 | flydsl | 1340.1 | 1111.0 (copy) | 1307.4 | 1307.6 | AITER flydsl contiguous 1340.1 |
| 16384x4608x8192 | bfloat16 | 1152 | one-tile owner, n = 2 |  | 1526.2 | 1545.7 | 1481.5 | 1484.1 | 1463.1 | 1474.1 | flydsl | 1420.6 | 1261.5 (copy) | 1210.0 | 1190.7 | AITER flydsl contiguous 1420.6 |
| 16384x7168x8192 | bfloat16 | 1792 | data-parallel |  | 1572.0 | 1592.1 | 1572.0 | 1592.1 | 1572.3 | 1579.7 | torch | 1515.1 | 1528.9 | 1527.0 | 1528.8 | AITER torch padded 1528.9 |
| 16384x8192x8192 | bfloat16 | 2048 | data-parallel |  | 1576.2 | 1599.8 | 1576.2 | 1599.8 | 1585.5 | 1588.9 | torch | 1541.3 | 1557.1 | 1571.6 | 1547.8 | hipBLASLt contiguous 1571.6 |
| 16384x10240x8192 | bfloat16 | 2560 | data-parallel |  | 1590.4 | 1604.6 | 1590.4 | 1604.6 | 1589.0 | 1593.0 | torch | 1560.8 | 1563.8 | 1577.9 | 1564.3 | hipBLASLt contiguous 1577.9 |
| 16384x14336x8192 | bfloat16 | 3584 | data-parallel |  | 1597.6 | 1606.2 | 1597.6 | 1606.2 | 1594.9 | 1596.3 | asm | 1547.2 | 1556.0 | 1552.3 | 1554.7 | AITER asm padded 1556.0 |
| 16384x16384x8192 | bfloat16 | 4096 | data-parallel |  | 1599.8 | 1609.2 | 1599.8 | 1609.2 | 1598.3 | 1597.3 | asm | 1558.7 | 1552.3 | 1539.1 | 1531.4 | AITER asm contiguous 1558.7 |
| 16384x57344x8192 | bfloat16 | 14336 | data-parallel |  | 1596.9 | 1608.8 | 1596.9 | 1608.8 | 1609.3 | 1601.7 | asm | 1560.0 | 1559.7 | 1504.5 | 1494.5 | AITER asm contiguous 1560.0 |
| 32768x1280x8192 | bfloat16 | 640 | one-tile owner, n = 2 |  | 1444.5 | 1458.2 | 1361.5 | 1371.6 | 1348.4 | 1363.1 | flydsl | 1313.1 | 962.4 (copy) | 1335.1 | 1330.7 | hipBLASLt contiguous 1335.1 |
| 32768x2560x8192 | bfloat16 | 1280 | data-parallel |  | 1534.5 | 1549.9 | 1534.5 | 1549.9 | 1529.5 | 1540.4 | torch | 1487.5 | 1504.4 | 1498.2 | 1508.8 | hipBLASLt padded 1508.8 |
| 32768x7168x8192 | bfloat16 | 3584 | data-parallel |  | 1583.8 | 1586.3 | 1583.8 | 1586.3 | 1577.5 | 1574.9 | torch | 1553.0 | 1566.0 | 1577.8 | 1568.5 | hipBLASLt contiguous 1577.8 |
| 32768x8192x8192 | bfloat16 | 4096 | data-parallel |  | 1595.4 | 1594.9 | 1595.4 | 1594.9 | 1588.1 | 1581.6 | flydsl | 1551.2 | 1444.3 (copy) | 1580.6 | 1580.6 | hipBLASLt contiguous 1580.6 |
| 32768x10240x8192 | bfloat16 | 5120 | data-parallel |  | 1598.7 | 1598.3 | 1598.7 | 1598.3 | 1594.6 | 1586.4 | torch | 1579.6 | 1583.4 | 1597.6 | 1590.6 | hipBLASLt contiguous 1597.6 |
| 32768x14336x8192 | bfloat16 | 7168 | data-parallel |  | 1601.4 | 1593.6 | 1601.4 | 1593.6 | 1598.2 | 1586.9 | asm | 1517.1 | 1562.2 | 1538.7 | 1545.8 | AITER asm padded 1562.2 |
| 32768x16384x8192 | bfloat16 | 8192 | data-parallel |  | 1587.6 | 1583.6 | 1587.6 | 1583.6 | 1596.9 | 1581.3 | asm | 1555.5 | 1553.2 | 1524.2 | 1499.6 | AITER asm contiguous 1555.5 |
| 32768x57344x8192 | bfloat16 | 28672 | data-parallel |  | 1561.6 | 1534.0 | 1561.6 | 1534.0 | 1577.7 | 1572.6 | asm | 1552.1 | 1546.4 | 1426.4 | 1443.5 | AITER asm contiguous 1552.1 |
| 512x6144x12288 | bfloat16 | 48 | one-tile owner, n = 4 |  | 860.1 | 874.9 | 347.8 | 361.6 | 347.7 | 359.7 | torch | 783.8 | 784.1 | 785.6 | 784.9 | hipBLASLt contiguous 785.6 |
| 1024x6144x12288 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 1061.0 | 1074.4 | 695.2 | 720.3 | 695.8 | 719.3 | torch | 1099.1 | 1135.3 | 1102.7 | 1126.6 | AITER torch padded 1135.3 |
| 2048x6144x12288 | bfloat16 | 192 | data-parallel | yes | 1303.9 | 1330.6 | 1307.6 | 1326.7 | 1314.7 | 1330.8 | torch | 1309.8 | 1341.8 | 1327.1 | 1340.7 | AITER torch padded 1341.8 |
| 4096x6144x12288 | bfloat16 | 384 | one-tile owner, n = 2 |  | 1500.2 | 1529.7 | 1355.9 | 1349.5 | 1350.5 | 1356.0 | torch | 1325.3 | 1352.4 | 1343.8 | 1351.4 | AITER torch padded 1352.4 |
| 8192x6144x12288 | bfloat16 | 768 | data-parallel |  | 1552.7 | 1584.1 | 1552.7 | 1584.1 | 1558.1 | 1581.2 | torch | 1509.2 | 1550.1 | 1526.1 | 1557.4 | hipBLASLt padded 1557.4 |
| 16384x6144x12288 | bfloat16 | 1536 | data-parallel |  | 1565.1 | 1592.9 | 1565.1 | 1592.9 | 1565.5 | 1585.2 | torch | 1512.9 | 1573.6 | 1530.8 | 1567.7 | AITER torch padded 1573.6 |
| 32768x6144x12288 | bfloat16 | 3072 | data-parallel |  | 1576.5 | 1598.9 | 1576.5 | 1598.9 | 1570.4 | 1589.1 | asm | 1543.6 | 1554.4 | 1557.8 | 1584.0 | hipBLASLt padded 1584.0 |
| 512x16384x13312 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1288.3 | 1273.8 | 922.6 | 911.8 | 929.9 | 914.2 | flydsl | 710.2 | 465.1 (copy) | 1108.5 | 1089.6 | hipBLASLt contiguous 1108.5 |
| 1024x16384x13312 | bfloat16 | 256 | data-parallel | yes | 1520.9 | 1536.5 | 1531.4 | 1528.8 | 1530.1 | 1537.8 | asm | 1482.2 | 1494.0 | 1489.5 | 1498.0 | hipBLASLt padded 1498.0 |
| 2048x16384x13312 | bfloat16 | 512 | data-parallel |  | 1570.7 | 1582.1 | 1570.7 | 1582.1 | 1559.1 | 1579.8 | flydsl | 1005.5 | 838.7 (copy) | 1509.8 | 1518.8 | hipBLASLt padded 1518.8 |
| 4096x16384x13312 | bfloat16 | 1024 | data-parallel |  | 1595.5 | 1608.5 | 1595.5 | 1608.5 | 1596.8 | 1608.7 | flydsl | 1540.0 | 1333.8 (copy) | 1517.9 | 1533.0 | AITER flydsl contiguous 1540.0 |
| 8192x16384x13312 | bfloat16 | 2048 | data-parallel |  | 1579.0 | 1597.8 | 1579.0 | 1597.8 | 1571.1 | 1588.5 | flydsl | 1518.9 | 1391.6 (copy) | 1517.8 | 1527.7 | hipBLASLt padded 1527.7 |
| 16384x16384x13312 | bfloat16 | 4096 | data-parallel |  | 1578.4 | 1592.5 | 1578.4 | 1592.5 | 1576.7 | 1586.0 | asm | 1518.1 | 1516.7 | 1425.0 | 1502.2 | AITER asm contiguous 1518.1 |
| 32768x16384x13312 | bfloat16 | 8192 | data-parallel |  | 1584.4 | 1594.9 | 1584.4 | 1594.9 | 1579.6 | 1589.8 | flydsl | 1526.8 | 1456.4 (copy) | 1474.7 | 1401.0 | AITER flydsl contiguous 1526.8 |
| 512x2304x16384 | bfloat16 | 18 | one-tile reduce-scatter, n = 7 |  | 546.7 | 508.1 | 139.3 | 129.5 | 138.8 | 131.5 | torch | 704.4 | 615.3 | 707.7 | 621.3 | hipBLASLt contiguous 707.7 |
| 512x4608x16384 | bfloat16 | 36 | one-tile owner, n = 6 |  | 837.3 | 759.3 | 281.1 | 261.3 | 276.4 | 266.8 | flydsl | 506.6 | 346.8 (copy) | 794.6 | 769.4 | hipBLASLt contiguous 794.6 |
| 512x9216x16384 | bfloat16 | 72 | one-tile owner, n = 3-4 |  | 1066.6 | 1015.7 | 550.9 | 509.6 | 514.4 | 520.0 | flydsl | 589.2 | 426.1 (copy) | 760.3 | 740.3 | hipBLASLt contiguous 760.3 |
| 512x13312x16384 | bfloat16 | 104 | one-tile owner, n = 2-3 |  | 1145.3 | 1111.9 | 745.9 | 715.4 | 734.2 | 733.6 | flydsl | 542.7 | 383.4 (copy) | 1089.6 | 1105.2 | hipBLASLt padded 1105.2 |
| 512x26624x16384 | bfloat16 | 208 | data-parallel | yes | 1303.4 | 1242.1 | 1301.8 | 1238.6 | 1223.7 | 1248.9 | flydsl | 763.1 | 493.8 (copy) | 1187.0 | 1206.5 | hipBLASLt padded 1206.5 |
| 512x53248x16384 | bfloat16 | 416 | data-parallel |  | 1304.2 | 1259.1 | 1304.2 | 1259.1 | 1299.3 | 1276.1 | flydsl | 631.2 | 442.1 (copy) | 1210.3 | 1215.8 | hipBLASLt padded 1215.8 |
| 1024x2304x16384 | bfloat16 | 36 | one-tile owner, n = 6 |  | 840.7 | 690.2 | 277.8 | 236.3 | 275.0 | 245.1 | torch | 870.6 | 762.4 | 876.5 | 764.5 | hipBLASLt contiguous 876.5 |
| 1024x4608x16384 | bfloat16 | 72 | one-tile owner, n = 3-4 |  | 1112.4 | 1037.6 | 547.0 | 483.4 | 551.8 | 498.0 | flydsl | 644.3 | 514.4 (copy) | 1037.2 | 975.7 | hipBLASLt contiguous 1037.2 |
| 1024x9216x16384 | bfloat16 | 144 | data-parallel | yes | 1058.0 | 954.3 | 1053.5 | 951.5 | 1055.7 | 972.5 | flydsl | 678.2 | 548.4 (copy) | 1166.2 | 1122.8 | hipBLASLt contiguous 1166.2 |
| 1024x13312x16384 | bfloat16 | 208 | data-parallel | yes | 1359.1 | 1316.1 | 1361.5 | 1322.3 | 1362.8 | 1330.5 | asm | 1328.2 | 1325.1 | 1242.5 | 1255.9 | AITER asm contiguous 1328.2 |
| 1024x26624x16384 | bfloat16 | 416 | data-parallel |  | 1377.4 | 1348.8 | 1377.4 | 1348.8 | 1375.0 | 1331.5 | flydsl | 944.3 | 710.1 (copy) | 1256.0 | 1260.3 | hipBLASLt padded 1260.3 |
| 1024x53248x16384 | bfloat16 | 832 | one-tile owner, n = 4 |  | 1525.6 | 1506.1 | 1374.8 | 1353.4 | 1361.5 | 1352.9 | triton | 986.4 | 958.2 | 1257.4 | 1268.3 | hipBLASLt padded 1268.3 |
| 2048x2304x16384 | bfloat16 | 72 | one-tile owner, n = 3-4 |  | 1137.3 | 1027.7 | 551.2 | 472.1 | 552.1 | 501.9 | torch | 871.9 | 825.2 | 876.6 | 824.8 | hipBLASLt contiguous 876.6 |
| 2048x4608x16384 | bfloat16 | 144 | data-parallel | yes | 1048.6 | 961.4 | 1053.5 | 954.6 | 1059.9 | 996.6 | flydsl | 715.3 | 613.4 (copy) | 1196.2 | 1152.8 | hipBLASLt contiguous 1196.2 |
| 2048x9216x16384 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 |  | 1409.1 | 1360.4 | 1073.4 | 995.6 | 1074.7 | 1005.0 | flydsl | 903.4 | 749.6 (copy) | 1162.0 | 1126.9 | hipBLASLt contiguous 1162.0 |
| 2048x13312x16384 | bfloat16 | 416 | data-parallel |  | 1403.5 | 1368.9 | 1403.5 | 1368.9 | 1405.3 | 1372.8 | flydsl | 1359.0 | 1072.4 (copy) | 1243.7 | 1244.3 | AITER flydsl contiguous 1359.0 |
| 2048x26624x16384 | bfloat16 | 832 | one-tile owner, n = 4 |  | 1558.9 | 1512.4 | 1405.3 | 1363.0 | 1397.4 | 1367.7 | flydsl | 904.8 | 778.0 (copy) | 1242.0 | 1247.2 | hipBLASLt padded 1247.2 |
| 2048x53248x16384 | bfloat16 | 1664 | one-tile owner, n = 2 |  | 1581.2 | 1566.7 | 1532.0 | 1496.8 | 1510.1 | 1497.4 | flydsl | 772.4 | 681.8 (copy) | 1290.1 | 1293.1 | hipBLASLt padded 1293.1 |
| 4096x2304x16384 | bfloat16 | 144 | data-parallel | yes | 1020.6 | 937.1 | 1017.0 | 933.6 | 1022.0 | 965.1 | torch | 1177.5 | 1143.6 | 1189.4 | 1149.5 | hipBLASLt contiguous 1189.4 |
| 4096x4608x16384 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 |  | 1416.5 | 1359.1 | 1076.5 | 1001.9 | 1066.3 | 1013.1 | torch | 1174.6 | 1136.2 | 1184.2 | 1138.3 | hipBLASLt contiguous 1184.2 |
| 4096x9216x16384 | bfloat16 | 576 | one-tile owner, n = 4 |  | 1533.6 | 1522.4 | 1367.0 | 1322.4 | 1354.2 | 1324.4 | torch | 1321.2 | 1339.8 | 1336.4 | 1341.2 | hipBLASLt padded 1341.2 |
| 4096x13312x16384 | bfloat16 | 832 | one-tile owner, n = 4 |  | 1560.2 | 1561.5 | 1422.2 | 1408.5 | 1412.5 | 1407.5 | flydsl | 1373.4 | 1185.8 (copy) | 1264.8 | 1276.6 | AITER flydsl contiguous 1373.4 |
| 4096x26624x16384 | bfloat16 | 1664 | one-tile owner, n = 2 |  | 1591.6 | 1566.1 | 1537.9 | 1501.1 | 1523.4 | 1499.9 | flydsl | 1485.8 | 1298.1 (copy) | 1318.9 | 1323.1 | AITER flydsl contiguous 1485.8 |
| 4096x53248x16384 | bfloat16 | 3328 | data-parallel |  | 1595.2 | 1615.7 | 1595.2 | 1615.7 | 1599.2 | 1609.1 | flydsl | 1555.3 | 1350.8 (copy) | 1487.5 | 1506.7 | AITER flydsl contiguous 1555.3 |
| 8192x2304x16384 | bfloat16 | 288 | one-tile reduce-scatter, n = 4 |  | 1371.8 | 1300.8 | 1016.1 | 928.2 | 963.9 | 917.6 | torch | 1176.2 | 1133.6 | 1185.1 | 1133.2 | hipBLASLt contiguous 1185.1 |
| 8192x4608x16384 | bfloat16 | 576 | one-tile owner, n = 4 |  | 1500.7 | 1440.2 | 1319.6 | 1219.2 | 1283.7 | 1213.5 | torch | 1339.3 | 1356.3 | 1357.3 | 1355.7 | hipBLASLt contiguous 1357.3 |
| 8192x9216x16384 | bfloat16 | 1152 | one-tile owner, n = 2 |  | 1546.8 | 1501.3 | 1474.5 | 1408.9 | 1451.9 | 1399.7 | flydsl | 1403.0 | 1263.5 (copy) | 1142.4 | 1142.5 | AITER flydsl contiguous 1403.0 |
| 8192x13312x16384 | bfloat16 | 1664 | one-tile owner, n = 2 |  | 1557.6 | 1502.9 | 1505.9 | 1440.2 | 1491.4 | 1434.5 | flydsl | 1447.9 | 1319.6 (copy) | 1306.9 | 1316.5 | AITER flydsl contiguous 1447.9 |
| 8192x26624x16384 | bfloat16 | 3328 | data-parallel |  | 1561.7 | 1518.9 | 1561.7 | 1518.9 | 1558.7 | 1510.4 | flydsl | 1520.9 | 1402.1 (copy) | 1479.4 | 1484.5 | AITER flydsl contiguous 1520.9 |
| 8192x53248x16384 | bfloat16 | 6656 | data-parallel |  | 1554.5 | 1563.4 | 1554.5 | 1563.4 | 1573.8 | 1551.3 | flydsl | 1527.4 | 1406.5 (copy) | 1455.7 | 1476.0 | AITER flydsl contiguous 1527.4 |
| 16384x2304x16384 | bfloat16 | 576 | one-tile owner, n = 4 |  | 1482.0 | 1432.1 | 1307.8 | 1223.5 | 1299.9 | 1213.3 | torch | 1338.2 | 1335.6 | 1349.5 | 1336.2 | hipBLASLt contiguous 1349.5 |
| 16384x4608x16384 | bfloat16 | 1152 | one-tile owner, n = 2 |  | 1536.7 | 1519.6 | 1473.0 | 1419.4 | 1462.2 | 1423.6 | flydsl | 1421.5 | 1248.9 (copy) | 1175.4 | 1162.0 | AITER flydsl contiguous 1421.5 |
| 16384x9216x16384 | bfloat16 | 2304 | data-parallel |  | 1566.8 | 1508.3 | 1566.8 | 1508.3 | 1566.6 | 1512.6 | flydsl | 1513.9 | 1399.7 (copy) | 1533.2 | 1487.9 | hipBLASLt contiguous 1533.2 |
| 16384x13312x16384 | bfloat16 | 3328 | data-parallel |  | 1561.9 | 1519.2 | 1561.9 | 1519.2 | 1570.5 | 1519.7 | asm | 1498.6 | 1536.1 | 1487.3 | 1476.9 | AITER asm padded 1536.1 |
| 16384x26624x16384 | bfloat16 | 6656 | data-parallel |  | 1558.5 | 1526.5 | 1558.5 | 1526.5 | 1571.6 | 1524.2 | flydsl | 1513.8 | 1432.1 (copy) | 1451.3 | 1481.6 | AITER flydsl contiguous 1513.8 |
| 16384x53248x16384 | bfloat16 | 13312 | data-parallel |  | 1551.8 | 1550.6 | 1551.8 | 1550.6 | 1566.9 | 1543.9 | asm | 1486.1 | 1539.9 | 1457.2 | 1482.1 | AITER asm padded 1539.9 |
| 32768x2304x16384 | bfloat16 | 1152 | one-tile owner, n = 2 |  | 1518.1 | 1524.4 | 1455.5 | 1425.7 | 1446.9 | 1405.2 | flydsl | 1410.4 | 1150.2 (copy) | 1185.9 | 1157.9 | AITER flydsl contiguous 1410.4 |
| 32768x4608x16384 | bfloat16 | 2304 | data-parallel |  | 1556.6 | 1548.4 | 1556.6 | 1548.4 | 1558.4 | 1529.4 | torch | 1508.2 | 1515.3 | 1514.3 | 1516.8 | hipBLASLt padded 1516.8 |
| 32768x9216x16384 | bfloat16 | 4608 | data-parallel |  | 1557.2 | 1547.6 | 1557.2 | 1547.6 | 1569.0 | 1543.5 | asm | 1492.8 | 1502.2 | 1484.4 | 1466.3 | AITER asm padded 1502.2 |
| 32768x13312x16384 | bfloat16 | 6656 | data-parallel |  | 1547.8 | 1551.8 | 1547.8 | 1551.8 | 1568.0 | 1540.8 | asm | 1511.9 | 1536.4 | 1444.1 | 1504.0 | AITER asm padded 1536.4 |
| 32768x26624x16384 | bfloat16 | 13312 | data-parallel |  | 1542.3 | 1562.2 | 1542.3 | 1562.2 | 1567.1 | 1551.4 | flydsl | 1511.0 | 1475.3 (copy) | 1432.3 | 1481.2 | AITER flydsl contiguous 1511.0 |
| 32768x53248x16384 | bfloat16 | 26624 | data-parallel |  | 1551.3 | 1571.0 | 1551.3 | 1571.0 | 1574.1 | 1557.2 | flydsl | 1532.9 | 1490.3 (copy) | 1445.9 | 1487.7 | AITER flydsl contiguous 1532.9 |
| 512x6144x18432 | bfloat16 | 48 | one-tile owner, n = 5-6 |  | 975.2 | 979.5 | 353.9 | 360.7 | 353.3 | 361.5 | flydsl | 785.0 | 486.4 (copy) | 839.4 | 817.4 | hipBLASLt contiguous 839.4 |
| 1024x6144x18432 | bfloat16 | 96 | one-tile owner, n = 2-3 |  | 1129.5 | 1144.2 | 707.7 | 697.6 | 713.7 | 702.4 | torch | 1173.6 | 1191.7 | 1181.8 | 1191.0 | AITER torch padded 1191.7 |
| 2048x6144x18432 | bfloat16 | 192 | data-parallel | yes | 1333.5 | 1343.7 | 1338.2 | 1335.8 | 1341.2 | 1343.4 | torch | 1333.3 | 1359.5 | 1350.3 | 1359.9 | hipBLASLt padded 1359.9 |
| 4096x6144x18432 | bfloat16 | 384 | one-tile owner, n = 2 |  | 1542.2 | 1563.7 | 1361.1 | 1362.6 | 1358.9 | 1364.0 | torch | 1325.8 | 1355.4 | 1345.5 | 1358.1 | hipBLASLt padded 1358.1 |
| 8192x6144x18432 | bfloat16 | 768 | data-parallel |  | 1572.7 | 1601.9 | 1572.7 | 1601.9 | 1584.1 | 1597.8 | torch | 1492.5 | 1540.3 | 1511.0 | 1546.3 | hipBLASLt padded 1546.3 |
| 16384x6144x18432 | bfloat16 | 1536 | data-parallel |  | 1586.3 | 1603.3 | 1586.3 | 1603.3 | 1587.0 | 1599.5 | asm | 1533.9 | 1555.4 | 1522.2 | 1562.0 | hipBLASLt padded 1562.0 |
| 32768x6144x18432 | bfloat16 | 3072 | data-parallel |  | 1582.6 | 1586.0 | 1582.6 | 1586.0 | 1588.1 | 1592.4 | asm | 1541.7 | 1546.4 | 1494.1 | 1507.0 | AITER asm padded 1546.4 |
| 512x5120x25600 | bfloat16 | 40 | one-tile owner, n = 6-7 |  | 969.3 | 1001.0 | 293.9 | 307.9 | 294.9 | 308.7 | flydsl | 427.7 | 312.5 (copy) | 875.6 | 886.2 | hipBLASLt padded 886.2 |
| 1024x5120x25600 | bfloat16 | 80 | one-tile owner, n = 3-4 |  | 1282.2 | 1309.1 | 590.7 | 617.2 | 594.4 | 620.3 | torch | 1144.9 | 1163.8 | 1153.6 | 1164.4 | hipBLASLt padded 1164.4 |
| 2048x5120x25600 | bfloat16 | 160 | data-parallel | yes | 1169.9 | 1224.3 | 1184.8 | 1228.3 | 1180.1 | 1228.5 | torch | 1281.5 | 1302.9 | 1294.3 | 1302.6 | AITER torch padded 1302.9 |
| 4096x5120x25600 | bfloat16 | 320 | one-tile owner, n = 4 |  | 1530.9 | 1551.1 | 1157.7 | 1201.4 | 1156.7 | 1199.0 | torch | 1268.3 | 1292.6 | 1280.9 | 1292.0 | AITER torch padded 1292.6 |
| 8192x5120x25600 | bfloat16 | 640 | one-tile owner, n = 2 |  | 1565.2 | 1581.2 | 1431.3 | 1438.1 | 1416.8 | 1434.5 | asm | 1390.7 | 1407.7 | 1347.2 | 1366.2 | AITER asm padded 1407.7 |
| 16384x5120x25600 | bfloat16 | 1280 | data-parallel |  | 1587.9 | 1599.0 | 1587.9 | 1599.0 | 1588.6 | 1594.4 | torch | 1500.5 | 1570.8 | 1510.1 | 1575.5 | hipBLASLt padded 1575.5 |
| 32768x5120x25600 | bfloat16 | 2560 | data-parallel |  | 1592.7 | 1589.4 | 1592.7 | 1589.4 | 1591.2 | 1589.1 | torch | 1442.2 | 1441.4 | 1434.3 | 1439.1 | AITER torch contiguous 1442.2 |
| 512x16384x26624 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1380.8 | 1387.8 | 958.2 | 948.7 | 924.9 | 946.7 | flydsl | 649.3 | 438.1 (copy) | 1202.7 | 1198.3 | hipBLASLt contiguous 1202.7 |
| 1024x16384x26624 | bfloat16 | 256 | data-parallel | yes | 1559.0 | 1569.7 | 1562.9 | 1565.2 | 1558.6 | 1570.2 | flydsl | 753.2 | 583.0 (copy) | 1522.0 | 1534.4 | hipBLASLt padded 1534.4 |
| 2048x16384x26624 | bfloat16 | 512 | data-parallel |  | 1599.1 | 1604.2 | 1599.1 | 1604.2 | 1591.8 | 1605.5 | flydsl | 809.2 | 703.1 (copy) | 1531.8 | 1541.8 | hipBLASLt padded 1541.8 |
| 4096x16384x26624 | bfloat16 | 1024 | data-parallel |  | 1615.6 | 1612.9 | 1615.6 | 1612.9 | 1611.5 | 1618.0 | flydsl | 1557.6 | 1347.2 (copy) | 1536.6 | 1547.6 | AITER flydsl contiguous 1557.6 |
| 8192x16384x26624 | bfloat16 | 2048 | data-parallel |  | 1584.7 | 1581.4 | 1584.7 | 1581.4 | 1598.0 | 1599.4 | asm | 231.4 | 241.5 | 1458.4 | 1472.8 | hipBLASLt padded 1472.8 |
| 16384x16384x26624 | bfloat16 | 4096 | data-parallel |  | 1607.5 | 1601.5 | 1607.5 | 1601.5 | 1605.2 | 1612.6 | asm | 1561.4 | 1568.9 | 1476.4 | 1432.2 | AITER asm padded 1568.9 |
| 32768x16384x26624 | bfloat16 | 8192 | data-parallel |  | 1601.8 | 1559.6 | 1601.8 | 1559.6 | 1600.3 | 1604.3 | asm | 1557.4 | 1562.5 | 1454.0 | 1442.4 | AITER asm padded 1562.5 |
| 512x8192x28672 | bfloat16 | 64 | one-tile owner, n = 4 |  | 1223.7 | 1240.9 | 477.3 | 487.2 | 474.7 | 479.4 | flydsl | 789.3 | 484.4 (copy) | 922.9 | 919.3 | hipBLASLt contiguous 922.9 |
| 1024x8192x28672 | bfloat16 | 128 | one-tile owner, n = 2 |  | 1440.8 | 1467.9 | 940.9 | 972.1 | 942.7 | 963.9 | triton | 695.0 | 694.2 | 1348.0 | 1356.1 | hipBLASLt padded 1356.1 |
| 2048x8192x28672 | bfloat16 | 256 | data-parallel | yes | 1594.7 | 1603.6 | 1600.8 | 1603.5 | 1583.2 | 1604.3 | flydsl | 1033.8 | 841.2 (copy) | 1564.2 | 1570.9 | hipBLASLt padded 1570.9 |
| 4096x8192x28672 | bfloat16 | 512 | data-parallel |  | 1610.8 | 1619.7 | 1610.8 | 1619.7 | 1608.2 | 1620.8 | flydsl | 1551.9 | 1296.9 (copy) | 1561.8 | 1577.9 | hipBLASLt padded 1577.9 |
| 8192x8192x28672 | bfloat16 | 1024 | data-parallel |  | 1598.7 | 1612.7 | 1598.7 | 1612.7 | 1600.4 | 1611.4 | asm | 1561.1 | 1576.2 | 1545.5 | 1577.5 | hipBLASLt padded 1577.5 |
| 16384x8192x28672 | bfloat16 | 2048 | data-parallel |  | 1598.3 | 1602.9 | 1598.3 | 1602.9 | 1603.5 | 1606.7 | asm | 1539.0 | 1570.2 | 1531.1 | 1525.0 | AITER asm padded 1570.2 |
| 32768x8192x28672 | bfloat16 | 4096 | data-parallel |  | 1598.5 | 1577.9 | 1598.5 | 1577.9 | 1604.4 | 1602.2 | asm | 1564.4 | 1567.8 | 1478.2 | 1475.4 | AITER asm padded 1567.8 |
| 512x7168x35840 | bfloat16 | 56 | one-tile owner, n = 4-5 |  | 1173.0 | 1131.0 | 435.9 | 407.5 | 437.2 | 406.1 | flydsl | 815.5 | 497.7 (copy) | 964.2 | 957.2 | hipBLASLt contiguous 964.2 |
| 1024x7168x35840 | bfloat16 | 112 | one-tile owner, n = 2-3 |  | 1355.0 | 1362.8 | 860.0 | 814.3 | 864.7 | 812.1 | flydsl | 572.6 | 461.9 (copy) | 1292.4 | 1302.8 | hipBLASLt padded 1302.8 |
| 2048x7168x35840 | bfloat16 | 224 | spread |  | 1506.2 | 1522.0 | 1492.3 | 1494.4 | 1487.8 | 1493.5 | asm | 1441.8 | 1459.9 | 1410.0 | 1412.4 | AITER asm padded 1459.9 |
| 4096x7168x35840 | bfloat16 | 448 | spread |  | 1565.5 | 1575.3 | 1503.0 | 1505.3 | 1498.6 | 1501.0 | flydsl | 1457.4 | 1218.1 (copy) | 1403.7 | 1406.8 | AITER flydsl contiguous 1457.4 |
| 8192x7168x35840 | bfloat16 | 896 | one-tile owner, n = 2 |  | 1588.8 | 1597.6 | 1477.4 | 1486.9 | 1472.4 | 1482.4 | asm | 1422.9 | 1432.6 | 1404.7 | 1410.8 | AITER asm padded 1432.6 |
| 16384x7168x35840 | bfloat16 | 1792 | data-parallel |  | 1510.5 | 1519.2 | 1510.5 | 1519.2 | 1535.2 | 1577.0 | torch | 1454.8 | 1487.4 | 1464.8 | 1499.7 | hipBLASLt padded 1499.7 |
| 32768x7168x35840 | bfloat16 | 3584 | data-parallel |  | 1594.4 | 1597.8 | 1594.4 | 1597.8 | 1598.8 | 1602.2 | flydsl | 1467.7 | 1370.9 (copy) | 1442.3 | 1456.9 | AITER flydsl contiguous 1467.7 |
