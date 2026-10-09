## Update 2026-10-05: stream-K Gluon GEMM (`streamk_wip`) vs v9, AITER and hipBLASLt, warm and cold caches

`streamk_wip` is the work-in-progress stream-K version of the Gluon a16w16 kernel: tutorial v20, snapshot `7c55c4017abfc6cc` (WORKLOG up to entry 9). It uses the same 256x256x64 tile as v9. Since the last update:
- its host rule defaults to `capped`, which caps the one-tile split count and runs data-parallel where splitting does not pay (mostly short K);
- the reduce-scatter ending reads its slices in lane order, 1-3% faster.

Setup:
- the same 630 shapes, torch 2.10.0+rocm7.2.4 with hipBLASLt 1.2.2, AITER at `main` `a1455f1c4`, one MI355X;
- AITER's pick is `aiter.tuned_gemm.gemm_a16w16`; hipBLASLt is the torch call AITER's `torch` libtype makes (`F.linear`, or `torch.mm(..., out_dtype=torch.float32)` for fp32 output), and its tile is read from the solution name;
- two cache states: **warm** (back to back on the same buffers, as in the earlier updates) and **cold** (a 2048 MB rotating buffer, so no launch reads the inputs of the previous one);
- `do_bench_cudagraph`, 2 interleaved rounds, median.

Every `streamk_wip`, v9 and hipBLASLt output matched the fp32 reference. AITER's FlyDSL pick is outside the tolerance on 29 shapes (max error 1-4), as before. In the tables, "faster" and "slower" mean a gap of more than 1%, and "within 1%" holds the rest. Positive medians mean the first kernel named is faster.

**Summary**

| Comparison | Warm: faster | Warm: within 1% | Warm: slower | Warm: median | Cold: faster | Cold: within 1% | Cold: slower | Cold: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| streamk_wip vs AITER's pick | 327 | 37 | 266 | +1.6% | 356 | 26 | 248 | +2.5% |
| streamk_wip vs hipBLASLt | 333 | 39 | 258 | +1.6% | 346 | 24 | 260 | +2.1% |
| v9 vs AITER's pick | 286 | 42 | 302 | -0.3% | 295 | 32 | 303 | -0.4% |
| v9 vs hipBLASLt | 263 | 53 | 314 | -0.9% | 255 | 39 | 336 | -2.3% |
| AITER's pick vs hipBLASLt | 219 | 236 | 175 | -0.2% | 209 | 238 | 183 | -0.3% |

**By v9 grid size (number of 256x256 output tiles), warm**

| Output tiles | Shapes | v9 vs AITER: median | streamk_wip vs AITER: faster | streamk_wip vs AITER: within 1% | streamk_wip vs AITER: slower | streamk_wip vs AITER: median | streamk_wip vs hipBLASLt: faster | streamk_wip vs hipBLASLt: within 1% | streamk_wip vs hipBLASLt: slower | streamk_wip vs hipBLASLt: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Under 256 (less than one wave) | 321 | -34.0% | 62 | 15 | 244 | -15.3% | 77 | 10 | 234 | -15.8% |
| 256 to 1023 | 152 | +3.0% | 122 | 9 | 21 | +5.7% | 119 | 11 | 22 | +5.8% |
| 1024 or more | 157 | +2.6% | 143 | 13 | 1 | +3.7% | 137 | 18 | 2 | +4.3% |

**By v9 grid size (number of 256x256 output tiles), cold**

| Output tiles | Shapes | v9 vs AITER: median | streamk_wip vs AITER: faster | streamk_wip vs AITER: within 1% | streamk_wip vs AITER: slower | streamk_wip vs AITER: median | streamk_wip vs hipBLASLt: faster | streamk_wip vs hipBLASLt: within 1% | streamk_wip vs hipBLASLt: slower | streamk_wip vs hipBLASLt: median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Under 256 (less than one wave) | 321 | -34.1% | 76 | 13 | 232 | -11.2% | 80 | 4 | 237 | -14.7% |
| 256 to 1023 | 152 | +2.8% | 130 | 8 | 14 | +6.3% | 119 | 12 | 21 | +5.5% |
| 1024 or more | 157 | +2.5% | 150 | 5 | 2 | +4.1% | 147 | 8 | 2 | +4.6% |

**By the configuration `capped` chose, warm**

| Configuration | Shapes | vs AITER: faster | vs AITER: within 1% | vs AITER: slower | vs AITER: median | vs v9: median | split vs data-parallel: median |
|---|---:|---:|---:|---:|---:|---:|---:|
| Data-parallel | 324 | 209 | 27 | 88 | +2.5% | -0.3% | - |
| One-tile reduce-scatter, K >= 2048 | 77 | 12 | 1 | 64 | -27.7% | +121.2% | +118.8% |
| One-tile owner, K >= 2048 | 219 | 104 | 9 | 106 | -0.4% | +19.1% | +19.0% |
| One-tile reduce-scatter, K < 2048 | 6 | 0 | 0 | 6 | -65.8% | +3.6% | +5.3% |
| One-tile owner, K < 2048 | 2 | 0 | 0 | 2 | -49.8% | -1.0% | +0.8% |
| Spread | 2 | 2 | 0 | 0 | +5.4% | +2.8% | +2.1% |

**hipBLASLt's tile choice, warm (the six most common of 46 tiles)**

| hipBLASLt tile (MxNxK) | Shapes | hipBLASLt median TFLOPS | streamk_wip faster | within 1% | streamk_wip slower | streamk_wip vs hipBLASLt: median |
|---|---:|---:|---:|---:|---:|---:|
| 256x256x64 | 225 | 1452 | 181 | 29 | 15 | +3.5% |
| 128x128x256 | 40 | 754 | 8 | 0 | 32 | -18.0% |
| 192x256x64 | 34 | 1279 | 13 | 3 | 18 | -2.1% |
| 224x256x64 | 32 | 1320 | 32 | 0 | 0 | +8.2% |
| 256x128x64 | 27 | 733 | 1 | 0 | 26 | -21.1% |
| 160x256x64 | 27 | 1252 | 13 | 2 | 12 | -0.8% |

**Cold against warm (median change in TFLOPS)**

| Shapes | Count | AITER's pick | hipBLASLt | v9 | streamk_wip |
|---|---:|---:|---:|---:|---:|
| All shapes | 630 | -5.6% | -4.7% | -6.5% | -5.3% |
| Under 256 output tiles | 321 | -11.2% | -9.8% | -10.3% | -8.8% |
| 256 to 1023 output tiles | 152 | -4.4% | -3.5% | -3.7% | -3.3% |
| 1024 or more output tiles | 157 | -0.9% | -1.3% | -0.5% | -0.4% |
| K < 2048 | 91 | -8.8% | -7.8% | -8.4% | -7.5% |

**Selected shapes (TFLOPS)**

| Shape (MxNxK) | Output tiles | streamk_wip config | hipBLASLt tile | Warm: AITER | Warm: hipBLASLt | Warm: v9 | Warm: streamk_wip | Cold: AITER | Cold: hipBLASLt | Cold: streamk_wip |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 512x2304x16384 | 18 | one-tile reduce-scatter, n = 7 | 256x128x64 | 734 | 733 | 162 | 635 | 705 | 706 | 545 |
| 512x5120x25600 | 40 | one-tile owner, n = 6-7 | 160x128x128 | 439 | 910 | 328 | 994 | 430 | 876 | 968 |
| 512x7168x35840 | 56 | one-tile owner, n = 4-5 | 256x176x64 | 820 | 967 | 440 | 1172 | 820 | 966 | 1162 |
| 8192x2304x16384 | 288 | one-tile reduce-scatter, n = 4 | 144x256x64 | 1203 | 1206 | 1036 | 1399 | 1182 | 1186 | 1382 |
| 4096x5120x25600 | 320 | one-tile owner, n = 4 | 160x256x64 | 1280 | 1282 | 1162 | 1548 | 1275 | 1282 | 1544 |
| 4096x4608x8192 | 288 | one-tile reduce-scatter, n = 4 | 144x256x64 | 1271 | 1274 | 1088 | 1397 | 1180 | 1186 | 1362 |
| 1024x3072x512 | 48 | data-parallel | 96x128x128 | 459 | 404 | 226 | 212 | 366 | 324 | 192 |
| 2048x2048x512 | 64 | data-parallel | 128x128x128 | 528 | 471 | 285 | 273 | 431 | 394 | 254 |
| 4096x5120x640 | 320 | data-parallel | 256x176x64 | 811 | 778 | 753 | 779 | 616 | 654 | 656 |
| 8192x2048x512 | 256 | data-parallel | 256x256x64 | 805 | 807 | 867 | 788 | 759 | 769 | 747 |

**What this shows:**
- **`capped` removes most of the short-K loss.** On the 91 shapes with K < 2048, `streamk_wip` against v9 goes from -7.6% (the `auto` rule, Oct 2) to -1.9%, and against AITER from -10.9% to -5.0%. `capped` runs data-parallel on 324 shapes, against 274 for `auto`. Elsewhere the two rules are level (median -0.1%); the reduce-scatter shapes are 25% faster, from the capped split count and the faster slice reads.
- **Ahead of both AITER and hipBLASLt overall.** Warm, `streamk_wip` is faster than AITER's pick on 327 shapes and than hipBLASLt on 333 (median +1.6% for both). Cold, that rises to 356 and 346. With 1024 or more output tiles it is faster than AITER's pick on 143 of 157 shapes warm and 150 cold.
- **AITER's pick is level with plain hipBLASLt** (median -0.2%). On the 229 shapes where AITER picks `torch` the two are the same call, and they agree to a median +0.3%.
- **Against hipBLASLt the tile decides.** hipBLASLt picks 256x256x64 on 225 shapes, and `streamk_wip` is faster on 181 of them (median +3.5%). It is also ahead where hipBLASLt picks the larger non-power-of-2 tiles (224x256, 256x224, 256x240: +8 to +21%). It loses where hipBLASLt picks small tiles for shapes under one wave: 128x128x256 (-18%), 256x128x64 (-21%) and smaller.
- **Cold caches cost every kernel about 5-6% (median), and about 10% under one wave.** `streamk_wip` loses a little less than the others, so its lead grows. Cold, its data-parallel shapes are also level with v9 (median +0.4%, against -0.3% warm).
- **The remaining losses are small shapes.** Under one wave AITER and hipBLASLt still lead (`streamk_wip` slower than AITER on 244 of 321 warm, median -15%), because they use smaller tiles. The 77 one-tile reduce-scatter shapes (65 of them under one wave, K = 2048-16384) are 2.2x faster than v9 but still 28% behind AITER.
- **Noise:** between Oct 2 and today the same data-parallel kernel moved by more than 1% on 44 of 272 shapes (median -0.3%), so gaps within about 1% are ties.

## Next

1. **More parallelism for small and short-K shapes.** These are now the main gap to AITER and hipBLASLt. Options:
   - smaller output tiles, giving more tiles per CU;
   - fewer, larger K chunks per split, so the fixup cost is shared over more work;
   - splitting along M or N within a tile instead of along K;
   - a cheaper ending for shallow splits.
2. **ATT traces.** Capture `rocprofv3 --att` traces of `streamk_wip` and v9 on representative losers: 8192x2048x512 (data-parallel, warm), 512x2304x16384 (one-tile reduce-scatter, under one wave) and 1024x3072x512 (data-parallel, under one wave). Look at MFMA utilisation, `s_waitcnt` and LDS stalls, time spent waiting on the fixup flags and barrier, and the store tail.
3. **Smaller tiles, alone and with stream-K.** Try 256x128, 128x256 and 128x128 tiles in `streamk_wip` (data-parallel only, then with stream-K) on the 321 shapes under one wave. hipBLASLt's tile table gives a first guide to which tile wins where.
