# The stream-K host rules on AITER's shapes (held-out test set)

rocprof kernel times, contiguous B and C, 1000 launches over rotating inputs, median. Loss = kernel time / the fastest single-rounding choice measured for that shape: data-parallel, one-tile reduce-scatter or owner (fp32) at any n, spread (S > 128), or v24 as deployed. The first run's cases are v24's (data-parallel: v9 at one wave or less, v13 above); the re-score's new ones are v25's, which times as v24 on aligned shapes (experiments/streamk_split_count/results/spot_v25.md). `capped`: v24 as deployed. `v1`: the rule frozen 2026-10-09 10:40 from sweep F only (sha256 of sweep_f.jsonl 8e29bf6acfb4). Version 2 frozen 2026-10-10 18:39, in two variants that differ only below one wave: `v2` (version 1's rule family) and `v2-sqrt` (the same with the reduce-scatter n also at most Q / sqrt(S)). Both use sweep F's constants for one wave or more, sweep G's (sha256 of sweep_g.jsonl a8605436cbdc) for less than one wave, and spread for S > 128 from k >= k_s per regime (one wave or more 512, less than one wave 512); its data-parallel kernel is v9_ragged at one wave or less (the v9 rows) and v13 above. Version 3 (`v3`) frozen 2026-10-12 10:37: `v2` with spread for S > 128 when (256 - S) k >= C per regime (one wave or more 10493, less than one wave 10493), fitted on sweep H (sha256 of sweep_h.jsonl f23a51fae2f2), its spread on v26 (fewer spills in the end-to-end loop; the `v26 spread` rows). `v26 deployed`: v26 as deployed (no STREAMK_* set) with version 3 applied as its host rule below one wave and for S > 128, timed on its own (the `v26 capped` rows). Nothing measured here was used to fit any of them.

| shapes | count | data-parallel: median / 90th pct / worst | capped | v1 | v2 | v2-sqrt | v3 | v26 deployed | v27 @256x128 | v27 @128x256 | v27 @128x128x128 | v1: more than 2% off | v2: more than 2% off | v2-sqrt: more than 2% off | v3: more than 2% off | v26 deployed: more than 2% off | v27 @256x128: more than 2% off | v27 @128x256: more than 2% off | v27 @128x128x128: more than 2% off |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| less than one wave | 314 | 1.464 / 3.124 / 4.997 | 1.043 / 1.165 / 1.277 | 1.053 / 1.177 / 1.435 | 1.018 / 1.104 / 1.221 | 1.022 / 1.126 / 1.221 | 1.012 / 1.098 / 1.221 | 1.012 / 1.105 / 1.231 | 0.987 / 1.187 / 1.330 | 0.986 / 1.222 / 1.413 | 1.008 / 1.297 / 1.491 | 203 | 150 | 163 | 135 | 137 | 133 | 139 | 156 |
| of which S <= 128 | 254 | 1.624 / 3.320 / 4.997 | 1.054 / 1.166 / 1.277 | 1.048 / 1.164 / 1.435 | 1.021 / 1.104 / 1.221 | 1.027 / 1.126 / 1.221 | 1.021 / 1.104 / 1.221 | 1.022 / 1.112 / 1.231 | 0.957 / 1.128 / 1.319 | 0.951 / 1.134 / 1.413 | 0.910 / 1.243 / 1.458 | 161 | 131 | 144 | 131 | 134 | 75 | 81 | 96 |
| of which S > 128 | 60 | 1.005 / 1.140 / 1.209 | 1.005 / 1.131 / 1.212 | 1.075 / 1.216 / 1.342 | 1.005 / 1.140 / 1.209 | 1.005 / 1.140 / 1.209 | 1.002 / 1.013 / 1.054 | 1.002 / 1.014 / 1.061 | 1.162 / 1.299 / 1.330 | 1.179 / 1.321 / 1.399 | 1.263 / 1.372 / 1.491 | 42 | 19 | 19 | 4 | 3 | 58 | 58 | 60 |
| one full wave or more | 126 | 1.046 / 1.214 / 1.417 | 1.003 / 1.016 / 1.058 | 1.003 / 1.026 / 1.091 | 1.002 / 1.022 / 1.076 | 1.002 / 1.022 / 1.076 | 1.002 / 1.017 / 1.076 | 1.003 / 1.015 / 1.039 | 1.158 / 1.224 / 1.342 | 1.218 / 1.292 / 1.328 | 1.341 / 1.429 / 1.542 | 16 | 13 | 13 | 10 | 10 | 122 | 123 | 125 |
| of which S <= 128 | 95 | 1.063 / 1.221 / 1.417 | 1.003 / 1.016 / 1.040 | 1.003 / 1.022 / 1.076 | 1.003 / 1.022 / 1.076 | 1.003 / 1.022 / 1.076 | 1.003 / 1.022 / 1.076 | 1.003 / 1.021 / 1.039 | 1.154 / 1.224 / 1.342 | 1.215 / 1.295 / 1.328 | 1.338 / 1.426 / 1.542 | 10 | 10 | 10 | 10 | 10 | 91 | 92 | 94 |
| of which S > 128 | 31 | 1.000 / 1.051 / 1.061 | 1.003 / 1.016 / 1.058 | 1.005 / 1.057 / 1.091 | 1.000 / 1.013 / 1.061 | 1.000 / 1.013 / 1.061 | 1.000 / 1.005 / 1.013 | 1.002 / 1.009 / 1.019 | 1.170 / 1.224 / 1.273 | 1.218 / 1.275 / 1.311 | 1.361 / 1.456 / 1.535 | 6 | 3 | 3 | 0 | 0 | 31 | 31 | 31 |
| all | 440 | 1.217 / 2.713 / 4.997 | 1.015 / 1.141 / 1.277 | 1.020 / 1.160 / 1.435 | 1.007 / 1.088 / 1.221 | 1.011 / 1.104 / 1.221 | 1.006 / 1.080 / 1.221 | 1.007 / 1.082 / 1.231 | 1.076 / 1.208 / 1.342 | 1.073 / 1.269 / 1.413 | 1.166 / 1.403 / 1.542 | 219 | 163 | 176 | 145 | 147 | 255 | 262 | 281 |

v2 against capped, less than one wave, shape by shape: more than 2% faster on 111 shapes, more than 2% slower on 36, of 314.

v2 against capped, one full wave or more, shape by shape: more than 2% faster on 3 shapes, more than 2% slower on 4, of 126.

v2-sqrt against capped, less than one wave, shape by shape: more than 2% faster on 104 shapes, more than 2% slower on 37, of 314.

v2-sqrt against capped, one full wave or more, shape by shape: more than 2% faster on 3 shapes, more than 2% slower on 4, of 126.

v3 against capped, less than one wave, shape by shape: more than 2% faster on 126 shapes, more than 2% slower on 35, of 314.

v3 against capped, one full wave or more, shape by shape: more than 2% faster on 6 shapes, more than 2% slower on 4, of 126.

## Every shape

| M | N | K | tiles | S | fastest (n) | TFLOPS | data-parallel | capped | v1 (choice) | v2 (choice) | v2-sqrt (choice) | v3 (choice) | AITER's tuned pick, TFLOPS (other tiles; do_bench timing) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 512 | 1024 | 4096 | 8 | 8 | v24 rs (16) | 188 | 3.124 | 1.054 | 1.000 (v24 rs 16) | 1.000 (v24 rs 16) | 1.030 (v25 rs 11) | 1.000 (v24 rs 16) | 284 (flydsl) |
| 512 | 1024 | 5120 | 8 | 8 | v25 rs (20) | 203 | 3.825 | 1.015 | 1.057 (v24 rs 16) | 1.000 (v25 rs 20) | 1.073 (v25 rs 11) | 1.000 (v25 rs 20) | 249 (asm) |
| 512 | 1024 | 7168 | 8 | 8 | v24 capped (-) | 265 | 4.663 | 1.000 | 1.077 (v24 rs 16) | 1.165 (v25 rs 24) | 1.123 (v25 rs 11) | 1.165 (v25 rs 24) | 387 (torch) |
| 512 | 1280 | 6144 | 10 | 10 | v24 rs (16) | 282 | 4.180 | 1.022 | 1.065 (v24 rs 12) | 1.164 (v25 rs 19) | 1.178 (v25 rs 10) | 1.164 (v25 rs 19) | 340 (flydsl) |
| 512 | 1280 | 8192 | 10 | 10 | v24 rs (16) | 340 | 4.356 | 1.074 | 1.075 (v24 rs 12) | 1.079 (v25 rs 19) | 1.099 (v25 rs 10) | 1.079 (v25 rs 19) | 422 (torch) |
| 512 | 1536 | 1536 | 12 | 12 | v24 rs (12) | 103 | 1.463 | 1.025 | 1.003 (v24 rs 6) | 1.003 (v24 rs 6) | 1.003 (v24 rs 6) | 1.003 (v24 rs 6) | 264 (flydsl) |
| 512 | 1536 | 6144 | 12 | 12 | v24 rs (12) | 313 | 3.856 | 1.131 | 1.137 (v24 rs 10) | 1.038 (v24 rs 16) | 1.147 (v25 rs 9) | 1.038 (v24 rs 16) | 384 (flydsl) |
| 512 | 1536 | 7168 | 12 | 12 | v24 rs (8) | 327 | 3.834 | 1.046 | 1.038 (v24 rs 10) | 1.005 (v24 rs 16) | 1.082 (v25 rs 9) | 1.005 (v24 rs 16) | 412 (torch) |
| 512 | 2048 | 1536 | 16 | 16 | v26 capped (-) | 138 | 1.483 | 1.062 | 1.009 (v24 rs 6) | 1.009 (v24 rs 6) | 1.009 (v24 rs 6) | 1.009 (v24 rs 6) | 311 (torch) |
| 512 | 2048 | 4096 | 16 | 16 | v24 rs (8) | 317 | 2.631 | 1.006 | 1.000 (v24 rs 8) | 1.118 (v24 rs 12) | 1.000 (v24 rs 8) | 1.118 (v24 rs 12) | 439 (flydsl) |
| 512 | 2048 | 6144 | 16 | 16 | v24 capped (-) | 373 | 3.459 | 1.000 | 1.003 (v24 rs 8) | 1.012 (v24 rs 12) | 1.003 (v24 rs 8) | 1.012 (v24 rs 12) | 342 (asm) |
| 512 | 2048 | 7168 | 16 | 16 | v24 rs (8) | 415 | 3.645 | 1.004 | 1.000 (v24 rs 8) | 1.036 (v24 rs 12) | 1.000 (v24 rs 8) | 1.036 (v24 rs 12) | 546 (torch) |
| 512 | 2304 | 1536 | 18 | 18 | v24 rs (4) | 149 | 1.420 | 1.095 | 1.002 (v24 rs 6) | 1.002 (v24 rs 6) | 1.002 (v24 rs 6) | 1.002 (v24 rs 6) | 318 (flydsl) |
| 512 | 2304 | 6144 | 18 | 18 | v24 rs (8) | 404 | 3.320 | 1.148 | 1.143 (v24 rs 7) | 1.081 (v25 rs 10) | 1.000 (v24 rs 8) | 1.081 (v25 rs 10) | 441 (flydsl) |
| 512 | 2304 | 16384 | 18 | 18 | v24 rs (12) | 691 | 4.939 | 1.199 | 1.216 (v24 rs 7) | 1.007 (v25 rs 10) | 1.059 (v24 rs 8) | 1.007 (v25 rs 10) | 705 (torch) |
| 512 | 2560 | 6144 | 20 | 20 | v24 rs (8) | 432 | 3.193 | 1.108 | 1.099 (v24 rs 6) | 1.073 (v25 rs 9) | 1.130 (v25 rs 7) | 1.073 (v25 rs 9) | 477 (torch) |
| 512 | 2560 | 8192 | 20 | 20 | v24 rs (8) | 547 | 3.497 | 1.125 | 1.130 (v24 rs 6) | 1.031 (v25 rs 9) | 1.082 (v25 rs 7) | 1.031 (v25 rs 9) | 559 (torch) |
| 512 | 3072 | 6144 | 24 | 24 | v24 rs (8) | 486 | 2.988 | 1.138 | 1.144 (v24 rs 5) | 1.000 (v24 rs 8) | 1.103 (v25 rs 7) | 1.000 (v24 rs 8) | 542 (flydsl) |
| 512 | 3584 | 7168 | 28 | 28 | v24 rs (8) | 595 | 2.950 | 1.166 | 1.167 (v24 rs 4) | 1.080 (v24 rs 6) | 1.080 (v24 rs 6) | 1.080 (v24 rs 6) | 556 (flydsl) |
| 512 | 4096 | 1024 | 32 | 32 | v24 rs (4) | 186 | 1.026 | 1.075 | 1.000 (v24 rs 4) | 1.000 (v24 rs 4) | 1.000 (v24 rs 4) | 1.000 (v24 rs 4) | 388 (torch) |
| 512 | 4096 | 1536 | 32 | 32 | v24 capped (-) | 246 | 1.314 | 1.000 | 1.009 (v24 rs 4) | 1.110 (v24 rs 6) | 1.110 (v24 rs 6) | 1.110 (v24 rs 6) | 443 (torch) |
| 512 | 4096 | 2048 | 32 | 32 | v24 capped (-) | 314 | 1.457 | 1.000 | 1.000 (v24 rs 4) | 1.051 (v24 rs 6) | 1.051 (v24 rs 6) | 1.051 (v24 rs 6) | 490 (torch) |
| 512 | 4096 | 4096 | 32 | 32 | v24 rs (6) | 477 | 1.979 | 1.010 | 1.007 (v24 rs 4) | 1.000 (v24 rs 6) | 1.000 (v24 rs 6) | 1.000 (v24 rs 6) | 601 (flydsl) |
| 512 | 4096 | 8192 | 32 | 32 | v24 rs (6) | 706 | 2.817 | 1.121 | 1.151 (v24 rs 4) | 1.000 (v24 rs 6) | 1.000 (v24 rs 6) | 1.000 (v24 rs 6) | 675 (flydsl) |
| 512 | 4608 | 8192 | 36 | 36 | v24 rs (6) | 756 | 2.689 | 1.228 | 1.286 (v24 rs 3) | 1.047 (v25 rs 5) | 1.047 (v25 rs 5) | 1.047 (v25 rs 5) | 721 (torch) |
| 512 | 4608 | 16384 | 36 | 36 | v24 rs (default) | 937 | 3.333 | 1.082 | 1.435 (v24 rs 3) | 1.063 (v25 rs 5) | 1.063 (v25 rs 5) | 1.063 (v25 rs 5) | 504 (flydsl) |
| 512 | 5120 | 1280 | 40 | 40 | v26 capped (-) | 220 | 1.081 | 1.081 | 1.092 (v24 rs 3) | 1.007 (v24 rs 4) | 1.025 (v25 rs 5) | 1.007 (v24 rs 4) | 450 (flydsl) |
| 512 | 5120 | 3200 | 40 | 40 | v25 rs (5) | 467 | 1.722 | 1.277 | 1.106 (v24 rs 3) | 1.010 (v24 rs 4) | 1.000 (v25 rs 5) | 1.010 (v24 rs 4) | 575 (flydsl) |
| 512 | 5120 | 5120 | 40 | 40 | v24 rs (6) | 607 | 2.250 | 1.233 | 1.201 (v24 rs 3) | 1.055 (v24 rs 4) | 1.033 (v25 rs 5) | 1.055 (v24 rs 4) | 651 (torch) |
| 512 | 5120 | 6400 | 40 | 40 | v24 rs (6) | 675 | 2.395 | 1.182 | 1.223 (v24 rs 3) | 1.073 (v24 rs 4) | 1.012 (v25 rs 5) | 1.073 (v24 rs 4) | 542 (flydsl) |
| 512 | 5120 | 25600 | 40 | 40 | v24 rs (6) | 1105 | 3.330 | 1.051 | 1.361 (v24 rs 3) | 1.138 (v24 rs 4) | 1.052 (v25 rs 5) | 1.138 (v24 rs 4) | 430 (flydsl) |
| 512 | 6144 | 1024 | 48 | 48 | dp (-) | 271 | 1.000 | 1.047 | 1.165 (v24 owner 2) | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 502 (flydsl) |
| 512 | 6144 | 1536 | 48 | 48 | v26 capped (-) | 310 | 1.112 | 1.112 | 1.065 (v24 owner 3) | 1.021 (v24 rs 4) | 1.126 (v24 rs default) | 1.021 (v24 rs 4) | 542 (flydsl) |
| 512 | 6144 | 2048 | 48 | 48 | v24 rs (4) | 399 | 1.230 | 1.118 | 1.045 (v24 owner 3) | 1.000 (v24 rs 4) | 1.177 (v24 rs default) | 1.000 (v24 rs 4) | 597 (flydsl) |
| 512 | 6144 | 3072 | 48 | 48 | v26 capped (-) | 491 | 1.551 | 1.223 | 1.083 (v24 owner 4) | 1.004 (v24 rs 4) | 1.036 (v24 rs default) | 1.004 (v24 rs 4) | 639 (flydsl) |
| 512 | 6144 | 4608 | 48 | 48 | v26 capped (-) | 641 | 1.900 | 1.157 | 1.161 (v24 owner default) | 1.007 (v24 rs 4) | 1.056 (v24 rs default) | 1.007 (v24 rs 4) | 686 (flydsl) |
| 512 | 6144 | 6144 | 48 | 48 | v24 rs (4) | 700 | 2.136 | 1.135 | 1.092 (v24 owner default) | 1.000 (v24 rs 4) | 1.112 (v24 rs default) | 1.000 (v24 rs 4) | 687 (flydsl) |
| 512 | 6144 | 7168 | 48 | 48 | v26 capped (-) | 772 | 2.212 | 1.179 | 1.069 (v24 owner default) | 1.004 (v24 rs 4) | 1.062 (v24 rs default) | 1.004 (v24 rs 4) | 715 (flydsl) |
| 512 | 6144 | 12288 | 48 | 48 | v26 capped (-) | 906 | 2.588 | 1.046 | 1.004 (v24 owner default) | 1.006 (v24 rs 4) | 1.016 (v24 rs default) | 1.006 (v24 rs 4) | 781 (torch) |
| 512 | 6144 | 18432 | 48 | 48 | v24 rs (default) | 1049 | 2.970 | 1.005 | 1.015 (v24 owner default) | 1.062 (v24 rs 4) | 1.000 (v24 rs default) | 1.062 (v24 rs 4) | 787 (flydsl) |
| 512 | 6400 | 5120 | 50 | 50 | v24 rs (4) | 673 | 1.992 | 1.118 | 1.103 (v24 owner default) | 1.100 (v24 rs 3) | 1.032 (v24 rs default) | 1.100 (v24 rs 3) | 614 (flydsl) |
| 512 | 7168 | 1024 | 56 | 56 | v26 capped (-) | 314 | 1.002 | 1.037 | 1.172 (v24 owner 2) | 1.002 (dp -) | 1.002 (dp -) | 1.002 (dp -) | 536 (flydsl) |
| 512 | 7168 | 1536 | 56 | 56 | v24 rs (2) | 340 | 1.058 | 1.059 | 1.014 (v24 owner 3) | 1.017 (v24 rs 3) | 1.014 (v24 owner 3) | 1.017 (v24 rs 3) | 589 (flydsl) |
| 512 | 7168 | 1792 | 56 | 56 | v24 rs (2) | 383 | 1.104 | 1.045 | 1.078 (v24 owner 3) | 1.035 (v24 rs 3) | 1.078 (v24 owner 3) | 1.035 (v24 rs 3) | 621 (flydsl) |
| 512 | 7168 | 3584 | 56 | 56 | v24 rs (4) | 601 | 1.590 | 1.165 | 1.190 (v24 owner default) | 1.046 (v24 rs 3) | 1.190 (v24 owner default) | 1.046 (v24 rs 3) | 717 (flydsl) |
| 512 | 7168 | 4224 | 56 | 56 | v24 rs (4) | 649 | 1.703 | 1.187 | 1.126 (v24 owner default) | 1.050 (v24 rs 3) | 1.126 (v24 owner default) | 1.050 (v24 rs 3) | 726 (flydsl) |
| 512 | 7168 | 8192 | 56 | 56 | v24 rs (4) | 902 | 2.107 | 1.089 | 1.139 (v24 owner default) | 1.085 (v24 rs 3) | 1.139 (v24 owner default) | 1.085 (v24 rs 3) | 795 (flydsl) |
| 512 | 7168 | 35840 | 56 | 56 | v24 rs (4) | 1231 | 2.803 | 1.018 | 1.011 (v24 owner default) | 1.145 (v24 rs 3) | 1.011 (v24 owner default) | 1.145 (v24 rs 3) | 820 (flydsl) |
| 512 | 8192 | 1024 | 64 | 64 | v26 capped (-) | 351 | 1.003 | 1.044 | 1.155 (v24 owner 2) | 1.003 (dp -) | 1.003 (dp -) | 1.003 (dp -) | 581 (flydsl) |
| 512 | 8192 | 2048 | 64 | 64 | v24 owner (3) | 480 | 1.151 | 1.044 | 1.000 (v24 owner 3) | 1.031 (v24 rs 3) | 1.000 (v24 owner 3) | 1.031 (v24 rs 3) | 708 (flydsl) |
| 512 | 8192 | 3584 | 64 | 64 | v24 owner (3) | 647 | 1.508 | 1.108 | 1.051 (v24 owner default) | 1.024 (v24 rs 3) | 1.051 (v24 owner default) | 1.024 (v24 rs 3) | 676 (flydsl) |
| 512 | 8192 | 4096 | 64 | 64 | v26 capped (-) | 708 | 1.503 | 1.121 | 1.047 (v24 owner default) | 1.002 (v24 rs 3) | 1.047 (v24 owner default) | 1.002 (v24 rs 3) | 808 (flydsl) |
| 512 | 8192 | 7168 | 64 | 64 | v24 rs (4) | 896 | 1.935 | 1.079 | 1.015 (v24 owner default) | 1.073 (v24 rs 3) | 1.015 (v24 owner default) | 1.073 (v24 rs 3) | 709 (flydsl) |
| 512 | 8192 | 8192 | 64 | 64 | v24 rs (4) | 951 | 1.990 | 1.043 | 1.020 (v24 owner default) | 1.059 (v24 rs 3) | 1.020 (v24 owner default) | 1.059 (v24 rs 3) | 680 (flydsl) |
| 512 | 8192 | 28672 | 64 | 64 | v24 rs (default) | 1267 | 2.618 | 1.001 | 1.000 (v24 owner default) | 1.139 (v24 rs 3) | 1.000 (v24 owner default) | 1.139 (v24 rs 3) | 788 (flydsl) |
| 512 | 8448 | 7168 | 66 | 66 | v24 rs (3) | 848 | 1.770 | 1.065 | 1.069 (v24 owner default) | 1.069 (v24 owner default) | 1.069 (v24 owner default) | 1.069 (v24 owner default) | 672 (flydsl) |
| 512 | 8704 | 4096 | 68 | 68 | v24 rs (3) | 728 | 1.453 | 1.113 | 1.149 (v24 owner default) | 1.149 (v24 owner default) | 1.149 (v24 owner default) | 1.149 (v24 owner default) | 752 (torch) |
| 512 | 9216 | 6144 | 72 | 72 | v24 owner (3) | 851 | 1.732 | 1.066 | 1.090 (v24 owner default) | 1.090 (v24 owner default) | 1.090 (v24 owner default) | 1.090 (v24 owner default) | 610 (flydsl) |
| 512 | 9216 | 16384 | 72 | 72 | v24 owner (3) | 1127 | 1.887 | 1.001 | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 638 (flydsl) |
| 512 | 10240 | 4096 | 80 | 80 | v24 rs (3) | 787 | 1.348 | 1.032 | 1.019 (v24 owner default) | 1.019 (v24 owner default) | 1.019 (v24 owner default) | 1.019 (v24 owner default) | 860 (torch) |
| 512 | 10240 | 8192 | 80 | 80 | v24 owner (3) | 1053 | 1.631 | 1.007 | 1.011 (v24 owner default) | 1.011 (v24 owner default) | 1.011 (v24 owner default) | 1.011 (v24 owner default) | 625 (flydsl) |
| 512 | 12800 | 5120 | 100 | 100 | v24 owner (2) | 908 | 1.381 | 1.086 | 1.065 (v24 owner default) | 1.065 (v24 owner default) | 1.065 (v24 owner default) | 1.065 (v24 owner default) | 662 (flydsl) |
| 512 | 13312 | 16384 | 104 | 104 | v24 owner (2) | 1196 | 1.748 | 1.019 | 1.022 (v24 owner default) | 1.022 (v24 owner default) | 1.022 (v24 owner default) | 1.022 (v24 owner default) | 546 (flydsl) |
| 512 | 14336 | 8192 | 112 | 112 | v24 capped (-) | 1142 | 1.309 | 1.000 | 1.017 (v24 owner default) | 1.017 (v24 owner default) | 1.017 (v24 owner default) | 1.017 (v24 owner default) | 1036 (torch) |
| 512 | 16384 | 2048 | 128 | 128 | v24 owner (default) | 786 | 1.051 | 1.048 | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 720 (flydsl) |
| 512 | 16384 | 4096 | 128 | 128 | v24 owner (default) | 1005 | 1.195 | 1.017 | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 634 (flydsl) |
| 512 | 16384 | 6656 | 128 | 128 | v24 owner (default) | 1168 | 1.321 | 1.007 | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 840 (flydsl) |
| 512 | 16384 | 8192 | 128 | 128 | v24 capped (-) | 1244 | 1.236 | 1.000 | 1.007 (v24 owner default) | 1.007 (v24 owner default) | 1.007 (v24 owner default) | 1.007 (v24 owner default) | 656 (flydsl) |
| 512 | 16384 | 13312 | 128 | 128 | v24 owner (2) | 1317 | 1.410 | 1.008 | 1.006 (v24 owner default) | 1.006 (v24 owner default) | 1.006 (v24 owner default) | 1.006 (v24 owner default) | 704 (flydsl) |
| 512 | 16384 | 26624 | 128 | 128 | v24 owner (2) | 1412 | 1.496 | 1.004 | 1.006 (v24 owner default) | 1.006 (v24 owner default) | 1.006 (v24 owner default) | 1.006 (v24 owner default) | 654 (flydsl) |
| 512 | 20480 | 7168 | 160 | 160 | v24 capped (-) | 1133 | 1.001 | 1.000 | 1.134 (v24 owner default) | 1.001 (dp -) | 1.001 (dp -) | 1.004 (v26 spread -) | 1033 (flydsl) |
| 512 | 26624 | 16384 | 208 | 208 | v26 spread (-) | 1309 | 1.012 | 1.032 | 1.025 (v24 owner default) | 1.012 (dp -) | 1.012 (dp -) | 1.000 (v26 spread -) | 769 (flydsl) |
| 512 | 37888 | 5120 | 296 | 40 | v24 rs (4) | 1220 | 1.238 | 1.029 | 1.026 (v24 rs 3) | 1.026 (v24 rs 3) | 1.026 (v24 rs 3) | 1.026 (v24 rs 3) | 1057 (torch) |
| 512 | 51200 | 5120 | 400 | 144 | dp (-) | 1230 | 1.000 | 1.003 | 1.046 (v24 owner default) | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1140 (flydsl) |
| 512 | 53248 | 16384 | 416 | 160 | v26 spread (-) | 1379 | 1.061 | 1.045 | 1.079 (v24 owner default) | 1.061 (dp -) | 1.061 (dp -) | 1.000 (v26 spread -) | 647 (flydsl) |
| 512 | 57344 | 8192 | 448 | 192 | v24 rs (1) | 1358 | 1.002 | 1.000 | 1.017 (v24 owner default) | 1.002 (dp -) | 1.002 (dp -) | 1.002 (dp -) | 1295 (flydsl) |
| 512 | 129280 | 4096 | 1010 | 242 | v24 capped (-) | 1393 | 1.000 | 1.000 | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1306 (asm) |
| 768 | 1024 | 4096 | 12 | 12 | v24 rs (8) | 256 | 2.820 | 1.010 | 1.072 (v24 rs 10) | 1.099 (v24 rs 16) | 1.068 (v25 rs 9) | 1.099 (v24 rs 16) | 371 (torch) |
| 768 | 1024 | 5120 | 12 | 12 | v24 capped (-) | 284 | 3.506 | 1.000 | 1.008 (v24 rs 10) | 1.140 (v24 rs 16) | 1.137 (v25 rs 9) | 1.140 (v24 rs 16) | 313 (asm) |
| 768 | 1024 | 7168 | 12 | 12 | v24 rs (8) | 331 | 3.843 | 1.038 | 1.039 (v24 rs 10) | 1.011 (v24 rs 16) | 1.096 (v25 rs 9) | 1.011 (v24 rs 16) | 409 (torch) |
| 768 | 2048 | 4096 | 24 | 24 | v26 capped (-) | 404 | 2.228 | 1.061 | 1.077 (v24 rs 5) | 1.034 (v24 rs 8) | 1.045 (v25 rs 7) | 1.034 (v24 rs 8) | 527 (flydsl) |
| 768 | 2048 | 7168 | 24 | 24 | v26 capped (-) | 549 | 3.177 | 1.156 | 1.153 (v24 rs 5) | 1.005 (v24 rs 8) | 1.024 (v25 rs 7) | 1.005 (v24 rs 8) | 560 (flydsl) |
| 768 | 4096 | 4096 | 48 | 48 | v24 rs (4) | 620 | 1.712 | 1.135 | 1.197 (v24 owner default) | 1.000 (v24 rs 4) | 1.096 (v24 rs default) | 1.000 (v24 rs 4) | 690 (flydsl) |
| 768 | 4096 | 8192 | 48 | 48 | v26 capped (-) | 847 | 2.240 | 1.064 | 1.101 (v24 owner default) | 1.006 (v24 rs 4) | 1.026 (v24 rs default) | 1.006 (v24 rs 4) | 744 (flydsl) |
| 768 | 8192 | 4096 | 96 | 96 | v24 owner (2) | 891 | 1.295 | 1.134 | 1.133 (v24 owner default) | 1.133 (v24 owner default) | 1.133 (v24 owner default) | 1.133 (v24 owner default) | 876 (torch) |
| 768 | 8704 | 4096 | 102 | 102 | v24 owner (2) | 933 | 1.272 | 1.097 | 1.102 (v24 owner default) | 1.102 (v24 owner default) | 1.102 (v24 owner default) | 1.102 (v24 owner default) | 870 (flydsl) |
| 768 | 10240 | 4096 | 120 | 120 | v24 owner (2) | 1026 | 1.217 | 1.024 | 1.011 (v24 owner default) | 1.011 (v24 owner default) | 1.011 (v24 owner default) | 1.011 (v24 owner default) | 973 (flydsl) |
| 768 | 37888 | 5120 | 444 | 188 | v26 capped (-) | 1345 | 1.001 | 1.002 | 1.018 (v24 owner default) | 1.001 (dp -) | 1.001 (dp -) | 1.001 (dp -) | 1226 (torch) |
| 768 | 129280 | 4096 | 1515 | 235 | dp (-) | 1458 | 1.000 | 1.004 | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1347 (torch) |
| 1024 | 512 | 4096 | 8 | 8 | v26 capped (-) | 190 | 3.161 | 1.060 | 1.005 (v24 rs 16) | 1.005 (v24 rs 16) | 1.048 (v25 rs 11) | 1.005 (v24 rs 16) | 283 (flydsl) |
| 1024 | 512 | 7168 | 8 | 8 | v24 capped (-) | 267 | 4.678 | 1.000 | 1.033 (v24 rs 16) | 1.161 (v25 rs 24) | 1.156 (v25 rs 11) | 1.161 (v25 rs 24) | 336 (flydsl) |
| 1024 | 512 | 8192 | 8 | 8 | v24 rs (16) | 314 | 4.997 | 1.001 | 1.000 (v24 rs 16) | 1.095 (v25 rs 24) | 1.095 (v25 rs 11) | 1.095 (v25 rs 24) | 334 (torch) |
| 1024 | 1024 | 4096 | 16 | 16 | v24 rs (8) | 320 | 2.628 | 1.006 | 1.000 (v24 rs 8) | 1.077 (v24 rs 12) | 1.000 (v24 rs 8) | 1.077 (v24 rs 12) | 443 (flydsl) |
| 1024 | 1024 | 5120 | 16 | 16 | v24 rs (8) | 337 | 3.116 | 1.010 | 1.000 (v24 rs 8) | 1.059 (v24 rs 12) | 1.000 (v24 rs 8) | 1.059 (v24 rs 12) | 334 (asm) |
| 1024 | 1024 | 7168 | 16 | 16 | v24 rs (8) | 424 | 3.678 | 1.003 | 1.000 (v24 rs 8) | 1.062 (v24 rs 12) | 1.000 (v24 rs 8) | 1.062 (v24 rs 12) | 562 (torch) |
| 1024 | 1280 | 8192 | 20 | 20 | v26 capped (-) | 542 | 3.419 | 1.098 | 1.099 (v24 rs 6) | 1.029 (v25 rs 9) | 1.054 (v25 rs 7) | 1.029 (v25 rs 9) | 583 (torch) |
| 1024 | 1536 | 1536 | 24 | 24 | v24 rs (4) | 196 | 1.391 | 1.119 | 1.107 (v24 rs 5) | 1.034 (v24 rs 6) | 1.034 (v24 rs 6) | 1.034 (v24 rs 6) | 401 (flydsl) |
| 1024 | 1536 | 7168 | 24 | 24 | v26 capped (-) | 547 | 3.171 | 1.141 | 1.145 (v24 rs 5) | 1.002 (v24 rs 8) | 1.025 (v25 rs 7) | 1.002 (v24 rs 8) | 564 (flydsl) |
| 1024 | 2048 | 1536 | 32 | 32 | v24 rs (4) | 245 | 1.318 | 1.002 | 1.000 (v24 rs 4) | 1.078 (v24 rs 6) | 1.078 (v24 rs 6) | 1.078 (v24 rs 6) | 455 (torch) |
| 1024 | 2048 | 4096 | 32 | 32 | v24 rs (6) | 476 | 1.966 | 1.007 | 1.014 (v24 rs 4) | 1.000 (v24 rs 6) | 1.000 (v24 rs 6) | 1.000 (v24 rs 6) | 604 (flydsl) |
| 1024 | 2048 | 6144 | 32 | 32 | v26 capped (-) | 578 | 2.630 | 1.096 | 1.094 (v24 rs 4) | 1.002 (v24 rs 6) | 1.002 (v24 rs 6) | 1.002 (v24 rs 6) | 411 (opus) |
| 1024 | 2048 | 7168 | 32 | 32 | v26 capped (-) | 627 | 2.713 | 1.093 | 1.091 (v24 rs 4) | 1.001 (v24 rs 6) | 1.001 (v24 rs 6) | 1.001 (v24 rs 6) | 735 (torch) |
| 1024 | 2304 | 1536 | 36 | 36 | v24 rs (4) | 261 | 1.251 | 1.211 | 1.107 (v24 rs 3) | 1.094 (v25 rs 5) | 1.094 (v25 rs 5) | 1.094 (v25 rs 5) | 458 (torch) |
| 1024 | 2304 | 6144 | 36 | 36 | v24 rs (6) | 612 | 2.495 | 1.188 | 1.282 (v24 rs 3) | 1.062 (v25 rs 5) | 1.062 (v25 rs 5) | 1.062 (v25 rs 5) | 604 (torch) |
| 1024 | 2304 | 16384 | 36 | 36 | v24 rs (6) | 952 | 3.404 | 1.103 | 1.420 (v24 rs 3) | 1.061 (v25 rs 5) | 1.061 (v25 rs 5) | 1.061 (v25 rs 5) | 873 (torch) |
| 1024 | 2560 | 6144 | 40 | 40 | v24 rs (6) | 653 | 2.395 | 1.229 | 1.246 (v24 rs 3) | 1.053 (v24 rs 4) | 1.036 (v25 rs 5) | 1.053 (v24 rs 4) | 694 (torch) |
| 1024 | 2560 | 8192 | 40 | 40 | v24 rs (6) | 806 | 2.564 | 1.171 | 1.247 (v24 rs 3) | 1.074 (v24 rs 4) | 1.017 (v25 rs 5) | 1.074 (v24 rs 4) | 788 (torch) |
| 1024 | 3072 | 512 | 48 | 48 | dp (-) | 209 | 1.000 | 1.036 | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 366 (flydsl) |
| 1024 | 3072 | 1536 | 48 | 48 | v26 capped (-) | 311 | 1.116 | 1.118 | 1.060 (v24 owner 3) | 1.026 (v24 rs 4) | 1.122 (v24 rs default) | 1.026 (v24 rs 4) | 555 (flydsl) |
| 1024 | 3072 | 6144 | 48 | 48 | v24 rs (4) | 713 | 2.169 | 1.140 | 1.087 (v24 owner default) | 1.000 (v24 rs 4) | 1.112 (v24 rs default) | 1.000 (v24 rs 4) | 462 (opus) |
| 1024 | 3584 | 7168 | 56 | 56 | v24 rs (4) | 857 | 2.113 | 1.132 | 1.139 (v24 owner default) | 1.117 (v24 rs 3) | 1.139 (v24 owner default) | 1.117 (v24 rs 3) | 778 (flydsl) |
| 1024 | 4096 | 512 | 64 | 64 | v24 capped (-) | 260 | 1.048 | 1.000 | 1.048 (dp -) | 1.048 (dp -) | 1.048 (dp -) | 1.048 (dp -) | 441 (flydsl) |
| 1024 | 4096 | 1024 | 64 | 64 | v26 capped (-) | 351 | 1.005 | 1.033 | 1.149 (v24 owner 2) | 1.005 (dp -) | 1.005 (dp -) | 1.005 (dp -) | 559 (torch) |
| 1024 | 4096 | 1536 | 64 | 64 | v24 owner (2) | 377 | 1.028 | 1.032 | 1.005 (v24 owner 3) | 1.026 (v24 rs 3) | 1.005 (v24 owner 3) | 1.026 (v24 rs 3) | 561 (torch) |
| 1024 | 4096 | 2048 | 64 | 64 | v24 owner (3) | 486 | 1.153 | 1.046 | 1.000 (v24 owner 3) | 1.029 (v24 rs 3) | 1.000 (v24 owner 3) | 1.029 (v24 rs 3) | 616 (torch) |
| 1024 | 4096 | 4096 | 64 | 64 | v26 capped (-) | 718 | 1.517 | 1.104 | 1.039 (v24 owner default) | 1.005 (v24 rs 3) | 1.039 (v24 owner default) | 1.005 (v24 rs 3) | 830 (flydsl) |
| 1024 | 4096 | 8192 | 64 | 64 | v24 rs (default) | 977 | 1.987 | 1.030 | 1.012 (v24 owner default) | 1.077 (v24 rs 3) | 1.012 (v24 owner default) | 1.077 (v24 rs 3) | 897 (flydsl) |
| 1024 | 4608 | 8192 | 72 | 72 | v24 owner (3) | 1016 | 1.821 | 1.088 | 1.088 (v24 owner default) | 1.088 (v24 owner default) | 1.088 (v24 owner default) | 1.088 (v24 owner default) | 941 (torch) |
| 1024 | 4608 | 16384 | 72 | 72 | v24 owner (3) | 1186 | 2.113 | 1.026 | 1.030 (v24 owner default) | 1.030 (v24 owner default) | 1.030 (v24 owner default) | 1.030 (v24 owner default) | 650 (flydsl) |
| 1024 | 5120 | 640 | 80 | 80 | v24 capped (-) | 329 | 1.002 | 1.000 | 1.002 (dp -) | 1.002 (dp -) | 1.002 (dp -) | 1.002 (dp -) | 448 (torch) |
| 1024 | 5120 | 1280 | 80 | 80 | v26 capped (-) | 397 | 1.005 | 1.004 | 1.062 (v24 owner default) | 1.005 (dp -) | 1.005 (dp -) | 1.005 (dp -) | 604 (torch) |
| 1024 | 5120 | 3200 | 80 | 80 | v24 owner (3) | 723 | 1.351 | 1.017 | 1.032 (v24 owner default) | 1.032 (v24 owner default) | 1.032 (v24 owner default) | 1.032 (v24 owner default) | 822 (torch) |
| 1024 | 5120 | 5120 | 80 | 80 | v24 owner (3) | 901 | 1.672 | 1.165 | 1.002 (v24 owner default) | 1.002 (v24 owner default) | 1.002 (v24 owner default) | 1.002 (v24 owner default) | 848 (torch) |
| 1024 | 5120 | 6400 | 80 | 80 | v24 owner (3) | 974 | 1.729 | 1.160 | 1.020 (v24 owner default) | 1.020 (v24 owner default) | 1.020 (v24 owner default) | 1.020 (v24 owner default) | 600 (flydsl) |
| 1024 | 5120 | 25600 | 80 | 80 | v24 owner (default) | 1335 | 2.052 | 1.003 | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1144 (torch) |
| 1024 | 6144 | 2048 | 96 | 96 | v24 owner (2) | 650 | 1.060 | 1.055 | 1.191 (v24 owner default) | 1.191 (v24 owner default) | 1.191 (v24 owner default) | 1.191 (v24 owner default) | 739 (flydsl) |
| 1024 | 6144 | 3072 | 96 | 96 | v24 owner (2) | 756 | 1.215 | 1.134 | 1.131 (v24 owner default) | 1.131 (v24 owner default) | 1.131 (v24 owner default) | 1.131 (v24 owner default) | 790 (flydsl) |
| 1024 | 6144 | 4608 | 96 | 96 | v24 owner (2) | 913 | 1.365 | 1.094 | 1.096 (v24 owner default) | 1.096 (v24 owner default) | 1.096 (v24 owner default) | 1.096 (v24 owner default) | 895 (torch) |
| 1024 | 6144 | 6144 | 96 | 96 | v24 owner (2) | 983 | 1.520 | 1.104 | 1.104 (v24 owner default) | 1.104 (v24 owner default) | 1.104 (v24 owner default) | 1.104 (v24 owner default) | 1024 (torch) |
| 1024 | 6144 | 7168 | 96 | 96 | v24 owner (2) | 1029 | 1.490 | 1.095 | 1.095 (v24 owner default) | 1.095 (v24 owner default) | 1.095 (v24 owner default) | 1.095 (v24 owner default) | 981 (torch) |
| 1024 | 6144 | 12288 | 96 | 96 | v24 owner (2) | 1124 | 1.479 | 1.029 | 1.019 (v24 owner default) | 1.019 (v24 owner default) | 1.019 (v24 owner default) | 1.019 (v24 owner default) | 1094 (torch) |
| 1024 | 6144 | 18432 | 96 | 96 | v24 owner (2) | 1239 | 1.589 | 1.033 | 1.032 (v24 owner default) | 1.032 (v24 owner default) | 1.032 (v24 owner default) | 1.032 (v24 owner default) | 1168 (torch) |
| 1024 | 6400 | 5120 | 100 | 100 | v24 owner (2) | 938 | 1.396 | 1.076 | 1.075 (v24 owner default) | 1.075 (v24 owner default) | 1.075 (v24 owner default) | 1.075 (v24 owner default) | 825 (flydsl) |
| 1024 | 7168 | 512 | 112 | 112 | v24 capped (-) | 425 | 1.018 | 1.000 | 1.018 (dp -) | 1.018 (dp -) | 1.018 (dp -) | 1.018 (dp -) | 512 (flydsl) |
| 1024 | 7168 | 768 | 112 | 112 | v24 capped (-) | 502 | 1.027 | 1.000 | 1.027 (dp -) | 1.027 (dp -) | 1.027 (dp -) | 1.027 (dp -) | 598 (flydsl) |
| 1024 | 7168 | 1024 | 112 | 112 | v26 capped (-) | 589 | 1.024 | 1.020 | 1.024 (dp -) | 1.024 (dp -) | 1.024 (dp -) | 1.024 (dp -) | 669 (flydsl) |
| 1024 | 7168 | 1536 | 112 | 112 | v26 capped (-) | 622 | 1.001 | 1.002 | 1.109 (v24 owner default) | 1.001 (dp -) | 1.001 (dp -) | 1.001 (dp -) | 777 (torch) |
| 1024 | 7168 | 1792 | 112 | 112 | v24 owner (2) | 677 | 1.007 | 1.008 | 1.123 (v24 owner default) | 1.123 (v24 owner default) | 1.123 (v24 owner default) | 1.123 (v24 owner default) | 784 (flydsl) |
| 1024 | 7168 | 3584 | 112 | 112 | v24 owner (2) | 933 | 1.248 | 1.051 | 1.057 (v24 owner default) | 1.057 (v24 owner default) | 1.057 (v24 owner default) | 1.057 (v24 owner default) | 959 (torch) |
| 1024 | 7168 | 4224 | 112 | 112 | v24 owner (2) | 1008 | 1.344 | 1.066 | 1.049 (v24 owner default) | 1.049 (v24 owner default) | 1.049 (v24 owner default) | 1.049 (v24 owner default) | 934 (flydsl) |
| 1024 | 7168 | 8192 | 112 | 112 | v24 owner (2) | 1180 | 1.408 | 1.007 | 1.006 (v24 owner default) | 1.006 (v24 owner default) | 1.006 (v24 owner default) | 1.006 (v24 owner default) | 737 (flydsl) |
| 1024 | 7168 | 35840 | 112 | 112 | v24 capped (-) | 1390 | 1.594 | 1.000 | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 575 (flydsl) |
| 1024 | 8192 | 1024 | 128 | 128 | dp (-) | 639 | 1.000 | 1.003 | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 757 (torch) |
| 1024 | 8192 | 2048 | 128 | 128 | v24 owner (2) | 790 | 1.014 | 1.023 | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 884 (flydsl) |
| 1024 | 8192 | 3584 | 128 | 128 | v24 owner (default) | 1024 | 1.225 | 1.234 | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 791 (flydsl) |
| 1024 | 8192 | 4096 | 128 | 128 | v24 owner (default) | 1054 | 1.190 | 1.006 | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1022 (flydsl) |
| 1024 | 8192 | 7168 | 128 | 128 | v24 capped (-) | 1225 | 1.361 | 1.000 | 1.007 (v24 owner default) | 1.007 (v24 owner default) | 1.007 (v24 owner default) | 1.007 (v24 owner default) | 1016 (flydsl) |
| 1024 | 8192 | 8192 | 128 | 128 | v24 owner (2) | 1275 | 1.368 | 1.022 | 1.017 (v24 owner default) | 1.017 (v24 owner default) | 1.017 (v24 owner default) | 1.017 (v24 owner default) | 916 (flydsl) |
| 1024 | 8192 | 28672 | 128 | 128 | v24 owner (2) | 1481 | 1.562 | 1.002 | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 694 (triton) |
| 1024 | 8448 | 7168 | 132 | 132 | v26 capped (-) | 1124 | 1.209 | 1.212 | 1.310 (v24 owner default) | 1.209 (dp -) | 1.209 (dp -) | 1.003 (v26 spread -) | 1039 (torch) |
| 1024 | 8704 | 4096 | 136 | 136 | dp (-) | 981 | 1.000 | 1.006 | 1.217 (v24 owner default) | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 998 (torch) |
| 1024 | 9216 | 6144 | 144 | 144 | v26 capped (-) | 1109 | 1.142 | 1.131 | 1.253 (v24 owner default) | 1.142 (dp -) | 1.142 (dp -) | 1.002 (v26 spread -) | 1066 (torch) |
| 1024 | 9216 | 16384 | 144 | 144 | v26 spread (-) | 1237 | 1.082 | 1.079 | 1.190 (v24 owner default) | 1.082 (dp -) | 1.082 (dp -) | 1.000 (v26 spread -) | 678 (flydsl) |
| 1024 | 10240 | 4096 | 160 | 160 | v24 capped (-) | 1117 | 1.004 | 1.000 | 1.172 (v24 owner default) | 1.004 (dp -) | 1.004 (dp -) | 1.004 (dp -) | 1123 (torch) |
| 1024 | 10240 | 8192 | 160 | 160 | v26 capped (-) | 1216 | 1.033 | 1.036 | 1.162 (v24 owner default) | 1.033 (dp -) | 1.033 (dp -) | 1.002 (v26 spread -) | 828 (flydsl) |
| 1024 | 12800 | 5120 | 200 | 200 | dp (-) | 1246 | 1.000 | 1.002 | 1.044 (v24 owner default) | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 723 (flydsl) |
| 1024 | 13312 | 16384 | 208 | 208 | v26 capped (-) | 1402 | 1.041 | 1.016 | 1.028 (v24 owner default) | 1.041 (dp -) | 1.041 (dp -) | 1.006 (v26 spread -) | 1334 (asm) |
| 1024 | 14336 | 8192 | 224 | 224 | v26 capped (-) | 1454 | 1.001 | 1.000 | 1.042 (v24 owner default) | 1.001 (dp -) | 1.001 (dp -) | 1.001 (dp -) | 1358 (flydsl) |
| 1024 | 20480 | 7168 | 320 | 64 | v24 rs (4) | 1360 | 1.214 | 1.010 | 1.006 (v24 owner default) | 1.006 (v24 owner default) | 1.006 (v24 owner default) | 1.006 (v24 owner default) | 747 (flydsl) |
| 1024 | 26624 | 16384 | 416 | 160 | v26 capped (-) | 1480 | 1.057 | 1.058 | 1.057 (v24 owner default) | 1.057 (dp -) | 1.057 (dp -) | 1.003 (v26 spread -) | 949 (flydsl) |
| 1024 | 37888 | 5120 | 592 | 80 | v24 owner (3) | 1390 | 1.091 | 1.023 | 1.006 (v24 owner default) | 1.006 (v24 owner default) | 1.006 (v24 owner default) | 1.006 (v24 owner default) | 1058 (torch) |
| 1024 | 51200 | 5120 | 800 | 32 | v24 rs (4) | 1408 | 1.096 | 1.001 | 1.000 (v24 rs 4) | 1.000 (v24 rs 4) | 1.000 (v24 rs 4) | 1.000 (v24 rs 4) | 1225 (asm) |
| 1024 | 53248 | 16384 | 832 | 64 | v26 capped (-) | 1550 | 1.106 | 1.001 | 1.003 (v24 owner default) | 1.003 (v24 owner default) | 1.003 (v24 owner default) | 1.003 (v24 owner default) | 990 (triton) |
| 1024 | 57344 | 8192 | 896 | 128 | v24 owner (default) | 1517 | 1.058 | 1.000 | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1383 (asm) |
| 1024 | 129280 | 4096 | 2020 | 228 | dp (-) | 1482 | 1.000 | 1.002 | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1369 (torch) |
| 1280 | 512 | 4096 | 10 | 10 | v24 rs (8) | 217 | 2.874 | 1.002 | 1.003 (v24 rs 12) | 1.018 (v24 rs 16) | 1.061 (v25 rs 10) | 1.018 (v24 rs 16) | 321 (flydsl) |
| 1280 | 512 | 7168 | 10 | 10 | v25 rs (19) | 304 | 4.240 | 1.063 | 1.053 (v24 rs 12) | 1.000 (v25 rs 19) | 1.127 (v25 rs 10) | 1.000 (v25 rs 19) | 365 (torch) |
| 1280 | 1024 | 4096 | 20 | 20 | v25 rs (9) | 350 | 2.303 | 1.033 | 1.021 (v24 rs 6) | 1.000 (v25 rs 9) | 1.046 (v25 rs 7) | 1.000 (v25 rs 9) | 487 (torch) |
| 1280 | 1024 | 7168 | 20 | 20 | v24 rs (8) | 490 | 3.381 | 1.151 | 1.153 (v24 rs 6) | 1.100 (v25 rs 9) | 1.064 (v25 rs 7) | 1.100 (v25 rs 9) | 536 (torch) |
| 1280 | 2048 | 4096 | 40 | 40 | v25 rs (5) | 549 | 1.808 | 1.186 | 1.108 (v24 rs 3) | 1.003 (v24 rs 4) | 1.000 (v25 rs 5) | 1.003 (v24 rs 4) | 622 (flydsl) |
| 1280 | 2048 | 7168 | 40 | 40 | v24 rs (6) | 729 | 2.507 | 1.164 | 1.265 (v24 rs 3) | 1.059 (v24 rs 4) | 1.027 (v25 rs 5) | 1.059 (v24 rs 4) | 747 (torch) |
| 1280 | 129280 | 4096 | 2525 | 221 | dp (-) | 1452 | 1.000 | 1.001 | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1387 (torch) |
| 1536 | 512 | 4096 | 12 | 12 | v24 capped (-) | 255 | 2.809 | 1.000 | 1.060 (v24 rs 10) | 1.098 (v24 rs 16) | 1.074 (v25 rs 9) | 1.098 (v24 rs 16) | 366 (torch) |
| 1536 | 512 | 7168 | 12 | 12 | v24 rs (8) | 331 | 3.819 | 1.050 | 1.058 (v24 rs 10) | 1.014 (v24 rs 16) | 1.095 (v25 rs 9) | 1.014 (v24 rs 16) | 404 (torch) |
| 1536 | 1024 | 4096 | 24 | 24 | v26 capped (-) | 401 | 2.193 | 1.052 | 1.076 (v24 rs 5) | 1.027 (v24 rs 8) | 1.034 (v25 rs 7) | 1.027 (v24 rs 8) | 520 (flydsl) |
| 1536 | 1024 | 5120 | 24 | 24 | v24 rs (8) | 442 | 2.711 | 1.060 | 1.059 (v24 rs 5) | 1.000 (v24 rs 8) | 1.069 (v25 rs 7) | 1.000 (v24 rs 8) | 437 (asm) |
| 1536 | 1024 | 7168 | 24 | 24 | v26 capped (-) | 551 | 3.168 | 1.146 | 1.139 (v24 rs 5) | 1.007 (v24 rs 8) | 1.030 (v25 rs 7) | 1.007 (v24 rs 8) | 648 (torch) |
| 1536 | 2048 | 4096 | 48 | 48 | v26 capped (-) | 621 | 1.699 | 1.129 | 1.195 (v24 owner default) | 1.014 (v24 rs 4) | 1.057 (v24 rs default) | 1.014 (v24 rs 4) | 701 (flydsl) |
| 1536 | 2048 | 7168 | 48 | 48 | v24 rs (4) | 788 | 2.242 | 1.188 | 1.055 (v24 owner default) | 1.000 (v24 rs 4) | 1.067 (v24 rs default) | 1.000 (v24 rs 4) | 842 (torch) |
| 1536 | 37888 | 5120 | 888 | 120 | v24 owner (2) | 1436 | 1.040 | 1.003 | 1.002 (v24 owner default) | 1.002 (v24 owner default) | 1.002 (v24 owner default) | 1.002 (v24 owner default) | 1271 (torch) |
| 1536 | 129280 | 4096 | 3030 | 214 | v26 capped (-) | 1474 | 1.002 | 1.001 | 1.002 (dp -) | 1.002 (dp -) | 1.002 (dp -) | 1.002 (dp -) | 1394 (torch) |
| 1792 | 512 | 4096 | 14 | 14 | v24 rs (8) | 287 | 2.705 | 1.008 | 1.044 (v24 rs 9) | 1.147 (v25 rs 13) | 1.044 (v24 rs 9) | 1.147 (v25 rs 13) | 396 (flydsl) |
| 1792 | 512 | 7168 | 14 | 14 | v24 rs (8) | 378 | 3.755 | 1.089 | 1.089 (v24 rs 9) | 1.056 (v25 rs 13) | 1.089 (v24 rs 9) | 1.056 (v25 rs 13) | 442 (torch) |
| 1792 | 1024 | 4096 | 28 | 28 | v24 rs (6) | 436 | 2.046 | 1.016 | 1.021 (v24 rs 4) | 1.000 (v24 rs 6) | 1.000 (v24 rs 6) | 1.000 (v24 rs 6) | 577 (flydsl) |
| 1792 | 1024 | 7168 | 28 | 28 | v24 rs (8) | 583 | 2.840 | 1.125 | 1.129 (v24 rs 4) | 1.035 (v24 rs 6) | 1.035 (v24 rs 6) | 1.035 (v24 rs 6) | 601 (flydsl) |
| 1792 | 2048 | 4096 | 56 | 56 | v24 rs (4) | 664 | 1.565 | 1.156 | 1.154 (v24 owner default) | 1.027 (v24 rs 3) | 1.154 (v24 owner default) | 1.027 (v24 rs 3) | 752 (flydsl) |
| 1792 | 2048 | 7168 | 56 | 56 | v24 rs (4) | 866 | 2.102 | 1.138 | 1.164 (v24 owner default) | 1.118 (v24 rs 3) | 1.164 (v24 owner default) | 1.118 (v24 rs 3) | 778 (flydsl) |
| 1792 | 129280 | 4096 | 3535 | 207 | v24 capped (-) | 1490 | 1.000 | 1.000 | 1.002 (v24 owner default) | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1438 (asm) |
| 2048 | 512 | 4096 | 16 | 16 | v24 capped (-) | 314 | 2.578 | 1.000 | 1.001 (v24 rs 8) | 1.107 (v24 rs 12) | 1.001 (v24 rs 8) | 1.107 (v24 rs 12) | 426 (flydsl) |
| 2048 | 512 | 7168 | 16 | 16 | v24 rs (8) | 421 | 3.633 | 1.003 | 1.000 (v24 rs 8) | 1.053 (v24 rs 12) | 1.000 (v24 rs 8) | 1.053 (v24 rs 12) | 507 (torch) |
| 2048 | 512 | 8192 | 16 | 16 | v24 capped (-) | 471 | 3.729 | 1.000 | 1.001 (v24 rs 8) | 1.050 (v24 rs 12) | 1.001 (v24 rs 8) | 1.050 (v24 rs 12) | 507 (torch) |
| 2048 | 1024 | 4096 | 32 | 32 | v24 capped (-) | 473 | 1.935 | 1.000 | 1.001 (v24 rs 4) | 1.003 (v24 rs 6) | 1.003 (v24 rs 6) | 1.003 (v24 rs 6) | 603 (flydsl) |
| 2048 | 1024 | 5120 | 32 | 32 | v24 rs (6) | 529 | 2.425 | 1.064 | 1.064 (v24 rs 4) | 1.000 (v24 rs 6) | 1.000 (v24 rs 6) | 1.000 (v24 rs 6) | 430 (opus) |
| 2048 | 1024 | 7168 | 32 | 32 | v24 rs (6) | 633 | 2.693 | 1.094 | 1.093 (v24 rs 4) | 1.000 (v24 rs 6) | 1.000 (v24 rs 6) | 1.000 (v24 rs 6) | 708 (torch) |
| 2048 | 1280 | 8192 | 40 | 40 | v24 rs (6) | 818 | 2.577 | 1.187 | 1.251 (v24 rs 3) | 1.085 (v24 rs 4) | 1.039 (v25 rs 5) | 1.085 (v24 rs 4) | 700 (torch) |
| 2048 | 1536 | 1536 | 48 | 48 | v26 capped (-) | 312 | 1.102 | 1.110 | 1.059 (v24 owner 3) | 1.035 (v24 rs 4) | 1.138 (v24 rs default) | 1.035 (v24 rs 4) | 553 (torch) |
| 2048 | 2048 | 512 | 64 | 64 | v24 capped (-) | 262 | 1.054 | 1.000 | 1.054 (dp -) | 1.054 (dp -) | 1.054 (dp -) | 1.054 (dp -) | 431 (flydsl) |
| 2048 | 2048 | 1536 | 64 | 64 | v24 rs (2) | 381 | 1.026 | 1.028 | 1.002 (v24 owner 3) | 1.034 (v24 rs 3) | 1.002 (v24 owner 3) | 1.034 (v24 rs 3) | 563 (torch) |
| 2048 | 2048 | 4096 | 64 | 64 | v24 rs (3) | 715 | 1.498 | 1.100 | 1.026 (v24 owner default) | 1.000 (v24 rs 3) | 1.026 (v24 owner default) | 1.000 (v24 rs 3) | 834 (flydsl) |
| 2048 | 2048 | 6144 | 64 | 64 | v24 rs (4) | 844 | 1.893 | 1.035 | 1.005 (v24 owner default) | 1.041 (v24 rs 3) | 1.005 (v24 owner default) | 1.041 (v24 rs 3) | 678 (opus) |
| 2048 | 2048 | 7168 | 64 | 64 | v24 rs (4) | 929 | 1.964 | 1.072 | 1.004 (v24 owner default) | 1.067 (v24 rs 3) | 1.004 (v24 owner default) | 1.067 (v24 rs 3) | 875 (flydsl) |
| 2048 | 2304 | 6144 | 72 | 72 | v24 owner (3) | 899 | 1.810 | 1.028 | 1.029 (v24 owner default) | 1.029 (v24 owner default) | 1.029 (v24 owner default) | 1.029 (v24 owner default) | 846 (torch) |
| 2048 | 2304 | 16384 | 72 | 72 | v24 owner (3) | 1193 | 2.133 | 1.016 | 1.014 (v24 owner default) | 1.014 (v24 owner default) | 1.014 (v24 owner default) | 1.014 (v24 owner default) | 876 (torch) |
| 2048 | 2560 | 6144 | 80 | 80 | v24 owner (3) | 970 | 1.747 | 1.135 | 1.019 (v24 owner default) | 1.019 (v24 owner default) | 1.019 (v24 owner default) | 1.019 (v24 owner default) | 909 (torch) |
| 2048 | 2560 | 8192 | 80 | 80 | v24 owner (3) | 1107 | 1.793 | 1.032 | 1.026 (v24 owner default) | 1.026 (v24 owner default) | 1.026 (v24 owner default) | 1.026 (v24 owner default) | 988 (torch) |
| 2048 | 3072 | 6144 | 96 | 96 | v24 owner (2) | 998 | 1.493 | 1.016 | 1.021 (v24 owner default) | 1.021 (v24 owner default) | 1.021 (v24 owner default) | 1.021 (v24 owner default) | 697 (opus) |
| 2048 | 3584 | 7168 | 112 | 112 | v24 owner (2) | 1171 | 1.423 | 1.018 | 1.018 (v24 owner default) | 1.018 (v24 owner default) | 1.018 (v24 owner default) | 1.018 (v24 owner default) | 1004 (opus) |
| 2048 | 4096 | 512 | 128 | 128 | dp (-) | 469 | 1.000 | 1.002 | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 560 (torch) |
| 2048 | 4096 | 1024 | 128 | 128 | dp (-) | 643 | 1.000 | 1.004 | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 768 (torch) |
| 2048 | 4096 | 1536 | 128 | 128 | dp (-) | 700 | 1.000 | 1.004 | 1.067 (v24 owner default) | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 849 (torch) |
| 2048 | 4096 | 2048 | 128 | 128 | v24 owner (default) | 809 | 1.025 | 1.028 | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 940 (torch) |
| 2048 | 4096 | 4096 | 128 | 128 | v24 owner (2) | 1074 | 1.206 | 1.003 | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1034 (opus) |
| 2048 | 4096 | 8192 | 128 | 128 | v24 owner (default) | 1304 | 1.378 | 1.004 | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1140 (opus) |
| 2048 | 4608 | 8192 | 144 | 144 | v26 capped (-) | 1197 | 1.086 | 1.086 | 1.186 (v24 owner default) | 1.086 (dp -) | 1.086 (dp -) | 1.004 (v26 spread -) | 1161 (torch) |
| 2048 | 4608 | 16384 | 144 | 144 | v26 spread (-) | 1261 | 1.140 | 1.136 | 1.216 (v24 owner default) | 1.140 (dp -) | 1.140 (dp -) | 1.000 (v26 spread -) | 718 (flydsl) |
| 2048 | 5120 | 640 | 160 | 160 | v24 capped (-) | 626 | 1.030 | 1.000 | 1.030 (dp -) | 1.030 (dp -) | 1.030 (dp -) | 1.030 (dp -) | 597 (flydsl) |
| 2048 | 5120 | 1280 | 160 | 160 | v26 capped (-) | 764 | 1.007 | 1.005 | 1.007 (dp -) | 1.007 (dp -) | 1.007 (dp -) | 1.007 (dp -) | 847 (torch) |
| 2048 | 5120 | 3200 | 160 | 160 | v24 capped (-) | 1053 | 1.001 | 1.000 | 1.101 (v24 owner default) | 1.001 (dp -) | 1.001 (dp -) | 1.001 (dp -) | 1092 (torch) |
| 2048 | 5120 | 5120 | 160 | 160 | v26 spread (-) | 1099 | 1.025 | 1.024 | 1.085 (v24 owner default) | 1.025 (dp -) | 1.025 (dp -) | 1.025 (dp -) | 1153 (torch) |
| 2048 | 5120 | 6400 | 160 | 160 | v26 spread (-) | 1174 | 1.048 | 1.042 | 1.142 (v24 owner default) | 1.048 (dp -) | 1.048 (dp -) | 1.048 (dp -) | 744 (flydsl) |
| 2048 | 5120 | 25600 | 160 | 160 | v26 capped (-) | 1431 | 1.189 | 1.187 | 1.342 (v24 owner default) | 1.189 (dp -) | 1.189 (dp -) | 1.000 (v26 spread -) | 1290 (torch) |
| 2048 | 6144 | 2048 | 192 | 192 | dp (-) | 1117 | 1.000 | 1.015 | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1046 (flydsl) |
| 2048 | 6144 | 3072 | 192 | 192 | v26 capped (-) | 1180 | 1.006 | 1.009 | 1.151 (v24 owner default) | 1.006 (dp -) | 1.006 (dp -) | 1.006 (dp -) | 1120 (flydsl) |
| 2048 | 6144 | 4608 | 192 | 192 | v24 capped (-) | 1255 | 1.003 | 1.000 | 1.086 (v24 owner default) | 1.003 (dp -) | 1.003 (dp -) | 1.003 (dp -) | 1192 (flydsl) |
| 2048 | 6144 | 6144 | 192 | 192 | dp (-) | 1274 | 1.000 | 1.014 | 1.087 (v24 owner default) | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1267 (torch) |
| 2048 | 6144 | 12288 | 192 | 192 | v26 spread (-) | 1403 | 1.052 | 1.048 | 1.090 (v24 owner default) | 1.052 (dp -) | 1.052 (dp -) | 1.000 (v26 spread -) | 1323 (torch) |
| 2048 | 6144 | 18432 | 192 | 192 | v26 spread (-) | 1481 | 1.086 | 1.086 | 1.131 (v24 owner default) | 1.086 (dp -) | 1.086 (dp -) | 1.000 (v26 spread -) | 1344 (torch) |
| 2048 | 6400 | 5120 | 200 | 200 | v24 capped (-) | 1273 | 1.000 | 1.000 | 1.060 (v24 owner default) | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1133 (flydsl) |
| 2048 | 7168 | 1024 | 224 | 224 | v24 capped (-) | 1003 | 1.003 | 1.000 | 1.003 (dp -) | 1.003 (dp -) | 1.003 (dp -) | 1.003 (dp -) | 960 (flydsl) |
| 2048 | 7168 | 1792 | 224 | 224 | dp (-) | 1175 | 1.000 | 1.009 | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1119 (flydsl) |
| 2048 | 7168 | 8192 | 224 | 224 | v24 capped (-) | 1442 | 1.004 | 1.000 | 1.022 (v24 owner default) | 1.004 (dp -) | 1.004 (dp -) | 1.004 (dp -) | 1377 (asm) |
| 2048 | 7168 | 35840 | 224 | 224 | v26 capped (-) | 1550 | 1.030 | 1.007 | 1.085 (v24 owner default) | 1.007 (v25 spread -) | 1.007 (v25 spread -) | 1.001 (v26 spread -) | 1452 (asm) |
| 2048 | 8704 | 4096 | 272 | 16 | v24 capped (-) | 1272 | 1.313 | 1.000 | 1.008 (v24 rs 8) | 1.008 (v24 rs 8) | 1.008 (v24 rs 8) | 1.008 (v24 rs 8) | 1083 (torch) |
| 2048 | 9216 | 6144 | 288 | 32 | v24 rs (6) | 1334 | 1.319 | 1.006 | 1.010 (v24 rs 4) | 1.010 (v24 rs 4) | 1.010 (v24 rs 4) | 1.010 (v24 rs 4) | 1102 (torch) |
| 2048 | 9216 | 16384 | 288 | 32 | v24 rs (default) | 1504 | 1.390 | 1.028 | 1.030 (v24 rs 4) | 1.030 (v24 rs 4) | 1.030 (v24 rs 4) | 1.030 (v24 rs 4) | 913 (flydsl) |
| 2048 | 10240 | 4096 | 320 | 64 | v24 rs (4) | 1265 | 1.121 | 1.012 | 1.003 (v24 owner default) | 1.003 (v24 owner default) | 1.003 (v24 owner default) | 1.003 (v24 owner default) | 1162 (torch) |
| 2048 | 10240 | 8192 | 320 | 64 | v24 rs (default) | 1422 | 1.211 | 1.004 | 1.005 (v24 owner default) | 1.005 (v24 owner default) | 1.005 (v24 owner default) | 1.005 (v24 owner default) | 933 (flydsl) |
| 2048 | 12800 | 5120 | 400 | 144 | v26 spread (-) | 1330 | 1.005 | 1.004 | 1.019 (v24 owner default) | 1.005 (dp -) | 1.005 (dp -) | 1.005 (dp -) | 803 (flydsl) |
| 2048 | 13312 | 16384 | 416 | 160 | v26 capped (-) | 1514 | 1.060 | 1.058 | 1.091 (v24 owner default) | 1.060 (dp -) | 1.060 (dp -) | 1.004 (v26 spread -) | 1365 (flydsl) |
| 2048 | 14336 | 8192 | 448 | 192 | v26 spread (-) | 1485 | 1.012 | 1.016 | 1.015 (v24 owner default) | 1.012 (dp -) | 1.012 (dp -) | 1.012 (dp -) | 1404 (flydsl) |
| 2048 | 20480 | 7168 | 640 | 128 | v24 capped (-) | 1509 | 1.068 | 1.000 | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 832 (flydsl) |
| 2048 | 26624 | 16384 | 832 | 64 | v24 rs (4) | 1586 | 1.108 | 1.003 | 1.002 (v24 owner default) | 1.002 (v24 owner default) | 1.002 (v24 owner default) | 1.002 (v24 owner default) | 910 (flydsl) |
| 2048 | 37888 | 5120 | 1184 | 160 | dp (-) | 1477 | 1.000 | 1.007 | 1.011 (v24 owner default) | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1210 (torch) |
| 2048 | 51200 | 5120 | 1600 | 64 | v24 rs (4) | 1495 | 1.031 | 1.011 | 1.002 (v24 owner default) | 1.002 (v24 owner default) | 1.002 (v24 owner default) | 1.002 (v24 owner default) | 213 (asm) |
| 2048 | 53248 | 16384 | 1664 | 128 | v26 capped (-) | 1608 | 1.046 | 1.005 | 1.003 (v24 owner default) | 1.003 (v24 owner default) | 1.003 (v24 owner default) | 1.003 (v24 owner default) | 801 (flydsl) |
| 2048 | 129280 | 4096 | 4040 | 200 | dp (-) | 1523 | 1.000 | 1.001 | 1.003 (v24 owner default) | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1431 (flydsl) |
| 2304 | 512 | 4096 | 18 | 18 | v24 rs (8) | 333 | 2.446 | 1.106 | 1.095 (v24 rs 7) | 1.050 (v25 rs 10) | 1.000 (v24 rs 8) | 1.050 (v25 rs 10) | 351 (flydsl) |
| 2304 | 1024 | 4096 | 36 | 36 | v24 rs (4) | 509 | 1.860 | 1.212 | 1.132 (v24 rs 3) | 1.024 (v25 rs 5) | 1.024 (v25 rs 5) | 1.024 (v25 rs 5) | 512 (flydsl) |
| 2304 | 2048 | 4096 | 72 | 72 | v24 rs (3) | 781 | 1.467 | 1.116 | 1.119 (v24 owner default) | 1.119 (v24 owner default) | 1.119 (v24 owner default) | 1.119 (v24 owner default) | 688 (opus) |
| 2560 | 512 | 4096 | 20 | 20 | v24 rs (8) | 350 | 2.313 | 1.035 | 1.039 (v24 rs 6) | 1.033 (v25 rs 9) | 1.081 (v25 rs 7) | 1.033 (v25 rs 9) | 390 (flydsl) |
| 2560 | 1024 | 4096 | 40 | 40 | v24 rs (4) | 553 | 1.816 | 1.189 | 1.116 (v24 rs 3) | 1.000 (v24 rs 4) | 1.029 (v25 rs 5) | 1.000 (v24 rs 4) | 560 (flydsl) |
| 2560 | 2048 | 4096 | 80 | 80 | v24 owner (3) | 817 | 1.387 | 1.054 | 1.011 (v24 owner default) | 1.011 (v24 owner default) | 1.011 (v24 owner default) | 1.011 (v24 owner default) | 770 (opus) |
| 2816 | 512 | 4096 | 22 | 22 | v26 capped (-) | 378 | 2.274 | 1.092 | 1.099 (v24 rs 5) | 1.001 (v24 rs 8) | 1.042 (v25 rs 7) | 1.001 (v24 rs 8) | 419 (flydsl) |
| 2816 | 1024 | 4096 | 44 | 44 | v24 rs (4) | 588 | 1.754 | 1.161 | 1.229 (v24 owner default) | 1.000 (v24 rs 4) | 1.064 (v24 rs default) | 1.000 (v24 rs 4) | 600 (flydsl) |
| 2816 | 2048 | 4096 | 88 | 88 | v24 owner (default) | 841 | 1.301 | 1.010 | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 824 (opus) |
| 3072 | 512 | 4096 | 24 | 24 | v24 rs (6) | 397 | 2.186 | 1.048 | 1.059 (v24 rs 5) | 1.017 (v24 rs 8) | 1.017 (v25 rs 7) | 1.017 (v24 rs 8) | 455 (flydsl) |
| 3072 | 1024 | 4096 | 48 | 48 | v24 rs (4) | 612 | 1.678 | 1.123 | 1.188 (v24 owner default) | 1.000 (v24 rs 4) | 1.038 (v24 rs default) | 1.000 (v24 rs 4) | 644 (flydsl) |
| 3072 | 2048 | 1536 | 96 | 96 | dp (-) | 543 | 1.000 | 1.003 | 1.221 (v24 owner default) | 1.221 (v24 owner default) | 1.221 (v24 owner default) | 1.221 (v24 owner default) | 690 (opus) |
| 3072 | 2048 | 4096 | 96 | 96 | v24 owner (2) | 899 | 1.294 | 1.128 | 1.130 (v24 owner default) | 1.130 (v24 owner default) | 1.130 (v24 owner default) | 1.130 (v24 owner default) | 915 (opus) |
| 3072 | 4096 | 1024 | 192 | 192 | dp (-) | 904 | 1.000 | 1.004 | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 913 (torch) |
| 3072 | 4096 | 1536 | 192 | 192 | dp (-) | 993 | 1.000 | 1.003 | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 996 (opus) |
| 3072 | 4096 | 2048 | 192 | 192 | dp (-) | 1099 | 1.000 | 1.003 | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1100 (torch) |
| 3328 | 512 | 4096 | 26 | 26 | v24 rs (6) | 420 | 2.141 | 1.048 | 1.046 (v24 rs 4) | 1.031 (v25 rs 7) | 1.000 (v24 rs 6) | 1.031 (v25 rs 7) | 485 (flydsl) |
| 3328 | 1024 | 4096 | 52 | 52 | v24 rs (4) | 636 | 1.618 | 1.193 | 1.147 (v24 owner default) | 1.038 (v24 rs 3) | 1.147 (v24 owner default) | 1.038 (v24 rs 3) | 683 (flydsl) |
| 3328 | 2048 | 4096 | 104 | 104 | v24 owner (2) | 948 | 1.266 | 1.101 | 1.114 (v24 owner default) | 1.114 (v24 owner default) | 1.114 (v24 owner default) | 1.114 (v24 owner default) | 942 (opus) |
| 3584 | 512 | 4096 | 28 | 28 | v24 rs (6) | 439 | 2.082 | 1.034 | 1.044 (v24 rs 4) | 1.000 (v24 rs 6) | 1.000 (v24 rs 6) | 1.000 (v24 rs 6) | 510 (flydsl) |
| 3584 | 1024 | 4096 | 56 | 56 | v24 rs (4) | 662 | 1.564 | 1.165 | 1.151 (v24 owner default) | 1.026 (v24 rs 3) | 1.151 (v24 owner default) | 1.026 (v24 rs 3) | 715 (flydsl) |
| 3584 | 2048 | 4096 | 112 | 112 | v24 owner (2) | 1001 | 1.254 | 1.055 | 1.042 (v24 owner default) | 1.042 (v24 owner default) | 1.042 (v24 owner default) | 1.042 (v24 owner default) | 1006 (opus) |
| 3840 | 512 | 4096 | 30 | 30 | v26 capped (-) | 458 | 2.015 | 1.027 | 1.035 (v24 rs 4) | 1.041 (v24 rs 6) | 1.041 (v24 rs 6) | 1.041 (v24 rs 6) | 539 (flydsl) |
| 3840 | 1024 | 4096 | 60 | 60 | v24 rs (3) | 685 | 1.512 | 1.129 | 1.072 (v24 owner default) | 1.000 (v24 rs 3) | 1.072 (v24 owner default) | 1.000 (v24 rs 3) | 742 (flydsl) |
| 3840 | 2048 | 4096 | 120 | 120 | v24 owner (2) | 1042 | 1.226 | 1.014 | 1.005 (v24 owner default) | 1.005 (v24 owner default) | 1.005 (v24 owner default) | 1.005 (v24 owner default) | 1038 (opus) |
| 4096 | 512 | 4096 | 32 | 32 | v24 capped (-) | 470 | 1.949 | 1.000 | 1.003 (v24 rs 4) | 1.025 (v24 rs 6) | 1.025 (v24 rs 6) | 1.025 (v24 rs 6) | 582 (flydsl) |
| 4096 | 512 | 7168 | 32 | 32 | v24 rs (6) | 619 | 2.687 | 1.086 | 1.085 (v24 rs 4) | 1.000 (v24 rs 6) | 1.000 (v24 rs 6) | 1.000 (v24 rs 6) | 626 (flydsl) |
| 4096 | 512 | 8192 | 32 | 32 | v26 capped (-) | 691 | 2.787 | 1.102 | 1.097 (v24 rs 4) | 1.006 (v24 rs 6) | 1.006 (v24 rs 6) | 1.006 (v24 rs 6) | 643 (flydsl) |
| 4096 | 1024 | 4096 | 64 | 64 | v26 capped (-) | 713 | 1.506 | 1.110 | 1.024 (v24 owner default) | 1.001 (v24 rs 3) | 1.024 (v24 owner default) | 1.001 (v24 rs 3) | 832 (flydsl) |
| 4096 | 1024 | 5120 | 64 | 64 | v24 rs (default) | 774 | 1.783 | 1.208 | 1.008 (v24 owner default) | 1.022 (v24 rs 3) | 1.008 (v24 owner default) | 1.022 (v24 rs 3) | 715 (opus) |
| 4096 | 1024 | 7168 | 64 | 64 | v24 rs (default) | 927 | 1.974 | 1.086 | 1.022 (v24 owner default) | 1.079 (v24 rs 3) | 1.022 (v24 owner default) | 1.079 (v24 rs 3) | 914 (torch) |
| 4096 | 1280 | 8192 | 80 | 80 | v24 owner (3) | 1092 | 1.784 | 1.019 | 1.020 (v24 owner default) | 1.020 (v24 owner default) | 1.020 (v24 owner default) | 1.020 (v24 owner default) | 986 (torch) |
| 4096 | 1536 | 1536 | 96 | 96 | v24 capped (-) | 537 | 1.001 | 1.000 | 1.170 (v24 owner default) | 1.170 (v24 owner default) | 1.170 (v24 owner default) | 1.170 (v24 owner default) | 767 (torch) |
| 4096 | 2048 | 512 | 128 | 128 | v24 capped (-) | 470 | 1.011 | 1.000 | 1.011 (dp -) | 1.011 (dp -) | 1.011 (dp -) | 1.011 (dp -) | 644 (torch) |
| 4096 | 2048 | 1536 | 128 | 128 | v26 capped (-) | 696 | 1.003 | 1.001 | 1.038 (v24 owner default) | 1.003 (dp -) | 1.003 (dp -) | 1.003 (dp -) | 856 (opus) |
| 4096 | 2048 | 4096 | 128 | 128 | v26 capped (-) | 1085 | 1.217 | 1.011 | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1094 (opus) |
| 4096 | 2048 | 6144 | 128 | 128 | v24 capped (-) | 1221 | 1.419 | 1.000 | 1.012 (v24 owner default) | 1.012 (v24 owner default) | 1.012 (v24 owner default) | 1.012 (v24 owner default) | 1031 (opus) |
| 4096 | 2048 | 7168 | 128 | 128 | v24 owner (2) | 1265 | 1.390 | 1.005 | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1182 (torch) |
| 4096 | 2304 | 6144 | 144 | 144 | v26 spread (-) | 1123 | 1.143 | 1.143 | 1.157 (v24 owner default) | 1.143 (dp -) | 1.143 (dp -) | 1.000 (v26 spread -) | 688 (flydsl) |
| 4096 | 2304 | 16384 | 144 | 144 | v26 spread (-) | 1261 | 1.201 | 1.198 | 1.233 (v24 owner default) | 1.201 (dp -) | 1.201 (dp -) | 1.000 (v26 spread -) | 1188 (torch) |
| 4096 | 2560 | 6144 | 160 | 160 | v26 spread (-) | 1145 | 1.054 | 1.055 | 1.180 (v24 owner default) | 1.054 (dp -) | 1.054 (dp -) | 1.054 (dp -) | 639 (flydsl) |
| 4096 | 2560 | 8192 | 160 | 160 | v26 capped (-) | 1231 | 1.036 | 1.038 | 1.174 (v24 owner default) | 1.036 (dp -) | 1.036 (dp -) | 1.003 (v26 spread -) | 1243 (torch) |
| 4096 | 3072 | 1536 | 192 | 192 | dp (-) | 995 | 1.000 | 1.004 | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 940 (flydsl) |
| 4096 | 3072 | 6144 | 192 | 192 | v26 capped (-) | 1262 | 1.006 | 1.004 | 1.077 (v24 owner default) | 1.006 (dp -) | 1.006 (dp -) | 1.006 (dp -) | 1148 (asm) |
| 4096 | 3584 | 7168 | 224 | 224 | dp (-) | 1425 | 1.000 | 1.002 | 1.022 (v24 owner default) | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1334 (flydsl) |
| 4096 | 4608 | 8192 | 288 | 32 | v24 rs (6) | 1442 | 1.355 | 1.016 | 1.014 (v24 rs 4) | 1.014 (v24 rs 4) | 1.014 (v24 rs 4) | 1.014 (v24 rs 4) | 1180 (torch) |
| 4096 | 4608 | 16384 | 288 | 32 | v24 rs (default) | 1515 | 1.395 | 1.019 | 1.015 (v24 rs 4) | 1.015 (v24 rs 4) | 1.015 (v24 rs 4) | 1.015 (v24 rs 4) | 1181 (torch) |
| 4096 | 5120 | 640 | 320 | 64 | v26 capped (-) | 662 | 1.004 | 1.000 | 1.004 (dp -) | 1.004 (dp -) | 1.004 (dp -) | 1.004 (dp -) | 616 (flydsl) |
| 4096 | 5120 | 1280 | 320 | 64 | v24 owner (2) | 813 | 1.007 | 1.012 | 1.023 (v24 owner 3) | 1.023 (v24 owner 3) | 1.023 (v24 owner 3) | 1.023 (v24 owner 3) | 881 (torch) |
| 4096 | 5120 | 3200 | 320 | 64 | v24 owner (3) | 1206 | 1.120 | 1.020 | 1.030 (v24 owner default) | 1.030 (v24 owner default) | 1.030 (v24 owner default) | 1.030 (v24 owner default) | 1130 (torch) |
| 4096 | 5120 | 5120 | 320 | 64 | v24 owner (3) | 1327 | 1.186 | 1.034 | 1.002 (v24 owner default) | 1.002 (v24 owner default) | 1.002 (v24 owner default) | 1.002 (v24 owner default) | 1179 (torch) |
| 4096 | 5120 | 6400 | 320 | 64 | v24 rs (default) | 1392 | 1.221 | 1.009 | 1.010 (v24 owner default) | 1.010 (v24 owner default) | 1.010 (v24 owner default) | 1.010 (v24 owner default) | 1227 (torch) |
| 4096 | 5120 | 25600 | 320 | 64 | v24 rs (default) | 1570 | 1.350 | 1.004 | 1.005 (v24 owner default) | 1.005 (v24 owner default) | 1.005 (v24 owner default) | 1.005 (v24 owner default) | 1275 (torch) |
| 4096 | 6144 | 2048 | 384 | 128 | dp (-) | 1182 | 1.000 | 1.010 | 1.025 (v24 owner default) | 1.025 (v24 owner default) | 1.025 (v24 owner default) | 1.025 (v24 owner default) | 893 (flydsl) |
| 4096 | 6144 | 3072 | 384 | 128 | v24 owner (default) | 1276 | 1.031 | 1.040 | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 930 (flydsl) |
| 4096 | 6144 | 4608 | 384 | 128 | v24 owner (2) | 1397 | 1.068 | 1.006 | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1271 (torch) |
| 4096 | 6144 | 6144 | 384 | 128 | v24 owner (2) | 1434 | 1.085 | 1.001 | 1.002 (v24 owner default) | 1.002 (v24 owner default) | 1.002 (v24 owner default) | 1.002 (v24 owner default) | 1301 (torch) |
| 4096 | 6144 | 12288 | 384 | 128 | v24 capped (-) | 1558 | 1.114 | 1.000 | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1336 (torch) |
| 4096 | 6144 | 18432 | 384 | 128 | v24 owner (2) | 1590 | 1.137 | 1.000 | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1339 (torch) |
| 4096 | 6400 | 5120 | 400 | 144 | v26 spread (-) | 1338 | 1.003 | 1.005 | 1.018 (v24 owner default) | 1.003 (dp -) | 1.003 (dp -) | 1.003 (dp -) | 1248 (flydsl) |
| 4096 | 7168 | 512 | 448 | 192 | dp (-) | 794 | 1.000 | 1.004 | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 748 (flydsl) |
| 4096 | 7168 | 1024 | 448 | 192 | dp (-) | 1062 | 1.000 | 1.004 | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 995 (flydsl) |
| 4096 | 7168 | 1792 | 448 | 192 | dp (-) | 1234 | 1.000 | 1.003 | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1156 (flydsl) |
| 4096 | 7168 | 8192 | 448 | 192 | v26 spread (-) | 1499 | 1.013 | 1.014 | 1.018 (v24 owner default) | 1.013 (dp -) | 1.013 (dp -) | 1.013 (dp -) | 1412 (flydsl) |
| 4096 | 7168 | 35840 | 448 | 192 | v26 capped (-) | 1599 | 1.051 | 1.001 | 1.060 (v24 owner default) | 1.002 (v25 spread -) | 1.002 (v25 spread -) | 1.000 (v26 spread -) | 1465 (flydsl) |
| 4096 | 8704 | 4096 | 544 | 32 | v26 capped (-) | 1409 | 1.151 | 1.001 | 1.005 (v24 rs 4) | 1.005 (v24 rs 4) | 1.005 (v24 rs 4) | 1.005 (v24 rs 4) | 1215 (torch) |
| 4096 | 9216 | 6144 | 576 | 64 | v24 owner (3) | 1467 | 1.113 | 1.003 | 1.018 (v24 owner default) | 1.018 (v24 owner default) | 1.018 (v24 owner default) | 1.018 (v24 owner default) | 1307 (torch) |
| 4096 | 9216 | 16384 | 576 | 64 | v24 rs (default) | 1584 | 1.165 | 1.002 | 1.003 (v24 owner default) | 1.003 (v24 owner default) | 1.003 (v24 owner default) | 1.003 (v24 owner default) | 1330 (torch) |
| 4096 | 10240 | 4096 | 640 | 128 | v24 capped (-) | 1439 | 1.034 | 1.000 | 1.002 (v24 owner default) | 1.002 (v24 owner default) | 1.002 (v24 owner default) | 1.002 (v24 owner default) | 1312 (flydsl) |
| 4096 | 10240 | 8192 | 640 | 128 | v24 owner (default) | 1554 | 1.063 | 1.002 | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1389 (flydsl) |
| 4096 | 12800 | 5120 | 800 | 32 | v24 owner (4) | 1488 | 1.117 | 1.003 | 1.011 (v24 rs 4) | 1.011 (v24 rs 4) | 1.011 (v24 rs 4) | 1.011 (v24 rs 4) | 1291 (asm) |
| 4096 | 13312 | 16384 | 832 | 64 | v24 rs (default) | 1601 | 1.114 | 1.002 | 1.002 (v24 owner default) | 1.002 (v24 owner default) | 1.002 (v24 owner default) | 1.002 (v24 owner default) | 1386 (flydsl) |
| 4096 | 14336 | 8192 | 896 | 128 | v24 capped (-) | 1575 | 1.052 | 1.000 | 1.003 (v24 owner default) | 1.003 (v24 owner default) | 1.003 (v24 owner default) | 1.003 (v24 owner default) | 1420 (flydsl) |
| 4096 | 26624 | 16384 | 1664 | 128 | v24 capped (-) | 1622 | 1.043 | 1.000 | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1502 (flydsl) |
| 4096 | 51200 | 5120 | 3200 | 128 | v26 capped (-) | 1548 | 1.010 | 1.001 | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1450 (asm) |
| 4096 | 129280 | 4096 | 8080 | 144 | dp (-) | 1548 | 1.000 | 1.003 | 1.004 (v24 owner default) | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1476 (asm) |
| 4352 | 512 | 4096 | 34 | 34 | v24 rs (4) | 484 | 1.904 | 1.215 | 1.148 (v24 rs 3) | 1.004 (v25 rs 5) | 1.004 (v25 rs 5) | 1.004 (v25 rs 5) | 448 (flydsl) |
| 4352 | 1024 | 4096 | 68 | 68 | v24 rs (3) | 740 | 1.476 | 1.105 | 1.074 (v24 owner default) | 1.074 (v24 owner default) | 1.074 (v24 owner default) | 1.074 (v24 owner default) | 644 (opus) |
| 4352 | 2048 | 4096 | 136 | 136 | dp (-) | 985 | 1.000 | 1.017 | 1.204 (v24 owner default) | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1047 (torch) |
| 5120 | 512 | 4096 | 40 | 40 | v24 rs (4) | 543 | 1.801 | 1.180 | 1.114 (v24 rs 3) | 1.000 (v24 rs 4) | 1.021 (v25 rs 5) | 1.000 (v24 rs 4) | 537 (flydsl) |
| 5120 | 1024 | 4096 | 80 | 80 | v24 owner (3) | 823 | 1.416 | 1.077 | 1.034 (v24 owner default) | 1.034 (v24 owner default) | 1.034 (v24 owner default) | 1.034 (v24 owner default) | 769 (opus) |
| 5120 | 2048 | 1536 | 160 | 160 | v26 capped (-) | 847 | 1.001 | 1.002 | 1.001 (dp -) | 1.001 (dp -) | 1.001 (dp -) | 1.001 (dp -) | 953 (torch) |
| 5120 | 2048 | 4096 | 160 | 160 | dp (-) | 1126 | 1.000 | 1.006 | 1.184 (v24 owner default) | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1166 (torch) |
| 5120 | 4096 | 1024 | 320 | 64 | dp (-) | 875 | 1.000 | 1.009 | 1.076 (v24 owner 2) | 1.076 (v24 owner 2) | 1.076 (v24 owner 2) | 1.076 (v24 owner 2) | 977 (torch) |
| 5120 | 4096 | 1536 | 320 | 64 | dp (-) | 975 | 1.000 | 1.001 | 1.050 (v24 owner 3) | 1.050 (v24 owner 3) | 1.050 (v24 owner 3) | 1.050 (v24 owner 3) | 1005 (torch) |
| 5120 | 4096 | 2048 | 320 | 64 | v24 owner (2) | 1091 | 1.038 | 1.006 | 1.015 (v24 owner 3) | 1.015 (v24 owner 3) | 1.015 (v24 owner 3) | 1.015 (v24 owner 3) | 1103 (torch) |
| 6144 | 512 | 4096 | 48 | 48 | v26 capped (-) | 602 | 1.666 | 1.105 | 1.196 (v24 owner default) | 1.004 (v24 rs 4) | 1.050 (v24 rs default) | 1.004 (v24 rs 4) | 682 (flydsl) |
| 6144 | 512 | 7168 | 48 | 48 | v26 capped (-) | 759 | 2.210 | 1.182 | 1.075 (v24 owner default) | 1.001 (v24 rs 4) | 1.052 (v24 rs default) | 1.001 (v24 rs 4) | 724 (flydsl) |
| 6144 | 1024 | 4096 | 96 | 96 | v24 owner (2) | 890 | 1.283 | 1.117 | 1.117 (v24 owner default) | 1.117 (v24 owner default) | 1.117 (v24 owner default) | 1.117 (v24 owner default) | 928 (opus) |
| 6144 | 1024 | 7168 | 96 | 96 | v24 owner (2) | 1022 | 1.497 | 1.083 | 1.082 (v24 owner default) | 1.082 (v24 owner default) | 1.082 (v24 owner default) | 1.082 (v24 owner default) | 967 (opus) |
| 6144 | 2048 | 4096 | 192 | 192 | v26 capped (-) | 1266 | 1.016 | 1.007 | 1.094 (v24 owner default) | 1.016 (dp -) | 1.016 (dp -) | 1.016 (dp -) | 1294 (opus) |
| 6144 | 2048 | 7168 | 192 | 192 | v26 capped (-) | 1294 | 1.005 | 1.005 | 1.055 (v24 owner default) | 1.005 (dp -) | 1.005 (dp -) | 1.005 (dp -) | 1367 (opus) |
| 6144 | 129280 | 4096 | 12120 | 88 | v24 rs (2) | 1527 | 1.002 | 1.003 | 1.005 (v24 owner default) | 1.005 (v24 owner default) | 1.005 (v24 owner default) | 1.005 (v24 owner default) | 1478 (asm) |
| 6400 | 512 | 4096 | 50 | 50 | v24 rs (4) | 619 | 1.654 | 1.108 | 1.174 (v24 owner default) | 1.061 (v24 rs 3) | 1.040 (v24 rs default) | 1.061 (v24 rs 3) | 561 (flydsl) |
| 6400 | 1024 | 4096 | 100 | 100 | v24 owner (2) | 916 | 1.287 | 1.119 | 1.111 (v24 owner default) | 1.111 (v24 owner default) | 1.111 (v24 owner default) | 1.111 (v24 owner default) | 892 (opus) |
| 6400 | 2048 | 4096 | 200 | 200 | v24 capped (-) | 1271 | 1.005 | 1.000 | 1.063 (v24 owner default) | 1.005 (dp -) | 1.005 (dp -) | 1.005 (dp -) | 1111 (torch) |
| 6912 | 512 | 4096 | 54 | 54 | v24 rs (4) | 648 | 1.611 | 1.198 | 1.150 (v24 owner default) | 1.048 (v24 rs 3) | 1.150 (v24 owner default) | 1.048 (v24 rs 3) | 649 (flydsl) |
| 6912 | 1024 | 4096 | 108 | 108 | v24 owner (2) | 968 | 1.246 | 1.092 | 1.075 (v24 owner default) | 1.075 (v24 owner default) | 1.075 (v24 owner default) | 1.075 (v24 owner default) | 938 (opus) |
| 6912 | 2048 | 4096 | 216 | 216 | dp (-) | 1316 | 1.000 | 1.002 | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1249 (torch) |
| 7168 | 512 | 4096 | 56 | 56 | v24 rs (4) | 660 | 1.581 | 1.168 | 1.134 (v24 owner default) | 1.032 (v24 rs 3) | 1.134 (v24 owner default) | 1.032 (v24 rs 3) | 582 (flydsl) |
| 7168 | 1024 | 4096 | 112 | 112 | v24 owner (2) | 987 | 1.217 | 1.045 | 1.041 (v24 owner default) | 1.041 (v24 owner default) | 1.041 (v24 owner default) | 1.041 (v24 owner default) | 1006 (opus) |
| 7168 | 2048 | 4096 | 224 | 224 | v24 capped (-) | 1355 | 1.013 | 1.000 | 1.013 (dp -) | 1.013 (dp -) | 1.013 (dp -) | 1.013 (dp -) | 1273 (torch) |
| 7424 | 512 | 4096 | 58 | 58 | v24 rs (4) | 668 | 1.544 | 1.161 | 1.102 (v24 owner default) | 1.027 (v24 rs 3) | 1.102 (v24 owner default) | 1.027 (v24 rs 3) | 649 (flydsl) |
| 7424 | 1024 | 4096 | 116 | 116 | v24 owner (2) | 1014 | 1.230 | 1.045 | 1.055 (v24 owner default) | 1.055 (v24 owner default) | 1.055 (v24 owner default) | 1.055 (v24 owner default) | 985 (opus) |
| 7424 | 2048 | 4096 | 232 | 232 | v24 capped (-) | 1374 | 1.008 | 1.000 | 1.008 (dp -) | 1.008 (dp -) | 1.008 (dp -) | 1.008 (dp -) | 1136 (torch) |
| 7680 | 512 | 4096 | 60 | 60 | v24 rs (4) | 676 | 1.524 | 1.138 | 1.105 (v24 owner default) | 1.005 (v24 rs 3) | 1.105 (v24 owner default) | 1.005 (v24 rs 3) | 572 (flydsl) |
| 7680 | 1024 | 4096 | 120 | 120 | v24 owner (2) | 1038 | 1.231 | 1.023 | 1.022 (v24 owner default) | 1.022 (v24 owner default) | 1.022 (v24 owner default) | 1.022 (v24 owner default) | 1038 (opus) |
| 7680 | 2048 | 1536 | 240 | 240 | v26 capped (-) | 1144 | 1.008 | 1.004 | 1.008 (dp -) | 1.008 (dp -) | 1.008 (dp -) | 1.008 (dp -) | 958 (torch) |
| 7680 | 2048 | 4096 | 240 | 240 | v24 capped (-) | 1401 | 1.004 | 1.000 | 1.004 (dp -) | 1.004 (dp -) | 1.004 (dp -) | 1.004 (dp -) | 1162 (torch) |
| 7680 | 4096 | 1024 | 480 | 224 | dp (-) | 1113 | 1.000 | 1.004 | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 947 (torch) |
| 7680 | 4096 | 1536 | 480 | 224 | v24 capped (-) | 1215 | 1.004 | 1.000 | 1.004 (dp -) | 1.004 (dp -) | 1.004 (dp -) | 1.004 (dp -) | 1034 (torch) |
| 7680 | 4096 | 2048 | 480 | 224 | v26 capped (-) | 1320 | 1.002 | 1.000 | 1.002 (dp -) | 1.002 (dp -) | 1.002 (dp -) | 1.002 (dp -) | 1104 (torch) |
| 7936 | 512 | 4096 | 62 | 62 | v24 rs (3) | 692 | 1.504 | 1.133 | 1.059 (v24 owner default) | 1.000 (v24 rs 3) | 1.059 (v24 owner default) | 1.000 (v24 rs 3) | 716 (flydsl) |
| 7936 | 1024 | 4096 | 124 | 124 | v24 capped (-) | 1047 | 1.169 | 1.000 | 1.016 (v24 owner default) | 1.016 (v24 owner default) | 1.016 (v24 owner default) | 1.016 (v24 owner default) | 1018 (opus) |
| 7936 | 2048 | 4096 | 248 | 248 | v26 capped (-) | 1424 | 1.002 | 1.001 | 1.002 (dp -) | 1.002 (dp -) | 1.002 (dp -) | 1.002 (dp -) | 1369 (torch) |
| 8192 | 512 | 4096 | 64 | 64 | v26 capped (-) | 703 | 1.494 | 1.115 | 1.043 (v24 owner default) | 1.006 (v24 rs 3) | 1.043 (v24 owner default) | 1.006 (v24 rs 3) | 778 (flydsl) |
| 8192 | 512 | 7168 | 64 | 64 | v24 rs (default) | 893 | 1.932 | 1.081 | 1.012 (v24 owner default) | 1.078 (v24 rs 3) | 1.012 (v24 owner default) | 1.078 (v24 rs 3) | 845 (flydsl) |
| 8192 | 512 | 8192 | 64 | 64 | v24 rs (4) | 920 | 1.892 | 1.010 | 1.010 (v24 owner default) | 1.028 (v24 rs 3) | 1.010 (v24 owner default) | 1.028 (v24 rs 3) | 863 (flydsl) |
| 8192 | 1024 | 4096 | 128 | 128 | v26 capped (-) | 1051 | 1.194 | 1.016 | 1.018 (v24 owner default) | 1.018 (v24 owner default) | 1.018 (v24 owner default) | 1.018 (v24 owner default) | 1083 (opus) |
| 8192 | 1024 | 5120 | 128 | 128 | v24 owner (default) | 1124 | 1.337 | 1.006 | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1025 (opus) |
| 8192 | 1024 | 7168 | 128 | 128 | v26 capped (-) | 1210 | 1.344 | 1.007 | 1.004 (v24 owner default) | 1.004 (v24 owner default) | 1.004 (v24 owner default) | 1.004 (v24 owner default) | 1098 (torch) |
| 8192 | 1280 | 8192 | 160 | 160 | v26 spread (-) | 1222 | 1.052 | 1.063 | 1.177 (v24 owner default) | 1.052 (dp -) | 1.052 (dp -) | 1.000 (v26 spread -) | 1228 (torch) |
| 8192 | 1536 | 1536 | 192 | 192 | v26 capped (-) | 983 | 1.003 | 1.003 | 1.003 (dp -) | 1.003 (dp -) | 1.003 (dp -) | 1.003 (dp -) | 997 (torch) |
| 8192 | 2304 | 6144 | 288 | 32 | v24 rs (6) | 1311 | 1.303 | 1.016 | 1.022 (v24 rs 4) | 1.022 (v24 rs 4) | 1.022 (v24 rs 4) | 1.022 (v24 rs 4) | 1106 (torch) |
| 8192 | 2304 | 16384 | 288 | 32 | v24 rs (6) | 1455 | 1.417 | 1.017 | 1.017 (v24 rs 4) | 1.017 (v24 rs 4) | 1.017 (v24 rs 4) | 1.017 (v24 rs 4) | 1182 (torch) |
| 8192 | 2560 | 6144 | 320 | 64 | v24 owner (default) | 1348 | 1.211 | 1.004 | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 409 (asm) |
| 8192 | 2560 | 8192 | 320 | 64 | v24 rs (4) | 1425 | 1.216 | 1.009 | 1.009 (v24 owner default) | 1.009 (v24 owner default) | 1.009 (v24 owner default) | 1.009 (v24 owner default) | 1260 (torch) |
| 8192 | 3072 | 6144 | 384 | 128 | v26 capped (-) | 1441 | 1.076 | 1.003 | 1.007 (v24 owner default) | 1.007 (v24 owner default) | 1.007 (v24 owner default) | 1.007 (v24 owner default) | 1175 (asm) |
| 8192 | 3584 | 7168 | 448 | 192 | v26 spread (-) | 1461 | 1.006 | 1.010 | 1.017 (v24 owner default) | 1.006 (dp -) | 1.006 (dp -) | 1.006 (dp -) | 1388 (flydsl) |
| 8192 | 4608 | 8192 | 576 | 64 | v26 capped (-) | 1538 | 1.152 | 1.002 | 1.016 (v24 owner default) | 1.016 (v24 owner default) | 1.016 (v24 owner default) | 1.016 (v24 owner default) | 1347 (torch) |
| 8192 | 4608 | 16384 | 576 | 64 | v24 rs (4) | 1542 | 1.160 | 1.003 | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1356 (torch) |
| 8192 | 5120 | 640 | 640 | 128 | dp (-) | 844 | 1.000 | 1.005 | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 867 (torch) |
| 8192 | 5120 | 1280 | 640 | 128 | v24 capped (-) | 1074 | 1.010 | 1.000 | 1.010 (dp -) | 1.010 (dp -) | 1.010 (dp -) | 1.010 (dp -) | 1004 (flydsl) |
| 8192 | 5120 | 3200 | 640 | 128 | v24 owner (2) | 1396 | 1.035 | 1.035 | 1.004 (v24 owner default) | 1.004 (v24 owner default) | 1.004 (v24 owner default) | 1.004 (v24 owner default) | 1267 (asm) |
| 8192 | 5120 | 5120 | 640 | 128 | v24 owner (2) | 1472 | 1.048 | 1.004 | 1.006 (v24 owner default) | 1.006 (v24 owner default) | 1.006 (v24 owner default) | 1.006 (v24 owner default) | 1306 (torch) |
| 8192 | 5120 | 6400 | 640 | 128 | v24 owner (default) | 1507 | 1.056 | 1.001 | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1355 (flydsl) |
| 8192 | 5120 | 25600 | 640 | 128 | v24 capped (-) | 1595 | 1.109 | 1.000 | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1400 (asm) |
| 8192 | 6400 | 5120 | 800 | 32 | v26 capped (-) | 1465 | 1.103 | 1.007 | 1.003 (v24 rs 4) | 1.003 (v24 rs 4) | 1.003 (v24 rs 4) | 1.003 (v24 rs 4) | 1256 (flydsl) |
| 8192 | 7168 | 1024 | 896 | 128 | dp (-) | 1119 | 1.000 | 1.006 | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1049 (asm) |
| 8192 | 7168 | 1792 | 896 | 128 | dp (-) | 1273 | 1.000 | 1.001 | 1.007 (v24 owner default) | 1.007 (v24 owner default) | 1.007 (v24 owner default) | 1.007 (v24 owner default) | 1200 (asm) |
| 8192 | 7168 | 8192 | 896 | 128 | v24 capped (-) | 1582 | 1.054 | 1.000 | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1427 (asm) |
| 8192 | 7168 | 35840 | 896 | 128 | v26 capped (-) | 1615 | 1.090 | 1.001 | 1.002 (v24 owner default) | 1.002 (v24 owner default) | 1.002 (v24 owner default) | 1.002 (v24 owner default) | 1430 (asm) |
| 8192 | 8704 | 4096 | 1088 | 64 | v24 owner (3) | 1502 | 1.063 | 1.007 | 1.019 (v24 owner default) | 1.019 (v24 owner default) | 1.019 (v24 owner default) | 1.019 (v24 owner default) | 1343 (asm) |
| 8192 | 9216 | 6144 | 1152 | 128 | v24 owner (default) | 1555 | 1.032 | 1.000 | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1425 (asm) |
| 8192 | 9216 | 16384 | 1152 | 128 | v26 capped (-) | 1579 | 1.064 | 1.001 | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1409 (flydsl) |
| 8192 | 12800 | 5120 | 1600 | 64 | v24 rs (4) | 1531 | 1.030 | 1.009 | 1.003 (v24 owner default) | 1.003 (v24 owner default) | 1.003 (v24 owner default) | 1.003 (v24 owner default) | 1388 (asm) |
| 8192 | 13312 | 16384 | 1664 | 128 | v24 capped (-) | 1591 | 1.046 | 1.000 | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1467 (flydsl) |
| 8192 | 129280 | 4096 | 16160 | 32 | v24 rs (6) | 1568 | 1.006 | 1.002 | 1.001 (v24 rs 4) | 1.001 (v24 rs 4) | 1.001 (v24 rs 4) | 1.001 (v24 rs 4) | 1472 (flydsl) |
| 10240 | 512 | 4096 | 80 | 80 | v24 owner (3) | 794 | 1.387 | 1.057 | 1.020 (v24 owner default) | 1.020 (v24 owner default) | 1.020 (v24 owner default) | 1.020 (v24 owner default) | 856 (torch) |
| 10240 | 512 | 7168 | 80 | 80 | v24 owner (3) | 977 | 1.724 | 1.012 | 1.014 (v24 owner default) | 1.014 (v24 owner default) | 1.014 (v24 owner default) | 1.014 (v24 owner default) | 826 (torch) |
| 10240 | 1024 | 4096 | 160 | 160 | v26 capped (-) | 1109 | 1.004 | 1.003 | 1.160 (v24 owner default) | 1.004 (dp -) | 1.004 (dp -) | 1.004 (dp -) | 1154 (torch) |
| 10240 | 1024 | 7168 | 160 | 160 | v26 capped (-) | 1172 | 1.054 | 1.049 | 1.164 (v24 owner default) | 1.054 (dp -) | 1.054 (dp -) | 1.004 (v26 spread -) | 1212 (torch) |
| 10240 | 2048 | 4096 | 320 | 64 | v24 owner (3) | 1266 | 1.129 | 1.011 | 1.015 (v24 owner default) | 1.015 (v24 owner default) | 1.015 (v24 owner default) | 1.015 (v24 owner default) | 1208 (torch) |
| 10240 | 2048 | 7168 | 320 | 64 | v24 rs (default) | 1388 | 1.201 | 1.003 | 1.005 (v24 owner default) | 1.005 (v24 owner default) | 1.005 (v24 owner default) | 1.005 (v24 owner default) | 1257 (torch) |
| 10240 | 7168 | 768 | 1120 | 96 | v24 capped (-) | 1080 | 1.006 | 1.000 | 1.006 (dp -) | 1.006 (dp -) | 1.006 (dp -) | 1.006 (dp -) | 390 (opus) |
| 10240 | 7168 | 1536 | 1120 | 96 | v26 capped (-) | 1287 | 1.005 | 1.002 | 1.033 (v24 owner default) | 1.033 (v24 owner default) | 1.033 (v24 owner default) | 1.033 (v24 owner default) | 1153 (asm) |
| 10240 | 7168 | 3584 | 1120 | 96 | v24 owner (default) | 1493 | 1.035 | 1.013 | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 786 (flydsl) |
| 10240 | 7168 | 4224 | 1120 | 96 | v24 owner (2) | 1517 | 1.041 | 1.009 | 1.007 (v24 owner default) | 1.007 (v24 owner default) | 1.007 (v24 owner default) | 1.007 (v24 owner default) | 915 (opus) |
| 10240 | 8448 | 7168 | 1320 | 40 | v24 rs (4) | 1527 | 1.051 | 1.003 | 1.010 (v24 rs 3) | 1.010 (v24 rs 3) | 1.010 (v24 rs 3) | 1.010 (v24 rs 3) | 1401 (asm) |
| 10240 | 20480 | 7168 | 3200 | 128 | v26 capped (-) | 1536 | 1.015 | 1.001 | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1509 (asm) |
| 10240 | 129280 | 4096 | 20200 | 232 | dp (-) | 1540 | 1.000 | 1.003 | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1491 (asm) |
| 12288 | 512 | 4096 | 96 | 96 | v24 owner (2) | 859 | 1.265 | 1.097 | 1.098 (v24 owner default) | 1.098 (v24 owner default) | 1.098 (v24 owner default) | 1.098 (v24 owner default) | 896 (opus) |
| 12288 | 512 | 7168 | 96 | 96 | v24 owner (2) | 982 | 1.465 | 1.080 | 1.081 (v24 owner default) | 1.081 (v24 owner default) | 1.081 (v24 owner default) | 1.081 (v24 owner default) | 931 (opus) |
| 12288 | 1024 | 4096 | 192 | 192 | v26 capped (-) | 1235 | 1.003 | 1.009 | 1.072 (v24 owner default) | 1.003 (dp -) | 1.003 (dp -) | 1.003 (dp -) | 1276 (opus) |
| 12288 | 1024 | 7168 | 192 | 192 | v26 capped (-) | 1263 | 1.004 | 1.002 | 1.042 (v24 owner default) | 1.004 (dp -) | 1.004 (dp -) | 1.004 (dp -) | 1344 (opus) |
| 12288 | 2048 | 4096 | 384 | 128 | v24 capped (-) | 1352 | 1.046 | 1.000 | 1.004 (v24 owner default) | 1.004 (v24 owner default) | 1.004 (v24 owner default) | 1.004 (v24 owner default) | 1315 (opus) |
| 12288 | 2048 | 7168 | 384 | 128 | v24 owner (default) | 1456 | 1.094 | 1.003 | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1388 (opus) |
| 12288 | 129280 | 4096 | 24240 | 176 | dp (-) | 1559 | 1.000 | 1.004 | 1.005 (v24 owner default) | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1494 (asm) |
| 14336 | 512 | 4096 | 112 | 112 | v24 owner (2) | 968 | 1.238 | 1.032 | 1.035 (v24 owner default) | 1.035 (v24 owner default) | 1.035 (v24 owner default) | 1.035 (v24 owner default) | 967 (opus) |
| 14336 | 512 | 7168 | 112 | 112 | v24 owner (2) | 1091 | 1.406 | 1.017 | 1.011 (v24 owner default) | 1.011 (v24 owner default) | 1.011 (v24 owner default) | 1.011 (v24 owner default) | 1001 (opus) |
| 14336 | 1024 | 4096 | 224 | 224 | dp (-) | 1313 | 1.000 | 1.006 | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1244 (torch) |
| 14336 | 1024 | 7168 | 224 | 224 | v24 capped (-) | 1366 | 1.006 | 1.000 | 1.017 (v24 owner default) | 1.006 (dp -) | 1.006 (dp -) | 1.006 (dp -) | 1325 (torch) |
| 14336 | 2048 | 4096 | 448 | 192 | dp (-) | 1409 | 1.000 | 1.005 | 1.034 (v24 owner default) | 1.000 (dp -) | 1.000 (dp -) | 1.000 (dp -) | 1316 (flydsl) |
| 14336 | 2048 | 7168 | 448 | 192 | v26 capped (-) | 1485 | 1.002 | 1.003 | 1.010 (v24 owner default) | 1.002 (dp -) | 1.002 (dp -) | 1.002 (dp -) | 1371 (flydsl) |
| 14336 | 129280 | 4096 | 28280 | 120 | dp (-) | 1537 | 1.000 | 1.004 | 1.007 (v24 owner default) | 1.007 (v24 owner default) | 1.007 (v24 owner default) | 1.007 (v24 owner default) | 1489 (asm) |
| 16384 | 512 | 4096 | 128 | 128 | v24 capped (-) | 1015 | 1.191 | 1.000 | 1.018 (v24 owner default) | 1.018 (v24 owner default) | 1.018 (v24 owner default) | 1.018 (v24 owner default) | 1023 (opus) |
| 16384 | 512 | 7168 | 128 | 128 | v24 owner (default) | 1184 | 1.354 | 1.004 | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1080 (opus) |
| 16384 | 512 | 8192 | 128 | 128 | v24 owner (2) | 1242 | 1.227 | 1.001 | 1.008 (v24 owner default) | 1.008 (v24 owner default) | 1.008 (v24 owner default) | 1.008 (v24 owner default) | 1065 (torch) |
| 16384 | 1280 | 8192 | 320 | 64 | v24 rs (4) | 1425 | 1.217 | 1.005 | 1.006 (v24 owner default) | 1.006 (v24 owner default) | 1.006 (v24 owner default) | 1.006 (v24 owner default) | 1252 (torch) |
| 16384 | 1536 | 1536 | 384 | 128 | v26 capped (-) | 1071 | 1.003 | 1.009 | 1.071 (v24 owner default) | 1.071 (v24 owner default) | 1.071 (v24 owner default) | 1.071 (v24 owner default) | 1064 (torch) |
| 16384 | 2304 | 6144 | 576 | 64 | v26 capped (-) | 1460 | 1.111 | 1.002 | 1.017 (v24 owner default) | 1.017 (v24 owner default) | 1.017 (v24 owner default) | 1.017 (v24 owner default) | 975 (flydsl) |
| 16384 | 2304 | 16384 | 576 | 64 | v24 rs (4) | 1517 | 1.142 | 1.003 | 1.003 (v24 owner default) | 1.003 (v24 owner default) | 1.003 (v24 owner default) | 1.003 (v24 owner default) | 1348 (torch) |
| 16384 | 2560 | 6144 | 640 | 128 | v26 capped (-) | 1498 | 1.042 | 1.002 | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1314 (flydsl) |
| 16384 | 2560 | 8192 | 640 | 128 | v24 owner (2) | 1518 | 1.070 | 1.003 | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1345 (flydsl) |
| 16384 | 3584 | 7168 | 896 | 128 | v24 owner (default) | 1538 | 1.045 | 1.001 | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1386 (flydsl) |
| 16384 | 4608 | 8192 | 1152 | 128 | v24 owner (default) | 1561 | 1.033 | 1.001 | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1432 (flydsl) |
| 16384 | 4608 | 16384 | 1152 | 128 | v24 capped (-) | 1568 | 1.056 | 1.000 | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1437 (flydsl) |
| 16384 | 6400 | 5120 | 1600 | 64 | v24 owner (3) | 1513 | 1.020 | 1.007 | 1.002 (v24 owner default) | 1.002 (v24 owner default) | 1.002 (v24 owner default) | 1.002 (v24 owner default) | 1370 (flydsl) |
| 16384 | 12800 | 5120 | 3200 | 128 | v24 owner (default) | 1557 | 1.016 | 1.004 | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1459 (torch) |
| 16384 | 129280 | 4096 | 32320 | 64 | v24 rs (2) | 1568 | 1.001 | 1.001 | 1.002 (v24 owner default) | 1.002 (v24 owner default) | 1.002 (v24 owner default) | 1.002 (v24 owner default) | 1490 (asm) |
| 32768 | 1280 | 8192 | 640 | 128 | v24 capped (-) | 1479 | 1.063 | 1.000 | 1.002 (v24 owner default) | 1.002 (v24 owner default) | 1.002 (v24 owner default) | 1.002 (v24 owner default) | 1315 (flydsl) |
| 32768 | 2304 | 6144 | 1152 | 128 | v24 owner (2) | 1500 | 1.036 | 1.001 | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 1.001 (v24 owner default) | 652 (flydsl) |
| 32768 | 2304 | 16384 | 1152 | 128 | v24 owner (default) | 1546 | 1.053 | 1.002 | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1414 (flydsl) |
| 32768 | 6400 | 5120 | 3200 | 128 | v24 owner (2) | 1545 | 1.011 | 1.000 | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1.000 (v24 owner default) | 1465 (torch) |
