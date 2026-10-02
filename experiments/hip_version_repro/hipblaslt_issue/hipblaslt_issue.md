## gfx950 BF16 TN: hipBLASLt 1.4.1 (ROCm 10.0.0) and 1.5.0 heuristic selects slower solutions than 1.2.2 (ROCm 7.2.4)

We observe a performance regression in the default heuristic solution selection for BF16 TN GEMM on gfx950 (MI355X) between hipBLASLt 1.2.2 (ROCm 7.2.4) and hipBLASLt 1.4.1 (ROCm 10.0.0) / 1.5.0 (develop). Could you confirm whether this is a known regression, the result of an incorrect configuration on our side (for example the workspace size, `HIPBLASLT_MATMUL_DESC_*` attributes or environment settings), or expected behaviour?

For BF16 GEMM, TN, fp32 compute (`Cijk_Alik_Bljk_BBS`), the newer libraries replace MT256x256x64 SK3 on large compute-bound problems with the non-power-of-2 subtile macro-tiles added in #8604 (MT256x320, MT224x384, MT288x288, MT192x448). The only subtile reject is `K < 512` (#8626), so nothing filters them at larger K, and they are 5-13% slower. 1.5.0 also regresses many small problems (fewer than 256 output tiles) by up to 62% where 1.4.1 does not.

Example: PyTorch 32768×8192×1024 (hipBLASLt m=8192, n=32768, k=1024; Llama 70B): 1.2.2 selects 256x256 MI16 SK3 at 1262 TFLOPS; 1.4.1 selects 256x320 MI16 SK5 at 1111 (-12.0%); 1.5.0 selects 256x320 MI16 SK5 at 1108 (-12.2%). An MT256x256x64 solution is still in both newer libraries and runs at 1287 / 1288.

**Environment:**

- MI355X, one GPU (PCI bus 21); every version measured on the same card, one after the other.
- hipBLASLt 1.2.2 (`dabb6df2b9`, ROCm 7.2.4 debs), 1.4.1 (`8d1ae90e`, TheRock ROCm 10.0.0 gfx950 tarball), 1.5.0 (`6054c511`, TheRock nightly 10.2.0a20260929).
- Standalone hipBLASLt benchmark (attached, no framework): `hipblasLtMatmulAlgoGetHeuristic` top-1, 128 MiB workspace (the `hipblaslt-bench` default). A/B BF16, C=D BF16 (fp32 where noted), alpha 1, beta 0, no bias. hipEvent timing, 20 warm-up + 100 timed calls, median of 3.
- Cross-checked with `hipblaslt-bench` from TheRock's tests tarballs for 1.4.1 and 1.5.0 (see Reproduce).
- Problems: 630 BF16 GEMMs from AITER's per-model tuning lists.

**Summary** (at least 5% slower than 1.2.2 counts as a regression):

- 1.4.1 (ROCm 10.0.0) regresses 66 of 630 problems, 1.5.0 regresses 97; 107 regress in at least one.
- **Large problems (256 or more output tiles): 33**, 31 of them where 1.2.2 selects MT256x256 SK3. 1.4.1 and 1.5.0 regress them by a similar amount (median -6.5% / -5.7%). Llama 70B, Llama 405B, Kimi-K3, Qwen 32B, MiniMax-M3, DeepSeek-V4.
- The other 2 large problems keep 1.2.2's macro-tile but select a different kernel: m=5120 n=8192 k=1280 (Qwen 32B): 256x224 MI16 SK3 1113 → 256x224 MI32 SK5 944 / 256x224 MI32 SK5 947; m=5120 n=8192 k=640 (Qwen 32B): 256x224 MI16 SK3 906 → 256x224 MI32 SK5 791 / 256x224 MI32 SK5 792. An MT256x256 solution in the same libraries is within 5% of 1.2.2 on 2 of 2 (1.4.1) and 2 of 2 (1.5.0).
- **Small problems (fewer than 256 output tiles): 74**; 40 regress in 1.5.0 only. On 40 of those, 1.5.0 selects the same macro-tile as 1.4.1 (40 with the same MFMA as well), so they are not a tile-ranking change. DeepSeek-V4, Kimi / Kimi-K2 / Kimi-K3, MiniMax-M3, Llama 405B.
- **The MT256x256 solutions are still present.** On the 33 regressed problems where 1.2.2 selects MT256x256, the best MT256x256x64 solution is within 5% of 1.2.2 on 33 in 1.4.1 and 31 in 1.5.0, and 1.5.0 with `ANALYTICAL_GEMM_PICK=256x256x64` on 31. So this is a selection (ranking) regression, not a missing kernel. `ANALYTICAL_GEMM_PICK` was added in #7228 (2026-08-06) and is not in 1.4.1; on the released version the only runtime workaround is a per-problem `HIPBLASLT_TUNING_OVERRIDE_FILE`.
- Not all changes are regressions: 1.5.0 is at least 5% faster than 1.2.2 on 210 of 630 problems, mostly with fewer than 256 output tiles, where smaller or non-square tiles raise CU occupancy. A fix should be scoped (for example subtile rejects for saturated grids), not a revert.

### Large problems (256 or more output tiles), 33

- **m, n, k:** hipBLASLt column-major problem size, `opA=T, opB=N`, `lda=ldb=k`, `ldc=ldd=m`. For PyTorch `F.linear(x[M,K], w[N,K])`: m = N, n = M, k = K.
- **Model(s):** models whose AITER tuning list (`aiter/configs/model_configs`) contains this exact GEMM; *layer* means only the weight shape matches (same m and k, different token count n).
- **Tiles:** output tiles at 256x256, (m/256)·(n/256); the GPU has 256 CUs.
- **1.2.2 / 1.4.1 (ROCm 10.0.0) / 1.5.0:** heuristic top-1 (`hipblasLtMatmulAlgoGetHeuristic`, 128 MiB workspace), shown as `macro-tile MFMA stream-K: TFLOPS (change vs 1.2.2)`. MI16 / MI32 = MI16x16x1 / MI32x32x1.
- **Best MT256x256:** the fastest of all MT256x256x64 solutions returned by `hipblaslt_ext::getAllAlgos`, each timed individually. Shows that the kernel still exists in that library and what it achieves.
- **1.5.0 + `ANALYTICAL_GEMM_PICK=256x256x64`:** heuristic top-1 with that environment variable set, which restricts Origami's ranking to MT256x256x64 configs. It exists only in develop (#7228), not in 1.4.1.

| Model(s) | m, n, k | Tiles | 1.2.2 | 1.4.1 (ROCm 10.0.0) | 1.5.0 | Best MT256x256 in 1.4.1 (ROCm 10.0.0) | Best MT256x256 in 1.5.0 | 1.5.0 + `ANALYTICAL_GEMM_PICK=256x256x64` |
|---|---|---:|---|---|---|---|---|---|
| Qwen 32B | 5120, 8192, 1280 | 640 | 256x224 MI16 SK3: 1113 | 256x224 MI32 SK5: 944 (-15.1%) | 256x224 MI32 SK5: 947 (-14.9%) | 1123 (+0.9%) | 1122 (+0.9%) | 972 (-12.7%) |
| Qwen 32B | 5120, 8192, 640 | 640 | 256x224 MI16 SK3: 906 | 256x224 MI32 SK5: 791 (-12.7%) | 256x224 MI32 SK5: 792 (-12.6%) | 890 (-1.8%) | 894 (-1.4%) | 734 (-19.0%) |
| Kimi-K3 | 7168, 32768, 1024 | 3584 | 256x256 MI16 SK3: 1265 | 224x384 MI16 SK5: 1106 (-12.6%) | 224x384 MI16 SK5: 1106 (-12.5%) | 1284 (+1.6%) | 1285 (+1.6%) | 1258 (-0.5%) |
| Llama 70B | 8192, 32768, 1024 | 4096 | 256x256 MI16 SK3: 1262 | 256x320 MI16 SK5: 1111 (-12.0%) | 256x320 MI16 SK5: 1108 (-12.2%) | 1287 (+2.0%) | 1288 (+2.1%) | 1259 (-0.2%) |
| MiniMax-M3 (EAGLE) | 6144, 16384, 12288 | 1536 | 256x256 MI16 SK3: 1590 | 256x320 MI16 SK5: 1456 (-8.4%) | 256x320 MI16 SK5: 1451 (-8.8%) | 1602 (+0.8%) | 1592 (+0.1%) | 1585 (-0.3%) |
| MiniMax-M3 (EAGLE) | 9216, 16384, 6144 | 2304 | 256x256 MI16 SK3: 1529 | 192x448 MI16 SK5: 1398 (-8.6%) | 192x448 MI16 SK5: 1408 (-7.9%) | 1544 (+1.0%) | 1548 (+1.3%) | 1472 (-3.7%) |
| Llama 70B | 14336, 8192, 8192 | 1792 | 256x256 MI16 SK3: 1560 | 256x320 MI16 SK5: 1432 (-8.2%) | 256x320 MI16 SK5: 1428 (-8.5%) | 1564 (+0.3%) | 1548 (-0.8%) | 1522 (-2.4%) |
| Kimi-K3 | 3584, 32768, 7168 | 1792 | 256x256 MI16 SK3: 1528 | 256x320 MI16 SK5: 1409 (-7.8%) | 256x320 MI16 SK5: 1402 (-8.2%) | 1565 (+2.4%) | 1522 (-0.4%) | 1496 (-2.1%) |
| MiniMax-M3 (EAGLE) | 6144, 32768, 12288 | 3072 | 256x256 MI16 SK3: 1591 | 256x320 MI16 SK5: 1464 (-8.0%) | 256x320 MI16 SK5: 1472 (-7.5%) | 1600 (+0.6%) | 1600 (+0.5%) | 1597 (+0.3%) |
| Llama 70B | 7168, 16384, 8192 | 1792 | 256x256 MI16 SK3: 1564 | 256x320 MI16 SK5: 1441 (-7.9%) | 256x320 MI16 SK5: 1441 (-7.9%) | 1567 (+0.2%) | 1562 (-0.1%) | 1570 (+0.4%) |
| Llama 70B | 10240, 16384, 8192 | 2560 | 256x256 MI16 SK3: 1567 | 224x384 MI16 SK5: 1452 (-7.4%) | 224x384 MI16 SK5: 1446 (-7.7%) | 1579 (+0.7%) | 1581 (+0.9%) | 1572 (+0.3%) |
| MiniMax-M3 (EAGLE); layer: GLM-5 | 6144, 32768, 2048 | 3072 | 256x256 MI16 SK3: 1351 | 256x320 MI16 SK5: 1252 (-7.4%) | 256x320 MI16 SK5: 1268 (-6.1%) | 1419 (+5.0%) | 1413 (+4.6%) | 1367 (+1.2%) |
| Llama 405B | 16384, 32768, 6656 | 8192 | 256x256 MI16 SK3: 1561 | 256x320 MI16 SK5: 1447 (-7.3%) | 256x320 MI16 SK5: 1445 (-7.4%) | 1570 (+0.6%) | 1571 (+0.6%) | 1559 (-0.1%) |
| Llama 70B | 10240, 8192, 8192 | 1280 | 256x256 MI16 SK3: 1570 | 224x384 MI16 SK5: 1474 (-6.1%) | 224x384 MI16 SK5: 1463 (-6.8%) | 1520 (-3.2%) | 1513 (-3.6%) | 1479 (-5.8%) |
| Qwen 32B | 12800, 32768, 5120 | 6400 | 256x256 MI16 SK3: 1540 | 288x288 MI16 SK5: 1437 (-6.7%) | 288x288 MI16 SK5: 1452 (-5.7%) | 1551 (+0.7%) | 1550 (+0.6%) | 1512 (-1.8%) |
| MiniMax-M3 (EAGLE); layer: FLUX.2, GLM-5 | 6144, 32768, 6144 | 3072 | 256x256 MI16 SK3: 1553 | 256x320 MI16 SK5: 1449 (-6.7%) | 256x320 MI16 SK5: 1464 (-5.7%) | 1562 (+0.6%) | 1562 (+0.5%) | 1515 (-2.5%) |
| MiniMax-M3 (EAGLE); layer: GLM-5 | 6144, 16384, 2048 | 1536 | 256x256 MI16 SK3: 1333 | 256x320 MI16 SK5: 1245 (-6.6%) | 256x320 MI16 SK5: 1261 (-5.4%) | 1386 (+4.0%) | 1385 (+3.9%) | 1342 (+0.7%) |
| Kimi-K3 | 7168, 32768, 1792 | 3584 | 256x256 MI16 SK3: 1339 | 224x384 MI16 SK5: 1252 (-6.5%) | 224x384 MI16 SK5: 1266 (-5.4%) | 1392 (+4.0%) | 1386 (+3.5%) | 1350 (+0.8%) |
| Llama 70B | 7168, 32768, 8192 | 3584 | 256x256 MI16 SK3: 1568 | 224x384 MI16 SK5: 1478 (-5.7%) | 224x384 MI16 SK5: 1475 (-5.9%) | 1583 (+0.9%) | 1581 (+0.8%) | 1572 (+0.2%) |
| Kimi-K3 | 7168, 16384, 35840 | 1792 | 256x256 MI16 SK3: 1524 | 288x288 MI16 SK5: 1446 (-5.1%) | 288x288 MI16 SK5: 1435 (-5.9%) | 1578 (+3.5%) | 1582 (+3.8%) | 1533 (+0.6%) |
| MiniMax-M3 (EAGLE) | 9216, 32768, 6144 | 4608 | 256x256 MI16 SK3: 1551 | 288x288 MI16 SK5: 1466 (-5.5%) | 288x288 MI16 SK5: 1464 (-5.6%) | 1564 (+0.9%) | 1563 (+0.8%) | 1564 (+0.9%) |
| Llama 70B | 14336, 16384, 8192 | 3584 | 256x256 MI16 SK3: 1549 | 224x384 MI16 SK5: 1465 (-5.4%) | 224x384 MI16 SK5: 1462 (-5.6%) | 1581 (+2.1%) | 1580 (+2.1%) | 1569 (+1.3%) |
| Llama 405B | 16384, 16384, 2048 | 4096 | 256x256 MI16 SK3: 1352 | 256x320 MI16 SK5: 1278 (-5.5%) | 256x320 MI16 SK5: 1300 (-3.9%) | 1424 (+5.3%) | 1419 (+4.9%) | 1366 (+1.0%) |
| Llama 70B | 8192, 32768, 2048 | 4096 | 256x256 MI16 SK3: 1359 | 256x320 MI16 SK5: 1285 (-5.5%) | 256x320 MI16 SK5: 1305 (-4.0%) | 1424 (+4.8%) | 1424 (+4.8%) | 1370 (+0.8%) |
| Llama 70B | 8192, 32768, 7168 | 4096 | 256x256 MI16 SK3: 1562 | 256x320 MI16 SK5: 1482 (-5.1%) | 256x320 MI16 SK5: 1477 (-5.4%) | 1576 (+0.9%) | 1574 (+0.8%) | 1560 (-0.1%) |
| MiniMax-M3 (EAGLE); layer: GLM-5 | 6144, 32768, 3072 | 3072 | 256x256 MI16 SK3: 1396 | 256x320 MI16 SK5: 1323 (-5.3%) | 256x320 MI16 SK5: 1348 (-3.4%) | 1485 (+6.3%) | 1478 (+5.8%) | 1419 (+1.6%) |
| DeepSeek-V4 | 129280, 1536, 4096 | 3030 | 256x256 MI16 SK3: 1410 | 256x320 MI16 SK5: 1336 (-5.3%) | 256x320 MI16 SK5: 1356 (-3.8%) | 1439 (+2.0%) | 1424 (+1.0%) | 1408 (-0.2%) |
| DeepSeek-V4 | 129280, 2048, 4096 | 4040 | 256x256 MI16 SK3: 1407 | 272x256 MI16 SK5: 1333 (-5.3%) | 272x256 MI16 SK5: 1365 (-3.0%) | 1451 (+3.1%) | 1456 (+3.5%) | 1417 (+0.7%) |
| Llama 405B | 16384, 16384, 6656 | 4096 | 256x256 MI16 SK3: 1557 | 256x320 MI16 SK5: 1474 (-5.3%) | 256x320 MI16 SK5: 1494 (-4.0%) | 1572 (+0.9%) | 1572 (+0.9%) | 1531 (-1.7%) |
| DeepSeek-V4 | 129280, 768, 4096 | 1515 | 256x256 MI16 SK3: 1375 | 176x384 MI16 SK5: 1303 (-5.2%) | 176x384 MI16 SK5: 1305 (-5.1%) | 1385 (+0.7%) | 1372 (-0.2%) | 1369 (-0.4%) |
| Llama 405B | 16384, 32768, 4096 | 8192 | 256x256 MI16 SK3: 1500 | 256x320 MI16 SK5: 1422 (-5.2%) | 256x320 MI16 SK5: 1447 (-3.5%) | 1542 (+2.9%) | 1540 (+2.7%) | 1446 (-3.6%) |
| Llama 405B | 16384, 32768, 2048 | 8192 | 256x256 MI16 SK3: 1368 | 256x320 MI16 SK5: 1300 (-5.0%) | 256x320 MI16 SK5: 1322 (-3.4%) | 1445 (+5.6%) | 1447 (+5.8%) | 1381 (+0.9%) |
| Qwen 32B | 12800, 16384, 5120 | 3200 | 256x256 MI16 SK3: 1488 | 224x384 MI16 SK5: 1417 (-4.8%) | 224x384 MI16 SK5: 1413 (-5.0%) | 1525 (+2.5%) | 1517 (+2.0%) | 1452 (-2.4%) |

### Small problems (fewer than 256 output tiles), 74

Same columns as above; the number in brackets is the TFLOPS change vs 1.2.2.

<details><summary>Table</summary>

| Model(s) | m, n, k | Tiles | 1.2.2 | 1.4.1 (ROCm 10.0.0) | 1.5.0 |
|---|---|---:|---|---|---|
| DeepSeek-V4, Kimi; layer: Kimi-K2 | 1024, 512, 7168 | 8 | 128x128 MI32 SK3: 424 | 128x128 MI32 SK5: 436 (+2.8%) | 128x128 MI32 SK5: 160 (-62.3%) |
| MiniMax-M3 (MXFP4) | 1536, 512, 6144 | 12 | 192x64 MI16 SK3: 417 | 192x128 MI32 SK5: 420 (+0.8%) | 192x128 MI32 SK5: 208 (-50.0%) |
| Kimi-K3 | 1536, 512, 7168 | 12 | 192x64 MI16 SK3: 453 | 192x128 MI32 SK5: 465 (+2.8%) | 192x128 MI32 SK5: 236 (-47.9%) |
| DeepSeek-V4; layer: Kimi, Kimi-K2 | 1024, 1280, 7168 | 20 | 256x80 MI16 SK3: 583 | 256x160 MI16 SK5: 536 (-8.1%) | 256x160 MI16 SK5: 309 (-47.0%) |
| DeepSeek-V4; layer: Kimi, Kimi-K2 | 1024, 1792, 7168 | 28 | 256x128 MI16 SK3: 636 | 256x224 MI32 SK5: 547 (-14.0%) | 256x224 MI32 SK5: 348 (-45.3%) |
| qwen3_8_2p4t_a95b | 512, 2048, 8192 | 16 | 256x128 MI16 SK3: 537 | 256x128 MI16 SK5: 551 (+2.6%) | 256x128 MI16 SK5: 310 (-42.3%) |
| Llama 405B | 4608, 512, 16384 | 36 | 160x128 MI16 SK3: 945 | 112x512 MI16 SK5: 911 (-3.7%) | 112x512 MI16 SK5: 565 (-40.2%) |
| DeepSeek-V4; layer: Kimi, Kimi-K2 | 1024, 768, 7168 | 12 | 256x96 MI16 SK3: 459 | 128x192 MI16 SK5: 472 (+2.8%) | 128x192 MI16 SK5: 282 (-38.6%) |
| MiniMax-M3 (MXFP4) | 1280, 512, 6144 | 10 | 160x64 MI16 SK3: 372 | 160x128 MI16 SK5: 360 (-3.4%) | 160x128 MI16 SK5: 229 (-38.4%) |
| DeepSeek-V4 | 512, 1536, 7168 | 12 | 256x96 MI16 SK3: 449 | 128x192 MI16 SK5: 482 (+7.5%) | 128x192 MI16 SK5: 284 (-36.8%) |
| layer: DeepSeek-V4, GLM-5.3-Flash, Qwen3.5-397B | 512, 3072, 4096 | 24 | 64x96 MI16 SK3: 574 | 256x96 MI16 SK5: 464 (-19.2%) | 256x96 MI16 SK5: 375 (-34.7%) |
| Llama 70B | 1280, 512, 8192 | 10 | 160x64 MI16 SK3: 489 | 160x128 MI16 SK5: 494 (+1.1%) | 160x128 MI16 SK5: 321 (-34.2%) |
| DeepSeek-V4 | 512, 1792, 7168 | 14 | 256x112 MI16 SK3: 498 | 128x224 MI16 SK5: 536 (+7.6%) | 128x224 MI16 SK5: 329 (-33.9%) |
| qwen3_8_2p4t_a95b | 512, 8192, 8192 | 64 | 128x128 MI16 SK3: 918 | 256x256 MI16 SK5: 815 (-11.2%) | 256x256 MI16 SK5: 619 (-32.6%) |
| DeepSeek-V4 | 512, 8192, 7168 | 64 | 128x128 MI16 SK3: 876 | 256x256 MI16 SK5: 764 (-12.8%) | 256x256 MI16 SK5: 617 (-29.5%) |
| MiniMax-M3 (EAGLE) | 2304, 512, 6144 | 18 | 192x128 MI16 SK3: 407 | 144x256 MI16 SK5: 490 (+20.4%) | 144x256 MI16 SK5: 289 (-29.1%) |
| DeepSeek-V4 | 512, 1280, 7168 | 10 | 256x80 MI16 SK3: 414 | 128x160 MI16 SK5: 463 (+11.8%) | 128x160 MI16 SK5: 296 (-28.4%) |
| Llama 405B | 2304, 1024, 16384 | 36 | 256x256 MI16 SK3: 890 | 384x208 MI16 SK5: 956 (+7.4%) | 384x208 MI16 SK5: 638 (-28.4%) |
| DeepSeek-V4 | 512, 1024, 7168 | 8 | 256x64 MI16 SK3: 329 | 128x128 MI16 SK5: 343 (+4.5%) | 128x128 MI16 SK5: 242 (-26.3%) |
| DeepSeek-V4, Kimi; layer: Kimi-K2 | 1024, 2048, 7168 | 32 | 128x128 MI32 SK3: 794 | 256x128 MI16 SK5: 766 (-3.5%) | 256x128 MI16 SK5: 595 (-25.1%) |
| AITER tuned list (model not recorded) | 2048, 1024, 6144 (fp32 D) | 32 | 128x128 MI16 SK3: 718 | 256x128 MI16 SK5: 627 (-12.6%) | 256x128 MI16 SK5: 542 (-24.6%) |
| DeepSeek-V4 | 2048, 1280, 7168 | 40 | 128x160 MI16 SK3: 844 | 256x160 MI16 SK5: 749 (-11.2%) | 256x160 MI16 SK5: 644 (-23.6%) |
| DeepSeek-V4.1 | 1024, 2048, 5120 (fp32 D) | 32 | 128x128 MI16 SK3: 650 | 256x128 MI16 SK5: 585 (-9.9%) | 256x128 MI16 SK5: 497 (-23.5%) |
| Llama 70B | 8192, 512, 7168 | 64 | 128x128 MI16 SK3: 898 | 256x256 MI16 SK5: 742 (-17.4%) | 256x256 MI16 SK5: 691 (-23.1%) |
| DeepSeek-V4; layer: GLM-5.3-Flash, Qwen3.5-397B | 512, 4096, 4096 | 32 | 128x128 MI16 SK3: 530 | 256x128 MI16 SK5: 530 (-0.2%) | 256x128 MI16 SK5: 408 (-23.0%) |
| qwen3_8_2p4t_a95b | 512, 4096, 8192 | 32 | 128x128 MI16 SK3: 763 | 256x128 MI16 SK5: 749 (-1.9%) | 256x128 MI16 SK5: 587 (-23.0%) |
| layer: DeepSeek-V4, GLM-5.3-Flash, Qwen3.5-397B | 512, 3840, 4096 | 30 | 128x128 MI16 SK3: 526 | 256x128 MI16 SK5: 510 (-3.0%) | 256x128 MI16 SK5: 406 (-22.7%) |
| Qwen3.5-397B | 8704, 512, 4096 | 68 | 160x128 MI16 SK3: 917 | 80x256 MI16 SK5: 743 (-19.0%) | 80x256 MI16 SK5: 716 (-21.9%) |
| Llama 405B | 2304, 512, 16384 | 18 | 256x128 MI16 SK3: 748 | 144x256 MI16 SK5: 821 (+9.8%) | 144x256 MI16 SK5: 589 (-21.3%) |
| DeepSeek-V4 | 2048, 1024, 7168 | 32 | 256x128 MI16 SK3: 742 | 256x128 MI16 SK5: 731 (-1.5%) | 256x128 MI16 SK5: 588 (-20.7%) |
| DeepSeek-V4 | 2048, 2048, 7168 | 64 | 128x128 MI16 SK3: 878 | 256x256 MI16 SK5: 790 (-10.0%) | 256x256 MI16 SK5: 702 (-20.0%) |
| qwen3_8_2p4t_a95b | 4608, 512, 8192 | 36 | 160x128 MI16 SK3: 835 | 144x256 MI16 SK5: 739 (-11.5%) | 144x256 MI16 SK5: 679 (-18.7%) |
| Llama 70B; layer: Kimi | 8192, 512, 8192 | 64 | 128x128 MI16 SK3: 903 | 256x256 MI16 SK5: 804 (-11.0%) | 256x256 MI16 SK5: 735 (-18.6%) |
| Qwen3.5-397B | 4096, 1024, 8192 | 64 | 128x128 MI16 SK3: 911 | 256x256 MI16 SK5: 845 (-7.3%) | 256x256 MI16 SK5: 751 (-17.5%) |
| DeepSeek-V4, GLM-5.3-Flash | 2048, 1024, 4096 | 32 | 128x128 MI16 SK3: 539 | 256x128 MI16 SK5: 558 (+3.4%) | 256x128 MI16 SK5: 447 (-17.2%) |
| DeepSeek-V4; layer: Qwen3.5-397B | 1024, 2048, 4096 | 32 | 128x128 MI16 SK3: 530 | 256x128 MI16 SK5: 553 (+4.2%) | 256x128 MI16 SK5: 442 (-16.7%) |
| layer: GLM-5, MiniMax-M3 (MXFP4) | 3072, 1024, 6144 (fp32 D) | 48 | 96x128 MI16 SK3: 839 | 192x256 MI16 SK5: 748 (-10.8%) | 192x256 MI16 SK5: 707 (-15.8%) |
| qwen3_8_2p4t_a95b | 512, 1024, 8192 | 8 | 256x64 MI16 SK3: 371 | 128x128 MI16 SK5: 417 (+12.3%) | 128x128 MI16 SK5: 314 (-15.5%) |
| Kimi-K3 | 1536, 1024, 7168 | 24 | 256x128 MI16 SK3: 628 | 192x128 MI32 SK5: 636 (+1.4%) | 192x128 MI32 SK5: 533 (-15.1%) |
| layer: GLM-5, MiniMax-M3 (MXFP4) | 3072, 512, 6144 (fp32 D) | 24 | 96x128 MI16 SK3: 620 | 96x256 MI16 SK5: 532 (-14.1%) | 96x256 MI16 SK5: 527 (-15.0%) |
| Kimi; layer: GLM-5, Kimi-K2 | 7168, 1024, 512 | 112 | 256x128 MI16 SK3: 572 | 224x128 MI16 SK5: 494 (-13.7%) | 224x128 MI16 SK5: 487 (-14.9%) |
| layer: DeepSeek-V4, GLM-5.3-Flash, Qwen3.5-397B | 512, 4352, 4096 | 34 | 128x80 MI16 SK3: 613 | 256x144 MI16 SK5: 556 (-9.3%) | 256x144 MI16 SK5: 522 (-14.8%) |
| MiniMax-M3 (EAGLE) | 2304, 1024, 6144 | 36 | 256x176 MI16 SK3: 673 | 160x256 MI16 SK5: 666 (-1.1%) | 160x256 MI16 SK5: 574 (-14.8%) |
| DeepSeek-V4 | 512, 4096, 7168 | 32 | 128x128 MI16 SK3: 672 | 256x128 MI16 SK5: 719 (+7.1%) | 256x128 MI16 SK5: 575 (-14.3%) |
| DeepSeek-V4 | 2048, 1792, 7168 | 56 | 128x128 MI16 SK3: 802 | 256x224 MI16 SK5: 860 (+7.3%) | 256x224 MI16 SK5: 688 (-14.1%) |
| Qwen 32B | 5120, 512, 6400 | 40 | 160x128 MI16 SK3: 779 | 160x256 MI16 SK5: 677 (-13.1%) | 160x256 MI16 SK5: 671 (-13.9%) |
| Kimi-K3 | 2048, 4096, 512 | 128 | 256x128 MI16 SK3: 655 | 256x128 MI16 SK5: 572 (-12.7%) | 256x128 MI16 SK5: 621 (-5.2%) |
| DeepSeek-V4; layer: GLM-5.3-Flash, Qwen3.5-397B | 512, 10240, 4096 | 80 | 128x160 MI16 SK3: 996 | 256x80 MI16 SK5: 875 (-12.2%) | 256x80 MI16 SK5: 891 (-10.6%) |
| DeepSeek-V4; layer: GLM-5.3-Flash, Qwen3.5-397B | 512, 1280, 4096 | 10 | 64x48 MI16 SK3: 357 | 64x80 MI16 SK5: 317 (-11.1%) | 64x80 MI16 SK5: 320 (-10.3%) |
| Qwen 32B | 5120, 512, 5120 | 40 | 96x128 MI16 SK3: 772 | 160x128 MI16 SK5: 713 (-7.6%) | 160x128 MI16 SK5: 688 (-10.9%) |
| qwen3_8_2p4t_a95b | 4608, 1024, 8192 | 72 | 160x256 MI16 SK3: 943 | 288x288 MI16 SK5: 923 (-2.2%) | 288x288 MI16 SK5: 845 (-10.5%) |
| Kimi-K3 | 7168, 2048, 1024 | 224 | 256x256 MI16 SK3: 1039 | 224x256 MI16 SK5: 934 (-10.1%) | 224x256 MI16 SK5: 940 (-9.5%) |
| Kimi-K3 | 1536, 512, 1536 | 12 | 64x64 MI16 SK3: 286 | 96x32 MI16 SK5: 261 (-8.7%) | 96x32 MI16 SK5: 265 (-7.4%) |
| MiniMax-M3 (EAGLE) | 2560, 1024, 6144 | 40 | 160x128 MI16 SK3: 760 | 160x256 MI16 SK5: 704 (-7.3%) | 160x256 MI16 SK5: 694 (-8.7%) |
| Kimi-K3 | 6144, 512, 7168 | 48 | 192x128 MI16 SK3: 824 | 192x256 MI16 SK5: 756 (-8.2%) | 192x256 MI16 SK5: 775 (-5.9%) |
| DeepSeek-V4.1 | 1024, 1536, 5120 (fp32 D) | 24 | 64x96 MI16 SK3: 606 | 128x192 MI16 SK5: 560 (-7.6%) | 128x192 MI16 SK5: 560 (-7.5%) |
| DeepSeek-V4; layer: GLM-5.3-Flash, Qwen3.5-397B | 512, 1024, 4096 | 8 | 64x32 MI16 SK3: 309 | 64x64 MI16 SK5: 298 (-3.6%) | 64x64 MI16 SK5: 286 (-7.4%) |
| DeepSeek-V4 | 512, 6144, 7168 | 48 | 128x192 MI16 SK3: 683 | 256x192 MI16 SK5: 750 (+9.8%) | 256x192 MI16 SK5: 634 (-7.3%) |
| Qwen 32B | 6400, 512, 5120 | 50 | 128x128 MI16 SK3: 699 | 320x176 MI16 SK5: 679 (-2.8%) | 320x176 MI16 SK5: 650 (-7.0%) |
| Llama 405B | 16384, 512, 2048 | 128 | 256x128 MI16 SK3: 992 | 160x256 MI16 SK5: 923 (-6.9%) | 160x256 MI16 SK5: 964 (-2.8%) |
| Kimi-K3 | 7168, 1024, 768 | 112 | 224x128 MI16 SK3: 667 | 224x128 MI16 SK5: 624 (-6.5%) | 224x128 MI16 SK5: 623 (-6.6%) |
| DeepSeek-V4; layer: GLM-5.3-Flash, Qwen3.5-397B | 512, 12288, 4096 | 96 | 128x192 MI16 SK3: 1013 | 256x96 MI16 SK5: 961 (-5.1%) | 256x96 MI16 SK5: 950 (-6.2%) |
| DeepSeek-V4 | 512, 2048, 7168 | 16 | 128x128 MI16 SK3: 602 | 128x128 MI16 SK5: 574 (-4.6%) | 128x128 MI16 SK5: 565 (-6.2%) |
| DeepSeek-V4; layer: Kimi, Kimi-K2 | 1024, 1536, 7168 | 24 | 128x96 MI16 SK3: 736 | 128x96 MI16 SK5: 736 (-0.1%) | 128x96 MI16 SK5: 691 (-6.1%) |
| Llama 405B | 4608, 1024, 16384 | 72 | 144x256 MI16 SK3: 1107 | 288x288 MI16 SK5: 1116 (+0.8%) | 288x288 MI16 SK5: 1042 (-5.9%) |
| MiniMax-M3 (EAGLE) | 2560, 512, 6144 | 20 | 256x128 MI16 SK3: 504 | 160x128 MI16 SK5: 552 (+9.5%) | 160x128 MI16 SK5: 476 (-5.7%) |
| Qwen3.5-397B | 4096, 512, 8192 | 32 | 128x128 MI16 SK3: 756 | 128x256 MI16 SK5: 713 (-5.7%) | 128x256 MI16 SK5: 737 (-2.5%) |
| Llama 70B | 14336, 512, 8192 | 112 | 224x256 MI16 SK3: 1034 | 112x256 MI16 SK5: 975 (-5.7%) | 112x256 MI16 SK5: 976 (-5.6%) |
| Qwen3.5-397B | 4096, 768, 8192 | 48 | 128x192 MI16 SK3: 780 | 256x192 MI16 SK5: 817 (+4.8%) | 256x192 MI16 SK5: 735 (-5.7%) |
| Kimi-K3 | 7168, 1024, 1024 | 112 | 224x128 MI16 SK3: 771 | 224x128 MI16 SK5: 731 (-5.2%) | 224x128 MI16 SK5: 729 (-5.4%) |
| DeepSeek-V4 | 2048, 512, 7168 | 16 | 128x128 MI32 SK3: 588 | 128x128 MI32 SK5: 596 (+1.3%) | 128x128 MI32 SK5: 557 (-5.4%) |
| MiniMax-M3 (EAGLE) | 6144, 1024, 12288 | 96 | 192x256 MI16 SK3: 1154 | 192x128 MI16 SK5: 1107 (-4.1%) | 192x128 MI16 SK5: 1092 (-5.4%) |
| DeepSeek-V4; layer: Qwen3.5-397B | 1024, 512, 4096 | 8 | 64x32 MI16 SK3: 305 | 64x64 MI16 SK5: 294 (-3.6%) | 64x64 MI16 SK5: 289 (-5.3%) |
| Kimi-K3 | 3584, 1024, 7168 | 56 | 128x128 MI16 SK3: 820 | 224x256 MI16 SK5: 850 (+3.6%) | 224x256 MI16 SK5: 778 (-5.1%) |

</details>

### Smaller workspace

Same 630 problems with `HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES` = 32 MiB. The heuristic top-1 is the same solution as at 128 MiB on 630 / 630 / 630 of 630 problems (1.2.2 / 1.4.1 / 1.5.0), i.e. the workspace limit does not change the selection, but:

- Under the 32 MiB limit the heuristic returns the same stream-K solutions with `workspaceSize = 0` (at 128 MiB the same solution reports e.g. 85 MB for m=2304 n=2048 k=16384). When `hipblasLtMatmul` is then called with the allocated 32 MiB workspace (what frameworks such as PyTorch pass), it returns `HIPBLAS_STATUS_INTERNAL_ERROR` (6) on 3 problems in 1.2.2, **37 in 1.4.1** and 0 in 1.5.0. 1.4.1 examples: m=512 n=4352 k=4096 (256x144 MI16 SK5), m=6400 n=512 k=5120 (320x176 MI16 SK5), m=2304 n=512 k=6144 (144x256 MI16 SK5), m=2304 n=1024 k=6144 (160x256 MI16 SK5), m=2560 n=1024 k=6144 (160x256 MI16 SK5) and 32 more. Passing the heuristic's reported size (0) instead, as `hipblaslt-bench` does, runs all 37 1.4.1 cases without error, with the same kernel; `hipblaslt-bench` therefore does not show the error.
- Problems at least 20% slower than 1.2.2 at the same 32 MiB workspace: 3 in 1.4.1, 54 in 1.5.0, e.g. m=4608 n=512 k=16384: 112x512 MI16 SK5 276 vs 944 (-71%); m=1024 n=1792 k=7168: 256x224 MI32 SK5 200 vs 654 (-70%); m=2304 n=512 k=16384: 144x256 MI16 SK5 232 vs 749 (-69%); m=1024 n=1280 k=7168: 256x160 MI16 SK5 226 vs 601 (-62%).
- m=2304 n=2048 k=16384: 1.4.1 and 1.5.0 select 288x288 MI16 SK5 at both workspace sizes. At 32 MiB: 1.2.2 256x256 MI16 SK3 660, 1.4.1 error status 6, 1.5.0 503; at 128 MiB 1.5.0 runs at 1092. Frameworks hit this: with PyTorch's default hipBLASLt workspace, `F.linear` on this GEMM (PyTorch 2048×2304×16384) runs at ~505 TFLOPS on 1.5.0 (1054 with `HIPBLASLT_WORKSPACE_SIZE=131072`), and on PyTorch 2.13 + ROCm 7.14 (hipBLASLt 1.4.1) it fails with `HIPBLAS_STATUS_INTERNAL_ERROR` and PyTorch falls back to hipBLAS.

### Reproduce

`hipblaslt-bench` is hipBLASLt's benchmark client (`projects/hipblaslt/clients/bench`). Each line below times a single problem and is an example; the full data set comes from the attached standalone benchmark (second block).

Verified with the `hipblaslt-bench` binaries from TheRock's gfx950 tests tarballs (`therock-dist-linux-gfx950-dcgpu-tests-10.0.0`, `...-tests-10.2.0a20260929`), using these exact arguments on every regressed problem. At 128 MiB it selects the same kernel as the attached benchmark on 107/107 (1.4.1) and 107/107 (1.5.0) problems. On the 33 large problems its TFLOPS are within 3% on 23/33 and 22/33 (median ratio 0.988 / 0.981); on the 74 small problems (tens of µs per call) it reports lower TFLOPS (median ratio 0.913 / 0.940) because it times each call individually, while the attached benchmark times 100 back-to-back calls. At 32 MiB: same kernel on 58/58 and 58/58, and the 1.5.0 slowdowns reproduce (median TFLOPS ratio 0.99). The 1.4.1 `INTERNAL_ERROR` needs the allocated workspace size passed to `hipblasLtMatmul` (see above), which `hipblaslt-bench` does not do; reproduce it with the attached benchmark (`HBL_WORKSPACE_KB=32768`).

```bash
hipblaslt-bench -m 7168 -n 32768 -k 1024 --transA T --transB N --lda 1024 --ldb 1024 --ldc 7168 --ldd 7168 --a_type bf16_r --b_type bf16_r --c_type bf16_r --d_type bf16_r --compute_type f32_r --alpha 1 --beta 0 -j 20 -i 100 --print_kernel_info
hipblaslt-bench -m 8192 -n 32768 -k 1024 --transA T --transB N --lda 1024 --ldb 1024 --ldc 8192 --ldd 8192 --a_type bf16_r --b_type bf16_r --c_type bf16_r --d_type bf16_r --compute_type f32_r --alpha 1 --beta 0 -j 20 -i 100 --print_kernel_info
hipblaslt-bench -m 1024 -n 512 -k 7168 --transA T --transB N --lda 7168 --ldb 7168 --ldc 1024 --ldd 1024 --a_type bf16_r --b_type bf16_r --c_type bf16_r --d_type bf16_r --compute_type f32_r --alpha 1 --beta 0 -j 20 -i 100 --print_kernel_info
# 32 MiB workspace (m=2304 n=2048 k=16384):
hipblaslt-bench -m 2304 -n 2048 -k 16384 --transA T --transB N --lda 16384 --ldb 16384 --ldc 2304 --ldd 2304 --a_type bf16_r --b_type bf16_r --c_type bf16_r --d_type bf16_r --compute_type f32_r --alpha 1 --beta 0 -j 20 -i 100 --workspace 33554432 --print_kernel_info
# Workaround check (1.5.0 / develop only): restrict Origami's ranking to the 256x256 macro-tile
ANALYTICAL_GEMM_PICK=256x256x64 hipblaslt-bench -m 7168 -n 32768 -k 1024 --transA T --transB N --lda 1024 --ldb 1024 --ldc 7168 --ldd 7168 --a_type bf16_r --b_type bf16_r --c_type bf16_r --d_type bf16_r --compute_type f32_r --alpha 1 --beta 0 -j 20 -i 100 --print_kernel_info
```

Output of these lines here (`hipblaslt-bench` from the TheRock tarballs, selected kernel and TFLOPS):

- m=7168 n=32768 k=1024, 1.4.1 (ROCm 10.0.0): 224x384 MI16 SK5, 1096 TFLOPS
- m=7168 n=32768 k=1024, 1.5.0: 224x384 MI16 SK5, 1088 TFLOPS
- m=8192 n=32768 k=1024, 1.4.1 (ROCm 10.0.0): 256x320 MI16 SK5, 1094 TFLOPS
- m=8192 n=32768 k=1024, 1.5.0: 256x320 MI16 SK5, 1098 TFLOPS
- m=1024 n=512 k=7168, 1.4.1 (ROCm 10.0.0): 128x128 MI32 SK5, 424 TFLOPS
- m=1024 n=512 k=7168, 1.5.0: 128x128 MI32 SK5, 159 TFLOPS
- m=2304 n=2048 k=16384 (32 MiB), 1.4.1 (ROCm 10.0.0): 288x288 MI16 SK5, 503 TFLOPS
- m=2304 n=2048 k=16384 (32 MiB), 1.5.0: 288x288 MI16 SK5, 504 TFLOPS
- m=7168 n=32768 k=1024, 1.5.0 + ANALYTICAL_GEMM_PICK: 256x256 MI16 SK5, 1219 TFLOPS

Full data set with the attached benchmark (`shapes_all.txt`: all 630 problems as `m n k dtypeD`; `regress_shapes.txt`: the regressed ones):

```bash
hipcc -O2 -std=c++17 --offload-arch=gfx950 hbl_bench.cpp -lhipblaslt -o hbl_bench
./hbl_bench shapes_all.txt heuristic                                   # heuristic top-1, 128 MiB
HBL_WORKSPACE_KB=32768 ./hbl_bench shapes_all.txt heuristic            # heuristic top-1, 32 MiB
./hbl_bench regress_shapes.txt best256                                 # fastest MT256x256x64 solution
ANALYTICAL_GEMM_PICK=256x256x64 ./hbl_bench regress_shapes.txt heuristic
```

### What I checked in rocm-libraries (`develop` 55e6a02)

- `shared/origami/src/origami/heuristics.cpp` (HEURISTIC 3) rejects gfx950 BF16 TN subtile kernels only for `K < 512` (`key.subtile = true; key.max_k = 511`). Every regressed large problem where 1.2.2 selects MT256x256 has k ≥ 1024.
- These problems miss the Equality table and fall through to the Origami Prediction library (`Logic/asm_full/gfx950/gfx950/Origami/gfx950_Cijk_Alik_Bljk_BBS_BH_BiasSB_HAS_SAV_UserArgs.yaml`, 56 `UseSubtileImpl: true` solutions), where `rank_configs` ranks the subtile tiles above MT256x256.
- No runtime flag disables subtile solutions or restores the 1.2.2 selection. `ANALYTICAL_GEMM_PICK` (develop only, #7228) forces one macro-tile globally; `HIPBLASLT_TUNING_OVERRIDE_FILE` pins per problem; `ANALYTICAL_GEMM_HEURISTICS=0` only removes the `K < 512` reject.
- CHANGELOG 1.5.0 lists gfx950 Origami improvements and a subtile out-of-bounds fix; no known issue for this. No commit after #8626 (2026-06-19) changes subtile ranking. #8648 ("Refine gfx950 BF16 TN subtile reject heuristic") said the `K < 512` rule was "too permissive" and proposed rejects to "recover the regressions", but was closed unmerged ("Not needed anymore"). None of these problems are in `shared/origami/python/tests/baselines/rankings/gfx950.yaml`.

**Attachments:** `hbl_pure_results.csv` (all 630 problems: selection and TFLOPS per version at 128 MiB and 32 MiB, best MT256x256 and `ANALYTICAL_GEMM_PICK` results for the regressed problems, full kernel names, model tags), `bench_check.csv` (the `hipblaslt-bench` cross-check), `hbl_bench.cpp`, `shapes_all.txt`, `regress_shapes.txt`.
