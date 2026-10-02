# Issue GEMM numbers on two hip stacks

## Environments

| | torch | HIP | hipBLASLt | GPU 0 PCI bus |
|---|---|---|---|---|
| rocm724 | 2.10.0+rocm7.2.4.git3d3aa833 | 7.2.53211 | 100202 (dabb6df2b9) | 117 |
| nightly | 2.15.0a0+rocm10.2.0a20260929 | 7.17.26386 | 100500 (6054c511) | 117 |

Both use Triton `gfx950-tutorial-v2.2` (llirSched plugin loaded), the same AITER checkout and tuned CSVs, and GPU 0 only. The issue's own run used torch 2.10.0+rocm7.2.4 as well.

## Tutorial shapes (TFLOPS, median of 3)

| M×N×K | Variant | Issue | rocm724 | nightly | nightly vs rocm724 |
|---|---|---:|---:|---:|---:|
| 4096×4096×8192 | Triton backend | 1305 | 1280 | 1281 | +0.1% |
| 4096×4096×8192 | Gluon, no plugin | 1354 | 1347 | 1344 | -0.2% |
| 4096×4096×8192 | Gluon + llirSched | 1589 | 1579 | 1580 | +0.1% |
| 4096×4096×8192 | hipBLASLt (F.linear) | 1536 | 1505 | 1499 | -0.4% |
| 8192×8192×8192 | Triton backend | 1248 | 1256 | 1255 | -0.1% |
| 8192×8192×8192 | Gluon, no plugin | 1338 | 1333 | 1328 | -0.3% |
| 8192×8192×8192 | Gluon + llirSched | 1590 | 1574 | 1579 | +0.3% |
| 8192×8192×8192 | hipBLASLt (F.linear) | 1586 | 1565 | 1565 | +0.0% |
| 4096×8192×4096 | Triton backend | 1224 | 1217 | 1218 | +0.0% |
| 4096×8192×4096 | Gluon, no plugin | 1316 | 1305 | 1308 | +0.3% |
| 4096×8192×4096 | Gluon + llirSched | 1526 | 1523 | 1526 | +0.2% |
| 4096×8192×4096 | hipBLASLt (F.linear) | 1492 | 1478 | 1472 | -0.4% |
| 4096×106496×16384 | Triton backend | 1201 | 1205 | 1207 | +0.1% |
| 4096×106496×16384 | Gluon, no plugin | 1326 | 1307 | 1297 | -0.8% |
| 4096×106496×16384 | Gluon + llirSched | 1589 | 1592 | 1593 | +0.1% |
| 4096×106496×16384 | hipBLASLt (F.linear) | 1498 | 1504 | 1458 | -3.1% |

## Scope summary per environment (the issue's table, recomputed)

**Issue (torch 2.10 + ROCm 7.2.4, as reported)**

| Shapes | Count | Gluon's gap to the winner |
|---|---:|---|
| Measured | 630 | |
| Out of scope (smaller tile: 23; split-K: 15) | 38 | |
| Compute-bound | 592 | |
| Gluon wins | 293 | 0 |
| Lost to another of the eight tiles | 144 | median 63%, p90 178% |
| Lost to a non-power-of-2 tile | 117 | median 35%, p90 143% |
| Lost to another 256x256 kernel | 38 | median 0.7% (up to 174%) |

At 512 or more output tiles Gluon wins (or ties to 0.1%) 195 of 225 shapes; its worst loss there is 3.8%. hipBLASLt wins 150 of the 630 shapes (stream-K mode of the winning kernel: SK3: 150).

**torch 2.10.0+rocm7.2.4 (rerun here)**

| Shapes | Count | Gluon's gap to the winner |
|---|---:|---|
| Measured | 630 | |
| Out of scope (smaller tile: 23; split-K: 15) | 38 | |
| Compute-bound | 592 | |
| Gluon wins | 314 | 0 |
| Lost to another of the eight tiles | 142 | median 61%, p90 173% |
| Lost to a non-power-of-2 tile | 117 | median 31%, p90 140% |
| Lost to another 256x256 kernel | 19 | median 1.1% (up to 179%) |

At 512 or more output tiles Gluon wins (or ties to 0.1%) 206 of 225 shapes; its worst loss there is 3.9%. hipBLASLt wins 139 of the 630 shapes (stream-K mode of the winning kernel: SK3: 139).

**TheRock nightly torch 2.15.0a0 + ROCm 10.2.0a20260929**

| Shapes | Count | Gluon's gap to the winner |
|---|---:|---|
| Measured | 630 | |
| Out of scope (smaller tile: 23; split-K: 15) | 38 | |
| Compute-bound | 592 | |
| Gluon wins | 310 | 0 |
| Lost to another of the eight tiles | 138 | median 59%, p90 156% |
| Lost to a non-power-of-2 tile | 133 | median 32%, p90 134% |
| Lost to another 256x256 kernel | 11 | median 1.1% (up to 71%) |

At 512 or more output tiles Gluon wins (or ties to 0.1%) 209 of 225 shapes; its worst loss there is 13.7%. hipBLASLt wins 144 of the 630 shapes (stream-K mode of the winning kernel: SK5: 144).

## Per-shape changes

- **issue → rocm724:** 31 shapes change between Gluon winning and AITER's pick winning; on 23 of them Gluon and the pick are within 2% of each other in both runs (near-ties), leaving 8 real flips.
  - 2048×6144×3072 float32 (asm): issue gluon 1147.4 vs pick 1175.5; rocm724 gluon 1198.9 vs pick 1149.7
  - 8192×9216×6144 bfloat16 (asm): issue gluon 1504.1 vs pick 1504.4; rocm724 gluon 1481.9 vs pick 1447.8
  - 8192×5120×25600 bfloat16 (asm): issue gluon 1431.4 vs pick 1431.7; rocm724 gluon 1431.9 vs pick 1401.6
  - 512×12800×5120 bfloat16 (flydsl): issue gluon 791.3 vs pick 810.3; rocm724 gluon 813.6 vs pick 801.8
  - 512×57344×8192 bfloat16 (flydsl): issue gluon 1260.7 vs pick 1285.5; rocm724 gluon 1346.5 vs pick 1306.9
  - 1024×6144×3072 float32 (opus): issue gluon 648.4 vs pick 696.8; rocm724 gluon 687.6 vs pick 631.2
  - 16384×1536×1536 bfloat16 (torch): issue gluon 1126.3 vs pick 1171.0; rocm724 gluon 1155.2 vs pick 1149.7
  - 3072×4096×2048 bfloat16 (torch): issue gluon 1158.1 vs pick 1183.0; rocm724 gluon 1190.8 vs pick 1183.4
- **rocm724 → nightly:** 26 shapes change between Gluon winning and AITER's pick winning; on 1 of them Gluon and the pick are within 2% of each other in both runs (near-ties), leaving 25 real flips.
  - 32768×7168×1024 bfloat16 (torch): rocm724 gluon 1240.0 vs pick 1261.5; nightly gluon 1246.9 vs pick 1106.4
  - 32768×8192×1024 bfloat16 (torch): rocm724 gluon 1227.1 vs pick 1253.7; nightly gluon 1234.4 vs pick 1102.9
  - 16384×1536×1536 bfloat16 (torch): rocm724 gluon 1155.2 vs pick 1149.7; nightly gluon 1166.8 vs pick 1192.1
  - 32768×7168×1792 bfloat16 (torch): rocm724 gluon 1373.9 vs pick 1397.5; nightly gluon 1402.8 vs pick 1278.9
  - 3072×4096×2048 bfloat16 (torch): rocm724 gluon 1190.8 vs pick 1183.4; nightly gluon 1191.3 vs pick 1244.3
  - 32768×8192×2048 bfloat16 (torch): rocm724 gluon 1419.9 vs pick 1420.9; nightly gluon 1418.1 vs pick 1316.4
  - 6400×2048×4096 bfloat16 (torch): rocm724 gluon 1335.5 vs pick 1229.6; nightly gluon 1334.5 vs pick 1363.2
  - 6912×2048×4096 bfloat16 (torch): rocm724 gluon 1370.4 vs pick 1327.2; nightly gluon 1378.9 vs pick 1415.5
  - 7168×2048×4096 bfloat16 (torch): rocm724 gluon 1406.7 vs pick 1349.5; nightly gluon 1408.8 vs pick 1438.2
  - 14336×1024×4096 bfloat16 (torch): rocm724 gluon 1396.1 vs pick 1351.8; nightly gluon 1399.2 vs pick 1450.1
  - 16384×16384×4096 bfloat16 (torch): rocm724 gluon 1521.2 vs pick 1535.6; nightly gluon 1522.1 vs pick 1462.4
  - 4096×6144×4608 bfloat16 (torch): rocm724 gluon 1333.5 vs pick 1324.5; nightly gluon 1336.9 vs pick 1424.2
  - 1024×37888×5120 bfloat16 (torch): rocm724 gluon 1265.6 vs pick 1066.7; nightly gluon 1273.3 vs pick 1309.7
  - 1536×37888×5120 bfloat16 (torch): rocm724 gluon 1367.0 vs pick 1275.7; nightly gluon 1361.6 vs pick 1367.7
  - 8192×5120×5120 bfloat16 (torch): rocm724 gluon 1433.0 vs pick 1355.2; nightly gluon 1431.0 vs pick 1457.6
  - 32768×9216×6144 bfloat16 (torch): rocm724 gluon 1551.6 vs pick 1561.2; nightly gluon 1558.0 vs pick 1447.4
  - 14336×1024×7168 bfloat16 (torch): rocm724 gluon 1462.9 vs pick 1405.5; nightly gluon 1444.1 vs pick 1518.9
  - 1024×4608×8192 bfloat16 (torch): rocm724 gluon 620.8 vs pick 987.5; nightly gluon 619.8 vs pick 483.5
  - 32768×7168×8192 bfloat16 (torch): rocm724 gluon 1594.5 vs pick 1595.0; nightly gluon 1592.5 vs pick 1484.0
  - 2048×6144×12288 bfloat16 (torch): rocm724 gluon 1401.1 vs pick 1392.5; nightly gluon 1413.7 vs pick 1520.2
  - 16384×6144×12288 bfloat16 (torch): rocm724 gluon 1572.9 vs pick 1584.6; nightly gluon 1575.6 vs pick 1469.2
  - 2048×2304×16384 bfloat16 (torch): rocm724 gluon 639.6 vs pick 943.7; nightly gluon 640.2 vs pick 505.9
  - 4096×9216×16384 bfloat16 (torch): rocm724 gluon 1375.4 vs pick 1336.5; nightly gluon 1373.8 vs pick 1456.4
  - 2048×6144×18432 bfloat16 (torch): rocm724 gluon 1348.8 vs pick 1346.1; nightly gluon 1352.8 vs pick 1403.0
  - 4096×6144×18432 bfloat16 (torch): rocm724 gluon 1385.6 vs pick 1341.0; nightly gluon 1387.2 vs pick 1508.1

- **hipBLASLt kernels, issue vs rocm724 rerun:** the issue records the hipBLASLt kernel for its 150 hipBLASLt wins; the rerun launches the identical kernel(s) on 150 of them.

### TFLOPS change by AITER pick libtype

Median per-shape ratio. The Gluon column is the same v9 kernel in every environment, so it shows the GPU/day drift; the pick's residual is the pick ratio divided by Gluon's ratio.

**issue → rocm724**

| Pick | Shapes | Pick | Gluon v9 | Residual | Shapes with residual below −3% |
|---|---:|---:|---:|---:|---:|
| flydsl | 257 | -1.2% | -0.4% | -0.7% | 29 |
| torch | 229 | -1.5% | -0.5% | -0.6% | 19 |
| asm | 87 | -1.7% | -1.2% | -0.7% | 2 |
| opus | 54 | -1.7% | +2.3% | -3.9% | 31 |
| triton | 3 | -0.8% | -1.1% | -0.4% | 0 |

**rocm724 → nightly**

| Pick | Shapes | Pick | Gluon v9 | Residual | Shapes with residual below −3% |
|---|---:|---:|---:|---:|---:|
| flydsl | 257 | -0.1% | +0.1% | -0.3% | 13 |
| torch | 229 | +0.3% | +0.1% | +0.3% | 61 |
| asm | 87 | +0.1% | +0.1% | -0.0% | 0 |
| opus | 54 | +0.0% | +0.3% | -0.5% | 1 |
| triton | 3 | +0.7% | +1.0% | -0.6% | 0 |

**issue → nightly**

| Pick | Shapes | Pick | Gluon v9 | Residual | Shapes with residual below −3% |
|---|---:|---:|---:|---:|---:|
| flydsl | 257 | -1.4% | -0.1% | -1.0% | 46 |
| torch | 229 | -1.0% | -0.4% | -0.7% | 72 |
| asm | 87 | -1.5% | -1.1% | -0.8% | 1 |
| opus | 54 | -1.7% | +2.8% | -4.0% | 34 |
| triton | 3 | -1.1% | +0.2% | +0.1% | 0 |

### hipBLASLt kernel changes on `torch`-picked shapes (rocm724 → nightly)

229 shapes are `torch` picks in both. hipBLASLt keeps the same macro tile on 122 and switches to a different one on 107. Stream-K mode: SK3→SK5: 228, Custom→Custom: 1 (hipBLASLt's SK3→SK5 rename; SK5 with hybrid mode off runs the SK3 path).
- same kernel tile: median residual +0.4%, 9 below −3%, 33 above +3%.
- different kernel tile: median residual -0.3%, 52 below −3%, 45 above +3%.
- power-of-2 tile → non-power-of-2 tile: 35 shapes, median residual -5.9%, 30 below −3%, 1 above +3%.

Shapes where the macro tile changed, worst first (Change = pick TFLOPS ratio; ordering uses the drift-corrected residual):

| Shape | Tiles | rocm724 tile | TFLOPS | nightly tile | TFLOPS | Change | Gluon change | Winner rocm724 → nightly |
|---|---:|---|---:|---|---:|---:|---:|---|
| 1024×4608×8192 | 72 | 160x256 | 988 | 288x288 | 484 | -51.0% | -0.2% | hipBLASLt → gluon |
| 512×1536×7168 | 12 | 192x64 | 458 | 192x128 | 230 | -49.8% | -0.3% | hipBLASLt → hipBLASLt |
| 1280×1024×7168 | 20 | 256x80 | 595 | 256x160 | 308 | -48.2% | +0.1% | hipBLASLt → hipBLASLt |
| 2048×2304×16384 | 72 | 256x256 | 944 | 288x288 | 506 | -46.4% | +0.1% | hipBLASLt → gluon |
| 1536×512×7168 | 12 | 256x96 | 446 | 128x192 | 280 | -37.3% | +0.4% | hipBLASLt → hipBLASLt |
| 768×1024×7168 | 12 | 256x96 | 446 | 128x192 | 280 | -37.3% | +0.1% | hipBLASLt → hipBLASLt |
| 1792×512×7168 | 14 | 256x112 | 491 | 128x224 | 320 | -34.9% | -0.2% | hipBLASLt → hipBLASLt |
| 512×1280×8192 | 10 | 160x64 | 469 | 160x128 | 308 | -34.3% | -0.1% | hipBLASLt → hipBLASLt |
| 1280×512×7168 | 10 | 256x80 | 412 | 128x160 | 293 | -28.9% | -0.1% | hipBLASLt → hipBLASLt |
| 1024×2304×16384 | 36 | 256x256 | 899 | 384x208 | 642 | -28.6% | -0.6% | hipBLASLt → hipBLASLt |
| 2048×1024×7168 | 32 | 128x128 | 797 | 256x128 | 596 | -25.2% | -0.2% | hipBLASLt → hipBLASLt |
| 512×2304×16384 | 18 | 256x128 | 733 | 144x256 | 561 | -23.5% | +0.7% | hipBLASLt → hipBLASLt |
| 1280×2048×7168 | 40 | 128x160 | 851 | 256x160 | 673 | -20.9% | +0.1% | hipBLASLt → hipBLASLt |
| 512×4608×8192 | 36 | 160x128 | 853 | 144x256 | 702 | -17.7% | +0.2% | hipBLASLt → hipBLASLt |
| 512×8704×4096 | 68 | 160x128 | 904 | 80x256 | 745 | -17.5% | +0.1% | hipBLASLt → hipBLASLt |
| 1024×512×8192 | 8 | 256x64 | 368 | 128x128 | 310 | -15.8% | +0.0% | hipBLASLt → hipBLASLt |
| 1024×2304×6144 | 36 | 256x176 | 676 | 160x256 | 582 | -13.8% | +0.3% | hipBLASLt → hipBLASLt |
| 10240×512×4096 | 80 | 128x160 | 1007 | 256x80 | 882 | -12.4% | +0.6% | hipBLASLt → hipBLASLt |
| 32768×7168×1024 | 3584 | 256x256 | 1262 | 224x384 | 1106 | -12.3% | +0.6% | hipBLASLt → gluon |
| 32768×8192×1024 | 4096 | 256x256 | 1254 | 256x320 | 1103 | -12.0% | +0.6% | hipBLASLt → gluon |
| 32768×7168×1792 | 3584 | 256x256 | 1398 | 224x384 | 1279 | -8.5% | +2.1% | hipBLASLt → gluon |
| 16384×9216×6144 | 2304 | 256x256 | 1548 | 192x448 | 1389 | -10.3% | +0.1% | gluon → gluon |
| 16384×7168×8192 | 1792 | 256x256 | 1584 | 256x320 | 1434 | -9.5% | +0.1% | gluon → gluon |
| 1024×2560×6144 | 40 | 160x128 | 778 | 160x256 | 705 | -9.4% | -0.1% | hipBLASLt → hipBLASLt |
| 16384×10240×8192 | 2560 | 256x256 | 1594 | 224x384 | 1456 | -8.7% | +0.2% | gluon → gluon |
| 1024×6144×12288 | 96 | 192x256 | 1189 | 192x128 | 1088 | -8.5% | -0.1% | hipBLASLt → hipBLASLt |
| 32768×3584×7168 | 1792 | 256x256 | 1521 | 256x320 | 1403 | -7.8% | +0.3% | gluon → gluon |
| 512×5120×5120 | 40 | 96x128 | 789 | 160x128 | 725 | -8.0% | -0.1% | hipBLASLt → hipBLASLt |
| 32768×9216×6144 | 4608 | 256x256 | 1561 | 288x288 | 1447 | -7.3% | +0.4% | hipBLASLt → gluon |
| 16384×6144×12288 | 1536 | 256x256 | 1585 | 256x320 | 1469 | -7.3% | +0.2% | hipBLASLt → gluon |
| 32768×8192×2048 | 4096 | 256x256 | 1421 | 256x320 | 1316 | -7.4% | -0.1% | hipBLASLt → gluon |
| 32768×7168×8192 | 3584 | 256x256 | 1595 | 224x384 | 1484 | -7.0% | -0.1% | hipBLASLt → gluon |
| 512×4096×1024 | 32 | 128x64 | 488 | 64x128 | 459 | -5.9% | +0.7% | hipBLASLt → hipBLASLt |
| 16384×6144×2048 | 1536 | 256x256 | 1408 | 256x320 | 1308 | -7.0% | -0.7% | gluon → gluon |
| 16384×7168×35840 | 1792 | 256x256 | 1502 | 288x288 | 1461 | -2.7% | +3.6% | gluon → gluon |
| 16384×12800×5120 | 3200 | 256x256 | 1495 | 224x384 | 1407 | -5.9% | +0.1% | gluon → gluon |
| 16384×6144×4608 | 1536 | 256x256 | 1528 | 256x320 | 1438 | -5.8% | +0.1% | gluon → gluon |
| 32768×8192×7168 | 4096 | 256x256 | 1576 | 256x320 | 1480 | -6.1% | -0.3% | gluon → gluon |
| 16384×6144×3072 | 1536 | 256x256 | 1471 | 256x320 | 1385 | -5.9% | -0.1% | gluon → gluon |
| 1024×2560×8192 | 40 | 160x128 | 874 | 160x256 | 824 | -5.7% | -0.1% | hipBLASLt → hipBLASLt |
| 32768×6144×6144 | 3072 | 256x256 | 1538 | 256x320 | 1457 | -5.2% | +0.1% | gluon → gluon |
| 512×14336×8192 | 112 | 224x256 | 1036 | 112x256 | 980 | -5.5% | -0.2% | hipBLASLt → hipBLASLt |
| 32768×6144×4608 | 3072 | 256x256 | 1506 | 256x320 | 1431 | -5.0% | +0.1% | gluon → gluon |
| 32768×12800×5120 | 6400 | 256x256 | 1526 | 288x288 | 1447 | -5.2% | -0.1% | gluon → gluon |
| 512×2560×6144 | 20 | 256x128 | 512 | 160x128 | 486 | -4.9% | +0.1% | hipBLASLt → hipBLASLt |
| 16384×16384×4096 | 4096 | 256x256 | 1536 | 256x320 | 1462 | -4.8% | +0.1% | hipBLASLt → gluon |
| 768×129280×4096 | 1515 | 256x256 | 1358 | 176x384 | 1300 | -4.2% | +0.4% | gluon → gluon |
| 8192×10240×4096 | 1280 | 256x256 | 1496 | 224x384 | 1429 | -4.5% | -0.4% | gluon → gluon |
| 1024×5120×25600 | 80 | 160x256 | 1165 | 256x256 | 1121 | -3.8% | -0.0% | hipBLASLt → hipBLASLt |
| 32768×4608×16384 | 2304 | 256x256 | 1555 | 192x448 | 1499 | -3.6% | -0.2% | gluon → gluon |
| 1536×129280×4096 | 3030 | 256x256 | 1403 | 256x320 | 1359 | -3.2% | +0.0% | gluon → gluon |
| 32768×6400×5120 | 3200 | 256x256 | 1483 | 224x384 | 1437 | -3.1% | +0.0% | gluon → gluon |
| 768×37888×5120 | 444 | 224x256 | 1236 | 160x384 | 1223 | -1.1% | -0.2% | gluon → gluon |
| 1280×129280×4096 | 2525 | 256x256 | 1393 | 256x320 | 1392 | -0.1% | +0.2% | gluon → gluon |
| 16384×5120×25600 | 1280 | 256x256 | 1516 | 224x384 | 1511 | -0.3% | +0.0% | gluon → gluon |
| 32768×5120×25600 | 2560 | 256x256 | 1499 | 224x384 | 1502 | +0.2% | -0.1% | gluon → gluon |
| 1024×6144×7168 | 96 | 192x256 | 1064 | 192x128 | 1064 | +0.0% | -0.4% | hipBLASLt → hipBLASLt |
| 512×6144×12288 | 48 | 192x128 | 936 | 192x256 | 945 | +0.9% | +0.1% | hipBLASLt → hipBLASLt |
| 10240×1024×4096 | 160 | 256x160 | 1254 | 256x192 | 1281 | +2.1% | +0.6% | hipBLASLt → hipBLASLt |
| 4096×2304×16384 | 144 | 144x256 | 1307 | 256x320 | 1343 | +2.8% | +0.3% | hipBLASLt → hipBLASLt |
| 1024×1280×8192 | 20 | 256x128 | 622 | 160x128 | 637 | +2.4% | -0.1% | hipBLASLt → hipBLASLt |
| 5120×2048×4096 | 160 | 256x160 | 1243 | 256x192 | 1282 | +3.1% | +0.2% | hipBLASLt → hipBLASLt |
| 10240×1024×7168 | 160 | 256x160 | 1315 | 256x192 | 1357 | +3.2% | +0.1% | hipBLASLt → hipBLASLt |
| 512×2560×8192 | 20 | 256x128 | 607 | 160x128 | 634 | +4.4% | +0.1% | hipBLASLt → hipBLASLt |
| 4096×6144×6144 | 384 | 192x256 | 1359 | 256x256 | 1446 | +6.4% | +2.0% | hipBLASLt → hipBLASLt |
| 2048×2560×8192 | 80 | 160x256 | 1032 | 160x128 | 1092 | +5.8% | +0.7% | hipBLASLt → hipBLASLt |
| 4096×1280×8192 | 80 | 160x256 | 1038 | 160x128 | 1095 | +5.4% | -0.1% | hipBLASLt → hipBLASLt |
| 8192×4608×16384 | 576 | 192x256 | 1369 | 288x288 | 1455 | +6.3% | +0.7% | hipBLASLt → hipBLASLt |
| 4096×9216×6144 | 576 | 192x256 | 1366 | 240x320 | 1445 | +5.8% | -0.2% | hipBLASLt → hipBLASLt |
| 8192×4608×8192 | 576 | 192x256 | 1376 | 288x288 | 1465 | +6.4% | +0.2% | hipBLASLt → hipBLASLt |
| 1536×37888×5120 | 888 | 224x256 | 1276 | 352x224 | 1368 | +7.2% | -0.4% | gluon → hipBLASLt |
| 8192×5120×5120 | 640 | 256x224 | 1355 | 224x384 | 1458 | +7.6% | -0.1% | gluon → hipBLASLt |
| 2048×4608×8192 | 144 | 144x256 | 1258 | 160x256 | 1364 | +8.4% | +0.6% | hipBLASLt → hipBLASLt |
| 16384×2304×16384 | 576 | 192x256 | 1362 | 288x288 | 1471 | +8.0% | +0.1% | hipBLASLt → hipBLASLt |
| 2048×37888×5120 | 1184 | 240x256 | 1213 | 384x208 | 1315 | +8.4% | +0.4% | gluon → gluon |
| 1024×9216×6144 | 144 | 144x256 | 1226 | 160x256 | 1329 | +8.5% | +0.1% | hipBLASLt → hipBLASLt |
| 1024×8704×4096 | 136 | 144x256 | 1121 | 160x256 | 1218 | +8.6% | +0.0% | hipBLASLt → hipBLASLt |
| 4352×2048×4096 | 136 | 192x192 | 1088 | 160x256 | 1189 | +9.3% | +0.2% | hipBLASLt → hipBLASLt |
| 4096×9216×16384 | 576 | 192x256 | 1336 | 240x320 | 1456 | +9.0% | -0.1% | gluon → hipBLASLt |
| 1024×8448×7168 | 132 | 144x256 | 1183 | 160x256 | 1301 | +9.9% | -0.4% | hipBLASLt → hipBLASLt |
| 10240×512×7168 | 80 | 256x160 | 844 | 256x80 | 933 | +10.5% | +0.2% | hipBLASLt → hipBLASLt |
| 6400×2048×4096 | 200 | 256x208 | 1230 | 256x224 | 1363 | +10.9% | -0.1% | gluon → hipBLASLt |
| 5120×4096×1536 | 320 | 256x160 | 1104 | 256x320 | 1217 | +10.3% | -1.0% | hipBLASLt → hipBLASLt |
| 512×37888×5120 | 296 | 160x256 | 1057 | 448x176 | 1183 | +11.9% | +0.1% | hipBLASLt → hipBLASLt |
| 2048×10240×4096 | 320 | 160x256 | 1234 | 448x192 | 1387 | +12.3% | +0.3% | hipBLASLt → hipBLASLt |
| 4096×5120×1280 | 320 | 160x256 | 993 | 224x384 | 1119 | +12.7% | +0.4% | hipBLASLt → hipBLASLt |
| 4096×6144×18432 | 384 | 192x256 | 1341 | 256x256 | 1508 | +12.5% | +0.1% | gluon → hipBLASLt |
| 4096×8704×4096 | 544 | 192x256 | 1263 | 272x256 | 1419 | +12.3% | -0.4% | hipBLASLt → hipBLASLt |
| 4096×5120×5120 | 320 | 160x256 | 1266 | 224x384 | 1430 | +13.0% | +0.2% | hipBLASLt → hipBLASLt |
| 7680×4096×2048 | 480 | 256x240 | 1147 | 256x256 | 1299 | +13.2% | +0.3% | gluon → gluon |
| 16384×512×8192 | 128 | 256x256 | 1060 | 256x128 | 1213 | +14.5% | +1.0% | hipBLASLt → hipBLASLt |
| 4096×5120×6400 | 320 | 160x256 | 1293 | 224x384 | 1468 | +13.5% | +0.1% | hipBLASLt → hipBLASLt |
| 4096×5120×3200 | 320 | 160x256 | 1191 | 224x384 | 1357 | +13.9% | +0.4% | hipBLASLt → hipBLASLt |
| 2048×1280×8192 | 40 | 320x192 | 722 | 160x256 | 824 | +14.2% | -0.0% | hipBLASLt → hipBLASLt |
| 4096×4608×8192 | 288 | 144x256 | 1273 | 256x320 | 1457 | +14.4% | -0.2% | hipBLASLt → hipBLASLt |
| 10240×2048×4096 | 320 | 256x160 | 1277 | 256x320 | 1460 | +14.3% | -0.3% | hipBLASLt → hipBLASLt |
| 7680×2048×1536 | 240 | 256x240 | 1043 | 256x256 | 1192 | +14.2% | -0.7% | gluon → gluon |
| 7680×2048×4096 | 240 | 256x240 | 1241 | 256x256 | 1431 | +15.3% | +0.1% | gluon → gluon |
| 10240×2048×7168 | 320 | 256x160 | 1327 | 256x320 | 1532 | +15.5% | +0.1% | hipBLASLt → hipBLASLt |
| 8192×2304×6144 | 288 | 144x256 | 1235 | 288x288 | 1428 | +15.6% | -0.2% | hipBLASLt → hipBLASLt |
| 4096×4608×16384 | 288 | 144x256 | 1255 | 256x320 | 1450 | +15.5% | -0.4% | hipBLASLt → hipBLASLt |
| 7424×2048×4096 | 232 | 256x240 | 1211 | 256x256 | 1408 | +16.3% | +0.2% | gluon → gluon |
| 2048×9216×6144 | 288 | 144x256 | 1238 | 384x208 | 1456 | +17.6% | +0.5% | hipBLASLt → hipBLASLt |
| 4096×5120×25600 | 320 | 160x256 | 1279 | 224x384 | 1538 | +20.2% | -0.3% | hipBLASLt → hipBLASLt |
| 1024×37888×5120 | 592 | 256x208 | 1067 | 384x208 | 1310 | +22.8% | +0.6% | gluon → hipBLASLt |
| 2048×8704×4096 | 272 | 224x160 | 1151 | 272x256 | 1413 | +22.8% | -0.2% | hipBLASLt → hipBLASLt |
| 8192×2304×16384 | 288 | 144x256 | 1204 | 288x288 | 1461 | +21.3% | -7.1% | hipBLASLt → hipBLASLt |

## Correctness and anomalies

- **rocm724:** 630 shapes; 29 with an output outside the fp32 tolerance; 0 with errors. Live picks: flydsl 257, torch 229, asm 87, opus 54, triton 3.
  - 1024×6144×2048 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=1 gluon_maxerr=1 
  - 8192×6144×2048 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=1 gluon_maxerr=1 
  - 1024×6144×3072 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=2 gluon_maxerr=1 
  - 32768×6144×3072 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=2 gluon_maxerr=1 
  - 512×1024×4096 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=2 gluon_maxerr=1 
  - 768×8704×4096 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=2 gluon_maxerr=1 
  - 1280×512×4096 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=2 gluon_maxerr=0.5 
  - 512×6400×5120 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=2 gluon_maxerr=1 
  - 512×12800×5120 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=2 gluon_maxerr=1 
  - 1024×6400×5120 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=2 gluon_maxerr=1 
  - 512×1280×6144 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=2 gluon_maxerr=1 
  - 512×1536×6144 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=2 gluon_maxerr=1 
  - 512×2304×6144 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=2 gluon_maxerr=1 
  - 512×3072×6144 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=2 gluon_maxerr=1 
  - 512×5120×6400 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=2 gluon_maxerr=1 
  - 768×2048×7168 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=2 gluon_maxerr=2 
  - 1024×512×7168 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=2 gluon_maxerr=1 
  - 1024×1536×7168 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=2 gluon_maxerr=2 
  - 4096×512×7168 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=2 gluon_maxerr=2 
  - 512×4096×8192 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=2 gluon_maxerr=2 
  - 512×16384×8192 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=2 gluon_maxerr=2 
  - 1024×10240×8192 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=2 gluon_maxerr=2 
  - 512×13312×16384 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=4 gluon_maxerr=2 
  - 512×6144×18432 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=4 gluon_maxerr=2 
  - 512×5120×25600 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=4 gluon_maxerr=2 
  - 512×16384×26624 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=4 gluon_maxerr=4 
  - 2048×16384×26624 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=4 gluon_maxerr=4 
  - 512×8192×28672 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=4 gluon_maxerr=4 
  - 512×7168×35840 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=4 gluon_maxerr=2 
- **nightly:** 630 shapes; 29 with an output outside the fp32 tolerance; 0 with errors. Live picks: flydsl 257, torch 229, asm 87, opus 54, triton 3.
  - 1024×6144×2048 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=1 gluon_maxerr=1 
  - 8192×6144×2048 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=1 gluon_maxerr=1 
  - 1024×6144×3072 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=2 gluon_maxerr=1 
  - 32768×6144×3072 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=2 gluon_maxerr=1 
  - 512×1024×4096 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=2 gluon_maxerr=1 
  - 768×8704×4096 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=2 gluon_maxerr=1 
  - 1280×512×4096 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=2 gluon_maxerr=1 
  - 512×6400×5120 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=2 gluon_maxerr=1 
  - 512×12800×5120 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=2 gluon_maxerr=1 
  - 1024×6400×5120 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=2 gluon_maxerr=1 
  - 512×1280×6144 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=2 gluon_maxerr=1 
  - 512×1536×6144 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=2 gluon_maxerr=1 
  - 512×2304×6144 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=2 gluon_maxerr=1 
  - 512×3072×6144 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=2 gluon_maxerr=1 
  - 512×5120×6400 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=2 gluon_maxerr=1 
  - 768×2048×7168 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=2 gluon_maxerr=2 
  - 1024×512×7168 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=2 gluon_maxerr=1 
  - 1024×1536×7168 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=2 gluon_maxerr=2 
  - 4096×512×7168 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=2 gluon_maxerr=2 
  - 512×4096×8192 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=2 gluon_maxerr=2 
  - 512×16384×8192 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=2 gluon_maxerr=2 
  - 1024×10240×8192 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=2 gluon_maxerr=2 
  - 512×13312×16384 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=4 gluon_maxerr=2 
  - 512×6144×18432 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=4 gluon_maxerr=2 
  - 512×5120×25600 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=4 gluon_maxerr=2 
  - 512×16384×26624 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=4 gluon_maxerr=4 
  - 2048×16384×26624 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=4 gluon_maxerr=4 
  - 512×8192×28672 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=4 gluon_maxerr=4 
  - 512×7168×35840 bfloat16: pick_ok=False gluon_ok=True pick_maxerr=4 gluon_maxerr=4 
- **issue:** picks flydsl 257, torch 229, asm 87, opus 54, triton 3.

The outputs outside tolerance are the same 29 FlyDSL picks in both environments, and none is a hipBLASLt shape. Their max error (1–4) is the same size as Gluon's on those shapes (outputs are around 100, where one bf16 step is 1). They all split K, inside the workgroup (`w…x2`) or with split-K (`ks>1`), so an extra bf16 rounding of the partial sums fails the element-wise `allclose(atol=0.1, rtol=0.01)` on outputs close to zero. The issue reported all outputs correct, so it presumably used a looser check.

## Method notes

- Timing, shapes, bias and output dtype follow the issue: `do_bench_cudagraph`, 2 interleaved rounds (median) of AITER's live `tuned_gemm.gemm_a16w16` pick and Gluon v9 (`backend="gluon", kernel_type="compute_bound"`), same inputs, fp32 reference check.
- Columns are rebuilt by `derive.py`, whose rules reproduce all 630 rows of the issue's CSV (`python derive.py --validate`); the scope summary reproduces the issue's table.
- hipBLASLt kernel names come from the torch profiler (the issue used rocprofv3).
- The ROCm 7.2.4 runtime needs `AMD_COMGR_CACHE=0` in this container: with comgr's cache on, it compiles its blit kernels without the device libraries and every GPU call crashes. No fallback was needed; the whole stack (HIP runtime, hipBLASLt, rocBLAS, hipcc for AITER's JIT) is ROCm 7.2.4.
- Both environments use their own AITER JIT and Triton cache directories, so `/opt/venv` is untouched. Everything ran on GPU 0 (PCI bus 117 in both), one environment at a time.

