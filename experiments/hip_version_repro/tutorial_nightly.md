# Tutorial shapes: TheRock nightly torch 2.15.0a0 + ROCm 10.2.0a20260929

`env=nightly torch=2.15.0a0+rocm10.2.0a20260929 hip=7.17.26386 hipblaslt=100500(6054c511) triton=3.8.0@/home/jbowley/repos/triton_gfx950-tutorial-v2.2/python/triton gpu_pci_bus=117 gpu=gfx950:sramecc+:xnack- cus=256`

TFLOPS, `bench_gemm_a16w16.py` (hipBLASLt: `F.linear` with bias, `do_bench_cudagraph`), one process per measurement, median of 3 interleaved rounds, GPU 0.

| M×N×K | Triton backend | Gluon, no plugin | Gluon + llirSched | hipBLASLt (F.linear) |
|---|---:|---:|---:|---:|
| 4096×4096×8192 | 1281 | 1344 | 1580 | 1499 |
| 8192×8192×8192 | 1255 | 1328 | 1579 | 1565 |
| 4096×8192×4096 | 1218 | 1308 | 1526 | 1472 |
| 4096×106496×16384 | 1207 | 1297 | 1593 | 1458 |

Individual rounds:

- 4096×4096×8192 Triton backend: 1279, 1281, 1283
- 4096×4096×8192 Gluon, no plugin: 1344, 1346, 1340
- 4096×4096×8192 Gluon + llirSched: 1588, 1580, 1563
- 4096×4096×8192 hipBLASLt (F.linear): 1508, 1497, 1499
- 8192×8192×8192 Triton backend: 1259, 1253, 1255
- 8192×8192×8192 Gluon, no plugin: 1328, 1331, 1328
- 8192×8192×8192 Gluon + llirSched: 1579, 1576, 1579
- 8192×8192×8192 hipBLASLt (F.linear): 1568, 1565, 1564
- 4096×8192×4096 Triton backend: 1220, 1218, 1218
- 4096×8192×4096 Gluon, no plugin: 1309, 1296, 1308
- 4096×8192×4096 Gluon + llirSched: 1521, 1526, 1526
- 4096×8192×4096 hipBLASLt (F.linear): 1472, 1472, 1471
- 4096×106496×16384 Triton backend: 1211, 1207, 1198
- 4096×106496×16384 Gluon, no plugin: 1295, 1309, 1297
- 4096×106496×16384 Gluon + llirSched: 1593, 1594, 1591
- 4096×106496×16384 hipBLASLt (F.linear): 1459, 1457, 1458
