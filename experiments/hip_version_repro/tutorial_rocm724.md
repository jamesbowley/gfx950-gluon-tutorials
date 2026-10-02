# Tutorial shapes: torch 2.10.0+rocm7.2.4 (rerun here)

`env=rocm724 torch=2.10.0+rocm7.2.4.git3d3aa833 hip=7.2.53211 hipblaslt=100202(dabb6df2b9) triton=3.8.0@/home/jbowley/repos/triton_gfx950-tutorial-v2.2/python/triton gpu_pci_bus=117 gpu=gfx950:sramecc+:xnack- cus=256`

TFLOPS, `bench_gemm_a16w16.py` (hipBLASLt: `F.linear` with bias, `do_bench_cudagraph`), one process per measurement, median of 3 interleaved rounds, GPU 0.

| M×N×K | Triton backend | Gluon, no plugin | Gluon + llirSched | hipBLASLt (F.linear) |
|---|---:|---:|---:|---:|
| 4096×4096×8192 | 1280 | 1347 | 1579 | 1505 |
| 8192×8192×8192 | 1256 | 1333 | 1574 | 1565 |
| 4096×8192×4096 | 1217 | 1305 | 1523 | 1478 |
| 4096×106496×16384 | 1205 | 1307 | 1592 | 1504 |

Individual rounds:

- 4096×4096×8192 Triton backend: 1281, 1280, 1279
- 4096×4096×8192 Gluon, no plugin: 1342, 1347, 1347
- 4096×4096×8192 Gluon + llirSched: 1586, 1579, 1576
- 4096×4096×8192 hipBLASLt (F.linear): 1505, 1503, 1505
- 8192×8192×8192 Triton backend: 1256, 1257, 1256
- 8192×8192×8192 Gluon, no plugin: 1333, 1331, 1333
- 8192×8192×8192 Gluon + llirSched: 1574, 1576, 1573
- 8192×8192×8192 hipBLASLt (F.linear): 1566, 1563, 1565
- 4096×8192×4096 Triton backend: 1220, 1214, 1217
- 4096×8192×4096 Gluon, no plugin: 1305, 1312, 1303
- 4096×8192×4096 Gluon + llirSched: 1524, 1516, 1523
- 4096×8192×4096 hipBLASLt (F.linear): 1481, 1478, 1478
- 4096×106496×16384 Triton backend: 1205, 1205, 1209
- 4096×106496×16384 Gluon, no plugin: 1301, 1312, 1307
- 4096×106496×16384 Gluon + llirSched: 1592, 1593, 1591
- 4096×106496×16384 hipBLASLt (F.linear): 1504, 1504, 1501
