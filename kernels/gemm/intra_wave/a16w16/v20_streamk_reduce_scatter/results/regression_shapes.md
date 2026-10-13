> v20 here is STREAMK_POLICY=one_tile STREAMK_FIXUP=rs, the defaults before WORKLOG entry 2.

dtype=bf16 bias=False HIP_VISIBLE_DEVICES=2

| shape (MxNxK) | version | correct | TFLOPS | vs first | VGPR | AGPR | VGPR spill | SGPR spill | scratch ops (kloop/tile/other) |
|---|---|---|---|---|---|---|---|---|---|
| 4096x4096x8192 | v19 | yes | 1570.4 |  | 480 | 256 | 0 | 0 | 0/0/0 |
| 4096x4096x8192 | v20 | yes | 1560.7 | -0.6% | 480 | 256 | 0 | 0 | 0/0/0 |
| 8192x8192x8192 | v19 | yes | 1622.4 |  | 480 | 256 | 0 | 0 | 0/0/0 |
| 8192x8192x8192 | v20 | yes | 1651.5 | +1.8% | 480 | 256 | 0 | 0 | 0/0/0 |
| 4096x8192x4096 | v19 | yes | 1546.7 |  | 480 | 256 | 0 | 0 | 0/0/0 |
| 4096x8192x4096 | v20 | yes | 1550.9 | +0.3% | 480 | 256 | 0 | 0 | 0/0/0 |
| 4096x106496x16384 | v19 | yes | 1690.7 |  | 480 | 256 | 0 | 0 | 0/0/0 |
| 4096x106496x16384 | v20 | yes | 1685.3 | -0.3% | 480 | 256 | 0 | 0 | 0/0/0 |
