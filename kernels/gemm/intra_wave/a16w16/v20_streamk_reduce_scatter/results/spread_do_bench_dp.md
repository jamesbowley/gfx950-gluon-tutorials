> v20 here is STREAMK_POLICY=dp (WORKLOG entry 3).

dtype=bf16 bias=False HIP_VISIBLE_DEVICES=2

| shape (MxNxK) | version | correct | TFLOPS | vs first | VGPR | AGPR | VGPR spill | SGPR spill | scratch ops (kloop/tile/other) |
|---|---|---|---|---|---|---|---|---|---|
| 4096x7168x8192 | v20 | yes | 1504.2 |  | 480 | 256 | 0 | 0 | 0/0/0 |
| 4096x7168x16384 | v20 | yes | 1517.9 |  | 480 | 256 | 0 | 0 | 0/0/0 |
| 4096x7168x32768 | v20 | yes | 1460.8 |  | 480 | 256 | 0 | 0 | 0/0/0 |
| 4096x7168x65536 | v20 | yes | 1353.3 |  | 480 | 256 | 0 | 0 | 0/0/0 |
| 8192x7936x32768 | v20 | yes | 1593.1 |  | 480 | 256 | 0 | 0 | 0/0/0 |
| 4096x7936x32768 | v20 | yes | 1559.1 |  | 480 | 256 | 0 | 0 | 0/0/0 |
| 4096x6144x8192 | v20 | yes | 1318.8 |  | 480 | 256 | 0 | 0 | 0/0/0 |
