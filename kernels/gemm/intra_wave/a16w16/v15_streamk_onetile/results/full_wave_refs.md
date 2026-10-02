dtype=bf16 bias=False HIP_VISIBLE_DEVICES=2

| shape (MxNxK) | version | correct | TFLOPS | vs first | VGPR | AGPR | VGPR spill | SGPR spill | scratch ops (kloop/tile/other) |
|---|---|---|---|---|---|---|---|---|---|
| 4096x4096x4096 | v13 | yes | 1230.4 |  | 480 | 256 | 0 | 0 | 0/0/0 |
| 4096x4096x8192 | v13 | yes | 1495.2 |  | 480 | 256 | 0 | 0 | 0/0/0 |
| 4096x4096x16384 | v13 | yes | 1437.3 |  | 480 | 256 | 0 | 0 | 0/0/0 |
| 8192x6144x8192 | v13 | yes | 1603.1 |  | 480 | 256 | 0 | 0 | 0/0/0 |
| 8192x8192x8192 | v13 | yes | 1621.4 |  | 480 | 256 | 0 | 0 | 0/0/0 |
