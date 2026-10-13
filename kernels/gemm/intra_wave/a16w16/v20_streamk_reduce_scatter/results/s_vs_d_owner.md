> v20 here is STREAMK_POLICY=one_tile STREAMK_FIXUP=owner: v17's ending in the v20 build (control).

dtype=bf16 bias=False HIP_VISIBLE_DEVICES=2

| shape (MxNxK) | version | correct | TFLOPS | vs first | VGPR | AGPR | VGPR spill | SGPR spill | scratch ops (kloop/tile/other) |
|---|---|---|---|---|---|---|---|---|---|
| 4096x16640x8192 | v20 | yes | 1544.7 |  | 492 | 256 | 0 | 4 | 0/0/0 |
| 4096x4608x4096 | v20 | yes | 1083.2 |  | 492 | 256 | 0 | 4 | 0/0/0 |
| 4096x4608x8192 | v20 | yes | 1320.3 |  | 492 | 256 | 0 | 4 | 0/0/0 |
| 4096x4608x16384 | v20 | yes | 1439.7 |  | 492 | 256 | 0 | 4 | 0/0/0 |
| 4352x4352x4096 | v20 | yes | 1103.8 |  | 492 | 256 | 0 | 4 | 0/0/0 |
| 4352x4352x16384 | v20 | yes | 1286.8 |  | 492 | 256 | 0 | 4 | 0/0/0 |
| 4096x5120x4096 | v20 | yes | 1232.4 |  | 492 | 256 | 0 | 4 | 0/0/0 |
| 4096x5120x16384 | v20 | yes | 1510.2 |  | 492 | 256 | 0 | 4 | 0/0/0 |
