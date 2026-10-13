## round 1: ragged, v25 (both flags on whenever either dimension is ragged)
| shape (MxNxK) | version | correct | us | TFLOPS | vs first | VGPR | AGPR | VGPR spill | SGPR spill | scratch ops (kloop/tile/other) |
|---|---|---|---|---|---|---|---|---|---|---|
| 8100x8192x8192 | v25 | yes | 669.6 | 1623.5 |  | 492 | 256 | 0 | 36 | 0/0/0 |
| 8192x8128x8192 | v25 | yes | 675.2 | 1615.7 |  | 492 | 256 | 0 | 36 | 0/0/0 |
| 4300x4000x8192 | v25 | yes | 204.6 | 1377.6 |  | 496 | 256 | 0 | 45 | 0/0/0 |
| 8150x8400x8192 | v25 | yes | 724.3 | 1548.7 |  | 496 | 256 | 0 | 43 | 0/0/0 |
| 4000x6080x8192 | v25 | yes | 266.8 | 1493.5 |  | 500 | 256 | 0 | 43 | 0/0/0 |
| 128x8192x8192 | v25 | yes | 76.6 | 224.2 |  | 496 | 256 | 0 | 43 | 0/0/0 |

## round 2: ragged, v25 (both flags on whenever either dimension is ragged)
| shape (MxNxK) | version | correct | us | TFLOPS | vs first | VGPR | AGPR | VGPR spill | SGPR spill | scratch ops (kloop/tile/other) |
|---|---|---|---|---|---|---|---|---|---|---|
| 8100x8192x8192 | v25 | yes | 670.6 | 1621.1 |  | 492 | 256 | 0 | 36 | 0/0/0 |
| 8192x8128x8192 | v25 | yes | 676.5 | 1612.6 |  | 492 | 256 | 0 | 36 | 0/0/0 |
| 4300x4000x8192 | v25 | yes | 204.1 | 1380.6 |  | 496 | 256 | 0 | 45 | 0/0/0 |
| 8150x8400x8192 | v25 | yes | 729.4 | 1537.8 |  | 496 | 256 | 0 | 43 | 0/0/0 |
| 4000x6080x8192 | v25 | yes | 268.1 | 1486.2 |  | 500 | 256 | 0 | 43 | 0/0/0 |
| 128x8192x8192 | v25 | yes | 76.7 | 223.9 |  | 496 | 256 | 0 | 43 | 0/0/0 |

