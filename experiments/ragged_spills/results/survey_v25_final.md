| case | variant | correct | VGPR | VGPR spill | SGPR spill | scratch ops (kloop/tile/other) |
|---|---|---|---|---|---|---|
| 4300x4000x8192 | v25_before | yes | 496 | 0 | 45 | 0/0/0 |
| 4300x4000x8192 | v25_final | yes | 488 | 0 | 45 | 0/0/0 |
| 4300x4000x8192:CASE_BIAS=1 | v25_before | yes | 512 | 4 | 44 | 0/0/8 |
| 4300x4000x8192:CASE_BIAS=1 | v25_final | yes | 512 | 4 | 44 | 0/0/8 |
| 8150x8400x8192 | v25_before | yes | 496 | 0 | 43 | 0/0/0 |
| 8150x8400x8192 | v25_final | yes | 488 | 0 | 43 | 0/0/0 |
| 4300x4000x8192:STREAMK_FIXUP=owner | v25_before | yes | 500 | 0 | 43 | 0/0/0 |
| 4300x4000x8192:STREAMK_FIXUP=owner | v25_final | yes | 492 | 0 | 43 | 0/0/0 |
| 1000x2000x8192 | v25_before | yes | 496 | 0 | 43 | 0/0/0 |
| 1000x2000x8192 | v25_final | yes | 488 | 0 | 43 | 0/0/0 |
| 128x8192x8192 | v25_before | yes | 496 | 0 | 43 | 0/0/0 |
| 128x8192x8192 | v25_final | yes | 488 | 0 | 43 | 0/0/0 |
| 131072x200x8192 | v25_before | yes | 512 | 12 | 336 | 0/6/0 |
| 131072x200x8192 | v25_final | yes | 512 | 11 | 323 | 0/6/4 |
| 4352x4096x8192 | v25_before | yes | 488 | 0 | 4 | 0/0/0 |
| 4352x4096x8192 | v25_final | yes | 488 | 0 | 4 | 0/0/0 |
