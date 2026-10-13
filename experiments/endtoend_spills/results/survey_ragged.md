| case | variant | correct | VGPR | VGPR spill | SGPR spill | scratch ops (stores / loads) | SGPR lanes |
|---|---|---|---|---|---|---|---|
| 4300x4000x8176:STREAMK_POLICY=two_tile | base | yes | 512 | 15 | 28 | L1.post 0/65, L1.pre 0/1, out 15/0 | 28 |
| 4300x4000x8176:STREAMK_POLICY=two_tile | crow | yes | 512 | 13 | 29 | L1.post 0/47, L1.pre 0/1, out 13/0 | 29 |
| 4300x4000x8176:STREAMK_POLICY=spread | base | yes | 512 | 36 | 67 | L1.post 2/2, L2.post 0/73, out 28/6 | 67 |
| 4300x4000x8176:STREAMK_POLICY=spread | crow | yes | 512 | 35 | 67 | L1.post 2/2, L2.post 0/68, out 27/6 | 67 |
| 1000x2000x8080:STREAMK_POLICY=spread | base | yes | 512 | 15 | 29 | L1.post 0/65, L1.pre 0/1, out 15/0 | 29 |
| 1000x2000x8080:STREAMK_POLICY=spread | crow | yes | 512 | 13 | 28 | L1.post 0/47, L1.pre 0/1, out 13/0 | 28 |
| 4352x4096x8080:STREAMK_POLICY=two_tile | base | yes | 512 | 15 | 22 | L1.post 0/29, L1.pre 0/1, out 15/0 | 22 |
| 4352x4096x8080:STREAMK_POLICY=two_tile | crow | yes | 512 | 8 | 28 | L1.post 0/22, L1.pre 0/1, out 8/0 | 28 |
| 4300x4000x8176:STREAMK_POLICY=two_tile:CASE_BIAS=1 | base | yes | 512 | 29 | 32 | L1.post 2/87, L1.pre 0/1, out 21/0 | 32 |
| 4300x4000x8176:STREAMK_POLICY=two_tile:CASE_BIAS=1 | crow | yes | 512 | 45 | 32 | L1.post 6/87, L1.pre 0/1, out 21/0 | 32 |
