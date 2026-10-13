| case | variant | correct | VGPR | VGPR spill | SGPR spill | scratch ops (kloop/tile/other) |
|---|---|---|---|---|---|---|
| 128x98304x8176 | base | yes | 512 | 11 | 50 | 0/5/27 |
| 128x98304x8176 | final+nopers | yes | 512 | 12 | 19 | 0/3/15 |
| 128x98304x8192 | base | yes | 500 | 0 | 43 | 0/0/0 |
| 128x98304x8192 | final+nopers | yes | 476 | 0 | 11 | 0/0/0 |
| 98304x128x8080 | base | yes | 512 | 12 | 50 | 0/6/28 |
| 98304x128x8080 | final+nopers | yes | 512 | 6 | 9 | 0/0/12 |
| 200x131072x8080 | base | yes | 512 | 2 | 37 | 0/2/2 |
| 200x131072x8080 | final+nopers | yes | 492 | 0 | 6 | 0/0/0 |
| 4300x4000x8080:STREAMK_POLICY=two_tile | base | yes | 512 | 34 | 66 | 0/73/40 |
| 4300x4000x8080:STREAMK_POLICY=two_tile | final+nopers | yes | 512 | 15 | 31 | 0/66/15 |
| 4352x4096x8192:STREAMK_POLICY=two_tile | base | yes | 512 | 14 | 21 | 0/0/43 |
| 4352x4096x8192:STREAMK_POLICY=two_tile | final+nopers | yes | 512 | 14 | 18 | 0/29/14 |
