| case | variant | correct | VGPR | VGPR spill | SGPR spill | scratch ops (kloop/tile/other) |
|---|---|---|---|---|---|---|
| 4352x4096x8192 | base | yes | 488 | 0 | 4 | 0/0/0 |
| 4352x4096x8192:CASE_BIAS=1 | base | yes | 496 | 0 | 4 | 0/0/0 |
| 8192x8192x8192 | base | yes | 480 | 0 | 0 | 0/0/0 |
| 8192x8192x8192:CASE_BIAS=1 | base | yes | 488 | 0 | 0 | 0/0/0 |
| 8192x8448x8192 | base | yes | 488 | 0 | 4 | 0/0/0 |
| 4096x6144x8192 | base | yes | 492 | 0 | 4 | 0/0/0 |
| 4096x6144x8192:CASE_BIAS=1 | base | yes | 496 | 0 | 4 | 0/0/0 |
| 3328x5120x8192 | base | yes | 488 | 0 | 4 | 0/0/0 |
| 4352x4352x8192 | base | yes | 492 | 0 | 4 | 0/0/0 |
| 1024x2048x4096 | base | yes | 488 | 0 | 4 | 0/0/0 |
| 256x8192x8192 | base | yes | 488 | 0 | 4 | 0/0/0 |
| 131072x256x8192 | base | yes | 480 | 0 | 0 | 0/0/0 |
| 128x98304x8192 | base | yes | 500 | 0 | 43 | 0/0/0 |
| 4352x4096x2048 | base | yes | 488 | 0 | 4 | 0/0/0 |
| 4352x4096x16384 | base | yes | 488 | 0 | 4 | 0/0/0 |
| 4352x4096x8192:STREAMK_FIXUP=owner | base | yes | 492 | 0 | 4 | 0/0/0 |
| 4352x4096x8192:STREAMK_FIXUP=atomic_iz | base | NO | 484 | 0 | 0 | 0/0/0 |
| 4352x4096x8192:STREAMK_EXCHANGE=l2 | base | yes | 488 | 0 | 4 | 0/0/0 |
| 4352x4096x8192:STREAMK_PARTIALS=bf16_pairs | base | yes | 488 | 0 | 4 | 0/0/0 |
| 4352x4096x8192:STREAMK_PLACEMENT=chunk | base | yes | 488 | 0 | 4 | 0/0/0 |
| 4352x4096x8192:STREAMK_POLICY=dp | base | yes | 480 | 0 | 0 | 0/0/0 |
| 4352x4096x8192:STREAMK_POLICY=two_tile | base | yes | 512 | 14 | 21 | 0/0/43 |
| 4352x4096x8192:STREAMK_POLICY=spread | base | yes | 512 | 14 | 21 | 0/0/43 |
