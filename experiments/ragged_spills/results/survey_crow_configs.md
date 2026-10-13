| case | variant | correct | VGPR | VGPR spill | SGPR spill | scratch ops (kloop/tile/other) |
|---|---|---|---|---|---|---|
| 4300x4000x8080:STREAMK_POLICY=two_tile | base | yes | 512 | 34 | 66 | 0/73/40 |
| 4300x4000x8080:STREAMK_POLICY=two_tile | crow | yes | 512 | 27 | 66 | 0/73/32 |
| 4300x4000x8080:STREAMK_POLICY=spread | base | yes | 512 | 34 | 67 | 0/73/40 |
| 4300x4000x8080:STREAMK_POLICY=spread | crow | yes | 512 | 27 | 67 | 0/73/32 |
| 4300x4000x8080:STREAMK_FIXUP=atomic_iz | base | NO | 512 | 10 | 145 | 0/4/16 |
| 4300x4000x8080:STREAMK_FIXUP=atomic_iz | crow | NO | 512 | 6 | 145 | 0/0/12 |
| 4300x4000x8080:STREAMK_EXCHANGE=l2 | base | yes | 512 | 10 | 48 | 0/3/11 |
| 4300x4000x8080:STREAMK_EXCHANGE=l2 | crow | yes | 512 | 9 | 48 | 0/2/10 |
| 4300x4000x8080:STREAMK_PLACEMENT=chunk | base | yes | 512 | 9 | 44 | 0/3/15 |
| 4300x4000x8080:STREAMK_PLACEMENT=chunk | crow | yes | 512 | 5 | 44 | 0/0/10 |
| 4300x4000x8080:STREAMK_PARTIALS=bf16_pairs | base | yes | 512 | 10 | 50 | 0/3/11 |
| 4300x4000x8080:STREAMK_PARTIALS=bf16_pairs | crow | yes | 512 | 9 | 49 | 0/2/10 |
| 8150x8400x8080 | base | yes | 512 | 9 | 47 | 0/3/15 |
| 8150x8400x8080 | crow | yes | 512 | 5 | 47 | 0/0/10 |
| 4000x6080x8080 | base | yes | 512 | 12 | 50 | 0/6/28 |
| 4000x6080x8080 | crow | yes | 512 | 5 | 50 | 0/0/10 |
| 4300x4000x8080:STREAMK_FORCE_EDGE=1 | base | yes | 512 | 9 | 50 | 0/3/15 |
| 4300x4000x8080:STREAMK_FORCE_EDGE=1 | crow | yes | 512 | 5 | 49 | 0/0/10 |
