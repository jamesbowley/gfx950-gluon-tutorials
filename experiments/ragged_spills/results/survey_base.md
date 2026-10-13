| case | variant | correct | VGPR | VGPR spill | SGPR spill | scratch ops (kloop/tile/other) | spill sites |
|---|---|---|---|---|---|---|---|
| 4300x4000x8080 | base | yes | 512 | 9 | 50 | 0/3/15 | base.py:0 x9, base.py:1927 x6, base.py:501 x2, base.py:1597 x1 |
| 4352x4096x8080 | base | yes | 496 | 0 | 12 | 0/0/0 |  |
| 4300x4000x8192 | base | yes | 496 | 0 | 45 | 0/0/0 |  |
| 4352x4096x8080:STREAMK_FORCE_EDGE=1 | base | yes | 512 | 9 | 50 | 0/3/15 | base.py:0 x9, base.py:1927 x6, base.py:501 x2, base.py:1597 x1 |
| 1000x2000x4032 | base | yes | 512 | 9 | 43 | 0/3/15 | base.py:0 x10, base.py:1927 x5, base.py:501 x2, base.py:1597 x1 |
| 128x8192x8176 | base | yes | 512 | 11 | 47 | 0/6/16 | base.py:0 x10, base.py:1927 x6, base.py:501 x5, base.py:1597 x1 |
| 1000x2000x4096 | base | yes | 496 | 0 | 43 | 0/0/0 |  |
| 4300x4000x8080:STREAMK_POLICY=dp | base | yes | 512 | 2 | 37 | 0/2/2 | base.py:501 x2, base.py:375 x1, base.py:1927 x1 |
| 4300x4000x8080:STREAMK_FIXUP=owner | base | yes | 512 | 12 | 50 | 0/6/28 | base.py:0 x11, base.py:1927 x8, base.py:452 x8, base.py:501 x5, base.py:1597 x1, base.py:1244 x1 |
| 4300x4000x8080:STREAMK_POLICY=two_tile | base | yes | 512 | 34 | 66 | 0/73/40 | base.py:0 x36, base.py:490 x20, base.py:452 x20, base.py:1966 x12, base.py:1963 x10, base.py:1489 x5, base.py:501 x5, base.py:697 x4, base.py:1587 x1 |
