| case | variant | correct | VGPR | VGPR spill | SGPR spill | scratch ops (kloop/tile/other) | spill sites |
|---|---|---|---|---|---|---|---|
| 4300x4000x8080 | base | yes | 512 | 9 | 50 | 0/3/15 | base.py:0 x9, base.py:1927 x6, base.py:501 x2, base.py:1597 x1 |
| 4300x4000x8080 | kbcast | yes | 512 | 9 | 50 | 0/3/15 | kbcast.py:0 x9, kbcast.py:1925 x6, kbcast.py:499 x2, kbcast.py:1595 x1 |
| 4300x4000x8080 | kopaque | yes | 512 | 9 | 50 | 0/3/15 | kopaque.py:0 x10, kopaque.py:1928 x5, kopaque.py:501 x2, kopaque.py:1598 x1 |
| 4300x4000x8080 | crow | yes | 512 | 5 | 49 | 0/0/10 | crow.py:1930 x5, crow.py:0 x5 |
| 4300x4000x8080 | crow+kopaque | yes | 512 | 5 | 50 | 0/0/10 | crow+kopaque.py:1931 x5, crow+kopaque.py:0 x5 |
| 4300x4000x8080:STREAMK_POLICY=dp | base | yes | 512 | 2 | 37 | 0/2/2 | base.py:501 x2, base.py:375 x1, base.py:1927 x1 |
| 4300x4000x8080:STREAMK_POLICY=dp | kbcast | yes | 512 | 2 | 37 | 0/2/2 | kbcast.py:499 x2, kbcast.py:375 x1, kbcast.py:1925 x1 |
| 4300x4000x8080:STREAMK_POLICY=dp | kopaque | yes | 512 | 2 | 36 | 0/2/2 | kopaque.py:501 x2, kopaque.py:375 x1, kopaque.py:1928 x1 |
| 4300x4000x8080:STREAMK_POLICY=dp | crow | yes | 512 | 0 | 35 | 0/0/0 |  |
| 4300x4000x8080:STREAMK_POLICY=dp | crow+kopaque | yes | 512 | 0 | 36 | 0/0/0 |  |
| 128x8192x8176 | base | yes | 512 | 11 | 47 | 0/6/16 | base.py:0 x10, base.py:1927 x6, base.py:501 x5, base.py:1597 x1 |
| 128x8192x8176 | kbcast | yes | 512 | 11 | 47 | 0/6/16 | kbcast.py:0 x10, kbcast.py:1925 x6, kbcast.py:499 x5, kbcast.py:1595 x1 |
| 128x8192x8176 | kopaque | yes | 512 | 11 | 52 | 0/6/16 | kopaque.py:0 x10, kopaque.py:1928 x6, kopaque.py:501 x5, kopaque.py:1598 x1 |
| 128x8192x8176 | crow | yes | 512 | 8 | 47 | 0/0/16 | crow.py:0 x11, crow.py:1930 x5 |
| 128x8192x8176 | crow+kopaque | yes | 512 | 8 | 52 | 0/0/16 | crow+kopaque.py:0 x11, crow+kopaque.py:1931 x5 |
| 1000x2000x4032 | base | yes | 512 | 9 | 43 | 0/3/15 | base.py:0 x10, base.py:1927 x5, base.py:501 x2, base.py:1597 x1 |
| 1000x2000x4032 | kbcast | yes | 512 | 9 | 43 | 0/3/15 | kbcast.py:0 x10, kbcast.py:1925 x5, kbcast.py:499 x2, kbcast.py:1595 x1 |
| 1000x2000x4032 | kopaque | yes | 512 | 9 | 43 | 0/3/15 | kopaque.py:0 x10, kopaque.py:1928 x5, kopaque.py:501 x2, kopaque.py:1598 x1 |
| 1000x2000x4032 | crow | yes | 512 | 5 | 43 | 0/0/10 | crow.py:1930 x5, crow.py:0 x5 |
| 1000x2000x4032 | crow+kopaque | yes | 512 | 5 | 43 | 0/0/10 | crow+kopaque.py:1931 x5, crow+kopaque.py:0 x5 |
| 4300x4000x8080:STREAMK_FIXUP=owner | base | yes | 512 | 12 | 50 | 0/6/28 | base.py:0 x11, base.py:1927 x8, base.py:452 x8, base.py:501 x5, base.py:1597 x1, base.py:1244 x1 |
| 4300x4000x8080:STREAMK_FIXUP=owner | kbcast | yes | 512 | 12 | 50 | 0/6/28 | kbcast.py:0 x11, kbcast.py:1925 x8, kbcast.py:450 x8, kbcast.py:499 x5, kbcast.py:1595 x1, kbcast.py:1242 x1 |
| 4300x4000x8080:STREAMK_FIXUP=owner | kopaque | yes | 512 | 12 | 46 | 0/6/28 | kopaque.py:0 x11, kopaque.py:1928 x8, kopaque.py:452 x8, kopaque.py:501 x5, kopaque.py:1598 x1, kopaque.py:1244 x1 |
| 4300x4000x8080:STREAMK_FIXUP=owner | crow | yes | 512 | 5 | 49 | 0/0/10 | crow.py:0 x5, crow.py:1930 x4, crow.py:375 x1 |
| 4300x4000x8080:STREAMK_FIXUP=owner | crow+kopaque | yes | 512 | 5 | 46 | 0/0/10 | crow+kopaque.py:0 x5, crow+kopaque.py:1931 x4, crow+kopaque.py:375 x1 |
| 4300x4000x8192 | base | yes | 496 | 0 | 45 | 0/0/0 |  |
| 4300x4000x8192 | kbcast | yes | 496 | 0 | 45 | 0/0/0 |  |
| 4300x4000x8192 | kopaque | yes | 496 | 0 | 45 | 0/0/0 |  |
| 4300x4000x8192 | crow | yes | 488 | 0 | 45 | 0/0/0 |  |
| 4300x4000x8192 | crow+kopaque | yes | 488 | 0 | 45 | 0/0/0 |  |
| 4352x4096x8080 | base | yes | 496 | 0 | 12 | 0/0/0 |  |
| 4352x4096x8080 | kbcast | yes | 496 | 0 | 12 | 0/0/0 |  |
| 4352x4096x8080 | kopaque | yes | 480 | 0 | 14 | 0/0/0 |  |
| 4352x4096x8080 | crow | yes | 496 | 0 | 12 | 0/0/0 |  |
| 4352x4096x8080 | crow+kopaque | yes | 480 | 0 | 14 | 0/0/0 |  |
