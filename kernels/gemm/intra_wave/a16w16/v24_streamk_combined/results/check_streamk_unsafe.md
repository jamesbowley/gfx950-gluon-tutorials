| shape | configuration | STREAMK_NUM_PROGRAMS | tile order | bias | launches | wrong results | flags or counts left up | largest error |
|---|---|---|---|---|---|---|---|---|
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 0 | 30 | 30 | 0 | 35.875 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 0 | 30 | 30 | 0 | 32.625 |

FAIL, as required (the negative control was detected)
