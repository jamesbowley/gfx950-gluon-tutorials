| shape | configuration | STREAMK_NUM_PROGRAMS | tile order | bias | launches | wrong results | flags or counts left up | largest error |
|---|---|---|---|---|---|---|---|---|
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | 10 | error: no result after 90 s (hang) | | |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | 10 | error: no result after 90 s (hang) | | |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | 10 | error: no result after 90 s (hang) | | |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | 10 | error: no result after 90 s (hang) | | |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | 10 | error: no result after 90 s (hang) | | |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | 10 | error: no result after 90 s (hang) | | |

FAIL, as required (the negative control was detected)
