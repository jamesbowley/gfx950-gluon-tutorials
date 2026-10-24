| shape | configuration | STREAMK_NUM_PROGRAMS | tile order | bias | launches | wrong results | flags or counts left up | largest error |
|---|---|---|---|---|---|---|---|---|
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 0 | 30 | 1 | 0 | 25.344 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 1 | 30 | 1 | 0 | 23.561 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 0 | 30 | 1 | 0 | 22.812 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 1 | 30 | 1 | 0 | 23.945 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 0 | 30 | 0 | 0 | 0.177 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 1 | 30 | 0 | 0 | 0.125 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 0 | 30 | 0 | 0 | 0.172 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 1 | 30 | 0 | 0 | 0.128 |

FAIL, as required (the negative control was detected)
