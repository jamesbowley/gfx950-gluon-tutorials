tile 128x128x128: kernel BLOCK 128x128x128, 4 waves, 256 programs, STREAMK_OWNER_OPT=bp

| shape | configuration | STREAMK_NUM_PROGRAMS | tile order | bias | kernel | launches | wrong results | flags or counts left up | largest error |
|---|---|---|---|---|---|---|---|---|---|
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner (grid 148) | 10 | 0 | 0 | 0.000 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner (grid 148) | 10 | 0 | 0 | 0.000 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner (grid 148) | 10 | 0 | 0 | 0.000 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner (grid 148) | 10 | 0 | 0 | 0.000 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 384x1408x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 384x1408x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 384x1408x8192 | POLICY=spread, PARTIALS=fp32 | 132 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x8192 | POLICY=spread, PARTIALS=fp32 | 132 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x8192 | POLICY=spread, PARTIALS=fp32 | 148 | v9 | 0 | v28_streamk_owner (grid 148) | 10 | 0 | 0 | 0.000 |
| 384x1408x8192 | POLICY=spread, PARTIALS=fp32 | 148 | v9 | 1 | v28_streamk_owner (grid 148) | 10 | 0 | 0 | 0.000 |
| 384x1408x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 384x1408x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=spread, PARTIALS=fp32 | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=spread, PARTIALS=fp32 | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=spread, PARTIALS=fp32 | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=spread, PARTIALS=fp32 | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x1024 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x1024 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x1024 | POLICY=spread, PARTIALS=fp32 | 132 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x1024 | POLICY=spread, PARTIALS=fp32 | 132 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x1024 | POLICY=spread, PARTIALS=fp32 | 148 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x1024 | POLICY=spread, PARTIALS=fp32 | 148 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x1024 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x1024 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |

PASS
