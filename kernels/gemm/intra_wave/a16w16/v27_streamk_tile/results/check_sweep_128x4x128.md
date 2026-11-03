tile 128x4x128: kernel BLOCK 4x128x128, 2 waves, 512 programs, swapped (C^T = B^T A^T)

| shape | configuration | STREAMK_NUM_PROGRAMS | tile order | bias | kernel | launches | wrong results | flags or counts left up | largest error |
|---|---|---|---|---|---|---|---|---|---|
| 4096x16x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 512 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4096x16x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4096x16x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 512 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4096x16x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4096x16x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4096x16x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4000x30x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4000x30x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4000x30x4080 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4000x30x4080 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 8192x1x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 256 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 8192x1x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 288 | v9 | 0 | v27_streamk_tile (grid 288) | 10 | 0 | 0 | 0.000 |
| 8192x1x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 8192x1x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 256 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 8192x1x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 288 | v9 | 0 | v27_streamk_tile (grid 288) | 10 | 0 | 0 | 0.000 |
| 8192x1x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 8192x1x4096 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 8192x1x4096 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4000x32x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4000x32x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4000x32x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4000x32x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4064x66x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4064x66x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4064x66x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4064x66x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4064x66x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4064x66x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4064x66x8080 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4064x66x8080 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |

PASS
