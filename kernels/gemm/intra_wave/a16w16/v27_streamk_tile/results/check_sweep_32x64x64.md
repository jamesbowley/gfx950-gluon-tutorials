tile 32x64x64: kernel BLOCK 32x64x64, 1 waves, 1024 programs

| shape | configuration | STREAMK_NUM_PROGRAMS | tile order | bias | kernel | launches | wrong results | flags or counts left up | largest error |
|---|---|---|---|---|---|---|---|---|---|
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 256 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 288 | v9 | 0 | v27_streamk_tile (grid 288) | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 256 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 288 | v9 | 0 | v27_streamk_tile (grid 288) | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 252 | v9 | 0 | v27_streamk_tile (grid 252) | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 283 | v9 | 0 | v27_streamk_tile (grid 283) | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 252 | v9 | 0 | v27_streamk_tile (grid 252) | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 283 | v9 | 0 | v27_streamk_tile (grid 283) | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 512 | v9 | 0 | v27_streamk_tile (grid 512) | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 576 | v9 | 0 | v27_streamk_tile (grid 576) | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 512 | v9 | 0 | v27_streamk_tile (grid 512) | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 576 | v9 | 0 | v27_streamk_tile (grid 576) | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 252 | v9 | 0 | v27_streamk_tile (grid 252) | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 283 | v9 | 0 | v27_streamk_tile (grid 283) | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 252 | v9 | 0 | v27_streamk_tile (grid 252) | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 283 | v9 | 0 | v27_streamk_tile (grid 283) | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 528x4080x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 256 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 528x4080x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 288 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 528x4080x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 528x4080x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 256 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 528x4080x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 288 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 528x4080x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 528x4080x8080 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 528x4080x8080 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |

PASS
