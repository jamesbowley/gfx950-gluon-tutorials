matmul_kernel.py sha256 1175b73525d3cc3f00e47159868c8eb109337e2b3f8df9ba0ff89c6d717103a7

tile 16x32x128: kernel BLOCK 16x32x128, 1 waves, 1024 programs

| shape | configuration | STREAMK_NUM_PROGRAMS | tile order | bias | kernel | launches | wrong results | flags or counts left up | largest error |
|---|---|---|---|---|---|---|---|---|---|
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 512 | v9 | 0 | v27_streamk_tile (grid 512) | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 576 | v9 | 0 | v27_streamk_tile (grid 576) | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 512 | v9 | 0 | v27_streamk_tile (grid 512) | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 576 | v9 | 0 | v27_streamk_tile (grid 576) | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 1000 | v9 | 0 | v27_streamk_tile (grid 1000) | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 1000 | v9 | 0 | v27_streamk_tile (grid 1000) | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 1024 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 1024 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 1000 | v9 | 0 | v27_streamk_tile (grid 1000) | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 1000 | v9 | 0 | v27_streamk_tile (grid 1000) | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 136x4088x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 512 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 136x4088x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 576 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 136x4088x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 136x4088x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 512 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 136x4088x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 576 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 136x4088x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 136x4088x8080 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 136x4088x8080 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |

PASS
