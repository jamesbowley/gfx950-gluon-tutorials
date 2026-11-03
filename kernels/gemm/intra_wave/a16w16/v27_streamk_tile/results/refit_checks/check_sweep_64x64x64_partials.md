matmul_kernel.py sha256 1175b73525d3cc3f00e47159868c8eb109337e2b3f8df9ba0ff89c6d717103a7

tile 64x64x64: kernel BLOCK 64x64x64, 4 waves, 256 programs

| shape | configuration | STREAMK_NUM_PROGRAMS | tile order | bias | kernel | launches | wrong results | flags or counts left up | largest error |
|---|---|---|---|---|---|---|---|---|---|
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.137 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.150 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.119 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.152 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.120 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.152 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.121 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1024x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.235 |
| 1024x1536x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.167 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.128 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.131 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.132 |
| 250x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.122 |
| 250x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.109 |
| 250x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.111 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.134 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.154 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |

PASS
