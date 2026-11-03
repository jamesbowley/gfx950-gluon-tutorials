matmul_kernel.py sha256 1175b73525d3cc3f00e47159868c8eb109337e2b3f8df9ba0ff89c6d717103a7

tile 64x128x128: kernel BLOCK 64x128x128, 4 waves, 256 programs

| shape | configuration | STREAMK_NUM_PROGRAMS | tile order | bias | kernel | launches | wrong results | flags or counts left up | largest error |
|---|---|---|---|---|---|---|---|---|---|
| 1088x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1088x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.140 |
| 1088x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.135 |
| 1088x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.164 |
| 1088x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.140 |
| 1088x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.149 |
| 1088x2048x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 1088x2048x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1088x2048x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.134 |
| 1088x2048x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1088x2048x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1088x2048x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.127 |
| 1088x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.147 |
| 1088x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.162 |
| 1088x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.147 |
| 1088x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.147 |
| 1088x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1088x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 1088x2176x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1088x2176x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.160 |
| 1088x2176x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.142 |
| 1088x2176x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.151 |
| 1088x2176x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.127 |
| 1088x2176x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.141 |
| 832x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 832x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 832x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 832x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 832x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.156 |
| 832x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 832x2560x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 832x2560x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.138 |
| 832x2560x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 832x2560x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 832x2560x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 832x2560x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.122 |
| 1024x3072x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.207 |
| 1024x3072x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.207 |
| 1024x3072x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.171 |
| 1024x3072x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.169 |
| 250x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.133 |
| 250x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.138 |
| 250x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.154 |
| 250x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.136 |
| 250x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.156 |
| 250x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.154 |
| 250x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.163 |
| 250x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.171 |
| 250x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.133 |
| 250x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.116 |
| 250x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.131 |
| 250x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.120 |
| 1075x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.172 |
| 1075x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 1075x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.138 |
| 1075x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.128 |
| 1075x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.181 |
| 1075x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1075x2000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1075x2000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.136 |
| 1075x2000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1075x2000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1075x2000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.146 |
| 1075x2000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.129 |

PASS
