matmul_kernel.py sha256 1175b73525d3cc3f00e47159868c8eb109337e2b3f8df9ba0ff89c6d717103a7

tile 64x256x64: kernel BLOCK 64x256x64, 4 waves, 256 programs

| shape | configuration | STREAMK_NUM_PROGRAMS | tile order | bias | kernel | launches | wrong results | flags or counts left up | largest error |
|---|---|---|---|---|---|---|---|---|---|
| 1088x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.155 |
| 1088x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.185 |
| 1088x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 1088x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.128 |
| 1088x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.154 |
| 1088x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.135 |
| 1088x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.149 |
| 1088x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 1088x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.151 |
| 1088x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.149 |
| 1088x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.145 |
| 1088x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.133 |
| 1088x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.154 |
| 1088x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.185 |
| 1088x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.127 |
| 1088x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.164 |
| 1088x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.135 |
| 1088x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.146 |
| 1088x4352x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.148 |
| 1088x4352x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.131 |
| 1088x4352x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.142 |
| 1088x4352x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.158 |
| 1088x4352x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.129 |
| 1088x4352x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.148 |
| 832x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 832x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.130 |
| 832x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 832x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.131 |
| 832x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.127 |
| 832x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 832x5120x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 832x5120x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 832x5120x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 832x5120x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.146 |
| 832x5120x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 832x5120x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 1024x6144x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.227 |
| 1024x6144x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.214 |
| 1024x6144x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.173 |
| 1024x6144x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.163 |
| 250x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.163 |
| 250x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.158 |
| 250x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.157 |
| 250x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.163 |
| 250x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.138 |
| 250x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.142 |
| 250x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.151 |
| 250x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.138 |
| 250x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.134 |
| 250x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.147 |
| 250x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.156 |
| 250x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.136 |
| 1075x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.155 |
| 1075x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.151 |
| 1075x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 1075x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.138 |
| 1075x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.144 |
| 1075x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 1075x4000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.142 |
| 1075x4000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.139 |
| 1075x4000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.137 |
| 1075x4000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.147 |
| 1075x4000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 1075x4000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |

PASS
