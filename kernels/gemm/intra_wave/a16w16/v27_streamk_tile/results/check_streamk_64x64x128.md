tile 64x64x128: kernel BLOCK 64x64x128, 4 waves, 256 programs

| shape | configuration | STREAMK_NUM_PROGRAMS | tile order | bias | kernel | launches | wrong results | flags or counts left up | largest error |
|---|---|---|---|---|---|---|---|---|---|
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.128 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.127 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.121 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.128 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.156 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.138 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.147 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.146 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.161 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.174 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.144 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.157 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.139 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.187 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.131 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.127 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.121 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.136 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.128 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.122 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.148 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.129 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.253 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.193 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.203 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.209 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.378 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.420 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.204 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.209 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.226 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.203 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.393 |
| 1088x1024x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.478 |
| 1088x1024x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.173 |
| 1088x1024x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.161 |
| 1088x1024x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1024x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1088x1024x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.129 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.136 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.135 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.126 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.172 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.162 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.154 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.136 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.179 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.164 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.158 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.142 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.130 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.132 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.153 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.162 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.144 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.133 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.154 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.129 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.130 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.136 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.133 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.161 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.164 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.133 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.129 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.128 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.128 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.121 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.150 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.144 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.149 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.226 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.222 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.283 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.236 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.320 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.291 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.213 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.199 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.237 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.227 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.301 |
| 1088x1088x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.310 |
| 1088x1088x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.235 |
| 1088x1088x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.148 |
| 1088x1088x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x1088x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.134 |
| 1088x1088x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.157 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 18 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 18 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.119 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.148 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.121 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 18 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.120 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.122 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.122 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.159 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.122 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.126 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.135 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 18 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.121 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.126 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.121 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.121 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.113 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.203 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.207 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.250 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.201 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.548 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.574 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.190 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.181 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.233 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.185 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.491 |
| 832x1280x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.478 |
| 832x1280x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.170 |
| 832x1280x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.146 |
| 832x1280x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x1280x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 832x1280x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 1024x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1024x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1024x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1024x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1024x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1024x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1024x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1024x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1024x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1024x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1024x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1024x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1024x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.223 |
| 1024x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.181 |
| 1024x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.187 |
| 1024x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.167 |
| 1024x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.200 |
| 1024x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.175 |
| 1024x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.199 |
| 1024x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.171 |
| 1024x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.192 |
| 1024x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.179 |
| 1024x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.173 |
| 1024x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.187 |
| 1024x1536x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1024x1536x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1024x1536x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1024x1536x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1024x1536x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.186 |
| 1024x1536x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.144 |
| 1024x1536x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.181 |
| 1024x1536x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.141 |
| 1024x1536x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.199 |
| 1024x1536x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.192 |
| 1024x1536x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.169 |
| 1024x1536x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.166 |
| 1024x1536x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1024x1536x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1024x1536x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.184 |
| 1024x1536x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.154 |
| 1024x1536x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1024x1536x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1024x1536x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.154 |
| 1024x1536x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.157 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.119 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.161 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.131 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.140 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.114 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.145 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.145 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.131 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.153 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.128 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.150 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.149 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.129 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.138 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.156 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.157 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.121 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.133 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.167 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.128 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.112 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.128 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.127 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.127 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.126 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.132 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.153 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.154 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.141 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.137 |
| 250x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.141 |
| 250x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.113 |
| 250x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.127 |
| 250x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.122 |
| 250x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.120 |
| 250x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.146 |
| 250x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.116 |
| 250x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.121 |
| 250x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.114 |
| 250x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.115 |
| 250x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.120 |
| 250x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.122 |
| 250x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.132 |
| 250x500x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 250x500x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 250x500x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 250x500x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 250x500x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 250x500x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 250x500x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 250x500x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 250x500x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 250x500x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 250x500x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 250x500x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 250x500x8080 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.117 |
| 250x500x8080 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.115 |
| 250x500x8080 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x500x8080 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 250x500x8080 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.117 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.150 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.121 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.121 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.150 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.163 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.155 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.129 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.127 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.141 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.143 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.149 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.129 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.143 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.131 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.139 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.138 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.173 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.122 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.143 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.131 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.147 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.121 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.122 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.126 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 1075x1000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 1075x1000x8176 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.165 |
| 1075x1000x8176 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.163 |
| 1075x1000x8176 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x1000x8176 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.122 |
| 1075x1000x8176 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.142 |

FAIL
