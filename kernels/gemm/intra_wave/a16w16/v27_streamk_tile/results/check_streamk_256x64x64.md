tile 256x64x64: kernel BLOCK 256x64x64, 4 waves, 256 programs

| shape | configuration | STREAMK_NUM_PROGRAMS | tile order | bias | kernel | launches | wrong results | flags or counts left up | largest error |
|---|---|---|---|---|---|---|---|---|---|
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.164 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.160 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.130 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.137 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.134 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.166 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.144 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.142 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.167 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.145 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.155 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.148 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.160 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.145 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.139 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.136 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.136 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.129 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.147 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.157 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.164 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.162 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.148 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.128 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.160 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.154 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.141 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.126 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.164 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.157 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.127 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.129 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.149 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.136 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.130 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.175 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.148 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.152 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.156 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.153 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.145 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.215 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.251 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.261 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.248 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.408 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.480 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.240 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.224 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.248 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.254 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.449 |
| 4352x1024x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.426 |
| 4352x1024x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.222 |
| 4352x1024x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.229 |
| 4352x1024x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1024x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 4352x1024x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.183 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.192 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.153 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.150 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.141 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.126 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.148 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.148 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.165 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.147 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.158 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.143 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.150 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.180 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.146 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.154 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.138 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.159 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.178 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.146 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.167 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.166 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.139 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.169 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.168 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.153 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.179 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.134 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.151 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.149 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.189 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.151 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.154 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.189 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.158 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.146 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.147 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.158 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.159 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.143 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.150 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.133 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.156 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.150 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.129 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.140 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.135 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.140 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.262 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.228 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.267 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.290 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.328 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.321 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.221 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.243 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.260 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.237 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.319 |
| 4352x1088x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.312 |
| 4352x1088x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.174 |
| 4352x1088x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.192 |
| 4352x1088x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4352x1088x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.134 |
| 4352x1088x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.135 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 18 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 18 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.134 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.122 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.129 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.126 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 18 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.126 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.137 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.142 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.157 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.126 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.138 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.130 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 18 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.165 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.137 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.141 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.146 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.216 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.200 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.224 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.233 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.712 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.831 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.196 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.194 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.252 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.237 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.807 |
| 3328x1280x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.930 |
| 3328x1280x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.196 |
| 3328x1280x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.186 |
| 3328x1280x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3328x1280x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 3328x1280x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 4096x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4096x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4096x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4096x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4096x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4096x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4096x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4096x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4096x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4096x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4096x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4096x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4096x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.223 |
| 4096x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.213 |
| 4096x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.240 |
| 4096x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.207 |
| 4096x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.243 |
| 4096x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.196 |
| 4096x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.211 |
| 4096x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.194 |
| 4096x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.201 |
| 4096x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.179 |
| 4096x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.198 |
| 4096x1536x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.206 |
| 4096x1536x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4096x1536x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4096x1536x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4096x1536x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4096x1536x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.173 |
| 4096x1536x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.159 |
| 4096x1536x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.171 |
| 4096x1536x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.175 |
| 4096x1536x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.226 |
| 4096x1536x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.196 |
| 4096x1536x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.216 |
| 4096x1536x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.236 |
| 4096x1536x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4096x1536x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4096x1536x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.179 |
| 4096x1536x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.163 |
| 4096x1536x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4096x1536x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4096x1536x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.175 |
| 4096x1536x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.180 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.156 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.143 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.130 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.143 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.133 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.129 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.183 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.153 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.140 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.168 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.127 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.129 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.151 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.144 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.141 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.147 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.168 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.140 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.140 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.144 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.138 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.148 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.150 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.131 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.152 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.162 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.141 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.150 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.157 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.168 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.192 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.129 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.145 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.151 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.151 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.132 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.142 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.156 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.128 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.178 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.144 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.149 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.141 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.144 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.120 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.145 |
| 1000x500x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 1000x500x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 1000x500x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 1000x500x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 1000x500x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 1000x500x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 1000x500x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 1000x500x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 1000x500x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 1000x500x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 1000x500x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 1000x500x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 1000x500x8080 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.147 |
| 1000x500x8080 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.148 |
| 1000x500x8080 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x500x8080 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.147 |
| 1000x500x8080 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.131 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.133 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.148 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.187 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.127 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.131 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.131 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.126 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.168 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.164 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.137 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.142 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.156 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.128 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.153 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.142 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.127 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.127 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.174 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.148 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.152 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.155 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.148 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.150 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.126 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.145 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.158 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.162 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.184 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.157 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.162 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.160 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.135 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.134 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.127 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.128 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.157 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.141 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.128 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.174 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.145 |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 4300x1000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | | 10 | error:   File "/home/jbowley/repos/triton/python/triton/backends/amd/compiler.py", line 586, in make_llir /     pm.run(mod, 'make_llir') / RuntimeError: PassManager::run failed | | |
| 4300x1000x8176 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.180 |
| 4300x1000x8176 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.184 |
| 4300x1000x8176 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4300x1000x8176 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 4300x1000x8176 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |

FAIL
