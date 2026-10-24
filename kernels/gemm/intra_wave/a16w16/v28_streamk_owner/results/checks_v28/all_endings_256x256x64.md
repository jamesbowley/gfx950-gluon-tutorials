tile 256x256x64: kernel BLOCK 256x256x64, 4 waves, 256 programs, STREAMK_OWNER_OPT=(none)

| shape | configuration | STREAMK_NUM_PROGRAMS | tile order | bias | kernel | launches | wrong results | flags or counts left up | largest error |
|---|---|---|---|---|---|---|---|---|---|
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.183 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.188 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.171 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.156 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.144 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.159 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.174 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.163 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.160 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.164 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.149 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.164 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.161 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.183 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.170 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.166 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.142 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.134 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.202 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.166 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.180 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.159 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.154 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.155 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.186 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.139 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.159 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.161 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.152 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.135 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.160 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.167 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.149 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.171 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.126 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.141 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.167 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.167 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.157 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.152 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.141 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.167 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.181 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.165 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.178 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.171 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.170 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.137 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.194 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.147 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.176 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.173 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.138 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.149 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.192 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.159 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.170 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.149 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.153 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.148 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.169 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.172 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.176 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.181 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.144 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.129 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.198 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.162 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.171 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.184 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.165 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.145 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.154 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.142 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.150 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.143 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.142 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.167 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.162 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.159 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.157 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.172 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.134 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.171 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.181 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.157 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.155 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.163 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.136 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.147 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.133 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.153 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.155 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.149 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.145 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.155 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.231 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.243 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.284 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.262 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.484 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.455 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.288 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.234 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.293 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.285 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.493 |
| 4352x4096x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.449 |
| 4352x4096x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.200 |
| 4352x4096x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.211 |
| 4352x4096x8192 | POLICY=two_tile, PARTIALS=bf16_pairs | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.182 |
| 4352x4096x8192 | POLICY=two_tile, PARTIALS=bf16_pairs | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.212 |
| 4352x4096x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4096x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.153 |
| 4352x4096x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.139 |
| 4352x4096x8192 | POLICY=spread, PARTIALS=bf16_pairs | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.155 |
| 4352x4096x8192 | POLICY=spread, PARTIALS=bf16_pairs | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.148 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.171 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.183 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.171 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.149 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.160 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.171 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.178 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.163 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.152 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.170 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.194 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.179 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.180 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.163 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.165 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.154 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.152 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.143 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.173 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.177 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.182 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.184 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.158 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.166 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.163 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.187 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.178 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.193 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.171 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.154 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.173 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.164 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.186 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.173 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.167 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.168 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.179 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.181 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.171 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.163 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.155 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.146 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.171 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.159 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.172 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.168 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.158 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.162 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.175 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.189 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.165 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.179 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.188 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.172 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.189 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.203 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.166 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.173 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.146 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.152 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.177 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.175 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.172 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.206 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.188 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.163 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.186 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.177 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.175 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.170 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.163 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.171 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.164 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.153 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.182 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.165 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.166 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.174 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.185 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.165 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.171 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.177 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.181 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.151 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.179 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.161 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.164 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.146 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.154 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.163 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.172 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.147 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.148 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.154 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.174 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.167 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.262 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.245 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.252 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.260 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.399 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.368 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.229 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.269 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.258 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.303 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.330 |
| 4352x4352x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.345 |
| 4352x4352x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.200 |
| 4352x4352x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.224 |
| 4352x4352x8192 | POLICY=two_tile, PARTIALS=bf16_pairs | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.204 |
| 4352x4352x8192 | POLICY=two_tile, PARTIALS=bf16_pairs | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.175 |
| 4352x4352x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4352x4352x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.143 |
| 4352x4352x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.143 |
| 4352x4352x8192 | POLICY=spread, PARTIALS=bf16_pairs | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.163 |
| 4352x4352x8192 | POLICY=spread, PARTIALS=bf16_pairs | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.148 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.170 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.129 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.149 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.127 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.141 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.126 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.148 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.143 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.151 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.162 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.134 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.170 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.147 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.150 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.151 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.139 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.131 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.143 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.141 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.162 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.126 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.147 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.136 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.145 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.175 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.131 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.127 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.182 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.127 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.131 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.155 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.140 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.135 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.166 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.147 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.165 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.155 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.130 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.140 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.133 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.127 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.164 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.173 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.149 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.129 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.183 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.128 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.139 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.174 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.137 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.135 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.144 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.150 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.136 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.129 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.150 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.176 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.150 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.141 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.178 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.137 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.149 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.151 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.154 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.142 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.144 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.146 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.156 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.128 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.158 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.147 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.149 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.133 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.248 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.220 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.262 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.245 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.913 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 1.095 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.222 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.245 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.248 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.232 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.898 |
| 3328x5120x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.934 |
| 3328x5120x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.239 |
| 3328x5120x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.223 |
| 3328x5120x8192 | POLICY=two_tile, PARTIALS=bf16_pairs | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.219 |
| 3328x5120x8192 | POLICY=two_tile, PARTIALS=bf16_pairs | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.238 |
| 3328x5120x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 3328x5120x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.138 |
| 3328x5120x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.128 |
| 3328x5120x8192 | POLICY=spread, PARTIALS=bf16_pairs | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.138 |
| 3328x5120x8192 | POLICY=spread, PARTIALS=bf16_pairs | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.142 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.232 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.204 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.218 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.224 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.232 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.212 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.230 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.233 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.223 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.188 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.216 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.197 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.234 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.214 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.220 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.195 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.227 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.206 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.231 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.220 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.246 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.224 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.213 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.203 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.177 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.177 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.180 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.179 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.179 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.173 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.179 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.180 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.215 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.242 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.229 |
| 4096x6144x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.226 |
| 4096x6144x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4096x6144x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4096x6144x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.183 |
| 4096x6144x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.177 |
| 4096x6144x8192 | POLICY=two_tile, PARTIALS=bf16_pairs | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.181 |
| 4096x6144x8192 | POLICY=two_tile, PARTIALS=bf16_pairs | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.177 |
| 4096x6144x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4096x6144x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4096x6144x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.177 |
| 4096x6144x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.181 |
| 4096x6144x8192 | POLICY=spread, PARTIALS=bf16_pairs | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.187 |
| 4096x6144x8192 | POLICY=spread, PARTIALS=bf16_pairs | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.173 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.172 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.173 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.178 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.173 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.156 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.170 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.159 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.182 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.195 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.185 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.162 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.160 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.167 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.142 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.185 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.179 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.142 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.176 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.175 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.186 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.189 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.174 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.151 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.150 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.165 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.179 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.173 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.186 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.185 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.151 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.175 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.170 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.149 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.173 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.147 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.154 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.199 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.167 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.162 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.167 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.153 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.179 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.168 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.164 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.179 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.190 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.167 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.164 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.162 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.181 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.174 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.176 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.148 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.154 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.180 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.162 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.157 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.187 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.165 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.136 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.193 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.173 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.166 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.196 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.170 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.151 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.153 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.171 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.167 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.162 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.164 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.156 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.135 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.170 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.155 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.185 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.157 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.154 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.155 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.172 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.146 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.152 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.169 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.168 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.173 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.175 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.166 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.171 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.149 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.166 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.142 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.157 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.158 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.157 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.164 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.140 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.298 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.272 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.252 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.298 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.345 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.375 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.281 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.257 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.296 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.286 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.454 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.377 |
| 1000x2000x8080 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.157 |
| 1000x2000x8080 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.143 |
| 1000x2000x8080 | POLICY=two_tile, PARTIALS=bf16_pairs | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.138 |
| 1000x2000x8080 | POLICY=two_tile, PARTIALS=bf16_pairs | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.155 |
| 1000x2000x8080 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.154 |
| 1000x2000x8080 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.167 |
| 1000x2000x8080 | POLICY=spread, PARTIALS=bf16_pairs | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.157 |
| 1000x2000x8080 | POLICY=spread, PARTIALS=bf16_pairs | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.141 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.179 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.175 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.149 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.150 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.142 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.135 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.169 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.160 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.185 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.161 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.143 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.143 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.153 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.147 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.175 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.158 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.143 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.156 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.153 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.136 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.156 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.155 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.153 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.150 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.166 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.158 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.152 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.127 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.137 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.162 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.158 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.176 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.152 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.155 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.134 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.212 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.144 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.156 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.182 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.166 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.150 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.141 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.169 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.176 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.200 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.150 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.153 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.190 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.160 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.157 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.162 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.146 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.137 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.148 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.168 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.134 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.188 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.146 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.142 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.153 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.151 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.142 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.166 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.156 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.152 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.187 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.171 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.158 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.170 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.151 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.165 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.179 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.152 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.134 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.151 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.155 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.127 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.158 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.141 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.164 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.168 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.138 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.152 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.157 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.168 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.155 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.158 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.145 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.149 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.139 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.155 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.153 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.175 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.149 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.146 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.231 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.246 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.271 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.256 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.488 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.446 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.250 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.235 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.283 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.285 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.476 |
| 4300x4000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.492 |
| 4300x4000x8176 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.218 |
| 4300x4000x8176 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.209 |
| 4300x4000x8176 | POLICY=two_tile, PARTIALS=bf16_pairs | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.204 |
| 4300x4000x8176 | POLICY=two_tile, PARTIALS=bf16_pairs | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.238 |
| 4300x4000x8176 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 4300x4000x8176 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.160 |
| 4300x4000x8176 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.146 |
| 4300x4000x8176 | POLICY=spread, PARTIALS=bf16_pairs | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.136 |
| 4300x4000x8176 | POLICY=spread, PARTIALS=bf16_pairs | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.132 |

PASS
