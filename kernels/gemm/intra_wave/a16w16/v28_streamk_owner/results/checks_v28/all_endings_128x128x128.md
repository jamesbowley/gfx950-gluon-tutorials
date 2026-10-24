tile 128x128x128: kernel BLOCK 128x128x128, 4 waves, 256 programs, STREAMK_OWNER_OPT=(none)

| shape | configuration | STREAMK_NUM_PROGRAMS | tile order | bias | kernel | launches | wrong results | flags or counts left up | largest error |
|---|---|---|---|---|---|---|---|---|---|
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.187 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.146 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.134 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.165 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.127 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.126 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.126 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.134 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.148 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.131 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.144 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.144 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.124 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.136 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.163 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.143 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.144 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.145 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.196 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.135 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.152 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.142 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.140 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.143 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.174 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.128 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.137 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.130 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.126 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.142 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.170 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.168 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.137 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.124 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.134 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.194 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.153 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.140 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.133 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.141 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.144 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.154 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.126 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.132 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.132 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.152 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.127 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.137 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.126 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.141 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.138 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.130 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.141 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.127 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.131 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.169 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.149 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.132 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.157 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.131 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.127 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.162 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.173 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.140 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.124 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.160 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.138 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.128 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.150 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.138 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.134 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.148 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.158 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.144 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.137 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.124 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.168 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.175 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.141 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.152 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.133 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.148 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.165 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.136 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.159 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.161 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.223 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.239 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.231 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.265 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.438 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.424 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.233 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.220 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.258 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.228 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.508 |
| 2176x2048x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.432 |
| 2176x2048x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.213 |
| 2176x2048x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.179 |
| 2176x2048x8192 | POLICY=two_tile, PARTIALS=bf16_pairs | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.201 |
| 2176x2048x8192 | POLICY=two_tile, PARTIALS=bf16_pairs | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.190 |
| 2176x2048x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2048x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.127 |
| 2176x2048x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 2176x2048x8192 | POLICY=spread, PARTIALS=bf16_pairs | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.136 |
| 2176x2048x8192 | POLICY=spread, PARTIALS=bf16_pairs | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.135 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.159 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.146 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.157 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.179 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.140 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.146 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.136 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.164 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.167 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.174 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.159 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.181 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.178 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.127 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.149 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.171 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.163 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.136 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.131 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.175 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.153 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.147 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.147 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.133 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.185 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.157 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.173 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.159 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.155 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.174 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.159 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.155 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.164 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.149 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.137 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.141 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.140 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.181 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.150 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.159 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.166 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.140 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.162 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.178 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.163 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.138 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.157 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.163 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.143 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.161 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.143 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.135 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.151 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.167 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.181 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.152 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.141 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.145 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.143 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.158 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.180 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.155 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.162 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.132 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.145 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.154 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.136 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.151 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.141 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.149 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.137 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.148 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.150 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.182 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.164 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.160 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.152 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.133 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.134 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.149 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.152 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.134 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.163 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.158 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.158 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.160 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.164 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.151 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.144 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.132 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.142 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.139 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.126 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.147 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.260 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.251 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.273 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.241 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.317 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.295 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.253 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.232 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.261 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.264 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.334 |
| 2176x2176x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.347 |
| 2176x2176x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.174 |
| 2176x2176x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.224 |
| 2176x2176x8192 | POLICY=two_tile, PARTIALS=bf16_pairs | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.230 |
| 2176x2176x8192 | POLICY=two_tile, PARTIALS=bf16_pairs | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.171 |
| 2176x2176x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2176x2176x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.148 |
| 2176x2176x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.161 |
| 2176x2176x8192 | POLICY=spread, PARTIALS=bf16_pairs | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.131 |
| 2176x2176x8192 | POLICY=spread, PARTIALS=bf16_pairs | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.138 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.151 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.140 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.144 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.134 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.124 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.124 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.124 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.151 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.128 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.145 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.124 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.126 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.149 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.137 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.124 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.124 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.138 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.124 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.128 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.128 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.124 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.124 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.163 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.141 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.150 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.132 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.131 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.151 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.123 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.123 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.129 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.124 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.124 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.137 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.124 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.124 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.124 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.136 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.135 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.136 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.156 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.124 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.124 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.124 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.123 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.134 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.132 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.189 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.216 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.221 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.233 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.604 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.546 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.223 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 16 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.208 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.218 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 18 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.259 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.596 |
| 1664x2560x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.513 |
| 1664x2560x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.169 |
| 1664x2560x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.169 |
| 1664x2560x8192 | POLICY=two_tile, PARTIALS=bf16_pairs | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.204 |
| 1664x2560x8192 | POLICY=two_tile, PARTIALS=bf16_pairs | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.163 |
| 1664x2560x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1664x2560x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.124 |
| 1664x2560x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=spread, PARTIALS=bf16_pairs | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 1664x2560x8192 | POLICY=spread, PARTIALS=bf16_pairs | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.214 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.226 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.192 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.203 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.214 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.196 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.195 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.184 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.242 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.193 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.236 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.197 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.204 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.206 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.216 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.200 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.238 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.163 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.194 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.218 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.209 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.187 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.228 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.203 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.177 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.170 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.174 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.166 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.171 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.160 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.167 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.168 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.236 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.214 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.197 |
| 2048x3072x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.189 |
| 2048x3072x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2048x3072x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2048x3072x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.173 |
| 2048x3072x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.179 |
| 2048x3072x8192 | POLICY=two_tile, PARTIALS=bf16_pairs | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.181 |
| 2048x3072x8192 | POLICY=two_tile, PARTIALS=bf16_pairs | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.175 |
| 2048x3072x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2048x3072x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2048x3072x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.168 |
| 2048x3072x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.179 |
| 2048x3072x8192 | POLICY=spread, PARTIALS=bf16_pairs | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.169 |
| 2048x3072x8192 | POLICY=spread, PARTIALS=bf16_pairs | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.164 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.150 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.160 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.131 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.181 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.157 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.144 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.136 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.140 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.155 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.165 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.146 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.143 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.183 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.144 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.175 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.168 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.145 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.161 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.158 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.132 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.159 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.149 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.153 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.151 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.173 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.156 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.150 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.151 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.149 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.143 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.160 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.136 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.154 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.140 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.143 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.147 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.169 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.193 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.156 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.190 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.136 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.165 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.170 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.141 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.147 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.148 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.145 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.123 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.162 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.155 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.183 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.142 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.156 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.134 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.145 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.154 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.165 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.172 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.140 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.126 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.135 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.165 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.142 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.147 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.144 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.157 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.151 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.169 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.135 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.163 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.158 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.123 |
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
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.152 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.149 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.159 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.152 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.148 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.161 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.154 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.146 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.166 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.131 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.171 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.153 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.134 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.138 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.150 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.132 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.142 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.136 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.163 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.152 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.175 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.137 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.165 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.129 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.253 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.236 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.264 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.237 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.329 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.303 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.280 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.263 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.241 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.254 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.320 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.308 |
| 500x1000x8080 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.131 |
| 500x1000x8080 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.122 |
| 500x1000x8080 | POLICY=two_tile, PARTIALS=bf16_pairs | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.131 |
| 500x1000x8080 | POLICY=two_tile, PARTIALS=bf16_pairs | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.127 |
| 500x1000x8080 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.150 |
| 500x1000x8080 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.164 |
| 500x1000x8080 | POLICY=spread, PARTIALS=bf16_pairs | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.156 |
| 500x1000x8080 | POLICY=spread, PARTIALS=bf16_pairs | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.157 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.128 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.139 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.193 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.124 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.137 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.154 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.158 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.124 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.124 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.141 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.132 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.167 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.161 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.157 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.132 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.145 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.134 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.181 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.133 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.135 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.149 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.143 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.141 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.147 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.129 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.148 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.135 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.152 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.140 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.128 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.173 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.134 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.145 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.155 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.134 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.177 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.151 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.153 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.133 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.135 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.137 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.149 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.147 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.152 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.129 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.155 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.142 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.124 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.136 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.142 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.171 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.166 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.144 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.162 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.136 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.151 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.161 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.133 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.124 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.126 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.161 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.124 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.127 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.124 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.146 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.138 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.172 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.140 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.129 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.167 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.124 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.143 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.127 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.138 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.130 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.139 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.135 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.221 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.202 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.254 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.216 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.420 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.399 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.237 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.220 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.224 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.228 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.418 |
| 2150x2000x8176 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.435 |
| 2150x2000x8176 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.180 |
| 2150x2000x8176 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.176 |
| 2150x2000x8176 | POLICY=two_tile, PARTIALS=bf16_pairs | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.247 |
| 2150x2000x8176 | POLICY=two_tile, PARTIALS=bf16_pairs | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.155 |
| 2150x2000x8176 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 2150x2000x8176 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.124 |
| 2150x2000x8176 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.124 |
| 2150x2000x8176 | POLICY=spread, PARTIALS=bf16_pairs | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.125 |
| 2150x2000x8176 | POLICY=spread, PARTIALS=bf16_pairs | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.150 |

PASS
