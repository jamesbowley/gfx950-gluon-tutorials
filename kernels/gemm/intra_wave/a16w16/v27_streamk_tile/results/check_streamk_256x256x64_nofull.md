tile 256x256x64: kernel BLOCK 256x256x64, 4 waves, 256 programs

| shape | configuration | STREAMK_NUM_PROGRAMS | tile order | bias | kernel | launches | wrong results | flags or counts left up | largest error |
|---|---|---|---|---|---|---|---|---|---|
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.171 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.187 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.189 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.168 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.161 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.175 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.173 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.175 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.187 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.214 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.154 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.183 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.166 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.179 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.169 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.162 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.152 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.160 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.183 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.165 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.175 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.179 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.141 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.146 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.197 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.160 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.159 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.172 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.195 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.162 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.189 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.176 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.163 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.140 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.143 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.159 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.165 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.166 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.165 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.172 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.145 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.161 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.184 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.178 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.194 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.172 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.154 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.160 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.158 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.169 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.188 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.184 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.153 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.160 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.176 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.159 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.172 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.189 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.168 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.143 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.200 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.173 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.166 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.153 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.174 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.170 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.191 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.171 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.156 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.181 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.172 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.179 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.166 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.170 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.173 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.168 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.155 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.157 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.163 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.163 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.184 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.166 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.147 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.154 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.162 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.177 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.154 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.161 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.142 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.155 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.164 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.164 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.181 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.159 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.177 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.154 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.281 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.274 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.333 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.341 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.396 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.350 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.276 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.269 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.323 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.290 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.384 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.332 |
| 768x2816x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.142 |
| 768x2816x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.170 |
| 768x2816x8192 | POLICY=two_tile, PARTIALS=bf16_pairs | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.164 |
| 768x2816x8192 | POLICY=two_tile, PARTIALS=bf16_pairs | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.153 |
| 768x2816x8192 | POLICY=spread, PARTIALS=fp32 | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=spread, PARTIALS=fp32 | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=spread, PARTIALS=fp32 | 148 | v9 | 0 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=spread, PARTIALS=fp32 | 148 | v9 | 1 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=spread, PARTIALS=bf16 | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.151 |
| 768x2816x8192 | POLICY=spread, PARTIALS=bf16 | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.156 |
| 768x2816x8192 | POLICY=spread, PARTIALS=bf16 | 148 | v9 | 0 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.183 |
| 768x2816x8192 | POLICY=spread, PARTIALS=bf16 | 148 | v9 | 1 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.213 |
| 768x2816x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.158 |
| 768x2816x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.153 |
| 768x2816x8192 | POLICY=spread, PARTIALS=bf16_pairs | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.195 |
| 768x2816x8192 | POLICY=spread, PARTIALS=bf16_pairs | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.170 |
| 768x2816x8192 | POLICY=spread, PARTIALS=bf16_pairs | 148 | v9 | 0 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.158 |
| 768x2816x8192 | POLICY=spread, PARTIALS=bf16_pairs | 148 | v9 | 1 | v27_streamk_tile (grid 148) | 10 | 0 | 0 | 0.178 |
| 768x2816x8192 | POLICY=spread, PARTIALS=bf16_pairs | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.160 |
| 768x2816x8192 | POLICY=spread, PARTIALS=bf16_pairs | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.190 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.167 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.159 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.159 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.159 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.159 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.151 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.173 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.170 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.196 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.177 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.173 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.159 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.174 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.159 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.183 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.156 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.165 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.154 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.187 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.178 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.176 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.162 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.155 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.185 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.183 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.164 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.167 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.175 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.143 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.175 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.166 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.185 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.157 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.191 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.177 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.161 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.169 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.193 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.181 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.167 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.149 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.158 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.168 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.180 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.192 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.170 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.159 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.153 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.192 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.170 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.163 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.161 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.160 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.135 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.174 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.176 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.177 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.154 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.155 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.146 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.187 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.172 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.180 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.158 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.154 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.158 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.174 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.152 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.182 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.175 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.147 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.160 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.174 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.179 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.157 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.176 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.144 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.164 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.177 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.154 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.175 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.193 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.152 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.140 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.162 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.162 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.155 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.163 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.140 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.150 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.166 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.161 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.155 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.177 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.176 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.158 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.253 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.268 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.264 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.280 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.341 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.336 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.259 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.270 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.275 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.314 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.355 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.379 |
| 1000x2000x8080 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.166 |
| 1000x2000x8080 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.152 |
| 1000x2000x8080 | POLICY=two_tile, PARTIALS=bf16_pairs | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.168 |
| 1000x2000x8080 | POLICY=two_tile, PARTIALS=bf16_pairs | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.148 |
| 1000x2000x8080 | POLICY=spread, PARTIALS=fp32 | 128 | v9 | 0 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=spread, PARTIALS=fp32 | 128 | v9 | 1 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=spread, PARTIALS=fp32 | 144 | v9 | 0 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=spread, PARTIALS=fp32 | 144 | v9 | 1 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=spread, PARTIALS=bf16 | 128 | v9 | 0 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.147 |
| 1000x2000x8080 | POLICY=spread, PARTIALS=bf16 | 128 | v9 | 1 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.168 |
| 1000x2000x8080 | POLICY=spread, PARTIALS=bf16 | 144 | v9 | 0 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.169 |
| 1000x2000x8080 | POLICY=spread, PARTIALS=bf16 | 144 | v9 | 1 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.157 |
| 1000x2000x8080 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.167 |
| 1000x2000x8080 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.166 |
| 1000x2000x8080 | POLICY=spread, PARTIALS=bf16_pairs | 128 | v9 | 0 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.182 |
| 1000x2000x8080 | POLICY=spread, PARTIALS=bf16_pairs | 128 | v9 | 1 | v27_streamk_tile (grid 128) | 10 | 0 | 0 | 0.170 |
| 1000x2000x8080 | POLICY=spread, PARTIALS=bf16_pairs | 144 | v9 | 0 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.133 |
| 1000x2000x8080 | POLICY=spread, PARTIALS=bf16_pairs | 144 | v9 | 1 | v27_streamk_tile (grid 144) | 10 | 0 | 0 | 0.183 |
| 1000x2000x8080 | POLICY=spread, PARTIALS=bf16_pairs | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.143 |
| 1000x2000x8080 | POLICY=spread, PARTIALS=bf16_pairs | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.159 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.043 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.041 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.047 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.044 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.039 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.051 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.047 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.045 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.043 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.045 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.041 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.042 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.042 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.055 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.046 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.050 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.042 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.044 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.049 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.048 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.045 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.043 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.045 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.043 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.041 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.047 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.044 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.049 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.044 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.044 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.044 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.049 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.041 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.042 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.043 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.049 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.039 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.043 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.040 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.042 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.045 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.038 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.043 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.049 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.047 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.046 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.039 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.045 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.041 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.044 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.045 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.041 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.049 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.044 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.046 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.044 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.045 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.045 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.045 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.047 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.046 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 132 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.040 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.044 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 148 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.044 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.044 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.042 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.046 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.044 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.048 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.041 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.047 |
| 768x2816x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16_pairs, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.044 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.035 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.039 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.049 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.046 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.045 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.038 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.041 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.042 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.043 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.039 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.045 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.039 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.038 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.042 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.042 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.038 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.041 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.040 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.043 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.044 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.046 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.043 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.042 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.043 |
| 768x2816x512 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.077 |
| 768x2816x512 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.062 |
| 768x2816x512 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.073 |
| 768x2816x512 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.078 |
| 768x2816x512 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.069 |
| 768x2816x512 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.062 |
| 768x2816x512 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.064 |
| 768x2816x512 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.069 |
| 768x2816x512 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.067 |
| 768x2816x512 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.068 |
| 768x2816x512 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.061 |
| 768x2816x512 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.068 |
| 768x2816x512 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.045 |
| 768x2816x512 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.043 |
| 768x2816x512 | POLICY=two_tile, PARTIALS=bf16_pairs | default | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.037 |
| 768x2816x512 | POLICY=two_tile, PARTIALS=bf16_pairs | default | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.041 |
| 768x2816x512 | POLICY=spread, PARTIALS=fp32 | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=spread, PARTIALS=fp32 | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=spread, PARTIALS=fp32 | 148 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=spread, PARTIALS=fp32 | 148 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=spread, PARTIALS=bf16 | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.038 |
| 768x2816x512 | POLICY=spread, PARTIALS=bf16 | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.043 |
| 768x2816x512 | POLICY=spread, PARTIALS=bf16 | 148 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.043 |
| 768x2816x512 | POLICY=spread, PARTIALS=bf16 | 148 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.043 |
| 768x2816x512 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.039 |
| 768x2816x512 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.044 |
| 768x2816x512 | POLICY=spread, PARTIALS=bf16_pairs | 132 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.040 |
| 768x2816x512 | POLICY=spread, PARTIALS=bf16_pairs | 132 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.040 |
| 768x2816x512 | POLICY=spread, PARTIALS=bf16_pairs | 148 | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.038 |
| 768x2816x512 | POLICY=spread, PARTIALS=bf16_pairs | 148 | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.039 |
| 768x2816x512 | POLICY=spread, PARTIALS=bf16_pairs | default | v9 | 0 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.039 |
| 768x2816x512 | POLICY=spread, PARTIALS=bf16_pairs | default | v9 | 1 | v27_streamk_tile (grid 132) | 10 | 0 | 0 | 0.041 |

PASS
