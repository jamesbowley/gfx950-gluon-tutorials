tile 256x256x64: kernel BLOCK 256x256x64, 4 waves, 256 programs, STREAMK_OWNER_OPT=bsf

| shape | configuration | STREAMK_NUM_PROGRAMS | tile order | bias | kernel | launches | wrong results | flags or counts left up | largest error |
|---|---|---|---|---|---|---|---|---|---|
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner (grid 148) | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner (grid 148) | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner (grid 148) | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner (grid 148) | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.177 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.164 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner (grid 148) | 10 | 0 | 0 | 0.147 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner (grid 148) | 10 | 0 | 0 | 0.166 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.157 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.155 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.181 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.169 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner (grid 148) | 10 | 0 | 0 | 0.154 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner (grid 148) | 10 | 0 | 0 | 0.182 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.153 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.141 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.172 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.173 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner (grid 148) | 10 | 0 | 0 | 0.162 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner (grid 148) | 10 | 0 | 0 | 0.177 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.150 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.143 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.167 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.153 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner (grid 148) | 10 | 0 | 0 | 0.147 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner (grid 148) | 10 | 0 | 0 | 0.159 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.167 |
| 768x2816x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.160 |
| 768x2816x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.157 |
| 768x2816x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.151 |
| 768x2816x8192 | POLICY=two_tile, PARTIALS=bf16_pairs | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.153 |
| 768x2816x8192 | POLICY=two_tile, PARTIALS=bf16_pairs | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.154 |
| 768x2816x8192 | POLICY=spread, PARTIALS=fp32 | 132 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=spread, PARTIALS=fp32 | 132 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=spread, PARTIALS=fp32 | 148 | v9 | 0 | v28_streamk_owner (grid 148) | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=spread, PARTIALS=fp32 | 148 | v9 | 1 | v28_streamk_owner (grid 148) | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 768x2816x8192 | POLICY=spread, PARTIALS=bf16 | 132 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.161 |
| 768x2816x8192 | POLICY=spread, PARTIALS=bf16 | 132 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.175 |
| 768x2816x8192 | POLICY=spread, PARTIALS=bf16 | 148 | v9 | 0 | v28_streamk_owner (grid 148) | 10 | 0 | 0 | 0.163 |
| 768x2816x8192 | POLICY=spread, PARTIALS=bf16 | 148 | v9 | 1 | v28_streamk_owner (grid 148) | 10 | 0 | 0 | 0.156 |
| 768x2816x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.169 |
| 768x2816x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.152 |
| 768x2816x8192 | POLICY=spread, PARTIALS=bf16_pairs | 132 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.181 |
| 768x2816x8192 | POLICY=spread, PARTIALS=bf16_pairs | 132 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.161 |
| 768x2816x8192 | POLICY=spread, PARTIALS=bf16_pairs | 148 | v9 | 0 | v28_streamk_owner (grid 148) | 10 | 0 | 0 | 0.152 |
| 768x2816x8192 | POLICY=spread, PARTIALS=bf16_pairs | 148 | v9 | 1 | v28_streamk_owner (grid 148) | 10 | 0 | 0 | 0.155 |
| 768x2816x8192 | POLICY=spread, PARTIALS=bf16_pairs | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.148 |
| 768x2816x8192 | POLICY=spread, PARTIALS=bf16_pairs | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.160 |
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
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.186 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.163 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.172 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.189 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.146 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.150 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.163 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.145 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.197 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.177 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.153 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.150 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.167 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.186 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.159 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.179 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.166 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.159 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.148 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.154 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.156 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.160 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.181 |
| 1000x2000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.149 |
| 1000x2000x8080 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.173 |
| 1000x2000x8080 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.159 |
| 1000x2000x8080 | POLICY=two_tile, PARTIALS=bf16_pairs | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.177 |
| 1000x2000x8080 | POLICY=two_tile, PARTIALS=bf16_pairs | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.170 |
| 1000x2000x8080 | POLICY=spread, PARTIALS=fp32 | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=spread, PARTIALS=fp32 | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=spread, PARTIALS=fp32 | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=spread, PARTIALS=fp32 | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 1000x2000x8080 | POLICY=spread, PARTIALS=bf16 | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.162 |
| 1000x2000x8080 | POLICY=spread, PARTIALS=bf16 | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.177 |
| 1000x2000x8080 | POLICY=spread, PARTIALS=bf16 | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.168 |
| 1000x2000x8080 | POLICY=spread, PARTIALS=bf16 | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.143 |
| 1000x2000x8080 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.154 |
| 1000x2000x8080 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.142 |
| 1000x2000x8080 | POLICY=spread, PARTIALS=bf16_pairs | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.162 |
| 1000x2000x8080 | POLICY=spread, PARTIALS=bf16_pairs | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.170 |
| 1000x2000x8080 | POLICY=spread, PARTIALS=bf16_pairs | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.153 |
| 1000x2000x8080 | POLICY=spread, PARTIALS=bf16_pairs | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.145 |
| 1000x2000x8080 | POLICY=spread, PARTIALS=bf16_pairs | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.162 |
| 1000x2000x8080 | POLICY=spread, PARTIALS=bf16_pairs | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.163 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.042 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.043 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.044 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.038 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.042 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.042 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.038 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.039 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.043 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.043 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.048 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.037 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.040 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.045 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.039 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.038 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.038 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.046 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.041 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.043 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.047 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.041 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.038 |
| 768x2816x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.038 |
| 768x2816x512 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.039 |
| 768x2816x512 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.041 |
| 768x2816x512 | POLICY=two_tile, PARTIALS=bf16_pairs | default | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.042 |
| 768x2816x512 | POLICY=two_tile, PARTIALS=bf16_pairs | default | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.045 |
| 768x2816x512 | POLICY=spread, PARTIALS=fp32 | 132 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=spread, PARTIALS=fp32 | 132 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=spread, PARTIALS=fp32 | 148 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=spread, PARTIALS=fp32 | 148 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 768x2816x512 | POLICY=spread, PARTIALS=bf16 | 132 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.039 |
| 768x2816x512 | POLICY=spread, PARTIALS=bf16 | 132 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.041 |
| 768x2816x512 | POLICY=spread, PARTIALS=bf16 | 148 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.040 |
| 768x2816x512 | POLICY=spread, PARTIALS=bf16 | 148 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.040 |
| 768x2816x512 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.039 |
| 768x2816x512 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.038 |
| 768x2816x512 | POLICY=spread, PARTIALS=bf16_pairs | 132 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.050 |
| 768x2816x512 | POLICY=spread, PARTIALS=bf16_pairs | 132 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.038 |
| 768x2816x512 | POLICY=spread, PARTIALS=bf16_pairs | 148 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.045 |
| 768x2816x512 | POLICY=spread, PARTIALS=bf16_pairs | 148 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.039 |
| 768x2816x512 | POLICY=spread, PARTIALS=bf16_pairs | default | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.037 |
| 768x2816x512 | POLICY=spread, PARTIALS=bf16_pairs | default | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.041 |

PASS
