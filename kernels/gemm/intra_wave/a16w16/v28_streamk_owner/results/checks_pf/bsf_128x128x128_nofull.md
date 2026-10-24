tile 128x128x128: kernel BLOCK 128x128x128, 4 waves, 256 programs, STREAMK_OWNER_OPT=bsf

| shape | configuration | STREAMK_NUM_PROGRAMS | tile order | bias | kernel | launches | wrong results | flags or counts left up | largest error |
|---|---|---|---|---|---|---|---|---|---|
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner (grid 148) | 10 | 0 | 0 | 0.000 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner (grid 148) | 10 | 0 | 0 | 0.000 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner (grid 148) | 10 | 0 | 0 | 0.000 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner (grid 148) | 10 | 0 | 0 | 0.000 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.136 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.154 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner (grid 148) | 10 | 0 | 0 | 0.147 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner (grid 148) | 10 | 0 | 0 | 0.153 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.149 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.142 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.140 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.165 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner (grid 148) | 10 | 0 | 0 | 0.128 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner (grid 148) | 10 | 0 | 0 | 0.144 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.133 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.121 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.132 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.156 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner (grid 148) | 10 | 0 | 0 | 0.150 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner (grid 148) | 10 | 0 | 0 | 0.132 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.155 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.141 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.150 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.134 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner (grid 148) | 10 | 0 | 0 | 0.134 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner (grid 148) | 10 | 0 | 0 | 0.140 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.144 |
| 384x1408x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.153 |
| 384x1408x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 384x1408x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 384x1408x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.143 |
| 384x1408x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.152 |
| 384x1408x8192 | POLICY=two_tile, PARTIALS=bf16_pairs | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.138 |
| 384x1408x8192 | POLICY=two_tile, PARTIALS=bf16_pairs | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.121 |
| 384x1408x8192 | POLICY=spread, PARTIALS=fp32 | 132 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x8192 | POLICY=spread, PARTIALS=fp32 | 132 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x8192 | POLICY=spread, PARTIALS=fp32 | 148 | v9 | 0 | v28_streamk_owner (grid 148) | 10 | 0 | 0 | 0.000 |
| 384x1408x8192 | POLICY=spread, PARTIALS=fp32 | 148 | v9 | 1 | v28_streamk_owner (grid 148) | 10 | 0 | 0 | 0.000 |
| 384x1408x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 384x1408x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 384x1408x8192 | POLICY=spread, PARTIALS=bf16 | 132 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.145 |
| 384x1408x8192 | POLICY=spread, PARTIALS=bf16 | 132 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.189 |
| 384x1408x8192 | POLICY=spread, PARTIALS=bf16 | 148 | v9 | 0 | v28_streamk_owner (grid 148) | 10 | 0 | 0 | 0.142 |
| 384x1408x8192 | POLICY=spread, PARTIALS=bf16 | 148 | v9 | 1 | v28_streamk_owner (grid 148) | 10 | 0 | 0 | 0.149 |
| 384x1408x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.151 |
| 384x1408x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.119 |
| 384x1408x8192 | POLICY=spread, PARTIALS=bf16_pairs | 132 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.166 |
| 384x1408x8192 | POLICY=spread, PARTIALS=bf16_pairs | 132 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.145 |
| 384x1408x8192 | POLICY=spread, PARTIALS=bf16_pairs | 148 | v9 | 0 | v28_streamk_owner (grid 148) | 10 | 0 | 0 | 0.139 |
| 384x1408x8192 | POLICY=spread, PARTIALS=bf16_pairs | 148 | v9 | 1 | v28_streamk_owner (grid 148) | 10 | 0 | 0 | 0.183 |
| 384x1408x8192 | POLICY=spread, PARTIALS=bf16_pairs | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.148 |
| 384x1408x8192 | POLICY=spread, PARTIALS=bf16_pairs | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.157 |
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
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.158 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.128 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.132 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.149 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.116 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.139 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.160 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.164 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.136 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.153 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.119 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.149 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.156 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.134 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.121 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.168 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.133 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.147 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.170 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.148 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.125 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.131 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.126 |
| 500x1000x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.144 |
| 500x1000x8080 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.139 |
| 500x1000x8080 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.146 |
| 500x1000x8080 | POLICY=two_tile, PARTIALS=bf16_pairs | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.138 |
| 500x1000x8080 | POLICY=two_tile, PARTIALS=bf16_pairs | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.135 |
| 500x1000x8080 | POLICY=spread, PARTIALS=fp32 | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=spread, PARTIALS=fp32 | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=spread, PARTIALS=fp32 | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=spread, PARTIALS=fp32 | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.000 |
| 500x1000x8080 | POLICY=spread, PARTIALS=bf16 | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.167 |
| 500x1000x8080 | POLICY=spread, PARTIALS=bf16 | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.170 |
| 500x1000x8080 | POLICY=spread, PARTIALS=bf16 | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.119 |
| 500x1000x8080 | POLICY=spread, PARTIALS=bf16 | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.169 |
| 500x1000x8080 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.146 |
| 500x1000x8080 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.140 |
| 500x1000x8080 | POLICY=spread, PARTIALS=bf16_pairs | 128 | v9 | 0 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.154 |
| 500x1000x8080 | POLICY=spread, PARTIALS=bf16_pairs | 128 | v9 | 1 | v28_streamk_owner (grid 128) | 10 | 0 | 0 | 0.152 |
| 500x1000x8080 | POLICY=spread, PARTIALS=bf16_pairs | 144 | v9 | 0 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.122 |
| 500x1000x8080 | POLICY=spread, PARTIALS=bf16_pairs | 144 | v9 | 1 | v28_streamk_owner (grid 144) | 10 | 0 | 0 | 0.144 |
| 500x1000x8080 | POLICY=spread, PARTIALS=bf16_pairs | default | v9 | 0 | v28_streamk_owner | 10 | 0 | 0 | 0.121 |
| 500x1000x8080 | POLICY=spread, PARTIALS=bf16_pairs | default | v9 | 1 | v28_streamk_owner | 10 | 0 | 0 | 0.121 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.053 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.050 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.048 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.051 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.054 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.050 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.053 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.048 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.052 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.051 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.050 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.052 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.050 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.051 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.054 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.047 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.053 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.052 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.047 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 132 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.049 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.053 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 148 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.055 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.056 |
| 384x1408x1024 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16_pairs, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.053 |
| 384x1408x1024 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x1024 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x1024 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.054 |
| 384x1408x1024 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.058 |
| 384x1408x1024 | POLICY=two_tile, PARTIALS=bf16_pairs | default | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.049 |
| 384x1408x1024 | POLICY=two_tile, PARTIALS=bf16_pairs | default | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.060 |
| 384x1408x1024 | POLICY=spread, PARTIALS=fp32 | 132 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x1024 | POLICY=spread, PARTIALS=fp32 | 132 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x1024 | POLICY=spread, PARTIALS=fp32 | 148 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x1024 | POLICY=spread, PARTIALS=fp32 | 148 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x1024 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x1024 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.000 |
| 384x1408x1024 | POLICY=spread, PARTIALS=bf16 | 132 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.049 |
| 384x1408x1024 | POLICY=spread, PARTIALS=bf16 | 132 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.048 |
| 384x1408x1024 | POLICY=spread, PARTIALS=bf16 | 148 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.050 |
| 384x1408x1024 | POLICY=spread, PARTIALS=bf16 | 148 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.049 |
| 384x1408x1024 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.052 |
| 384x1408x1024 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.053 |
| 384x1408x1024 | POLICY=spread, PARTIALS=bf16_pairs | 132 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.048 |
| 384x1408x1024 | POLICY=spread, PARTIALS=bf16_pairs | 132 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.052 |
| 384x1408x1024 | POLICY=spread, PARTIALS=bf16_pairs | 148 | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.050 |
| 384x1408x1024 | POLICY=spread, PARTIALS=bf16_pairs | 148 | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.049 |
| 384x1408x1024 | POLICY=spread, PARTIALS=bf16_pairs | default | v9 | 0 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.050 |
| 384x1408x1024 | POLICY=spread, PARTIALS=bf16_pairs | default | v9 | 1 | v28_streamk_owner (grid 132) | 10 | 0 | 0 | 0.051 |

PASS
