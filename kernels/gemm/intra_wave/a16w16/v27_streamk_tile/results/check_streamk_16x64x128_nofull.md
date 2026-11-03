tile 16x64x128: kernel BLOCK 16x64x128, 1 waves, 1024 programs

| shape | configuration | STREAMK_NUM_PROGRAMS | tile order | bias | kernel | launches | wrong results | flags or counts left up | largest error |
|---|---|---|---|---|---|---|---|---|---|
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 256 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 256 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 288 | v9 | 0 | v27_streamk_tile (grid 288) | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 288 | v9 | 1 | v27_streamk_tile (grid 288) | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 256 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 256 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 288 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 288 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 256 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 256 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 288 | v9 | 0 | v27_streamk_tile (grid 288) | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 288 | v9 | 1 | v27_streamk_tile (grid 288) | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 256 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 256 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 288 | v9 | 0 | v27_streamk_tile (grid 288) | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 288 | v9 | 1 | v27_streamk_tile (grid 288) | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 256 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 256 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 288 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 288 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 256 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 256 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 288 | v9 | 0 | v27_streamk_tile (grid 288) | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 288 | v9 | 1 | v27_streamk_tile (grid 288) | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 256 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.148 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 256 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.127 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 288 | v9 | 0 | v27_streamk_tile (grid 288) | 10 | 0 | 0 | 0.126 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 288 | v9 | 1 | v27_streamk_tile (grid 288) | 10 | 0 | 0 | 0.122 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.116 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.103 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 256 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 256 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.134 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 288 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.134 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 288 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.162 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.122 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.108 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 256 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.157 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 256 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.117 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 288 | v9 | 0 | v27_streamk_tile (grid 288) | 10 | 0 | 0 | 0.126 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 288 | v9 | 1 | v27_streamk_tile (grid 288) | 10 | 0 | 0 | 0.132 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.113 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.133 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 256 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.114 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 256 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.125 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 288 | v9 | 0 | v27_streamk_tile (grid 288) | 10 | 0 | 0 | 0.124 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 288 | v9 | 1 | v27_streamk_tile (grid 288) | 10 | 0 | 0 | 0.157 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.136 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.118 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 256 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.134 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 256 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.115 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 288 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.115 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 288 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.115 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.132 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.108 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 256 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.114 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 256 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.122 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 288 | v9 | 0 | v27_streamk_tile (grid 288) | 10 | 0 | 0 | 0.128 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 288 | v9 | 1 | v27_streamk_tile (grid 288) | 10 | 0 | 0 | 0.121 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.106 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.129 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 256 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 256 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 288 | v9 | 0 | v27_streamk_tile (grid 288) | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 288 | v9 | 1 | v27_streamk_tile (grid 288) | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 256 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 256 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 288 | v9 | 0 | v27_streamk_tile (grid 288) | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 288 | v9 | 1 | v27_streamk_tile (grid 288) | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 256 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.107 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 256 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.123 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 288 | v9 | 0 | v27_streamk_tile (grid 288) | 10 | 0 | 0 | 0.120 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 288 | v9 | 1 | v27_streamk_tile (grid 288) | 10 | 0 | 0 | 0.112 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.116 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.110 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 256 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.108 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 256 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.110 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 288 | v9 | 0 | v27_streamk_tile (grid 288) | 10 | 0 | 0 | 0.109 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 288 | v9 | 1 | v27_streamk_tile (grid 288) | 10 | 0 | 0 | 0.164 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.114 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.111 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 256 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.216 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 256 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.204 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 288 | v9 | 0 | v27_streamk_tile (grid 288) | 10 | 0 | 0 | 0.233 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 288 | v9 | 1 | v27_streamk_tile (grid 288) | 10 | 0 | 0 | 0.229 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.439 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.367 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 256 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.199 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 256 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.222 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 288 | v9 | 0 | v27_streamk_tile (grid 288) | 10 | 0 | 0 | 0.222 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 288 | v9 | 1 | v27_streamk_tile (grid 288) | 10 | 0 | 0 | 0.217 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.378 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.422 |
| 16x4096x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.113 |
| 16x4096x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.114 |
| 16x4096x8192 | POLICY=spread, PARTIALS=fp32 | 256 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=spread, PARTIALS=fp32 | 256 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=spread, PARTIALS=fp32 | 288 | v9 | 0 | v27_streamk_tile (grid 288) | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=spread, PARTIALS=fp32 | 288 | v9 | 1 | v27_streamk_tile (grid 288) | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=spread, PARTIALS=bf16 | 256 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.125 |
| 16x4096x8192 | POLICY=spread, PARTIALS=bf16 | 256 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.133 |
| 16x4096x8192 | POLICY=spread, PARTIALS=bf16 | 288 | v9 | 0 | v27_streamk_tile (grid 288) | 10 | 0 | 0 | 0.127 |
| 16x4096x8192 | POLICY=spread, PARTIALS=bf16 | 288 | v9 | 1 | v27_streamk_tile (grid 288) | 10 | 0 | 0 | 0.114 |
| 16x4096x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.114 |
| 16x4096x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.140 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 504 | v9 | 0 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 504 | v9 | 1 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 567 | v9 | 0 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 567 | v9 | 1 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 504 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 504 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 567 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 567 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 504 | v9 | 0 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 504 | v9 | 1 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 567 | v9 | 0 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 567 | v9 | 1 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 504 | v9 | 0 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 504 | v9 | 1 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 567 | v9 | 0 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 567 | v9 | 1 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 504 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 504 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 567 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 567 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 504 | v9 | 0 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 504 | v9 | 1 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 567 | v9 | 0 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 567 | v9 | 1 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 504 | v9 | 0 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.100 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 504 | v9 | 1 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.096 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 567 | v9 | 0 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.099 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 567 | v9 | 1 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.090 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.090 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.099 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 504 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.113 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 504 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.098 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 567 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.103 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 567 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.097 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.084 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.093 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 504 | v9 | 0 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.101 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 504 | v9 | 1 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.098 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 567 | v9 | 0 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.101 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 567 | v9 | 1 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.107 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.095 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.091 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 504 | v9 | 0 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.100 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 504 | v9 | 1 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.121 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 567 | v9 | 0 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.101 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 567 | v9 | 1 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.094 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.090 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.090 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 504 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.095 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 504 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.095 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 567 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.103 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 567 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.098 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.098 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.091 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 504 | v9 | 0 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.104 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 504 | v9 | 1 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.099 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 567 | v9 | 0 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.098 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 567 | v9 | 1 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.100 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.088 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.088 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 504 | v9 | 0 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 504 | v9 | 1 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 567 | v9 | 0 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 567 | v9 | 1 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 504 | v9 | 0 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 504 | v9 | 1 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 567 | v9 | 0 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 567 | v9 | 1 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 504 | v9 | 0 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.106 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 504 | v9 | 1 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.092 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 567 | v9 | 0 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.088 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 567 | v9 | 1 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.097 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.094 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.092 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 504 | v9 | 0 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.092 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 504 | v9 | 1 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.095 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 567 | v9 | 0 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.097 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 567 | v9 | 1 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.093 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.090 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.093 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 504 | v9 | 0 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.170 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 504 | v9 | 1 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.174 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 567 | v9 | 0 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.178 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 567 | v9 | 1 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.192 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.257 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.257 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 504 | v9 | 0 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.177 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 504 | v9 | 1 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.149 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 567 | v9 | 0 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.180 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 567 | v9 | 1 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.146 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.248 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.217 |
| 30x4000x4080 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.089 |
| 30x4000x4080 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.099 |
| 30x4000x4080 | POLICY=spread, PARTIALS=fp32 | 504 | v9 | 0 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=spread, PARTIALS=fp32 | 504 | v9 | 1 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=spread, PARTIALS=fp32 | 567 | v9 | 0 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=spread, PARTIALS=fp32 | 567 | v9 | 1 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=spread, PARTIALS=bf16 | 504 | v9 | 0 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.095 |
| 30x4000x4080 | POLICY=spread, PARTIALS=bf16 | 504 | v9 | 1 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.087 |
| 30x4000x4080 | POLICY=spread, PARTIALS=bf16 | 567 | v9 | 0 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.091 |
| 30x4000x4080 | POLICY=spread, PARTIALS=bf16 | 567 | v9 | 1 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.096 |
| 30x4000x4080 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.093 |
| 30x4000x4080 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.096 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 512 | v9 | 0 | v27_streamk_tile (grid 512) | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 512 | v9 | 1 | v27_streamk_tile (grid 512) | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 576 | v9 | 0 | v27_streamk_tile (grid 576) | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 576 | v9 | 1 | v27_streamk_tile (grid 576) | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 512 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 512 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 576 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 576 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 512 | v9 | 0 | v27_streamk_tile (grid 512) | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 512 | v9 | 1 | v27_streamk_tile (grid 512) | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 576 | v9 | 0 | v27_streamk_tile (grid 576) | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 576 | v9 | 1 | v27_streamk_tile (grid 576) | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 512 | v9 | 0 | v27_streamk_tile (grid 512) | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 512 | v9 | 1 | v27_streamk_tile (grid 512) | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 576 | v9 | 0 | v27_streamk_tile (grid 576) | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 576 | v9 | 1 | v27_streamk_tile (grid 576) | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 512 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 512 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 576 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 576 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 512 | v9 | 0 | v27_streamk_tile (grid 512) | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 512 | v9 | 1 | v27_streamk_tile (grid 512) | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 576 | v9 | 0 | v27_streamk_tile (grid 576) | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 576 | v9 | 1 | v27_streamk_tile (grid 576) | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 512 | v9 | 0 | v27_streamk_tile (grid 512) | 10 | 0 | 0 | 0.090 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 512 | v9 | 1 | v27_streamk_tile (grid 512) | 10 | 0 | 0 | 0.104 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 576 | v9 | 0 | v27_streamk_tile (grid 576) | 10 | 0 | 0 | 0.102 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 576 | v9 | 1 | v27_streamk_tile (grid 576) | 10 | 0 | 0 | 0.082 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.085 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.075 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 512 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.086 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 512 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.108 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 576 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.092 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 576 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.083 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.090 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.089 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 512 | v9 | 0 | v27_streamk_tile (grid 512) | 10 | 0 | 0 | 0.090 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 512 | v9 | 1 | v27_streamk_tile (grid 512) | 10 | 0 | 0 | 0.093 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 576 | v9 | 0 | v27_streamk_tile (grid 576) | 10 | 0 | 0 | 0.069 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 576 | v9 | 1 | v27_streamk_tile (grid 576) | 10 | 0 | 0 | 0.074 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.064 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.070 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 512 | v9 | 0 | v27_streamk_tile (grid 512) | 10 | 0 | 0 | 0.099 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 512 | v9 | 1 | v27_streamk_tile (grid 512) | 10 | 0 | 0 | 0.082 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 576 | v9 | 0 | v27_streamk_tile (grid 576) | 10 | 0 | 0 | 0.086 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 576 | v9 | 1 | v27_streamk_tile (grid 576) | 10 | 0 | 0 | 0.079 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.062 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.102 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 512 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.079 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 512 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.077 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 576 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.083 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 576 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.088 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.073 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.083 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 512 | v9 | 0 | v27_streamk_tile (grid 512) | 10 | 0 | 0 | 0.095 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 512 | v9 | 1 | v27_streamk_tile (grid 512) | 10 | 0 | 0 | 0.077 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 576 | v9 | 0 | v27_streamk_tile (grid 576) | 10 | 0 | 0 | 0.083 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 576 | v9 | 1 | v27_streamk_tile (grid 576) | 10 | 0 | 0 | 0.094 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.086 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.071 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 512 | v9 | 0 | v27_streamk_tile (grid 512) | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 512 | v9 | 1 | v27_streamk_tile (grid 512) | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 576 | v9 | 0 | v27_streamk_tile (grid 576) | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 576 | v9 | 1 | v27_streamk_tile (grid 576) | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 512 | v9 | 0 | v27_streamk_tile (grid 512) | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 512 | v9 | 1 | v27_streamk_tile (grid 512) | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 576 | v9 | 0 | v27_streamk_tile (grid 576) | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 576 | v9 | 1 | v27_streamk_tile (grid 576) | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 512 | v9 | 0 | v27_streamk_tile (grid 512) | 10 | 0 | 0 | 0.072 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 512 | v9 | 1 | v27_streamk_tile (grid 512) | 10 | 0 | 0 | 0.077 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 576 | v9 | 0 | v27_streamk_tile (grid 576) | 10 | 0 | 0 | 0.084 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 576 | v9 | 1 | v27_streamk_tile (grid 576) | 10 | 0 | 0 | 0.072 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.071 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.073 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 512 | v9 | 0 | v27_streamk_tile (grid 512) | 10 | 0 | 0 | 0.084 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 512 | v9 | 1 | v27_streamk_tile (grid 512) | 10 | 0 | 0 | 0.085 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 576 | v9 | 0 | v27_streamk_tile (grid 576) | 10 | 0 | 0 | 0.077 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 576 | v9 | 1 | v27_streamk_tile (grid 576) | 10 | 0 | 0 | 0.089 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.078 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.083 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 512 | v9 | 0 | v27_streamk_tile (grid 512) | 10 | 0 | 0 | 0.124 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 512 | v9 | 1 | v27_streamk_tile (grid 512) | 10 | 0 | 0 | 0.127 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 576 | v9 | 0 | v27_streamk_tile (grid 576) | 10 | 0 | 0 | 0.126 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 576 | v9 | 1 | v27_streamk_tile (grid 576) | 10 | 0 | 0 | 0.165 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.176 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.200 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 512 | v9 | 0 | v27_streamk_tile (grid 512) | 10 | 0 | 0 | 0.146 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 512 | v9 | 1 | v27_streamk_tile (grid 512) | 10 | 0 | 0 | 0.114 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 576 | v9 | 0 | v27_streamk_tile (grid 576) | 10 | 0 | 0 | 0.133 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 576 | v9 | 1 | v27_streamk_tile (grid 576) | 10 | 0 | 0 | 0.130 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.140 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.167 |
| 1x8192x4096 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.071 |
| 1x8192x4096 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.081 |
| 1x8192x4096 | POLICY=spread, PARTIALS=fp32 | 512 | v9 | 0 | v27_streamk_tile (grid 512) | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=spread, PARTIALS=fp32 | 512 | v9 | 1 | v27_streamk_tile (grid 512) | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=spread, PARTIALS=fp32 | 576 | v9 | 0 | v27_streamk_tile (grid 576) | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=spread, PARTIALS=fp32 | 576 | v9 | 1 | v27_streamk_tile (grid 576) | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=spread, PARTIALS=bf16 | 512 | v9 | 0 | v27_streamk_tile (grid 512) | 10 | 0 | 0 | 0.076 |
| 1x8192x4096 | POLICY=spread, PARTIALS=bf16 | 512 | v9 | 1 | v27_streamk_tile (grid 512) | 10 | 0 | 0 | 0.085 |
| 1x8192x4096 | POLICY=spread, PARTIALS=bf16 | 576 | v9 | 0 | v27_streamk_tile (grid 576) | 10 | 0 | 0 | 0.082 |
| 1x8192x4096 | POLICY=spread, PARTIALS=bf16 | 576 | v9 | 1 | v27_streamk_tile (grid 576) | 10 | 0 | 0 | 0.088 |
| 1x8192x4096 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.085 |
| 1x8192x4096 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.072 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 504 | v9 | 0 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 504 | v9 | 1 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 567 | v9 | 0 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 567 | v9 | 1 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 504 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 504 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 567 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 567 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 504 | v9 | 0 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 504 | v9 | 1 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 567 | v9 | 0 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 567 | v9 | 1 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 504 | v9 | 0 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 504 | v9 | 1 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 567 | v9 | 0 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 567 | v9 | 1 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 504 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 504 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 567 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 567 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 504 | v9 | 0 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 504 | v9 | 1 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 567 | v9 | 0 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 567 | v9 | 1 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 504 | v9 | 0 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.187 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 504 | v9 | 1 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.121 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 567 | v9 | 0 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.128 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 567 | v9 | 1 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.124 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.111 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.109 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 504 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.137 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 504 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.171 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 567 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 567 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.121 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.155 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.138 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 504 | v9 | 0 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.150 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 504 | v9 | 1 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.148 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 567 | v9 | 0 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.118 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 567 | v9 | 1 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.179 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.146 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.113 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 504 | v9 | 0 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.121 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 504 | v9 | 1 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.139 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 567 | v9 | 0 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.124 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 567 | v9 | 1 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.143 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.108 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.134 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 504 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.140 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 504 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.119 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 567 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.120 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 567 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.157 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.157 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.144 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 504 | v9 | 0 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.120 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 504 | v9 | 1 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.122 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 567 | v9 | 0 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.174 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 567 | v9 | 1 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.121 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.134 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.127 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 504 | v9 | 0 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 504 | v9 | 1 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 567 | v9 | 0 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 567 | v9 | 1 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 504 | v9 | 0 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 504 | v9 | 1 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 567 | v9 | 0 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 567 | v9 | 1 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 504 | v9 | 0 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.117 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 504 | v9 | 1 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.178 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 567 | v9 | 0 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.135 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 567 | v9 | 1 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.127 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.132 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.151 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 504 | v9 | 0 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.113 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 504 | v9 | 1 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.112 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 567 | v9 | 0 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.131 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 567 | v9 | 1 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.138 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.116 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.129 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 504 | v9 | 0 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.206 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 504 | v9 | 1 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.237 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 567 | v9 | 0 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.223 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 567 | v9 | 1 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.243 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.323 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.273 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 504 | v9 | 0 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.250 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 504 | v9 | 1 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.216 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 567 | v9 | 0 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.238 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 567 | v9 | 1 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.239 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.331 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.268 |
| 32x4000x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.159 |
| 32x4000x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.116 |
| 32x4000x8192 | POLICY=spread, PARTIALS=fp32 | 504 | v9 | 0 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=spread, PARTIALS=fp32 | 504 | v9 | 1 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=spread, PARTIALS=fp32 | 567 | v9 | 0 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=spread, PARTIALS=fp32 | 567 | v9 | 1 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=spread, PARTIALS=bf16 | 504 | v9 | 0 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.139 |
| 32x4000x8192 | POLICY=spread, PARTIALS=bf16 | 504 | v9 | 1 | v27_streamk_tile (grid 504) | 10 | 0 | 0 | 0.128 |
| 32x4000x8192 | POLICY=spread, PARTIALS=bf16 | 567 | v9 | 0 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.162 |
| 32x4000x8192 | POLICY=spread, PARTIALS=bf16 | 567 | v9 | 1 | v27_streamk_tile (grid 567) | 10 | 0 | 0 | 0.116 |
| 32x4000x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 32x4000x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.138 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 252 | v9 | 0 | v27_streamk_tile (grid 252) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 252 | v9 | 1 | v27_streamk_tile (grid 252) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 283 | v9 | 0 | v27_streamk_tile (grid 283) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 283 | v9 | 1 | v27_streamk_tile (grid 283) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 1008) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 1008) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 252 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 252 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 283 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 283 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 252 | v9 | 0 | v27_streamk_tile (grid 252) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 252 | v9 | 1 | v27_streamk_tile (grid 252) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 283 | v9 | 0 | v27_streamk_tile (grid 283) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 283 | v9 | 1 | v27_streamk_tile (grid 283) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 1008) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 1008) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 252 | v9 | 0 | v27_streamk_tile (grid 252) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 252 | v9 | 1 | v27_streamk_tile (grid 252) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 283 | v9 | 0 | v27_streamk_tile (grid 283) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 283 | v9 | 1 | v27_streamk_tile (grid 283) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 1008) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 1008) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 252 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 252 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 283 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 283 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 252 | v9 | 0 | v27_streamk_tile (grid 252) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 252 | v9 | 1 | v27_streamk_tile (grid 252) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 283 | v9 | 0 | v27_streamk_tile (grid 283) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 283 | v9 | 1 | v27_streamk_tile (grid 283) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 1008) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 1008) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 252 | v9 | 0 | v27_streamk_tile (grid 252) | 10 | 0 | 0 | 0.095 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 252 | v9 | 1 | v27_streamk_tile (grid 252) | 10 | 0 | 0 | 0.092 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 283 | v9 | 0 | v27_streamk_tile (grid 283) | 10 | 0 | 0 | 0.078 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 283 | v9 | 1 | v27_streamk_tile (grid 283) | 10 | 0 | 0 | 0.076 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 1008) | 10 | 0 | 0 | 0.075 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 1008) | 10 | 0 | 0 | 0.097 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 252 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.086 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 252 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.090 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 283 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.101 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 283 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.093 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.081 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.078 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 252 | v9 | 0 | v27_streamk_tile (grid 252) | 10 | 0 | 0 | 0.088 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 252 | v9 | 1 | v27_streamk_tile (grid 252) | 10 | 0 | 0 | 0.070 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 283 | v9 | 0 | v27_streamk_tile (grid 283) | 10 | 0 | 0 | 0.083 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 283 | v9 | 1 | v27_streamk_tile (grid 283) | 10 | 0 | 0 | 0.094 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 1008) | 10 | 0 | 0 | 0.084 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 1008) | 10 | 0 | 0 | 0.075 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 252 | v9 | 0 | v27_streamk_tile (grid 252) | 10 | 0 | 0 | 0.092 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 252 | v9 | 1 | v27_streamk_tile (grid 252) | 10 | 0 | 0 | 0.077 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 283 | v9 | 0 | v27_streamk_tile (grid 283) | 10 | 0 | 0 | 0.098 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 283 | v9 | 1 | v27_streamk_tile (grid 283) | 10 | 0 | 0 | 0.082 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 1008) | 10 | 0 | 0 | 0.082 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 1008) | 10 | 0 | 0 | 0.077 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 252 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.100 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 252 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.076 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 283 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.087 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 283 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.099 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.087 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.068 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 252 | v9 | 0 | v27_streamk_tile (grid 252) | 10 | 0 | 0 | 0.084 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 252 | v9 | 1 | v27_streamk_tile (grid 252) | 10 | 0 | 0 | 0.082 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 283 | v9 | 0 | v27_streamk_tile (grid 283) | 10 | 0 | 0 | 0.085 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 283 | v9 | 1 | v27_streamk_tile (grid 283) | 10 | 0 | 0 | 0.089 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 1008) | 10 | 0 | 0 | 0.080 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 1008) | 10 | 0 | 0 | 0.078 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 252 | v9 | 0 | v27_streamk_tile (grid 252) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 252 | v9 | 1 | v27_streamk_tile (grid 252) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 283 | v9 | 0 | v27_streamk_tile (grid 283) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 283 | v9 | 1 | v27_streamk_tile (grid 283) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 1008) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 1008) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 252 | v9 | 0 | v27_streamk_tile (grid 252) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 252 | v9 | 1 | v27_streamk_tile (grid 252) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 283 | v9 | 0 | v27_streamk_tile (grid 283) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 283 | v9 | 1 | v27_streamk_tile (grid 283) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 1008) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 1008) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 252 | v9 | 0 | v27_streamk_tile (grid 252) | 10 | 0 | 0 | 0.077 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 252 | v9 | 1 | v27_streamk_tile (grid 252) | 10 | 0 | 0 | 0.084 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 283 | v9 | 0 | v27_streamk_tile (grid 283) | 10 | 0 | 0 | 0.076 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 283 | v9 | 1 | v27_streamk_tile (grid 283) | 10 | 0 | 0 | 0.082 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 1008) | 10 | 0 | 0 | 0.079 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 1008) | 10 | 0 | 0 | 0.072 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 252 | v9 | 0 | v27_streamk_tile (grid 252) | 10 | 0 | 0 | 0.097 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 252 | v9 | 1 | v27_streamk_tile (grid 252) | 10 | 0 | 0 | 0.083 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 283 | v9 | 0 | v27_streamk_tile (grid 283) | 10 | 0 | 0 | 0.090 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 283 | v9 | 1 | v27_streamk_tile (grid 283) | 10 | 0 | 0 | 0.083 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 1008) | 10 | 0 | 0 | 0.073 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 1008) | 10 | 0 | 0 | 0.070 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 252 | v9 | 0 | v27_streamk_tile (grid 252) | 10 | 0 | 0 | 0.167 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 252 | v9 | 1 | v27_streamk_tile (grid 252) | 10 | 0 | 0 | 0.141 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 283 | v9 | 0 | v27_streamk_tile (grid 283) | 10 | 0 | 0 | 0.128 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 283 | v9 | 1 | v27_streamk_tile (grid 283) | 10 | 0 | 0 | 0.126 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 1008) | 10 | 0 | 0 | 0.246 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 1008) | 10 | 0 | 0 | 0.230 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 252 | v9 | 0 | v27_streamk_tile (grid 252) | 10 | 0 | 0 | 0.160 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 252 | v9 | 1 | v27_streamk_tile (grid 252) | 10 | 0 | 0 | 0.121 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 283 | v9 | 0 | v27_streamk_tile (grid 283) | 10 | 0 | 0 | 0.146 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 283 | v9 | 1 | v27_streamk_tile (grid 283) | 10 | 0 | 0 | 0.135 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 1008) | 10 | 0 | 0 | 0.219 |
| 3x4000x4080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 1008) | 10 | 0 | 0 | 0.225 |
| 3x4000x4080 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile (grid 1008) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile (grid 1008) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile (grid 1008) | 10 | 0 | 0 | 0.080 |
| 3x4000x4080 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile (grid 1008) | 10 | 0 | 0 | 0.099 |
| 3x4000x4080 | POLICY=spread, PARTIALS=fp32 | 252 | v9 | 0 | v27_streamk_tile (grid 252) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=spread, PARTIALS=fp32 | 252 | v9 | 1 | v27_streamk_tile (grid 252) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=spread, PARTIALS=fp32 | 283 | v9 | 0 | v27_streamk_tile (grid 283) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=spread, PARTIALS=fp32 | 283 | v9 | 1 | v27_streamk_tile (grid 283) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile (grid 1008) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile (grid 1008) | 10 | 0 | 0 | 0.000 |
| 3x4000x4080 | POLICY=spread, PARTIALS=bf16 | 252 | v9 | 0 | v27_streamk_tile (grid 252) | 10 | 0 | 0 | 0.082 |
| 3x4000x4080 | POLICY=spread, PARTIALS=bf16 | 252 | v9 | 1 | v27_streamk_tile (grid 252) | 10 | 0 | 0 | 0.101 |
| 3x4000x4080 | POLICY=spread, PARTIALS=bf16 | 283 | v9 | 0 | v27_streamk_tile (grid 283) | 10 | 0 | 0 | 0.077 |
| 3x4000x4080 | POLICY=spread, PARTIALS=bf16 | 283 | v9 | 1 | v27_streamk_tile (grid 283) | 10 | 0 | 0 | 0.090 |
| 3x4000x4080 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile (grid 1008) | 10 | 0 | 0 | 0.082 |
| 3x4000x4080 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile (grid 1008) | 10 | 0 | 0 | 0.074 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 512 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 512 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 576 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 576 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 512 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 512 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 576 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 576 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 512 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 512 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 576 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 576 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 512 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 512 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 576 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 576 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 512 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 512 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 576 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 576 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 512 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 512 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 576 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 576 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 512 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.034 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 512 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.030 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 576 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.030 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 576 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.028 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.033 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.029 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 512 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.032 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 512 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.029 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 576 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.029 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 576 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.029 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.033 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.031 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 512 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.031 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 512 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.029 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 576 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.030 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 576 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.028 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.030 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.030 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 512 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.031 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 512 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.028 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 576 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.031 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 576 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.031 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.034 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.040 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 512 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.031 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 512 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.030 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 576 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.037 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 576 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.029 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.029 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.041 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 512 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.029 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 512 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.031 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 576 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.027 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 576 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.030 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.029 |
| 1x8192x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.033 |
| 1x8192x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 512 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 512 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 576 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 576 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 512 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 512 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 576 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 576 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 512 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.028 |
| 1x8192x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 512 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.025 |
| 1x8192x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 576 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.029 |
| 1x8192x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 576 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.040 |
| 1x8192x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.023 |
| 1x8192x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.026 |
| 1x8192x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 512 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.029 |
| 1x8192x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 512 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.028 |
| 1x8192x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 576 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.023 |
| 1x8192x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 576 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.042 |
| 1x8192x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.024 |
| 1x8192x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.025 |
| 1x8192x512 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 512 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.031 |
| 1x8192x512 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 512 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.031 |
| 1x8192x512 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 576 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.029 |
| 1x8192x512 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 576 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.034 |
| 1x8192x512 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.029 |
| 1x8192x512 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.032 |
| 1x8192x512 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 512 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.030 |
| 1x8192x512 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 512 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.036 |
| 1x8192x512 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 576 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.030 |
| 1x8192x512 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 576 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.058 |
| 1x8192x512 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.030 |
| 1x8192x512 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.030 |
| 1x8192x512 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.024 |
| 1x8192x512 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.027 |
| 1x8192x512 | POLICY=spread, PARTIALS=fp32 | 512 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=spread, PARTIALS=fp32 | 512 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=spread, PARTIALS=fp32 | 576 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=spread, PARTIALS=fp32 | 576 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.000 |
| 1x8192x512 | POLICY=spread, PARTIALS=bf16 | 512 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.027 |
| 1x8192x512 | POLICY=spread, PARTIALS=bf16 | 512 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.028 |
| 1x8192x512 | POLICY=spread, PARTIALS=bf16 | 576 | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.028 |
| 1x8192x512 | POLICY=spread, PARTIALS=bf16 | 576 | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.030 |
| 1x8192x512 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.027 |
| 1x8192x512 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile (grid 256) | 10 | 0 | 0 | 0.026 |

PASS
