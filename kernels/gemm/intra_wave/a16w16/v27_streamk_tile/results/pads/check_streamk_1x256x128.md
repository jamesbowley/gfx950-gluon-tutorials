tile 1x256x128: kernel BLOCK 1x256x128, 4 waves, 256 programs

| shape | configuration | STREAMK_NUM_PROGRAMS | tile order | bias | kernel | launches | wrong results | flags or counts left up | largest error |
|---|---|---|---|---|---|---|---|---|---|
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.106 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.106 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.085 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.117 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.116 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.089 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.100 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.062 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.102 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.116 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.075 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.098 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.069 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.097 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.104 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.078 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.118 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.121 |
| 16x4096x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.075 |
| 16x4096x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.109 |
| 16x4096x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.110 |
| 16x4096x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 16x4096x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.120 |
| 16x4096x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.102 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.098 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.062 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.062 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.101 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.095 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.112 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.102 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.062 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.062 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.101 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.097 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.069 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.076 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.082 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.110 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.096 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.098 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.097 |
| 30x4000x4080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.093 |
| 30x4000x4080 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.089 |
| 30x4000x4080 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.092 |
| 30x4000x4080 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 30x4000x4080 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.098 |
| 30x4000x4080 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.098 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.096 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.094 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.077 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.070 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.085 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.084 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.085 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.079 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.093 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.079 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.072 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.077 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.096 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.098 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.093 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.079 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.076 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.080 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.097 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.098 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.089 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.078 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.079 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.097 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.074 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.086 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.081 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.084 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.075 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.083 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.079 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.087 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.082 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.095 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.079 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.072 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.083 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.091 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.088 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.076 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.083 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.078 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.080 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.080 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.080 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.095 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.074 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.084 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.145 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.139 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.133 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.182 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.178 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.193 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.122 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.137 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.137 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.142 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.153 |
| 1x8192x4096 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.169 |
| 1x8192x4096 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.080 |
| 1x8192x4096 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.074 |
| 1x8192x4096 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1x8192x4096 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.066 |
| 1x8192x4096 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.076 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.111 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.104 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.097 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.107 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.107 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.119 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.110 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.122 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.116 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.119 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.125 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.120 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.123 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.099 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.114 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.122 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.075 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.108 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.117 |
| 32x4000x8192 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.120 |
| 32x4000x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.118 |
| 32x4000x8192 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.108 |
| 32x4000x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 32x4000x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.106 |
| 32x4000x8192 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.091 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.116 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.100 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.100 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.104 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.095 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.119 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.111 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.102 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.111 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.116 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.093 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.100 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.115 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.107 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.094 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.119 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.101 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.120 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.096 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.108 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.118 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.089 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.103 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.114 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.093 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.102 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.088 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.082 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.110 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.092 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.119 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.103 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.097 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.090 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.116 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.128 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.124 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.100 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.114 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.138 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.111 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.122 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.119 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.121 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.122 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.109 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.155 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.172 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.166 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.168 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.301 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.265 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.173 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 64 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.179 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.189 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 72 | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.193 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.298 |
| 17x4032x8080 | POLICY=one_tile, FIXUP=atomic_iz, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.339 |
| 17x4032x8080 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=two_tile, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.122 |
| 17x4032x8080 | POLICY=two_tile, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.128 |
| 17x4032x8080 | POLICY=spread, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=spread, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 17x4032x8080 | POLICY=spread, PARTIALS=bf16 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.088 |
| 17x4032x8080 | POLICY=spread, PARTIALS=bf16 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.119 |

PASS
