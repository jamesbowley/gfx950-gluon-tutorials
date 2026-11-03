matmul_kernel.py sha256 6c5ab21df9acab9176ffb37076050e3ef0d88b5613e997369b0f14204786146e

tile 64x128x64: kernel BLOCK 64x128x64, 4 waves, 512 programs

| shape | configuration | STREAMK_NUM_PROGRAMS | tile order | bias | kernel | launches | wrong results | flags or counts left up | largest error |
|---|---|---|---|---|---|---|---|---|---|
| 1088x2048x8192 | POLICY=dp, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x2048x8192 | POLICY=dp, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x2176x8192 | POLICY=dp, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1088x2176x8192 | POLICY=dp, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x2560x8192 | POLICY=dp, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 832x2560x8192 | POLICY=dp, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1024x3072x8192 | POLICY=dp, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1024x3072x8192 | POLICY=dp, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x1000x8080 | POLICY=dp, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 250x1000x8080 | POLICY=dp, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x2000x8176 | POLICY=dp, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1075x2000x8176 | POLICY=dp, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1536x4096x4096 | POLICY=dp, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1536x4096x4096 | POLICY=dp, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 2234x4072x3984 | POLICY=dp, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 2234x4072x3984 | POLICY=dp, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |

PASS
