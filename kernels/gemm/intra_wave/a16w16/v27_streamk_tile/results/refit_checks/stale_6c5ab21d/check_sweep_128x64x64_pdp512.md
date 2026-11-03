matmul_kernel.py sha256 6c5ab21df9acab9176ffb37076050e3ef0d88b5613e997369b0f14204786146e

tile 128x64x64: kernel BLOCK 128x64x64, 4 waves, 512 programs

| shape | configuration | STREAMK_NUM_PROGRAMS | tile order | bias | kernel | launches | wrong results | flags or counts left up | largest error |
|---|---|---|---|---|---|---|---|---|---|
| 2176x1024x8192 | POLICY=dp, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 2176x1024x8192 | POLICY=dp, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 2176x1088x8192 | POLICY=dp, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 2176x1088x8192 | POLICY=dp, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1664x1280x8192 | POLICY=dp, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 1664x1280x8192 | POLICY=dp, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 2048x1536x8192 | POLICY=dp, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 2048x1536x8192 | POLICY=dp, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 500x500x8080 | POLICY=dp, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 500x500x8080 | POLICY=dp, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 2150x1000x8176 | POLICY=dp, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 2150x1000x8176 | POLICY=dp, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3072x2048x4096 | POLICY=dp, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 3072x2048x4096 | POLICY=dp, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4468x2036x3984 | POLICY=dp, PARTIALS=fp32 | default | v9 | 0 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |
| 4468x2036x3984 | POLICY=dp, PARTIALS=fp32 | default | v9 | 1 | v27_streamk_tile | 10 | 0 | 0 | 0.000 |

PASS
