Stack: `llir+amdgcnas`, K = 8192, one loop iteration = two k-steps.

| tile | VGPRs | AGPRs | spills | loop insts | MFMAs | buffer loads | ds_reads | s_waitcnt | s_barrier | s_nop | v_accvgpr | scratch | longest non-MFMA run | MFMAs between buffer loads |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 256x256 | 448 | 256 | 0 | 423 | 256 | 32 | 64 | 20 | 8 | 0 | 0 | 0 | 5 | 4-20 |
| 256x128 | 304 | 160 | 0 | 251 | 128 | 24 | 48 | 8 | 8 | 0 | 0 | 0 | 8 | 2-12 |
| 128x256 | 404 | 168 | 0 | 251 | 128 | 24 | 48 | 8 | 8 | 0 | 0 | 0 | 8 | 2-12 |
| 128x128 | 176 | 76 | 0 | 155 | 64 | 16 | 32 | 8 | 8 | 0 | 0 | 0 | 8 | 2-6 |
| 128x128r2 | 252 | 88 | 0 | 157 | 64 | 16 | 32 | 4 | 4 | 4 | 0 | 0 | 17 | 0-12 |
| 128x128x128 | 260 | 80 | 0 | 283 | 128 | 32 | 64 | 8 | 8 | 0 | 0 | 0 | 12 | 3-5 |
