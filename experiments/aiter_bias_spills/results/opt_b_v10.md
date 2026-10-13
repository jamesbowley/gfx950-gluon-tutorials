M=4096 N=4096, bf16 inputs.

| Kernel | Stack | Bias | Out | K | Arch VGPRs | AGPRs | .vgpr_count | Spills | Scratch B | Spill ops (where) | Correct |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| v10 | m1 | off | bf16 | 8192 | 224 | 256 | 480 | 0 | 0 | - | yes |
| v10 | m1 | on | bf16 | 8192 | 256 | 256 | 512 | 0 | 0 | - | yes |
| v10 | m1 | off | fp32 | 8192 | 244 | 256 | 500 | 0 | 0 | - | yes |
| v10 | m1 | on | fp32 | 8192 | 256 | 256 | 512 | 14 | 60 | tile_loop 10, outside 6 | yes |
