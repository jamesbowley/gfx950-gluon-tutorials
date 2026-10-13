M=4096 N=4096, bf16 inputs.

| Kernel | Stack | Bias | Out | K | Arch VGPRs | AGPRs | .vgpr_count | Spills | Scratch B | Spill ops (where) | Correct |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| v9 | m2 | off | bf16 | 8192 | 190 | 256 | 448 | 0 | 0 | - | yes |
| v9 | m2 | on | bf16 | 8192 | 238 | 256 | 496 | 0 | 0 | - | yes |
| v9 | m2 | off | fp32 | 8192 | 214 | 256 | 472 | 0 | 0 | - | yes |
| v9 | m2 | on | fp32 | 8192 | 240 | 256 | 496 | 0 | 0 | - | yes |
| v10 | m2 | off | bf16 | 8192 | 224 | 256 | 480 | 0 | 0 | - | yes |
| v10 | m2 | on | bf16 | 8192 | 256 | 256 | 512 | 6 | 28 | tile_loop 6, outside 6 | yes |
| v10 | m2 | off | fp32 | 8192 | 244 | 256 | 500 | 0 | 0 | - | yes |
| v10 | m2 | on | fp32 | 8192 | 256 | 256 | 512 | 24 | 100 | tile_loop 16, outside 8 | yes |
