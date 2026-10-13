M=4096 N=4096, bf16 inputs.

| Kernel | Stack | Bias | Out | K | Arch VGPRs | AGPRs | .vgpr_count | Spills | Scratch B | Spill ops (where) | Correct |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| v9 | m3 | off | bf16 | 8192 | 232 | 256 | 488 | 0 | 0 | - | yes |
| v9 | m3 | on | bf16 | 8192 | 256 | 256 | 512 | 0 | 0 | - | yes |
| v9 | m3 | off | fp32 | 8192 | 216 | 256 | 472 | 0 | 0 | - | yes |
| v9 | m3 | on | fp32 | 8192 | 256 | 256 | 512 | 20 | 84 | outside 12 | yes |
| v10 | m3 | off | bf16 | 8192 | 232 | 256 | 488 | 0 | 0 | - | yes |
| v10 | m3 | on | bf16 | 8192 | 256 | 256 | 512 | 18 | 76 | tile_loop 13, outside 9 | yes |
| v10 | m3 | off | fp32 | 8192 | 216 | 256 | 472 | 0 | 0 | - | yes |
| v10 | m3 | on | fp32 | 8192 | 256 | 256 | 512 | 40 | 164 | tile_loop 24, outside 12 | yes |
| tut_v10 | m3 | off | bf16 | 8192 | 232 | 256 | 488 | 0 | 0 | - | yes |
| tut_v10 | m3 | on | bf16 | 8192 | 256 | 256 | 512 | 18 | 76 | tile_loop 13, outside 9 | yes |
| tut_v10 | m3 | off | fp32 | 8192 | 216 | 256 | 472 | 0 | 0 | - | yes |
| tut_v10 | m3 | on | fp32 | 8192 | 256 | 256 | 512 | 40 | 164 | tile_loop 24, outside 12 | yes |
| v9 | m3a | off | bf16 | 8192 | 232 | 256 | 488 | 0 | 0 | - | yes |
| v9 | m3a | on | bf16 | 8192 | 256 | 256 | 512 | 0 | 0 | - | yes |
| v9 | m3a | off | fp32 | 8192 | 216 | 256 | 472 | 0 | 0 | - | yes |
| v9 | m3a | on | fp32 | 8192 | 256 | 256 | 512 | 20 | 84 | outside 12 | yes |
| v10 | m3a | off | bf16 | 8192 | 232 | 256 | 488 | 0 | 0 | - | yes |
| v10 | m3a | on | bf16 | 8192 | 256 | 256 | 512 | 18 | 76 | tile_loop 13, outside 9 | yes |
| v10 | m3a | off | fp32 | 8192 | 216 | 256 | 472 | 0 | 0 | - | yes |
| v10 | m3a | on | fp32 | 8192 | 256 | 256 | 512 | 40 | 164 | tile_loop 24, outside 12 | yes |
| tut_v10 | m3a | off | bf16 | 8192 | 232 | 256 | 488 | 0 | 0 | - | yes |
| tut_v10 | m3a | on | bf16 | 8192 | 256 | 256 | 512 | 18 | 76 | tile_loop 13, outside 9 | yes |
| tut_v10 | m3a | off | fp32 | 8192 | 216 | 256 | 472 | 0 | 0 | - | yes |
| tut_v10 | m3a | on | fp32 | 8192 | 256 | 256 | 512 | 40 | 164 | tile_loop 24, outside 12 | yes |
