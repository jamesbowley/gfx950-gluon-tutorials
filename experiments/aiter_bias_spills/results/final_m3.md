M=4096 N=4096, bf16 inputs.

| Kernel | Stack | Bias | Out | K | Arch VGPRs | AGPRs | .vgpr_count | Spills | Scratch B | Spill ops (where) | Correct |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| v10 | m3 | off | bf16 | 8192 | 232 | 256 | 488 | 0 | 0 | - | yes |
| v10 | m3 | on | bf16 | 8192 | 252 | 256 | 508 | 0 | 0 | - | yes |
| v10 | m3 | off | fp32 | 8192 | 216 | 256 | 472 | 0 | 0 | - | yes |
| v10 | m3 | on | fp32 | 8192 | 244 | 256 | 500 | 0 | 0 | - | yes |
| v11 | m3 | off | bf16 | 8192 | 232 | 256 | 488 | 0 | 0 | - | yes |
| v11 | m3 | on | bf16 | 8192 | 256 | 256 | 512 | 19 | 80 | tile_loop 18, outside 12 | yes |
| v11 | m3 | off | fp32 | 8192 | 222 | 256 | 480 | 0 | 0 | - | yes |
| v11 | m3 | on | fp32 | 8192 | 256 | 256 | 512 | 49 | 200 | tile_loop 30, outside 10 | yes |
| v12 | m3 | off | bf16 | 8192 | 242 | 256 | 500 | 0 | 0 | - | yes |
| v12 | m3 | on | bf16 | 8192 | 252 | 256 | 508 | 0 | 0 | - | yes |
| v12 | m3 | off | fp32 | 8192 | 198 | 256 | 456 | 0 | 0 | - | yes |
| v12 | m3 | on | fp32 | 8192 | 256 | 256 | 512 | 27 | 112 | tile_loop 18, outside 10 | yes |
| v13 | m3 | off | bf16 | 8192 | 202 | 256 | 460 | 0 | 0 | - | yes |
| v13 | m3 | on | bf16 | 8192 | 224 | 256 | 480 | 0 | 0 | - | yes |
| v13 | m3 | off | fp32 | 8192 | 198 | 256 | 456 | 0 | 0 | - | yes |
| v13 | m3 | on | fp32 | 8192 | 202 | 256 | 460 | 0 | 0 | - | yes |
