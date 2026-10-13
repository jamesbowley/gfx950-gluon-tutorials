| spec | shape | correct | dot layout (TTGIR) | LDS encodings | MFMA | MFMAs | v_dot2 | v_fma | ds_bpermute | async LDS loads | global loads | VGPRs | LDS B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tri1x256x128 | 2048x4096x8192 | True | `none (FMA)` | swizzled_shared | - | 0 | 128 | 0 | 0 | 32 | 0 | 106 | 131584 |
| tri2x256x128 | 2048x4096x8192 | True | `none (FMA)` | swizzled_shared | - | 0 | 256 | 0 | 0 | 32 | 0 | 172 | 132096 |
| tri4x256x128 | 2048x4096x8192 | True | `#ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 4], instrShape = [4, 64, 64], isTransposed = false}>` | swizzled_shared | v_mfma_f32_4x4x4_16b_bf16 | 64 | 0 | 0 | 130 | 130 | 0 | 128 | 133120 |
| tri8x256x128 | 2048x4096x8192 | True | `#ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 4], instrShape = [4, 64, 64], isTransposed = false}>` | swizzled_shared | v_mfma_f32_4x4x4_16b_bf16 | 128 | 0 | 0 | 132 | 132 | 0 | 131 | 135168 |
| tri16x256x128 | 2048x4096x8192 | True | `#ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 4], instrShape = [16, 16, 32], isTransposed = true}>` | padded_shared,swizzled_shared | v_mfma_f32_16x16x32_bf16 | 32 | 0 | 0 | 2 | 34 | 0 | 94 | 143328 |
| tri256x4x128 | 2048x256x8192 | True | `#ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [64, 4, 64], isTransposed = true}>` | swizzled_shared | v_mfma_f32_4x4x4_16b_bf16 | 64 | 0 | 0 | 130 | 130 | 0 | 259 | 133120 |
| tri256x8x128 | 2048x256x8192 | True | `#ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [64, 4, 64], isTransposed = true}>` | swizzled_shared | v_mfma_f32_4x4x4_16b_bf16 | 128 | 0 | 0 | 132 | 132 | 0 | 327 | 135168 |
| tri16x16x128w1 | 2048x4096x8192 | True | `#ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>` | swizzled_shared | v_mfma_f32_16x16x32_bf16 | 8 | 0 | 0 | 16 | 16 | 0 | 54 | 16384 |
| tri16x16x128 | 2048x4096x8192 | True | `#ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 32], isTransposed = true}>` | swizzled_shared | v_mfma_f32_16x16x32_bf16 | 16 | 0 | 0 | 8 | 8 | 0 | 58 | 16384 |
| tri16x64x128w1 | 2048x4096x8192 | True | `#ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>` | padded_shared,swizzled_shared | v_mfma_f32_16x16x32_bf16 | 32 | 0 | 0 | 8 | 40 | 0 | 94 | 41952 |
