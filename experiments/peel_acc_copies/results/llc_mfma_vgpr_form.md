| input | llc options | accvgpr_mov | s_nop | accvgpr_read | accvgpr_write | vgpr_count | agpr_count | spills |
|---|---|---|---|---|---|---|---|---|
| v9.llir | baseline | 128 | 76 | 288 | 288 | 448 | 256 | 0 |
| v9.llir | amdgpu-mfma-vgpr-form=0 | 0 | 96 | 276 | 276 | 440 | 256 | 0 |
| v10.llir | baseline | 128 | 98 | 288 | 288 | 476 | 256 | 0 |
| v10.llir | amdgpu-mfma-vgpr-form=0 | 0 | 95 | 280 | 280 | 452 | 256 | 0 |
| v11.llir | baseline | 0 | 68 | 256 | 256 | 452 | 256 | 0 |
| v11.llir | amdgpu-mfma-vgpr-form=0 | 0 | 71 | 256 | 256 | 500 | 256 | 0 |
| v12.llir | baseline | 0 | 104 | 256 | 256 | 440 | 256 | 0 |
| v12.llir | amdgpu-mfma-vgpr-form=0 | 0 | 67 | 300 | 300 | 468 | 256 | 0 |
| v13.llir | baseline | 312 | 217 | 320 | 64 | 496 | 256 | 0 |
| v13.llir | amdgpu-mfma-vgpr-form=0 | 0 | 92 | 320 | 64 | 480 | 256 | 0 |
| v9_bias.llir | baseline | 188 | 81 | 300 | 300 | 460 | 256 | 0 |
| v9_bias.llir | amdgpu-mfma-vgpr-form=0 | 0 | 96 | 288 | 288 | 460 | 256 | 0 |
| v10_bias.llir | baseline | 152 | 88 | 296 | 296 | 488 | 256 | 0 |
| v10_bias.llir | amdgpu-mfma-vgpr-form=0 | 0 | 97 | 288 | 288 | 472 | 256 | 0 |
| v11_bias.llir | baseline | 132 | 95 | 264 | 264 | 464 | 256 | 0 |
| v11_bias.llir | amdgpu-mfma-vgpr-form=0 | 0 | 68 | 272 | 272 | 496 | 256 | 0 |
| v12_bias.llir | baseline | 0 | 87 | 256 | 256 | 444 | 256 | 0 |
| v12_bias.llir | amdgpu-mfma-vgpr-form=0 | 0 | 61 | 300 | 300 | 476 | 256 | 0 |
| v13_bias.llir | baseline | 276 | 186 | 320 | 96 | 508 | 256 | 0 |
| v13_bias.llir | amdgpu-mfma-vgpr-form=0 | 0 | 162 | 360 | 136 | 512 | 256 | 15 |
| v13_K8192.llir | baseline | 312 | 217 | 320 | 64 | 496 | 256 | 0 |
| v13_K8192.llir | amdgpu-mfma-vgpr-form=0 | 0 | 92 | 320 | 64 | 480 | 256 | 0 |
