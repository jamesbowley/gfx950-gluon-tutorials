| round | K | version | config | kernel us | accvgpr_mov | s_nop | vgpr | spills | correct |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1024 | v9 | inproc | 30.40 | 128 | 19 | 448 | 0 | True |
| 1 | 1024 | v9 | llc | 31.84 | 128 | 19 | 448 | 0 | True |
| 1 | 1024 | v9 | llc:-amdgpu-mfma-vgpr-form=0 | 33.16 | 0 | 39 | 440 | 0 | True |
| 1 | 1024 | v10 | inproc | 30.92 | 128 | 49 | 480 | 0 | True |
| 1 | 1024 | v10 | llc | 31.16 | 128 | 51 | 476 | 0 | True |
| 1 | 1024 | v10 | llc:-amdgpu-mfma-vgpr-form=0 | 30.80 | 0 | 66 | 464 | 0 | True |
| 1 | 1024 | v11 | inproc | 30.96 | 0 | 9 | 452 | 0 | True |
| 1 | 1024 | v11 | llc | 32.02 | 0 | 21 | 452 | 0 | True |
| 1 | 1024 | v11 | llc:-amdgpu-mfma-vgpr-form=0 | 30.84 | 0 | 19 | 500 | 0 | True |
| 1 | 1024 | v12 | inproc | 30.96 | 0 | 47 | 440 | 0 | True |
| 1 | 1024 | v12 | llc | 31.08 | 0 | 48 | 440 | 0 | True |
| 1 | 1024 | v12 | llc:-amdgpu-mfma-vgpr-form=0 | 31.28 | 0 | 10 | 452 | 0 | True |
| 1 | 1024 | v13 | inproc | 31.24 | 348 | 121 | 480 | 0 | True |
| 1 | 1024 | v13 | llc | 31.24 | 348 | 121 | 476 | 0 | True |
| 1 | 1024 | v13 | llc:-amdgpu-mfma-vgpr-form=0 | 30.82 | 0 | 68 | 460 | 0 | True |
| 1 | 8192 | v9 | inproc | 175.00 | 128 | 19 | 448 | 0 | True |
| 1 | 8192 | v9 | llc | 175.12 | 128 | 19 | 448 | 0 | True |
| 1 | 8192 | v9 | llc:-amdgpu-mfma-vgpr-form=0 | 175.72 | 0 | 39 | 440 | 0 | True |
| 1 | 8192 | v10 | inproc | 175.22 | 128 | 49 | 480 | 0 | True |
| 1 | 8192 | v10 | llc | 176.08 | 128 | 51 | 476 | 0 | True |
| 1 | 8192 | v10 | llc:-amdgpu-mfma-vgpr-form=0 | 175.92 | 0 | 66 | 464 | 0 | True |
| 1 | 8192 | v11 | inproc | 174.60 | 0 | 9 | 452 | 0 | True |
| 1 | 8192 | v11 | llc | 174.54 | 0 | 21 | 452 | 0 | True |
| 1 | 8192 | v11 | llc:-amdgpu-mfma-vgpr-form=0 | 175.72 | 0 | 19 | 500 | 0 | True |
| 1 | 8192 | v12 | inproc | 176.86 | 0 | 47 | 440 | 0 | True |
| 1 | 8192 | v12 | llc | 176.08 | 0 | 48 | 440 | 0 | True |
| 1 | 8192 | v12 | llc:-amdgpu-mfma-vgpr-form=0 | 176.32 | 0 | 10 | 452 | 0 | True |
| 1 | 8192 | v13 | inproc | 177.40 | 348 | 121 | 480 | 0 | True |
| 1 | 8192 | v13 | llc | 177.10 | 348 | 121 | 476 | 0 | True |
| 1 | 8192 | v13 | llc:-amdgpu-mfma-vgpr-form=0 | 175.74 | 0 | 68 | 460 | 0 | True |
| 2 | 1024 | v9 | inproc | 30.46 | 128 | 19 | 448 | 0 | True |
| 2 | 1024 | v9 | llc | 30.64 | 128 | 19 | 448 | 0 | True |
| 2 | 1024 | v9 | llc:-amdgpu-mfma-vgpr-form=0 | 30.74 | 0 | 39 | 440 | 0 | True |
| 2 | 1024 | v10 | inproc | 30.96 | 128 | 49 | 480 | 0 | True |
| 2 | 1024 | v10 | llc | 31.08 | 128 | 51 | 476 | 0 | True |
| 2 | 1024 | v10 | llc:-amdgpu-mfma-vgpr-form=0 | 31.12 | 0 | 66 | 464 | 0 | True |
| 2 | 1024 | v11 | inproc | 30.92 | 0 | 9 | 452 | 0 | True |
| 2 | 1024 | v11 | llc | 30.92 | 0 | 21 | 452 | 0 | True |
| 2 | 1024 | v11 | llc:-amdgpu-mfma-vgpr-form=0 | 31.00 | 0 | 19 | 500 | 0 | True |
| 2 | 1024 | v12 | inproc | 30.96 | 0 | 47 | 440 | 0 | True |
| 2 | 1024 | v12 | llc | 30.96 | 0 | 48 | 440 | 0 | True |
| 2 | 1024 | v12 | llc:-amdgpu-mfma-vgpr-form=0 | 33.56 | 0 | 10 | 452 | 0 | True |
| 2 | 1024 | v13 | inproc | 31.44 | 348 | 121 | 480 | 0 | True |
| 2 | 1024 | v13 | llc | 33.54 | 348 | 121 | 476 | 0 | True |
| 2 | 1024 | v13 | llc:-amdgpu-mfma-vgpr-form=0 | 30.72 | 0 | 68 | 460 | 0 | True |
| 2 | 8192 | v9 | inproc | 175.56 | 128 | 19 | 448 | 0 | True |
| 2 | 8192 | v9 | llc | 175.14 | 128 | 19 | 448 | 0 | True |
| 2 | 8192 | v9 | llc:-amdgpu-mfma-vgpr-form=0 | 175.60 | 0 | 39 | 440 | 0 | True |
| 2 | 8192 | v10 | inproc | 174.80 | 128 | 49 | 480 | 0 | True |
| 2 | 8192 | v10 | llc | 175.56 | 128 | 51 | 476 | 0 | True |
| 2 | 8192 | v10 | llc:-amdgpu-mfma-vgpr-form=0 | 177.32 | 0 | 66 | 464 | 0 | True |
| 2 | 8192 | v11 | inproc | 175.74 | 0 | 9 | 452 | 0 | True |
| 2 | 8192 | v11 | llc | 176.12 | 0 | 21 | 452 | 0 | True |
| 2 | 8192 | v11 | llc:-amdgpu-mfma-vgpr-form=0 | 175.84 | 0 | 19 | 500 | 0 | True |
| 2 | 8192 | v12 | inproc | 176.06 | 0 | 47 | 440 | 0 | True |
| 2 | 8192 | v12 | llc | 176.40 | 0 | 48 | 440 | 0 | True |
| 2 | 8192 | v12 | llc:-amdgpu-mfma-vgpr-form=0 | 177.62 | 0 | 10 | 452 | 0 | True |
| 2 | 8192 | v13 | inproc | 176.90 | 348 | 121 | 480 | 0 | True |
| 2 | 8192 | v13 | llc | 176.12 | 348 | 121 | 476 | 0 | True |
| 2 | 8192 | v13 | llc:-amdgpu-mfma-vgpr-form=0 | 174.88 | 0 | 68 | 460 | 0 | True |
| 3 | 1024 | v9 | inproc | 30.56 | 128 | 19 | 448 | 0 | True |
| 3 | 1024 | v9 | llc | 30.44 | 128 | 19 | 448 | 0 | True |
| 3 | 1024 | v9 | llc:-amdgpu-mfma-vgpr-form=0 | 32.36 | 0 | 39 | 440 | 0 | True |
| 3 | 1024 | v10 | inproc | 31.04 | 128 | 49 | 480 | 0 | True |
| 3 | 1024 | v10 | llc | 30.84 | 128 | 51 | 476 | 0 | True |
| 3 | 1024 | v10 | llc:-amdgpu-mfma-vgpr-form=0 | 30.80 | 0 | 66 | 464 | 0 | True |
| 3 | 1024 | v11 | inproc | 30.84 | 0 | 9 | 452 | 0 | True |
| 3 | 1024 | v11 | llc | 30.84 | 0 | 21 | 452 | 0 | True |
| 3 | 1024 | v11 | llc:-amdgpu-mfma-vgpr-form=0 | 31.04 | 0 | 19 | 500 | 0 | True |
| 3 | 1024 | v12 | inproc | 31.08 | 0 | 47 | 440 | 0 | True |
| 3 | 1024 | v12 | llc | 31.56 | 0 | 48 | 440 | 0 | True |
| 3 | 1024 | v12 | llc:-amdgpu-mfma-vgpr-form=0 | 31.00 | 0 | 10 | 452 | 0 | True |
| 3 | 1024 | v13 | inproc | 31.24 | 348 | 121 | 480 | 0 | True |
| 3 | 1024 | v13 | llc | 31.56 | 348 | 121 | 476 | 0 | True |
| 3 | 1024 | v13 | llc:-amdgpu-mfma-vgpr-form=0 | 32.68 | 0 | 68 | 460 | 0 | True |
| 3 | 8192 | v9 | inproc | 175.30 | 128 | 19 | 448 | 0 | True |
| 3 | 8192 | v9 | llc | 176.04 | 128 | 19 | 448 | 0 | True |
| 3 | 8192 | v9 | llc:-amdgpu-mfma-vgpr-form=0 | 175.04 | 0 | 39 | 440 | 0 | True |
| 3 | 8192 | v10 | inproc | 176.00 | 128 | 49 | 480 | 0 | True |
| 3 | 8192 | v10 | llc | 175.54 | 128 | 51 | 476 | 0 | True |
| 3 | 8192 | v10 | llc:-amdgpu-mfma-vgpr-form=0 | 175.64 | 0 | 66 | 464 | 0 | True |
| 3 | 8192 | v11 | inproc | 175.54 | 0 | 9 | 452 | 0 | True |
| 3 | 8192 | v11 | llc | 175.56 | 0 | 21 | 452 | 0 | True |
| 3 | 8192 | v11 | llc:-amdgpu-mfma-vgpr-form=0 | 177.18 | 0 | 19 | 500 | 0 | True |
| 3 | 8192 | v12 | inproc | 176.60 | 0 | 47 | 440 | 0 | True |
| 3 | 8192 | v12 | llc | 176.64 | 0 | 48 | 440 | 0 | True |
| 3 | 8192 | v12 | llc:-amdgpu-mfma-vgpr-form=0 | 176.06 | 0 | 10 | 452 | 0 | True |
| 3 | 8192 | v13 | inproc | 175.68 | 348 | 121 | 480 | 0 | True |
| 3 | 8192 | v13 | llc | 177.88 | 348 | 121 | 476 | 0 | True |
| 3 | 8192 | v13 | llc:-amdgpu-mfma-vgpr-form=0 | 200.76 | 0 | 68 | 460 | 0 | True |

| K | version | config | median of rounds (us) | min | max |
|---|---|---|---|---|---|
| 1024 | v9 | inproc | 30.46 | 30.40 | 30.56 |
| 1024 | v9 | llc | 30.64 | 30.44 | 31.84 |
| 1024 | v9 | llc:-amdgpu-mfma-vgpr-form=0 | 32.36 | 30.74 | 33.16 |
| 1024 | v10 | inproc | 30.96 | 30.92 | 31.04 |
| 1024 | v10 | llc | 31.08 | 30.84 | 31.16 |
| 1024 | v10 | llc:-amdgpu-mfma-vgpr-form=0 | 30.80 | 30.80 | 31.12 |
| 1024 | v11 | inproc | 30.92 | 30.84 | 30.96 |
| 1024 | v11 | llc | 30.92 | 30.84 | 32.02 |
| 1024 | v11 | llc:-amdgpu-mfma-vgpr-form=0 | 31.00 | 30.84 | 31.04 |
| 1024 | v12 | inproc | 30.96 | 30.96 | 31.08 |
| 1024 | v12 | llc | 31.08 | 30.96 | 31.56 |
| 1024 | v12 | llc:-amdgpu-mfma-vgpr-form=0 | 31.28 | 31.00 | 33.56 |
| 1024 | v13 | inproc | 31.24 | 31.24 | 31.44 |
| 1024 | v13 | llc | 31.56 | 31.24 | 33.54 |
| 1024 | v13 | llc:-amdgpu-mfma-vgpr-form=0 | 30.82 | 30.72 | 32.68 |
| 8192 | v9 | inproc | 175.30 | 175.00 | 175.56 |
| 8192 | v9 | llc | 175.14 | 175.12 | 176.04 |
| 8192 | v9 | llc:-amdgpu-mfma-vgpr-form=0 | 175.60 | 175.04 | 175.72 |
| 8192 | v10 | inproc | 175.22 | 174.80 | 176.00 |
| 8192 | v10 | llc | 175.56 | 175.54 | 176.08 |
| 8192 | v10 | llc:-amdgpu-mfma-vgpr-form=0 | 175.92 | 175.64 | 177.32 |
| 8192 | v11 | inproc | 175.54 | 174.60 | 175.74 |
| 8192 | v11 | llc | 175.56 | 174.54 | 176.12 |
| 8192 | v11 | llc:-amdgpu-mfma-vgpr-form=0 | 175.84 | 175.72 | 177.18 |
| 8192 | v12 | inproc | 176.60 | 176.06 | 176.86 |
| 8192 | v12 | llc | 176.40 | 176.08 | 176.64 |
| 8192 | v12 | llc:-amdgpu-mfma-vgpr-form=0 | 176.32 | 176.06 | 177.62 |
| 8192 | v13 | inproc | 176.90 | 175.68 | 177.40 |
| 8192 | v13 | llc | 177.10 | 176.12 | 177.88 |
| 8192 | v13 | llc:-amdgpu-mfma-vgpr-form=0 | 175.74 | 174.88 | 200.76 |
