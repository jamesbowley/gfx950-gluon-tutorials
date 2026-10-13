MAC fit, reduce-scatter (60 stamped GEMM runs): o = 1.72 us per segment, rms residual 3.46 us; t(A), us per k-step: 2 CUs 1.01, 4 CUs 1.00, 8 CUs 1.03, 16 CUs 1.07, 32 CUs 1.07, 64 CUs 1.11, 128 CUs 1.19, 256 CUs 1.40
MAC fit, owner (60 stamped GEMM runs): o = 3.87 us per segment, rms residual 4.96 us; t(A), us per k-step: 2 CUs 1.01, 4 CUs 1.06, 8 CUs 1.10, 16 CUs 1.14, 32 CUs 1.14, 64 CUs 1.15, 128 CUs 1.22, 256 CUs 1.42

Best split by the model, per S and K: pattern n (model tail us / measured data-parallel tail us).

| S | K = 512 | K = 1024 | K = 2048 | K = 4096 | K = 8192 | K = 16384 |
|---|---|---|---|---|---|---|
| 1 | data-parallel | rs 8 (11 / 14) | rs 12 (13 / 25) | rs 16 (15 / 52) | rs 16 (20 / 110) | rs 32 (27 / 232) |
| 2 | data-parallel | rs 8 (12 / 15) | rs 12 (13 / 27) | rs 16 (16 / 55) | rs 16 (20 / 119) | rs 32 (28 / 238) |
| 3 | data-parallel | rs 8 (12 / 15) | rs 12 (14 / 27) | rs 16 (16 / 56) | rs 16 (21 / 119) | rs 32 (29 / 239) |
| 4 | data-parallel | rs 8 (12 / 15) | rs 12 (14 / 28) | rs 16 (17 / 56) | rs 16 (22 / 118) | rs 16 (30 / 242) |
| 6 | data-parallel | rs 8 (12 / 15) | rs 12 (14 / 28) | rs 16 (17 / 56) | rs 16 (22 / 118) | rs 16 (31 / 244) |
| 8 | data-parallel | rs 8 (12 / 15) | rs 8 (15 / 28) | rs 12 (18 / 56) | rs 16 (23 / 118) | rs 16 (33 / 251) |
| 12 | data-parallel | rs 6 (14 / 15) | rs 8 (17 / 28) | rs 8 (21 / 56) | rs 12 (30 / 118) | rs 16 (42 / 246) |
| 16 | data-parallel | rs 6 (15 / 15) | rs 8 (18 / 27) | rs 8 (22 / 56) | rs 8 (32 / 119) | rs 16 (46 / 248) |
| 24 | data-parallel | rs 4 (16 / 16) | rs 4 (20 / 28) | rs 6 (27 / 56) | rs 8 (39 / 119) | rs 8 (60 / 238) |
| 32 | data-parallel | data-parallel | rs 4 (22 / 28) | rs 6 (31 / 56) | rs 8 (44 / 119) | rs 8 (67 / 236) |
| 38 | data-parallel | data-parallel | rs 4 (24 / 28) | rs 6 (31 / 56) | rs 6 (46 / 119) | rs 6 (75 / 234) |
| 48 | data-parallel | data-parallel | rs 4 (27 / 28) | rs 4 (38 / 56) | rs 4 (59 / 119) | rs 4 (101 / 239) |
| 64 | data-parallel | data-parallel | data-parallel | rs 4 (44 / 56) | rs 4 (66 / 119) | rs 4 (111 / 241) |
| 96 | data-parallel | data-parallel | owner 2 (35 / 35) | owner 2 (56 / 60) | owner 2 (99 / 120) | owner 2 (184 / 235) |
| 128 | data-parallel | data-parallel | data-parallel | owner 2 (60 / 61) | owner 2 (106 / 123) | owner 2 (197 / 283) |

/home/jbowley/repos/gfx950-gluon-tutorials/experiments/memory_model_litmus/images/split_optimum.png
