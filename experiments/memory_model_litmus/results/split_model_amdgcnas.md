MAC fit, reduce-scatter (60 stamped GEMM runs): o = 1.85 us per segment, rms residual 3.23 us; t(A), us per k-step: 2 CUs 1.03, 4 CUs 1.04, 8 CUs 1.06, 16 CUs 1.09, 32 CUs 1.07, 64 CUs 1.11, 128 CUs 1.18, 256 CUs 1.39
MAC fit, owner (60 stamped GEMM runs): o = 3.68 us per segment, rms residual 4.99 us; t(A), us per k-step: 2 CUs 1.04, 4 CUs 1.07, 8 CUs 1.11, 16 CUs 1.13, 32 CUs 1.14, 64 CUs 1.15, 128 CUs 1.22, 256 CUs 1.43

Best split by the model, per S and K: pattern n (model tail us / measured data-parallel tail us).

| S | K = 2048 | K = 4096 | K = 8192 | K = 16384 |
|---|---|---|---|---|
| 1 | rs 12 (13 / 26) | rs 16 (16 / 52) | rs 16 (20 / 106) | rs 32 (27 / 231) |
| 2 | rs 12 (14 / 28) | rs 16 (16 / 55) | rs 16 (20 / 114) | rs 32 (28 / 236) |
| 3 | rs 12 (14 / 29) | rs 16 (16 / 55) | rs 16 (21 / 114) | rs 32 (29 / 239) |
| 4 | rs 12 (14 / 29) | rs 16 (17 / 56) | rs 16 (22 / 114) | rs 16 (31 / 243) |
| 6 | rs 12 (15 / 29) | rs 16 (17 / 56) | rs 16 (22 / 114) | rs 16 (31 / 245) |
| 8 | rs 8 (15 / 28) | rs 12 (18 / 56) | rs 16 (23 / 114) | rs 16 (33 / 249) |
| 12 | rs 8 (17 / 28) | rs 8 (21 / 56) | rs 12 (30 / 114) | rs 16 (42 / 251) |
| 16 | rs 8 (18 / 28) | rs 8 (22 / 56) | rs 8 (32 / 114) | rs 16 (46 / 252) |
| 24 | rs 4 (20 / 29) | rs 6 (27 / 56) | rs 8 (39 / 114) | rs 8 (59 / 235) |
| 32 | rs 4 (22 / 29) | rs 6 (31 / 56) | rs 8 (44 / 114) | rs 8 (67 / 237) |
| 38 | rs 4 (24 / 29) | rs 6 (31 / 57) | rs 6 (46 / 114) | rs 6 (75 / 235) |
| 48 | rs 4 (27 / 29) | rs 4 (38 / 57) | rs 4 (58 / 115) | rs 4 (100 / 242) |
| 64 | data-parallel | rs 4 (44 / 57) | rs 4 (66 / 115) | rs 4 (110 / 242) |
| 96 | owner 2 (34 / 35) | owner 2 (56 / 61) | owner 2 (99 / 116) | rs 2 (183 / 236) |
| 128 | data-parallel | owner 2 (60 / 64) | owner 2 (106 / 119) | owner 2 (197 / 284) |

/home/jbowley/repos/gfx950-gluon-tutorials/experiments/memory_model_litmus/images/split_optimum.png
