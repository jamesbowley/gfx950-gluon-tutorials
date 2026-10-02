# Reversed against forward two-tile processing (HIP_VISIBLE_DEVICES=2, bf16, v9 tile order)

Scratch build: v19 with the two-tile segment loop in k order (owner = the segment holding
k-step 0, peers counted up), everything else identical (not kept). do_bench TFLOPS in one
process, one discarded warm-up per shape; both builds checked against torch, flags re-armed.

| shape | leftover tiles | forward two-tile | reversed two-tile (v19) | reversed vs forward | both correct |
|---|---|---|---|---|---|
| 1536x11008x8192 | 2 | 1187 | 1288 | +8.5% | yes |
| 3328x5120x8192 | 4 | 1134 | 1253 | +10.5% | yes |
| 4352x4096x8192 | 16 | 821 | 1272 | +55.0% | yes |
| 8192x8448x8192 | 32 | 1348 | 1590 | +17.9% | yes |
| 4096x5120x8192 | 64 | 1104 | 1236 | +11.9% | yes |
| 4096x6144x8192 | 128 | 1357 | 1389 | +2.4% | yes |
| 4096x7168x4096 | 192 | 1071 | 1177 | +9.9% | yes |
| 4096x7168x8192 | 192 | 1150 | 1237 | +7.6% | yes |
| 4096x7168x16384 | 192 | 1227 | 1338 | +9.0% | yes |
| 9472x10240x8192 | 200 | 1398 | 1463 | +4.7% | yes |
| 8192x7936x8192 | 224 | 1166 | 1393 | +19.5% | yes |
| 4096x7936x8192 | 240 | 922 | 1118 | +21.2% | yes |
| 4608x7168x8192 | 248 | 1011 | 1149 | +13.6% | yes |
| 1792x18688x8192 | 255 | 1272 | 1385 | +8.9% | yes |
