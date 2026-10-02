# Rotating owner against owner = chunk 0 (HIP_VISIBLE_DEVICES=2, bf16, v9 tile order)

Scratch build: v18 with `owner_j = t * 0` instead of `t % n` (not kept), timed against v18 in the
same process with do_bench, one discarded warm-up timing per shape. Both builds checked against
torch on every shape.

| shape | owner = chunk 0 | rotating owner | rotating vs chunk 0 | both correct |
|---|---|---|---|---|
| 4352x4096x8192 | 1143 | 1184 | +3.6% | yes |
| 4352x4352x8192 | 1205 | 1305 | +8.3% | yes |
| 4352x4096x16384 | 1269 | 1298 | +2.3% | yes |
| 3328x5120x8192 | 773 | 775 | +0.2% | yes |
| 8192x8448x8192 | 1534 | 1572 | +2.5% | yes |
