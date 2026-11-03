# Held-out test: the REFIT rule on 617 real-model shapes

Rule `refit_rule.py` (sha256 cd5676840ce5, table 9473c50b1c89), picks frozen in `results/heldout_manifest.json` before any timing. Timed like REFIT (rounds 1-2 on different GPUs, round 3 for close calls; `results/heldout_report.md`), noise sigma 1.24%. Regret: time over the best point timed for the shape, which is a reduced factorial (the REFIT quadrant tiles x dp, pdp, pdp{P}, cap and their 4 most often near-best split configurations), so it is a lower bound on the regret against every configuration.

| set | shapes | median | 90th | worst | geomean | > sigma |
|---|---|---|---|---|---|---|
| REFIT rule | 617 | 1.004 | 1.183 | 1.644 | 1.047 | 239 |
| v26 deployed (cap at 256x256x64) | 617 | 1.004 | 1.315 | 2.599 | 1.093 | 247 |

By group (rule, then v26):

| set | shapes | median | 90th | worst | geomean | > sigma |
|---|---|---|---|---|---|---|
| 256x256 waves 0: rule | 314 | 1.039 | 1.239 | 1.644 | 1.077 | 195 |
| 256x256 waves 0: v26 | 314 | 1.112 | 1.516 | 2.599 | 1.186 | 222 |
| 256x256 waves 1-3: rule | 148 | 1.002 | 1.183 | 1.436 | 1.030 | 35 |
| 256x256 waves 1-3: v26 | 148 | 1.000 | 1.017 | 1.101 | 1.006 | 19 |
| 256x256 waves 4+: rule | 155 | 1.000 | 1.008 | 1.267 | 1.006 | 9 |
| 256x256 waves 4+: v26 | 155 | 1.000 | 1.007 | 1.030 | 1.002 | 6 |
| inside REFIT's ranges: rule | 425 | 1.010 | 1.195 | 1.472 | 1.055 | 205 |
| inside REFIT's ranges: v26 | 425 | 1.031 | 1.418 | 2.599 | 1.135 | 231 |
| outside (M > 8192, N > 16384 or K > 32768): rule | 192 | 1.000 | 1.151 | 1.644 | 1.031 | 34 |
| outside (M > 8192, N > 16384 or K > 32768): v26 | 192 | 1.000 | 1.010 | 1.160 | 1.005 | 16 |

Rule faster than v26 by more than sigma on 145 shapes, slower on 56.

The 10 largest rule regrets (shape, pick, best timed):

| regret | shape | pick | best |
|---|---|---|---|
| 1.644 | 16384x512x4096 | cap @ 64x128x128 | cap @ 128x256x64 |
| 1.571 | 14336x512x7168 | sp @ 64x128x128 | cap @ 256x256x64 |
| 1.496 | 14336x512x4096 | sp @ 64x128x128 | pdp @ 128x256x64 |
| 1.472 | 7936x512x4096 | sp @ 64x128x128 | cap @ 128x128x128 |
| 1.441 | 10240x512x7168 | sp @ 64x128x128 | cap @ 256x256x64 |
| 1.436 | 32768x512x7168 | cap @ 128x128x128 | dp @ 256x256x64 |
| 1.433 | 7424x512x4096 | sp @ 64x128x128 | dp @ 128x128x128 |
| 1.433 | 512x4096x1536 | dp @ 128x128x64 | cap @ 128x64x128 |
| 1.422 | 1024x6144x2048 | cap @ 256x256x64 | dp @ 256x128x64 |
| 1.408 | 4096x4096x1536 | pdp512 @ 128x128x64 | cap @ 256x256x64 |

No held-out shape is memory-bound (M >= 512): the memory-bound rule is only tested by leaving shapes and K bins out of REFIT (`results/refit_fit.md`).
