# Held-out AITER shapes: results/heldout_r1.csv, results/heldout_r2.csv, rule results/rule.json

597 shapes. Time over the per-shape best, geomean / median / 90th percentile / worst.

| set | default | pick + dp | pick + best | pick + tuned (pruned space) | best tile, dp |
|---|---|---|---|---|---|
| all (597) | 1.0268 / 1.000 / 1.119 / 1.730 | 1.0368 / 1.000 / 1.147 / 1.730 | 1.0258 / 1.000 / 1.119 / 1.730 | 1.0258 / 1.000 / 1.119 / 1.730 | 1.0327 / 1.000 / 1.139 / 1.679 |
| below one wave (215) | 1.0405 / 1.000 / 1.139 / 1.730 | 1.0405 / 1.000 / 1.139 / 1.730 | 1.0405 / 1.000 / 1.139 / 1.730 | 1.0405 / 1.000 / 1.139 / 1.730 | 1.0356 / 1.000 / 1.133 / 1.679 |
| one wave or more (382) | 1.0191 / 1.000 / 1.098 / 1.259 | 1.0347 / 1.000 / 1.147 / 1.556 | 1.0176 / 1.000 / 1.098 / 1.259 | 1.0176 / 1.000 / 1.098 / 1.259 | 1.0311 / 1.000 / 1.139 / 1.452 |
| v31 within 10% of AITER's best library (450) | 1.0279 / 1.000 / 1.124 / 1.730 | 1.0417 / 1.000 / 1.158 / 1.730 | 1.0271 / 1.000 / 1.124 / 1.730 | 1.0271 / 1.000 / 1.124 / 1.730 | 1.0377 / 1.000 / 1.146 / 1.679 |

AITER's best library / v31's best (TFLOPS ratio; above 1: the library is faster): median 1.037, 90th percentile 1.190; 147 of 597 shapes more than 10% faster in the library.

Default slower than the picked tile's dp by more than 1% on 2 shapes.

## Worst 15 defaults

| shape | regime | tile | rule | default / best | pick + dp / best |
|---|---|---|---|---|---|
| 512x2304x16384 | below | 128x64x128 | dp | 1.730 | 1.730 |
| 1024x1280x8192 | below | 128x64x128 | dp | 1.533 | 1.533 |
| 512x2560x8192 | below | 128x64x128 | dp | 1.435 | 1.435 |
| 1280x1024x7168 | below | 128x64x128 | dp | 1.421 | 1.421 |
| 512x2304x6144 | below | 128x64x128 | dp | 1.316 | 1.316 |
| 512x2560x6144 | below | 128x64x128 | dp | 1.308 | 1.308 |
| 2304x512x4096 | below | 128x64x128 | dp | 1.286 | 1.286 |
| 1280x1024x4096 | below | 128x64x128 | dp | 1.277 | 1.277 |
| 4096x5120x25600 | after | 256x128x64 | ow2 | 1.259 | 1.431 |
| 2560x512x4096 | below | 128x64x128 | dp | 1.253 | 1.253 |
| 2048x2304x16384 | below | 256x128x64 | dp | 1.250 | 1.250 |
| 8192x2304x16384 | after | 256x128x64 | ow3 | 1.249 | 1.556 |
| 512x8448x7168 | below | 256x128x64 | dp | 1.242 | 1.242 |
| 1024x4608x8192 | below | 256x128x64 | dp | 1.222 | 1.222 |
| 512x9216x16384 | below | 256x128x64 | dp | 1.217 | 1.217 |
