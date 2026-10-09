# Stream-K workspace placement (warm, stream-K split shapes, 5 fresh-buffer rounds)

| Variant | Layout | Kernel | Shapes | median spread | p90 spread | over 2% | over 5% | max |
|---|---|---|---:|---:|---:|---:|---:|---:|
| workspace fixed per process | contiguous | streamk_wip (split) | 306 | 0.63% | 1.62% | 19 | 1 | 6.0% |
| workspace fixed per process | contiguous | data-parallel only (no P) | 306 | 0.34% | 1.10% | 10 | 3 | 19.1% |
| workspace fixed per process | padded | streamk_wip (split) | 306 | 0.86% | 2.80% | 52 | 6 | 6.3% |
| workspace fixed per process | padded | data-parallel only (no P) | 306 | 0.47% | 1.42% | 18 | 0 | 4.5% |
| fresh workspace every round | contiguous | streamk_wip (split) | 306 | 0.64% | 1.63% | 19 | 2 | 10.3% |
| fresh workspace every round | contiguous | data-parallel only (no P) | 306 | 0.38% | 1.24% | 14 | 2 | 9.3% |
| fresh workspace every round | padded | streamk_wip (split) | 306 | 0.82% | 2.70% | 47 | 10 | 6.8% |
| fresh workspace every round | padded | data-parallel only (no P) | 306 | 0.55% | 1.32% | 14 | 2 | 7.2% |
| fresh workspace every round, slots padded 256 B | contiguous | streamk_wip (split) | 306 | 0.69% | 2.37% | 36 | 3 | 7.8% |
| fresh workspace every round, slots padded 256 B | contiguous | data-parallel only (no P) | 306 | 0.35% | 1.38% | 19 | 6 | 18.2% |
| fresh workspace every round, slots padded 256 B | padded | streamk_wip (split) | 306 | 0.86% | 2.53% | 46 | 8 | 7.3% |
| fresh workspace every round, slots padded 256 B | padded | data-parallel only (no P) | 306 | 0.52% | 1.54% | 18 | 2 | 6.2% |

Median time against ws_base, per shape (this variant TFLOPS / ws_base TFLOPS - 1):

| Variant | Layout | Kernel | median | p10 | p90 |
|---|---|---|---:|---:|---:|
| fresh workspace every round | contiguous | streamk_wip (split) | -0.04% | -0.4% | +0.4% |
| fresh workspace every round | contiguous | data-parallel only (no P) | -0.14% | -0.7% | +0.3% |
| fresh workspace every round | padded | streamk_wip (split) | -0.00% | -0.5% | +0.4% |
| fresh workspace every round | padded | data-parallel only (no P) | -0.02% | -0.5% | +0.4% |
| fresh workspace every round, slots padded 256 B | contiguous | streamk_wip (split) | +0.11% | -0.5% | +0.9% |
| fresh workspace every round, slots padded 256 B | contiguous | data-parallel only (no P) | -0.14% | -0.8% | +0.3% |
| fresh workspace every round, slots padded 256 B | padded | streamk_wip (split) | +0.13% | -0.5% | +0.8% |
| fresh workspace every round, slots padded 256 B | padded | data-parallel only (no P) | -0.03% | -0.5% | +0.4% |
