# refit timing (refit)

Manifest `results/refit_manifest.json`, created 2026-10-14 17:12:43 +0000: 350 shapes, 89033 points on GPUs 0, 0, 0, 0; Triton 3.9.0 872c90a6a0574fc30071885186d64c725c1c3ea7; repository 4a4959d27b47a80df4a8ee099926160f8a9777de. Frozen files (sha256, checked before every chunk):

- `v27_streamk_tile/matmul_kernel.py` b96067e239445a19dfaad8392f173699dadc8be631d3ed658fc401626fcca788
- `v27_streamk_tile/sweep.py` ea0e6091c8194f228faeb29155ff1fd65be39e54a33e354fce8bcff0ad85e285
- `v27_streamk_tile/refit.py` d4d376cd626cc140ba478c125ed596ea59ea93162605142f439d84bbbf959817
- `v9_tile/matmul_kernel.py` e0255f25efa32281d6976dd4d5c16e6f1b8e85bea63d2baccb6e55cb7933c691

Rounds timed: 1 (89033 points), 2 (89033 points), 3 (16080 points), 2026-10-14 17:12 to 05:22.
Rows not correct or with an error: 0.

Points with no correct timing: 0.

## 1. Noise

|rel_1 / rel_2 - 1| over 89033 points (median / 90th / 99th percentile / max): 0.62% / 2.09% / 6.20% / 51.34%. The noise threshold sigma (90th percentile, from rounds 1 and 2) is **1.89%**.
FULL's repeats, for comparison: median 0.49%, 90th percentile 1.85%, max 10% (single timings on one GPU against another, GPU offsets included).

| group | points | median / 90th / 99th / max |
|---|---|---|
| compute-bound, quadrant | 28203 | 0.58% / 1.87% / 3.81% / 24.20% |
| iters 1000 | 78825 | 0.62% / 2.12% / 6.47% / 51.34% |
| iters 200 | 3898 | 0.56% / 1.65% / 3.04% / 7.73% |
| iters 300 | 1772 | 0.49% / 1.50% / 4.27% / 12.94% |
| iters 400 | 1709 | 0.62% / 2.12% / 3.88% / 24.20% |
| iters 500 | 455 | 0.84% / 2.65% / 6.71% / 14.51% |
| iters 600 | 1191 | 0.60% / 2.65% / 4.39% / 6.25% |
| iters 700 | 496 | 0.43% / 1.28% / 2.07% / 2.40% |
| iters 900 | 687 | 1.11% / 2.53% / 4.61% / 12.74% |
| memory-bound, quadrant | 33227 | 0.60% / 2.04% / 6.43% / 40.26% |
| memory-bound, small P=1024 | 10558 | 0.82% / 2.97% / 7.69% / 28.82% |
| memory-bound, small P=256 | 7637 | 0.59% / 2.00% / 9.78% / 51.34% |
| memory-bound, small P=512 | 9408 | 0.66% / 2.22% / 8.33% / 49.76% |

Per-GPU offset (a shape block's geometric-mean time on the GPU over its mean across the shape's rounds; within-block ratios cancel it):

| GPU | blocks | median | 10th | 90th |
|---|---|---|---|---|
| 0 | 437 | +0.01% | -0.19% | +0.22% |
| 1 | 88 | -0.10% | -0.37% | +0.04% |
| 2 | 88 | -0.03% | -0.20% | +0.08% |
| 3 | 87 | +0.12% | -0.07% | +0.55% |

Position in the chunk (each process's lead-in point is dropped): a timing over its point's merged value, median and 90th percentile of the absolute difference:

| position | timings | median | 90th percentile of abs |
|---|---|---|---|
| 0 | 3241 | +0.00% | 0.95% |
| 1-4 | 12964 | +0.00% | 0.97% |
| 5-19 | 48597 | +0.00% | 0.96% |
| 20+ | 129344 | +0.00% | 0.96% |

## 2. Round 3

16080 points (18.1%) on 350 shapes: (shape, tile) pairs whose best and runner-up configurations are within 2 sigma: 3798; shapes whose best and runner-up tiles are: 159; points whose rounds 1 and 2 differ by more than sigma: 8903; plus each shape's best points at its 3 best tiles as anchors.

Round 3 against the mean of rounds 1 and 2 (same normalisation): 0.81% / 2.89% / 8.85% / 35.52% (median / 90th / 99th / max); within sigma for 80%.
Decisions round 3 changed: the best tile of 26 of 350 shapes, the best configuration of 521 (shape, tile) pairs.

## 3. How close the winners are

After the merge: shapes whose runner-up tile, and (shape, tile) pairs whose runner-up configuration, is within sigma (1.89%) and 2 sigma of the best.

| regime | shapes | runner-up tile within sigma | within 2 sigma | (shape, tile) pairs | runner-up configuration within sigma | within 2 sigma |
|---|---|---|---|---|---|---|
| compute-bound | 198 | 34 | 54 | 2574 | 1304 | 1690 |
| memory-bound | 152 | 79 | 109 | 3952 | 1342 | 2101 |

v26 as deployed (cap at 256x256x64) over each shape's best: median 1.238, 90th percentile 3.419, worst 4.646 (350 shapes).

## 4. Coverage per tile class

(shape, tile) pairs per class in the tiles' own coordinates: waves of the tile's programs, leftover tiles S over programs P, and k-steps.

| class | pairs | w 0 | w 1-3 | w 4+ | S 0 | S <1/64 | S 1/64-1/16 | S 1/16-1/4 | S 1/4-1 | k <=8 | k 9-32 | k 33-128 | k >128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| compute-bound, quadrant | 2574 | 538 | 354 | 1682 | 124 | 0 | 128 | 740 | 1582 | 441 | 675 | 713 | 745 |
| memory-bound, quadrant | 1976 | 1823 | 147 | 6 | 2 | 167 | 546 | 640 | 621 | 108 | 753 | 818 | 297 |
| memory-bound, small P=1024 | 912 | 642 | 149 | 121 | 78 | 36 | 158 | 249 | 391 | 222 | 364 | 326 | 0 |
| memory-bound, small P=256 | 456 | 425 | 28 | 3 | 1 | 76 | 123 | 135 | 121 | 54 | 177 | 192 | 33 |
| memory-bound, small P=512 | 608 | 546 | 56 | 6 | 2 | 71 | 154 | 185 | 196 | 81 | 237 | 257 | 33 |
