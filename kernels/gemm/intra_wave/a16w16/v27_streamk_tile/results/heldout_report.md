# heldout timing (heldout)

Manifest `results/heldout_manifest.json`, created 2026-10-15 05:25:08 +0000: 617 shapes, 36355 points on GPUs 0, 1, 2, 3; Triton 3.9.0 872c90a6a0574fc30071885186d64c725c1c3ea7; repository 4a4959d27b47a80df4a8ee099926160f8a9777de. Frozen files (sha256, checked before every chunk):

- `v27_streamk_tile/matmul_kernel.py` b96067e239445a19dfaad8392f173699dadc8be631d3ed658fc401626fcca788
- `v27_streamk_tile/sweep.py` ea0e6091c8194f228faeb29155ff1fd65be39e54a33e354fce8bcff0ad85e285
- `v27_streamk_tile/refit.py` 6d8cbca8c6bcfcf9bc71a3ca2499e4c8a67115b33d966699e3e45dde3874acce
- `v9_tile/matmul_kernel.py` e0255f25efa32281d6976dd4d5c16e6f1b8e85bea63d2baccb6e55cb7933c691

Rounds timed: 1 (36355 points), 2 (36355 points), 3 (11319 points), 2026-10-15 05:29 to 09:19.
Rows not correct or with an error: 0.

Points with no correct timing: 0.

## 1. Noise

|rel_1 / rel_2 - 1| over 36355 points (median / 90th / 99th percentile / max): 0.46% / 1.31% / 3.34% / 399.35%. The noise threshold sigma (90th percentile, from rounds 1 and 2) is **1.24%**.
FULL's repeats, for comparison: median 0.49%, 90th percentile 1.85%, max 10% (single timings on one GPU against another, GPU offsets included).

| group | points | median / 90th / 99th / max |
|---|---|---|
| compute-bound, quadrant | 36355 | 0.46% / 1.31% / 3.34% / 399.35% |
| iters 1000 | 25874 | 0.45% / 1.26% / 2.63% / 9.28% |
| iters 200 | 4245 | 0.53% / 1.75% / 7.27% / 61.21% |
| iters 300 | 853 | 0.46% / 1.59% / 4.86% / 399.35% |
| iters 400 | 1527 | 0.43% / 1.22% / 3.14% / 23.69% |
| iters 500 | 842 | 0.38% / 1.14% / 2.07% / 5.93% |
| iters 600 | 190 | 0.57% / 1.69% / 9.80% / 12.55% |
| iters 700 | 1109 | 0.46% / 1.33% / 3.75% / 11.85% |
| iters 800 | 969 | 0.50% / 1.31% / 2.20% / 6.31% |
| iters 900 | 746 | 0.46% / 1.38% / 3.16% / 6.43% |

Per-GPU offset (a shape block's geometric-mean time on the GPU over its mean across the shape's rounds; within-block ratios cancel it):

| GPU | blocks | median | 10th | 90th |
|---|---|---|---|---|
| 0 | 308 | -0.03% | -0.40% | +0.21% |
| 1 | 308 | -0.14% | -0.30% | +0.01% |
| 2 | 311 | +0.06% | -0.20% | +0.27% |
| 3 | 309 | +0.17% | -0.06% | +0.42% |

Position in the chunk (each process's lead-in point is dropped): a timing over its point's merged value, median and 90th percentile of the absolute difference:

| position | timings | median | 90th percentile of abs |
|---|---|---|---|
| 0 | 1407 | +0.00% | 0.67% |
| 1-4 | 5625 | +0.00% | 0.67% |
| 5-19 | 21055 | +0.00% | 0.66% |
| 20+ | 55942 | +0.00% | 0.67% |

## 2. Round 3

11319 points (31.1%) on 616 shapes: (shape, tile) pairs whose best and runner-up configurations are within 2 sigma: 3963; shapes whose best and runner-up tiles are: 152; points whose rounds 1 and 2 differ by more than sigma: 3635; plus each shape's best points at its 3 best tiles as anchors.

Round 3 against the mean of rounds 1 and 2 (same normalisation): 0.44% / 1.42% / 3.90% / 65.71% (median / 90th / 99th / max); within sigma for 87%.
Decisions round 3 changed: the best tile of 15 of 617 shapes, the best configuration of 320 (shape, tile) pairs.

## 3. How close the winners are

After the merge: shapes whose runner-up tile, and (shape, tile) pairs whose runner-up configuration, is within sigma (1.24%) and 2 sigma of the best.

| regime | shapes | runner-up tile within sigma | within 2 sigma | (shape, tile) pairs | runner-up configuration within sigma | within 2 sigma |
|---|---|---|---|---|---|---|
| compute-bound | 617 | 92 | 152 | 8021 | 2444 | 3894 |

v26 as deployed (cap at 256x256x64) over each shape's best: median 1.004, 90th percentile 1.315, worst 2.599 (617 shapes).

## 4. Coverage per tile class

(shape, tile) pairs per class in the tiles' own coordinates: waves of the tile's programs, leftover tiles S over programs P, and k-steps.

| class | pairs | w 0 | w 1-3 | w 4+ | S 0 | S <1/64 | S 1/64-1/16 | S 1/16-1/4 | S 1/4-1 | k <=8 | k 9-32 | k 33-128 | k >128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| compute-bound, quadrant | 8021 | 1550 | 2118 | 4353 | 4597 | 0 | 26 | 493 | 2905 | 306 | 1870 | 4886 | 959 |
