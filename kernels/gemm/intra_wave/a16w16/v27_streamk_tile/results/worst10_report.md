# worst10 timing (worst10)

Manifest `results/worst10_manifest.json`, created 2026-10-17 05:56:16 +0000: 60 shapes, 15475 points on GPUs 3; Triton 3.9.0 872c90a6a0574fc30071885186d64c725c1c3ea7; repository None. Frozen files (sha256, checked before every chunk):

- `v27_streamk_tile/matmul_kernel.py` 1175b73525d3cc3f00e47159868c8eb109337e2b3f8df9ba0ff89c6d717103a7
- `v27_streamk_tile/sweep.py` 9cdc7918435ba13d3f3d644fb98dac5c04e331ff122806ad9fd11258025bc193
- `v27_streamk_tile/refit.py` dd1a7f7511f58d1ef63fd833c60b5c7d78b1fee6809bc07e3d0dec781ca12371
- `v9_tile/matmul_kernel.py` d757b2ea8326314bd8de9f62f0f48f62c22dde4bb4c05d987188e939924af30a

Rounds timed: 1 (15475 points), 2 (15475 points), 3 (2378 points), 2026-10-17 06:01 to 08:25.
Rows not correct or with an error: 0.

Points with no correct timing: 0.

## 1. Noise

|rel_1 / rel_2 - 1| over 15475 points (median / 90th / 99th percentile / max): 0.36% / 1.00% / 2.23% / 10.94%. The noise threshold sigma (90th percentile, from rounds 1 and 2) is **0.95%**.
FULL's repeats, for comparison: median 0.49%, 90th percentile 1.85%, max 10% (single timings on one GPU against another, GPU offsets included).

| group | points | median / 90th / 99th / max |
|---|---|---|
| compute-bound, quadrant | 15475 | 0.36% / 1.00% / 2.23% / 10.94% |
| iters 1000 | 15475 | 0.36% / 1.00% / 2.23% / 10.94% |

Per-GPU offset (a shape block's geometric-mean time on the GPU over its mean across the shape's rounds; within-block ratios cancel it):

| GPU | blocks | median | 10th | 90th |
|---|---|---|---|---|
| 3 | 120 | +0.00% | -0.05% | +0.04% |

Position in the chunk (each process's lead-in point is dropped): a timing over its point's merged value, median and 90th percentile of the absolute difference:

| position | timings | median | 90th percentile of abs |
|---|---|---|---|
| 0 | 557 | +0.00% | 0.44% |
| 1-4 | 2224 | +0.00% | 0.47% |
| 5-19 | 8340 | +0.00% | 0.47% |
| 20+ | 22207 | +0.00% | 0.47% |

## 2. Round 3

2378 points (15.4%) on 60 shapes: (shape, tile) pairs whose best and runner-up configurations are within 2 sigma: 464; shapes whose best and runner-up tiles are: 27; points whose rounds 1 and 2 differ by more than sigma: 1547; plus each shape's best points at its 3 best tiles as anchors.

Round 3 against the mean of rounds 1 and 2 (same normalisation): 0.44% / 1.32% / 2.74% / 6.08% (median / 90th / 99th / max); within sigma for 80%.
Decisions round 3 changed: the best tile of 2 of 60 shapes, the best configuration of 72 (shape, tile) pairs.

## 3. How close the winners are

After the merge: shapes whose runner-up tile, and (shape, tile) pairs whose runner-up configuration, is within sigma (0.95%) and 2 sigma of the best.

| regime | shapes | runner-up tile within sigma | within 2 sigma | (shape, tile) pairs | runner-up configuration within sigma | within 2 sigma |
|---|---|---|---|---|---|---|
| compute-bound | 60 | 17 | 28 | 780 | 349 | 464 |

v26 as deployed (cap at 256x256x64) over each shape's best: median 1.247, 90th percentile 1.464, worst 2.246 (60 shapes).

## 4. Coverage per tile class

(shape, tile) pairs per class in the tiles' own coordinates: waves of the tile's programs, leftover tiles S over programs P, and k-steps.

| class | pairs | w 0 | w 1-3 | w 4+ | S 0 | S <1/64 | S 1/64-1/16 | S 1/16-1/4 | S 1/4-1 | k <=8 | k 9-32 | k 33-128 | k >128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| compute-bound, quadrant | 780 | 429 | 275 | 76 | 104 | 0 | 9 | 159 | 508 | 4 | 146 | 568 | 62 |
