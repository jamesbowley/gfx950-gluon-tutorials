## Update 2026-09-29: BF16 GEMM comparison across hipBLASLt versions

For the stream-K work I need representative macro-tiles. Compute-bound shapes where hipBLASLt wins with MT256x256 are the reference set. v9 uses a 256x256 tile because it was tuned on square shapes, and hipBLASLt's heuristic converging on the same macro-tile is strong evidence that 256x256 is optimal there.

Reproducing the issue's timings showed a large delta that tracks the hipBLASLt version, so I re-ran the full 630-shape sweep. I assume the CSV results used torch 2.10.0+rocm7.2.4.lw.git3d3aa833, because that's what the container I found uses, and it's the version I'm running. The nightly column is TheRock torch 2.15.0a0+rocm10.2.0a20260929. Both columns were measured on the same GPU with the issue's method.

### Macro-tile selection changed between hipBLASLt 1.2.2 and 1.5.0

- hipBLASLt 1.2.2 selects MT256x256x64 on 225 of the 630 shapes.
- hipBLASLt 1.5.0 moves 98 of those to non-power-of-2 macro-tiles (256x320, 224x384, 288x288, 160x256, ...). These are the subtile solutions the gfx950 BF16 TN Origami library added in 1.5.0.
- The effect depends on output-tile occupancy.

Examples:

- **Regression:** 32768×8192×1024 (4096 output tiles, 16 waves): MT256x256 SK3 → MT256x320 SK5, 1268 → 1098 TFLOPS (−13%). AITER dispatches `torch` for this shape, so it inherits the regression, and Gluon (1234) becomes the winner.
- **Improvement:** 1024×8192×8192 (128 output tiles, 0.5 wave at 256x256): MT256x256 → MT160x256, 1029 → 1276 TFLOPS (+24%). The smaller macro-tile raises the tile count and CU occupancy, and it now beats Gluon (1080) by 18%.

| Gluon + llirSched vs AITER's pick (630 shapes) | Issue | torch 2.10.0+rocm7.2.4 | Nightly 2.15.0a0+rocm10.2.0 |
|---|---|---|---|
| hipBLASLt version | 1.2.x | 1.2.2 | 1.5.0 |
| hipBLASLt (`F.linear`) selects MT256x256 | – | 225 | 131 |
| Out of scope (smaller tile / split-K) | 38 (23 / 15) | 38 (23 / 15) | 38 (23 / 15) |
| Compute-bound | 592 | 592 | 592 |
| Gluon wins | 293 | 314 | 310 |
| Lost to another of the eight tiles | 144: median 63%, p90 178% | 142: median 61%, p90 173% | 138: median 59%, p90 156% |
| Lost to a non-power-of-2 tile | 117: median 35%, p90 143% | 117: median 31%, p90 140% | 133: median 32%, p90 134% |
| Lost to another 256x256 kernel | 38: median 0.6%, up to 174% | 19: median 1.1%, up to 179% | 11: median 1.1%, up to 71% |
| Gluon wins at ≥512 output tiles | 195 of 225, worst loss 3.8% | 206 of 225, worst loss 3.9% | 209 of 225, worst loss 13.7% |
| hipBLASLt wins | 150 (SK3) | 139 (SK3) | 144 (SK5) |

**What this shows:**

- **The torch 2.10.0+rocm7.2.4 rerun reproduces the issue.**
  - The rerun launches the identical hipBLASLt solution on all 150 of the issue's hipBLASLt wins.
  - 23 of the 31 winner flips are within 2%, which is measurement noise at parity.
  - Gluon v9 agrees between the two reruns to within about 1%, so the deltas in the nightly column are attributable to hipBLASLt.
- **The 1.5.0 non-power-of-2 macro-tiles regress once the GPU is saturated.**
  - At ≥256 output tiles (85 shapes): median −3.4%, and 46 regress by more than 3%.
  - Below 256 output tiles (13 shapes), 256x256 under-fills the 256 CUs; the smaller or non-square tiles improve occupancy and gain a median 24%.
- **Gluon's worst loss at ≥512 output tiles rises from 3.9% to 13.7%.** 4096×8704×4096 is now won by an MT272x256 kernel.
- **SK3 is renamed SK5 in 1.5.0.** With hybrid mode off, SK5 executes the SK3 path, so the stream-K scheduling itself is unchanged.
- **Stream-K reference macro-tiles should come from the 1.2.2 results.** They match the issue, and MT256x256 is still the heuristic's choice on the large compute-bound shapes.

**Attachments:**

- `supported_status_scoped_rocm724.csv` and `supported_status_scoped_nightly.csv`: the 630 shapes, in the same format as `supported_status_scoped.csv`.
- `hbl_all_rocm724.csv` and `hbl_all_nightly.csv`: hipBLASLt (`F.linear`) TFLOPS and solution name on all 630 shapes, independent of AITER's pick. The MT256x256 counts and both examples come from these.
