# The stream-K host rules on AITER's shapes (held-out test set)

How well the host rules do on shapes they were not fitted on, and how much is lost by not tuning each
shape. Nothing here is fitted: the rules are v24's `capped` (as deployed) and the rule fitted on sweep F
([experiments/streamk_split_count](../streamk_split_count/README.md), section 5), frozen in
[rules_frozen.json](../streamk_split_count/results/rules_frozen.json) before any of these timings.

```bash
# AITER's stack (Triton v2.2 + deepbind shim, LLIR scheduler, no amdgcnas hook), HIP device numbering
python experiments/streamk_aiter_test/test.py cases                                    # 440 shapes, 4,738 cases
HIP_VISIBLE_DEVICES=1 python experiments/streamk_aiter_test/test.py check --jobs 3     # results/check.md
experiments/streamk_split_count/run_f.sh start aiter                                    # timing on device 0, then the report
python experiments/streamk_aiter_test/test.py report                                   # results/test.md
```

**Shapes.** AITER's 630 bf16 shapes ([supported_status_scoped.csv](../aiter_lixun_shapes/supported_status_scoped.csv))
with a partial last wave of 256x256 tiles: 440 distinct shapes, 314 with less than one wave (every tile is
leftover) and 126 with one full wave or more. The other 190 are whole waves.

**Per shape**, every single-rounding choice the rules can make, timed with rocprof (1000 launches over a
rotating 512 MB, median kernel time) with contiguous B and C, as AITER passes them: data-parallel (v9 at one wave
or less, which has less overhead when no program takes a second tile; v13 above), v24's one-tile reduce-scatter and
owner at every n (S n <= 256, n <= K / 128) and the default split, v24 as deployed, and the frozen rule's choice.
The fastest is the per-shape tuned choice. Every case first passed a correctness check against torch with B in
AITER's (N, K) layout ([results/check.md](results/check.md)). Rows record the tile, the program count and the
layout, so later sweeps at other tiles share the format.

## Result ([results/test.md](results/test.md))

Kernel time / the per-shape tuned choice, median / 90th percentile / worst:

| shapes | count | data-parallel | v24 capped (deployed) | fitted rule (frozen) |
|---|---|---|---|---|
| less than one wave | 314 | 1.464 / 3.124 / 4.997 | 1.029 / 1.161 / 1.264 | 1.044 / 1.154 / 1.435 |
| one full wave or more | 126 | 1.041 / 1.214 / 1.417 | 1.003 / 1.012 / 1.040 | 1.003 / 1.025 / 1.076 |

- **With a full wave, the regime sweep F covers, per-shape tuning gains little**: `capped` is within 1.2% at the
  90th percentile and 4% at worst. The fitted rule is level at the median and slightly worse in the tail, as its
  cross-validation predicted.
- **With less than one wave the rules leave up to 16% (90th percentile) to 26% (worst) on the table.** Neither
  rule has seen this regime: every tail in sweep F follows a full wave. The fastest choice there is reduce-scatter
  in 135 shapes, the owner ending in 77, `capped`'s own choice in 65 and data-parallel in 37.
- **Most of the loss is the ending switch.** `capped` picks the owner ending from S = 33 (fewer than 8 programs
  per tile possible), the fitted rule from S = 43 (fewer than 6), while up to S = 64 reduce-scatter at n = 4-6
  is fastest: `capped` loses a median 9.1% and 18.8% at the
  90th percentile for S < 64, against 0.1% and 1.5% for S >= 128. With no full wave to overlap, the flat
  reduce-scatter ending pays from fewer programs per tile than after one.

**Next.** A sweep without a full wave (`GEMM_TILE_LIMIT = S` on sweep C's matrix), fitted there, then this test
again. Changing the rules to fit these shapes would spend the test set.
