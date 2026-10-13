# The stream-K host rules on AITER's shapes (held-out test set)

How well the host rules do on shapes they were not fitted on, and how much is lost by not tuning each
shape. Nothing here is fitted: the rules are v24's `capped` (as deployed), the rule fitted on sweep F
([experiments/streamk_split_count](../streamk_split_count/README.md), section 5), frozen in
[rules_frozen_v1.json](../streamk_split_count/results/rules_frozen_v1.json) before any of these timings, and the
two-regime rule fitted on sweeps F and G (section 6), frozen in
[rules_frozen.json](../streamk_split_count/results/rules_frozen.json) before the re-score's timings (last section).

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

## Re-score with the two-regime rule (version 2)

Sweep G ([experiments/streamk_split_count](../streamk_split_count/README.md), section 6) measured the regime without
a full wave on its own matrix, and version 2 of
[rules_frozen.json](../streamk_split_count/results/rules_frozen.json) was frozen from it before any of the cases below
were timed, in two variants that differ only below one wave: `v2` (version 1's rule family, constants fitted on
sweep G) and `v2-sqrt` (the same with the reduce-scatter n also at most Q / sqrt(S), the trend sweep G shows).
Both keep version 1's constants for one wave or more, and add spread for S > 128 from k >= 512 in both regimes.

```bash
HIP_VISIBLE_DEVICES=1 python experiments/streamk_aiter_test/test.py check --jobs 3   # the untimed cases: results/check_v2.md
experiments/streamk_split_count/run_f.sh start aiter                                 # times them, rewrites results/test.md
```

New cases (161, on v25, which times as v24 on these aligned shapes): v25 spread for all 91 shapes with S > 128 (the
first run had it only where `capped` picked it) and the 70 reduce-scatter n of the version 2 rules that no timed case
covered. All passed against torch before timing. Version 1 is read from
[rules_frozen_v1.json](../streamk_split_count/results/rules_frozen_v1.json). Nothing here fed back into a constant.

| shapes | count | capped (deployed) | v1 | v2 | v2-sqrt |
|---|---|---|---|---|---|
| less than one wave | 314 | 1.035 / 1.165 / 1.277 | 1.047 / 1.170 / 1.435 | **1.013 / 1.102 / 1.221** | 1.019 / 1.117 / 1.221 |
| of which S = 32-63 | 75 | 1.123 / 1.198 | 1.106 / 1.246 | **1.004 / 1.085** | 1.043 / 1.139 |
| of which S > 128 | 60 | 1.003 / 1.089 / 1.184 | 1.066 / 1.204 / 1.321 | 1.002 / 1.099 / 1.181 | 1.002 / 1.099 / 1.181 |
| one full wave or more | 126 | 1.003 / 1.016 / 1.050 | 1.003 / 1.026 / 1.082 | 1.002 / 1.022 / 1.076 | 1.002 / 1.022 / 1.076 |
| all | 440 | 1.011 / 1.138 / 1.277 | 1.019 / 1.151 / 1.435 | 1.005 / 1.088 / 1.221 | 1.008 / 1.102 / 1.221 |

(Median / 90th percentile / worst; the S = 32-63 row median / 90th percentile. Adding spread to every S > 128 shape
moved `capped`'s and version 1's numbers from the first run: spread is the fastest choice on 24 of the 91, 19 of them
below one wave.)

- **Below one wave, version 2 cuts the loss at the 90th percentile from 16.5% (`capped`) to 10.2%**, and the median
  from 3.5% to 1.3%. Shape by shape, `v2` is more than 2% faster than `capped` on 111 of the 314 and more than 2%
  slower on 36. With a full wave nothing changes (version 1's constants; `capped` is still level).
- **The gain is where sweep G pointed: S = 32-63**, the ending switch. `v2` runs reduce-scatter up to S = 64 with
  n = min(192 // S, k / 4), and loses a median 0.4% there against 12.3% for `capped`.
- **The sqrt(S) cap did not carry over.** It was ahead on sweep G (1.094 against 1.103 at the 90th percentile), but on
  AITER it is behind `v2` (1.117 against 1.102). At S = 48 it gives n = round(32 / sqrt(48)) = 5, which is the default
  split (all 256 programs), and that is 3-18% slower than n = 4 on these contiguous shapes; at S = 40 its n = 5 beats
  `v2`'s 4 at large K. A one-program rounding step at a split boundary is beyond what one sweep fixes.
- **S > 128 below one wave is not solved**: up to 18% is still lost there by every rule (90th percentile 9-10%).
  Spread pays from S = 160-192 at smaller k than `capped`'s 512 but not at S = 240-255, and no single threshold on k
  separates them, so the fitted threshold stayed at 512. A rule there needs S as well as k.

**Proposed host-rule change for v25 (not applied):** below one wave (tiles < programs), `capped` with `v2`'s constants
(T = 14, E = 4, m = 4, P = 192, c0 = 1.5): data-parallel if (256 - S) k / 256 < 14, reduce-scatter while 256 // S >= 4
with n = min(192 // S, k / 4), else owner with n = round(sqrt(k / 1.5)); one wave or more unchanged. Applying it means a
new kernel version (or a v25 entry) and re-checking its fallback paths, as a separate step.

## Re-score with version 3: spread by idle k-steps

Version 3 ([rules_frozen.json](../streamk_split_count/results/rules_frozen.json), frozen before these timings) is
`v2` with the S > 128 test replaced by "spread if (256 - S) k >= 10,493" in both regimes, fitted on sweep H only
([experiments/streamk_split_count](../streamk_split_count/README.md), section 7), its spread on v26. New cases: v26
spread on all 91 shapes with S > 128 (all passed against torch first, [results/check_v3.md](results/check_v3.md)).

| shapes with S > 128 | count | capped | v2 | v3 |
|---|---|---|---|---|
| less than one wave | 60 | 1.004 / 1.129 / 1.209 | 1.004 / 1.140 / 1.206 | **1.000 / 1.009 / 1.054** |
| one full wave or more | 31 | 1.003 / 1.016 / 1.056 | 1.000 / 1.013 / 1.061 | **1.000 / 1.005 / 1.013** |
| all 440 shapes | 440 | 1.013 / 1.139 / 1.277 | 1.006 / 1.088 / 1.221 | **1.004 / 1.080 / 1.221** |

(The S > 128 rows moved slightly from the version 2 section: v26's spread is now among each shape's choices.)

- **The S > 128 loss is gone**: 90th percentile 0.9% below one wave (12.9% for `capped`), 0.5% with a full wave. Spread
  where the data-parallel last wave leaves at least about 10,000 idle k-steps, data-parallel otherwise.
- **Below one wave overall**: `capped` 1.039 / 1.165 / 1.277, v3 1.008 / 1.098 / 1.221; v3 is more than 2% faster than
  `capped` on 126 of the 314 shapes and more than 2% slower on 35. What remains is at S <= 128 (as `v2`).

**Applied to v26's `capped_policy`** (v26 WORKLOG entry 5): `v2`'s constants below one wave, and for S > 128 in both
regimes spread if (256 - S) k >= 10,493 instead of k >= 512; one wave or more at S <= 128 unchanged. Timed as
deployed (`v26 deployed`): less than one wave 1.012 / 1.105 / 1.231 (S > 128: 1.002 / 1.014 / 1.061), one full wave
or more 1.003 / 1.015 / 1.039. All AITER shapes are aligned; on ragged M / N the spread build spills more, and spread
lost to data-parallel on 3000x3000x16384 (4%, 17% with bias). Ragged shapes may need their own rule; first
results are in experiments/streamk_split_count (section 8) and v26's WORKLOG entry 5.
