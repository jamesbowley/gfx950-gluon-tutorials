# Capped vs auto: which host rule should v20 use?

v20 had two host rules for its stream-K tail
([kernels/gemm/intra_wave/a16w16/v20_streamk_reduce_scatter](../../kernels/gemm/intra_wave/a16w16/v20_streamk_reduce_scatter)):

- `auto` (WORKLOG entry 2), the default: data-parallel, two-tile, or one-tile split across all 256 programs.
- `capped` (entry 5): no two-tile, and the one-tile split capped to n programs per leftover tile.

Entry 5 found `capped` within 1.5% of the best on 26 of 30 shapes, but it was scored on the shapes its
coefficients were fit to, and two-tile still won by 0.2-1.5% on three of them. This experiment tests
both rules, and rules that use two-tile in part of the space (hybrids), on shapes `capped` was not fit to.

**Outcome:** `capped` replaces `auto` as the default (v20 WORKLOG entry 7). There is one change to it:
a split of one program per tile now runs as data-parallel. No hybrid is worth adding.

```bash
# from the repo root, with the published v10+ stack (see the a16w16 WORKLOGs)
export PYTHONPATH=$PWD/scripts/triton_deepbind_shim:<triton_gfx950-tutorial-v2.2>/python
for i in 0 1 2; do   # one shard per GPU: main, split counts, then 3-process medians
  (for m in "main $i/3" "split $i/3" "repeat $i/3 3"; do
     HIP_VISIBLE_DEVICES=$((i+1)) python experiments/streamk_capped_vs_auto/sweep.py $m; done) &
done
cd experiments/aiter_lixun_shapes && HIP_VISIBLE_DEVICES=0 python run_v20_vs_aiter.py sweep --tag default --v20-only
cd ../.. && python experiments/streamk_capped_vs_auto/sweep.py score experiments/aiter_lixun_shapes/results/details_default.csv
```

## Method

**Held-out grid** (`sweep.py`, 32 shapes, results in `results/shard*/`).
- Leftover tiles S = 8, 24, 48 and 96, none of which is in entry 5's fit set.
- 1, 4, 8 and 16 full waves, at K = 6144; also K = 3072 at 1 and 8 waves, and K = 8192 at 8 and 16 waves.
- Where `256 * waves + S` only factors into a very skinny shape, the nearest S that factors well is used
  (26, 10 and 52).
- Configurations: v13, v19, and v20 `auto`, `capped`, `two_tile`, `rs`, `owner`, `spread` and `dp`.
  The v20 `rs` and `owner` endings were also timed at every split count n in {1, 2, 3, 4, 6, 8, 16}.
  This gives the best measured configuration of every shape (the oracle) and the best n.
- Timing is `streamk_version_tables`' method: rocprof kernel time, median of 1000 launches on a rotating
  512 MB of inputs (cold caches). Every main configuration is the median of 3 processes; the split
  counts are timed in one process.

**Real-model shapes** (`experiments/aiter_lixun_shapes`, 630 shapes).
- `run_v20_vs_aiter.py sweep --v20-only` times the v20 policies with `do_bench_cudagraph` (back to back),
  without the AITER worker. The ROCm 7.2.4 stack of the AITER comparison is not mounted in this
  container, so this run uses the container's stack (torch 2.13 on ROCm 7.14), as the held-out grid does.
- Columns: `auto`, `dp`, `capped`, and `two_tile` where a hybrid could pick it (a full wave and
  `256 // S >= 7`). Only 12 of the 630 shapes qualify for two-tile.
- `results/details_capped.csv` is the run before the n = 1 change; `details_default.csv` is after it.

**Scoring** (`sweep.py score`, [results/score.md](results/score.md)). A rule only chooses among measured
configurations, so its time on a shape is the time of its pick. The candidates:
- `auto` and `capped`;
- `hybrid(W, D)`: capped, but two-tile where there are at least W full waves, `256 // S >= 7`, and
  two-tile's lag `d = k-tiles * S / 256` is at most D;
- the best of capped and two-tile on each shape, which bounds every hybrid.

**Between launches.** `between_launches.py` timed v13, capped and two-tile on the 22 shapes where
two-tile is allowed and within 3% of capped ([results/between_launches_raw.md](results/between_launches_raw.md)).

## Results

Held-out grid, against the best measured configuration of each shape:

| rule | within 1.5% | worst | geomean |
|---|---:|---:|---:|
| auto | 24 of 32 | -11.7% | -1.56% |
| capped | 27 | -4.0% | -0.90% |
| hybrid(8, 12), the best hybrid | 28 | -4.0% | -0.82% |
| best of capped and two-tile | 30 | -3.8% | -0.52% |

Real-model shapes, against the best v20 column of each shape (after the n = 1 change):

| rule | within 1.5% | worst | geomean |
|---|---:|---:|---:|
| auto | 457 of 630 | -51.0% | -4.50% |
| capped | 609 | -10.6% | -0.15% |
| every hybrid | 609 | -10.6% | -0.15% |
| best of capped and two-tile | 613 | -10.6% | -0.14% |

What the numbers say:

1. **Capped beats auto on both sets.**
   - On the real-model set, capped is faster than auto on 292 shapes and within 1% on 429. The median
     gain is +0.8% on the 321 shapes with no full wave (p90 +27%) and 0.0% elsewhere.
   - Auto's worst losses are splits it should not make. On the 49 shapes with K < 2048 where auto
     picks one-tile, the split is 22% slower than data-parallel at the median. Capped's minimum-K rule
     runs data-parallel on 41 of them.
   - On the held-out grid, capped's median gain over auto is +2.0% at one full wave, +0.7% at 4 waves,
     and level at 8-16 waves.
2. **A hybrid buys almost nothing.**
   - On the real-model set, two-tile can only be picked on 12 shapes. Even the best possible choice
     between capped and two-tile adds 4 shapes and 0.01% geomean.
   - On the held-out grid, hybrid(8, 12) adds one shape and 0.08% geomean. In its two-tile region
     (10 shapes) two-tile averages +0.3% over capped (-0.9% to +1.6%), below the plan's 1% bar.
   - Two-tile's held-out wins are small (up to +1.7%), apart from 5120x3584x6144 (+4.1%, one wave,
     S = 24). It loses heavily elsewhere at one wave: -6% to -15% at S = 48-96, and -6% at S = 8,
     K = 3072.
3. **Between launches, two-tile wins only with an idle gap.**
   - Two-tile against capped on the 22 close calls:

     | between launches | two-tile ahead by > 1% | capped ahead by > 1% | median |
     |---|---:|---:|---:|
     | nothing (back to back) | 4 | 4 | -0.1% |
     | zero 256 MB (do_bench) | 1 | 8 | -0.6% |
     | read 256 MB (clean) | 2 | 5 | -0.2% |
     | zero 64 MB | 1 | 6 | -0.3% |
     | sleep 100 us (idle) | 12 | 0 | +1.8% |

   - With an idle GPU between launches, two-tile leads by 2-5% on the smaller shapes. With dirty caches,
     capped leads by up to 12%. Entry 2 chose two-tile from the clean-cache column; with the capped split,
     that column is now level.
4. **Capped's split count is close to the best n.** It is within 1% of the best n of the same ending on
   27 of 32 held-out shapes. The worst gap is 2.9% (5120x3584x3072: rs at n = 5 against n = 4), and part
   of each gap is that the split counts were timed in one process only.
5. **One change: n = 1 runs data-parallel.**
   - The owner formula gives n = 1 when S is large and K small, for example S = 128 at 56 k-tiles.
     One program per tile is the data-parallel last wave's work plus stream-K's restart and fixup
     machinery.
   - It lost to data-parallel on all 6 real-model shapes where capped picked it, by 0.8-5.6%. That
     agrees with the split-count experiment, where n = 1 never beat data-parallel.
   - `capped_policy` now returns data-parallel instead: 604 to 609 shapes within 1.5% on the
     real-model set.
   - On two of those shapes auto's n = 2 is better still (1024x8192x3584 by 7.2%, 8192x5120x3200 by
     2.0%). On the others data-parallel matches or beats n = 2.

Correctness:
- `check_streamk.py --policies capped` passes 36 configurations: 9 shapes covering every branch,
  including the two n = 1 shapes, both tile orders, and with and without bias
  ([results/check_streamk.md](results/check_streamk.md)).
- Every held-out run and all 630 real-model shapes matched the reference, with flags and counts re-armed.
- The measured default (STREAMK_POLICY unset) matches `capped` on the held-out grid, with a median
  difference of +0.01%.

## Where capped still loses

The real-model shapes where capped is more than 1.5% behind the best v20 column (21 of 630) fall into
four groups. None is changed here, because fixing them would mean refitting capped's coefficients to
these measurements.

- **The reduce-scatter cap at S = 28-32** (11 shapes, 1.5-6% behind auto). `n = min(128 // S, ...)`
  gives n = 4 there, and auto's n = 8 is faster on these shapes. It is not one-sided: over all 25
  real-model shapes with S = 32 where capped runs rs at n = 4, the median is +0.3% against auto, with a
  range of -6% to +19%. Entry 5's fit set also had n = 6 ahead of n = 4 at S = 32.
- **The minimum-K threshold** (5 shapes, 2-11% behind data-parallel). These are small no-full-wave
  shapes at 24 k-tiles, where `(256 - S) * k` is 5280-5712, just above the 5120 cut.
- **The owner's n = 2 at S = 64-80** (3 shapes, 1.5-3% behind auto's default split of 3-4 ways).
- **n = 1 as data-parallel where n = 2 is better** (2 shapes, above).

Open: shapes with bias, and fp32 C beyond the 32 such real-model shapes.
