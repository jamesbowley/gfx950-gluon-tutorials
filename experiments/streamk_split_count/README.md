# Split count and small K for the stream-K tail

Two questions about the one-tile stream-K tail (v15-v20 in
[kernels/gemm/intra_wave/a16w16](../../kernels/gemm/intra_wave/a16w16)):

1. **How many ways should each leftover tile be split?** With S leftover tiles the versions use
   all 256 programs, so each tile is split about `256 / S` ways. For S = 38 that is 6-7. Fewer
   splits mean longer chunks but fewer partials to collect. Section 9 of
   [STREAMK_EXPLAINED.md](../../kernels/gemm/intra_wave/a16w16/v14_streamk/STREAMK_EXPLAINED.md)
   calls this option (b) and models the best split as `n = sqrt(iters_per_tile / c)` for a serial
   owner, where `c` is the cost of one partial read in k-steps.
2. **When is K too small to split at all?**

```bash
# from the repo root, with the published v10+ stack (see the a16w16 WORKLOGs)
HIP_VISIBLE_DEVICES=3 python experiments/streamk_split_count/check_all.py   # correctness, all versions
HIP_VISIBLE_DEVICES=2 python experiments/streamk_split_count/sweep.py a     # split count at K = 8192
HIP_VISIBLE_DEVICES=0 python experiments/streamk_split_count/sweep.py b     # K = 512-8192
python experiments/streamk_split_count/sweep.py report                      # results/sweeps.md
```

## The knob

Every stream-K version (v15-v20) took a `STREAMK_NUM_PROGRAMS` constexpr marked "not used yet".
It now works in all of them. The host reads the environment variable `STREAMK_NUM_PROGRAMS`
(default 256, all programs), and only that many programs take part in the stream-K phase.

- **Tile-aligned split** (v17, v18, v20 `one_tile`): `n = STREAMK_NUM_PROGRAMS // S` programs per
  tile, so `STREAMK_NUM_PROGRAMS = S * n` gives exactly n.
- **End-to-end split** (v15, v16, v19/v20 two-tile, v20 `spread`): the ranges are cut into
  `STREAMK_NUM_PROGRAMS` pieces. With `S * n` programs, each program gets exactly 1/n of a tile.
- **Which programs idle** (`streamk_compact_pid`): each XCD keeps the same number of active
  programs, numbered contiguously, so the tail still runs on all eight L2s.
  `STREAMK_BALANCE_XCDS=0` instead idles the last spids, which leaves whole XCDs empty.

Correctness ([results/check_all.md](results/check_all.md)):
- Each version's own `check_streamk.py` (10 rotating launches) was run on S = 4, 16, 38 and 64:
  - at n = 1, n = 2 and the default;
  - in v20, with every policy and ending, the unbalanced mapping, and K = 512 / 1024.

  All 258 configurations are correct with every flag and count re-armed.
- v20's `check_partition.py` covers the compacted ids for every count from 1 to 256, and the
  end-to-end partition with `S * n` programs.

All times below are rocprof kernel times: 1000 back-to-back launches over a rotating 512 MB of
inputs, median. Small-K kernels take 30-150 us, and CUDA-event timings would include Python
launch gaps. Full tables are in [results/sweeps.md](results/sweeps.md), raw rows in
`results/sweep_a.jsonl` and `results/sweep_b.jsonl`.

## 1. Split count (K = 8192, all versions)

Best n per version against the default split, in us (v13, data-parallel, is 279-294):

| S | v13 | v15 | v16 | v17 | v18 | v20 owner | v20 rs | v20 spread |
|---|---|---|---|---|---|---|---|---|
| 4 | 278.6 | 627 -> 233 (n 4) | 346 -> 211 (n 8) | 341 -> 207 (n 8) | 351 -> 209 (n 6) | 342 -> 208 (n 8) | 209 -> **195** (n 16) | 339 -> 210 (n 8) |
| 16 | 278.7 | 318 -> 234 (n 4) | 243 -> 218 (n 8) | 237 -> 216 (n 4-6) | 245 -> 215 (n 6) | 237 -> 214 (n 4) | 216 -> **210** (n 8) | 244 -> 217 (n 4) |
| 38 | 280.2 | 285 -> 240 (n 4) | 251 -> 228 (n 4) | 236 -> 226 (n 4) | 230 -> **225** (n 4) | 235 -> 226 (n 4) | 236 -> 228 (n 4) | 249 -> 228 (n 4) |
| 64 | 282.5 | 252 -> 251 (n 2) | 239 (d) | 238 -> **237.6** (n 4) | 240 -> 239 (n 3) | 238 (n 3 / d) | 244 (n 4) | 240 (n 4) |
| 128 | 294.5 | 281 (d) | 276 (d) | **271.9** (d = n 2) | 272 (d) | 272 (d) | 290 (d) | 273 (d) |

What the numbers say:
1. **Capping the split is a large win when every tile is split many ways.** For the serial
   owner ending (v17, v18, v20 owner) at S = 4 it cuts the kernel time by 39% (342 to 208 us).
   At S = 16 the gain is 10%, at S = 38 4%, and none at S = 64-128, where the default is
   already 2-4 ways.
2. **The serial owner's best n matches section 9's model.** `sqrt(128 / c)` with `c` = 2.2 k-steps
   per read gives 7.6, and the measured best at S = 4 is n = 8. With more leftover tiles the best n
   falls (6 at S = 16, 4 at S = 38, 3 at S = 64): more programs publish at once, so each read
   costs more. `c = max(2.2, S / 5)` fits every measured S and K within 2% (sweep B).
3. **Reduce-scatter wants more splits than the owner** (n = 16 at S = 4, 8 at S = 16), because
   its ending costs about the same at any n. It is the best ending wherever tiles can be split at
   least 8 ways (S <= 32), and the owner's is best from S = 38.
4. **Every version gains in the same way.** v15's row-major partials make its fixup about twice as
   expensive, so it gains the most from capping (627 to 233 us at S = 4) and is still the slowest.
5. **XCD balance matters**: at S = 38, n = 2, leaving the idle programs on the last XCDs costs
   11% (owner, 236 to 262 us) and 15% (rs, 243 to 280 us).
6. **Two-tile with fewer programs only gets slower**: 213 to 315 us at S = 16 as the programs go
   from 256 to 128, and the same at S = 38. Every tile already has at most one peer, so fewer
   programs remove no fixups and only lengthen each range.

**Against two-tile.** With the split capped, one-tile is level with v19's two-tile or ahead of it.
- In rocprof (sweep B, K = 8192):
  - S = 4: rs at n 16 takes 194.6 us against two-tile's 203.3;
  - S = 16: rs at n 8 takes 210.2 us against 215.3;
  - S = 38: owner at n 4 takes 227.4 us against 233.9.
- With clean caches ([results/between_launches_k8192.md](results/between_launches_k8192.md),
  read 256 MB between launches):
  - S = 4: rs n 16 at 183.6 us against v19 at 185.0;
  - S = 16: 201.6 against 200.3;
  - S = 38: owner n 4 at 213.4 against 218.6.

  v19 is up to 4% ahead when the GPU idles between launches.

v20 entry 2 chose two-tile for short lags because v20's one-tile ending at the default split
lost to it by 3-15%. Most of that gap was the split count.

## 2. Small K

v20, S = 4-192, K = 512-8192. Gain is the best stream-K configuration against data-parallel (the
faster of v13 and v20 `dp`):

| S | K = 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|
| 4 | -5.8% | -4.0% | **+17.0%** (rs n 16) | **+34.6%** (rs n 16) | **+47.5%** (rs n 16) |
| 16 | -5.9% | -3.9% | **+8.3%** (owner n 3) | **+24.9%** (rs n 8) | **+36.6%** (rs n 8) |
| 38 | -4.6% | -3.6% | **+4.6%** (owner n 3) | **+16.9%** (owner n 3) | **+26.7%** (owner n 4) |
| 64 | -3.1% | -0.5% | **+3.1%** (owner n 2) | **+14.3%** (owner n 3) | **+21.5%** (owner, d) |
| 128 | -4.0% | -1.8% | -1.2% | **+5.5%** (owner, d) | **+7.7%** (owner n 2) |
| 192 | -4.0% | -2.3% | -1.0% | -0.9% | -1.1% |

- **K <= 1024 (16 k-steps or fewer per tile): never split.** The best "stream-K" configuration is
  n = 1, the stream-K machinery with no split, and it still loses 1-6%. The default splits lose
  30-80% (for example S = 16, K = 512: 49-56 us against 32).
  - The data-parallel last wave is short: 16 k-steps run on mostly idle CUs in about 12 us.
  - A split pays at least a pipeline restart (1-3 k-steps) plus its fixup (owner: 10 or more
    k-steps; reduce-scatter: about 25; two-tile: one read).
- **K = 2048 (32 k-steps): split only small S, and only a few ways.** The best n is 2-3 for the
  owner ending, and only reduce-scatter at S = 4 wants 16. The default splits lose here (S = 16:
  owner default 109 us against data-parallel's 80). So the split cap decides whether stream-K
  helps at all.
- **Two-tile is the worst choice at small K**: 11-68% behind data-parallel at K <= 1024, and
  behind or level with the capped one-tile split at every K here.
- **S = 192** is level with data-parallel at every K up to 8192, as in v20 entry 3. Its gain
  needs K >= 32768 and the end-to-end `spread` split.

**Threshold.** A tile-aligned split pays when the idle share of the data-parallel last wave,
`(256 - S) / 256 * iters_per_tile`, is at least about 20 k-steps. That separates every measured
point except S = 192 (where the tile-aligned split cannot help and `spread` needs about 128):
- wins at 24 or more (S = 64, K = 2048: 24; S = 128, K = 4096: 32);
- loses at 16 or less (S = 128, K = 2048: 16; S = 4, K = 1024: 15.75).

## Proposed host rule (not applied)

For v20's `streamk_policy`, with `ipt = iters_per_tile` and S leftover tiles:
1. **Data-parallel** if `(256 - S) * ipt < 20 * 256`. In practice that means K <= 1024 always;
   K = 2048 for S > 96; and S > 128, unless entry 3's spread rule applies (K >= 32768).
2. **One-tile with a capped split**, otherwise:
   - **reduce-scatter** if `256 // S >= 8`, with `n = min(256 // S, max(4, ipt / 8))`. That is
     n = 16 at K = 8192, 8 at 4096 and 4 at 2048. It is within 4% of the best measured n, except
     S = 4 at K = 2048, where n = 16 is 7% faster than the rule's 4;
   - **otherwise the owner's ending**, with `n = min(256 // S, round(sqrt(ipt / c)))` and
     `c = max(2.2, S / 5)` k-steps per partial read. It is within 2% of the best measured n.
     A fixed `c = 2.2` misses by up to 6% at S = 38-64, K = 2048, where reads cost more because
     more programs publish at once.
3. **Two-tile is no longer needed.** It ties the capped one-tile split at best. This reverses
   v20 entry 2's two-tile branch. Keep it only for shapes measured to favour it.

Points not covered: S between the measured values (an S = 24 or S = 96 shape would test the
thresholds), K = 16384 with capped splits, and bias. The trends are smooth, but the coefficients
(2.2, 1/8, 20) are fits to these shapes.

**Default behaviour is unchanged.** With `STREAMK_NUM_PROGRAMS` unset, `regress.py` on 4352x4096x8192 and
4352x4352x8192 gives every version's earlier TFLOPS within noise. Register counts and spills are the
same as before. For example v17 is at 1203 TFLOPS against 1213 earlier, with 492 VGPRs and no
spills; v15 is at 941 against 932.
