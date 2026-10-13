# Spills in v26 (ragged M, N, K)

Can v26_streamk_ragged_k's VGPR spills be removed without costing time? Investigated on copies of v26's kernel
(`variants/`); the versions are unchanged. Triton v2.2 + deepbind shim and the LLIR scheduler plugin, no amdgcnas
hook, GPU 4.

```bash
python experiments/ragged_spills/make_variants.py crow                      # variants/base.py + named patches
HIP_VISIBLE_DEVICES=4 python experiments/ragged_spills/survey.py base,noshift 4300x4000x8080 --lines
HIP_VISIBLE_DEVICES=4 python experiments/ragged_spills/time_case.py base:4300x4000x8080:8256 noshift:4300x4000x8080:8256
```

`survey.py` compiles each (variant, case) in a fresh Triton cache (`compile_case.py`; `CASE_BIAS=1` adds a bias),
checks C against torch, and reports registers, spills, scratch ops by region (`check_spills.parse_amdgcn`: K loop
/ the largest loop / other) and by source line. `time_case.py`: cold rocprof kernel time over rotating copies with
a fixed row stride (8256 unless noted), median of 300 launches.

## Summary

- **v25 / v26 spill on several classes of real shapes**, all at the 256 architectural-VGPR limit (the other 256
  are the accumulators' AGPRs): M, N and K ragged (9-12 spills), M / N ragged with a bias (4; with ragged K too,
  18-23), N below 256 with many M tiles (131072x200x8192: 12, plus 336 SGPR spills; with ragged K 44, 22 per tile).
- **None of these spills is measurable.** 131072x200x8192 runs in 492-495 us, its aligned twin 131072x256x8192
  (same tile grid, no spills) in 498-500; 131072x200x8176 (44 spills) in 494-499. Earlier, removing tile-loop
  spills (`crow`) gained at most 1.3%.
- **Removing all of them was not possible without a cost.** A redesign that shifts ragged edge tiles back (below)
  removes the spills on most classes, but its overlapping loads cost 1-3% on M / N-ragged shapes, more than the
  spills did; and at the register limit every fix moves spills elsewhere (one change fixes one class and
  regresses another), so no variant is spill-free on every class. N below 256 with many M tiles spills in every
  variant.
- **What is free** (`noshift`, timed against base): `crow`, `kopaque` when K_PAD - K > 64, and `HAS_FULL`. Fewer
  spills everywhere, none in the tile loop on most classes, and no regression: ties on multi-wave shapes, 1% faster
  on 4000x6080x8080, 3-6% faster on shapes of less than one wave (all spills gone there).

## Where the spills come from

- **In the persistent tile loop**: `stride_cm * row` for the epilogue's C store offsets. v24 keeps the column in
  those offsets so they are not hoisted above the tile loop, but the row half of the product is loop-invariant,
  so LLVM hoists it and holds it across the loop; with the edges' extra live values it is spilled.
- **Outside both loops** (once per program): thread-id derivatives (`tid & 63`, `& 48`, `& 15`, `& 0x3ff` and ORs)
  that the stream-K phase shares with the prologue, held (spilled) across the persistent loop. LLVM cannot
  rematerialise them (the thread-id register does not survive) and Gluon gives no hold on where its aranges'
  thread ids come from; they fit only if the persistent loop leaves room.
- **Hoisted K-tail masks**: the persistent epilogue's prefetch masks are loop-invariant; with K_PAD - K > 64 (both
  K-steps masked) LLVM holds the masked offsets across the loop.

## Variants ([survey_variants.md](results/survey_variants.md), [survey_final.md](results/survey_final.md), [survey_oddrule.md](results/survey_oddrule.md), [survey_noshift.md](results/survey_noshift.md))

| variant | change | effect |
|---|---|---|
| `kbcast` | K-tail masks over K only, broadcast by the load | none |
| `kopaque` | the epilogue's K-tail masks depend on the next tile (not hoisted) | K_PAD - K > 64: fewer registers (K-only 496 -> 480); **K_PAD - K <= 64: 3-11 more spills** |
| `crow` | under `EDGE_M`, C row offsets `stride_cm * min(row, rows_left - 1)`: tile-dependent, same stored values | tile-loop spills gone on the default endings |
| `shift` | ragged edge tiles start at `extent - 256` (overlapping the tile before), store masks get a lower bound; every load in bounds, so v24's kernel-wide offsets and no per-tile offset tensors | with `kopaque`: M, N and K ragged spill-free at 468-476 VGPRs; but **1-3% slower** on M / N-ragged shapes (the overlap re-reads up to 255 rows / columns) |
| `HAS_FULL` (`nopers`) | host-known: no full tiles (less than one wave) -> the persistent loop and its prologue are not compiled | shapes below one wave: 412-424 VGPRs, no spills (not even SGPR), 3-6% faster |
| `SMALL_M / N` | a dimension below 256 clamps its loads once (one tile row / column), not per tile | helps with K aligned; with ragged K and many tiles still spills |
| `oddshift`, `oddrule` | edge-free operands take the odd K-step as a scalar shift | with K aligned: M / N-ragged builds spill-free (468-492, with bias too); with ragged K: worse |
| `lateops` | the next tile's K-step 0 operands read at the next trip's start instead of the epilogue | 4-12 fewer VGPRs at most; does not remove the remaining spills |
| **`noshift`** | `crow` + `kopaque` only when K_PAD - K > 64 + `HAS_FULL` | below |

`noshift` against base (VGPR spills, tile-loop scratch ops; [survey_noshift.md](results/survey_noshift.md)):

| build | base | noshift |
|---|---|---|
| M, N, K ragged (4300x4000x8080, 8150x8400x8080, 4000x6080x8080, owner) | 9-12 (3-6) | 5 (0) |
| ... K_PAD - K = 16 (4300x4000x8176) | 11 (6) | 8 (0) |
| ... with bias (4300x4000x8080 / 8176) | 18 / 23 (12 / 11) | 22 / 28 (12 / 14) |
| M / N ragged, K aligned (4300x4000x8192), with bias | 0, with bias 4 (0) | 0, with bias 4 (0) |
| less than one wave (128x8192x8176, 1000x2000x4032, 4096x128x8080) | 9-11 (3-6) | **0** |
| dimension below 256, many tiles, ragged K (128x98304x8176, 98304x128x8176) | 11 (5) | 9 (2) |
| N below 256, many M tiles (131072x200x8192 / 8176) | 12 / 44 (6 / 22) | 11 / 47 (6 / 24) |
| K only (4352x4096x8080 / 8176), aligned | 0 | 0 (8080: 496 -> 480 VGPRs) |

## Timing ([timing_noshift.md](results/timing_noshift.md), [timing_final.md](results/timing_final.md), [timing_nsmall.md](results/timing_nsmall.md); cold kernel us, two rounds)

| shape | base | noshift | shift-based (`final+nopers+oddrule2+kkeep`) |
|---|---|---|---|
| 4300x4000x8080 | 215.2 / 214.0 | 214.7 / 214.6 | 216.6 / 216.8 (`final+nopers`) |
| 4000x6080x8080 | 277.2 / 276.9 | 274.5 / 274.2 | 276.5 / 274.1 (`final+nopers`) |
| 8150x8400x8080 | 745.6 / 741.4 | 742.6 / 742.7 | 744.7 / 743.8 (`final+nopers`) |
| 4300x4000x8176 | 214.6 / 215.8 | 215.3 / 214.4 | 220.1 / 219.2 |
| 4300x4000x8192 | 213.6 / 214.6 | 214.8 / 214.3 | 217.9 / 216.0 |
| 4352x4096x8080 | 218.8 / 218.2 | 218.8 / 217.0 | |
| 1000x2000x4032 (row stride 4160) | 36.1 / 35.9 | 35.7 / 35.4 | |
| 128x8192x8176 | 64.5 / 64.9 | 62.7 / 63.0 | |
| 4096x128x8080 | 40.0 / 39.8 | 37.6 / 37.4 | |
| 128x98304x8176 | 353.0 / 349.9 | 350.4 / 352.3 | |
| 131072x200x8192 (aligned twin 131072x256: 497.5 / 500.2) | 494.6 / 491.9 | | 500.1 / 497.6 |

## Conclusions

- The spills seen so far are harmless in time: those left are once per program or a few per tile.
- **Adopted**: `crow` and `HAS_FULL` in v26 (entry 2) and `crow` in v25 (entry 4), each compiled only with an edge
  flag on. `kopaque` was dropped: v26's asm check builds (`GEMM_TILE_LIMIT`, all three edges) went from 500 VGPRs
  and no spills to 512 and 6-14 spills with it, and without it from 500 to 492, no spills; on real shapes it never
  changed a spill count. Final surveys: [survey_v26_final.md](results/survey_v26_final.md),
  [survey_v25_final.md](results/survey_v25_final.md); timings: [timing_v25_v26_final.md](results/timing_v25_v26_final.md),
  [timing_bias_final.md](results/timing_bias_final.md) (ties, and 2-6% faster below one wave).
- Zero spills on every class needs headroom the persistent kernel does not have at 256 VGPRs. Options outside this
  experiment: compile the persistent and stream-K phases as separate kernels (each gets its own register budget,
  at the cost of a launch boundary in the tail), or get LLVM to rematerialise the thread-id values (a compiler
  change). Shifting edge tiles removes most spills but costs 1-3% in memory traffic.
