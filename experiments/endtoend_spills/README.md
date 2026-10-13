# Spills in v26's end-to-end stream-K builds (two-tile, spread)

`STREAMK_POLICY=two_tile` and `STREAMK_POLICY=spread` both compile v26's end-to-end path (`END_TO_END`), which
has spilled since v19 (14-16 VGPRs). Two-tile also was not faster than the default policy. Where are the spills,
what do they cost, and can they be removed? Investigated on copies of v26's kernel (`variants/`); one change was
adopted in v26 (WORKLOG entry 3). Triton v2.2 + deepbind shim and the LLIR scheduler plugin, no amdgcnas hook,
GPU 3.

```bash
cd kernels/gemm/intra_wave/a16w16
python ../../../../experiments/endtoend_spills/make_variants.py crow crow+split   # variants/base.py + patches
HIP_VISIBLE_DEVICES=3 python ../../../../experiments/endtoend_spills/survey.py base,crow \
    4352x4096x8192:STREAMK_POLICY=two_tile --lines --mem
HIP_VISIBLE_DEVICES=3 python ../../../../experiments/endtoend_spills/time_case.py --rounds 3 \
    base:4352x4096x8192:8192:STREAMK_POLICY=two_tile crow:4352x4096x8192:8192:STREAMK_POLICY=two_tile
python ../../../../experiments/endtoend_spills/regions.py a.amdgcn b.amdgcn   # per-region diff, no GPU
```

`survey.py` compiles each (variant, case) in a fresh Triton cache (`compile_case.py`), checks C against torch, and
reports the scratch ops per loop region (`regions.py`: `out` once per program, `L1` the persistent tile loop, `L2`
the end-to-end segment loop, each split into `pre` / `k` / `post` around its K loop), SGPR lane spills, and with
`--mem` the widths of the buffer accesses after the K loops. `time_case.py`: cold rocprofv3 kernel time, rotating
copies over 512 MB, median of 300 launches, each case once per round, rounds in alternating order.

## Summary

- **Where:** all 14 spilled VGPRs (aligned real shapes; 15 in the asm check's `GEMM_TILE_LIMIT` builds) are
  thread-id values that LLVM computes once, before the segment loop, and holds across every segment's K loop:
  the C store's row offsets (`stride_cm * row`, 8 values) and column offset (1), and 5 LDS addresses of the C
  store's layout conversion (`store_c`). No K loop has a scratch op: the 14 stores run once per program and
  29 reloads run in every segment's ending, around the C stores. The persistent loop is not the cause:
  without it (`nofull`) the segment loop alone spills the same 14.
- **What they cost:** 4-6 us per kernel. `crow` makes the C row offsets tile-dependent (clamped to the tile's
  last row, as v26 entry 2 does in the persistent loop under `EDGE_M`). The K loops stay instruction-identical
  and the ending keeps the same loads and stores; the spills go from 14 to 5 and the ending's reloads from 29
  to 9. Every end-to-end shape timed got faster: 1.5-3.4% on one-wave two-tile shapes, 1.4% on spread at K = 8192,
  0.2-0.4% on spread at K = 32768, 1-5% on ragged shapes, with or without bias.
- **Why so much for 20 loads (a hypothesis, not measured directly):** the reloads run right after a K loop
  that has streamed the tile's A and B through the caches, so the spill slots, written once at the program's
  start, have been evicted. The reloads come in dependent batches (once per quadrant for the LDS addresses),
  and each batch waits for memory.
- **Two-tile against the default:** with `crow`, two-tile beats the default policy (`capped`, which never picks
  two-tile) on all nine shapes timed, S = 2 to 64: by 0.5-7.5%. Before, it lost on three of them.
- **The last 5** (4 partial slot offsets, 1 column offset): every attempt either moved spills elsewhere or broke
  the 16-byte accesses (table below). Fewer spills is not the goal in itself: `crow` raises the bias builds'
  count (22 to 31) but still cuts their reloads in the ending, and is faster there too.

## Variants ([survey_variants.md](results/survey_variants.md), [survey_first.md](results/survey_first.md))

Every patch applies only in the segment loop (`streamk_segment`'s `SEG_LOOP`). VGPR spills, and scratch loads in
the ending, on 4352x4096x8192 two-tile (spread 4096x7168x32768 is the same):

| variant | change | spills (ending loads) | with bias | result |
|---|---|---|---|---|
| base | v26 entry 2 | 14 (29) | 22 (43) | |
| **`crow`** | C row offsets clamped to the tile's last row (`rows_left - 1`) | **5 (9)** | 31 (34) | **adopted**, faster everywhere |
| `ccol` | columns clamped too, per column | 69 (28) | 93 | C stores become `buffer_store_short`: 2-20% slower |
| `cgrp` | `crow` + columns clamped by 8-column group (keeps 16-byte stores) | 5 (9) | 32 | moves the column's spill to the prologue |
| `crow+pclamp` | partial slot rows clamped to a bound the compiler cannot see as >= 127 | 7 (11) | 36 | more spills |
| `crow+pclamp2` | `pclamp` + slot columns clamped by group of 4 | 15 (9, + 6 stores) | 35 | partial accesses scalarised: 1.5-37% slower |
| `crow+plin` | slot map built from a 1-D arange in its own blocked layout | 6 (11) | 22 | partial accesses fall to 8 bytes |
| `crow+bgrp` | bias offsets clamped by group of 4 | 5 (9) | 31 | no change |
| `nofull` | no persistent loop when there are no full tiles, aligned shapes too | 14 (29) | 26 | the persistent loop is not the cause |
| `split` / `crow+split` | the loop's two endings as two calls with a constant ending | 13 / 6 | 33 / 26 | two K loop copies; ties `crow` without bias, at most 0.5% faster with it |

`ccol` and `crow+pclamp2` change the ending's access widths (the `--mem` column), so their times measure that,
not spills. `split`'s two K loops make the region labels approximate.

## Timing (cold kernel us, median of 3 rounds)

[timing_first.md](results/timing_first.md), [timing_bias.md](results/timing_bias.md),
[timing_ragged.md](results/timing_ragged.md), [timing_split.md](results/timing_split.md); row stride K (8256 for
the ragged shapes):

| shape | policy | base | `crow` | change |
|---|---|---|---|---|
| 4352x4096x8192 (S = 16) | two-tile | 220.1 | 215.2 | -2.2% |
| 4352x4096x4096 (S = 16) | two-tile | 125.3 | 121.0 | -3.4% |
| 8192x8448x8192 (S = 32, 4 waves) | two-tile | 788.8 | 784.3 | -0.6% |
| 4096x7168x8192 (S = 192) | spread | 346.6 | 341.9 | -1.4% |
| 4096x7168x32768 (S = 192) | spread | 1289.1 | 1286.7 | -0.2% |
| 4096x7936x32768 (S = 240) | spread | 1427.7 | 1422.1 | -0.4% |
| 4352x4096x8192, bias | two-tile | 225.8 | 222.8 | -1.3% |
| 4352x4096x4096, bias | two-tile | 128.9 | 126.5 | -1.9% |
| 8192x8448x8192, bias | two-tile | 796.9 | 792.7 | -0.5% |
| 4096x7168x8192, bias | spread | 350.8 | 346.7 | -1.2% |
| 4096x7168x32768, bias | spread | 1298.5 | 1296.0 | -0.2% |
| 4300x4000x8176 (M, N, K ragged) | two-tile | 225.4 | 220.1 | -2.4% |
| 4300x4000x8176 | spread | 265.7 | 261.3 | -1.7% |
| 4300x4000x8176, bias | two-tile | 230.8 | 228.4 | -1.0% |
| 4352x4096x8080 (K ragged) | two-tile | 225.8 | 223.8 | -0.9% |
| 1000x2000x8080 (fewer tiles than a wave) | spread | 71.5 | 67.9 | -5.0% |

Rounds agree within about 1 us. On ragged shapes `crow` removes fewer reloads (all three edges: 65 to 47 per
ending, [survey_ragged.md](results/survey_ragged.md)) but is still faster.

**Two-tile against the default policy** ([timing_two_tile_vs_default.md](results/timing_two_tile_vs_default.md);
v19's sweep shapes; the default is `capped`'s one-tile split):

| shape | S | default | two-tile, base | two-tile, `crow` | `crow` vs default |
|---|---|---|---|---|---|
| 1536x11008x8192 | 2 | 208.4 | 204.6 | 200.5 | -3.8% |
| 3328x5120x8192 | 4 | 207.8 | 212.8 | 206.6 | -0.6% |
| 4352x4096x8192 | 16 | 221.5 | 220.2 | 214.5 | -3.2% |
| 4352x4096x4096 | 16 | 123.1 | 124.3 | 120.8 | -1.9% |
| 4096x16640x8192 (4 waves) | 16 | 777.5 | 776.1 | 772.6 | -0.6% |
| 4096x4608x8192 | 32 | 234.8 | 230.7 | 226.5 | -3.5% |
| 4096x4608x4096 | 32 | 131.2 | 130.0 | 126.9 | -3.3% |
| 4352x4352x8192 | 33 | 246.9 | 232.4 | 228.4 | -7.5% |
| 4096x5120x8192 | 64 | 255.4 | 257.8 | 254.0 | -0.5% |

## Checks

- `asm_check.py` (v26 [results/asm_check.md](../../kernels/gemm/intra_wave/a16w16/v26_streamk_ragged_k/results/asm_check.md)):
  every build except the two end-to-end ones still compiles to v25's instructions. End-to-end: two-tile 15 -> 6
  spills (K - 16 / K - 112: 16 -> 9 / 9; all edges forced 16 -> 14), spread 15 -> 6 (16 -> 7 / 5; 23 -> 22).
- `check_streamk.py --policies two_tile spread --orders v9 split` ([check_streamk_end_to_end.md](results/check_streamk_end_to_end.md)):
  seven shapes (v26's four, 4300x4000x8176, 1000x2000x8080, 4096x7168x8192), fp32 and bf16-pair partials, with
  and without bias: 112 configurations, no wrong results, no stray writes, every flag re-armed.

## Conclusions and next steps

- Spills in the segment ending are not free: each ending reload batch seems to wait for memory, and removing
  20 reloads is worth 4-6 us per kernel. When looking at spills, count the reloads that run after a K loop,
  not the spill count.
- Adopted: `crow` in v26 (entry 3), compiled only in the end-to-end segment loop.
- The policy: `capped` never picks two-tile, but with `crow` two-tile won on every shape timed from S = 2 to 64
  with a full wave to borrow. Choosing it needs the same kind of sweep as v20's entries 5-9 (more S, K and wave
  counts, real-model shapes) before changing the default.
- The last 5 spills need a cheaper slot-offset map (one register instead of 4) that keeps 16-byte accesses, or
  fewer registers across the segment's K loop; none of the tries here found one.
