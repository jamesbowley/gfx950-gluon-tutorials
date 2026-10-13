# To do: open items from the a16w16 studies

Things measured but not explained, or fixes not yet made, with enough detail to pick them up later.
Newest first.

## 64x64x128 is faster with the old hand-built LDS layout (2026-10-13)

**What.** v9_tile switched to aiter's derived layouts for every tile (v9_tile WORKLOG §15). At
64x64x128, the hand-built layout it replaced was 0.4-2.3% faster, in all four runs, although it has
two-way LDS bank conflicts and the derived one has none. Every other tile was equal or faster with the
derived layout (128x128x128: +1-4%).

**The two layouts** (A's 32-row half-tile at BLOCK_K = 128; B's is the same, transposed). Offset bases in
LDS address order, 16 elements of padding after every 512:

| | address bits 0-5 | bits 6-8 (lanes) | bits 9-11 | warps | one wave's load | LDS reads |
|---|---|---|---|---|---|---|
| hand-built | K 1-32 | rows 4, 8, 16 | K 64, rows 1, 2 | rows 1, 2 (K 64 is a second load) | 8 rows x 128 B | 2-way conflicts |
| derived (now) | K 1-32 | K 64, rows 8, 16 | rows 1, 2, 4 | rows 1, 2 (row 4 is a second load) | 4 rows x 256 B | conflict-free |

The hand-built layout came from `_half_tile_bases` in `v9_tile/matmul_kernel.py` as of commit 73976e5:
the top three row bits on lanes, rows 1 and 2 on warps, K bits past 64 next. The compiled loops are the
same instruction for instruction (131 instructions, 32 MFMAs, 16 buffer loads, 32 ds_reads); only the
addresses differ, plus 116 against 112 VGPRs. Both run 2 workgroups per CU (65.9 KB of LDS each).

**Measurement.** 2048x4096x{4096, 8192}, bf16, 2 GB rotating inputs, back to back.

d / ours throughput ("ours" is the hand-built layout):

| K | GPU 3 | GPU 4, repeat | GPU 3, sustained | GPU 4, with amdgcnas |
|---|---|---|---|---|
| 4096 | 0.983 | 0.986 | 0.983 | 0.977 |
| 8192 | 0.995 | 0.996 | 0.994 | 0.988 |

K = 8192, GPU 3. ATT on CU 0 gives stalls per loop iteration (2 k-steps) and wave; the LDS columns come
from the counters:

| layout | TFLOPS | ATT loop MFMA eff | vmcnt | buffer_load issue | lgkmcnt + ds_read | s_barrier | cycles issued | LDS conflict cycles per access | LDS busy |
|---|---|---|---|---|---|---|---|---|---|
| hand-built | 679 | 21.8% | 200 | 724 | 364 | 324 | 2448 | 0.99 | 64% |
| derived | 676 | 21.1% | 75 | 987 | 191 | 461 | 2546 | 0 | 31% |

Clock (2.18 against 2.22 GHz), power (1400 W cap) and L2 hits (81.2%) match.

- **Data:** `v9_tile/results/layout_ab_{timing_graph,rep_timing_graph,clock,counters,att,gcn_timing_graph}.csv`.
  The rows `64x64x128` are the hand-built layout and `64x64x128d` the derived one.
- **ATT traces:** `v9_tile/tmp/layout_ab/sweep_att/64x64x128{,d}_2048x4096x8192/`.
- **Table:** `python v9_tile/report.py layout_ab`.

**What the numbers say.**
- Conflict-free reads halve the LDS read stalls (364 to 191) and the `vmcnt` waits (200 to 75).
- Issue stalls on `buffer_load` grow by 263 and barrier stalls by 137, more than the reads save.
- So the TCP is fuller, or the two workgroups on a CU line up worse at the barriers.
- 128x128x128 makes the same change to its load shape and gains. The difference may be the two
  workgroups per CU (128x128x128 runs one), or the share of the loop the loads take.

**To try.**
1. **Hybrid layout.** Lanes on rows 4, 8, 16 (8 rows x 128 B per wave, as hand-built), rows 1 and 2 on
   warps, K 64 last:
   `[[0,1],[0,2],[0,4],[0,8],[0,16],[0,32],[4,0],[8,0],[16,0],[1,0],[2,0],[0,64]]`.
   Under the bank model of WORKLOG §15.3 (128 banks of 4 B, 32 lanes per pass) it is conflict-free. It
   separates the load shape from the bank conflicts: if it matches the hand-built speed, the load shape
   is what costs.
2. **Counters for the load side**, hand-built against derived:
   - TA / TD busy and TCP request counts (`TCP_TOTAL_CACHE_ACCESSES`, `TCP_TCC_READ_REQ`)
   - TCC channel balance
   - `SQ_INSTS_LDS` per wave
3. **One workgroup per CU.** Pin 64x64x128 to one workgroup per CU, with `waves_per_eu` or an LDS pad
   as v9_tile_small does. If the difference vanishes, the two workgroups' interleaving is the cause.
4. **Other shapes and K**, to see whether it is shape-specific: the full wave 2048x2048, and K = 16384.

## v9_tile_2r with ragged K fails under main's amdgcnas (2026-10-13)

128x128r2, 32x256r2 and 128x128x128r2 fail 27 ragged-K cases (K = 400, 4032, 8080, 8176) with
`TRITON_AMDGCNAS_PLUGIN=1`, and pass without it, with either layout
(`v9_tile/results/check_tiles_derived.md`). Main's amdgcnas lacks the fixes of v9_tile WORKLOG §2 and
§14.3, which are in commit d6873fa. Restoring them, or upstreaming them to main, should fix it; recheck
with `python v9_tile/check_tiles.py 128x128r2,32x256r2,128x128x128r2`.
