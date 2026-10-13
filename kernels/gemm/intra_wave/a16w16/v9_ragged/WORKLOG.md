# v9_ragged work log

v9_beyond_hotloop for M and N that are not multiples of the 256x256 tile, as the fallback of
v25_streamk_ragged_mn for data-parallel shapes of one wave or less. Results come from Triton v2.2 +
deepbind shim and the LLIR scheduler plugin, without the amdgcnas hook (as v24 and v25), run from
`a16w16/`, on GPU 0. Raw tables are in `results/`.

## 1. Edge handling (2026-10-10)

v9 has no edge masks: on a ragged M or N its loads read past A and B and its C stores wrap into the
next row or past the end of C (`regress.py` reports wrong results on 4000x4000x8192). v9_ragged takes
v25's edge handling: with `EDGE_M` / `EDGE_N` (the host sets both when either dimension is ragged;
`V9_FORCE_EDGE=1` for any shape) A rows past M and B columns past N are clamped to the tile's last
in-bounds one, so every load reads valid memory, and the four quadrant stores and the bias load are
masked to M x N. v9 already holds eight offset tensors (top, bottom, left, right, even and odd K-steps),
so clamping only changes how they are computed. With both flags off the kernel is v9's.

- **K is checked on the host**: a multiple of 128, at least 256 (v9 checks nothing, and 4096x4096x8000
  silently gives a wrong C, through v9 and through v24 / v25's one-wave fallback). The stream-K kernels
  check it with a `static_assert`, which the fallback path never compiles. (Entry 2 handles ragged K.)
- **Code** ([asm_check.md](results/asm_check.md)): with the flags off, 4096x4096x8192, 2048x4096x4096 and
  256x8192x8192, with and without bias, compile to v9's instructions and registers (448 VGPRs). The
  edge builds take 460 VGPRs, no spills.
- **Correctness** (`regress.py`, against torch): 4000x4000x8192, 4096x4000x8192, 4000x4096x8192,
  128x8192x8192 and 1000x2000x8192, and 4000x4000x8192 and 1000x2000x8192 with bias.
- **Timing**, cold kernel-only (v25's `rocprof_cold.py`: rocprofv3 over `bench.py --rocprof`, 1000 launches
  over a rotating 512 MB, median; [rocprof_cold.md](../v25_streamk_ragged_mn/results/rocprof_cold.md)):
  v9 on 4096x4096x8192 176.1 us; v9_ragged 176.7 us with the flags off and 175.4 us forced on
  (`V9_FORCE_EDGE=1`); 4000x4000x8192 (the same 16x16 grid) 175.0 us. `regress.py` (do_bench) agrees:
  181.8 / 184.1 us against 182.6 / 186.0 us.

## 2. Ragged K: the last pair first (2026-10-10)

`EDGE_K` (set by the host when K % 128 != 0; off, the kernel is entry 1's): K is rounded up to K_PAD, a whole
number of K-step pairs, and the prologue loads the last, partial pair (columns K_PAD - 128 .. K_PAD) with the
lanes at or past K masked to zero (`k_tail_mask`); the main loop then covers columns 0 .. K_PAD - 128 from the
row's start, unmasked. The host now requires K % 16 == 0 (16-byte rows for the async copies) and K_PAD >= 256.
This is also v26's handling ([v26 WORKLOG](../v26_streamk_ragged_k/WORKLOG.md), with the measurements).

- **First attempt, residue first with the loop shifted** by K_PAD - K: correct, but every main-loop read then
  starts 2 (K_PAD - K) bytes off the 128-byte lines; with a fixed row stride of 8256, 4096x4096x8176 took
  287.9 us and 8080 283.8 us against 183.5 us for 8192 (8128, shifted by a whole line, tied).
- **As committed**, same rows: 8192 183.2 us, 8176 182.4, 8128 183.4, 8080 183.3
  ([stride_timing.md](../v26_streamk_ragged_k/results/stride_timing.md)).
- **Contiguous inputs** with K not a multiple of 64 also misalign the rows themselves: 221.4 us at
  4096x4096x8176, as unchanged v9 at K = 8192 with a row stride of 8208 (220.7 us). That is the layout, not the
  kernel; pad the row stride to a multiple of 64 elements.
- **Correctness** (`regress.py`): 4096x4096xK for K = 8176, 8128, 8080, 4000x4000x8080 (with and without bias),
  4096x4096x144, 1000x2000x192, 128x8192x4000. **Code** ([asm_check.md](results/asm_check.md)): with the flags
  off still v9's instructions; ragged K builds 448 VGPRs, with M and N edges and bias 468, no spills.

## 3. A `GEMM_TILE_LIMIT` test hook (2026-10-10)

For sweep G (data-parallel with fewer tiles than programs), v9_ragged takes v13's test hook: with
`GEMM_TILE_LIMIT = T` (constexpr `TILE_LIMIT`, default 0) the grid is T programs and program p takes tile id p, with
the XCD remap over T (`get_pids` with T in place of the grid size) and the GROUP_SIZE_M swizzle over the full grid's
tile counts, so it computes exactly the grid's first T tile ids. **Code** ([asm_check.md](results/asm_check.md)): with
the hook off, still v9's instructions and registers on every shape; with it on (4096x8192x8192, T = 40 and 400), 8
instruction lines change (the remap constants), 448 VGPRs, no spills. **Correctness**: all 165 sweep G cases
(T = S and T = 256 + S) write exactly the first T tiles and match torch
([check_tile_limit_g.md](../../../../../experiments/streamk_split_count/results/check_tile_limit_g.md)).
