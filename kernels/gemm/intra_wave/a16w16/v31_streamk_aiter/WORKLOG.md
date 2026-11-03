# v31 worklog

Everything runs on rocm-smi GPU 1 (HIP 3, PCI 0000:15:00.0) under
`experiments/streamk_v31_rule/gpu_guard.py`, in tmux session `v31`; no foreign process
appeared during any run.

## 1. From v30: one file, AITER entry points, parameter reference

- `matmul_kernel.py` now holds the topology query (`streamk_topology`), the workspace
  (`_flags`, `_persistent_partials`, `streamk_workspace`), the config parsing
  (`TileSpec`, `StreamKConfig.from_dict`, exactly the eight CONFIG_KEYS) and the
  reduce-scatter reader rule (`reduce_scatter_config`, `rs_readers`). `runtime.py` is
  gone; `hardware.py`, `selection.py` and `heuristics.py` remain as offline tooling for
  the checks and sweeps, and the kernel imports none of them.
- No reference to v9, the tutorial, a worklog, `results/` or a `.md` file remains in
  the kernel module; error messages say "gfx950 gluon a16w16".
- The module docstring is the parameter reference: notation, then each of
  STREAMK_POLICY, STREAMK_FIXUP, STREAMK_EXCHANGE, SPLIT_N and GROUP_SIZE_M with a text
  diagram, and when each pays.
- `unsupported_reason(M, N, K, x, w, bias, activation, config)`,
  `gemm_a16w16_streamk(x, w, y, bias, config)` and `_KERNEL_MAP = {"compute_bound": ...}`
  follow the v9 port's interface on AITER's `llirSched_integration` branch.
- `env.sh` adds the AITER checkout to `PYTHONPATH` (the kernel imports AITER's
  `device_info` and `graph_alloc`; the venv has no installed AITER).

## 2. Non-persistent data-parallel launch (reverted, section 7)

`PERSISTENT` (kernel constexpr, default True) is False when the launcher sees policy
`dp` with T <= P: the program runs `persistent_tile` once with `PREFETCH_NEXT=False`,
which drops the next tile's coordinates, addressing and masks, the 8 epilogue
`buffer_load_to_shared` / `commit_group` pairs and the final operand reads, and drains
the pipeline with `wait_group(5, 4, 3, 2, 1, 0)`, the sequence v9's epilogue uses.
`gemm_a16w16(..., persistent=True)` forces the loop for A/B timing.

**Checks** (`checks_all.sh`, [results/checks.log](results/checks.log)): the 24 CPU tests
(new: the kernel's launch constants equal the offline resolver's on 100+ configs, the
PERSISTENT derivation, and the XCD remap equals v9's `_get_pids` on ragged grids and
GROUP_SIZE_M 1, 4, 8), then `run_checks.sh` (every case at five tiles, graph replay,
the stale-flag and stale-count negative controls), `--streams`, `--persistent-partials`
and `--extra` at three tiles. New cases: `dp_single` (T = 1), `dp_ragged_wave` (ragged
M, N, K, bias, T < P), and the forced-loop twins `pdp_wave` (T = P) and
`pdp_ragged_wave`. All pass.

**ISA** ([results/dp_asm_nonpersistent.md](results/dp_asm_nonpersistent.md)): without the loop the build issues a
quarter fewer async LDS loads than the persistent one (96 against 128 at 256x256x64;
v9: 64, v31 keeps its peeled first K iteration) and far fewer SGPRs (69 against 98);
the ragged 256x256x64 build loses the persistent one's 8 VGPR spills.

**Timing** ([results/dp_timing_nonpersistent.md](results/dp_timing_nonpersistent.md), 5 BLOCK_K-64 tiles x 12
single-wave cases, 9 rounds of graphs over 2 GB of rotating inputs):
- against the forced persistent loop: geomean 0.988 (0.958-1.001), most at K = 1024
  (0.96-0.98) where the discarded prefetch is a larger share;
- against v9: geomean 1.008 (0.970-1.047). v31 is faster on every ragged-K case (its
  masked tail is pipelined, v9's is not) and up to 4.7% slower on the shortest kernels
  (64x64 and 128x128 at K = 1024, 0.3-0.4 us), within 1-2% elsewhere.

## 3. The default Stream-K plan

[experiments/streamk_v31_rule](../../../../../experiments/streamk_v31_rule/README.md): about
55,000 points over seven sweeps, two rounds each, all correct. Given the tile, the plan is

```text
dp              if S == 0 or cap < 2                       (m = P // S, pairs = ceil(K / 2 BLOCK_K))
owner           if m <= 3:  n = sqrt(pairs / 1.5),   dp if pairs < 10
reduce-scatter  otherwise:  n = sqrt(pairs * m / 6), dp if pairs * (1 - 1/n) < 6
```

with the same constants for all 13 specs and both regimes; leaving out a whole spec does
not change its score (1.0104 geomean, 1.036 at the 90th percentile of each pair's best).
On the 597 held-out AITER shapes at every spec it is 1.0024 / 1.000 / 1.000 / 1.292 of each
spec's best (dp: 1.078 / 1.000 / 1.320 / 4.969). The tile is the larger loss: with
`pick_gemm_tile`'s tile the default is 1.066 geomean, 1.224 at the 90th percentile of the
shape's best over every spec; a Stream-K-aware tile pick gives 1.015 / 1.055.

## 4. The default plan in the kernel

`matmul_kernel.py` now holds the plan: `STREAMK_PLAN` (m_ow 3, c_ow 1.5, pairs_ow 10, b 6,
gain 6), `streamk_plan(M, N, K, tile, num_programs)` and `complete_config`. A config that
names only the tile (BLOCK_M, BLOCK_N, BLOCK_K, optionally GROUP_SIZE_M, default 4) gets the
plan with the `wt` exchange; the four plan keys are all given (and kept: a tuned config) or
all absent, a partial set is rejected. `_launch_constants`, `gemm_a16w16`,
`gemm_a16w16_streamk` and `unsupported_reason` all take either. The tile is never chosen
here: AITER always supplies it (a tuned row's `kernelName`, or the config JSON with
`pick_gemm_tile`), and the plan is the Stream-K counterpart of `compute_splitk_params`.

- **CPU** (`test_selection.py`, 29 tests): `streamk_plan` equals the experiment's
  `candidates.streamk_default` on every recorded (shape, spec) pair of the sweeps (over
  10,000), the constants equal `results/streamk_rule.json`, edge cases (S = 0, cap < 2,
  short K, the owner capped at 2, BLOCK_K 128 and ragged K), the completion rules, and every
  completed config launches.
- **GPU** (`checks_default.sh`, [results/checks_default.log](results/checks_default.log)):
  tile-only configs at 256x256x64, 128x64x64 and 128x128x128 through dp (whole waves, short
  K, below one wave), reduce-scatter and the owner after a full wave and below one wave, and
  a ragged reduce-scatter: every case runs the expected plan, matches torch, keeps the
  guards intact and leaves the flags re-armed. `checks_all.sh` passes again.

## 5. Inside AITER ([results/aiter_smoke.md](results/aiter_smoke.md))

A scratch worktree of `upstream/llirSched_integration` with this file next to the v9 kernel
and a pytest shim routing the branch's gfx950 entry points to it (the branch's default
tile, v31's default plan): 89 of the branch's gfx950 gluon tests pass, 6 fail.

- 4 are fp16 with bias at K = 8192: the reference is wrong. Against fp32, torch's
  `F.linear` is off by 4.0, v31 and v9 by 0.126; v9 fails them too.
- 2 are the 64x64 tile at 1024x1024x512: correct, but with no `sched_barrier` in the
  ISA. K is a constexpr, and at 8 K-steps LLVM unrolls the 64x64 builds' whole K loop
  (64 MFMAs inline). From K = 1024 every tile keeps its barriers.
- A cold compile takes 0.7-1.3 s per variant (v9 0.56 s); a new K, or a new S on a
  Stream-K path, costs one.

## 6. The remaining gap to v9 (non-persistent dp; reverted with section 2)

On even-K single-wave shapes v31's non-persistent launch is 1.4% slower than v9
(geomean, up to 4.7%; [results/dp_timing_nonpersistent.md](results/dp_timing_nonpersistent.md)), and 1.1% faster on
ragged K. The K loop is not unrolled at these K (barriers present from K = 1024), so that
is not the cause. Running a program's only tile forward on an even K (pair 0 in flight,
the loop from column 2 BLOCK_K, as v9) instead of last-pair-first changed v31 by 0.9954
(0.976-1.004) between runs while v9 itself moved 0.9971 (0.986-1.002): within noise
([results/dp_timing_forward.md](results/dp_timing_forward.md)), so it was reverted and
v31 keeps one K order. Not tried: v31's extra kernel arguments (the workspace pointers
and Stream-K constexprs) and its peeled first K iteration.

## 7. The non-persistent dp path removed

Section 2's `PERSISTENT` / `PREFETCH_NEXT` path is gone: every launch runs the persistent
loop again (one trip when T <= P), as v30 did. It added a compile variant for every
data-parallel launch with at most one tile per program, and it was still behind v9 (0.8%
geomean, up to 4.7%); for simplicity the kernel keeps one dp build.

- **Kernel:** `PERSISTENT` and `PREFETCH_NEXT` removed from `_gemm_a16w16_compute_bound_kernel`
  and `persistent_tile`, the `persistent=` argument from `_launch_constants` and
  `gemm_a16w16`. `_gemm_a16w16_compute_bound_kernel` is again identical to v30's (syntax
  trees compared, docstrings aside); `persistent_tile` differs from today's v30 only in
  comments and in v30's later coalesced store layout, which v31 never took.
- **Tools:** the `pdp` twins left `check_streamk.py`, the forced-loop variant left
  `dp_ab.py`, the PERSISTENT test left `test_selection.py`; the rule sweep's `grid.py` /
  `sweep.py` and `experiments/tune_aiter_shapes/tune.py` no longer time a `pdp` kind (the
  recorded CSVs keep their pdp rows, which the fits already skip).
- **Checks:** `checks_all.sh` and `checks_default.sh` pass ([results/checks.log](results/checks.log),
  [results/checks_default.log](results/checks_default.log)). A one-wave and a four-wave dp
  at the same K share one compiled kernel.
- **ISA** ([results/dp_asm.md](results/dp_asm.md)): the single-wave builds are the persistent
  ones again (256x256x64 ragged: 512 VGPRs, 8 spills, 128 async LDS loads; v9 428, 0, 64).
- **Timing against v9** ([results/dp_timing.md](results/dp_timing.md)): 2.7% slower geomean on
  even-K single-wave shapes (0.998-1.088, the most at K = 1024), level on ragged K (1.0008).

## 8. v30 entry 19 ported: layouts and offsets without switches on the tile

v30's WORKLOG entry 19, the one v30 kernel change v31 did not have (v30's entries 14, 17
and 18 and its vpid / e2e refactor were already in). Each v31 target was v30's code before
that entry, so this copies v30's code; v31's docstring wording stays.

- **Kernel:**
  - `coalesced_store_layout(rows, cols, elem_bytes, num_warps)` replaces both
    `STORE_CONTIG` `gStoreLayoutC` blocks (`persistent_tile`, `quadrant_segment`) and
    `reduce_scatter_slice`'s `SL` switch.
  - The `MFMA_*` size constants, and `in_block_row` / `in_block_col`.
  - `lane_contiguous_offsets` is one closed form instead of `HALF_M` / `HALF_N` branches.
  - `reduce_scatter_slice`'s `in_blk` uses the constants (still without the modulo on
    bit 16), and its bias column uses `block_col`.
  - `block_col` is `(v % VN) * 32`; `to_band` keeps only its general path, and `VN` has
    no default.
- **Checks** ([results/entry19_checks.log](results/entry19_checks.log)):
  - `checks_default.sh` passes.
  - `run_checks.sh` at eight tiles (64x64x64, 128x64x64, 64x128x64, 128x128x64,
    128x128x128, 256x128x64, 128x256x64, 256x256x64) passes: 168 cases and graph
    replay; the stale-flag control fails and stale-count hangs, as they must.
  - `--extra`, `--streams` and `--persistent-partials` at 64x64x64, 128x128x128 and
    256x256x64 pass.
- **ISA** ([results/entry19_asm.md](results/entry19_asm.md)): the default and extra
  cases at all eight tiles, built before and after on the same inputs.
  - All 240 outputs are bitwise equal, and 186 builds are instruction-identical.
  - Store instructions change only at 64x64x64 and 128x64x64:
    - at 64x64, half-predicated `buffer_store_dwordx4` becomes full-lane `dwordx2`;
    - at 128x64, a band's two `dwordx2` become one `dwordx4`.
  - Elsewhere the builds differ only in scalar block-column arithmetic: `(v % VN) * 32`
    compiles to `s_bfe_i32` instead of shifts, at most 17 instructions (256x256x64
    `l2_compact`).
  - No new spills. Outside the two small tiles the only VGPR change is 256x256x64
    `compact_rs`, 448 to 444.
  - The 64x64 VGPR changes are v30's: ragged_m 280 to 88, kres_gt_rs 168 to 84,
    two_tile_kres 252 to 96, and ragged_n up from 96 to 176.
- **Timing** ([results/entry19_timing.md](results/entry19_timing.md)): before against after
  in one process, `dp_ab.py`'s method, on HIP 3 (9 rounds) and HIP 5 (15 rounds).
  - The points: the twelve 64x64x64 dp shapes of
    [results/dp_timing.md](results/dp_timing.md), v30 entry 19's 64x64 and 128x64
    Stream-K points, and one 256x256x64 point.
  - New / old is 0.991-1.004 in both runs (geomean 0.999), inside the spreads. The
    recorded 64x64 dp timings stand.
- `persistent_tile` now matches v30's apart from comments (section 7's note on the
  store layout no longer applies).
