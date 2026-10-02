# wip_streamk on the v2.2 main: regression checks and a16w16 bias

`wip_streamk` rebased onto `origin/main` (`ef48b5f`, re-pin to `gfx950-tutorial-v2.2` with
`cd_regclass="a"`, #54). On top: v10-v13 moved to main's conventions, a `bias[N]` option for
a16w16 v9-v13, and one amdgcnas fix. Everything ran on GPU 3 only (`HIP_VISIBLE_DEVICES=3`),
Triton v2.2 via `PYTHONPATH=<triton_gfx950-tutorial-v2.2>/python` plus the deepbind shim; the
venv's editable Triton install (v2.1) was not changed.

Scripts: [run_matrix.sh](run_matrix.sh) (run_perf_table `--rocprof` matrix),
[dump_asm.sh](dump_asm.sh) + [compare_asm.py](compare_asm.py) (assembly identity, debug info
ignored), [check_correctness.sh](check_correctness.sh), [summarize.py](summarize.py). Raw logs
and tables: [results/](results/).

## 1. Rebase

One branch commit, replayed onto `ef48b5f`; the pre-rebase commit is kept as
`backup/wip_streamk-pre-rebase`. The scripts (`run_perf_table.py`, `run_counter_collection.py`,
`scripts/README.md`) merged cleanly. Five kernels (a16w16 v8/v9, a8w8, a4w4 v0/v1) conflicted on
the imports only: upstream dropped `import os`, the branch had moved `import triton` ahead of
`import torch`; resolved to upstream minus `os`, with the branch's import order.

Adapting the branch to main: `cd_regclass="a"` on all 16 MFMAs of v10-v13 and the
`amdgpu-agpr-alloc=256` / `TRITON_FORCE_MFMA_AGPR` launch line removed; v10-v13 added to
`REPORTED_COMBINATIONS` under `llir` and `llir+amdgcnas`; `wave_quant_sweep.py` and
`plot_pipeline_schedule.py` moved to the new config names; the v10 README placeholder (an
unmodified copy of the old v9 README) re-synced with upstream's v9 README; the untracked
`libLlirSched_linked.so` rebuilt against v2.2's `libtriton`.

## 2. Baseline (rebased branch, before bias) against upstream

Two rounds, [results/baseline_r1](results/baseline_r1), [results/baseline_r2](results/baseline_r2):
a16w16 FP16 K=8192 every published pair plus v10-v13, a16w16 BF16 v9-v13, a8w8 K=16384, a4w4
K=32768. Round-to-round: up to 1.3% TFLOPS, up to ~0.9pp MFMA efficiency.

VGPRs and spills match upstream's published v2.2 tables everywhere, and MFMA efficiency (the
clock-independent number) is within noise: e.g. v9 `llir+amdgcnas` 448 / 0 / 98.3% (published
448 / 0 / 97.82%), a8w8 72.1 / 96.3 / 99.7% (72.2 / 96.3 / 99.7%), a4w4 v0 16 / 28 / 28 spills and
82.8% (16 / 28 / 28, 82.13%), v6 `base` 241 spills. Absolute TFLOPS on GPU 3 sit ~17% below the
published ones (a slower-clocking part), so they are only compared against this GPU.

## 3. The branch's amdgcnas change: a correctness bug, fixed

The branch changes `plugins/amdgcnas/amdgcnas_ext.py` (upstream does not). Compiling v7-v9,
a8w8 and a4w4 under `llir` and `llir+amdgcnas` from `origin/main` and from the branch:

- `llir`: identical everywhere (amdgcnas is not involved).
- `llir+amdgcnas`: v8 and a4w4 v1 differed only in which VGPRs the LICM-hoisted LDS addresses
  land in; **a4w4 v0 produced wrong results** (7 of 8 K values differ from torch, max diff
  632-3536), while `origin/main` passes.

Cause: the branch's new rule keeps a hoistable def in the loop when a user after the loop would
read the renamed register (`v_add_u32 v59, 0, v58`), but its in-loop user
(`v_add_u32 v61, 0x217a0, v59`) was still hoisted into the prologue, where it reads `v59` before
the loop writes it. `hoist_loop_invariants` decided every invariant independently. Fix: visit
the invariants in program order and keep any whose operands include a def that stays in the loop
(checked before `can_hoist`, which renames as a side effect).

After the fix ([results/asm/fixed_amdgcnas](results/asm/fixed_amdgcnas) against
[results/asm/main_amdgcnas](results/asm/main_amdgcnas)): a4w4 v0 correct at all K; every kernel
identical to `origin/main` up to VGPR renaming except a4w4 v0, where the copy and its user (two
VALU instructions) stay in the loop. Upstream's `main` rows against the branch's, two rounds each
(TFLOPS / MFMA eff.), measured before the fix, with the renaming-only differences:

| Row | origin/main | branch |
|---|---|---|
| a16w16 v8 `llir+amdgcnas` | 1325-1328 / 98.6-98.9% | 1324-1327 / 98.4-98.8% |
| a4w4 v0 `llir+amdgcnas` | 4604-4605 / 82.5-82.6% | 4593-4600 / 82.8-82.9% (then wrong results) |
| a4w4 v1 `llir+amdgcnas` | 4833-4835 / 93.0% | 4818-4820 / 93.0-93.1% |

With the fix, a4w4 v0 `llir+amdgcnas` is correct but gives up a little: 4584 TFLOPS / 81.64% MFMA
efficiency against `origin/main`'s 4604 / 82.5% (-0.4%, about -1pp), from the two instructions
that stay in the loop. Upstream's hoist of that copy is correct for this kernel; the branch's
"users after the loop" rule is conservative, so it costs a4w4 v0 this much and nothing elsewhere.

## 4. Bias

`init_acc` in `kernels/gemm/utils/common.py` starts the accumulators from the bias (see the
a16w16 README). Checks:

- **No-bias code unchanged:** v9-v13, FP16 and BF16, K=8192, under `base`, `llir` and
  `llir+amdgcnas`: 30 of 30 `.amdgcn` identical to the baseline
  ([results/asm/baseline_nobias](results/asm/baseline_nobias) against
  [results/asm/bias_nobias](results/asm/bias_nobias), both with the pre-fix amdgcnas).
- **Correctness** ([results/correctness.txt](results/correctness.txt), with the amdgcnas fix):
  77 of 77 runs pass: a16w16 v0-v13 FP16/BF16 over bench.py's K sweep under their published
  configs; v10-v13 also at 8192x8192 (4 tiles per program) and 4608x4096 (uneven); `--bias` for
  v9-v13 under all three configs at all those shapes; a8w8, a4w4 v0/v1, inter_wave a16w16 / a8w8 /
  a4w4 v0-v2, attention fmha_v3 / fmha_v4.
- **Spills with bias:** none, in any version, dtype or config ([results/bias.md](results/bias.md)).
  Bias adds 0-12 VGPRs.
- **Cost at 4096x4096x8192:** -0.9% to +0.2% TFLOPS, two rounds each
  ([results/bias.md](results/bias.md)).
- **Few-tile / short-K shapes** (BF16, `llir+amdgcnas`, two rounds,
  [results/fewtile](results/fewtile)): bias cost 0.4-2.5%, round-to-round noise ~1.5%; v11/v12
  are not worse than v10/v13 (4096x8192x1024: v10 -0.9%, v11 -2.0%, v12 -1.8%, v13 -2.0%), and
  match what AITER's epilogue add cost them there (v11 -2.2%, v12 -1.5%). So all five keep the
  accumulator-init bias rather than an epilogue add for v11/v12.

## 5. Before / after

[results/before_after.md](results/before_after.md): baseline (two rounds) against the final tree
(two rounds, [results/after_r1](results/after_r1), [results/after_r2](results/after_r2)), every
row of the matrix. VGPRs and spills are unchanged in every row. TFLOPS change within ±0.8%
everywhere except `base` v3 (-2.0%) and v4 (+1.0%); neither kernel nor `base` (no amdgcnas) is
touched by this change, and baseline round-to-round was already 1.3%, so both are noise. The one
real change is a4w4 v0 `llir+amdgcnas` from the amdgcnas fix (section 3): -0.3% TFLOPS, MFMA
efficiency 82.84% -> 81.64%.
