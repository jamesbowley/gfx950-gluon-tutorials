# v31_streamk_aiter

v31 is v30's Stream-K kernel made ready for AITER: `matmul_kernel.py` is
self-contained (it imports only Triton, torch and AITER's `device_info` and
`graph_alloc`), its docstring is the parameter reference, and a config that names
only the tile gets a default Stream-K plan.

## What changed from v30

- **One file.** The topology query, the workspace (flags and partials), the config
  parsing and the reduce-scatter reader rule are inlined from v30's `hardware.py`,
  `runtime.py` and `selection.py`. The kernel module no longer refers to v9, the
  tutorial, the worklogs or any `.md` file.
- **Parameter reference.** The module docstring explains every config key with
  text diagrams of the schedules and endings.
- **AITER entry points.** `unsupported_reason(M, N, K, x, w, bias, activation,
  config)`, `gemm_a16w16_streamk(x, w, y, bias, config)` and `_KERNEL_MAP`, in the
  shape of the v9 port's on AITER's `llirSched_integration` branch, so either port
  (replace v9 or add a kernel type) is a few lines. They run inside AITER
  ([results/aiter_smoke.md](results/aiter_smoke.md)).
- **Default plan.** The tile is always supplied (as AITER supplies it: a tuned row's
  `kernelName` or the config JSON). The plan is either given too (a tuned config)
  or left out, and `streamk_plan` fills it in, the Stream-K counterpart of AITER's
  `compute_splitk_params`.

## Configs

```python
gemm_a16w16_streamk(x, w, y, config={"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64})   # default plan
gemm_a16w16_streamk(x, w, y, config={"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 4,
                                     "STREAMK_POLICY": "one_tile", "STREAMK_FIXUP": "rs",
                                     "SPLIT_N": 8, "STREAMK_EXCHANGE": "wt"})            # tuned
```

`complete_config` fills a tile-only config: GROUP_SIZE_M 4 and the default plan
with the `wt` exchange. The four plan keys (STREAMK_POLICY, STREAMK_FIXUP, SPLIT_N,
STREAMK_EXCHANGE) are all given or all absent; a partial set is rejected. The default
plan, with m = P // S programs per tail tile, pairs = ceil(K / (2 BLOCK_K)) and
cap = min(m, pairs):

```text
dp              if S == 0 or cap < 2
owner           if m <= 3:   n = sqrt(pairs / 1.5)     dp if pairs < 10
reduce-scatter  otherwise:   n = sqrt(pairs * m / 6)   dp if pairs * (1 - 1/n) < 6
```

n is clamped to cap and rounded to the nearest listed split. The constants
(`STREAMK_PLAN`) are the same for every tile spec and both regimes; how they were
fitted and tested is in
[experiments/streamk_v31_rule](../../../../../experiments/streamk_v31_rule/README.md)
(on 597 held-out AITER shapes at all 13 specs: 1.0024 geomean, 1.000 at the 90th
percentile of each spec's best).

**Compiles.** K, the tail tile count (an M dependence) and the split are constexprs:
every new K, and every new S on a Stream-K path, compiles once, about 1.3 s cold
(v9: 0.6 s), [results/aiter_smoke.md](results/aiter_smoke.md).

## Files

- `matmul_kernel.py`: the kernel and its launcher (the only file AITER needs).
- `selection.py`, `heuristics.py`, `hardware.py`: offline tooling for the checks
  and sweeps (exact-config resolver, v30's capped rule and auto tile, superseded for
  the plan by `streamk_plan`); the kernel does not import them. `test_selection.py`
  checks the resolver against the kernel's own launch constants, and `streamk_plan`
  against the experiment's formula on every recorded pair.
- `check_streamk.py`, `run_checks.sh`, `checks_all.sh`, `checks_default.sh`:
  correctness and synchronization checks (graph replay, negative controls, streams,
  persistent partials, and `--default`: tile-only configs through every branch of the
  default plan). `asm_check.py`: synchronization instructions.
- `dp_ab.py`: one-tile-per-program dp (a one-trip persistent loop) against v9
  (`v9_ref.py`): ISA counts and timing.
- `env.sh`: the v29 stack plus the AITER checkout on `PYTHONPATH`, HIP 3
  (rocm-smi GPU 1) by default. `run_guarded.sh LOG CMD...` runs a command under
  `experiments/streamk_v31_rule/gpu_guard.py`, which stops it as soon as another
  process uses the GPU.

## Checks

```bash
bash v31_streamk_aiter/run_guarded.sh v31_streamk_aiter/results/checks.log bash v31_streamk_aiter/checks_all.sh
bash v31_streamk_aiter/run_guarded.sh v31_streamk_aiter/results/checks_default.log bash v31_streamk_aiter/checks_default.sh
```

All pass ([results/checks.log](results/checks.log),
[results/checks_default.log](results/checks_default.log)); see [WORKLOG.md](WORKLOG.md).

## Known limits

- **fp16 with bias** is right, but AITER's test compares with torch's `F.linear`,
  which is off by up to 4.0 at K = 8192 (v9 fails the same test).
- **Short K on small tiles:** at K = 512 LLVM unrolls the whole K loop of the 64x64
  builds (K is a constexpr) and the MFMA schedule's barriers are dropped.
- **Against v9**, with at most one tile per program, v31's dp (a one-trip persistent
  loop) is 2.7% slower (geomean) on even-K single-wave shapes, up to 8.8% on the
  shortest kernels (K = 1024), and level on ragged K ([results/dp_timing.md](results/dp_timing.md)).
  A non-persistent dp path closed about half of that but cost a compile variant per
  single-wave dp and was still behind v9, so it was removed (WORKLOG section 7).
