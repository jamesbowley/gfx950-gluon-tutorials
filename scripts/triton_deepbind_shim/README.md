# triton_deepbind_shim

Local workaround for running this tutorial's tooling on a **pip-installed ROCm**
(`rocm-sdk-core` / `rocm-sdk-devel` wheels) rather than a `/opt/rocm` install.
Not needed on a conventional ROCm install.

## Usage

```bash
export PYTHONPATH=$(git rev-parse --show-toplevel)/scripts/triton_deepbind_shim:$PYTHONPATH
```

Then everything works unmodified — `bench.py`, `scripts/run_perf_table.py`,
`scripts/run_att.py`, the manual command lines in the kernel READMEs. The shim
is a no-op unless `rocprofv3` is in use or `LLVM_PASS_PLUGIN_PATH` is set, so
plain unprofiled runs are untouched.

## The problem

Two complete, different LLVMs end up in one process:

| | LLVM | how it gets loaded |
|---|---|---|
| `libtriton.so` | statically linked, pin `b010a18d`, exports all ~23k `llvm::` symbols (required by `TRITON_EXT_ENABLED=1` / default visibility) | `import triton` |
| `libLLVM.so.23.0git` | **shared**, also exports `llvm::` | `libamd_comgr.so.3`, which arrives via HIP (any `import torch`) and via `rocprofiler-sdk` (anything under `rocprofv3`) |

On a `/opt/rocm` install this never bites, because there `comgr` hides its LLVM.
The wheels ship it as a shared library instead, so whichever object sits earlier
in the global symbol scope captures the other's symbol references. Both pairings
break:

- **`rocprofv3` LD_PRELOADs `rocprofiler-sdk`**, so `libLLVM` is in the global
  scope before `libtriton` is dlopened. ~17k of `libtriton`'s references to its
  *own* LLVM bind to ROCm's copy, and `libtriton` dies in its static
  initializers — `SIGSEGV` in `__libc_realloc`, before a single kernel runs.
- **`LLVM_PASS_PLUGIN_PATH` makes `bench.py` load `libtriton` `RTLD_GLOBAL`**
  (so the LLIR plugin can resolve symbols). That inverts it: `libLLVM` loads
  later, its `cl::opt` registrations land in `libtriton`'s registry, and LLVM
  aborts with `Option 'default' already exists!`.

## The fix

Keep each LLVM bound to itself, using `RTLD_DEEPBIND` (an object's own
dependencies take precedence over the global scope):

1. Load `libtriton` with `RTLD_DEEPBIND` and **without** `RTLD_GLOBAL`, before
   the benchmark script runs. It prefers its own LLVM; `libLLVM`, loaded later
   and with `libtriton` kept out of the global scope, prefers its own. Because
   the shim imports `triton` first, `bench.py`'s own
   `sys.setdlopenflags(RTLD_GLOBAL)` is a no-op.
2. Give the LLIR pass plugin a `DT_NEEDED` for `libtriton` and pre-dlopen it
   with `RTLD_DEEPBIND`, so its 83 undefined `llvm::` symbols resolve to
   `libtriton` instead of ROCm's `libLLVM`.

Step 2 needs both halves. The stock `libLlirSched.so` deliberately links no
LLVM and has no `DT_NEEDED` for `libtriton`, so `DEEPBIND` has nothing to
prefer and it still binds to ROCm's `libLLVM` — at which point it hangs inside
the wrong `AnalysisManager<Module>::getResultImpl` instead of crashing.
`plugins/llir_scheduler/build_linked.sh` rebuilds it with
`-l:libtriton.so -Wl,-rpath,...` as `libLlirSched_linked.so`; the shim
transparently redirects `LLVM_PASS_PLUGIN_PATH` to that sibling when it exists,
so no script or README command needs changing.

Rebuild the plugin whenever Triton's LLVM pin moves (it is ABI-locked):

```bash
plugins/llir_scheduler/build_linked.sh
```

## Verified

Recorded on the `gfx950-tutorial-v2.1` pin, whose `llir+force-agpr+amdgcnas` config is
today's `llir+amdgcnas`. a16w16, the full documented config matrix, 4096x4096x8192 FP16,
`--rocprof`, against the values published in the kernel READMEs at the time:

| config / version | measured | published |
|---|---|---|
| base v0 | 525 TFLOPS, 450 VGPR, 0 spill | 525, 450, 0 |
| base v5 MFMA eff | 57.74% | 57.7% |
| llir v5 MFMA eff | 68.57% | 68.6% |
| llir v6 | 230 TFLOPS, **129 spills** | 228, **129 spills** |
| llir+force-agpr+amdgcnas v7 | 98.14% | 98.0% |
| llir+force-agpr+amdgcnas v8 | 98.04% | — |
| llir+force-agpr+amdgcnas v9 | 1415 TFLOPS, 97.91% | — |

The other kernel families, all on `llir+force-agpr+amdgcnas`:

| kernel | measured | published MFMA eff |
|---|---|---|
| a8w8, K=16384 | 3253 TFLOPS, 484 VGPR, 0 spill, 99.24% | 99.2% |
| a4w4 v1, K=32768 | 5190 TFLOPS, 500 VGPR, 0 spill, 93.67% | 93.7% |
| attention fmha_v4, B32 HQ8 HK8 D128 N8192 | 1224 TFLOPS, output matches torch | — |

Plus: plain unprofiled runs are unaffected (the shim no-ops), and the manual
`LLVM_PASS_PLUGIN_PATH=.../libLlirSched.so` command lines from the kernel
READMEs work via the redirect.

MFMA efficiency is the meaningful check, being clock-independent, and it lands
within 0.1pp of the published figures on every kernel. Absolute TFLOPS sit a
few percent below, which is expected: those were taken on GPU[7] of the
reference machine.
