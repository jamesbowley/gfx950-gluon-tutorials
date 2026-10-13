# amdgcnas — out-of-tree post-assembly peephole

`amdgcnas` is a pure-Python peephole over the final `amdgcn` assembly text: it
applies LICM and interleaves MFMA with scalar instructions to compress the
remaining non-MFMA gaps in the GEMM hot loop. It attaches at the `amdgcn` compile
stage through a stages-inspection hook, so it needs no `.so`, no LLVM symbols, and
no Triton rebuild — it works on stock Triton.

## Files
- `amdgcnas_ext.py` — the peephole itself: the LICM / save-restore /
  loop-scheduling passes, with the VGPR-count directives computed from the kernel's
  real register usage. Its entry point `amdgcn_as(text, verbose=False) -> text` is a
  pure-Python transform on the assembly string (stdlib only — no LLVM, no libtriton).
- `amdgcnas_plugin.py` — a `knobs.runtime.add_stages_inspection_hook` that wraps
  the `amdgcn` compile stage: run in-tree codegen, then `amdgcn_as` on the result.

## Use
`bench.py` installs the hook when `TRITON_AMDGCNAS_PLUGIN=1` (set it to `2` for
verbose peephole logging). The peephole runs on top of the LLIR scheduler; to
reproduce the full `llir+amdgcnas` stack:

```bash
LLVM_PASS_PLUGIN_PATH=.../plugins/llir_scheduler/libLlirSched.so \
TRITON_AMDGCNAS_PLUGIN=1 \
    python bench.py --version 8 --K 8192 --dtype fp16
```

The kernels it is used on (a16w16 v7 and later, a8w8, a4w4) keep their MFMA accumulators in
AGPRs with `cd_regclass="a"`, and the peephole assumes that, so it is only used on these pinned
kernels.

The loop passes reorder instructions, and keep these dependences (found on v9_tile's small tiles,
[WORKLOG §14.3](../../kernels/gemm/intra_wave/a16w16/v9_tile/WORKLOG.md)):

- **MFMA interleave:** an instruction crosses only MFMAs it shares no register with, and an `s_waitcnt`
  never moves below an MFMA.
- **lgkmcnt / barrier rotation:** a merge never crosses an lgkm op, and a rotation never crosses an LDS access.
- **LICM:** a use is invariant only through its reaching definition.

`scripts/run_perf_table.py` wires these into the tutorial's configs; see
[gemm/README §2.1](../../kernels/gemm/intra_wave/README.md#21-triton-build-and-the-out-of-tree-plugins)
for the component stack.
