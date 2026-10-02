#!/usr/bin/env python3
"""VGPR / AGPR / spill counts of the gfx950 a16w16 Gluon kernels, with and without bias.

    python check_spills.py                      # default matrix, writes results/<tag>.md
    python check_spills.py --tag opt_a --kernels v10 --stacks m1
    python check_spills.py --worker '<json case>'   # one compile, prints one JSON line

Every case compiles in its own subprocess with a fresh TRITON_CACHE_DIR, because the
Triton pin and the LLVM pass plugin are fixed once per process. Stacks:

- m1: Triton gfx950-tutorial-v2.2, AITER's kernels as shipped (AITER's llirSched plugin,
  per-MFMA cd_regclass="a" pins, no fixed AGPR/VGPR split).
- m2: m1 plus llvm_fn_attrs amdgpu-agpr-alloc=256 (the tutorial's fixed split). v2.2 can
  only set LLVM flags to true, so amdgpu-mfma-vgpr-form=0 has no equivalent here.
- m3: Triton gfx950-tutorial-v2.1 with the tutorial's llir+force-agpr stack
  (TRITON_FORCE_MFMA_AGPR=1: amdgpu-agpr-alloc=256 + amdgpu-mfma-vgpr-form=0). AITER's
  kernel source runs with cd_regclass dropped (v2.1 has no such argument); kernel
  "tut_v10" is the tutorial's own v10 with an AITER-style bias epilogue instead.
- m3a: m3 plus the amdgcnas post-assembly peephole.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
REPOS = os.path.dirname(REPO)
AITER = os.environ.get("AITER_DIR", os.path.join(REPOS, "aiter"))
TRITON_V21 = os.path.join(REPOS, "triton_gfx950-tutorial-v2.1", "python")
TRITON_V22 = os.path.join(REPOS, "triton_gfx950-tutorial-v2.2", "python")
SHIM = os.path.join(REPO, "scripts", "triton_deepbind_shim")
LLIR_PLUGIN = os.path.join(REPO, "plugins", "llir_scheduler", "libLlirSched.so")
AMDGCNAS = os.path.join(REPO, "plugins", "amdgcnas")

KERNEL_TYPES = {
    "v9": "compute_bound",
    "v10": "compute_bound_v10",
    "v11": "compute_bound_v11",
    "v12": "compute_bound_v12",
    "v13": "compute_bound_v13",
}
STACKS = ("m1", "m2", "m3", "m3a")


def stack_env(stack):
    env = dict(os.environ)
    for k in (
        "LLVM_PASS_PLUGIN_PATH",
        "LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE",
        "TRITON_FORCE_MFMA_AGPR",
        "TRITON_AMDGCNAS_PLUGIN",
    ):
        env.pop(k, None)
    if stack in ("m1", "m2"):
        env["PYTHONPATH"] = os.pathsep.join([TRITON_V22, AITER, HERE])
    else:
        env["PYTHONPATH"] = os.pathsep.join([SHIM, TRITON_V21, AITER, HERE, AMDGCNAS])
        # AITER would otherwise point LLVM_PASS_PLUGIN_PATH at its own build; the
        # tutorial's plugin schedules every kernel, no opt-in attribute needed.
        env["AITER_LLIR_SCHED"] = "0"
        env["LLVM_PASS_PLUGIN_PATH"] = LLIR_PLUGIN
        env["LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE"] = "1"
        env["TRITON_FORCE_MFMA_AGPR"] = "1"
        if stack == "m3a":
            env["TRITON_AMDGCNAS_PLUGIN"] = "1"
    return env


# ---------------------------------------------------------------------------
# .amdgcn parsing
# ---------------------------------------------------------------------------

_META = {
    "vgpr_count": r"^\s*\.vgpr_count:\s*(\d+)",
    "agpr_count": r"^\s*-?\s*\.agpr_count:\s*(\d+)",
    "vgpr_spill": r"^\s*\.vgpr_spill_count:\s*(\d+)",
    "sgpr_spill": r"^\s*\.sgpr_spill_count:\s*(\d+)",
    "arch_vgprs": r"^\s*;\s*NumVgprs:\s*(\d+)",
    "scratch": r"^\s*;\s*ScratchSize:\s*(\d+)",
}
_LABEL = re.compile(r"^(\.LBB\d+_\d+):")
_BRANCH = re.compile(r"^\s*s_(?:c)?branch\w*\s+(\.LBB\d+_\d+)")
_MFMA = re.compile(r"^\s*v_mfma")
_SPILL = re.compile(r"^\s*scratch_(store|load)_")


def parse_amdgcn(path):
    with open(path) as f:
        lines = f.read().splitlines()
    out = {}
    for key, pat in _META.items():
        rx = re.compile(pat)
        for line in lines:
            m = rx.search(line)
            if m:
                out[key] = int(m.group(1))
                break

    # Loops are backward branches; the K loop is the smallest one holding >= 16 MFMAs,
    # the tile loop the largest one.
    labels = {}
    loops = []
    for i, line in enumerate(lines):
        m = _LABEL.match(line)
        if m:
            labels[m.group(1)] = i
        m = _BRANCH.match(line)
        if m and m.group(1) in labels:
            loops.append((labels[m.group(1)], i))
    mfma_lines = [i for i, l in enumerate(lines) if _MFMA.match(l)]

    def n_mfma(lo, hi):
        return sum(lo <= i <= hi for i in mfma_lines)

    mfma_loops = [lp for lp in loops if n_mfma(*lp) >= 16]
    kloop = min(mfma_loops, key=lambda lp: lp[1] - lp[0]) if mfma_loops else None
    tloop = max(loops, key=lambda lp: lp[1] - lp[0]) if loops else None

    spills = {"kloop": 0, "tile_loop": 0, "outside": 0}
    for i, line in enumerate(lines):
        if not _SPILL.match(line):
            continue
        if kloop and kloop[0] <= i <= kloop[1]:
            spills["kloop"] += 1
        elif tloop and tloop != kloop and tloop[0] <= i <= tloop[1]:
            spills["tile_loop"] += 1
        else:
            spills["outside"] += 1
    out["spill_ops"] = spills
    out["kloop_mfma"] = n_mfma(*kloop) if kloop else 0
    return out


# ---------------------------------------------------------------------------
# worker: one compile
# ---------------------------------------------------------------------------


def _strip_cd_regclass():
    """v2.1 has no mfma(cd_regclass=...); drop the argument so AITER's source compiles."""
    import inspect

    from triton.experimental.gluon.language._core import builtin
    from triton.experimental.gluon.language.amd import cdna3

    if "cd_regclass" in inspect.signature(cdna3.mfma).parameters:
        return
    orig = cdna3.mfma

    @builtin
    def mfma(a, b, acc, cd_regclass=None, _semantic=None):
        return orig(a, b, acc, _semantic=_semantic)

    cdna3.mfma = mfma


def _add_fn_attr(name, value):
    from aiter.ops.triton import llir_sched

    orig = llir_sched.compile_options

    def compile_options():
        opts = dict(orig())
        opts["llvm_fn_attrs"] = tuple(opts.get("llvm_fn_attrs", ())) + ((name, value),)
        return opts

    llir_sched.compile_options = compile_options


def _bare_aiter_package():
    """Register `aiter` without running aiter/__init__.py, which pulls in the whole library
    (flydsl, JIT modules, ...). The Gluon GEMM modules only need aiter.ops.triton."""
    import types

    pkg = types.ModuleType("aiter")
    pkg.__path__ = [os.path.join(AITER, "aiter")]
    sys.modules["aiter"] = pkg


def _deepbind_triton_and_aiter_plugin():
    """What scripts/triton_deepbind_shim does, for AITER's plugin, which AITER only puts in
    LLVM_PASS_PLUGIN_PATH at launch time (too late for the shim to see it)."""
    import ctypes

    flags = sys.getdlopenflags()
    sys.setdlopenflags(os.RTLD_NOW | os.RTLD_DEEPBIND)
    try:
        import triton  # noqa: F401
    finally:
        sys.setdlopenflags(flags)
    from aiter.ops.triton import llir_sched

    so = llir_sched.plugin_path()
    if so is None:
        raise RuntimeError("AITER llirSched plugin unavailable")
    ctypes.CDLL(so, mode=os.RTLD_NOW | os.RTLD_DEEPBIND | os.RTLD_LOCAL)


def worker(case):
    stack, kernel = case["stack"], case["kernel"]
    _bare_aiter_package()
    if stack in ("m1", "m2"):
        _deepbind_triton_and_aiter_plugin()
    import torch

    M, N, K = case["M"], case["N"], case["K"]
    out_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[case["out"]]

    if stack in ("m3", "m3a"):
        if os.environ.get("TRITON_AMDGCNAS_PLUGIN"):
            import amdgcnas_plugin
            from triton import knobs

            knobs.runtime.add_stages_inspection_hook = amdgcnas_plugin.inspect_stages_hook
        if kernel != "tut_v10":
            _strip_cd_regclass()
            _add_fn_attr("amdgpu-agpr-alloc", "256")
    elif stack == "m2":
        _add_fn_attr("amdgpu-agpr-alloc", "256")

    torch.manual_seed(0)
    x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
    bias = torch.randn((N,), dtype=torch.bfloat16, device="cuda") if case["bias"] else None
    y = torch.empty((M, N), dtype=out_dtype, device="cuda")

    if kernel == "tut_v10":
        import tutorial_v10_bias

        tutorial_v10_bias.matmul(x, w.T, y, bias)
    elif kernel == "v9":
        from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.gemm_a16w16 import (
            gemm_a16w16_compute_bound,
        )

        gemm_a16w16_compute_bound(x, w, y, bias)
    else:
        from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.gemm_a16w16_persistent import (
            _PERSISTENT_KERNEL_MAP,
        )

        _PERSISTENT_KERNEL_MAP[KERNEL_TYPES[kernel]](x, w, y, bias)
    torch.cuda.synchronize()

    ref = torch.nn.functional.linear(x.float(), w.float(), None if bias is None else bias.float())
    err = (y.float() - ref).abs().max().item()
    tol = 1e-1 + 1e-2 * ref.abs().max().item()

    files = glob.glob(os.path.join(os.environ["TRITON_CACHE_DIR"], "*", "*.amdgcn"))
    files = [f for f in files if "gemm" in os.path.basename(f) or "v10_persistant" in f]
    if len(files) != 1:
        raise RuntimeError(f"expected one kernel .amdgcn, got {files}")
    res = parse_amdgcn(files[0])
    res.update(case, ok=err <= tol, max_err=err, amdgcn=files[0])
    print("RESULT " + json.dumps(res), flush=True)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def run_case(case, keep):
    cache = tempfile.mkdtemp(prefix=f"spills_{case['stack']}_{case['kernel']}_", dir=keep)
    env = stack_env(case["stack"])
    env["TRITON_CACHE_DIR"] = cache
    env["TRITON_ALWAYS_COMPILE"] = "1"
    p = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--worker", json.dumps(case)],
        env=env,
        cwd=HERE,
        capture_output=True,
        text=True,
    )
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT ") :])
    tail = "\n".join((p.stdout + p.stderr).strip().splitlines()[-15:])
    return dict(case, error=f"exit {p.returncode}\n{tail}")


def fmt_row(r):
    name = f"{r['kernel']} | {r['stack']} | {'on' if r['bias'] else 'off'} | {r['out']} | {r['K']}"
    if "error" in r:
        return f"| {name} | error | | | | | | |"
    s = r["spill_ops"]
    where = ", ".join(f"{k} {v}" for k, v in s.items() if v) or "-"
    return (
        f"| {name} | {r.get('arch_vgprs', '?')} | {r.get('agpr_count', '?')} | "
        f"{r.get('vgpr_count', '?')} | {r.get('vgpr_spill', '?')} | {r.get('scratch', '?')} | "
        f"{where} | {'yes' if r['ok'] else 'NO (%.3g)' % r['max_err']} |"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker")
    ap.add_argument("--kernels", nargs="+", default=list(KERNEL_TYPES))
    ap.add_argument("--stacks", nargs="+", default=["m1"], choices=STACKS)
    ap.add_argument("--bias", nargs="+", default=["off", "on"], choices=["off", "on"])
    ap.add_argument("--out", nargs="+", default=["bf16", "fp32"], choices=["bf16", "fp32"])
    ap.add_argument("--K", nargs="+", type=int, default=[8192, 1024])
    ap.add_argument("--M", type=int, default=4096)
    ap.add_argument("--N", type=int, default=4096)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--keep", default=os.path.join(HERE, "tmp"), help="cache root")
    args = ap.parse_args()

    if args.worker:
        worker(json.loads(args.worker))
        return

    os.makedirs(args.keep, exist_ok=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    rows = []
    for stack in args.stacks:
        for kernel in args.kernels:
            for K in args.K:
                for out in args.out:
                    for b in args.bias:
                        case = dict(stack=stack, kernel=kernel, bias=b == "on", out=out,
                                    M=args.M, N=args.N, K=K)
                        r = run_case(case, args.keep)
                        rows.append(r)
                        print(fmt_row(r), flush=True)
                        if "error" in r:
                            print(r["error"], flush=True)

    header = (
        "| Kernel | Stack | Bias | Out | K | Arch VGPRs | AGPRs | .vgpr_count | Spills | "
        "Scratch B | Spill ops (where) | Correct |\n"
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|"
    )
    md = os.path.join(HERE, "results", f"{args.tag}.md")
    with open(md, "w") as f:
        f.write(f"M={args.M} N={args.N}, bf16 inputs.\n\n{header}\n")
        f.write("\n".join(fmt_row(r) for r in rows) + "\n")
    with open(os.path.join(HERE, "results", f"{args.tag}.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print(f"wrote {md}")


if __name__ == "__main__":
    main()
