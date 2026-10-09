#!/usr/bin/env python3
"""Accumulator-copy statistics of an a16w16 kernel build.

    python compile_stats.py <kernel> [<kernel> ...] [--K 1024] [--plugins] [--bias] [--env K=V ...]
                            [--keep DIR]
    python compile_stats.py --asm file.s [...]        # statistics of existing assembly

<kernel> is a version number (13 -> kernels/.../v13_*/matmul_kernel.py) or a path to a
matmul_kernel.py variant. Each build runs in its own subprocess with a fresh TRITON_CACHE_DIR
(the plugin environment is fixed once per process), at 4096x4096xK bf16. --keep copies the
.llir and .amdgcn there as <name>.llir / <name>.amdgcn for the llc experiments.

Copies are split by position relative to the K loop, the innermost loop that contains MFMAs:
before it (peeled steps and loop entry), inside it, and after it (the drain and epilogue).
"""

import argparse
import glob
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
A16 = os.path.join(REPO, "kernels", "gemm", "intra_wave", "a16w16")
UTILS = os.path.join(REPO, "kernels", "gemm", "utils")
PLUGIN = os.path.join(REPO, "plugins", "llir_scheduler", "libLlirSched.so")


def kernel_path(spec):
    if os.path.exists(spec):
        return os.path.abspath(spec)
    d = glob.glob(os.path.join(A16, f"v{int(spec)}_*", "matmul_kernel.py"))
    d = [x for x in d if "rocprof_trace" not in x]
    assert len(d) == 1, d
    return d[0]


def asm_stats(text):
    lines = [l.split(";")[0].strip() for l in text.splitlines()]
    labels = {l[:-1]: i for i, l in enumerate(lines) if re.match(r"^\.LBB\d+_\d+:$", l)}
    is_mfma = [l.startswith("v_mfma") for l in lines]
    loops = []
    for i, l in enumerate(lines):
        m = re.match(r"^s_c?branch\w*\s+(\.LBB\d+_\d+)$", l)
        if m and m.group(1) in labels and labels[m.group(1)] < i:
            lo = labels[m.group(1)]
            n = sum(is_mfma[lo:i])
            if n:
                loops.append((n, lo, i))
    lo, hi = (min(loops)[1], min(loops)[2]) if loops else (0, -1)
    ops = {"accvgpr_mov": "v_accvgpr_mov_b32", "accvgpr_read": "v_accvgpr_read_b32",
           "accvgpr_write": "v_accvgpr_write_b32", "s_nop": "s_nop", "mfma": "v_mfma"}
    out = {k: sum(l.startswith(p) for l in lines) for k, p in ops.items()}
    mov = [l.startswith("v_accvgpr_mov_b32") for l in lines]
    out["mov_before_loop"] = sum(mov[:lo])
    out["mov_in_loop"] = sum(mov[lo:hi + 1])
    out["mov_after_loop"] = sum(mov[hi + 1:])
    out["mfma_in_loop"] = sum(is_mfma[lo:hi + 1])
    for key in ("vgpr_count", "agpr_count", "sgpr_count", "vgpr_spill_count", "sgpr_spill_count"):
        m = re.search(rf"\.{key}:\s+(\d+)", text)
        out[key] = int(m.group(1)) if m else None
    return out


def worker(case):
    sys.path[:0] = [UTILS, A16]
    import triton  # noqa: F401  (before torch: libtriton must bind its own LLVM first)
    import torch
    for k, v in case["env"].items():
        os.environ[k] = v
    spec = importlib.util.spec_from_file_location("variant_kernel", case["path"])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    M = N = 4096
    K = case["K"]
    a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16).T
    kw = {"bias": torch.randn((N,), device="cuda", dtype=torch.bfloat16)} if case["bias"] else {}
    c = mod.matmul(a, b, **kw)
    torch.cuda.synchronize()
    ref = a.float() @ b.float()
    if case["bias"]:
        ref += kw["bias"].float()
    err = (c.float() - ref).abs().max().item()
    print(json.dumps({"max_abs_err": err}))


def build(path, K, plugins, bias, env, keep, name):
    cache = tempfile.mkdtemp(prefix="pac_cache_")
    e = {**os.environ, "TRITON_CACHE_DIR": cache, **env}
    e.pop("LLVM_PASS_PLUGIN_PATH", None)
    e.pop("TRITON_AMDGCNAS_PLUGIN", None)
    if plugins:
        e["LLVM_PASS_PLUGIN_PATH"] = PLUGIN
        e["TRITON_AMDGCNAS_PLUGIN"] = "1"
    case = {"path": path, "K": K, "bias": bias, "env": env}
    pre = []
    if plugins:
        pre = ["-c", "import os,sys; sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL); "
               f"sys.path.insert(0, {os.path.join(REPO, 'plugins', 'amdgcnas')!r}); "
               "import triton, amdgcnas_plugin; from triton import knobs; "
               "knobs.runtime.add_stages_inspection_hook = amdgcnas_plugin.inspect_stages_hook; "
               f"sys.argv = ['x', '--worker', {json.dumps(case)!r}]; __file__ = {os.path.abspath(__file__)!r}; "
               f"exec(open({os.path.abspath(__file__)!r}).read())"]
    cmd = [sys.executable, *(pre or [os.path.abspath(__file__), "--worker", json.dumps(case)])]
    p = subprocess.run(cmd, env=e, capture_output=True, text=True)
    if p.returncode:
        sys.stderr.write(f"{name}: worker exit {p.returncode}\n{p.stderr[-3000:]}")
        return None
    err = json.loads(p.stdout.strip().splitlines()[-1])["max_abs_err"]
    asm = [f for f in glob.glob(os.path.join(cache, "*", "*.amdgcn")) if "hip_utils" not in f]
    asm = max(asm, key=os.path.getsize)
    llir = asm[:-len(".amdgcn")] + ".llir"
    st = asm_stats(open(asm).read())
    st["max_abs_err"] = err
    if keep:
        os.makedirs(keep, exist_ok=True)
        shutil.copy(asm, os.path.join(keep, name + ".amdgcn"))
        shutil.copy(llir, os.path.join(keep, name + ".llir"))
    shutil.rmtree(cache, ignore_errors=True)
    return st


COLS = ["accvgpr_mov", "mov_before_loop", "mov_in_loop", "mov_after_loop", "s_nop", "accvgpr_read",
        "accvgpr_write", "mfma", "mfma_in_loop", "vgpr_count", "agpr_count", "vgpr_spill_count"]


def print_row(name, st):
    print(f"| {name} | " + " | ".join(str(st.get(c)) for c in COLS) +
          (f" | {st['max_abs_err']:.3g}" if "max_abs_err" in st else " | -") + " |", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kernels", nargs="*")
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--plugins", action="store_true")
    ap.add_argument("--bias", action="store_true")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--keep")
    ap.add_argument("--asm", nargs="+")
    ap.add_argument("--worker")
    a = ap.parse_args()
    if a.worker:
        return worker(json.loads(a.worker))
    print("| build | " + " | ".join(COLS) + " | max abs err |")
    print("|---" * (len(COLS) + 2) + "|")
    for f in a.asm or []:
        print_row(os.path.basename(f), asm_stats(open(f).read()))
    env = dict(x.split("=", 1) for x in a.env)
    for k in a.kernels:
        path = kernel_path(k)
        name = (f"v{k}" if k.isdigit() else os.path.splitext(os.path.basename(k))[0]) + \
            ("_plugins" if a.plugins else "") + ("_bias" if a.bias else "") + (f"_K{a.K}" if a.K != 1024 else "")
        st = build(path, a.K, a.plugins, a.bias, env, a.keep, name)
        if st:
            print_row(name, st)


if __name__ == "__main__":
    main()
