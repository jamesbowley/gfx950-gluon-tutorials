##############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
##############################################################################
"""Regression check: correctness, TFLOPS and register / spill counts per version and shape.

    HIP_VISIBLE_DEVICES=2 python regress.py                       # v13 vs v14, fixed shapes
    HIP_VISIBLE_DEVICES=2 python regress.py --extra 4352x4096x8192 --bias
    HIP_VISIBLE_DEVICES=2 python regress.py --versions 13 14 15
    HIP_VISIBLE_DEVICES=2 python regress.py --versions 13 9 14 15 --streamk   # partial last waves
    HIP_VISIBLE_DEVICES=2 python regress.py --versions 13 17 19 --leftover-sweep  # S = 2 .. 255

Needs v10+'s published stack (Triton v2.2 + llir + amdgcnas); the venv's default Triton is
v2.1, which has no mfma(cd_regclass=...):

    export PYTHONPATH=<repo>/scripts/triton_deepbind_shim:<triton_gfx950-tutorial-v2.2>/python
    export LLVM_PASS_PLUGIN_PATH=<repo>/plugins/llir_scheduler/libLlirSched.so
    export TRITON_AMDGCNAS_PLUGIN=1

Prints one markdown table; TFLOPS deltas are against the first version, per shape. Every
(version, shape) runs in its own subprocess with a fresh TRITON_CACHE_DIR, so the one
.amdgcn it produces is that shape's specialization.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
REGRESSION_SHAPES = [
    (4096, 4096, 8192),
    (8192, 8192, 8192),
    (4096, 8192, 4096),
    (4096, 106496, 16384),
]
# Shapes with a partial last wave (256x256 tiles, 256 programs), one property of the stream-K
# tail each; see v15_streamk_onetile/WORKLOG.md.
STREAMK_SHAPES = [
    (4352, 4096, 8192),   # 16 SK tiles, 16-way split, 15 peers
    (4352, 4352, 8192),   # 33 SK tiles, drifting seams, crossing programs, 2-k-step segments
    (4352, 4096, 4096),   # as 4352x4096x8192 at half the k-steps per program
    (4352, 4096, 16384),  # ... and at twice
    (3328, 5120, 8192),   # 4 SK tiles: every segment 2 k-steps, 63 peers
    (4096, 6144, 8192),   # 128 SK tiles, 2-way split, 1 peer
    (3840, 4096, 8192),   # 240 tiles, no full wave
    (8192, 8448, 8192),   # 4 full waves + 32 SK tiles
    (8192, 7936, 8192),   # 3 full waves + 224 SK tiles
]
# Leftover tiles S from 2 to 255 (256x256 tiles, 256 programs), each with at least one full
# wave unless noted; see v19_streamk_two_tile_reversed/WORKLOG.md. Under two-tile the
# programs of an XCD read k-slices up to d = iters_per_tile * S / 256 k-steps apart.
LEFTOVER_SWEEP_SHAPES = [
    (1536, 11008, 8192),   # S = 2
    (3328, 5120, 8192),    # S = 4
    (4352, 4096, 8192),    # S = 16
    (8192, 8448, 8192),    # S = 32, 4 full waves
    (4096, 5120, 8192),    # S = 64
    (4096, 6144, 8192),    # S = 128
    (4096, 7168, 4096),    # S = 192, K = 4096
    (4096, 7168, 8192),    # S = 192
    (4096, 7168, 16384),   # S = 192, K = 16384
    (9472, 10240, 8192),   # S = 200, 5 full waves
    (8192, 7936, 8192),    # S = 224, 3 full waves
    (4096, 7936, 8192),    # S = 240
    (4608, 7168, 8192),    # S = 248
    (1792, 18688, 8192),   # S = 255
    (3840, 4096, 8192),    # 240 tiles, no full wave
]
RESULT_TAG = "REGRESS_RESULT "


def parse_shape(s):
    m, n, k = (int(x) for x in s.lower().split("x"))
    return m, n, k


def worker(case):
    # bench sets up the LLVM plugin / amdgcnas hook and the path to common.py.
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(REPO, "experiments", "aiter_bias_spills"))
    import importlib

    import bench
    import torch
    import triton
    from check_spills import parse_amdgcn

    version_dir = bench.VERSION_MAP[case["version"]]
    matmul = importlib.import_module(f"{version_dir}.matmul_kernel").matmul
    M, N, K = case["shape"]
    dtype = bench.name_to_torch_type[case["dtype"]]
    device = bench.DEVICE

    a = torch.rand((M, K), device=device, dtype=dtype) - 0.5
    b = torch.rand((N, K), device=device, dtype=dtype).T - 0.5
    kwargs = {}
    if case["bias"]:
        kwargs["bias"] = torch.rand((N,), device=device, dtype=dtype) - 0.5
        ref = (torch.matmul(a.float(), b.float()) + kwargs["bias"].float()).to(dtype)
    else:
        ref = torch.matmul(a, b)
    out = matmul(a, b, **kwargs)
    # Same tolerances as bench.test_correctness.
    rtol = 1e-2 if case["dtype"] == "bf16" else 0
    correct = bool(torch.allclose(out, ref, atol=1e-1, rtol=rtol))
    del ref, out

    ms = triton.testing.do_bench(lambda: matmul(a, b, **kwargs), quantiles=[0.5])
    ms = ms[0] if isinstance(ms, (list, tuple)) else ms
    tflops = 2 * M * N * K * 1e-12 / (ms * 1e-3)

    files = glob.glob(os.path.join(os.environ["TRITON_CACHE_DIR"], "**", "*.amdgcn"), recursive=True)
    named = [f for f in files if os.path.basename(f) == f"{version_dir}.amdgcn"]
    regs = parse_amdgcn(named[0]) if len(named) == 1 else {"error": f"{len(named)} .amdgcn files"}

    print(RESULT_TAG + json.dumps({"correct": correct, "tflops": tflops, **regs}), flush=True)


def run_case(case):
    env = dict(os.environ)
    env["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="regress_")
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--worker", json.dumps(case)],
        env=env,
        cwd=HERE,
        capture_output=True,
        text=True,
    )
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_TAG):
            return json.loads(line[len(RESULT_TAG):])
    tail = (proc.stderr or proc.stdout).strip().splitlines()[-5:]
    return {"error": " / ".join(tail) or f"exit {proc.returncode}"}


def fmt_row(shape, version, r, base_tflops):
    shape_s = "x".join(str(x) for x in shape)
    if "tflops" not in r:
        return f"| {shape_s} | v{version} | error: {r.get('error', '?')} | | | | | | | |"
    delta = "" if base_tflops is None else f"{100 * (r['tflops'] / base_tflops - 1):+.1f}%"
    s = r.get("spill_ops", {})
    spill_ops = f"{s.get('kloop', '?')}/{s.get('tile_loop', '?')}/{s.get('outside', '?')}"
    return (
        f"| {shape_s} | v{version} | {'yes' if r['correct'] else 'NO'} | {r['tflops']:.1f} | "
        f"{delta} | {r.get('vgpr_count', '?')} | {r.get('agpr_count', '?')} | "
        f"{r.get('vgpr_spill', '?')} | {r.get('sgpr_spill', '?')} | {spill_ops} |"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--versions", type=int, nargs="+", default=[13, 14])
    parser.add_argument("--extra", nargs="*", default=[], help="Extra MxNxK shapes")
    parser.add_argument("--only-extra", action="store_true", help="Skip the fixed regression shapes")
    parser.add_argument(
        "--streamk", action="store_true", help="Run STREAMK_SHAPES instead of the regression shapes"
    )
    parser.add_argument(
        "--leftover-sweep", action="store_true",
        help="Run LEFTOVER_SWEEP_SHAPES instead of the regression shapes",
    )
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        worker(json.loads(args.worker))
        return

    if args.only_extra:
        shapes = []
    elif args.leftover_sweep:
        shapes = list(LEFTOVER_SWEEP_SHAPES)
    elif args.streamk:
        shapes = list(STREAMK_SHAPES)
    else:
        shapes = list(REGRESSION_SHAPES)
    shapes += [parse_shape(s) for s in args.extra]
    print(
        f"dtype={args.dtype} bias={args.bias} "
        f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', 'unset')}\n"
    )
    print("| shape (MxNxK) | version | correct | TFLOPS | vs first | VGPR | AGPR | VGPR spill | SGPR spill | scratch ops (kloop/tile/other) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for shape in shapes:
        base = None
        for i, version in enumerate(args.versions):
            case = {"version": version, "shape": shape, "dtype": args.dtype, "bias": args.bias}
            r = run_case(case)
            print(fmt_row(shape, version, r, base), flush=True)
            if i == 0 and "tflops" in r:
                base = r["tflops"]


if __name__ == "__main__":
    main()
