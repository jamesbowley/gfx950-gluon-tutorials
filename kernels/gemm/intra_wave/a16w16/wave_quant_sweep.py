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

"""
Wave-quantization sweep for v9_beyond_hotloop: fixed M = K = 4096, N from 512 to
16384 in steps of 256 (one BLOCK_N column = +16 tiles per step).

Timing is rocprofv3 kernel-trace over cold rotating buffers with the shipping
`llir+amdgcnas` stack. The script relaunches itself once under
rocprofv3, then reduces the trace to
v9_beyond_hotloop/images/wave_quant.csv, which scripts/plot_wave_quant.py plots.

    python wave_quant_sweep.py --gpu 1
"""

import argparse
import csv
import glob
import json
import os
import re
import shutil
import statistics as S
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
KERNEL = "v9_beyond_hotloop"
TRACE_DIR = os.path.join(HERE, "tmp", "wave_quant_trace")
MANIFEST = os.path.join(TRACE_DIR, "manifest.json")
OUT_CSV = os.path.join(HERE, KERNEL, "images", "wave_quant.csv")

M = K = 4096
BLOCK = 256
N_VALUES = list(range(512, 16384 + 1, 256))
NUM_CUS = 256
THREADS_PER_WG = 256  # num_warps=4 x wave64


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--gpu", type=int, default=1, help="GPU index (HIP_VISIBLE_DEVICES), default 1")
    p.add_argument("--iters", type=int, default=200, help="timed dispatches per N (default 200)")
    p.add_argument("--keep", type=int, default=100, help="last-K dispatches averaged (default 100)")
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    p.add_argument("--rotating-buffer-size", type=int, default=1024, help="MB (default 1024)")
    p.add_argument("--force", action="store_true", help="skip the GPU-idle check")
    p.add_argument("--parse-only", action="store_true", help="re-reduce an existing trace")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def gpu_busy_pct(gpu):
    out = subprocess.run(["rocm-smi", "-d", str(gpu), "--showuse"], capture_output=True, text=True).stdout
    m = re.search(r"GPU use \(%\):\s*(\d+)", out)
    return int(m.group(1)) if m else None


def check_idle(gpu):
    samples = []
    for _ in range(5):
        samples.append(gpu_busy_pct(gpu))
        time.sleep(0.5)
    if any(s is None for s in samples):
        print(f"warning: could not read utilization of GPU {gpu}; continuing")
    elif max(samples) > 5:
        sys.exit(f"GPU {gpu} is busy (use% samples {samples}); pick another --gpu or pass --force")
    else:
        print(f"GPU {gpu} idle (use% samples {samples})")


def worker(args):
    sys.path.insert(0, HERE)
    import bench  # sets RTLD_GLOBAL / amdgcnas hook / utils path before triton use
    import importlib
    import torch

    matmul = importlib.import_module(f"{KERNEL}.matmul_kernel").matmul
    torch_dtype = bench.name_to_torch_type[args.dtype]
    manifest = []
    for N in N_VALUES:
        a = torch.rand((M, K), device=bench.DEVICE, dtype=torch_dtype) - 0.5
        b = torch.rand((N, K), device=bench.DEVICE, dtype=torch_dtype).T - 0.5
        correct = bool(torch.allclose(matmul(a, b), torch.matmul(a, b), atol=1e-1, rtol=0))
        del a, b

        a_list, b_list, c_list, copies = bench.gen_rotating_tensors(
            M, N, K, torch_dtype, args.rotating_buffer_size
        )
        matmul(a_list[0], b_list[0], c_list[0])
        torch.cuda.synchronize()
        for i in range(args.iters):
            idx = i % copies
            matmul(a_list[idx], b_list[idx], c_list[idx])
        torch.cuda.synchronize()
        del a_list, b_list, c_list
        torch.cuda.empty_cache()

        tiles = (M // BLOCK) * (N // BLOCK)
        manifest.append({"N": N, "tiles": tiles, "correct": correct, "dispatches": 2 + args.iters})
        print(f"N={N:5d} tiles={tiles:4d} copies={copies:3d} correct={correct}", flush=True)
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=1)


def profile(args):
    shutil.rmtree(TRACE_DIR, ignore_errors=True)
    os.makedirs(TRACE_DIR)
    env = dict(
        os.environ,
        HIP_VISIBLE_DEVICES=str(args.gpu),
        LLVM_PASS_PLUGIN_PATH=os.path.join(REPO, "plugins", "llir_scheduler", "libLlirSched.so"),
        TRITON_AMDGCNAS_PLUGIN="1",
    )
    cmd = ["rocprofv3", "--kernel-trace", "--kernel-include-regex", KERNEL,
           "-d", TRACE_DIR, "-o", "run", "--output-format", "csv", "--",
           sys.executable, os.path.abspath(__file__), "--worker",
           "--iters", str(args.iters), "--dtype", args.dtype,
           "--rotating-buffer-size", str(args.rotating_buffer_size)]  # fmt: skip
    subprocess.run(cmd, cwd=HERE, env=env, check=True)


def reduce(args):
    manifest = json.load(open(MANIFEST))
    trace = glob.glob(os.path.join(TRACE_DIR, "**", "*kernel_trace.csv"), recursive=True)
    assert len(trace) == 1, f"expected one kernel trace, found {trace}"
    rows = [r for r in csv.DictReader(open(trace[0])) if KERNEL in r["Kernel_Name"]]
    rows.sort(key=lambda r: int(r["Dispatch_Id"]))
    expected = sum(m["dispatches"] for m in manifest)
    assert len(rows) == expected, f"trace has {len(rows)} {KERNEL} dispatches, expected {expected}"

    out, pos = [], 0
    for m in manifest:
        chunk = rows[pos : pos + m["dispatches"]]
        pos += m["dispatches"]
        grids = {int(r["Grid_Size_X"]) for r in chunk}
        assert grids == {m["tiles"] * THREADS_PER_WG}, f"N={m['N']}: grid sizes {grids}"
        us = [(int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3 for r in chunk[-args.keep :]]
        q = S.quantiles(us, n=10)
        med = S.median(us)
        out.append({
            "N": m["N"],
            "tiles": m["tiles"],
            "waves": -(-m["tiles"] // NUM_CUS),
            "us_median": round(med, 3),
            "us_p20": round(q[1], 3),
            "us_p80": round(q[7], 3),
            "tflops": round(2 * M * m["N"] * K / (med * 1e-6) / 1e12, 1),
            "correct": m["correct"],
        })  # fmt: skip

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"{'N':>6} {'tiles':>5} {'waves':>5} {'us':>9} {'TFLOPS':>7} ok")
    for r in out:
        print(f"{r['N']:6d} {r['tiles']:5d} {r['waves']:5d} {r['us_median']:9.2f} {r['tflops']:7.1f} "
              f"{'yes' if r['correct'] else 'NO'}")  # fmt: skip
    print(f"wrote {OUT_CSV}")


def main():
    args = parse_args()
    if args.worker:
        worker(args)
        return
    if not args.parse_only:
        if not args.force:
            check_idle(args.gpu)
        profile(args)
    reduce(args)


if __name__ == "__main__":
    main()
