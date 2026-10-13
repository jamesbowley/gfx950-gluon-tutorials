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
"""Kernel-only cold times of variants: rocprofv3 --kernel-trace over 300 launches on rotating copies
(> 512 MB) of A and B with row stride S (as v26's stride_rp.py), median, per V:MxNxK:S[:ENV=VAL].

    HIP_VISIBLE_DEVICES=4 python experiments/ragged_spills/time_case.py base:8150x8400x8080:8256 \\
        crow:8150x8400x8080:8256

Runs itself under rocprofv3 with --launch for each case; CASE_BIAS=1 (as an ENV=VAL) adds a bias.
"""

import importlib.util
import os
import statistics
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
A16 = os.path.join(REPO, "kernels", "gemm", "intra_wave", "a16w16")
sys.path.insert(0, A16)
sys.path.insert(0, os.path.join(A16, "v20_streamk_reduce_scatter"))
REGEX = "v26_streamk_ragged_k|v25_streamk_ragged_mn|v9_ragged"


def launch(variant, M, N, K, S, iters=300):
    import bench  # noqa: F401
    import torch

    spec = importlib.util.spec_from_file_location("variant", os.path.join(HERE, "variants", f"{variant}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    per = (M * S + N * S + M * N) * 2
    copies = max(1, (512 << 20) // per + 1)
    sets = []
    for _ in range(copies):
        a = torch.randn((M, S), device="cuda", dtype=torch.bfloat16)[:, :K]
        b = torch.randn((N, S), device="cuda", dtype=torch.bfloat16)[:, :K].T
        sets.append((a, b, torch.empty((M, N), device="cuda", dtype=torch.bfloat16)))
    bias = torch.randn((N,), device="cuda", dtype=torch.bfloat16) if os.environ.get("CASE_BIAS") == "1" else None
    for i in range(iters):
        a, b, c = sets[i % copies]
        mod.matmul(a, b, c, bias=bias)
    torch.cuda.synchronize()


def main():
    if sys.argv[1] == "--launch":
        launch(sys.argv[2], *map(int, sys.argv[3:7]))
        return
    import rocprof_check as rc

    print("| variant | shape | row stride | env | cold kernel us |")
    print("|---|---|---|---|---|")
    for case in sys.argv[1:]:
        v, shape, S, *env = case.split(":")
        M, N, K = shape.split("x")
        d = tempfile.mkdtemp(prefix="rst_")
        subprocess.run(["rocprofv3", "--kernel-trace", "--kernel-include-regex", REGEX, "-d", d, "-o", "run",
                        "--output-format", "csv", "--", sys.executable, os.path.abspath(__file__), "--launch",
                        v, M, N, K, S], env={**os.environ, **dict(e.split("=", 1) for e in env)},
                       capture_output=True, text=True)
        rows = [r for r in rc.kernel_rows(d, REGEX) if "End_Timestamp" in r]
        us = statistics.median((int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3 for r in rows[1:]) \
            if rows else float("nan")
        print(f"| {v} | {shape} | {S} | {' '.join(env) or '-'} | {us:.1f} |", flush=True)


if __name__ == "__main__":
    main()
