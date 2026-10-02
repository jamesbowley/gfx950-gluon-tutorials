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
"""GEMM time with different work between launches: CUDA events around each GEMM only, 100
launches on the same inputs, median of launches 10-99.

    cd kernels/gemm/intra_wave/a16w16
    HIP_VISIBLE_DEVICES=4 python v20_streamk_reduce_scatter/between_launches.py 4352x4096x8192,3328x5120x8192 13,17,19,20
    HIP_VISIBLE_DEVICES=4 python v20_streamk_reduce_scatter/between_launches.py 4096x7168x32768 13,16,20:STREAMK_POLICY=spread

Versions are specs as in version_spec.py. All versions of a shape share one set of inputs.

- nothing: back to back. For kernels under about 200 us the GPU then waits on the Python
  launch path between GEMMs, so this column is only valid for the longer kernels.
- zero 256 MB: what triton.testing.do_bench does before every launch; leaves L2 and the
  last-level cache full of dirty lines that are written back while the GEMM starts.
- read 256 MB: as cold, but clean.
- zero 64 MB: a milder dirty state.
- sleep 100 us: an idle gap of about 100 us.
"""
import importlib
import os
import sys

A16 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, A16)
import bench  # noqa: E402
import torch  # noqa: E402

from v20_streamk_reduce_scatter import version_spec  # noqa: E402

shapes = [tuple(int(x) for x in s.split("x")) for s in sys.argv[1].split(",")]
versions = sys.argv[2].split(",")
buf = torch.empty(256 * 2**20 // 4, device="cuda", dtype=torch.int32)
small = torch.empty(64 * 2**20 // 4, device="cuda", dtype=torch.int32)
modes = {
    "nothing": lambda: None,
    "zero 256 MB (do_bench)": lambda: buf.zero_(),
    "read 256 MB": lambda: buf.sum(),
    "zero 64 MB": lambda: small.zero_(),
    "sleep 100 us": lambda: torch.cuda._sleep(200000),
}
print("| shape | version | " + " | ".join(modes) + " |")
print("|---|---|" + "---|" * len(modes))
for M, N, K in shapes:
    a = torch.rand((M, K), device="cuda", dtype=torch.bfloat16) - 0.5
    b = torch.rand((N, K), device="cuda", dtype=torch.bfloat16).T - 0.5
    c = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    for spec in versions:
        v, env = version_spec.parse(spec)
        mod = importlib.import_module(bench.VERSION_MAP[v] + ".matmul_kernel")
        res = []
        with version_spec.environment(env):
            mod.matmul(a, b, c)
            for name, between in modes.items():
                n = 100
                ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(n)]
                for i in range(n):
                    between()
                    ev[i][0].record()
                    mod.matmul(a, b, c)
                    ev[i][1].record()
                torch.cuda.synchronize()
                t = sorted(s.elapsed_time(e) * 1e3 for s, e in ev[10:])
                res.append(t[len(t) // 2])
        print(f"| {M}x{N}x{K} | {version_spec.label(spec)} | " + " | ".join(f"{x:.1f}" for x in res) + " |",
              flush=True)
