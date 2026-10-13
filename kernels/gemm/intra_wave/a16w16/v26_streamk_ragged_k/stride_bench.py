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
"""Launch a version's matmul on A (M, K) and B (K, N) whose rows (A's, and B's as an (N, K) tensor) have
an explicit stride S >= K, over rotating copies (> 512 MB), for rocprofv3 --kernel-trace (stride_rp.py).

    python v26_streamk_ragged_k/stride_bench.py VERSION M N K S [ITERS]

The first launch is checked against torch."""
import importlib
import os
import sys

sys.path.insert(0, os.getcwd())
import bench  # noqa: E402
import torch  # noqa: E402

v, M, N, K, S = sys.argv[1], *map(int, sys.argv[2:6])
iters = int(sys.argv[6]) if len(sys.argv) > 6 else 300
mod = importlib.import_module(bench.VERSION_MAP[int(v) if v.isdigit() else v] + ".matmul_kernel")
per = (M * S + N * S + M * N) * 2
copies = max(1, (512 << 20) // per + 1)
sets = []
for _ in range(copies):
    a = torch.randn((M, S), device="cuda", dtype=torch.bfloat16)[:, :K]
    b = torch.randn((N, S), device="cuda", dtype=torch.bfloat16)[:, :K].T
    c = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    sets.append((a, b, c))
a, b, c = sets[0]
ref = (a.float() @ b.float()).to(torch.bfloat16)
mod.matmul(a, b, c)
assert torch.allclose(c.float(), ref.float(), atol=1e-1, rtol=1e-2), "wrong result"
for i in range(iters):
    a, b, c = sets[i % copies]
    mod.matmul(a, b, c)
torch.cuda.synchronize()
