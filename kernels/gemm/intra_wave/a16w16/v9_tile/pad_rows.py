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
"""Back-to-back time of a tile with A's (a) or B's (b) rows contiguous against padded
(bench.empty_rows / make_b), everything else as the sweeps run it; V9_TILE_GM to try a GROUP_SIZE_M.

    python v9_tile/pad_rows.py a|b TILE[,TILE] MxNxK
"""
import os
import statistics
import sys

sys.path.insert(0, os.getcwd())
import bench_ext as bench  # noqa: E402
import torch  # noqa: E402
from v9_tile import matmul_kernel as mk  # noqa: E402

which = sys.argv[1]
M, N, K = (int(x) for x in sys.argv[3].split("x"))
dt = torch.bfloat16
for tile in sys.argv[2].split(","):
    for pad in (False, True):
        pad_a, pad_b = (pad, True) if which == "a" else (False, pad)
        copies = max(4, 2048 * 2**20 // (2 * (M * K + K * N + M * N)))
        a_l = [bench.empty_rows(M, K, dt, "cuda", pad_a).uniform_(-0.5, 0.5) for _ in range(copies)]
        b_l = [bench.make_b(N, K, dt, pad_b, lambda t: t.uniform_(-0.5, 0.5)) for _ in range(copies)]
        c_l = [bench.make_c(M, N, dt, True) for _ in range(copies)]
        for i in range(copies):
            mk.matmul(a_l[i], b_l[i], c_l[i], tile=tile)
        ok = torch.allclose(c_l[0].float(), a_l[0].float() @ b_l[0].float(), atol=1e-1, rtol=1e-2)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for i in range(50):
                mk.matmul(a_l[i % copies], b_l[i % copies], c_l[i % copies], tile=tile)
        g.replay()
        torch.cuda.synchronize()
        us = []
        for _ in range(10):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            g.replay()
            e.record()
            e.synchronize()
            us.append(s.elapsed_time(e) * 1e3 / 50)
        med = statistics.median(us)
        stride = a_l[0].stride(0) if which == "a" else b_l[0].stride(1)
        print(f"{tile} {M}x{N}x{K} {which.upper()} row stride {stride}: {med:.1f} us, "
              f"{2 * M * N * K / med / 1e6:.0f} TFLOPS, correct={ok}", flush=True)
        del a_l, b_l, c_l, g
        torch.cuda.empty_cache()
