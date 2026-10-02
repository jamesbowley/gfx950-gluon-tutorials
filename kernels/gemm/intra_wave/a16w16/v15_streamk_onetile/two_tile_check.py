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
"""v15's kernel under the one-tile and two-tile policies, against v13.

    cd kernels/gemm/intra_wave/a16w16
    HIP_VISIBLE_DEVICES=2 python v15_streamk_onetile/two_tile_check.py [MxNxK ...]

The policy is only the host's STREAMK_TILES: one-tile is total % 256, two-tile moves one full
wave into stream-K (total % 256 + 256), which needs at least one full wave. The kernel runs
either unchanged. Each shape: 10 two-tile launches on alternating inputs are checked, the
flags must be re-armed, then do_bench TFLOPS for v13, one-tile and two-tile.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import bench  # noqa: E402,F401  (plugin hooks and the path to common.py)
import regress  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402

from v13_persistant_peel_acc.matmul_kernel import matmul as v13  # noqa: E402
from v15_streamk_onetile import matmul_kernel as mk  # noqa: E402

NUM_PROGRAMS = 256
DEFAULT_SHAPES = [(4352, 4096, 8192), (4352, 4352, 8192), (4352, 4096, 16384), (3328, 5120, 8192),
                  (4096, 6144, 8192), (8192, 8448, 8192), (8192, 7936, 8192)]


def launch(a, b, c, streamk_tiles):
    M, K = a.shape
    _, N = b.shape
    P, locks = mk.streamk_workspace(a.device, NUM_PROGRAMS, 256, 256)
    mk.v15_streamk_onetile[(NUM_PROGRAMS, 1)](
        a, b, c, None, P, locks, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=256, BLOCK_N=256, BLOCK_K=64, NUM_PROGRAMS=NUM_PROGRAMS,
        STREAMK_TILES=streamk_tiles, STREAMK_NUM_PROGRAMS=NUM_PROGRAMS, NUM_XCDS=8,
        GROUP_SIZE_M=4, TILE_ORDER_V9=True, MASK_TAIL_PREFETCH=False, ADD_BIAS=False,
        num_warps=4,
    )
    return c


def main():
    shapes = [regress.parse_shape(s) for s in sys.argv[1:]] or DEFAULT_SHAPES
    print("| shape | full waves left | v13 | one-tile | two-tile | two-tile correct |")
    print("|---|---|---|---|---|---|")
    for M, N, K in shapes:
        total = (M // 256) * (N // 256)
        assert total >= NUM_PROGRAMS, "two-tile needs at least one full wave"
        one, two = total % NUM_PROGRAMS, total % NUM_PROGRAMS + NUM_PROGRAMS
        a = torch.rand((M, K), device="cuda", dtype=torch.bfloat16) - 0.5
        b = torch.rand((N, K), device="cuda", dtype=torch.bfloat16).T - 0.5
        a2 = torch.rand_like(a) - 0.5
        refs = [(a, torch.matmul(a, b)), (a2, torch.matmul(a2, b))]
        c = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
        ok = True
        for i in range(10):
            x, ref = refs[i % 2]
            c.fill_(float("nan"))
            launch(x, b, c, two)
            ok &= torch.allclose(c, ref, atol=1e-1, rtol=1e-2)
        _, locks = mk.streamk_workspace(a.device, NUM_PROGRAMS, 256, 256)
        ok &= int((locks != 0).sum()) == 0

        def tflops(fn):
            return 2 * M * N * K * 1e-12 / (triton.testing.do_bench(fn) * 1e-3)

        # The first do_bench of a run can read 25-30% low (v13 at 986 against 1322 here), so one
        # is thrown away.
        tflops(lambda: v13(a, b))
        t13 = tflops(lambda: v13(a, b))
        t1 = tflops(lambda: launch(a, b, c, one))
        t2 = tflops(lambda: launch(a, b, c, two))
        print(f"| {M}x{N}x{K} | {(total - two) // NUM_PROGRAMS} | {t13:.0f} | {t1:.0f} | {t2:.0f} | "
              f"{'yes' if ok else 'NO'} |", flush=True)


if __name__ == "__main__":
    main()
