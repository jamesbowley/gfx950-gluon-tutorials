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
"""Stream-K stress check for v15: back-to-back launches on rotating inputs, every result
checked, then the flags must all be re-armed (zero).

    cd kernels/gemm/intra_wave/a16w16
    HIP_VISIBLE_DEVICES=2 python v15_streamk_onetile/check_streamk.py
    HIP_VISIBLE_DEVICES=2 python v15_streamk_onetile/check_streamk.py --shapes 4352x4096x8192 --launches 100

regress.py checks one launch on fresh workspace. A stale flag (never reset, or reset too
late) lets an owner read a partial before this launch's contributor has written it; with
inputs that change every launch that partial is the previous launch's, and C is wrong.
Needs the same stack as regress.py.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import bench  # noqa: E402  (plugin hooks and the path to common.py)
import regress  # noqa: E402
import torch  # noqa: E402

from v15_streamk_onetile import matmul_kernel  # noqa: E402

N_INPUTS = 3


def check_shape(M, N, K, bias, launches):
    dtype = torch.bfloat16
    inputs = []
    for _ in range(N_INPUTS):
        a = torch.rand((M, K), device=bench.DEVICE, dtype=dtype) - 0.5
        b = torch.rand((N, K), device=bench.DEVICE, dtype=dtype).T - 0.5
        bias_n = torch.rand((N,), device=bench.DEVICE, dtype=dtype) - 0.5 if bias else None
        if bias:
            ref = (torch.matmul(a.float(), b.float()) + bias_n.float()).to(dtype)
        else:
            ref = torch.matmul(a, b)
        inputs.append((a, b, bias_n, ref))
    outs = [torch.empty((M, N), device=bench.DEVICE, dtype=dtype) for _ in range(N_INPUTS)]

    bad = 0
    for i in range(launches):
        a, b, bias_n, ref = inputs[i % N_INPUTS]
        c = outs[i % N_INPUTS]
        c.fill_(float("nan"))
        matmul_kernel.matmul(a, b, c, bias=bias_n)
        # Same tolerances as bench.test_correctness.
        if not torch.allclose(c, ref, atol=1e-1, rtol=1e-2):
            bad += 1
    torch.cuda.synchronize()
    _, locks = matmul_kernel.streamk_workspace(bench.DEVICE, 256, 256, 256)
    flags_up = int((locks != 0).sum())
    return bad, flags_up


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shapes", nargs="*", help="MxNxK shapes (default: regress.STREAMK_SHAPES)")
    parser.add_argument("--launches", type=int, default=30)
    parser.add_argument("--orders", nargs="+", default=["v9", "split"], choices=["v9", "split"])
    parser.add_argument("--bias", nargs="+", type=int, default=[0, 1], choices=[0, 1])
    args = parser.parse_args()

    shapes = [regress.parse_shape(s) for s in args.shapes] if args.shapes else regress.STREAMK_SHAPES
    failures = 0
    print("| shape | tile order | bias | launches | wrong results | flags left up |")
    print("|---|---|---|---|---|---|")
    for M, N, K in shapes:
        for order in args.orders:
            os.environ["PERSISTENT_TILE_ORDER"] = order
            for bias in args.bias:
                bad, flags_up = check_shape(M, N, K, bool(bias), args.launches)
                failures += bad + flags_up
                print(f"| {M}x{N}x{K} | {order} | {bias} | {args.launches} | {bad} | {flags_up} |", flush=True)
    print("\nPASS" if failures == 0 else "\nFAIL")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
