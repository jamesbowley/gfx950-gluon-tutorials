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
"""Correctness of v9_tile against an fp32 torch reference, per tile, shape, dtype and bias.

    source v9_tile/env.sh
    python v9_tile/check_tiles.py 256x256,256x128,128x256 > v9_tile/results/check_tiles.md

B and C rows are padded as bench.py does; `--no-pad` uses contiguous tensors. A failure prints
the largest error and where it is, which tells a layout bug (whole rows or columns wrong) from
rounding.
"""

import os
import sys

A16 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, A16)
import bench  # noqa: E402
import torch  # noqa: E402
from v9_tile import matmul_kernel as mk  # noqa: E402

SHAPES = [
    (4096, 4096, 8192),
    (4096, 2048, 4096),
    (2048, 4096, 4096),
    (2048, 2048, 1024),
    (4096, 1024, 4096),
    (1024, 2304, 16384),
    (16384, 512, 8192),
    (512, 512, 256),
]


def check(tile, M, N, K, dtype, bias, pad):
    BM, BN, BK, _ = mk.parse_tile(tile)
    if M % BM or N % BN or K % BK or K // BK < 4:
        return None
    torch.manual_seed(0)
    a = torch.rand((M, K), device="cuda", dtype=dtype) - 0.5
    b = bench.make_b(N, K, dtype, pad, lambda t: t.uniform_(-0.5, 0.5))
    c = bench.make_c(M, N, dtype, pad)
    c.fill_(float("nan"))
    bias_n = (torch.rand((N,), device="cuda", dtype=dtype) - 0.5) if bias else None
    mk.matmul(a, b, c, bias=bias_n, tile=tile)
    ref = a.float() @ b.float()
    if bias:
        ref = ref + bias_n.float()
    ref = ref.to(dtype).float()
    err = (c.float() - ref).abs()
    rtol = 1e-2 if dtype == torch.bfloat16 else 0
    ok = bool(torch.isfinite(c).all()) and bool((err <= 1e-1 + rtol * ref.abs()).all())
    worst = err.nan_to_num(float("inf")).argmax().item()
    return ok, err.nan_to_num(float("inf")).max().item(), divmod(worst, N)


def main():
    tiles = sys.argv[1].split(",")
    pad = "--no-pad" not in sys.argv
    print(f"| tile | shape | dtype | bias | result | max abs error | at (row, col) |")
    print("|---|---|---|---|---|---|---|")
    fails = 0
    for tile in tiles:
        for M, N, K in SHAPES:
            for dtype in (torch.bfloat16, torch.float16):
                for bias in (False, True):
                    r = check(tile, M, N, K, dtype, bias, pad)
                    if r is None:
                        continue
                    ok, e, at = r
                    fails += not ok
                    name = "bf16" if dtype == torch.bfloat16 else "fp16"
                    print(f"| {tile} | {M}x{N}x{K} | {name} | {bias} | {'ok' if ok else 'FAIL'} | {e:.3g} | {at} |",
                          flush=True)
    print(f"\n{fails} failures")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
