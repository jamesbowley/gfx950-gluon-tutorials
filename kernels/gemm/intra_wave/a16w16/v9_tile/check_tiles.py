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

B and C rows are contiguous, as in bench_ext.py; `--pad` pads them as bench_ext.py --pad does. A failure prints
the largest error and where it is, which tells a layout bug (whole rows or columns wrong) from
rounding. RAGGED_SHAPES have M, N or K that are not multiples of the tile: C is a window of a buffer
256 rows and columns larger, filled with a sentinel, and a case also fails if anything outside the
window changes. Tiles with a side of 32 or less (v9_tile_small) run SMALL_SHAPES, SHAPES and
SMALL_RAGGED_SHAPES (the long side and K ragged, EDGE_N and EDGE_K; checked in the same sentinel
window as RAGGED_SHAPES, which every case uses). Tiles with BLOCK_M a multiple of 16 between 128 and
256 (v9_tile's row bands) also run BAND_SHAPES, whose M is a multiple of the tile, and tiles with
such a BLOCK_N (column bands) BAND_N_SHAPES.
"""

import os
import sys

A16 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, A16)
import bench_ext as bench  # noqa: E402
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
RAGGED_SHAPES = [
    (4000, 4000, 8192),
    (1000, 2000, 4096),
    (4096, 2032, 4096),
    (130, 1040, 1024),
    (4096, 4096, 8176),
    (4000, 4000, 8080),
    (1000, 2000, 4032),
    (130, 1040, 400),
]
# For v9_tile_small (a tile side of 32 or less): skinny M, M not a multiple of the tile (130),
# and the mirror images, which run as C^T = B^T A^T.
SMALL_SHAPES = [
    (1, 4096, 4096),
    (2, 2048, 2048),
    (4, 1024, 4096),
    (8, 4096, 2048),
    (16, 4096, 4096),
    (32, 2048, 4096),
    (130, 1024, 1024),
    (512, 512, 1024),
    (2048, 1024, 4096),
    (4096, 1, 4096),
    (2048, 8, 2048),
    (1024, 16, 4096),
    (1024, 130, 1024),
]
# Ragged long side and K for v9_tile_small (EDGE_N, EDGE_K), and their mirror images.
SMALL_RAGGED_SHAPES = [
    (16, 4000, 4096),
    (30, 4032, 4096),
    (16, 4096, 4080),
    (32, 4096, 4000),
    (30, 4000, 4080),
    (1, 4032, 4000),
    (4000, 16, 4096),
    (4032, 30, 4080),
    (4000, 1, 4000),
]
# M a multiple of a banded BLOCK_M (192, 224, 240; v9_tile's row bands), where EDGE_M is off.
BAND_SHAPES = [
    (3072, 4096, 4096),
    (3584, 4096, 4096),
    (3840, 4096, 4096),
    (1920, 2048, 8192),
]
# N a multiple of a banded BLOCK_N (column bands), where EDGE_N is off.
BAND_N_SHAPES = [
    (4096, 3072, 4096),
    (4096, 3584, 4096),
    (4096, 3840, 4096),
    (2048, 2304, 8192),
]
SENTINEL = 1234.0


def check(tile, M, N, K, dtype, bias, pad):
    BM, BN, BK, regions = mk.parse_tile(tile)
    if mk.is_small(BM, BN, regions):
        if K % 16:
            return None
    elif K % 16 or -(-K // (2 * BK)) * 2 * BK < 4 * BK:
        return None
    torch.manual_seed(0)
    a = torch.rand((M, K), device="cuda", dtype=dtype) - 0.5
    b = bench.make_b(N, K, dtype, pad, lambda t: t.uniform_(-0.5, 0.5))
    big = bench.make_c(M + 256, N + 256, dtype, pad)
    big.fill_(SENTINEL)
    c = big[:M, :N]
    c.fill_(float("nan"))
    outside = torch.ones(big.shape, device="cuda", dtype=torch.bool)
    outside[:M, :N] = False
    bias_n = (torch.rand((N,), device="cuda", dtype=dtype) - 0.5) if bias else None
    mk.matmul(a, b, c, bias=bias_n, tile=tile)
    ref = a.float() @ b.float()
    if bias:
        ref = ref + bias_n.float()
    ref = ref.to(dtype).float()
    err = (c.float() - ref).abs()
    rtol = 1e-2 if dtype == torch.bfloat16 else 0
    ok = bool(torch.isfinite(c).all()) and bool((err <= 1e-1 + rtol * ref.abs()).all())
    ok = ok and bool((big[outside] == SENTINEL).all())
    worst = err.nan_to_num(float("inf")).argmax().item()
    return ok, err.nan_to_num(float("inf")).max().item(), divmod(worst, N)


def main():
    tiles = sys.argv[1].split(",")
    pad = "--pad" in sys.argv
    print(f"| tile | shape | dtype | bias | result | max abs error | at (row, col) |")
    print("|---|---|---|---|---|---|---|")
    fails = 0
    for tile in tiles:
        BM, BN, _, regions = mk.parse_tile(tile)
        if mk.is_small(BM, BN, regions):
            shapes = SMALL_SHAPES + SHAPES + SMALL_RAGGED_SHAPES
        else:
            shapes = (SHAPES + RAGGED_SHAPES + (BAND_SHAPES if mk.row_bands(BM) else [])
                      + (BAND_N_SHAPES if mk.col_bands(BN) else []))
        for M, N, K in shapes:
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
