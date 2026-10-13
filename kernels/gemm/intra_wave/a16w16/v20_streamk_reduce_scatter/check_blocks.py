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
"""Pure-Python check of v20's reduce-scatter address maps (no GPU).

    python v20_streamk_reduce_scatter/check_blocks.py

1. Decode. The writer stores element (row, col) of quadrant q at
   q * 16384 + lane_contiguous_offsets(row, col). reduce_scatter_slice reads block
   blk = address // 1024 and decodes it back to a tile row and column. Every one of the
   65536 elements of a slot must come back to where it was written, and every address must
   be hit once.
2. Coverage. For every split count n from 2 to 128, the blocks [64 j / n, 64 (j + 1) / n)
   of programs j = 0 .. n - 1 must cover the 64 blocks of a tile exactly once.
3. Group shape. reduce_scatter_shape must give a power-of-two NB, and NB * U loads in
   flight, for every leftover-tile count and K the host can pass.
"""

import itertools
import sys


def lane_contiguous(r, c):
    """lane_contiguous_offsets in matmul_kernel.py, for one element of a 128x128 quadrant."""
    f = (r % 16) * 4 + (r // 16 % 2) * 512 + (r // 32 % 2) * 4096 + (r // 64) * 8192
    g = c + (c // 4 % 4) * 60 + (c // 16 % 2) * 240 + (c // 32 % 2) * 992 + (c // 64) * 1984
    return f + g


def in_block(r, c):
    """reduce_scatter_slice's in_blk, for one element of a 32x32 block."""
    return (r % 16) * 4 + (r // 16) * 512 + c + (c // 4 % 4) * 60 + (c // 16) * 240


def block_origin(blk):
    """reduce_scatter_slice's (row, col) of block blk's first element in the tile."""
    q, v = blk // 16, blk % 16
    row = (q % 2) * 128 + (v // 4 % 2) * 32 + (v // 8) * 64
    col = (q // 2) * 128 + (v % 2) * 32 + (v // 2 % 2) * 64
    return row, col


def reduce_scatter_shape(streamk_tiles, num_programs, pairs_per_tile, loads_in_flight=16):
    """matmul_kernel.reduce_scatter_shape, without importing triton."""
    n_lo = max(min(num_programs // max(streamk_tiles, 1), pairs_per_tile), 2)
    nb = min(1 << (-(-64 // n_lo) - 1).bit_length(), loads_in_flight)
    return nb, max(loads_in_flight // nb, 1)


def check_decode():
    decode = {in_block(r, c): (r, c) for r, c in itertools.product(range(32), range(32))}
    if sorted(decode) != list(range(1024)):
        return "in_blk is not a permutation of 0..1023"
    seen = set()
    for q, r, c in itertools.product(range(4), range(128), range(128)):
        addr = q * 16384 + lane_contiguous(r, c)
        seen.add(addr)
        row0, col0 = block_origin(addr // 1024)
        dr, dc = decode[addr % 1024]
        want = ((q % 2) * 128 + r, (q // 2) * 128 + c)
        if (row0 + dr, col0 + dc) != want:
            return f"q={q} r={r} c={c}: written at {addr}, read back as {(row0 + dr, col0 + dc)}"
    if seen != set(range(65536)):
        return "the writer does not cover the slot exactly once"
    return None


def check_coverage():
    for n in range(2, 129):
        blocks = [b for j in range(n) for b in range((64 * j) // n, (64 * (j + 1)) // n)]
        if blocks != list(range(64)):
            return f"n={n}: blocks are not covered exactly once"
    return None


def check_group_shape():
    for s, k in itertools.product(range(1, 256), (2048, 4096, 8192, 16384, 32768)):
        nb, u = reduce_scatter_shape(s, 256, k // 128)
        if nb & (nb - 1) or not 1 <= nb <= 16 or nb * u != 16:
            return f"S={s} K={k}: NB={nb} U={u}"
    return None


def main():
    failures = 0
    for name, check in (("decode", check_decode), ("coverage", check_coverage), ("group shape", check_group_shape)):
        err = check()
        print(f"{name}: {'ok' if err is None else 'FAIL: ' + err}")
        failures += err is not None
    print("\nPASS" if failures == 0 else "\nFAIL")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
