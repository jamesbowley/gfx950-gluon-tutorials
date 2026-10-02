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
"""Check and draw the stream-K tile-to-XCD mapping (v15-v17's streamk_tile_id). No GPU.

    python v15_streamk_onetile/check_mapping.py

The persistent loop and stream-K split the tiles between them; stream-K numbers its programs
with spid, which is already grouped by XCD (32 consecutive spids per XCD). Checks, for every
regress.STREAMK_SHAPES shape and random shapes, both tile orders and the one-tile and two-tile
policies, that the stream-K tiles are exactly the tiles the persistent loop leaves over. Then
draws, for 3840x4096 and 4352x4096, which XCD computes each stream-K tile under the old lookup
(persistent_tile_id(total_full_tiles + t), which groups by XCD a second time under v9 order)
and the fixed one.

The formulas replicate the kernel's xcd_remap_tiles, get_logical_chiplet_mapped_pids,
streamk_tile_id and get_group_m_tile_ids for 256x256 tiles, 256 programs, 8 XCDs and
GROUP_SIZE_M = 4. The XCD of a stream-K tile is shown for v17's tile-aligned split; '+' marks
a tile whose programs span two XCDs.
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

P, X, G = 256, 8, 4


def xcd_remap_tiles(v, count):
    per = (count + X - 1) // X
    tall = count % X or X
    x, local = v % X, v // X
    return x * per + local if x < tall else tall * per + (x - tall) * (per - 1) + local


def spid_of(p):
    """get_logical_chiplet_mapped_pids for 256 programs (256 % 8 == 0)."""
    return (p % X) * (P // X) + p // X


def streamk_tile_id(t, total, full, v9):
    if not v9:
        return full + t
    per = (total + X - 1) // X
    tall = total % X or X
    l0 = full // X
    n_tall = per - l0
    n_short = max(per - 1 - l0, 1)
    head = tall * n_tall
    if t < head:
        return (t // n_tall) * per + l0 + t % n_tall
    t2 = t - head
    return tall * per + (t2 // n_short) * (per - 1) + l0 + t2 % n_short


def old_streamk_tile_id(t, total, full, v9):
    return xcd_remap_tiles(full + t, total) if v9 else full + t


def persistent_leftover(total, full, v9):
    covered = {xcd_remap_tiles(v, total) if v9 else v for v in range(full)}
    return set(range(total)) - covered


def check(total, sk):
    full = total - sk
    for v9 in (True, False):
        want = persistent_leftover(total, full, v9)
        got = [streamk_tile_id(t, total, full, v9) for t in range(sk)]
        assert len(set(got)) == sk and set(got) == want, (total, sk, v9)
        if v9:
            assert got == sorted(got), "v9 order: stream-K tiles must ascend with t"


def group_m(tile_id, tm, tn):
    npg = G * tn
    first = (tile_id // npg) * G
    gs = min(tm - first, G)
    return first + (tile_id % npg) % gs, (tile_id % npg) // gs


def v17_tile_of_spid(sk):
    """v17's streamk_chunk_of: spid -> stream-K tile index t (programs beyond the cap idle)."""
    n_lo, n_hi = P // sk, P // sk + 1
    extra = P % sk
    head = extra * n_hi
    return lambda s: s // n_hi if s < head else extra + (s - head) // n_lo


def draw(M, N):
    tm, tn = M // 256, N // 256
    total = tm * tn
    sk = total % P
    full = total - sk
    t_of = v17_tile_of_spid(sk)
    xcds = {}
    for p in range(P):
        xcds.setdefault(t_of(spid_of(p)), set()).add(p % X)
    print(f"## {M}x{N}: {tm}x{tn} tiles, {full} persistent, {sk} stream-K (v9 order)\n")
    for title, f in (("old: persistent_tile_id(total_full_tiles + t), grouped twice", old_streamk_tile_id),
                     ("fixed: streamk_tile_id, grouped once", streamk_tile_id)):
        grid = {}
        for v in range(full):
            grid[group_m(xcd_remap_tiles(v, total), tm, tn)] = "."
        for t in range(sk):
            x = xcds[t]
            grid[group_m(f(t, total, full, True), tm, tn)] = str(next(iter(x))) if len(x) == 1 else "+"
        print(f"{title} (digit = XCD of the tile's programs, . = persistent tile)\n")
        print("```text")
        for m in range(tm):
            print(f"m{m:<3}" + " ".join(grid[(m, n)] for n in range(tn)))
        print("```\n")


def main():
    import regress

    cases = [((M // 256) * (N // 256)) for M, N, _ in regress.STREAMK_SHAPES]
    rng = random.Random(0)
    cases += [rng.randint(1, 5000) for _ in range(2000)]
    n = 0
    for total in cases:
        if total % P:
            check(total, total % P)  # one-tile
            n += 1
        if total >= P:
            check(total, total % P + P)  # two-tile
            n += 1
    print(f"leftover-set checks passed: {n} (shape, policy) cases, both tile orders\n")
    draw(3840, 4096)
    draw(4352, 4096)


if __name__ == "__main__":
    main()
