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
"""Check v21's tile-aligned placements in pure Python (no GPU).

    python v21_streamk_l2_partials/check_partition.py

1. The XCD-local split (xcd_local_split + tile_groups with xcd_local, the host's replay of the
   kernel's xcd_local_chunk_of), for every S from 1 to 128, requested split n from 2 to 64 and
   2 * K / 128 = 16 to 256 pairs per tile: every stream-K tile gets exactly the chosen number
   of programs, they are consecutive spids (the reduce-scatter slots and flags b .. b + n - 1)
   on one XCD, and no program has two tiles. It also compares the map with an independent
   construction: tiles dealt to the XCDs in contiguous blocks, then each XCD's programs dealt
   to its tiles in order.
2. v20's balanced placement (streamk_compact_pid + streamk_chunk_of) under the capped host
   rule: for which S do a tile's programs span XCDs (so STREAMK_PARTIALS=l2 needs the
   XCD-local split), and the XCD-local split's per-XCD partial footprint at the capped n.
"""

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "..", "..", "utils"))

import triton  # noqa: E402,F401  (before torch)
from v21_streamk_l2_partials.matmul_kernel import capped_policy, tile_groups, xcd_local_split  # noqa: E402

P, X = 256, 8
PER = P // X
SLOT_MIB = 256 * 256 * 4 / 2**20


def independent(S, n):
    """Tiles dealt to XCDs in contiguous blocks (the first S % X XCDs one more), then every
    XCD's programs dealt to its tiles n at a time."""
    groups = []
    t = 0
    for x in range(X):
        for i in range(S // X + (x < S % X)):
            groups.append(list(range(x * PER + i * n, x * PER + (i + 1) * n)))
            t += 1
    assert t == S
    return groups


def check_xcd_local():
    cases = 0
    for S in range(1, 129):
        for n_req in (2, 3, 4, 6, 8, 12, 16, 24, 32, 64):
            for ppt in (16, 32, 64, 128, 256):
                n = xcd_local_split(S, n_req, ppt, P, X)
                assert 1 <= n <= min(n_req, ppt) and math.ceil(S / X) * n <= PER, (S, n_req, ppt, n)
                g = tile_groups(S, S * n, ppt, P, X, True, True, n)
                assert len(g) == S
                seen = set()
                for members in g:
                    assert len(members) == n, (S, n, members)
                    assert members == list(range(members[0], members[0] + n)), members
                    assert len({s // PER for s in members}) == 1, members
                    assert not seen & set(members)
                    seen |= set(members)
                assert g == independent(S, n), (S, n)
                cases += 1
    return cases


def capped_table():
    rows = []
    for S in range(1, 129):
        for ppt in (64, 128):
            K = ppt * 128
            total = P + S
            policy, fixup, programs = capped_policy(total, K // 64, P)
            if policy != "one_tile" or fixup != "rs":
                continue
            snp = programs or P
            g20 = tile_groups(S, snp, ppt, P, X, True, False, 1)
            straddle = sum(len({s // PER for s in m}) > 1 for m in g20)
            n = xcd_local_split(S, snp // S, ppt, P, X)
            mib = math.ceil(S / X) * n * SLOT_MIB
            rows.append((S, K, snp // S, straddle, n, mib, min(S, X)))
    return rows


def main():
    cases = check_xcd_local()
    print(f"XCD-local split: {cases} cases (S 1-128, 10 requested splits, 5 tile lengths) ok\n")
    print("Capped rule, reduce-scatter cases (the only ones STREAMK_PARTIALS=l2 changes). 'v20 straddling' "
          "counts tiles whose programs span XCDs under v20's balanced placement; the XCD-local split's n "
          "and per-XCD partial footprint (busiest XCD, 4 MiB of L2 per XCD) and the XCDs it uses.\n")
    print("| S | K | capped n | v20 straddling tiles | XCD-local n | MiB per XCD | XCDs used |")
    print("|---|---|---|---|---|---|---|")
    for S, K, n20, straddle, n, mib, xcds in capped_table():
        print(f"| {S} | {K} | {n20} | {straddle} | {n} | {mib:g} | {xcds} |")


if __name__ == "__main__":
    main()
