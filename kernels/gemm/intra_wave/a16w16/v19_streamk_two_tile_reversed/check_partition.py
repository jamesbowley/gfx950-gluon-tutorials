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
"""Check v19's reversed two-tile partition in pure Python (no GPU), for every shape in
regress.STREAMK_SHAPES and regress.LEFTOVER_SWEEP_SHAPES that has a full wave to borrow.

    python v19_streamk_two_tile_reversed/check_partition.py

Replicates the kernel's partition (pairs of k-steps, L = SK_PAIRS // 256, R = SK_PAIRS % 256,
ranges in spid order) and its segment loop (from the end of the range to its start). Checks:
- every pair is covered exactly once;
- every tile has one owner (the segment holding its last pair) with at most one peer, which is
  a lower spid and the owner's n_peers = spid - owner_of(tile start);
- every program has at most one partial, and it is the first segment it processes;
- the k-steps read by the 32 programs of an XCD at the same time are at most d apart (lockstep
  model, one pair per time step), where d = iters_per_tile * S / 256 rounded up to pairs.

It also prints, for reversed and forward (k-order) processing, that spread and the average
number of distinct k-steps an XCD's programs read at once. The spread overstates forward
order for small S: neighbouring programs differ by only a few k-steps there, but a program
near the end of a tile and one at the start of the next count as far apart.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

P, X = 256, 8


def owner_of(i, L, R):
    head = R * (L + 1)
    return i // (L + 1) if (L == 0 or i < head) else R + (i - head) // L


def segments(spid, L, R, ppt, reversed_order):
    """(tile, first pair, end pair, role, n_peers) in processing order."""
    lo = spid * L + min(spid, R)
    hi = (spid + 1) * L + min(spid + 1, R)
    out = []
    if reversed_order:
        cur = hi
        while cur > lo:
            t = (cur - 1) // ppt
            ts = t * ppt
            s = max(ts, lo)
            owner = cur == ts + ppt
            out.append((t, s, cur, "owner" if owner else "partial",
                        spid - owner_of(ts, L, R) if owner else 0))
            cur = s
    else:
        cur = lo
        while cur < hi:
            t = cur // ppt
            ts = t * ppt
            e = min(ts + ppt, hi)
            owner = cur == ts
            out.append((t, cur, e, "owner" if owner else "partial", 0))
            cur = e
    return out


def spread(L, R, ppt, reversed_order):
    """Lockstep model, one pair per time step. Returns the largest difference between the
    k-steps (within their tiles) read at the same time by the programs of one XCD, and the
    average number of distinct k-steps they read at the same time (fewer means more of them
    can share A/B slices in L2)."""
    reads = {}
    for spid in range(P):
        tau = 0
        for t, s, e, _, _ in segments(spid, L, R, ppt, reversed_order):
            for pair in range(s, e):
                reads.setdefault((spid // (P // X), tau), []).append((pair - t * ppt) * 2)
                tau += 1
    full = [v for v in reads.values() if len(v) == P // X]
    return (max(max(v) - min(v) for v in reads.values()),
            sum(len(set(v)) for v in full) / len(full))


def check(M, N, K):
    total = (M // 256) * (N // 256)
    S = total % P
    sk = S + P
    ppt = K // 128
    L, R = (sk * ppt) // P, (sk * ppt) % P
    covered = [0] * (sk * ppt)
    owners = {}
    for spid in range(P):
        segs = segments(spid, L, R, ppt, True)
        partials = [i for i, sg in enumerate(segs) if sg[3] == "partial"]
        assert partials in ([], [0]), (M, N, K, spid, segs)
        for t, s, e, role, n_peers in segs:
            for pair in range(s, e):
                covered[pair] += 1
            if role == "owner":
                assert t not in owners, (M, N, K, t)
                owners[t] = (spid, s, n_peers)
    assert all(c == 1 for c in covered), (M, N, K)
    assert sorted(owners) == list(range(sk)), (M, N, K)
    for t, (spid, s, n_peers) in owners.items():
        holders = {owner_of(pair, L, R) for pair in range(t * ppt, t * ppt + ppt)}
        peers = holders - {spid}
        assert n_peers == len(peers) <= 1, (M, N, K, t, holders)
        assert all(q == spid - 1 for q in peers), (M, N, K, t, holders)
    d = 2 * -(-(S * ppt) // P)
    return (S, total // P, d) + spread(L, R, ppt, True) + spread(L, R, ppt, False)


def main():
    import regress

    shapes = []
    for sh in regress.STREAMK_SHAPES + regress.LEFTOVER_SWEEP_SHAPES:
        total = (sh[0] // 256) * (sh[1] // 256)
        if sh not in shapes and total % P and total >= P:
            shapes.append(sh)
    shapes.sort(key=lambda sh: ((sh[0] // 256) * (sh[1] // 256) % P, sh[2]))
    print("| shape | leftover tiles S | full waves | lag d (k-steps) | XCD spread, reversed | "
          "XCD spread, forward | distinct k per XCD, reversed | distinct k per XCD, forward |")
    print("|---|---|---|---|---|---|---|---|")
    for M, N, K in shapes:
        S, waves, d, rev, rev_n, fwd, fwd_n = check(M, N, K)
        assert rev <= d, (M, N, K, rev, d)
        print(f"| {M}x{N}x{K} | {S} | {waves} | {d} | {rev} | {fwd} | {rev_n:.1f} | {fwd_n:.1f} |")
    print(f"\nall checks passed for {len(shapes)} two-tile shapes")


if __name__ == "__main__":
    main()
