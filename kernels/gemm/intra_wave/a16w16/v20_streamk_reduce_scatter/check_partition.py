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
"""Check v20's end-to-end reversed partition in pure Python (no GPU), for both policies that
use it: two-tile (S + 256 stream-K tiles) and spread (S stream-K tiles, the experiment for
S > 128).

    python v20_streamk_reduce_scatter/check_partition.py

Replicates the kernel's partition (pairs of k-steps, L = SK_PAIRS // 256, R = SK_PAIRS % 256,
ranges in spid order) and its segment loop (from the end of the range to its start). For
every S from 1 to 255 and K from 8192 to 65536 (two-tile only where there is a wave to borrow,
which the check assumes), it checks:
- every pair is covered exactly once;
- every tile has one owner (the segment holding its last pair), and its peers are the
  contiguous lower spids holding the tile's other pairs, n_peers = spid - owner_of(tile start);
  two-tile has at most one peer, spread at most 2 for S > 128;
- every program has at most one partial, and it is the first segment it processes, so a
  program publishes before it waits and waits only point to lower spids.
The same checks run with fewer stream-K programs (STREAMK_NUM_PROGRAMS = SNP < 256) for spread
and two-tile, and streamk_compact_pid must map the active programs onto 0 .. SNP - 1, keeping
order and contiguity within each XCD, spread evenly over the XCDs.

It then prints, for a few spread cases, the average number of distinct k-steps an XCD's 32
programs read at the same time (lockstep model, one pair per time step) in reversed and in
forward (v16) order. q is the denominator of S / 256 in lowest terms, the period of the
pattern of start offsets.
"""

import math
import sys

P, X = 256, 8


def owner_of(i, L, R):
    head = R * (L + 1)
    return i // (L + 1) if (L == 0 or i < head) else R + (i - head) // L


def segments(spid, L, R, ppt, reversed_order):  # spid: the stream-K (compacted) id
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


def distinct_k(L, R, ppt, reversed_order):
    """Lockstep model: the average number of distinct k-steps (within their tiles) that the
    programs of one XCD read at the same time, over the steps where all of them are busy."""
    reads = {}
    for spid in range(P):
        tau = 0
        for t, s, e, _, _ in segments(spid, L, R, ppt, reversed_order):
            for pair in range(s, e):
                reads.setdefault((spid // (P // X), tau), []).append(pair - t * ppt)
                tau += 1
    full = [v for v in reads.values() if len(v) == P // X]
    return sum(len(set(v)) for v in full) / len(full)


def compact_pid(spid, snp, balance=True):
    """streamk_compact_pid in matmul_kernel.py: stream-K id, or P if the program idles."""
    if not balance:
        return spid if spid < snp else P
    per, base, tall = P // X, snp // X, snp % X
    xcd, local = spid // per, spid % per
    return xcd * base + min(xcd, tall) + local if local < base + (xcd < tall) else P


def check_compaction(snp):
    for balance in (True, False):
        cids = [compact_pid(spid, snp, balance) for spid in range(P)]
        active = [c for c in cids if c != P]
        assert active == list(range(snp)), (snp, balance)
    per = P // X
    counts = [sum(compact_pid(x * per + i, snp) != P for i in range(per)) for x in range(X)]
    assert max(counts) - min(counts) <= 1, (snp, counts)


def check(sk, ppt, max_peers, programs=P):
    L, R = (sk * ppt) // programs, (sk * ppt) % programs
    intervals, owners = [], {}
    for spid in range(programs):
        segs = segments(spid, L, R, ppt, True)
        partials = [i for i, sg in enumerate(segs) if sg[3] == "partial"]
        assert partials in ([], [0]), (sk, ppt, spid, segs)
        for t, s, e, role, n_peers in segs:
            intervals.append((s, e))
            if role == "owner":
                assert t not in owners, (sk, ppt, t)
                owners[t] = (spid, n_peers)
    intervals.sort()
    assert intervals[0][0] == 0 and intervals[-1][1] == sk * ppt, (sk, ppt)
    assert all(a[1] == b[0] for a, b in zip(intervals, intervals[1:])), (sk, ppt)
    assert sorted(owners) == list(range(sk)), (sk, ppt)
    for t, (spid, n_peers) in owners.items():
        # Ranges are in spid order, so a tile's holders are the contiguous spids from the holder
        # of its first pair to the holder of its last.
        first, last = owner_of(t * ppt, L, R), owner_of(t * ppt + ppt - 1, L, R)
        assert last == spid, (sk, ppt, t)
        assert n_peers == spid - first <= max_peers, (sk, ppt, t, first, spid)


def main():
    cases = 0
    for K in (8192, 16384, 32768, 65536):
        ppt = K // 128
        for S in range(1, P):
            check(S, ppt, 2 if 2 * S > P else P)
            check(S + P, ppt, 1)
            cases += 2
    print(f"all partition checks passed: {cases} cases (S = 1-255, two-tile and spread, K = 8192-65536)")
    for snp in range(1, P + 1):
        check_compaction(snp)
    sub = 0
    for K in (512, 1024, 8192):
        ppt = K // 128
        for S in range(1, P):
            for n in (1, 2, 3, 4, 6, 8):
                snp = S * n
                if snp <= P:
                    check(S, ppt, P, programs=snp)
                    check(S + P, ppt, P, programs=snp)
                    sub += 2
    print(f"compaction checks passed for SNP = 1-256; partition checks passed for {sub} cases with SNP = S * n "
          "(n = 1-8, K = 512-8192)\n")

    print("Spread: distinct k-steps an XCD's programs read at once (lockstep model)\n")
    print("| S | K | q | pairs per program | distinct k, reversed | distinct k, forward (v16) |")
    print("|---|---|---|---|---|---|")
    for S in (128, 160, 192, 200, 224, 240, 255):
        for K in (8192, 32768):
            ppt = K // 128
            L, R = (S * ppt) // P, (S * ppt) % P
            q = P // math.gcd(S, P)
            rev, fwd = distinct_k(L, R, ppt, True), distinct_k(L, R, ppt, False)
            print(f"| {S} | {K} | {q} | {L}{'+' if R else ''} | {rev:.1f} | {fwd:.1f} |", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
