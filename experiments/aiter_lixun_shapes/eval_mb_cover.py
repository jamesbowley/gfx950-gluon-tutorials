#!/usr/bin/env python3
"""The "cover the small side, keep the long side large, split to half the CUs" memory-bound rule, CPU only.

    python eval_mb_cover.py

With a = min(M, N) (for N < M the shape is solved transposed and the tile spec transposed):
- short side BM: the smallest of (16, 32, 64, 128, 256) that covers a (1 for a = 1 with ONE_ROW);
- long side: 256 first, then 128, 64, 32, 16 (the tiles REFIT has for that BM): the first whose tiles T, split
  s = ceil(TARGET / T) ways (none if T >= TARGET), keep at least K_MIN of K per split, i.e. s <= K // K_MIN;
- if none can reach TARGET busy CUs, the tile with the most busy CUs, min(T s_max, 256) (ties: the larger long side).
The split runs as the one-tile reduce-scatter on s programs per tile (the timed n at or below s), or as the §17 table's
pick at that tile ("table"). Scored over REFIT's 152 memory-bound shapes (every tile x every configuration timed), with
the best point timed as the reference, and reported next to the time model (eval_mb_time.py) and the simple rule
(eval_mb_simple.py). Also: per shape, what the best point is, so the rule's misses can be read.
"""

import itertools
import math
from collections import Counter

import eval_joint_rule as e
import eval_joint_rule_mb as m
import eval_mb_traffic as tr

LONG = (256, 128, 64, 32, 16)


def tiles_for(bm):
    """{long side: spec} of REFIT's tiles with BM = bm (BK 128 preferred, else 64, else 256)."""
    out = {}
    for t in m.TILES:
        b, bn, bk = e.dims(t)
        if b == bm and (bn not in out or abs(bk - 128) < abs(e.dims(out[bn])[2] - 128)):
            out[bn] = t
    return out


def rule(M, N, K, target, k_min, one_row):
    a = M
    sides = ((1,) if one_row else ()) + (16, 32, 64, 128, 256)
    bm = next(b for b in sides if b >= min(a, 256))
    avail = tiles_for(bm)
    best = None
    for bn in LONG:
        t = avail.get(bn)
        if not t:
            continue
        T = -(-M // bm) * -(-N // bn)
        smax = max(1, K // k_min)
        s = 1 if T >= target else math.ceil(target / T)
        if s <= smax:
            return t, s
        busy = min(256, T * smax)
        if best is None or busy > best[0]:
            best = (busy, t, smax)
    return best[1], best[2]


def kinds(M, N, K, t, s, split):
    if split == "table":
        return [m.table_kind_mb(M, N, K, t), "cap"]
    if s <= 1:
        return ["dp", "cap"]
    return [f"rs{n}" for n in sorted((n for n in e.NS if n <= s), reverse=True)] + ["rsall", "cap"]


def transpose(t):
    a, b, k = t.split("x")
    return f"{b}x{a}x{k}"


def picker(target, k_min, one_row, split):
    def pick(M, N, K):
        if N < M:
            t, s = rule(N, M, K, target, k_min, one_row)
            return transpose(t), kinds(N, M, K, t, s, split)
        t, s = rule(M, N, K, target, k_min, one_row)
        return t, kinds(M, N, K, t, s, split)
    return pick


if __name__ == "__main__":
    mb, ts = tr.load_refit_mb(), tr.load_tilesel()
    grid = list(itertools.product((64, 128, 192, 256), (512, 1024, 2048, 4096), (False, True), ("own", "table")))
    sc = lambda pts, p: m.gm(tr.regret(pts, picker(*p))[0])
    fits = sorted((sc(mb, p), p) for p in grid)
    print("top fits (TARGET CUs, K_MIN, ONE_ROW, split):", [(round(g, 3), p) for g, p in fits[:6]])
    for p in ((128, 1024, False, "own"), (128, 1024, False, "table"), (128, 2048, True, "table")):
        print(f"  {p}: REFIT {sc(mb, p):.3f}")
    best = fits[0][1]
    xs, fb = tr.regret(mb, picker(*best))
    print("REFIT MB (in sample at the fitted point):", m.stats(xs), "| fallbacks", fb)
    loo = []
    for sh in mb:
        rest = {s: d for s, d in mb.items() if s != sh}
        loo += tr.regret({sh: mb[sh]}, picker(*min(grid, key=lambda p: sc(rest, p))))[0]
    print("REFIT MB leave one out:", m.stats(loo))
    for g, pts in sorted(ts.items()):
        xs, fb = tr.regret(pts, picker(*best))
        print(f"tilesel {g} (held out): {len(xs)} shapes {m.stats(xs)}, table kind not timed {fb}")
    # the misses
    rows = []
    for sh, d in mb.items():
        t, ks = picker(*best)(*sh)
        k = next((k for k in ks if k and (t, k) in d), None)
        if k:
            bt, bk = min(d, key=d.get)
            rows.append((d[(t, k)] / min(d.values()), sh, t, k, bt, bk))
    rows.sort(reverse=True)
    print("worst misses (regret, shape, rule tile/kind, best tile/kind):")
    for r in rows[:15]:
        print(f"  {r[0]:.2f} {r[1]} rule {r[2]} {r[3]} | best {r[4]} {r[5]}")
    lost = [r for r in rows if r[0] > 1.15]
    print(f"misses > 1.15: {len(lost)}; their K: {Counter('<=2048' if r[1][2] <= 2048 else '<=8192' if r[1][2] <= 8192 else '>8192' for r in lost)}; "
          f"best tile programs per CU: {Counter(m.P_of(r[4]) // 256 for r in lost)}")
