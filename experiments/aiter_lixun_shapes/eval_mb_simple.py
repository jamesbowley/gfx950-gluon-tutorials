#!/usr/bin/env python3
"""A simple memory-bound tile rule, CPU only: the short side covers the small dimension (up to COVER), the long side
is the largest that still fills the tile's programs, and a split only when no tile fills.

    python eval_mb_simple.py

With a = min(M, N) (for N < M the shape is solved transposed and the tile spec transposed):
- short side: the smallest tile side in (1, 16, 32, 64, 128, 256) that is >= min(a, COVER);
- among tiles with that short side, the largest long side whose tiles alone fill FILL of the tile's programs P
  (1024 for 16x16 / 16x32 / 16x64 / 32x32 / 1x16, 512 for 32x128 / 16x128, 256 otherwise);
- if none does, the best fill with the largest split s = min(P // T, K // K_MIN), smaller long side first.
The split is the §17 table's at that tile. Fitted (FILL, K_MIN, COVER) on REFIT's 152 memory-bound shapes; scored on
them leaving each shape out, and on §17 tilesel's small-M and small-N shapes (held out).
Also scored: the first version ("cover M fully, largest long side that keeps half the CUs busy"), for comparison.
"""
import itertools

import eval_joint_rule as e
import eval_joint_rule_mb as m
import eval_mb_traffic as tr

SIDES = sorted({e.dims(t)[0] for t in m.TILES})


def transpose(t):
    a, b, k = t.split("x")
    return f"{b}x{a}x{k}"


def rule(M, N, K, fill, k_min, cover):
    bm = next((b for b in SIDES if b >= min(M, cover)), 256)
    c = []
    for t in m.TILES:
        b, bn, bk = e.dims(t)
        if b != bm:
            continue
        T = -(-M // b) * -(-N // bn)
        P = m.P_of(t)
        s = 1 if T >= P else max(1, min(P // T, K // k_min, -(-K // (2 * bk))))
        c.append((T / P, min(1.0, T * s / P), bn, bk, t))
    ok = [x for x in c if x[0] >= fill]
    if ok:
        return max(ok, key=lambda x: (x[2], x[3]))[4]
    return max(c, key=lambda x: (x[1], x[0], -x[2]))[4]


def rule_v1(M, N, K, fill, k_min):
    bm = next((b for b in SIDES if b >= min(M, 256)), 256)
    c = []
    for t in m.TILES:
        b, bn, bk = e.dims(t)
        if b != bm:
            continue
        T = -(-M // b) * -(-N // bn)
        s = 1 if T >= m.P_of(t) else max(1, min(m.P_of(t) // T, K // k_min, -(-K // (2 * bk))))
        c.append((min(256, T * s) / 256, bn, bk, t))
    ok = [x for x in c if x[0] >= fill] or [max(c)]
    return max(ok, key=lambda x: (x[1], x[0], x[2]))[3]


def picker(rule_fn, *p):
    def pick(M, N, K):
        if N < M:
            t = rule_fn(N, M, K, *p)
            return transpose(t), [m.table_kind_mb(N, M, K, t), "cap"]
        t = rule_fn(M, N, K, *p)
        return t, [m.table_kind_mb(M, N, K, t), "cap"]
    return pick


def evaluate(name, rule_fn, grid, mb, ts):
    sc = lambda pts, p: m.gm(tr.regret(pts, picker(rule_fn, *p))[0])
    fits = sorted((sc(mb, p), p) for p in grid)
    best = fits[0][1]
    loo = []
    for sh in mb:
        rest = {s: d for s, d in mb.items() if s != sh}
        loo += tr.regret({sh: mb[sh]}, picker(rule_fn, *min(grid, key=lambda p: sc(rest, p))))[0]
    print(f"## {name}: fitted {best} (top fits {[(round(g, 3), p) for g, p in fits[:3]]})")
    print("  REFIT MB, leave one out:", m.stats(loo))
    for g, pts in sorted(ts.items()):
        xs, fb = tr.regret(pts, picker(rule_fn, *best))
        print(f"  tilesel {g} (held out): {len(xs)} shapes {m.stats(xs)}, table kind not timed {fb}")


if __name__ == "__main__":
    mb, ts = tr.load_refit_mb(), tr.load_tilesel()
    evaluate("simple rule (cover min(M, N) up to COVER, fill the programs)", rule,
             list(itertools.product((0.5, 0.75, 1.0), (1024, 2048, 4096), (16, 32, 64, 128, 256))), mb, ts)
    evaluate("first version (cover M fully, largest long side keeping FILL of the CUs busy)", rule_v1,
             list(itertools.product((0.25, 0.5, 0.75, 1.0), (512, 1024, 2048, 4096))), mb, ts)
