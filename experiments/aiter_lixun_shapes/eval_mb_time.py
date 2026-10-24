#!/usr/bin/env python3
"""A closed-form memory-bound tile + split rule: least estimated time from traffic and padded MFMA work, CPU only.

    python eval_mb_time.py [--out results/mb_time_eval.md]

For every (tile, split s) of REFIT's 26 tiles with s <= s_max = min(P // T, K // K_MIN, pairs):
    est = (traffic / BW + 2 Mpad Npad K / RATE) / busy + SPLIT_US [s > 1]
    traffic = 2 M K ceil(N / BN) + 2 N K ceil(M / BM) + 2 M N + 8 (s - 1) T BM BN bytes (eval_mb_traffic.traffic)
    Mpad, Npad = M, N rounded up to the tile; busy = min(256, T s) / 256; BW = 5 TB/s
the least est, ties within 2% to the least padded work, then the fewer splits. The split runs as the §17 table's at
that tile (bracket lookup), cap without a table. Fitted (K_MIN, RATE, SPLIT_US) on REFIT's 152 memory-bound shapes,
scored leaving each shape out, and on §17 tilesel's small-M (M = 1-448) and small-N shapes, held out.
"""

import argparse
import itertools
import os
from collections import Counter

import eval_joint_rule as e
import eval_joint_rule_mb as m
import eval_mb_traffic as tr

BW = 5e12


def rule(M, N, K, k_min, rate, split_us, tie=0.02):
    cands = []
    for t in m.TILES:
        bm, bn, bk = e.dims(t)
        T = -(-M // bm) * -(-N // bn)
        mpad, npad = -(-M // bm) * bm, -(-N // bn) * bn
        smax = 1 if T >= m.P_of(t) else max(1, min(m.P_of(t) // T, K // k_min, -(-K // (2 * bk))))
        for s in range(1, smax + 1):
            busy = min(256, T * s) / 256
            est = (tr.traffic(M, N, K, t, s) / BW + 2 * mpad * npad * K / rate) / busy * 1e6 + (split_us if s > 1 else 0)
            cands.append((est, mpad * npad, s, t))
    best = min(c[0] for c in cands)
    _, _, s, t = min((c for c in cands if c[0] <= best * (1 + tie)), key=lambda c: (c[1], c[2]))
    return t, s


def picker(*p):
    def pick(M, N, K):
        t, _ = rule(M, N, K, *p)
        return t, [m.table_kind_mb(M, N, K, t), "cap"]
    return pick


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(e.HERE, "results", "mb_time_eval.md"))
    args = ap.parse_args()
    mb, ts = tr.load_refit_mb(), tr.load_tilesel()
    grid = list(itertools.product((512, 1024, 2048), (0.5e15, 1e15, 2e15, 4e15, 8e15), (0, 1, 2, 4)))
    score = lambda pts, p: m.gm(tr.regret(pts, picker(*p))[0])
    fits = sorted(((score(mb, p), p) for p in grid), key=lambda x: x[0])
    best = fits[0][1]
    loo = []
    for sh in mb:
        rest = {s: d for s, d in mb.items() if s != sh}
        loo += tr.regret({sh: mb[sh]}, picker(*min(grid, key=lambda p: score(rest, p))))[0]

    def best_tile_table(pts):
        def f(M, N, K):
            d = pts[(M, N, K)]
            t = min(m.TILES, key=lambda t: min([v for (tt, _), v in d.items() if tt == t] or [1e30]))
            return t, [m.table_kind_mb(M, N, K, t), "cap"]
        return f

    cur = lambda M, N, K: ("256x256x64", ["cap"])
    L = ["# Memory-bound tile + split by estimated time (traffic + padded MFMA work)", "",
         "Regret: median / 90th / worst / geomean over the best point timed per shape, shapes above 1.05 / 1.15.", "",
         "Top fits on REFIT memory-bound: " + "; ".join(f"{g:.3f} (K_MIN {p[0]}, RATE {p[1]:.0e}, SPLIT_US {p[2]})"
                                                      for g, p in fits[:5]), "",
         "| set | picker | shapes | regret | table kind not timed (cap) |", "|---|---|---:|---|---:|"]
    L.append(f"| REFIT MB (152) | 256x256x64 + cap (current) | 152 | {m.stats(tr.regret(mb, cur)[0])} | |")
    xs, fb = tr.regret(mb, picker(*best))
    L.append(f"| REFIT MB | rule, in sample | {len(xs)} | {m.stats(xs)} | {fb} |")
    L.append(f"| REFIT MB | rule, leave one shape out | {len(loo)} | {m.stats(loo)} | |")
    L.append(f"| REFIT MB | best tile + table split | 152 | {m.stats(tr.regret(mb, best_tile_table(mb))[0])} | |")
    for g, pts in sorted(ts.items()):
        L.append(f"| tilesel {g} ({len(pts)}) | 256x256x64 + cap (current) | {len(pts)} | {m.stats(tr.regret(pts, cur)[0])} | |")
        xs, fb = tr.regret(pts, picker(*best))
        L.append(f"| tilesel {g} | rule (REFIT-fitted, held out) | {len(xs)} | {m.stats(xs)} | {fb} |")
        L.append(f"| tilesel {g} | best tile + table split | {len(pts)} | {m.stats(tr.regret(pts, best_tile_table(pts))[0])} | |")
    picks = Counter()
    for sh in mb:
        t, s = rule(*sh, *best)
        picks[(t, s > 1)] += 1
    L += ["", "For comparison: REFIT nearest neighbours left one shape out 1.051 (WORKLOG §15); §17's tree fitted to small "
          "shapes 1.101 cross-validated; the fill rule 1.66 (eval_joint_rule_mb.py); least traffic with a fill floor "
          "1.162 (eval_mb_traffic.py).", "",
          "Tilesel shapes are scored only where the rule's tile was timed there.", "",
          "Tiles the rule picks on REFIT MB (tile, split): "
          + ", ".join(f"{t} {'split' if s else 'dp'} {c}" for (t, s), c in picks.most_common(14))]
    with open(args.out, "w") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
