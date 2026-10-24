#!/usr/bin/env python3
"""The joint tile + split rule on REFIT's memory-bound shapes (M = 1-256), CPU only.

    python eval_joint_rule_mb.py [--out results/joint_rule_eval_mb.md]

As eval_joint_rule.py's form A, over REFIT's 26 tiles, each on its own number of programs P (1024 for the 16x16 /
16x32 / 16x64 / 32x32 / 1x16 classes, 512 for 32x128 / 128x32 / 16x128, 256 for the rest): under one wave of the
tile's programs the largest split s = min(P // T, K // K_MIN, pairs); the largest-area tile whose T * s fills
FILL * P (else the best fill). The split is the §17 table's where the tile has one (bracket lookup), else cap; from a
wave up, cap. Fitted (K_MIN, FILL) by leaving each shape out of the fit, so the regret is out of sample.
"""

import argparse
import itertools
import math
import os
import statistics

import eval_joint_rule as e

SMALL1024 = ("16x16x256", "16x16x128", "1x16x256", "32x32x128", "16x32x128", "16x64x128")
SMALL512 = ("32x128x128", "128x32x128", "16x128x128", "16x128x64")
TILES = e.QUAD + SMALL1024 + SMALL512 + ("32x256x128", "16x256x128", "32x256x64")


def P_of(t):
    return 1024 if t in SMALL1024 else 512 if t in SMALL512 else 256


def rule_mb(M, N, K, k_min, fill):
    cands = []
    for t in TILES:
        bm, bn, bk = e.dims(t)
        T = -(-M // bm) * -(-N // bn)
        P = P_of(t)
        if T >= P:
            cands.append((t, None, 1.0))
            continue
        s = max(1, min(P // T, K // k_min, -(-K // (2 * bk))))
        cands.append((t, s, min(1.0, T * s / P)))
    good = [c for c in cands if c[2] >= fill]
    if good:
        big = max(e.area(c[0]) for c in good)
        return max((c for c in good if e.area(c[0]) == big), key=lambda c: (c[2], e.dims(c[0])[2]))[:2]
    return max(cands, key=lambda c: (c[2], e.area(c[0])))[:2]


def table_kind_mb(M, N, K, t):
    P = P_of(t)
    bm, bn, bk = e.dims(t)
    T = -(-M // bm) * -(-N // bn)
    waves = T // P
    wc = 0 if waves == 0 else (1 if waves < 4 else 4)
    grid = e.grid_for(t, wc)
    if grid is None:
        return None
    S, k = T % P, -(-K // (2 * bk)) * 2
    if S == 0:
        return "dp"
    Ss = sorted({s for s, _ in grid})
    lo = max((x for x in Ss if x <= S), default=Ss[0])
    hi = min((x for x in Ss if x >= S), default=Ss[-1])
    cells = []
    for s in {lo, hi}:
        ks = sorted({kk for s_, kk in grid if s_ == s})
        cells += [grid[(s, kk)] for kk in {max((x for x in ks if x <= k), default=ks[0]),
                                           min((x for x in ks if x >= k), default=ks[-1])}]
    common = set(cells[0]).intersection(*cells[1:])
    if not common:
        return None
    score = {kd: sum(math.log(c[kd]) for c in cells) for kd in common}
    best = min(score.values())
    tie = [kd for kd in common if score[kd] <= best + len(cells) * math.log(1 + e.TABLE_DELTA)]
    return min(tie, key=lambda kd: (e.SIMPLE.index(kd) if kd in e.SIMPLE else 99, score[kd]))


def regret(pts, pick):
    out = []
    for sh, d in pts.items():
        t, kinds = pick(*sh)
        for k in kinds:
            if k and (t, k) in d:
                out.append(d[(t, k)] / min(d.values()))
                break
        else:
            out.append(float("nan"))
    return [x for x in out if x == x]


def gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def stats(xs):
    xs = sorted(xs)
    return (f"{statistics.median(xs):.3f} / {xs[int(0.9 * (len(xs) - 1))]:.3f} / {xs[-1]:.3f} / {gm(xs):.3f}, "
            f"{sum(x > 1.05 for x in xs)} > 1.05, {sum(x > 1.15 for x in xs)} > 1.15")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(e.HERE, "results", "joint_rule_eval_mb.md"))
    args = ap.parse_args()
    mb = {}
    import csv
    for r in csv.DictReader(open(os.path.join(e.A16R, "refit.csv"))):
        if r["correct"] != "True" or int(r["M"]) >= 512:
            continue
        mb.setdefault((int(r["M"]), int(r["N"]), int(r["K"])), {})[(r["tile"], r["kind"])] = float(r["us_median"])

    grid = list(itertools.product((256, 512, 1024, 2048, 4096), (0.25, 0.375, 0.5, 0.625, 0.75, 0.875)))

    def picker(k_min, fill, split="table"):
        def pick(M, N, K):
            t, s = rule_mb(M, N, K, k_min, fill)
            if split == "cap":
                return t, ["cap"]
            return t, [table_kind_mb(M, N, K, t), "cap"]
        return pick

    # leave one shape out: fit (k_min, fill) on the other shapes, score the left-out one
    loo = []
    picks = []
    for sh in mb:
        rest = {s: d for s, d in mb.items() if s != sh}
        kb, fb = min(grid, key=lambda p: gm(regret(rest, picker(*p))))
        picks.append((kb, fb))
        loo += regret({sh: mb[sh]}, picker(kb, fb))
    full = min(grid, key=lambda p: gm(regret(mb, picker(*p))))

    def best_tile(M, N, K):
        d = mb[(M, N, K)]
        return min(TILES, key=lambda t: min([v for (tt, _), v in d.items() if tt == t] or [1e30]))

    L = [f"# Joint rule on REFIT's {len(mb)} memory-bound shapes (M = 1-256)", "",
         "Regret: median / 90th / worst / geomean over the best point timed per shape.", "",
         "| picker | regret |", "|---|---|",
         f"| 256x256x64 + cap (auto_tile's memory-bound branch, v26) | {stats(regret(mb, lambda M, N, K: ('256x256x64', ['cap'])))} |",
         f"| joint rule + table split, fitted on all ({full}, in sample) | {stats(regret(mb, picker(*full)))} |",
         f"| joint rule + table split, leave one shape out | {stats(loo)} |",
         f"| joint rule's tile + cap ({full}) | {stats(regret(mb, picker(*full, split='cap')))} |",
         f"| best tile + table split | {stats(regret(mb, lambda M, N, K: (best_tile(M, N, K), [table_kind_mb(M, N, K, best_tile(M, N, K)), 'cap'])))} |",
         f"| best tile + cap | {stats(regret(mb, lambda M, N, K: (best_tile(M, N, K), ['cap'])))} |",
         "", "For comparison (WORKLOG §15/§16): the REFIT nearest-neighbour rule left one shape out 1.051 geomean, worst "
         "1.235; the depth-5 tree of results/split/tile_rule.md on small shapes 1.07-1.09.", "",
         "Parameters chosen in the leave-one-out folds: "
         + ", ".join(f"{k} {c}" for k, c in sorted(__import__('collections').Counter(picks).items(), key=lambda x: -x[1])[:6])]
    with open(args.out, "w") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
