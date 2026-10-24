#!/usr/bin/env python3
"""A memory-bound tile + split rule by least memory traffic, CPU only.

    python eval_mb_traffic.py [--out results/mb_traffic_eval.md]

Among every (tile, split s) of REFIT's 26 tiles with s <= s_max = min(P // T, K // K_MIN, pairs) whose fill reaches
FILL, the one with the least modelled traffic:
    A re-read per column of tiles + B re-read per row of tiles + C + the split's fp32 partials,
    2 M K ceil(N / BN) + 2 N K ceil(M / BM) + 2 M N + 8 (s - 1) T BM BN bytes,
ties within TIE to the fewer splits, then the larger tile. Fill is busy CUs, min(256, T s) / 256 ("cu"), or programs
over the tile's P ("prog"). The split runs as the §17 table's at that tile (bracket lookup; cap without a table), or
as rs{s} ("own").

Data: REFIT's 152 memory-bound shapes (M = 1-256, every tile x every configuration; fitting, and leave one shape out),
and §17 tilesel's small-M (100, M = 1-448) and small-N (60) shapes (each tile at its table pick, runner-up and cap, one
session; held out). A tilesel pick that was not timed falls back to cap at its tile (counted).
"""

import argparse
import csv
import itertools
import json
import math
import os
import statistics
from collections import Counter

import eval_joint_rule as e
import eval_joint_rule_mb as m

SPLIT = os.path.join(e.A16R, "split")


def traffic(M, N, K, t, s):
    bm, bn, _ = e.dims(t)
    T = -(-M // bm) * -(-N // bn)
    return (2 * M * K * -(-N // bn) + 2 * N * K * -(-M // bm) + 2 * M * N
            + (8 * (s - 1) * T * bm * bn if s > 1 else 0))


def rule(M, N, K, k_min, fill, mode="cu", tie=0.02):
    cands = []
    for t in m.TILES:
        bm, bn, bk = e.dims(t)
        T = -(-M // bm) * -(-N // bn)
        P = m.P_of(t)
        smax = 1 if T >= P else max(1, min(P // T, K // k_min, -(-K // (2 * bk))))
        for s in range(1, smax + 1):
            f = min(256, T * s) / 256 if mode == "cu" else min(1.0, T * s / P)
            cands.append((traffic(M, N, K, t, s), s, -e.area(t), t, f))
    ok = [c for c in cands if c[4] >= fill]
    if not ok:
        top = max(c[4] for c in cands)
        ok = [c for c in cands if c[4] >= top - 1e-9]
    best = min(c[0] for c in ok)
    near = [c for c in ok if c[0] <= best * (1 + tie)]
    tr, s, _, t, _ = min(near, key=lambda c: (c[1], c[2]))
    return t, s


def kinds(M, N, K, t, s, split):
    if split == "own":
        bm, bn, _ = e.dims(t)
        T = -(-M // bm) * -(-N // bn)
        if s == 1:
            return ["dp", "cap"]
        return ([f"rsall"] if s >= m.P_of(t) // T else []) + [f"rs{n}" for n in sorted(
            (n for n in e.NS if n <= s), reverse=True)] + ["cap"]
    return [m.table_kind_mb(M, N, K, t), "cap"]


def load_refit_mb():
    mb = {}
    for r in csv.DictReader(open(os.path.join(e.A16R, "refit.csv"))):
        if r["correct"] == "True" and int(r["M"]) < 512:
            mb.setdefault((int(r["M"]), int(r["N"]), int(r["K"])), {})[(r["tile"], r["kind"])] = float(r["us_median"])
    return mb


def load_tilesel():
    """{group: {shape: {(tile, kind): us}}} for tilesel's small-M and small-N shapes (median over T rounds)."""
    groups = {tuple(int(v) for v in k.split("x")) if isinstance(k, str) else tuple(k): g
              for k, g in json.load(open(os.path.join(SPLIT, "tilesel_blocks.json")))["groups"].items()}
    raw = {}
    for rnd in (1, 2, 3):
        path = os.path.join(SPLIT, f"T_r{rnd}.csv")
        if not os.path.exists(path):
            continue
        for r in csv.DictReader(open(path)):
            if r["correct"] != "True" or r.get("is_ref") == "1":
                continue
            raw.setdefault(((int(r["M"]), int(r["N"]), int(r["K"])), r["tile"], r["kind"]), []).append(
                float(r["us_median"]))
    out = {}
    for (sh, t, k), v in raw.items():
        g = groups.get(sh)
        if g in ("small-M", "small-N"):
            out.setdefault(g, {}).setdefault(sh, {})[(t, k)] = statistics.median(v)
    return out


def regret(pts, pick):
    out, fell = [], 0
    for sh, d in pts.items():
        t, ks = pick(*sh)
        for i, k in enumerate(ks):
            if k and (t, k) in d:
                out.append(d[(t, k)] / min(d.values()))
                fell += k == "cap" and ks[0] != "cap"
                break
        else:
            at = [v for (tt, _), v in d.items() if tt == t]
            if at:
                out.append(min(at) / min(d.values()))
                fell += 1
    return out, fell


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(e.HERE, "results", "mb_traffic_eval.md"))
    args = ap.parse_args()
    mb = load_refit_mb()
    ts = load_tilesel()
    grid = list(itertools.product((256, 512, 1024, 2048, 4096), (0.125, 0.25, 0.375, 0.5, 0.75, 1.0), ("cu", "prog"),
                                  ("table", "own")))

    def picker(k_min, fill, mode, split):
        return lambda M, N, K: (lambda ts_: (ts_[0], kinds(M, N, K, ts_[0], ts_[1], split)))(
            rule(M, N, K, k_min, fill, mode))

    score = lambda pts, p: m.gm(regret(pts, picker(*p))[0])
    fits = sorted(((score(mb, p), p) for p in grid), key=lambda x: x[0])
    best = fits[0][1]
    L = ["# Memory-bound tile + split by least traffic", "",
         "Regret: median / 90th / worst / geomean over the best point timed per shape, shapes above 1.05 / 1.15.", "",
         "## Fit on REFIT memory-bound (152 shapes)", ""]
    for g, p in fits[:8]:
        L.append(f"- {p}: {g:.3f}")
    L += ["", f"Chosen: K_MIN {best[0]}, FILL {best[1]} ({best[2]}), split {best[3]}.", ""]

    # leave one out on REFIT
    loo = []
    for sh in mb:
        rest = {s: d for s, d in mb.items() if s != sh}
        p = min(grid, key=lambda p: score(rest, p))
        loo += regret({sh: mb[sh]}, picker(*p))[0]

    def best_tile_table(pts):
        def f(M, N, K):
            d = pts[(M, N, K)]
            t = min(m.TILES, key=lambda t: min([v for (tt, _), v in d.items() if tt == t] or [1e30]))
            return t, [m.table_kind_mb(M, N, K, t), "cap"]
        return f

    L += ["## Results", "", "| set | picker | shapes | regret | picks falling back to cap |", "|---|---|---:|---|---:|"]
    L.append(f"| REFIT MB | 256x256x64 + cap (current) | {len(mb)} | {m.stats(regret(mb, lambda M, N, K: ('256x256x64', ['cap']))[0])} | |")
    xs, fb = regret(mb, picker(*best))
    L.append(f"| REFIT MB | traffic rule, in sample | {len(xs)} | {m.stats(xs)} | {fb} |")
    L.append(f"| REFIT MB | traffic rule, leave one shape out | {len(loo)} | {m.stats(loo)} | |")
    xs, fb = regret(mb, best_tile_table(mb))
    L.append(f"| REFIT MB | best tile + table split | {len(xs)} | {m.stats(xs)} | {fb} |")
    for g, pts in sorted(ts.items()):
        L.append(f"| tilesel {g} | 256x256x64 + cap (current) | {len(pts)} | "
                 f"{m.stats(regret(pts, lambda M, N, K: ('256x256x64', ['cap']))[0])} | |")
        xs, fb = regret(pts, picker(*best))
        L.append(f"| tilesel {g} | traffic rule (REFIT-fitted, held out) | {len(xs)} | {m.stats(xs)} | {fb} |")
        xs, fb = regret(pts, best_tile_table(pts))
        L.append(f"| tilesel {g} | best tile + table split | {len(xs)} | {m.stats(xs)} | {fb} |")
    L += ["", "For comparison: REFIT nearest neighbours left one shape out 1.051 (WORKLOG §15); results/split/tile_rule.md's "
          "tree fitted to small shapes 1.101 cross-validated; the fill rule (eval_joint_rule_mb.py) 1.66.", ""]

    # what the rule picks
    picks = Counter()
    for sh in mb:
        t, s = rule(*sh, best[0], best[1], best[2])
        picks[(t, "split" if s > 1 else "dp")] += 1
    L.append("Tiles the rule picks on REFIT MB: " + ", ".join(f"{t} {k} {c}" for (t, k), c in picks.most_common(12)))
    with open(args.out, "w") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
