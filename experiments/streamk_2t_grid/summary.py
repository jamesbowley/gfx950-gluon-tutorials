"""Summarise results/matrix.csv: per (shape, K) the best of dp, one-tile, the default
plan, two-tile on 256 programs and two-tile on a reduced grid, and time against G.

    python experiments/streamk_2t_grid/summary.py > experiments/streamk_2t_grid/results/summary.md
"""

import csv
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from points import exact  # noqa: E402


def main():
    rows = [r for r in csv.DictReader(open(os.path.join(HERE, "results", "matrix.csv"))) if r["us_median"]]
    bad = [r["point"] for r in rows if r["correct"] != "True"]
    by = defaultdict(dict)
    for r in rows:
        by[(int(r["T"]), int(r["M"]), int(r["N"]), int(r["K"]))][r["kind"]] = r
    print(f"{len(rows)} points, {len(bad)} incorrect {bad[:5]}\n")
    print("Times in us; ratios are time / best of (dp, one-tile, default). "
          "b: the denominator of T / G (exact: every cut on a multiple of pairs / b).\n")
    print("| T | S | shape | K | dp | best 1T | default | 2t (256) | best 2tg (G, b) | 2t / base | 2tg / base |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    curves = {}
    ratios = []
    for (T, M, N, K), k in sorted(by.items()):
        us = {kind: float(r["us_median"]) for kind, r in k.items()}
        pairs = int(next(iter(k.values()))["pairs"])
        one = {kind: t for kind, t in us.items() if kind.startswith(("ow", "rs"))}
        b1 = min(one, key=one.get) if one else None
        base = min([us["dp"], us.get("def", math.inf)] + list(one.values()))
        g = {int(kind[3:]): t for kind, t in us.items() if kind.startswith("2tg")}
        if "2t" in us:
            g[256] = us["2t"]
        bg = min(g, key=g.get)
        bb, ex = exact(T, pairs, bg)
        curves[(T, M, N, K)] = g
        ratios.append(g[bg] / base)
        two = f"{us['2t']:.1f}" if "2t" in us else "-"
        r2 = f"{us['2t'] / base:.3f}" if "2t" in us else "-"
        print(f"| {T} | {T % 256 if T > 256 else '-'} | {M}x{N} | {K} | {us['dp']:.1f} | "
              f"{f'{one[b1]:.1f} ({b1})' if b1 else '-'} | {us.get('def', math.nan):.1f} | {two} | "
              f"{g[bg]:.1f} ({bg}, {bb}{'' if ex else ' inexact'}) | {r2} | {g[bg] / base:.3f} |")
    wins = sum(x < 1 for x in ratios)
    print(f"\nBest two-tile at any G beats the best of dp / one-tile / default on {wins} of "
          f"{len(ratios)} cells; geomean ratio {math.exp(sum(map(math.log, ratios)) / len(ratios)):.3f}.\n")
    print("## Time against G (us), each row normalised to its G = 256 or largest G\n")
    for key, g in sorted(curves.items()):
        T, M, N, K = key
        ref = g[max(g)]
        cells = " ".join(f"{G}:{g[G] / ref:.2f}" for G in sorted(g, reverse=True))
        print(f"- T={T} K={K}: {cells}")


if __name__ == "__main__":
    main()
