"""A Stream-K-aware tile pick: pick_gemm_tile's cost model with the tail of a tile
that the rule splits costing 1/n of a wave plus F k-steps (F per regime), instead of
TAIL_WAVE_COST plus its fill, and a data-parallel launch below one wave (every tile
on its own CU, nothing after it) costing `below_tail` plus its fill.

    python experiments/streamk_v31_rule/tile_pick.py --rule results/rule_safe.json \\
        --fit results/grid_r1.csv results/grid_r2.csv --test results/heldout_r1.csv results/heldout_r2.csv

F is fitted on the sweep's shapes timed at two or more tiles (the default pipeline's
time over the shape's best, geometric mean), then scored on the held-out shapes.
"""

import argparse
import collections
import csv
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import candidates  # noqa: E402
import grid  # noqa: E402
from evaluate import kind_of, summary  # noqa: E402

F_VALUES = tuple(range(0, 41, 2))
BELOW_TAIL = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def load(paths):
    t = collections.defaultdict(list)
    for path in paths:
        for r in csv.DictReader(open(path)):
            if r["error"] or r["correct"] != "True" or r["kind"] == "pdp":
                continue
            t[(int(r["M"]), int(r["N"]), int(r["K"]))].append((r["tile"], r["kind"], float(r["us_median"])))
    shapes = {}
    for key, rows in t.items():
        per = collections.defaultdict(list)
        for tile, kind, us in rows:
            per[(tile, kind)].append(us)
        shapes[key] = {k: statistics.median(v) for k, v in per.items()}
    return shapes


def tile_spec(shape_tiles, bm, bn):
    """The timed spec of (bm, bn) for a shape (the sweep has 128x128 at two BLOCK_Ks)."""
    specs = sorted(t for t in shape_tiles if t.startswith(f"{bm}x{bn}x"))
    model = grid.model_tile_spec(bm, bn)
    return model if model in specs else specs[0]


def tile_cost(M, N, K, spec, rule, F):
    bm, bn, bk = (int(x) for x in spec.split("x"))
    tflops = dict(zip(grid.MODEL_TILES, grid.MODEL_TFLOPS))[(bm, bn)]
    T = grid.cdiv(M, bm) * grid.cdiv(N, bn)
    W, S = divmod(T, grid.P)
    cfg = candidates.streamk_default(M, N, K, (bm, bn, bk), grid.P, rule)
    if S == 0:
        waves = W
    elif cfg["STREAMK_POLICY"] == "one_tile" and F is not None:
        k = grid.cdiv(K, 2 * bk) * 2
        waves = W + 1 / cfg["SPLIT_N"] + F["below" if T < grid.P else "after"] / k
    else:
        tail = F["below_tail"] if F is not None and T < grid.P else grid.TAIL_WAVE_COST
        waves = W + tail + (1 - tail) * S / grid.P
    return waves * bm * bn / tflops


def pick(M, N, K, specs, rule, F):
    """Lowest cost among the timed specs (in MODEL_TILES order), 1% ties to the
    earlier, larger tile; F None is pick_gemm_tile's model."""
    order = sorted(specs, key=lambda s: grid.MODEL_TILES.index(tuple(int(x) for x in s.split("x")[:2])))
    costs = {s: tile_cost(M, N, K, s, rule, F) for s in order}
    best = min(costs.values())
    return next(s for s in order if costs[s] <= 1.01 * best)


def score(shapes, rule, F):
    out = []
    for (M, N, K), d in shapes.items():
        tiles = {t for t, _ in d}
        specs = {tile_spec(tiles, *(int(x) for x in t.split("x")[:2])) for t in tiles}
        if len({s.rsplit("x", 1)[0] for s in specs}) < 2:
            continue
        spec = pick(M, N, K, specs, rule, F)
        cfg = candidates.streamk_default(M, N, K, tuple(int(x) for x in spec.split("x")), grid.P, rule)
        t = d.get((spec, kind_of(cfg)))
        if t is not None:
            out.append(t / min(d.values()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rule", required=True)
    ap.add_argument("--fit", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()
    rule = json.load(open(args.rule))
    train, test = load(args.fit), load(args.test)
    best = None
    for bt in BELOW_TAIL:
        for fb in F_VALUES:
            for fa in F_VALUES:
                F = {"below": fb, "after": fa, "below_tail": bt}
                r = score(train, rule, F)
                o = sum(math.log(x) for x in r) / len(r)
                if best is None or o < best[0] - 1e-12:
                    best = (o, F)
    F = best[1]
    lines = [f"# Stream-K-aware tile pick (rule {args.rule})", "",
             f"Fitted on {', '.join(args.fit)} (shapes timed at 2+ tiles): {json.dumps(F)} "
             "(below / after: k-steps per split tail; below_tail: a dp launch's partial wave below one wave).", "",
             "Default pipeline (tile pick, then the rule) time over the shape's best among its timed tiles "
             "(geomean / median / 90th / worst):", "",
             "| set | pick_gemm_tile + rule | Stream-K-aware pick + rule |", "|---|---|---|"]
    for name, shapes in (("sweep (fit)", train), ("held-out AITER shapes", test)):
        base, aware = score(shapes, rule, None), score(shapes, rule, F)
        lines.append(f"| {name} ({len(aware)}) | {summary(base)} | {summary(aware)} |")
    text = "\n".join(lines)
    print(text)
    if args.out:
        open(args.out, "w").write(text + "\n")
        json.dump(F, open(os.path.splitext(args.out)[0] + ".json", "w"))


if __name__ == "__main__":
    main()
