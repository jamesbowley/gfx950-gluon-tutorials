"""The tile choice on shapes timed at every spec, with the formula giving the plan.

    python experiments/streamk_v31_rule/tile_choice.py --rule results/formula_rule.json \\
        --fit 'results/tilesweep_r*.csv' 'results/joint_train_r*.csv' \\
        --test 'results/heldout_r*.csv' 'results/joint_heldout_r*.csv' --out results/tile_choice.md

Per shape, time over the best of every (spec, method, n) timed:
- pick_gemm_tile, dp: AITER's tile model and no Stream-K;
- pick_gemm_tile, formula: its tile with the formula's plan;
- aware pick, formula: the lowest cost over every spec, each spec's work over its
  measured full-wave TFLOPS per wave, the tail priced by the formula's plan (a split
  tail 1/n of a wave plus F / k, a data-parallel partial wave tail + (1 - tail) S / P,
  `below_tail` below one wave); F, tail and below_tail fitted on --fit shapes;
- best spec, formula: the fastest spec when every spec runs the formula's plan (what a
  perfect tile choice would give);
- best spec, dp.
"""

import argparse
import collections
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fit_formula as ff  # noqa: E402
import grid  # noqa: E402

P = grid.P


def by_shape(blocks):
    out = collections.defaultdict(dict)
    for (spec, M, N, K), b in blocks.items():
        out[(M, N, K)][spec] = b
    return out


def plan(b, rule):
    form, c = rule["form"], rule["constants"]
    return ff.formula(b["info"], c, form, {}, 1.0)


def plan_time(b, rule):
    kind, n = plan(b, rule)
    return b["times"]["dp"] if kind == "dp" else b["times"].get(f"{kind}{n}")


def aware_cost(b, rule, tf, F, tail, below_tail):
    i = b["info"]
    bm, bn, bk = grid.parse_tile(i["spec"])
    kind, n = plan(b, rule)
    if i["S"] == 0:
        waves = i["W"]
    elif kind != "dp":
        waves = i["W"] + 1 / n + F / i["k"]
    else:
        t = below_tail if i["T"] < P else tail
        waves = i["W"] + t + (1 - t) * i["S"] / P
    return waves * bm * bn * bk * i["k"] / tf[i["spec"]]


def aware_pick(specs, rule, tf, F, tail, below_tail):
    costs = {s: aware_cost(b, rule, tf, F, tail, below_tail) for s, b in specs.items() if s in tf}
    best = min(costs.values())
    order = sorted(costs, key=lambda s: grid.ALL_SPECS.index(s))
    return next(s for s in order if costs[s] <= 1.01 * best)


def regret_of(specs, spec, rule):
    best = min(b["best"] for b in specs.values())
    t = plan_time(specs[spec], rule) if spec in specs else None
    return None if t is None else t / best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rule", required=True)
    ap.add_argument("--fit", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()
    rule = json.load(open(args.rule))
    tf = rule["spec_tflops"]
    train, test = by_shape(ff.load(args.fit)), by_shape(ff.load(args.test))
    train = {k: v for k, v in train.items() if len(v) >= 2}
    best = None
    for F in range(0, 25, 2):
        for tail in (0.3, 0.4, 0.5, 0.6, 0.7):
            for below in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
                r = [regret_of(s, aware_pick(s, rule, tf, F, tail, below), rule) for s in train.values()]
                r = [x for x in r if x]
                o = sum(math.log(x) for x in r) / len(r)
                if best is None or o < best[0]:
                    best = (o, F, tail, below)
    _, F, tail, below = best
    full = {k: v for k, v in test.items() if len(v) == len(grid.ALL_SPECS)}
    rows = collections.defaultdict(list)
    for (M, N, K), specs in full.items():
        overall = min(b["best"] for b in specs.values())
        bm, bn = grid.pick_gemm_tile(M, N)
        pick = grid.model_tile_spec(bm, bn)
        rows["pick_gemm_tile, dp"].append(specs[pick]["times"]["dp"] / overall)
        rows["pick_gemm_tile, formula"].append(regret_of(specs, pick, rule))
        rows["aware pick, formula"].append(regret_of(specs, aware_pick(specs, rule, tf, F, tail, below), rule))
        rows["best spec, formula"].append(min(x for x in (regret_of(specs, s, rule) for s in specs) if x))
        rows["best spec, dp"].append(min(b["times"]["dp"] for b in specs.values()) / overall)
    lines = ["# Tile choice with the formula's plan", "",
             f"{len(full)} held-out shapes timed at all {len(grid.ALL_SPECS)} specs; aware pick fitted on "
             f"{len(train)} training shapes with 2+ specs: F = {F} K-steps, tail = {tail}, below one wave {below}.", "",
             "Time over the shape's best (every spec, method and n), geomean / median / 90th / worst:", "",
             "| choice | result |", "|---|---|"]
    for name, r in rows.items():
        lines.append(f"| {name} | {ff.summary([x for x in r if x])} |")
    text = "\n".join(lines)
    print(text)
    if args.out:
        open(args.out, "w").write(text + "\n")


if __name__ == "__main__":
    main()
