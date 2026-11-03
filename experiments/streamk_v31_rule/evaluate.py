"""Score the default (pick_gemm_tile, then the Stream-K rule) on the held-out AITER
shapes.

    python experiments/streamk_v31_rule/evaluate.py results/heldout_r1.csv results/heldout_r2.csv \\
        --rule results/rule.json --out results/heldout.md

Per shape, every timed point (the HELDOUT_TILES cheapest tiles under the tile model,
every valid dp / owner / rs configuration) gives the per-shape best. Reported, as
time over that best (median over rounds):
- default: pick_gemm_tile's tile and the rule's configuration;
- pick + dp: the picked tile without Stream-K;
- pick + best: the picked tile with its best configuration (the tile model's share
  of the loss);
- best tile, dp: the fastest data-parallel tile.
The best v31 time is also compared with AITER's best library on the shape
(supported_status_scoped.csv's aiter_pick_tflops, an earlier measurement): shapes
where v31 is far behind are outside the kernel's domain whatever the rule does.
"""

import argparse
import collections
import csv
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import candidates  # noqa: E402
import grid  # noqa: E402


def kind_of(cfg):
    if cfg["STREAMK_POLICY"] == "dp":
        return "dp"
    return ("rs" if cfg["STREAMK_FIXUP"] == "rs" else "ow") + str(cfg["SPLIT_N"])


def summary(rs):
    rs = sorted(rs)
    if not rs:
        return "-"
    q = lambda f: rs[min(len(rs) - 1, int(f * len(rs)))]
    return f"{statistics.geometric_mean(rs):.4f} / {q(0.5):.3f} / {q(0.9):.3f} / {rs[-1]:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--rule", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()
    rule = json.load(open(args.rule))
    times = collections.defaultdict(list)
    for path in args.csv:
        for r in csv.DictReader(open(path)):
            if r["error"] or r["correct"] != "True":
                continue
            times[(int(r["M"]), int(r["N"]), int(r["K"]))].append((r["tile"], r["kind"], float(r["us_median"])))
    lib = {}
    for r in csv.DictReader(open(grid.AITER_SHAPES_CSV)):
        if r["aiter_pick_tflops"]:
            key = (int(r["M"]), int(r["N"]), int(r["K"]))
            lib[key] = max(lib.get(key, 0.0), float(r["aiter_pick_tflops"]))
    rows = []
    for (M, N, K), ts in times.items():
        per = collections.defaultdict(list)
        for tile, kind, us in ts:
            per[(tile, kind)].append(us)
        med = {k: statistics.median(v) for k, v in per.items() if k[1] != "pdp"}
        best = min(med.values())
        bm, bn = grid.pick_gemm_tile(M, N)
        tile = grid.model_tile_spec(bm, bn)
        t3 = tuple(int(x) for x in tile.split("x"))
        cfg = candidates.streamk_default(M, N, K, t3, grid.P, rule)
        default = med.get((tile, kind_of(cfg)))
        pick_dp = med.get((tile, "dp"))
        pick_best = min((v for (t, k), v in med.items() if t == tile), default=None)
        best_dp = min((v for (t, k), v in med.items() if k == "dp"), default=None)
        cand = [kind_of(c) for c in candidates.streamk_candidates(M, N, K, t3, grid.P, rule)]
        tuned = min((med[(tile, k)] for k in cand if (tile, k) in med), default=None)
        T = grid.cdiv(M, bm) * grid.cdiv(N, bn)
        v31_tflops = 2 * M * N * K / (best * 1e-6) / 1e12
        rows.append(dict(shape=(M, N, K), regime="below" if T < grid.P else "after", best=best,
                         default=default, pick_dp=pick_dp, pick_best=pick_best, best_dp=best_dp,
                         tuned=tuned, kind=kind_of(cfg), tile=tile,
                         lib=(lib[(M, N, K)] / v31_tflops) if (M, N, K) in lib else None))
    lines = [f"# Held-out AITER shapes: {', '.join(args.csv)}, rule {args.rule}", "",
             f"{len(rows)} shapes. Time over the per-shape best, geomean / median / 90th percentile / worst.", "",
             "| set | default | pick + dp | pick + best | pick + tuned (pruned space) | best tile, dp |",
             "|---|---|---|---|---|---|"]
    for name, sel in (("all", lambda r: True), ("below one wave", lambda r: r["regime"] == "below"),
                      ("one wave or more", lambda r: r["regime"] == "after"),
                      ("v31 within 10% of AITER's best library", lambda r: r["lib"] is not None and r["lib"] <= 1.1)):
        rs = [r for r in rows if sel(r)]
        f = lambda key: summary([r[key] / r["best"] for r in rs if r[key]])
        lines.append(f"| {name} ({len(rs)}) | {f('default')} | {f('pick_dp')} | {f('pick_best')} | {f('tuned')} | {f('best_dp')} |")
    libs = sorted(r["lib"] for r in rows if r["lib"])
    if libs:
        lines += ["", f"AITER's best library / v31's best (TFLOPS ratio; above 1: the library is faster): "
                  f"median {statistics.median(libs):.3f}, 90th percentile {libs[int(0.9 * len(libs))]:.3f}; "
                  f"{sum(x > 1.1 for x in libs)} of {len(libs)} shapes more than 10% faster in the library."]
    lose = [r for r in rows if r["default"] and r["pick_dp"] and r["default"] > 1.01 * r["pick_dp"]]
    lines += ["", f"Default slower than the picked tile's dp by more than 1% on {len(lose)} shapes.", "",
              "## Worst 15 defaults", "", "| shape | regime | tile | rule | default / best | pick + dp / best |",
              "|---|---|---|---|---|---|"]
    for r in sorted((r for r in rows if r["default"]), key=lambda r: -r["default"] / r["best"])[:15]:
        lines.append(f"| {'x'.join(map(str, r['shape']))} | {r['regime']} | {r['tile']} | {r['kind']} | "
                     f"{r['default'] / r['best']:.3f} | {r['pick_dp'] / r['best']:.3f} |")
    text = "\n".join(lines)
    print(text)
    if args.out:
        open(args.out, "w").write(text + "\n")


if __name__ == "__main__":
    main()
