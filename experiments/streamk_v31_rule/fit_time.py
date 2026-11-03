"""A time model of the v31 kernel, and the default it gives: the (tile spec, method,
split) of the lowest predicted time.

    python experiments/streamk_v31_rule/fit_time.py --train results/grid_r*.csv results/tilesweep_r*.csv \\
        results/gap_r*.csv results/joint_train_r*.csv --test results/heldout_r*.csv results/joint_heldout_r*.csv \\
        --out results/time_model.md --model-out results/time_model.json

Per tile spec s (BLOCK_M x BLOCK_N x BLOCK_K) a K-step takes t_ks[s] us on a program of a
full wave. A shape has T tiles, W = T // P full waves, S = T % P tail tiles, k K-steps
per tile (pairs = k / 2), and each split program's partial is B = BLOCK_M BLOCK_N 4
bytes (in MB below). Predicted times, per regime r (below one wave: T < P; else after):

    dp      = W k t_ks + [S > 0] (a_r + (1 - a_r) S / P) k t_ks
    rs on n = W k t_ks + c_rs,r pairs 2 t_ks / n + f_rs,r + q_rs,r S n B
    owner n = W k t_ks + c_ow,r pairs 2 t_ks / n + f_ow,r + q_ow,r (n - 1) B

dp's tail is a partial wave; the reduce-scatter's cost grows with the S n partials all
programs store and read at once (the chip's bandwidth), the owner's with its n - 1
serial reads (one CU's). t_ks and a come from the dp times (t_ks the median of the
implied K-step time per spec); the split terms, given t_ks, are a robust linear fit
of each method's times. Scored on the test shapes, where every spec and every valid
point is timed: the model's choice over every spec, method and n, against the
shape's best measured time.
"""

import argparse
import collections
import csv
import glob
import json
import math
import os
import statistics
import sys

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import grid  # noqa: E402

P = grid.P
OWNER_N = (2, 3, 4, 6, 8, 12, 16)
RS_N = (2, 4, 6, 8, 12, 16, 32, 64)


def load(patterns):
    t = collections.defaultdict(list)
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            for r in csv.DictReader(open(path)):
                if r["error"] or r["correct"] != "True" or r["kind"] == "pdp" or r["exchange"] == "l2":
                    continue
                if r["policy"] in ("two_tile", "spread"):
                    continue
                t[r["point"]].append(float(r["us_median"]))
    shapes = collections.defaultdict(dict)
    for p, v in t.items():
        kind, tile, M, N, K = grid.parse_point(p)
        shapes[(M, N, K)][(tile, kind)] = statistics.median(v)
    return shapes


def geometry(spec, M, N, K):
    bm, bn, bk = grid.parse_tile(spec)
    T = grid.cdiv(M, bm) * grid.cdiv(N, bn)
    k = grid.cdiv(K, 2 * bk) * 2
    return dict(T=T, W=T // P, S=T % P, k=k, pairs=k // 2, mb=bm * bn * 4 / 1e6,
                regime="below" if T < P else "after")


def method(kind):
    if kind == "dp":
        return "dp", 1
    return ("rs", int(kind[2:])) if kind.startswith("rs") else ("ow", int(kind[2:]))


def dp_waves(g, a):
    if g["S"] == 0:
        return g["W"]
    return g["W"] + a[g["regime"]] + (1 - a[g["regime"]]) * g["S"] / P


def predict(model, spec, kind, g):
    t_ks = model["t_ks"][spec]
    m, n = method(kind)
    if m == "dp":
        return dp_waves(g, model["a"]) * g["k"] * t_ks
    c, f, q = model[m][g["regime"]]
    base = g["W"] * g["k"] * t_ks + c * g["pairs"] * 2 * t_ks / n + f
    return base + q * (g["S"] * n * g["mb"] if m == "rs" else (n - 1) * g["mb"])


def fit_dp(shapes):
    """t_ks per spec and the tail weights a (grid search) from the dp times."""
    rows = [(spec, geometry(spec, *mnk), t) for mnk, d in shapes.items() for (spec, kind), t in d.items() if kind == "dp"]
    best = None
    for ab in np.arange(0.3, 1.01, 0.05):
        for aa in np.arange(0.3, 1.01, 0.05):
            a = {"below": ab, "after": aa}
            t_ks = {}
            for spec in grid.ALL_SPECS:
                v = [t / (dp_waves(g, a) * g["k"]) for s, g, t in rows if s == spec]
                if v:
                    t_ks[spec] = statistics.median(v)
            err = statistics.mean(abs(math.log(t / (dp_waves(g, a) * g["k"] * t_ks[s]))) for s, g, t in rows)
            if best is None or err < best[0]:
                best = (err, a, t_ks)
    return best[2], {k: round(float(v), 3) for k, v in best[1].items()}, best[0]


def fit_split(shapes, t_ks, m):
    out = {}
    for regime in ("below", "after"):
        X, y = [], []
        for mnk, d in shapes.items():
            for (spec, kind), t in d.items():
                mm, n = method(kind)
                if mm != m or spec not in t_ks:
                    continue
                g = geometry(spec, *mnk)
                if g["regime"] != regime:
                    continue
                tk = t_ks[spec]
                X.append((g["pairs"] * 2 * tk / n, 1.0, g["S"] * n * g["mb"] if m == "rs" else (n - 1) * g["mb"]))
                y.append(t - g["W"] * g["k"] * tk)
        X, y = np.array(X), np.array(y)
        # Relative residuals: a split's error matters in proportion to its time.
        w = 1 / np.maximum(np.abs(y), 1.0)
        res = least_squares(lambda p: (X @ p - y) * w, x0=[1.0, 5.0, 1.0], loss="soft_l1", f_scale=0.05,
                            bounds=([0, -50, 0], [3, 200, 1e4]))
        out[regime] = [round(float(v), 4) for v in res.x]
    return out


def candidates(spec, M, N, K):
    kinds = ["dp"] + [f"ow{n}" for n in OWNER_N] + [f"rs{n}" for n in RS_N]
    return [k for k in kinds if grid.valid(k, spec, M, N, K)]


def choose(model, M, N, K, specs=None):
    best = None
    for spec in specs or grid.ALL_SPECS:
        if spec not in model["t_ks"]:
            continue
        g = geometry(spec, M, N, K)
        for kind in candidates(spec, M, N, K):
            t = predict(model, spec, kind, g)
            if best is None or t < best[0]:
                best = (t, spec, kind)
    return best[1], best[2]


def summary(rs):
    rs = sorted(rs)
    if not rs:
        return "-"
    q = lambda f: rs[min(len(rs) - 1, int(f * len(rs)))]
    return f"{statistics.geometric_mean(rs):.4f} / {q(0.5):.3f} / {q(0.9):.3f} / {rs[-1]:.3f}"


def prediction_errors(model, shapes):
    err = collections.defaultdict(list)
    for mnk, d in shapes.items():
        for (spec, kind), t in d.items():
            if spec in model["t_ks"]:
                err[method(kind)[0]].append(predict(model, spec, kind, geometry(spec, *mnk)) / t)
    return {m: summary([max(x, 1 / x) for x in v]) for m, v in err.items()}


def score(model, shapes, full_only):
    """Measured time of the model's choice over the shape's best (shapes with every
    spec timed when full_only), and how often the choice was not timed."""
    regs, missing = [], 0
    for (M, N, K), d in shapes.items():
        specs = {s for s, _ in d}
        if full_only and len(specs) < len(grid.ALL_SPECS):
            continue
        spec, kind = choose(model, M, N, K, None if full_only else sorted(specs))
        if (spec, kind) not in d:
            missing += 1
            continue
        regs.append(d[(spec, kind)] / min(d.values()))
    return regs, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--out")
    ap.add_argument("--model-out")
    args = ap.parse_args()
    train, test = load(args.train), load(args.test)
    t_ks, a, dp_err = fit_dp(train)
    model = {"t_ks": {s: round(v, 4) for s, v in t_ks.items()}, "a": a}
    model["rs"] = fit_split(train, t_ks, "rs")
    model["ow"] = fit_split(train, t_ks, "ow")
    tflops = {s: round(2 * np.prod(grid.parse_tile(s)) * P / (v * 1e-6) / 1e12) for s, v in t_ks.items()}
    lines = ["# v31 time model", "", f"Train: {' '.join(args.train)}; test: {' '.join(args.test)}.", "",
             "## Constants", "", "```json", json.dumps(model, indent=1), "```", "",
             "Full-wave TFLOPS implied by t_ks: " + ", ".join(f"{s} {v}" for s, v in tflops.items()), "",
             "## Prediction error, max(pred / time, time / pred): geomean / median / 90th / worst", ""]
    for name, shapes in (("train", train), ("test", test)):
        for m, s in sorted(prediction_errors(model, shapes).items()):
            lines.append(f"- {name} {m}: {s}")
    lines += ["", "## The model's choice (every spec, method and n) over the shape's best", ""]
    for name, shapes, full in (("test, every spec timed", test, True), ("train, among timed specs", train, False)):
        regs, missing = score(model, shapes, full)
        lines.append(f"- {name}: {summary(regs)} on {len(regs)} shapes ({missing} choices not timed)")
    text = "\n".join(lines)
    print(text)
    if args.out:
        open(args.out, "w").write(text + "\n")
    if args.model_out:
        json.dump(model, open(args.model_out, "w"), indent=1)


if __name__ == "__main__":
    main()
