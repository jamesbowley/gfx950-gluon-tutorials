"""Fit the Stream-K formula given the tile, choose its form, and test how robust it is.

    python experiments/streamk_v31_rule/fit_formula.py --train 'results/grid_r*.csv' 'results/tilesweep_r*.csv' \\
        'results/gap_r*.csv' 'results/joint_train_r*.csv' 'results/formula_r*.csv' \\
        --test 'results/heldout_r*.csv' 'results/joint_heldout_r*.csv' --out results/formula.md

The formula (formula()): with m = P // S programs per tail tile, pairs = ceil(K / 2 BK),
cap = min(m, pairs), and u the tile term,

    dp             if S == 0 or cap < 2
    owner          if m <= m_ow:  n = sqrt(pairs / c_ow),    dp if pairs < pairs_ow u
    reduce-scatter otherwise:     n = sqrt(pairs m / (b v)), dp if pairs (1 - 1/n) < gain u

n clamped to [2, cap] and rounded to the nearest listed split. Forms, simplest first:
- shared: one set of constants, u = v = 1;
- regimes: constants per regime (below one wave: T < P; else after), u = v = 1;
- the best of those with a tile term: u = 64 / BK (the fixed cost in this tile's
  K-pairs if it scales like the K-pair's length), or u = v = tf (partial bytes per
  K-pair time, from each spec's measured full-wave throughput).
A more complex form is kept only if leaving out whole specs improves the geometric
mean by MIN_GAIN or the 90th percentile by MIN_GAIN_P90. The chosen form's constants
are bootstrapped over shapes, then rounded to readable values (the cost reported).

Objective: mean log regret (the formula's time over the pair's best timed time) plus
LOSS_WEIGHT times the mean excess of its time over dp's beyond NOISE. A sweep that did
not time the formula's n is scored at the nearest timed n (log distance).
"""

import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import csv
import glob
import json
import math
import random
import statistics
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import grid  # noqa: E402

P = grid.P
OWNER_N = (2, 3, 4, 6, 8, 12, 16)
RS_N = (2, 4, 6, 8, 12, 16, 32, 64)
NOISE = 1.01
LOSS_WEIGHT = 5.0
MIN_GAIN = 0.005
MIN_GAIN_P90 = 0.01
RESTARTS = 24
SEED = 3110
SPACE = {
    "m_ow": (1, 2, 3, 4, 5),
    "c_ow": (1, 1.5, 2, 3, 4, 6, 8),
    "pairs_ow": (2, 4, 6, 8, 10, 12, 16, 24),
    "b": (2, 3, 4, 6, 8, 12, 16, 24),
    "gain": (0, 1, 2, 3, 4, 5, 6, 8, 10, 12),
}
BASE = dict(m_ow=3, c_ow=2, pairs_ow=12, b=8, gain=5)
ROUND = {"c_ow": (1, 2, 3, 4, 6, 8), "b": (2, 4, 6, 8, 12, 16), "pairs_ow": (2, 4, 6, 8, 10, 12, 16, 24),
         "gain": (2, 3, 4, 5, 6, 8, 10)}


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
    blocks = collections.defaultdict(dict)
    for p, v in t.items():
        kind, spec, M, N, K = grid.parse_point(p)
        blocks[(spec, M, N, K)][kind] = statistics.median(v)
    out = {}
    for (spec, M, N, K), b in blocks.items():
        if "dp" not in b:
            continue
        g = grid.shape_info(spec, M, N, K)
        bm, bn, bk = grid.parse_tile(spec)
        info = dict(spec=spec, T=g["T"], S=g["S"], W=g["W"], k=g["k"], pairs=g["pairs"], bk=bk,
                    regime="below" if g["T"] < P else "after", shape=(M, N, K),
                    ow=sorted(int(k[2:]) for k in b if k.startswith("ow")),
                    rs=sorted(int(k[2:]) for k in b if k.startswith("rs")))
        out[(spec, M, N, K)] = dict(times=b, info=info, best=min(b.values()))
    return out


def spec_tf(blocks):
    """Full-wave TFLOPS per spec from the dp times of pairs with S == 0 or W >= 2 (tail
    effects small): 2 BM BN BK P k (W + tail) / time, tail = 0.5 + 0.5 S / P."""
    v = collections.defaultdict(list)
    for b in blocks.values():
        i = b["info"]
        if i["W"] < 2 and i["S"]:
            continue
        bm, bn, bk = grid.parse_tile(i["spec"])
        waves = i["W"] + (0.5 + 0.5 * i["S"] / P if i["S"] else 0)
        v[i["spec"]].append(2 * bm * bn * bk * P * i["k"] * waves / (b["times"]["dp"] * 1e-6) / 1e12)
    return {s: statistics.median(x) for s, x in v.items()}


def nearest(n, avail):
    return min(avail, key=lambda a: (abs(math.log(a / n)), a)) if avail else None


def formula(info, c, form, tf, ref_tf):
    """The formula's (kind, n) for a pair; avail: snap to the timed n of the block when
    info carries them (fitting), else to the listed splits (deployment)."""
    S, pairs = info["S"], info["pairs"]
    if S == 0:
        return "dp", 1
    m = P // S
    cap = min(m, pairs)
    if cap < 2:
        return "dp", 1
    cc = c[info["regime"]] if form["regimes"] else c["all"]
    u = v = 1.0
    if form["tile"] == "bk":
        u = 64 / info["bk"]
    elif form["tile"] == "tf":
        u = v = tf.get(info["spec"], ref_tf) / ref_tf * 64 / info["bk"]
    if m <= cc["m_ow"]:
        if pairs < cc["pairs_ow"] * u:
            return "dp", 1
        n = max(2.0, math.sqrt(pairs / cc["c_ow"]))
        allowed = [a for a in (info.get("ow") or OWNER_N) if a <= cap]
        return ("ow", nearest(n, allowed)) if allowed else ("dp", 1)
    n = max(2.0, math.sqrt(pairs * m / (cc["b"] * v)))
    allowed = [a for a in (info.get("rs") or RS_N) if a <= cap]
    n = nearest(min(n, cap), allowed) if allowed else None
    if not n or pairs * (1 - 1 / n) < cc["gain"] * u:
        return "dp", 1
    return "rs", n


def time_of(b, c, form, tf, ref_tf):
    kind, n = formula(b["info"], c, form, tf, ref_tf)
    return b["times"]["dp"] if kind == "dp" else b["times"].get(f"{kind}{n}")


def objective(blocks, keys, c, form, tf, ref_tf):
    total, n = 0.0, 0
    for key in keys:
        b = blocks[key]
        t = time_of(b, c, form, tf, ref_tf)
        if t is None:
            total += 0.1
        else:
            total += math.log(t / b["best"]) + LOSS_WEIGHT * max(0.0, math.log(t / (NOISE * b["times"]["dp"])))
        n += 1
    return total / max(n, 1)


def fit(blocks, keys, form, tf, ref_tf, restarts=RESTARTS, start=None):
    groups = ("below", "after") if form["regimes"] else ("all",)
    c = {g: dict(start[g]) if start else dict(BASE) for g in groups}
    rng = random.Random(SEED)
    for g in groups:
        gkeys = [k for k in keys if not form["regimes"] or blocks[k]["info"]["regime"] == g]

        def descend(cc):
            best = objective(blocks, gkeys, dict(c, **{g: cc}), form, tf, ref_tf)
            for _ in range(6):
                changed = False
                for name, values in SPACE.items():
                    for val in values:
                        trial = dict(cc, **{name: val})
                        o = objective(blocks, gkeys, dict(c, **{g: trial}), form, tf, ref_tf)
                        if o < best - 1e-12:
                            best, cc, changed = o, trial, True
                if not changed:
                    break
            return best, cc

        starts = [dict(c[g])] + [{n: rng.choice(v) for n, v in SPACE.items()} for _ in range(restarts)]
        c[g] = min((descend(s) for s in starts), key=lambda r: r[0])[1]
    return c


def regrets(blocks, keys, c, form, tf, ref_tf):
    out = []
    for key in keys:
        b = blocks[key]
        t = time_of(b, c, form, tf, ref_tf)
        if t is not None:
            out.append((t / b["best"], t / b["times"]["dp"]))
    return out


def summary(rs):
    rs = sorted(rs)
    if not rs:
        return "-"
    q = lambda f: rs[min(len(rs) - 1, int(f * len(rs)))]
    return f"{statistics.geometric_mean(rs):.4f} / {q(0.5):.3f} / {q(0.9):.3f} / {rs[-1]:.3f}"


def dp_losses(r):
    d = [x[1] for x in r]
    return f"vs dp: >5% slower {sum(x > 1.05 for x in d)}, >10% {sum(x > 1.10 for x in d)} of {len(d)}, worst {max(d):.3f}"


def cross_validate(blocks, form, tf, ref_tf, by):
    groups = collections.defaultdict(list)
    for key, b in blocks.items():
        i = b["info"]
        g = {"spec": i["spec"], "k": grid.cover_cell(i["spec"], *i["shape"]) and grid.cover_cell(i["spec"], *i["shape"])[3],
             "m": grid.cover_cell(i["spec"], *i["shape"]) and grid.cover_cell(i["spec"], *i["shape"])[2]}[by]
        groups[g].append(key)
    out = []
    for g, held in groups.items():
        held_set = set(held)
        c = fit(blocks, [k for k in blocks if k not in held_set], form, tf, ref_tf, restarts=8)
        out += regrets(blocks, held, c, form, tf, ref_tf)
    return out


DATA = {}


def _form_job(form):
    train, tf, ref_tf = DATA["train"], DATA["tf"], DATA["ref_tf"]
    keys = list(train)
    c = fit(train, keys, form, tf, ref_tf)
    ins = [x[0] for x in regrets(train, keys, c, form, tf, ref_tf)]
    loso = [x[0] for x in cross_validate(train, form, tf, ref_tf, "spec")]
    return form, c, ins, loso


def _boot_job(args):
    seed, form, c = args
    train, tf, ref_tf = DATA["train"], DATA["tf"], DATA["ref_tf"]
    shapes, by_shape = DATA["shapes"], DATA["by_shape"]
    rng = random.Random(seed)
    ks = [k for sh in rng.choices(shapes, k=len(shapes)) for k in by_shape[sh]]
    return fit(train, ks, form, tf, ref_tf, restarts=2, start=c)


def _cv_job(args):
    form, c, by = args
    return by, cross_validate(DATA["train"], form, DATA["tf"], DATA["ref_tf"], by)


def rounded(c):
    return {g: {k: (min(ROUND[k], key=lambda x: abs(x - v)) if k in ROUND else v) for k, v in cc.items()}
            for g, cc in c.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--out")
    ap.add_argument("--rule-out")
    ap.add_argument("--bootstrap", type=int, default=100)
    args = ap.parse_args()
    train, test = load(args.train), load(args.test)
    tf = spec_tf({**train, **test})
    ref_tf = tf["256x256x64"]
    keys = list(train)
    lines = [f"# Stream-K formula given the tile", "",
             f"Train: {len(train)} (shape, spec) pairs; test: {len(test)} pairs.", "",
             "Measured full-wave TFLOPS per spec (dp, S = 0 or W >= 2): "
             + ", ".join(f"{s} {round(v)}" for s, v in sorted(tf.items(), key=lambda kv: -kv[1])), "",
             "## Form selection (leave one spec out: time over the pair's best, geomean / median / 90th / worst)", "",
             "| form | in-sample | leave one spec out |", "|---|---|---|"]
    forms = [dict(name="shared", regimes=False, tile="none"), dict(name="regimes", regimes=True, tile="none"),
             dict(name="shared + 64/BK", regimes=False, tile="bk"), dict(name="regimes + 64/BK", regimes=True, tile="bk"),
             dict(name="shared + tf", regimes=False, tile="tf"), dict(name="regimes + tf", regimes=True, tile="tf")]
    DATA.update(train=train, tf=tf, ref_tf=ref_tf)
    with ProcessPoolExecutor(len(forms)) as ex:
        results = list(ex.map(_form_job, forms))
    for form, c, ins, loso in results:
        lines.append(f"| {form['name']} | {summary(ins)} | {summary(loso)} |")
        print(lines[-1], flush=True)
    # Simplest form unless a more complex one wins clearly out of sample.
    chosen = results[0]
    for r in results[1:]:
        gm = lambda x: statistics.geometric_mean(x)
        p90 = lambda x: sorted(x)[int(0.9 * len(x))]
        if (math.log(gm(chosen[3])) - math.log(gm(r[3])) >= MIN_GAIN
                or p90(chosen[3]) - p90(r[3]) >= MIN_GAIN_P90):
            chosen = r
    form, c, _, _ = chosen
    cr = rounded(c)
    lines += ["", f"Chosen form: **{form['name']}**.", "", "## Constants (fitted, then rounded)", "",
              "```json", json.dumps({"form": form, "fitted": c, "rounded": cr}, indent=1), "```", ""]
    rr_fit = regrets(train, keys, c, form, tf, ref_tf)
    rr = regrets(train, keys, cr, form, tf, ref_tf)
    lines += [f"- train, fitted: {summary([x[0] for x in rr_fit])}; {dp_losses(rr_fit)}",
              f"- train, rounded: {summary([x[0] for x in rr])}; {dp_losses(rr)}"]
    shapes = sorted({b["info"]["shape"] for b in train.values()})
    by_shape = collections.defaultdict(list)
    for key, b in train.items():
        by_shape[b["info"]["shape"]].append(key)
    DATA.update(shapes=shapes, by_shape=by_shape)
    with ProcessPoolExecutor(32) as ex:
        cvs = list(ex.map(_cv_job, [(form, c, "k"), (form, c, "m")]))
        boots = list(ex.map(_boot_job, [(SEED + 1 + i, form, c) for i in range(args.bootstrap)]))
    for by, cv in cvs:
        lines.append(f"- leave one {by} bin out: {summary([x[0] for x in cv])}")
    boot = collections.defaultdict(list)
    for cb in boots:
        for g, cc in cb.items():
            for name, val in cc.items():
                boot[(g, name)].append(val)
    lines += ["", f"## Bootstrap over shapes ({args.bootstrap} resamples): 10th / median / 90th percentile", ""]
    for (g, name), vals in sorted(boot.items()):
        vals = sorted(vals)
        lines.append(f"- {g} {name}: {vals[len(vals) // 10]} / {statistics.median(vals)} / {vals[9 * len(vals) // 10]}")
    lines += ["", "## Test: held-out AITER shapes, every spec (time over the pair's best)", ""]
    per_spec = collections.defaultdict(list)
    for key, b in test.items():
        t = time_of(b, cr, form, tf, ref_tf)
        if t is not None:
            per_spec[b["info"]["spec"]].append((t / b["best"], t / b["times"]["dp"]))
    allr = [x for v in per_spec.values() for x in v]
    lines.append(f"- all specs: {summary([x[0] for x in allr])}; {dp_losses(allr)}")
    dpr = [b["times"]["dp"] / b["best"] for b in test.values()]
    lines.append(f"- dp for comparison: {summary(dpr)}")
    for spec in grid.ALL_SPECS:
        if per_spec[spec]:
            lines.append(f"- {spec}: {summary([x[0] for x in per_spec[spec]])}; {dp_losses(per_spec[spec])}")
    text = "\n".join(lines)
    print(text)
    if args.out:
        open(args.out, "w").write(text + "\n")
    if args.rule_out:
        json.dump({"form": form, "constants": cr, "spec_tflops": {s: round(v) for s, v in tf.items()}},
                  open(args.rule_out, "w"), indent=1)


if __name__ == "__main__":
    main()
