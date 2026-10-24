#!/usr/bin/env python3
"""What a joint tile + split rule would buy, CPU only, from timings already taken.

    python eval_joint_rule.py [--out results/joint_rule_eval.md]

The rule (quadrant tiles only, 256 programs): under one wave, each candidate tile gets the largest K split that
fits, s = min(256 // T, K // K_MIN, pairs), where T is its tile count and K_MIN the least K one program may own;
from one wave up it keeps cap (v26's policy, near-best there). Two ways to choose the tile:
- A, "largest that fills": the largest-area tile whose T * s reaches FILL * 256 (else the best fill);
- B, "origami-lite": estimated time = (one tile's full-K time at the tile's FULL_TFLOPS per CU) / s, plus SPLIT_US
  for any split, or T / 256 of a tile's time from one wave up; the largest-area tile within DELTA of the best.
The split runs as the one-tile split with the reduce-scatter ending (rs{s}), or the owner ending for s = 2
(OWNER2), at the timed n at or below s ("all" when s takes every program).

Data (each point's merged cold rocprof time, sweep.py's timer):
- REFIT, compute-bound (198 shapes, every quadrant tile x every configuration): fitting, and in-sample regret;
- WORST10 (2026-10-17: 61 AITER shapes, every quadrant tile x every configuration, fp32 partials only): held out;
- GE512 (2026-10-21: the 167 both-sides->=512 all-tuned losers; auto, the REFIT rule's pick, cap and rs4 at
  256x256x64, 256x128x64 and 128x128x128, AITER's pick and hipBLASLt in the same harness session): held out,
  where the rule's tile was timed. A split not timed there takes the timed point at that tile whose n is nearest.
Regret is the pick's time over the best point timed for the shape.
"""

import argparse
import csv
import itertools
import math
import os
import statistics
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
A16R = os.path.join(HERE, "..", "..", "kernels", "gemm", "intra_wave", "a16w16", "v27_streamk_tile", "results")
STUDY = "/home/jbowley/repos/aiter/streamk_study"
P = 256
QUAD = ("256x256x64", "128x256x64", "256x128x64", "128x128x128", "128x128x64", "64x256x64", "256x64x64",
        "64x128x128", "128x64x128", "64x128x64", "128x64x64", "64x64x128", "64x64x64")
# sweep.FULL_TFLOPS: each quadrant tile's best TFLOPS in the FULL sweep (full occupancy), the throughput model.
FULL_TFLOPS = {"256x256x64": 1529, "256x128x64": 1332, "128x256x64": 1274, "128x128x64": 1176, "128x128x128": 1142,
               "256x64x64": 938, "128x64x64": 887, "64x128x64": 875, "128x64x128": 852, "64x256x64": 826,
               "64x128x128": 802, "64x64x128": 729, "64x64x64": 689}
NS = (2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32)


def dims(t):
    return tuple(int(v) for v in t.split("x"))


def ntiles(M, N, t):
    bm, bn, _ = dims(t)
    return -(-M // bm) * -(-N // bn)


def area(t):
    bm, bn, _ = dims(t)
    return bm * bn


# ---------------------------------------------------------------- the current rule (auto_tile + cap)

def auto_tile(M, N, K):
    """matmul_kernel.auto_tile (1175b735)."""
    if M < 512 or ntiles(M, N, "256x256x64") >= 256:
        return "256x256x64"
    ok = [t for t in QUAD if ntiles(M, N, t) >= 128]
    if not ok:
        return "64x64x128"
    big = max(area(t) for t in ok)
    pick = min((t for t in ok if area(t) == big), key=lambda t: (ntiles(M, N, t), dims(t)[2] != 128))
    if ntiles(M, N, pick) == 128 and K <= 4096:
        halves = [t for t in QUAD if 2 * area(t) == area(pick) and ntiles(M, N, t) == 256]
        if halves:
            pick = min(halves, key=lambda t: dims(t)[2] != 128)
    return pick


def capped_n(T, k_steps):
    """capped_policy's programs per leftover tile under one wave (1 = data-parallel), matmul_kernel 1175b735."""
    s = T % P
    if s == 0 or 2 * s > P:
        return 1
    if T < P:
        if (P - s) * k_steps < 14 * P:
            return 1
        cap = min(P // s, k_steps // 2)
        n = min(cap, 192 // s, round(k_steps / 4)) if P // s >= 4 else min(cap, round(math.sqrt(k_steps / 1.5)))
        return max(1, n)
    return 1


# ---------------------------------------------------------------- the joint rule

def split_for(M, N, K, t, k_min):
    """(T, s): the largest K split under one wave (s = 1: data-parallel; None from one wave up)."""
    T = ntiles(M, N, t)
    if T >= P:
        return T, None
    bk = dims(t)[2]
    pairs = -(-K // (2 * bk))
    return T, max(1, min(P // T, K // k_min, pairs))


def rule(M, N, K, form, k_min, fill=0.75, split_us=2.0, delta=0.02, per_split_us=0.0):
    """(tile, s) by form A (largest that fills), B (origami-lite, the largest split) or C (origami-lite over every
    split up to the largest, each program of a split costing PER_SPLIT_US more)."""
    if form == "C":
        est = []
        for t in QUAD:
            T, smax = split_for(M, N, K, t, k_min)
            bm, bn, _ = dims(t)
            tau = 2.0 * bm * bn * K / (FULL_TFLOPS[t] * 1e12 / P) * 1e6
            if smax is None:
                est.append((tau * T / P, t, None))
                continue
            for s in range(1, smax + 1):
                est.append((tau / s + (split_us + per_split_us * s if s > 1 else 0.0), t, s))
        best = min(e for e, _, _ in est)
        near = [x for x in est if x[0] <= best * (1 + delta)]
        big = max(area(t) for _, t, _ in near)
        e, t, s = min((x for x in near if area(x[1]) == big), key=lambda x: x[0])
        return t, s
    if form == "A":
        cands = []
        for t in QUAD:
            T, s = split_for(M, N, K, t, k_min)
            f = 1.0 if s is None else min(1.0, T * s / P)
            cands.append((t, s, f))
        good = [c for c in cands if c[2] >= fill]
        if good:
            big = max(area(c[0]) for c in good)
            return max((c for c in good if area(c[0]) == big), key=lambda c: (c[2], dims(c[0])[2]))[:2]
        return max(cands, key=lambda c: (c[2], area(c[0])))[:2]
    est = []
    for t in QUAD:
        T, s = split_for(M, N, K, t, k_min)
        bm, bn, _ = dims(t)
        tau = 2.0 * bm * bn * K / (FULL_TFLOPS[t] * 1e12 / P) * 1e6  # us, one tile, full K, one CU
        e = tau * T / P if s is None else tau / s + (split_us if s > 1 else 0.0)
        est.append((e, t, s))
    best = min(e for e, _, _ in est)
    near = [x for x in est if x[0] <= best * (1 + delta)]
    big = max(area(t) for _, t, _ in near)
    e, t, s = min((x for x in near if area(x[1]) == big), key=lambda x: x[0])
    return t, s


def kind_for(T, s, owner2):
    if s is None:
        return ["cap"]
    if s == 1:
        return ["dp", "cap"]
    end = "ow" if (s == 2 and owner2) else "rs"
    ks = ([f"{end}all"] if s >= P // T else []) + [f"{end}{n}" for n in sorted((n for n in NS if n <= s), reverse=True)]
    return ks + ["cap"]


# ---------------------------------------------------------------- the split tables (WORKLOG §17)

TABLES = os.path.join(A16R, "split", "tables.json")
SIMPLE = ("dp", "cap", "pdp", "pdp512", "rs2", "rs3", "rs4", "rs5", "rs6", "rs7", "rs8", "rs10", "rs12", "rs16",
          "rs24", "rs32", "rsall", "ow2", "ow3", "ow4", "owall", "2t", "sp")
TABLE_DELTA = 0.01
_GRIDS = {}


def grid_for(t, wc):
    if not _GRIDS:
        import json
        for key, v in json.load(open(TABLES)).items():
            _GRIDS[key] = {(c["S"], c["k"]): c["vals"] for c in v["cells"]}
    return _GRIDS.get(f"{t} w{wc}")


def table_kind(M, N, K, t):
    """split_report.bracket's method for the tile at the shape: the kind with the smallest summed log time over the
    cells bracketing (S, k-steps), ties within 1% per cell to the simpler kind."""
    T = ntiles(M, N, t)
    waves = T // P
    wc = 0 if waves == 0 else (1 if waves < 4 else 4)
    grid = grid_for(t, wc)
    if grid is None:
        return "cap"
    S, k = T % P, -(-K // (2 * dims(t)[2])) * 2
    if S == 0:
        return "dp" if waves <= 1 else "pdp"
    Ss = sorted({s for s, _ in grid})
    lo = max((x for x in Ss if x <= S), default=Ss[0])
    hi = min((x for x in Ss if x >= S), default=Ss[-1])
    cells = []
    for s in {lo, hi}:
        ks = sorted({kk for s_, kk in grid if s_ == s})
        klo = max((x for x in ks if x <= k), default=ks[0])
        khi = min((x for x in ks if x >= k), default=ks[-1])
        cells += [grid[(s, kk)] for kk in {klo, khi}]
    common = set(cells[0]).intersection(*cells[1:])
    if not common:
        return "cap"
    score = {kd: sum(math.log(c[kd]) for c in cells) for kd in common}
    best = min(score.values())
    tie = [kd for kd in common if score[kd] <= best + len(cells) * math.log(1 + TABLE_DELTA)]
    return min(tie, key=lambda kd: (SIMPLE.index(kd) if kd in SIMPLE else 99, score[kd]))


def table_picker(tile_fn):
    def pick(M, N, K):
        t = tile_fn(M, N, K)
        return t, [table_kind(M, N, K, t), "cap"]
    return pick


# ---------------------------------------------------------------- data

def load_points(path, keep_shape, keep_kind=lambda k: True):
    pts = {}
    for r in csv.DictReader(open(path)):
        if r["correct"] != "True" or r["tile"] not in QUAD or not keep_kind(r["kind"]):
            continue
        sh = (int(r["M"]), int(r["N"]), int(r["K"]))
        if keep_shape(sh):
            pts.setdefault(sh, {})[(r["tile"], r["kind"])] = float(r["us_median"])
    return pts


def regret_full(pts, pick):
    """Regret per shape for a (tile, [kinds]) picker over full-factorial points; snapped counts the fallbacks."""
    out, snapped = [], 0
    for sh, d in pts.items():
        best = min(d.values())
        t, kinds = pick(*sh)
        for i, k in enumerate(kinds):
            if (t, k) in d:
                out.append(d[(t, k)] / best)
                snapped += i > 0 and k != "cap" or (k == "cap" and kinds[0] != "cap")
                break
    return out, snapped


def stats(xs):
    xs = sorted(xs)
    gm = math.exp(sum(math.log(x) for x in xs) / len(xs))
    return (f"{statistics.median(xs):.3f} / {xs[int(0.9 * (len(xs) - 1))]:.3f} / {xs[-1]:.3f} / {gm:.3f}, "
            f"{sum(x > 1.05 for x in xs)} > 1.05")


def pickers(form, k_min, fill, split_us, delta, owner2, per_split_us=0.0):
    def joint(M, N, K):
        t, s = rule(M, N, K, form, k_min, fill, split_us, delta, per_split_us)
        return t, kind_for(ntiles(M, N, t), s, owner2)
    return joint


def diag_tile_only(pts, joint):
    """The rule's tile with the best configuration timed at it: the tile error alone."""
    out = []
    for sh, d in pts.items():
        t, _ = joint(*sh)
        at = [v for (tt, _), v in d.items() if tt == t]
        if at:
            out.append(min(at) / min(d.values()))
    return out


def diag_split_only(pts, joint, owner2):
    """The best tile (by its best configuration) with the rule's split for that tile: the split error alone."""
    out = []
    for sh, d in pts.items():
        tb = min(QUAD, key=lambda t: min([v for (tt, _), v in d.items() if tt == t] or [1e30]))
        params = joint.params
        T, s = rule_split_at(sh, tb, params)
        for k in kind_for(T, s, owner2):
            if (tb, k) in d:
                out.append(d[(tb, k)] / min(d.values()))
                break
    return out


def rule_split_at(sh, t, params):
    """The rule's split for a given tile: largest (A, B) or the model's best (C)."""
    M, N, K = sh
    T, smax = split_for(M, N, K, t, params["k_min"])
    if smax is None or params["form"] != "C":
        return T, smax
    bm, bn, _ = dims(t)
    tau = 2.0 * bm * bn * K / (FULL_TFLOPS[t] * 1e12 / P) * 1e6
    s = min(range(1, smax + 1), key=lambda s: tau / s + (params["split_us"] + params["per_split_us"] * s if s > 1 else 0))
    return T, s


def auto_pick(M, N, K):
    return auto_tile(M, N, K), ["cap"]


def oracle_tile_cap(pts):
    def f(M, N, K):
        d = pts[(M, N, K)]
        t = min(QUAD, key=lambda t: d.get((t, "cap"), 1e30))
        return t, ["cap"]
    return f


# ---------------------------------------------------------------- GE512 (partial points, AITER in the same session)

def ms(r, n):
    xs = [float(x) for x in (r.get(n + "_ms") or "").split()]
    return statistics.median(xs) if xs else float("nan")


def variant_rule(params, variant):
    """(M, N, K) -> (tile, s, use_cap) for a form-A variant: "cap" runs cap at the rule's tile; ("max", m) runs the
    largest split at most m at the rule's tile; ("maxfill", m) also judges the fill with that split."""
    q = {k: v for k, v in params.items() if k != "owner2"}

    def f(M, N, K):
        if isinstance(variant, tuple) and variant[0] == "maxfill":
            m = variant[1]
            cands = []
            for t in QUAD:
                T, s = split_for(M, N, K, t, q["k_min"])
                s = None if s is None else min(s, m)
                cands.append((t, s, 1.0 if s is None else min(1.0, T * s / P)))
            good = [c for c in cands if c[2] >= q["fill"]]
            if good:
                big = max(area(c[0]) for c in good)
                t, s, _ = max((c for c in good if area(c[0]) == big), key=lambda c: (c[2], dims(c[0])[2]))
            else:
                t, s, _ = max(cands, key=lambda c: (c[2], area(c[0])))
            return t, s, False
        t, s = rule(M, N, K, **q)
        if variant == "cap":
            return t, s, True
        if isinstance(variant, tuple) and variant[0] == "max":
            return t, (None if s is None else min(s, variant[1])), False
        return t, s, False
    return f


def variant_picker(params, variant):
    f = variant_rule(params, variant)

    def pick(M, N, K):
        t, s, cap = f(M, N, K)
        return t, (["cap"] if cap else kind_for(ntiles(M, N, t), s, params["owner2"]))
    return pick


def ge512_eval(params, variant=None):
    rows = list(csv.DictReader(open(os.path.join(STUDY, "2026-10-20_all_tuned/results/ge512/details_alltuned_ge512.csv"))))
    slots = ("best", "tile2", "tile3", "fp32", "x5", "x6", "x7")
    res = []
    for r in rows:
        M, N, K = int(r["M"]), int(r["N"]), int(r["K"])
        pts = {}
        if r["v20_ok"] == "True":
            pts[(r["v20_tile"], "cap")] = ms(r, "v20")
        for s in slots:
            if r.get(f"pk_{s}_ok") == "True":
                pts[(r[f"pk_{s}_tile"], r[f"pk_{s}_kind"])] = ms(r, "pk_" + s)
        best = min(pts.values())
        pick = ms(r, "pick")
        if variant is None:
            t, s = rule(M, N, K, **params)
            use_cap = False
        else:
            t, s, use_cap = variant_rule(dict(params, owner2=True), variant)(M, N, K)
        T = ntiles(M, N, t)
        at_t = {k: v for (tt, k), v in pts.items() if tt == t}
        if not at_t:
            res.append(dict(shape=(M, N, K), covered=False, tile=t, s=s))
            continue

        def n_of(k):
            if k == "cap":
                return capped_n(T, -(-K // (2 * dims(t)[2])) * 2) if T < P else None
            if k == "dp" or k.startswith("pdp"):
                return 1
            if k.endswith("all"):
                return P // T if T < P else None
            if k[:2] in ("rs", "ow") and k[2:].isdigit():
                return int(k[2:])
            return None  # sp, 2t: no fixed n
        if use_cap:
            want = capped_n(T, -(-K // (2 * dims(t)[2])) * 2) if T < P else 1
            k = "cap" if "cap" in at_t else min(at_t, key=lambda k: abs((n_of(k) or want) - want))
        else:
            want = 1 if s is None else s
            k = min(at_t, key=lambda k: (abs((n_of(k) or want) - want), k != f"rs{want}"))
        res.append(dict(shape=(M, N, K), covered=True, tile=t, s=s, kind=k, us=at_t[k], best=best, pick=pick,
                        auto=pts.get((r["v20_tile"], "cap"), float("nan")), exact=(n_of(k) == want)))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "joint_rule_eval.md"))
    args = ap.parse_args()

    refit = load_points(os.path.join(A16R, "refit.csv"), lambda sh: sh[0] >= 512)
    worst = load_points(os.path.join(STUDY, "2026-10-17/results/sweep/worst10.csv"), lambda sh: True,
                        lambda k: not (k[-1] in "bp" and k[:2] in ("rs", "ow")))
    under = lambda pts: {sh: d for sh, d in pts.items() if ntiles(*sh[:2], "256x256x64") < P}
    L = ["# Joint tile + split rule, CPU-only evaluation", "",
         "Regret: the pick's time over the best point timed for the shape (median / 90th / worst / geomean, "
         "shapes above 1.05).", ""]

    # fit on REFIT compute-bound
    KMINS = (512, 768, 1024, 1536, 2048, 3072, 4096)
    grid = []
    for form, k_min, owner2 in itertools.product("ABC", KMINS, (False, True)):
        if form == "A":
            for fill in (0.375, 0.5, 0.625, 0.75, 0.875):
                grid.append(dict(form=form, k_min=k_min, fill=fill, owner2=owner2))
        elif form == "B":
            for split_us, delta in itertools.product((0.0, 2.0, 4.0, 8.0, 16.0), (0.0, 0.02, 0.05, 0.10)):
                grid.append(dict(form=form, k_min=k_min, split_us=split_us, delta=delta, owner2=owner2))
        else:
            for split_us, per, delta in itertools.product((0.0, 2.0, 4.0, 8.0), (0.0, 0.5, 1.0, 2.0, 4.0),
                                                          (0.0, 0.02, 0.05, 0.10)):
                grid.append(dict(form=form, k_min=k_min, split_us=split_us, per_split_us=per, delta=delta,
                                 owner2=owner2))

    def make(p):
        f = pickers(p["form"], p["k_min"], p.get("fill", 0.75), p.get("split_us", 2.0), p.get("delta", 0.02),
                    p["owner2"], p.get("per_split_us", 0.0))
        f.params = dict(p, split_us=p.get("split_us", 2.0), per_split_us=p.get("per_split_us", 0.0))
        return f

    def gmean(xs):
        return math.exp(sum(math.log(x) for x in xs) / len(xs))

    def score(p, pts):
        xs, _ = regret_full(pts, make(p))
        return gmean(xs), xs

    fits = sorted(((score(p, refit)[0], i, p) for i, p in enumerate(grid)), key=lambda x: x[:2])
    best_by_form = {f: next(p for _, _, p in fits if p["form"] == f) for f in "ABC"}
    L += ["## Fitted parameters (geomean regret on REFIT's 198 compute-bound shapes)", ""]
    for f, p in best_by_form.items():
        L.append(f"- Form {f}: `{p}`, geomean {score(p, refit)[0]:.3f}")
    L.append("- Top 5 overall: " + "; ".join(f"{g:.3f} {p}" for g, _, p in fits[:5]))
    L.append("")

    def row(name, xs, n):
        return f"| {name} | {n} | {stats(xs)} |"

    for title, pts in (("REFIT compute-bound (fitting set)", refit), ("REFIT, under one 256x256 wave", under(refit)),
                       ("WORST10 (held out)", worst), ("WORST10, under one wave", under(worst))):
        L += [f"## {title}: {len(pts)} shapes", "", "| picker | shapes | regret |", "|---|---:|---|"]
        xs, _ = regret_full(pts, auto_pick)
        L.append(row("auto_tile + cap (current)", xs, len(xs)))
        xs, _ = regret_full(pts, oracle_tile_cap(pts))
        L.append(row("oracle tile + cap", xs, len(xs)))
        for f, p in best_by_form.items():
            j = make(p)
            xs, sn = regret_full(pts, j)
            L.append(row(f"joint rule, form {f} ({sn} picks at a nearby n)", xs, len(xs)))
            L.append(row(f"  form {f}: its tile, best configuration there (tile error alone)", diag_tile_only(pts, j),
                         len(pts)))
            L.append(row(f"  form {f}: the best tile, its split (split error alone)",
                         diag_split_only(pts, j, p["owner2"]), len(pts)))
        L.append("")

    # sensitivity on WORST10
    L += ["## Sensitivity (held-out WORST10 geomean; REFIT geomean in brackets)", ""]
    for f in "ABC":
        sub = [p for _, _, p in fits if p["form"] == f][:8]
        for p in sub:
            L.append(f"- {p}: WORST10 {score(p, worst)[0]:.3f} (REFIT {score(p, refit)[0]:.3f})")
    L.append("")

    # Form A split variants (form A's fitted k_min and fill)
    pa = best_by_form["A"]
    variants = [(None, "form A as fitted (the largest split)"), ("cap", "form A's tile, cap (capped's split)")]
    variants += [(("max", m), f"form A's tile, the largest split up to {m}") for m in (3, 4, 6, 8)]
    variants += [(("maxfill", m), f"form A with splits up to {m}, also in the fill test") for m in (4, 6, 8)]
    L += ["## Form A split variants (geomean / shapes above 1.05)", "",
          "| variant | REFIT | REFIT under one wave | WORST10 | GE512: vs AITER median, behind > 4% (covered) |",
          "|---|---|---|---|---|"]
    for v, name in variants:
        cells = []
        for pts in (refit, under(refit), worst):
            xs, _ = regret_full(pts, variant_picker(pa, v) if v else make(pa))
            cells.append(f"{gmean(xs):.3f} / {sum(x > 1.05 for x in xs)}")
        q = {k: val for k, val in pa.items() if k != "owner2"}
        res = [x for x in ge512_eval(q, v) if x["covered"]]
        g = [x["pick"] / x["us"] - 1 for x in res]
        cells.append(f"{100 * statistics.median(g):+.1f}%, {sum(x < -0.04 for x in g)} of {len(g)}")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L.append("")

    # the split tables at each tile rule
    def best_tile(pts):
        return lambda M, N, K: min(QUAD, key=lambda t: min([v for (tt, _), v in pts[(M, N, K)].items() if tt == t]
                                                            or [1e30]))
    qa = {k: v for k, v in pa.items() if k != "owner2"}
    tile_rules = (("auto_tile", lambda M, N, K: auto_tile(M, N, K)),
                  ("form A's tile", lambda M, N, K: rule(M, N, K, **qa)[0]))
    L += ["## The split tables (WORKLOG §17, bracket lookup) as the split, at each tile rule (geomean / shapes > 1.05; "
          "a table kind not timed falls back to cap)", "",
          "| tile rule | REFIT | REFIT under one wave | WORST10 |", "|---|---|---|---|"]
    for name, tf in tile_rules + (("best tile (oracle)", None),):
        cells = []
        for pts in (refit, under(refit), worst):
            xs, _ = regret_full(pts, table_picker(tf or best_tile(pts)))
            cells.append(f"{gmean(xs):.3f} / {sum(x > 1.05 for x in xs)}")
        L.append(f"| {name} + table split | " + " | ".join(cells) + " |")
    L.append("")

    # GE512
    for f, p in best_by_form.items():
        q = dict(p)
        q.pop("owner2")
        res = ge512_eval(q)
        cov = [x for x in res if x["covered"]]
        L += [f"## GE512 (167 both-sides->=512 losers), form {f}", "",
              f"- The rule's tile was timed on {len(cov)} of {len(res)} shapes ({sum(x['exact'] for x in cov)} at "
              f"exactly its split). Untimed tiles: "
              + ", ".join(f"{t} {c}" for t, c in Counter(x['tile'] for x in res if not x['covered']).most_common()) + "."]
        gap = lambda a, b: b / a - 1
        for name, key in (("auto_tile + cap (as run)", "auto"), ("joint rule", "us"), ("best timed point", "best")):
            g = [gap(x[key], x["pick"]) for x in cov if x[key] == x[key]]
            L.append(f"- {name} against AITER's pick, on those {len(g)}: median {100 * statistics.median(g):+.1f}%, "
                     f"behind by > 4% on {sum(v < -0.04 for v in g)}, by > 10% on {sum(v < -0.10 for v in g)}")
        rg = [x["us"] / x["best"] for x in cov]
        L.append(f"- joint rule regret against the best timed point: {stats(rg)}")
        L.append("- tiles chosen: " + ", ".join(f"{t} {c}" for t, c in Counter(x['tile'] for x in res).most_common()))
        L.append("")
    with open(args.out, "w") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
