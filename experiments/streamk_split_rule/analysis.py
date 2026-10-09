##############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
##############################################################################
"""Maps, rule fitting and validation scoring for the split-rule sweep (called through sweep.py).

A rule maps a shape's geometry to one configuration: ("dp", None), ("two_tile", None),
("rs", n) or ("owner", n). On the grid a rule is scored from the measurements alone: its time
on a shape is the measured time of its pick, against the best measured configuration.
"""

import csv
import itertools
import json
import math
import os
import statistics
import subprocess
import sys

import sweep

NUM_CUS = sweep.NUM_CUS
GRID_CONFIGS = ("dp", "two_tile", "rs", "owner")
WCLASS_LABEL = {0: "no full wave", 1: "1-3 full waves", 4: "4-7 full waves"}


# ---------------------------------------------------------------- data


def load_shapes(which="main"):
    """{shape: {"g": geometry, "rows": {(config, n): row}}} with only correct, timed rows."""
    shapes = {}
    for r in sweep.load(which):
        e = shapes.setdefault(r["shape"], {"g": {k: r[k] for k in ("M", "N", "K", "wclass", "waves", "S", "k",
                                                                  "aspect")}, "rows": {}})
        if sweep.good(r):
            e["rows"][(r["config"], r["n"])] = r
    return shapes


def label(config, n):
    return {"dp": "dp", "two_tile": "2T"}.get(config, f"{'rs' if config == 'rs' else 'ow'}{n}")


def best_row(e, grid_only=False):
    rows = [r for (c, _), r in e["rows"].items() if not grid_only or c in GRID_CONFIGS]
    return max(rows, key=lambda r: r["tflops"]) if rows else None


def spread(e):
    """Median slowest/fastest - 1 over the shape's repeated configurations, in %."""
    s = [100 * (max(r["us_runs"]) / min(r["us_runs"]) - 1) for r in e["rows"].values() if len(r.get("us_runs", [])) > 1]
    return statistics.median(s) if s else None


def pct(a, b):
    return 100 * (a / b - 1)


def family_best(e, config):
    rows = [r for (c, _), r in e["rows"].items() if c == config]
    return max(rows, key=lambda r: r["tflops"]) if rows else None


# ---------------------------------------------------------------- maps


def cell_table(shapes, wclass, cell, title, note=""):
    S_vals = sorted({e["g"]["S"] for e in shapes.values() if e["g"]["wclass"] == wclass})
    k_vals = sorted({e["g"]["k"] for e in shapes.values() if e["g"]["wclass"] == wclass})
    by = {(e["g"]["S"], e["g"]["k"]): e for e in shapes.values() if e["g"]["wclass"] == wclass}
    out = [f"#### {title}", ""] + ([note, ""] if note else [])
    out += ["| S \\ k-tiles | " + " | ".join(str(k) for k in k_vals) + " |", "|---|" + "---:|" * len(k_vals)]
    for S in S_vals:
        out.append(f"| {S} | " + " | ".join(cell(by[(S, k)]) if (S, k) in by else "" for k in k_vals) + " |")
    return out + [""]


def heatmap(shapes, wclass, value, title, path, cmap="RdBu", center=0.0, fmt="{:+.0f}"):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False
    S_vals = sorted({e["g"]["S"] for e in shapes.values() if e["g"]["wclass"] == wclass})
    k_vals = sorted({e["g"]["k"] for e in shapes.values() if e["g"]["wclass"] == wclass})
    by = {(e["g"]["S"], e["g"]["k"]): e for e in shapes.values() if e["g"]["wclass"] == wclass}
    data = np.full((len(S_vals), len(k_vals)), np.nan)
    for i, S in enumerate(S_vals):
        for j, k in enumerate(k_vals):
            if (S, k) in by:
                v = value(by[(S, k)])
                if v is not None:
                    data[i, j] = v
    fig, ax = plt.subplots(figsize=(1.0 + 0.6 * len(k_vals), 1.0 + 0.42 * len(S_vals)))
    finite = data[np.isfinite(data)]
    lim = max(1.0, float(np.max(np.abs(finite - center)))) if finite.size else 1.0
    im = ax.imshow(data, cmap=cmap, vmin=center - lim, vmax=center + lim, aspect="auto", origin="lower")
    ax.set_xticks(range(len(k_vals)), [str(k) for k in k_vals])
    ax.set_yticks(range(len(S_vals)), [str(S) for S in S_vals])
    ax.set_xlabel("k-tiles (K / 64)")
    ax.set_ylabel("leftover tiles S")
    ax.set_title(title, fontsize=10)
    for i in range(len(S_vals)):
        for j in range(len(k_vals)):
            if np.isfinite(data[i, j]):
                ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def maps():
    shapes = load_shapes()
    L = ["# Split-rule sweep: maps", "",
         "Generated by `sweep.py maps` from `results/main.jsonl`. v20, bf16, no bias, v9 tile order, HIP device 0, "
         "cold rocprof (median kernel time of 1000 launches over a 2048 MB rotating buffer); where a configuration "
         "was repeated (each shape's top 3, capped and auto), the median of 3 processes. Configurations: `dp`, "
         "`2T` (two-tile), `rsN` (one-tile reduce-scatter on N programs per leftover tile), `owN` (one-tile, "
         "owner ending).", ""]
    for wc in (1, 0, 4):
        sub = {s: e for s, e in shapes.items() if e["g"]["wclass"] == wc}
        if not sub:
            continue
        L += [f"## {WCLASS_LABEL[wc]}", ""]

        def winner(e):
            b = best_row(e, grid_only=True)
            return label(b["config"], b["n"]) if b else ""

        def split_gain(e):
            dp, sp = e["rows"].get(("dp", None)), [r for (c, _), r in e["rows"].items() if c in ("rs", "owner")]
            if not dp or not sp:
                return None
            return pct(max(r["tflops"] for r in sp), dp["tflops"])

        def best_n(config):
            def f(e):
                b = family_best(e, config)
                return "" if b is None else str(b["n"])
            return f

        def rs_vs_owner(e):
            a, b = family_best(e, "rs"), family_best(e, "owner")
            return None if a is None or b is None else pct(a["tflops"], b["tflops"])

        def tt_vs_onetile(e):
            t = e["rows"].get(("two_tile", None))
            one = [r for (c, _), r in e["rows"].items() if c in ("rs", "owner")]
            return None if t is None or not one else pct(t["tflops"], max(r["tflops"] for r in one))

        def rule_vs_best(config):
            def f(e):
                r, b = e["rows"].get((config, None)), best_row(e)
                return None if r is None or b is None else pct(r["tflops"], b["tflops"])
            return f

        def fmt(v):
            return "" if v is None else f"{v:+.1f}"

        L += cell_table(sub, wc, winner, "Winner (fastest grid configuration)")
        L += cell_table(sub, wc, lambda e: fmt(split_gain(e)), "Best one-tile split against dp, %",
                        "Positive: splitting pays. The boundary of the positive region is k_min(S).")
        L += cell_table(sub, wc, best_n("rs"), "Best reduce-scatter n")
        L += cell_table(sub, wc, best_n("owner"), "Best owner n")
        L += cell_table(sub, wc, lambda e: fmt(rs_vs_owner(e)), "Best reduce-scatter against best owner, %")
        if wc:
            L += cell_table(sub, wc, lambda e: fmt(tt_vs_onetile(e)), "Two-tile against the best one-tile split, %")
        L += cell_table(sub, wc, lambda e: fmt(rule_vs_best("capped")(e)), "capped against the best configuration, %")
        L += cell_table(sub, wc, lambda e: fmt(rule_vs_best("auto")(e)), "auto against the best configuration, %")
        L += cell_table(sub, wc, lambda e: "" if spread(e) is None else f"{spread(e):.1f}",
                        "Process-to-process spread, %", "Median over the repeated configurations of slowest/fastest - 1.")

        for name, val, title in (("split_gain", split_gain, "best split vs dp, %"),
                                 ("capped_vs_best", rule_vs_best("capped"), "capped vs best, %"),
                                 ("two_tile", tt_vs_onetile if wc else None, "two-tile vs best one-tile, %")):
            if val is None:
                continue
            png = f"{name}_w{wc}.png"
            if heatmap(sub, wc, val, f"{WCLASS_LABEL[wc]}: {title}", os.path.join(sweep.RESULTS, png)):
                L += [f"![{title}]({png})", ""]

    noise = [abs(pct(e["rows"][("default", None)]["tflops"], e["rows"][("capped", None)]["tflops"]))
             for e in shapes.values() if ("default", None) in e["rows"] and ("capped", None) in e["rows"]]
    if noise:
        L += ["## Noise", "",
              f"`default` (STREAMK_POLICY unset) runs the same kernel as `capped`: over {len(noise)} shapes their "
              f"difference is median {statistics.median(noise):.2f}%, p90 {sorted(noise)[int(0.9 * (len(noise) - 1))]:.2f}%, "
              f"max {max(noise):.2f}%.", ""]
    write("maps.md", L)


# ---------------------------------------------------------------- rules


def feasible(S, k, n):
    return 2 <= n and S * n <= NUM_CUS and n <= k // 2


def snap(config, n, S, k):
    """The largest measured n of the ending that is <= n and feasible, else the smallest feasible."""
    ns = [m for m in (sweep.RS_N if config == "rs" else sweep.OWNER_N) if feasible(S, k, m)]
    if not ns:
        return None
    lower = [m for m in ns if m <= n]
    return max(lower) if lower else min(ns)


def pick_time(e, pick):
    """TFLOPS of a rule's pick on a grid shape (None if not measured)."""
    config, n = pick
    g = e["g"]
    if config in ("rs", "owner"):
        n = snap(config, n, g["S"], g["k"])
        if n is None:
            config, n = "dp", None
    if config == "two_tile" and g["waves"] == 0:
        config = "dp"
    r = e["rows"].get((config, n))
    return None if r is None else r["tflops"]


def score(shapes, rule):
    """(shapes scored, within 1.5% of the best, worst %, geomean %) of a rule over grid shapes."""
    gaps = []
    for e in shapes:
        b = best_row(e)
        t = pick_time(e, rule(e["g"]))
        if b is not None and t is not None:
            gaps.append(t / b["tflops"])
    if not gaps:
        return 0, 0, float("nan"), float("nan")
    return (len(gaps), sum(x >= 0.985 for x in gaps), 100 * (min(gaps) - 1),
            100 * (math.exp(statistics.fmean(math.log(x) for x in gaps)) - 1))


def measured_rule(config):
    """A baseline that was timed directly (capped, auto): its own row, not a grid lookup."""
    def rule(g):
        return (config, None)
    return rule


# Formula family: capped's structure with every constant free.
FORMULA_GRID = {
    "T": (8, 12, 16, 20, 24, 28, 32, 40, 48),  # split when (256 - S) * k >= T * 256
    "F": (4, 6, 8, 12, 16),  # reduce-scatter when 256 // S >= F
    "P": (64, 96, 128, 160, 192, 256),  # reduce-scatter programs: n <= P // S
    "ck": (2, 4, 6, 8, 12, 16),  # reduce-scatter n >= k // ck ...
    "nmin": (2, 4, 6, 8),  # ... and >= nmin
    "c0": (1.0, 1.5, 2.2, 3.0, 4.0),  # owner: n = round(sqrt(k / max(c0, S / d)))
    "d": (2, 3, 5, 8, 12),
    "tt": (None, (1, 8), (1, 12), (1, 18), (4, 8), (4, 12), (4, 18)),  # two-tile from W waves, d <= D
}
FORMULA_BLOCKS = (("T",), ("F",), ("P", "ck", "nmin"), ("c0", "d"), ("tt",))
CAPPED_PARAMS = {"T": 20, "F": 8, "P": 128, "ck": 8, "nmin": 8, "c0": 2.2, "d": 5, "tt": None}


def formula_rule(p):
    def rule(g):
        S, k, w = g["S"], g["k"], g["waves"]
        if S == 0 or 2 * S > NUM_CUS or (NUM_CUS - S) * k < p["T"] * NUM_CUS:
            return ("dp", None)
        if p["tt"] is not None and w >= p["tt"][0] and NUM_CUS // S >= 7 and k * S <= p["tt"][1] * NUM_CUS:
            return ("two_tile", None)
        if NUM_CUS // S >= p["F"]:
            return ("rs", max(2, min(p["P"] // S, NUM_CUS // S, max(p["nmin"], k // p["ck"]))))
        n = min(NUM_CUS // S, round(math.sqrt(k / max(p["c0"], S / p["d"]))))
        return ("dp", None) if n < 2 else ("owner", n)
    return rule


def fit_formula(train):
    """Coordinate descent over FORMULA_BLOCKS from capped's constants, maximising the geomean."""
    p = dict(CAPPED_PARAMS)
    best = score(train, formula_rule(p))[3]
    for _ in range(4):
        improved = False
        for block in FORMULA_BLOCKS:
            for values in itertools.product(*(FORMULA_GRID[b] for b in block)):
                q = {**p, **dict(zip(block, values))}
                s = score(train, formula_rule(q))[3]
                if s > best + 1e-9:
                    best, p, improved = s, q, True
        if not improved:
            break
    return p


def cell_key(g):
    return (g["wclass"], g["S"], g["k"])


def fit_table(train, smooth=False):
    """{(wclass, S, k): (config, n)}: the cell's fastest grid configuration, or with smooth, the one
    with the best geomean over the cell and its grid neighbours (by S and k index)."""
    by_cell = {cell_key(e["g"]): e for e in train}
    table = {}
    for (wc, S, k), e in by_cell.items():
        cands = {(c, n) for (c, n) in e["rows"] if c in GRID_CONFIGS}
        if not smooth:
            b = best_row(e, grid_only=True)
            table[(wc, S, k)] = (b["config"], b["n"])
            continue
        Si, ki = sweep.S_GRID.index(S), sweep.K_GRID.index(k)
        neigh = [by_cell.get((wc, sweep.S_GRID[i], sweep.K_GRID[j]))
                 for i in range(max(0, Si - 1), min(len(sweep.S_GRID), Si + 2))
                 for j in range(max(0, ki - 1), min(len(sweep.K_GRID), ki + 2))]
        neigh = [x for x in neigh if x is not None]

        def value(cn):
            logs = []
            for x in neigh:
                t, b = pick_time(x, cn), best_row(x)
                if t is not None and b is not None:
                    logs.append(math.log(t / b["tflops"]) * (2 if x is e else 1))
            return statistics.fmean(logs) if logs else -1e9

        table[(wc, S, k)] = max(cands, key=value)
    return table


def table_rule(table):
    """Nearest lower grid cell in S and in k (the smallest if there is none below); configurations
    that need a full wave fall back to dp without one. S > 128 runs dp."""
    cells = {}
    for (wc, S, k) in table:
        cells.setdefault(wc, (set(), set()))
        cells[wc][0].add(S)
        cells[wc][1].add(k)

    def lower(vals, x):
        below = [v for v in vals if v <= x]
        return max(below) if below else min(vals)

    def rule(g):
        S, k = g["S"], g["k"]
        if S == 0 or 2 * S > NUM_CUS:
            return ("dp", None)
        wc = 0 if g["waves"] == 0 else (1 if g["waves"] < 4 else 4)
        if wc not in cells:
            wc = min(cells, key=lambda c: abs(c - wc))
        Ss, ks = cells[wc]
        cs, ck = lower(Ss, S), lower(ks, k)
        cand = table.get((wc, cs, ck))
        if cand is None:
            return ("dp", None)
        return cand
    return rule


def cross_validate(shapes, fit_fn, make_rule, by):
    """Leave-one-value-out on S or k: fit without the value, predict the shapes that have it."""
    preds = []
    for v in sorted({e["g"][by] for e in shapes}):
        train = [e for e in shapes if e["g"][by] != v]
        test = [e for e in shapes if e["g"][by] == v]
        rule = make_rule(fit_fn(train))
        preds += [(e, rule(e["g"])) for e in test]

    def rule_from_preds(g, _cache={id(e["g"]): p for e, p in preds}):
        return _cache[id(g)]
    return score([e for e, _ in preds], rule_from_preds)


def fit():
    shapes = list(load_shapes().values())
    shapes = [e for e in shapes if best_row(e) is not None]
    candidates = {
        "formula": (fit_formula, formula_rule),
        "table": (lambda tr: fit_table(tr), table_rule),
        "table, smoothed": (lambda tr: fit_table(tr, smooth=True), table_rule),
    }
    L = ["# Split-rule sweep: rule fitting", "",
         "Generated by `sweep.py fit`. Each rule's time on a shape is the measured time of its pick; gaps are "
         "against the best measured configuration of the shape. \"within 1.5%\" counts shapes no more than 1.5% "
         "behind it. Leave-one-S-out and leave-one-k-out fit the rule without one grid value and score it on the "
         "shapes that have that value, so every shape is predicted by a rule that never saw its S (or its k).", "",
         "| rule | in-sample: within 1.5% | worst | geomean | leave-one-S-out: within 1.5% | worst | geomean "
         "| leave-one-k-out: within 1.5% | worst | geomean |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]

    def row(name, ins, cs, ck):
        f = lambda s: f"{s[1]} of {s[0]} | {s[2]:+.1f}% | {s[3]:+.2f}%"  # noqa: E731
        return f"| {name} | {f(ins)} | {f(cs)} | {f(ck)} |"

    results = {}
    for name in ("capped", "auto"):
        s = score(shapes, measured_rule(name))
        L.append(row(f"{name} (measured)", s, s, s))
    s = score(shapes, formula_rule(CAPPED_PARAMS))
    L.append(row("capped's formula on the grid's n", s, s, s))
    for name, (fit_fn, make_rule) in candidates.items():
        full = fit_fn(shapes)
        ins = score(shapes, make_rule(full))
        cs = cross_validate(shapes, fit_fn, make_rule, "S")
        ck = cross_validate(shapes, fit_fn, make_rule, "k")
        results[name] = (full, ins, cs, ck)
        L.append(row(name, ins, cs, ck))
    L += [""]

    # The choice: the best mean of the two cross-validated geomeans; the formula wins ties within 0.1%.
    cv = {name: (r[2][3] + r[3][3]) / 2 for name, r in results.items()}
    choice = max(cv, key=cv.get)
    if choice != "formula" and cv["formula"] >= cv[choice] - 0.1:
        choice = "formula"
    full = results[choice][0]
    L += [f"**Chosen: {choice}** (mean cross-validated geomean {cv[choice]:+.2f}%; "
          + ", ".join(f"{n} {v:+.2f}%" for n, v in cv.items()) + ").", ""]
    p = results["formula"][0]
    L += ["Formula constants fitted on the whole grid (capped's in brackets):", "",
          "| constant | meaning | fitted | capped |", "|---|---|---:|---:|"]
    meaning = {"T": "split when (256 - S) * k >= T * 256", "F": "reduce-scatter when 256 // S >= F",
               "P": "reduce-scatter n <= P // S", "ck": "reduce-scatter n >= k // ck", "nmin": "reduce-scatter n >= nmin",
               "c0": "owner n = round(sqrt(k / max(c0, S / d)))", "d": "(owner)", "tt": "two-tile from (W waves, d <= D)"}
    L += [f"| {c} | {meaning[c]} | {p[c]} | {CAPPED_PARAMS[c]} |" for c in FORMULA_GRID]
    L += [""]

    by_wc = {}
    rule = (formula_rule if choice == "formula" else table_rule)(full)
    for e in shapes:
        by_wc.setdefault(e["g"]["wclass"], []).append(e)
    L += ["Chosen rule (in-sample) against capped, by wave class:", "",
          "| wave class | shapes | rule: within 1.5% | geomean | capped: within 1.5% | geomean |", "|---|---:|---:|---:|---:|---:|"]
    for wc in (0, 1, 4):
        es = by_wc.get(wc, [])
        if es:
            a, c = score(es, rule), score(es, measured_rule("capped"))
            L.append(f"| {WCLASS_LABEL[wc]} | {a[0]} | {a[1]} | {a[3]:+.2f}% | {c[1]} | {c[3]:+.2f}% |")
    L += [""]

    out = {"family": choice}
    if choice == "formula":
        out["params"] = full
    else:
        out["table"] = [[wc, S, k, c, n] for (wc, S, k), (c, n) in sorted(full.items())]
    with open(os.path.join(sweep.RESULTS, "rule.json"), "w") as f:
        json.dump(out, f, indent=1)
    L += ["Written to `results/rule.json`."]
    write("fit.md", L)


# ---------------------------------------------------------------- validation

RULE_POLICY = "split_rule"
AITER = os.path.join(sweep.REPO, "experiments", "aiter_lixun_shapes")
# A new tag reruns every shape (the runner skips the shapes a tag already has).
REAL_TAG = os.environ.get("SPLIT_RULE_REAL_TAG", "split_rule")


KERNEL = os.path.join(sweep.REPO, "kernels", "gemm", "intra_wave", "a16w16", "v20_streamk_reduce_scatter",
                      "matmul_kernel.py")


def kernel_policy(src):
    """split_rule_policy (and what it needs) executed from the kernel's source, without Triton."""
    import ast

    tree = ast.parse(src)
    body = [x for x in tree.body if (isinstance(x, ast.FunctionDef) and x.name in ("capped_policy", "split_rule_policy"))
            or (isinstance(x, ast.Assign) and getattr(x.targets[0], "id", "").startswith("SPLIT_RULE"))]
    scope = {"math": math}
    exec(compile(ast.Module(body=body, type_ignores=[]), KERNEL, "exec"), scope)
    return scope["split_rule_policy"]


def install():
    """Write results/rule.json's table into the kernel's SPLIT_RULE_TABLE (host code only), check
    that split_rule_policy then picks what the table picks on every grid shape, and drop the
    off-grid split_rule rows (results/validate.jsonl is backed up first) so validate re-times them."""
    import re

    rule = json.load(open(os.path.join(sweep.RESULTS, "rule.json")))
    if rule["family"] != "table":
        sys.exit(f"install: the fitted rule is {rule['family']!r}; only a table can be installed")
    table = {(wc, S, k): (c, n) for wc, S, k, c, n in rule["table"]}
    lines = ["SPLIT_RULE_TABLE = {"]
    for wc in (0, 1, 4):
        lines.append(f"    {wc}: {{")
        for S in sweep.S_GRID:
            cells = " ".join(f"{label(*table[(wc, S, k)]):>4s}" for k in sweep.K_GRID)
            lines.append(f'        {S}: "{cells}",')
        lines.append("    },")
    lines.append("}")
    src = open(KERNEL).read()
    new, count = re.subn(r"SPLIT_RULE_TABLE = \{\n.*?\n\}\n", "\n".join(lines) + "\n", src, flags=re.S)
    assert count == 1, count
    policy = kernel_policy(new)
    shapes = load_shapes()
    rule_fn = table_rule(table)
    mismatches = 0
    for e in shapes.values():
        g = e["g"]
        p, fixup, programs = policy((g["M"] // 256) * (g["N"] // 256), g["k"], NUM_CUS)
        got = {"dp": ("dp", None), "two_tile": ("two_tile", None)}.get(
            p, (fixup, programs // g["S"] if programs else NUM_CUS // g["S"]))
        want = rule_fn(g)
        if want[0] in ("rs", "owner"):
            want = (want[0], snap(want[0], want[1], g["S"], g["k"]))
        mismatches += got != want
    assert mismatches == 0, f"{mismatches} grid shapes where the kernel and the table disagree"
    if new != src:
        with open(KERNEL + ".tmp", "w") as f:
            f.write(new)
        os.replace(KERNEL + ".tmp", KERNEL)
    path = os.path.join(sweep.RESULTS, "validate.jsonl")
    if os.path.exists(path):
        backup = os.path.join(sweep.RESULTS, "validate.pre_lane_order.jsonl")
        if not os.path.exists(backup):
            import shutil

            shutil.copy2(path, backup)
        rows = sweep.load("validate")
        kept = [r for r in rows if r["config"] != RULE_POLICY]
        with open(path + ".tmp", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in kept)
        os.replace(path + ".tmp", path)
        print(f"install: dropped {len(rows) - len(kept)} off-grid {RULE_POLICY} rows", flush=True)
    print(f"install: SPLIT_RULE_TABLE {'updated' if new != src else 'unchanged'}; the kernel matches the table on "
          f"{len(shapes)} grid shapes", flush=True)


def validate():
    """Off-grid shapes: dp, capped, auto and the new rule, each to 3 processes, then the score."""
    sweep.sweep("validate", list(sweep.offgrid_cases((RULE_POLICY,))))
    sweep.repeat("validate", 3, everything=True)
    rows = sweep.load("validate")
    configs = ("dp", "capped", "auto", RULE_POLICY)
    by, scores = validation_score(rows, configs)
    L = ["# Split-rule sweep: off-grid validation", "",
         f"Generated by `sweep.py validate`. {len(by)} shapes with S and k between the grid's values (S = "
         f"{', '.join(map(str, sweep.OFF_S))}; k-tiles = {', '.join(map(str, sweep.OFF_K))}) and other aspect "
         "ratios, timed like the grid (cold rocprof, 2048 MB rotating buffer, GPU 0), median of 3 processes. "
         "Gaps are against the best of the four measured configurations.", "",
         "| rule | shapes | within 1.5% | worst | geomean |", "|---|---:|---:|---:|---:|"]
    L += [f"| {c} | {s[0]} | {s[1]} | {s[2]:+.1f}% | {s[3]:+.2f}% |" for c, s in scores.items()]
    L += ["", "By wave class (geomean against the best):", "",
          "| wave class | shapes | " + " | ".join(configs) + " |", "|---|---:|" + "---:|" * len(configs)]
    for wc in (0, 1, 4):
        sub = [r for r in rows if r.get("wclass") == wc]
        b2, s2 = validation_score(sub, configs)
        if b2:
            L.append(f"| {WCLASS_LABEL[wc]} | {len(b2)} | " + " | ".join(
                f"{s2[c][3]:+.2f}%" if c in s2 else "-" for c in configs) + " |")
    L += ["", "Per shape (TFLOPS; the rule's gap to capped):", "",
          "| shape | waves | S | k-tiles | dp | capped | auto | rule | rule vs capped |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, s in sorted(by.items(), key=lambda kv: (next(iter(kv[1].values()))["waves"], next(iter(kv[1].values()))["S"])):
        r0 = next(iter(s.values()))
        t = {c: s[c]["tflops"] if c in s else float("nan") for c in configs}
        L.append(f"| {name} | {r0['waves']} | {r0['S']} | {r0['k']} | " + " | ".join(f"{t[c]:.0f}" for c in configs)
                 + f" | {pct(t[RULE_POLICY], t['capped']):+.1f}% |")
    bad = [r for r in rows if not sweep.good(r)]
    L += ["", f"Failed or wrong: {len(bad)}."] + [f"- {r['shape']} {r['config']}: {r['correct']}, {r['error']}" for r in bad]
    write("validate.md", L)


def realmodel():
    """The 630 real-model shapes with the rule as a column (v20 only, do_bench_cudagraph), then the score."""
    runner = os.path.join(AITER, "run_v20_vs_aiter.py")
    subprocess.run([sys.executable, runner, "sweep", "--tag", REAL_TAG, "--v20-only", "--rule-policy", RULE_POLICY],
                   cwd=AITER, check=True)
    subprocess.run([sys.executable, runner, "report", "--tag", REAL_TAG,
                    "--baseline", os.path.join(AITER, "results", "details_default.csv")], cwd=AITER, check=True)
    rows = list(csv.DictReader(open(os.path.join(AITER, "results", f"details_{REAL_TAG}.csv"), newline="")))

    def f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return float("nan")

    scored = []
    for r in rows:
        if r["v20_ok"] != "True" or r.get("v20_rule_ok") == "False" or r.get("v20_capped_ok") == "False":
            continue
        auto = f(r["v20_tflops"])
        dp = auto if r["v20_policy"] == "dp" else f(r["v20_dp_tflops"])
        capped = dp if r["v20_capped_policy"] == "dp" else f(r["v20_capped_tflops"])
        rule = dp if r["v20_rule_policy"] == "dp" else f(r["v20_rule_tflops"])
        two = auto if r["v20_policy"] == "two_tile" else f(r["v20_two_tile_tflops"])
        cols = {"auto": auto, "dp": dp, "capped": capped, "rule": rule, "two_tile": two}
        cols = {c: x for c, x in cols.items() if x == x}
        if not {"auto", "capped", "rule"} <= set(cols):
            continue
        scored.append((r, cols, max(cols.values())))
    L = ["# Split-rule sweep: real-model validation", "",
         f"Generated by `sweep.py realmodel`. The 630 shapes of `experiments/aiter_lixun_shapes` with "
         f"`run_v20_vs_aiter.py sweep --tag {REAL_TAG} --v20-only --rule-policy {RULE_POLICY}` (do_bench_cudagraph, "
         "back to back, 2 interleaved rounds). Gaps are against the best v20 column of each shape (auto, dp, "
         "capped, the rule, and two-tile where a full wave and 256 // S >= 7 allow it).", "",
         "| rule | shapes | within 1.5% | worst | geomean |", "|---|---:|---:|---:|---:|"]
    for c in ("auto", "capped", "rule"):
        gaps = [cols[c] / b for _, cols, b in scored]
        L.append(f"| {c} | {len(gaps)} | {sum(x >= 0.985 for x in gaps)} | {100 * (min(gaps) - 1):+.1f}% | "
                 f"{100 * (math.exp(statistics.fmean(math.log(x) for x in gaps)) - 1):+.2f}% |")

    def bucket(r):
        w = int(r["waves"])
        return "no full wave" if w == 0 else "1-3 full waves" if w < 4 else "4+ full waves"

    L += ["", "By wave class (geomean against the best; rule against capped, median):", "",
          "| wave class | shapes | auto | capped | rule | rule vs capped, median | worst | best |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for b in ("no full wave", "1-3 full waves", "4+ full waves"):
        sub = [(r, cols, best) for r, cols, best in scored if bucket(r) == b]
        if not sub:
            continue
        geo = {c: 100 * (math.exp(statistics.fmean(math.log(cols[c] / best) for _, cols, best in sub)) - 1)
               for c in ("auto", "capped", "rule")}
        rc = [pct(cols["rule"], cols["capped"]) for _, cols, _ in sub]
        L.append(f"| {b} | {len(sub)} | {geo['auto']:+.2f}% | {geo['capped']:+.2f}% | {geo['rule']:+.2f}% | "
                 f"{statistics.median(rc):+.1f}% | {min(rc):+.1f}% | {max(rc):+.1f}% |")
    worst = sorted(scored, key=lambda t: t[1]["rule"] / t[1]["capped"])[:15]
    L += ["", "Where the rule is furthest behind capped:", "",
          "| shape | waves | S | k-tiles | rule picks | capped picks | dp | capped | rule | rule vs capped |",
          "|---|---:|---:|---:|---|---|---:|---:|---:|---:|"]
    for r, cols, _ in worst:
        L.append(f"| {r['M']}x{r['N']}x{r['K']} | {r['waves']} | {r['S']} | {r['k_steps']} | {r['v20_rule_label']} | "
                 f"{r['v20_capped_label']} | {cols['dp']:.0f} | {cols['capped']:.0f} | {cols['rule']:.0f} | "
                 f"{pct(cols['rule'], cols['capped']):+.1f}% |")
    write("realmodel.md", L)


def validation_score(rows, configs):
    """Per shape: the best of the measured configurations; per rule: within 1.5%, worst, geomean."""
    by = {}
    for r in rows:
        if sweep.good(r):
            by.setdefault(r["shape"], {})[r["config"]] = r
    out = {}
    for c in configs:
        gaps = [s[c]["tflops"] / max(x["tflops"] for x in s.values()) for s in by.values() if c in s]
        if gaps:
            out[c] = (len(gaps), sum(x >= 0.985 for x in gaps), 100 * (min(gaps) - 1),
                      100 * (math.exp(statistics.fmean(math.log(x) for x in gaps)) - 1))
    return by, out


def write(name, L):
    os.makedirs(sweep.RESULTS, exist_ok=True)
    text = "\n".join(L) + "\n"
    with open(os.path.join(sweep.RESULTS, name), "w") as f:
        f.write(text)
    print(text, flush=True)
