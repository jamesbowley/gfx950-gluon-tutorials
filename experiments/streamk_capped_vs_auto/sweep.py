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
"""v20's capped and auto host rules on shapes that capped was not fit to (v20 WORKLOG entry 7).

    export PYTHONPATH=<repo>/scripts/triton_deepbind_shim:<triton_gfx950-tutorial-v2.2>/python
    HIP_VISIBLE_DEVICES=1 python experiments/streamk_capped_vs_auto/sweep.py main 0/3
    HIP_VISIBLE_DEVICES=1 python experiments/streamk_capped_vs_auto/sweep.py split 0/3
    HIP_VISIBLE_DEVICES=1 python experiments/streamk_capped_vs_auto/sweep.py repeat 0/3 3
    python experiments/streamk_capped_vs_auto/sweep.py score [real-model details CSV]   # results/score.md

The measurement is streamk_version_tables/sweep.py's (cold rocprof over 1000 launches on a
rotating 512 MB, plus warm L2 counters), on a held-out grid: leftover tiles S = 8, 24, 48 and
96 with 1, 4, 8 and 16 full waves. Where 256 * waves + S only factors into a very skinny
shape, the nearest S that factors well is used (26, 10 and 52). None of these S values is in
the fit set.

`i/n` runs shard i of n: every n-th shape, in its own results/shard<i>_of<n>/ directory, so
each GPU runs whole shapes and the configurations of one shape are always compared on one GPU.

`score` reads every shard, and the real-model sweep in experiments/aiter_lixun_shapes
(results/details_default.csv, from `run_v20_vs_aiter.py sweep --tag default --v20-only`), and
scores auto, capped and hybrid(W, D) against the best measured configuration of each shape.
"""

import csv
import glob
import importlib.util
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(HERE, "results")
REAL_CSV = os.path.join(REPO, "experiments", "aiter_lixun_shapes", "results", "details_default.csv")

_spec = importlib.util.spec_from_file_location(
    "streamk_version_tables_sweep", os.path.join(REPO, "experiments", "streamk_version_tables", "sweep.py"))
vt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vt)

NUM_CUS = vt.NUM_CUS


def factor(tiles):
    """(M, N), multiples of 256, with M * N / 256**2 = tiles and the aspect ratio closest to 1."""
    tm = min((a for a in range(1, tiles + 1) if tiles % a == 0), key=lambda a: abs(math.log(a * a / tiles)))
    return 256 * tm, 256 * (tiles // tm)


# (full waves, S, K). S = 26, 10 and 52 stand in for 24, 8 and 48, whose tile counts only factor
# into shapes 8-32 times longer than wide.
GRID = [(w, s, 6144) for w, ss in ((1, (8, 24, 48, 96)), (4, (8, 26, 48, 96)), (8, (10, 24, 52, 96)),
                                   (16, (8, 24, 48, 96))) for s in ss]
GRID += [(w, s, 3072) for w, ss in ((1, (8, 24, 48, 96)), (8, (10, 24, 52, 96))) for s in ss]
GRID += [(w, s, 8192) for w, ss in ((8, (10, 24, 52, 96)), (16, (8, 24, 48, 96))) for s in ss]
SHAPES = []
for w, s, K in GRID:
    M, N = factor(NUM_CUS * w + s)
    g = vt.geometry(M, N, K)
    assert (g["waves"], g["S"]) == (w, s), (M, N, K, g)
    SHAPES.append((M, N, K, "held-out"))

# "v20 default" leaves STREAMK_POLICY unset: the confirmatory run of the new default (capped, with
# n = 1 run as data-parallel).
CONFIGS = {"v13": (13, {}), "v19": (19, {})} | {k: (20, env) for k, env in vt.V20.items()} | {"v20 default": (20, {})}
ONE_TILE = ["v20 owner", "v20 rs"]

vt.SHAPES = SHAPES
vt.MAIN = CONFIGS
vt.ONE_TILE = ONE_TILE


def shard(spec):
    """Shape names and results directory of shard "i/n". Shapes are dealt out largest first, so
    every shard gets a similar amount of work."""
    i, n = (int(x) for x in spec.split("/"))
    order = sorted(SHAPES, key=lambda s: -s[0] * s[1] * s[2])
    names = {vt.shape_name(M, N, K) for M, N, K, _ in order[i::n]}
    return names, os.path.join(RESULTS, f"shard{i}_of{n}")


# ---------------------------------------------------------------- score


def load_heldout():
    """{shape: {"g": geometry, "main": {config: row}, "split": {(config, n): row}}} over every shard."""
    shapes = {}
    for which in ("main", "split"):
        for path in sorted(glob.glob(os.path.join(RESULTS, "shard*", f"{which}.jsonl"))):
            vt.RESULTS = os.path.dirname(path)
            for r in vt.load(which):
                e = shapes.setdefault(r["shape"], {"g": vt.geometry(r["M"], r["N"], r["K"]), "main": {}, "split": {}})
                if which == "main":
                    e["main"][r["config"]] = r
                else:
                    e["split"][(r["config"], r["n"])] = r
    return shapes


def hybrid_two_tile(g, W, D):
    """hybrid(W, D) runs two-tile where it has W or more full waves, every tile would be split at
    least 7 ways and the two-tile lag d = k-tiles * S / 256 is at most D k-steps."""
    S = g["S"]
    return 0 < S and g["waves"] >= W and NUM_CUS // S >= 7 and g["k_tiles"] * S <= D * NUM_CUS


W_GRID = (1, 2, 4, 8)
D_GRID = (8, 12, 18)


def score_heldout(shapes, auto_fn, capped_fn):
    """Per shape: TFLOPS of every rule's pick, the best measured configuration, and the spread."""
    out = []
    for name, e in shapes.items():
        main = {c: r for c, r in e["main"].items() if vt.good(r)}
        if not all(c in main for c in ("v20 auto", "v20 capped", "v20 two_tile", "v13")):
            continue
        g = e["g"]
        cands = list(main.values()) + [r for r in e["split"].values() if vt.good(r)]
        best = max(cands, key=lambda r: r["tflops"])
        spreads = [100 * (max(r["us_runs"]) / min(r["us_runs"]) - 1) for r in main.values() if len(r.get("us_runs", [])) > 1]
        c_policy, c_fixup, c_programs = capped_fn(g["tiles"], g["k_tiles"], NUM_CUS)
        rules = {"auto": main["v20 auto"]["tflops"], "capped": main["v20 capped"]["tflops"]}
        one_split = c_policy == "one_tile" and c_programs == g["S"]
        rules["capped, n = 1 as dp"] = main["v20 dp"]["tflops"] if one_split and "v20 dp" in main else rules["capped"]
        for W in W_GRID:
            for D in D_GRID:
                pick = "v20 two_tile" if hybrid_two_tile(g, W, D) else "v20 capped"
                rules[f"hybrid({W}, {D})"] = main[pick]["tflops"]
        rules["best of capped, two-tile"] = max(main["v20 capped"]["tflops"], main["v20 two_tile"]["tflops"])
        if "v20 default" in main:
            rules["default (measured)"] = main["v20 default"]["tflops"]
        out.append(dict(shape=name, g=g, rules=rules, best=best["tflops"],
                        best_cfg=f"{best['config']}" + ("" if best["n"] == "default" else f" n {best['n']}"),
                        v13=main["v13"]["tflops"], two_tile=main["v20 two_tile"]["tflops"],
                        capped=main["v20 capped"]["tflops"], auto=main["v20 auto"]["tflops"],
                        auto_pick=auto_fn(g["tiles"], g["k_tiles"], NUM_CUS),
                        capped_pick=vt.capped_label(g, (c_policy, c_fixup, c_programs)),
                        split=e["split"], main=main,
                        spread=statistics.median(spreads) if spreads else None))
    return sorted(out, key=lambda r: (r["g"]["k_tiles"], r["g"]["waves"], r["g"]["S"]))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_real(path):
    """Per real-model shape: TFLOPS of every rule's pick and the best measured v20 configuration.
    A column that was not timed because it would run the same schedule as another takes that
    column's time."""
    if not os.path.exists(path):
        return []
    out = []
    for r in csv.DictReader(open(path, newline="")):
        if r["v20_ok"] != "True" or r.get("v20_capped_ok") not in ("True", ""):
            continue
        M, N, K = int(r["M"]), int(r["N"]), int(r["K"])
        g = vt.geometry(M, N, K)
        auto_t = fnum(r["v20_tflops"])
        policy = r["v20_policy"]
        dp_t = auto_t if policy == "dp" else fnum(r["v20_dp_tflops"])
        capped_t = fnum(r["v20_capped_tflops"])
        if capped_t != capped_t:
            capped_t = dp_t if r["v20_capped_policy"] == "dp" else float("nan")
        two_t = auto_t if policy == "two_tile" else fnum(r["v20_two_tile_tflops"])
        if capped_t != capped_t or auto_t != auto_t:
            continue
        rules = {"auto": auto_t, "capped": capped_t,
                 "capped, n = 1 as dp": dp_t if r["v20_capped_label"].endswith("n = 1") else capped_t}
        for W in W_GRID:
            for D in D_GRID:
                use_two = hybrid_two_tile(g, W, D) and two_t == two_t
                rules[f"hybrid({W}, {D})"] = two_t if use_two else capped_t
        rules["best of capped, two-tile"] = max(capped_t, two_t) if two_t == two_t else capped_t
        measured = {c: x for c, x in (("auto", auto_t), ("dp", dp_t), ("capped", capped_t), ("two-tile", two_t))
                    if x == x}
        best_cfg = max(measured, key=measured.get)
        out.append(dict(shape=f"{M}x{N}x{K}", out=r["out"], g=g, rules=rules, best=measured[best_cfg],
                        best_cfg=best_cfg, dp=dp_t,
                        aiter=fnum(r["aiter_pick_tflops"]), auto_pick=policy, capped_pick=r["v20_capped_label"],
                        two_tile=two_t, capped=capped_t, auto=auto_t))
    return out


def rule_stats(rows, rule, tol=1.5):
    """(shapes, within tol% of the best, worst loss %, geomean against the best %)."""
    gaps = [100 * (r["rules"][rule] / r["best"] - 1) for r in rows]
    if not gaps:
        return 0, 0, float("nan"), float("nan")
    geo = 100 * (math.exp(statistics.fmean(math.log(r["rules"][rule] / r["best"]) for r in rows)) - 1)
    return len(gaps), sum(g >= -tol for g in gaps), min(gaps), geo


def bucket(g):
    w = g["waves"]
    return "no full wave" if w == 0 else "1 wave" if w == 1 else "2-7 waves" if w < 8 else "8+ waves"


def between_launches():
    """Two-tile against capped in each between-launch mode of v20's between_launches.py
    (results/between_launches_raw.md, kernel times in us)."""
    path = os.path.join(RESULTS, "between_launches_raw.md")
    if not os.path.exists(path):
        return []
    lines = [l for l in open(path) if l.startswith("| ") and not l.startswith("| shape")]
    head = next(l for l in open(path) if l.startswith("| shape"))
    modes = [m.strip() for m in head.strip().strip("|").split("|")[2:]]
    us = {}
    for l in lines:
        cells = [c.strip() for c in l.strip().strip("|").split("|")]
        us[(cells[0], cells[1])] = [float(x) for x in cells[2:]]
    shapes = sorted({s for s, _ in us}, key=lambda s: math.prod(int(x) for x in s.split("x")))
    L = ["## Two-tile against capped between launches", "",
         "`between_launches.py`: CUDA events around each GEMM, 100 launches on the same inputs, median of "
         "launches 10-99, with different work between launches. Positive means two-tile is faster than capped. "
         "\"nothing\" is only valid for kernels over about 200 us.", "",
         "| shape | " + " | ".join(modes) + " |", "|---|" + "---:|" * len(modes)]
    wins = {m: [] for m in modes}
    for s in shapes:
        c, t = us.get((s, "v20 policy=capped")), us.get((s, "v20 policy=two_tile"))
        if not c or not t:
            continue
        gaps = [100 * (a / b - 1) for a, b in zip(c, t)]
        for m, x in zip(modes, gaps):
            wins[m].append(x)
        L.append(f"| {s} | " + " | ".join(f"{x:+.1f}%" for x in gaps) + " |")
    L += ["", "| mode | shapes | two-tile ahead by > 1% | capped ahead by > 1% | median |", "|---|---:|---:|---:|---:|"]
    for m in modes:
        xs = wins[m]
        if xs:
            L.append(f"| {m} | {len(xs)} | {sum(x > 1 for x in xs)} | {sum(x < -1 for x in xs)} | "
                     f"{statistics.median(xs):+.1f}% |")
    return L + [""]


def score(real_csv=REAL_CSV):
    auto_fn, capped_fn = vt.v20_host_functions("streamk_policy", "capped_policy")
    held = score_heldout(load_heldout(), auto_fn, capped_fn)
    real = load_real(real_csv)
    rules = (["auto", "capped", "capped, n = 1 as dp"] + [f"hybrid({W}, {D})" for W in W_GRID for D in D_GRID]
             + ["best of capped, two-tile"])
    L = ["# Capped vs auto on held-out and real-model shapes", "",
         "Generated by `sweep.py score`. Gaps are against the best measured configuration of each shape: on the "
         "held-out grid every main configuration and every split count n, on the real-model set the v20 columns "
         "that were timed (auto, dp, capped, two-tile). \"Within 1.5%\" counts shapes no more than 1.5% behind "
         "it. \"capped, n = 1 as dp\" runs data-parallel where capped's split is one program per tile. "
         "hybrid(W, D) is capped's pick, except two-tile where there are at least W full waves, "
         "`256 // S >= 7` and `d = k-tiles * S / 256 <= D`. \"Best of capped, two-tile\" is the bound for any "
         "rule that chooses between the two. \"default (measured)\" is v20 with STREAMK_POLICY unset after "
         "entry 7 (capped, n = 1 as data-parallel), timed in its own processes.", ""]

    def stats_table(rows, title):
        out = [f"## {title}", "", "| rule | shapes | within 1.5% | worst | geomean |", "|---|---:|---:|---:|---:|"]
        measured = ["default (measured)"] if rows and all("default (measured)" in r["rules"] for r in rows) else []
        for rule in rules + measured:
            n, ok, worst, geo = rule_stats(rows, rule)
            out.append(f"| {rule} | {n} | {ok} | {worst:+.1f}% | {geo:+.2f}% |")
        return out + [""]

    def bucket_table(rows, title):
        out = [f"### {title}: capped against auto and two-tile, by full waves", "",
               "| full waves | shapes | capped vs auto, median | worst | best | capped vs two-tile, median | "
               "two-tile ahead by > 1% |", "|---|---:|---:|---:|---:|---:|---:|"]
        groups = {}
        for r in rows:
            groups.setdefault(bucket(r["g"]), []).append(r)
        for b in ("no full wave", "1 wave", "2-7 waves", "8+ waves"):
            rs = groups.get(b, [])
            if not rs:
                continue
            ca = [100 * (r["capped"] / r["auto"] - 1) for r in rs]
            ct = [100 * (r["capped"] / r["two_tile"] - 1) for r in rs if r["two_tile"] == r["two_tile"]]
            out.append(f"| {b} | {len(rs)} | {statistics.median(ca):+.1f}% | {min(ca):+.1f}% | {max(ca):+.1f}% | "
                       + (f"{statistics.median(ct):+.1f}% | {sum(x < -1 for x in ct)} of {len(ct)} |" if ct else "- | - |"))
        return out + [""]

    L += stats_table(held, f"Held-out grid ({len(held)} shapes, cold rocprof)")
    L += bucket_table(held, "Held-out grid")
    L += stats_table(real, f"Real-model shapes ({len(real)} shapes, do_bench_cudagraph, "
                           f"`{os.path.relpath(real_csv, REPO)}`)")
    L += bucket_table(real, "Real-model shapes")

    L += ["## Held-out grid, per shape", "",
          "TFLOPS. Spread is the median over configurations of slowest/fastest process - 1.", "",
          "| shape | waves | S | k-tiles | d | auto picks | capped picks | v13 | auto | capped | two-tile | best "
          "| best config | capped vs best | two-tile vs capped | spread |",
          "|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|"]
    for r in held:
        g = r["g"]
        d = g["k_tiles"] * g["S"] / NUM_CUS
        L.append(f"| {r['shape']} | {g['waves']} | {g['S']} | {g['k_tiles']} | {d:g} | {' '.join(r['auto_pick'])} | "
                 f"{r['capped_pick']} | {r['v13']:.0f} | {r['auto']:.0f} | {r['capped']:.0f} | {r['two_tile']:.0f} | "
                 f"{r['best']:.0f} | {r['best_cfg']} | {vt.pct(r['capped'], r['best'])} | "
                 f"{vt.pct(r['two_tile'], r['capped'])} | "
                 + ("-" if r["spread"] is None else f"{r['spread']:.1f}%") + " |")
    L += [""]

    L += ["## Held-out grid: capped's split count against the best n", "",
          "For shapes where capped picks one-tile: capped's TFLOPS, and the best n of the same ending.", "",
          "| shape | S | k-tiles | capped picks | capped | best n of that ending | its TFLOPS | capped vs it |",
          "|---|---:|---:|---|---:|---|---:|---:|"]
    for r in held:
        pick = r["capped_pick"]
        if not pick.startswith("one-tile"):
            continue
        cfg = "v20 rs" if " rs" in pick else "v20 owner"
        cands = [(n, x) for (c, n), x in r["split"].items() if c == cfg and vt.good(x)]
        if cfg in r["main"]:
            cands.append(("default", r["main"][cfg]))
        if not cands:
            continue
        bn, bx = max(cands, key=lambda e: e[1]["tflops"])
        L.append(f"| {r['shape']} | {r['g']['S']} | {r['g']['k_tiles']} | {pick} | {r['capped']:.0f} | {cfg} n {bn} | "
                 f"{bx['tflops']:.0f} | {vt.pct(r['capped'], bx['tflops'])} |")
    L += [""]

    lost = sorted((r for r in real if r["capped"] < r["best"] * 0.985), key=lambda r: r["capped"] / r["best"])
    L += [f"## Real-model shapes where capped is more than 1.5% behind the best v20 column ({len(lost)})", "",
          "`(256 - S) * k` is the idle share of the data-parallel last wave in k-steps * 256; capped splits when "
          "it is at least 20 * 256 = 5120.", "",
          "| shape | out | waves | S | k-tiles | (256 - S) * k | auto picks | capped picks | dp | auto | capped "
          "| two-tile | best | capped vs best |",
          "|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---|---:|"]
    for r in lost:
        g = r["g"]
        L.append(f"| {r['shape']} | {r['out']} | {g['waves']} | {g['S']} | {g['k_tiles']} | "
                 f"{(NUM_CUS - g['S']) * g['k_tiles']} | {r['auto_pick']} | {r['capped_pick']} | {r['dp']:.0f} | "
                 f"{r['auto']:.0f} | {r['capped']:.0f} | "
                 + ("-" if r["two_tile"] != r["two_tile"] else f"{r['two_tile']:.0f}")
                 + f" | {r['best_cfg']} | {vt.pct(r['capped'], r['best'])} |")
    L += [""]

    close = [r for r in held + real if r["two_tile"] == r["two_tile"] and abs(r["two_tile"] / r["capped"] - 1) <= 0.03
             and hybrid_two_tile(r["g"], 1, 18)]
    L += [f"## Close calls for the clean-cache check ({len(close)})", "",
          "Shapes where two-tile is allowed and within 3% of capped.", "",
          ", ".join(sorted({r["shape"] for r in close})), ""]
    L += between_launches()

    text = "\n".join(L) + "\n"
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "score.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode in ("main", "split"):
        names, vt.RESULTS = shard(sys.argv[2])
        vt.sweep(mode, names)
    elif mode == "repeat":
        names, vt.RESULTS = shard(sys.argv[2])
        vt.repeat("main", int(sys.argv[3]), names)
    elif mode == "shapes":
        for M, N, K, _ in SHAPES:
            g = vt.geometry(M, N, K)
            print(f"{vt.shape_name(M, N, K)}: {g['tiles']} tiles, {g['waves']} waves, S = {g['S']}, "
                  f"{g['k_tiles']} k-tiles")
    elif mode == "score":
        score(*sys.argv[2:3])
    else:
        sys.exit(f"unknown mode {mode}")
