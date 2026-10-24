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
"""The split tables from split_sweep.py's timings (results/split), with their win/loss matrices.

    python v27_streamk_tile/split_report.py        # results/split/tables.json, tables.md

Per tile and wave class: the S x k-steps matrix of picks (the lowest merged time, ties within DELTA to the simpler
method), cap's time over the cell's best in each cell, and per method its wins, the cells within sigma of the best,
and its geomean and worst loss over the cells where it was timed. The table is scored as the kernel would use it (the
cell at or below S and k, split_rule_policy's lookup), leaving out one S row or one k column at a time and predicting
the left-out cells from the rest. Stage N's extra n are put on their cell's scale through the leader and runner-up
they were timed with.
"""

import collections
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import split_sweep as ss  # noqa: E402
import sweep  # noqa: E402

SIGMA = ss.SIGMA_DEFAULT


def gm(xs):
    xs = list(xs)
    return math.exp(statistics.mean(math.log(x) for x in xs)) if xs else float("nan")


def q90(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(0.9 * len(xs)))] if xs else float("nan")


def second_geometry():
    """{cell id: {kind: time over the best}} at the cell's other geometry (split_extra.py's GEO, split_geo.py's G2)."""
    out = {}
    path = os.path.join(ss.OUT, "extra_cells.json")
    if os.path.exists(path):
        cs = json.load(open(path))["GEO"]
        blocks = {c["id"]: ss.block_points(c, ss.methods(c["tile"], c["M"], c["N"], c["K"], c["wc"])) for c in cs}
        v, _ = ss.relative(blocks, {r: ss.round_us("GEO", r) for r in (1, 2, 3)})
        out.update({c["ref"]: ss.cell_val(c, v) for c in cs})
    path = os.path.join(ss.OUT, "geo2_cells.json")
    if os.path.exists(path):
        for st, cs in json.load(open(path)).items():
            blocks = {c["id"]: ss.block_points(c, c["kinds"]) for c in cs}
            v, _ = ss.relative(blocks, {r: ss.round_us(st, r) for r in (1, 2, 3)})
            out.update({c["ref"]: ss.cell_val(c, v) for c in cs})
    return {k: v for k, v in out.items() if v}


def load(geometries=True):
    """[cell], {cell id: {kind: time over the cell's best}}; geometries: over both of the cell's geometries where the
    other one was timed (the geometric mean of the two, over the kinds timed at both)."""
    stages = [s for s in ("A", "B1", "B2") if ss.load_cells(s)]
    cells = [c for s in stages for c in ss.load_cells(s)]
    val = {}
    for s in stages:
        v, _ = ss.relative(ss.stage_blocks(ss.load_cells(s)), {r: ss.round_us(s, r) for r in (1, 2, 3)})
        val.update(v)
    nblocks = collections.defaultdict(list)
    for r in ss.read_rows(ss.round_path("N", 1)):
        if r.get("is_ref") == "0":
            nblocks[r["block"]].append(sweep.point_key(r))
    valn, _ = ss.relative(nblocks, {r: ss.round_us("N", r) for r in (1, 2, 3)})
    nextra = {b[:-len("|n")]: {sweep.parse_point(p)[0]: valn[p] for p in pts if p in valn} for b, pts in nblocks.items()}
    out = {}
    for c in cells:
        v = ss.cell_val(c, val)
        ne = nextra.get(c["id"])
        common = [k for k in (ne or {}) if k in v]
        if common:
            scale = gm(v[k] / ne[k] for k in common)
            v.update({k: x * scale for k, x in ne.items() if k not in v})
            b = min(v.values())
            v = {k: x / b for k, x in v.items()}
        if v:
            out[c["id"]] = v
    if geometries:
        for cid, v2 in second_geometry().items():
            v1 = out.get(cid)
            common = [k for k in v1 or {} if k in v2]
            if common:
                v = {k: math.sqrt(v1[k] * v2[k]) for k in common}
                b = min(v.values())
                out[cid] = {k: x / b for k, x in v.items()}
    return [c for c in cells if c["id"] in out], out


def lookup(table, S, k):
    """split_rule_policy's lookup: the cell at or below S and k (the smallest if none is below)."""
    Ss = sorted({s for s, _ in table})
    s_cell = max((x for x in Ss if x <= S), default=Ss[0])
    ks = sorted({kk for s, kk in table if s == s_cell})
    k_cell = max((x for x in ks if x <= k), default=ks[0])
    return table[(s_cell, k_cell)]


def bracket(grid, S, k):
    """The method for (S, k) from {(S, k): {kind: time over the cell's best}}: the kind with the smallest summed log
    time over the cells bracketing S and k (the rows at or just below and just above S, the same for k; one cell
    when (S, k) is on the grid), among the kinds timed in all of them; the at-or-below cell's pick if none is."""
    Ss = sorted({s for s, _ in grid})
    lo = max((x for x in Ss if x <= S), default=Ss[0])
    hi = min((x for x in Ss if x >= S), default=Ss[-1])
    cells = []
    for s in {lo, hi}:
        ks = sorted({kk for ss_, kk in grid if ss_ == s})
        klo = max((x for x in ks if x <= k), default=ks[0])
        khi = min((x for x in ks if x >= k), default=ks[-1])
        cells += [grid[(s, kk)] for kk in {klo, khi}]
    common = set(cells[0]).intersection(*cells[1:])
    if not common:
        return ss.pick_kind(grid[(lo, max(kk for s_, kk in grid if s_ == lo and kk <= k) if any(
            s_ == lo and kk <= k for s_, kk in grid) else min(kk for s_, kk in grid if s_ == lo))])
    score = {kd: sum(math.log(c[kd]) for c in cells) for kd in common}
    best = min(score.values())
    tie = [kd for kd in common if score[kd] <= best + len(cells) * math.log(1 + ss.DELTA)]
    return min(tie, key=lambda kd: (ss.SIMPLE.index(kd) if kd in ss.SIMPLE else 99, score[kd]))


def cross_validate(cs, vals, axis, rule="lower"):
    """Losses of the left-out cells, each predicted from the table without its S row (axis S) or k column (axis k),
    by the at-or-below cell's pick (rule lower) or bracket()."""
    grid = {(c["S"], c["k"]): vals[c["id"]] for c in cs}
    out = []
    for c in cs:
        rest = {key: v for key, v in grid.items() if (key[0] if axis == "S" else key[1]) != c[axis]}
        if not rest:
            continue
        if rule == "lower":
            p = lookup({key: ss.pick_kind(v) for key, v in rest.items()}, c["S"], c["k"])
        else:
            p = bracket(rest, c["S"], c["k"])
        out.append(vals[c["id"]].get(p, max(vals[c["id"]].values())))
    return out


def main():
    cells, vals = load()
    tables, L, summary = {}, ["# Split tables", "", "From results/split (split_sweep.py), physical GPU 2, one session. "
                              "Every value is a time over its cell's best, within the cell's block.", ""], []
    for tile in ss.QUAD_TILES + ss.SMALL_TILES:
        for wc in ss.WAVE_CLASSES:
            cs = [c for c in cells if c["tile"] == tile and c["wc"] == wc]
            if not cs:
                continue
            Ss, ks = sorted({c["S"] for c in cs}), sorted({c["k"] for c in cs})
            grid = {(c["S"], c["k"]): vals[c["id"]] for c in cs}
            ent = []
            for c in sorted(cs, key=lambda c: (c["S"], c["k"])):
                v = vals[c["id"]]
                pk = ss.pick_kind(v)
                others = sorted((x for k_, x in v.items() if k_ != pk))
                ent.append(dict(S=c["S"], k=c["k"], M=c["M"], N=c["N"], K=c["K"], pick=pk,
                                runner_up=min((k_ for k_ in v if k_ != pk), key=v.get, default=None),
                                margin=round(others[0] / v[pk], 4) if others else None,
                                vals={k_: round(x, 4) for k_, x in sorted(v.items(), key=lambda kv: kv[1])}))
            tables[f"{tile} w{wc}"] = dict(S=Ss, k=ks, cells=ent)
            caps = [v["cap"] for v in grid.values() if "cap" in v]
            dps = [v["dp"] for v in grid.values() if "dp" in v]
            cvS, cvk = cross_validate(cs, vals, "S"), cross_validate(cs, vals, "k")
            bS, bk = cross_validate(cs, vals, "S", "bracket"), cross_validate(cs, vals, "k", "bracket")
            summary.append((tile, wc, len(cs), caps, dps, cvS, cvk, bS, bk))
            L += [f"## {tile}, wave class {wc} ({len(cs)} cells)", "",
                  "Pick per cell (rows S, columns k-steps), and cap's time over the cell's best in brackets.", "",
                  "| S \\ k | " + " | ".join(map(str, ks)) + " |", "|---" * (len(ks) + 1) + "|"]
            for S in Ss:
                row = []
                for k in ks:
                    v = grid.get((S, k))
                    row.append(f"{ss.pick_kind(v)} ({v['cap']:.2f})" if v and "cap" in v else
                               (ss.pick_kind(v) or "") if v else "")
                L.append(f"| {S} | " + " | ".join(row) + " |")
            stats = collections.defaultdict(lambda: [0, 0, []])
            for v in grid.values():
                pk = ss.pick_kind(v)
                stats[pk][0] += 1
                for k_, x in v.items():
                    stats[k_][2].append(x)
                    stats[k_][1] += x <= 1 + SIGMA
            L += ["", "| method | wins | within sigma | cells timed | geomean loss | 90th | worst |",
                  "|---|---|---|---|---|---|---|"]
            for k_, (w, ws, xs) in sorted(stats.items(), key=lambda kv: (-kv[1][0], -kv[1][1])):
                L.append(f"| {k_} | {w} | {ws} | {len(xs)} | {gm(xs):.3f} | {q90(xs):.3f} | {max(xs):.3f} |")
            L.append("")
    head = ["## Summary", "",
            "Geomean / worst time over each cell's best (over both geometries where both were timed): cap and dp in "
            "every cell; the table leaving out its S row or k column (cross-validated), read at or below (the cell at "
            "or below S and k) or by bracket (the method best over the cells either side).", "",
            "| tile | waves | cells | cap | dp | S out, at or below | S out, bracket | k out, at or below | "
            "k out, bracket |", "|---|---|---|---|---|---|---|---|---|"]
    f2 = lambda xs: f"{gm(xs):.3f} / {max(xs or [1]):.2f}"  # noqa: E731
    for tile, wc, n, caps, dps, cvS, cvk, bS, bk in summary:
        head.append(f"| {tile} | {wc} | {n} | {f2(caps)} | {f2(dps)} | {f2(cvS)} | {f2(bS)} | {f2(cvk)} | {f2(bk)} |")
    allc = lambda i: [x for row in summary for x in row[i]]  # noqa: E731
    head.append(f"| all | | {sum(r[2] for r in summary)} | {f2(allc(3))} | {f2(allc(4))} | {f2(allc(5))} | "
                f"{f2(allc(7))} | {f2(allc(6))} | {f2(allc(8))} |")
    L[4:4] = head + [""]
    json.dump(tables, open(os.path.join(ss.OUT, "tables.json"), "w"), indent=0)
    open(os.path.join(ss.OUT, "tables.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(head))


if __name__ == "__main__":
    main()
