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
"""Checks on split_sweep.py's shortcuts, timed after it in the same session on the same GPU (results/split).

    python v27_streamk_tile/split_extra.py run       # AUD, GEO, ORI, WAV stages, resumable
    python v27_streamk_tile/split_extra.py report    # results/split/extra_report.md

- AUD: a seeded random AUDIT_SHARE of stage A's cells, every method (all n, both endings, all, pdp{P}) timed in
  rounds 1 and 2 without elimination, round 3 for close calls. The adaptive pick's loss there is an unbiased estimate
  of what racing, coarse n and the coarse grid cost.
- GEO: GEO_SHARE of stage A's cells at their other geometry (square against skinny, one row of small tiles against
  several), raced like stage A: does the pick hold?
- ORI: ORI_CELLS small-tile cells as the transposed shape at the twin tile (1x16x256 -> 16x1x256, ...): is the
  table the same for both orientations?
- WAV: WAV_CELLS quadrant cells at the other wave counts of their class (2 and 3 for class 1, 5-7 for class 4).
"""

import argparse
import collections
import json
import math
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import split_report as sr  # noqa: E402
import split_sweep as ss  # noqa: E402
import sweep  # noqa: E402

AUDIT_SHARE = 0.03
GEO_SHARE = 0.08
ORI_CELLS = 36
WAV_CELLS = 60
SEED = ss.SEED + 1
TWIN = {"1x16x256": "16x1x256", "16x16x256": "16x16x256", "16x16x128": "16x16x128", "16x32x128": "32x16x128",
        "32x32x128": "32x32x128", "32x128x128": "128x32x128"}
ALL_N = (2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32)


def full_methods(tile, M, N, K, wc):
    kinds = ["dp", "pdp", "cap", "all", "2t", "sp"] + [f"pdp{p}" for p in sweep.PDP_PROGRAMS]
    kinds += [f"{e}{n}" for e in ("rs", "ow") for n in ALL_N + ("all",)]
    out = []
    for k in kinds:
        if sweep.pdp_programs(k) and sweep.pdp_programs(k) > sweep.PDP_MAX_PROGRAMS.get(tile, 256):
            continue
        if k == "2t" and wc == 0:
            continue
        if sweep.valid(k, tile, M, N, K):
            out.append(k)
    return out


def other_shape(c):
    """The cell's shape at its other geometry, or None if the tile count has only one."""
    bm, bn, bk, P, swap, small = ss.geom(c["tile"])
    for w in ss.WAVE_CLASSES[c["wc"]]:
        T = w * P + c["S"]
        opts = ("row", "square") if small else ("square", "skinny-m", "skinny-n")
        for which in opts:
            sh = ss.shape_for(c["tile"], T, c["K"], which)
            if sh and sh != (c["M"], c["N"], c["K"]) and sweep.shape_info(c["tile"], *sh)["waves"] == \
                    sweep.shape_info(c["tile"], c["M"], c["N"], c["K"])["waves"]:
                return sh
    return None


def stage_cells():
    rng = random.Random(SEED)
    A = ss.load_cells("A")
    out = {}
    by_tile = collections.defaultdict(list)
    for c in A:
        by_tile[c["tile"]].append(c)
    aud = [c for t in sorted(by_tile) for c in rng.sample(by_tile[t], max(1, round(AUDIT_SHARE * len(by_tile[t]))))]
    out["AUD"] = [dict(c, stage="AUD", kinds=full_methods(c["tile"], c["M"], c["N"], c["K"], c["wc"])) for c in aud]
    geo = []
    for c in rng.sample(A, round(GEO_SHARE * len(A))):
        sh = other_shape(c)
        if sh:
            geo.append(dict(ss.make_cell(c["tile"], c["wc"], c["S"], c["k"], sh, "GEO"), ref=c["id"]))
    out["GEO"] = geo
    ori = []
    small = [c for c in A if c["tile"] in TWIN]
    for c in rng.sample(small, min(ORI_CELLS, len(small))):
        t2 = TWIN[c["tile"]]
        sh = (c["N"], c["M"], c["K"])
        info = sweep.shape_info(t2, *sh)
        if (info["S"], info["k_steps"]) == (c["S"], sweep.shape_info(c["tile"], c["M"], c["N"], c["K"])["k_steps"]):
            ori.append(dict(ss.make_cell(t2, c["wc"], c["S"], c["k"], sh, "ORI"), ref=c["id"]))
    out["ORI"] = ori
    wav = []
    quad = [c for c in A if c["tile"] in ss.QUAD_TILES and c["wc"] > 0]
    for c in rng.sample(quad, min(WAV_CELLS, len(quad))):
        P = ss.geom(c["tile"])[3]
        w0 = sweep.shape_info(c["tile"], c["M"], c["N"], c["K"])["waves"]
        for w in rng.sample([w for w in ss.WAVE_CLASSES[c["wc"]] if w != w0], len(ss.WAVE_CLASSES[c["wc"]]) - 1):
            sh = ss.shape_for(c["tile"], w * P + c["S"], c["K"], "square")
            if sh:
                wav.append(dict(ss.make_cell(c["tile"], c["wc"], c["S"], c["k"], sh, "WAV"), ref=c["id"], w=w))
                break
    out["WAV"] = wav
    return out


def blocks_of(cells, full=False):
    return {c["id"]: ss.block_points(c, c["kinds"] if full else ss.methods(c["tile"], c["M"], c["N"], c["K"], c["wc"]))
            for c in cells}


def time_all(stage, blocks, m, jobs):
    """Rounds 1 and 2 of every point, round 3 for the two leaders of a block when within sigma."""
    ss.warm([p for v in blocks.values() for p in v], jobs)
    ss.time_blocks(dict(blocks), stage, 1, m)
    ss.time_blocks(dict(blocks), stage, 2, m)
    val, _ = ss.relative(blocks, {1: ss.round_us(stage, 1), 2: ss.round_us(stage, 2)})
    close = {}
    for b, pts in blocks.items():
        s = sorted((p for p in pts if p in val), key=val.get)
        if len(s) > 1 and val[s[1]] <= val[s[0]] * (1 + ss.SIGMA_DEFAULT):
            close[b] = s[:2]
    ss.time_blocks(dict(close), stage, 3, m)


def run(args):
    os.environ.setdefault("TRITON_CACHE_DIR", ss.TRITON_CACHE)
    m = ss.manifest()
    ss.check_frozen(m)
    path = os.path.join(ss.OUT, "extra_cells.json")
    cells = json.load(open(path)) if os.path.exists(path) else stage_cells()
    json.dump(cells, open(path, "w"), indent=0)
    for st in ("AUD", "GEO", "ORI", "WAV"):
        ss.log(f"{st}: {len(cells[st])} cells")
        if st == "AUD":
            time_all(st, blocks_of(cells[st], full=True), m, args.jobs)
        else:
            ss.race(st, blocks_of(cells[st]), m, args.jobs)
    ss.log("extra stages done")


def report(args):
    cells = json.load(open(os.path.join(ss.OUT, "extra_cells.json")))
    _, vals = sr.load()
    L = ["# Checks on the split sweep's shortcuts (split_extra.py)", ""]
    for st in ("AUD", "GEO", "ORI", "WAV"):
        cs = cells[st]
        blocks = blocks_of(cs, full=st == "AUD")
        v, _ = ss.relative(blocks, {r: ss.round_us(st, r) for r in (1, 2, 3)})
        losses, flips, alls = [], 0, []
        for c in cs:
            cv = ss.cell_val(c, v)
            if not cv:
                continue
            ref = vals.get(c.get("ref", c["id"]))
            if not ref:
                continue
            pk = ss.pick_kind(ref)
            losses.append(cv.get(pk, max(cv.values())))
            flips += ss.pick_kind(cv) != pk
            if st == "AUD" and "all" in cv and c["S"] * 2 > ss.geom(c["tile"])[3]:
                alls.append(cv["all"])
        L += [f"## {st}: {len(losses)} cells", "",
              f"The sweep's pick at these cells: time over the best timed here, median {statistics.median(losses):.3f}, "
              f"90th {sr.q90(losses):.3f}, worst {max(losses):.3f}, geomean {sr.gm(losses):.3f}; above sigma "
              f"{sum(x > 1 + ss.SIGMA_DEFAULT for x in losses)}, above 1.05 {sum(x > 1.05 for x in losses)}; a "
              f"different pick in {flips}." if losses else "nothing timed", ""]
        if alls:
            L += [f"The uneven split (`all`) at S > P/2: {len(alls)} cells, geomean {sr.gm(alls):.3f}, best "
                  f"{min(alls):.3f}, within sigma of the best in {sum(x <= 1 + ss.SIGMA_DEFAULT for x in alls)}.", ""]
    open(os.path.join(ss.OUT, "extra_report.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["run", "report", "plan"])
    ap.add_argument("--jobs", type=int, default=32)
    args = ap.parse_args()
    if args.mode == "plan":
        cells = stage_cells()
        for st, cs in cells.items():
            b = blocks_of(cs, full=st == "AUD")
            n = sum(map(len, b.values()))
            est = sum(sweep.point_cost(p) for v in b.values() for p in v)
            print(f"{st}: {len(cs)} cells, {n} points, one pass {est / 3600:.2f} h")
        return
    {"run": run, "report": report}[args.mode](args)


if __name__ == "__main__":
    main()
