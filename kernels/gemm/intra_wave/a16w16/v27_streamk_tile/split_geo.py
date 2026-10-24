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
"""A second geometry for every split-table cell (results/split, the same GPU and manifest as split_sweep.py).

    python v27_streamk_tile/split_geo.py plan
    python v27_streamk_tile/split_geo.py run      # stages G2a, G2b, G2c, resumable

Each cell of stages A, B1 and B2 not already in split_extra.py's GEO stage is timed at its other geometry
(split_extra.other_shape: square against skinny; one row of small tiles against several), with the methods within
TOP_LIMIT of the cell's best at its first geometry (at most TOP_N) plus dp and cap. On GEO's 248 cells that set held
the other geometry's best within 0.02% (geomean; worst 2%). Raced as split_sweep.py's stages, in order of value:
G2a small tiles and quadrant tiles without a full wave, G2b quadrant tiles with 1-3 waves, G2c with 4 or more.
split_report.py then picks each cell's method over both geometries.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import split_extra as se  # noqa: E402
import split_report as sr  # noqa: E402
import split_sweep as ss  # noqa: E402
import sweep  # noqa: E402

TOP_LIMIT, TOP_N = 1.25, 10
STAGES = ("G2a", "G2b", "G2c")


def stage_of(c):
    if c["tile"] in ss.SMALL_TILES or c["wc"] == 0:
        return "G2a"
    return "G2b" if c["wc"] == 1 else "G2c"


def cells():
    path = os.path.join(ss.OUT, "geo2_cells.json")
    if os.path.exists(path):
        return json.load(open(path))
    _, vals = sr.load(geometries=False)
    done = {c["ref"] for c in json.load(open(os.path.join(ss.OUT, "extra_cells.json")))["GEO"]}
    out = {s: [] for s in STAGES}
    for st in ("A", "B1", "B2"):
        for c in ss.load_cells(st):
            v = vals.get(c["id"])
            if c["id"] in done or not v:
                continue
            sh = se.other_shape(c)
            if not sh:
                continue
            top = [k for k in sorted(v, key=v.get) if v[k] <= TOP_LIMIT][:TOP_N]
            kinds = [k for k in dict.fromkeys(top + ["dp", "cap"]) if sweep.valid(k, c["tile"], *sh)]
            g = ss.make_cell(c["tile"], c["wc"], c["S"], c["k"], sh, stage_of(c))
            out[stage_of(c)].append(dict(g, ref=c["id"], kinds=kinds))
    json.dump(out, open(path, "w"), indent=0)
    return out


def blocks_of(cs):
    return {c["id"]: ss.block_points(c, c["kinds"]) for c in cs}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["plan", "run"])
    ap.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args()
    cs = cells()
    if args.mode == "plan":
        for st in STAGES:
            b = blocks_of(cs[st])
            n = sum(map(len, b.values()))
            est = sum(sweep.point_cost(p) for v in b.values() for p in v)
            print(f"{st}: {len(cs[st])} cells, {n} points, one pass {est / 3600:.2f} h")
        return
    os.environ.setdefault("TRITON_CACHE_DIR", ss.TRITON_CACHE)
    m = ss.manifest()
    ss.check_frozen(m)
    for st in STAGES:
        ss.race(st, blocks_of(cs[st]), m, args.jobs)
        ss.log(f"{st} done")
    ss.log("second geometry done")


if __name__ == "__main__":
    main()
