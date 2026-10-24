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
"""The two-stage plan's second stage: the best tile per shape, every tile at its own split table's pick.

    python v27_streamk_tile/tilesel.py plan     # shapes, points, estimate
    python v27_streamk_tile/tilesel.py run      # results/split/T_r*.csv, resumable
    python v27_streamk_tile/tilesel.py report   # results/split/tilesel_report.md

Shapes (seeded): a stratified re-timed subset of REFIT's (so the old and new runs meet on the same shapes), gap
shapes REFIT did not cover (tall-skinny, wide-short, large, M or N of 1-448), and a stratified subset of the AITER
real-model shapes, held out from any fitting. Each shape is one block: every candidate tile at its table pick
(results/split/tables.json, the cell at or below the tile's S and k-steps in its wave class), the table's runner-up
and cap, plus v26 (cap at 256x256x64); small tiles for M (or N, at the transposed twin tile) of at most 448. Raced as
split_sweep.py's stages, on the same GPU, in the same session.
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
import split_extra as se  # noqa: E402
import split_report as sr  # noqa: E402
import split_sweep as ss  # noqa: E402
import sweep  # noqa: E402

SEED = ss.SEED + 2
STAGE = "T"
REFIT_CB, REFIT_MB, AITER = 70, 50, 150
GAP = (  # (group, count, M range, N range, K range); M or N given as a tuple of values is drawn from it
    ("tall-skinny", 60, (8192, 65536), (256, 2048), (512, 32768)),
    ("wide-short", 50, (512, 2048), (16384, 131072), (512, 32768)),
    ("large", 30, (8192, 32768), (8192, 65536), (512, 16384)),
    ("small-M", 100, (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 320, 384, 448), (256, 131072),
     (1024, 32768)),
    ("small-N", 60, (4096, 131072), (1, 2, 4, 8, 16, 32, 64, 128, 256, 448), (1024, 32768)),
)
MNK_MAX, NK_MAX = 4e12, 2 ** 30
SMALL_DIM = 448
FEW_QUAD = ("64x64x128", "64x128x128", "128x128x64")


def log_uniform(rng, lo, hi, step=256):
    return max(lo, min(hi, round(2 ** rng.uniform(math.log2(lo), math.log2(hi)) / step) * step))


def shapes():
    """{(M, N, K): group}."""
    rng = random.Random(SEED)
    out = {}
    cb, mb = sweep.refit_shapes()
    bins = collections.defaultdict(list)
    for sh in cb:
        bins[min(sweep.shape_info("256x256x64", *sh)["waves"], 4)].append(sh)
    for w, v in sorted(bins.items()):
        for sh in rng.sample(v, min(len(v), round(REFIT_CB * len(v) / len(cb)))):
            out[sh] = "refit-cb"
    for sh in rng.sample(mb, REFIT_MB):
        out[sh] = "refit-mb"
    for g, n, mr, nr, kr in GAP:
        made = 0
        while made < n:
            M = rng.choice(mr) if len(mr) > 2 else log_uniform(rng, *mr)
            N = rng.choice(nr) if len(nr) > 2 else log_uniform(rng, *nr)
            K = log_uniform(rng, *kr)
            if M * N * K > MNK_MAX:
                K = max(kr[0], int(MNK_MAX / (M * N)) // 256 * 256)
            if min(M, N) <= SMALL_DIM and max(M, N) * K > NK_MAX:
                K = max(kr[0], NK_MAX // max(M, N) // 256 * 256)
            if (M, N, K) in out or M * N * K > MNK_MAX:
                continue
            out[(M, N, K)] = g
            made += 1
    held = sweep.heldout_shapes()
    hb = collections.defaultdict(list)
    for sh in held:
        hb[min(sweep.shape_info("256x256x64", *sh)["waves"], 4)].append(sh)
    for w, v in sorted(hb.items()):
        for sh in rng.sample(v, min(len(v), round(AITER * len(v) / len(held)))):
            out.setdefault(sh, "aiter")
    return out


def tiles_for(M, N):
    small_m, small_n = M <= SMALL_DIM, N <= SMALL_DIM and N < M
    quad = list(ss.QUAD_TILES) if min(M, N) >= 64 else list(FEW_QUAD)
    if small_m:
        return quad + list(ss.SMALL_TILES)
    if small_n:
        return quad + [se.TWIN[t] for t in ss.SMALL_TILES]
    return quad


TABLE_TILE = {v: k for k, v in se.TWIN.items()}


def table_kinds(tables, tile, M, N, K, top=2):
    """The tile's table pick and runner-up at the shape (the cell at or below its S and k-steps), the next ones in the
    cell's order where those are not valid at the shape; dp and pdp when there are no leftover tiles."""
    sh = sweep.shape_info(tile, M, N, K)
    if sh["S"] == 0:
        return ["dp", "pdp"]
    wc = 0 if sh["waves"] == 0 else (1 if sh["waves"] < 4 else 4)
    t = tables.get(f"{TABLE_TILE.get(tile, tile)} w{wc}")
    if not t:
        return ["cap"]
    grid = {(c["S"], c["k"]): c for c in t["cells"]}
    cell = sr.lookup(grid, sh["S"], sh["k_steps"])
    order = sorted(cell["vals"], key=cell["vals"].get)
    pick = cell["pick"]
    order = [pick] + [k for k in order if k != pick]
    return [k for k in order if sweep.valid(k, tile, M, N, K)][:top]


def blocks(tables):
    out, groups = {}, {}
    for (M, N, K), g in sorted(shapes().items()):
        pts = {sweep.point_name("cap", "256x256x64", M, N, K)}
        for t in tiles_for(M, N):
            for k in table_kinds(tables, t, M, N, K) + ["cap"]:
                if sweep.valid(k, t, M, N, K):
                    pts.add(sweep.point_name(k, t, M, N, K))
        b = f"{M}x{N}x{K}"
        out[b], groups[b] = sorted(pts), g
    return out, groups


def load_tables():
    """The tables the tile selection was timed with (tables_tilesel.json once it exists, else the current ones)."""
    frozen = os.path.join(ss.OUT, "tables_tilesel.json")
    return json.load(open(frozen if os.path.exists(frozen) else os.path.join(ss.OUT, "tables.json")))


def plan(args):
    b, g = blocks(load_tables())
    n = sum(map(len, b.values()))
    est = sum(sweep.point_cost(p) for v in b.values() for p in v)
    print(f"{len(b)} shapes ({dict(collections.Counter(g.values()))}), {n} points, one pass {est / 3600:.2f} h")


def run(args):
    os.environ.setdefault("TRITON_CACHE_DIR", ss.TRITON_CACHE)
    m = ss.manifest()
    ss.check_frozen(m)
    path = os.path.join(ss.OUT, "tilesel_blocks.json")
    if os.path.exists(path):
        d = json.load(open(path))
        b, g = d["blocks"], d["groups"]
    else:
        b, g = blocks(load_tables())
        json.dump(dict(blocks=b, groups=g), open(path, "w"), indent=0)
    ss.race(STAGE, b, m, args.jobs)
    ss.log("tile selection done")


def report(args):
    d = json.load(open(os.path.join(ss.OUT, "tilesel_blocks.json")))
    b, g = d["blocks"], d["groups"]
    val, _ = ss.relative(b, {r: ss.round_us(STAGE, r) for r in (1, 2, 3)})
    tables = load_tables()
    v27 = sweep.v27_module()
    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    for blk, pts in b.items():
        M, N, K = (int(x) for x in blk.split("x"))
        vv = {p: val[p] for p in pts if p in val}
        if not vv:
            continue
        best = min(vv.values())
        get = lambda k, t: vv.get(sweep.point_name(k, t, M, N, K))  # noqa: E731
        per_tile = {}
        for t in tiles_for(M, N):
            ks = table_kinds(tables, t, M, N, K, top=1)
            if ks and get(ks[0], t):
                per_tile[t] = get(ks[0], t)
        grp = g[blk]
        rows[grp]["v26 (cap at 256x256x64)"].append(get("cap", "256x256x64") / best)
        auto = v27.auto_tile(M, N, K) if M >= 512 else "256x256x64"
        if get("cap", auto):
            rows[grp]["auto_tile + cap"].append(get("cap", auto) / best)
        if per_tile:
            rows[grp]["oracle tile at its table pick"].append(min(per_tile.values()) / best)
        for t in ("256x256x64",):
            if t in per_tile:
                rows[grp][f"{t} at its table pick"].append(per_tile[t] / best)
    L = ["# Tile selection (tilesel.py)", "", "Time over the best point timed for the shape (same block): median / 90th / "
         "worst / geomean, and shapes above 1.05.", "", "| group | selection | shapes | regret |", "|---|---|---|---|"]
    for grp in sorted(rows):
        for sel, xs in rows[grp].items():
            L.append(f"| {grp} | {sel} | {len(xs)} | {statistics.median(xs):.3f} / {sr.q90(xs):.3f} / {max(xs):.3f} / "
                     f"{sr.gm(xs):.3f}, {sum(x > 1.05 for x in xs)} |")
    open(os.path.join(ss.OUT, "tilesel_report.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["plan", "run", "report"])
    ap.add_argument("--jobs", type=int, default=32)
    args = ap.parse_args()
    {"plan": plan, "run": run, "report": report}[args.mode](args)


if __name__ == "__main__":
    main()
