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
"""Figures for the v9_tile worklog, from results/*.csv.

    python v9_tile/plot_tiles.py     # writes v9_tile/images/*.png

- tile_count.png (measurement C): back-to-back TFLOPS against N, M = K = 4096, one curve per
  tile, with each tile's ceiling from the per-CU load limit: a CU moves at most ~60 B/ns, so it
  computes at most 60 B/ns x BM*BN/(BM+BN) FLOP/B (and at most its 2.4 GHz peak), and a launch
  of T tiles in ceil(T/256) waves of non-persistent workgroups averages T / ceil(T/256) busy CUs.
  The flat line is the power-capped compute ceiling, v9's best full-chip number on this GPU.
- same_shape.png (measurement B): each shape under each tile.
"""

import csv
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
IMG = os.path.join(HERE, "images")
CU_BNS = 60.0  # bytes per ns per CU
NUM_CUS = 256
COLORS = {"256x256": "tab:blue", "256x128": "tab:orange", "128x256": "tab:green", "128x128": "tab:red",
          "128x128x128": "tab:purple"}


def rows(name):
    p = os.path.join(RES, name)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def dims(tile):
    d = [int(x) for x in tile.removesuffix("r2").split("x")]
    return d[0], d[1], d[2] if len(d) > 2 else 64


def ceiling(tile, tiles, compute_cap):
    BM, BN, _ = dims(tile)
    per_cu = CU_BNS * BM * BN / (BM + BN) / 1e3  # TFLOPS per CU
    busy = tiles / math.ceil(tiles / NUM_CUS)
    return min(busy * per_cu, compute_cap * busy / NUM_CUS)


def tile_count(compute_cap):
    data = rows("c_timing_graph.csv") + rows("c128_timing_graph.csv")
    if not data:
        return
    fig, ax = plt.subplots(figsize=(10, 5.5))
    peak_24 = 1048.576 * 2.4  # TFLOPS at the 2.4 GHz clock limit
    for tile in dict.fromkeys(r["tile"] for r in data):
        BM, BN, _ = dims(tile)
        pts = sorted((int(r["N"]), float(r["tflops"])) for r in data if r["tile"] == tile)
        x = [p[0] for p in pts]
        ax.plot(x, [p[1] for p in pts], "o-", ms=3, color=COLORS.get(tile), label=f"{tile} measured")
        if tile in ("128x256", "128x128x128"):
            continue  # same ceiling as its mirror / its BK=64 twin
        xs = list(range(BN, max(x) + 1, BN))
        ax.plot(xs, [ceiling(tile, (4096 // BM) * (n // BN), peak_24) for n in xs], "--", lw=1,
                color=COLORS.get(tile), label=f"{tile} per-CU load ceiling (60 B/ns)")
    ax.axhline(compute_cap, color="gray", lw=1, ls=":", label=f"power-capped compute (v9 best, {compute_cap:.0f})")
    for n in (4096, 8192):
        ax.axvline(n, color="lightgray", lw=0.8)
    ax.set_xlabel("N (M = K = 4096); grey lines: one and two full waves of 256x256")
    ax.set_ylabel("TFLOPS, back to back (bf16)")
    ax.set_title("Throughput against N, per tile size (GPU 2, v9_tile, llir+amdgcnas, bf16)")
    ax.set_ylim(0, compute_cap * 1.15)
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "tile_count.png"), dpi=130)


def same_shape():
    data = rows("b_timing_graph.csv") + [r for r in rows("b128_timing_graph.csv") if not r["tile"].endswith("r2")]
    if not data:
        return
    shapes = list(dict.fromkeys(f"{r['M']}x{r['N']}x{r['K']}" for r in data))
    tiles = list(dict.fromkeys(r["tile"] for r in data))
    by = {(f"{r['M']}x{r['N']}x{r['K']}", r["tile"]): r for r in data}
    fig, ax = plt.subplots(figsize=(12, 5))
    w = 0.8 / len(tiles)
    for i, t in enumerate(tiles):
        xs, ys = [], []
        for j, s in enumerate(shapes):
            r = by.get((s, t))
            if r:
                xs.append(j + (i - (len(tiles) - 1) / 2) * w)
                ys.append(float(r["tflops"]))
        bars = ax.bar(xs, ys, w, color=COLORS.get(t), label=t)
        for b, (x, y) in zip(bars, zip(xs, ys)):
            tl = by[(shapes[round(x)], t)]["tiles"]
            ax.text(x, y + 15, tl, ha="center", fontsize=6, rotation=90)
    ax.set_xticks(range(len(shapes)))
    ax.set_xticklabels(shapes, rotation=20, fontsize=8)
    ax.set_ylabel("TFLOPS, back to back (bf16)")
    ax.set_title("Same shape, different tile (numbers on bars: tile count)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "same_shape.png"), dpi=130)


def main():
    os.makedirs(IMG, exist_ok=True)
    best = [float(r["tflops"]) for r in rows("a_timing_graph.csv") if r["tile"] in ("v9", "256x256")]
    cap = max(best) if best else 1600.0
    tile_count(cap)
    same_shape()


if __name__ == "__main__":
    main()
