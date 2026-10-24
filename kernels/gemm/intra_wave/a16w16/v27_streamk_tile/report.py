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
"""Tables and figures from sweep.py's CSVs (no GPU).

    python v27_streamk_tile/report.py wq    # results/wq_report.md, images/wq_k{4096,8192}.png
    python v27_streamk_tile/report.py mb    # results/mb_report.md, images/mb.png
"""

import collections
import csv
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
IMG = os.path.join(HERE, "images")
WQ_TILES = ("256x256", "256x128", "128x256", "128x128x128")


def load(name):
    path = os.path.join(RES, name)
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def wq():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(IMG, exist_ok=True)
    out = ["# Wave quantization per tile: v27 stream-K against v9_tile data-parallel\n",
           "M = 4096, N = 256 .. 16384, bf16, back to back (HIP graph), 2 GB of rotating inputs, rocprof median of "
           "100 dispatches ([sweep.py](../sweep.py) WQ4096 / WQ8192). `dp`: v9_tile at the tile; `cap`: v27 at the "
           "tile as deployed (capped policy; a data-parallel choice of one wave or less runs v9_tile); `all`: v27's "
           "one-tile split over all 256 programs (v9_tile §13's sk256).\n"]
    for K in (4096, 8192):
        rows = [r for r in load(f"wq{K}.csv") if r["correct"] == "True"]
        if not rows:
            continue
        bad = [r for r in load(f"wq{K}.csv") if r["correct"] != "True"]
        d = collections.defaultdict(dict)
        for r in rows:
            d[(r["tile"], r["kind"])][int(r["N"])] = float(r["tflops"])
        Ns = sorted({int(r["N"]) for r in rows})
        fig, ax = plt.subplots(figsize=(13, 5))
        for i, t in enumerate(WQ_TILES):
            col = f"C{i}"
            for kind, style in (("dp", ":"), ("cap", "-")):
                xs = sorted(d[(t, kind)])
                ax.plot(xs, [d[(t, kind)][x] for x in xs], style, color=col, label=f"{t} {kind}")
            xs = sorted(d[(t, "all")])
            ax.plot(xs, [d[(t, "all")][x] for x in xs], "o", color=col, ms=4, label=f"{t} all")
        ax.set_xlabel("N (M = 4096)")
        ax.set_ylabel("TFLOPS")
        ax.set_title(f"K = {K}: dotted data-parallel (v9_tile), solid v27 capped, dots v27 one-tile on all programs")
        ax.grid(alpha=0.3)
        ax.legend(ncol=4, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(IMG, f"wq_k{K}.png"), dpi=110)
        plt.close(fig)
        out.append(f"\n## K = {K}\n")
        out.append(f"{len(rows)} points correct, {len(bad)} not.\n")
        out.append(f"![K = {K}](../images/wq_k{K}.png)\n")
        # Per N: best dp tile, best v27 capped tile; v27 capped at each tile over dp at the same tile.
        out.append("| N | best dp (tile) | best v27 capped (tile) | capped / best dp | "
                   + " | ".join(f"cap / dp @{t}" for t in WQ_TILES) + " |")
        out.append("|---|---|---|---|" + "---|" * len(WQ_TILES))
        ratios = collections.defaultdict(list)
        best_ratio = []
        for n in Ns:
            dps = {t: d[(t, "dp")].get(n) for t in WQ_TILES if d[(t, "dp")].get(n)}
            caps = {t: d[(t, "cap")].get(n) for t in WQ_TILES if d[(t, "cap")].get(n)}
            if not dps or not caps:
                continue
            bd = max(dps, key=dps.get)
            bc = max(caps, key=caps.get)
            best_ratio.append((caps[bc] / dps[bd], n))
            cells = []
            for t in WQ_TILES:
                if t in dps and t in caps:
                    ratios[t].append(caps[t] / dps[t])
                    cells.append(f"{caps[t] / dps[t]:.3f}")
                else:
                    cells.append("")
            if n % 1024 == 256 or n % 4096 in (512, 1024, 2048) or n % 4096 == 0:
                out.append(f"| {n} | {dps[bd]:.0f} ({bd}) | {caps[bc]:.0f} ({bc}) | {caps[bc] / dps[bd]:.3f} | "
                           + " | ".join(cells) + " |")
        out.append("\nOver every N (table rows are a subset): v27 capped against data-parallel at the same tile, "
                   "median / min / max:\n")
        for t in WQ_TILES:
            xs = ratios[t]
            out.append(f"- {t}: {statistics.median(xs):.3f} / {min(xs):.3f} / {max(xs):.3f}")
        xs = [r for r, _ in best_ratio]
        out.append(f"- best v27 capped tile against best data-parallel tile, per N: median {statistics.median(xs):.3f}, "
                   f"min {min(xs):.3f} (N = {min(best_ratio)[1]}), max {max(xs):.3f} (N = {max(best_ratio)[1]})")
        # The 'all' points against capped and dp at the same tile.
        out.append("\nOne-tile on all programs (`all`, §13's 16 N), TFLOPS: dp / cap / all per tile.\n")
        out.append("| N | " + " | ".join(WQ_TILES) + " |")
        out.append("|---|" + "---|" * len(WQ_TILES))
        for n in sorted(d[(WQ_TILES[0], "all")]):
            out.append(f"| {n} | " + " | ".join(
                f"{d[(t, 'dp')].get(n, 0):.0f} / {d[(t, 'cap')].get(n, 0):.0f} / {d[(t, 'all')].get(n, 0):.0f}"
                for t in WQ_TILES) + " |")
    text = "\n".join(out) + "\n"
    open(os.path.join(RES, "wq_report.md"), "w").write(text)
    print(text)


def mb():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(IMG, exist_ok=True)
    rows = load("mb.csv")
    d = collections.defaultdict(dict)
    for r in rows:
        bn = int(r["tile"].split("x")[1])
        d[(int(r["M"]), int(r["N"]), bn)][r["kind"]] = r
    out = ["# Memory-bound small-M grid: v27 split against v9_tile data-parallel\n",
           "v9_tile §14.7's grid: M = BM = 1, 4, 16, 32, N = 4096 / 8192 / 16384, BN = 16 .. 256, BK = 128, "
           "K = 8192, bf16, back to back, 2 GB of rotating inputs (weights from HBM). TB/s counts A, B and C once. "
           "`dp`: v9_tile with one wave per SIMD (waves_per_eu = 1, as v27); `cap`: v27 as deployed; `all`: v27 "
           "one-tile over all 1024 / W programs. Every small tile runs with v9_tile's default waves (one per 64 "
           "columns) except 32x64x128, which needs w2 for its LDS to fit 4 / W workgroups per CU.\n",
           "| M | N | BN | tiles | dp | cap (choice) | all (ending) | best / dp |",
           "|---|---|---|---|---|---|---|---|"]
    pts = []
    for (m, n, bn), v in sorted(d.items()):
        if not all(k in v for k in ("dp", "cap", "all")):
            continue
        dp, cap, al = (float(v[k]["tbps"]) for k in ("dp", "cap", "all"))
        tiles = n // bn
        choice = "v9_tile" if v["cap"]["kernel"] == "v9_tile" else v["cap"]["policy"] + " " + v["cap"]["fixup"]
        if v["cap"]["kernel"] != "v9_tile" and v["cap"]["policy"] == "dp":
            choice = "persistent dp"
        best = max(cap, al)
        pts.append((m, n, bn, tiles, dp, cap, al))
        out.append(f"| {m} | {n} | {bn} | {tiles} | {dp:.2f} | {cap:.2f} ({choice}) | {al:.2f} ({v['all']['fixup']}) | "
                   f"{best / dp:.2f} |")
    by_tiles = collections.defaultdict(list)
    for m, n, bn, tiles, dp, cap, al in pts:
        by_tiles[tiles].append((dp, cap, al))
    out.append("\nBy tile count, median TB/s (dp / cap / all) and the best of cap and all over dp:\n")
    out.append("| tiles | points | dp | cap | all | best / dp, median (range) |")
    out.append("|---|---|---|---|---|---|")
    for tiles in sorted(by_tiles):
        v = by_tiles[tiles]
        rat = [max(c, a) / dp for dp, c, a in v]
        out.append(f"| {tiles} | {len(v)} | {statistics.median(x[0] for x in v):.2f} | "
                   f"{statistics.median(x[1] for x in v):.2f} | {statistics.median(x[2] for x in v):.2f} | "
                   f"{statistics.median(rat):.2f} ({min(rat):.2f}-{max(rat):.2f}) |")
    fig, ax = plt.subplots(figsize=(8, 5))
    for kind, idx, mk in (("dp", 4, "o"), ("cap", 5, "s"), ("all", 6, "^")):
        ax.scatter([p[3] for p in pts], [p[idx] for p in pts], marker=mk, label=kind, alpha=0.7)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("tiles")
    ax.set_ylabel("TB/s (A, B, C once)")
    ax.axhline(6.0, color="gray", ls=":", label="HBM ~6.0-6.3")
    ax.set_title("M = 1-32, K = 8192: data-parallel against v27's splits")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "mb.png"), dpi=110)
    plt.close(fig)
    out.append("\n![memory-bound](../images/mb.png)\n")
    text = "\n".join(out) + "\n"
    open(os.path.join(RES, "mb_report.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    {"wq": wq, "mb": mb}[sys.argv[1]]()
