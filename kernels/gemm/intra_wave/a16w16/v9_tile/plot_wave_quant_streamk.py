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
"""Wave quantization at 256x256x64 against v31 Stream-K, and a tile that fits the waves better.

    source v31_streamk_aiter/env.sh   # v31's streamk_plan tells which N it splits
    python v9_tile/plot_wave_quant_streamk.py [--k 4096] [--small 128x256]

Same layout as v9_beyond_hotloop/images/wave_quant.png, from one session on GPU 6
(run_wave_quant_v31.sh): results/wq31dp{K}_{timing_graph,clock}.csv (v9_tile data-parallel),
results/wq31_{K}_timing_graph.csv (v31_streamk_aiter: "v31" default plan, "v31spread") and
results/wq31t_{K}_timing_graph.csv (v31's default plan at the smaller tile, sweep.py WQ31T_{K}).
Writes images/wave_quant_tiles/streamk_k{K}_{small}.{png,svg}.

The dashed line is the least-squares line through the full-wave points (N a multiple of
4096): time = fixed + per_wave x tiles / 256, what a partly full last wave would cost if
it cost exactly its share of a full one.
"""

import argparse
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
IMG = os.path.join(HERE, "images", "wave_quant_tiles")

M = 4096
NUM_CUS = 256
N_PER_WAVE = NUM_CUS // (M // 256) * 256
HALF = N_PER_WAVE // 2 + 128  # between S = 128 and S = 144 leftover tiles
LABEL = {"128x256": "128×256"}  # tiles with a v31t{tile} run in results/wq31t_{K}_timing_graph.csv

C_TFLOPS = "#2E6FBA"
C_TIME = "#6A3D9A"
C_SMALL = "#2CA58D"
C_SK = "#D62728"
C_CLIFF = "#D62728"
C_BAND = "#EEF1F5"
C_GUIDE = "#8C8C8C"
C_TEXT = "#333333"


def load(name, tile):
    rows = sorted((r for r in csv.DictReader(open(os.path.join(RES, name))) if r["tile"] == tile),
                  key=lambda r: int(r["N"]))  # fmt: skip
    col = lambda k, t=float: np.array([t(r[k]) for r in rows])  # noqa: E731
    d = {"N": col("N", int), "tiles": col("tiles", int)}
    for k in ("us_median", "tflops", "sclk_ghz", "power_w"):
        if rows and k in rows[0]:
            d[k] = col(k)
    return d


def style_axis(ax, n_max):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9A9A9A")
    ax.grid(axis="y", color="#E3E3E3", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=C_TEXT, labelsize=10)
    ax.set_xlim(0, n_max + 256)
    ax.xaxis.set_major_locator(MultipleLocator(1024))
    ax.xaxis.set_minor_locator(MultipleLocator(256))
    for w in range(int(np.ceil(n_max / N_PER_WAVE))):
        lo, hi = w * N_PER_WAVE + 128, (w + 1) * N_PER_WAVE + 128
        if w % 2 == 0:
            ax.axvspan(0 if w == 0 else lo, hi, color=C_BAND, zorder=0, linewidth=0)
        if w > 0:
            ax.axvline(lo, color=C_GUIDE, linestyle=(0, (4, 3)), linewidth=0.9, zorder=1)
        ax.axvline(w * N_PER_WAVE + HALF, color="#B8B8B8", linestyle=":", linewidth=1.0, zorder=1)


def panel_title(ax, letter, title, hint):
    ax.set_title(f"({letter})  {title}", loc="left", fontsize=13, fontweight="bold", color=C_TEXT, pad=10)
    ax.set_title(hint, loc="right", fontsize=10, color=C_GUIDE, style="italic", pad=10)


def plot_series(ax, d, y, color, full, cliff, lw=2.0, alpha=1.0):
    ax.plot(d["N"], y, color=color, linewidth=lw, zorder=3, alpha=alpha)
    ax.scatter(d["N"], y, s=16, color=color, zorder=4, linewidths=0, alpha=alpha)
    if full is not None:
        ax.scatter(d["N"][full], y[full], s=70, color=color, edgecolor="white", linewidth=1.5, zorder=5)
        ax.scatter(d["N"][cliff], y[cliff], s=70, facecolor="white", edgecolor=C_CLIFF, linewidth=2.0, zorder=5)


def plot_sk(ax, n, y, alpha=1.0):
    ax.scatter(n, y, s=55, marker="D", color=C_SK, edgecolor="white", linewidth=1.0, zorder=6, alpha=alpha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--small", default="128x256", choices=sorted(LABEL))
    args = ap.parse_args()
    k, small = args.k, args.small

    sys.path.insert(0, os.path.dirname(HERE))
    from v31_streamk_aiter.matmul_kernel import streamk_plan

    d = load(f"wq31dp{k}_timing_graph.csv", "256x256")
    clk = load(f"wq31dp{k}_clock.csv", "256x256")
    sm = load(f"wq31dp{k}_timing_graph.csv", small)
    sm_v31 = load(f"wq31t_{k}_timing_graph.csv", f"v31t{small}")
    v31 = load(f"wq31_{k}_timing_graph.csv", "v31")
    spread = load(f"wq31_{k}_timing_graph.csv", "v31spread")
    N, tiles, us, tf = d["N"], d["tiles"], d["us_median"], d["tflops"]
    assert np.array_equal(v31["N"], N) and np.array_equal(sm["N"], N) and np.array_equal(sm_v31["N"], N)
    waves = -(-tiles // NUM_CUS)
    full = tiles % NUM_CUS == 0
    cliff = np.r_[False, waves[1:] > waves[:-1]]
    n_max = int(N.max())
    i_cliff = int(np.flatnonzero(cliff)[0])
    at = {int(n): i for i, n in enumerate(N)}

    per_wave, fixed = np.polyfit(tiles[full] / NUM_CUS, us[full], 1)
    ideal = fixed + per_wave * tiles / NUM_CUS
    kstep = per_wave / (k // 64)
    sk = np.array([streamk_plan(M, int(n), k, (256, 256, 64), NUM_CUS)[0] != "dp" for n in N])
    got = np.where(sk, np.minimum(us, v31["us_median"]), us)
    sp_i = np.array([at[int(n)] for n in spread["N"]])
    left = tiles % NUM_CUS

    cost = (v31["us_median"][sk] - ideal[sk]) / kstep
    won = (us[sk] - v31["us_median"][sk]) / us[sk]
    print(f"K = {k}: dashed = {fixed:.1f} + {per_wave:.1f} x tiles / 256 us; full-chip k-step {kstep:.2f} us")
    print(f"  v31 Stream-K on {sk.sum()} N: cost over the dashed line {cost.min():.1f} to {cost.max():.1f} k-steps "
          f"(median {np.median(cost):.1f}); time saved {100 * won.min():.0f}-{100 * won.max():.0f}%")
    for s in (16, 32, 64, 128):
        m = sk & (left == s)
        print(f"  S = {s}: time saved {100 * np.median((us[m] - v31['us_median'][m]) / us[m]):.1f}% (median over waves)")
    sp_gain = us[sp_i] / spread["us_median"] - 1
    print(f"  spread past half a wave: {100 * sp_gain.min():+.1f}% to {100 * sp_gain.max():+.1f}% against data-parallel")
    full_clk = clk["sclk_ghz"][full]
    print(f"  clock: full waves {full_clk.min():.2f}-{full_clk.max():.2f} GHz, one step past "
          f"{clk['sclk_ghz'][cliff].min():.2f}-{clk['sclk_ghz'][cliff].max():.2f}")

    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        3, 1, figsize=(11, 12.5), dpi=200, sharex=True, gridspec_kw={"hspace": 0.38}
    )
    fig.patch.set_facecolor("white")
    for ax in (ax_a, ax_b, ax_c):
        style_axis(ax, n_max)

    # (a) TFLOPS
    ax = ax_a
    panel_title(ax, "a", "Throughput", "TFLOPS, higher is better")
    plot_series(ax, d, tf, C_TFLOPS, full, cliff)
    plot_sk(ax, N[sk], v31["tflops"][sk])
    n_split = streamk_plan(M, int(N[i_cliff]), k, (256, 256, 64), NUM_CUS)[2]
    ax.annotate(
        f"Stream-K: the {tiles[i_cliff] % NUM_CUS} tiles\nsplit over {n_split} CUs each",
        xy=(N[i_cliff], v31["tflops"][i_cliff]), xytext=(N[i_cliff] + 2000, 850),
        fontsize=9, color=C_SK, va="center",
        arrowprops=dict(arrowstyle="-|>", color=C_SK, lw=1.1, shrinkA=2, shrinkB=5),
    )  # fmt: skip
    ax.set_ylabel("TFLOPS", fontsize=11, color=C_TEXT)
    ax.set_ylim(0, 1850)
    for w in range(int(waves.max())):
        label = "1 wave" if w == 0 else f"{w + 1} waves"
        ax.text(w * N_PER_WAVE + 300, 1800, label, ha="left", va="top",
                fontsize=11, fontweight="bold", color="#5A5A5A")  # fmt: skip
        ax.text(w * N_PER_WAVE + HALF + 80, 1800, "last wave\nhalf full", ha="left", va="top",
                fontsize=8, color=C_GUIDE, style="italic", linespacing=1.1)  # fmt: skip
    for n_ in N[cliff]:
        i = at[int(n_)]
        drop = 100 * (1 - tf[i] / tf[i - 1])
        ax.text(n_ + 120, tf[i] - 60, f"−{drop:.0f}%", fontsize=11, fontweight="bold", color=C_CLIFF,
                ha="left", va="top")  # fmt: skip
    ax.annotate(
        f"+16 tiles opens a new wave\nwith only 16 of {NUM_CUS} CUs busy",
        xy=(N[i_cliff], tf[i_cliff]), xytext=(N[i_cliff] + 450, 560),
        fontsize=10, color=C_CLIFF, va="center",
        arrowprops=dict(arrowstyle="-|>", color=C_CLIFF, lw=1.3, shrinkA=2, shrinkB=6),
    )  # fmt: skip
    ax.text(N[cliff][-1] - 100, 1000, "the fewer waves,\nthe bigger the drop", fontsize=10, color=C_CLIFF,
            ha="center", va="top", style="italic")  # fmt: skip
    ax.annotate(
        f"{tiles[0]} tiles: {100 * tiles[0] / NUM_CUS:.0f}% of CUs busy",
        xy=(N[0], tf[0]), xytext=(N[0] + 350, 60),
        fontsize=10, color=C_TEXT, va="center",
        arrowprops=dict(arrowstyle="-|>", color=C_GUIDE, lw=1.1, shrinkA=2, shrinkB=6),
    )  # fmt: skip
    ax.legend(
        handles=[
            Line2D([], [], color=C_TFLOPS, lw=2.0, marker="o", ms=4, label="256×256×64 data-parallel"),
            Line2D([], [], color=C_SK, lw=0, marker="D", ms=6, markeredgecolor="white",
                   label="Stream-K (used while the last wave is at most half full)"),
        ],
        loc="lower right", bbox_to_anchor=(1.0, 0.02), frameon=True, facecolor="white", edgecolor="#DDDDDD",
        fontsize=9,
    )  # fmt: skip

    # (b) Duration: the idle-CU time, and the part of it Stream-K wins back
    ax = ax_b
    panel_title(ax, "b", "Kernel duration", "µs, lower is better")
    ax.fill_between(N, ideal, got, where=got >= ideal, interpolate=True,
                    color=C_CLIFF, alpha=0.10, linewidth=0, zorder=2)  # fmt: skip
    ax.fill_between(N, got, us, where=sk, interpolate=False, step=None,
                    color=C_CLIFF, alpha=0.45, linewidth=0, zorder=2)  # fmt: skip
    ax.plot(N, ideal, color=C_GUIDE, linestyle=(0, (5, 3)), linewidth=1.4, zorder=2)
    ax.step(N, fixed + per_wave * waves, where="mid", color=C_GUIDE, linestyle=":", linewidth=1.3, zorder=2)
    plot_series(ax, d, us, C_TIME, full, cliff)
    plot_sk(ax, N[sk], v31["us_median"][sk])
    ax.set_ylabel("Duration (µs)", fontsize=11, color=C_TEXT)
    ax.set_ylim(0, 480)

    j = at[11776]
    ax.annotate("dashed = straight line through the full-wave times", xy=(N[j], ideal[j]),
                xytext=(N[j] - 1300, 100), fontsize=10, color=C_GUIDE, style="italic", va="center",
                ha="center", arrowprops=dict(arrowstyle="-|>", color=C_GUIDE, lw=1.1, shrinkA=2,
                                             shrinkB=3))  # fmt: skip
    j = at[14592]
    ax.annotate("dotted = naive guess: one extra\ntile costs a whole wave", xy=(N[j], fixed + per_wave * waves[j]),
                xytext=(N[j] - 300, 455), fontsize=10, color=C_GUIDE, style="italic", va="center",
                ha="center", arrowprops=dict(arrowstyle="-|>", color=C_GUIDE, lw=1.1, shrinkA=2,
                                             shrinkB=3))  # fmt: skip
    ax.text(
        200, 470,
        "Shaded = time lost because the last wave is only partly full\n"
        "(the real time minus the dashed line). It is less than the naive\n"
        "guess: a partly full wave runs faster than a full one.\n"
        "Dark = what Stream-K wins back.  Light = what it cannot:\n"
        "splitting a tile has a fixed cost (store partial sums, wait, add\n"
        "them up). As the last wave fills, the lost time shrinks toward\n"
        "that cost, so Stream-K is shown only up to half a wave.",
        fontsize=9.3, color=C_TEXT, va="top", ha="left", linespacing=1.35, zorder=7,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#CCCCCC"),
    )  # fmt: skip
    ax.legend(
        handles=[Patch(color=C_CLIFF, alpha=0.45, label="won back by Stream-K"),
                 Patch(color=C_CLIFF, alpha=0.10, label="not recoverable")],
        loc="lower right", bbox_to_anchor=(1.0, 0.02), frameon=True, facecolor="white", edgecolor="#DDDDDD",
        fontsize=9,
    )  # fmt: skip

    # (c) Start from a tile that fits, then Stream-K: each tile at its best of data-parallel and v31
    ax = ax_c
    panel_title(ax, "c", f"Half a wave left: split it, or use a tile that fills the wave?",
                "TFLOPS, best of data-parallel and Stream-K")
    best_big = np.maximum(tf, v31["tflops"])
    best_sm = np.maximum(sm["tflops"], sm_v31["tflops"])
    bm, bn = (int(x) for x in small.split("x")[:2])
    fits = (sm["tiles"] % NUM_CUS == 0) & ~full
    ax.plot(N, best_big, color=C_TFLOPS, linewidth=2.0, zorder=3)
    ax.scatter(N, best_big, s=12, color=C_TFLOPS, zorder=4, linewidths=0)
    ax.plot(N, best_sm, color=C_SMALL, linewidth=2.0, zorder=3)
    ax.scatter(N, best_sm, s=12, color=C_SMALL, zorder=4, linewidths=0)
    ax.scatter(N[fits], best_sm[fits], s=90, facecolor="white", edgecolor=C_SMALL, linewidth=2.0, zorder=5)
    ax.set_ylabel("TFLOPS", fontsize=11, color=C_TEXT)
    ax.set_ylim(0, 1850)
    n0, n1 = (int(n) for n in N[fits][:2])
    n_half = streamk_plan(M, n0, k, (256, 256, 64), NUM_CUS)[2]
    ax.annotate(
        f"N = {n0}: {LABEL[small]} fills the wave: {best_sm[at[n0]]:.0f} TFLOPS\n"
        f"against {best_big[at[n0]]:.0f} for 256×256 split n = {n_half}. The smaller tile wins",
        xy=(n0, best_sm[at[n0]]), xytext=(250, 1680),
        fontsize=10, color=C_SMALL, va="center",
        arrowprops=dict(arrowstyle="-|>", color=C_SMALL, lw=1.2, shrinkA=2, shrinkB=6),
    )  # fmt: skip
    ax.annotate(
        f"N = {n1}: {LABEL[small]} fills {sm['tiles'][at[n1]] // NUM_CUS} waves, but 256×256\n"
        f"split n = {streamk_plan(M, n1, k, (256, 256, 64), NUM_CUS)[2]} wins: "
        f"{best_big[at[n1]]:.0f} against {best_sm[at[n1]]:.0f}",
        xy=(n1, best_sm[at[n1]]), xytext=(n1 + 4000, 960),
        fontsize=10, color=C_SMALL, va="center",
        arrowprops=dict(arrowstyle="-|>", color=C_SMALL, lw=1.2, shrinkA=2, shrinkB=6),
    )  # fmt: skip
    ax.text(
        2700, 900,
        "With half a wave of 256×256 tiles left, there are two options:\n"
        f"split each leftover tile over n = {n_half} CUs (Stream-K), or use a smaller\n"
        "tile that fills the wave. The smaller tile has lower MFMA utilization,\n"
        "so neither wins by default: here the smaller tile wins at one wave\n"
        "and Stream-K wins after that. hipBLASLt makes this trade-off: it\n"
        "ranks tiles, mostly non-power-of-two, with the Stream-K split in\n"
        "the cost model, so a tile that fills the GPU usually wins.",
        fontsize=9.3, color=C_TEXT, va="top", ha="left", linespacing=1.35, zorder=7,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#CCCCCC"),
    )  # fmt: skip
    ax.legend(
        handles=[
            Line2D([], [], color=C_TFLOPS, lw=2.0, marker="o", ms=4, label="256×256×64"),
            Line2D([], [], color=C_SMALL, lw=2.0, marker="o", ms=4, label=f"{LABEL[small]}×64"),
            Line2D([], [], color=C_SMALL, lw=0, marker="o", ms=8, markerfacecolor="white", markeredgewidth=2,
                   label=f"{LABEL[small]} fills its waves exactly"),
        ],
        loc="lower right", bbox_to_anchor=(1.0, 0.02), frameon=True, facecolor="white", edgecolor="#DDDDDD",
        fontsize=9,
    )  # fmt: skip

    ax.set_xlabel(f"N  (M = {M}, K = {k})", fontsize=11, color=C_TEXT)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    fig.canvas.draw()
    for lbl in ax.get_xticklabels():
        if lbl.get_text() and int(lbl.get_text()) % N_PER_WAVE == 0 and int(lbl.get_text()) > 0:
            lbl.set_fontweight("bold")

    fig.suptitle("Wave quantization and Stream-K on MI355X",
                 x=0.07, y=0.975, ha="left", fontsize=17, fontweight="bold", color="#1A1A1A")  # fmt: skip
    fig.text(0.07, 0.953,
             f"BF16 GEMM, M = {M}, K = {k}, 256×256×64 tiles. {NUM_CUS} CUs × 1 workgroup each → "
             f"{NUM_CUS} tiles run at once. N steps by 256 (+16 tiles).\n"
             "Filled dots = exactly full waves.  Hollow red = one step past a full wave.  "
             "Dotted lines = last wave half full.",
             ha="left", va="top", fontsize=10.5, color="#555555", linespacing=1.5)  # fmt: skip
    fig.subplots_adjust(left=0.07, right=0.97, top=0.89, bottom=0.05)

    os.makedirs(IMG, exist_ok=True)
    for ext in ("png", "svg"):
        out = os.path.join(IMG, f"streamk_k{k}_{small}.{ext}")
        fig.savefig(out, facecolor="white")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
