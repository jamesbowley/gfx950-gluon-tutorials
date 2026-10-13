#!/usr/bin/env python3
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

"""
Regenerate kernels/gemm/intra_wave/a16w16/v9_beyond_hotloop/images/wave_quant.{png,svg}
— the wave-quantization figure for v9_beyond_hotloop (after NVIDIA's "Matrix
Multiplication Background" Fig. 8): TFLOPS, duration and tile count vs N at
M = K = 4096.

Data comes from wave_quant.csv, produced by
kernels/gemm/intra_wave/a16w16/wave_quant_sweep.py.

    python scripts/plot_wave_quant.py
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMG_DIR = os.path.join(REPO, "kernels", "gemm", "intra_wave", "a16w16", "v9_beyond_hotloop", "images")
CSV_PATH = os.path.join(IMG_DIR, "wave_quant.csv")

M = K = 4096
NUM_CUS = 256
N_PER_WAVE = NUM_CUS // (M // 256) * 256  # N at which the tile count hits one full wave

C_TFLOPS = "#2E6FBA"
C_TIME = "#6A3D9A"
C_TILES = "#E08E0B"
C_CLIFF = "#D62728"
C_BAND = "#EEF1F5"
C_GUIDE = "#8C8C8C"
C_TEXT = "#333333"


def load():
    rows = list(csv.DictReader(open(CSV_PATH)))
    col = lambda k, t=float: np.array([t(r[k]) for r in rows])  # noqa: E731
    return {
        "N": col("N", int),
        "tiles": col("tiles", int),
        "waves": col("waves", int),
        "us": col("us_median"),
        "us_lo": col("us_p20"),
        "us_hi": col("us_p80"),
        "tflops": col("tflops"),
    }


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
    n_waves = int(np.ceil(n_max / N_PER_WAVE))
    for w in range(n_waves):
        lo, hi = w * N_PER_WAVE + 128, (w + 1) * N_PER_WAVE + 128
        if w % 2 == 0:
            ax.axvspan(0 if w == 0 else lo, hi, color=C_BAND, zorder=0, linewidth=0)
        if w > 0:
            ax.axvline(lo, color=C_GUIDE, linestyle=(0, (4, 3)), linewidth=0.9, zorder=1)


def panel_title(ax, letter, title, hint):
    ax.set_title(f"({letter})  {title}", loc="left", fontsize=13, fontweight="bold", color=C_TEXT, pad=10)
    ax.set_title(hint, loc="right", fontsize=10, color=C_GUIDE, style="italic", pad=10)


def plot_series(ax, d, y, color, full, cliff):
    ax.plot(d["N"], y, color=color, linewidth=2.0, zorder=3)
    ax.scatter(d["N"], y, s=16, color=color, zorder=4, linewidths=0)
    ax.scatter(d["N"][full], y[full], s=70, color=color, edgecolor="white", linewidth=1.5, zorder=5)
    ax.scatter(d["N"][cliff], y[cliff], s=70, facecolor="white", edgecolor=C_CLIFF, linewidth=2.0, zorder=5)


def main():
    d = load()
    N, tiles = d["N"], d["tiles"]
    full = tiles % NUM_CUS == 0
    cliff = np.r_[False, d["waves"][1:] > d["waves"][:-1]]
    n_max = int(N.max())
    i_full = int(np.flatnonzero(full)[0])
    i_cliff = int(np.flatnonzero(cliff)[0])

    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        3, 1, figsize=(11, 12), dpi=200, sharex=True, gridspec_kw={"hspace": 0.38}
    )
    fig.patch.set_facecolor("white")
    for ax in (ax_a, ax_b, ax_c):
        style_axis(ax, n_max)

    # (a) TFLOPS
    ax = ax_a
    panel_title(ax, "a", "Throughput", "TFLOPS, higher is better")
    plot_series(ax, d, d["tflops"], C_TFLOPS, full, cliff)
    ax.set_ylabel("TFLOPS", fontsize=11, color=C_TEXT)
    ax.set_ylim(0, 1700)
    for w in range(int(d["waves"].max())):
        label = "1 wave" if w == 0 else f"{w + 1} waves"
        ax.text((w + 0.5) * N_PER_WAVE + 128, 1640, label, ha="center", va="top",
                fontsize=11, fontweight="bold", color="#5A5A5A")  # fmt: skip

    drop = 100 * (1 - d["tflops"][i_cliff] / d["tflops"][i_full])
    ax.annotate(
        f"N = {N[i_full]} → {N[i_cliff]}: just +16 tiles\n"
        f"opens a 2nd wave with only 16 of {NUM_CUS} CUs busy\n"
        f"→ throughput drops {drop:.0f}%",
        xy=(N[i_cliff], d["tflops"][i_cliff]), xytext=(N[i_cliff] + 450, 330),
        fontsize=10, color=C_CLIFF, va="center",
        arrowprops=dict(arrowstyle="-|>", color=C_CLIFF, lw=1.3, shrinkA=2, shrinkB=6),
    )  # fmt: skip
    ax.annotate(
        f"{tiles[0]} tiles: only {100 * tiles[0] / NUM_CUS:.1f}% of CUs busy",
        xy=(N[0], d["tflops"][0]), xytext=(N[0] + 300, 110),
        fontsize=10, color=C_TEXT, va="center",
        arrowprops=dict(arrowstyle="-|>", color=C_GUIDE, lw=1.1, shrinkA=2, shrinkB=6),
    )  # fmt: skip

    # (b) Duration
    ax = ax_b
    panel_title(ax, "b", "Kernel duration", "µs, lower is better")
    t_wave = d["us"][i_full]
    ideal = tiles / NUM_CUS * t_wave
    ax.fill_between(N, ideal, d["us"], where=d["us"] >= ideal, interpolate=True,
                    color=C_CLIFF, alpha=0.10, linewidth=0, zorder=2)  # fmt: skip
    ax.plot(N, ideal, color=C_GUIDE, linestyle=(0, (5, 3)), linewidth=1.4, zorder=2)
    ax.fill_between(N, d["us_lo"], d["us_hi"], color=C_TIME, alpha=0.25, linewidth=0, zorder=3)
    plot_series(ax, d, d["us"], C_TIME, full, cliff)
    ax.set_ylabel("Duration (µs)", fontsize=11, color=C_TEXT)
    ax.set_ylim(0, 440)

    j = int(np.searchsorted(N, 9216))
    ax.annotate(
        "dashed = time if it scaled with tile count",
        xy=(N[j], ideal[j]), xytext=(N[j] - 200, 150),
        fontsize=10, color=C_GUIDE, style="italic", va="center",
        arrowprops=dict(arrowstyle="-|>", color=C_GUIDE, lw=1.1, shrinkA=2, shrinkB=3),
    )  # fmt: skip
    j = int(np.searchsorted(N, 4608))
    ax.annotate(
        "shaded = idle-CU time\n(last wave only partly full)",
        xy=(N[j], (ideal[j] + d["us"][j]) / 2), xytext=(N[j] + 700, 45),
        fontsize=10, color=C_CLIFF, va="center",
        arrowprops=dict(arrowstyle="-|>", color=C_CLIFF, lw=1.1, shrinkA=2, shrinkB=3),
    )  # fmt: skip
    jump = d["us"][i_cliff] - d["us"][i_full]
    ax.annotate(
        f"+{jump:.0f} µs for +16 tiles",
        xy=(N[i_cliff], d["us"][i_cliff]), xytext=(N[i_cliff] - 3000, 245),
        fontsize=10, color=C_CLIFF, va="center",
        arrowprops=dict(arrowstyle="-|>", color=C_CLIFF, lw=1.3, shrinkA=2, shrinkB=6,
                        connectionstyle="arc3,rad=-0.2"),
    )  # fmt: skip

    # (c) Tiles
    ax = ax_c
    panel_title(ax, "c", "Number of 256×256 output tiles", f"1 wave = {NUM_CUS} tiles (1 per CU)")
    colors = np.where(cliff, C_CLIFF, C_TILES)
    ax.bar(N, tiles, width=190, color=colors, zorder=3, linewidth=0)
    ax.set_ylabel("Tiles", fontsize=11, color=C_TEXT)
    ax.set_ylim(0, 1100)
    ax.yaxis.set_major_locator(MultipleLocator(NUM_CUS))
    for w in range(1, int(d["waves"].max()) + 1):
        ax.axhline(w * NUM_CUS, color=C_GUIDE, linestyle=":", linewidth=1.1, zorder=2)
    ax_r = ax.twinx()
    ax_r.set_ylim(ax.get_ylim())
    ax_r.set_yticks([w * NUM_CUS for w in range(1, int(d["waves"].max()) + 1)])
    ax_r.set_yticklabels([f"{w} wave{'s' if w > 1 else ''}" for w in range(1, int(d["waves"].max()) + 1)])
    ax_r.tick_params(colors="#5A5A5A", labelsize=10, length=0)
    for s in ax_r.spines.values():
        s.set_visible(False)
    ax.annotate(
        "red = first N that spills into a new wave",
        xy=(N[i_cliff], tiles[i_cliff]), xytext=(N[i_cliff] - 3900, 470),
        fontsize=10, color=C_CLIFF, va="center",
        arrowprops=dict(arrowstyle="-|>", color=C_CLIFF, lw=1.1, shrinkA=2, shrinkB=4),
    )  # fmt: skip

    ax.set_xlabel("N  (M = K = 4096)", fontsize=11, color=C_TEXT)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(0)
    fig.canvas.draw()
    for lbl in ax.get_xticklabels():
        if lbl.get_text() and int(lbl.get_text()) % N_PER_WAVE == 0 and int(lbl.get_text()) > 0:
            lbl.set_fontweight("bold")

    fig.suptitle("Wave quantization in v9_beyond_hotloop on MI355X",
                 x=0.07, y=0.975, ha="left", fontsize=17, fontweight="bold", color="#1A1A1A")  # fmt: skip
    fig.text(0.07, 0.953,
             f"FP16 GEMM, M = K = 4096, 256×256 tiles. {NUM_CUS} CUs × 1 workgroup each → "
             f"{NUM_CUS} tiles run at once. N steps by 256 (+16 tiles).\n"
             "Filled dots = exactly full waves.  Hollow red = one step past a full wave.",
             ha="left", va="top", fontsize=10.5, color="#555555", linespacing=1.5)  # fmt: skip
    fig.subplots_adjust(left=0.07, right=0.93, top=0.885, bottom=0.05)

    for ext in ("png", "svg"):
        out = os.path.join(IMG_DIR, f"wave_quant.{ext}")
        fig.savefig(out, facecolor="white")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
