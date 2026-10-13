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
"""Wave quantization with smaller tiles against stream-K (WORKLOG section 13).

    python v9_tile/plot_wave_quant_tiles.py

Reads results/wq{K}_{timing_graph,clock,att}.csv and results/wqsk{K}_{timing_graph,clock}.csv
(from run_wave_quant_tiles.sh and sweep.py att) and writes, per K, to images/wave_quant_tiles/:

- throughput_k{K}.png: TFLOPS, duration and average busy CUs against N;
- power_k{K}.png: MFMA utilization of the busy CUs, graphics clock and socket power against N.

Then prints the WORKLOG's summary tables.
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator, PercentFormatter  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
IMG = os.path.join(HERE, "images", "wave_quant_tiles")

M = 4096
KS = (4096, 8192)
GPU = {4096: 6, 8192: 7}
NUM_CUS = 256
N_PER_WAVE = 4096  # one full wave of 256x256 tiles at M = 4096
POWER_CAP_W = 1400
SUMMARY_N = (1024, 2048, 4096, 4352, 6144)

TILES = ("256x256", "128x256", "128x128x128")
COLOR = {"256x256": "#2E6FBA", "128x256": "#E08E0B", "128x128x128": "#2CA58D", "sk256": "#D62728"}
LABEL = {"256x256": "256×256", "128x256": "128×256", "128x128x128": "128×128×128",
         "sk256": "256×256 Stream-K"}
C_BAND = "#EEF1F5"
C_GUIDE = "#8C8C8C"
C_TEXT = "#333333"


def rows(name):
    p = os.path.join(RES, name)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def series(data, tile, key):
    """(N, values) of one tile, sorted by N."""
    pts = sorted((int(r["N"]), float(r[key])) for r in data if r["tile"] == tile and r.get(key) not in (None, ""))
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def busy_frac(tile, n):
    bm, bn = (256, 256) if tile == "sk256" else (int(x) for x in tile.split("x")[:2])
    if tile == "sk256":
        return np.ones_like(n, dtype=float)
    tiles = (M // bm) * (n // bn)
    return tiles / (np.ceil(tiles / NUM_CUS) * NUM_CUS)


def style_axis(ax, n_max):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9A9A9A")
    ax.grid(axis="y", color="#E3E3E3", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=C_TEXT, labelsize=10)
    ax.set_xlim(0, n_max + 256)
    ax.xaxis.set_major_locator(MultipleLocator(2048))
    ax.xaxis.set_minor_locator(MultipleLocator(256))
    for w in range(int(np.ceil(n_max / N_PER_WAVE))):
        lo, hi = w * N_PER_WAVE + 128, (w + 1) * N_PER_WAVE + 128
        if w % 2 == 0:
            ax.axvspan(0 if w == 0 else lo, hi, color=C_BAND, zorder=0, linewidth=0)
        if w > 0:
            ax.axvline(lo, color=C_GUIDE, linestyle=(0, (4, 3)), linewidth=0.9, zorder=1)


def panel_title(ax, letter, title, hint):
    ax.set_title(f"({letter})  {title}", loc="left", fontsize=13, fontweight="bold", color=C_TEXT, pad=10)
    ax.set_title(hint, loc="right", fontsize=10, color=C_GUIDE, style="italic", pad=10)


def end_labels(ax, ends, min_gap):
    """Label each line at its right end, pushed apart vertically by at least min_gap."""
    ends = sorted(ends, key=lambda e: e[1])
    ys = []
    for _, y, _ in ends:
        ys.append(max(y, ys[-1] + min_gap) if ys else y)
    for (x, y, tile), y_lbl in zip(ends, ys):
        ax.annotate(LABEL[tile], xy=(x, y), xytext=(x + 330, y_lbl), fontsize=10, fontweight="bold",
                    color=COLOR[tile], va="center", annotation_clip=False)


def plot_tiles(ax, data, key, scale=1.0, min_gap=None):
    ends = []
    for tile in TILES:
        n, y = series(data, tile, key)
        if not len(n):
            continue
        y = y * scale
        ax.plot(n, y, color=COLOR[tile], linewidth=2.0, zorder=3)
        ax.scatter(n, y, s=12, color=COLOR[tile], zorder=4, linewidths=0)
        ends.append((n[-1], y[-1], tile))
    if min_gap is not None:
        end_labels(ax, ends, min_gap)


def plot_sk(ax, data, key, scale=1.0, label_n=False):
    n, y = series(data, "sk256", key)
    if not len(n):
        return
    y = y * scale
    ax.scatter(n, y, s=75, marker="D", color=COLOR["sk256"], edgecolor="white", linewidth=1.2, zorder=6)
    if label_n:
        # Neighbouring points are 256 apart in N: alternate the label height so they do not collide.
        for i, (ni, yi) in enumerate(zip(n, y)):
            s = (M // 256) * (ni // 256) % NUM_CUS
            ax.annotate(f"n={NUM_CUS // s}", xy=(ni, yi), xytext=(0, 9 if i % 2 == 0 else 21),
                        textcoords="offset points", ha="center", fontsize=8, color=COLOR["sk256"],
                        fontweight="bold", zorder=7)


def legend(fig):
    handles = [Line2D([], [], color=COLOR[t], lw=2.0, marker="o", ms=4, label=f"{LABEL[t]} data-parallel")
               for t in TILES]
    handles.append(Line2D([], [], color=COLOR["sk256"], lw=0, marker="D", ms=8, markeredgecolor="white",
                          label="256×256 Stream-K, leftover tiles split over all 256 CUs"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.065, 0.93), ncol=2, frameon=False,
               fontsize=10)


def finish(fig, ax, k, title, out):
    ax.set_xlabel(f"N  (M = 4096, K = {k})", fontsize=11, color=C_TEXT)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    fig.suptitle(title, x=0.07, y=0.985, ha="left", fontsize=17, fontweight="bold", color="#1A1A1A")
    fig.text(0.07, 0.962,
             f"BF16, back-to-back launches over cold rotating inputs, MI355X (GPU {GPU[k]}). Shaded bands: "
             "waves of 256×256 tiles (256 per wave);\n"
             "128×256 fills a wave every 2048 in N, 128×128×128 every 1024. "
             "Stream-K split n = 256 / leftover tiles.",
             ha="left", va="top", fontsize=10.5, color="#555555", linespacing=1.5)
    legend(fig)
    fig.subplots_adjust(left=0.07, right=0.86, top=0.855, bottom=0.05)
    os.makedirs(IMG, exist_ok=True)
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def new_fig(n_max):
    fig, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True, gridspec_kw={"hspace": 0.34})
    fig.patch.set_facecolor("white")
    for ax in axes:
        style_axis(ax, n_max)
    return fig, axes


def throughput(k, timing, sk_timing):
    n_max = max(int(r["N"]) for r in timing)
    fig, (ax_a, ax_b, ax_c) = new_fig(n_max)

    panel_title(ax_a, "a", "Throughput", "TFLOPS, higher is better")
    plot_tiles(ax_a, timing, "tflops", min_gap=70)
    plot_sk(ax_a, sk_timing, "tflops", label_n=True)
    ax_a.set_ylabel("TFLOPS", fontsize=11, color=C_TEXT)
    ax_a.set_ylim(0, 1800)

    panel_title(ax_b, "b", "Kernel duration", "µs, lower is better")
    plot_tiles(ax_b, timing, "us_median")
    plot_sk(ax_b, sk_timing, "us_median")
    ax_b.set_ylabel("Duration (µs)", fontsize=11, color=C_TEXT)
    ax_b.set_ylim(0, None)
    _, top = ax_b.get_ylim()
    ends = [(series(timing, t, "us_median")[0][-1], series(timing, t, "us_median")[1][-1], t) for t in TILES]
    end_labels(ax_b, ends, 0.05 * top)

    panel_title(ax_c, "c", "Average CUs busy", "tiles / (waves × 256); Stream-K keeps all 256 busy")
    for tile in TILES:
        n, _ = series(timing, tile, "tflops")
        ax_c.step(n, busy_frac(tile, n), where="mid", color=COLOR[tile], linewidth=2.0, zorder=3)
    n, _ = series(sk_timing, "sk256", "tflops")
    ax_c.scatter(n, np.ones(len(n)), s=75, marker="D", color=COLOR["sk256"], edgecolor="white",
                 linewidth=1.2, zorder=6, clip_on=False)
    ax_c.set_ylim(0, 1.05)
    ax_c.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax_c.set_ylabel("CUs busy", fontsize=11, color=C_TEXT)

    finish(fig, ax_c, k, f"Wave quantization: smaller tiles against Stream-K, K = {k}",
           os.path.join(IMG, f"throughput_k{k}.png"))


def power(k, clock, sk_clock):
    n_max = max(int(r["N"]) for r in clock)
    fig, (ax_a, ax_b, ax_c) = new_fig(n_max)

    panel_title(ax_a, "a", "MFMA utilization of the busy CUs",
                "TFLOPS / (busy CUs × 4096 FLOP/clk × clock)")
    plot_tiles(ax_a, clock, "mfma_util_busy", scale=100, min_gap=4.5)
    plot_sk(ax_a, sk_clock, "mfma_util_busy", scale=100)
    ax_a.set_ylabel("MFMA utilization (%)", fontsize=11, color=C_TEXT)
    ax_a.set_ylim(0, 100)

    panel_title(ax_b, "b", "Graphics clock", "GHz, sustained under back-to-back launches")
    plot_tiles(ax_b, clock, "sclk_ghz", min_gap=0.05)
    plot_sk(ax_b, sk_clock, "sclk_ghz")
    ax_b.axhline(2.4, color=C_GUIDE, linestyle=":", linewidth=1.1, zorder=2)
    ax_b.text(150, 2.42, "2.4 GHz limit", fontsize=9, color=C_GUIDE, va="bottom")
    ax_b.set_ylabel("Clock (GHz)", fontsize=11, color=C_TEXT)
    ax_b.set_ylim(1.4, 2.6)

    panel_title(ax_c, "c", "Socket power", "W")
    plot_tiles(ax_c, clock, "power_w")
    plot_sk(ax_c, sk_clock, "power_w")
    ax_c.axhline(POWER_CAP_W, color=C_GUIDE, linestyle=":", linewidth=1.1, zorder=2)
    ax_c.text(150, POWER_CAP_W + 15, f"{POWER_CAP_W} W cap", fontsize=9, color=C_GUIDE, va="bottom")
    ax_c.set_ylabel("Power (W)", fontsize=11, color=C_TEXT)
    ax_c.set_ylim(0, 1550)

    finish(fig, ax_c, k, f"Why: MFMA utilization, clock and power, K = {k}",
           os.path.join(IMG, f"power_k{k}.png"))


def summary(k, timing, clock, att, sk_timing, sk_clock):
    by = {}
    for src, name in ((timing, "t"), (clock, "c"), (att, "a"), (sk_timing, "t"), (sk_clock, "c")):
        for r in src:
            by.setdefault((r["tile"], int(r["N"])), {})[name] = r
    print(f"\n#### K = {k}\n")
    print("| N | kernel | tiles | CUs busy | TFLOPS | us | MFMA util, chip | MFMA util, busy CUs | "
          "ATT loop MFMA eff | clock GHz | power W |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for n in SUMMARY_N:
        for tile in TILES + ("sk256",):
            d = by.get((tile, n))
            if not d or "t" not in d:
                continue
            t, c, a = d["t"], d.get("c", {}), d.get("a", {})
            pct = lambda r, key: f"{100 * float(r[key]):.0f}%" if r.get(key) else ""  # noqa: E731
            name = LABEL[tile] if tile == "sk256" else f"{LABEL[tile]} DP"
            if tile == "sk256":
                name += f" (n={NUM_CUS // (int(t['tiles']) % NUM_CUS)})"
            print(f"| {n} | {name} | {t['tiles']} | {100 * float(busy_frac(tile, np.array([n]))[0]):.0f}% | "
                  f"{float(t['tflops']):.0f} | {float(t['us_median']):.1f} | {pct(c, 'mfma_util')} | "
                  f"{pct(c, 'mfma_util_busy')} | {pct(a, 'att_mfma_eff')} | {c.get('sclk_ghz', '')} | "
                  f"{c.get('power_w', '')} |")


def check(k, timing, clock):
    """Flag points whose clock run disagrees with the timed run: another tenant on the GPU slows one
    of them, and the clock and power it reports are then not the kernel's."""
    timed = {(r["tile"], r["N"]): float(r["tflops"]) for r in timing}
    bad = [(r["tile"], r["N"], timed[(r["tile"], r["N"])], float(r["sustained_tflops"])) for r in clock
           if (r["tile"], r["N"]) in timed and abs(float(r["sustained_tflops"]) / timed[(r["tile"], r["N"])] - 1) > 0.07]
    wrong = [(r["tile"], r["N"]) for r in timing if r.get("correct") not in (None, "", "True")]
    print(f"K = {k}: {len(clock)} clock points, {len(bad)} disagree with timing by > 7%, "
          f"{len(wrong)} incorrect")
    for b in bad:
        print(f"  {b[0]} N={b[1]}: timed {b[2]:.0f}, sustained {b[3]:.0f} TFLOPS")


def main():
    for k in KS:
        timing, clock = rows(f"wq{k}_timing_graph.csv"), rows(f"wq{k}_clock.csv")
        sk_timing, sk_clock = rows(f"wqsk{k}_timing_graph.csv"), rows(f"wqsk{k}_clock.csv")
        att = rows(f"wq{k}_att.csv")
        check(k, timing + sk_timing, clock + sk_clock)
        if timing:
            throughput(k, timing, sk_timing)
        if clock:
            power(k, clock, sk_clock)
        if timing:
            summary(k, timing, clock, att, sk_timing, sk_clock)


if __name__ == "__main__":
    main()
