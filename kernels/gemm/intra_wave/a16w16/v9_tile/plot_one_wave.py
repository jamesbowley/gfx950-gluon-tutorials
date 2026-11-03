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
"""One wave of 256x256x64: the clock against busy CUs, the memory traffic's share, and the ideal Stream-K line.

    python v9_tile/plot_one_wave.py [--k 4096]

Data from run_wave_quant_v31.sh's session (results/wq31dp{K}_*, wq31_{K}_*: the wave-quantization figure)
and run_one_wave.sh's (results/ow_*{K}_*): SQ (near-square grids), WQ1W (the figure's M = 4096 first wave,
re-run), SHAPEGRID (square to 1 x T at T busy CUs, cold and warm), SKIDEAL (4096x4096xK', K' = K T / 256)
and v31 again. Pinned-clock passes, if present, are results/ow_shapegrid{K}_pin{MHz}_clock.csv.
Writes images/wave_quant_tiles/one_wave_k{K}.{png,svg}.
"""

import argparse
import csv
import glob
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
IMG = os.path.join(HERE, "images", "wave_quant_tiles")
NUM_CUS = 256

C_SQ = "#2CA58D"
C_WQ = "#2E6FBA"
C_SK = "#D62728"
C_IDEAL = "#E08E0B"
C_GUIDE = "#8C8C8C"
C_TEXT = "#333333"
T_COLORS = {32: "#9ECAE1", 64: "#6BAED6", 128: "#3182BD", 192: "#08519C", 256: "#08306B"}


def rows(name, tile=None):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return []
    out = []
    for r in csv.DictReader(open(path)):
        if tile and r["tile"] != tile:
            continue
        out.append({k: (float(v) if re.fullmatch(r"-?[\d.]+(e-?\d+)?", v or "") else v) for k, v in r.items()})
    return out


def key(r):
    return int(r["M"]), int(r["N"]), int(r["K"])


def by_shape(rs):
    return {key(r): r for r in rs}


def col(rs, k):
    return np.array([r[k] for r in rs], dtype=float)


def fabric_tbps(cnt, tim):
    """The L2's reads beyond itself (MALL or HBM: TCC_EA0_RDREQ_DRAM counts both) per second of the
    back-to-back time. None without counters."""
    c = cnt.get(key(tim))
    if not c or not c.get("ea_rd_MB"):
        return None
    return c["ea_rd_MB"] * 1e6 / (tim["us_median"] * 1e-6) / 1e12


def style(ax, title, hint=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9A9A9A")
    ax.grid(color="#E3E3E3", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=C_TEXT, labelsize=9)
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=C_TEXT, pad=8)
    if hint:
        ax.set_title(hint, loc="right", fontsize=9, color=C_GUIDE, style="italic", pad=8)


def drift(new, old, k):
    o = by_shape(old)
    d = [new_r[k] / o[key(new_r)][k] - 1 for new_r in new if key(new_r) in o]
    return (min(d), max(d)) if d else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4096)
    args = ap.parse_args()
    K = args.k
    sys.path.insert(0, os.path.dirname(HERE))
    from v31_streamk_aiter.matmul_kernel import streamk_plan

    sq_t, sq_c = rows(f"ow_sq{K}_timing_graph.csv"), rows(f"ow_sq{K}_clock.csv")
    wq_t, wq_c = rows(f"ow_wq1w{K}_timing_graph.csv"), rows(f"ow_wq1w{K}_clock.csv")
    sg_t, sg_c = rows(f"ow_shapegrid{K}_timing_graph.csv"), rows(f"ow_shapegrid{K}_clock.csv")
    sg_w = by_shape(rows(f"ow_shapegrid{K}_warm_clock.csv"))
    sk_t, sk_c = rows(f"ow_skideal{K}_timing_graph.csv"), rows(f"ow_skideal{K}_clock.csv")
    v31_t, v31_c = rows(f"ow_wq31{K}_timing_graph.csv"), rows(f"ow_wq31{K}_clock.csv")
    cnt = {**by_shape(rows(f"ow_wq1w{K}_counters.csv")), **by_shape(rows(f"ow_sq{K}_counters.csv")),
           **by_shape(rows(f"ow_shapegrid{K}_counters.csv")), **by_shape(rows(f"ow_skideal{K}_counters.csv"))}
    old_dp = rows(f"wq31dp{K}_timing_graph.csv", "256x256")
    old_dp_c = rows(f"wq31dp{K}_clock.csv", "256x256")
    old_v31 = rows(f"wq31_{K}_timing_graph.csv", "v31")
    old_v31_c = rows(f"wq31_{K}_clock.csv", "v31")
    assert all(r["correct"] == "True" for r in sq_t + wq_t + sg_t + sk_t + v31_t), "a point failed its check"

    # Shapes measured again (the clock at 144 tiles is bimodal): the median over every reading.
    reps = {}
    for r in rows(f"ow_recheck{K}_clock.csv") + rows(f"ow_repeat{K}_clock.csv"):
        reps.setdefault(key(r), []).append(r)
    for rs in (sq_c, wq_c, sg_c):
        for r in rs:
            if key(r) in reps:
                allr = [r] + reps[key(r)]
                g = col(allr, "sclk_ghz")
                print(f"  {int(r['M'])}x{int(r['N'])}: {len(allr)} clock readings, {g.min():.2f}-{g.max():.2f} GHz, "
                      f"{col(allr, 'power_w').min():.0f}-{col(allr, 'power_w').max():.0f} W; median used")
                for k in ("sclk_ghz", "power_w", "sustained_tflops"):
                    r[k] = float(np.median(col(allr, k)))

    # Overlay check against the figure's session.
    print(f"K = {K}: drift of today's first wave against the wave-quantization figure's session")
    for name, new, old, k in (("256x256 time", wq_t, old_dp, "us_median"), ("v31 time", v31_t, old_v31, "us_median"),
                              ("256x256 clock", wq_c, old_dp_c, "sclk_ghz"), ("v31 clock", v31_c, old_v31_c,
                                                                              "sclk_ghz")):
        d = drift(new, old, k)
        if d:
            print(f"  {name}: {100 * d[0]:+.1f}% to {100 * d[1]:+.1f}%")

    # Clock against busy CUs.
    print("\n  tiles | square: shape GHz W ns/k-step TB/s | M = 4096: shape GHz W ns/k-step TB/s")
    sq_tm, wq_tm = by_shape(sq_t), by_shape(wq_t)
    for a, b in zip(sq_c, wq_c):
        line = f"  {int(a['tiles']):5d}"
        for c, tm in ((a, sq_tm), (b, wq_tm)):
            t = tm[key(c)]
            hb = fabric_tbps(cnt, t)
            line += (f" | {int(c['M'])}x{int(c['N'])} {c['sclk_ghz']:.2f} {c['power_w']:.0f} "
                     f"{1e3 * t['us_median'] / (K // 64):.0f} {hb if hb is None else round(hb, 2)}")
        print(line)

    print("\n  SHAPEGRID: T shape | cold: us GHz W L2-read TB/s L2-hit | warm: GHz W")
    sg_tm = by_shape(sg_t)
    for c in sg_c:
        t = sg_tm[key(c)]
        w = sg_w.get(key(c))
        hb = fabric_tbps(cnt, t)
        l2 = cnt.get(key(c), {}).get("l2_hit")
        print(f"  {int(c['tiles']):3d} {int(c['M'])}x{int(c['N'])} | {t['us_median']:.1f} {c['sclk_ghz']:.2f} "
              f"{c['power_w']:.0f} {hb if hb is None else round(hb, 2)} {l2} | "
              + (f"{w['sclk_ghz']:.2f} {w['power_w']:.0f}" if w else "-"))

    pins = {}
    for f in sorted(glob.glob(os.path.join(RES, f"ow_shapegrid{K}_pin*_clock.csv"))):
        mhz = int(re.search(r"_pin(\d+)_", f).group(1))
        pins[mhz] = rows(os.path.basename(f))
    fits = {}
    for mhz, pr in pins.items():
        X = np.array([[1.0, r["tiles"], fabric_tbps(cnt, sg_tm[key(r)]) or 0.0] for r in pr])
        y = col(pr, "power_w")
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        fits[mhz] = coef
        rms = np.sqrt(np.mean((X @ coef - y) ** 2))
        print(f"\n  pinned at {mhz} MHz: P = {coef[0]:.0f} W + {coef[1]:.2f} W per busy CU + {coef[2]:.0f} W per "
              f"L2-read TB/s (rms {rms:.0f} W); clocks {col(pr, 'sclk_ghz').min():.2f}-{col(pr, 'sclk_ghz').max():.2f}")

    # The ideal split against data-parallel and v31 at the same N.
    full = [r for r in old_dp if int(r["tiles"]) % NUM_CUS == 0]
    per_wave, fixed = np.polyfit(col(full, "tiles") / NUM_CUS, col(full, "us_median"), 1)
    ks = per_wave / (K // 64)
    sk_n = {int(r["K"]) * NUM_CUS // K * 16: r for r in sk_t}
    skc_n = {int(r["K"]) * NUM_CUS // K * 16: r for r in sk_c}
    wq_cn = {int(r["N"]): r for r in wq_c}
    v31_n = {int(r["N"]): r for r in v31_t}
    print(f"\n  ideal split (4096x4096xK') against the first wave; dashed = {fixed:.1f} + {per_wave:.1f} x tiles / 256 us,"
          f" k-step {ks:.2f} us")
    print("  N  K' | us: DP v31 ideal dashed | ideal-dashed, v31-ideal in k-steps | GHz: DP ideal")
    for r in wq_t:
        n = int(r["N"])
        if n not in sk_n:
            continue
        i, d = sk_n[n], fixed + per_wave * r["tiles"] / NUM_CUS
        v = v31_n[n]["us_median"]
        print(f"  {n:5d} {int(i['K']):4d} | {r['us_median']:.1f} {v:.1f} {i['us_median']:.1f} {d:.1f} | "
              f"{(i['us_median'] - d) / ks:+.1f} {(v - i['us_median']) / ks:+.1f} | "
              f"{wq_cn[n]['sclk_ghz']:.2f} {skc_n[n]['sclk_ghz']:.2f}")

    def draw_cus(ax_a, ax_b, la, lb, notes=True):
        """Clock and power against busy CUs; notes: the wave-quantization figure's name and the 144-tile note."""
        wq_name = "M = 4096, N = 256..4096" + (" (the figure's first wave)" if notes else "")
        for ax, k, lab, letter in ((ax_a, "sclk_ghz", "Clock (GHz)", la), (ax_b, "power_w", "Power (W)", lb)):
            style(ax, f"({letter})  Clock against busy CUs" if k == "sclk_ghz" else f"({letter})  Power against busy CUs",
                  "same K, one tile per busy CU")
            for rs, c, name in ((wq_c, C_WQ, wq_name),
                                (sq_c, C_SQ, "near-square grid (aspect at most 2)")):
                ax.plot(col(rs, "tiles"), col(rs, k), color=c, lw=2, marker="o", ms=4, label=name)
            ax.set_xlabel("busy CUs (tiles)", fontsize=10, color=C_TEXT)
            ax.set_ylabel(lab, fontsize=10, color=C_TEXT)
            ax.set_xlim(0, 264)
            ax.xaxis.set_major_locator(plt.MultipleLocator(32))
            if k == "power_w":
                ax.axhline(1400, color=C_GUIDE, ls=(0, (4, 3)), lw=1)
                ax.text(8, 1385, "1400 W cap", fontsize=8.5, color=C_GUIDE, va="top")
            ax.legend(fontsize=8.5, loc="lower left" if k == "sclk_ghz" else "lower right", frameon=False)
            bim = [r for r in sq_c if key(r) in reps and len(reps[key(r)]) >= 4 and int(r["tiles"]) == 144]
            if bim and k == "sclk_ghz" and notes:
                g = col([bim[0]] + reps[key(bim[0])], k)
                ax.annotate(f"3072x3072 is bimodal:\n{g.min():.2f}-{g.max():.2f} GHz over {len(g)} runs (median)",
                            (144, bim[0][k]), xytext=(14, 0), textcoords="offset points", fontsize=8, color=C_TEXT,
                            va="center")

    N = col(wq_t, "N")
    split = np.array([streamk_plan(4096, int(n), K, (256, 256, 64), NUM_CUS)[0] != "dp" for n in N])
    ideal_n = np.array(sorted(sk_n))

    def draw_first_wave(ax_e, ax_f, le, lf):
        """Throughput and duration over the first wave: data-parallel, v31 and the ideal split."""
        for ax, k, letter in ((ax_e, "tflops", le), (ax_f, "us_median", lf)):
            style(ax, f"({letter})  First wave: throughput" if k == "tflops" else f"({letter})  First wave: duration",
                  "TFLOPS, higher is better" if k == "tflops" else "µs, lower is better")
            ax.plot(N, col(wq_t, k), color=C_WQ, lw=2, marker="o", ms=4, label="256×256×64 data-parallel")
            ax.scatter(N[split], col(v31_t, k)[split], marker="D", s=40, color=C_SK, edgecolor="white", lw=0.8,
                       zorder=5, label="Stream-K (default plan)")
            ax.plot(ideal_n, [sk_n[n][k] for n in ideal_n], color=C_IDEAL, lw=2, marker="s", ms=4, zorder=4,
                    label="ideal split: 4096×4096×K', same k-steps on 256 CUs")
            if k == "us_median":
                ax.plot(N, fixed + per_wave * col(wq_t, "tiles") / NUM_CUS, color=C_GUIDE, ls=(0, (5, 3)), lw=1.4,
                        label="dashed line of the wave-quantization figure")
            kp = "N" if K == 4096 else f"{K // 4096} N"
            ax.set_xlabel(f"N  (M = 4096, K = {K}; ideal split: 4096×4096×K', K' = {kp})", fontsize=10,
                          color=C_TEXT)
            ax.set_ylabel("TFLOPS" if k == "tflops" else "Duration (µs)", fontsize=10, color=C_TEXT)
            ax.set_xlim(0, 4096 + 128)
            ax.set_ylim(0, None)
            ax.xaxis.set_major_locator(plt.MultipleLocator(512))
            ax.legend(fontsize=8.5, loc="lower right", frameon=False)

    # Figure
    fig, axs = plt.subplots(3, 2, figsize=(13, 13.5), dpi=180, gridspec_kw={"hspace": 0.42, "wspace": 0.22})
    fig.patch.set_facecolor("white")
    (ax_a, ax_b), (ax_c, ax_d), (ax_e, ax_f) = axs
    draw_cus(ax_a, ax_b, "a", "b")

    have_cnt = all(fabric_tbps(cnt, sg_tm[key(r)]) is not None for r in sg_c)
    for ax, k, lab in ((ax_c, "sclk_ghz", "Clock (GHz)"), (ax_d, "power_w", "Power (W)")):
        style(ax, "(c)  Clock against memory traffic" if k == "sclk_ghz" else "(d)  Power against memory traffic",
              "T busy CUs, grid square to 1 x T")
        for T, color in T_COLORS.items():
            rs = [r for r in sg_c if int(r["tiles"]) == T]
            x = [fabric_tbps(cnt, sg_tm[key(r)]) if have_cnt else r["M"] / r["N"] for r in rs]
            x, rs = (list(v) for v in zip(*sorted(zip(x, rs), key=lambda p: p[0])))
            ax.plot(x, col(rs, k), color=color, lw=2, marker="o", ms=5, label=f"T = {T}")
            for xi, r in zip(x, rs):
                if r is rs[-1] or r is rs[0]:
                    ax.annotate(f"{int(r['M'] // 256)}x{int(r['N'] // 256)}", (xi, r[k]), textcoords="offset points",
                                xytext=(4, 4), fontsize=7, color=color)
                w = sg_w.get(key(r))
                if w:
                    ax.scatter([xi], [w[k]], s=28, facecolor="white", edgecolor=color, lw=1.4, zorder=4)
        if k == "power_w":
            ax.axhline(1400, color=C_GUIDE, ls=(0, (4, 3)), lw=1)
            for mhz, pr in pins.items():
                x = [fabric_tbps(cnt, sg_tm[key(r)]) for r in pr]
                ax.scatter(x, col(pr, "power_w"), marker="x", s=24, color=C_GUIDE, zorder=4)
                c0, c1, c2 = fits[mhz]
                ax.text(0.02, 0.03, f"x = pinned at {mhz} MHz: P ≈ {c0:.0f} W + {c1:.2f} W per CU + {c2:.0f} W per TB/s",
                        transform=ax.transAxes, fontsize=8, color=C_TEXT)
        ax.set_xlabel("L2 reads from MALL or HBM, TB/s (counters, cold run)" if have_cnt else "grid aspect M / N",
                      fontsize=10, color=C_TEXT)
        ax.set_ylabel(lab, fontsize=10, color=C_TEXT)
        handles = [Line2D([], [], color=c, lw=2, marker="o", ms=4, label=f"T = {T}") for T, c in T_COLORS.items()]
        if sg_w:
            handles.append(Line2D([], [], lw=0, marker="o", ms=5, markerfacecolor="white", markeredgecolor=C_TEXT,
                                  label="warm: inputs in the MALL, no HBM reads"))
        ax.legend(handles=handles, fontsize=8, loc="lower left" if k == "sclk_ghz" else "lower right", frameon=False)
        if k == "sclk_ghz":
            mb = [r for r in sg_c if int(r["tiles"]) == 256 and int(r["N"]) == 256]
            if mb and have_cnt:
                r = mb[0]
                ax.annotate(f"1x256 is memory-bound: {sg_tm[key(r)]['us_median']:.0f} µs against "
                            f"{sg_tm[(4096, 4096, K)]['us_median']:.0f}\nfor 16x16; the CUs stall, so the clock rises",
                            (fabric_tbps(cnt, sg_tm[key(r)]), r[k]), xytext=(3.55, 1.9), ha="right", va="center",
                            fontsize=8, color=C_TEXT,
                            arrowprops=dict(arrowstyle="-|>", color=C_GUIDE, lw=1, shrinkA=2, shrinkB=4))

    draw_first_wave(ax_e, ax_f, "e", "f")

    fig.suptitle("One wave of 256×256×64 on MI355X: clock, memory traffic and the ideal split",
                 x=0.06, y=0.975, ha="left", fontsize=15, fontweight="bold", color="#1A1A1A")
    fig.text(0.06, 0.952, f"BF16, K = {K}, back-to-back launches (HIP graph), hwmon clock and power, GPU 6. "
             "Hollow = warm inputs.", ha="left", va="top", fontsize=10, color="#555555")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.915, bottom=0.05)
    os.makedirs(IMG, exist_ok=True)
    for ext in ("png", "svg"):
        out = os.path.join(IMG, f"one_wave_k{K}.{ext}")
        fig.savefig(out, facecolor="white")
        print(f"wrote {out}")

    # Supplement: the busy-CU panels and the first-wave panels, each pair with its explanation.
    t_cus, ghz, watts = col(wq_c, "tiles"), col(wq_c, "sclk_ghz"), col(wq_c, "power_w")
    t_cap = int(t_cus[np.argmax(watts >= 1395)])
    n_sp = [int(n) for n in N[split] if int(n) in sk_n]
    gap = np.array([(v31_n[n]["us_median"] - sk_n[n]["us_median"]) / ks for n in n_sp])
    over = np.array([(sk_n[n]["us_median"] - (fixed + per_wave * n / 16 / NUM_CUS)) / ks for n in sk_n])
    n_ex = max(n_sp)
    tf = lambda rs, n: next(r["tflops"] for r in rs if int(r["N"]) == n)  # noqa: E731
    text_cus = (
        f"(a) and (b): every busy CU does the same work (one 256×256 tile, K = {K}); only the number of busy CUs "
        f"changes. As CUs are added, power rises from {watts[0]:.0f} W to the 1400 W cap (reached at {t_cap} CUs) and "
        f"the clock falls from {ghz[0]:.2f} to {ghz[-1]:.2f} GHz. So a partly full wave runs each tile faster than a "
        "full one: filling the idle CUs, with Stream-K or any other method, also lowers the clock, and the gain "
        "shrinks as the last wave approaches full. The near-square grids and the wave-quantization figure's M = 4096 "
        "shapes give the same curves, so this is the number of busy CUs, not the shape of the problem."
    )
    text_split = (
        f"(c) and (d): the ideal split is the first wave's total work spread evenly over all 256 CUs as a plain GEMM "
        f"(4096×4096×K', K' = {'N' if K == 4096 else f'{K // 4096} N'}): Stream-K with no partial sums, no waiting "
        "and no fixup. Like the roof of a roofline, it is the best a split can reach with this GEMM kernel. "
        f"Wherever v31's Stream-K splits, it is {gap.min():.0f}-{gap.max():.0f} k-steps "
        f"({gap.min() * ks:.0f}-{gap.max() * ks:.0f} µs) slower than the ideal; at N = {n_ex} it gives "
        f"{tf(v31_t, n_ex):.0f} TFLOPS against the ideal's {sk_n[n_ex]['tflops']:.0f} "
        f"(data-parallel: {tf(wq_t, n_ex):.0f}). That gap is the cost of the split itself. The ideal sits "
        f"{over.min():.0f}-{over.max():.0f} k-steps above the dashed line of the wave-quantization figure, because a "
        "CU with little work still pays the prologue and epilogue."
    )
    title = "Supporting experiments: why the gain shrinks near a full wave, and how close Stream-K gets"
    sfig, ((s_a, s_b), (s_c, s_d)) = plt.subplots(2, 2, figsize=(13, 9.5), dpi=180,
                                                  gridspec_kw={"hspace": 0.38, "wspace": 0.22})
    sfig.patch.set_facecolor("white")
    draw_cus(s_a, s_b, "a", "b", notes=False)
    draw_first_wave(s_c, s_d, "c", "d")
    sfig.suptitle(title, x=0.06, y=0.975, ha="left", fontsize=15, fontweight="bold", color="#1A1A1A")
    sfig.text(0.06, 0.945, f"BF16 GEMM, 256×256×64 tiles, one wave (at most 256 tiles, one per CU), K = {K}. "
              "Back-to-back launches (HIP graph), hwmon clock and power, MI355X GPU 6.",
              ha="left", va="top", fontsize=10, color="#555555")
    sfig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.07)
    for ext in ("png", "svg"):
        out = os.path.join(IMG, f"one_wave_supplement_k{K}.{ext}")
        sfig.savefig(out, facecolor="white")
        print(f"wrote {out}")
    out = os.path.join(IMG, f"one_wave_supplement_k{K}.md")
    with open(out, "w") as f:
        f.write(f"**{title}.**\n\n{text_cus}\n\n{text_split}\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
