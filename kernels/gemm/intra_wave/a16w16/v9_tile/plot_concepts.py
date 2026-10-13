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
"""Concept figures for the v9_tile worklog (§10), from the measured CSVs in results/.

    python v9_tile/plot_concepts.py     # writes v9_tile/images/concepts/*.png

1. window.png     where the prefetch distance d must sit: L/C <= d <= Q/B, per tile
2. roofline.png   per-CU roofline: compute roofs per clock, load roofs Q/L (cold, warm)
3. power.png      clock against MFMA utilisation at the power cap; throughput = product
4. budget.png     where the chip's CU-time goes for 4096x2048x8192, three ways
5. split_k.png    the tile against split-K across K (split-K estimated)

Model constants: Q = 32 KB in flight per CU; L = 550 ns cold, ~385 ns warm (both inferred
from the measured bytes per ns with Q = 32 KB); per-CU peak 4096 FLOP per clock.
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "images", "concepts")

Q = 32768  # bytes in flight per CU
L_COLD, L_WARM = 550.0, 385.0  # ns
FLOP_PER_CLK_CU = 4096
PEAK_PER_GHZ = 1048.576  # chip TFLOPS per GHz
COLORS = {"256x256": "tab:blue", "256x128": "tab:orange", "128x256": "tab:green", "128x128": "tab:red",
          "128x128x128": "tab:purple"}


def load(*names):
    out = {}
    for n in names:
        p = os.path.join(RES, n)
        if os.path.exists(p):
            for r in csv.DictReader(open(p)):
                out.setdefault((r["tile"], int(r["M"]), int(r["N"]), int(r["K"])), {}).update(r)
    return out


def dims(tile):
    d = [int(x) for x in tile.removesuffix("r2").split("x")]
    return d[0], d[1], d[2] if len(d) > 2 else 64


def kstep(tile):
    """(MFMA cycles per k-step per wave, bytes per k-step)."""
    bm, bn, bk = dims(tile)
    return (bm // 32) * (bn // 32) * (bk // 32) * 16, (bm + bn) * bk * 2


FULL = {"256x256": (4096, 4096), "256x128": (4096, 2048), "128x128": (2048, 2048), "128x128x128": (2048, 2048)}


# --------------------------------------------------------------------------------------------
def fig_window(cold):
    tiles = ["256x256", "256x128", "128x128", "128x128x128"]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for i, t in enumerate(tiles):
        y = len(tiles) - 1 - i
        cyc, B = kstep(t)
        r = cold[(t, *FULL[t], 8192)]
        f = float(r["sclk_ghz"])
        C = cyc / f  # ns
        lo_c, lo_w, hi = L_COLD / C, L_WARM / C, Q / B
        eff = min(1, hi / lo_c)
        ax.barh(y, hi, height=0.36, color=COLORS[t], alpha=0.35)
        ax.plot([hi, hi], [y - 0.25, y + 0.25], color=COLORS[t], lw=3)
        ax.plot(lo_c, y, "v", color="black", ms=11)
        ax.plot(lo_w, y, "v", color="gray", ms=9)
        if lo_c > hi:
            ax.plot([hi, lo_c], [y, y], color="crimson", lw=6, alpha=0.6, solid_capstyle="butt")
        else:
            ax.plot([lo_c, hi], [y, y], color="green", lw=6, alpha=0.6, solid_capstyle="butt")
        ax.text(2.75, y + 0.08,
                f"C = {cyc} cyc / {f:.2f} GHz = {C:.0f} ns,  B = {B // 1024} KB\n"
                f"need L/C = {lo_c:.2f},  allowed Q/B = {hi:.2f}  ->  eff <= {eff:.0%}"
                f"   (measured loop {float(r['att_mfma_eff']):.0%})",
                va="center", fontsize=8.5)
    ax.set_yticks(range(len(tiles)))
    ax.set_yticklabels(tiles[::-1])
    ax.set_xlim(0, 5.6)
    ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5])
    ax.set_xlabel("prefetch distance d, in k-steps in flight")
    ax.set_title("Where d must sit:  L/C <= d <= Q/B   (full wave, K = 8192, cold)", fontsize=11)
    handles = [plt.Line2D([], [], marker="v", ls="", color="black", ms=10, label="L/C, cold (L = 550 ns): steps needed to hide latency"),
               plt.Line2D([], [], marker="v", ls="", color="gray", ms=9, label="L/C, warm (L = 385 ns)"),
               plt.Rectangle((0, 0), 1, 1, color="gray", alpha=0.35, label="0 ... Q/B: steps the 32 KB TCP can hold"),
               plt.Line2D([], [], color="green", lw=6, alpha=0.6, label="feasible gap: latency hidden"),
               plt.Line2D([], [], color="crimson", lw=6, alpha=0.6, label="shortfall: eff <= (Q/B) / (L/C)")]
    ax.legend(handles=handles, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
    ax.axvline(2.65, color="lightgray", lw=0.8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "window.png"), dpi=130, bbox_inches="tight")


# --------------------------------------------------------------------------------------------
def fig_roofline(cold, warm):
    fig, ax = plt.subplots(figsize=(10, 6))
    xs = [i for i in range(1, 301)]
    for f, ls in ((1.71, "-"), (1.94, "--"), (2.4, ":")):
        R = FLOP_PER_CLK_CU * f / 1e3
        ax.axhline(R, color="dimgray", ls=ls, lw=1.2)
        ax.text(302, R, f"compute roof\n{f} GHz: {R:.1f}", va="center", fontsize=8)
    for bns, lab, col, x0, txt in ((Q / L_COLD, "cold", "black", 170, (185, 10.4)),
                                   (Q / L_WARM, "warm", "gray", 110, (8, 10.4))):
        ax.plot(xs, [bns * x / 1e3 for x in xs], color=col, lw=1.5)
        ax.annotate(f"load roof, {lab}: Q/L = {bns:.0f} B/ns, x I", xy=(x0, bns * x0 / 1e3), xytext=txt,
                    color=col, fontsize=8.5, arrowprops=dict(arrowstyle="-", color=col, lw=0.8))
    for t in ["256x256", "256x128", "128x128", "128x128x128"]:
        bm, bn, _ = dims(t)
        I = bm * bn / (bm + bn)
        ax.axvline(I, color=COLORS[t], lw=0.6, alpha=0.5)
        rc = cold.get((t, *FULL[t], 8192))
        if rc:
            y = float(rc["att_mfma_eff"]) * FLOP_PER_CLK_CU * float(rc["sclk_ghz"]) / 1e3
            ax.plot(I, y, "o", color=COLORS[t], ms=9, label=f"{t}  (I = {I:.0f})  cold, K = 8192")
        rw = warm.get((t, *FULL[t], 4096))
        if rw:
            y = float(rw["att_mfma_eff"]) * FLOP_PER_CLK_CU * float(rw["sclk_ghz"]) / 1e3
            ax.plot(I, y, "^", color=COLORS[t], ms=9, mfc="white", mew=2, label=f"{t}  warm, K = 4096")
    R = FLOP_PER_CLK_CU * 1.94 / 1e3
    ridge = R * 1e3 / (Q / L_COLD)
    ax.annotate(f"ridge at 1.94 GHz, cold:\nI = R / (Q/L) = {ridge:.0f} FLOP/B", xy=(ridge, R), xytext=(150, 5.2),
                arrowprops=dict(arrowstyle="->"), fontsize=8.5)
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 11)
    ax.set_xlabel("tile intensity I = BM*BN / (BM+BN)   (FLOP per byte a CU loads)")
    ax.set_ylabel("TFLOPS per CU, main loop (ATT efficiency x clock x peak)")
    ax.set_title("Per-CU roofline: the load roof is Q / L (32 KB over the round trip), not HBM")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "roofline.png"), dpi=130)


# --------------------------------------------------------------------------------------------
def fig_power(full, low):
    fig, ax = plt.subplots(figsize=(10, 6))
    for T in (600, 800, 1000, 1200, 1400, 1600):
        us = [u / 100 for u in range(15, 101)]
        fs = [T / (PEAK_PER_GHZ * u) for u in us]
        pts = [(u, f) for u, f in zip(us, fs) if 1.5 <= f <= 2.5]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color="lightgray", lw=1)
            ax.text(pts[0][0], 2.49, f"{T}", fontsize=7.5, color="gray", ha="center", va="top")
    for (t, M, N, K), r in sorted(full.items()):
        if t not in COLORS or not r.get("sclk_ghz") or not r.get("tflops") or K < 2048:
            continue
        f, T = float(r["sclk_ghz"]), float(r["tflops"])
        u = T / (PEAK_PER_GHZ * f)
        ax.plot(u, f, "o", color=COLORS[t], ms=7)
        ax.annotate(f"K={K // 1024}k", (u, f), textcoords="offset points", xytext=(4, -9), fontsize=6.5)
    for (t, M, N, K), r in sorted(low.items()):
        if int(r["tiles"]) >= 256:
            continue
        f, T = float(r["sclk_ghz"]), float(r["sustained_tflops"])
        u = T / (PEAK_PER_GHZ * f)
        ax.plot(u, f, "s", color=COLORS[t], ms=7, mfc="white", mew=1.8)
        if int(r["tiles"]) == 128:
            ax.annotate(f"{t}, 128 tiles, {float(r['power_w']):.0f} W", (u, f), textcoords="offset points",
                        xytext=(6, -12), fontsize=7.5)
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS[t], label=f"{t}, full wave, at the 1400 W cap")
               for t in ["256x256", "256x128", "128x256", "128x128", "128x128x128"]]
    handles += [plt.Line2D([], [], marker="s", ls="", color="k", mfc="white",
                           label="8 / 32 / 128 tiles, K = 8192 (power to spare)"),
                plt.Line2D([], [], color="lightgray", label="constant TFLOPS (labelled at top) = 1.05 x clock x util")]
    ax.legend(handles=handles, fontsize=8, loc="upper right")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(1.6, 2.5)
    ax.set_xlabel("chip MFMA utilisation (back-to-back TFLOPS / (peak per GHz x clock); idle CUs count as idle)")
    ax.set_ylabel("graphics clock under sustained load (GHz)")
    ax.set_title("At the power cap, idle MFMAs buy clock, but clock does not shorten a ns-latency limit")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "power.png"), dpi=130)


# --------------------------------------------------------------------------------------------
def fig_budget(b, a, ks):
    M, N, K = 4096, 2048, 8192
    flop = 2 * M * N * K
    rows = []
    r = ks.get(("256x256", M, N, K)) or b[("256x256", M, N, K)]
    c = b[("256x256", M, N, K)]
    T, f = flop / float(r["tflops"]) / 1e6, float(c["sclk_ghz"])
    util = float(r["tflops"]) / (PEAK_PER_GHZ * f)
    rows.append(("256x256, 128 tiles\n(half the CUs idle)", T, f, float(c["power_w"]),
                 [("MFMA busy", util), ("CUs with no tile", 0.5), ("load stalls, prologue, epilogue", 0.5 - util)]))
    r, c = b[("256x128", M, N, K)], b[("256x128", M, N, K)]
    T, f = flop / float(r["tflops"]) / 1e6, float(c["sclk_ghz"])
    util = float(r["tflops"]) / (PEAK_PER_GHZ * f)
    rows.append(("256x128, 256 tiles", T, f, float(c["power_w"]),
                 [("MFMA busy", util), ("load stalls, prologue, epilogue", 1 - util)]))
    # split-K 2: a full wave of 256x256 at K/2 (measured 4096x4096x4096) plus a ~12 k-step owner ending
    full = a[("256x256", 4096, 4096, K // 2)]
    T0, f = float(full["us_median"]), float(full["sclk_ghz"])
    u0 = float(full["tflops"]) / (PEAK_PER_GHZ * f)
    steps = K // 2 // 64
    T = T0 * (steps + 12) / steps
    s = steps / (steps + 12)
    rows.append(("256x256, split-K 2\n(estimate)", T, f, float(full["power_w"]),
                 [("MFMA busy", u0 * s), ("load stalls, prologue, epilogue", (1 - u0) * s), ("fixup (owner ending)", 1 - s)]))
    seg_col = {"MFMA busy": "tab:green", "CUs with no tile": "lightgray", "load stalls, prologue, epilogue": "tab:orange",
               "fixup (owner ending)": "tab:red"}
    fig, ax = plt.subplots(figsize=(11, 4.3))
    seen = set()
    for i, (name, T, f, p, segs) in enumerate(rows):
        y = len(rows) - 1 - i
        x = 0
        for lab, frac in segs:
            w = T * frac
            ax.barh(y, w, left=x, color=seg_col[lab], label=None if lab in seen else lab, edgecolor="white")
            if frac > 0.07:
                ax.text(x + w / 2, y, f"{frac:.0%}", ha="center", va="center", fontsize=8)
            seen.add(lab)
            x += w
        ax.text(T + 1.5, y, f"{T:.0f} us  ({flop / T / 1e6:.0f} TFLOPS)\n{f:.2f} GHz, {p:.0f} W", va="center", fontsize=8.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=9)
    ax.set_xlim(0, 185)
    ax.set_xlabel("kernel time (us); each bar split by where the chip's CU-time goes")
    ax.set_title(f"{M}x{N}x{K}: the shortest bar wins, not the most 'compute-bound' one")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "budget.png"), dpi=130)


# --------------------------------------------------------------------------------------------
def fig_split_k(a, ks):
    M, N = 4096, 2048
    Ks = [2048, 4096, 8192, 16384]
    fig, ax = plt.subplots(figsize=(9, 5))
    for t, lab in (("256x256", "256x256, 128 tiles (measured)"), ("256x128", "256x128, 256 tiles (measured)")):
        pts = [(K, float((ks.get((t, M, N, K)) or a.get((t, M, N, K)))["tflops"])) for K in Ks
               if ks.get((t, M, N, K)) or a.get((t, M, N, K))]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=COLORS[t], label=lab)
    est = []
    for K in Ks:
        full = a.get(("256x256", 4096, 4096, K // 2))
        if not full:
            continue
        steps = K // 2 // 64
        T = float(full["us_median"]) * (steps + 12) / steps
        est.append((K, 2 * M * N * K / T / 1e6))
    ax.plot([p[0] for p in est], [p[1] for p in est], "s--", color="tab:red",
            label="256x256 split-K 2 (estimate: full wave at K/2 + 12 k-step ending)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ks)
    ax.set_xticklabels([str(k) for k in Ks])
    ax.set_xlabel("K")
    ax.set_ylabel("TFLOPS, back to back")
    ax.set_title(f"{M}x{N}xK: the fixup is a fixed cost, so K decides tile against split-K")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "split_k.png"), dpi=130)


def main():
    os.makedirs(OUT, exist_ok=True)
    cold = load("a_att.csv", "a128_att.csv", "a_clock.csv", "a128_clock.csv")
    warm = load("warm_att.csv", "warm_clock.csv")
    full = load("a_timing_graph.csv", "a128_timing_graph.csv", "a_clock.csv", "a128_clock.csv")
    low = load("a_lowload_clock.csv")
    a = load("a_timing_graph.csv", "a_clock.csv")
    b = load("b_timing_graph.csv", "b_clock.csv")
    ks = load("k_timing_graph.csv")
    fig_window(cold)
    fig_roofline(cold, warm)
    fig_power(full, low)
    fig_budget(b, a, ks)
    fig_split_k(a, ks)


if __name__ == "__main__":
    main()
