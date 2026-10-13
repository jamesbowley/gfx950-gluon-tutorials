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
"""Small tiles (WORKLOG §14): measured against model_small.py's ceilings.

    python v9_tile/report_small.py cb      # compute-bound table (SMALL_CB, K = 4096 and 8192)
    python v9_tile/report_small.py mb      # memory-bound table (SMALL_MB)
    python v9_tile/report_small.py warm    # SMALL_WARM against the cold runs
    python v9_tile/report_small.py plots   # images/small_tiles/*.png

Columns, per point:
- busy:   MFMA busy over the whole kernel and every SIMD (SQ_VALU_MFMA_BUSY_CYCLES / (cycles x 1024)),
          padded rows included.
- useful: back-to-back TFLOPS of the real M x N x K over the peak at the measured clock (= busy x the
          padding BM BN / (Mp Np), less the prologue and epilogue).
- B/ns:   bytes the CU loads (the real rows of A and B, every k-step of every tile) per ns per CU.
- model:  model_small.ceilings at the measured clock, cold (L = 550 ns), with the tiles per CU of the
          run: 4 / W pinned; natural, the measured mean waves per CU over W.
"""

import csv
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model_small as ms  # noqa: E402
import sweep  # noqa: E402

RES = os.path.join(HERE, "results")
ATT_DIR = os.path.join(HERE, "tmp", "small", "sweep_att")
OUT = os.path.join(HERE, "images", "small_tiles")


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return {}
    return {(r["tile"], int(r["M"]), int(r["N"]), int(r["K"])): r for r in csv.DictReader(open(p))}


def f(r, k, default=None):
    v = (r or {}).get(k, "")
    return float(v) if v not in ("", None) else default


def kernel_tile(tile):
    """(km, kn, bk, W, pin, small): the kernel's own tile (the small side as M)."""
    bm, bn, bk, regions, w, pin = sweep.tile_spec(tile)
    if sweep.is_small(tile):
        return min(bm, bn), max(bm, bn), bk, w, pin, True
    return bm, bn, bk, 4, False, False


def stalls(point):
    """att_stalls.analyze on the point's ATT directory: (cycles per iteration, {class: stall})."""
    import att_stalls

    d = os.path.join(ATT_DIR, point.replace(":", "_"))
    if not glob.glob(os.path.join(d, "**", "ui_*"), recursive=True):
        return None
    try:
        r = att_stalls.analyze(d)
    except Exception:
        return None
    return r["total_latency"], {k: v[1] for k, v in r["by"].items()}


def cb_rows(prefix="small_cb"):
    timing, clock, counters, att = (load(f"{prefix}_{s}.csv") for s in ("timing_graph", "clock", "counters", "att"))
    rows = []
    for key, t in timing.items():
        tile, M, N, K = key
        km, kn, bk, w, pin, small = kernel_tile(tile)
        c, n, a = clock.get(key), counters.get(key), att.get(key)
        ghz = f(c, "sclk_ghz", ms.CLOCK)
        us = f(t, "us_median")
        tiles = int(t["tiles"])
        bytes_loaded = tiles * (K // bk) * (km + kn) * bk * 2
        waves_simd = f(n, "waves_per_simd", f(n, "occupancy_waves_per_cu"))
        tiles_cu = 4 // w if pin else max(1.0, 4 * (waves_simd or 1) / w)
        m = ms.ceilings(km, kn, bk, w, tiles_cu=tiles_cu, clock=ghz)
        # Whole kernel: a launch of T tiles keeps T / ceil(T / slots) of the CU slots busy on average.
        slots = 256 * tiles_cu
        busy_frac = tiles / -(-tiles // slots) / slots
        for k in ("window", "stages", "busy", "useful"):
            m[k] *= busy_frac
        st = stalls(f"{tile}:{M}x{N}x{K}")
        rows.append({
            "tile": tile, "K": K, "W": w, "pin": "yes" if pin else "", "tiles": tiles,
            "tflops": f(c, "sustained_tflops", f(t, "tflops")), "ghz": ghz, "W_power": f(c, "power_w"),
            "useful": f(c, "mfma_util"), "busy": f(n, "mfma_busy"), "att_loop": f(a, "att_mfma_eff"),
            "bns": bytes_loaded / (us * 1e3) / 256, "waves_cu": waves_simd, "l2_hit": f(n, "l2_hit"),
            "m_window": m["window"], "m_stages": m["stages"], "m_busy": m["busy"], "m_useful": m["useful"],
            "m_binds": m["binds"], "pad": m["pad"], "stalls": st,
        })
    return rows


def pct(x):
    return "" if x is None else f"{100 * x:.1f}%"


def num(x, fmt="{:.0f}"):
    return "" if x is None else fmt.format(x)


def cb_table():
    rows = cb_rows()
    print("| tile | K | W | pinned | waves/SIMD | TFLOPS | GHz | W | MFMA busy | useful | ATT loop eff | B/ns per CU "
          "| L2 hit | model busy (window / 2 stages) | model useful | stalls per iteration: vmcnt / buffer_load / "
          "lgkm+ds_read / barrier |")
    print("|" + "---|" * 16)
    for r in sorted(rows, key=lambda r: (r["K"], sweep.SMALL_CB_TILES.index(r["tile"].removesuffix("p"))
                                         if r["tile"].removesuffix("p") in sweep.SMALL_CB_TILES else -1, r["pin"])):
        st = r["stalls"]
        sp = ""
        if st:
            s = st[1]
            sp = (f"{s.get('s_waitcnt vmcnt', 0) + s.get('s_waitcnt vmcnt+lgkmcnt', 0):.0f} / {s.get('buffer_load', 0):.0f}"
                  f" / {s.get('s_waitcnt lgkmcnt', 0) + s.get('ds_read', 0):.0f} / {s.get('s_barrier', 0):.0f}"
                  f" (of {st[0]:.0f})")
        print(f"| {r['tile']} | {r['K']} | {r['W']} | {r['pin']} | {num(r['waves_cu'], '{:.1f}')} | {num(r['tflops'])} "
              f"| {num(r['ghz'], '{:.2f}')} | {num(r['W_power'])} | {pct(r['busy'])} | {pct(r['useful'])} "
              f"| {pct(r['att_loop'])} | {num(r['bns'])} | {pct(r['l2_hit'])} | {pct(r['m_busy'])} "
              f"({pct(r['m_window'])} / {pct(r['m_stages'])}) | {pct(r['m_useful'])} | {sp} |")


def mb_rows():
    timing, counters = load("small_mb_timing_graph.csv"), load("small_mb_counters.csv")
    rows = []
    for key, t in timing.items():
        tile, M, N, K = key
        km, kn, bk, w, pin, _ = kernel_tile(tile)
        n = counters.get(key)
        bw, busy = ms.hbm_bw(M, N, K, km, kn, bk, w, tiles_cu=4 // w)
        tiles = int(t["tiles"])
        # Bytes the chip can hold in flight: per busy CU, 2 stages of each of its tiles, at most 32 KB.
        per_cu = -(-tiles // busy)
        inflight = busy * min(ms.Q, 2 * per_cu * (km + kn) * bk * 2)
        rows.append({"tile": tile, "M": M, "N": N, "K": K, "BN": kn, "BK": bk, "W": w, "pin": "yes" if pin else "",
                     "tiles": tiles, "us": f(t, "us_median"), "tbps": f(t, "tbps"), "model": bw,
                     "inflight": inflight,
                     "busy_cus": busy, "ea_ratio": f(n, "ea_rd_over_unique"), "l2_hit": f(n, "l2_hit"),
                     "waves_cu": f(n, "waves_per_simd", f(n, "occupancy_waves_per_cu"))})
    return rows


def mb_table():
    rows = mb_rows()
    print("| M x N x K | tile | W | pinned | tiles | us | TB/s | model TB/s | read beyond L2 / unique | L2 hit "
          "| waves/SIMD |")
    print("|" + "---|" * 11)
    for r in sorted(rows, key=lambda r: (r["M"], r["N"], r["BK"], r["BN"], r["pin"])):
        print(f"| {r['M']}x{r['N']}x{r['K']} | {r['tile']} | {r['W']} | {r['pin']} | {r['tiles']} | {num(r['us'], '{:.1f}')} "
              f"| {num(r['tbps'], '{:.2f}')} | {r['model']:.1f} | {num(r['ea_ratio'], '{:.2f}')} | {pct(r['l2_hit'])} "
              f"| {num(r['waves_cu'], '{:.1f}')} |")


def warm_table():
    cold = {(r["tile"], r["K"]): r for r in cb_rows()}
    warm = cb_rows("small_warm")
    print("| tile | TFLOPS cold | TFLOPS warm | B/ns cold | B/ns warm | ATT loop cold | ATT loop warm | model window warm |")
    print("|---|---|---|---|---|---|---|---|")
    for r in warm:
        c = cold.get((r["tile"], r["K"]), {})
        km, kn, bk, w, pin, small = kernel_tile(r["tile"])
        mw = ms.ceilings(km, kn, bk, w, L=ms.L_WARM, clock=r["ghz"])["window"] if small else None
        print(f"| {r['tile']} | {num(c.get('tflops'))} | {num(r['tflops'])} | {num(c.get('bns'))} | {num(r['bns'])} "
              f"| {pct(c.get('att_loop'))} | {pct(r['att_loop'])} | {pct(mw)} |")


def alt_table():
    """The alternatives (SMALL_ALT_*, ours) and the Triton reference (SMALL_REF_*), per group of
    ALT_GROUPS on the compute-bound shapes, then waves_per_eu against the pin, then memory-bound."""
    peak = sweep.PEAK_TFLOPS_PER_GHZ * 2.4  # the small tiles all ran at 2.4 GHz (§14.4)
    data = {"alt": alt_rows("alt", "cb"), "ref": alt_rows("ref", "cb")}
    att = {**load("small_alt_cb_att.csv"), **load("small_ref_cb_att.csv")}
    print("| group | variant | spec | TFLOPS | of peak at 2.4 GHz | MFMA busy | waves/SIMD | L2 hit | ATT loop eff "
          "| stalls per iteration: vmcnt / buffer_load / lgkm+ds_read / of total |")
    print("|" + "---|" * 10)
    rows = list(ALT_GROUPS) + [(t, [(lab, t + suf, "alt") for suf, lab in
                                     (("", "natural"), ("p", "LDS pin"), ("e1", "waves_per_eu 1"),
                                      ("e2", "waves_per_eu 2"))]) for t in ("16x64x64", "16x32x128", "32x256x64")]
    for title, items in rows:
        for lab, tile, pre in items:
            r = data[pre].get(tile)
            if r is None:
                continue
            t, n, key = r
            a = att.get(key)
            st = stalls(f"{tile}:{key[1]}x{key[2]}x{key[3]}")
            sp = ""
            if st:
                s = st[1]
                sp = (f"{s.get('s_waitcnt vmcnt', 0) + s.get('s_waitcnt vmcnt+lgkmcnt', 0):.0f} / "
                      f"{s.get('buffer_load', 0):.0f} / {s.get('s_waitcnt lgkmcnt', 0) + s.get('ds_read', 0):.0f} / "
                      f"{st[0]:.0f}")
            v = float(t["tflops"])
            print(f"| {title} | {lab} | {tile} | {v:.0f} | {v / peak:.1%} | {pct(f(n, 'mfma_busy'))} "
                  f"| {num(f(n, 'waves_per_simd'), '{:.2f}')} | {pct(f(n, 'l2_hit'))} | {pct(f(a, 'att_mfma_eff'))} | {sp} |")
    print("\nMemory-bound (TB/s, A + B + C once over the kernel time):\n")
    mb = {**{k: ("alt", t) for k, t in load("small_alt_mb_timing_graph.csv").items()},
          **{k: ("ref", t) for k, t in load("small_ref_mb_timing_graph.csv").items()}}
    print("| shape | BN | pad 16 | 4x64 | Triton |")
    print("|---|---|---|---|---|")
    shapes = sorted({(k[1], k[2], k[3]) for k in mb}, key=lambda s: (s[0] > s[1], min(s[0], s[1]), max(s[0], s[1])))
    for M, N, K in shapes:
        small = min(M, N)
        for bn in (64, 256):
            spec = f"{small}x{bn}x128" if M <= N else f"{bn}x{small}x128"
            w = "w1" if bn == 64 else ""
            cells = [mb.get((spec, M, N, K)), mb.get((spec + "m4", M, N, K)),
                     mb.get((f"tri{spec}{w}", M, N, K))]
            print(f"| {M}x{N}x{K} | {bn} | " + " | ".join(f"{float(c[1]['tbps']):.2f}" if c else "" for c in cells) + " |")


def mfma_cycles():
    """Cycles per MFMA from the counters: busy SIMD-cycles (mfma_busy x kernel cycles x 1024 SIMDs)
    over the MFMAs the kernel executes (tiles x k-steps x waves x MFMAs per wave per k-step). The
    16x16x32 points check the method (16 cycles); the m4 points measure mfma_f32_4x4x4bf16_1k."""
    timing, counters = load("small_alt_cb_timing_graph.csv"), load("small_alt_cb_counters.csv")
    print("| tile | MFMA | MFMAs executed (M) | busy SIMD-cycles (M) | cycles per MFMA |")
    print("|---|---|---|---|---|")
    for key, t in timing.items():
        tile, M, N, K = key
        n = counters.get(key)
        if not n or not kernel_tile(tile)[5] or "e" in tile.split("x")[-1]:
            continue
        km, kn, bk, w, _, _ = kernel_tile(tile)
        mfma4 = tile.endswith("m4")
        if mfma4:
            per_wave = (-(-km // 4)) * (kn // w // 64) * (bk // 4)
        else:
            per_wave = (-(-km // 16)) * max(1, kn // w // 16) * (bk // 32)
        executed = int(t["tiles"]) * (K // bk) * w * per_wave
        busy = float(n["mfma_busy"]) * float(n["kcycles"]) * 1e3 * 1024
        name = "4x4x4_16b" if mfma4 else "16x16x32"
        print(f"| {tile} | {name} | {executed / 1e6:.1f} | {busy / 1e6:.1f} | {busy / executed:.1f} |")


def plots():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUT, exist_ok=True)
    cb_plot(plt)
    mb_plot(plt)
    if load("small_alt_cb_timing_graph.csv"):
        alt_plot(plt)


def plain_log_ticks(ax):
    """Log axes labelled 5, 10, 20 ... instead of 5 x 10^0."""
    from matplotlib.ticker import FuncFormatter

    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(fmt)
        axis.set_minor_formatter(fmt)


def handle(marker, color, label, **kw):
    """A legend entry drawn exactly as the plotted points."""
    from matplotlib.lines import Line2D

    kw.setdefault("ls", "")
    return Line2D([], [], marker=marker, color=color, markerfacecolor=color, label=label, **kw)


def cb_plot(plt):
    """Measured MFMA busy against the model, one shape (2048x4096x8192, K = 8192): each small tile
    pinned (circle), the same tile at natural occupancy (cross, joined to its circle), the anchors."""
    rows = [r for r in cb_rows() if r["K"] == 8192 and r["busy"] is not None]
    pinned = {r["tile"].removesuffix("p"): r for r in rows if r["pin"]}
    fig, ax = plt.subplots(figsize=(14, 7.5))
    xy = lambda r: (100 * r["m_busy"], 100 * r["busy"])  # noqa: E731
    labelled = []
    for r in rows:
        small = kernel_tile(r["tile"])[5]
        if r["pin"] or not small:
            ax.plot(*xy(r), ls="", marker="o", color="tab:blue" if small else "tab:green", ms=6)
            labelled.append((r["tile"].removesuffix("p"), xy(r)))
            continue
        ax.plot(*xy(r), ls="", marker="x", color="tab:orange", ms=7, mew=1.5)
        twin = pinned.get(r["tile"])
        if twin is None:  # no pinned run (LDS or registers do not allow it): label the cross itself
            labelled.append((r["tile"] + " (natural only)", xy(r)))
        else:
            ax.plot([xy(twin)[0], xy(r)[0]], [xy(twin)[1], xy(r)[1]], color="tab:orange", lw=0.7, alpha=0.6)
            if abs(r["busy"] / twin["busy"] - 1) > 0.15:
                labelled.append((f"{r['tile']} natural", xy(r)))
    # Too many points sit within a few percent of each other for text labels: number them (from the
    # top) and list the numbers beside the plot.
    key = []
    for i, (text, (x, y)) in enumerate(sorted(labelled, key=lambda t: -t[1][1]), 1):
        ax.annotate(str(i), (x, y), fontsize=7, xytext=(4, 1), textcoords="offset points")
        key.append(f"{i:2d}  {text}  ({y:.1f}% / {x:.1f}%)")
    fig.text(0.735, 0.95, "measured / model\n\n" + "\n".join(key), fontsize=7.5, va="top", family="monospace")
    fig.subplots_adjust(right=0.72)
    ax.plot([4, 60], [4, 60], "k--", lw=0.8)
    ax.legend(handles=[
        handle("", "k", "measured = model", ls="--"),
        handle("o", "tab:blue", "small tile, pinned (one wave per SIMD)"),
        handle("x", "tab:orange", "same tile, natural occupancy (line to its pinned run)", mew=1.5),
        handle("o", "tab:green", "v9_tile anchors (256x256: 128 tiles, half the CUs)"),
    ], fontsize=8, loc="upper left")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(4, 60)
    ax.set_ylim(5, 60)
    plain_log_ticks(ax)
    ax.set_xlabel("model MFMA busy, cold L = 550 ns, at the measured clock (%)")
    ax.set_ylabel("measured MFMA busy, whole kernel, all SIMDs, padded rows included (%)")
    ax.set_title("One shape, 2048x4096x8192 bf16 (K = 8192, cold): measured MFMA busy against the model",
                 fontsize=10)
    fig.savefig(os.path.join(OUT, "cb_model.png"), dpi=130)


def mb_fit(rows):
    """Least squares (relative) over t = t0 + bytes / min(HBM, in flight / L): a fixed cost per
    kernel (launch ramp, pipeline fill, epilogue) and the round trip. Returns (t0 us, L ns, rms)."""
    best = None
    for t0 in [0.25 * i for i in range(61)]:
        for L in range(300, 1500, 5):
            err = 0.0
            for r in rows:
                bw = min(ms.HBM_TBS, r["inflight"] / L / 1e3) * 1e12
                err += ((t0 + r["bytes"] / bw * 1e6) / r["us"] - 1) ** 2
            if best is None or err < best[0]:
                best = (err, t0, L)
    return best[1], best[2], (best[0] / len(rows)) ** 0.5


def mb_plot(plt):
    """Memory-bound points, each a different shape (M x N x 8192) and tile. Left: measured kernel time
    against the fitted model's. Middle: TB/s against the tile count at BK = 128. Right: the same for
    the BK variants (M = 16). Colour is N, the marker BN; M barely matters (B is streamed)."""
    rows = [r for r in mb_rows() if not r["pin"]]  # pinned runs are the same within noise
    for r in rows:
        r["bytes"] = r["tbps"] * 1e12 * r["us"] * 1e-6
    t0, l_fit, rms = mb_fit(rows)
    ncol = {4096: "tab:blue", 8192: "tab:orange", 16384: "tab:green"}
    bkcol = {64: "tab:purple", 128: "tab:orange", 256: "tab:brown"}
    markers = {16: "o", 32: "s", 64: "^", 128: "D", 256: "v"}
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.8), gridspec_kw={"width_ratios": [1, 1.15, 0.9]})

    for r in rows:
        pred = t0 + r["bytes"] / (min(ms.HBM_TBS, r["inflight"] / l_fit / 1e3) * 1e12) * 1e6
        ax1.plot(pred, r["us"], ls="", marker=markers[r["BN"]], color=ncol[r["N"]], alpha=0.8)
    lo, hi = 10, 120
    ax1.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax1.fill_between([lo, hi], [lo * 0.85, hi * 0.85], [lo * 1.15, hi * 1.15], color="grey", alpha=0.15)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)
    plain_log_ticks(ax1)
    ax1.set_xlabel(f"model time (us) = {t0:g} + bytes / min(6.3 TB/s, in flight / {l_fit} ns)")
    ax1.set_ylabel("measured time (us), back to back")
    ax1.set_title(f"Fit over all {len(rows)} points: {100 * rms:.0f}% rms (band: +-15%)", fontsize=10)
    ax1.legend(handles=[handle("o", c, f"N = {n}") for n, c in ncol.items()]
               + [handle(m, "grey", f"BN = {bn}") for bn, m in markers.items()], fontsize=8, loc="upper left")

    for r in rows:
        if r["BK"] == 128:
            ax2.plot(r["tiles"], r["tbps"], ls="", marker=markers[r["BN"]], color=ncol[r["N"]], alpha=0.8)
    for ax in (ax2, ax3):
        ax.axhline(ms.HBM_TBS, c="k", lw=0.8, ls=":")
        ax.axvline(256, c="grey", lw=0.8, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("tiles (workgroups); dotted: one per CU")
        ax.set_ylim(0, 6.8)
    ax2.set_ylabel("TB/s (A + B + C once, over the kernel time)")
    ax2.set_title("BK = 128, M = 1-32: HBM (dotted, ~6.3 TB/s) from ~2 tiles per CU;\n"
                  "below, larger N (longer kernels) reach more of it", fontsize=10)
    ax2.legend(handles=[handle("o", c, f"N = {n}") for n, c in ncol.items()]
               + [handle(m, "grey", f"BN = {bn}") for bn, m in markers.items()], fontsize=8, loc="upper left")

    for r in rows:
        if r["M"] == 16:
            ax3.plot(r["tiles"], r["tbps"], ls="", marker=markers[r["BN"]], color=bkcol[r["BK"]], alpha=0.85)
    ax3.set_title("M = 16, all N: BK (bytes in flight per tile) matters", fontsize=10)
    ax3.legend(handles=[handle("o", c, f"BK = {bk}") for bk, c in bkcol.items()]
               + [handle(m, "grey", f"BN = {bn}") for bn, m in markers.items()], fontsize=8, loc="upper left")
    fig.suptitle("Memory-bound small tiles: each point is one shape (M x N x 8192, M = 1-32, N = 4096-16384) "
                 "and tile; bf16, cold (2 GB rotating), natural occupancy")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mb_bandwidth.png"), dpi=130)
    return t0, l_fit, rms


ALT_GROUPS = [
    # (title, [(label, tile, prefix)]); prefix "ref" for the Triton reference CSVs
    ("BM = 1", [("pad 16", "1x256x128", "alt"), ("4x64", "1x256x128m4", "alt"), ("Triton (v_dot2)", "tri1x256x128", "ref")]),
    ("BM = 2", [("pad 16", "2x256x128", "alt"), ("4x64", "2x256x128m4", "alt"), ("Triton (v_dot2)", "tri2x256x128", "ref")]),
    ("BM = 4", [("pad 16", "4x256x128", "alt"), ("4x64", "4x256x128m4", "alt"), ("Triton (4x64)", "tri4x256x128", "ref")]),
    ("BM = 8", [("pad 16", "8x256x128", "alt"), ("4x64", "8x256x128m4", "alt"), ("Triton (4x64)", "tri8x256x128", "ref")]),
    ("BM = 16", [("ours", "16x256x128", "alt"), ("Triton", "tri16x256x128", "ref")]),
    ("BN = 4", [("swap, pad 16", "256x4x128", "alt"), ("swap, 4x64", "256x4x128m4", "alt"),
                ("Triton (64x4)", "tri256x4x128", "ref")]),
    ("BN = 8", [("swap, pad 16", "256x8x128", "alt"), ("swap, 4x64", "256x8x128m4", "alt"),
                ("Triton (64x4)", "tri256x8x128", "ref")]),
    ("16x16", [("1 wave", "16x16x128", "alt"), ("1 wave, pinned", "16x16x128p", "alt"),
               ("1 wave, wpe 1", "16x16x128e1", "alt"), ("4 waves, replicated", "16x16x128w4", "alt"),
               ("Triton 1 warp", "tri16x16x128w1", "ref"), ("Triton 4 warps", "tri16x16x128", "ref")]),
]


def alt_rows(prefix, kind):
    timing, counters = load(f"small_{prefix}_{kind}_timing_graph.csv"), load(f"small_{prefix}_{kind}_counters.csv")
    return {k[0]: (t, counters.get(k), k) for k, t in timing.items()}


def alt_plot(plt):
    """Useful TFLOPS of each alternative on the compute-bound shapes, and waves_per_eu against the pin."""
    data = {"alt": alt_rows("alt", "cb"), "ref": alt_rows("ref", "cb")}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 5.8), gridspec_kw={"width_ratios": [2.2, 1]})
    x, ticks, labels = 0, [], []
    colors = {"pad 16": "tab:blue", "ours": "tab:blue", "swap, pad 16": "tab:blue", "1 wave": "tab:blue",
              "4x64": "tab:orange", "swap, 4x64": "tab:orange", "1 wave, pinned": "tab:cyan",
              "1 wave, wpe 1": "tab:olive", "4 waves, replicated": "tab:purple"}
    for title, items in ALT_GROUPS:
        start = x
        for lab, tile, pre in items:
            t = data[pre].get(tile)
            if t is None:
                continue
            v = float(t[0]["tflops"])
            ax1.bar(x, v, color=colors.get(lab, "tab:red"), width=0.8)
            ax1.text(x, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
            ax1.text(x, -0.02, lab, rotation=60, ha="right", va="top", fontsize=7, transform=ax1.get_xaxis_transform())
            x += 1
        ticks.append((start + x - 1) / 2)
        labels.append(title)
        x += 0.8
    ax1.set_yscale("log")
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.tick_params(axis="x", pad=70)
    ax1.set_ylabel("useful TFLOPS (real M x N x K), back to back")
    ax1.set_title("Small-tile alternatives, 2048x4096x8192 (BN = 4, 8: 2048x256x8192); red: Triton tl.dot, stock",
                  fontsize=10)

    tiles = ("16x64x64", "16x32x128", "16x16x128", "32x256x64")
    modes = (("", "natural", "tab:blue"), ("p", "LDS pin", "tab:cyan"), ("e1", "waves_per_eu 1", "tab:olive"),
             ("e2", "waves_per_eu 2", "tab:gray"))
    for i, t in enumerate(tiles):
        for j, (suf, lab, c) in enumerate(modes):
            r = data["alt"].get(t + suf)
            if r is None:
                continue
            v = float(r[0]["tflops"])
            occ = r[1].get("waves_per_simd") if r[1] else None
            ax2.bar(i * 5 + j, v, color=c, width=0.85, label=lab if i == 0 else None)
            ax2.text(i * 5 + j, v, f"{v:.0f}\n{float(occ):.1f}" if occ else f"{v:.0f}", ha="center", va="bottom",
                     fontsize=6.5)
    ax2.set_xticks([i * 5 + 1.5 for i in range(len(tiles))])
    ax2.set_xticklabels(tiles, fontsize=9)
    ax2.set_ylabel("useful TFLOPS")
    ax2.set_title("LDS pin against waves_per_eu\n(labels: TFLOPS, waves per SIMD)", fontsize=10)
    ax2.set_ylim(0, ax2.get_ylim()[1] * 1.3)
    ax2.legend(fontsize=8, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "alternatives.png"), dpi=130)


if __name__ == "__main__":
    {"cb": cb_table, "mb": mb_table, "warm": warm_table, "plots": plots, "alt": alt_table, "cycles": mfma_cycles}[
        sys.argv[1] if len(sys.argv) > 1 else "cb"]()
