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
"""Per-tile load latency and MFMA efficiency (WORKLOG §16): LAT_FW and LAT_CB against the window model.

    python v9_tile/report_latency.py           # results/latency_report.md
    python v9_tile/report_latency.py plots     # images/latency/*.png

Per point (sweep.py's LAT_FW / LAT_CB runs, results/lat_{fw,cb}_*.csv, ATT under tmp/latency/{fw,cb}):

- Latency: L_tcp, the mean TCP->L2 round trip of a 128 B request (TCP_TCC_READ_REQ_LATENCY), and
  L_issue, the mean VMEM instruction round trip from issue (VmemLatency), both in ns at the counter
  run's clock. L_miss (TCC_EA0_RDREQ_LEVEL / RDREQ) reads 1.5-2x the miss time implied on the TCP side
  in calib_latency.py, so it is kept as a relative number.
- In flight per CU (Little's law on the same counters): KB of 128 B requests between the TCP and L2,
  and KB of 1 KB VMEM instructions issued and not returned. The calibration's one CU reading HBM holds
  45.7 KB from issue (the ~44-entry cap of the tcp_latency README) and 20 KB TCP-side.
- Efficiency: ATT main-loop MFMA efficiency on CU 0, MFMA busy over the whole kernel (counters) and
  sustained useful utilisation (clock run).
- Model: the README §3 ceilings (model_small's formulas; the four-quadrant tiles' waves own (BM/2) x
  (BN/2)) at the clock run's clock: once with the documented window (Q = 32 KB, L = 550 ns) and once
  with the measured pair (Q = 45 KB from issue, L = L_issue). The whole kernel's busy CU share scales
  both.
- Two stages: the k-step time T_k from the timed run against L_issue (2 T_k must cover it), and the
  ATT `vmcnt` share of the loop's stalls.
- Old stack: the same point (or its mirror) from §15.4 (layout_ab, the derived layouts) or else §4,
  §7, §14.4 (Triton v2.2 + llirSched + amdgcnas, hand-built layouts).
"""

import csv
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sweep  # noqa: E402

RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "images", "latency")
ATT_DIR = {"fw": os.path.join(HERE, "tmp", "latency", "fw", "sweep_att"),
           "cb": os.path.join(HERE, "tmp", "latency", "cb", "sweep_att")}
PRESETS = {"fw": "one full wave per tile, K = 8192", "cb": "2048x4096x8192"}
Q_DOC, L_DOC = 32768, 550.0
Q_ISSUE = 45 * 1024  # calib_latency.py: one CU reading HBM, KB in flight from issue
L2_HIT_CYC = 110  # calib_latency.py: TCP-side round trip of an L2 hit, unloaded
OLD = [("§15.4", "layout_ab", "d"), ("§15.4", "layout_ab", ""), ("§4", "a", ""), ("§7", "a128", ""),
       ("§14.4", "small_cb", "")]


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return {}
    return {(r["tile"], int(r["M"]), int(r["N"]), int(r["K"])): r for r in csv.DictReader(open(p))}


def f(r, k, default=None):
    v = (r or {}).get(k, "")
    return float(v) if v not in ("", None) else default


def kernel_tile(tile):
    """(km, kn, bk, W, small): the kernel's tile, the small side as M for v9_tile_small."""
    bm, bn, bk, _, w, _ = sweep.tile_spec(tile)
    if sweep.is_small(tile):
        return min(bm, bn), max(bm, bn), bk, w, True
    return bm, bn, bk, 4, False


def ceilings(tile, L, Q, clock):
    """README §3 / model_small.ceilings with Q a parameter; the quadrant tiles' LDS reads per wave."""
    km, kn, bk, w, small = kernel_tile(tile)
    Mp, Np = max(16, km), max(16, kn)
    tiles_cu = sweep.wg_per_cu(tile)
    B = (km + kn) * bk * 2
    cyc = 2 * Mp * Np * bk / (1024 * w)
    C = cyc / clock
    t_issue = tiles_cu * B / 1024 * 16 / clock
    wm, wn = (Mp, Np // w) if small else (km // 2, kn // 2)
    share = w / 4
    r = {"pad": km * kn / (Mp * Np), "proc": min(1.0, 64 * cyc * share / B),
         "window": min(1.0, Q * C * share / (B * L)),
         "stages": min(1.0, 2 * tiles_cu * share * C / (L + t_issue)),
         "lds": min(1.0, wm * wn / (16 * (wm + wn)))}
    r["busy"] = min(r["proc"], r["window"], r["stages"], r["lds"])
    r["binds"] = min(("proc", "window", "stages", "lds"), key=lambda k: r[k])
    r.update(B=B, C_ns=C, tiles_cu=tiles_cu)
    return r


def stalls(preset, point):
    import att_stalls

    d = os.path.join(ATT_DIR[preset], point.replace(":", "_"))
    if not glob.glob(os.path.join(d, "**", "ui_*"), recursive=True):
        return None
    try:
        r = att_stalls.analyze(d)
    except Exception:
        return None
    s = {k: v[1] for k, v in r["by"].items()}
    return {"total": r["total_latency"], "vmcnt": s.get("s_waitcnt vmcnt", 0) + s.get("s_waitcnt vmcnt+lgkmcnt", 0),
            "issue": s.get("buffer_load", 0), "lds": s.get("s_waitcnt lgkmcnt", 0) + s.get("ds_read", 0),
            "barrier": s.get("s_barrier", 0), "stall": r["total_stall"]}


def old_point(tile, M, N, K):
    """(section, TFLOPS back to back, ATT loop eff) of the same point on an earlier run, or its mirror."""
    for sec, prefix, suf in OLD:
        timing, att = load(f"{prefix}_timing_graph.csv"), load(f"{prefix}_att.csv")
        for key in ((tile + suf, M, N, K), (tile + suf, N, M, K)):
            if key in timing:
                return sec + (" (mirror)" if key[1] != M else ""), f(timing[key], "tflops"), f(att.get(key), "att_mfma_eff")
    return None, None, None


def rows(preset):
    timing, clock, counters, att = (load(f"lat_{preset}_{s}.csv") for s in ("timing_graph", "clock", "counters", "att"))
    out = []
    for key, t in timing.items():
        tile, M, N, K = key
        km, kn, bk, w, small = kernel_tile(tile)
        c, n, a = clock.get(key), counters.get(key), att.get(key)
        ghz = f(c, "sclk_ghz", 2.1)
        us = f(t, "us_median")
        tiles = int(t["tiles"])
        tiles_cu = sweep.wg_per_cu(tile)
        slots = 256 * tiles_cu
        waves = -(-tiles // slots)
        busy_frac = tiles / waves / slots
        cus = sweep.loaded_cus(tiles)
        bytes_loaded = tiles * (K // bk) * (km + kn) * bk * 2
        lat_ghz = f(n, "lat_clock_ghz")
        l_issue = f(n, "l_vmem_ns")
        m_doc = ceilings(tile, L_DOC, Q_DOC, ghz)
        m_own = ceilings(tile, l_issue, Q_ISSUE, ghz) if l_issue else None
        t_k = us * 1e3 / (waves * K / bk)
        st = stalls(preset, f"{tile}:{M}x{N}x{K}")
        sec, old_tf, old_att = old_point(tile, M, N, K)
        h = f(n, "l2_hit")
        l_tcp_cyc = f(n, "l_tcp_cyc")
        out.append({
            "preset": preset, "tile": tile, "shape": f"{M}x{N}x{K}", "tiles": tiles, "tiles_cu": tiles_cu,
            "tflops": f(c, "sustained_tflops", f(t, "tflops")), "tflops_graph": f(t, "tflops"), "ghz": ghz,
            "power": f(c, "power_w"), "useful": f(c, "mfma_util"), "busy": f(n, "mfma_busy"),
            "att": f(a, "att_mfma_eff"), "bns": bytes_loaded / (us * 1e3) / cus, "l2_hit": h,
            "l_implied": Q_DOC / (bytes_loaded / (us * 1e3) / cus),
            "l_tcp_cyc": l_tcp_cyc, "l_tcp_ns": f(n, "l_tcp_ns"), "l_issue_cyc": f(n, "l_vmem_cyc"), "l_issue": l_issue,
            "l_miss_cyc": f(n, "l_miss_cyc"), "lat_ghz": lat_ghz, "pmc_ghz": f(n, "pmc_clock_ghz"),
            "inflight_tcp": f(n, "inflight_tcp_kb"), "inflight_issue": f(n, "inflight_vmem_kb"),
            "pending": f(n, "tcp_pending_stall"),
            "hit_part_ns": h * L2_HIT_CYC / lat_ghz if h is not None and lat_ghz else None,
            "m_doc": m_doc["busy"] * busy_frac, "m_doc_binds": m_doc["binds"], "m_doc_window": m_doc["window"],
            "m_doc_stages": m_doc["stages"],
            "m_own": m_own["busy"] * busy_frac if m_own else None, "m_own_binds": m_own["binds"] if m_own else "",
            "m_own_stages": m_own["stages"] if m_own else None, "busy_frac": busy_frac, "pad": m_doc["pad"],
            "t_k": t_k, "lead": 2 * t_k / l_issue if l_issue else None, "stalls": st,
            "old_sec": sec, "old_tflops": old_tf, "old_att": old_att,
        })
    order = {t: i for i, t in enumerate(sweep.LAT_TILES)}
    return sorted(out, key=lambda r: order.get(r["tile"], 99))


def pct(x):
    return "" if x is None else f"{100 * x:.1f}%"


def num(x, fmt="{:.0f}"):
    return "" if x is None else fmt.format(x)


def report():
    lines = ["# Per-tile load latency and MFMA efficiency (WORKLOG §16)", "",
             "Generated by `python v9_tile/report_latency.py` from `results/lat_{fw,cb}_*.csv`; "
             "the counter calibration is `results/latency_calib.csv` (`calib_latency.py`).", ""]
    for p, title in PRESETS.items():
        rs = rows(p)
        if not rs:
            continue
        lines += [f"## {title}", "", "### Latency and bytes in flight", "",
                  "| tile | shape | tiles/CU | L2 hit | B/ns per CU (timed) | L inferred: 32 KB / (B/ns) | L_tcp cyc "
                  "| L_tcp ns | L_issue cyc | L_issue ns | L_miss cyc (rel.) | counter-run GHz | in flight TCP-side KB/CU "
                  "| in flight from issue KB/CU | TCP pending stall |",
                  "|" + "---|" * 15]
        for r in rs:
            lines.append(
                f"| {r['tile']} | {r['shape']} | {r['tiles_cu']} | {pct(r['l2_hit'])} | {num(r['bns'])} "
                f"| {num(r['l_implied'])} | {num(r['l_tcp_cyc'])} "
                f"| {num(r['l_tcp_ns'])} | {num(r['l_issue_cyc'])} | {num(r['l_issue'])} | {num(r['l_miss_cyc'])} "
                f"| {num(r['lat_ghz'], '{:.2f}')} | {num(r['inflight_tcp'], '{:.1f}')} "
                f"| {num(r['inflight_issue'], '{:.1f}')} | {pct(r['pending'])} |")
        lines += ["", "### MFMA efficiency against the model", "",
                  "| tile | TFLOPS (sustained) | GHz | W | useful | MFMA busy | ATT loop eff | model busy at 550 ns, 32 KB "
                  "(binds) | model busy at L_issue, 45 KB (binds) | old stack: TFLOPS, ATT loop | TFLOPS now / old |",
                  "|" + "---|" * 11]
        for r in rs:
            ratio = r["tflops_graph"] / r["old_tflops"] if r["old_tflops"] else None
            flag = " **(-)**" if ratio is not None and ratio < 0.97 else ""
            old = (f"{num(r['old_tflops'])}, {pct(r['old_att'])} ({r['old_sec']})" if r["old_sec"] else "")
            lines.append(
                f"| {r['tile']} | {num(r['tflops'])} | {num(r['ghz'], '{:.2f}')} | {num(r['power'])} | {pct(r['useful'])} "
                f"| {pct(r['busy'])} | {pct(r['att'])} | {pct(r['m_doc'])} ({r['m_doc_binds']}) "
                f"| {pct(r['m_own'])} ({r['m_own_binds']}) | {old} | {num(ratio, '{:.3f}')}{flag} |")
        lines += ["", "### Two stages", "",
                  "| tile | T_k ns (timed) | 2 T_k / L_issue | ATT stalls per iteration: vmcnt / buffer_load issue / "
                  "lgkm+ds_read / barrier (of cycles issued) | vmcnt share of stalls | 2 stages enough |",
                  "|" + "---|" * 6]
        for r in rs:
            st = r["stalls"]
            if st:
                share = st["vmcnt"] / max(1.0, st["stall"])
                sp = (f"{st['vmcnt']:.0f} / {st['issue']:.0f} / {st['lds']:.0f} / {st['barrier']:.0f} "
                      f"({st['total']:.0f})")
            else:
                share, sp = None, ""
            ok = ""
            if share is not None and r["lead"] is not None:
                ok = "yes" if share < 0.1 and r["lead"] >= 1.0 else "borderline" if share < 0.2 else "no"
            lines.append(f"| {r['tile']} | {num(r['t_k'])} | {num(r['lead'], '{:.2f}')} | {sp} | {pct(share)} | {ok} |")
        lines.append("")
    open(os.path.join(RES, "latency_report.md"), "w").write("\n".join(lines))
    print("\n".join(lines))


def plots():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUT, exist_ok=True)
    data = {p: rows(p) for p in PRESETS}
    colors = {"fw": "tab:blue", "cb": "tab:orange"}

    # 1. L against tile: the TCP-side round trip split into the part an unloaded L2 hit explains and the
    # rest (misses and queueing), and the round trip from issue.
    fig, ax = plt.subplots(figsize=(13, 5))
    tiles = list(sweep.LAT_TILES)
    x = range(len(tiles))
    wbar = 0.2
    for j, p in enumerate(PRESETS):
        by = {r["tile"]: r for r in data[p]}
        off = (j - 0.5) * 2 * wbar
        hit = [by[t]["hit_part_ns"] if t in by else 0 for t in tiles]
        tcp = [by[t]["l_tcp_ns"] if t in by else 0 for t in tiles]
        iss = [by[t]["l_issue"] if t in by else 0 for t in tiles]
        ax.bar([i + off - wbar / 2 for i in x], hit, wbar, color=colors[p], alpha=0.9,
               label=f"{p}: L_tcp, hit part (h x unloaded L2 hit)")
        ax.bar([i + off - wbar / 2 for i in x], [a - b for a, b in zip(tcp, hit)], wbar, bottom=hit, color=colors[p],
               alpha=0.45, label=f"{p}: L_tcp, misses and queueing")
        ax.bar([i + off + wbar / 2 for i in x], iss, wbar, color=colors[p], alpha=0.2, edgecolor=colors[p],
               hatch="//", label=f"{p}: L_issue (from issue)")
    ax.axhline(L_DOC, color="k", ls="--", lw=1, label="550 ns (inferred, README)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(tiles, rotation=35, ha="right")
    ax.set_ylabel("mean round trip, ns (counter run's clock)")
    ax.set_title("Load latency per tile: TCP->L2 round trip and VMEM round trip from issue")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "latency_by_tile.png"), dpi=130)
    plt.close(fig)

    # 2. What moves L: the miss rate, and the bandwidth each CU pulls.
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for p in PRESETS:
        rs = [r for r in data[p] if r["l_tcp_ns"] is not None]
        for ax, xk, xl in ((axs[0], "miss", "L2 miss rate (1 - hit)"), (axs[1], "bns", "B/ns per CU (timed run)")):
            xs = [(1 - r["l2_hit"]) if xk == "miss" else r["bns"] for r in rs]
            ax.scatter(xs, [r["l_tcp_ns"] for r in rs], color=colors[p], label=f"{p}: L_tcp")
            ax.scatter(xs, [r["l_issue"] for r in rs], color=colors[p], marker="^", alpha=0.6, label=f"{p}: L_issue")
            for xv, r in zip(xs, rs):
                ax.annotate(r["tile"], (xv, r["l_tcp_ns"]), fontsize=6, xytext=(2, 2), textcoords="offset points")
            ax.set_xlabel(xl)
    for ax in axs:
        ax.axhline(L_DOC, color="k", ls="--", lw=1)
        ax.set_ylabel("ns")
        ax.legend(fontsize=7)
    axs[0].set_title("Round trip against the L2 miss rate")
    axs[1].set_title("Round trip against the bytes each CU pulls")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "latency_drivers.png"), dpi=130)
    plt.close(fig)

    # 3. Bytes in flight per CU against the two capacities.
    fig, ax = plt.subplots(figsize=(13, 4.5))
    for j, p in enumerate(PRESETS):
        by = {r["tile"]: r for r in data[p]}
        off = (j - 0.5) * 2 * wbar
        ax.bar([i + off - wbar / 2 for i in x], [by[t]["inflight_tcp"] or 0 if t in by else 0 for t in tiles], wbar,
               color=colors[p], alpha=0.5, label=f"{p}: TCP->L2 (128 B requests)")
        ax.bar([i + off + wbar / 2 for i in x], [by[t]["inflight_issue"] or 0 if t in by else 0 for t in tiles], wbar,
               color=colors[p], alpha=0.9, label=f"{p}: from issue (1 KB instructions)")
    ax.axhline(32, color="k", ls="--", lw=1, label="32 KB (README Q)")
    ax.axhline(45, color="r", ls=":", lw=1, label="45 KB (one CU from HBM, from issue)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(tiles, rotation=35, ha="right")
    ax.set_ylabel("mean KB in flight per CU")
    ax.set_title("Bytes in flight per CU (Little's law on the latency counters)")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "inflight.png"), dpi=130)
    plt.close(fig)

    # 4. Measured MFMA utilisation (the clock run: the model's clock) against the model at 550 ns and at
    # the tile's own L.
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mk, title in ((axs[0], "m_doc", "model: Q = 32 KB, L = 550 ns"),
                          (axs[1], "m_own", "model: Q = 45 KB, L = L_issue (measured)")):
        for p in PRESETS:
            rs = [r for r in data[p] if r[mk] is not None and r["useful"] is not None]
            ax.scatter([r[mk] for r in rs], [r["useful"] for r in rs], color=colors[p], label=p)
            for r in rs:
                ax.annotate(r["tile"], (r[mk], r["useful"]), fontsize=6, xytext=(2, 2), textcoords="offset points")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.04, 1.1)
        ax.set_ylim(0.04, 1.1)
        ax.set_xlabel("predicted MFMA busy (whole kernel)")
        ax.set_ylabel("measured MFMA utilisation (clock run, whole kernel)")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "model.png"), dpi=130)
    plt.close(fig)
    print(f"wrote {OUT}/latency_by_tile.png, latency_drivers.png, inflight.png, model.png")


if __name__ == "__main__":
    if sys.argv[1:] == ["plots"]:
        plots()
    else:
        report()
