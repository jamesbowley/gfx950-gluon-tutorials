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
"""results/*.csv and results/att_*.json -> images/*.png and results/summary.md.

    python plot.py
"""

import collections
import csv
import json
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
IMG = os.path.join(HERE, "images")
TIERS = ["L1", "L2", "MALL", "HBM"]
COLORS = dict(L1="#767676", L2="#3b7dd8", MALL="#4c9a5a", HBM="#c8443c")
TCP_BPC = 64


def rows(name):
    return list(csv.DictReader(open(os.path.join(RES, f"{name}.csv"))))


def med(rs, key):
    return st.median(float(r[key]) for r in rs)


def unloaded():
    """Median over reps of each tier's p10/p50/p90, one wave (num_warps=1)."""
    g = collections.defaultdict(list)
    for r in rows("lat"):
        if r["num_warps"] == "1":
            g[r["tier"]].append(r)
    return {t: {k: med(g[t], k) for k in ("p10_cyc", "p50_cyc", "p90_cyc", "p50_ns", "local_ghz")} for t in TIERS}


def fig_tiers(u):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for i, t in enumerate(TIERS):
        d = u[t]
        ax.bar(i, d["p50_cyc"], color=COLORS[t], alpha=0.8)
        ax.errorbar(i, d["p50_cyc"], yerr=[[d["p50_cyc"] - d["p10_cyc"]], [d["p90_cyc"] - d["p50_cyc"]]], color="black",
                    capsize=4, lw=1)
        ax.text(i, d["p90_cyc"] + 25, f"{d['p50_cyc']:.0f} cyc\n{d['p50_ns']:.0f} ns", ha="center", fontsize=9)
    ax.axhline(1000, color="black", ls="--", lw=0.8)
    ax.text(3.45, 1010, "branch: L = 1000", fontsize=8, ha="right", va="bottom")
    ax.set_xticks(range(4), TIERS)
    ax.set_ylim(0, 1350)
    ax.set_ylabel("issue to data, cycles")
    ax.set_title(f"One 1 KB buffer_load in flight, idle GPU, {u['HBM']['local_ghz']:.2f} GHz (median, 10-90%)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "latency_tiers.png"), dpi=130)
    plt.close(fig)


def loaded():
    g = collections.defaultdict(list)
    for r in rows("loaded"):
        g[(r["stream"], int(r["W"]), r["tier"])].append(r)
    out = {}
    for (s, W, t), rs in g.items():
        out[(s, W, t)] = dict(tbs=med(rs, "stream_gbps") / 1000 if rs[0]["stream_gbps"] else 0.0,
                              p50_cyc=med(rs, "p50_cyc"), p90_cyc=med(rs, "p90_cyc"), p50_ns=med(rs, "p50_ns"),
                              ghz=med(rs, "local_ghz"))
    return out


def fig_loaded(ld):
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, stream, tiers, title in [
        (axs[0], "hbm", ["HBM", "L1"], "W CUs streaming from HBM (1 GB)"),
        (axs[1], "l2", ["L2", "MALL", "HBM", "L1"], "W CUs streaming from L2 (64 KB each)"),
    ]:
        for t in tiers:
            pts = sorted((v["tbs"], v["p50_cyc"], v["p90_cyc"], W) for (s, W, tt), v in ld.items()
                         if s == stream and tt == t)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=COLORS[t], label=f"{t} probe, median")
            ax.plot([p[0] for p in pts], [p[2] for p in pts], ":", color=COLORS[t], lw=1)
            for p in pts:
                if t == tiers[0]:
                    ax.annotate(f"W={p[3]}", (p[0], p[1]), textcoords="offset points", xytext=(4, -10), fontsize=7)
        ax.axhline(1000, color="black", ls="--", lw=0.8)
        ax.set_xlabel("stream bandwidth, TB/s (chip)")
        ax.set_ylabel("probe load, cycles (dotted: 90th percentile)")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("Latency of one probe load while other CUs stream (the HBM stream also evicts the probe's L2 and MALL "
                 "data, so only its HBM and L1 tiers are shown)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "latency_loaded.png"), dpi=130)
    plt.close(fig)


def clock_rows():
    return [r for r in rows("clock")]


def fig_clock(cr):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for t in TIERS:
        pts = [(float(r["local_ghz"]), float(r["p50_cyc"]), float(r["p50_ns"])) for r in cr if r["tier"] == t]
        axs[0].plot([p[0] for p in pts], [p[1] for p in pts], "o", color=COLORS[t], label=t, ms=4)
        axs[1].plot([p[0] for p in pts], [p[2] for p in pts], "o", color=COLORS[t], label=t, ms=4)
    axs[0].set_ylabel("median, cycles")
    axs[1].set_ylabel("median, ns")
    for ax in axs:
        ax.set_xlabel("shader clock around the probe loads, GHz")
        ax.legend(fontsize=8)
    fig.suptitle("Probe latency at the clocks MFMA-only co-runners cause (0-255 CUs). L1 is fixed in cycles; the rest "
                 "moves less in ns than in cycles, but the clock only spans 2.1-2.4 GHz", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "latency_clock.png"), dpi=130)
    plt.close(fig)


def inflight():
    """Valid points only: every workgroup of the CU ran concurrently, and no wpc = 5 (its fifth workgroup does not
    fit the LDS and lands on another CU with the same HW_ID key: one 'CU' then reads more than 64 B/clk from L2)."""
    out = []
    for r in rows("inflight"):
        wpc = int(r["wpc"])
        if wpc == 5 or float(r["conc_per_cu"]) < 0.9 * wpc:
            continue
        out.append(dict(scope=r["scope"], tier=r["tier"], kb=int(r["kb_inflight"]), bpc=float(r["bpc_per_cu"]),
                        gbps=float(r["gbps_per_cu"]), ghz=float(r["sclk_mhz"]) / 1000))
    return out


def inflight_fit(pts, scope, tier):
    p = [x for x in pts if x["scope"] == scope and x["tier"] == tier]
    by = collections.defaultdict(list)
    for x in p:
        by[x["kb"]].append(x["bpc"])
    curve = sorted((kb, max(v)) for kb, v in by.items())
    plateau = st.median(b for kb, b in curve if kb >= 64)
    # Round trip from the linear part: B/clk = KB in flight / L, from the points well under the plateau
    lin = [(kb, b) for kb, b in curve if b < 0.6 * plateau]
    L = st.median(kb * 1024 / b for kb, b in lin) if lin else None
    ghz = st.median(x["ghz"] for x in p)
    return dict(curve=curve, plateau=plateau, L_lin=L, ghz=ghz, gbps=plateau * ghz)


def fig_inflight(pts, u):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, scope in zip(axs, ["1cu", "256cu"]):
        for t in ["L2", "MALL", "HBM"]:
            f = inflight_fit(pts, scope, t)
            ax.plot([c[0] for c in f["curve"]], [c[1] for c in f["curve"]], "o-", color=COLORS[t], ms=3,
                    label=f"{t}: plateau {f['plateau']:.1f} B/clk ({f['gbps']:.0f} GB/s)")
            if scope == "1cu":
                L = u[t]["p50_cyc"]
                xs = [0, 140]
                ax.plot(xs, [x * 1024 / L for x in xs], ":", color=COLORS[t], lw=1)
        for kb in (32, 44):
            ax.axvline(kb, color="grey", lw=0.8, ls="--")
            ax.text(kb, 66, f"{kb} KB", fontsize=8, ha="center")
        ax.axhline(TCP_BPC, color="black", lw=0.8)
        ax.set_xlim(0, 132)
        ax.set_ylim(0, 70)
        ax.set_xlabel("nominal KB in flight per CU (2 x waves x unroll x 1 KB)")
        ax.set_ylabel("B/clk per CU")
        ax.set_title("one CU (dotted: KB / unloaded probe L)" if scope == "1cu" else "all 256 CUs", fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Bandwidth per CU against loads in flight. Black: the TCP's 64 B/clk processing rate", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "inflight.png"), dpi=130)
    plt.close(fig)


ATT_ORDER = ["stream_8_4_HBM_xcd0", "stream_16_4_HBM_xcd0", "stream_16_4_MALL_xcd0", "stream_16_4_HBM_all",
             "stream_4_4_HBM_xcd0", "stream_4_2_HBM_xcd0", "v7_K8192", "v7_K16384", "v8_K8192", "v8_K16384",
             "v7_K8192_sep29", "v8_K8192_sep29"]


def att():
    out = {}
    for n in ATT_ORDER:
        p = os.path.join(RES, f"att_{n}.json")
        if os.path.exists(p):
            out[n] = json.load(open(p))
    return out


def fig_att(a):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, names, title in [
        (axs[0], [n for n in a if n.startswith("stream") and a[n]["stretched_loads_per_cu"] > 0.2 * a[n]["loads_per_cu"]],
         "stream kernel, slot-bound (stretched loads > 20%)"),
        (axs[1], [n for n in a if n.startswith("v")], "a16w16 v7 / v8 hot loops"),
    ]:
        for n in names:
            dk = {int(k): v for k, v in a[n]["dk_median"].items()}
            ks = sorted(dk)
            line, = ax.plot(ks, [dk[k] for k in ks], label=n)
            L = a[n]["L_exact"]["p50"]
            if L and a[n]["L_exact"]["n"] >= 50:
                ax.axhline(L, color=line.get_color(), ls=":", lw=1)
                if a[n]["k_match"]:
                    ax.plot([a[n]["k_match"]], [L], "o", color=line.get_color())
        for kb in (32, 44):
            ax.axvline(kb, color="grey", lw=0.8, ls="--")
        ax.set_xlabel("k: loads issued earlier on the same CU")
        ax.set_ylabel("median issue(n) - issue(n-k), cycles")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)
    fig.suptitle("A buffer_load that waits for a slot issues when the load k earlier returns: D_k = L at the true k "
                 "(dotted: each trace's median exact L; dot: the crossing)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "att_dk.png"), dpi=130)
    plt.close(fig)


def summary(u, ld, cr, pts, a):
    L = []
    L.append("# Summary (generated by plot.py)\n")
    L.append("## Unloaded latency, one 1 KB load in flight\n")
    L.append("| tier | p10 cyc | p50 cyc | p90 cyc | p50 ns | clock GHz |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for t in TIERS:
        d = u[t]
        L.append(f"| {t} | {d['p10_cyc']:.0f} | {d['p50_cyc']:.0f} | {d['p90_cyc']:.0f} | {d['p50_ns']:.0f} | "
                 f"{d['local_ghz']:.2f} |")
    L.append("\n## Probe under load (median cycles; stream TB/s)\n")
    L.append("| stream | W | TB/s | HBM probe | L2 probe | MALL probe | clock GHz |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for s in ("hbm", "l2", "mfma"):
        for W in (0, 8, 32, 64, 128, 255):
            g = {t: ld.get((s, W, t)) for t in TIERS}
            if not g["HBM"]:
                continue
            ev = " (evicted)" if s == "hbm" and W > 0 else ""
            L.append(f"| {s} | {W} | {g['HBM']['tbs']:.1f} | {g['HBM']['p50_cyc']:.0f} | {g['L2']['p50_cyc']:.0f}{ev} | "
                     f"{g['MALL']['p50_cyc']:.0f}{ev} | {g['HBM']['ghz']:.2f} |")
    L.append("\n## Clock (MFMA co-runners, 0.1 s settle)\n")
    L.append("| W | tier | clock GHz | p50 cyc | p50 ns |")
    L.append("|---:|---|---:|---:|---:|")
    g = collections.defaultdict(list)
    for r in cr:
        g[(int(r["W"]), r["tier"])].append(r)
    for (W, t) in sorted(g):
        rs = g[(W, t)]
        L.append(f"| {W} | {t} | {med(rs, 'local_ghz'):.2f} | {med(rs, 'p50_cyc'):.0f} | {med(rs, 'p50_ns'):.0f} |")
    L.append("\n## Loads in flight against bandwidth\n")
    L.append("| scope | tier | plateau B/clk | GB/s | L from the linear part, cyc | plateau x unloaded L, KB |")
    L.append("|---|---|---:|---:|---:|---:|")
    for scope in ("1cu", "256cu"):
        for t in ("L2", "MALL", "HBM"):
            f = inflight_fit(pts, scope, t)
            Ll = f"{f['L_lin']:.0f}" if f["L_lin"] else ""
            L.append(f"| {scope} | {t} | {f['plateau']:.1f} | {f['gbps']:.0f} | {Ll} | "
                     f"{f['plateau'] * u[t]['p50_cyc'] / 1024:.1f} |")
    L.append("\n## ATT\n")
    L.append("| trace | clock GHz | stretched loads | exact L n | exact L p50 cyc | ns | D_32 | D_44 | k where D_k = L |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for n, r in a.items():
        le = r["L_exact"]
        dk = r["dk_median"]
        L.append(f"| {n} | {r['ghz_realtime']} | {100 * r['stretched_loads_per_cu'] / r['loads_per_cu']:.0f}% | "
                 f"{le['n']} | {le['p50'] or ''} | {le['p50_ns'] or ''} | {dk.get('32', '')} | {dk.get('44', '')} | "
                 f"{r['k_match'] or ''} |")
    with open(os.path.join(RES, "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    os.makedirs(IMG, exist_ok=True)
    u = unloaded()
    fig_tiers(u)
    ld = loaded()
    fig_loaded(ld)
    cr = clock_rows()
    fig_clock(cr)
    pts = inflight()
    fig_inflight(pts, u)
    a = att()
    fig_att(a)
    summary(u, ld, cr, pts, a)


if __name__ == "__main__":
    main()
