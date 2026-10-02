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
"""Plots for results/*.csv produced by bench.py. Writes images/*.png."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
IMAGES = os.path.join(HERE, "images")
OPS = ["read", "write", "copy"]
PEAK_TBPS = 8.0


def load(name, results=RESULTS):
    path = os.path.join(results, f"{name}.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def save(fig, name):
    os.makedirs(IMAGES, exist_ok=True)
    path = os.path.join(IMAGES, f"{name}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def plot_A(df):
    for scaling in ["weak", "strong"]:
        d = df[df.sweep_tag == scaling]
        fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True)
        for j, op in enumerate(OPS):
            for mode, style in [("rr", "-"), ("packed", "--")]:
                s = d[(d.op == op) & (d["mode"] == mode)].sort_values("W")
                label = "round-robin" if mode == "rr" else "XCD-packed"
                axes[0, j].plot(s.W, s.gbps / 1000, style, label=label)
                axes[1, j].plot(s.W, s.gbps / s.W, style, label=label)
            p = df[(df.sweep_tag == scaling + "_plain") & (df.op == op)]
            if len(p):
                axes[0, j].plot(p.W, p.gbps / 1000, "k.", label="round-robin, plain grid of W")
            for x in range(32, 257, 32):
                axes[0, j].axvline(x, color="0.85", lw=0.8, zorder=0)
                axes[1, j].axvline(x, color="0.85", lw=0.8, zorder=0)
            axes[0, j].axhline(PEAK_TBPS, color="r", lw=0.8, ls=":", label="8 TB/s theoretical")
            axes[0, j].set_title(f"{op}")
            axes[0, j].set_ylabel("TB/s (read+write bytes)")
            axes[1, j].set_ylabel("GB/s per active CU")
            axes[1, j].set_xlabel("active workgroups (= CUs)")
            axes[0, j].legend(fontsize=8)
            axes[0, j].grid(alpha=0.3)
            axes[1, j].grid(alpha=0.3)
        what = "32 MB per workgroup" if scaling == "weak" else "2 GB total per buffer"
        fig.suptitle(f"Sweep A ({scaling} scaling, {what}): bandwidth vs active CUs, 8 waves x 4 x 16 B in flight")
        save(fig, f"A_{scaling}")

    # zoom on the split-k region
    d = df[df.sweep_tag == "strong"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for j, op in enumerate(OPS):
        for mode, style in [("rr", "o-"), ("packed", "s--")]:
            s = d[(d.op == op) & (d["mode"] == mode) & (d.W <= 96)].sort_values("W")
            axes[j].plot(s.W, s.time_ms, style, ms=3, label="round-robin" if mode == "rr" else "XCD-packed")
        axes[j].set_yscale("log")
        axes[j].set_title(f"{op}: time for 2 GB (lower is better)")
        axes[j].set_xlabel("active workgroups")
        axes[j].set_ylabel("ms")
        axes[j].grid(alpha=0.3, which="both")
        axes[j].legend()
    save(fig, "A_strong_time_zoom")


def plot_B(df):
    g = df[(df.sweep_tag == "grid") & (df.wpc == 1)]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for i, W in enumerate([32, 64]):
        for j, op in enumerate(OPS):
            s = g[(g.W == W) & (g.op == op)].pivot_table(index="num_warps", columns="unroll", values="gbps")
            ax = axes[i, j]
            im = ax.imshow(s.values / 1000, origin="lower", cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(s.columns)), s.columns)
            ax.set_yticks(range(len(s.index)), s.index)
            for (r, c), v in np.ndenumerate(s.values / 1000):
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", color="w" if v < s.values.max() / 1500 else "k",
                        fontsize=8)
            ax.set_xlabel("UNROLL (16 B loads per thread per tile)")
            ax.set_ylabel("waves per workgroup")
            ax.set_title(f"{op}, W={W} round-robin (TB/s)")
            fig.colorbar(im, ax=ax)
    fig.suptitle("Sweep B: bandwidth vs memory-level parallelism (1 GB per buffer, 1 workgroup per CU)")
    save(fig, "B_heatmap")

    h = df[df.sweep_tag.isin(["32wg_32cu", "64wg_32cu", "64wg_64cu", "32wg_32cu_2xwaves", "32wg_32cu_2xunroll"])]
    if not len(h):
        return
    tags = ["32wg_32cu", "32wg_32cu_2xwaves", "32wg_32cu_2xunroll", "64wg_32cu", "64wg_64cu"]
    labels = {
        "32wg_32cu": "32 WG on 32 CUs",
        "32wg_32cu_2xwaves": "32 WG on 32 CUs, 2x waves",
        "32wg_32cu_2xunroll": "32 WG on 32 CUs, 2x unroll",
        "64wg_32cu": "64 WG on 32 CUs (2 per CU)",
        "64wg_64cu": "64 WG on 64 CUs",
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), sharey=True)
    bases = h[["base_warps", "base_unroll"]].drop_duplicates().sort_values(["base_unroll", "base_warps"]).values
    for j, op in enumerate(OPS):
        ax = axes[j]
        width = 0.16
        for k, tag in enumerate(tags):
            vals = []
            for nw, u in bases:
                r = h[(h.op == op) & (h.sweep_tag == tag) & (h.base_warps == nw) & (h.base_unroll == u)]
                vals.append(r.gbps.iloc[0] / 1000 if len(r) else np.nan)
            ax.bar(np.arange(len(bases)) + (k - 2) * width, vals, width, label=labels[tag])
        ax.set_xticks(range(len(bases)), [f"{nw}w x u{u}" for nw, u in bases], rotation=30)
        ax.set_title(f"{op}")
        ax.set_ylabel("TB/s")
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8)
    fig.suptitle("Sweep B: explicit CU placement (hardware-id selected CUs, 4 or 8 per XCD). "
                 "x axis: base waves per WG x UNROLL")
    save(fig, "B_placement")


def plot_C(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    for j, op in enumerate(OPS):
        ax = axes[j]
        for W, color in [(32, "C0"), (64, "C1"), (256, "C2")]:
            for tag, style in [("cold", "-"), ("warm", "--")]:
                s = df[(df.op == op) & (df.W == W) & (df.sweep_tag == tag)].sort_values("size_mb")
                ax.plot(s.size_mb, s.gbps_in_kernel / 1000, style, color=color, marker=".", label=f"W={W} {tag}")
        ax.axvline(32, color="0.6", ls=":", lw=1)
        ax.axvline(256, color="0.3", ls=":", lw=1)
        ax.text(32, ax.get_ylim()[1] * 0.95, " 8x4 MB L2", fontsize=8)
        ax.text(256, ax.get_ylim()[1] * 0.95, " 256 MB MALL", fontsize=8)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("working set per buffer (MB)")
        ax.set_ylabel("TB/s (in-kernel timestamps)")
        ax.set_title(op)
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Sweep C: working-set size (cold = L2/MALL flushed before each run, warm = back-to-back). "
                 "In-kernel time, so launch overhead is excluded")
    save(fig, "C_working_set")


def plot_D(df):
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.8), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.2]))
    for j, op in enumerate(OPS):
        d = df[df.op == op]
        m = np.full((8, 8), np.nan)
        for _, r in d[d.n_sub <= 2].iterrows():
            xs = [int(x) for x in str(r.xcds).split("-")]
            if len(xs) == 1:
                m[xs[0], xs[0]] = r.gbps / 1000
            else:
                m[xs[0], xs[1]] = m[xs[1], xs[0]] = r.gbps / 1000
        ax = axes[j]
        im = ax.imshow(m, cmap="viridis")
        for (a, b), v in np.ndenumerate(m):
            ax.text(b, a, f"{v:.2f}", ha="center", va="center", fontsize=7, color="w")
        ax.set_title(f"{op}: 1 XCD (diag) / XCD pairs, TB/s")
        ax.set_xlabel("XCD")
        ax.set_ylabel("XCD")
        fig.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[3]
    q = df[df.n_sub >= 4]
    subs = q.xcds.drop_duplicates().tolist()
    for k, op in enumerate(OPS):
        vals = [q[(q.op == op) & (q.xcds == s)].gbps.iloc[0] / 1000 for s in subs]
        ax.bar(np.arange(len(subs)) + (k - 1) * 0.27, vals, 0.27, label=op)
    ax.set_xticks(range(len(subs)), subs, rotation=30)
    ax.set_ylabel("TB/s")
    ax.set_title("4 and 8 XCDs (32 CUs each)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Sweep D: XCD subsets, all 32 CUs of each selected XCD, 32 MB per workgroup")
    save(fig, "D_xcd_subsets")


def plot_E(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    for j, op in enumerate(OPS):
        ax = axes[j]
        for pat, style in [("chunked", "o-"), ("grid", "s--")]:
            s = df[(df.op == op) & (df.sweep_tag == pat)].sort_values("W")
            ax.plot(s.W, s.gbps / s.W, style, label=pat)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("active workgroups (round-robin)")
        ax.set_ylabel("GB/s per CU")
        ax.set_title(op)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Sweep E: contiguous chunk per workgroup vs grid-stride")
    save(fig, "E_access_pattern")


def plot_F(df):
    rs = sorted(df.read_mb_per_tile.unique())
    fig, axes = plt.subplots(1, len(rs), figsize=(5 * len(rs), 4.5))
    for j, r_mb in enumerate(rs):
        ax = axes[j]
        d = df[df.read_mb_per_tile == r_mb]
        base = d[d.splits == 1].time_ms.iloc[0]
        ax.axhline(base * 1e3, color="k", lw=1, label="S=1 (32 WG, direct write)")
        for var, style in [("twopass", "o-"), ("atomic", "s--")]:
            s = d[d.sweep_tag == var].sort_values("splits")
            s = pd.concat([d[d.splits == 1], s])
            ax.plot(s.splits, s.time_ms * 1e3, style, label=var)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
        ax.set_xlabel("split factor S (grid = 32 x S)")
        ax.set_ylabel("end-to-end time (us)")
        ax.set_title(f"{r_mb} MB read per output tile ({32 * r_mb} MB total)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Sweep F: split-k proxy, 32 output tiles of 256 KB fp32")
    save(fig, "F_splitk")


def _tier_markers(ax, labels=True):
    # per-CU footprint at which each level's capacity is exhausted (256 CUs, 32 per XCD)
    for x, name in [(32, "32 KB L1"), (128, "4 MB L2 per XCD"), (1024, "256 MB MALL")]:
        ax.axvline(x, color="0.5", ls=":", lw=1)
        if labels:
            ax.text(x * 1.05, ax.get_ylim()[1] * 0.92, name, fontsize=8, color="0.3")


def plot_G(df):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for op, style in [("read", "o-"), ("write", "s--")]:
        s = df[df.op == op].sort_values("footprint_per_wg_kb")
        ax.plot(s.footprint_per_wg_kb, s.slope_gbps / 1000, style, label=f"{op} (steady state, P vs 4P passes)")
    ax.axhline(PEAK_TBPS, color="r", lw=0.8, ls=":", label="8 TB/s HBM theoretical")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_yticks([4, 8, 16, 32], ["4", "8", "16", "32"])
    ax.set_xlabel("footprint per CU (KB); total = 256 x this")
    ax.set_ylabel("TB/s")
    ax.grid(alpha=0.3, which="both")
    _tier_markers(ax)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("Sweep G: steady-state bandwidth vs footprint, 256 CUs, 4 waves x UNROLL=4 (GPU 6)")
    save(fig, "G_tiers")


def plot_H(h, d):
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.8), gridspec_kw=dict(width_ratios=[2, 1]))
    sub = h[h.sweep_tag.str.startswith("xcds_")]
    cold = {} if d is None else dict(d[d.op == "read"][["xcds", "gbps"]].values)
    x = np.arange(len(sub))
    ax = axes[0]
    ax.bar(x - 0.2, sub.slope_gbps / 1000, 0.4, label="MALL hits (512 KB per CU, steady state)")
    ax.bar(x + 0.2, [cold.get(k, np.nan) / 1000 for k in sub.xcds], 0.4, label="cold HBM (sweep D)")
    for xi, v in zip(x, sub.slope_gbps / 1000):
        ax.text(xi - 0.2, v + 0.05, f"{v:.2f}", ha="center", fontsize=7, rotation=90)
    ax.set_xticks(x, sub.xcds, rotation=30)
    ax.set_ylabel("TB/s")
    ax.set_title("XCD subsets, 32 CUs per XCD")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    ax = axes[1]
    c = h[h.sweep_tag.str.startswith("all8_")].sort_values("cus_per_xcd")
    full = h[h.sweep_tag == "xcds_0-1-2-3-4-5-6-7"]
    c = pd.concat([c, full.assign(cus_per_xcd=32)])
    ax.plot(c.cus_per_xcd, c.slope_gbps / 1000, "o-", label="all 8 XCDs, 16 MB per XCD")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
    ax.set_xlabel("CUs per XCD")
    ax.set_ylabel("TB/s")
    ax.set_title("MALL hits vs CUs per XCD")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle("Sweep H: MALL-hit bandwidth is capped by the same fabric limits as HBM traffic (GPU 6)")
    save(fig, "H_mall_placement")


def plot_I(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, name, n_xcd in [(axes[0], "x0", 1), (axes[1], "all8", 8)]:
        s = df[df.sweep_tag.str.startswith(name + "_")]
        for cfg in ["4w_u4", "8w_u4", "16w_u4"]:
            t = s[s.sweep_tag.str.endswith(cfg)].sort_values("cus_per_xcd")
            ax.plot(t.cus_per_xcd, t.slope_gbps / 1000, "o-", label=cfg.replace("_", " x "))
        cus = np.array([1, 2, 4, 8, 16, 32])
        ax.plot(cus, cus * n_xcd * 64 * 2.4e9 / 1e12, "k:", lw=1, label="64 B/clk per CU at 2.4 GHz")
        ax.axhline(n_xcd * 2048 * 2.4e9 / 1e12, color="r", ls=":", lw=1, label="L2 peak 2 KB/clk per XCD")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xticks(cus, [str(c) for c in cus])
        ax.set_xlabel("CUs per XCD")
        ax.set_ylabel("TB/s")
        ax.set_title("one XCD" if n_xcd == 1 else "all 8 XCDs")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle("Sweep I: L2-hit bandwidth (64 KB per CU, 2 MB per XCD), steady state (GPU 6)")
    save(fig, "I_l2_scaling")


F_THEO, F_MEAS = 2.52e15, 1.56e15  # bf16 dense: 256 CUs x 4096 FLOP/clk x 2.4 GHz; best a16w16 GEMM
B_THEO = 8.0e12


def ridge_inputs(g, h, i):
    return dict(hbm=g[g.sweep_tag == "read_4096KB"].slope_gbps.iloc[0] * 1e9, mall=h.slope_gbps.max() * 1e9,
                l2=i.slope_gbps.max() * 1e9)


def plot_ridge(b):
    ai = np.logspace(0, 4, 400)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(ai, np.minimum(F_THEO, ai * B_THEO) / 1e12, "C0-", lw=2, label="theoretical: 2.52 PFLOPS, 8.0 TB/s")
    ax.loglog(ai, np.minimum(F_MEAS, ai * b["hbm"]) / 1e12, "C1-", lw=2,
              label=f"measured: 1.56 PFLOPS, HBM {b['hbm'] / 1e12:.2f} TB/s")
    ax.loglog(ai, np.minimum(F_MEAS, ai * b["mall"]) / 1e12, "C2--",
              label=f"measured, MALL hits {b['mall'] / 1e12:.2f} TB/s")
    ax.loglog(ai, np.minimum(F_MEAS, ai * b["l2"]) / 1e12, "C3--", label=f"measured, L2 hits {b['l2'] / 1e12:.1f} TB/s")
    lo, hi = F_MEAS / B_THEO, F_THEO / b["hbm"]
    ax.axvspan(lo, hi, color="0.85", zorder=0, label=f"HBM ridge band {lo:.0f}-{hi:.0f} FLOP/B")
    for x, c in [(F_THEO / B_THEO, "C0"), (F_MEAS / b["hbm"], "C1")]:
        ax.axvline(x, color=c, ls=":", lw=1)
    ax.set_xlabel("arithmetic intensity (FLOP/byte at the named boundary)")
    ax.set_ylabel("TFLOPS")
    ax.set_ylim(10, 4000)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Ridge band, bf16 dense (MI355X, NPS1)")
    save(fig, "ridge_band")


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--tiers-dir", default=os.path.join(RESULTS, "gpu6"), help="CSVs for sweeps G-J (and their D)")
    args = p.parse_args()
    for name, fn in [("A", plot_A), ("B", plot_B), ("C", plot_C), ("D", plot_D), ("E", plot_E), ("F", plot_F)]:
        df = load(name)
        if df is not None:
            fn(df)
    g, h, i = (load(n, args.tiers_dir) for n in "GHI")
    if g is not None:
        plot_G(g)
    if h is not None:
        plot_H(h, load("D", args.tiers_dir))
    if i is not None:
        plot_I(i)
    if g is not None and h is not None and i is not None:
        b = ridge_inputs(g, h, i)
        plot_ridge(b)
        for name, bw in b.items():
            print(f"ridge vs {name} {bw / 1e12:.2f} TB/s: theo/meas {F_THEO / bw:.0f}, meas/meas {F_MEAS / bw:.0f}")
        print(f"HBM band: {F_MEAS / B_THEO:.0f} (meas/theo) .. {F_THEO / b['hbm']:.0f} (theo/meas), "
              f"theo/theo {F_THEO / B_THEO:.0f}")


if __name__ == "__main__":
    main()
