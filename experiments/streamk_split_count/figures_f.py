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
"""Figures for sweep F (no GPU), from results/sweep_f.jsonl.

    python experiments/streamk_split_count/figures_f.py     # images/sweep_f_*.png

- sweep_f_winners.png: per (S, K), the configuration with the shortest tail and its speed-up over
  data-parallel (v13), allowing only single-rounding configurations (left) or every one (right).
- sweep_f_ratios.png: one panel per comparison, the tail ratio A / B at each one's best n
  (diverging colours around 1; blue: A shorter).
- sweep_f_tail_vs_n_endings.png and sweep_f_tail_vs_n_placement.png: tail against programs per tile
  n, rows S and columns K, with data-parallel as a dashed line; the shaded span is the n at which
  the panel's fastest single-rounding configuration is within NEAR of its best (how flat the
  optimum is).
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm, TwoSlopeNorm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sweep  # noqa: E402

IMG = os.path.join(HERE, "images")
NEAR = 1.02
SHORT = {"v13": "dp", "v24 two_tile": "2T", "v20 rs": "v20 rs", "v24 owner": "ow", "v24 owner chunk": "ow ch",
         "v24 owner bf16_pairs": "ow bf16", "v24 rs": "rs", "v24 rs chunk": "rs ch", "v24 rs l2": "rs l2",
         "v24 rs bf16_pairs": "rs bf16", "v24 atomic_iz": "atom", "v24 atomic_iz chunk": "atom ch",
         "v24 spread": "spread"}
COMPARISONS = [
    ("v24 atomic_iz", "v24 rs bf16_pairs", "atomics / bf16 reduce-scatter"),
    ("v24 rs bf16_pairs", "v24 rs", "bf16 / fp32 partials, reduce-scatter"),
    ("v24 owner bf16_pairs", "v24 owner", "bf16 / fp32 partials, owner"),
    ("v24 rs chunk", "v24 rs", "chunk-major / tile-major, reduce-scatter"),
    ("v24 owner chunk", "v24 owner", "chunk-major / tile-major, owner"),
    ("v24 atomic_iz chunk", "v24 atomic_iz", "chunk-major / tile-major, atomics"),
    ("v24 rs l2", "v24 rs", "l2 + XCD-local / write-through, reduce-scatter"),
    ("v24 rs", "v20 rs", "v24 / v20 reduce-scatter"),
    ("v24 rs", "v24 owner", "reduce-scatter / owner (fp32)"),
]
ENDINGS = {"v24 owner": "#c0504d", "v24 owner bf16_pairs": "#e8a09e", "v24 rs": "#4c9a4c",
           "v24 rs bf16_pairs": "#9fd39f", "v24 atomic_iz": "#17becf", "v20 rs": "#888888"}
PLACEMENT = {"v24 owner": "#c0504d", "v24 owner chunk": "#7f2a28", "v24 rs": "#4c9a4c", "v24 rs chunk": "#1f5f1f",
             "v24 rs l2": "#d4a017", "v24 atomic_iz": "#17becf", "v24 atomic_iz chunk": "#0b6f7a"}
GRID_S = (1, 4, 16, 38, 128)


def load():
    rows = sweep.load("f")
    base = {r["K"]: r["us"] for r in rows if r["config"] == "v13 base"}
    tails = {}
    for r in rows:
        if r["config"] != "v13 base" and r["K"] in base:
            tails.setdefault((r["S"], r["K"], r["config"]), {})[r["n"]] = r["us"] - base[r["K"]]
    return tails


def best(tails, S, K, cfg):
    t = tails.get((S, K, cfg))
    if not t:
        return None
    n = min(t, key=t.get)
    return t[n], n


def cells(tails):
    Ss = sorted({S for S, _, _ in tails if S <= 128})
    Ks = sorted({K for S, K, _ in tails if S <= 128})
    return Ss, Ks


def winners(tails):
    Ss, Ks = cells(tails)
    single = ["v13", "v24 two_tile", *[c for c in sweep.SPLIT_F if c not in sweep.MULTI_ROUNDING_F]]
    every = ["v13", "v24 two_tile", *sweep.SPLIT_F]
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    for ax, cfgs, title in ((axes[0], single, "Single rounding (fp32 partials or no split)"),
                            (axes[1], every, "Any rounding (adds bf16 partials and atomics)")):
        grid = np.full((len(Ss), len(Ks)), np.nan)
        for i, S in enumerate(Ss):
            for j, K in enumerate(Ks):
                bs = {c: best(tails, S, K, c) for c in cfgs}
                bs = {c: b for c, b in bs.items() if b}
                dp = best(tails, S, K, "v13")
                if not bs or not dp:
                    continue
                c = min(bs, key=lambda x: bs[x][0])
                t, n = bs[c]
                grid[i, j] = dp[0] / t if t > 0 else np.nan
                nn = "" if n == "-" else (" d" if n == "default" else f" {n}")
                ax.text(j, i, f"{SHORT[c]}{nn}\n{grid[i, j]:.2f}x", ha="center", va="center", fontsize=7.5,
                        color="w" if grid[i, j] > 4 else "k")
        im = ax.imshow(grid, cmap="YlGn", aspect="auto", norm=LogNorm(vmin=1, vmax=16))
        ax.set_xticks(range(len(Ks)), [str(K) for K in Ks])
        ax.set_yticks(range(len(Ss)), [str(S) for S in Ss])
        ax.set_xlabel("K")
        ax.set_ylabel("leftover tiles S")
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.04, label="data-parallel tail / fastest tail")
    fig.suptitle("Sweep F: fastest tail per cell (configuration, programs per tile n; d = default) and its "
                 "speed-up over data-parallel", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(IMG, "sweep_f_winners.png")
    fig.savefig(out, dpi=120)
    return out


def ratios(tails):
    Ss, Ks = cells(tails)
    cols = 3
    rows = -(-len(COMPARISONS) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 5.4 * rows), squeeze=False)
    norm = TwoSlopeNorm(vmin=0.6, vcenter=1.0, vmax=1.5)
    for k, (a, b, title) in enumerate(COMPARISONS):
        ax = axes[k // cols][k % cols]
        grid = np.full((len(Ss), len(Ks)), np.nan)
        for i, S in enumerate(Ss):
            for j, K in enumerate(Ks):
                x, y = best(tails, S, K, a), best(tails, S, K, b)
                if x and y and x[0] > 0 and y[0] > 0:
                    grid[i, j] = x[0] / y[0]
                    ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=7.5)
        im = ax.imshow(np.clip(grid, 0.6, 1.5), cmap="RdBu_r", aspect="auto", norm=norm)
        ax.set_xticks(range(len(Ks)), [str(K) for K in Ks], fontsize=8)
        ax.set_yticks(range(len(Ss)), [str(S) for S in Ss], fontsize=8)
        ax.set_xlabel("K")
        ax.set_ylabel("S")
        vals = grid[~np.isnan(grid)]
        med = f"median {np.median(vals):.3f}, A shorter in {int((vals < 1).sum())} of {vals.size}" if vals.size else ""
        ax.set_title(f"{title}\n{med}", fontsize=10)
    for k in range(len(COMPARISONS), rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.colorbar(im, ax=axes, fraction=0.02, label="tail A / tail B (each at its best n)")
    fig.suptitle("Sweep F: tail ratios, each configuration at its best n (blue: the first is shorter)", fontsize=12)
    out = os.path.join(IMG, "sweep_f_ratios.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    return out


def tail_vs_n(tails, cfgs, name, title):
    _, Ks = cells(tails)
    Ks = [K for K in (1024, 4096, 8192, 16384) if K in Ks]
    ns = sweep.N_C
    single = [c for c in cfgs if c not in sweep.MULTI_ROUNDING_F]
    fig, axes = plt.subplots(len(GRID_S), len(Ks), figsize=(4.4 * len(Ks), 3.3 * len(GRID_S)), squeeze=False)
    for i, S in enumerate(GRID_S):
        for j, K in enumerate(Ks):
            ax = axes[i][j]
            top = 0
            dp = best(tails, S, K, "v13")
            if dp:
                ax.axhline(dp[0], color="0.3", ls=(0, (3, 2)), lw=1.2, label="data-parallel (v13)")
                top = dp[0]
            fastest = None
            for cfg, color in cfgs.items():
                t = tails.get((S, K, cfg), {})
                xs = [n for n in ns if n in t]
                if not xs:
                    continue
                ax.plot(xs, [t[n] for n in xs], marker="o", ms=3.5, lw=1.5, color=color, label=sweep_label(cfg))
                top = max(top, min(t[n] for n in xs) * 2.2)
                if cfg in single:
                    b = min(t[n] for n in xs)
                    if fastest is None or b < fastest[0]:
                        fastest = (b, [n for n in xs if t[n] <= NEAR * b], color)
            if fastest:
                lo, hi = min(fastest[1]), max(fastest[1])
                ax.axvspan(lo / 1.15, hi * 1.15, color=fastest[2], alpha=0.12, lw=0)
            ax.set_xscale("log", base=2)
            ax.set_xticks(ns, [str(n) for n in ns], fontsize=7)
            ax.set_xlim(0.8, 80)
            ax.set_ylim(0, top * 1.05 if top else 1)
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(f"K = {K} ({K // 64} k-steps per tile)", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"S = {S}: tail us")
            if i == len(GRID_S) - 1:
                ax.set_xlabel("programs per tile n")
    h, labels = [], []
    for ax in axes.flat:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in labels:
                h.append(hh)
                labels.append(ll)
    fig.legend(h, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle(f"{title}. Shaded: the n at which the panel's fastest single-rounding configuration is within "
                 f"{round(100 * (NEAR - 1))}% of its best", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    out = os.path.join(IMG, f"sweep_f_tail_vs_n_{name}.png")
    fig.savefig(out, dpi=110)
    return out


def sweep_label(cfg):
    return {"v24 owner": "owner (fp32)", "v24 owner bf16_pairs": "owner, bf16 partials", "v24 rs": "reduce-scatter (fp32)",
            "v24 rs bf16_pairs": "reduce-scatter, bf16 partials", "v24 atomic_iz": "atomic_iz (bf16 adds into C)",
            "v20 rs": "v20 reduce-scatter", "v24 owner chunk": "owner, chunk-major",
            "v24 rs chunk": "reduce-scatter, chunk-major", "v24 rs l2": "reduce-scatter, l2 + XCD-local",
            "v24 atomic_iz chunk": "atomic_iz, chunk-major"}.get(cfg, cfg)


def main():
    os.makedirs(IMG, exist_ok=True)
    tails = load()
    outs = [winners(tails), ratios(tails),
            tail_vs_n(tails, ENDINGS, "endings", "Sweep F: endings and partial types (tile-major placement)"),
            tail_vs_n(tails, PLACEMENT, "placement", "Sweep F: placement and exchange (fp32 partials)")]
    print("\n".join(outs))


if __name__ == "__main__":
    main()
