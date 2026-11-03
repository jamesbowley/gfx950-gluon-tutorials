"""Plots of the tile spread metrics (wave fill, target waves, target fill) from
results/all_points.csv, into images/:

    python experiments/tune_aiter_shapes/plot_fill.py

fill_sweep.png        the metrics against M at N=6288 for a few tiles (no timing data)
fill_by_version.png   target fill and extra waves of each version's best point per shape
fill_hbl_vs_ours.png  hipBLASLt's target fill against our best tile's, coloured by speed
fill_gap_vs_time.png  the target-fill gap to hipBLASLt against the time ratio
"""

import csv
import math
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "images")
P = 256
A_MAX = 224 * 384


def cdiv(a, b):
    return -(-a // b)


def spread(M, N, bm, bn):
    T = cdiv(M, bm) * cdiv(N, bn)
    W = cdiv(T, P)
    wt = max(1, cdiv(M * N, P * A_MAX))
    return dict(T=T, W=W, waves=T / P, fill=T / (P * W), wt=wt,
                target_fill=min(T / (P * wt), 1.0) * wt / W)


def sweep():
    N = 6288
    Ms = np.arange(128, 8193, 16)
    tiles = [((256, 256), "256x256 (ours)"), ((224, 320), "224x320 (hipBLASLt-like)"),
             ((128, 128), "128x128 (ours)"), ((64, 256), "64x256 (FlyDSL-like)")]
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for (bm, bn), label in tiles:
        s = [spread(M, N, bm, bn) for M in Ms]
        ax[0].plot(Ms, [x["waves"] for x in s], label=label)
        ax[1].plot(Ms, [x["W"] - x["wt"] for x in s], label=label, drawstyle="steps-post")
        ax[2].plot(Ms, [x["target_fill"] for x in s], label=label)
    wt = [spread(M, N, 1, 1)["wt"] for M in Ms]
    ax[0].plot(Ms, wt, "k--", lw=1, label="target waves")
    ax[0].set_ylabel("waves (tiles / 256)")
    ax[1].set_ylabel("extra waves")
    ax[2].set_ylabel("target fill")
    ax[2].set_xlabel(f"M (N = {N})")
    ax[0].legend(fontsize=8, ncol=2)
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle("Tile spread against M: waves needed, extra waves over target, target fill")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fill_sweep.png"), dpi=120)
    plt.close(fig)


def load():
    best = {}
    for r in csv.DictReader(open(os.path.join(HERE, "results", "all_points.csv"))):
        if r["best_in_version"] == "yes" or r["best_ours"] == "yes":
            d = best.setdefault(r["shape"], {})
            if r["best_in_version"] == "yes":
                d[r["version"]] = r
            if r["best_ours"] == "yes":
                d["ours"] = r
    return best


def by_version(best):
    vers = [("torch", "hipBLASLt"), ("v31", "v31"), ("v9", "v9"), ("flydsl", "FlyDSL")]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    data = [[float(d[v]["tile_target_fill"]) for d in best.values() if v in d] for v, _ in vers]
    ax[0].hist(data, bins=np.linspace(0, 1, 21), label=[n for _, n in vers])
    ax[0].set_xlabel("target fill of the version's best point")
    ax[0].set_ylabel("shapes")
    ax[0].legend()
    extra = [[int(d[v]["tile_extra_waves"]) for d in best.values() if v in d] for v, _ in vers]
    top = max(max(e) for e in extra)
    ax[1].hist(extra, bins=np.arange(-0.5, top + 1.5), label=[n for _, n in vers])
    ax[1].set_xlabel("extra waves over target")
    ax[1].set_ylabel("shapes")
    ax[1].legend()
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle("Best point per shape: target fill and extra waves by version")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fill_by_version.png"), dpi=120)
    plt.close(fig)


def hbl_vs_ours(best):
    x, y, c = [], [], []
    for d in best.values():
        if "torch" in d and "ours" in d:
            x.append(float(d["ours"]["tile_target_fill"]))
            y.append(float(d["torch"]["tile_target_fill"]))
            c.append(math.log2(float(d["torch"]["us_median"]) / float(d["ours"]["us_median"])))
    fig, ax = plt.subplots(figsize=(7, 6))
    lim = max(abs(v) for v in c)
    sc = ax.scatter(x, y, c=c, cmap="coolwarm_r", vmin=-lim, vmax=lim, s=14)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("log2(hipBLASLt us / our us): > 0 we are faster")
    ax.set_xlabel("target fill of our best tile (v31 or v9)")
    ax.set_ylabel("target fill of hipBLASLt's tile")
    ax.set_title("Per shape: hipBLASLt's fill against ours")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fill_hbl_vs_ours.png"), dpi=120)
    plt.close(fig)


def gap_vs_time(best):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for v, name, marker in (("ours", "ours (best v31 or v9)", "o"), ("flydsl", "FlyDSL", "^")):
        g, t = [], []
        for d in best.values():
            if "torch" in d and v in d:
                g.append(float(d["torch"]["tile_target_fill"]) - float(d[v]["tile_target_fill"]))
                t.append(float(d[v]["us_median"]) / float(d["torch"]["us_median"]))
        r = np.corrcoef(g, np.log(t))[0, 1]
        ax.scatter(g, t, s=12, marker=marker, alpha=0.6, label=f"{name} (corr with log ratio {r:.2f})")
    ax.axhline(1, color="k", lw=1)
    ax.axvline(0, color="k", lw=1, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("hipBLASLt target fill - this tile's target fill")
    ax.set_ylabel("time / hipBLASLt time (> 1 slower)")
    ax.set_title("Fill gap to hipBLASLt against time ratio, per shape")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fill_gap_vs_time.png"), dpi=120)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    sweep()
    best = load()
    by_version(best)
    hbl_vs_ours(best)
    gap_vs_time(best)
    print("wrote", sorted(f for f in os.listdir(OUT) if f.startswith("fill_")))


if __name__ == "__main__":
    main()
