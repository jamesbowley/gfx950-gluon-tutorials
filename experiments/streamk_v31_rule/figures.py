"""Maps of the sweeps in the formula's coordinates: programs per tail tile (m = P // S,
with the matching S) against K-steps per tile (k), per regime and spec group.

    python experiments/streamk_v31_rule/figures.py --rule results/streamk_rule.json \\
        'results/*_r1.csv' 'results/*_r2.csv' --out images/

For every (m bin, k) cell, over all (shape, spec) pairs of the group in it:
- left: colour = median speedup of the pair's best configuration over dp, label = the
  most common best configuration (dp, ow{n}, rs{n}) and the pair count;
- right: label = the formula's most common choice, colour = the median of its time
  over the pair's best.
"""

import argparse
import collections
import json
import os
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fit_formula as ff  # noqa: E402

P = 256
M_BINS = ((128, 256), (64, 127), (32, 63), (16, 31), (10, 15), (8, 9), (6, 7), (5, 5), (4, 4), (3, 3), (2, 2))
K_BINS = (4, 8, 12, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256)
GROUPS = {
    "large": ("256x256x64", "256x128x64", "128x256x64", "128x128x64"),
    "side64": ("256x64x64", "64x256x64", "128x64x64", "64x128x64", "64x64x64"),
    "bk128": ("128x128x128", "128x64x128", "64x128x128", "64x64x128"),
}
TITLES = {"large": "BLOCK_K 64, sides 128-256", "side64": "BLOCK_K 64, a side of 64",
          "bk128": "BLOCK_K 128"}


def m_label(lo, hi):
    s_lo, s_hi = (-(-P // (hi + 1)) if hi < P else 1), P // lo
    m = f"{lo}" if lo == hi else f"{lo}-{hi}"
    s = f"{s_lo}" if s_lo == s_hi else f"{s_lo}-{s_hi}"
    return f"{m}  (S {s})"


def kind_name(kind, n):
    return "dp" if kind == "dp" else f"{kind}{n}"


def cells(blocks, specs, regime):
    out = collections.defaultdict(list)
    for b in blocks.values():
        i = b["info"]
        if i["spec"] not in specs or not i["S"] or i["regime"] != regime:
            continue
        m = P // i["S"]
        row = next((r for r, (lo, hi) in enumerate(M_BINS) if lo <= m <= hi), None)
        if row is None or i["k"] not in K_BINS:
            continue
        out[(row, K_BINS.index(i["k"]))].append(b)
    return out


def best_kind(b):
    return min(b["times"], key=b["times"].get)


def draw(ax, gc, value, label, cmap, norm, title):
    data = np.full((len(M_BINS), len(K_BINS)), np.nan)
    for (r, c), bs in gc.items():
        data[r, c] = value(bs)
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
    for (r, c), bs in gc.items():
        rgba = im.cmap(im.norm(data[r, c]))
        dark = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2] < 0.5
        ax.text(c, r, label(bs), ha="center", va="center", fontsize=6.5, color="white" if dark else "black")
    ax.set_xticks(range(len(K_BINS)), [str(k) for k in K_BINS])
    ax.set_yticks(range(len(M_BINS)), [m_label(*b) for b in M_BINS])
    ax.set_xlabel("K-steps per tile, k (K / BLOCK_K)")
    ax.set_ylabel("programs per tail tile, m = P // S")
    ax.set_title(title, fontsize=10)
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--rule", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "images"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    blocks = ff.load(args.csv)
    rule = json.load(open(args.rule))
    form, consts = {"regimes": False, "tile": "none"}, {"all": rule["formula"]}
    mode = lambda xs: collections.Counter(xs).most_common(1)[0][0]

    def chosen(b):
        return kind_name(*ff.formula(b["info"], consts, form, {}, 1.0))

    def regret(bs):
        v = [b["times"][chosen(b)] / b["best"] for b in bs if chosen(b) in b["times"]]
        return statistics.median(v) if v else np.nan

    for group, specs in GROUPS.items():
        for regime, name in (("after", "one wave or more"), ("below", "below one wave")):
            gc = cells(blocks, specs, regime)
            if not gc:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(19, 7), constrained_layout=True)
            im = draw(axes[0], gc, lambda bs: statistics.median(b["times"]["dp"] / b["best"] for b in bs),
                      lambda bs: f"{mode(best_kind(b) for b in bs)}\n{len(bs)}", "viridis", Normalize(1.0, 1.6),
                      f"Measured, {TITLES[group]}, {name}: best (most common), pairs")
            fig.colorbar(im, ax=axes[0], label="median speedup of the best over dp")
            im = draw(axes[1], gc, lambda bs: max(regret(bs), 1.0), lambda bs: mode(chosen(b) for b in bs),
                      "Reds", Normalize(1.0, 1.15), f"Formula, {TITLES[group]}, {name}: its choice")
            fig.colorbar(im, ax=axes[1], label="median time of the formula's choice over the best")
            path = os.path.join(args.out, f"formula_{group}_{regime}.png")
            fig.savefig(path, dpi=120)
            plt.close(fig)
            print(path)


if __name__ == "__main__":
    main()
