"""Conceptual figures for the CU drift write-up (no GPU, a few seconds).

    python make_figures.py     # writes images/concept_*.png

Model of one XCD: 32 CUs, tiles in the XCD's local order l = 0, 1, ...; step s = l // 32 is the
32 tiles that run together in lockstep. Within a step, row = l % 4 and column = l // 4 (mod 8), so
a tile shares its A row with l + 4, l + 8, ... and its B column with l + 1, l + 2, l + 3.
A tile takes 128 K-steps; each tile's time varies by SIGMA (fraction) at random, plus a small
fixed per-CU offset.
  static  (persistent): CU j runs l = j, j + 32, j + 64, ... back to back
  dynamic (v9):         whoever finishes next takes the next l
The values below are the ones fitted from the measured v12 records on 16384x57344x8192.
"""

import heapq
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "images")
CUS = 32
KSTEPS = 128  # K = 8192, BLOCK_K = 64
STEPS = 56  # trips per program on 16384x57344x8192
SIGMA = 0.0039  # per-tile jitter (fraction of a tile)
OFFSET_SD = 0.0006  # per-CU systematic offset (fraction)
L2_WINDOW = 10  # K-steps of shared data a 4 MB L2 holds for one step (4 A rows + 8 B columns)


def simulate(policy, n_tiles, delta, sigma, rng):
    """Start time (in tiles) of every local tile under the given policy."""
    start = np.zeros(n_tiles)

    def dur(cu):
        return (1 + delta[cu]) * (1 + sigma * rng.standard_normal())

    if policy == "static":
        for cu in range(CUS):
            t = 0.0
            for l in range(cu, n_tiles, CUS):
                start[l] = t
                t += dur(cu)
    else:
        heap = [(dur(cu), cu) for cu in range(CUS)]
        heapq.heapify(heap)
        for l in range(CUS, n_tiles):
            t, cu = heapq.heappop(heap)
            start[l] = t
            heapq.heappush(heap, (t + dur(cu), cu))
    return start


def partner_gap_p90(start):
    """Per step: p90 start gap (K-steps) for B-column and A-row partner pairs."""
    n = len(start)
    l = np.arange(CUS)
    row, col = l % 4, l // 4
    i, j = np.triu_indices(CUS, 1)
    b_pair, a_pair = col[i] == col[j], row[i] == row[j]
    out = {"b_col": [], "a_row": []}
    for s in range(n // CUS):
        st = start[s * CUS:(s + 1) * CUS]
        g = np.abs(st[i] - st[j]) * KSTEPS
        out["b_col"].append(np.percentile(g[b_pair], 90))
        out["a_row"].append(np.percentile(g[a_pair], 90))
    return out


def fig_ab_reuse():
    """Tile numbering within a step, and the dynamic-dispatch start gap vs index distance."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1.1, 1]})
    for r in range(4):
        for c in range(8):
            l = c * 4 + r
            color = "white"
            if l == 0:
                color = "#9e9e9e"
            elif c == 0:
                color = "#8ecae6"  # B-column partners of tile 0
            elif r == 0:
                color = "#ffb703"  # A-row partners of tile 0
            ax.add_patch(Rectangle((c, 3 - r), 1, 1, facecolor=color, edgecolor="black"))
            ax.text(c + 0.5, 3.5 - r, str(l), ha="center", va="center", fontsize=11)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 4)
    ax.set_xticks(np.arange(8) + 0.5, [f"col {c}" for c in range(8)])
    ax.set_yticks(np.arange(4) + 0.5, [f"row {r}" for r in range(3, -1, -1)])
    ax.set_title("Local tile index l in one step: row = l % 4, col = l // 4\n"
                 "blue: B-column partners of tile 0 (1-3 apart), orange: A-row partners (4-28 apart)",
                 fontsize=9)  # fmt: skip
    ax.set_aspect("equal")

    d = np.arange(1, 32)
    for w, ls, label in ((16, "-", "W = 16 K-steps (random-walk model)"),
                         (6, "--", "W = 6 K-steps (measured v9)")):  # fmt: skip
        bx.plot(d, d * w / 32, ls=ls, c="k", label=label)
    bx.axvspan(0.5, 3.5, color="#8ecae6", alpha=0.5, label="B-column partner distances")
    for a in range(4, 29, 4):
        bx.axvline(a, color="#ffb703", lw=2, alpha=0.8)
    bx.plot([], [], color="#ffb703", lw=2, label="A-row partner distances")
    bx.axhline(L2_WINDOW, ls=":", c="gray", label="~L2 window")
    bx.set_xlabel("index distance between two tiles of a step")
    bx.set_ylabel("start gap (K-steps)")
    bx.set_title("Dynamic dispatch: tiles go out one per finish, W/32 apart,\n"
                 "so the start gap is (index distance) x W / 32", fontsize=9)  # fmt: skip
    bx.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "concept_ab_reuse.png"), dpi=110)
    plt.close(fig)


def fig_random_walk():
    """32 CUs: running speed error converges, position (lateness) spread grows like sqrt(n)."""
    rng = np.random.default_rng(0)
    err = SIGMA * rng.standard_normal((CUS, STEPS))  # fraction of a tile, per tile
    pos = np.cumsum(err, axis=1) * KSTEPS  # K-steps
    n = np.arange(1, STEPS + 1)
    speed = np.cumsum(err, axis=1) / n * 100  # running mean, % of a tile
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 3.8))
    ax.plot(n, speed.T, lw=0.8, alpha=0.6)
    ax.set_title("Speed error (running mean per tile) evens out: ~1/sqrt(n)", fontsize=10)
    ax.set_xlabel("tiles")
    ax.set_ylabel("% of a tile")
    bx.plot(n, pos.T, lw=0.8, alpha=0.6)
    bx.plot(n, pos.max(0) - pos.min(0), c="k", lw=2, label="spread (slowest - fastest)")
    bx.plot(n, 4 * SIGMA * KSTEPS * np.sqrt(n), c="k", ls="--", label="4 sigma sqrt(n)")
    bx.axhline(L2_WINDOW, ls=":", c="gray", label="~L2 window")
    bx.set_title("Position (lateness) wanders: ~sqrt(n)", fontsize=10)
    bx.set_xlabel("tiles")
    bx.set_ylabel("K-steps")
    bx.legend(fontsize=8)
    fig.suptitle(f"32 CUs, no slow CU, per-tile jitter {SIGMA:.2%} ({SIGMA * KSTEPS:.1f} K-steps)",
                 fontsize=10)  # fmt: skip
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "concept_random_walk.png"), dpi=110)
    plt.close(fig)


def fig_pairing(seeds=8):
    """Monte Carlo: p90 partner gap vs step, static vs dynamic, B-column vs A-row."""
    n_tiles = STEPS * CUS
    res = {}
    for pol in ("static", "dynamic"):
        runs = []
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            delta = rng.normal(0, OFFSET_SD, CUS)
            runs.append(partner_gap_p90(simulate(pol, n_tiles, delta, SIGMA, rng)))
        res[pol] = {k: np.mean([r[k] for r in runs], 0) for k in ("b_col", "a_row")}
    s = np.arange(1, STEPS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, k, name in ((axes[0], "b_col", "B-column partners"), (axes[1], "a_row", "A-row partners")):
        ax.plot(s, res["static"][k], label="static (persistent)")
        ax.plot(s, res["dynamic"][k], label="dynamic (v9 model)")
        ax.axhline(L2_WINDOW, ls=":", c="gray", label="~L2 window")
        ax.set_title(f"{name}: p90 start gap", fontsize=10)
        ax.set_xlabel("step (trip)")
    axes[0].set_ylabel("K-steps")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Monte Carlo, jitter {SIGMA:.2%}/tile, CU offset sd {OFFSET_SD:.2%}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "concept_pairing.png"), dpi=110)
    plt.close(fig)
    for pol in res:
        print(f"{pol}: final-step p90 B / A = {res[pol]['b_col'][-1]:.1f} / {res[pol]['a_row'][-1]:.1f} K-steps")


if __name__ == "__main__":
    os.makedirs(IMG, exist_ok=True)
    fig_ab_reuse()
    fig_random_walk()
    fig_pairing()
    print("wrote images/concept_ab_reuse.png, concept_random_walk.png, concept_pairing.png")
