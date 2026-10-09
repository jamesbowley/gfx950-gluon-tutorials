"""How many ways to split a stream-K leftover tile, for any S and K: a model (no GPU).

    python split_model.py      # prints the tables, writes images/split_optimum.png

tail(S, n, K) = MAC(ipt / n k-steps on S n CUs) + exchange(pattern, S, n), against the measured
data-parallel tail of the same S and K:
- MAC: the split programs' K loop, stamped in the GEMM by v20's fixup_timing.py --tile-limit
  (kernels/.../v20_streamk_reduce_scatter/results/fixup_timing_limited.jsonl: one 4096x8192 matrix,
  the first 256 + S tiles, S = 1-128, n = 2-32, K = 4096 and 16384). Fitted as o + k t(A): a fixed
  cost o per segment (prologue, pipeline fill and drain) plus k k-steps at t(A) us each, where
  A = S n is the number of CUs running the tail. t(A) grows with A because a split program reads
  its own k-slices, so the tail's K loop becomes bandwidth-bound once many CUs run it.
- exchange: this experiment's synthetic exchange for every S (results/exchange_sn.md): 256 KiB
  partials, write-through, each group on one XCD, n a power of 2 up to 32; other n are interpolated
  linearly in n, and n = 64 extrapolated from n = 16 and 32. It has no C store or bias.
- data-parallel: the measured tail of v13 on the same matrix (experiments/streamk_split_count
  sweep C: kernel time with 256 + S tiles minus with 256), so it carries no model at all.
There is no skew term: every measured tail is a kernel time minus the full wave's, and what the
model leaves out (C store, bias, the programs' spread of start times) shows up as its residual
against those measurements (tail_variants.py).
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MAC_ROWS = os.path.join(REPO, "kernels", "gemm", "intra_wave", "a16w16", "v20_streamk_reduce_scatter", "results",
                        "fixup_timing_limited.jsonl")
EXCHANGE = os.path.join(HERE, "results", "exchange_sn.md")
SWEEP_C = os.path.join(REPO, "experiments", "streamk_split_count", "results", "sweep_c.jsonl")
IMG = os.path.join(HERE, "images")
BLOCK_K = 64
NS = (2, 3, 4, 6, 8, 12, 16, 32, 64)
PATTERNS = {"reduce-scatter": "#6aa56a", "owner": "#c0504d"}


def mac_rows(fixup):
    """(S, n, k-steps per program, MAC us) from the stamped one-tile runs with this ending. The rs
    rows are medians over all of a tile's programs, the owner rows over the owners (chunk 0)."""
    rows = []
    with open(MAC_ROWS) as f:
        for line in f:
            r = json.loads(line)
            if r["wrong"] == 0 and r["fixup"] == fixup:
                rows.append((r["S"], r["n"], r["K"] // BLOCK_K / r["n"], r["mac"]))
    return rows


def fit_mac(rows):
    """Least squares o + k t(A) over every row; t(A) for each A = S n present. Returns (o, {A: t}, rms)."""
    As = sorted({Sv * n for Sv, n, *_ in rows})
    X = np.zeros((len(rows), 1 + len(As)))
    y = np.zeros(len(rows))
    for i, (Sv, n, k, mac) in enumerate(rows):
        X[i, 0] = 1
        X[i, 1 + As.index(Sv * n)] = k
        y[i] = mac
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return coef[0], dict(zip(As, coef[1:])), float(np.sqrt(np.mean(resid**2)))


def t_of(A, t):
    """t(A), interpolated in log2(A) between the fitted points, flat outside them."""
    As = sorted(t)
    return float(np.interp(np.log2(A), np.log2(As), [t[a] for a in As]))


def exchange_sn():
    """{pattern: {S: {n: us}}} from results/exchange_sn.md."""
    out = {}
    with open(EXCHANGE) as f:
        for line in f:
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if len(c) > 6 and c[0] in PATTERNS and c[1].isdigit():
                out.setdefault(c[0], {}).setdefault(int(c[1]), {})[int(c[2])] = float(c[5])
    return out


def exchange_us(ex, pattern, S, n):
    """The exchange of S tiles split n ways: measured at powers of 2, linear in n between them and
    beyond the last two."""
    pts = ex[pattern].get(S)
    if not pts:
        Ss = sorted(ex[pattern])
        lo = max((s for s in Ss if s <= S), default=Ss[0])
        pts = ex[pattern][lo]
    ns = sorted(pts)
    if n in pts:
        return pts[n]
    if len(ns) == 1:
        return pts[ns[0]]
    if n > ns[-1]:
        a, b = ns[-2], ns[-1]
        return pts[b] + (pts[b] - pts[a]) * (n - b) / (b - a)
    return float(np.interp(n, ns, [pts[x] for x in ns]))


def sweep_rows():
    with open(SWEEP_C) as f:
        return [json.loads(line) for line in f]


def dp_tails(rows):
    """{(S, K): measured v13 tail us} on the limited matrix: kernel(256 + S) - kernel(256)."""
    base = {r["K"]: r["us"] for r in rows if r["config"] == "v13 base" and r.get("limit") == 256}
    return {(r["S"], r["K"]): r["us"] - base[r["K"]] for r in rows
            if r["config"] == "v13" and r.get("limit") and r["K"] in base}


FIXUP = {"reduce-scatter": "rs", "owner": "owner"}


class Model:
    def __init__(self):
        self.fits = {p: fit_mac(mac_rows(f)) for p, f in FIXUP.items()}
        self.ex = exchange_sn()

    def mac(self, pattern, S, n, K):
        o, t, _ = self.fits[pattern]
        return o + K / BLOCK_K / n * t_of(S * n, t)

    def tail(self, pattern, S, n, K):
        """(tail us, MAC us, exchange us) for S tiles split n ways at K."""
        mac = self.mac(pattern, S, n, K)
        e = exchange_us(self.ex, pattern, S, n)
        return mac + e, mac, e

    def splits(self, S, K):
        return [n for n in NS if S * n <= 256 and n <= K // (2 * BLOCK_K)]

    def best(self, pattern, S, K, dp):
        """(n, tail us) of the best split, or (1, dp) if no split beats data-parallel."""
        vals = {n: self.tail(pattern, S, n, K)[0] for n in self.splits(S, K)}
        if not vals:
            return 1, dp
        n = min(vals, key=vals.get)
        return (n, vals[n]) if vals[n] < dp else (1, dp)


def main():
    m = Model()
    rows = sweep_rows()
    dp = dp_tails(rows)
    for p, (o, t, rms) in m.fits.items():
        print(f"MAC fit, {p} ({len(mac_rows(FIXUP[p]))} stamped GEMM runs): o = {o:.2f} us per segment, rms residual "
              f"{rms:.2f} us; t(A), us per k-step: " + ", ".join(f"{a} CUs {v:.2f}" for a, v in sorted(t.items())))
    print()
    Ss = sorted({S for S, _ in dp if S <= 128})
    Ks = sorted({K for S, K in dp if S <= 128})
    print("Best split by the model, per S and K: pattern n (model tail us / measured data-parallel tail us).\n")
    print("| S | " + " | ".join(f"K = {K}" for K in Ks) + " |")
    print("|---|" + "---|" * len(Ks))
    best = {}
    for S in Ss:
        cells = []
        for K in Ks:
            d = dp[(S, K)]
            b = {p: m.best(p, S, K, d) for p in PATTERNS}
            best[(S, K)] = b
            p = min(b, key=lambda x: b[x][1])
            n, v = b[p]
            cells.append("data-parallel" if n == 1 else f"{'rs' if p == 'reduce-scatter' else 'owner'} {n} "
                         f"({v:.0f} / {d:.0f})")
        print(f"| {S} | " + " | ".join(cells) + " |")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
    for ax, p in zip(axes[:2], PATTERNS):
        grid = np.array([[np.log2(best[(S, K)][p][0]) for K in Ks] for S in Ss])
        im = ax.imshow(grid, cmap="viridis", aspect="auto", vmin=0, vmax=6)
        for i, S in enumerate(Ss):
            for j, K in enumerate(Ks):
                n = best[(S, K)][p][0]
                ax.text(j, i, "dp" if n == 1 else str(n), ha="center", va="center", fontsize=8,
                        color="w" if grid[i, j] < 3 else "k")
        ax.set_xticks(range(len(Ks)), [str(K) for K in Ks])
        ax.set_yticks(range(len(Ss)), [str(S) for S in Ss])
        ax.set_xlabel("K")
        ax.set_ylabel("leftover tiles S")
        ax.set_title(f"Best n by the model, {p}", fontsize=10)
    ax = axes[2]
    grid = np.array([[dp[(S, K)] / min(best[(S, K)][p][1] for p in PATTERNS) for K in Ks] for S in Ss])
    im = ax.imshow(grid, cmap="YlGn", aspect="auto", norm=LogNorm(vmin=1, vmax=16))
    for i in range(len(Ss)):
        for j in range(len(Ks)):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="w" if grid[i, j] > 5 else "k")
    ax.set_xticks(range(len(Ks)), [str(K) for K in Ks])
    ax.set_yticks(range(len(Ss)), [str(S) for S in Ss])
    ax.set_xlabel("K")
    ax.set_title("Model: measured data-parallel tail / best split tail", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.text(0.01, 0.01, "Model tail = MAC (o + k-steps x t(S n), fitted per ending to v20's stamped K loop, rms "
             + " / ".join(f"{m.fits[p][2]:.1f}" for p in PATTERNS) + " us) + exchange (results/exchange_sn.md, no C "
             "store). Data-parallel: v13 measured on the same matrix (sweep C). 'dp': no split beats data-parallel.",
             fontsize=8, color="0.3")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    os.makedirs(IMG, exist_ok=True)
    out = os.path.join(IMG, "split_optimum.png")
    fig.savefig(out, dpi=150)
    print("\n" + out)


if __name__ == "__main__":
    main()
