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
"""The preliminary sweep's report (no GPU): results/prelim_<i>.csv (sweep.py PRELIM, run_prelim.sh; shard i ran
on one GPU) to results/prelim_report.md and images/prelim_*.png.

    python v27_streamk_tile/prelim_report.py

Sections: noise (the repeated points), per-shape winners and the regret of fixed choices, the
separability test (does a few configurations per tile, chosen without looking at the other tiles, keep
the global best?), cutoff maps (best tile per shape; best configuration per tile in its own
coordinates), n curves (time against programs per tile with the ending fixed) and the memory-bound
    shapes against v9_tile §14.7's model. A configuration is a sweep.py kind; `cap` (the deployed rule) is
reported, but the separability test is over the explicit configurations only.
"""

import collections
import csv
import glob
import math
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
IMG = os.path.join(HERE, "images")
QUAD = ("256x256", "256x128", "128x256", "128x128x128")
SMALL = ("16x256x128", "16x64x128", "32x128x128")
TILES = QUAD + SMALL
V26 = ("256x256", "cap")
NEAR = 1.02
TOP = 3
# v9_tile WORKLOG §14.7: time = T0 + unique bytes / min(HBM, in flight / L), in flight = busy CUs x min(Q, 2 stages x
# programs per CU x bytes per k-step).
T0_US, L_NS, Q_BYTES, HBM_TBS = 7.75, 615.0, 32768, 6.3
FAMILY_COLOR = {"dp": "tab:gray", "rs": "tab:blue", "ow": "tab:orange", "2t": "tab:red", "sp": "tab:purple",
                "cap": "black"}
TILE_COLOR = dict(zip(TILES, ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown",
                              "tab:pink")))


def dims(tile):
    """(BM, BN, BK, programs) of a spec (BK 64 unless given; a small tile has BN / 64 waves, 1024 / W programs)."""
    p = [int(x) for x in tile.split("x")]
    bm, bn, bk = p[0], p[1], (p[2] if len(p) > 2 else 64)
    small = min(bm, bn) <= 32
    return bm, bn, bk, (1024 // max(1, bn // 64) if small else 256)


def regime(M):
    return "compute-bound" if M >= 1024 else "memory-bound"


def opt_in(kind):
    """The atomic ending and 16-bit partials: only ever selected explicitly, so not part of the search."""
    return kind.startswith("az") or (kind[:2] in ("rs", "ow") and kind[-1] in "bp")


def family(kind):
    return "az" if kind.startswith("az") else kind[:2] if kind[:2] in ("rs", "ow", "2t", "sp", "dp") else kind


def shape_key(sh):
    M, N, K = sh
    return (M < 1024, K, M, N)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))]


def load():
    """The shards' rows, each tagged with its shard (GPU), or the merged prelim.csv's."""
    shards = sorted(glob.glob(os.path.join(RES, "prelim_[0-9]*.csv")))
    rows = []
    for f in shards or [os.path.join(RES, "prelim.csv")]:
        gpu = os.path.basename(f)[7:-4] if shards else ""
        rows += [dict(r, gpu=gpu) for r in csv.DictReader(open(f)) if not opt_in(r["kind"])]
    for r in rows:
        for k in ("M", "N", "K", "rep", "tiles", "S", "waves", "k_steps"):
            r[k] = int(r[k])
        for k in ("us_median", "tflops", "tbps", "n", "n_hi", "pairs_per_prog", "programs", "busy_cus"):
            r[k] = float(r[k]) if r[k] not in ("", None) else None
        r["ok"] = r["correct"] == "True" and not r["error"] and r["us_median"]
    return rows


def ratio_stats(xs):
    return f"{statistics.median(xs):.3f} / {pct(xs, 0.9):.3f} / {max(xs):.3f}" if xs else "-"


class Data:
    def __init__(self, rows):
        self.rows = rows
        self.base = [r for r in rows if r["rep"] == 0]
        self.good = [r for r in self.base if r["ok"]]
        self.T = collections.defaultdict(dict)  # shape -> (tile, kind) -> us
        self.R = collections.defaultdict(dict)
        for r in self.good:
            sh = (r["M"], r["N"], r["K"])
            self.T[sh][(r["tile"], r["kind"])] = r["us_median"]
            self.R[sh][(r["tile"], r["kind"])] = r
        self.shapes = sorted(self.T, key=shape_key)

    def explicit(self, sh, tile=None):
        return {c: t for c, t in self.T[sh].items() if c[1] != "cap" and (tile is None or c[0] == tile)}

    def best(self, sh, tile=None, explicit=False):
        t = self.explicit(sh, tile) if explicit else {c: v for c, v in self.T[sh].items()
                                                       if tile is None or c[0] == tile}
        if not t:
            return None, None
        c = min(t, key=t.get)
        return c, t[c]


def noise_section(d, out):
    orig = {(r["kind"], r["tile"], r["M"], r["N"], r["K"]): r for r in d.good}
    pairs = []
    for r in d.rows:
        k = (r["kind"], r["tile"], r["M"], r["N"], r["K"])
        if r["rep"] == 1 and r["ok"] and k in orig:
            pairs.append((orig[k], r))
    diffs = [abs(b["us_median"] / a["us_median"] - 1) for a, b in pairs]
    out.append("\n## 1. Noise\n")
    if not diffs:
        out.append("No repeated points.\n")
        return 0.02
    spread = statistics.median(diffs)
    sig = pct(diffs, 0.9)
    signed = collections.defaultdict(list)
    for a, b in pairs:
        signed[b["gpu"]].append(b["us_median"] / a["us_median"])
    out.append(f"{len(pairs)} points timed twice, on different GPUs and at different times. |second / first - 1|: "
               f"median {100 * spread:.2f}%, 75th percentile {100 * pct(diffs, 0.75):.2f}%, 90th percentile "
               f"{100 * sig:.2f}%, max {100 * max(diffs):.2f}%. No GPU is biased: the median second / first per "
               f"repeating GPU is {min(statistics.median(v) for v in signed.values()):.3f}-"
               f"{max(statistics.median(v) for v in signed.values()):.3f}. Below, a difference under "
               f"**{100 * sig:.1f}%** (the 90th percentile) is treated as not significant (marked ~). Twice the "
               f"median ({100 * 2 * spread:.1f}%) would be too tight: a quarter of the repeats differ by more.\n")
    out.append("| kernel time | pairs | median | 90th percentile | max |")
    out.append("|---|---|---|---|---|")
    for lo, hi in ((0, 20), (20, 60), (60, 1e9)):
        xs = [abs(b["us_median"] / a["us_median"] - 1) for a, b in pairs if lo <= a["us_median"] < hi]
        if xs:
            label = f"{lo}-{hi} us" if hi < 1e9 else f">= {lo} us"
            out.append(f"| {label} | {len(xs)} | {100 * statistics.median(xs):.2f}% | {100 * pct(xs, 0.9):.2f}% | "
                       f"{100 * max(xs):.2f}% |")
    worst = sorted(pairs, key=lambda p: -abs(p[1]["us_median"] / p[0]["us_median"] - 1))[:5]
    out.append("\nLargest differences: " + "; ".join(
        f"{a['kind']}:{a['tile']}:{a['M']}x{a['N']}x{a['K']} {a['us_median']:.1f} / {b['us_median']:.1f} us"
        for a, b in worst) + ".\n")
    return sig


def coverage_section(d, out):
    bad = [r for r in d.base if r["correct"] == "False"]
    err = [r for r in d.base if r["error"]]
    out.append(f"{len(d.base)} points ({len(d.good)} correct and timed), {len(d.T)} shapes. "
               f"Wrong results: {len(bad)}; failed or hung: {len(err)}.\n")
    for r in bad + err:
        out.append(f"- {r['kind']}:{r['tile']}:{r['M']}x{r['N']}x{r['K']}: "
                   f"{'wrong result' if r in bad else r['error']}")
    if bad or err:
        out.append("")


def fmt_cfg(c):
    return f"{c[1]} @{c[0]}"


def winners_section(d, sig, out):
    out.append("\n## 2. Per-shape winners and regrets\n")
    out.append("Per shape: the fastest (tile, configuration); v26 as deployed (`cap` at 256x256, which v27 at "
               "256x256 is); capped at its best tile; the best data-parallel tile; and each tile's best "
               "configuration, all as time over the winner's. ~ marks a winner whose runner-up tile is within "
               f"noise ({100 * sig:.1f}%). Throughput is TFLOPS for M >= 1024, TB/s (A, B, C once) below.\n")
    reg = collections.defaultdict(lambda: collections.defaultdict(list))
    wins = collections.defaultdict(collections.Counter)
    for name, tiles in (("compute-bound", QUAD), ("memory-bound", TILES)):
        shapes = [sh for sh in d.shapes if regime(sh[0]) == name]
        if not shapes:
            continue
        out.append(f"\n### {name} (M {'1024, 4096' if name == 'compute-bound' else '1-128'})\n")
        out.append("| M | N | K | 256x256 tiles | winner | us | thr | v26 | best cap | best dp | "
                   + " | ".join(tiles) + " |")
        out.append("|---|---|---|---|---|---|---|---|---|---|" + "---|" * len(tiles))
        for sh in shapes:
            M, N, K = sh
            c, t = d.best(sh)
            other = [d.best(sh, ti)[1] for ti in tiles if ti != c[0] and d.best(sh, ti)[1]]
            close = "~" if other and min(other) <= (1 + sig) * t else ""
            r = d.R[sh][c]
            thr = f"{r['tflops']:.0f}" if M >= 1024 else f"{r['tbps']:.2f}"
            v26 = d.T[sh].get(V26)
            caps = [d.T[sh][(ti, "cap")] for ti in tiles if (ti, "cap") in d.T[sh]]
            dps = [d.T[sh][(ti, "dp")] for ti in tiles if (ti, "dp") in d.T[sh]]
            per = [d.best(sh, ti)[1] for ti in tiles]
            cells = [f"{x / t:.2f}" if x else "" for x in per]
            out.append(f"| {M} | {N} | {K} | {-(-M // 256) * -(-N // 256)} | {fmt_cfg(c)}{close} | {t:.1f} | {thr} | "
                       f"{v26 / t:.2f} | {min(caps) / t:.2f} | {min(dps) / t:.2f} | " + " | ".join(cells) + " |")
            wins[name][c[0]] += 1
            reg[name]["v26 as deployed"].append(v26 / t)
            reg[name]["capped, best tile per shape"].append(min(caps) / t)
            reg[name]["data-parallel, best tile per shape"].append(min(dps) / t)
            for ti, x in zip(tiles, per):
                if x:
                    reg[name][f"best configuration @{ti}"].append(x / t)
                if (ti, "cap") in d.T[sh]:
                    reg[name][f"capped @{ti}"].append(d.T[sh][(ti, "cap")] / t)
        out.append("\nWinning tiles: " + ", ".join(f"{ti} {n}" for ti, n in wins[name].most_common()) + ".\n")
        out.append("| choice | median / 90th percentile / worst time over the winner | shapes more than "
                   f"{100 * sig:.1f}% slower |")
        out.append("|---|---|---|")
        for k, xs in reg[name].items():
            out.append(f"| {k} | {ratio_stats(xs)} | {sum(x > 1 + sig for x in xs)} of {len(xs)} |")
    return reg


def top_configs(d, tile, shapes):
    """Explicit configurations of a tile by how many shapes they are within NEAR of the tile's best on."""
    cnt, n = collections.Counter(), 0
    for sh in shapes:
        t = d.explicit(sh, tile)
        if not t:
            continue
        n += 1
        b = min(t.values())
        for (ti, k), v in t.items():
            if v <= NEAR * b:
                cnt[k] += 1
    return cnt, n


def coords(d, sh, ti):
    """A shape in a tile's own coordinates: log2 of leftover tiles over programs (-9 for none), log2 k-steps,
    whether there is a full wave, the share of the tile's rows that M fills."""
    r = next(r for (t2, _), r in d.R[sh].items() if t2 == ti)
    bm, _, _, P = dims(ti)
    s = r["S"] / P
    return (math.log2(s) if s else -9.0, math.log2(r["k_steps"]), min(r["waves"], 1), min(sh[0], bm) / bm)


def nn_config(d, sh, ti):
    """The best explicit configuration at ti of the nearest other shape (coords) whose best also runs at sh."""
    here, c0 = d.explicit(sh, ti), coords(d, sh, ti)
    cands = []
    for o in d.shapes:
        c, _ = d.best(o, ti, explicit=True)
        if o != sh and c is not None:
            cands.append((sum((a - b) ** 2 for a, b in zip(c0, coords(d, o, ti))), shape_key(o), c[1]))
    return next((k for _, _, k in sorted(cands) if (ti, k) in here), "dp")


def separability_section(d, sig, out):
    out.append("\n## 3. Separability\n")
    out.append(f"Stage 1 of a two-stage sweep keeps a few configurations per tile, chosen on that tile alone; stage 2 "
               "picks the tile among those. For each tile, the explicit configurations (not `cap`) ranked by how "
               f"many shapes they are within {100 * (NEAR - 1):.0f}% of the tile's best on (count / shapes the tile "
               "ran):\n")
    out.append("| tile | shapes | top configurations (count) |")
    out.append("|---|---|---|")
    tops = {}
    for ti in TILES:
        cnt, n = top_configs(d, ti, d.shapes)
        tops[ti] = cnt
        out.append(f"| {ti} | {n} | " + ", ".join(f"{k} ({v})" for k, v in cnt.most_common(6)) + " |")
    by_regime = {}
    for name in ("compute-bound", "memory-bound"):
        shapes = [sh for sh in d.shapes if regime(sh[0]) == name]
        by_regime[name] = {ti: top_configs(d, ti, shapes)[0] for ti in TILES}

    def restricted(sh, keep):
        return min(v for (ti, k), v in d.explicit(sh).items() if k in keep[ti] or k == "dp")

    out.append("\nRestricting every tile to data-parallel and its top k split configurations (ranked over all shapes, "
               "or per regime), against the best explicit configuration of any tile:\n")
    out.append("| sets | k | shapes whose best is outside | loss: median / worst | shapes losing more than "
               f"{100 * sig:.1f}% | worst shapes |")
    out.append("|---|---|---|---|---|---|")
    summary = {}
    for label in ("all shapes", "per regime"):
        for k in (1, 2, 3, 4):
            losses, outside, worst = [], 0, []
            for sh in d.shapes:
                src = tops if label == "all shapes" else by_regime[regime(sh[0])]
                keep = {ti: [c for c, _ in src[ti].most_common() if c != "dp"][:k] for ti in TILES}
                c, b = d.best(sh, explicit=True)
                rb = restricted(sh, keep)
                outside += rb > b
                losses.append(rb / b)
                worst.append((rb / b, sh, c))
            worst.sort(reverse=True)
            ws = "; ".join(f"{sh[0]}x{sh[1]}x{sh[2]} {x:.3f} ({fmt_cfg(c)})" for x, sh, c in worst[:3] if x > 1)
            summary[(label, k)] = (outside, len(losses), losses)
            out.append(f"| {label} | {k} | {outside} of {len(losses)} | {statistics.median(losses):.3f} / "
                       f"{max(losses):.3f} | {sum(x > 1 + sig for x in losses)} | {ws} |")
    # A per-tile rule in the tile's own coordinates instead of a fixed set: leave-one-shape-out nearest neighbour.
    losses, outside, worst, hits, tile_loss = [], 0, [], 0, []
    for sh in d.shapes:
        pred = {ti: nn_config(d, sh, ti) for ti in TILES if d.explicit(sh, ti)}
        for ti, k in pred.items():
            hits += k == d.best(sh, ti, explicit=True)[0][1]
            tile_loss.append(d.T[sh][(ti, k)] / d.best(sh, ti, explicit=True)[1])
        c, b = d.best(sh, explicit=True)
        rb = min(d.T[sh][(ti, k)] for ti, k in pred.items())
        outside += rb > b
        losses.append(rb / b)
        worst.append((rb / b, sh, c))
    worst.sort(reverse=True)
    ws = "; ".join(f"{sh[0]}x{sh[1]}x{sh[2]} {x:.3f} ({fmt_cfg(c)})" for x, sh, c in worst[:3] if x > 1)
    out.append(f"| nearest-neighbour rule per tile | 1 | {outside} of {len(losses)} | "
               f"{statistics.median(losses):.3f} / {max(losses):.3f} | {sum(x > 1 + sig for x in losses)} | {ws} |")
    out.append(f"\nThe last row replaces the fixed sets by a rule in each tile's own coordinates (leftover tiles over "
               "programs, k-steps, whether there is a full wave, the share of the tile's rows in M): at each tile, the "
               "best configuration of the nearest other shape that also runs here (leave one shape out), then the "
               f"fastest tile among those. It picks the tile's own best configuration on {hits} of {len(tile_loss)} "
               f"(shape, tile) pairs; over all pairs it costs the tile a median {statistics.median(tile_loss):.3f}, "
               f"90th percentile {pct(tile_loss, 0.9):.3f}.\n")
    out.append("\nThe same question per tile: the tile's best configuration on the shapes it wins against those it "
               "does not (the three most frequent, with counts). If stage 1 is enough, the two columns agree:\n")
    out.append("| tile | shapes won | best configuration there | shapes not won | best configuration there |")
    out.append("|---|---|---|---|---|")
    for ti in TILES:
        won, lost = collections.Counter(), collections.Counter()
        for sh in d.shapes:
            c, _ = d.best(sh, ti, explicit=True)
            if c is None:
                continue
            g, _ = d.best(sh, explicit=True)
            (won if g[0] == ti else lost)[c[1]] += 1
        out.append(f"| {ti} | {sum(won.values())} | " + ", ".join(f"{k} {v}" for k, v in won.most_common(3))
                   + f" | {sum(lost.values())} | " + ", ".join(f"{k} {v}" for k, v in lost.most_common(3)) + " |")
    capwin = []
    for sh in d.shapes:
        for ti in TILES:
            if (ti, "cap") in d.T[sh]:
                b = d.best(sh, ti, explicit=True)[1]
                if b and d.T[sh][(ti, "cap")] < b / (1 + sig):
                    capwin.append((b / d.T[sh][(ti, "cap")], sh, ti, d.R[sh][(ti, "cap")]))
    capwin.sort(reverse=True)
    out.append(f"\n`cap` beats every explicit configuration of its tile by more than noise on {len(capwin)} "
               "(shape, tile) pairs" + (": " + "; ".join(
                   f"{sh[0]}x{sh[1]}x{sh[2]} @{ti} by {x:.3f} ({r['policy']} {r['fixup']} n = {r['n']:g})"
                   for x, sh, ti, r in capwin[:6]) if capwin else "") + ".\n")
    return summary


def cutoff_maps(d, sig, plt, out):
    from matplotlib.patches import Patch

    panels = [("compute-bound", 2048), ("compute-bound", 8192), ("memory-bound", 4096), ("memory-bound", 8192)]
    fig, axes = plt.subplots(2, 2, figsize=(17, 9))
    for ax, (name, K) in zip(axes.flat, panels):
        shapes = [sh for sh in d.shapes if regime(sh[0]) == name and sh[2] == K]
        Ms = sorted({sh[0] for sh in shapes})
        Ns = sorted({sh[1] for sh in shapes})
        tiles = QUAD if name == "compute-bound" else TILES
        for sh in shapes:
            c, t = d.best(sh)
            other = [d.best(sh, ti)[1] for ti in tiles if ti != c[0] and d.best(sh, ti)[1]]
            x, y = Ns.index(sh[1]), Ms.index(sh[0])
            ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color=TILE_COLOR[c[0]], alpha=0.75))
            close = "~" if other and min(other) <= (1 + sig) * t else ""
            ax.text(x, y, f"{c[1]}{close}\n{-(-sh[0] // 256) * -(-sh[1] // 256)}", ha="center", va="center", fontsize=7)
        ax.set_xlim(-0.5, len(Ns) - 0.5)
        ax.set_ylim(-0.5, len(Ms) - 0.5)
        ax.set_xticks(range(len(Ns)), [str(n) for n in Ns], fontsize=8)
        ax.set_yticks(range(len(Ms)), [str(m) for m in Ms], fontsize=8)
        ax.set_xlabel("N")
        ax.set_ylabel("M")
        ax.set_title(f"{name}, K = {K}: colour the winning tile; text its configuration and the 256x256 tile count "
                     "(~: runner-up tile within noise)", fontsize=8)
    fig.legend(handles=[Patch(color=TILE_COLOR[t], alpha=0.75, label=t) for t in TILES], ncol=7, loc="lower center")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(IMG, "prelim_cutoffs.png"), dpi=110)
    plt.close(fig)
    out.append("\n## 4. Cutoff maps\n")
    out.append("![best tile per shape](../images/prelim_cutoffs.png)\n")

    fig, axes = plt.subplots(len(TILES), 1, figsize=(22, 2.4 * len(TILES)))
    for ax, ti in zip(axes, TILES):
        cells = collections.defaultdict(collections.Counter)
        for sh in d.shapes:
            c, _ = d.best(sh, ti, explicit=True)
            if c is not None:
                r = d.R[sh][c]
                cells[(sh[0] < 1024, r["k_steps"]), r["S"]][c[1]] += 1
        rows = sorted({k[0] for k in cells})
        cols = sorted({k[1] for k in cells})
        for (rw, s), cnt in cells.items():
            x, y = cols.index(s), rows.index(rw)
            ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, alpha=0.35,
                                       color=FAMILY_COLOR[family(cnt.most_common(1)[0][0])]))
            ax.text(x, y, "\n".join(f"{k} {v}" if v > 1 else k for k, v in cnt.most_common()), ha="center",
                    va="center", fontsize=6)
        ax.set_xlim(-0.5, len(cols) - 0.5)
        ax.set_ylim(-0.5, len(rows) - 0.5)
        ax.set_xticks(range(len(cols)), [str(s) for s in cols], fontsize=7)
        ax.set_yticks(range(len(rows)), [f"{'memory' if mb else 'compute'}, {k} k-steps" for mb, k in rows], fontsize=7)
        ax.set_title(f"{ti}: columns S, leftover tiles of {dims(ti)[3]} programs", fontsize=9)
    fig.legend(handles=[Patch(color=c, alpha=0.35, label=f) for f, c in FAMILY_COLOR.items() if f != "cap"], ncol=6,
               loc="lower center")
    fig.suptitle("Best explicit configuration per tile in the tile's own coordinates "
                 "(count of shapes if more than one)")
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    fig.savefig(os.path.join(IMG, "prelim_configs.png"), dpi=100)
    plt.close(fig)
    out.append("Best explicit configuration at each tile against the tile's own coordinates: leftover tiles S "
               "(columns) and regime and k-steps per tile (rows). A cell lists the winners of the shapes that fall in "
               "it, shaded by the most frequent one.\n")
    out.append("![best configuration per tile](../images/prelim_configs.png)\n")


def n_curves(d, plt, out):
    import matplotlib.colors as mcolors
    import matplotlib.ticker as mticker
    import numpy as np

    out.append("\n## 5. n curves\n")
    out.append("Time over the tile's best explicit configuration at the shape, against n (programs per tile; for "
               "`all` with uneven tiles, the smaller count), the ending held fixed. One line per shape, colour "
               "S / programs; black: the median over shapes at each n. Top axis: pairs of k-steps per program at "
               "K = 8192.\n")
    table = []
    for name, tiles, fname in (("compute-bound", QUAD, "prelim_n_cb.png"), ("memory-bound", TILES, "prelim_n_mb.png")):
        fig, axes = plt.subplots(len(tiles), 2, figsize=(12, 3.4 * len(tiles)), squeeze=False)
        fig.subplots_adjust(hspace=0.75)
        norm = mcolors.LogNorm(1 / 256, 1)
        for row, ti in enumerate(tiles):
            pairs8k = 8192 // (2 * dims(ti)[2])
            for col, (ending, label) in enumerate((("rs", "reduce-scatter"), ("owner", "owner"))):
                ax = axes[row][col]
                per_n = collections.defaultdict(list)
                for sh in d.shapes:
                    if regime(sh[0]) != name:
                        continue
                    best = d.best(sh, ti, explicit=True)[1]
                    pts = {}
                    for (t2, k), r in d.R[sh].items():
                        if t2 == ti and k != "cap" and family(k) in ("rs", "ow") and r["ending"] == ending:
                            pts[r["n"]] = min(pts.get(r["n"], 1e9), r["us_median"] / best)
                    if len(pts) < 2:
                        continue
                    xs = sorted(pts)
                    S_P = d.R[sh][d.best(sh, ti)[0]]["S"] / dims(ti)[3]
                    ax.plot(xs, [pts[x] for x in xs], "-o", ms=2.5, lw=0.8, alpha=0.6,
                            color=plt.cm.viridis(norm(max(S_P, 1 / 256))))
                    for x in xs:
                        per_n[x].append(pts[x])
                if per_n:
                    xs = sorted(per_n)
                    ax.plot(xs, [statistics.median(per_n[x]) for x in xs], "-", color="k", lw=2)
                    for x in xs:
                        table.append((name, ti, ending, int(x), len(per_n[x]), statistics.median(per_n[x])))
                ax.set_xscale("log", base=2)
                ns = [2 ** i for i in range(1, 7)]
                ax.set_xticks(ns, [str(n) for n in ns])
                ax.xaxis.set_minor_locator(mticker.NullLocator())
                ax.set_xlim(1.8, 72)
                ax.set_ylim(0.95, 2.0)
                ax.grid(alpha=0.3)
                ax.set_title(f"{ti}, {label}", fontsize=9, pad=14)
                ax.set_ylabel("time / tile's best")
                if row == len(tiles) - 1:
                    ax.set_xlabel("n (programs per tile)")
                sec = ax.secondary_xaxis("top", functions=(lambda n, p=pairs8k: p / np.maximum(n, 1e-9),
                                                           lambda q, p=pairs8k: p / np.maximum(q, 1e-9)))
                sec.set_xticks([pairs8k / n for n in ns], [f"{pairs8k / n:g}" for n in ns])
                sec.xaxis.set_minor_locator(mticker.NullLocator())
                sec.tick_params(labelsize=7)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=axes, label="S / programs", shrink=0.4)
        fig.suptitle(f"{name}: time against n, ending fixed")
        fig.savefig(os.path.join(IMG, fname), dpi=100, bbox_inches="tight")
        plt.close(fig)
        out.append(f"![n curves, {name}](../images/{fname})\n")
    out.append("Median time over the tile's best, per n (shapes):\n")
    out.append("| regime | tile | ending | " + " | ".join(f"n = {n}" for n in (2, 3, 4, 6, 8, 16, 32, 64)) + " |")
    out.append("|---|---|---|" + "---|" * 8)
    groups = collections.defaultdict(dict)
    for name, ti, ending, n, cnt, med in table:
        groups[(name, ti, ending)][n] = (cnt, med)
    for (name, ti, ending), v in groups.items():
        cells = []
        for n in (2, 3, 4, 6, 8, 16, 32, 64):
            cells.append(f"{v[n][1]:.3f} ({v[n][0]})" if n in v else "")
        out.append(f"| {name} | {ti} | {ending} | " + " | ".join(cells) + " |")


def model_tbps(r):
    """v9_tile §14.7's model throughput for a row: the programs of its last wave (or stream-K phase) in flight."""
    bm, bn, bk, P = dims(r["tile"])
    progs = min(r["tiles"], P) if r["ending"] == "dp" else r["programs"]
    busy = min(progs, 256)
    inflight = busy * min(Q_BYTES, 2 * -(-progs // busy) * (bm + bn) * bk * 2)
    nbytes = 2 * (r["M"] * r["K"] + r["K"] * r["N"] + r["M"] * r["N"])
    t = T0_US + nbytes / (min(HBM_TBS, inflight / L_NS / 1e3) * 1e12) * 1e6
    return nbytes / (t * 1e-6) / 1e12


def memory_bound(d, plt, out):
    rows = [r for r in d.good if r["M"] < 1024 and r["kind"] != "cap"]
    out.append("\n## 6. Memory-bound shapes\n")
    if not rows:
        return
    endings = ("dp", "rs", "owner", "e2e")
    colors = dict(zip(endings, ("tab:gray", "tab:blue", "tab:orange", "tab:purple")))
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(19, 5.5))
    for e in endings:
        rs = [r for r in rows if r["ending"] == e]
        if not rs:
            continue
        a1.scatter([r["busy_cus"] for r in rs], [r["tbps"] for r in rs], s=10, alpha=0.6, color=colors[e], label=e)
        a2.scatter([r["pairs_per_prog"] for r in rs], [r["tbps"] for r in rs], s=10, alpha=0.6, color=colors[e])
        a3.scatter([model_tbps(r) for r in rs], [r["tbps"] for r in rs], s=10, alpha=0.6, color=colors[e])
    for ax in (a1, a2):
        ax.axhline(HBM_TBS, color="k", ls=":", lw=0.8)
        ax.set_ylabel("TB/s (A, B, C once)")
        ax.grid(alpha=0.3)
    a1.set_xlabel("busy CUs (programs in the stream-K phase or last wave, at most 256)")
    a1.axvline(110, color="grey", ls="--", lw=0.8)
    a1.legend(title="ending")
    a2.set_xscale("log", base=2)
    a2.set_xlabel("pairs of k-steps per program")
    a3.plot([0, 6.3], [0, 6.3], "k--", lw=0.8)
    a3.set_xlabel(f"model TB/s: {T0_US} us + bytes / min({HBM_TBS} TB/s, in flight / {L_NS:g} ns) (v9_tile §14.7)")
    a3.set_ylabel("measured TB/s")
    a3.grid(alpha=0.3)
    fig.suptitle("M = 1-128: every tile and explicit configuration")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "prelim_mb.png"), dpi=110)
    plt.close(fig)
    out.append("Every explicit configuration at M = 1-128, all 7 tiles. Busy CUs is the estimate in sweep.py; the "
               "model takes those programs' bytes in flight (2 stages, at most 32 KB per CU) and charges §14.7's "
               "fixed 7.75 us once, so a split's extra fixed cost shows as measured below model.\n")
    out.append("![memory-bound](../images/prelim_mb.png)\n")
    out.append("| tiles | ending | points | measured / model: median (10th-90th percentile) | median TB/s |")
    out.append("|---|---|---|---|---|")
    for fam, tiles in (("small", SMALL), ("quadrant", QUAD)):
        for e in endings:
            rs = [r for r in rows if r["ending"] == e and r["tile"] in tiles]
            if rs:
                x = [r["tbps"] / model_tbps(r) for r in rs]
                out.append(f"| {fam} | {e} | {len(rs)} | {statistics.median(x):.2f} ({pct(x, 0.1):.2f}-"
                           f"{pct(x, 0.9):.2f}) | {statistics.median(r['tbps'] for r in rs):.2f} |")
    out.append("\nBest TB/s per shape: data-parallel (any tile) against the best split (any tile, configuration):\n")
    out.append("| M | N | K | best dp (tile) | best split (configuration) | split / dp |")
    out.append("|---|---|---|---|---|---|")
    for sh in d.shapes:
        if sh[0] >= 1024:
            continue
        dp = [(d.R[sh][c]["tbps"], c) for c in d.T[sh] if c[1] == "dp"]
        sp = [(d.R[sh][c]["tbps"], c) for c in d.T[sh] if c[1] not in ("dp", "cap") and d.R[sh][c]["ending"] != "dp"]
        if dp and sp:
            (a, ca), (b, cb) = max(dp), max(sp)
            out.append(f"| {sh[0]} | {sh[1]} | {sh[2]} | {a:.2f} ({ca[0]}) | {b:.2f} ({fmt_cfg(cb)}) | {b / a:.2f} |")


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(IMG, exist_ok=True)
    d = Data(load())
    out = ["# v27 preliminary sweep: tiles x stream-K configurations\n",
           "PRELIMINARY: a coarse factorial (WORKLOG §11) to decide between a two-stage and a factorial re-fit; not "
           "a rule fit. Aligned N and K, bf16, back to back (HIP graph), 2 GB of rotating inputs, rocprof median of "
           "100 dispatches ([sweep.py](../sweep.py) PRELIM, [run_prelim.sh](../run_prelim.sh)). Tiles: 256x256, "
           "256x128, 128x256 and 128x128x128 on the compute-bound shapes (M = 1024, 4096, N = 256-8448, K = 2048, "
           "8192), and those plus 16x256x128, 16x64x128 and 32x128x128 on the memory-bound and crossover shapes "
           "(M = 1-128, N = 4096-16384, K = 4096, 8192). Configurations: `dp` (v9_tile at the tile), `cap` (v27 as "
           "deployed), one-tile `rs{n}` / `ow{n}` (reduce-scatter, owner, fp32 partials, on n programs per "
           "leftover tile; `all` = every program), `2t` (two-tile) and `sp` (spread). The atomic ending and 16-bit "
           "partials are opt-in (selected explicitly, never by a rule), so they are not part of the search; the run "
           "also timed `azall`, and its rows are left out.\n"]
    coverage_section(d, out)
    sig = noise_section(d, out)
    winners_section(d, sig, out)
    separability_section(d, sig, out)
    cutoff_maps(d, sig, plt, out)
    n_curves(d, plt, out)
    memory_bound(d, plt, out)
    text = "\n".join(out) + "\n"
    open(os.path.join(RES, "prelim_report.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
