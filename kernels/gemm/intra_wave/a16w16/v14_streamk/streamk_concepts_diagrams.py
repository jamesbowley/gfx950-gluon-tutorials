"""One small figure per stream-K concept, used by ../STREAMK_VARIANTS.md.

Each figure shows a single contrast on a toy problem; the detailed replays of each version are in
streamk_versions_diagrams.py. Timeline costs are illustrative k-steps shaped on STREAMK.md's
measurements (owner ending about 10 + 2.2 (n - 1), reduce-scatter about 24 at any n). C2's curve is
measured: experiments/streamk_split_count/results/sweep_a.jsonl, S = 4, K = 8192.

    python streamk_concepts_diagrams.py     # images/concepts/c*.png
"""

import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "images", "concepts")
SWEEP_A = os.path.join(HERE, "..", "..", "..", "..", "..", "experiments", "streamk_split_count", "results",
                       "sweep_a.jsonl")

TILE = ["#66c2a5", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494", "#80b1d3", "#bc80bd", "#fb8072",
        "#ccebc5"]
FULL_WAVE = "#d9d9d9"
STORE = "#d95f02"
READ = "#4d4d4d"
SLICE = "#1b7837"
CSTORE = "#7b3294"
WAIT = dict(facecolor="#eeeeee", edgecolor="grey", hatch="..")
PROG = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
XCD_SHADE = ["#eef4fb", "#fdf3e9"]
XCD_EDGE = ["#2166ac", "#c2410c"]

# Ending costs in k-steps after a program's MAC phase.
OWNER_STORE, OWNER_WAIT, OWNER_READ, OWNER_C = 1.5, 6.0, 2.2, 2.0
RS_STORE, RS_BARRIER, RS_SLICE, RS_C = 8.0, 6.0, 8.0, 1.0
BF16_STORE, BF16_SLICE = 5.0, 5.5  # v23: publish 30-46% and slice reads 17-44% shorter


def bar(ax, row, t0, t1, colour=None, label=None, h=0.8, fontsize=7, textcolour="black", **kw):
    style = dict(facecolor=colour, edgecolor="black", linewidth=0.6) | kw
    ax.add_patch(Rectangle((t0, row - h / 2), t1 - t0, h, **style))
    if label:
        ax.text((t0 + t1) / 2, row, label, ha="center", va="center", fontsize=fontsize, color=textcolour)


def rows_axis(ax, labels, xlabel="time (k-steps)"):
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_ylim(len(labels) - 0.4, -0.6)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(path)


def e2e(tiles, programs, ipt):
    """End-to-end cuts: program p's pieces as (tile, k0, k1), in forward order."""
    per = tiles * ipt // programs
    assert per * programs == tiles * ipt
    out = []
    for p in range(programs):
        a, b, pieces = p * per, (p + 1) * per, []
        while a < b:
            t, k0 = divmod(a, ipt)
            k1 = min(ipt, k0 + b - a)
            pieces.append((t, k0, k1))
            a += k1 - k0
        out.append(pieces)
    return out


def aligned(tiles, programs, ipt):
    """Aligned cuts: tile t gets programs // tiles programs (the first programs % tiles one more)."""
    out, p = [], 0
    for t in range(tiles):
        n = programs // tiles + (t < programs % tiles)
        step = ipt / n
        out.append([(p + j, j * step, (j + 1) * step) for j in range(n)])
        p += n
    return out


# ----------------------------------------------------------------------------- C1


def c01_dp():
    fig, ax = plt.subplots(figsize=(6.5, 2.4))
    for r in range(8):
        bar(ax, r, 0, 16, FULL_WAVE, f"T{r}")
        if r < 2:
            bar(ax, r, 16, 32, TILE[r], f"T{8 + r}")
        else:
            bar(ax, r, 16, 32, "white", edgecolor="grey", linestyle="--", hatch="//", linewidth=0.5)
    ax.text(24, 4.5, "idle: 256 - S CUs", ha="center", va="center", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none"))
    ax.text(8, -1.1, "full wave", ha="center", fontsize=8)
    ax.text(24, -1.1, "last wave: S = 2 leftover tiles", ha="center", fontsize=8)
    ax.set_xlim(0, 32)
    rows_axis(ax, [f"CU {r}" for r in range(8)])
    ax.set_ylim(7.6, -1.6)
    ax.set_title("Data-parallel: the last wave leaves most CUs idle", fontsize=10, loc="left")
    save(fig, "c01_dp.png")


# ----------------------------------------------------------------------------- C2


def c02_one_tile():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.5, 2.8), gridspec_kw=dict(width_ratios=[1.1, 1]))
    for r in range(8):
        bar(ax, r, 0, 16, FULL_WAVE, f"T{r}")
        t, j = divmod(r, 4)
        bar(ax, r, 16, 20, TILE[t], f"T{8 + t}.{j}", fontsize=6)
        bar(ax, r, 20, 22, READ)
    ax.text(8, -1.0, "full wave", ha="center", fontsize=7)
    ax.text(18, -1.0, "S = 2 tiles,\nn = 4 CUs each", ha="center", fontsize=7)
    ax.text(22.4, 3.5, "add the\npartials", ha="left", va="center", fontsize=7)
    ax.set_xlim(0, 26)
    rows_axis(ax, [f"CU {r}" for r in range(8)])
    ax.set_ylim(7.6, -1.9)
    ax.set_title("Split only the leftover tiles over n CUs", fontsize=10, loc="left")

    rows = [json.loads(line) for line in open(SWEEP_A)]
    dp = next(r["us"] for r in rows if r["S"] == 4 and r["config"] == "v13")
    pts = sorted((r["n"], r["us"]) for r in rows
                 if r["S"] == 4 and r["config"] == "v20 owner" and r["n"] != "default")
    ns, us = zip(*pts)
    bx.plot(ns, us, "o-", color=PROG[2], label="one-tile, owner ending")
    bx.axhline(dp, color="grey", linestyle="--", label="data-parallel")
    best = min(pts, key=lambda p: p[1])
    bx.annotate(f"n = {best[0]}: {best[1]:.0f} us", best, xytext=(best[0] * 1.5, best[1] + 35), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.6))
    bx.annotate(f"n = {ns[-1]}: {us[-1]:.0f} us\n2 k-steps of MAC,\n63 partials", (ns[-1], us[-1]),
                xytext=(9, 315), fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.6))
    bx.set_xscale("log", base=2)
    bx.set_xticks(ns, [str(n) for n in ns], fontsize=7)
    bx.tick_params(axis="y", labelsize=7)
    bx.set_xlabel("n, CUs per leftover tile", fontsize=8)
    bx.set_ylabel("kernel time (us)", fontsize=8)
    bx.legend(fontsize=7, loc="upper left", frameon=False)
    bx.set_title("Measured, S = 4, K = 8192: n must be tuned", fontsize=10, loc="left")
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)
    fig.tight_layout()
    save(fig, "c02_one_tile.png")


# ----------------------------------------------------------------------------- C3


def c03_two_tile():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.5, 2.6), sharey=True)
    for r in range(8):
        bar(ax, r, 0, 16, TILE[r], f"T{r}")
        t, j = divmod(r, 4)
        bar(ax, r, 16, 20, TILE[8 + t], f"T{8 + t}", fontsize=6)
    ax.set_title("dp + one-tile: tiles 8-9 have 4 CUs each", fontsize=10, loc="left")
    for p, pieces in enumerate(e2e(10, 8, 16)):
        t0 = 0
        for t, k0, k1 in pieces:
            bar(bx, p, t0, t0 + k1 - k0, TILE[t], f"T{t}", fontsize=7 if k1 - k0 > 2 else 6)
            t0 += k1 - k0
    bx.set_title("two-tile: every tile has at most 2 CUs", fontsize=10, loc="left")
    for a in (ax, bx):
        a.set_xlim(0, 21)
        a.set_xticks(range(0, 21, 4))
        rows_axis(a, [f"CU {r}" for r in range(8)])
    fig.tight_layout()
    save(fig, "c03_two_tile.png")


# ----------------------------------------------------------------------------- C4, C5

IPT45 = 24


def tiles_frame(ax, title):
    for t in range(3):
        ax.add_patch(Rectangle((0, t - 0.4), IPT45, 0.8, facecolor="none", edgecolor="black", linewidth=1.2))
    ax.set_xlim(-0.5, IPT45 + 0.5)
    ax.set_xticks(range(0, IPT45 + 1, 4))
    rows_axis(ax, [f"tile {t}" for t in range(3)], xlabel="k-step within the tile")
    ax.set_title(title, fontsize=10, loc="left")


def c04_e2e():
    fig, ax = plt.subplots(figsize=(6.5, 2.3))
    for p, pieces in enumerate(e2e(3, 8, IPT45)):
        for t, k0, k1 in pieces:
            split = len(pieces) > 1
            bar(ax, t, k0, k1, TILE[t], hatch="////" if split else None)
            ax.text((k0 + k1) / 2, t, f"p{p}", ha="center", va="center", fontsize=7,
                    fontweight="bold" if split else "normal",
                    bbox=dict(facecolor="white", edgecolor="none", pad=1) if split else None)
    tiles_frame(ax, "End-to-end: 3 tiles over 8 CUs, 9 k-steps each; cuts ignore tile boundaries\n"
                    "hatched: p2 and p5 span two tiles, so each pays a store, a drain and a restart")
    save(fig, "c04_e2e.png")


def c05_aligned():
    fig, ax = plt.subplots(figsize=(6.5, 2.3))
    for t, chunks in enumerate(aligned(3, 8, IPT45)):
        for p, k0, k1 in chunks:
            bar(ax, t, k0, k1, TILE[t], f"p{p}")
    for k in (8, 16):
        ax.axvline(k, ymin=0.25, ymax=0.97, color="black", linestyle=":", lw=0.8)
    tiles_frame(ax, "Aligned: cuts only inside tiles; one tile per CU, chunk j at the same k-steps")
    save(fig, "c05_aligned.png")


# ----------------------------------------------------------------------------- C6


def c06_order():
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 2.9), sharey=True)
    work = e2e(10, 8, 16)[:4]
    for ax, reverse, title in ((axes[0], False, "forward: 4 different k-steps at once"),
                               (axes[1], True, "reversed: 2 groups, d = 4 apart")):
        for p, pieces in enumerate(work):
            t, dy = 0, (p - 1.5) * 0.3 if reverse else 0
            for i, (tile, k0, k1) in enumerate(pieces[::-1] if reverse else pieces):
                ax.plot([t, t + k1 - k0], [k0 + dy, k1 + dy], color=PROG[p], lw=2.2, solid_capstyle="butt",
                        label=f"CU {p}" if i == 0 else None)
                if reverse and i == 0:
                    ax.plot(t + k1 - k0, k1 + dy, "o", color="black", ms=4, zorder=3)
                t += k1 - k0
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("time (k-steps)", fontsize=8)
        ax.set_xlim(0, 20)
        ax.set_ylim(-0.8, 16.8)
        ax.set_xticks(range(0, 21, 4))
        ax.set_yticks(range(0, 17, 4))
        ax.tick_params(labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("k-step being read", fontsize=8)
    axes[1].plot([], [], "o", color="black", ms=4, label="partial published, early")
    axes[1].legend(fontsize=7, frameon=False, loc="lower right", ncol=2)
    fig.suptitle("Two-tile on one XCD (4 CUs, 16 k-steps per tile): the k-step each CU reads over time",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "c06_order.png")


# ----------------------------------------------------------------------------- C7, C8, C11


def owner_end(n):
    return OWNER_WAIT + OWNER_READ * (n - 1) + OWNER_C


def draw_owner(ax, n):
    bar(ax, 0, -3, 0, FULL_WAVE, "MAC")
    bar(ax, 0, 0, OWNER_WAIT, label="wait", **WAIT)
    t = OWNER_WAIT
    for i in range(1, n):
        bar(ax, 0, t, t + OWNER_READ, READ, str(i), textcolour="white")
        t += OWNER_READ
    bar(ax, 0, t, t + OWNER_C, CSTORE, "C", textcolour="white")
    for r in range(1, n):
        bar(ax, r, -3, 0, FULL_WAVE)
        bar(ax, r, 0, OWNER_STORE, STORE)
    rows_axis(ax, ["owner"] + [f"CU {r}" for r in range(1, n)], xlabel="k-steps after the MAC phase")


def draw_rs(ax, n, store=RS_STORE, slice_=RS_SLICE):
    for r in range(n):
        bar(ax, r, -3, 0, FULL_WAVE, "MAC" if r == 0 else None)
        bar(ax, r, 0, store, STORE, "store" if r == 0 else None, textcolour="white")
        bar(ax, r, store, store + RS_BARRIER, label="barrier" if r == 0 else None, **WAIT)
        t = store + RS_BARRIER
        bar(ax, r, t, t + slice_, SLICE, f"read 1/{n} of all" if r == 0 else None, textcolour="white")
        bar(ax, r, t + slice_, t + slice_ + RS_C, CSTORE)
    rows_axis(ax, [f"CU {r}" for r in range(n)], xlabel="k-steps after the MAC phase")
    return store + RS_BARRIER + slice_ + RS_C


ENDING_LEGEND = [Patch(color=STORE, label="store partial"), Patch(**WAIT, label="wait"),
                 Patch(color=READ, label="read one partial"), Patch(color=SLICE, label="read a slice of every partial"),
                 Patch(color=CSTORE, label="store C")]


def stacked(title, ns=(2, 8)):
    """One panel per n, stacked on one time axis, each as tall as its number of CUs."""
    fig, axes = plt.subplots(len(ns), 1, figsize=(7.5, 0.9 + 0.3 * sum(ns) + 0.25 * len(ns)), sharex=True,
                             gridspec_kw=dict(height_ratios=list(ns)))
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    return fig, axes


def c07_owner():
    fig, axes = stacked("Owner ending: one CU reads the other n - 1 partials one after another")
    for ax, n in zip(axes, (2, 8)):
        draw_owner(ax, n)
        ax.set_xlim(-3.5, 28)
        ax.set_title(f"n = {n}: ending {owner_end(n):.0f} k-steps", fontsize=9, loc="left")
    axes[0].set_xlabel("")
    axes[1].legend(handles=[ENDING_LEGEND[i] for i in (0, 1, 2, 4)], fontsize=7, frameon=False,
                   loc="lower right")
    fig.tight_layout()
    save(fig, "c07_owner.png")


def c08_rs():
    fig, axes = stacked("Reduce-scatter: every CU stores its partial, then adds 1/n of the tile from all n")
    for ax, n in zip(axes, (2, 8)):
        end = draw_rs(ax, n)
        o = owner_end(n)
        ax.axvline(o, color="black", linestyle="--", lw=1.0)
        ax.set_xlim(-3.5, 28)
        ax.set_title(f"n = {n}: ending {end:.0f} k-steps; the owner ending's is {o:.0f}", fontsize=9, loc="left")
    axes[0].set_xlabel("")
    owner_line = plt.Line2D([], [], color="black", linestyle="--", label="owner ending done, same n")
    axes[1].legend(handles=[ENDING_LEGEND[i] for i in (0, 1, 3, 4)] + [owner_line], fontsize=7,
                   frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    save(fig, "c08_rs.png")


def c11_bf16():
    fig, axes = stacked("bf16 partials: half the bytes to store and read; the sums stay fp32", ns=(4, 4))
    for ax, (name, st, sl) in zip(axes, (("fp32 partials, 1 rounding", RS_STORE, RS_SLICE),
                                         ("bf16 partials, about 1.6 roundings", BF16_STORE, BF16_SLICE))):
        end = draw_rs(ax, 4, st, sl)
        ax.set_xlim(-3.5, 25)
        ax.set_title(f"reduce-scatter, n = 4, {name}: ending {end:.1f} k-steps", fontsize=9, loc="left")
    axes[0].set_xlabel("")
    fig.tight_layout()
    save(fig, "c11_bf16.png")


# ----------------------------------------------------------------------------- C9


def c09_chunk_major():
    S, n = 4, 2
    cases = [("tile-major", False, False), ("chunk-major, owner chunk 0", True, False),
             ("chunk-major, owner rotates", True, True)]
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.5))
    for ax, (title, chunk, rotate) in zip(axes, cases):
        for x in range(2):
            ax.add_patch(Rectangle((x * 2.3 - 0.1, -0.15), 2.1, 2.6, facecolor=XCD_SHADE[x],
                                   edgecolor=XCD_EDGE[x], lw=1.2))
            ax.text(x * 2.3 + 0.95, 2.6, f"XCD {x}", ha="center", fontsize=8, color=XCD_EDGE[x])
        owners, kranges = [0, 0], [set(), set()]
        for w in range(S * n):
            t, j = (w % S, w // S) if chunk else divmod(w, n)
            owner = j == ((t % n) if rotate else 0)
            x, slot = divmod(w, 4)
            cx, cy = x * 2.3 + (slot % 2) * 1.0, (1 - slot // 2) * 1.2
            ax.add_patch(Rectangle((cx, cy), 0.9, 1.0, facecolor=TILE[t], edgecolor="black",
                                   lw=2.2 if owner else 0.6))
            ax.text(cx + 0.45, cy + 0.62, f"T{t}", ha="center", va="center", fontsize=8)
            ax.text(cx + 0.45, cy + 0.3, f"k {j * 64}-{j * 64 + 63}", ha="center", va="center", fontsize=6)
            owners[x] += owner
            kranges[x].add(j)
        for x in range(2):
            k = len(kranges[x])
            ax.text(x * 2.3 + 0.95, -0.55, f"{k} k-range{'s' if k > 1 else ''}, {owners[x]} owners",
                    ha="center", fontsize=7)
        ax.set_xlim(-0.3, 4.6)
        ax.set_ylim(-0.8, 2.9)
        ax.axis("off")
        ax.set_title(title, fontsize=9, loc="left")
    fig.suptitle("Placement: S = 4 tiles x n = 2 chunks on 2 XCDs (thick border: owner)", fontsize=10,
                 x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "c09_chunk_major.png")


# ----------------------------------------------------------------------------- C10, C12 helpers


def box(ax, x, y, w, h, text, colour="white", edge="black", fontsize=7, **kw):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=colour, edgecolor=edge, lw=0.8, **kw))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, a, b, text=None, colour="black", style="-|>", ls="-", rad=0.0, fontsize=7, offset=(0, 0)):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, color=colour, lw=1.0, linestyle=ls,
                                 mutation_scale=9, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((a[0] + b[0]) / 2 + offset[0], (a[1] + b[1]) / 2 + offset[1], text, fontsize=fontsize,
                ha="center", va="center", color=colour)


def c10_atomic():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.5, 2.9), gridspec_kw=dict(width_ratios=[1, 1.05]))
    for j in range(4):
        box(ax, j * 1.2, 3.0, 1.0, 0.7, f"CU {j}", TILE[0])
        arrow(ax, (j * 1.2 + 0.5, 3.0), (j * 1.2 + 0.5, 1.15), "+=" if j == 0 else None, offset=(0.25, 0))
    box(ax, 0, 0.3, 4.6, 0.8, "C tile (bf16), zeroed first, 1/4 by each CU", "#f2f2f2")
    ax.text(5.0, 2.2, "no workspace\nno flags\nno barrier\nno slice read", ha="left", va="center",
            fontsize=8, color="#b2182b")
    ax.text(2.3, 2.1, "buffer_atomic_pk_add_bf16", ha="center", fontsize=7,
            bbox=dict(facecolor="white", edgecolor="none"))
    ax.set_xlim(-0.2, 6.6)
    ax.set_ylim(0, 4.0)
    ax.axis("off")
    ax.set_title("Each CU adds its partial straight into C", fontsize=10, loc="left")

    ns = [2, 4, 8, 16, 32, 64]
    bx.plot(ns, [0.5 + 0.75 * math.sqrt(n) for n in ns], "o-", color=STORE,
            label="bf16 atomics: 0.5 + 0.75 sqrt(n), not reproducible")
    bx.plot(ns, [1.6] * len(ns), "s-", color=SLICE, label="bf16 partials, fp32 sums: about 1.6")
    bx.plot(ns, [1.0] * len(ns), "-", color="grey", label="fp32 partials: 1 (one rounding)")
    bx.set_xscale("log", base=2)
    bx.set_xticks(ns, [str(n) for n in ns], fontsize=7)
    bx.tick_params(axis="y", labelsize=7)
    bx.set_xlabel("n, CUs per tile", fontsize=8)
    bx.set_ylabel("mean error of C\n(in roundings)", fontsize=8)
    bx.legend(fontsize=7, frameon=False, loc="upper left")
    bx.set_ylim(0, 7.5)
    bx.set_title("The price: error grows with n", fontsize=10, loc="left")
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)
    fig.tight_layout()
    save(fig, "c10_atomic.png")


def c12_l2():
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.0))
    for ax, local in zip(axes, (False, True)):
        for x in range(2):
            box(ax, x * 3.2, 1.3, 2.9, 2.3, None, XCD_SHADE[x], edge=XCD_EDGE[x])
            ax.text(x * 3.2 + 1.45, 3.75, f"XCD {x}", ha="center", fontsize=8, color=XCD_EDGE[x])
            box(ax, x * 3.2 + 0.2, 1.5, 2.5, 0.5, "L2", "white", edge=XCD_EDGE[x])
            for c in range(4):
                mine = x == 0 if local else c in ((2, 3) if x == 0 else (0, 1))
                box(ax, x * 3.2 + 0.2 + c * 0.65, 2.6, 0.55, 0.7, None, TILE[0] if mine else "white")
        box(ax, 0, 0.0, 6.1, 0.5, "Infinity Cache / HBM", "#f2f2f2")
        if local:
            for c in range(4):
                arrow(ax, (0.47 + c * 0.65, 2.6), (0.47 + c * 0.65, 2.0), style="<|-|>")
            arrow(ax, (1.45, 0.5), (1.45, 1.5), colour="grey", ls="--")
            ax.text(1.6, 0.9, "but the first read of a written line\nstill fetches it from HBM", fontsize=7,
                    color="dimgrey", va="center")
            ax.set_title("l2: a tile's CUs on one XCD; partials and flags\nstay in its L2. Barrier about 1 us",
                         fontsize=9, loc="left")
        else:
            for x, cs in ((0, (2, 3)), (1, (0, 1))):
                for c in cs:
                    cx = x * 3.2 + 0.47 + c * 0.65
                    arrow(ax, (cx, 2.6), (cx, 0.5), style="<|-|>")
            ax.set_title("wt: partials written through to memory, flags as\ndevice-wide atomics. Barrier 3-9 us",
                         fontsize=9, loc="left")
        ax.set_xlim(-0.2, 6.4)
        ax.set_ylim(-0.1, 4.0)
        ax.axis("off")
    fig.suptitle("Where a tile's partials are exchanged (coloured: the 4 CUs of one tile)", fontsize=10,
                 x=0.01, ha="left", y=1.04)
    fig.tight_layout()
    save(fig, "c12_l2.png")


def main():
    for f in (c01_dp, c02_one_tile, c03_two_tile, c04_e2e, c05_aligned, c06_order, c07_owner, c08_rs,
              c09_chunk_major, c10_atomic, c11_bf16, c12_l2):
        f()


if __name__ == "__main__":
    main()
