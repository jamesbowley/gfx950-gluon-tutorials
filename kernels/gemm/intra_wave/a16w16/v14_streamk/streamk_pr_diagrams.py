"""Uniform schematic figures for the stream-K PR description, used by ../STREAMK_PR.md.

Every figure uses one toy problem: 8 CUs standing for 256, a full wave of tiles T0-T7, S = 3 leftover
tiles T8-T10 and 48 k-steps per tile. Ending costs are v23's measured phase medians converted to k-steps
(v23_streamk_rs_band/results/fixup_timing_e2.md, 1 k-step = 1.21 us): a partial store of about 4.7 us
for owner peers and rs alike, a flag or barrier of about 2 us, 2.7 us per owner read, and a slice
read of about 4.7 us when the n producers read (owner ending about 7.5 + 2.2 (n - 1), rs about 10).

    python streamk_pr_diagrams.py     # images/pr/p*.png
"""

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle  # noqa: E402

from streamk_concepts_diagrams import aligned, e2e  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "images", "pr")

# ----------------------------------------------------------------------------- style

WIDTH, DPI = 8.0, 300
TITLE_PT, LABEL_PT, SMALL_PT = 11, 9, 8

# matplotlib's font cache does not pick up the system's Inter OTFs, so register them directly.
for _path in glob.glob("/usr/share/fonts/opentype/inter/Inter-*.otf"):
    fm.fontManager.addfont(_path)
FONT = "Inter" if any(f.name == "Inter" for f in fm.fontManager.ttflist) else "DejaVu Sans"

INK, SOFT = "#222222", "#555555"
plt.rcParams.update({
    "font.family": FONT,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": SOFT,
    "ytick.color": INK,
    "axes.edgecolor": "#888888",
    "axes.spines.left": False,
    "axes.grid": True,
    "axes.grid.axis": "x",
    "axes.axisbelow": True,
    "grid.color": "#e8e8e8",
    "grid.linewidth": 0.8,
    "font.size": LABEL_PT,
    "axes.titlesize": LABEL_PT,
    "axes.titlelocation": "left",
    "axes.labelsize": LABEL_PT,
    "xtick.labelsize": SMALL_PT,
    "ytick.labelsize": SMALL_PT,
    "legend.fontsize": SMALL_PT,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "hatch.linewidth": 0.6,
})

FULL = "#d9d9d9"                                  # full-wave tiles and the MAC phase
LEFT = ["#66c2a5", "#8da0cb", "#e78ac3"]           # leftover tiles T8, T9, T10
IDLE = dict(facecolor="white", edgecolor="#9a9a9a", hatch="///", linewidth=0.5)
STORE, WAITC, READ, SLICE, CSTORE = "#d95f02", "#f0f0f0", "#4d4d4d", "#1b7837", "#7b3294"
WAIT = dict(facecolor=WAITC, edgecolor="#9a9a9a", hatch="..", linewidth=0.5)
CU_LINE = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
XCD_SHADE, XCD_EDGE = ["#eef4fb", "#fdf3e9"], ["#2166ac", "#c2410c"]
DASH = dict(color="#333333", linestyle=(0, (4, 2)), lw=1.1)
GHOST = dict(color="#bbbbbb", linestyle=(0, (4, 2)), lw=1.1)
EDGE, SEP = "white", 1.0                           # bar segments are separated by thin white gaps
CALLOUT = "#c0392b"

CUS, FULL_TILES, S, IPT = 8, 8, 3, 48
CU_LABELS = [f"CU {c}" for c in range(CUS)]


def tile_colour(t):
    return FULL if t < FULL_TILES else LEFT[t - FULL_TILES]


def bar(ax, row, t0, t1, colour=None, label=None, fontsize=SMALL_PT, textcolour=INK, h=0.72, **kw):
    style = dict(facecolor=colour, edgecolor=EDGE, linewidth=SEP) | kw
    ax.add_patch(Rectangle((t0, row - h / 2), t1 - t0, h, **style))
    if label:
        ax.text((t0 + t1) / 2, row, label, ha="center", va="center", fontsize=fontsize, color=textcolour)


def gap_callout(ax, x0, x1, y, text):
    """A red double arrow from x0 to x1 at row y, with the takeaway written above it."""
    ax.annotate("", (x0, y), (x1, y), arrowprops=dict(arrowstyle="<|-|>", color=CALLOUT, lw=1.2,
                                                      shrinkA=0, shrinkB=0))
    ax.text((x0 + x1) / 2, y - 0.35, text, ha="center", va="bottom", fontsize=SMALL_PT, color=CALLOUT,
            fontweight="semibold", linespacing=1.1)


def note(ax, x, y, text, ha="center", **kw):
    ax.text(x, y, text, ha=ha, va="center", fontsize=SMALL_PT, color=CALLOUT, fontweight="semibold",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5), zorder=5, **kw)


def rows(ax, labels, xlabel="time (k-steps)"):
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(xlabel)


def figure(title, nrows=1, ncols=1, height=2.6, **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=(WIDTH, height), layout="constrained", **kw)
    fig.suptitle(title, fontsize=TITLE_PT, x=0.0, ha="left", fontweight="semibold")
    fig.get_layout_engine().set(h_pad=0.06, w_pad=0.06)
    return fig, axes


def legend(fig, handles):
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), handlelength=1.6,
               columnspacing=1.6)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(path)


def tile_patches():
    return ([Patch(facecolor=FULL, edgecolor=EDGE, lw=SEP, label="full-wave tile")]
            + [Patch(facecolor=c, edgecolor=EDGE, lw=SEP, label=f"T{FULL_TILES + i}")
               for i, c in enumerate(LEFT)])


IDLE_PATCH = Patch(**IDLE, label="idle CU")

# ----------------------------------------------------------------------------- toy schedules

TAIL_X = (0, 2 * IPT + 2)


def draw_dp(ax):
    for c in range(CUS):
        bar(ax, c, 0, IPT, FULL, f"T{c}")
        if c < S:
            bar(ax, c, IPT, 2 * IPT, LEFT[c], f"T{FULL_TILES + c}")
        else:
            bar(ax, c, IPT, 2 * IPT, **IDLE)
    return 2 * IPT


def one_tile_e2e():
    """Per CU, its (leftover tile, k0, k1) pieces with the leftover tiles cut end to end over all CUs."""
    return e2e(S, CUS, IPT)


def one_tile_aligned(n=CUS // S):
    """Per CU, its (leftover tile, k0, k1) piece: every tile cut the same way over n CUs, the rest idle."""
    out = [[] for _ in range(CUS)]
    for t, chunks in enumerate(aligned(S, S * n, IPT)):
        for c, k0, k1 in chunks:
            out[c] = [(t, k0, k1)]
    return out


def draw_one_tile(ax):
    """dp's full wave, then the leftover tiles' e2e pieces."""
    end = 0
    for c, pieces in enumerate(one_tile_e2e()):
        bar(ax, c, 0, IPT, FULL, f"T{c}")
        t0 = IPT
        for t, k0, k1 in pieces:
            w = k1 - k0
            bar(ax, c, t0, t0 + w, LEFT[t], f"T{FULL_TILES + t}" if w >= 6 else None)
            t0 += w
        end = max(end, t0)
    return end


def draw_two_tile(ax):
    """Full wave plus leftover tiles laid end to end and cut evenly over the CUs."""
    work = e2e(FULL_TILES + S, CUS, IPT)
    for c, pieces in enumerate(work):
        t0 = 0
        for t, k0, k1 in pieces:
            w = k1 - k0
            bar(ax, c, t0, t0 + w, tile_colour(t), f"T{t}" if w >= 6 else None,
                fontsize=SMALL_PT if w >= 10 else 7)
            t0 += w
    return t0


def tail_axis(ax):
    ax.set_xlim(*TAIL_X)
    ax.set_xticks(range(0, TAIL_X[1] + 1, IPT // 2))
    rows(ax, CU_LABELS)


# ----------------------------------------------------------------------------- P1


def p1_dp():
    fig, ax = figure("Data-parallel: the last wave leaves 256 - S CUs idle", height=2.7)
    end = draw_dp(ax)
    tail_axis(ax)
    ax.set_title("dp, S = 3 leftover tiles")
    ax.axvline(end, **DASH)
    note(ax, 1.5 * IPT, 5, f"{CUS - S} of {CUS} CUs idle for {IPT} k-steps")
    legend(fig, tile_patches() + [IDLE_PATCH, Line2D([], [], **DASH, label="dp done")])
    save(fig, "p1_dp.png")


# ----------------------------------------------------------------------------- P2


def p2_split():
    fig, axes = figure("Which tiles are split: only the leftovers, or one full wave as well", ncols=2,
                       height=2.7, sharey=True)
    dp_end = 2 * IPT
    for ax, draw, name in ((axes[0], draw_one_tile, "one-tile: T8-T10 cut over 8 CUs"),
                           (axes[1], draw_two_tile, "two-tile: T0-T10 cut over 8 CUs")):
        end = draw(ax)
        tail_axis(ax)
        ax.axvline(dp_end, **DASH)
        gap_callout(ax, end, dp_end, 3.5, f"{dp_end - end:.0f} k-steps\nsooner than dp")
        ax.set_title(f"{name}, done at {end:.0f}")
    axes[1].set_ylabel("")
    legend(fig, tile_patches() + [Line2D([], [], **DASH, label=f"dp done ({dp_end})")])
    save(fig, "p2_split.png")


# ----------------------------------------------------------------------------- P3


SWITCH = 3.4    # k-steps for a CU to move to its next tile: store the partial, drain, reload (measured)
POOR_L2 = 1.2   # illustrative MAC slowdown when a CU's k-range drifts away from its XCD neighbours' (forward e2e)
SWITCH_STYLE = dict(facecolor="#4d4d4d", edgecolor=EDGE, linewidth=SEP)


def tint(colour, f=0.6):
    r, g, b = matplotlib.colors.to_rgb(colour)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def mac_style(colour, poor):
    if not poor:
        return dict(facecolor=colour, edgecolor=EDGE, linewidth=SEP)
    return dict(facecolor=tint(colour), edgecolor=EDGE, hatchcolor=colour, hatch="////", linewidth=SEP)


def draw_pieces(ax, c, t0, pieces, poor, colour=tile_colour, first=0, range_from=12, fontsize=None):
    """One CU's (tile, k0, k1) pieces from t0, with a tile switch between pieces; returns the end time.

    Pieces at least range_from k-steps wide get "tile k-range", narrower ones the tile alone. A fixed
    fontsize labels every piece with "tile k-range" at that size."""
    for i, (t, k0, k1) in enumerate(pieces):
        if i:
            bar(ax, c, t0, t0 + SWITCH, **SWITCH_STYLE)
            t0 += SWITCH
        w = (k1 - k0) * (POOR_L2 if poor else 1.0)
        tile, ks = f"T{first + t}", f"{k0:.0f}-{k1:.0f}"
        if fontsize:
            name, size = f"{tile} {ks}", fontsize
        elif w >= range_from:
            name, size = f"{tile} {ks}", SMALL_PT
        else:
            name, size = (tile if w >= 6 else None), 7
        bar(ax, c, t0, t0 + w, label=name, fontsize=size, **mac_style(colour(first + t), poor))
        t0 += w
    return t0


def mac_legend(poor=True):
    return [Patch(**mac_style(LEFT[0], False), label="MAC, good L2 reuse"),
            Patch(**mac_style(LEFT[0], True), label="MAC, poor L2 reuse: slower")][:2 if poor else 1] + \
           [Patch(**SWITCH_STYLE, label="tile switch: store, drain, reload")]


def p3_cuts():
    fig, axes = figure("Where the cuts fall: e2e uses every CU, aligned keeps L2 reuse", ncols=2,
                       height=2.8, sharey=True)
    n = CUS // S
    for ax, work, poor, name in (
            (axes[0], one_tile_e2e(), True, f"e2e (forward): {CUS} CUs, {S * IPT // CUS} k-steps each"),
            (axes[1], one_tile_aligned(n), False, f"aligned: n = {n} per tile, {IPT // n} k-steps each")):
        end = 0
        for c, pieces in enumerate(work):
            bar(ax, c, IPT - 4, IPT, FULL)
            if pieces:
                end = max(end, draw_pieces(ax, c, IPT, pieces, poor, colour=lambda t: LEFT[t - FULL_TILES],
                                           first=FULL_TILES))
            else:
                bar(ax, c, IPT, IPT + 34, **IDLE)
        ax.axvline(end, **DASH)
        if not poor:
            note(ax, IPT + 12, CUS - 1.5, f"{CUS - S * n} CUs idle, still done first")
        ax.set_xlim(IPT - 4, IPT + 32)
        ax.set_xticks(range(IPT, IPT + 33, 8))
        rows(ax, CU_LABELS)
        ax.set_title(f"{name}, done at {end:.0f}")
    legend(fig, mac_legend() + [IDLE_PATCH])
    save(fig, "p3_cuts.png")


# ----------------------------------------------------------------------------- P4


def p4a_order_kranges():
    fig, axes = figure("Order: the k-step each CU reads over time (section 2's two-tile, CUs 0-3)",
                       ncols=2, height=2.8, sharey=True)
    work = e2e(FULL_TILES + S, CUS, IPT)[:4]
    per = (FULL_TILES + S) * IPT // CUS
    for ax, reverse, name in ((axes[0], False, "forward: 4 different k-steps at once"),
                              (axes[1], True, "reversed: every CU starts at 0, ranges stay grouped")):
        for c, pieces in enumerate(work):
            t, dy = 0, (c - 1.5) * IPT / 70 if reverse else 0
            seq = pieces[::-1] if reverse else pieces
            for i, (_, k0, k1) in enumerate(seq):
                if i:
                    t += SWITCH
                ax.plot([t, t + k1 - k0], [k0 + dy, k1 + dy], color=CU_LINE[c], lw=2.4, solid_capstyle="butt")
                t += k1 - k0
                if i == (0 if reverse else len(seq) - 1):
                    ax.plot(t, k1 + dy, "o", color="black", ms=4.5, zorder=3)
        ax.set_title(name)
        ax.set_xlabel("time (k-steps)")
        ax.set_xlim(0, per + 2 * SWITCH + 1)
        ax.set_xticks(range(0, per + 1, IPT // 3))
        ax.set_ylim(-2, IPT + 2)
        ax.set_yticks(range(0, IPT + 1, IPT // 3))
    axes[0].set_ylabel("k-step being read")
    for ax in axes:
        ax.grid(axis="y")
    axes[1].annotate("all 4 CUs start at k-step 0", (0.4, 1.5), (1.5, 0.85 * IPT), ha="left", va="center",
                     fontsize=SMALL_PT, color=CALLOUT, fontweight="semibold",
                     arrowprops=dict(arrowstyle="-|>", color=CALLOUT, lw=1.1, shrinkA=2, shrinkB=3))
    legend(fig, [Line2D([], [], color=CU_LINE[c], lw=2.4, label=f"CU {c}") for c in range(4)]
           + [Line2D([], [], marker="o", color="black", ms=4.5, lw=0, label="partial for the peer published")])
    save(fig, "p4a_order_kranges.png")


def p4b_order_timeline():
    fig, axes = figure("Order: forward loses L2 reuse, reversed keeps it (section 2's two-tile)", nrows=2,
                       height=5.0, sharex=True)
    work = e2e(FULL_TILES + S, CUS, IPT)
    for ax, reverse, name in ((axes[0], False, "forward: CUs start at scattered k-steps"),
                              (axes[1], True, "reversed: every CU starts at k-step 0")):
        end = max(draw_pieces(ax, c, 0, pieces[::-1] if reverse else pieces, poor=not reverse, fontsize=7)
                  for c, pieces in enumerate(work))
        ax.axvline(end, **DASH)
        ax.set_xlim(0, 88)
        ax.set_xticks(range(0, 89, 8))
        rows(ax, CU_LABELS, xlabel="time (k-steps)" if reverse else "")
        ax.set_title(f"{name}, done at {end:.0f}")
        if reverse:
            ax.axvline(fwd_end, **GHOST)
            gap_callout(ax, end, fwd_end, 3.5, f"{fwd_end - end:.0f} k-steps\nsooner")
        fwd_end = end
    legend(fig, mac_legend() + [Line2D([], [], **DASH, label="done")])
    save(fig, "p4b_order_timeline.png")


# ----------------------------------------------------------------------------- P5

# k-steps; a reader with R readers per tile reads SLICE_FIXED + SLICE_BYTES * n / R and stores 1/R of C.
STORE_PARTIAL, FLAG, READ_PARTIAL, STORE_C = 4.0, 1.5, 2.2, 2.0
SLICE_FIXED, SLICE_BYTES = 1.0, 3.0
MAC0 = -3


def draw_owner(ax, n):
    t = STORE_PARTIAL + FLAG
    bar(ax, 0, MAC0, 0, FULL)
    bar(ax, 0, 0, t, **WAIT)
    for i in range(1, n):
        bar(ax, 0, t, t + READ_PARTIAL, READ, str(i), textcolour="white", fontsize=7)
        t += READ_PARTIAL
    bar(ax, 0, t, t + STORE_C, CSTORE)
    for r in range(1, n):
        bar(ax, r, MAC0, 0, FULL)
        bar(ax, r, 0, STORE_PARTIAL, STORE)
    return t + STORE_C


def draw_rs(ax, producers, readers):
    t_read = STORE_PARTIAL + FLAG
    read = SLICE_FIXED + SLICE_BYTES * producers / readers
    c = STORE_C / readers
    for r in range(readers):
        if r < producers:
            bar(ax, r, MAC0, 0, FULL)
            bar(ax, r, 0, STORE_PARTIAL, STORE)
            bar(ax, r, STORE_PARTIAL, t_read, **WAIT)
        else:
            bar(ax, r, MAC0, 0, **IDLE)
            bar(ax, r, 0, t_read, **WAIT)
        bar(ax, r, t_read, t_read + read, SLICE)
        bar(ax, r, t_read + read, t_read + read + c, CSTORE)
    return t_read + read + c


def p5_endings():
    fig, axes = figure("Endings: how a tile's n partials are added into C", nrows=2, ncols=3, height=5.0,
                       sharex=True, sharey=True)
    for row, n in zip(axes, (2, 8)):
        idle = CUS - n
        ends = [draw_owner(row[0], n), draw_rs(row[1], n, n), draw_rs(row[2], n, CUS)]
        names = ["owner", "rs", f"rs all (+{idle} idle)" if idle else "rs all (= rs)"]
        best = min(ends)
        for ax, name, end in zip(row, names, ends):
            ax.axvline(best, **DASH)
            ax.set_title(f"{name}, n = {n}: {end:.0f} k-steps", fontweight="semibold" if end == best else "normal")
            ax.set_xlim(MAC0 - 0.5, 26)
            ax.set_xticks(range(0, 26, 5))
            rows(ax, CU_LABELS, xlabel="k-steps after the MAC phase" if n == 8 else "")
        if n == 8:
            gap_callout(row[0], best, ends[0], 4.5, f"+{ends[0] - best:.0f} k-steps:\n{n - 1} reads in series")
    legend(fig, [Patch(facecolor=FULL, edgecolor=EDGE, lw=SEP, label="MAC"),
                 Patch(facecolor=STORE, edgecolor=EDGE, lw=SEP, label="store partial"),
                 Patch(**WAIT, label="wait"),
                 Patch(facecolor=READ, edgecolor=EDGE, lw=SEP, label="read one partial"),
                 Patch(facecolor=SLICE, edgecolor=EDGE, lw=SEP, label="read 1/R of every partial"),
                 Patch(facecolor=CSTORE, edgecolor=EDGE, lw=SEP, label="store C"),
                 Patch(**IDLE, label="idle CU"),
                 Line2D([], [], **DASH, label="fastest")])
    save(fig, "p5_endings.png")


# ----------------------------------------------------------------------------- P6


def box(ax, x, y, w, h, text=None, colour="white", edge="#666666", lw=0.8, zorder=1):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=colour, edgecolor=edge, lw=lw, zorder=zorder))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=SMALL_PT, zorder=zorder + 1)


def arrow(ax, a, b, colour="black", ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", color=colour, lw=1.1, linestyle=ls, mutation_scale=9,
                                 shrinkA=0, shrinkB=0, zorder=3))


WRITE_C, READ_C = "black", "#2166ac"
CU_TOP, CU_Y, L2_TOP, L2_Y, MEM_TOP = 3.3, 2.6, 2.0, 1.5, 0.5


def p6_l2():
    fig, axes = figure("Barrier: keeping flags in one XCD's L2 cuts it from 3-9 us to about 1 us", ncols=2,
                       height=2.9)
    for ax, local in zip(axes, (False, True)):
        for x in range(2):
            box(ax, x * 3.2, 1.3, 2.9, 2.3, None, XCD_SHADE[x], edge=XCD_EDGE[x], lw=1.0)
            ax.text(x * 3.2 + 1.45, 3.75, f"XCD {x}", ha="center", fontsize=SMALL_PT, color=XCD_EDGE[x])
            l2_text = "L2: flags, written partials" if local and x == 0 else "L2"
            box(ax, x * 3.2 + 0.2, L2_Y, 2.5, L2_TOP - L2_Y, l2_text, "white", edge=XCD_EDGE[x],
                zorder=4)
            for c in range(4):
                mine = x == 0 if local else c in ((2, 3) if x == 0 else (0, 1))
                box(ax, x * 3.2 + 0.2 + c * 0.65, CU_Y, 0.55, CU_TOP - CU_Y, None, LEFT[0] if mine else "white",
                    lw=0.6)
                if mine:
                    cx = x * 3.2 + 0.475 + c * 0.65
                    arrow(ax, (cx - 0.12, CU_Y), (cx - 0.12, L2_TOP if local else MEM_TOP), WRITE_C)
                    arrow(ax, (cx + 0.12, MEM_TOP), (cx + 0.12, CU_Y), READ_C, ls="--")
        box(ax, 0, 0.0, 6.1, MEM_TOP, "Infinity Cache / HBM" + ("" if local else ": partials, flags"), "#f2f2f2")
        ax.set_title("L2 exchange: tile on one XCD, barrier about 1 us" if local
                     else "write-through: tile on any XCD, barrier 3-9 us")
        if local:
            ax.annotate("reads still come\nfrom memory", (2.75, 1.0), (3.3, 1.0), ha="left", va="center",
                        fontsize=SMALL_PT, color=CALLOUT, fontweight="semibold",
                        arrowprops=dict(arrowstyle="-|>", color=CALLOUT, lw=1.1, shrinkA=2, shrinkB=2))
        ax.set_xlim(-0.1, 6.2)
        ax.set_ylim(-0.1, 4.0)
        ax.axis("off")
    legend(fig, [Patch(facecolor=LEFT[0], edgecolor=EDGE, lw=SEP, label="the 4 CUs of one tile"),
                 Line2D([], [], color=WRITE_C, lw=1.1, label="write partial"),
                 Line2D([], [], color=READ_C, lw=1.1, ls="--", label="read partials")])
    save(fig, "p6_l2.png")


def main():
    for f in (p1_dp, p2_split, p3_cuts, p4a_order_kranges, p4b_order_timeline, p5_endings, p6_l2):
        f()


if __name__ == "__main__":
    main()
