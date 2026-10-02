"""Before/after diagrams for the stream-K versions v15-v20, used by ../STREAMK.md.

Every figure replays the work assignment of two versions on a small toy problem (8 programs on
2 XCDs of 4, 16 k-steps per tile) and draws both with the same grammar:

- top: one row per program, time in k-steps along x. A bar is a segment, coloured by tile and
  labelled with its k-range; the owner's segment is hatched. Store, wait, read, refill, slice
  and C-store glyphs follow it.
- below, one strip per XCD: the k-step each program of that XCD is reading at each moment. The
  band around each line is the ~2 k-step L2 residency window on MI355X (128 KiB of L2 per CU
  against 64 KiB of A/B read per k-step). Overlapping bands can share L2 lines, assuming the
  tiles share A rows, as the leftover tiles of the real shapes do.

The partition, the peer walk and the tile-aligned split come from streamk_diagrams.py; the
timeline adds a pipeline refill between segments, a C store and the reduce-scatter ending.
Costs are illustrative k-steps (COST), not measured ones.

    python streamk_versions_diagrams.py     # checks, a summary table, images/versions/*.png
"""

import argparse
import math
import os
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from streamk_diagrams import (  # noqa: E402
    ORIGINAL,
    REVERSED,
    Problem,
    aligned_split,
    build_schedule,
    even_chunks,
    self_check,
)

P, PPX, IPT = 8, 4, 16
NUM_XCDS = P // PPX
COST = dict(store=1.0, read=1.0, refill=0.5)
WINDOW = 2.0

# No oranges: the partial store is orange.
TILE_COLOURS = ["#66c2a5", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494", "#80b1d3",
                "#bc80bd", "#b3de69", "#fccde5", "#ccebc5", "#bebada"]
STYLE = {
    "store": dict(facecolor="#d95f02", edgecolor="black"),
    "wait": dict(facecolor="#e6e6e6", edgecolor="grey", hatch=".."),
    "read": dict(facecolor="#4d4d4d", edgecolor="black"),
    "refill": dict(facecolor="white", edgecolor="grey", hatch="xxx"),
    "slice": dict(facecolor="#1b7837", edgecolor="black"),
}
XCD_SHADE = ["#eef4fb", "#fdf3e9"]
XCD_LINE = ["#2166ac", "#c2410c"]
FULL_WAVE = "#d9d9d9"


@dataclass
class Seg:
    pid: int
    tile: int
    k0: int
    k1: int
    role: str  # "complete", "owner", "partial" or "rs" (reduce-scatter member)
    peers: list = field(default_factory=list)


@dataclass
class Ev:
    kind: str  # "mac", "refill", "store", "wait", "read", "slice" or "cstore"
    t0: float
    t1: float
    seg: Seg
    peer: int = -1


def xcd(p):
    return p // PPX


# ----------------------------------------------------------------------------- work assignment
#
# A work assignment maps each program to its segments in the order it processes them.


def data_parallel(S):
    """v13's last wave: program p < S runs leftover tile p whole; the rest idle."""
    return {p: [Seg(p, p, 0, IPT, "complete")] if p < S else [] for p in range(P)}


def end_to_end(S, variant):
    """v15/v16 (forward), v19 two-tile and v20 spread (reversed): the k-steps of the S tiles laid
    end to end and cut evenly over the programs."""
    pr = Problem(f"{S} tiles", P, S, IPT, pids_per_xcd=PPX)
    sched = build_schedule(pr, variant)
    self_check(sched)
    roles = {"reduce": "owner"}
    work = {}
    for p, ss in sched.segs.items():
        order = ss if variant == ORIGINAL else ss[::-1]
        work[p] = [Seg(p, s.tile, s.k0, s.k1, roles.get(s.role, s.role), list(s.peers)) for s in order]
    return work


def tile_aligned(S, ending="owner", chunk_major=False):
    """v17 (tile-major, chunk-0 owner), v18 (chunk-major, owner t % n) and v20 (reduce-scatter)."""
    items = []
    for t, chunks in enumerate(aligned_split(S, P, IPT)):
        k = 0
        for j, n in enumerate(chunks):
            items.append((t, j, k, k + n))
            k += n
    if chunk_major:
        items.sort(key=lambda it: (it[1], it[0]))
    by_tile = {}
    for p, (t, j, k0, k1) in enumerate(items):
        by_tile.setdefault(t, []).append((p, j, k0, k1))
    work = {p: [] for p in range(P)}
    for t, members in by_tile.items():
        n = len(members)
        owner_j = t % n if chunk_major else 0
        for p, j, k0, k1 in members:
            others = [q for q, jq, *_ in sorted(members, key=lambda m: (m[1] - owner_j) % n) if q != p]
            if n == 1:
                role, peers = "complete", []
            elif ending == "rs":
                role, peers = "rs", others
            elif j == owner_j:
                role, peers = "owner", others
            else:
                role, peers = "partial", []
            work[p].append(Seg(p, t, k0, k1, role, peers))
    return work


def check_work(work, S):
    """Invariants every version relies on: each k-step of each tile is computed once, each split
    tile has one owner whose peers are exactly its other programs (or all programs reduce-scatter),
    each program publishes at most one partial and publishes it first, and an owner collects only
    after all its own work."""
    cover = {(t, k): 0 for t in range(S) for k in range(IPT)}
    by_tile = {}
    for p, ss in work.items():
        assert sum(s.role in ("partial", "rs") for s in ss) <= 1, f"p{p} needs two workspace slots"
        for i, s in enumerate(ss):
            assert s.pid == p
            if s.role in ("partial", "rs"):
                assert i == 0, f"p{p}: its partial is not the first thing it does"
            if s.role == "owner":
                assert i == len(ss) - 1, f"p{p}: owner segment is not its last"
            for k in range(s.k0, s.k1):
                cover[(s.tile, k)] += 1
            by_tile.setdefault(s.tile, []).append(s)
    assert all(v == 1 for v in cover.values()), "k-steps not covered exactly once"
    for t, ss in by_tile.items():
        pids = {s.pid for s in ss}
        if len(ss) == 1:
            assert ss[0].role == "complete" and (ss[0].k0, ss[0].k1) == (0, IPT)
        elif any(s.role == "rs" for s in ss):
            assert all(s.role == "rs" and set(s.peers) == pids - {s.pid} for s in ss)
        else:
            owners = [s for s in ss if s.role == "owner"]
            assert len(owners) == 1, f"tile {t}: {len(owners)} owners"
            assert set(owners[0].peers) == pids - {owners[0].pid}
            assert all(s.role == "partial" for s in ss if s is not owners[0])
    return work


# ----------------------------------------------------------------------------- timeline


def timeline(work, cost=COST):
    """Lockstep replay: one k-step per unit of time, a refill before every segment but the
    first, then the segment's ending."""
    publish = {}
    for p, ss in work.items():
        if ss and ss[0].role in ("partial", "rs"):
            publish[p] = ss[0].k1 - ss[0].k0 + cost["store"]
    release = {s.tile: max(publish[q] for q in [p] + s.peers)
               for p, ss in work.items() for s in ss if s.role == "rs"}
    events = {}
    for p, ss in work.items():
        t, ev = 0.0, []
        for i, s in enumerate(ss):
            if i:
                ev.append(Ev("refill", t, t + cost["refill"], s))
                t += cost["refill"]
            ev.append(Ev("mac", t, t + s.k1 - s.k0, s))
            t += s.k1 - s.k0
            if s.role in ("partial", "rs"):
                ev.append(Ev("store", t, t + cost["store"], s))
                t += cost["store"]
            if s.role == "owner":
                for q in s.peers:
                    if publish[q] > t:
                        ev.append(Ev("wait", t, publish[q], s, q))
                        t = publish[q]
                    ev.append(Ev("read", t, t + cost["read"], s, q))
                    t += cost["read"]
            elif s.role == "rs":
                if release[s.tile] > t:
                    ev.append(Ev("wait", t, release[s.tile], s))
                    t = release[s.tile]
                n = len(s.peers) + 1
                ev.append(Ev("slice", t, t + (n - 1) / n * cost["read"], s))
                t += (n - 1) / n * cost["read"]
            if s.role != "partial":
                ev.append(Ev("cstore", t, t, s))
        events[p] = ev
    return events


def finish(events):
    return max((ev[-1].t1 for ev in events.values() if ev), default=0.0)


def distinct_k(events, x, step=0.25):
    """Mean and max number of distinct k-steps the programs of XCD x read at the same moment."""
    macs = [e for p, ev in events.items() if xcd(p) == x for e in ev if e.kind == "mac"]
    counts, t = [], step / 2
    while t < finish(events):
        ks = {math.floor(e.seg.k0 + t - e.t0) for e in macs if e.t0 <= t < e.t1}
        if ks:
            counts.append(len(ks))
        t += step
    return (sum(counts) / len(counts), max(counts)) if counts else (0.0, 0)


# ----------------------------------------------------------------------------- drawing


def seg_label(s):
    return f"T{s.tile} k{s.k0}" if s.k1 - s.k0 == 1 else f"T{s.tile} k{s.k0}-{s.k1 - 1}"


def xcd_bands(ax):
    for x in range(NUM_XCDS):
        ax.axhspan(x * PPX - 0.5, (x + 1) * PPX - 0.5, color=XCD_SHADE[x % 2], zorder=0)
        ax.text(-0.065, (x + 0.5) * PPX - 0.5, f"XCD {x}", transform=ax.get_yaxis_transform(),
                rotation=90, ha="center", va="center", fontsize=8, color=XCD_LINE[x % 2], weight="bold")


def titles(fig, title, subtitle, x=0.07):
    h = fig.get_figheight()
    fig.text(x, 1 - 0.15 / h, title, fontsize=12, weight="bold", va="top")
    fig.text(x, 1 - 0.45 / h, subtitle, fontsize=9.5, color="#444444", va="top")


def title_top(fig):
    """Figure fraction below the title block."""
    return 1 - 0.8 / fig.get_figheight()


def draw_timeline(ax, events, xmax, title, offset=0.0, arrows=True):
    h = 0.64
    xcd_bands(ax)
    publish = {p: e.t1 for p, ev in events.items() for e in ev if e.kind == "store"}
    for p, ev in events.items():
        if not ev:
            ax.text(offset + 0.4, p, "idle", va="center", fontsize=8, color="#b2182b", style="italic")
        for e in ev:
            t0, w = e.t0 + offset, e.t1 - e.t0
            if e.kind == "mac":
                s = e.seg
                ax.add_patch(Rectangle((t0, p - h / 2), w, h, facecolor=TILE_COLOURS[s.tile % 12],
                                       edgecolor="black", lw=0.9, zorder=2,
                                       hatch="///" if s.role == "owner" else None))
                if w >= 2.4:
                    ax.text(t0 + w / 2, p, seg_label(s), ha="center", va="center", fontsize=7, zorder=3,
                            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))
            elif e.kind == "cstore":
                ax.plot(t0, p, marker="D", ms=4.5, color="black", zorder=4)
            else:
                ax.add_patch(Rectangle((t0, p - h / 2), w, h, zorder=2, **STYLE[e.kind]))
                if e.kind == "read":
                    if w >= 0.9:
                        ax.text(t0 + w / 2, p, f"+p{e.peer}", ha="center", va="center", fontsize=6,
                                color="white", zorder=3)
                    if arrows:
                        ax.annotate("", xy=(t0 + 0.05, p), xytext=(publish[e.peer] + offset, e.peer),
                                    arrowprops=dict(arrowstyle="->", color="#7a0177", lw=0.8,
                                                    shrinkA=0, shrinkB=0), zorder=5)
    ax.axvline(finish(events) + offset, color="#b2182b", ls=":", lw=1.3, zorder=1)
    ax.set_xlim(0, xmax)
    ax.set_ylim(P - 0.5, -0.5)
    ax.set_yticks(range(P), [f"p{p}" for p in range(P)], fontsize=7.5)
    ax.set_title(title, fontsize=9.5, loc="left")
    ax.grid(axis="x", color="#e5e5e5", zorder=0)
    ax.set_axisbelow(True)


def draw_klines(ax, events, x, xmax):
    colour = XCD_LINE[x % 2]
    for p, ev in events.items():
        if xcd(p) != x:
            continue
        for e in ev:
            if e.kind != "mac":
                continue
            ts, ks = [e.t0, e.t1], [e.seg.k0, e.seg.k1]
            ax.fill_between(ts, [k - WINDOW / 2 for k in ks], [k + WINDOW / 2 for k in ks],
                            color=colour, alpha=0.12, lw=0)
            ax.plot(ts, ks, color=colour, lw=1.5, alpha=0.75)
    mean, mx = distinct_k(events, x)
    ax.set_facecolor(XCD_SHADE[x % 2])
    ax.set_xlim(0, xmax)
    ax.set_ylim(-1.2, IPT + 1.2)
    ax.set_yticks([0, IPT // 2, IPT])
    ax.tick_params(labelsize=7)
    ax.set_ylabel(f"XCD {x}\nk-step", fontsize=7.5, color=colour)
    note = f"distinct k-steps read at once: mean {mean:.1f}, max {mx}" if mx else "no reads: CUs idle"
    ax.text(0.995, 0.93, note, transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
    ax.grid(axis="x", color="#e5e5e5")


def legend(fig, tiles, kinds, klines=True, y=0.0):
    handles = [Patch(facecolor=TILE_COLOURS[t % 12], edgecolor="black", label=f"tile T{t}") for t in tiles]
    handles.append(Patch(facecolor="white", edgecolor="black", hatch="///", label="owner's segment"))
    names = dict(store="store partial + flag", wait="wait", read="owner reads a partial",
                 refill="pipeline refill (tile switch)", slice="sum 1/n of the tile (reduce-scatter)")
    handles += [Patch(**STYLE[k], label=names[k]) for k in ("store", "wait", "read", "refill", "slice")
                if k in kinds]
    handles.append(Line2D([], [], marker="D", ms=5, color="black", lw=0, label="write C"))
    if klines:
        handles.append(Patch(facecolor=XCD_LINE[0], alpha=0.25, label=f"k-step read, +-{WINDOW / 2:g} band"))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 8), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, y))


def kinds_of(*evsets):
    return {e.kind for events in evsets for ev in events.values() for e in ev}


def tiles_of(*evsets):
    return sorted({e.seg.tile for events in evsets for ev in events.values() for e in ev})


def fig_pair(path, title, subtitle, left, right):
    """Previous version on the left, this version on the right, on identical axes."""
    xmax = max(finish(ev) for _, ev in (left, right)) + 1.0
    fig = plt.figure(figsize=(15, 7.6))
    gs = fig.add_gridspec(1 + NUM_XCDS, 2, height_ratios=[3.3] + [1.15] * NUM_XCDS, hspace=0.12,
                          wspace=0.17, left=0.07, right=0.99, top=title_top(fig) - 0.03, bottom=0.14)
    for col, (panel, events) in enumerate((left, right)):
        ax = fig.add_subplot(gs[0, col])
        draw_timeline(ax, events, xmax, f"{panel}\ntail: {finish(events):.1f} k-steps")
        ax.tick_params(labelbottom=False)
        for x in range(NUM_XCDS):
            axk = fig.add_subplot(gs[1 + x, col], sharex=ax)
            draw_klines(axk, events, x, xmax)
            if x < NUM_XCDS - 1:
                axk.tick_params(labelbottom=False)
        axk.set_xlabel("time since the stream-K tail started (k-steps)", fontsize=8.5)
    legend(fig, tiles_of(left[1], right[1]), kinds_of(left[1], right[1]))
    titles(fig, title, subtitle)
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- figures

TOY = f"{P} programs on {NUM_XCDS} XCDs, {IPT} k-steps per tile"
COSTS = (f"Illustrative costs: store a partial {COST['store']:g}, read one {COST['read']:g}, "
         f"refill {COST['refill']:g} k-steps")


def fig_f0(path, S=3):
    """Data-parallel last wave against one-tile stream-K, whole kernel (one full wave + tail)."""
    dp, sk = timeline(check_work(data_parallel(S), S)), timeline(check_work(end_to_end(S, ORIGINAL), S))
    xmax = IPT + max(finish(dp), finish(sk)) + 1.0
    fig, axes = plt.subplots(1, 2, figsize=(15, 3.9), sharey=True)
    for ax, events, title in ((axes[0], dp, f"Data-parallel: the last wave runs {S} tiles on {P} CUs"),
                              (axes[1], sk, f"Stream-K: the last wave's {S * IPT} k-steps split over all {P} CUs")):
        for p in range(P):
            ax.add_patch(Rectangle((0, p - 0.32), IPT, 0.64, facecolor=FULL_WAVE, edgecolor="black", lw=0.6))
            ax.text(IPT / 2, p, "full wave: one whole tile", ha="center", va="center", fontsize=7, color="#555555")
        draw_timeline(ax, events, xmax, f"{title}\nkernel: {IPT + finish(events):.1f} k-steps", offset=IPT)
        ax.axvline(IPT, color="black", lw=1.0)
        ax.set_xlabel("time since the kernel started (k-steps)", fontsize=8.5)
    legend(fig, tiles_of(dp, sk), kinds_of(dp, sk), klines=False)
    fig.tight_layout(rect=(0.02, 0.08, 1, title_top(fig)))
    titles(fig, "F0. Stream-K fills the idle CUs of a partial last wave",
           f"{TOY}; one full wave, then {S} leftover tiles. {COSTS}.", x=0.03)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return dp, sk


def fig_f1(path, S=3):
    """Iteration space and per-CU load: data-parallel, split-K 2 and 3 ways, stream-K."""

    def split_k(s):
        pieces, unit = [], 0
        for t in range(S):
            k = 0
            for n in even_chunks(IPT, s):
                pieces.append((unit % P, t, k, k + n, unit >= P))
                k, unit = k + n, unit + 1
        return pieces

    sk = end_to_end(S, ORIGINAL)
    rows = [
        ("data-parallel: one program per tile", split_k(1)),
        ("split-K, 2 ways: every tile cut in 2", split_k(2)),
        ("split-K, 3 ways: 9 pieces on 8 CUs", split_k(3)),
        ("stream-K: k-steps split evenly, cuts fall anywhere",
         [(p, s.tile, s.k0, s.k1, False) for p, ss in sk.items() for s in ss]),
    ]
    fig, axes = plt.subplots(len(rows), 2, figsize=(15, 8.2), gridspec_kw=dict(width_ratios=[2.2, 1]))
    for (title, pieces), (ax_s, ax_l) in zip(rows, axes):
        cover = [0] * (S * IPT)
        load = [0] * P
        for p, t, k0, k1, second in pieces:
            for k in range(k0, k1):
                cover[t * IPT + k] += 1
            load[p] += k1 - k0
            x = t * IPT + k0
            ax_s.add_patch(Rectangle((x, 0), k1 - k0, 1, facecolor=TILE_COLOURS[t], edgecolor="black", lw=1.3))
            ax_s.text(x + (k1 - k0) / 2, 0.5, f"p{p}" + (" (2nd)" if second else ""), ha="center",
                      va="center", fontsize=7.5 if k1 - k0 > 2 else 6.5)
        assert all(c == 1 for c in cover), title
        for t in range(S + 1):
            ax_s.axvline(t * IPT, color="black", lw=2, ls="--")
        for t in range(S):
            ax_s.text(t * IPT + IPT / 2, 1.18, f"tile T{t}", ha="center", fontsize=8)
        ax_s.set_xlim(-0.3, S * IPT + 0.3)
        ax_s.set_ylim(-0.2, 1.45)
        ax_s.set_yticks([])
        ax_s.set_xticks(range(0, S * IPT + 1, 4))
        ax_s.tick_params(labelsize=7)
        for spine in ("left", "top", "right"):
            ax_s.spines[spine].set_visible(False)
        ax_s.set_title(title, fontsize=9.5, loc="left")
        idle = sum(v == 0 for v in load)
        ax_l.barh(range(P), load, color=["#bdbdbd" if v else "white" for v in load], edgecolor="black")
        for p, v in enumerate(load):
            if not v:
                ax_l.text(0.3, p, "idle", va="center", fontsize=7, color="#b2182b", style="italic")
        ax_l.axvline(max(load), color="#b2182b", ls=":", lw=1.3)
        ax_l.set_xlim(0, IPT + 1)
        ax_l.set_ylim(P - 0.5, -0.5)
        ax_l.set_yticks(range(P), [f"p{p}" for p in range(P)], fontsize=7)
        ax_l.tick_params(axis="x", labelsize=7)
        ax_l.set_title(f"k-steps per CU: longest {max(load)}, {idle} idle", fontsize=9.5, loc="left")
    loads = [sum(s.k1 - s.k0 for s in ss) for ss in sk.values()]
    assert max(loads) - min(loads) <= 1, "stream-K does not balance"
    axes[-1][0].set_xlabel("iteration space: tile x 16 + k-step", fontsize=8.5)
    axes[-1][1].set_xlabel("k-steps of MAC work (fixup not shown)", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, title_top(fig)))
    titles(fig, "F1. Split-K against stream-K",
           f"{S} leftover tiles x {IPT} k-steps on {P} CUs. Left: who computes which k-steps. "
           "Right: the MAC work each CU gets.", x=0.03)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_f3(path):
    """One wave's partial store instruction: v15 row-major against v16 lane-contiguous."""
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw=dict(width_ratios=[1, 1.25]))
    lane_cols = plt.cm.Blues([0.45, 0.6, 0.75, 0.9])
    rows, lines_per_row = 16, 4

    # Row-major: quadrant rows are 128 fp32 = 512 B = 4 cache lines; one instruction writes a
    # 16 x 16 fp32 block, 64 B in each of 16 rows.
    for r in range(rows):
        for c in range(lines_per_row):
            ax_l.add_patch(Rectangle((c, -r), 1, 0.8, facecolor="white", edgecolor="#999999", lw=0.8))
        for q in range(4):
            ax_l.add_patch(Rectangle((q * 0.125, -r), 0.125, 0.8, facecolor=lane_cols[q], edgecolor="none"))
        ax_l.add_patch(Rectangle((0, -r), 0.5, 0.8, facecolor="none", edgecolor="black", lw=1.2))
        ax_l.text(-0.1, -r + 0.4, f"row {r}", ha="right", va="center", fontsize=6.5)
        ax_l.text(4.08, -r + 0.4, f"lanes {r}, {r + 16}, {r + 32}, {r + 48}", va="center", fontsize=6.5,
                  color="#2166ac")
    for c in range(lines_per_row):
        ax_l.text(c + 0.5, 1.05, f"line {c}\n128 B", ha="center", fontsize=7)
    ax_l.text(2.4, -rows - 0.5, "each row: 128 fp32 = 512 B = 4 cache lines\n"
              "the instruction writes 64 B (4 lanes x 16 B) into 16 different rows:\n"
              "16 half-written cache lines, 512 B apart", ha="center", va="top", fontsize=8.5)
    ax_l.set_xlim(-1.2, 5.6)
    ax_l.set_ylim(-rows - 2.6, 1.8)
    ax_l.set_axis_off()
    ax_l.set_title("v15: row-major partial (row x 128 + col)", fontsize=10, loc="left")

    # Lane-contiguous: [16 register vectors][4 warps][64 lanes][4 fp32]; lane l writes bytes
    # 16l..16l+15 of a 1 KiB block, 8 whole lines.
    for c in range(8):
        ax_r.add_patch(Rectangle((c, 0), 1, 1.2, facecolor="white", edgecolor="#999999", lw=0.8))
        for q in range(8):
            ax_r.add_patch(Rectangle((c + q / 8, 0), 1 / 8, 1.2, facecolor=lane_cols[(c * 8 + q) // 16],
                                     edgecolor="white", lw=0.3))
        ax_r.add_patch(Rectangle((c, 0), 1, 1.2, facecolor="none", edgecolor="black", lw=1.2))
        ax_r.text(c + 0.5, 1.35, f"line {c}", ha="center", fontsize=7)
        ax_r.text(c + 0.5, -0.25, f"lanes {8 * c}-{8 * c + 7}", ha="center", fontsize=6.5)
    ax_r.annotate("", xy=(8, 1.85), xytext=(0, 1.85), arrowprops=dict(arrowstyle="<->", lw=1.1))
    ax_r.text(4, 2.0, "1 KiB, contiguous", ha="center", fontsize=8.5)
    for i, (cost, v15, v16) in enumerate((("store one partial", 4.1, 1.5), ("owner reads one partial", 4.9, 1.8))):
        ax_r.text(0, -1.4 - 0.55 * i, f"{cost}: {v15} -> {v16} k-steps", fontsize=9)
    ax_r.text(0, -2.6, "4352x4096x8192 tail: 131 us (v15) -> 67 us (v16)", fontsize=9)
    ax_r.text(0, -3.4, "the same instruction writes 8 whole cache lines; the owner reads the\n"
              "partial back with the same map, so neither side converts layouts", fontsize=8.5, va="top")
    ax_r.set_xlim(-0.3, 8.3)
    ax_r.set_ylim(-5.0, 2.6)
    ax_r.set_axis_off()
    ax_r.set_title("v16: lane-contiguous partial [16 register vectors][4 warps][64 lanes][4 fp32]",
                   fontsize=10, loc="left")
    handles = [Patch(facecolor=lane_cols[q], label=f"lanes {16 * q}-{16 * q + 15}") for q in range(4)]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, title_top(fig)))
    titles(fig, "F3. v15 -> v16: lane-contiguous partials",
           "Where one wave's store instruction lands in the workspace: 64 lanes x 16 B (4 fp32) = 1 KiB. "
           "Lane l holds row l % 16, columns 4 (l // 16) .. +3 of a 16 x 16 block.", x=0.03)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def summary_row(fig, panel, events):
    ks = [distinct_k(events, x) for x in range(NUM_XCDS)]
    return (f"| {fig} | {panel} | {finish(events):.1f} | "
            + ", ".join(f"{m:.1f} / {mx}" for m, mx in ks) + " |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "versions"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    out = lambda name: os.path.join(args.out, name)  # noqa: E731
    rows = []

    dp, sk = fig_f0(out("f0_dp_vs_streamk.png"))
    assert sum(not ev for ev in dp.values()) == P - 3 and all(sk.values())
    rows += [summary_row("F0", "data-parallel tail", dp), summary_row("F0", "stream-K tail", sk)]

    fig_f1(out("f1_splitk_vs_streamk.png"))

    # F2: v13 -> v15, three leftover tiles.
    v13 = timeline(check_work(data_parallel(3), 3))
    v15w = check_work(end_to_end(3, ORIGINAL), 3)
    assert any(len(ss) > 1 for ss in v15w.values()), "toy should have a tile-crossing program"
    v15 = timeline(v15w)
    fig_pair(out("f2_v13_to_v15.png"), "F2. v13 -> v15: one-tile stream-K, cut end to end",
             f"3 leftover tiles, {TOY}. {COSTS}.",
             ("v13, data-parallel last wave: 3 CUs busy, 5 idle", v13),
             ("v15, end-to-end cuts: seams drift, crossings, serial owner fixup", v15))
    rows += [summary_row("F2", "v13", v13), summary_row("F2", "v15", v15)]

    fig_f3(out("f3_v15_to_v16_layout.png"))

    # F4: v16 -> v17, seven leftover tiles: long ranges, where drift costs L2.
    v16 = timeline(check_work(end_to_end(7, ORIGINAL), 7))
    v17w = check_work(tile_aligned(7), 7)
    assert all(len(ss) <= 1 for ss in v17w.values()), "tile-aligned: one segment per program"
    v17 = timeline(v17w)
    assert max(distinct_k(v17, x)[1] for x in range(NUM_XCDS)) < max(distinct_k(v16, x)[1] for x in range(NUM_XCDS))
    fig_pair(out("f4_v16_to_v17.png"), "F4. v16 -> v17: tile-aligned cuts",
             f"7 leftover tiles (long ranges, like 3840x4096x8192), {TOY}. {COSTS}.",
             ("v16, end-to-end cuts: 14 k-steps each from drifting offsets", v16),
             ("v17, tile-aligned: T0 split 2 ways, the rest whole from k-step 0", v17))
    rows += [summary_row("F4", "v16", v16), summary_row("F4", "v17", v17)]

    # F5: v17 -> v18, four tiles x two chunks.
    tm = timeline(check_work(tile_aligned(4), 4))
    cmw = check_work(tile_aligned(4, chunk_major=True), 4)
    cm = timeline(cmw)
    owner_xcds = {xcd(s.pid) for ss in cmw.values() for s in ss if s.role == "owner"}
    assert owner_xcds == set(range(NUM_XCDS)), "rotating owner should spread owners over the XCDs"
    assert all(distinct_k(cm, x)[1] == 1 for x in range(NUM_XCDS))
    fig_pair(out("f5_v17_to_v18.png"), "F5. v17 -> v18: chunk-major placement with a rotating owner",
             f"4 leftover tiles, 2 programs each, {TOY}. {COSTS}.",
             ("v17, tile-major: an XCD holds whole tiles (chunks 0 and 1)", tm),
             ("v18, chunk-major: an XCD holds one chunk of every tile; owner = chunk t % n", cm))
    rows += [summary_row("F5", "v17", tm), summary_row("F5", "v18", cm)]

    # F6: forward two-tile -> v19 reversed. One full wave (8 tiles) plus 2 leftover: all 10 tiles
    # are stream-K, each program gets 20 k-steps, lag d = 16 x 2 / 8 = 4.
    fwd = timeline(check_work(end_to_end(10, ORIGINAL), 10))
    revw = check_work(end_to_end(10, REVERSED), 10)
    assert all(ss[0].k0 == 0 for ss in revw.values()), "reversed: every program starts at k-step 0"
    assert all(len(s.peers) <= 1 for ss in revw.values() for s in ss), "two-tile: at most one peer"
    rev = timeline(revw)
    assert all(distinct_k(rev, x)[1] <= 2 for x in range(NUM_XCDS)), "reversed: two groups per XCD"
    assert sum(distinct_k(fwd, x)[0] for x in range(NUM_XCDS)) > sum(distinct_k(rev, x)[0] for x in range(NUM_XCDS))
    fig_pair(out("f6_forward_to_v19.png"), "F6. Forward two-tile -> v19 reversed two-tile",
             f"One full wave + 2 leftover tiles, all 10 in stream-K (20 k-steps each, lag d = 4), {TOY}. "
             f"{COSTS}.",
             ("forward two-tile (v15/v16 policy): every program starts at its own k offset", fwd),
             ("v19, reversed: heads first from k-step 0; two groups d apart per XCD", rev))
    rows += [summary_row("F6", "forward", fwd), summary_row("F6", "v19 reversed", rev)]

    # F7: v17 owner ending -> v20 reduce-scatter, two tiles x four chunks.
    own = timeline(check_work(tile_aligned(2), 2))
    rs = timeline(check_work(tile_aligned(2, ending="rs"), 2))
    assert finish(rs) < finish(own)
    fig_pair(out("f7_v17_to_v20_reduce_scatter.png"), "F7. v17 -> v20: reduce-scatter ending",
             f"2 leftover tiles, 4 programs each, {TOY}. {COSTS}.",
             ("v17, owner ending: chunk 0 reads 3 partials one after another", own),
             ("v20, reduce-scatter: all store, wait for the tile, each sums 1/4", rs))
    rows += [summary_row("F7", "v17 owner", own), summary_row("F7", "v20 rs", rs)]

    # F8: S > P/2. Tile-aligned leaves most tiles whole; v20's spread cuts end to end, reversed.
    al = timeline(check_work(tile_aligned(6), 6))
    spw = check_work(end_to_end(6, REVERSED), 6)
    sp = timeline(spw)
    assert finish(sp) < finish(al)
    fig_pair(out("f8_v20_spread.png"), "F8. More than half a wave left over: v17 tile-aligned -> v20 spread",
             f"6 leftover tiles (S > P / 2), {TOY}. {COSTS}.",
             ("v17, tile-aligned: 4 tiles stay whole, so the tail is a full tile", al),
             ("v20 spread: 12 k-steps each, reversed order", sp))
    rows += [summary_row("F8", "v17 tile-aligned", al), summary_row("F8", "v20 spread", sp)]

    print("all checks passed\n")
    print("| figure | panel | tail (k-steps) | distinct k per XCD, mean / max |")
    print("|---|---|---|---|")
    print("\n".join(rows))
    print(f"\nimages written to {args.out}")


if __name__ == "__main__":
    main()
