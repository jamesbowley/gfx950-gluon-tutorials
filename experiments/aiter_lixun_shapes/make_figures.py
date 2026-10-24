#!/usr/bin/env python3
"""Figures for the v20 (streamk_wip, capped) comparison on the 630 shapes.

    python make_figures.py [--warm CSV --cold CSV] [--layout plain|pad] [--out DIR]

Reads results/details_capped_hbl.csv (warm) and results/details_capped_hbl_rot2048.csv (cold,
2048 MB rotating buffer) by default and writes results/figures/*.png plus results/figures/figures.md.

Every comparison is against the best library on each shape: the faster of AITER's pick and
hipBLASLt (the torch call AITER's torch libtype makes), with that kernel's tile. "Same tile"
means the faster library uses a 256x256 tile, like v9 and streamk_wip.

For a `sweep --pad512` run, --layout pad draws v9 and streamk_wip on the padded layout against the
best valid library time over both layouts (pad_report.best_fields), and adds figure 7: the spread
between fresh-buffer rounds and the padded/contiguous ratio per kernel.
"""

import argparse
import math
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from run_v20_vs_aiter import aiter_tile, fnum, key, read_rows  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(RESULTS, "figures")
RUNS = {"warm": os.path.join(RESULTS, "details_capped_hbl.csv"),
        "cold": os.path.join(RESULTS, "details_capped_hbl_rot2048.csv")}
TIE = 1.0  # percent

GRID = ("under 1 wave", "256-1023 tiles", "1024+ tiles")
GRID_COLOR = {"under 1 wave": "tab:red", "256-1023 tiles": "tab:orange", "1024+ tiles": "tab:blue"}
TILE_CAT = ("256x256", "smaller, >= half the area", "smaller, < half the area", "larger")
TILE_COLOR = {"256x256": "tab:green", "smaller, >= half the area": "tab:orange",
              "smaller, < half the area": "tab:red", "larger": "tab:purple"}
CONFIG_MARKER = {"data-parallel": "o", "one-tile owner": "^", "one-tile reduce-scatter": "s", "spread": "D"}


def pct(a, b):
    return (a / b - 1.0) * 100.0


def grid(tiles):
    return GRID[0] if tiles < 256 else (GRID[1] if tiles < 1024 else GRID[2])


def tile_cat(tile):
    try:
        m, n = (int(x) for x in tile.split("x")[:2])
    except ValueError:
        return None
    if (m, n) == (256, 256):
        return TILE_CAT[0]
    area = m * n
    if area > 256 * 256:
        return TILE_CAT[3]
    return TILE_CAT[1] if area >= 128 * 256 else TILE_CAT[2]


def config(r):
    p = r["v20_policy"]
    if p == "one_tile":
        return "one-tile reduce-scatter" if r["v20_fixup"] == "rs" else "one-tile owner"
    return {"dp": "data-parallel"}.get(p, p)


def mn_tile(spec):
    """The MxN part of a tile spec."""
    return "x".join(spec.split("x")[:2])


def output_tile(tile, library):
    """Tile in torch output MxN order. hipBLASLt names MT as column-major N x M."""
    parts = tile.split("x")
    if len(parts) >= 2 and library in ("hipBLASLt", "AITER torch", "AITER hipblaslt"):
        return f"{parts[1]}x{parts[0]}"
    return mn_tile(tile)


def load(path, layout="plain"):
    from pad_report import best_fields, times

    # plain: everything contiguous; pad: v9 and streamk_wip padded against the best library over both
    # layouts; mixed: v9 and streamk_wip padded against the libraries as AITER calls them (contiguous)
    lay = "" if layout == "plain" else "_pad"
    rows = []
    for r in read_rows(path):
        b = best_fields(r, "_pad" if layout == "pad" else "")
        best, tile = b["best"], b["best_tile"]
        # streamk_wip forced data-parallel: timed where capped splits or hands the launch to v9, and
        # the headline kernel elsewhere
        t = times(r, lay)
        our_tile = mn_tile(r.get("v20_tile", "256x256x64"))
        best_output_tile = output_tile(tile, b["best_lib"])
        rows.append(dict(
            key=key(r["M"], r["N"], r["K"], r["out"]), shape=f"{r['M']}x{r['N']}x{r['K']}",
            M=int(r["M"]), N=int(r["N"]), K=int(r["K"]), tiles=int(r["tiles"]), waves=int(r["waves"]),
            S=int(r["S"]), k=int(r["k_steps"]), config=config(r), label=r["v20_label"],
            best=best, best_lib=b["best_lib"], best_tile=tile, best_output_tile=best_output_tile,
            our_tile=our_tile, same=best_output_tile == our_tile if CFG["v27_view"] else tile == "256x256",
            tile_cat=tile_cat(tile),
            g9=pct(t["v9"], best), g20=pct(t["v20"], best), g20dp=pct(t["dp"], best),
            g20_v9=pct(t["v20"], t["v9"]),
            g20_pick=pct(t["v20"], fnum(r["aiter_pick_tflops"])), raw=r))
        if "v9new_tflops" in r:
            v9new_tile = mn_tile(r["v9new_tile"])
            rows[-1].update(g9new=pct(fnum(r["v9new_tflops"]), best), g26=pct(fnum(r["v26_tflops"]), best),
                            v9new_tile=v9new_tile, same_v9new=best_output_tile == v9new_tile)
    return rows


def counts(gaps):
    return (sum(g > TIE for g in gaps), sum(abs(g) <= TIE for g in gaps), sum(g < -TIE for g in gaps),
            statistics.median(gaps))


LO, HI = -90, 60  # every gap against the best library falls inside (-86% .. +57%)


def clip(x):
    return max(min(x, HI), LO)


def symlog_axis(ax, which, lo=LO, hi=HI):
    set_scale = ax.set_xscale if which == "x" else ax.set_yscale
    set_lim = ax.set_xlim if which == "x" else ax.set_ylim
    set_scale("symlog", linthresh=10, linscale=1.0)
    set_lim(lo, hi)
    ticks = [-80, -40, -20, -10, -5, 0, 5, 10, 20, 40]
    labels = [f"{t:+d}%" if t else "0" for t in ticks]
    if which == "x":
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=45)
    else:
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)


# ---------------------------------------------------------------- figure 0


SUMMARY_GROUPS = (
    ("All shapes", lambda r: True),
    ("Under 256 tiles (less than one wave)", lambda r: r["tiles"] < 256),
    ("256 to 1023 tiles", lambda r: 256 <= r["tiles"] < 1024),
    ("1024 or more tiles", lambda r: r["tiles"] >= 1024),
    ("Best library uses 256x256", lambda r: r["same"]),
    ("Best library uses another tile", lambda r: not r["same"]),
)


def same_own_tile(r, col):
    """Best library uses the tile of the kernel represented by this bar."""
    return r["same_v9new"] if col == "g9new" else r["same"]


same_own_tile.by_kernel = True


def different_own_tile(r, col):
    return not same_own_tile(r, col)


different_own_tile.by_kernel = True


SUMMARY_GROUPS_V27 = SUMMARY_GROUPS[:4] + (
    ("Best library uses the same MxN tile", same_own_tile),
    ("Best library uses a different MxN tile", different_own_tile),
)


KERNELS3 = (("v9", "g9", 0.4), ("streamk_wip, data-parallel only", "g20dp", 0.7), ("streamk_wip", "g20", 1.0))
KERNELS2 = (("v9", "g9", 0.4), ("streamk_wip", "g20", 1.0))
KERNELS_V27 = (("new AITER v9", "g9new", 0.5), ("tile-rule data-parallel", "g20dp", 0.7),
               ("streamk_wip", "g20", 1.0))
# Figure 0 of a --kernel v27 run: the new AITER v9 and v26 (timed in the same run) as well.
KERNELS5 = (("v9", "g9", 0.4), ("new v9", "g9new", 0.55), ("v26", "g26", 0.55),
            ("streamk_wip, data-parallel only", "g20dp", 0.7), ("streamk_wip", "g20", 1.0))
STATE_TITLE = {"warm": "Warm caches (back to back)", "cold": "Cold caches (2048 MB rotating buffer)"}
STATES = ("warm", "cold")  # ("cold",) with --cold-only


def wide():
    """Width factor of the bar figures: room for the counts column."""
    return 1.4 if CFG["counts"] else 1.0


def diverging(ax, rows, groups, kernels, title, show_labels=True):
    """One bar per (group, kernel): share of shapes behind the best library (left, red), within 1%
    (grey, centred) and ahead (right, blue), with the median on the right. Returns the numbers."""
    step = len(kernels) + 1
    out, ypos, labels = {}, [], []
    for gi, (gname, pred) in enumerate(groups):
        for ki, (kname, col, alpha) in enumerate(kernels):
            sel = ([r for r in rows if pred(r, col)] if getattr(pred, "by_kernel", False)
                   else [r for r in rows if pred(r)])
            y = -(gi * step + ki)
            ypos.append(y)
            labels.append(f"{gname}: {kname}" if ki == 0 else kname)
            if not sel:
                continue
            ahead, tie, behind, med = counts([r[col] for r in sel])
            n = len(sel)
            a, t, b = 100 * ahead / n, 100 * tie / n, 100 * behind / n
            ax.barh(y, -b, left=-t / 2, color="tab:red", alpha=alpha, height=0.8)
            ax.barh(y, t, left=-t / 2, color="0.7", alpha=alpha, height=0.8)
            ax.barh(y, a, left=t / 2, color="tab:blue", alpha=alpha, height=0.8)
            for val, x, cnt in ((b, -t / 2 - b / 2, behind), (a, t / 2 + a / 2, ahead)):
                if val >= 7:
                    ax.text(x, y, str(cnt), ha="center", va="center", fontsize=7.5, color="white")
            weight = "bold" if kname == "streamk_wip" else "normal"
            ax.text(103, y, f"{med:+.1f}%", va="center", fontsize=8.5, fontweight=weight)
            if CFG["counts"]:
                ax.text(170, y, f"{behind} / {tie} / {ahead}", ha="center", va="center", fontsize=8.5,
                        fontweight=weight)
            out[(gname, kname)] = (n, ahead, tie, behind, med)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlim(-100, 205 if CFG["counts"] else 115)
    ax.set_xticks([-100, -75, -50, -25, 0, 25, 50, 75, 100])
    ax.set_xticklabels(["100%", "75%", "50%", "25%", "0", "25%", "50%", "75%", "100%"])
    ax.text(-50, max(ypos) + 1.0, "behind by more than 1%", ha="center", fontsize=9, color="tab:red")
    ax.text(50, max(ypos) + 1.0, "ahead by more than 1%", ha="center", fontsize=9, color="tab:blue")
    ax.text(108, max(ypos) + 1.0, "median", ha="center", fontsize=9)
    if CFG["counts"]:
        ax.text(170, max(ypos) + 1.0, "behind / within 1% / ahead", ha="center", fontsize=9)
    ax.set_xlabel("share of shapes (numbers in the bars are shape counts; grey = within 1%)")
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="x")
    for gi in range(1, len(groups)):
        ax.axhline(-(gi * step) + 1, color="0.6", lw=0.6)
    if show_labels:
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, fontsize=8.5)
    else:
        ax.tick_params(labelleft=False)
    ax.set_ylim(min(ypos) - 0.8, max(ypos) + 1.6)
    return out


def figure0(runs, md):
    """v9, streamk_wip forced data-parallel and streamk_wip against the best library, per group,
    warm and cold side by side."""
    kernels = KERNELS_V27 if CFG["v27_view"] else KERNELS5 if "g9new" in runs[STATES[0]][0] else KERNELS3
    tall = 8.6 * (len(kernels) + 1) / 4
    fig, axes = plt.subplots(1, len(STATES), figsize=((15 if len(STATES) == 2 else 9.5) * wide(), tall), sharey=True,
                             squeeze=False)
    rows_out = {}
    groups = SUMMARY_GROUPS_V27 if CFG["v27_view"] else SUMMARY_GROUPS
    for i, (ax, state) in enumerate(zip(axes[0], STATES)):
        res = diverging(ax, runs[state], groups, kernels, STATE_TITLE[state], show_labels=i == 0)
        rows_out.update({(state,) + k: v for k, v in res.items()})
    if kernels is KERNELS_V27:
        fig.suptitle("New AITER v9, tile-rule data-parallel, and streamk_wip,\nagainst the best of AITER's pick and "
                     f"hipBLASLt ({len(runs[STATES[0]])} shapes)", fontsize=12)
    elif kernels is KERNELS5:
        fig.suptitle("v9 (existing and new), v26, streamk_wip forced data-parallel, and streamk_wip,\nagainst the "
                     f"best of AITER's pick and hipBLASLt ({len(runs[STATES[0]])} shapes)", fontsize=12)
    else:
        fig.suptitle("v9, streamk_wip forced data-parallel, and streamk_wip, against the best of AITER's pick and "
                     "hipBLASLt (630 shapes;\nv9 to data-parallel = the persistent kernel, data-parallel to streamk_wip = "
                     "stream-K)", fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT, "fig0_summary.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    md["fig0"] = rows_out
    return path


# ---------------------------------------------------------------- figures 4-6: S = 0 apart from S > 0

WAVE_GROUPS = (
    ("1 full wave", lambda r: r["waves"] == 1),
    ("2-3 full waves", lambda r: 2 <= r["waves"] <= 3),
    ("4-7 full waves", lambda r: 4 <= r["waves"] <= 7),
    ("8 or more full waves", lambda r: r["waves"] >= 8),
)
S_BUCKETS = (("S 1-32", 1, 32), ("S 33-64", 33, 64), ("S 65-128", 65, 128), ("S 129-255", 129, 255))
K_BUCKETS = (("k <= 32", 1, 32), ("k 33-64", 33, 64), ("k 65-128", 65, 128), ("k > 128", 129, 10 ** 6))
WAVE_SPLIT = (("Under one wave (fewer tiles than CUs)", lambda r: r["waves"] == 0),
              ("One full wave or more", lambda r: r["waves"] >= 1))


def s_groups():
    return tuple((name, (lambda lo, hi: lambda r: lo <= r["S"] <= hi)(lo, hi)) for name, lo, hi in S_BUCKETS)


def figure4(state, runs, md):
    """S = 0: no leftover tiles, so capped always runs data-parallel and streamk_wip is the
    persistent data-parallel kernel. v9 against streamk_wip by number of full waves."""
    rows = [r for r in runs[state] if r["S"] == 0]
    groups = (("All S = 0 shapes", lambda r: True),) + WAVE_GROUPS
    fig, ax = plt.subplots(figsize=(10 * wide(), 5.8))
    res = diverging(ax, rows, groups, KERNELS2, STATE_TITLE[state])
    fig.suptitle(f"Shapes with no leftover tiles (S = 0, {len(rows)} shapes): v9 against streamk_wip, which runs "
                 "data-parallel here,\nboth against the best of AITER's pick and hipBLASLt", fontsize=11)
    fig.tight_layout()
    path = os.path.join(OUT, f"fig4_s0_by_waves_{state}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    md.setdefault("fig4", {})[state] = res
    return path


def figure5(state, runs, md):
    """S > 0: rows are S buckets; under one wave and one wave or more in separate panels."""
    rows = [r for r in runs[state] if r["S"] > 0]
    fig, axes = plt.subplots(1, 2, figsize=(15 * wide(), 9.5 if CFG["v27_view"] else 7.2), sharey=True)
    res = {}
    for i, (ax, (wname, wpred)) in enumerate(zip(axes, WAVE_SPLIT)):
        sel = [r for r in rows if wpred(r)]
        tile_groups = ()
        if CFG["v27_view"]:
            tile_groups = (
                ("Best library: same MxN tile", same_own_tile),
                ("Best library: different MxN tile", different_own_tile),
            )
        groups = ((f"All ({len(sel)} shapes)", lambda r: True),) + tile_groups + s_groups()
        kernels = KERNELS_V27 if CFG["v27_view"] else KERNELS3
        out = diverging(ax, sel, groups, kernels, f"{wname}, {len(sel)} shapes", show_labels=i == 0)
        res.update({(wname,) + k: v for k, v in out.items()})
    fig.suptitle(f"Shapes with leftover tiles (S > 0, {len(rows)} shapes, {state} caches), by S = output tiles mod "
                 "256,\nagainst the best of AITER's pick and hipBLASLt; data-parallel only to streamk_wip = stream-K",
                 fontsize=11)
    fig.tight_layout()
    path = os.path.join(OUT, f"fig5_s_buckets_{state}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    md.setdefault("fig5", {})[state] = res
    return path


def figure6(state, runs, md):
    """S > 0: median by S bucket (rows) and k-step bucket (columns), under one wave and one wave or
    more, for streamk_wip against the best library and for stream-K's own gain."""
    rows = [r for r in runs[state] if r["S"] > 0]
    metrics = (("g20", "streamk_wip against the best library", 20),
               ("sk_gain", "stream-K's gain: streamk_wip against data-parallel only", 40))
    for r in rows:
        r["sk_gain"] = pct(1 + r["g20"] / 100, 1 + r["g20dp"] / 100)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    res = {}
    for row_i, (wname, wpred) in enumerate(WAVE_SPLIT):
        sel = [r for r in rows if wpred(r)]
        for col_i, (metric, mname, lim) in enumerate(metrics):
            ax = axes[row_i][col_i]
            grid_vals = [[float("nan")] * len(K_BUCKETS) for _ in S_BUCKETS]
            for si, (sname, slo, shi) in enumerate(S_BUCKETS):
                for ki, (kname, klo, khi) in enumerate(K_BUCKETS):
                    cell = [r[metric] for r in sel if slo <= r["S"] <= shi and klo <= r["k"] <= khi]
                    if cell:
                        med = statistics.median(cell)
                        grid_vals[si][ki] = med
                        behind = sum(x < -TIE for x in cell)
                        ax.text(ki, si, f"{med:+.1f}%\nn = {len(cell)}" + (f", {behind} behind" if metric == "g20"
                                                                           else ""),
                                ha="center", va="center", fontsize=7.5,
                                color="white" if abs(med) > lim * 0.6 else "black")
                        res[(wname, metric, sname, kname)] = (len(cell), med, behind)
            im = ax.imshow(grid_vals, cmap="RdBu", norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim), aspect="auto")
            ax.set_xticks(range(len(K_BUCKETS)))
            ax.set_xticklabels([k[0] for k in K_BUCKETS])
            ax.set_yticks(range(len(S_BUCKETS)))
            ax.set_yticklabels([s[0] for s in S_BUCKETS])
            ax.set_title(f"{wname}\n{mname}", fontsize=9.5)
            ax.set_xlabel("k-steps per output tile (K / 64)")
            cb = fig.colorbar(im, ax=ax, pad=0.02)
            cb.set_label(f"median % (colour clipped at +-{lim}%)", fontsize=8)
    fig.suptitle(f"Shapes with leftover tiles (S > 0, {len(rows)} shapes, {state} caches): median by S and k-steps",
                 fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT, f"fig6_s_k_heatmap_{state}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    md.setdefault("fig6", {})[state] = res
    return path


# ---------------------------------------------------------------- figure 1


def figure1(state, runs, md):
    rows = runs[state]
    other = "cold" if state == "warm" else "warm"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2))

    for st, ls in ((state, "-"), (other, "--")):
        if st not in runs:
            continue
        for name, col, color in (("v9", "g9", "tab:gray"), ("streamk_wip", "g20", "tab:blue")):
            g = sorted(r[col] for r in runs[st])
            n = len(g)
            ahead = sum(x > TIE for x in g)
            ax1.plot([clip(x) for x in g], [100 * (i + 1) / n for i in range(n)], ls=ls, color=color,
                     lw=2 if st == state else 1.3,
                     label=f"{name}, {st}: median {statistics.median(g):+.1f}%, ahead on {ahead} of {n}")
    ax1.axvline(0, color="k", lw=0.8)
    ax1.axvspan(-TIE, TIE, color="0.85", zorder=0)
    symlog_axis(ax1, "x")
    ax1.set_xlabel("TFLOPS against the best of AITER's pick and hipBLASLt (per shape)")
    ax1.set_ylabel("% of the 630 shapes at or below")
    ax1.set_title("(a) Distribution against the best library")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper left", fontsize=8.5)

    for gname in GRID:
        for same, face in ((True, None), (False, "none")):
            pts = [r for r in rows if grid(r["tiles"]) == gname and r["same"] == same]
            ax2.scatter([clip(r["g9"]) for r in pts], [clip(r["g20"]) for r in pts],
                        s=18, marker="o", facecolors=face or GRID_COLOR[gname], edgecolors=GRID_COLOR[gname],
                        linewidths=0.8, alpha=0.75)
    ax2.plot([LO, HI], [LO, HI], color="k", lw=0.8, ls=":")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.axvline(0, color="k", lw=0.8)
    symlog_axis(ax2, "x")
    symlog_axis(ax2, "y")
    q = {
        "behind with v9, ahead now": sum(r["g9"] < -TIE and r["g20"] > TIE for r in rows),
        "behind with both": sum(r["g9"] < -TIE and r["g20"] < -TIE for r in rows),
        "ahead with both": sum(r["g9"] > TIE and r["g20"] > TIE for r in rows),
        "ahead with v9, behind now": sum(r["g9"] > TIE and r["g20"] < -TIE for r in rows),
    }
    box = dict(boxstyle="round", fc="white", alpha=0.85)
    t = ax2.transAxes
    ax2.text(0.02, 0.97, f"behind with v9,\nahead now: {q['behind with v9, ahead now']}", fontsize=9, bbox=box,
             transform=t, va="top")
    ax2.text(0.02, 0.03, f"behind with both: {q['behind with both']}", fontsize=9, bbox=box, transform=t)
    ax2.text(0.98, 0.97, f"ahead with both: {q['ahead with both']}", fontsize=9, bbox=box, transform=t,
             va="top", ha="right")
    ax2.text(0.98, 0.03, f"ahead with v9,\nbehind now: {q['ahead with v9, behind now']}", fontsize=9, bbox=box,
             transform=t, ha="right")
    improved = sum(r["g20"] > r["g9"] + TIE for r in rows)
    worse = sum(r["g20"] < r["g9"] - TIE for r in rows)
    ax2.set_xlabel("v9 against the best library")
    ax2.set_ylabel("streamk_wip against the best library")
    ax2.set_title(f"(b) Per shape: above the diagonal, streamk_wip is faster than v9\n"
                  f"({improved} shapes faster than v9 by more than 1%, {worse} slower)", fontsize=10)
    ax2.grid(alpha=0.3)
    handles = [Line2D([], [], ls="", marker="o", color=GRID_COLOR[g], label=g) for g in GRID]
    handles += [Line2D([], [], ls="", marker="o", color="k", label="best library\nuses 256x256"),
                Line2D([], [], ls="", marker="o", mfc="none", color="k", label="best library uses\nanother tile")]
    ax2.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.suptitle(f"streamk_wip vs v9, both against the best of AITER's pick and hipBLASLt ({state} caches)",
                 fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT, f"fig1_progress_{state}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)

    md.setdefault("fig1", {})[state] = dict(
        v9=counts([r["g9"] for r in rows]), v20=counts([r["g20"] for r in rows]), q=q, improved=improved,
        worse=worse, same=sum(r["same"] for r in rows),
        hbl_best=sum(r["best_lib"] == "hipBLASLt" for r in rows))
    return path


# ---------------------------------------------------------------- figure 2


def s_x(r, max_waves):
    """S on x; whole-wave shapes (S = 0) go in a strip left of 0, spread by their wave count."""
    if r["S"]:
        return r["S"]
    jitter = (sum(map(ord, r["shape"])) % 5 - 2) * 0.9
    return -58 + 50 * math.log2(max(r["waves"], 1)) / math.log2(max_waves) + jitter


def figure2(state, runs, md):
    rows = [r for r in runs[state] if r["same"]]
    max_waves = max(max(r["waves"] for r in rows), 2)
    band = [r for r in rows if r["k"] <= 48 and (r["S"] == 0 or r["S"] > 128)]
    rest = [r for r in rows if r not in band]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4), sharey=True)
    for ax, col, title in ((axes[0], "g20", "(a) streamk_wip against the best library"),
                           (axes[1], "g20_v9", "(b) streamk_wip against v9")):
        norm = TwoSlopeNorm(vmin=-10, vcenter=0, vmax=10)
        for cfg, marker in CONFIG_MARKER.items():
            pts = [r for r in rows if r["config"] == cfg]
            if not pts:
                continue
            sc = ax.scatter([s_x(r, max_waves) for r in pts], [r["k"] for r in pts], c=[r[col] for r in pts],
                            cmap="RdBu", norm=norm, marker=marker, edgecolors="k", linewidths=0.4,
                            s=[25 + 18 * math.log2(1 + r["waves"]) for r in pts])
        ax.add_patch(plt.Rectangle((-63, 6), 61, 48 - 6 + 4, color="gold", alpha=0.25, zorder=0))
        ax.add_patch(plt.Rectangle((128.5, 6), 128, 48 - 6 + 4, color="gold", alpha=0.25, zorder=0))
        ax.axvline(-2, color="k", lw=0.6)
        ax.set_yscale("log", base=2)
        ax.set_yticks([8, 16, 32, 64, 128, 256, 512])
        ax.set_yticklabels(["8", "16", "32", "64", "128", "256", "512"])
        ax.set_xlim(-64, 258)
        ax.set_xticks([-58, -33, -8, 32, 64, 96, 128, 160, 192, 224, 255])
        ax.set_xticklabels([f"1", f"{round(2 ** (math.log2(max_waves) / 2))}", f"{max_waves}", "32", "64", "96",
                            "128", "160", "192", "224", "255"])
        ax.text(-33, 4.6, "S = 0: full waves", ha="center", fontsize=8)
        ax.set_xlabel("leftover tiles S = output tiles mod 256   (left strip: S = 0, placed by number of full waves)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        cb = fig.colorbar(sc, ax=ax, pad=0.01)
        cb.set_label("% (clipped at +-10%)")
    axes[0].set_ylabel("k-steps per output tile (K / 64)")
    handles = [Line2D([], [], ls="", marker=m, mfc="0.7", mec="k", label=c) for c, m in CONFIG_MARKER.items()
               if any(r["config"] == c for r in rows)]
    handles.append(plt.Rectangle((0, 0), 1, 1, color="gold", alpha=0.4, label="k <= 48 and (S = 0 or S > 128)"))
    axes[0].legend(handles=handles, loc="lower center", bbox_to_anchor=(0.36, 0.0), fontsize=7.5,
                   title="config capped ran; size = full waves", title_fontsize=7.5, framealpha=0.9)

    def summ(rs, col):
        return counts([r[col] for r in rs]) if rs else (0, 0, 0, float("nan"))

    fig.suptitle(f"Shapes where the best library also uses a 256x256 tile ({len(rows)} shapes, {state} caches)",
                 fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT, f"fig2_same_tile_S_k_{state}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    losses = [r for r in rows if r["g20"] < -TIE]
    md.setdefault("fig2", {})[state] = dict(
        n=len(rows), band=len(band), band_best=summ(band, "g20"), band_v9=summ(band, "g20_v9"),
        rest_best=summ(rest, "g20"), rest_v9=summ(rest, "g20_v9"), losses=len(losses),
        losses_in_band=sum(r in band for r in losses), losses_dp=sum(r["config"] == "data-parallel" for r in losses),
        worst=sorted(losses, key=lambda r: r["g20"])[:6])
    return path


# ---------------------------------------------------------------- figure 3


def figure3(state, runs, md):
    rows = [r for r in runs[state] if r["tile_cat"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw=dict(width_ratios=[1.6, 1]))
    for cat in TILE_CAT:
        pts = [r for r in rows if r["tile_cat"] == cat]
        if not pts:
            continue
        ax1.scatter([r["tiles"] for r in pts], [clip(r["g20"]) for r in pts], s=16,
                    color=TILE_COLOR[cat], alpha=0.75, label=f"{cat} ({len(pts)} shapes)")
    ax1.set_xscale("log", base=2)
    ax1.axvline(256, color="k", lw=0.8, ls=":")
    ax1.text(260, -85, "one wave (256 tiles)", fontsize=8)
    ax1.axhline(0, color="k", lw=0.8)
    ax1.axhspan(-TIE, TIE, color="0.85", zorder=0)
    symlog_axis(ax1, "y")
    ax1.set_xlabel("output tiles at 256x256 (log scale)")
    ax1.set_ylabel("streamk_wip against the best library")
    ax1.set_title("(a) Per shape, coloured by the best library's tile")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, loc="lower right", title="best library's tile", title_fontsize=8, framealpha=0.9)

    losses = [r for r in rows if r["g20"] < -TIE]
    bottom = [0] * len(GRID)
    table = {}
    for cat in TILE_CAT:
        vals = [sum(1 for r in losses if r["tile_cat"] == cat and grid(r["tiles"]) == g) for g in GRID]
        table[cat] = vals
        if not any(r["tile_cat"] == cat for r in rows):
            continue
        ax2.bar(GRID, vals, bottom=bottom, color=TILE_COLOR[cat], label=cat)
        bottom = [b + v for b, v in zip(bottom, vals)]
    for i, g in enumerate(GRID):
        total = sum(1 for r in rows if grid(r["tiles"]) == g)
        ax2.text(i, bottom[i] + 3, f"{bottom[i]} of {total} shapes", ha="center", fontsize=9)
    ax2.set_ylim(0, max(bottom) * 1.12)
    ax2.set_ylabel("shapes where streamk_wip is more than 1% behind")
    ax2.set_title(f"(b) The {len(losses)} losses, by grid size and the best library's tile")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, axis="y")
    fig.suptitle(f"Where streamk_wip still loses to the best of AITER's pick and hipBLASLt ({state} caches)",
                 fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT, f"fig3_losses_{state}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    md.setdefault("fig3", {})[state] = dict(
        losses=len(losses), table=table,
        by_cat={c: counts([r["g20"] for r in rows if r["tile_cat"] == c]) for c in TILE_CAT
                if any(r["tile_cat"] == c for r in rows)},
        n_cat={c: sum(r["tile_cat"] == c for r in rows) for c in TILE_CAT},
        unknown=len(runs[state]) - len(rows))
    return path


# ---------------------------------------------------------------- captions


def write_md(md, paths):
    def c(t):
        return f"{t[0]} ahead, {t[1]} within 1%, {t[2]} behind, median {t[3]:+.1f}%"

    L = ["# Figures: streamk_wip (v20, capped) against the best of AITER's pick and hipBLASLt", "",
         CFG["data"], "",
         "The baseline on each shape is the faster of AITER's pick and hipBLASLt (the torch call AITER's `torch` "
         "libtype makes), with that kernel's tile. AITER itself dispatches the winner of its offline tuning, so this "
         "is a stricter bar than what AITER runs today. \"Ahead\" and \"behind\" mean a gap of more than 1%.", ""]
    if CFG["layout"] == "pad":
        L += ["Layout: v9 and streamk_wip on the padded layout (each row stride that is a multiple of 512 bytes "
              "gets 256 more). The best library is the fastest valid time of AITER's pick and hipBLASLt over both "
              "layouts; a padded call that copies its inputs (FlyDSL) does not count.", ""]
    elif CFG["layout"] == "mixed":
        L += ["Layout: v9 and streamk_wip on the padded layout (each row stride that is a multiple of 512 bytes "
              "gets 256 more); AITER's pick and hipBLASLt contiguous, as AITER calls them.", ""]
    names = (("v9", "v9"), ("streamk_wip, data-parallel only", "data-parallel only"), ("streamk_wip", "streamk_wip"))
    if (STATES[0], SUMMARY_GROUPS[0][0], "v26") in md["fig0"]:
        names = (("v9", "v9"), ("new v9", "new v9"), ("v26", "v26")) + names[1:]
    m0, both = STATES[0], len(STATES) > 1

    def alt(*lines):
        """The pointer to the other cache state's file, when there is one."""
        return ["", *lines, ""] if both else [""]

    L += ["## Figure 0: summary", "", "![fig0](fig0_summary.png)", "",
          "\"Data-parallel only\" is streamk_wip with `STREAMK_POLICY=dp`: the same persistent kernel without the "
          "stream-K split. v9 to data-parallel only is the persistent kernel's change; data-parallel only to "
          "streamk_wip is what stream-K adds.", "",
          *(["In Figures 0, 4 and 5, the right-hand count column is `behind / within 1% / ahead`.", ""]
            if CFG["counts"] else []),
          "| Caches | Shapes | Count | " + " | ".join(f"{s}: faster | {s}: within 1% | {s}: slower | {s}: median"
                                                   for _, s in names) + " |",
          "|---|---|---:|" + "---:|---:|---:|---:|" * len(names)]
    for st in STATES:
        for gname, _ in SUMMARY_GROUPS:
            if (st, gname, names[0][0]) not in md["fig0"]:
                continue
            cells, n = [], 0
            for kname, _ in names:
                n, a, t, b, med = md["fig0"][(st, gname, kname)]
                cells.append(f"{a} | {t} | {b} | {med:+.1f}%")
            L.append(f"| {st.capitalize()} | {gname} | {n} | " + " | ".join(cells) + " |")
    L += [""]
    f1 = md["fig1"]
    L += ["## Figure 1: progress over v9", "", f"![fig1](fig1_progress_{m0}.png)", ""]
    for st in STATES:
        d = f1[st]
        L += [f"- {st.capitalize()}: v9 {c(d['v9'])}; streamk_wip {c(d['v20'])}. "
              f"streamk_wip improved on v9 by more than 1% on {d['improved']} shapes and fell back on {d['worse']}. "
              f"{d['q']['behind with v9, ahead now']} shapes went from behind to ahead, "
              f"{d['q']['ahead with v9, behind now']} the other way. "
              f"hipBLASLt is the faster library on {d['hbl_best']} shapes; the best library uses 256x256 on {d['same']}."]
    L += alt("Cold version: `fig1_progress_cold.png` (both states are drawn in panel (a) of each).")

    f2 = md["fig2"]
    L += ["## Figure 2: the same-tile losses, by S and k-steps", "", f"![fig2](fig2_same_tile_S_k_{m0}.png)", ""]
    for st in STATES:
        d = f2[st]
        L += [f"- {st.capitalize()}: {d['n']} shapes where the best library uses 256x256; streamk_wip is behind on "
              f"{d['losses']}, {d['losses_dp']} of them data-parallel and {d['losses_in_band']} in the shaded band "
              f"(k <= 48 and S = 0 or S > 128).",
              f"  - In the band ({d['band']} shapes): against the best library {c(d['band_best'])}; "
              f"against v9 {c(d['band_v9'])}.",
              f"  - Outside it ({d['n'] - d['band']} shapes): against the best library {c(d['rest_best'])}; "
              f"against v9 {c(d['rest_v9'])}.",
              "  - Worst: " + "; ".join(f"{r['shape']} (S = {r['S']}, k = {r['k']}, {r['label']}, "
                                         f"{r['g20']:+.1f}% vs best, {r['g20_v9']:+.1f}% vs v9)" for r in d["worst"]) + "."]
    L += alt("Cold version: `fig2_same_tile_S_k_cold.png`.")

    f3 = md["fig3"]
    L += ["## Figure 3: where the remaining losses are", "", f"![fig3](fig3_losses_{m0}.png)", ""]
    for st in STATES:
        d = f3[st]
        L += [f"- {st.capitalize()}: streamk_wip is behind the best library on {d['losses']} shapes."]
        for cat, t in d["by_cat"].items():
            L += [f"  - Best library tile {cat} ({d['n_cat'][cat]} shapes): {c(t)}. Losses by grid size "
                  f"(under 1 wave / 256-1023 / 1024+): {' / '.join(str(v) for v in d['table'][cat])}."]
        if d["unknown"]:
            L += [f"  - {d['unknown']} shapes have no parsable tile (AITER's Triton pick) and are left out."]
    L += alt("Cold version: `fig3_losses_cold.png`.")

    L += ["## Figure 4: shapes with no leftover tiles (S = 0)", "", f"![fig4](fig4_s0_by_waves_{m0}.png)", "",
          "With S = 0 capped always runs data-parallel, so streamk_wip here is the persistent data-parallel kernel "
          "and stream-K plays no part.", ""]
    for st in STATES:
        d = md["fig4"][st]
        L += [f"- {st.capitalize()}: " + "; ".join(
            f"{g}: v9 {d[(g, 'v9')][4]:+.1f}% ({d[(g, 'v9')][3]} behind), streamk_wip {d[(g, 'streamk_wip')][4]:+.1f}% "
            f"({d[(g, 'streamk_wip')][3]} behind) of {d[(g, 'v9')][0]}" for g in
            ["All S = 0 shapes"] + [w[0] for w in WAVE_GROUPS]) + "."]
    L += alt("Cold version: `fig4_s0_by_waves_cold.png`.")[:-1] + [
          "", "## Figure 5: shapes with leftover tiles (S > 0), by S", "", f"![fig5](fig5_s_buckets_{m0}.png)", "",
          "Rows are S buckets; under one wave and one wave or more are separate panels. Data-parallel only to "
          "streamk_wip is stream-K's own effect." + (" Cold version: `fig5_s_buckets_cold.png`." if both else ""), "",
          "## Figure 6: shapes with leftover tiles (S > 0), by S and k-steps", "", f"![fig6](fig6_s_k_heatmap_{m0}.png)",
          "", "Each cell is the median over the shapes in it (n), with the number more than 1% behind the best "
          "library. Left: streamk_wip against the best library. Right: stream-K's gain, streamk_wip against the same "
          "kernel forced data-parallel; 0.0% means capped ran data-parallel on every shape in the cell."
          + (" Cold version: `fig6_s_k_heatmap_cold.png`." if both else ""), ""]
    if "fig7" in md:
        L += ["## Figure 7: padding, spread between fresh-buffer rounds and padded against contiguous", "",
              "![fig7](fig7_padding.png)", "",
              "Top: share of shapes (with padded x and w strides) whose spread between rounds exceeds x, per kernel; "
              "dashed contiguous, solid padded. Bottom: median padded / contiguous TFLOPS - 1 by grid size, and on "
              "the control shapes (K % 256 == 128, x and w not padded). AITER's pick leaves out, on both layouts, "
              "the shapes where its padded call copied its inputs.", ""]
        for st, d in md["fig7"].items():
            L += [f"- {st.capitalize()}: " + "; ".join(
                f"{k}: median spread {v['sp_plain']:.2f}% contiguous, {v['sp_pad']:.2f}% padded; over 2%: "
                f"{v['gt2_plain']} and {v['gt2_pad']} of {v['n']}" for k, v in d.items()) + "."]
        L += [""]
    with open(os.path.join(OUT, "figures.md"), "w") as f:
        f.write("\n".join(L))
    return os.path.join(OUT, "figures.md")


# ---------------------------------------------------------------- figure 7: padding

PAD_SPREAD = (("v9", "v9", "tab:gray"), ("streamk_wip", "v20", "tab:blue"), ("AITER's pick", "pick", "tab:green"),
              ("hipBLASLt", "hbl", "tab:orange"))
PAD_RATIO = (("v9", "v9_tflops", "v9_pad_tflops", None, "tab:gray"),
             ("streamk_wip", "v20_tflops", "v20_pad_tflops", None, "tab:blue"),
             ("AITER's pick", "aiter_pick_tflops", "pick_pad_tflops", "pick_pad_copy", "tab:green"),
             ("hipBLASLt", "hbl_tflops", "hbl_pad_tflops", "hbl_pad_copy", "tab:orange"))


def figure7(runs, md):
    """Top: spread between fresh-buffer rounds, contiguous against padded, per kernel (shapes whose
    x and w strides are padded). Bottom: median padded / contiguous - 1 by grid size and control."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))
    res = {}
    for col, state in enumerate(("warm", "cold")):
        rows = [r["raw"] for r in runs[state]]
        padded = [r for r in rows if int(r["K"]) % 256 == 0]
        ax = axes[0][col]
        xs = [0.25 * i for i in range(0, 41)]
        res[state] = {}
        for name, c, color in PAD_SPREAD:
            sel = [r for r in padded if c != "pick" or str(r["pick_pad_copy"]) != "True"]
            plain = [fnum(r[f"{c}_spread"]) for r in sel]
            pad = [fnum(r[f"{c}_pad_spread"]) for r in sel]
            plain, pad = [x for x in plain if x == x], [x for x in pad if x == x]
            for vals, ls in ((plain, "--"), (pad, "-")):
                ax.plot(xs, [100 * sum(v > x for v in vals) / len(vals) for x in xs], ls=ls, color=color,
                        lw=1.8 if ls == "-" else 1.2,
                        label=f"{name}, {'padded' if ls == '-' else 'contiguous'}: median {statistics.median(vals):.2f}%")
            res[state][name] = dict(sp_plain=statistics.median(plain), sp_pad=statistics.median(pad),
                                    gt2_plain=sum(v > 2 for v in plain), gt2_pad=sum(v > 2 for v in pad),
                                    n=len(plain))
        ax.set_yscale("log")
        ax.set_ylim(0.1, 100)
        ax.set_xlabel("spread between rounds, (slowest - fastest) / median (%)")
        ax.set_ylabel("% of shapes with spread above x")
        ax.set_title(f"{STATE_TITLE[state]}: {len(padded)} shapes with padded x and w strides")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5)

        ax = axes[1][col]
        groups = [(g, lambda r, g=g: int(r["K"]) % 256 == 0 and grid(int(r["tiles"])) == g) for g in GRID]
        groups.append(("control\n(K % 256 = 128)", lambda r: int(r["K"]) % 256 != 0))
        width = 0.2
        for ki, (name, cc, pc, flag, color) in enumerate(PAD_RATIO):
            meds = []
            for gi, (gname, pred) in enumerate(groups):
                g = [pct(fnum(r[pc]), fnum(r[cc])) for r in rows if pred(r) and (flag is None or str(r[flag]) != "True")]
                g = [x for x in g if x == x]
                meds.append(statistics.median(g) if g else float("nan"))
            ax.bar([gi + (ki - 1.5) * width for gi in range(len(groups))], meds, width=width, color=color, label=name)
            for gi, m in enumerate(meds):
                if m == m:
                    ax.annotate(f"{m:+.1f}", (gi + (ki - 1.5) * width, m), xytext=(0, 2 if m >= 0 else -2),
                                textcoords="offset points", ha="center", va="bottom" if m >= 0 else "top", fontsize=7)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g[0] for g in groups], fontsize=8.5)
        ax.set_ylabel("median padded / contiguous TFLOPS - 1 (%)")
        ax.set_title(f"{STATE_TITLE[state]}: padded against contiguous")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    fig.suptitle("Stride padding (+256 bytes on every row stride that is a multiple of 512 bytes): "
                 "round-to-round spread and speed", fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT, "fig7_padding.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    md["fig7"] = res
    return path


CFG = dict(layout="plain", counts=False, v27_view=False,
           data="Data: `details_capped_hbl.csv` (warm: back to back on the same buffers) and "
           "`details_capped_hbl_rot2048.csv` (cold: 2048 MB rotating buffer), 630 shapes, MI355X, torch "
           "2.10.0+rocm7.2.4 with hipBLASLt 1.2.2, AITER `main` `a1455f1c4`, v20 snapshot `7c55c4017abfc6cc`.")


def main():
    global OUT
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--warm", default=RUNS["warm"])
    p.add_argument("--cold", default=RUNS["cold"])
    p.add_argument("--layout", choices=["plain", "pad", "mixed"], default="plain")
    p.add_argument("--out", default=OUT)
    p.add_argument("--data", default=None, help="the data line of figures.md")
    p.add_argument("--cold-only", action="store_true", help="a run with no warm sweep: cold figures only")
    p.add_argument("--counts", action="store_true",
                   help="the bar figures (0, 4, 5) get a column after the median: behind / within 1% / ahead")
    p.add_argument("--v27-view", action="store_true",
                   help="bar figures compare new AITER v9, tile-rule data-parallel and streamk_wip")
    p.add_argument("--issue-only", action="store_true",
                   help="write only Figure 0 and cold Figure 5 (requires --cold-only)")
    args = p.parse_args()
    global STATES
    OUT = args.out
    CFG["layout"] = args.layout
    CFG["counts"] = args.counts
    CFG["v27_view"] = args.v27_view
    if args.data:
        CFG["data"] = args.data
    os.makedirs(OUT, exist_ok=True)
    if args.cold_only:
        assert args.layout == "plain", "--cold-only draws the contiguous layout"
        STATES = ("cold",)
        runs = {"cold": load(args.cold, args.layout)}
    else:
        runs = {"warm": load(args.warm, args.layout), "cold": load(args.cold, args.layout)}
        assert {r["key"] for r in runs["warm"]} == {r["key"] for r in runs["cold"]}, "warm and cold shapes differ"
    md = {}
    if args.issue_only:
        assert args.cold_only, "--issue-only requires --cold-only"
        paths = [figure0(runs, md), figure5("cold", runs, md)]
        for path in paths:
            print("wrote", path)
        return
    paths = [figure0(runs, md)]
    for st in STATES:
        paths += [figure1(st, runs, md), figure2(st, runs, md), figure3(st, runs, md),
                  figure4(st, runs, md), figure5(st, runs, md), figure6(st, runs, md)]
    if "v20_pad_tflops" in runs[STATES[0]][0]["raw"]:
        paths.append(figure7(runs, md))
    paths.append(write_md(md, paths))
    for p in paths:
        print("wrote", p)


if __name__ == "__main__":
    main()
