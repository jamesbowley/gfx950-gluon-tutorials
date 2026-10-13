"""Figures for the version-table sweep (results/main.jsonl, results/split.jsonl).

    python experiments/streamk_version_tables/plot.py      # images/g1-g5_*.png, summary on stdout

G1  speed against v13, every version and v20 policy (heatmap)
G2  where stream-K pays: best stream-K against best data-parallel, by S (scatter)
G3  L2 hit rate against v13 (heatmap)
G4  TFLOPS against programs per leftover tile n, v20 owner and rs (small multiples)
G5  v20 auto and capped against the best of everything measured (bars)

Labels and schedules come from sweep.py, so they match results/tables.md. No GPU or Triton needed.
"""

import math
import os
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sweep  # noqa: E402

IMAGES = os.path.join(HERE, "images")
CLASSES = ["compute", "short K", "skinny"]
DP_CONFIGS = ["v9", "v10", "v11", "v12", "v13", "v14", "v20 dp"]
COLUMNS = ["best dp", "v15", "v16", "v17", "v18", "v19", "v20 auto", "v20 rs", "v20 owner",
           "v20 two_tile", "v20 spread", "v20 capped"]
NOISY = {"256x32768x8192", "256x49152x8192"}  # 13-30% process-to-process spread (section 9)
SCHED_COLOURS = {"data-parallel": "#7f7f7f", "two-tile": "#d95f02", "one-tile rs": "#1b7837",
                 "one-tile owner": "#2166ac", "one-tile end-to-end": "#984ea3", "spread": "#e7298a"}
CLASS_MARKERS = {"compute": "o", "short K": "s", "skinny": "^"}


def load():
    main = {(r["shape"], r["config"]): r for r in sweep.load("main")}
    split = {(r["shape"], r["config"], r["n"]): r for r in sweep.load("split")}
    policy, capped_policy = sweep.v20_host_functions("streamk_policy", "capped_policy")
    shapes = []
    for M, N, K, cls in sweep.SHAPES:
        name = sweep.shape_name(M, N, K)
        g = sweep.geometry(M, N, K)
        for c in sweep.MAIN:
            r = main.get((name, c))
            assert sweep.good(r), f"missing or wrong main case {name} {c}"
        auto = policy(g["tiles"], g["k_tiles"], sweep.NUM_CUS)
        capped = capped_policy(g["tiles"], g["k_tiles"], sweep.NUM_CUS)
        sched = {c: sweep.schedule(c, g, auto, capped) for c in sweep.MAIN}
        shapes.append(dict(name=name, M=M, N=N, K=K, cls=cls, sched=sched, **g))
    shapes.sort(key=lambda s: (CLASSES.index(s["cls"]), s["S"], s["waves"], s["K"]))
    return shapes, main, split


def row_label(s):
    label = f"{s['name']}   S {s['S']}, {s['waves']} w, {s['k_tiles']} k"
    return label + " (noisy)" if s["name"] in NOISY else label


def best_dp(s, main):
    return max((main[(s["name"], c)] for c in DP_CONFIGS), key=lambda r: r["tflops"])


def cell_row(s, main, c):
    return best_dp(s, main) if c == "best dp" else main[(s["name"], c)]


def split_entries(s, split, config):
    """(n, row) for one configuration on one shape, the default split last."""
    ns = sweep.splits(s["S"], s["K"]) if s["S"] else []
    out = [(n, split.get((s["name"], config, n))) for n in ns]
    return [(n, r) for n, r in out if sweep.good(r)]


def best_overall(s, main, split):
    rows = [main[(s["name"], c)] for c in sweep.MAIN]
    rows += [r for c in sweep.ONE_TILE for _, r in split_entries(s, split, c)]
    return max(rows, key=lambda r: r["tflops"])


def spread_pct(r):
    runs = r.get("us_runs") or [r["us"]]
    return 100 * (max(runs) / min(runs) - 1)


def class_bands(ax, shapes, width):
    """Separators and class names on the left of a row-per-shape axis."""
    start = 0
    for cls in CLASSES:
        n = sum(s["cls"] == cls for s in shapes)
        if not n:
            continue
        if start:
            ax.axhline(start - 0.5, color="black", lw=1.6)
        ax.text(-0.42, start + n / 2 - 0.5, cls, transform=ax.get_yaxis_transform(), rotation=90,
                ha="center", va="center", fontsize=10, weight="bold")
        start += n


def titles(fig, title, subtitle):
    h = fig.get_figheight()
    fig.text(0.01, 1 - 0.12 / h, title, fontsize=13, weight="bold", va="top")
    fig.text(0.01, 1 - 0.42 / h, subtitle, fontsize=9.5, color="#444444", va="top")


def heatmap(path, shapes, main, value, text, norm, cmap, cbar_label, title, subtitle):
    nr, nc = len(shapes), len(COLUMNS)
    fig, ax = plt.subplots(figsize=(15, 0.36 * nr + 2.2))
    grid = [[value(s, c) for c in COLUMNS] for s in shapes]
    im = ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto")
    for i, s in enumerate(shapes):
        rows = {c: cell_row(s, main, c) for c in COLUMNS}
        best_c = max(COLUMNS, key=lambda c: rows[c]["tflops"])
        for j, c in enumerate(COLUMNS):
            v = grid[i][j]
            dark = abs(norm(v) - 0.5) > 0.33
            ax.text(j, i, text(s, c), ha="center", va="center", fontsize=7,
                    color="white" if dark else "black")
            if c == best_c:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, lw=2.2, ec="black"))
    ax.set_xticks(range(nc), [c.removeprefix("v20 ") if c.startswith("v20") else c for c in COLUMNS],
                  fontsize=8.5)
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
    for j, c in enumerate(COLUMNS):
        if c == "v20 auto":
            ax.axvline(j - 0.5, color="black", lw=1.2)
            ax.text(j - 0.5, -1.6, "v20 policies ->", fontsize=8, va="bottom")
    ax.axvline(0.5, color="black", lw=1.2)
    ax.set_yticks(range(nr), [row_label(s) for s in shapes], fontsize=7.5, family="monospace")
    class_bands(ax, shapes, nc)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label(cbar_label, fontsize=8.5)
    fig.tight_layout(rect=(0.03, 0, 1, 1 - 0.75 / fig.get_figheight()))
    titles(fig, title, subtitle)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def g1(path, shapes, main):
    def value(s, c):
        return 100 * (cell_row(s, main, c)["tflops"] / main[(s["name"], "v13")]["tflops"] - 1)

    def text(s, c):
        return f"{value(s, c):+.0f}"

    heatmap(path, shapes, main, value, text, TwoSlopeNorm(vcenter=0, vmin=-50, vmax=50), "RdBu",
            "% against v13 (clipped at +-50)",
            "G1. Speed against v13 on every test shape",
            "Cold rocprof TFLOPS, % against v13; 'best dp' is the fastest data-parallel build (v9-v14, v20 dp). "
            "Outlined: the fastest column on the row. Rows: S, full waves (w), k-steps per tile (k).")


def g3(path, shapes, main):
    def value(s, c):
        return 100 * (cell_row(s, main, c)["l2_hit"] - main[(s["name"], "v13")]["l2_hit"])

    def text(s, c):
        return f"{100 * cell_row(s, main, c)['l2_hit']:.0f}"

    heatmap(path, shapes, main, value, text, TwoSlopeNorm(vcenter=0, vmin=-40, vmax=10), "RdBu",
            "L2 hit rate, points against v13",
            "G3. L2 hit rate on every test shape",
            "Cells print the hit rate (%), coloured by points against v13. Outlined: the fastest column "
            "on the row, so a lower hit rate can still win.")


def g2(path, shapes, main, split):
    fig, ax = plt.subplots(figsize=(13, 6.2))
    pts = []
    for s in shapes:
        if s["S"] == 0:
            continue
        dp = best_dp(s, main)
        sk = [(main[(s["name"], c)], s["sched"][c]) for c in sweep.MAIN if s["sched"][c] != "data-parallel"]
        for c in sweep.ONE_TILE:
            sched = {"v20 rs": "one-tile rs"}.get(c, "one-tile owner")
            if c in ("v15", "v16"):
                sched = "one-tile end-to-end"
            sk += [(r, sched) for _, r in split_entries(s, split, c)]
        r, sched = max(sk, key=lambda e: e[0]["tflops"])
        sched = sched.split(",")[0]
        y = 100 * (r["tflops"] / dp["tflops"] - 1)
        pts.append((s, y, sched))
        ax.scatter(s["S"], y, s=110, color=SCHED_COLOURS[sched], marker=CLASS_MARKERS[s["cls"]],
                   edgecolor="black", lw=0.8, zorder=3)
        tag = f"K {s['K'] // 1024}k" + (f", {s['waves']} w" if s["waves"] != 1 else "")
        if abs(y) >= 2.5:
            ax.annotate(tag, (s["S"], y), textcoords="offset points", xytext=(7, 4), fontsize=7.5)
    near = [p for p in pts if abs(p[1]) < 2.5]
    ax.annotate(f"{len(near)} shapes within 2.5% of data-parallel\n(S >= 128 at K <= 8192, and 4096x5120x2048)"
                if near else "", (160, -4.5), fontsize=7.5, ha="center", va="top", color="#444444")
    noise = statistics.median(spread_pct(main[(s["name"], c)]) for s in shapes for c in sweep.MAIN)
    ax.axhspan(-noise, noise, color="#dddddd", zorder=0)
    ax.axhline(0, color="black", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256], ["1", "2", "4", "8", "16", "32", "64", "128", "256"])
    ax.set_xlabel("S, leftover tiles (log scale)", fontsize=9)
    ax.set_ylabel("best stream-K against best data-parallel (%)", fontsize=9)
    ax.grid(color="#eeeeee", zorder=0)
    handles = [Patch(facecolor=SCHED_COLOURS[k], edgecolor="black", label=k) for k in SCHED_COLOURS
               if any(p[2] == k for p in pts)]
    handles += [Line2D([], [], marker=m, color="white", markerfacecolor="white", markeredgecolor="black",
                       markersize=9, label=cls) for cls, m in CLASS_MARKERS.items()]
    handles.append(Patch(facecolor="#dddddd", label=f"median run-to-run spread, +-{noise:.1f}%"))
    ax.legend(handles=handles, fontsize=8, loc="upper right", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / fig.get_figheight()))
    titles(fig, "G2. Where stream-K pays",
           "Per shape, the fastest stream-K configuration (any version, policy or capped split) against the "
           "fastest data-parallel build. Colour: its schedule. Label: K, and full waves when not 1.")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return pts


def g4(path, shapes, main, split):
    sel = [s for s in shapes if any(split_entries(s, split, c) for c in ("v20 owner", "v20 rs"))]
    ncols = 5
    nrows = math.ceil(len(sel) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.1 * nrows + 1.2), squeeze=False)
    for ax, s in zip(axes.flat, sel):
        default_n = 256 / s["S"]
        for c, colour, label in (("v20 owner", "#2166ac", "owner"), ("v20 rs", "#1b7837", "reduce-scatter")):
            pts = [(n, r["tflops"]) for n, r in split_entries(s, split, c)]
            pts.append((default_n, main[(s["name"], c)]["tflops"]))
            xs, ys = zip(*sorted(pts))
            ax.plot(xs, ys, marker="o", ms=4, color=colour, lw=1.5, label=label)
            ax.plot(default_n, main[(s["name"], c)]["tflops"], marker="o", ms=8, mfc="white", mec=colour,
                    mew=1.8, zorder=4)
        ax.axhline(best_dp(s, main)["tflops"], color="black", ls="--", lw=1)
        tt = [main[(s["name"], c)]["tflops"] for c in ("v19", "v20 two_tile") if s["sched"][c] == "two-tile"]
        if tt:
            ax.axhline(max(tt), color="#d95f02", ls=":", lw=1.4)
        ax.set_xscale("log", base=2)
        ticks = [t for t in (1, 2, 4, 8, 16, 32, 64, 128, 256) if t <= default_n * 1.01]
        ax.set_xticks(ticks, [str(t) for t in ticks], fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(color="#eeeeee")
        ax.set_title(f"{s['name']}  S {s['S']}, {s['k_tiles']} k" + (f", {s['waves']} w" if s["waves"] != 1 else ""),
                     fontsize=8.5, loc="left")
    for ax in list(axes.flat)[len(sel):]:
        ax.set_axis_off()
    for ax in axes[:, 0]:
        ax.set_ylabel("TFLOPS", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("programs per leftover tile, n", fontsize=8)
    handles = [Line2D([], [], color="#2166ac", marker="o", label="owner ending"),
               Line2D([], [], color="#1b7837", marker="o", label="reduce-scatter ending"),
               Line2D([], [], color="grey", marker="o", mfc="white", ls="", ms=8, label="default split (all 256 programs)"),
               Line2D([], [], color="black", ls="--", label="best data-parallel"),
               Line2D([], [], color="#d95f02", ls=":", lw=1.4, label="two-tile (v19 / v20 two_tile)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1 - 0.75 / fig.get_figheight()))
    titles(fig, "G4. Programs per leftover tile",
           "One-tile stream-K in v20 with STREAMK_NUM_PROGRAMS = S * n. Right-most point: the default split. "
           "Too few programs leave the tail long; too many make the collection long.")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return sel


def g5(path, shapes, main, split):
    nr = len(shapes)
    fig, ax = plt.subplots(figsize=(13, 0.34 * nr + 2.2))
    res = {}
    for i, s in enumerate(shapes):
        best = best_overall(s, main, split)
        for k, (c, colour, dy) in enumerate((("v20 auto", "#7570b3", -0.2), ("v20 capped", "#1b9e77", 0.2))):
            v = 100 * (main[(s["name"], c)]["tflops"] / best["tflops"] - 1)
            res[(s["name"], c)] = v
            ax.barh(i + dy, v, height=0.38, color=colour, label=c if i == 0 else None)
            if v < -2.5:
                ax.text(v - 0.3, i + dy, f"{v:.1f}", va="center", ha="right", fontsize=6.5)
    ax.axvline(-1.5, color="#1b9e77", ls=":", lw=1, label="-1.5%")
    ax.axvline(-2.5, color="#7570b3", ls=":", lw=1, label="-2.5%")
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(range(nr), [row_label(s) for s in shapes], fontsize=7.5, family="monospace")
    ax.set_ylim(nr - 0.5, -0.5)
    ax.set_xlim(-25, 1)
    ax.set_xlabel("% against the fastest configuration measured on the shape, including every capped split",
                  fontsize=8.5)
    ax.grid(axis="x", color="#eeeeee")
    class_bands(ax, shapes, 1)
    n_cap = sum(res[(s["name"], "v20 capped")] >= -1.5 for s in shapes)
    n_auto = sum(res[(s["name"], "v20 auto")] >= -2.5 for s in shapes)
    ax.legend(fontsize=8.5, loc="lower left")
    fig.tight_layout(rect=(0.03, 0, 1, 1 - 0.75 / fig.get_figheight()))
    titles(fig, "G5. Host rules against the best",
           f"capped is within 1.5% on {n_cap} of {nr} shapes, auto within 2.5% on {n_auto}. Bars below -2.5% "
           "are labelled; bars past -25% are clipped.")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return n_cap, n_auto


def main():
    os.makedirs(IMAGES, exist_ok=True)
    shapes, main_rows, split = load()
    assert len(shapes) == len(sweep.SHAPES) == 30
    g1(os.path.join(IMAGES, "g1_speed.png"), shapes, main_rows)
    pts = g2(os.path.join(IMAGES, "g2_where_streamk_pays.png"), shapes, main_rows, split)
    g3(os.path.join(IMAGES, "g3_l2.png"), shapes, main_rows)
    sel = g4(os.path.join(IMAGES, "g4_split_count.png"), shapes, main_rows, split)
    n_cap, n_auto = g5(os.path.join(IMAGES, "g5_host_rules.png"), shapes, main_rows, split)
    print(f"{len(shapes)} shapes, {len(sel)} with a split sweep")
    print(f"capped within 1.5% of the best on {n_cap}, auto within 2.5% on {n_auto}")
    print("G2: best stream-K against best data-parallel")
    for s, y, sched in pts:
        print(f"  {s['name']:>16} S {s['S']:>3}: {y:+6.1f}%  {sched}")
    print(f"images written to {IMAGES}")


if __name__ == "__main__":
    main()
