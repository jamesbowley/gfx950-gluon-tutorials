"""Figures from results/probe.md and results/slice.md (no GPU).

    python make_figures.py     # images/probe.png, images/slice.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "images")


def tables(path):
    out, section, header = {}, None, None
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith("#"):
            section, header = line.lstrip("#").strip(), None
        elif line.startswith("|") and section:
            cells = [c.strip() for c in line.strip("|").split("|")]
            if header is None:
                header = cells
            elif not set(line) <= set("|-"):
                out.setdefault(section, []).append(dict(zip(header, cells)))
        else:
            header = None
    return out


def probe_figure():
    rows = [r for r in tables(os.path.join(HERE, "results", "probe.md"))["probe"] if "add" not in r["case"]]
    fig, ax = plt.subplots(figsize=(11, 5))
    labels = [r["conversion"] for r in rows]
    ns = [float(r["ns per conversion"]) for r in rows]
    rounds = [float(r["LDS rounds"]) for r in rows]
    colors = ["#6aa56a" if x == 0 else "#4878a8" if x <= 2 else "#e0a040" if x <= 4 else "#c0504d" for x in rounds]
    y = range(len(rows))
    ax.barh(list(y), ns, color=colors)
    for i, (v, rd, r) in enumerate(zip(ns, rounds, rows)):
        ax.text(v + 8, i, f"{v:.0f} ns, {rd:g} LDS rounds, {int(r['LDS bytes']) // 1024} KiB LDS", va="center",
                fontsize=8)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1250)
    ax.set_xlabel("ns per conversion of one 128x128 bf16 quadrant (one CU, 4 warps)")
    ax.set_title("Every non-trivial conversion moves the same 32 KiB through LDS\n(8 ds_write_b128 + 8 ds_read_b128 "
                 "per lane); the cost follows the scratch buffer the compiler picks:\n16 KiB = 2 rounds, 8 KiB = 4, "
                 "4 KiB = 8 (each round: write, barrier, read, barrier)", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save(fig, "probe.png")


VAR_STYLE = {
    "v20": ("#7f7f7f", "v20: L3 slice, 8 B per lane C store"),
    "v20_old_cvt": ("#c0504d", "L3 -> columns-first L3 (v21's dropped attempt)"),
    "v20_store2d": ("#4878a8", "L3 reshaped -> gStoreLayoutC"),
    "mfma_read": ("#6aa56a", "load in MFMA layout -> gStoreLayoutC (owner path)"),
    "writer_cvt": ("#e0a040", "writer converts, row-major 16 B per lane partial"),
    "writer_cvt8": ("#8064a2", "writer converts, 32 B per lane partial (holes)"),
}


def slice_figure():
    rows = tables(os.path.join(HERE, "results", "slice.md"))["slice"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5))
    for col, setup in enumerate(("tiles 16", "CUs 256")):
        for row, (field, title) in enumerate((("C store us", "C store (convert + bf16 store)"),
                                              ("publish us", "publish (incl. the writer's conversion)"))):
            ax = axes[row, col]
            ns = [2, 4, 8, 16]
            width = 0.13
            for k, (v, (color, label)) in enumerate(VAR_STYLE.items()):
                vals = [float(next(r[field] for r in rows if r["variant"] == v and r["setup"] == setup
                                   and r["n"] == str(n))) for n in ns]
                xs = [i + (k - 2.5) * width for i in range(len(ns))]
                ax.bar(xs, vals, width, color=color, label=label)
            ax.set_xticks(range(len(ns)))
            ax.set_xticklabels([f"n = {n}" for n in ns])
            ax.set_ylabel("us")
            ax.set_title(f"{title}, {setup}", fontsize=10)
            ax.grid(axis="y", alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.text(0.01, 0.01, "Synthetic one-tile reduce-scatter, every tile on one XCD, cold caches, medians of 30 launches; "
             "every C element checked. Source: results/slice.md.", fontsize=8, color="0.3")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "slice.png")


def splits_figure():
    rows = tables(os.path.join(HERE, "results", "splits.md"))["splits"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6), sharey=True)
    styles = [("R = N, plain (v20)", "#7f7f7f"), ("R = N, block", "#c0504d"),
              ("R = 2^floor(log2 N), plain", "#e0a040"), ("R = 2^floor(log2 N), band", "#4878a8")]
    for ax, setup in zip(axes, ("tiles 16", "CUs 256")):
        ns = sorted({int(r["N"]) for r in rows if r["setup"] == setup})
        width = 0.2
        for i, n in enumerate(ns):
            rs = [r for r in rows if r["setup"] == setup and int(r["N"]) == n]
            for k, (label, color) in enumerate(styles):
                all_read = k <= 1  # styles 0, 1: R = N; 2, 3: R = the largest power of two <= N
                store = ("plain", "block", "plain", "band")[k]
                if n & (n - 1) == 0:  # a power of two: R = N for both plain and band
                    all_read, store = True, ("plain", None, None, "band")[k]
                hit = [r for r in rs if store and r["store"] == store and (int(r["R"]) == n) == all_read]
                if hit:
                    ax.bar(i + (k - 1.5) * width, float(hit[0]["slice us"]), width, color=color,
                           label=label if i == 0 else None)
        ax.set_xticks(range(len(ns)))
        ax.set_xticklabels([f"N = {n}" for n in ns])
        ax.set_title(f"{setup}: the critical reader's reads + C store", fontsize=10)
        ax.set_ylabel("us")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.text(0.01, 0.01, "fp32 partials. Power-of-two N: R = N only (plain and band). Cold caches, medians of 30 "
             "launches; every C element checked. Source: results/splits.md.", fontsize=8, color="0.3")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "splits.png")


def bf16_figure():
    rows = tables(os.path.join(HERE, "results", "bf16.md"))["bf16"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    styles = {"fp32": "#4878a8", "bf16": "#e0a040", "bf16_pairs": "#6aa56a"}
    labels = {"fp32": "fp32 partials (16 B per lane)", "bf16": "bf16, v20's map (8 B per lane)",
              "bf16_pairs": "bf16, register pairs (16 B per lane)"}
    for ax, setup in zip(axes, ("tiles 16", "CUs 256")):
        ns = [2, 4, 8, 16]
        for k, (part, color) in enumerate(styles.items()):
            sl = [float(next(r["slice us"] for r in rows if r["setup"] == setup and r["n"] == str(n)
                             and r["partials"] == part)) for n in ns]
            ax.bar([i + (k - 1) * 0.27 for i in range(len(ns))], sl, 0.27, color=color, label=labels[part])
        ax.set_xticks(range(len(ns)))
        ax.set_xticklabels([f"n = {n}" for n in ns])
        ax.set_title(f"{setup}: the critical reader's reads + C store", fontsize=10)
        ax.set_ylabel("us")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.text(0.01, 0.01, "Slice loaded in the MFMA layout (or the pair layout, reshaped to it for free), summed in fp32, "
             "converted like v20's owner store. Cold caches, medians of 30 launches; every C element checked. "
             "Source: results/bf16.md.", fontsize=8, color="0.3")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "bf16.png")


def save(fig, name):
    os.makedirs(IMG, exist_ok=True)
    out = os.path.join(IMG, name)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    probe_figure()
    slice_figure()
    splits_figure()
    bf16_figure()
