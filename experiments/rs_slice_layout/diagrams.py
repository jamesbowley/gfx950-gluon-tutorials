"""Explanatory diagrams (no GPU).

    python diagrams.py     # images/writer_cvt_widths.png, register_pairs.png, store_patterns.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "images")
LANE = ["#c0392b", "#e67e22", "#27ae60", "#2e86c1"]  # lanes 0-3: one 4-lane group (one L1 cycle)
OTHER = "#d5d8dc"


def strip(ax, y, chunks, n, unit, label, note):
    """A row of n memory chunks of `unit` bytes; chunks: {chunk index: lane}."""
    for k in range(n):
        lane = chunks.get(k)
        color = OTHER if lane is None or lane >= 4 else LANE[lane]
        ax.add_patch(Rectangle((k * unit, y), unit, 0.8, facecolor=color if lane is not None else "white",
                               edgecolor="0.5", lw=0.6))
        if lane is not None:
            ax.text(k * unit + unit / 2, y + 0.4, str(lane), ha="center", va="center", fontsize=7,
                    color="white" if lane < 4 else "0.3", fontweight="bold" if lane < 4 else "normal")
    for b in range(0, n * unit + 1, 128):
        ax.plot([b, b], [y - 0.1, y + 0.9], color="k", lw=1.2)
    ax.text(-8, y + 0.4, label, ha="right", va="center", fontsize=8.5)
    ax.text(n * unit + 8, y + 0.4, note, ha="left", va="center", fontsize=8.5)


def writer_cvt_widths():
    fig, ax = plt.subplots(figsize=(16, 6.2))
    W = 256  # bytes of one row shown
    # partial: fp32 row-major, 16-byte chunks = 4 columns
    strip(ax, 6, {k: k for k in range(16)}, 16, 16, "writer stores (W_ROW, 4 fp32 per lane)",
          "lanes 0-3: 64 contiguous bytes. Coalesced.")
    strip(ax, 4.6, {2 * l: l for l in range(8)}, 16, 16, "reader, 8 fp32 per lane: load 1",
          "lane l holds columns 8l .. 8l+7 = 32 bytes,\nso each 16-byte load leaves 16-byte gaps")
    strip(ax, 3.6, {2 * l + 1: l for l in range(8)}, 16, 16, "reader, 8 fp32 per lane: load 2", "")
    strip(ax, 2.2, {l: l for l in range(16)}, 16, 16, "  ...then the C store (bf16, 16 B per lane)",
          "lanes 0-3: 64 contiguous bytes of C")
    strip(ax, 0.8, {l: l for l in range(16)}, 16, 16, "or: reader, 4 fp32 per lane (coalesced load)",
          "lanes 0-3: 64 contiguous bytes. Coalesced...")
    strip(ax, -0.6, {l: l for l in range(32)}, 32, 8, "  ...then the C store (bf16, 8 B per lane)",
          "...but each lane has only 4 values = 8 bytes:\nlanes 0-3 store 32 bytes (dwordx2)")
    ax.set_xlim(-330, W + 330)
    ax.set_ylim(-1.2, 7.5)
    ax.axis("off")
    ax.set_title("Why the reader cannot both load and store coalesced after the writer converts: one partial row "
                 "(fp32, 256 bytes shown; black lines: 128-byte cache lines)\n"
                 "fp32 -> bf16 halves the bytes. A lane's 16 contiguous bytes of fp32 are only 8 bytes of bf16. To "
                 "store 16 bytes of bf16, a lane needs 8 values = 32 bytes of fp32: two loads,\nand the lanes of each "
                 "load then sit 32 bytes apart. Without moving values between lanes (a convert_layout), one side "
                 "loses.", fontsize=10)
    save(fig, "writer_cvt_widths.png")


def register_pairs():
    fig, ax = plt.subplots(figsize=(16, 5.6))
    # lane 0's registers in the MFMA layout: vector v holds 4 values; v's bit 0 is +32 columns
    for k in range(8):
        v = k // 4
        ax.add_patch(Rectangle((k * 1.0, 4), 1.0, 0.8, facecolor=["#4878a8", "#6aa56a"][v], edgecolor="white"))
        ax.text(k + 0.5, 4.4, f"r{k}\ncol {k % 4 + 32 * v}", ha="center", va="center", fontsize=7, color="white")
    ax.text(-0.3, 4.4, "lane 0's first 8 registers\n(MFMA layout, row 0)", ha="right", va="center", fontsize=9)
    ax.text(8.3, 4.4, "registers 0-3: columns 0-3 of block v0;  registers 4-7: columns 32-35 (block v1).\n"
            "The MFMA layout's register bits are col 1, col 2, col 32, col 64, row 32, row 64.", va="center",
            fontsize=9)
    # v20 map in bf16
    for k in range(8):
        v = k // 4
        x = (k % 4) * 0.5 + v * 6
        ax.add_patch(Rectangle((x, 2.2), 0.5, 0.8, facecolor=["#4878a8", "#6aa56a"][v], edgecolor="white"))
    ax.add_patch(Rectangle((2.0, 2.2), 4.0, 0.8, facecolor="white", edgecolor="0.6", ls="--"))
    ax.text(4.0, 2.6, "the rest of v0: lanes 1-63,\nthen warps 1-3 (2 KiB in all)", ha="center", va="center",
            fontsize=7.5, color="0.4")
    ax.text(-0.3, 2.6, "bf16, v20's map:\none store per register vector", ha="right", va="center", fontsize=9)
    ax.text(8.3, 2.6, "each vector's 4 bf16 = 8 bytes per lane (dwordx2); v1's 8 bytes come 2 KiB later.\n"
            "Contiguous across the wave (512 bytes per instruction), but 8-byte lanes.", va="center", fontsize=9)
    for k in range(8):
        v = k // 4
        ax.add_patch(Rectangle((k * 0.5, 0.4), 0.5, 0.8, facecolor=["#4878a8", "#6aa56a"][v], edgecolor="white"))
    ax.add_patch(Rectangle((4.0, 0.4), 4.0, 0.8, facecolor="white", edgecolor="0.6", ls="--"))
    ax.text(6.0, 0.8, "lane 1's 16 bytes", ha="center", va="center", fontsize=7.5, color="0.4")
    ax.text(-0.3, 0.8, "bf16, register pairs:\none store for both vectors", ha="right", va="center", fontsize=9)
    ax.text(8.3, 0.8, "registers 0-7 (two vectors 32 columns apart) = 16 contiguous bytes per lane (dwordx4),\n"
            "1 KiB per instruction. A reshape/permute makes them one tensor dimension so the compiler sees it;\n"
            "it moves no data. The reader loads the same map and reshapes back to the MFMA layout for free.",
            va="center", fontsize=9)
    ax.set_xlim(-6, 24)
    ax.set_ylim(0, 5.3)
    ax.axis("off")
    ax.set_title("Register pairing for bf16 partials (each small box: one bf16 value; widths to scale in bytes)",
                 fontsize=10)
    save(fig, "register_pairs.png")


def store_patterns():
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
    cases = [
        ("plain (v20): straight from the slice layout",
         "4 bf16 = 8 B per lane; lanes 0-3 on 4 rows.\nOne instruction: 16 rows x 32 B.\nNo conversion. Works for any slice.",
         [(r, 0, 8, r) for r in range(4)], [(r, 8 * (l // 16) % 32, 8) for l in range(64) for r in [l % 16]]),
        ("block: each 32x32 block converted to row order",
         "8 bf16 = 16 B per lane; lanes 0-3 on one row:\n64 contiguous bytes (one block row). One instruction:\n"
         "16 rows x 64 B. 4-round conversion (509 ns per\n128x128). Works for any group of blocks.",
         [(0, 16 * l, 16, l) for l in range(4)], [(l // 4, 16 * (l % 4), 16) for l in range(64)]),
        ("band: the group reshaped to its 2-D band, converted",
         "8 bf16 = 16 B per lane; 16 lanes on one row:\n256 contiguous bytes. One instruction: 4 rows x 256 B.\n"
         "2-round conversion (366 ns). Needs a group of blocks\nthat forms a rectangle: aligned, a power of two.",
         [(0, 16 * l, 16, l) for l in range(4)], [(l // 16, 16 * (l % 16), 16) for l in range(64)]),
    ]
    for ax, (title, note, group, wave) in zip(axes, cases):
        for r in range(16):
            for k in range(4):
                ax.add_patch(Rectangle((k * 128, r), 128, 1, facecolor="white", edgecolor="0.8", lw=0.5))
        for r, x, w in wave:
            ax.add_patch(Rectangle((x, r), w, 1, facecolor=OTHER, edgecolor="white", lw=0.4))
        for r, x, w, lane in group:
            ax.add_patch(Rectangle((x, r), w, 1, facecolor=LANE[lane], edgecolor="white", lw=0.4))
        ax.set_xlim(0, 512)
        ax.set_ylim(16, 0)
        ax.set_xticks(range(0, 513, 128))
        ax.set_yticks([0.5, 3.5, 7.5, 15.5])
        ax.set_yticklabels(["row 0", "row 3", "row 7", "row 15"], fontsize=7)
        ax.set_xlabel("byte within the C row (grid: 128-byte lines; rows are 8 KiB apart)")
        ax.set_title(title, fontsize=10)
        ax.text(0.98, 0.04, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                bbox=dict(facecolor="white", edgecolor="0.7"))
    fig.suptitle("The C bytes one wave's store instruction writes (grey) and those of lanes 0-3, one 4-lane group "
                 "(coloured)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "store_patterns.png")


BLOCK_BIT = {1: ("col", 32), 2: ("col", 64), 4: ("row", 32), 8: ("row", 64)}  # v20's block map
BITS = [("col", 1 << k) for k in range(7)] + [("row", 1 << k) for k in range(7)]


def read_layouts():
    """{name: {(axis, value): (kind, index)}} from results/layouts.txt, in quadrant (row, col) terms."""
    import ast
    import re
    out = {}
    for line in open(os.path.join(HERE, "results", "layouts.txt")):
        m = re.match(r"(.+?): DistributedLinearLayout\(reg_bases=(\[.*?\]\]|\[\]), lane_bases=(\[.*?\]\]|\[\]), "
                     r"warp_bases=(\[.*?\]\]|\[\])", line)
        if not m:
            continue
        bits = {}
        for kind, txt in zip(("reg", "lane", "warp"), m.groups()[1:]):
            for i, b in enumerate(ast.literal_eval(txt)):
                if len(b) == 3:  # (block, row, col)
                    blk, r, c = b
                    key = BLOCK_BIT[blk] if blk else ("row", r) if r else ("col", c)
                else:
                    key = ("row", b[0]) if b[0] else ("col", b[1])
                bits[key] = (kind, i)
        out[m.group(1)] = bits
    return out


def footprint(bits, lane):
    """Quadrant (row, col) of every element the given lane of warp 0 holds."""
    regs = sorted((i, key) for key, (kind, i) in bits.items() if kind == "reg")
    lanes = {i: key for key, (kind, i) in bits.items() if kind == "lane"}
    r0 = sum(v for k, (ax, v) in lanes.items() if lane >> k & 1 and ax == "row")
    c0 = sum(v for k, (ax, v) in lanes.items() if lane >> k & 1 and ax == "col")
    pts = []
    for mask in range(1 << len(regs)):
        r, c = r0, c0
        for j, (_, (ax, v)) in enumerate(regs):
            if mask >> j & 1:
                r, c = (r + v, c) if ax == "row" else (r, c + v)
        pts.append((r, c))
    return pts


def layout_bits():
    L = read_layouts()
    rows = [
        ("L3 reshaped to (128, 128)", "L3: v20's slice read (source)", None),
        ("MFMA (128, 128)", "MFMA: the accumulator (owner / DP source)", None),
        ("L3 reshaped to (128, 128)", "-> MFMA (target)", ("MFMA (128, 128)", 8)),
        ("L3_OLD (16, 32, 32)", "-> columns-first L3 (v21's target)", ("L3_OLD (16, 32, 32)", 8)),
        ("S3 (16, 32, 32)", "-> S3 ('block' target)", ("S3 (16, 32, 32)", 4)),
        ("gStoreLayoutC (128, 128)", "-> gStoreLayoutC ('band' / DP target)", ("gStoreLayoutC (128, 128)", 2)),
    ]
    color = {"reg": "#6aa56a", "lane": "#4878a8", "warp": "#e0a040"}
    src_regs = {k for k, (kind, _) in L["L3 reshaped to (128, 128)"].items() if kind == "reg"}
    fig = plt.figure(figsize=(18, 13))
    ax = fig.add_axes([0.19, 0.56, 0.55, 0.36])
    for y, (name, label, tgt) in enumerate(rows):
        bits = L[tgt[0] if tgt else name]
        for x, key in enumerate(BITS):
            kind, i = bits[key]
            shared = tgt is not None and kind == "reg" and key in src_regs
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor=color[kind], edgecolor="k" if shared else "white",
                                   lw=2.5 if shared else 1))
            ax.text(x + 0.5, y + 0.5, f"{kind[0]}{i}", ha="center", va="center", fontsize=8, color="white")
        ax.text(-0.3, y + 0.5, label, ha="right", va="center", fontsize=9)
        if tgt:
            n = sum(1 for key in BITS if bits[key][0] == "reg" and key in src_regs)
            ax.text(len(BITS) + 0.3, y + 0.5, f"{n} register bits shared with L3 -> {tgt[1]} LDS rounds",
                    va="center", fontsize=9)
    ax.axhline(2, color="k", lw=1.5)
    ax.set_xlim(0, len(BITS))
    ax.set_ylim(len(rows), 0)
    ax.set_xticks([x + 0.5 for x in range(len(BITS))])
    ax.set_xticklabels([f"{a} {v}" for a, v in BITS], fontsize=8, rotation=45)
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Which hardware index holds each bit of a quadrant element's (row, col): r = register, l = lane, "
                 "w = warp\nBlack border: a register bit in both L3 and the target. Triton splits a conversion into "
                 "one LDS round per such bit beyond the 16-byte vector (buildReps): rounds = 2^(shared - 3)",
                 fontsize=10)
    for k, (key, title) in enumerate((("L3 reshaped to (128, 128)", "L3 (v20's read) and MFMA"),
                                      ("L3_OLD (16, 32, 32)", "columns-first L3 (v21's target)"),
                                      ("gStoreLayoutC (128, 128)", "gStoreLayoutC ('band' / DP target)"))):
        a = fig.add_axes([0.05 + k * 0.32, 0.08, 0.27, 0.36])
        a.add_patch(Rectangle((0, 0), 128, 128, facecolor="white", edgecolor="0.6"))
        for v in range(0, 129, 16):
            a.axhline(v, color="0.92", lw=0.5)
            a.axvline(v, color="0.92", lw=0.5)
        for lane, col in ((0, "#c0392b"), (1, "#2e86c1")):
            for r, c in footprint(L[key], lane):
                a.add_patch(Rectangle((c, r - 1), 1, 3, facecolor=col, edgecolor=col, lw=1.5))
        a.set_xlim(0, 128)
        a.set_ylim(128, 0)
        a.set_aspect("equal")
        a.set_xticks([0, 32, 64, 96, 128])
        a.set_yticks([0, 32, 64, 96, 128])
        a.set_title(f"{title}\nlane 0 (red) and lane 1 (blue) of warp 0: 64 elements each", fontsize=9)
    fig.text(0.05, 0.01, "Lane 0 holds the same 64 elements in L3, MFMA and the columns-first L3 (same register "
             "bits); only which lane holds the others changes, so each register slice can be exchanged on its own: "
             "many small rounds.\ngStoreLayoutC gives every lane 8 consecutive columns on 8 rows 16 apart: different "
             "register bits, so most of the data must move at once: 2 large rounds.", fontsize=9, color="0.25")
    save(fig, "layout_bits.png")


def owners(bits):
    """{(row, col): (warp, lane)} over the 128x128 quadrant."""
    by_kind = {"reg": {}, "lane": {}, "warp": {}}
    for key, (kind, i) in bits.items():
        by_kind[kind][i] = key
    out = {}
    nr, nl, nw = len(by_kind["reg"]), len(by_kind["lane"]), len(by_kind["warp"])
    for w in range(1 << nw):
        for ln in range(1 << nl):
            for rg in range(1 << nr):
                r = c = 0
                for kind, val in (("reg", rg), ("lane", ln), ("warp", w)):
                    for i, (ax, v) in by_kind[kind].items():
                        if val >> i & 1:
                            if ax == "row":
                                r += v
                            else:
                                c += v
                out[(r, c)] = (w, ln)
    return out


def layout_pictures():
    import matplotlib.colors as mcolors
    L = read_layouts()
    panels = [
        ("L3 (16, 32, 32)", "v20's slice layout: how the partials are read (the source)", ""),
        ("L3_OLD (16, 32, 32)", "columns-first L3: v21's dropped C-store target", "8 LDS rounds, 650 ns"),
        ("S3 (16, 32, 32)", "S3, 3-D: the 'block' C-store target", "4 LDS rounds, 510 ns"),
        ("S3_ROW (16, 32, 32)", "S3_ROW, 3-D: lanes continue into the next block", "2 LDS rounds, 381 ns"),
        ("gStoreLayoutC (128, 128)", "gStoreLayoutC, 2-D: the 'band' target (and v20's data-parallel C store)",
         "2 LDS rounds, 366 ns"),
    ]
    lane_colors = [mcolors.to_rgb(c) for c in plt.get_cmap("tab10").colors[:8]]
    warp_tint = [mcolors.to_rgb(c) for c in ("#cfcfcf", "#d6e4f2", "#dcecd5", "#f6e8c8")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(16, 17))
    for ax, (name, title, cost) in zip(axes, panels):
        own = owners(L[name])
        img = [[None] * 128 for _ in range(32)]
        for r in range(32):
            for c in range(128):
                w, ln = own[(r, c)]
                img[r][c] = lane_colors[ln] if (w == 0 and ln < 8) else warp_tint[w]
        ax.imshow(img, interpolation="nearest", aspect="equal", extent=(0, 128, 32, 0))
        for x in (32, 64, 96):
            ax.axvline(x, color="k", lw=2)
        for y in (16,):
            ax.axhline(y, color="0.5", lw=0.6, ls=":")
        ax.set_xticks([0, 32, 64, 96, 128])
        ax.set_yticks([0, 8, 16, 24, 32])
        ax.tick_params(labelsize=8)
        ax.set_title(f"{title}" + (f"   ->   {cost}" if cost else ""), fontsize=10, loc="left")
        ax.set_xticks([16, 48, 80, 112], minor=True)
        ax.set_xticklabels([f"block {k}" for k in range(4)], minor=True, fontsize=8)
        ax.tick_params(axis="x", which="minor", length=0, pad=14)
    handles = [Rectangle((0, 0), 1, 1, color=lane_colors[i]) for i in range(8)] + \
        [Rectangle((0, 0), 1, 1, color=warp_tint[w]) for w in range(4)]
    labels = [f"warp 0, lane {i}" for i in range(8)] + ["warp 0, other lanes"] + [f"warp {w}" for w in (1, 2, 3)]
    fig.legend(handles, labels, loc="lower center", ncol=12, fontsize=8, frameon=False)
    fig.suptitle("Who holds each element of the first 32 rows of a 128x128 quadrant (blocks 0-3 side by side, "
                 "black lines: 32x32 block edges)\nThe 3-D layouts name an element (block, row, col); here every "
                 "element is drawn at its place in C. S3_ROW and gStoreLayoutC are the same picture.", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    save(fig, "layout_pictures.png")


def save(fig, name):
    os.makedirs(IMG, exist_ok=True)
    out = os.path.join(IMG, name)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    writer_cvt_widths()
    register_pairs()
    store_patterns()
    layout_bits()
    layout_pictures()
