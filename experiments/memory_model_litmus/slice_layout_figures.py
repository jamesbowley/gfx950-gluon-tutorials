"""Why v20's reduce-scatter C store is not coalesced (no GPU, about a second).

    python slice_layout_figures.py     # images/slice_ownership.png, slice_memory_order.png,
                                       # slice_lane_order.png, slice_store_footprint.png

Every address comes from the formulas in v20's reduce_scatter_slice
(kernels/gemm/intra_wave/a16w16/v20_streamk_reduce_scatter/matmul_kernel.py):
- a partial slot is 64 blocks of 1024 fp32; block blk is the 32x32 square of the 256x256 tile at
  (row0, col0) below, and element (r, c) of it sits at float in_blk(r, c) of the block;
- program j of a tile split n ways sums blocks [64 j / n, 64 (j + 1) / n) and stores them to C;
- the slice's register layout is BlockedLayout([1, 1, 4], [1, 16, 4], [1, 2, 2], order): each lane
  holds 4 neighbouring columns of one row. v20 uses order [1, 2, 0] (rows first in lane order); before
  2026-10-04 it was [2, 1, 0] (columns first).
C is row-major bf16; the figures use N = 4096 columns, so C rows are 8192 bytes apart.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "images")
LINE = 128  # bytes per cache line
C_ROW = 8192  # bytes between C rows (N = 4096 bf16)
LANE_COLORS = ["#c0392b", "#e67e22", "#27ae60", "#2e86c1"]  # lanes 0-3: one L1 cycle
OTHER = "#d5d8dc"


def block_origin(blk):
    q, v = blk // 16, blk % 16
    return (q % 2) * 128 + (v // 4 % 2) * 32 + (v // 8) * 64, (q // 2) * 128 + (v % 2) * 32 + (v // 2 % 2) * 64


def in_blk(r, c):
    return (r % 16) * 4 + (r // 16) * 512 + c + (c // 4 % 4) * 60 + (c // 16) * 240


def lane_rc(lane, rows_first):
    """(row, first column) of warp 0's lane in a block; the lane holds columns c .. c + 3."""
    if rows_first:
        return lane % 16, 4 * (lane // 16)
    return lane // 4, 4 * (lane % 4)


def owner_map(n):
    """256x256 grid of the member that stores each C element."""
    own = [[-1] * 256 for _ in range(256)]
    for j in range(n):
        for blk in range(64 * j // n, 64 * (j + 1) // n):
            r0, c0 = block_origin(blk)
            for r in range(32):
                for c in range(32):
                    assert own[r0 + r][c0 + c] == -1
                    own[r0 + r][c0 + c] = j
    assert all(x >= 0 for row in own for x in row), "the members' blocks must cover the tile"
    return own


def checks():
    for lane in range(64):
        r, c = lane_rc(lane, True)
        assert [in_blk(r, c + k) for k in range(4)] == [4 * lane + k for k in range(4)]
    old = [in_blk(*lane_rc(lane, False)) * 4 for lane in range(4)]
    assert old == [0, 256, 512, 768], old
    for n in (2, 4, 8, 16):
        own = owner_map(n)
        for j in range(n):
            for row in own:
                cols = [c for c, x in enumerate(row) if x == j]
                if cols:
                    assert len(cols) == 128 and cols[-1] - cols[0] == 127 and cols[0] % 128 == 0, (n, j)


# ---------------------------------------------------------------------------------------------


def ownership():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    cmap = plt.get_cmap("tab20")
    for ax, n in zip(axes, (4, 8, 16)):
        own = owner_map(n)
        ax.imshow([[cmap(x % 20) for x in row] for row in own], interpolation="nearest")
        for k in range(0, 257, 32):
            ax.axhline(k - 0.5, color="white", lw=0.6)
            ax.axvline(k - 0.5, color="white", lw=0.6)
        for j in range(n):
            pts = [(r, c) for r in range(0, 256, 8) for c in range(0, 256, 8) if own[r][c] == j]
            rm = sum(p[0] for p in pts) / len(pts)
            cm = sum(p[1] for p in pts) / len(pts)
            ax.text(cm, rm, f"p{j}", ha="center", va="center", fontsize=9, fontweight="bold")
        rows = 256 * 256 // (128 * n)
        assert sum(x == 0 for row in own for x in row) == rows * 128
        ax.set_title(f"n = {n}: program pj stores {rows} rows x 128 columns\n"
                     f"of C ({128 // n} KiB of bf16, 256 B per row)", fontsize=10)
        ax.set_xticks([0, 64, 128, 192, 255])
        ax.set_yticks([0, 64, 128, 192, 255])
        ax.set_xlabel("column of the 256x256 tile")
        ax.set_ylabel("row")
    fig.suptitle("One 256x256 C tile split n ways: which part of C each program stores (white lines: the 32x32 "
                 "blocks)", fontsize=12)
    fig.text(0.01, 0.01, "p0 .. p(n-1) are the n programs that split the tile's K loop. Each holds a partial of the "
             "WHOLE tile; in the reduce-scatter step program j sums blocks [64 j / n, 64 (j + 1) / n) of all n partials\n"
             "and stores that part of C. With v20's block map these are full-quadrant-width rectangles, so a program's "
             "share of C is not the obstacle to coalesced stores: every one of its rows is 256 contiguous bytes.",
             fontsize=8.5, color="0.3")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    return save(fig, "slice_ownership.png")


def memory_order():
    """One wave's 16x16 piece: where each element lives in the partial and in C."""
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(15, 7.4))
    cmap = plt.get_cmap("tab20")
    for r in range(16):
        for c in range(16):
            f = in_blk(r, c)
            a0.add_patch(Rectangle((c, r), 1, 1, facecolor=cmap(f // 32 * 2 % 20), edgecolor="white", lw=0.5))
            a0.text(c + 0.5, r + 0.5, str(f), ha="center", va="center", fontsize=6)
            a1.add_patch(Rectangle((c, r), 1, 1, facecolor=cmap(r % 20), edgecolor="white", lw=0.5))
            a1.text(c + 0.5, r + 0.5, str(2 * c), ha="center", va="center", fontsize=6)
    for k in range(0, 17, 4):
        a0.axvline(k, color="k", lw=1)
    for g in range(4):
        a0.text(4 * g + 2, -0.35, f"lanes {16 * g}-{16 * g + 15}", ha="center", va="bottom", fontsize=8)
    for r in range(16):
        a0.text(16.1, r + 0.5, f"lane {r}, {r + 16}, {r + 32}, {r + 48}", va="center", fontsize=6, color="0.35")
    for ax, title in ((a0, "Partial slot (fp32, MFMA register order): float offset in the block\n"
                           "lane l holds row l % 16, columns 4 (l // 16) .. + 3 and stores them at floats 4 l .. 4 l + 3\n"
                           "colour = 128-byte line: addresses run DOWN the rows, one line is 8 rows x 4 columns\n"),
                      (a1, "C (row-major bf16): byte offset within the row\n"
                           "colour = 128-byte line. Addresses run ALONG a row: the 16 columns are 32 bytes\n"
                           "of one line, and the next row is 8192 bytes away\n")):
        ax.set_xlim(0, 16)
        ax.set_ylim(16, 0)
        ax.set_aspect("equal")
        ax.set_xticks([k + 0.5 for k in range(16)])
        ax.set_xticklabels(range(16), fontsize=7)
        ax.set_yticks([k + 0.5 for k in range(16)])
        ax.set_yticklabels(range(16), fontsize=7)
        ax.set_xlabel("column of C (within this 16 x 16 MFMA output tile)")
        ax.set_ylabel("row of C (within this 16 x 16 MFMA output tile)")
        ax.set_title(title, fontsize=9.5)
    fig.suptitle("One wave's 16 x 16 MFMA output tile (one register vector of 4 fp32 per lane): the same elements "
                 "in the two memories", fontsize=12)
    fig.text(0.01, 0.01, "A lane holds 4 neighbouring columns of one row. Lanes that are neighbours in the partial "
             "(next 16 bytes) are on the next ROW of C;\nlanes that are neighbours in C (next 8 bytes) are 256 bytes "
             "apart in the partial. One lane order cannot make both contiguous.", fontsize=8.5, color="0.3")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    return save(fig, "slice_memory_order.png")


def read_chunks(rows_first):
    """{lane: (line, 16-byte chunk)} of warp 0's partial read (the wave reads 1 KiB = 8 lines)."""
    out = {}
    for lane in range(64):
        f = in_blk(*lane_rc(lane, rows_first))
        out[lane] = (f // 32, f % 32 // 4)
    return out


def store_chunks(rows_first):
    """{lane: (C row, 8-byte chunk of the row's first line)} of warp 0's C store (block at the tile origin)."""
    out = {}
    for lane in range(64):
        r, c = lane_rc(lane, rows_first)
        out[lane] = (r, 2 * c // 8)
    return out


def draw_lines(ax, chunks, nlines, nchunks, line_label):
    for line in range(nlines):
        for k in range(nchunks):
            ax.add_patch(Rectangle((k, line), 1, 1, facecolor="white", edgecolor="0.8", lw=0.5))
    for lane, (line, k) in chunks.items():
        color = LANE_COLORS[lane] if lane < 4 else OTHER
        ax.add_patch(Rectangle((k, line), 1, 1, facecolor=color, edgecolor="white", lw=0.8))
        ax.text(k + 0.5, line + 0.5, str(lane), ha="center", va="center", fontsize=6,
                color="white" if lane < 4 else "0.35", fontweight="bold" if lane < 4 else "normal")
    ax.set_xlim(0, nchunks)
    ax.set_ylim(nlines, 0)
    ax.set_yticks([k + 0.5 for k in range(nlines)])
    ax.set_yticklabels([line_label(k) for k in range(nlines)], fontsize=7)
    ax.set_xticks([])
    touched = {chunks[lane][0] for lane in range(4)}
    return len(touched)


def lane_order():
    fig, axes = plt.subplots(2, 2, figsize=(16, 9.6), gridspec_kw={"width_ratios": [8, 16]})
    for row, (rows_first, name) in enumerate(((False, "before 2026-10-04: columns first, order [2, 1, 0]"),
                                              (True, "v20 now: rows first, order [1, 2, 0]"))):
        a_r, a_c = axes[row]
        nr = draw_lines(a_r, read_chunks(rows_first), 8, 8, lambda k: f"line {k} (bytes {128 * k}-{128 * k + 127})")
        a_r.set_xlabel("16-byte chunk of the line")
        a_r.set_title(f"{name}\nPARTIAL READ (16 B per lane, the wave reads 1 KiB): lanes 0-3 touch {nr} "
                      f"line{'s' if nr > 1 else ''}", fontsize=9.5)
        sc = store_chunks(rows_first)
        nc = draw_lines(a_c, sc, 16, 16, lambda k: f"C row {k} (+{k * C_ROW} B)")
        a_c.set_xlabel("8-byte chunk of the row's first 128-byte line (the wave writes the first 32 bytes of 16 rows)")
        a_c.set_title(f"{name}\nC STORE (8 B per lane): lanes 0-3 touch {nc} line{'s' if nc > 1 else ''}",
                      fontsize=9.5)
    fig.suptitle("Lanes 0-3 (coloured) are what the L1 handles in one cycle; it does not merge across cycles",
                 fontsize=12)
    fig.text(0.01, 0.012, "Measured in v21's slice timing (same slice code as v20): the change cut the slice's reads "
             "from 7.8-8.4 to 4.5-5.0 us and made bias + C store 0.3-1.3 us slower; the kernel got 1.9-4.5 us faster. "
             "The reads carry about 16x the bytes of the C store\n(each member reads 256 KiB of fp32 slices and stores "
             "128 / n KiB of bf16), so the reads get the contiguous order. Either way the wave's C store writes the "
             "same 512 bytes: 32 bytes of 16 rows.", fontsize=8.5, color="0.3")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    return save(fig, "slice_lane_order.png")


def store_footprint():
    """One wave store instruction's C bytes, for three store layouts (rows 0-15, bytes 0-1023)."""
    cases = [
        ("v20 reduce-scatter slice store (no convert)", "4 bf16 = 8 B per lane; 16 rows x 32 B = 512 B;\n"
         "16 lines, each a quarter written", [(r, 0, 32) for r in range(16)]),
        ("v20 owner / data-parallel C store, after convert_layout\n(gStoreLayoutC = BlockedLayout([1, 8], [4, 16], "
         "[4, 1], [1, 0]))", "8 bf16 = 16 B per lane; 4 rows x 256 B = 1 KiB;\n8 whole lines",
         [(r, 0, 256) for r in range(4)]),
        ("synthetic exchange _store_c (C tile as a linear buffer)", "8 bf16 = 16 B per lane; 1 KiB contiguous;\n"
         "8 whole lines", [(0, 0, 1024)]),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(14, 9.5))
    for ax, (title, note, spans) in zip(axes, cases):
        for r in range(16):
            for k in range(8):
                ax.add_patch(Rectangle((k * LINE, r), LINE, 1, facecolor="white", edgecolor="0.75", lw=0.5))
        for r, b0, nb in spans:
            ax.add_patch(Rectangle((b0, r), nb, 1, facecolor="#2e86c1", edgecolor="white", lw=0.5))
        ax.set_xlim(0, 1024)
        ax.set_ylim(16, 0)
        ax.set_xticks(range(0, 1025, 128))
        ax.set_yticks([0.5, 3.5, 7.5, 15.5])
        ax.set_yticklabels(["row 0", "row 3", "row 7", "row 15"], fontsize=7)
        ax.set_title(title, fontsize=10, loc="left")
        ax.text(1.01, 0.5, note, transform=ax.transAxes, fontsize=9, va="center")
    axes[-1].set_xlabel("byte within the row (grid: 128-byte lines); rows are 8192 bytes apart")
    fig.suptitle("The C bytes one wave's store instruction writes", fontsize=12)
    fig.text(0.01, 0.01, "Tried for the slice and dropped (v21 entry 2): loading rows first and convert_layout to the "
             "store layout before the bias and C store. The conversion goes through LDS, and bias + C rose from "
             "1.0-3.9 us to 2.4-10.5 us.", fontsize=8.5, color="0.3")
    fig.tight_layout(rect=(0, 0.035, 0.8, 0.96))
    return save(fig, "slice_store_footprint.png")


def save(fig, name):
    os.makedirs(IMG, exist_ok=True)
    out = os.path.join(IMG, name)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(out)
    return out


if __name__ == "__main__":
    checks()
    ownership()
    memory_order()
    lane_order()
    store_footprint()
