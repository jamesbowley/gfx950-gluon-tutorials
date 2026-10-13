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
"""Writes variants/<name>.py: variants/base.py (v26's matmul_kernel.py, unchanged) with named patches.

    python experiments/endtoend_spills/make_variants.py crow crow+ccol

A name joins patches with '+'. Each patch is a list of (old, new[, count]) replacements that must match once
(or count times).
Every patch is compiled only in the end-to-end segment loop (streamk_segment's SEG_LOOP, which the `seg`
patch adds and every other patch pulls in), so the other builds are unchanged.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

C_OFFSETS = """    col = pid_n * BLOCK_N + offs_cn
    c_offsets = stride_cm * offs_cm[:, None] + stride_cn * col[None, :]
    c_tr_base = c_tl_base + BLOCK_N * stride_cn // 2
    c_bl_base = c_tl_base + BLOCK_M * stride_cm // 2
    c_br_base = c_bl_base + BLOCK_N * stride_cn // 2

    if ATOMIC_C:"""

PATCHES = {
    # streamk_segment's SEG_LOOP: set by the end-to-end path, whose segments run in a loop.
    "seg": [
        ("""    K_PAD: gl.constexpr = 0,
    K_RES: gl.constexpr = 0,
):
    \"\"\"K-steps [k0, k0 + 2 * n_pairs)""",
         """    K_PAD: gl.constexpr = 0,
    K_RES: gl.constexpr = 0,
    SEG_LOOP: gl.constexpr = False,
):
    \"\"\"K-steps [k0, k0 + 2 * n_pairs)"""),
        ("""                False, False, NUM_PROGRAMS, epoch, M, N, EDGE_M, EDGE_N, gLoadLayoutA, gLoadLayoutB, EDGE_K, K_PAD, K_RES,
            )
            cur = seg_start""",
         """                False, False, NUM_PROGRAMS, epoch, M, N, EDGE_M, EDGE_N, gLoadLayoutA, gLoadLayoutB, EDGE_K, K_PAD, K_RES,
                SEG_LOOP=True,
            )
            cur = seg_start"""),
    ],
    # C row offsets clamped to the tile's last row (only rows the stores mask off change, in the top and
    # the bottom quadrants), so stride_cm * row is not hoisted above the segment loop and spilled.
    "crow": [(C_OFFSETS, C_OFFSETS.replace(
        "    c_offsets = stride_cm * offs_cm[:, None] + stride_cn * col[None, :]\n",
        """    if SEG_LOOP:
        c_offsets = stride_cm * gl.minimum(offs_cm, rows_left - 1)[:, None] + stride_cn * col[None, :]
    else:
        c_offsets = stride_cm * offs_cm[:, None] + stride_cn * col[None, :]
"""))],
    # Both halves tile-dependent: columns clamped to the tile's last column as well.
    "ccol": [(C_OFFSETS, C_OFFSETS.replace(
        "    col = pid_n * BLOCK_N + offs_cn\n",
        """    if SEG_LOOP:
        col = pid_n * BLOCK_N + gl.minimum(offs_cn, cols_left - 1)
    else:
        col = pid_n * BLOCK_N + offs_cn
""").replace(
        "    c_offsets = stride_cm * offs_cm[:, None] + stride_cn * col[None, :]\n",
        """    if SEG_LOOP:
        c_offsets = stride_cm * gl.minimum(offs_cm, rows_left - 1)[:, None] + stride_cn * col[None, :]
    else:
        c_offsets = stride_cm * offs_cm[:, None] + stride_cn * col[None, :]
"""))],
    # The C column offsets tile-dependent too, clamped by 8-column group (a lane's 16-byte store) so the
    # stores stay vectorised (ccol's per-column clamp scalarises them).
    "cgrp": [("""    col = pid_n * BLOCK_N + offs_cn
    if SEG_LOOP:
""", """    if SEG_LOOP:
        col = pid_n * BLOCK_N + gl.minimum(offs_cn // 8, (cols_left - 1) // 8) * 8 + offs_cn % 8
    else:
        col = pid_n * BLOCK_N + offs_cn
    if SEG_LOOP:
""")],
    # fixup_quadrant's SEG_LOOP and hi, a bound HALF_M - 1 + (n_pairs - 1): never below HALF_M - 1
    # (n_pairs >= 1), which the compiler cannot know, so clamping an index below HALF_M to it changes nothing
    # but makes the result depend on the segment. No change on its own.
    "fqseg": [
        ("""    EDGE_N: gl.constexpr = False,
):
    \"\"\"Quadrant q (tl, bl, tr, br)""",
         """    EDGE_N: gl.constexpr = False,
    hi=0,
    SEG_LOOP: gl.constexpr = False,
):
    \"\"\"Quadrant q (tl, bl, tr, br)"""),
        ("""            ".wt", rows_left, cols_left, EDGE_M, EDGE_N,
        )""",
         """            ".wt", rows_left, cols_left, EDGE_M, EDGE_N, BLOCK_M // 2 - 1 + n_pairs - 1, SEG_LOOP,
        )""", 4),
    ],
    # The partial slot offsets (lane_contiguous_offsets) of the owner ending depend on the segment: the row
    # index is clamped to hi, so the offsets are not hoisted above the segment loop.
    "pclamp": [
        ("""def lane_contiguous_offsets(HALF_M: gl.constexpr, HALF_N: gl.constexpr, mfmaLayout: gl.constexpr):""",
         """def lane_contiguous_offsets(HALF_M: gl.constexpr, HALF_N: gl.constexpr, mfmaLayout: gl.constexpr,
                            row_hi=0, CLAMP: gl.constexpr = False):"""),
        ("""    r = gl.arange(0, HALF_M, gl.SliceLayout(1, mfmaLayout))
    c = gl.arange(0, HALF_N, gl.SliceLayout(0, mfmaLayout))
    # The column term""",
         """    r = gl.arange(0, HALF_M, gl.SliceLayout(1, mfmaLayout))
    if CLAMP:
        r = gl.minimum(r, row_hi)
    c = gl.arange(0, HALF_N, gl.SliceLayout(0, mfmaLayout))
    # The column term"""),
        ("""        p_offsets = lane_contiguous_offsets(HALF_M, HALF_N, mfmaLayout)""",
         """        p_offsets = lane_contiguous_offsets(HALF_M, HALF_N, mfmaLayout, hi, SEG_LOOP)"""),
    ],
    # The same slot map built from a 1-D arange whose blocked layout is the map itself ([16 register vectors]
    # [4 warps][64 lanes][4 fp32]: element tid * 4 + 1024 v + e), reshaped and permuted into the quadrant's
    # (row, col) bits and converted to the MFMA layout with no data movement. Each register's offset is then
    # the thread's tid * 4 plus a constant.
    "plin": [
        ("""@gluon.constexpr_function
def pair_layout(rows):""",
         """@gluon.jit
def lane_contiguous_offsets_lin(HALF_M: gl.constexpr, HALF_N: gl.constexpr, mfmaLayout: gl.constexpr):
    \"\"\"lane_contiguous_offsets, from an arange in the map's own layout.\"\"\"
    gl.static_assert(HALF_M == 128 and HALF_N == 128, "bit map is for 128x128 quadrants")
    o = gl.arange(0, HALF_M * HALF_N, gl.BlockedLayout([4], [64], [4], [0]))
    # Bits of o, high to low: row 64, row 32, col 64, col 32 (register vector), row 16, col 16 (warp), col 8,
    # col 4, row 8, row 4, row 2, row 1 (lane), col 2, col 1 (element).
    x = gl.reshape(o, [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    x = gl.permute(x, [0, 1, 4, 8, 9, 10, 11, 2, 3, 5, 6, 7, 12, 13])
    return gl.convert_layout(gl.reshape(x, [HALF_M, HALF_N]), mfmaLayout, assert_trivial=True)


@gluon.constexpr_function
def pair_layout(rows):"""),
        ("""        p_offsets = lane_contiguous_offsets(HALF_M, HALF_N, mfmaLayout)""",
         """        if SEG_LOOP:
            p_offsets = lane_contiguous_offsets_lin(HALF_M, HALF_N, mfmaLayout)
        else:
            p_offsets = lane_contiguous_offsets(HALF_M, HALF_N, mfmaLayout)"""),
    ],
    # The owner's bias offsets depend on the segment (clamped to hi by group of 4, a lane's contiguous
    # columns), so they are not hoisted above the segment loop.
    "bgrp": [
        ("""                bias = gl.amd.cdna3.buffer_load(
                    ptr=bias_ptr + pid_n * BLOCK_N + (q // 2) * HALF_N, offsets=offs_pn
                )""",
         """                b_offs = offs_pn
                if SEG_LOOP:
                    b_offs = gl.minimum(offs_pn // 4, hi // 4) * 4 + offs_pn % 4
                bias = gl.amd.cdna3.buffer_load(
                    ptr=bias_ptr + pid_n * BLOCK_N + (q // 2) * HALF_N, offsets=b_offs
                )"""),
    ],
    # pclamp with the column index clamped too, by group of 4 (a lane's 16-byte access): no part of the
    # offsets is left to hoist.
    "pclamp2": [
        ("""        r = gl.minimum(r, row_hi)
    c = gl.arange(0, HALF_N, gl.SliceLayout(0, mfmaLayout))
""", """        r = gl.minimum(r, row_hi)
    c = gl.arange(0, HALF_N, gl.SliceLayout(0, mfmaLayout))
    if CLAMP:
        c = gl.minimum(c // 4, row_hi // 4) * 4 + c % 4
"""),
    ],
    # The segment loop's two endings as two calls with a constant ending (partial, or owner), so neither copy
    # carries the other's code.
    "split": [lambda s: split_segment_call(s)],
    # No persistent loop whenever there are no full tiles, on aligned shapes too (v26 only does it with an edge
    # flag on, to keep aligned builds unchanged).
    "nofull": [("""        HAS_FULL=total_tiles > STREAMK_TILES or not (EDGE_M or EDGE_N or EDGE_K),""",
                """        HAS_FULL=total_tiles > STREAMK_TILES,""")],
}
REQUIRES = {"crow": ["seg"], "ccol": ["seg"], "cgrp": ["seg", "crow"], "fqseg": ["seg"],
            "pclamp": ["seg", "fqseg"], "pclamp2": ["seg", "fqseg", "pclamp"], "bgrp": ["seg", "fqseg"], "plin": ["seg", "fqseg"]}


def split_segment_call(s):
    head = """            streamk_segment(
                streamk_tile_id(t, total_tiles, total_full_tiles, NUM_XCDS, TILE_ORDER_V9),"""
    tail = "            cur = seg_start\n"
    loop = s.index("        while cur > sk_start:\n")
    i, j = s.index(head, loop), s.index(tail, loop)
    call = s[i:j]
    assert call.count("1 - is_owner, n_peers,") == 1
    ind = "".join("    " + ln if ln else ln for ln in call.splitlines(keepends=True))
    new = ("            if is_owner != 0:\n" + ind.replace("1 - is_owner, n_peers,", "0, n_peers,")
           + "            else:\n" + ind.replace("1 - is_owner, n_peers,", "1, 0,"))
    return s[:i] + new + s[j:]


def build(name):
    s = open(os.path.join(HERE, "variants", "base.py")).read()
    patches = []
    for p in name.split("+"):
        patches += [r for r in REQUIRES.get(p, []) if r not in patches] + [p]
    for patch in patches:
        for rep in PATCHES[patch]:
            if callable(rep):
                s = rep(s)
                continue
            old, new, *count = rep
            assert s.count(old) == (count or [1])[0], (patch, s.count(old), old[:80])
            s = s.replace(old, new)
    with open(os.path.join(HERE, "variants", f"{name}.py"), "w") as f:
        f.write(s)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        build(n)
        print("wrote", n)
