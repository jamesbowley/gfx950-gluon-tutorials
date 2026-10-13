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

    python experiments/ragged_spills/make_variants.py kbcast kopaque kbcast+kopaque

A name joins patches with '+'. Each patch is a list of (old, new) replacements that must match once.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

PATCHES = {
    # The persistent epilogue's C row offsets depend on the tile (clamped to its last row, which only
    # changes rows the store masks drop, in both the top and the bottom quadrants), so the compiler
    # cannot hoist stride_cm * row above the tile loop and hold it, spilled, across the loop.
    "crow": [(
        """    # share one offset tensor rather than holding four in VGPRs through the epilogue.
    c_tl_base = c_ptr + pid_m.to(gl.int64) * BLOCK_M * stride_cm
    col = pid_n * BLOCK_N + offs_cn
    c_offsets = stride_cm * offs_cm[:, None] + stride_cn * col[None, :]
    c_tr_base = c_tl_base + BLOCK_N * stride_cn // 2""",
        """    # share one offset tensor rather than holding four in VGPRs through the epilogue.
    c_tl_base = c_ptr + pid_m.to(gl.int64) * BLOCK_M * stride_cm
    col = pid_n * BLOCK_N + offs_cn
    if EDGE_M:
        c_offsets = stride_cm * gl.minimum(offs_cm, rows_left - 1)[:, None] + stride_cn * col[None, :]
    else:
        c_offsets = stride_cm * offs_cm[:, None] + stride_cn * col[None, :]
    c_tr_base = c_tl_base + BLOCK_N * stride_cn // 2""")],
    # K-tail masks over K only ([1, BLOCK_K] for A, [BLOCK_K, 1] for B); the load broadcasts them.
    "kbcast": [(
        """    if K_FIRST:
        k = gl.arange(0, BLOCK_K, gl.SliceLayout(1, layout))[:, None]
        r = gl.arange(0, ROWS, gl.SliceLayout(0, layout))[None, :]
    else:
        k = gl.arange(0, BLOCK_K, gl.SliceLayout(0, layout))[None, :]
        r = gl.arange(0, ROWS, gl.SliceLayout(1, layout))[:, None]
    return (k + STEP * BLOCK_K < k_hi) & (r >= 0)""",
        """    if K_FIRST:
        k = gl.arange(0, BLOCK_K, gl.SliceLayout(1, layout))[:, None]
    else:
        k = gl.arange(0, BLOCK_K, gl.SliceLayout(0, layout))[None, :]
    return k + STEP * BLOCK_K < k_hi""")],
    # The persistent epilogue's tail masks depend on the next tile, so they are not hoisted out of
    # the tile loop and held across it.
    "kopaque": [(
        """        pa0, pb0, pa1, pb1 = tail_pair_masks(2 * BLOCK_K - K_RES, K_RES, BLOCK_M, BLOCK_N, BLOCK_K,
                                             gLoadLayoutA, gLoadLayoutB)""",
        """        k_hi = gl.where(next_m >= 0, 2 * BLOCK_K - K_RES, 0)
        pa0, pb0, pa1, pb1 = tail_pair_masks(k_hi, K_RES, BLOCK_M, BLOCK_N, BLOCK_K,
                                             gLoadLayoutA, gLoadLayoutB)""")],
}


def build(name):
    s = open(os.path.join(HERE, "variants", "base.py")).read()
    for patch in name.split("+"):
        for old, new in PATCHES[patch]:
            assert s.count(old) == 1, (patch, s.count(old), old[:80])
            s = s.replace(old, new)
    with open(os.path.join(HERE, "variants", f"{name}.py"), "w") as f:
        f.write(s)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        build(n)
        print("wrote", n)
