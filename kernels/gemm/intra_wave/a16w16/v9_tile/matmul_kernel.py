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
"""v9_beyond_hotloop with the tile size as a parameter.

The kernel is v9's: four quadrant accumulators, two LDS buffers, the same region order and
wait counts. Only the layouts that encoded 256x256x64 are now built on the host from the tile
size, and GROUP_SIZE_M follows v9's footprint rule for the tile.

The tile comes from `matmul(..., tile="256x128")` or, when `tile` is None, from the V9_TILE
environment variable ("256x128" or "128x128x128", read at call time); the default is 256x256x64.
A trailing "r2" ("128x128r2") selects v9_tile_2r, two regions per k-step (v7's N slice) instead
of four quadrants. V9_TILE_GM overrides GROUP_SIZE_M and V9_TILE_MFMA=32 picks the 32x32x16 MFMA.

M and N need not be multiples of the tile (v25's edge handling): when either is ragged the host sets
EDGE_M and EDGE_N, which clamp A rows past M and B columns past N to the tile's last in-bounds one
(every load reads valid memory) and mask the C stores and the bias load to M x N. With both off the
kernels are unchanged. V9_FORCE_EDGE=1 sets them for any shape.

K need not be a multiple of 2 BLOCK_K (EDGE_K, set by the host): K is rounded up to K_PAD, a whole
number of K-step pairs, and the last, partial pair goes first (v9_ragged's handling): the prologue
loads it with the lanes at or past K masked to zero, and the main loop then covers columns
0 .. K_PAD - 2 BLOCK_K unmasked. K must be a multiple of 16.

Tiles with a side of 32 or less ("16x256x128", "1x256x128", "256x8x128") run v9_tile_small: the
small side is the kernel's M, padded to a 16-row MFMA, and W waves ("w1", "w2", "w4"; by default one
per 64 columns) sit side by side along the other. A small BN runs as C^T = B^T A^T. A trailing "p"
(or V9_TILE_PIN=1) pads LDS so that exactly 4 / W workgroups fit in a CU, one wave per SIMD. N (M
when swapped) must be a multiple of the long side and K of 2 BLOCK_K. Comparisons with Triton's own
choices (WORKLOG §14.9): "m4" uses the 4x64 MFMA instead of the 16-row padding, "wW" above BN / 16
replicates the tile over the waves, and "eE" launches with waves_per_eu = E (E = 4 / W pins one wave
per SIMD without the LDS pad).
"""

import math
import os
import re

import torch
import triton
from bias import init_acc
from common import get_pids
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def v9_tile(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K: gl.constexpr,
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,  #
    GRID_MN: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,  #
    ADD_BIAS: gl.constexpr,
    MFMA_32: gl.constexpr,
    EDGE_M: gl.constexpr = False,
    EDGE_N: gl.constexpr = False,
    EDGE_K: gl.constexpr = False,
):
    pid_m, pid_n = get_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    gLoadLayoutA: gl.constexpr = half_tile_load_layout(BLOCK_M // 2, BLOCK_K, False)
    gLoadLayoutB: gl.constexpr = half_tile_load_layout(BLOCK_N // 2, BLOCK_K, True)
    sharedLayoutA: gl.constexpr = half_tile_shared_layout(BLOCK_M // 2, BLOCK_K, False)
    sharedLayoutB: gl.constexpr = half_tile_shared_layout(BLOCK_N // 2, BLOCK_K, True)
    mfmaLayout: gl.constexpr = mfma_layout(MFMA_32)
    gStoreLayoutC: gl.constexpr = store_layout(BLOCK_N // 2)

    nBuffers: gl.constexpr = 2
    smemA_top = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [nBuffers, BLOCK_M // 2, BLOCK_K], sharedLayoutA
    )
    smemA_bot = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [nBuffers, BLOCK_M // 2, BLOCK_K], sharedLayoutA
    )
    smemB_left = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, BLOCK_N // 2], sharedLayoutB
    )
    smemB_right = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, BLOCK_N // 2], sharedLayoutB
    )

    offs_am = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, gLoadLayoutA))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutA))

    offs_bn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gLoadLayoutB))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadLayoutB))

    a_base = a_ptr + pid_m * BLOCK_M * stride_am
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn

    if EDGE_M:
        lim_m = M - 1 - pid_m * BLOCK_M
        a_top_offsets = gl.minimum(offs_am, lim_m)[:, None] * stride_am + offs_ak[None, :] * stride_ak
        a_bot_offsets = gl.minimum(offs_am + BLOCK_M // 2, lim_m)[:, None] * stride_am + offs_ak[None, :] * stride_ak
    else:
        a_top_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
        a_bot_offsets = a_top_offsets + BLOCK_M * stride_am // 2
    if EDGE_N:
        lim_n = N - 1 - pid_n * BLOCK_N
        b_left_offsets = offs_bk[:, None] * stride_bk + gl.minimum(offs_bn, lim_n)[None, :] * stride_bn
        b_right_offsets = offs_bk[:, None] * stride_bk + gl.minimum(offs_bn + BLOCK_N // 2, lim_n)[None, :] * stride_bn
    else:
        b_left_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        b_right_offsets = b_left_offsets + BLOCK_N * stride_bn // 2

    a_top_offsets_next = a_top_offsets + BLOCK_K * stride_ak
    a_bot_offsets_next = a_bot_offsets + BLOCK_K * stride_ak
    b_left_offsets_next = b_left_offsets + BLOCK_K * stride_bk
    b_right_offsets_next = b_right_offsets + BLOCK_K * stride_bk

    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfmaLayout, k_width=8)
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfmaLayout, k_width=8)

    acc_tl, acc_bl, acc_tr, acc_br = init_acc_edge(
        bias_ptr, pid_n, N - pid_n * BLOCK_N, BLOCK_M, BLOCK_N, mfmaLayout, ADD_BIAS, EDGE_N
    )

    # Ragged K: the prologue loads the last, partial pair (masked at K); the main loop then starts
    # at column 0. K-step 1 of that pair always reaches past K, K-step 0 only when K_RES > BLOCK_K.
    K_PAD: gl.constexpr = (K + 2 * BLOCK_K - 1) // (2 * BLOCK_K) * (2 * BLOCK_K) if EDGE_K else K
    K_RES: gl.constexpr = K_PAD - K
    iterMax = gl.cdiv(K_PAD, BLOCK_K)
    a_first = a_base
    b_first = b_base
    mA0, mB0, mA1, mB1 = None, None, None, None
    if EDGE_K:
        a_first = a_base + (K_PAD - 2 * BLOCK_K) * stride_ak
        b_first = b_base + (K_PAD - 2 * BLOCK_K) * stride_bk
        mA1 = k_tail_mask(2 * BLOCK_K - K_RES, 1, BLOCK_M // 2, BLOCK_K, gLoadLayoutA, False)
        mB1 = k_tail_mask(2 * BLOCK_K - K_RES, 1, BLOCK_N // 2, BLOCK_K, gLoadLayoutB, True)
        if K_RES > BLOCK_K:
            mA0 = k_tail_mask(2 * BLOCK_K - K_RES, 0, BLOCK_M // 2, BLOCK_K, gLoadLayoutA, False)
            mB0 = k_tail_mask(2 * BLOCK_K - K_RES, 0, BLOCK_N // 2, BLOCK_K, gLoadLayoutB, True)

    g_idx = 0
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(g_idx), b_first, b_left_offsets, mask=mB0)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_top.index(g_idx), a_first, a_top_offsets, mask=mA0)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_bot.index(g_idx), a_first, a_bot_offsets, mask=mA0)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_right.index(g_idx), b_first, b_right_offsets, mask=mB0)
    gl.amd.cdna4.async_copy.commit_group()

    g_idx = 1
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_left.index(g_idx), b_first, b_left_offsets_next, mask=mB1
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_top.index(g_idx), a_first, a_top_offsets_next, mask=mA1
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_bot.index(g_idx), a_first, a_bot_offsets_next, mask=mA1
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_right.index(g_idx), b_first, b_right_offsets_next, mask=mB1
    )
    gl.amd.cdna4.async_copy.commit_group()

    if not EDGE_K:
        a_base += BLOCK_K * stride_ak * 2
        b_base += BLOCK_K * stride_bk * 2

    gl.amd.cdna4.async_copy.wait_group(6)
    b_left = smemB_left.index(0).load(dotOpLayoutB)
    a_top = smemA_top.index(0).load(dotOpLayoutA)

    gl.assume(iterMax > 3)

    for k in range(0, iterMax - 2, 2):
        # Sub-iteration 0: consume buffer 0, prefetch into buffer 0.
        acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")

        gl.amd.cdna4.async_copy.wait_group(5)
        a_bot = smemA_bot.index(0).load(dotOpLayoutA)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(0), b_base, b_left_offsets)
        gl.amd.cdna4.async_copy.commit_group()

        acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")

        gl.amd.cdna4.async_copy.wait_group(5)
        b_right = smemB_right.index(0).load(dotOpLayoutB)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_top.index(0), a_base, a_top_offsets)
        gl.amd.cdna4.async_copy.commit_group()

        acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")

        gl.amd.cdna4.async_copy.wait_group(5)
        b_left = smemB_left.index(1).load(dotOpLayoutB)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_bot.index(0), a_base, a_bot_offsets)
        gl.amd.cdna4.async_copy.commit_group()

        acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")

        gl.amd.cdna4.async_copy.wait_group(5)
        a_top = smemA_top.index(1).load(dotOpLayoutA)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_right.index(0), b_base, b_right_offsets)
        gl.amd.cdna4.async_copy.commit_group()

        # Sub-iteration 1: consume buffer 1, prefetch into buffer 1 (odd K-step offsets).
        acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")

        gl.amd.cdna4.async_copy.wait_group(5)
        a_bot = smemA_bot.index(1).load(dotOpLayoutA)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemB_left.index(1), b_base, b_left_offsets_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")

        gl.amd.cdna4.async_copy.wait_group(5)
        b_right = smemB_right.index(1).load(dotOpLayoutB)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemA_top.index(1), a_base, a_top_offsets_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")

        gl.amd.cdna4.async_copy.wait_group(5)
        b_left = smemB_left.index(0).load(dotOpLayoutB)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemA_bot.index(1), a_base, a_bot_offsets_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")

        gl.amd.cdna4.async_copy.wait_group(5)
        a_top = smemA_top.index(0).load(dotOpLayoutA)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemB_right.index(1), b_base, b_right_offsets_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        a_base += BLOCK_K * stride_ak * 2
        b_base += BLOCK_K * stride_bk * 2

    offs_cm = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gStoreLayoutC))
    c_base = c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
    c_tl_offsets = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_tr_offsets = c_tl_offsets + BLOCK_N * stride_cn // 2
    c_bl_offsets = c_tl_offsets + BLOCK_M * stride_cm // 2
    c_br_offsets = c_bl_offsets + BLOCK_N * stride_cn // 2
    rows_left = M - pid_m * BLOCK_M
    cols_left = N - pid_n * BLOCK_N

    acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    l_idx = (iterMax - 2) % 2
    a_bot = smemA_bot.index(l_idx).load(dotOpLayoutA)

    acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(4)
    b_right = smemB_right.index(l_idx).load(dotOpLayoutB)

    acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(3)
    g_idx = 1 - l_idx
    b_left = smemB_left.index(g_idx).load(dotOpLayoutB)

    acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(2)
    a_top = smemA_top.index(g_idx).load(dotOpLayoutA)

    acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(1)
    a_bot = smemA_bot.index(g_idx).load(dotOpLayoutA)

    acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(0)
    b_right = smemB_right.index(g_idx).load(dotOpLayoutB)

    c_tl = acc_tl.to(c_ptr.dtype.element_ty)
    c_tl = gl.convert_layout(c_tl, layout=gStoreLayoutC)
    store_edge(c_tl, c_base, c_tl_offsets, gStoreLayoutC, rows_left, cols_left, EDGE_M, EDGE_N)

    acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")

    c_bl = acc_bl.to(c_ptr.dtype.element_ty)
    c_bl = gl.convert_layout(c_bl, layout=gStoreLayoutC)
    store_edge(c_bl, c_base, c_bl_offsets, gStoreLayoutC, rows_left - BLOCK_M // 2, cols_left, EDGE_M, EDGE_N)

    acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")

    c_tr = acc_tr.to(c_ptr.dtype.element_ty)
    c_tr = gl.convert_layout(c_tr, layout=gStoreLayoutC)
    store_edge(c_tr, c_base, c_tr_offsets, gStoreLayoutC, rows_left, cols_left - BLOCK_N // 2, EDGE_M, EDGE_N)

    c_br = acc_br.to(c_ptr.dtype.element_ty)
    c_br = gl.convert_layout(c_br, layout=gStoreLayoutC)
    store_edge(c_br, c_base, c_br_offsets, gStoreLayoutC, rows_left - BLOCK_M // 2, cols_left - BLOCK_N // 2, EDGE_M,
               EDGE_N)


@gluon.jit
def v9_tile_2r(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K: gl.constexpr,
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,  #
    GRID_MN: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,  #
    ADD_BIAS: gl.constexpr,
    MFMA_32: gl.constexpr,
    EDGE_M: gl.constexpr = False,
    EDGE_N: gl.constexpr = False,
    EDGE_K: gl.constexpr = False,
):
    """Two regions per k-step (v7's N slice): the whole A tile against each B half.

    For tiles small enough that four quadrants leave too few MFMAs per region (128x128x64: 8).
    Each region holds half the tile's MFMAs. Groups as in v7: {A, B_left} then {B_right}, so A and
    B_left land three regions after they are issued.
    """
    pid_m, pid_n = get_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    gLoadLayoutA: gl.constexpr = half_tile_load_layout(BLOCK_M, BLOCK_K, False)
    gLoadLayoutB: gl.constexpr = half_tile_load_layout(BLOCK_N // 2, BLOCK_K, True)
    sharedLayoutA: gl.constexpr = half_tile_shared_layout(BLOCK_M, BLOCK_K, False)
    sharedLayoutB: gl.constexpr = half_tile_shared_layout(BLOCK_N // 2, BLOCK_K, True)
    mfmaLayout: gl.constexpr = mfma_layout(MFMA_32)
    gStoreLayoutC: gl.constexpr = store_layout(BLOCK_N // 2)

    nBuffers: gl.constexpr = 2
    smemA = gl.allocate_shared_memory(a_ptr.dtype.element_ty, [nBuffers, BLOCK_M, BLOCK_K], sharedLayoutA)
    smemB_left = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, BLOCK_N // 2], sharedLayoutB
    )
    smemB_right = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, BLOCK_N // 2], sharedLayoutB
    )

    offs_am = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gLoadLayoutA))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutA))
    offs_bn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gLoadLayoutB))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadLayoutB))

    a_base = a_ptr + pid_m * BLOCK_M * stride_am
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn
    if EDGE_M:
        a_offsets = gl.minimum(offs_am, M - 1 - pid_m * BLOCK_M)[:, None] * stride_am + offs_ak[None, :] * stride_ak
    else:
        a_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    if EDGE_N:
        lim_n = N - 1 - pid_n * BLOCK_N
        b_left_offsets = offs_bk[:, None] * stride_bk + gl.minimum(offs_bn, lim_n)[None, :] * stride_bn
        b_right_offsets = offs_bk[:, None] * stride_bk + gl.minimum(offs_bn + BLOCK_N // 2, lim_n)[None, :] * stride_bn
    else:
        b_left_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        b_right_offsets = b_left_offsets + BLOCK_N * stride_bn // 2

    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfmaLayout, k_width=8)
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfmaLayout, k_width=8)

    acc_left, acc_right = init_acc_2r(bias_ptr, pid_n, BLOCK_M, BLOCK_N, mfmaLayout, ADD_BIAS,
                                      N - pid_n * BLOCK_N, EDGE_N)

    K_PAD: gl.constexpr = (K + 2 * BLOCK_K - 1) // (2 * BLOCK_K) * (2 * BLOCK_K) if EDGE_K else K
    K_RES: gl.constexpr = K_PAD - K
    iterMax = gl.cdiv(K_PAD, BLOCK_K)

    if EDGE_K:
        # The last, partial pair first (masked at K); the main loop then starts at column 0.
        for g_idx in gl.static_range(2):
            mA = k_tail_mask(2 * BLOCK_K - K_RES, g_idx, BLOCK_M, BLOCK_K, gLoadLayoutA, False)
            mB = k_tail_mask(2 * BLOCK_K - K_RES, g_idx, BLOCK_N // 2, BLOCK_K, gLoadLayoutB, True)
            a_tail = a_base + (K_PAD - 2 * BLOCK_K + g_idx * BLOCK_K) * stride_ak
            b_tail = b_base + (K_PAD - 2 * BLOCK_K + g_idx * BLOCK_K) * stride_bk
            gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(g_idx), a_tail, a_offsets, mask=mA)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(g_idx), b_tail, b_left_offsets, mask=mB)
            gl.amd.cdna4.async_copy.commit_group()
            gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_right.index(g_idx), b_tail, b_right_offsets, mask=mB)
            gl.amd.cdna4.async_copy.commit_group()
    else:
        for g_idx in gl.static_range(2):
            gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(g_idx), b_base, b_left_offsets)
            gl.amd.cdna4.async_copy.commit_group()
            gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_right.index(g_idx), b_base, b_right_offsets)
            gl.amd.cdna4.async_copy.commit_group()
            a_base += BLOCK_K * stride_ak
            b_base += BLOCK_K * stride_bk

    gl.amd.cdna4.async_copy.wait_group(3)
    a = smemA.index(0).load(dotOpLayoutA)
    b_left = smemB_left.index(0).load(dotOpLayoutB)

    gl.assume(iterMax > 3)

    for k in range(0, iterMax - 2, 2):
        for s in gl.static_range(2):
            # Sub-iteration s consumes buffer s and refills it with k-step k + s + 2.
            acc_left = gl.amd.cdna3.mfma(a, b_left, acc_left, cd_regclass="a")

            gl.amd.cdna4.async_copy.wait_group(2)
            b_right = smemB_right.index(s).load(dotOpLayoutB)

            gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(s), a_base, a_offsets)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(s), b_base, b_left_offsets)
            gl.amd.cdna4.async_copy.commit_group()

            acc_right = gl.amd.cdna3.mfma(a, b_right, acc_right, cd_regclass="a")

            gl.amd.cdna4.async_copy.wait_group(2)
            a = smemA.index(1 - s).load(dotOpLayoutA)
            b_left = smemB_left.index(1 - s).load(dotOpLayoutB)

            gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_right.index(s), b_base, b_right_offsets)
            gl.amd.cdna4.async_copy.commit_group()

            a_base += BLOCK_K * stride_ak
            b_base += BLOCK_K * stride_bk

    offs_cm = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gStoreLayoutC))
    c_base = c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
    c_left_offsets = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_right_offsets = c_left_offsets + BLOCK_N * stride_cn // 2
    rows_left = M - pid_m * BLOCK_M
    cols_left = N - pid_n * BLOCK_N

    # k-step iterMax - 2 (buffer 0)
    acc_left = gl.amd.cdna3.mfma(a, b_left, acc_left, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(0)
    b_right = smemB_right.index(0).load(dotOpLayoutB)
    acc_right = gl.amd.cdna3.mfma(a, b_right, acc_right, cd_regclass="a")
    a = smemA.index(1).load(dotOpLayoutA)
    b_left = smemB_left.index(1).load(dotOpLayoutB)

    # k-step iterMax - 1 (buffer 1)
    acc_left = gl.amd.cdna3.mfma(a, b_left, acc_left, cd_regclass="a")
    b_right = smemB_right.index(1).load(dotOpLayoutB)

    c_left = acc_left.to(c_ptr.dtype.element_ty)
    c_left = gl.convert_layout(c_left, layout=gStoreLayoutC)
    store_edge(c_left, c_base, c_left_offsets, gStoreLayoutC, rows_left, cols_left, EDGE_M, EDGE_N)

    acc_right = gl.amd.cdna3.mfma(a, b_right, acc_right, cd_regclass="a")

    c_right = acc_right.to(c_ptr.dtype.element_ty)
    c_right = gl.convert_layout(c_right, layout=gStoreLayoutC)
    store_edge(c_right, c_base, c_right_offsets, gStoreLayoutC, rows_left, cols_left - BLOCK_N // 2, EDGE_M, EDGE_N)


@gluon.jit
def v9_tile_small(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K: gl.constexpr,
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,  #
    GRID_MN: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,  #
    ADD_BIAS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    EDGE_M: gl.constexpr = False,
    C_T: gl.constexpr = False,
    PAD0: gl.constexpr = 0,
    PAD1: gl.constexpr = 0,
    PAD2: gl.constexpr = 0,
    MFMA4: gl.constexpr = False,
):
    """Tiles with BLOCK_M = 1 .. 32: one region per k-step, the W waves side by side along N.

    The MFMA is 16x16x32, so A is held in LDS as MP = BLOCK_M rounded up to 16 rows; the rows at or
    past BLOCK_M (and past M under EDGE_M) are masked lanes in the loads, which write zeros and
    read no memory, and are masked out of the C stores. Two LDS buffers; k-step j + 2 is issued into
    buffer j % 2 as soon as k-step j has been read out of it. C_T: the host passed C^T = B^T A^T, so
    C is stored contiguous along this kernel's M and the bias runs along M. PAD0..2: unused LDS
    allocations of that many 16-byte rows (powers of two), so that LDS sets how many workgroups
    share a CU. MFMA4: the 4x64 MFMA (mfma_f32_4x4x4bf16_1k, 16 blocks along N, A broadcast to all of
    them), as Triton picks for 4 <= M < 16: the MFMA rows are BLOCK_M rounded up to 4, read from the
    top of the 16-row strip the copies fill.
    """
    pid_m, pid_n = get_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)
    MP: gl.constexpr = (BLOCK_M + 15) // 16 * 16
    MASK_A: gl.constexpr = EDGE_M or MP != BLOCK_M
    MR: gl.constexpr = (BLOCK_M + 3) // 4 * 4 if MFMA4 else MP

    gLoadLayoutA: gl.constexpr = strip_load_layout(MP, BLOCK_K, False, NUM_WARPS)
    gLoadLayoutB: gl.constexpr = strip_load_layout(BLOCK_N, BLOCK_K, True, NUM_WARPS)
    sharedLayoutA: gl.constexpr = strip_shared_layout(BLOCK_K, False, False)
    sharedLayoutB: gl.constexpr = strip_shared_layout(BLOCK_K, True, False)
    # 4x64 reads a 4-row slice of A, which a swizzle over 16 rows cannot be split into: A unswizzled.
    readLayoutA: gl.constexpr = strip_shared_layout(BLOCK_K, False, not MFMA4)
    readLayoutB: gl.constexpr = strip_shared_layout(BLOCK_K, True, True)
    mfmaLayout: gl.constexpr = small_mfma_layout(MFMA4, NUM_WARPS)
    gStoreLayoutC: gl.constexpr = small_store_layout(MR, BLOCK_N, NUM_WARPS, C_T)
    KW: gl.constexpr = 4 if MFMA4 else 8
    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfmaLayout, k_width=KW)
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfmaLayout, k_width=KW)

    smemA = gl.allocate_shared_memory(a_ptr.dtype.element_ty, [2, MP, BLOCK_K], sharedLayoutA)
    smemB = gl.allocate_shared_memory(b_ptr.dtype.element_ty, [2, BLOCK_K, BLOCK_N], sharedLayoutB)
    # The copies write row by row (a warp's lanes must fill 1 KB in order); the MFMA operands are
    # read through a swizzled view of the same bytes, the swizzle folded into the copies' offsets.
    smemA_rd = smemA.reinterpret(a_ptr.dtype.element_ty, [2, MP, BLOCK_K], readLayoutA)
    smemB_rd = smemB.reinterpret(b_ptr.dtype.element_ty, [2, BLOCK_K, BLOCK_N], readLayoutB)
    lds_pads(a_ptr.dtype.element_ty, PAD0, PAD1, PAD2, NUM_WARPS)

    offs_am = gl.arange(0, MP, gl.SliceLayout(1, gLoadLayoutA))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutA))
    offs_bn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, gLoadLayoutB))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadLayoutB))

    rows_left = M - pid_m * BLOCK_M
    a_base = a_ptr + pid_m * BLOCK_M * stride_am
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn
    if MFMA4:
        a_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    else:
        a_offsets = offs_am[:, None] * stride_am + swizzled_k(offs_ak[None, :], offs_am[:, None], BLOCK_K) * stride_ak
    b_offsets = swizzled_k(offs_bk[:, None], offs_bn[None, :], BLOCK_K) * stride_bk + offs_bn[None, :] * stride_bn
    mA = None
    if MASK_A:
        a_rows = offs_am < BLOCK_M
        if EDGE_M:
            a_rows = a_rows & (offs_am < rows_left)
        mA = a_rows[:, None] & (offs_ak < BLOCK_K)[None, :]

    acc = gl.zeros((MR, BLOCK_N), gl.float32, mfmaLayout)
    if ADD_BIAS:
        if C_T:
            offs_bias = gl.arange(0, MR, gl.SliceLayout(1, mfmaLayout))
            bias = gl.amd.cdna3.buffer_load(ptr=bias_ptr + pid_m * BLOCK_M, offsets=offs_bias,
                                            mask=(offs_bias < BLOCK_M) & (offs_bias < rows_left), other=0.0)
            acc = acc + bias.to(gl.float32)[:, None]
        else:
            offs_bias = gl.arange(0, BLOCK_N, gl.SliceLayout(0, mfmaLayout))
            bias = gl.amd.cdna3.buffer_load(ptr=bias_ptr + pid_n * BLOCK_N, offsets=offs_bias)
            acc = acc + bias.to(gl.float32)[None, :]

    for s in gl.static_range(2):
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(s), a_base, a_offsets, mask=mA)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB.index(s), b_base, b_offsets)
        gl.amd.cdna4.async_copy.commit_group()
        a_base += BLOCK_K * stride_ak
        b_base += BLOCK_K * stride_bk

    gl.amd.cdna4.async_copy.wait_group(1)
    a = a_rows_view(smemA_rd.index(0), MR, MP).load(dotOpLayoutA)
    b = smemB_rd.index(0).load(dotOpLayoutB)

    iterMax: gl.constexpr = K // BLOCK_K
    for k in range(0, iterMax - 2, 2):
        for s in gl.static_range(2):
            gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(s), a_base, a_offsets, mask=mA)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB.index(s), b_base, b_offsets)
            gl.amd.cdna4.async_copy.commit_group()

            acc = gl.amd.cdna3.mfma(a, b, acc, cd_regclass="a")

            gl.amd.cdna4.async_copy.wait_group(1)
            a = a_rows_view(smemA_rd.index(1 - s), MR, MP).load(dotOpLayoutA)
            b = smemB_rd.index(1 - s).load(dotOpLayoutB)

            a_base += BLOCK_K * stride_ak
            b_base += BLOCK_K * stride_bk

    acc = gl.amd.cdna3.mfma(a, b, acc, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(0)
    a = a_rows_view(smemA_rd.index(1), MR, MP).load(dotOpLayoutA)
    b = smemB_rd.index(1).load(dotOpLayoutB)
    acc = gl.amd.cdna3.mfma(a, b, acc, cd_regclass="a")

    c = gl.convert_layout(acc.to(c_ptr.dtype.element_ty), layout=gStoreLayoutC)
    offs_cm = gl.arange(0, MR, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, gStoreLayoutC))
    c_base = c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
    c_offsets = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    if MASK_A:
        c_rows = offs_cm < BLOCK_M
        if EDGE_M:
            c_rows = c_rows & (offs_cm < rows_left)
        gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_offsets, stored_value=c,
                                  mask=c_rows[:, None] & (offs_cn < BLOCK_N)[None, :])
    else:
        gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_offsets, stored_value=c)


@gluon.constexpr_function
def small_mfma_layout(mfma4, num_warps):
    """v9_tile_small's accumulator layout: 16x16x32 (transposed, for 8-wide C stores along N), or the
    4x64 instruction, which Triton only builds untransposed because A is the broadcast operand."""
    if mfma4:
        return gl.amd.AMDMFMALayout(version=4, instr_shape=[4, 64, 64], transposed=False,
                                    warps_per_cta=[1, num_warps])
    return gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, num_warps])


@gluon.jit
def a_rows_view(smem, ROWS: gl.constexpr, STRIP: gl.constexpr):
    """The first ROWS rows of an A buffer of STRIP rows (all of it unless the MFMA has fewer rows)."""
    if ROWS == STRIP:
        return smem
    return smem.slice(0, ROWS)


@gluon.jit
def lds_pads(dtype: gl.constexpr, PAD0: gl.constexpr, PAD1: gl.constexpr, PAD2: gl.constexpr,
             NUM_WARPS: gl.constexpr):
    """Allocate PAD0 + PAD1 + PAD2 16-byte rows of LDS (each a power of two or 0), all before
    writing one row of each: the writes keep the allocations, and their order keeps them all
    live at once, so the allocator cannot overlap them."""
    layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    row = gl.zeros([1, 8], dtype, gl.BlockedLayout([1, 8], [64, 1], [NUM_WARPS, 1], [1, 0]))
    if PAD0 > 0:
        p0 = gl.allocate_shared_memory(dtype, [PAD0, 8], layout)
    if PAD1 > 0:
        p1 = gl.allocate_shared_memory(dtype, [PAD1, 8], layout)
    if PAD2 > 0:
        p2 = gl.allocate_shared_memory(dtype, [PAD2, 8], layout)
    if PAD0 > 0:
        p0.slice(0, 1).store(row)
    if PAD1 > 0:
        p1.slice(0, 1).store(row)
    if PAD2 > 0:
        p2.slice(0, 1).store(row)


@gluon.jit
def init_acc_2r(bias_ptr, pid_n, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, MFMA: gl.constexpr,
                ADD_BIAS: gl.constexpr, cols_left=0, EDGE_N: gl.constexpr = False):
    """The two [BLOCK_M, BLOCK_N / 2] accumulators of v9_tile_2r, from bias[N] or zero; with EDGE_N
    the bias is loaded only for the tile's first cols_left columns."""
    acc_l = gl.zeros((BLOCK_M, BLOCK_N // 2), gl.float32, MFMA)
    acc_r = acc_l
    if ADD_BIAS:
        offs_bias = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, MFMA))
        bias_base = bias_ptr + pid_n * BLOCK_N
        if EDGE_N:
            bias_l = gl.amd.cdna3.buffer_load(ptr=bias_base, offsets=offs_bias, mask=offs_bias < cols_left, other=0.0)
            bias_r = gl.amd.cdna3.buffer_load(ptr=bias_base + BLOCK_N // 2, offsets=offs_bias,
                                              mask=offs_bias < cols_left - BLOCK_N // 2, other=0.0)
        else:
            bias_l = gl.amd.cdna3.buffer_load(ptr=bias_base, offsets=offs_bias)
            bias_r = gl.amd.cdna3.buffer_load(ptr=bias_base + BLOCK_N // 2, offsets=offs_bias)
        acc_l = acc_l + bias_l.to(gl.float32)[None, :]
        acc_r = acc_r + bias_r.to(gl.float32)[None, :]
    return acc_l, acc_r


@gluon.jit
def init_acc_edge(bias_ptr, pid_n, cols_left, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, MFMA: gl.constexpr,
                  ADD_BIAS: gl.constexpr, EDGE_N: gl.constexpr):
    """bias.init_acc, with the bias loaded only for the tile's first cols_left columns under
    EDGE_N."""
    if ADD_BIAS and EDGE_N:
        acc_l = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, MFMA)
        offs_bias = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, MFMA))
        bias_base = bias_ptr + pid_n * BLOCK_N
        bias_l = gl.amd.cdna3.buffer_load(ptr=bias_base, offsets=offs_bias, mask=offs_bias < cols_left, other=0.0)
        bias_r = gl.amd.cdna3.buffer_load(ptr=bias_base + BLOCK_N // 2, offsets=offs_bias,
                                          mask=offs_bias < cols_left - BLOCK_N // 2, other=0.0)
        acc_r = acc_l + bias_r.to(gl.float32)[None, :]
        acc_l = acc_l + bias_l.to(gl.float32)[None, :]
        tl, bl, tr, br = acc_l, acc_l, acc_r, acc_r
    else:
        tl, bl, tr, br = init_acc(bias_ptr, pid_n, BLOCK_M, BLOCK_N, MFMA, ADD_BIAS)
    return tl, bl, tr, br


@gluon.jit
def k_tail_mask(k_hi, STEP: gl.constexpr, ROWS: gl.constexpr, BLOCK_K: gl.constexpr, layout: gl.constexpr,
                K_FIRST: gl.constexpr):
    """Load mask of K-step STEP (0 or 1) of a block in the last pair, of whose 2 BLOCK_K K values only
    the first k_hi lie inside the matrix: lanes holding the others load zeros. K_FIRST: B's
    [BLOCK_K, ROWS] block, else A's [ROWS, BLOCK_K]."""
    if K_FIRST:
        k = gl.arange(0, BLOCK_K, gl.SliceLayout(1, layout))[:, None]
        r = gl.arange(0, ROWS, gl.SliceLayout(0, layout))[None, :]
    else:
        k = gl.arange(0, BLOCK_K, gl.SliceLayout(0, layout))[None, :]
        r = gl.arange(0, ROWS, gl.SliceLayout(1, layout))[:, None]
    return (k + STEP * BLOCK_K < k_hi) & (r >= 0)


@gluon.jit
def store_edge(c, c_base, c_offsets, layout: gl.constexpr, rows_left, cols_left, EDGE_M: gl.constexpr,
               EDGE_N: gl.constexpr):
    """buffer_store of a block of C; with EDGE_M / EDGE_N only its first rows_left rows and
    cols_left columns (either may exceed the block)."""
    if EDGE_M or EDGE_N:
        r = gl.arange(0, c.shape[0], gl.SliceLayout(1, layout))[:, None]
        cc = gl.arange(0, c.shape[1], gl.SliceLayout(0, layout))[None, :]
        gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_offsets, stored_value=c,
                                  mask=(r < rows_left) & (cc < cols_left))
    else:
        gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_offsets, stored_value=c)


@gluon.constexpr_function
def _pow2_bases(lo, hi):
    """[lo, 2 lo, ..., hi / 2]: the powers of two in [lo, hi)."""
    out, x = [], lo
    while x < hi:
        out.append(x)
        x *= 2
    return out


@gluon.constexpr_function
def _half_tile_bases(rows, block_k, k_first):
    """Bases of a [rows, block_k] half-tile, K contiguous (k_first: the [block_k, rows] half-tile
    of B, the same bases transposed).

    One buffer_load_to_lds_dwordx4 per wave covers 64 K values (8 per lane over 8 lanes) of 8
    rows, 1 KB; the shared layout keeps that 1 KB contiguous (lanes write consecutive LDS
    addresses) and pads 16 elements after each. Warps take rows 1 and 2, the lanes the top
    three row bits, and the registers the K bits past 64 and the rows in between. At rows = 128
    and block_k = 64 this is v9's layout.
    """
    assert rows >= 32 and block_k >= 64 and rows & (rows - 1) == 0 and block_k & (block_k - 1) == 0
    lane_rows = [rows // 8, rows // 4, rows // 2]
    reg_rows = _pow2_bases(4, rows // 8)
    extra_k = _pow2_bases(64, block_k)

    def t(bs):
        return [[b[1], b[0]] for b in bs] if k_first else bs

    return {
        "shape": [block_k, rows] if k_first else [rows, block_k],
        "reg": t([[0, 1], [0, 2], [0, 4]] + [[0, k] for k in extra_k] + [[r, 0] for r in reg_rows]),
        "lane": t([[0, 8], [0, 16], [0, 32]] + [[r, 0] for r in lane_rows]),
        "warp": t([[1, 0], [2, 0]]),
        "offset": t(
            [[0, k] for k in _pow2_bases(1, 64)]
            + [[r, 0] for r in lane_rows]
            + [[0, k] for k in extra_k]
            + [[1, 0], [2, 0]]
            + [[r, 0] for r in reg_rows]
        ),
    }


@gluon.constexpr_function
def half_tile_load_layout(rows, block_k, k_first):
    b = _half_tile_bases(rows, block_k, k_first)
    return gl.DistributedLinearLayout(
        reg_bases=b["reg"], lane_bases=b["lane"], warp_bases=b["warp"], block_bases=[], shape=b["shape"]
    )


@gluon.constexpr_function
def half_tile_shared_layout(rows, block_k, k_first):
    b = _half_tile_bases(rows, block_k, k_first)
    return gl.PaddedSharedLayout([[512, 16]], b["offset"], [], b["shape"])


@gluon.constexpr_function
def mfma_layout(mfma_32):
    instr = [32, 32, 16] if mfma_32 else [16, 16, 32]
    return gl.amd.AMDMFMALayout(version=4, instr_shape=instr, transposed=True, warps_per_cta=[2, 2])


@gluon.constexpr_function
def store_layout(cols):
    """Blocked C-store layout of a [*, cols] quadrant: 8 values per lane along N."""
    tpw_n = min(64, cols // 8)
    return gl.BlockedLayout([1, 8], [64 // tpw_n, tpw_n], [4, 1], [1, 0])


@gluon.constexpr_function
def strip_load_layout(rows, block_k, k_first, num_warps):
    """Global-load layout of a [rows, block_k] strip (k_first: B's [block_k, rows]), K contiguous:
    8 K values per lane, min(64, block_k / 8) lanes along K, the rest of the lanes and then the
    warps along the rows. A warp's buffer_load_to_lds_dwordx4 then covers whole rows, 1 KB."""
    tk = min(64, block_k // 8)
    assert rows >= num_warps * (64 // tk), f"{rows} rows x {block_k} K is less than one load per warp"
    if k_first:
        return gl.BlockedLayout([8, 1], [tk, 64 // tk], [1, num_warps], [0, 1])
    return gl.BlockedLayout([1, 8], [64 // tk, tk], [num_warps, 1], [1, 0])


@gluon.constexpr_function
def strip_swizzle(block_k):
    """(per_phase, max_phase) of the strips' XOR swizzle: 16-byte chunk c of row r sits at
    c ^ ((r // per_phase) % max_phase), so the 16 rows one ds_read_b128 reads (16x32 MFMA operand,
    8 K per lane) land in 16 different bank groups; rows of 128 B (block_k = 64) pair up."""
    return max(1, 128 // block_k), min(16, block_k // 8)


@gluon.constexpr_function
def strip_shared_layout(block_k, k_first, swizzled):
    """The strips' LDS layout: row by row, 16-byte chunks XOR-swizzled as strip_swizzle (the view
    the MFMA operands are read through), or not (the view the async copies write through)."""
    order = [0, 1] if k_first else [1, 0]
    if not swizzled:
        return gl.SwizzledSharedLayout(1, 1, 1, order=order)
    per_phase, max_phase = strip_swizzle(block_k)
    return gl.SwizzledSharedLayout(vec=8, per_phase=per_phase, max_phase=max_phase, order=order)


@gluon.jit
def swizzled_k(offs_k, offs_r, BLOCK_K: gl.constexpr):
    """The K index each lane loads so that a plain row-by-row copy lands the strip swizzled: the
    copy writes chunk c of row r where the swizzled view reads chunk c ^ phase(r)."""
    PER_PHASE: gl.constexpr = strip_swizzle(BLOCK_K)[0]
    MAX_PHASE: gl.constexpr = strip_swizzle(BLOCK_K)[1]
    return ((offs_k // 8) ^ ((offs_r // PER_PHASE) % MAX_PHASE)) * 8 + offs_k % 8


@gluon.constexpr_function
def small_store_layout(rows, cols, num_warps, c_t):
    """C-store layout of v9_tile_small's [rows, cols] tile: 8 values per lane along N, or along M
    when C is stored transposed (c_t), the warps side by side along N."""
    if c_t:
        sm = min(8, rows)
        tm = rows // sm
        return gl.BlockedLayout([sm, 1], [tm, 64 // tm], [1, num_warps], [0, 1])
    tn = max(1, min(64, cols // (8 * num_warps)))
    return gl.BlockedLayout([1, 8], [64 // tn, tn], [1, num_warps], [1, 0])


def group_size_m(block_m, block_n, grid_m, grid_mn, num_xcds=8, num_cus=256, tiles_per_cu=1):
    """v9's rule with tile sizes: minimise an XCD's A + B footprint GM * BM + ceil(P / GM) * BN,
    P = the programs an XCD holds at once (tiles_per_cu of them per CU). Ties go to the smaller GM
    (v9's 4 at P = 32)."""
    p = max(1, min(num_cus * tiles_per_cu, grid_mn) // num_xcds)
    best = None
    for gm in (1, 2, 4, 8, 16, 32, 64, 128):
        if gm > max(1, grid_m):
            break
        cost = gm * block_m + math.ceil(p / gm) * block_n
        if best is None or cost < best[0]:
            best = (cost, gm)
    return best[1]


TILE_RE = re.compile(r"^(\d+)x(\d+)(?:x(\d+))?(r2)?(?:w([124]))?(m4)?(?:e(\d))?(p)?$")
LDS_PER_CU = 160 * 1024
LAST_LAUNCH = {}  # the last launch's kernel name and compiled metadata (shared bytes), for the checks


def parse_spec(spec):
    """"BMxBN[xBK][r2][wW][m4][eE][p]" -> (BM, BN, BK, regions, W or None, pin). r2 selects
    v9_tile_2r; wW sets v9_tile_small's waves per workgroup (more than BN / 16 of them replicate the
    tile, as Triton's warps do on a 16x16 dot); p pins it to one wave per SIMD. m4 and eE: see
    spec_options."""
    m = TILE_RE.match(spec.lower())
    assert m, f"tile spec {spec!r}: expected BMxBN[xBK][r2][wW][m4][eE][p]"
    bm, bn, bk, r2, w, _, _, pin = m.groups()
    return int(bm), int(bn), int(bk or 64), 2 if r2 else 4, int(w) if w else None, bool(pin)


def spec_options(spec):
    """v9_tile_small's options in a spec: m4, the 4x64 MFMA instead of the 16-row padded 16x16x32;
    eE, launch with waves_per_eu = E (LLVM's register budget for E waves per SIMD)."""
    m = TILE_RE.match(spec.lower())
    return {"mfma4": bool(m.group(6)), "waves_per_eu": int(m.group(7) or 0)}


def parse_tile(spec):
    """"BMxBN[xBK][r2][wW][p]" -> (BM, BN, BK, regions)."""
    return parse_spec(spec)[:4]


def tile_config(tile=None):
    """(BM, BN, BK, regions) from the argument, else V9_TILE, else 256x256x64."""
    if tile is None:
        tile = os.environ.get("V9_TILE", "256x256x64")
    if isinstance(tile, str):
        return parse_tile(tile)
    return (*tile, 4) if len(tile) == 3 else tuple(tile)


def is_small(bm, bn, regions=4):
    """Tiles with a side of 32 or less run v9_tile_small (r2 keeps v9_tile_2r)."""
    return regions != 2 and min(bm, bn) <= 32


def small_config(bm, bn, bk, w=None):
    """v9_tile_small's (BM, BN, BK, W, swap) for a BM x BN tile: the small side becomes the kernel's
    M (swap: the host runs C^T = B^T A^T), and W defaults to one wave per 64 columns, 1 to 4."""
    swap = bn < bm
    km, kn = (bn, bm) if swap else (bm, bn)
    return km, kn, bk, w or max(1, min(4, kn // 64)), swap


def small_lds_bytes(km, kn, bk):
    return 2 * (-(-km // 16) * 16 + kn) * bk * 2


def small_pin_pads(km, kn, bk, w):
    """LDS pads (PAD0..2, 16-byte rows, powers of two) so that exactly 4 / W workgroups fit in a CU,
    one wave per SIMD: the total lands near the middle of (LDS / (t + 1), LDS / t], t = 4 / W."""
    t = 4 // w
    base = small_lds_bytes(km, kn, bk)
    assert base <= LDS_PER_CU // t, f"{km}x{kn}x{bk} uses {base} B of LDS: fewer than {t} fit in a CU"
    rows = max(0, (LDS_PER_CU // (t + 1) + LDS_PER_CU // t) // 2 - base) // 16
    pads = []
    for _ in range(3):
        p = 1 << (rows.bit_length() - 1) if rows else 0
        pads.append(p)
        rows -= p
    total = base + 16 * sum(pads)
    assert LDS_PER_CU // (t + 1) < total <= LDS_PER_CU // t, f"pads {pads} give {total} B"
    return dict(zip(("PAD0", "PAD1", "PAD2"), pads))


def matmul_small(a, b, c, bias, bm, bn, bk, w, pin, mfma4=False, waves_per_eu=0, schedule_hint="mfma-schedule"):
    M, K = a.shape
    N = b.shape[1]
    km, kn, bk, w, swap = small_config(bm, bn, bk, w)
    if swap:
        a, b, c = b.T, a.T, c.T
        M, N = N, M
    assert a.stride(1) == 1 and b.stride(0) == 1, "A and B must be K-contiguous"
    assert N % kn == 0, f"{'M' if swap else 'N'}={N} must be a multiple of {kn}"
    assert K % (2 * bk) == 0, f"K={K} must be a multiple of {2 * bk}"
    if mfma4:
        assert km <= 8 and kn % (64 * w) == 0, f"4x64 MFMA: BM <= 8 and 64 columns per wave ({km}x{kn}, {w} waves)"
    else:
        # W x 16 > BN replicates: the waves hold copies of the tile (Triton's warps on a 16x16 dot).
        assert kn % (16 * w) == 0 or (16 * w) % kn == 0, f"{kn} columns do not split over {w} waves"
    grid_m, grid_n = triton.cdiv(M, km), N // kn
    GRID_MN = grid_m * grid_n
    gm_env = os.environ.get("V9_TILE_GM")
    GROUP_SIZE_M = int(gm_env) if gm_env else group_size_m(km, kn, grid_m, GRID_MN, 8, tiles_per_cu=4 // w)
    pin = pin or os.environ.get("V9_TILE_PIN") == "1"
    k = v9_tile_small[(GRID_MN, 1)](
        a, b, c, bias, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=km, BLOCK_N=kn, BLOCK_K=bk, GRID_MN=GRID_MN, NUM_XCDS=8, GROUP_SIZE_M=GROUP_SIZE_M,
        ADD_BIAS=bias is not None, NUM_WARPS=w,
        EDGE_M=os.environ.get("V9_FORCE_EDGE") == "1" or M % km != 0, C_T=swap,
        MFMA4=mfma4,
        num_warps=w,
        waves_per_eu=waves_per_eu,
        schedule_hint=schedule_hint,
        **(small_pin_pads(km, kn, bk, w) if pin else {}),
    )
    LAST_LAUNCH.update(kernel="v9_tile_small", shared=k.metadata.shared, tile=(km, kn, bk, w), swap=swap,
                       pin=pin, mfma4=mfma4, waves_per_eu=waves_per_eu)


def matmul(a, b, c=None, bias=None, tile=None, schedule_hint="mfma-schedule"):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.stride(1) == 1, "Matrix A must be row-major (its row stride may be padded)"
    M, K = a.shape
    K, N = b.shape
    assert bias is None or (bias.shape == (N,) and bias.is_contiguous()), "bias must be contiguous (N,)"
    if c is None:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    opts = {}
    if isinstance(tile, str) or (tile is None and "V9_TILE" in os.environ):
        BLOCK_M, BLOCK_N, BLOCK_K, regions, w, pin = parse_spec(tile or os.environ["V9_TILE"])
        opts = spec_options(tile or os.environ["V9_TILE"])
    else:
        (BLOCK_M, BLOCK_N, BLOCK_K, regions), w, pin = tile_config(tile), None, False
    if is_small(BLOCK_M, BLOCK_N, regions):
        matmul_small(a, b, c, bias, BLOCK_M, BLOCK_N, BLOCK_K, w, pin, schedule_hint=schedule_hint, **opts)
        return c
    # The pipeline consumes K-steps in pairs, at least two pairs; other K run over K rounded up to a
    # pair (EDGE_K). K must keep 16-byte rows: Triton only knows a stride is aligned when it is a
    # multiple of 16.
    assert K % 16 == 0, f"K={K} must be a multiple of 16 (pad A and B)"
    k_pad = triton.cdiv(K, 2 * BLOCK_K) * 2 * BLOCK_K
    assert k_pad >= 4 * BLOCK_K, f"K={K} needs at least {4 * BLOCK_K} after rounding up to {2 * BLOCK_K}"
    EDGE_K = K != k_pad
    num_warps = 4
    grid_m, grid_n = triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N)
    GRID_MN = grid_m * grid_n
    NUM_XCDS = 8
    gm_env = os.environ.get("V9_TILE_GM")
    GROUP_SIZE_M = int(gm_env) if gm_env else group_size_m(BLOCK_M, BLOCK_N, grid_m, GRID_MN, NUM_XCDS)

    # Ragged M / N: clamped loads and masked stores, on both dimensions if either is ragged (as v25).
    EDGE_M = EDGE_N = os.environ.get("V9_FORCE_EDGE") == "1" or M % BLOCK_M != 0 or N % BLOCK_N != 0
    kernel = v9_tile_2r if regions == 2 else v9_tile
    kernel[(GRID_MN, 1)](
        a,
        b,
        c,  #
        bias,
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GRID_MN=GRID_MN,
        NUM_XCDS=NUM_XCDS,
        GROUP_SIZE_M=GROUP_SIZE_M,
        ADD_BIAS=bias is not None,
        MFMA_32=os.environ.get("V9_TILE_MFMA") == "32",
        EDGE_M=EDGE_M,
        EDGE_N=EDGE_N,
        EDGE_K=EDGE_K,
        schedule_hint=schedule_hint,
        num_warps=num_warps,
    )
    return c
