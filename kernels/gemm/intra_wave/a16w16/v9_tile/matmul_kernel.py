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
wait counts. Only the layouts that encoded 256x256x64 are now derived from the tile size, as aiter's
gfx950 Gluon GEMM derives them: the shared layouts from Triton's compute_efficient_padded_shared_layout
for the MFMA operands, the global-load layouts cut from those (tile_shared_layout, tile_load_layout).
At 256x256x64 they are v9's. GROUP_SIZE_M follows v9's footprint rule for the tile.

The tile comes from `matmul(..., tile="256x128")` or, when `tile` is None, from the V9_TILE
environment variable ("256x128" or "128x128x128", read at call time); the default is 256x256x64.
A trailing "r2" ("128x128r2") selects v9_tile_2r, two regions per k-step (v7's N slice) instead
of four quadrants. V9_TILE_GM overrides GROUP_SIZE_M and V9_TILE_MFMA=32 picks the 32x32x16 MFMA.

M and N need not be multiples of the tile (v25's edge handling): when either is ragged the host sets
EDGE_M and EDGE_N, which clamp A rows past M and B columns past N to the tile's last in-bounds one
(every load reads valid memory) and mask the C stores and the bias load to M x N. With both off the
kernels are unchanged. V9_FORCE_EDGE=1 sets them for any shape.

BLOCK_M may also be a multiple of 16 between 128 and 256 ("192x128", "240x128"; row_bands): Gluon
tensors have power-of-two sides, so v9_tile runs the quadrants of a 128-row tile and row bands of
64, 32 and 16 rows under them, every band split over all four waves (WORKLOG §17). BLOCK_N may be
one instead ("256x192", "128x240"; col_bands): column bands right of a 128-column tile (WORKLOG
§18), not both. Power-of-two tiles compile to the same code as before.

K need not be a multiple of 2 BLOCK_K (EDGE_K, set by the host): K is rounded up to K_PAD, a whole
number of K-step pairs, and the last, partial pair goes first (v9_ragged's handling): the prologue
loads it with the lanes at or past K masked to zero, and the main loop then covers columns
0 .. K_PAD - 2 BLOCK_K unmasked. K must be a multiple of 16.

Tiles with a side of 32 or less ("16x256x128", "1x256x128", "256x8x128") run v9_tile_small: the
small side is the kernel's M, padded to a 16-row MFMA, and W waves ("w1", "w2", "w4"; by default one
per 64 columns) sit side by side along the other. A small BN runs as C^T = B^T A^T. A trailing "p"
(or V9_TILE_PIN=1) pads LDS so that exactly 4 / W workgroups fit in a CU, one wave per SIMD. N (M
when swapped) need not be a multiple of the long side, nor K of 2 BLOCK_K (EDGE_N and EDGE_K, as
v9_tile's); K must be a multiple of 16. Comparisons with Triton's own
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
from triton.experimental.gluon.language.amd import cdna4


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
    MAIN_M: gl.constexpr = 0,
    BAND64: gl.constexpr = False,
    BAND32: gl.constexpr = False,
    BAND16: gl.constexpr = False,
    MAIN_N: gl.constexpr = 0,
    NBAND64: gl.constexpr = False,
    NBAND32: gl.constexpr = False,
    NBAND16: gl.constexpr = False,
):
    """Four quadrants of (MAIN_M / 2) x (MAIN_N / 2), MAIN_M = BLOCK_M and MAIN_N = BLOCK_N unless set.

    BLOCK_M = 128 + 64 BAND64 + 32 BAND32 + 16 BAND16 (MAIN_M = 128) adds row bands under the
    quadrants, each split into a left and right half against the same B: a 64-row band (A_mid) and
    a 32-row band (A_s32) with the quadrants' 2x2 waves and B registers, and a 16-row band (A_s16)
    with the waves side by side (v9_tile_small's strip layouts) and its own read of B. A_mid's copy
    shares A_top's commit group, A_s32's and A_s16's A_bot's, so the wait counts are v9's.

    BLOCK_N = 128 + 64 NBAND64 + 32 NBAND32 + 16 NBAND16 (MAIN_N = 128) mirrors them: column bands to
    the right of the quadrants, each split into a top and bottom half against A_top and A_bot. B_mid
    (64 columns) and B_s32 (32) take the 2x2 waves and the quadrants' A registers; B_s16 (16) stacks
    the waves along M and reads A again. B_mid's copy shares B_left's commit group, B_s32's and
    B_s16's B_right's. Row and column bands do not combine.
    """
    pid_m, pid_n = get_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)
    QUAD_M: gl.constexpr = MAIN_M if MAIN_M > 0 else BLOCK_M
    HALF_M: gl.constexpr = QUAD_M // 2
    R_MID: gl.constexpr = QUAD_M
    R_32: gl.constexpr = R_MID + HALF_M if BAND64 else R_MID
    R_16: gl.constexpr = R_32 + 32 if BAND32 else R_32
    QUAD_N: gl.constexpr = MAIN_N if MAIN_N > 0 else BLOCK_N
    HALF_N: gl.constexpr = QUAD_N // 2
    C_MID: gl.constexpr = QUAD_N
    C_32: gl.constexpr = C_MID + HALF_N if NBAND64 else C_MID
    C_16: gl.constexpr = C_32 + 32 if NBAND32 else C_32

    sharedLayoutA: gl.constexpr = tile_shared_layout(HALF_M, BLOCK_K, False, MFMA_32, a_ptr.dtype.element_ty)
    sharedLayoutB: gl.constexpr = tile_shared_layout(HALF_N, BLOCK_K, True, MFMA_32, b_ptr.dtype.element_ty)
    gLoadLayoutA: gl.constexpr = tile_load_layout(sharedLayoutA)
    gLoadLayoutB: gl.constexpr = tile_load_layout(sharedLayoutB)
    mfmaLayout: gl.constexpr = mfma_layout(MFMA_32)
    gStoreLayoutC: gl.constexpr = store_layout(HALF_N)

    nBuffers: gl.constexpr = 2
    smemA_top = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [nBuffers, HALF_M, BLOCK_K], sharedLayoutA
    )
    smemA_bot = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [nBuffers, HALF_M, BLOCK_K], sharedLayoutA
    )
    smemB_left = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, HALF_N], sharedLayoutB
    )
    smemB_right = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, HALF_N], sharedLayoutB
    )

    offs_am = gl.arange(0, HALF_M, gl.SliceLayout(1, gLoadLayoutA))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutA))

    offs_bn = gl.arange(0, HALF_N, gl.SliceLayout(0, gLoadLayoutB))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadLayoutB))

    a_base = a_ptr + pid_m.to(gl.int64) * BLOCK_M * stride_am
    b_base = b_ptr + pid_n.to(gl.int64) * BLOCK_N * stride_bn

    if EDGE_M:
        lim_m = M - 1 - pid_m * BLOCK_M
        a_top_offsets = gl.minimum(offs_am, lim_m)[:, None] * stride_am + offs_ak[None, :] * stride_ak
        a_bot_offsets = gl.minimum(offs_am + HALF_M, lim_m)[:, None] * stride_am + offs_ak[None, :] * stride_ak
    else:
        a_top_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
        a_bot_offsets = a_top_offsets + QUAD_M * stride_am // 2
    if EDGE_N:
        lim_n = N - 1 - pid_n * BLOCK_N
        b_left_offsets = offs_bk[:, None] * stride_bk + gl.minimum(offs_bn, lim_n)[None, :] * stride_bn
        b_right_offsets = offs_bk[:, None] * stride_bk + gl.minimum(offs_bn + HALF_N, lim_n)[None, :] * stride_bn
    else:
        b_left_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        b_right_offsets = b_left_offsets + QUAD_N * stride_bn // 2

    a_top_offsets_next = a_top_offsets + BLOCK_K * stride_ak
    a_bot_offsets_next = a_bot_offsets + BLOCK_K * stride_ak
    b_left_offsets_next = b_left_offsets + BLOCK_K * stride_bk
    b_right_offsets_next = b_right_offsets + BLOCK_K * stride_bk

    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfmaLayout, k_width=8)
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfmaLayout, k_width=8)

    if MAIN_N > 0:
        acc_tl = init_band_n(bias_ptr, pid_n, 0, HALF_N, HALF_M, BLOCK_N, mfmaLayout, ADD_BIAS, N - pid_n * BLOCK_N,
                             EDGE_N)
        acc_tr = init_band_n(bias_ptr, pid_n, HALF_N, HALF_N, HALF_M, BLOCK_N, mfmaLayout, ADD_BIAS,
                             N - pid_n * BLOCK_N, EDGE_N)
        acc_bl = acc_tl
        acc_br = acc_tr
    else:
        acc_tl, acc_bl, acc_tr, acc_br = init_acc_edge(
            bias_ptr, pid_n, N - pid_n * BLOCK_N, QUAD_M, BLOCK_N, mfmaLayout, ADD_BIAS, EDGE_N
        )

    # Row bands under the quadrants (rows R_MID, R_32, R_16 of the tile); EDGE_M clamps their rows
    # to the tile's last in-bounds one, as A_top's and A_bot's.
    if BAND64:
        smemA_mid = gl.allocate_shared_memory(a_ptr.dtype.element_ty, [nBuffers, HALF_M, BLOCK_K], sharedLayoutA)
        if EDGE_M:
            a_mid_offsets = gl.minimum(offs_am + R_MID, lim_m)[:, None] * stride_am + offs_ak[None, :] * stride_ak
        else:
            a_mid_offsets = a_top_offsets + R_MID * stride_am
        a_mid_offsets_next = a_mid_offsets + BLOCK_K * stride_ak
        acc_ml = acc_tl
        acc_mr = acc_tr
    if BAND32:
        sharedLayoutA32: gl.constexpr = tile_shared_layout(32, BLOCK_K, False, MFMA_32, a_ptr.dtype.element_ty)
        gLoadLayoutA32: gl.constexpr = tile_load_layout(sharedLayoutA32)
        smemA_s32 = gl.allocate_shared_memory(a_ptr.dtype.element_ty, [nBuffers, 32, BLOCK_K], sharedLayoutA32)
        rows32 = gl.arange(0, 32, gl.SliceLayout(1, gLoadLayoutA32)) + R_32
        if EDGE_M:
            rows32 = gl.minimum(rows32, lim_m)
        a_s32_offsets = (rows32[:, None] * stride_am
                         + gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutA32))[None, :] * stride_ak)
        a_s32_offsets_next = a_s32_offsets + BLOCK_K * stride_ak
        acc_s32l, _, acc_s32r, _ = init_acc_edge(
            bias_ptr, pid_n, N - pid_n * BLOCK_N, 64, BLOCK_N, mfmaLayout, ADD_BIAS, EDGE_N
        )
    if BAND16:
        # 16 rows: the waves sit side by side along N (16 columns of each B half each), so the band
        # reads B again in its own operand layout. A wave's copy (16 B per lane) covers
        # 512 / BLOCK_K rows: at BLOCK_K = 64 two waves fill the band and the other two are predicated off.
        mfma16: gl.constexpr = small_mfma_layout(False, 4)
        dotOpLayoutA16: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma16, k_width=8)
        dotOpLayoutB16: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma16, k_width=8)
        gLoadLayoutA16: gl.constexpr = strip_load_layout(16, BLOCK_K, False, 4, replicate=True)
        smemA_s16 = gl.allocate_shared_memory(a_ptr.dtype.element_ty, [nBuffers, 16, BLOCK_K],
                                              strip_shared_layout(BLOCK_K, False, False))
        smemA_s16_rd = smemA_s16.reinterpret(a_ptr.dtype.element_ty, [nBuffers, 16, BLOCK_K],
                                             strip_shared_layout(BLOCK_K, False, True))
        offs_a16r = gl.arange(0, 16, gl.SliceLayout(1, gLoadLayoutA16))
        a16_k = swizzled_k(gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutA16))[None, :], offs_a16r[:, None],
                           BLOCK_K)
        rows16 = offs_a16r + R_16
        if EDGE_M:
            rows16 = gl.minimum(rows16, lim_m)
        a_s16_offsets = rows16[:, None] * stride_am + a16_k * stride_ak
        a_s16_offsets_next = a_s16_offsets + BLOCK_K * stride_ak
        acc_s16l, _, acc_s16r, _ = init_acc_edge(
            bias_ptr, pid_n, N - pid_n * BLOCK_N, 32, BLOCK_N, mfma16, ADD_BIAS, EDGE_N
        )

    # Column bands right of the quadrants (columns C_MID, C_32, C_16 of the tile); EDGE_N clamps
    # their columns to the tile's last in-bounds one, as B_left's and B_right's.
    if NBAND64:
        smemB_mid = gl.allocate_shared_memory(b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, HALF_N], sharedLayoutB)
        if EDGE_N:
            b_mid_offsets = offs_bk[:, None] * stride_bk + gl.minimum(offs_bn + C_MID, lim_n)[None, :] * stride_bn
        else:
            b_mid_offsets = b_left_offsets + C_MID * stride_bn
        b_mid_offsets_next = b_mid_offsets + BLOCK_K * stride_bk
        acc_n64t = init_band_n(bias_ptr, pid_n, C_MID, HALF_N, HALF_M, BLOCK_N, mfmaLayout, ADD_BIAS,
                               N - pid_n * BLOCK_N, EDGE_N)
        acc_n64b = acc_n64t
    if NBAND32:
        sharedLayoutB32: gl.constexpr = tile_shared_layout(32, BLOCK_K, True, MFMA_32, b_ptr.dtype.element_ty)
        gLoadLayoutB32: gl.constexpr = tile_load_layout(sharedLayoutB32)
        smemB_s32 = gl.allocate_shared_memory(b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, 32], sharedLayoutB32)
        cols32 = gl.arange(0, 32, gl.SliceLayout(0, gLoadLayoutB32)) + C_32
        if EDGE_N:
            cols32 = gl.minimum(cols32, lim_n)
        b_s32_offsets = (gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadLayoutB32))[:, None] * stride_bk
                         + cols32[None, :] * stride_bn)
        b_s32_offsets_next = b_s32_offsets + BLOCK_K * stride_bk
        acc_n32t = init_band_n(bias_ptr, pid_n, C_32, 32, HALF_M, BLOCK_N, mfmaLayout, ADD_BIAS, N - pid_n * BLOCK_N,
                               EDGE_N)
        acc_n32b = acc_n32t
    if NBAND16:
        # 16 columns: the waves stack along M (HALF_M / 4 rows of each A half each), so the band reads
        # A again in its own operand layout. B_s16 is v9_tile_small's swizzled strip, copied by
        # 512 / BLOCK_K columns per wave (the waves past 16 columns predicated off).
        mfma16n: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                                     warps_per_cta=[4, 1])
        dotOpLayoutA16n: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma16n, k_width=8)
        dotOpLayoutB16n: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma16n, k_width=8)
        gLoadLayoutB16: gl.constexpr = strip_load_layout(16, BLOCK_K, True, 4, replicate=True)
        smemB_s16 = gl.allocate_shared_memory(b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, 16],
                                              strip_shared_layout(BLOCK_K, True, False))
        smemB_s16_rd = smemB_s16.reinterpret(b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, 16],
                                             strip_shared_layout(BLOCK_K, True, True))
        offs_b16c = gl.arange(0, 16, gl.SliceLayout(0, gLoadLayoutB16))
        b16_k = swizzled_k(gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadLayoutB16))[:, None], offs_b16c[None, :],
                           BLOCK_K)
        cols16 = offs_b16c + C_16
        if EDGE_N:
            cols16 = gl.minimum(cols16, lim_n)
        b_s16_offsets = b16_k * stride_bk + cols16[None, :] * stride_bn
        b_s16_offsets_next = b_s16_offsets + BLOCK_K * stride_bk
        acc_n16t = init_band_n(bias_ptr, pid_n, C_16, 16, HALF_M, BLOCK_N, mfma16n, ADD_BIAS, N - pid_n * BLOCK_N,
                               EDGE_N)
        acc_n16b = acc_n16t

    # Ragged K: the prologue loads the last, partial pair (masked at K); the main loop then starts
    # at column 0. K-step 1 of that pair always reaches past K, K-step 0 only when K_RES > BLOCK_K.
    K_PAD: gl.constexpr = (K + 2 * BLOCK_K - 1) // (2 * BLOCK_K) * (2 * BLOCK_K) if EDGE_K else K
    K_RES: gl.constexpr = K_PAD - K
    iterMax = gl.cdiv(K_PAD, BLOCK_K)
    a_first = a_base
    b_first = b_base
    mA0, mB0, mA1, mB1 = None, None, None, None
    m32_0, m32_1, m16_0, m16_1 = None, None, None, None
    mn32_0, mn32_1, mn16_0, mn16_1 = None, None, None, None
    if EDGE_K:
        a_first = a_base + (K_PAD - 2 * BLOCK_K) * stride_ak
        b_first = b_base + (K_PAD - 2 * BLOCK_K) * stride_bk
        mA1 = k_tail_mask(2 * BLOCK_K - K_RES, 1, HALF_M, BLOCK_K, gLoadLayoutA, False)
        mB1 = k_tail_mask(2 * BLOCK_K - K_RES, 1, HALF_N, BLOCK_K, gLoadLayoutB, True)
        if K_RES > BLOCK_K:
            mA0 = k_tail_mask(2 * BLOCK_K - K_RES, 0, HALF_M, BLOCK_K, gLoadLayoutA, False)
            mB0 = k_tail_mask(2 * BLOCK_K - K_RES, 0, HALF_N, BLOCK_K, gLoadLayoutB, True)
        if BAND32:
            m32_1 = k_tail_mask(2 * BLOCK_K - K_RES, 1, 32, BLOCK_K, gLoadLayoutA32, False)
            if K_RES > BLOCK_K:
                m32_0 = k_tail_mask(2 * BLOCK_K - K_RES, 0, 32, BLOCK_K, gLoadLayoutA32, False)
        if BAND16:
            # The column a lane loads is the swizzled one (as v9_tile_small).
            m16_1 = (a16_k + BLOCK_K < 2 * BLOCK_K - K_RES) & (offs_a16r >= 0)[:, None]
            if K_RES > BLOCK_K:
                m16_0 = (a16_k < 2 * BLOCK_K - K_RES) & (offs_a16r >= 0)[:, None]
        if NBAND32:
            mn32_1 = k_tail_mask(2 * BLOCK_K - K_RES, 1, 32, BLOCK_K, gLoadLayoutB32, True)
            if K_RES > BLOCK_K:
                mn32_0 = k_tail_mask(2 * BLOCK_K - K_RES, 0, 32, BLOCK_K, gLoadLayoutB32, True)
        if NBAND16:
            mn16_1 = (b16_k + BLOCK_K < 2 * BLOCK_K - K_RES) & (offs_b16c >= 0)[None, :]
            if K_RES > BLOCK_K:
                mn16_0 = (b16_k < 2 * BLOCK_K - K_RES) & (offs_b16c >= 0)[None, :]

    g_idx = 0
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(g_idx), b_first, b_left_offsets, mask=mB0)
    if NBAND64:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_mid.index(g_idx), b_first, b_mid_offsets, mask=mB0)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_top.index(g_idx), a_first, a_top_offsets, mask=mA0)
    if BAND64:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_mid.index(g_idx), a_first, a_mid_offsets, mask=mA0)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_bot.index(g_idx), a_first, a_bot_offsets, mask=mA0)
    if BAND32:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_s32.index(g_idx), a_first, a_s32_offsets, mask=m32_0)
    if BAND16:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_s16.index(g_idx), a_first, a_s16_offsets, mask=m16_0)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_right.index(g_idx), b_first, b_right_offsets, mask=mB0)
    if NBAND32:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_s32.index(g_idx), b_first, b_s32_offsets, mask=mn32_0)
    if NBAND16:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_s16.index(g_idx), b_first, b_s16_offsets, mask=mn16_0)
    gl.amd.cdna4.async_copy.commit_group()

    g_idx = 1
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_left.index(g_idx), b_first, b_left_offsets_next, mask=mB1
    )
    if NBAND64:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_mid.index(g_idx), b_first, b_mid_offsets_next, mask=mB1)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_top.index(g_idx), a_first, a_top_offsets_next, mask=mA1
    )
    if BAND64:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_mid.index(g_idx), a_first, a_mid_offsets_next, mask=mA1)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_bot.index(g_idx), a_first, a_bot_offsets_next, mask=mA1
    )
    if BAND32:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_s32.index(g_idx), a_first, a_s32_offsets_next,
                                                      mask=m32_1)
    if BAND16:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_s16.index(g_idx), a_first, a_s16_offsets_next,
                                                      mask=m16_1)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_right.index(g_idx), b_first, b_right_offsets_next, mask=mB1
    )
    if NBAND32:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_s32.index(g_idx), b_first, b_s32_offsets_next,
                                                      mask=mn32_1)
    if NBAND16:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_s16.index(g_idx), b_first, b_s16_offsets_next,
                                                      mask=mn16_1)
    gl.amd.cdna4.async_copy.commit_group()

    if not EDGE_K:
        a_base += BLOCK_K * stride_ak * 2
        b_base += BLOCK_K * stride_bk * 2

    gl.amd.cdna4.async_copy.wait_group(6)
    b_left = smemB_left.index(0).load(dotOpLayoutB)
    a_top = smemA_top.index(0).load(dotOpLayoutA)
    if BAND64:
        a_mid = smemA_mid.index(0).load(dotOpLayoutA)
    if BAND16:
        b_left_s = smemB_left.index(0).load(dotOpLayoutB16)
    if NBAND64:
        b_mid = smemB_mid.index(0).load(dotOpLayoutB)
    if NBAND16:
        a_top_s = smemA_top.index(0).load(dotOpLayoutA16n)

    gl.assume(iterMax > 3)

    for k in range(0, iterMax - 2, 2):
        # Sub-iteration 0: consume buffer 0, prefetch into buffer 0. A band's operands are read with
        # the quadrant operand of the same commit group, before the copy that refills their buffer.
        acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")
        if NBAND64:
            acc_n64t = gl.amd.cdna3.mfma(a_top, b_mid, acc_n64t, cd_regclass="a")

        gl.amd.cdna4.async_copy.wait_group(5)
        a_bot = smemA_bot.index(0).load(dotOpLayoutA)
        if BAND32:
            a_s32 = smemA_s32.index(0).load(dotOpLayoutA)
        if BAND16:
            a_s16 = smemA_s16_rd.index(0).load(dotOpLayoutA16)
        if NBAND16:
            a_bot_s = smemA_bot.index(0).load(dotOpLayoutA16n)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(0), b_base, b_left_offsets)
        if NBAND64:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_mid.index(0), b_base, b_mid_offsets)
        gl.amd.cdna4.async_copy.commit_group()

        acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")
        if BAND64:
            acc_ml = gl.amd.cdna3.mfma(a_mid, b_left, acc_ml, cd_regclass="a")
        if BAND32:
            acc_s32l = gl.amd.cdna3.mfma(a_s32, b_left, acc_s32l, cd_regclass="a")
        if BAND16:
            acc_s16l = gl.amd.cdna3.mfma(a_s16, b_left_s, acc_s16l, cd_regclass="a")
        if NBAND64:
            acc_n64b = gl.amd.cdna3.mfma(a_bot, b_mid, acc_n64b, cd_regclass="a")

        gl.amd.cdna4.async_copy.wait_group(5)
        b_right = smemB_right.index(0).load(dotOpLayoutB)
        if BAND16:
            b_right_s = smemB_right.index(0).load(dotOpLayoutB16)
        if NBAND32:
            b_s32 = smemB_s32.index(0).load(dotOpLayoutB)
        if NBAND16:
            b_s16 = smemB_s16_rd.index(0).load(dotOpLayoutB16n)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_top.index(0), a_base, a_top_offsets)
        if BAND64:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_mid.index(0), a_base, a_mid_offsets)
        gl.amd.cdna4.async_copy.commit_group()

        acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")
        if NBAND32:
            acc_n32t = gl.amd.cdna3.mfma(a_top, b_s32, acc_n32t, cd_regclass="a")
        if NBAND16:
            acc_n16t = gl.amd.cdna3.mfma(a_top_s, b_s16, acc_n16t, cd_regclass="a")

        gl.amd.cdna4.async_copy.wait_group(5)
        b_left = smemB_left.index(1).load(dotOpLayoutB)
        if BAND16:
            b_left_s = smemB_left.index(1).load(dotOpLayoutB16)
        if NBAND64:
            b_mid = smemB_mid.index(1).load(dotOpLayoutB)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_bot.index(0), a_base, a_bot_offsets)
        if BAND32:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_s32.index(0), a_base, a_s32_offsets)
        if BAND16:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_s16.index(0), a_base, a_s16_offsets)
        gl.amd.cdna4.async_copy.commit_group()

        acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")
        if BAND64:
            acc_mr = gl.amd.cdna3.mfma(a_mid, b_right, acc_mr, cd_regclass="a")
        if BAND32:
            acc_s32r = gl.amd.cdna3.mfma(a_s32, b_right, acc_s32r, cd_regclass="a")
        if BAND16:
            acc_s16r = gl.amd.cdna3.mfma(a_s16, b_right_s, acc_s16r, cd_regclass="a")
        if NBAND32:
            acc_n32b = gl.amd.cdna3.mfma(a_bot, b_s32, acc_n32b, cd_regclass="a")
        if NBAND16:
            acc_n16b = gl.amd.cdna3.mfma(a_bot_s, b_s16, acc_n16b, cd_regclass="a")

        gl.amd.cdna4.async_copy.wait_group(5)
        a_top = smemA_top.index(1).load(dotOpLayoutA)
        if BAND64:
            a_mid = smemA_mid.index(1).load(dotOpLayoutA)
        if NBAND16:
            a_top_s = smemA_top.index(1).load(dotOpLayoutA16n)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_right.index(0), b_base, b_right_offsets)
        if NBAND32:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_s32.index(0), b_base, b_s32_offsets)
        if NBAND16:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_s16.index(0), b_base, b_s16_offsets)
        gl.amd.cdna4.async_copy.commit_group()

        # Sub-iteration 1: consume buffer 1, prefetch into buffer 1 (odd K-step offsets).
        acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")
        if NBAND64:
            acc_n64t = gl.amd.cdna3.mfma(a_top, b_mid, acc_n64t, cd_regclass="a")

        gl.amd.cdna4.async_copy.wait_group(5)
        a_bot = smemA_bot.index(1).load(dotOpLayoutA)
        if BAND32:
            a_s32 = smemA_s32.index(1).load(dotOpLayoutA)
        if BAND16:
            a_s16 = smemA_s16_rd.index(1).load(dotOpLayoutA16)
        if NBAND16:
            a_bot_s = smemA_bot.index(1).load(dotOpLayoutA16n)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemB_left.index(1), b_base, b_left_offsets_next
        )
        if NBAND64:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_mid.index(1), b_base, b_mid_offsets_next)
        gl.amd.cdna4.async_copy.commit_group()

        acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")
        if BAND64:
            acc_ml = gl.amd.cdna3.mfma(a_mid, b_left, acc_ml, cd_regclass="a")
        if BAND32:
            acc_s32l = gl.amd.cdna3.mfma(a_s32, b_left, acc_s32l, cd_regclass="a")
        if BAND16:
            acc_s16l = gl.amd.cdna3.mfma(a_s16, b_left_s, acc_s16l, cd_regclass="a")
        if NBAND64:
            acc_n64b = gl.amd.cdna3.mfma(a_bot, b_mid, acc_n64b, cd_regclass="a")

        gl.amd.cdna4.async_copy.wait_group(5)
        b_right = smemB_right.index(1).load(dotOpLayoutB)
        if BAND16:
            b_right_s = smemB_right.index(1).load(dotOpLayoutB16)
        if NBAND32:
            b_s32 = smemB_s32.index(1).load(dotOpLayoutB)
        if NBAND16:
            b_s16 = smemB_s16_rd.index(1).load(dotOpLayoutB16n)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemA_top.index(1), a_base, a_top_offsets_next
        )
        if BAND64:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_mid.index(1), a_base, a_mid_offsets_next)
        gl.amd.cdna4.async_copy.commit_group()

        acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")
        if NBAND32:
            acc_n32t = gl.amd.cdna3.mfma(a_top, b_s32, acc_n32t, cd_regclass="a")
        if NBAND16:
            acc_n16t = gl.amd.cdna3.mfma(a_top_s, b_s16, acc_n16t, cd_regclass="a")

        gl.amd.cdna4.async_copy.wait_group(5)
        b_left = smemB_left.index(0).load(dotOpLayoutB)
        if BAND16:
            b_left_s = smemB_left.index(0).load(dotOpLayoutB16)
        if NBAND64:
            b_mid = smemB_mid.index(0).load(dotOpLayoutB)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemA_bot.index(1), a_base, a_bot_offsets_next
        )
        if BAND32:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_s32.index(1), a_base, a_s32_offsets_next)
        if BAND16:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_s16.index(1), a_base, a_s16_offsets_next)
        gl.amd.cdna4.async_copy.commit_group()

        acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")
        if BAND64:
            acc_mr = gl.amd.cdna3.mfma(a_mid, b_right, acc_mr, cd_regclass="a")
        if BAND32:
            acc_s32r = gl.amd.cdna3.mfma(a_s32, b_right, acc_s32r, cd_regclass="a")
        if BAND16:
            acc_s16r = gl.amd.cdna3.mfma(a_s16, b_right_s, acc_s16r, cd_regclass="a")
        if NBAND32:
            acc_n32b = gl.amd.cdna3.mfma(a_bot, b_s32, acc_n32b, cd_regclass="a")
        if NBAND16:
            acc_n16b = gl.amd.cdna3.mfma(a_bot_s, b_s16, acc_n16b, cd_regclass="a")

        gl.amd.cdna4.async_copy.wait_group(5)
        a_top = smemA_top.index(0).load(dotOpLayoutA)
        if BAND64:
            a_mid = smemA_mid.index(0).load(dotOpLayoutA)
        if NBAND16:
            a_top_s = smemA_top.index(0).load(dotOpLayoutA16n)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemB_right.index(1), b_base, b_right_offsets_next
        )
        if NBAND32:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_s32.index(1), b_base, b_s32_offsets_next)
        if NBAND16:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_s16.index(1), b_base, b_s16_offsets_next)
        gl.amd.cdna4.async_copy.commit_group()

        a_base += BLOCK_K * stride_ak * 2
        b_base += BLOCK_K * stride_bk * 2

    offs_cm = gl.arange(0, HALF_M, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = gl.arange(0, HALF_N, gl.SliceLayout(0, gStoreLayoutC))
    c_base = c_ptr + pid_m.to(gl.int64) * BLOCK_M * stride_cm + pid_n.to(gl.int64) * BLOCK_N * stride_cn
    c_tl_offsets = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_tr_offsets = c_tl_offsets + QUAD_N * stride_cn // 2
    c_bl_offsets = c_tl_offsets + QUAD_M * stride_cm // 2
    c_br_offsets = c_bl_offsets + QUAD_N * stride_cn // 2
    rows_left = M - pid_m * BLOCK_M
    cols_left = N - pid_n * BLOCK_N

    acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")
    if NBAND64:
        acc_n64t = gl.amd.cdna3.mfma(a_top, b_mid, acc_n64t, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    l_idx = (iterMax - 2) % 2
    a_bot = smemA_bot.index(l_idx).load(dotOpLayoutA)
    if BAND32:
        a_s32 = smemA_s32.index(l_idx).load(dotOpLayoutA)
    if BAND16:
        a_s16 = smemA_s16_rd.index(l_idx).load(dotOpLayoutA16)
    if NBAND16:
        a_bot_s = smemA_bot.index(l_idx).load(dotOpLayoutA16n)

    acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")
    if BAND64:
        acc_ml = gl.amd.cdna3.mfma(a_mid, b_left, acc_ml, cd_regclass="a")
    if BAND32:
        acc_s32l = gl.amd.cdna3.mfma(a_s32, b_left, acc_s32l, cd_regclass="a")
    if BAND16:
        acc_s16l = gl.amd.cdna3.mfma(a_s16, b_left_s, acc_s16l, cd_regclass="a")
    if NBAND64:
        acc_n64b = gl.amd.cdna3.mfma(a_bot, b_mid, acc_n64b, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(4)
    b_right = smemB_right.index(l_idx).load(dotOpLayoutB)
    if BAND16:
        b_right_s = smemB_right.index(l_idx).load(dotOpLayoutB16)
    if NBAND32:
        b_s32 = smemB_s32.index(l_idx).load(dotOpLayoutB)
    if NBAND16:
        b_s16 = smemB_s16_rd.index(l_idx).load(dotOpLayoutB16n)

    acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")
    if NBAND32:
        acc_n32t = gl.amd.cdna3.mfma(a_top, b_s32, acc_n32t, cd_regclass="a")
    if NBAND16:
        acc_n16t = gl.amd.cdna3.mfma(a_top_s, b_s16, acc_n16t, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(3)
    g_idx = 1 - l_idx
    b_left = smemB_left.index(g_idx).load(dotOpLayoutB)
    if BAND16:
        b_left_s = smemB_left.index(g_idx).load(dotOpLayoutB16)
    if NBAND64:
        b_mid = smemB_mid.index(g_idx).load(dotOpLayoutB)

    acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")
    if BAND64:
        acc_mr = gl.amd.cdna3.mfma(a_mid, b_right, acc_mr, cd_regclass="a")
    if BAND32:
        acc_s32r = gl.amd.cdna3.mfma(a_s32, b_right, acc_s32r, cd_regclass="a")
    if BAND16:
        acc_s16r = gl.amd.cdna3.mfma(a_s16, b_right_s, acc_s16r, cd_regclass="a")
    if NBAND32:
        acc_n32b = gl.amd.cdna3.mfma(a_bot, b_s32, acc_n32b, cd_regclass="a")
    if NBAND16:
        acc_n16b = gl.amd.cdna3.mfma(a_bot_s, b_s16, acc_n16b, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(2)
    a_top = smemA_top.index(g_idx).load(dotOpLayoutA)
    if BAND64:
        a_mid = smemA_mid.index(g_idx).load(dotOpLayoutA)
    if NBAND16:
        a_top_s = smemA_top.index(g_idx).load(dotOpLayoutA16n)

    acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")
    if NBAND64:
        acc_n64t = gl.amd.cdna3.mfma(a_top, b_mid, acc_n64t, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(1)
    a_bot = smemA_bot.index(g_idx).load(dotOpLayoutA)
    if BAND32:
        a_s32 = smemA_s32.index(g_idx).load(dotOpLayoutA)
    if BAND16:
        a_s16 = smemA_s16_rd.index(g_idx).load(dotOpLayoutA16)
    if NBAND16:
        a_bot_s = smemA_bot.index(g_idx).load(dotOpLayoutA16n)

    acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")
    if BAND64:
        acc_ml = gl.amd.cdna3.mfma(a_mid, b_left, acc_ml, cd_regclass="a")
    if BAND32:
        acc_s32l = gl.amd.cdna3.mfma(a_s32, b_left, acc_s32l, cd_regclass="a")
    if BAND16:
        acc_s16l = gl.amd.cdna3.mfma(a_s16, b_left_s, acc_s16l, cd_regclass="a")
    if NBAND64:
        acc_n64b = gl.amd.cdna3.mfma(a_bot, b_mid, acc_n64b, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(0)
    b_right = smemB_right.index(g_idx).load(dotOpLayoutB)
    if BAND16:
        b_right_s = smemB_right.index(g_idx).load(dotOpLayoutB16)
    if NBAND32:
        b_s32 = smemB_s32.index(g_idx).load(dotOpLayoutB)
    if NBAND16:
        b_s16 = smemB_s16_rd.index(g_idx).load(dotOpLayoutB16n)

    c_tl = acc_tl.to(c_ptr.dtype.element_ty)
    c_tl = gl.convert_layout(c_tl, layout=gStoreLayoutC)
    store_edge(c_tl, c_base, c_tl_offsets, gStoreLayoutC, rows_left, cols_left, EDGE_M, EDGE_N)

    acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")

    c_bl = acc_bl.to(c_ptr.dtype.element_ty)
    c_bl = gl.convert_layout(c_bl, layout=gStoreLayoutC)
    store_edge(c_bl, c_base, c_bl_offsets, gStoreLayoutC, rows_left - HALF_M, cols_left, EDGE_M, EDGE_N)

    acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")

    c_tr = acc_tr.to(c_ptr.dtype.element_ty)
    c_tr = gl.convert_layout(c_tr, layout=gStoreLayoutC)
    store_edge(c_tr, c_base, c_tr_offsets, gStoreLayoutC, rows_left, cols_left - HALF_N, EDGE_M, EDGE_N)

    c_br = acc_br.to(c_ptr.dtype.element_ty)
    c_br = gl.convert_layout(c_br, layout=gStoreLayoutC)
    store_edge(c_br, c_base, c_br_offsets, gStoreLayoutC, rows_left - HALF_M, cols_left - HALF_N, EDGE_M,
               EDGE_N)

    if BAND64:
        acc_mr = gl.amd.cdna3.mfma(a_mid, b_right, acc_mr, cd_regclass="a")
        store_band(acc_ml, acc_mr, c_base, R_MID, stride_cm, stride_cn, rows_left, cols_left, BLOCK_N, gStoreLayoutC,
                   EDGE_M, EDGE_N)
    if BAND32:
        acc_s32r = gl.amd.cdna3.mfma(a_s32, b_right, acc_s32r, cd_regclass="a")
        store_band(acc_s32l, acc_s32r, c_base, R_32, stride_cm, stride_cn, rows_left, cols_left, BLOCK_N,
                   gStoreLayoutC, EDGE_M, EDGE_N)
    if BAND16:
        acc_s16r = gl.amd.cdna3.mfma(a_s16, b_right_s, acc_s16r, cd_regclass="a")
        store_band(acc_s16l, acc_s16r, c_base, R_16, stride_cm, stride_cn, rows_left, cols_left, BLOCK_N,
                   gStoreLayoutC, EDGE_M, EDGE_N)

    if NBAND64:
        store_col_band(acc_n64t, acc_n64b, c_base, C_MID, HALF_M, stride_cm, stride_cn, rows_left, cols_left,
                       gStoreLayoutC, EDGE_M, EDGE_N)
    if NBAND32:
        acc_n32t = gl.amd.cdna3.mfma(a_top, b_s32, acc_n32t, cd_regclass="a")
        acc_n32b = gl.amd.cdna3.mfma(a_bot, b_s32, acc_n32b, cd_regclass="a")
        store_col_band(acc_n32t, acc_n32b, c_base, C_32, HALF_M, stride_cm, stride_cn, rows_left, cols_left,
                       store_layout(32), EDGE_M, EDGE_N)
    if NBAND16:
        acc_n16t = gl.amd.cdna3.mfma(a_top_s, b_s16, acc_n16t, cd_regclass="a")
        acc_n16b = gl.amd.cdna3.mfma(a_bot_s, b_s16, acc_n16b, cd_regclass="a")
        store_col_band(acc_n16t, acc_n16b, c_base, C_16, HALF_M, stride_cm, stride_cn, rows_left, cols_left,
                       store_layout(16), EDGE_M, EDGE_N)


@gluon.jit
def store_col_band(acc_t, acc_b, c_base, COL0: gl.constexpr, HALF_M: gl.constexpr, stride_cm, stride_cn, rows_left,
                   cols_left, layout: gl.constexpr, EDGE_M: gl.constexpr, EDGE_N: gl.constexpr):
    """C store of a column band of v9_tile: its top and bottom halves, columns COL0 .. COL0 + width."""
    offs_m = gl.arange(0, HALF_M, gl.SliceLayout(1, layout))
    offs_n = gl.arange(0, acc_t.shape[1], gl.SliceLayout(0, layout))
    offs_t = stride_cm * offs_m[:, None] + stride_cn * (offs_n + COL0)[None, :]
    c_t = gl.convert_layout(acc_t.to(c_base.dtype.element_ty), layout=layout)
    store_edge(c_t, c_base, offs_t, layout, rows_left, cols_left - COL0, EDGE_M, EDGE_N)
    c_b = gl.convert_layout(acc_b.to(c_base.dtype.element_ty), layout=layout)
    store_edge(c_b, c_base, offs_t + HALF_M * stride_cm, layout, rows_left - HALF_M, cols_left - COL0, EDGE_M, EDGE_N)


@gluon.jit
def store_band(acc_l, acc_r, c_base, ROW0: gl.constexpr, stride_cm, stride_cn, rows_left, cols_left,
               BLOCK_N: gl.constexpr, layout: gl.constexpr, EDGE_M: gl.constexpr, EDGE_N: gl.constexpr):
    """C store of a row band of v9_tile: its left and right halves, rows ROW0 .. ROW0 + rows of the tile."""
    offs_m = gl.arange(0, acc_l.shape[0], gl.SliceLayout(1, layout))
    offs_n = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, layout))
    offs_l = stride_cm * (offs_m + ROW0)[:, None] + stride_cn * offs_n[None, :]
    c_l = gl.convert_layout(acc_l.to(c_base.dtype.element_ty), layout=layout)
    store_edge(c_l, c_base, offs_l, layout, rows_left - ROW0, cols_left, EDGE_M, EDGE_N)
    c_r = gl.convert_layout(acc_r.to(c_base.dtype.element_ty), layout=layout)
    store_edge(c_r, c_base, offs_l + BLOCK_N * stride_cn // 2, layout, rows_left - ROW0, cols_left - BLOCK_N // 2,
               EDGE_M, EDGE_N)


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

    sharedLayoutA: gl.constexpr = tile_shared_layout(BLOCK_M, BLOCK_K, False, MFMA_32, a_ptr.dtype.element_ty)
    sharedLayoutB: gl.constexpr = tile_shared_layout(BLOCK_N // 2, BLOCK_K, True, MFMA_32, b_ptr.dtype.element_ty)
    gLoadLayoutA: gl.constexpr = tile_load_layout(sharedLayoutA)
    gLoadLayoutB: gl.constexpr = tile_load_layout(sharedLayoutB)
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

    a_base = a_ptr + pid_m.to(gl.int64) * BLOCK_M * stride_am
    b_base = b_ptr + pid_n.to(gl.int64) * BLOCK_N * stride_bn
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
    c_base = c_ptr + pid_m.to(gl.int64) * BLOCK_M * stride_cm + pid_n.to(gl.int64) * BLOCK_N * stride_cn
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
    EDGE_N: gl.constexpr = False,
    EDGE_K: gl.constexpr = False,
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

    EDGE_N: B's columns past N are clamped to the tile's last in-bounds one (every load reads valid
    memory), and the C store and, without C_T, the bias load are masked to N. EDGE_K: K is rounded up
    to K_PAD, a whole number of K-step pairs, and the prologue loads the last, partial pair first (as
    v9_tile), its lanes at or past K masked to zero; the mask compares the column a lane actually
    loads (the swizzle is folded into the copy offsets). The main loop then covers 0 .. K_PAD - 2 BLOCK_K.
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
    cols_left = N - pid_n * BLOCK_N
    a_base = a_ptr + pid_m.to(gl.int64) * BLOCK_M * stride_am
    b_base = b_ptr + pid_n.to(gl.int64) * BLOCK_N * stride_bn
    if MFMA4:
        a_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    else:
        a_offsets = offs_am[:, None] * stride_am + swizzled_k(offs_ak[None, :], offs_am[:, None], BLOCK_K) * stride_ak
    if EDGE_N:
        b_offsets = (swizzled_k(offs_bk[:, None], offs_bn[None, :], BLOCK_K) * stride_bk
                     + gl.minimum(offs_bn, cols_left - 1)[None, :] * stride_bn)
    else:
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
            if EDGE_N:
                bias = gl.amd.cdna3.buffer_load(ptr=bias_ptr + pid_n * BLOCK_N, offsets=offs_bias,
                                                mask=offs_bias < cols_left, other=0.0)
            else:
                bias = gl.amd.cdna3.buffer_load(ptr=bias_ptr + pid_n * BLOCK_N, offsets=offs_bias)
            acc = acc + bias.to(gl.float32)[None, :]

    K_PAD: gl.constexpr = (K + 2 * BLOCK_K - 1) // (2 * BLOCK_K) * (2 * BLOCK_K) if EDGE_K else K
    K_RES: gl.constexpr = K_PAD - K
    if EDGE_K:
        # The last, partial pair first; K-step 1 of it always reaches past K, K-step 0 only when
        # K_RES > BLOCK_K. The main loop then starts at column 0 (a_base, b_base unmoved).
        if MFMA4:
            a_k = offs_ak[None, :]
        else:
            a_k = swizzled_k(offs_ak[None, :], offs_am[:, None], BLOCK_K)
        b_k = swizzled_k(offs_bk[:, None], offs_bn[None, :], BLOCK_K)
        for s in gl.static_range(2):
            tail = K_PAD - 2 * BLOCK_K + s * BLOCK_K
            if s == 1 or K_RES > BLOCK_K:
                k_hi = 2 * BLOCK_K - K_RES - s * BLOCK_K
                mA_s = (a_k < k_hi) & (offs_am >= 0)[:, None]
                if MASK_A:
                    mA_s = mA_s & mA
                gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(s), a_base + tail * stride_ak, a_offsets,
                                                              mask=mA_s)
                gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB.index(s), b_base + tail * stride_bk, b_offsets,
                                                              mask=b_k < k_hi)
            else:
                gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(s), a_base + tail * stride_ak, a_offsets,
                                                              mask=mA)
                gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB.index(s), b_base + tail * stride_bk, b_offsets)
            gl.amd.cdna4.async_copy.commit_group()
    else:
        for s in gl.static_range(2):
            gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(s), a_base, a_offsets, mask=mA)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB.index(s), b_base, b_offsets)
            gl.amd.cdna4.async_copy.commit_group()
            a_base += BLOCK_K * stride_ak
            b_base += BLOCK_K * stride_bk

    gl.amd.cdna4.async_copy.wait_group(1)
    a = a_rows_view(smemA_rd.index(0), MR, MP).load(dotOpLayoutA)
    b = smemB_rd.index(0).load(dotOpLayoutB)

    iterMax: gl.constexpr = K_PAD // BLOCK_K
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
    c_base = c_ptr + pid_m.to(gl.int64) * BLOCK_M * stride_cm + pid_n.to(gl.int64) * BLOCK_N * stride_cn
    c_offsets = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    if MASK_A or EDGE_N:
        c_rows = offs_cm < BLOCK_M
        if EDGE_M:
            c_rows = c_rows & (offs_cm < rows_left)
        c_cols = offs_cn < BLOCK_N
        if EDGE_N:
            c_cols = offs_cn < cols_left
        gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_offsets, stored_value=c,
                                  mask=c_rows[:, None] & c_cols[None, :])
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
def init_band_n(bias_ptr, pid_n, C0: gl.constexpr, WIDTH: gl.constexpr, ROWS: gl.constexpr, BLOCK_N: gl.constexpr,
                MFMA: gl.constexpr, ADD_BIAS: gl.constexpr, cols_left, EDGE_N: gl.constexpr):
    """The [ROWS, WIDTH] accumulator of the columns C0 .. C0 + WIDTH of a v9_tile tile, from bias[N]
    or zero; with EDGE_N the bias is loaded only for the tile's first cols_left columns."""
    acc = gl.zeros((ROWS, WIDTH), gl.float32, MFMA)
    if ADD_BIAS:
        offs = gl.arange(0, WIDTH, gl.SliceLayout(0, MFMA))
        base = bias_ptr + pid_n * BLOCK_N + C0
        if EDGE_N:
            bias = gl.amd.cdna3.buffer_load(ptr=base, offsets=offs, mask=offs < cols_left - C0, other=0.0)
        else:
            bias = gl.amd.cdna3.buffer_load(ptr=base, offsets=offs)
        acc = acc + bias.to(gl.float32)[None, :]
    return acc


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
def tile_shared_layout(rows, block_k, k_first, mfma_32, dtype):
    """Padded shared layout of a [rows, block_k] block of A (k_first: B's [block_k, rows]), as aiter's
    gfx950 GEMM builds it: Triton's compute_efficient_padded_shared_layout for the MFMA operand that
    reads it. 16 elements of padding after every 512, so each wave's 1 KB load lands contiguously,
    with the row bits one ds_read_b128 spans 32, 64, 128 and 256 B apart: no bank conflicts (WORKLOG
    §15). At 128 and 32 rows with BLOCK_K = 64 this is v9's layout. Triton has none below 32 rows,
    nor at 32 rows for the 32x32x16 MFMA."""
    op = gl.DotOperandLayout(operand_index=1 if k_first else 0, parent=mfma_layout(mfma_32), k_width=8)
    shape = [block_k, rows] if k_first else [rows, block_k]
    layout = cdna4.compute_efficient_padded_shared_layout(op, shape, dtype)
    assert layout is not None, f"no efficient padded layout for {shape} (fewer than 32 rows?)"
    return layout


@gluon.constexpr_function
def tile_load_layout(shared, num_warps=4):
    """The global-load layout that fills a tile_shared_layout (aiter's compute_gload_layout): its offset
    bases, in LDS address order, split into 3 register bases (8 elements, 16 B per lane), 6 lane bases
    and 2 warp bases, the rest registers again (further loads). Each warp's buffer_load_to_lds then
    fills 1 KB of LDS in order."""
    bases = [list(b) for b in shared.offset_bases]
    n_warp = int(math.log2(num_warps))
    warp = bases[9:9 + n_warp]
    warp += [[0] * len(shared.shape)] * (n_warp - len(warp))
    return gl.DistributedLinearLayout(reg_bases=bases[:3] + bases[9 + n_warp:], lane_bases=bases[3:9],
                                      warp_bases=warp, block_bases=[], shape=list(shared.shape))


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
def strip_load_layout(rows, block_k, k_first, num_warps, replicate=False):
    """Global-load layout of a [rows, block_k] strip (k_first: B's [block_k, rows]), K contiguous:
    8 K values per lane, min(64, block_k / 8) lanes along K, the rest of the lanes and then the
    warps along the rows. A warp's buffer_load_to_lds_dwordx4 then covers whole rows, 1 KB.
    replicate: allow fewer rows than one load per warp (the warps past them hold copies, which
    the copies predicate off)."""
    tk = min(64, block_k // 8)
    assert replicate or rows >= num_warps * (64 // tk), f"{rows} rows x {block_k} K is less than one load per warp"
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


def row_bands(bm):
    """v9_tile's arguments for BLOCK_M = 128 + 64 a + 32 b + 16 c (a, b, c in {0, 1}): the quadrants
    of a 128-row tile and the row bands under them. Empty for a power of two."""
    if bm & (bm - 1) == 0:
        return {}
    assert 128 < bm < 256 and bm % 16 == 0, f"BLOCK_M={bm}: a power of two or a multiple of 16 in (128, 256)"
    extra = bm - 128
    return dict(MAIN_M=128, BAND64=bool(extra & 64), BAND32=bool(extra & 32), BAND16=bool(extra & 16))


def col_bands(bn):
    """row_bands for BLOCK_N: the quadrants of a 128-column tile and the column bands to their right."""
    if bn & (bn - 1) == 0:
        return {}
    assert 128 < bn < 256 and bn % 16 == 0, f"BLOCK_N={bn}: a power of two or a multiple of 16 in (128, 256)"
    extra = bn - 128
    return dict(MAIN_N=128, NBAND64=bool(extra & 64), NBAND32=bool(extra & 32), NBAND16=bool(extra & 16))


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
    # Other N and K run with EDGE_N / EDGE_K. K must keep 16-byte rows (as matmul's tiles).
    assert K % 16 == 0, f"K={K} must be a multiple of 16 (pad A and B)"
    if mfma4:
        assert km <= 8 and kn % (64 * w) == 0, f"4x64 MFMA: BM <= 8 and 64 columns per wave ({km}x{kn}, {w} waves)"
    else:
        # W x 16 > BN replicates: the waves hold copies of the tile (Triton's warps on a 16x16 dot).
        assert kn % (16 * w) == 0 or (16 * w) % kn == 0, f"{kn} columns do not split over {w} waves"
    grid_m, grid_n = triton.cdiv(M, km), triton.cdiv(N, kn)
    GRID_MN = grid_m * grid_n
    gm_env = os.environ.get("V9_TILE_GM")
    GROUP_SIZE_M = int(gm_env) if gm_env else group_size_m(km, kn, grid_m, GRID_MN, 8, tiles_per_cu=4 // w)
    pin = pin or os.environ.get("V9_TILE_PIN") == "1"
    force_edge = os.environ.get("V9_FORCE_EDGE") == "1"
    k = v9_tile_small[(GRID_MN, 1)](
        a, b, c, bias, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=km, BLOCK_N=kn, BLOCK_K=bk, GRID_MN=GRID_MN, NUM_XCDS=8, GROUP_SIZE_M=GROUP_SIZE_M,
        ADD_BIAS=bias is not None, NUM_WARPS=w,
        EDGE_M=force_edge or M % km != 0, C_T=swap,
        EDGE_N=force_edge or N % kn != 0,
        EDGE_K=force_edge or K % (2 * bk) != 0,
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
    mfma_32 = os.environ.get("V9_TILE_MFMA") == "32"
    bands = row_bands(BLOCK_M)
    if bands:
        assert regions == 4, f"BLOCK_M={BLOCK_M}: row bands need the four-quadrant kernel (no r2)"
        # The 32x32x16 MFMA needs 32 rows per wave, and Triton has no 32-row padded layout for it.
        assert not (mfma_32 and (bands["BAND32"] or bands["BAND16"])), "V9_TILE_MFMA=32: BLOCK_M = 128 + 64 only"
        # The 16-row band puts the 4 waves side by side over each B half: 16 columns each.
        assert not bands["BAND16"] or BLOCK_N >= 128, f"BLOCK_M={BLOCK_M} needs BLOCK_N >= 128"
        assert BLOCK_M * BLOCK_N <= 256 * 256, f"{BLOCK_M}x{BLOCK_N}: more than 256 accumulator registers"
    nbands = col_bands(BLOCK_N)
    if nbands:
        assert regions == 4, f"BLOCK_N={BLOCK_N}: column bands need the four-quadrant kernel (no r2)"
        # A band of rows and one of columns would meet in corner pieces, which v9_tile does not build.
        assert not bands, f"{BLOCK_M}x{BLOCK_N}: BLOCK_M and BLOCK_N cannot both be banded"
        assert not (mfma_32 and (nbands["NBAND32"] or nbands["NBAND16"])), "V9_TILE_MFMA=32: BLOCK_N = 128 + 64 only"
        # The 16-column band stacks the 4 waves over each A half: BLOCK_M / 8 rows each.
        assert not nbands["NBAND16"] or BLOCK_M >= 128, f"BLOCK_N={BLOCK_N} needs BLOCK_M >= 128"
        assert BLOCK_M * BLOCK_N <= 256 * 256, f"{BLOCK_M}x{BLOCK_N}: more than 256 accumulator registers"
        # amdgcnas's lgkmcnt rotation leaves 46 MFMA operands of 256x192's loop unwaited after their
        # ds_read, and C comes out wrong (WORKLOG §18). Without the plugin the kernel is correct.
        assert not ((BLOCK_M, BLOCK_N) == (256, 192) and os.environ.get("TRITON_AMDGCNAS_PLUGIN")), \
            "256x192 is miscompiled by the amdgcnas plugin: unset TRITON_AMDGCNAS_PLUGIN"
        bands = nbands
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
        MFMA_32=mfma_32,
        EDGE_M=EDGE_M,
        EDGE_N=EDGE_N,
        EDGE_K=EDGE_K,
        schedule_hint=schedule_hint,
        num_warps=num_warps,
        **bands,
    )
    return c
