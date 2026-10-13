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

import os

import triton
import torch
from bias import init_acc
from common import get_pids
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def v9_ragged(
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
    EDGE_M: gl.constexpr = False,
    EDGE_N: gl.constexpr = False,
    EDGE_K: gl.constexpr = False,
    TILE_LIMIT: gl.constexpr = 0,
):
    """
    v9_beyond_hotloop for M and N that need not be multiples of the tile. With EDGE_M / EDGE_N off
    the kernel is v9's.

    TILE_LIMIT (an experiment, 0 = off): run as a GEMM of the grid's first TILE_LIMIT tile ids, as
    v13's hook does: TILE_LIMIT programs, program p takes tile id p with the XCD remap over
    TILE_LIMIT, and the GROUP_SIZE_M swizzle over the full grid's tile counts; the other tiles of C
    are not written. With them on (the host sets both when either dimension is ragged), A rows
    past M and B columns past N are clamped to the tile's last in-bounds one, so every load reads
    valid memory, and the C stores and the bias load are masked to M x N (v25's edge handling).
    EDGE_K (the host sets it when K is not a multiple of 2 * BLOCK_K): K is rounded up to K_PAD, a
    whole number of K-step pairs, and the last, partial pair goes first: the prologue loads it
    (columns K_PAD - 2 BLOCK_K .. K_PAD) with the lanes at or past K masked to zero, and the main loop
    then covers columns 0 .. K_PAD - 2 BLOCK_K unmasked, from the start of each row as in v9. Shifting
    the whole loop to start K_PAD - K before the matrix instead would misalign every main-loop read
    with the 128-byte lines unless K_PAD - K is a multiple of 64 (55% slower at 4096x4096x8176).

    v9_beyond_hotloop's notes:

    Beyond the hot loop: L2 locality + interleaved epilogue on top of v8_sliceMN.

    Builds on v8's slice-both-M-and-N design (4 quadrant accumulators, unrolled by 2,
    pre-computed _next offsets) and adds:

    1. XCD-aware PID remapping + GROUP_SIZE_M workgroup swizzling for L2 cache locality.
    2. Interleaved epilogue using extract_slice: each 128x128 accumulator is split into
       two 64x128 sub-tiles along M. Stores for sub-tile i are pipelined with the MFMA
       computing sub-tile i+1, spreading write traffic over time.
    """

    if TILE_LIMIT > 0:
        pid_m, pid_n = get_pids(M, N, BLOCK_M, BLOCK_N, TILE_LIMIT, NUM_XCDS, GROUP_SIZE_M)
    else:
        pid_m, pid_n = get_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    # Half-M global load layout
    gLoadLayoutA: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0]],
        lane_bases=[[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [64, 0]],
        warp_bases=[[1, 0], [2, 0]],
        block_bases=[],
        shape=[BLOCK_M // 2, BLOCK_K],
    )
    # Half-N global load layout
    gLoadLayoutB: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 4], [0, 8]],
        lane_bases=[[8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 64]],
        warp_bases=[[0, 1], [0, 2]],
        block_bases=[],
        shape=[BLOCK_K, BLOCK_N // 2],
    )

    # Half-M padded shared layout
    sharedLayoutA: gl.constexpr = gl.PaddedSharedLayout(
        [[512, 16]],
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [16, 0],
            [32, 0],
            [64, 0],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
        ],
        [],
        [BLOCK_M // 2, BLOCK_K],
    )
    # Half-N padded shared layout
    sharedLayoutB: gl.constexpr = gl.PaddedSharedLayout(
        [[512, 16]],
        [
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [16, 0],
            [32, 0],
            [0, 16],
            [0, 32],
            [0, 64],
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
        ],
        [],
        [BLOCK_K, BLOCK_N // 2],
    )

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
    # Ragged K: the prologue loads the last, partial pair (masked at K); the main loop then starts at
    # column 0.
    K_PAD: gl.constexpr = (K + 2 * BLOCK_K - 1) // (2 * BLOCK_K) * (2 * BLOCK_K) if EDGE_K else K
    K_RES: gl.constexpr = K_PAD - K
    a_first = a_base
    b_first = b_base
    if EDGE_K:
        a_first = a_base + (K_PAD - 2 * BLOCK_K) * stride_ak
        b_first = b_base + (K_PAD - 2 * BLOCK_K) * stride_bk

    # Two sets of offsets: base (even K-steps) and _next (odd K-steps). With EDGE_M / EDGE_N the
    # rows / columns past the matrix are clamped to the tile's last in-bounds one: their loads read
    # valid memory and only feed C elements that the store masks drop.
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

    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 2]
    )

    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfmaLayout, k_width=8)
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfmaLayout, k_width=8)

    # Ahead of the prologue loads, so the prologue's wait covers the bias load.
    acc_tl, acc_bl, acc_tr, acc_br = init_acc_edge(
        bias_ptr, pid_n, N - pid_n * BLOCK_N, BLOCK_M, BLOCK_N, mfmaLayout, ADD_BIAS, EDGE_N
    )

    iterMax = gl.cdiv(K_PAD, BLOCK_K)

    # The last pair's masks (EDGE_K): K-step 1 always reaches past K, K-step 0 only when
    # K_RES > BLOCK_K.
    mA0, mB0, mA1, mB1 = None, None, None, None
    if EDGE_K:
        mA1 = k_tail_mask(2 * BLOCK_K - K_RES, 1, BLOCK_M // 2, BLOCK_K, gLoadLayoutA, False)
        mB1 = k_tail_mask(2 * BLOCK_K - K_RES, 1, BLOCK_N // 2, BLOCK_K, gLoadLayoutB, True)
        if K_RES > BLOCK_K:
            mA0 = k_tail_mask(2 * BLOCK_K - K_RES, 0, BLOCK_M // 2, BLOCK_K, gLoadLayoutA, False)
            mB0 = k_tail_mask(2 * BLOCK_K - K_RES, 0, BLOCK_N // 2, BLOCK_K, gLoadLayoutB, True)

    ## Prologue — same as v8
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

    ## Main loop — same as v8
    for k in range(0, iterMax - 2, 2):

        ## =============================================================
        ## Sub-iteration 0: consume buffer 0, prefetch into buffer 0
        ## =============================================================

        ########################################
        ## Region 0: C_tl = DOT(a_top, b_left)
        ########################################
        acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")

        gl.amd.cdna4.async_copy.wait_group(5)
        a_bot = smemA_bot.index(0).load(dotOpLayoutA)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(0), b_base, b_left_offsets)
        gl.amd.cdna4.async_copy.commit_group()

        ########################################
        ## Region 1: C_bl = DOT(a_bot, b_left)
        ########################################
        acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")

        gl.amd.cdna4.async_copy.wait_group(5)
        b_right = smemB_right.index(0).load(dotOpLayoutB)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_top.index(0), a_base, a_top_offsets)
        gl.amd.cdna4.async_copy.commit_group()

        ########################################
        ## Region 2: C_tr = DOT(a_top, b_right)
        ########################################
        acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")

        gl.amd.cdna4.async_copy.wait_group(5)
        b_left = smemB_left.index(1).load(dotOpLayoutB)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_bot.index(0), a_base, a_bot_offsets)
        gl.amd.cdna4.async_copy.commit_group()

        ########################################
        ## Region 3: C_br = DOT(a_bot, b_right)
        ########################################
        acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")

        gl.amd.cdna4.async_copy.wait_group(5)
        a_top = smemA_top.index(1).load(dotOpLayoutA)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_right.index(0), b_base, b_right_offsets)
        gl.amd.cdna4.async_copy.commit_group()

        ## =============================================================
        ## Loop unroll: Sub-iteration 1: consume buffer 1, prefetch
        ## into buffer 1. AC uses _next offsets (odd K-step).
        ## =============================================================

        ########################################
        ## Region 0: C_tl = DOT(a_top, b_left)
        ########################################
        acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")

        gl.amd.cdna4.async_copy.wait_group(5)
        a_bot = smemA_bot.index(1).load(dotOpLayoutA)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemB_left.index(1), b_base, b_left_offsets_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        ########################################
        ## Region 1: C_bl = DOT(a_bot, b_left)
        ########################################
        acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")

        gl.amd.cdna4.async_copy.wait_group(5)
        b_right = smemB_right.index(1).load(dotOpLayoutB)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemA_top.index(1), a_base, a_top_offsets_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        ########################################
        ## Region 2: C_tr = DOT(a_top, b_right)
        ########################################
        acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")

        gl.amd.cdna4.async_copy.wait_group(5)
        b_left = smemB_left.index(0).load(dotOpLayoutB)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemA_bot.index(1), a_base, a_bot_offsets_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        ########################################
        ## Region 3: C_br = DOT(a_bot, b_right)
        ########################################
        acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")

        gl.amd.cdna4.async_copy.wait_group(5)
        a_top = smemA_top.index(0).load(dotOpLayoutA)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemB_right.index(1), b_base, b_right_offsets_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        a_base += BLOCK_K * stride_ak * 2
        b_base += BLOCK_K * stride_bk * 2

    ## Epilogue: 4-quadrant stores with natural-pipeline ordering (matches v8).
    ## v9's contribution lives in the prologue / pid remapping (§3); the epilogue
    ## is unchanged from v8 because the sub-tile variant produced only ~200 cycles
    ## of additional savings — within noise relative to the full kernel.

    gStoreLayoutC: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])

    offs_cm = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gStoreLayoutC))
    c_base = c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
    c_tl_offsets = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_tr_offsets = c_tl_offsets + BLOCK_N * stride_cn // 2
    c_bl_offsets = c_tl_offsets + BLOCK_M * stride_cm // 2
    c_br_offsets = c_bl_offsets + BLOCK_N * stride_cn // 2
    rows_left = M - pid_m * BLOCK_M
    cols_left = N - pid_n * BLOCK_N

    ## Iter iterMax - 2: same 4-region pattern as main loop, no AC
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

    ## Iter iterMax - 1
    ## Natural-pipeline epilogue: each store follows its MFMA with one
    ## MFMA cycle of gap, yielding uniform MFMA-store interleaving.
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
def k_tail_mask(k_hi, STEP: gl.constexpr, ROWS: gl.constexpr, BLOCK_K: gl.constexpr, layout: gl.constexpr,
                K_FIRST: gl.constexpr):
    """Load mask of K-step STEP (0 or 1) of a half-tile in the last pair, of whose 2 BLOCK_K K values
    only the first k_hi lie inside the matrix: lanes holding the others load zeros. K_FIRST: B's
    [BLOCK_K, ROWS] half-tile, else A's [ROWS, BLOCK_K]."""
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
    """buffer_store of a quadrant of C; with EDGE_M / EDGE_N only its first rows_left rows and
    cols_left columns (either may exceed the quadrant)."""
    if EDGE_M or EDGE_N:
        r = gl.arange(0, c.shape[0], gl.SliceLayout(1, layout))[:, None]
        cc = gl.arange(0, c.shape[1], gl.SliceLayout(0, layout))[None, :]
        gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_offsets, stored_value=c,
                                  mask=(r < rows_left) & (cc < cols_left))
    else:
        gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_offsets, stored_value=c)


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


def matmul(a, b, c=None, bias=None, schedule_hint="mfma-schedule"):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.stride(1) == 1, "Matrix A must be row-major (its row stride may be padded)"
    M, K = a.shape
    K, N = b.shape
    assert bias is None or (bias.shape == (N,) and bias.is_contiguous()), "bias must be contiguous (N,)"
    BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 64
    # The prologue's two K-steps, the 2x-unrolled loop and the 2-step epilogue need an even number
    # of at least 4 K-steps (v9 does not check, and silently computes a wrong C otherwise). Other K
    # run over K rounded up to a pair (EDGE_K). K must keep 16-byte rows: Triton only knows a stride
    # is aligned when it is a multiple of 16.
    assert K % 16 == 0, f"K={K} must be a multiple of 16 (pad A and B)"
    k_pad = triton.cdiv(K, 2 * BLOCK_K) * 2 * BLOCK_K
    assert k_pad >= 4 * BLOCK_K, f"K={K} needs at least {4 * BLOCK_K} after rounding up to {2 * BLOCK_K}"
    EDGE_K = K != k_pad
    num_warps = 4
    if c is None:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    GRID_MN = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    # GEMM_TILE_LIMIT (an experiment): compute only the first T tile ids, as a T-tile GEMM.
    TILE_LIMIT = int(os.environ.get("GEMM_TILE_LIMIT", "0"))
    assert 0 <= TILE_LIMIT <= GRID_MN
    grid = (TILE_LIMIT or GRID_MN, 1)
    NUM_XCDS = 8
    GROUP_SIZE_M = 4
    # Ragged M / N: clamped loads and masked stores, on both dimensions if either is ragged (as
    # v25). V9_FORCE_EDGE=1 turns them on for any shape, to measure their cost.
    EDGE_M = EDGE_N = os.environ.get("V9_FORCE_EDGE") == "1" or M % BLOCK_M != 0 or N % BLOCK_N != 0
    v9_ragged[grid](
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
        EDGE_M=EDGE_M,
        EDGE_N=EDGE_N,
        EDGE_K=EDGE_K,
        TILE_LIMIT=TILE_LIMIT,
        schedule_hint=schedule_hint,
        num_warps=num_warps,
    )
    return c
