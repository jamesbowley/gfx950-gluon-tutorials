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
"""

import math
import os

import torch
import triton
from common import get_pids, init_acc
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

    a_top_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    a_bot_offsets = a_top_offsets + BLOCK_M * stride_am // 2
    b_left_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    b_right_offsets = b_left_offsets + BLOCK_N * stride_bn // 2

    a_top_offsets_next = a_top_offsets + BLOCK_K * stride_ak
    a_bot_offsets_next = a_bot_offsets + BLOCK_K * stride_ak
    b_left_offsets_next = b_left_offsets + BLOCK_K * stride_bk
    b_right_offsets_next = b_right_offsets + BLOCK_K * stride_bk

    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfmaLayout, k_width=8)
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfmaLayout, k_width=8)

    acc_tl, acc_bl, acc_tr, acc_br = init_acc(
        bias_ptr, pid_n, BLOCK_M, BLOCK_N, mfmaLayout, ADD_BIAS
    )

    iterMax = gl.cdiv(K, BLOCK_K)

    g_idx = 0
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(g_idx), b_base, b_left_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_top.index(g_idx), a_base, a_top_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_bot.index(g_idx), a_base, a_bot_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_right.index(g_idx), b_base, b_right_offsets)
    gl.amd.cdna4.async_copy.commit_group()

    g_idx = 1
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_left.index(g_idx), b_base, b_left_offsets_next
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_top.index(g_idx), a_base, a_top_offsets_next
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_bot.index(g_idx), a_base, a_bot_offsets_next
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_right.index(g_idx), b_base, b_right_offsets_next
    )
    gl.amd.cdna4.async_copy.commit_group()

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
    gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_tl_offsets, stored_value=c_tl)

    acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")

    c_bl = acc_bl.to(c_ptr.dtype.element_ty)
    c_bl = gl.convert_layout(c_bl, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_bl_offsets, stored_value=c_bl)

    acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")

    c_tr = acc_tr.to(c_ptr.dtype.element_ty)
    c_tr = gl.convert_layout(c_tr, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_tr_offsets, stored_value=c_tr)

    c_br = acc_br.to(c_ptr.dtype.element_ty)
    c_br = gl.convert_layout(c_br, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_br_offsets, stored_value=c_br)


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
    a_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    b_left_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    b_right_offsets = b_left_offsets + BLOCK_N * stride_bn // 2

    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfmaLayout, k_width=8)
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfmaLayout, k_width=8)

    acc_left, acc_right = init_acc_2r(bias_ptr, pid_n, BLOCK_M, BLOCK_N, mfmaLayout, ADD_BIAS)

    iterMax = gl.cdiv(K, BLOCK_K)

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
    gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_left_offsets, stored_value=c_left)

    acc_right = gl.amd.cdna3.mfma(a, b_right, acc_right, cd_regclass="a")

    c_right = acc_right.to(c_ptr.dtype.element_ty)
    c_right = gl.convert_layout(c_right, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_right_offsets, stored_value=c_right)


@gluon.jit
def init_acc_2r(bias_ptr, pid_n, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, MFMA: gl.constexpr,
                ADD_BIAS: gl.constexpr):
    """The two [BLOCK_M, BLOCK_N / 2] accumulators of v9_tile_2r, from bias[N] or zero."""
    acc_l = gl.zeros((BLOCK_M, BLOCK_N // 2), gl.float32, MFMA)
    acc_r = acc_l
    if ADD_BIAS:
        offs_bias = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, MFMA))
        bias_base = bias_ptr + pid_n * BLOCK_N
        bias_l = gl.amd.cdna3.buffer_load(ptr=bias_base, offsets=offs_bias)
        bias_r = gl.amd.cdna3.buffer_load(ptr=bias_base + BLOCK_N // 2, offsets=offs_bias)
        acc_l = acc_l + bias_l.to(gl.float32)[None, :]
        acc_r = acc_r + bias_r.to(gl.float32)[None, :]
    return acc_l, acc_r


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


def group_size_m(block_m, block_n, grid_m, grid_mn, num_xcds=8, num_cus=256):
    """v9's rule with tile sizes: minimise an XCD's A + B footprint GM * BM + ceil(P / GM) * BN,
    P = the programs an XCD holds at once. Ties go to the smaller GM (v9's 4 at P = 32)."""
    p = max(1, min(num_cus, grid_mn) // num_xcds)
    best = None
    for gm in (1, 2, 4, 8, 16, 32):
        if gm > max(1, grid_m):
            break
        cost = gm * block_m + math.ceil(p / gm) * block_n
        if best is None or cost < best[0]:
            best = (cost, gm)
    return best[1]


def parse_tile(spec):
    """"BMxBN[xBK][r2]" -> (BM, BN, BK, regions); r2 selects v9_tile_2r."""
    spec = spec.lower()
    regions = 2 if spec.endswith("r2") else 4
    parts = [int(x) for x in spec.removesuffix("r2").split("x")]
    if len(parts) == 2:
        parts.append(64)
    assert len(parts) == 3, f"tile spec {spec!r}: expected BMxBN[xBK][r2]"
    return (*parts, regions)


def tile_config(tile=None):
    """(BM, BN, BK, regions) from the argument, else V9_TILE, else 256x256x64."""
    if tile is None:
        tile = os.environ.get("V9_TILE", "256x256x64")
    if isinstance(tile, str):
        return parse_tile(tile)
    return (*tile, 4) if len(tile) == 3 else tuple(tile)


def matmul(a, b, c=None, bias=None, tile=None):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.stride(1) == 1, "Matrix A must be row-major (its row stride may be padded)"
    M, K = a.shape
    K, N = b.shape
    assert bias is None or (bias.shape == (N,) and bias.is_contiguous()), "bias must be contiguous (N,)"
    BLOCK_M, BLOCK_N, BLOCK_K, regions = tile_config(tile)
    assert M % BLOCK_M == 0 and N % BLOCK_N == 0, f"{M}x{N} is not a multiple of the {BLOCK_M}x{BLOCK_N} tile"
    assert K % BLOCK_K == 0 and K // BLOCK_K >= 4, f"K={K} needs at least 4 steps of {BLOCK_K}"
    num_warps = 4
    if c is None:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid_m, grid_n = triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N)
    GRID_MN = grid_m * grid_n
    NUM_XCDS = 8
    gm_env = os.environ.get("V9_TILE_GM")
    GROUP_SIZE_M = int(gm_env) if gm_env else group_size_m(BLOCK_M, BLOCK_N, grid_m, GRID_MN, NUM_XCDS)

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
        num_warps=num_warps,
    )
    return c
