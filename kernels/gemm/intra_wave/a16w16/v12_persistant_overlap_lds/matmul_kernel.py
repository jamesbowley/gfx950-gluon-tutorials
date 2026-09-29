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
# from common import get_pids
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

@gluon.jit
def get_group_m_tile_ids(
    tile_ids,
    num_tiles_m,
    num_tiles_n,
    GROUP_SIZE_M: gl.constexpr,
):
    """GROUP_SIZE_M swizzle."""

    if GROUP_SIZE_M == 1:
        tiles_m = tile_ids // num_tiles_n
        tiles_n = tile_ids % num_tiles_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_tiles_n
        group_id = tile_ids // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_tiles_m - first_pid_m, GROUP_SIZE_M)
        tiles_m = first_pid_m + ((tile_ids % num_pid_in_group) % group_size_m)
        tiles_n = (tile_ids % num_pid_in_group) // group_size_m

    return tiles_m, tiles_n

@gluon.jit
def get_logical_chiplet_mapped_pids(
    NUM_PROGRAMS: gl.constexpr,
    NUM_XCDS: gl.constexpr,
):
    """XCD-aware PID remapping."""
    pid = gl.program_id(axis=0)

    if NUM_XCDS != 1:
        ## pid remapping on xcds
        # Number of pids per XCD in the new arrangement
        pids_per_xcd = (NUM_PROGRAMS + NUM_XCDS - 1) // NUM_XCDS
        # When GRID_MN cannot divide NUM_XCDS, some xcds will have
        # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
        # We calculate the number of xcds that have pids_per_xcd pids as
        # tall_xcds
        tall_xcds = NUM_PROGRAMS % NUM_XCDS
        tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
        # Compute current XCD and local pid within the XCD
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        # Calculate new pid based on the new grouping
        if xcd < tall_xcds:
            pid = xcd * pids_per_xcd + local_pid
        else:
            pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid
    return pid


@gluon.jit
def xcd_remap_tiles(vpid, count, NUM_XCDS: gl.constexpr):
    """XCD remap over `count` tiles: XCD x owns a contiguous 1/NUM_XCDS of them (v9's get_pids)."""
    per_xcd = (count + NUM_XCDS - 1) // NUM_XCDS
    tall = count % NUM_XCDS
    tall = gl.where(tall == 0, NUM_XCDS, tall)
    xcd = vpid % NUM_XCDS
    local = vpid // NUM_XCDS
    return gl.where(
        xcd < tall,
        xcd * per_xcd + local,
        tall * per_xcd + (xcd - tall) * (per_xcd - 1) + local,
    )


@gluon.jit
def persistent_tile_id(vpid, total_tiles, NUM_XCDS: gl.constexpr, TILE_ORDER_V9: gl.constexpr):
    """Tile id (before the GROUP_SIZE_M swizzle) for virtual pid `vpid` = start + i * NUM_PROGRAMS.

    Split order: the XCD remap was applied once to the program id, so vpid already is the tile id.
    v9 order: vpid is the pid v9 would have launched, remapped over all tiles like v9's get_pids,
    so every XCD processes exactly v9's tiles.
    """
    if TILE_ORDER_V9:
        tile_id = xcd_remap_tiles(vpid, total_tiles, NUM_XCDS)
    else:
        tile_id = vpid
    return tile_id



@gluon.jit
def v12_persistant_overlap_lds(
    a_ptr,
    b_ptr,
    c_ptr,
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
    NUM_PROGRAMS: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,  #
    TILE_ORDER_V9: gl.constexpr,
    MASK_TAIL_PREFETCH: gl.constexpr,
):
    """
    v11_persistant_overlap_global plus cross-tile LDS reads: the next tile's K-step 0 LDS
    reads (b_left, a_top) move from the top of the tile loop into the previous tile's
    epilogue, so their latency overlaps the last MFMAs and C stores. The four C quadrants
    share one vector offset tensor, with the quadrant shift in the scalar base, to keep the
    extra live operands spill-free.
    """

    # Programs step through virtual pids start, start + NUM_PROGRAMS, ...; persistent_tile_id
    # turns each into a tile id in either the split or the v9 order.
    if TILE_ORDER_V9:
        start = gl.program_id(axis=0)
    else:
        start = get_logical_chiplet_mapped_pids(NUM_PROGRAMS, NUM_XCDS)
    num_tiles_m = gl.cdiv(M, BLOCK_M)
    num_tiles_n = gl.cdiv(N, BLOCK_N)
    total_tiles = num_tiles_m * num_tiles_n
    # pid_m, pid_n = get_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

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

    # Two offset tensors per operand: even and odd K-steps. The half-tile (bot/right)
    # shift is uniform, so it goes in the scalar base pointer instead of 4 more offset
    # tensors held in VGPRs for the whole kernel.
    a_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    b_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    a_half = BLOCK_M // 2 * stride_am
    b_half = BLOCK_N // 2 * stride_bn
    a_offsets_next = a_offsets + BLOCK_K * stride_ak
    b_offsets_next = b_offsets + BLOCK_K * stride_bk

    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 2]
    )

    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfmaLayout, k_width=8)
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfmaLayout, k_width=8)

    # The 2x-unrolled loop and the 2-step epilogue always leave buffer 0 as the one to read
    # next, which only holds for an even number of full K-steps.
    gl.static_assert(K % (2 * BLOCK_K) == 0, "K must be a multiple of 2 * BLOCK_K")
    iterMax = gl.cdiv(K, BLOCK_K)
    gl.assume(iterMax > 3)

    ## Prologue, first tile only: every later tile's first two K-steps are prefetched by the
    ## previous tile's epilogue. Programs with no tile still load a valid one, then skip the loop.
    first_id = persistent_tile_id(min(start, total_tiles - 1), total_tiles, NUM_XCDS, TILE_ORDER_V9)
    tile_m, tile_n = get_group_m_tile_ids(first_id, num_tiles_m, num_tiles_n, GROUP_SIZE_M)
    a_first = a_ptr + tile_m.to(gl.int64) * BLOCK_M * stride_am
    b_first = b_ptr + tile_n.to(gl.int64) * BLOCK_N * stride_bn

    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(0), b_first, b_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_top.index(0), a_first, a_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_bot.index(0), a_first + a_half, a_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_right.index(0), b_first + b_half, b_offsets)
    gl.amd.cdna4.async_copy.commit_group()

    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(1), b_first, b_offsets_next)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_top.index(1), a_first, a_offsets_next)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_bot.index(1), a_first + a_half, a_offsets_next
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_right.index(1), b_first + b_half, b_offsets_next
    )
    gl.amd.cdna4.async_copy.commit_group()

    # First tile's K-step 0 operands; later tiles get them from the previous tile's epilogue.
    gl.amd.cdna4.async_copy.wait_group(6)
    b_left = smemB_left.index(0).load(dotOpLayoutB)
    a_top = smemA_top.index(0).load(dotOpLayoutA)

    for vpid in range(start, total_tiles, NUM_PROGRAMS):
        tile_id = persistent_tile_id(vpid, total_tiles, NUM_XCDS, TILE_ORDER_V9)

        tile_m, tile_n = get_group_m_tile_ids(tile_id, num_tiles_m, num_tiles_n, GROUP_SIZE_M)

        pid_m = tile_m
        pid_n = tile_n

        # On the last trip there is no next tile: prefetch from a valid (clamped) tile so every
        # trip issues the same instructions and wait counts. MASK_TAIL_PREFETCH masks those
        # loads off; that costs a per-lane select per offset VGPR, so it is optional.
        next_id = persistent_tile_id(
            min(vpid + NUM_PROGRAMS, total_tiles - 1), total_tiles, NUM_XCDS, TILE_ORDER_V9
        )
        has_next = (vpid + NUM_PROGRAMS < total_tiles) if MASK_TAIL_PREFETCH else None
        next_m, next_n = get_group_m_tile_ids(next_id, num_tiles_m, num_tiles_n, GROUP_SIZE_M)
        a_next = a_ptr + next_m.to(gl.int64) * BLOCK_M * stride_am
        b_next = b_ptr + next_n.to(gl.int64) * BLOCK_N * stride_bn

        # K-steps 0 and 1 are already in flight, so the main loop starts at K-step 2.
        a_base = a_ptr + pid_m.to(gl.int64) * BLOCK_M * stride_am + BLOCK_K * stride_ak * 2
        b_base = b_ptr + pid_n.to(gl.int64) * BLOCK_N * stride_bn + BLOCK_K * stride_bk * 2

        acc_tl = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfmaLayout)
        acc_bl = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfmaLayout)
        acc_tr = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfmaLayout)
        acc_br = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfmaLayout)

        ## Main loop — same as v8
        for k in range(0, iterMax - 2, 2):

            ## =============================================================
            ## Sub-iteration 0: consume buffer 0, prefetch into buffer 0
            ## =============================================================

            ########################################
            ## Region 0: C_tl = DOT(a_top, b_left)
            ########################################
            acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl)

            gl.amd.cdna4.async_copy.wait_group(5)
            a_bot = smemA_bot.index(0).load(dotOpLayoutA)

            gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(0), b_base, b_offsets)
            gl.amd.cdna4.async_copy.commit_group()

            ########################################
            ## Region 1: C_bl = DOT(a_bot, b_left)
            ########################################
            acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl)

            gl.amd.cdna4.async_copy.wait_group(5)
            b_right = smemB_right.index(0).load(dotOpLayoutB)

            gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_top.index(0), a_base, a_offsets)
            gl.amd.cdna4.async_copy.commit_group()

            ########################################
            ## Region 2: C_tr = DOT(a_top, b_right)
            ########################################
            acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr)

            gl.amd.cdna4.async_copy.wait_group(5)
            b_left = smemB_left.index(1).load(dotOpLayoutB)

            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                smemA_bot.index(0), a_base + a_half, a_offsets
            )
            gl.amd.cdna4.async_copy.commit_group()

            ########################################
            ## Region 3: C_br = DOT(a_bot, b_right)
            ########################################
            acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br)

            gl.amd.cdna4.async_copy.wait_group(5)
            a_top = smemA_top.index(1).load(dotOpLayoutA)

            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                smemB_right.index(0), b_base + b_half, b_offsets
            )
            gl.amd.cdna4.async_copy.commit_group()

            ## =============================================================
            ## Loop unroll: Sub-iteration 1: consume buffer 1, prefetch
            ## into buffer 1. AC uses _next offsets (odd K-step).
            ## =============================================================

            ########################################
            ## Region 0: C_tl = DOT(a_top, b_left)
            ########################################
            acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl)

            gl.amd.cdna4.async_copy.wait_group(5)
            a_bot = smemA_bot.index(1).load(dotOpLayoutA)

            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                smemB_left.index(1), b_base, b_offsets_next
            )
            gl.amd.cdna4.async_copy.commit_group()

            ########################################
            ## Region 1: C_bl = DOT(a_bot, b_left)
            ########################################
            acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl)

            gl.amd.cdna4.async_copy.wait_group(5)
            b_right = smemB_right.index(1).load(dotOpLayoutB)

            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                smemA_top.index(1), a_base, a_offsets_next
            )
            gl.amd.cdna4.async_copy.commit_group()

            ########################################
            ## Region 2: C_tr = DOT(a_top, b_right)
            ########################################
            acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr)

            gl.amd.cdna4.async_copy.wait_group(5)
            b_left = smemB_left.index(0).load(dotOpLayoutB)

            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                smemA_bot.index(1), a_base + a_half, a_offsets_next
            )
            gl.amd.cdna4.async_copy.commit_group()

            ########################################
            ## Region 3: C_br = DOT(a_bot, b_right)
            ########################################
            acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br)

            gl.amd.cdna4.async_copy.wait_group(5)
            a_top = smemA_top.index(0).load(dotOpLayoutA)

            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                smemB_right.index(1), b_base + b_half, b_offsets_next
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
        # The tile's row goes in the 64-bit scalar base, so the 32-bit byte offsets only span
        # BLOCK_M rows. Its column stays in the vector offsets, which keeps them tile-dependent:
        # the compiler can't hoist them above the tile loop and keep them live (and spilled)
        # through the K loop. The quadrant shifts go in the scalar base too, so all four stores
        # share one offset tensor rather than holding four in VGPRs through the epilogue.
        c_tl_base = c_ptr + pid_m.to(gl.int64) * BLOCK_M * stride_cm
        col = pid_n * BLOCK_N + offs_cn
        c_offsets = stride_cm * offs_cm[:, None] + stride_cn * col[None, :]
        c_tr_base = c_tl_base + BLOCK_N * stride_cn // 2
        c_bl_base = c_tl_base + BLOCK_M * stride_cm // 2
        c_br_base = c_bl_base + BLOCK_N * stride_cn // 2

        ## Iter iterMax - 2: same 4-region pattern as main loop. Each async copy prefetches the
        ## next tile's K-step 0 into the buffer the main loop would refill here, so every wait
        ## keeps the main loop's count of 5 groups in flight.
        acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl)
        gl.amd.cdna4.async_copy.wait_group(5)
        l_idx = (iterMax - 2) % 2
        a_bot = smemA_bot.index(l_idx).load(dotOpLayoutA)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemB_left.index(0), b_next, b_offsets, mask=has_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl)
        gl.amd.cdna4.async_copy.wait_group(5)
        b_right = smemB_right.index(l_idx).load(dotOpLayoutB)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemA_top.index(0), a_next, a_offsets, mask=has_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr)
        gl.amd.cdna4.async_copy.wait_group(5)
        g_idx = 1 - l_idx
        b_left = smemB_left.index(g_idx).load(dotOpLayoutB)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemA_bot.index(0), a_next + a_half, a_offsets, mask=has_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br)
        gl.amd.cdna4.async_copy.wait_group(5)
        a_top = smemA_top.index(g_idx).load(dotOpLayoutA)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemB_right.index(0), b_next + b_half, b_offsets, mask=has_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        ## Iter iterMax - 1: prefetch the next tile's K-step 1.
        ## Natural-pipeline epilogue: each store follows its MFMA with one
        ## MFMA cycle of gap, yielding uniform MFMA-store interleaving.
        acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl)
        gl.amd.cdna4.async_copy.wait_group(5)
        a_bot = smemA_bot.index(g_idx).load(dotOpLayoutA)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemB_left.index(1), b_next, b_offsets_next, mask=has_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl)
        gl.amd.cdna4.async_copy.wait_group(5)
        b_right = smemB_right.index(g_idx).load(dotOpLayoutB)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemA_top.index(1), a_next, a_offsets_next, mask=has_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        c_tl = acc_tl.to(a_ptr.dtype.element_ty)
        c_tl = gl.convert_layout(c_tl, layout=gStoreLayoutC)
        gl.amd.cdna3.buffer_store(ptr=c_tl_base, offsets=c_offsets, stored_value=c_tl)

        acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemA_bot.index(1), a_next + a_half, a_offsets_next, mask=has_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        c_bl = acc_bl.to(a_ptr.dtype.element_ty)
        c_bl = gl.convert_layout(c_bl, layout=gStoreLayoutC)
        gl.amd.cdna3.buffer_store(ptr=c_bl_base, offsets=c_offsets, stored_value=c_bl)

        acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br)

        # Next tile's K-step 0 operands (prefetched in iter iterMax - 2), read while the last
        # MFMAs and stores drain; this tile's operand registers are all dead after acc_br.
        gl.amd.cdna4.async_copy.wait_group(5)
        b_left = smemB_left.index(0).load(dotOpLayoutB)
        a_top = smemA_top.index(0).load(dotOpLayoutA)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemB_right.index(1), b_next + b_half, b_offsets_next, mask=has_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        c_tr = acc_tr.to(a_ptr.dtype.element_ty)
        c_tr = gl.convert_layout(c_tr, layout=gStoreLayoutC)
        gl.amd.cdna3.buffer_store(ptr=c_tr_base, offsets=c_offsets, stored_value=c_tr)

        c_br = acc_br.to(a_ptr.dtype.element_ty)
        c_br = gl.convert_layout(c_br, layout=gStoreLayoutC)
        gl.amd.cdna3.buffer_store(ptr=c_br_base, offsets=c_offsets, stored_value=c_br)

    # Drain the last trip's redundant prefetch before the wave exits.
    gl.amd.cdna4.async_copy.wait_group(0)


def matmul(a, b, c=None):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 64
    num_warps = 4
    if c is None:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # Buffer offsets are 32-bit bytes. Tile positions, half-tile shifts and the K advance live
    # in the 64-bit scalar bases, so the per-lane offsets only span one tile (plus C's width).
    def fits(*spans, t):
        return sum(spans) * t.element_size() < 2**31 - 2

    assert fits(BLOCK_M * a.stride(0), 2 * BLOCK_K * a.stride(1), t=a), "A strides too large"
    assert fits(2 * BLOCK_K * b.stride(0), BLOCK_N * b.stride(1), t=b), "B strides too large"
    assert fits(BLOCK_M * c.stride(0), N * c.stride(1), t=c), "C strides too large"
    # GRID_MN means the same thing as NUM_PROGRAMS, change when using persistant.
    GRID_MN = 256# triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (GRID_MN, 1)
    NUM_XCDS = 8
    GROUP_SIZE_M = 4
    v12_persistant_overlap_lds[grid](
        a,
        b,
        c,  #
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
        NUM_PROGRAMS=GRID_MN,
        NUM_XCDS=NUM_XCDS,
        GROUP_SIZE_M=GROUP_SIZE_M,
        # Tiles are walked in v9's order; PERSISTENT_TILE_ORDER=split selects the split order.
        TILE_ORDER_V9=os.environ.get("PERSISTENT_TILE_ORDER", "v9") == "v9",
        # The masked prefetch costs about 12 VGPRs, which here pushes the kernel into spills,
        # so the last trip loads the clamped tile unmasked.
        MASK_TAIL_PREFETCH=False,
        num_warps=num_warps,
        # force-agpr RA hint: reserve 256 AGPRs for MFMA accumulators, enabled by
        # TRITON_FORCE_MFMA_AGPR (paired in llvm.cc with amdgpu-mfma-vgpr-form=0).
        llvm_fn_attrs=("amdgpu-agpr-alloc=256" if os.environ.get("TRITON_FORCE_MFMA_AGPR") else ""),
    )
    return c
