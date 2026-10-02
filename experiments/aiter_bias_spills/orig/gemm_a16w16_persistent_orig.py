# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Persistent Gluon BF16/FP16 GEMMs for gfx950, compute-bound variants v10-v13.

Ported from ROCm/gfx950-gluon-tutorials, kernels/gemm/intra_wave/a16w16:

- v10_persistant: v9 (``gemm_a16w16.py``) turned into a persistent kernel. One program
  per CU loops over the output tiles; every tile still runs v9's prologue, K loop and
  epilogue.
- v11_persistant_overlap_global: each tile's epilogue issues the next tile's K-step 0-1
  global->LDS loads.
- v12_persistant_overlap_lds: the next tile's K-step 0 LDS reads also move into the
  previous tile's epilogue.
- v13_persistant_peel_acc: K-steps 0-1 of every tile are peeled out of the K loop with a
  constant zero accumulator, so no AGPRs are zeroed per tile.

As in the v9 port, the accumulators are pinned to AGPRs with ``cd_regclass="a"``, bias
and fp32 output are added, and the instruction schedule comes from the llirSched plugin.
The tile, masking and layout limits are v9's; see ``unsupported_reason``.
"""

import functools

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton import llir_sched
from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.gemm_a16w16 import (
    BLOCK_K,
    BLOCK_M,
    BLOCK_N,
    GROUP_SIZE_M,
    NUM_WARPS,
    NUM_XCDS,
    unsupported_reason,
)


def _make_layouts():
    """v9's layouts for the 256x256x64 tile, passed to the kernels as constexprs."""
    mfma = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 2]
    )
    return dict(
        # Half-M / half-N global load layouts.
        G_LOAD_A=gl.DistributedLinearLayout(
            reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0]],
            lane_bases=[[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [64, 0]],
            warp_bases=[[1, 0], [2, 0]],
            block_bases=[],
            shape=[BLOCK_M // 2, BLOCK_K],
        ),
        G_LOAD_B=gl.DistributedLinearLayout(
            reg_bases=[[1, 0], [2, 0], [4, 0], [0, 4], [0, 8]],
            lane_bases=[[8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 64]],
            warp_bases=[[0, 1], [0, 2]],
            block_bases=[],
            shape=[BLOCK_K, BLOCK_N // 2],
        ),
        # Half-M / half-N padded shared layouts.
        SHARED_A=gl.PaddedSharedLayout(
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
        ),
        SHARED_B=gl.PaddedSharedLayout(
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
        ),
        MFMA=mfma,
        DOT_A=gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=8),
        DOT_B=gl.DotOperandLayout(operand_index=1, parent=mfma, k_width=8),
        G_STORE_C=gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0]),
    )


@gluon.jit
def _get_group_m_tile_ids(
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
def _get_logical_chiplet_mapped_pids(
    NUM_PROGRAMS: gl.constexpr,
    NUM_XCDS: gl.constexpr,
):
    """XCD-aware PID remapping of the program id."""
    pid = gl.program_id(axis=0)
    if NUM_XCDS != 1:
        pids_per_xcd = (NUM_PROGRAMS + NUM_XCDS - 1) // NUM_XCDS
        tall_xcds = NUM_PROGRAMS % NUM_XCDS
        tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        if xcd < tall_xcds:
            pid = xcd * pids_per_xcd + local_pid
        else:
            pid = (
                tall_xcds * pids_per_xcd
                + (xcd - tall_xcds) * (pids_per_xcd - 1)
                + local_pid
            )
    return pid


@gluon.jit
def _xcd_remap_tiles(vpid, count, NUM_XCDS: gl.constexpr):
    """XCD remap over `count` tiles: XCD x owns a contiguous 1/NUM_XCDS of them."""
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
def _persistent_tile_id(
    vpid, total_tiles, NUM_XCDS: gl.constexpr, TILE_ORDER_V9: gl.constexpr
):
    """Tile id (before the GROUP_SIZE_M swizzle) for virtual pid `vpid`.

    Split order: the XCD remap was applied once to the program id, so vpid is the tile id.
    v9 order: vpid is the pid v9 would have launched, remapped over all tiles like v9, so
    every XCD processes exactly v9's tiles.
    """
    if TILE_ORDER_V9:
        tile_id = _xcd_remap_tiles(vpid, total_tiles, NUM_XCDS)
    else:
        tile_id = vpid
    return tile_id


@gluon.jit
def _first_vpid(
    NUM_PROGRAMS: gl.constexpr, NUM_XCDS: gl.constexpr, TILE_ORDER_V9: gl.constexpr
):
    """Programs step through virtual pids start, start + NUM_PROGRAMS, ..."""
    if TILE_ORDER_V9:
        start = gl.program_id(axis=0)
    else:
        start = _get_logical_chiplet_mapped_pids(NUM_PROGRAMS, NUM_XCDS)
    return start


@gluon.jit
def _tile_bases(
    a_ptr,
    b_ptr,
    tile_id,
    num_tiles_m,
    num_tiles_n,
    stride_am,
    stride_bn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
):
    """64-bit A and B base pointers of a tile, and its (pid_m, pid_n)."""
    pid_m, pid_n = _get_group_m_tile_ids(
        tile_id, num_tiles_m, num_tiles_n, GROUP_SIZE_M
    )
    a_base = a_ptr + pid_m.to(gl.int64) * BLOCK_M * stride_am
    b_base = b_ptr + pid_n.to(gl.int64) * BLOCK_N * stride_bn
    return a_base, b_base, pid_m, pid_n


@gluon.jit
def _load_k_pair(
    smemA_top,
    smemA_bot,
    smemB_left,
    smemB_right,
    a_base,
    b_base,
    a_offsets,
    b_offsets,
    a_offsets_next,
    b_offsets_next,
    a_half,
    b_half,
):
    """Async-copy K-steps 0 (buffer 0) and 1 (buffer 1) at a_base / b_base: 8 groups."""
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_left.index(0), b_base, b_offsets
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_top.index(0), a_base, a_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_bot.index(0), a_base + a_half, a_offsets
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_right.index(0), b_base + b_half, b_offsets
    )
    gl.amd.cdna4.async_copy.commit_group()

    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_left.index(1), b_base, b_offsets_next
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_top.index(1), a_base, a_offsets_next
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_bot.index(1), a_base + a_half, a_offsets_next
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_right.index(1), b_base + b_half, b_offsets_next
    )
    gl.amd.cdna4.async_copy.commit_group()


@gluon.jit
def _k_step_pair(
    acc_tl,
    acc_bl,
    acc_tr,
    acc_br,
    a_top,
    b_left,
    smemA_top,
    smemA_bot,
    smemB_left,
    smemB_right,
    a_base,
    b_base,
    a_offsets,
    b_offsets,
    a_half,
    b_half,
    a_offsets_next,
    b_offsets_next,
    DOT_A: gl.constexpr,
    DOT_B: gl.constexpr,
):
    """One unrolled main-loop iteration (two K-steps), same as v8/v9.

    Consumes a_top / b_left (already in registers) plus buffers 0 and 1, prefetches the
    two K-steps at a_base / b_base into the same buffers, and returns the next a_top /
    b_left. Passing constant zeros as the accumulators turns the first four MFMAs into
    zero-C MFMAs (v13).
    """
    ## Sub-iteration 0: consume buffer 0, prefetch into buffer 0.
    acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    a_bot = smemA_bot.index(0).load(DOT_A)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_left.index(0), b_base, b_offsets
    )
    gl.amd.cdna4.async_copy.commit_group()

    acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    b_right = smemB_right.index(0).load(DOT_B)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_top.index(0), a_base, a_offsets)
    gl.amd.cdna4.async_copy.commit_group()

    acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    b_left = smemB_left.index(1).load(DOT_B)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_bot.index(0), a_base + a_half, a_offsets
    )
    gl.amd.cdna4.async_copy.commit_group()

    acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    a_top = smemA_top.index(1).load(DOT_A)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_right.index(0), b_base + b_half, b_offsets
    )
    gl.amd.cdna4.async_copy.commit_group()

    ## Sub-iteration 1: consume buffer 1, prefetch into buffer 1 (odd K-step offsets).
    acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    a_bot = smemA_bot.index(1).load(DOT_A)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_left.index(1), b_base, b_offsets_next
    )
    gl.amd.cdna4.async_copy.commit_group()

    acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    b_right = smemB_right.index(1).load(DOT_B)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_top.index(1), a_base, a_offsets_next
    )
    gl.amd.cdna4.async_copy.commit_group()

    acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    b_left = smemB_left.index(0).load(DOT_B)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_bot.index(1), a_base + a_half, a_offsets_next
    )
    gl.amd.cdna4.async_copy.commit_group()

    acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    a_top = smemA_top.index(0).load(DOT_A)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_right.index(1), b_base + b_half, b_offsets_next
    )
    gl.amd.cdna4.async_copy.commit_group()

    return acc_tl, acc_bl, acc_tr, acc_br, a_top, b_left


@gluon.jit
def _load_bias(
    bias_ptr, pid_n, BLOCK_N: gl.constexpr, MFMA: gl.constexpr, ADD_BIAS: gl.constexpr
):
    """fp32 bias for the tile's left and right N halves (0.0 without bias)."""
    if ADD_BIAS:
        offs_bias = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, MFMA))
        bias_l = gl.load(bias_ptr + pid_n * BLOCK_N + offs_bias).to(gl.float32)
        bias_r = gl.load(bias_ptr + pid_n * BLOCK_N + BLOCK_N // 2 + offs_bias).to(
            gl.float32
        )
    else:
        bias_l = 0.0
        bias_r = 0.0
    return bias_l, bias_r


@gluon.jit
def _store_quadrant(
    acc,
    bias,
    c_base,
    c_offsets,
    ADD_BIAS: gl.constexpr,
    G_STORE_C: gl.constexpr,
):
    """Add the bias in fp32, downcast to C's dtype and store one 128x128 quadrant."""
    if ADD_BIAS:
        acc = acc + bias[None, :]
    c = acc.to(c_base.dtype.element_ty)
    c = gl.convert_layout(c, layout=G_STORE_C)
    gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_offsets, stored_value=c)


@gluon.jit
def _gemm_a16w16_v10_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K: gl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    TILE_ORDER_V9: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    G_LOAD_A: gl.constexpr,
    G_LOAD_B: gl.constexpr,
    SHARED_A: gl.constexpr,
    SHARED_B: gl.constexpr,
    MFMA: gl.constexpr,
    DOT_A: gl.constexpr,
    DOT_B: gl.constexpr,
    G_STORE_C: gl.constexpr,
):
    """
    v9 turned into a persistent kernel: NUM_PROGRAMS programs loop over the output tiles
    (tile_id += NUM_PROGRAMS). Every tile still runs v9's own prologue, K loop and
    epilogue; nothing overlaps across tiles yet.
    """
    start = _first_vpid(NUM_PROGRAMS, NUM_XCDS, TILE_ORDER_V9)
    num_tiles_m = gl.cdiv(M, BLOCK_M)
    num_tiles_n = gl.cdiv(N, BLOCK_N)
    total_tiles = num_tiles_m * num_tiles_n
    # The 2x-unrolled loop and the 2-step epilogue always leave buffer 0 as the one to
    # read next, which only holds for an even number of full K-steps.
    gl.static_assert(K % (2 * BLOCK_K) == 0, "K must be a multiple of 2 * BLOCK_K")

    nBuffers: gl.constexpr = 2
    smemA_top = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [nBuffers, BLOCK_M // 2, BLOCK_K], SHARED_A
    )
    smemA_bot = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [nBuffers, BLOCK_M // 2, BLOCK_K], SHARED_A
    )
    smemB_left = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, BLOCK_N // 2], SHARED_B
    )
    smemB_right = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, BLOCK_N // 2], SHARED_B
    )

    offs_am = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, G_LOAD_A))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, G_LOAD_A))
    offs_bn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, G_LOAD_B))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, G_LOAD_B))

    for vpid in range(start, total_tiles, NUM_PROGRAMS):
        tile_id = _persistent_tile_id(vpid, total_tiles, NUM_XCDS, TILE_ORDER_V9)
        a_base, b_base, pid_m, pid_n = _tile_bases(
            a_ptr,
            b_ptr,
            tile_id,
            num_tiles_m,
            num_tiles_n,
            stride_am,
            stride_bn,
            BLOCK_M,
            BLOCK_N,
            GROUP_SIZE_M,
        )

        # Two offset tensors per operand: even and odd K-steps. The half-tile (bot/right)
        # shift is uniform, so it goes in the scalar base pointer.
        a_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
        b_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        a_half = BLOCK_M // 2 * stride_am
        b_half = BLOCK_N // 2 * stride_bn
        a_offsets_next = a_offsets + BLOCK_K * stride_ak
        b_offsets_next = b_offsets + BLOCK_K * stride_bk

        acc_tl = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, MFMA)
        acc_bl = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, MFMA)
        acc_tr = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, MFMA)
        acc_br = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, MFMA)

        iterMax = gl.cdiv(K, BLOCK_K)

        ## Prologue, same as v9.
        _load_k_pair(
            smemA_top,
            smemA_bot,
            smemB_left,
            smemB_right,
            a_base,
            b_base,
            a_offsets,
            b_offsets,
            a_offsets_next,
            b_offsets_next,
            a_half,
            b_half,
        )
        a_base += BLOCK_K * stride_ak * 2
        b_base += BLOCK_K * stride_bk * 2

        gl.amd.cdna4.async_copy.wait_group(6)
        b_left = smemB_left.index(0).load(DOT_B)
        a_top = smemA_top.index(0).load(DOT_A)

        gl.assume(iterMax > 3)

        ## Main loop, same as v9.
        for k in range(0, iterMax - 2, 2):
            acc_tl, acc_bl, acc_tr, acc_br, a_top, b_left = _k_step_pair(
                acc_tl,
                acc_bl,
                acc_tr,
                acc_br,
                a_top,
                b_left,
                smemA_top,
                smemA_bot,
                smemB_left,
                smemB_right,
                a_base,
                b_base,
                a_offsets,
                b_offsets,
                a_half,
                b_half,
                a_offsets_next,
                b_offsets_next,
                DOT_A,
                DOT_B,
            )
            a_base += BLOCK_K * stride_ak * 2
            b_base += BLOCK_K * stride_bk * 2

        ## Epilogue: 4-quadrant stores with natural-pipeline ordering, same as v9.
        offs_cm = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, G_STORE_C))
        offs_cn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, G_STORE_C))
        c_base = c_ptr + pid_m.to(gl.int64) * BLOCK_M * stride_cm
        col = pid_n * BLOCK_N + offs_cn
        c_tl_offsets = stride_cm * offs_cm[:, None] + stride_cn * col[None, :]
        c_tr_offsets = c_tl_offsets + BLOCK_N * stride_cn // 2
        c_bl_offsets = c_tl_offsets + BLOCK_M * stride_cm // 2
        c_br_offsets = c_bl_offsets + BLOCK_N * stride_cn // 2

        # Loaded here so its latency overlaps the last MFMAs.
        bias_l, bias_r = _load_bias(bias_ptr, pid_n, BLOCK_N, MFMA, ADD_BIAS)

        ## Iter iterMax - 2: same 4-region pattern as the main loop, no async copies.
        acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")
        gl.amd.cdna4.async_copy.wait_group(5)
        l_idx = (iterMax - 2) % 2
        a_bot = smemA_bot.index(l_idx).load(DOT_A)

        acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")
        gl.amd.cdna4.async_copy.wait_group(4)
        b_right = smemB_right.index(l_idx).load(DOT_B)

        acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")
        gl.amd.cdna4.async_copy.wait_group(3)
        g_idx = 1 - l_idx
        b_left = smemB_left.index(g_idx).load(DOT_B)

        acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")
        gl.amd.cdna4.async_copy.wait_group(2)
        a_top = smemA_top.index(g_idx).load(DOT_A)

        ## Iter iterMax - 1: each store follows its MFMA with one MFMA of gap.
        acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")
        gl.amd.cdna4.async_copy.wait_group(1)
        a_bot = smemA_bot.index(g_idx).load(DOT_A)

        acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")
        gl.amd.cdna4.async_copy.wait_group(0)
        b_right = smemB_right.index(g_idx).load(DOT_B)

        _store_quadrant(acc_tl, bias_l, c_base, c_tl_offsets, ADD_BIAS, G_STORE_C)

        acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")

        _store_quadrant(acc_bl, bias_l, c_base, c_bl_offsets, ADD_BIAS, G_STORE_C)

        acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")

        _store_quadrant(acc_tr, bias_r, c_base, c_tr_offsets, ADD_BIAS, G_STORE_C)
        _store_quadrant(acc_br, bias_r, c_base, c_br_offsets, ADD_BIAS, G_STORE_C)


@gluon.jit
def _gemm_a16w16_v11_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K: gl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    TILE_ORDER_V9: gl.constexpr,
    MASK_TAIL_PREFETCH: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    G_LOAD_A: gl.constexpr,
    G_LOAD_B: gl.constexpr,
    SHARED_A: gl.constexpr,
    SHARED_B: gl.constexpr,
    MFMA: gl.constexpr,
    DOT_A: gl.constexpr,
    DOT_B: gl.constexpr,
    G_STORE_C: gl.constexpr,
):
    """
    v10 plus cross-tile global prefetch: each tile's epilogue issues the next tile's
    K-step 0-1 global->LDS loads into the buffers the last two K-steps free, so the next
    tile starts with its first loads already in flight. Only the first tile loads its own
    K-steps 0-1, before the tile loop. The last trip's prefetch is masked off.
    """
    start = _first_vpid(NUM_PROGRAMS, NUM_XCDS, TILE_ORDER_V9)
    num_tiles_m = gl.cdiv(M, BLOCK_M)
    num_tiles_n = gl.cdiv(N, BLOCK_N)
    total_tiles = num_tiles_m * num_tiles_n

    nBuffers: gl.constexpr = 2
    smemA_top = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [nBuffers, BLOCK_M // 2, BLOCK_K], SHARED_A
    )
    smemA_bot = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [nBuffers, BLOCK_M // 2, BLOCK_K], SHARED_A
    )
    smemB_left = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, BLOCK_N // 2], SHARED_B
    )
    smemB_right = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, BLOCK_N // 2], SHARED_B
    )

    offs_am = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, G_LOAD_A))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, G_LOAD_A))
    offs_bn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, G_LOAD_B))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, G_LOAD_B))

    a_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    b_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    a_half = BLOCK_M // 2 * stride_am
    b_half = BLOCK_N // 2 * stride_bn
    a_offsets_next = a_offsets + BLOCK_K * stride_ak
    b_offsets_next = b_offsets + BLOCK_K * stride_bk

    gl.static_assert(K % (2 * BLOCK_K) == 0, "K must be a multiple of 2 * BLOCK_K")
    iterMax = gl.cdiv(K, BLOCK_K)
    gl.assume(iterMax > 3)

    ## Prologue, first tile only: every later tile's first two K-steps are prefetched by
    ## the previous tile's epilogue. Programs with no tile still load a valid one, then
    ## skip the loop.
    first_id = _persistent_tile_id(
        min(start, total_tiles - 1), total_tiles, NUM_XCDS, TILE_ORDER_V9
    )
    a_first, b_first, _, _ = _tile_bases(
        a_ptr,
        b_ptr,
        first_id,
        num_tiles_m,
        num_tiles_n,
        stride_am,
        stride_bn,
        BLOCK_M,
        BLOCK_N,
        GROUP_SIZE_M,
    )
    _load_k_pair(
        smemA_top,
        smemA_bot,
        smemB_left,
        smemB_right,
        a_first,
        b_first,
        a_offsets,
        b_offsets,
        a_offsets_next,
        b_offsets_next,
        a_half,
        b_half,
    )

    for vpid in range(start, total_tiles, NUM_PROGRAMS):
        tile_id = _persistent_tile_id(vpid, total_tiles, NUM_XCDS, TILE_ORDER_V9)
        a_base, b_base, pid_m, pid_n = _tile_bases(
            a_ptr,
            b_ptr,
            tile_id,
            num_tiles_m,
            num_tiles_n,
            stride_am,
            stride_bn,
            BLOCK_M,
            BLOCK_N,
            GROUP_SIZE_M,
        )
        # K-steps 0 and 1 are already in flight, so the main loop starts at K-step 2.
        a_base += BLOCK_K * stride_ak * 2
        b_base += BLOCK_K * stride_bk * 2

        # On the last trip there is no next tile: prefetch from a valid (clamped) tile so
        # every trip issues the same instructions and wait counts. MASK_TAIL_PREFETCH
        # masks those loads off.
        next_id = _persistent_tile_id(
            min(vpid + NUM_PROGRAMS, total_tiles - 1),
            total_tiles,
            NUM_XCDS,
            TILE_ORDER_V9,
        )
        has_next = (vpid + NUM_PROGRAMS < total_tiles) if MASK_TAIL_PREFETCH else None
        a_next, b_next, _, _ = _tile_bases(
            a_ptr,
            b_ptr,
            next_id,
            num_tiles_m,
            num_tiles_n,
            stride_am,
            stride_bn,
            BLOCK_M,
            BLOCK_N,
            GROUP_SIZE_M,
        )

        acc_tl = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, MFMA)
        acc_bl = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, MFMA)
        acc_tr = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, MFMA)
        acc_br = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, MFMA)

        gl.amd.cdna4.async_copy.wait_group(6)
        b_left = smemB_left.index(0).load(DOT_B)
        a_top = smemA_top.index(0).load(DOT_A)

        ## Main loop, same as v9.
        for k in range(0, iterMax - 2, 2):
            acc_tl, acc_bl, acc_tr, acc_br, a_top, b_left = _k_step_pair(
                acc_tl,
                acc_bl,
                acc_tr,
                acc_br,
                a_top,
                b_left,
                smemA_top,
                smemA_bot,
                smemB_left,
                smemB_right,
                a_base,
                b_base,
                a_offsets,
                b_offsets,
                a_half,
                b_half,
                a_offsets_next,
                b_offsets_next,
                DOT_A,
                DOT_B,
            )
            a_base += BLOCK_K * stride_ak * 2
            b_base += BLOCK_K * stride_bk * 2

        ## Epilogue. The tile's row goes in the 64-bit scalar base; its column stays in the
        ## vector offsets, which keeps them tile-dependent so the compiler can't hoist them
        ## above the tile loop and keep them live through the K loop.
        offs_cm = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, G_STORE_C))
        offs_cn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, G_STORE_C))
        c_base = c_ptr + pid_m.to(gl.int64) * BLOCK_M * stride_cm
        col = pid_n * BLOCK_N + offs_cn
        c_tl_offsets = stride_cm * offs_cm[:, None] + stride_cn * col[None, :]
        c_tr_offsets = c_tl_offsets + BLOCK_N * stride_cn // 2
        c_bl_offsets = c_tl_offsets + BLOCK_M * stride_cm // 2
        c_br_offsets = c_bl_offsets + BLOCK_N * stride_cn // 2

        bias_l, bias_r = _load_bias(bias_ptr, pid_n, BLOCK_N, MFMA, ADD_BIAS)

        ## Iter iterMax - 2: each async copy prefetches the next tile's K-step 0 into the
        ## buffer the main loop would refill here, so every wait keeps 5 groups in flight.
        acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")
        gl.amd.cdna4.async_copy.wait_group(5)
        l_idx = (iterMax - 2) % 2
        a_bot = smemA_bot.index(l_idx).load(DOT_A)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemB_left.index(0), b_next, b_offsets, mask=has_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")
        gl.amd.cdna4.async_copy.wait_group(5)
        b_right = smemB_right.index(l_idx).load(DOT_B)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemA_top.index(0), a_next, a_offsets, mask=has_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")
        gl.amd.cdna4.async_copy.wait_group(5)
        g_idx = 1 - l_idx
        b_left = smemB_left.index(g_idx).load(DOT_B)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemA_bot.index(0), a_next + a_half, a_offsets, mask=has_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")
        gl.amd.cdna4.async_copy.wait_group(5)
        a_top = smemA_top.index(g_idx).load(DOT_A)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemB_right.index(0), b_next + b_half, b_offsets, mask=has_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        ## Iter iterMax - 1: prefetch the next tile's K-step 1; each store follows its
        ## MFMA with one MFMA of gap.
        acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")
        gl.amd.cdna4.async_copy.wait_group(5)
        a_bot = smemA_bot.index(g_idx).load(DOT_A)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemB_left.index(1), b_next, b_offsets_next, mask=has_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")
        gl.amd.cdna4.async_copy.wait_group(5)
        b_right = smemB_right.index(g_idx).load(DOT_B)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemA_top.index(1), a_next, a_offsets_next, mask=has_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        _store_quadrant(acc_tl, bias_l, c_base, c_tl_offsets, ADD_BIAS, G_STORE_C)

        acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemA_bot.index(1), a_next + a_half, a_offsets_next, mask=has_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        _store_quadrant(acc_bl, bias_l, c_base, c_bl_offsets, ADD_BIAS, G_STORE_C)

        acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemB_right.index(1), b_next + b_half, b_offsets_next, mask=has_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        _store_quadrant(acc_tr, bias_r, c_base, c_tr_offsets, ADD_BIAS, G_STORE_C)
        _store_quadrant(acc_br, bias_r, c_base, c_br_offsets, ADD_BIAS, G_STORE_C)

    # Drain the last trip's redundant prefetch before the wave exits.
    gl.amd.cdna4.async_copy.wait_group(0)


@gluon.jit
def _epilogue_overlap_lds(
    acc_tl,
    acc_bl,
    acc_tr,
    acc_br,
    a_top,
    b_left,
    smemA_top,
    smemA_bot,
    smemB_left,
    smemB_right,
    a_next,
    b_next,
    a_offsets,
    b_offsets,
    a_offsets_next,
    b_offsets_next,
    a_half,
    b_half,
    has_next,
    c_ptr,
    bias_ptr,
    pid_m,
    pid_n,
    stride_cm,
    stride_cn,
    iterMax,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    MFMA: gl.constexpr,
    DOT_A: gl.constexpr,
    DOT_B: gl.constexpr,
    G_STORE_C: gl.constexpr,
):
    """v12/v13 tile epilogue: the last two K-steps, the next tile's K-step 0-1 prefetch
    and its K-step 0 LDS reads, and the four C stores. Returns the next a_top / b_left.

    The four C quadrants share one vector offset tensor, with the quadrant shift in the
    scalar base, to keep the extra live operands spill-free.
    """
    offs_cm = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, G_STORE_C))
    offs_cn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, G_STORE_C))
    c_tl_base = c_ptr + pid_m.to(gl.int64) * BLOCK_M * stride_cm
    col = pid_n * BLOCK_N + offs_cn
    c_offsets = stride_cm * offs_cm[:, None] + stride_cn * col[None, :]
    c_tr_base = c_tl_base + BLOCK_N * stride_cn // 2
    c_bl_base = c_tl_base + BLOCK_M * stride_cm // 2
    c_br_base = c_bl_base + BLOCK_N * stride_cn // 2

    bias_l, bias_r = _load_bias(bias_ptr, pid_n, BLOCK_N, MFMA, ADD_BIAS)

    ## Iter iterMax - 2: each async copy prefetches the next tile's K-step 0 into the
    ## buffer the main loop would refill here, so every wait keeps 5 groups in flight.
    acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    l_idx = (iterMax - 2) % 2
    a_bot = smemA_bot.index(l_idx).load(DOT_A)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_left.index(0), b_next, b_offsets, mask=has_next
    )
    gl.amd.cdna4.async_copy.commit_group()

    acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    b_right = smemB_right.index(l_idx).load(DOT_B)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_top.index(0), a_next, a_offsets, mask=has_next
    )
    gl.amd.cdna4.async_copy.commit_group()

    acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    g_idx = 1 - l_idx
    b_left = smemB_left.index(g_idx).load(DOT_B)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_bot.index(0), a_next + a_half, a_offsets, mask=has_next
    )
    gl.amd.cdna4.async_copy.commit_group()

    acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    a_top = smemA_top.index(g_idx).load(DOT_A)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_right.index(0), b_next + b_half, b_offsets, mask=has_next
    )
    gl.amd.cdna4.async_copy.commit_group()

    ## Iter iterMax - 1: prefetch the next tile's K-step 1; each store follows its MFMA
    ## with one MFMA of gap.
    acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    a_bot = smemA_bot.index(g_idx).load(DOT_A)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_left.index(1), b_next, b_offsets_next, mask=has_next
    )
    gl.amd.cdna4.async_copy.commit_group()

    acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    b_right = smemB_right.index(g_idx).load(DOT_B)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_top.index(1), a_next, a_offsets_next, mask=has_next
    )
    gl.amd.cdna4.async_copy.commit_group()

    _store_quadrant(acc_tl, bias_l, c_tl_base, c_offsets, ADD_BIAS, G_STORE_C)

    acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_bot.index(1), a_next + a_half, a_offsets_next, mask=has_next
    )
    gl.amd.cdna4.async_copy.commit_group()

    _store_quadrant(acc_bl, bias_l, c_bl_base, c_offsets, ADD_BIAS, G_STORE_C)

    acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")

    # Next tile's K-step 0 operands (prefetched in iter iterMax - 2), read while the last
    # MFMAs and stores drain; this tile's operand registers are all dead after acc_br.
    gl.amd.cdna4.async_copy.wait_group(5)
    b_left = smemB_left.index(0).load(DOT_B)
    a_top = smemA_top.index(0).load(DOT_A)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_right.index(1), b_next + b_half, b_offsets_next, mask=has_next
    )
    gl.amd.cdna4.async_copy.commit_group()

    _store_quadrant(acc_tr, bias_r, c_tr_base, c_offsets, ADD_BIAS, G_STORE_C)
    _store_quadrant(acc_br, bias_r, c_br_base, c_offsets, ADD_BIAS, G_STORE_C)

    return a_top, b_left


@gluon.jit
def _gemm_a16w16_v12_v13_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K: gl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    TILE_ORDER_V9: gl.constexpr,
    MASK_TAIL_PREFETCH: gl.constexpr,
    PEEL_ACC: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    G_LOAD_A: gl.constexpr,
    G_LOAD_B: gl.constexpr,
    SHARED_A: gl.constexpr,
    SHARED_B: gl.constexpr,
    MFMA: gl.constexpr,
    DOT_A: gl.constexpr,
    DOT_B: gl.constexpr,
    G_STORE_C: gl.constexpr,
):
    """
    v12 (PEEL_ACC=False): v11 plus cross-tile LDS reads. The next tile's K-step 0 LDS
    reads (b_left, a_top) move from the top of the tile loop into the previous tile's
    epilogue, so their latency overlaps the last MFMAs and C stores.

    v13 (PEEL_ACC=True): v12 plus a peeled first iteration. K-steps 0-1 of every tile run
    outside the K loop with a constant zero accumulator, so the first MFMA of each output
    block uses an inline-0 C operand and no AGPRs are zeroed per tile.
    """
    start = _first_vpid(NUM_PROGRAMS, NUM_XCDS, TILE_ORDER_V9)
    num_tiles_m = gl.cdiv(M, BLOCK_M)
    num_tiles_n = gl.cdiv(N, BLOCK_N)
    total_tiles = num_tiles_m * num_tiles_n

    nBuffers: gl.constexpr = 2
    smemA_top = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [nBuffers, BLOCK_M // 2, BLOCK_K], SHARED_A
    )
    smemA_bot = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [nBuffers, BLOCK_M // 2, BLOCK_K], SHARED_A
    )
    smemB_left = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, BLOCK_N // 2], SHARED_B
    )
    smemB_right = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, BLOCK_N // 2], SHARED_B
    )

    offs_am = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, G_LOAD_A))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, G_LOAD_A))
    offs_bn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, G_LOAD_B))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, G_LOAD_B))

    a_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    b_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    a_half = BLOCK_M // 2 * stride_am
    b_half = BLOCK_N // 2 * stride_bn
    a_offsets_next = a_offsets + BLOCK_K * stride_ak
    b_offsets_next = b_offsets + BLOCK_K * stride_bk

    gl.static_assert(K % (2 * BLOCK_K) == 0, "K must be a multiple of 2 * BLOCK_K")
    iterMax = gl.cdiv(K, BLOCK_K)
    gl.assume(iterMax > 3)

    ## Prologue, first tile only (see v11).
    first_id = _persistent_tile_id(
        min(start, total_tiles - 1), total_tiles, NUM_XCDS, TILE_ORDER_V9
    )
    a_first, b_first, _, _ = _tile_bases(
        a_ptr,
        b_ptr,
        first_id,
        num_tiles_m,
        num_tiles_n,
        stride_am,
        stride_bn,
        BLOCK_M,
        BLOCK_N,
        GROUP_SIZE_M,
    )
    _load_k_pair(
        smemA_top,
        smemA_bot,
        smemB_left,
        smemB_right,
        a_first,
        b_first,
        a_offsets,
        b_offsets,
        a_offsets_next,
        b_offsets_next,
        a_half,
        b_half,
    )

    # First tile's K-step 0 operands; later tiles get them from the previous epilogue.
    gl.amd.cdna4.async_copy.wait_group(6)
    b_left = smemB_left.index(0).load(DOT_B)
    a_top = smemA_top.index(0).load(DOT_A)

    for vpid in range(start, total_tiles, NUM_PROGRAMS):
        tile_id = _persistent_tile_id(vpid, total_tiles, NUM_XCDS, TILE_ORDER_V9)
        a_base, b_base, pid_m, pid_n = _tile_bases(
            a_ptr,
            b_ptr,
            tile_id,
            num_tiles_m,
            num_tiles_n,
            stride_am,
            stride_bn,
            BLOCK_M,
            BLOCK_N,
            GROUP_SIZE_M,
        )
        # K-steps 0 and 1 are already in flight, so the first prefetch is K-step 2.
        a_base += BLOCK_K * stride_ak * 2
        b_base += BLOCK_K * stride_bk * 2

        next_id = _persistent_tile_id(
            min(vpid + NUM_PROGRAMS, total_tiles - 1),
            total_tiles,
            NUM_XCDS,
            TILE_ORDER_V9,
        )
        has_next = (vpid + NUM_PROGRAMS < total_tiles) if MASK_TAIL_PREFETCH else None
        a_next, b_next, _, _ = _tile_bases(
            a_ptr,
            b_ptr,
            next_id,
            num_tiles_m,
            num_tiles_n,
            stride_am,
            stride_bn,
            BLOCK_M,
            BLOCK_N,
            GROUP_SIZE_M,
        )

        if PEEL_ACC:
            ## K-steps 0 and 1, peeled: a constant-zero accumulator becomes the MFMA's
            ## inline-0 C operand. A loop-carried zero would be materialized.
            zero = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, MFMA)
            acc_tl, acc_bl, acc_tr, acc_br, a_top, b_left = _k_step_pair(
                zero,
                zero,
                zero,
                zero,
                a_top,
                b_left,
                smemA_top,
                smemA_bot,
                smemB_left,
                smemB_right,
                a_base,
                b_base,
                a_offsets,
                b_offsets,
                a_half,
                b_half,
                a_offsets_next,
                b_offsets_next,
                DOT_A,
                DOT_B,
            )
            a_base += BLOCK_K * stride_ak * 2
            b_base += BLOCK_K * stride_bk * 2
            k_first = 2
        else:
            acc_tl = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, MFMA)
            acc_bl = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, MFMA)
            acc_tr = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, MFMA)
            acc_br = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, MFMA)
            k_first = 0

        ## Main loop, same as v9.
        for k in range(k_first, iterMax - 2, 2):
            acc_tl, acc_bl, acc_tr, acc_br, a_top, b_left = _k_step_pair(
                acc_tl,
                acc_bl,
                acc_tr,
                acc_br,
                a_top,
                b_left,
                smemA_top,
                smemA_bot,
                smemB_left,
                smemB_right,
                a_base,
                b_base,
                a_offsets,
                b_offsets,
                a_half,
                b_half,
                a_offsets_next,
                b_offsets_next,
                DOT_A,
                DOT_B,
            )
            a_base += BLOCK_K * stride_ak * 2
            b_base += BLOCK_K * stride_bk * 2

        a_top, b_left = _epilogue_overlap_lds(
            acc_tl,
            acc_bl,
            acc_tr,
            acc_br,
            a_top,
            b_left,
            smemA_top,
            smemA_bot,
            smemB_left,
            smemB_right,
            a_next,
            b_next,
            a_offsets,
            b_offsets,
            a_offsets_next,
            b_offsets_next,
            a_half,
            b_half,
            has_next,
            c_ptr,
            bias_ptr,
            pid_m,
            pid_n,
            stride_cm,
            stride_cn,
            iterMax,
            BLOCK_M,
            BLOCK_N,
            ADD_BIAS,
            MFMA,
            DOT_A,
            DOT_B,
            G_STORE_C,
        )

    # Drain the last trip's redundant prefetch before the wave exits.
    gl.amd.cdna4.async_copy.wait_group(0)


@functools.cache
def _layouts():
    return _make_layouts()


@functools.cache
def _num_cus(device_index):
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _offsets_fit_reason(x, b, y, N):
    """Buffer offsets are 32-bit bytes. Tile positions, half-tile shifts and the K advance
    live in the 64-bit scalar bases, so the per-lane offsets only span one tile (plus C's
    width)."""

    def fits(*spans, t):
        return sum(spans) * t.element_size() < 2**31 - 2

    if not fits(BLOCK_M * x.stride(0), 2 * BLOCK_K * x.stride(1), t=x):
        return "x strides too large for 32-bit buffer offsets"
    if not fits(2 * BLOCK_K * b.stride(0), BLOCK_N * b.stride(1), t=b):
        return "w strides too large for 32-bit buffer offsets"
    if not fits(BLOCK_M * y.stride(0), N * y.stride(1), t=y):
        return "y strides too large for 32-bit buffer offsets"
    return None


def _launch(name, kernel, x, w, y, bias, **variant):
    """y = x @ w.T (+ bias) with x (M, K) and w (N, K), both row-major; y (M, N)."""
    M, K = x.shape
    N, _ = w.shape
    b = w.T  # (K, N) view, K contiguous: the layout the kernel loads
    reason = unsupported_reason(M, N, K, x, w) or _offsets_fit_reason(x, b, y, N)
    if reason is not None:
        raise ValueError(f"gfx950 gluon {name} a16w16: {reason}")
    device = (
        x.device.index if x.device.index is not None else torch.cuda.current_device()
    )
    # One program per CU, but no idle programs: v11-v13 would still issue a first-tile
    # prefetch from each of them.
    num_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    num_programs = min(_num_cus(device), num_tiles)
    kernel[(num_programs, 1)](
        x,
        b,
        y,
        bias,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        b.stride(0),
        b.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        NUM_PROGRAMS=num_programs,
        NUM_XCDS=NUM_XCDS,
        GROUP_SIZE_M=GROUP_SIZE_M,
        TILE_ORDER_V9=True,
        ADD_BIAS=bias is not None,
        num_warps=NUM_WARPS,
        **variant,
        **_layouts(),
        **llir_sched.compile_options(),
    )
    return y


def gemm_a16w16_v10(x, w, y, bias=None):
    return _launch("compute_bound_v10", _gemm_a16w16_v10_kernel, x, w, y, bias)


def gemm_a16w16_v11(x, w, y, bias=None):
    return _launch(
        "compute_bound_v11",
        _gemm_a16w16_v11_kernel,
        x,
        w,
        y,
        bias,
        MASK_TAIL_PREFETCH=True,
    )


def gemm_a16w16_v12(x, w, y, bias=None):
    # The masked tail prefetch costs about 12 VGPRs, which pushes v12/v13 into spills,
    # so the last trip loads the clamped tile unmasked.
    return _launch(
        "compute_bound_v12",
        _gemm_a16w16_v12_v13_kernel,
        x,
        w,
        y,
        bias,
        MASK_TAIL_PREFETCH=False,
        PEEL_ACC=False,
    )


def gemm_a16w16_v13(x, w, y, bias=None):
    return _launch(
        "compute_bound_v13",
        _gemm_a16w16_v12_v13_kernel,
        x,
        w,
        y,
        bias,
        MASK_TAIL_PREFETCH=False,
        PEEL_ACC=True,
    )


_PERSISTENT_KERNEL_MAP = {
    "compute_bound_v10": gemm_a16w16_v10,
    "compute_bound_v11": gemm_a16w16_v11,
    "compute_bound_v12": gemm_a16w16_v12,
    "compute_bound_v13": gemm_a16w16_v13,
}
