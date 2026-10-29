# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon BF16/FP16 GEMM for gfx950, compute-bound variant with a Stream-K tail.

Ported from ROCm/gfx950-gluon-tutorials (ef48b5f),
kernels/gemm/intra_wave/a16w16/v9_beyond_hotloop: one wave per SIMD (4 warps), a
256x256x64 tile split into four 128x128 quadrants, a 2-deep async-copy LDS pipeline
with local prefetch, the K loop unrolled by 2, MFMA accumulators pinned to AGPRs with
``cd_regclass="a"``, and an XCD-aware PID remap. Its instruction schedule comes from
Triton's MFMA scheduler, enabled per launch with ``schedule_hint="mfma-schedule"``;
``AITER_MFMA_SCHED=0`` turns it off for A/B runs, which costs about 15%. The kernel
needs an upstream Triton newer than triton-lang/triton#12209 (``3b0c7f081``), which
brought the scheduler; ``mfma(..., cd_regclass=)`` (triton-lang/triton#11792) was
already there. The tutorial explains every step:
https://github.com/ROCm/gfx950-gluon-tutorials/tree/main/kernels/gemm/intra_wave/a16w16

The tile is any (BLOCK_M, BLOCK_N) with sides 64, 128 or 256, with BLOCK_K 64 or 128;
every layout is derived from it. Which tile and split a shape runs with comes from a
complete AITER configuration dictionary that the caller must supply; the
launcher derives only shape- and topology-dependent launch
constants. Each program has four waves, with one resident program per queried CU.
Ragged shapes are specialized at compile time: a ragged M or N clamps the loads of the
last tile to the matrix edge and masks its stores; a ragged K (EVEN_K false) is
virtually padded to a whole number of K-step pairs (K_PAD). Each tile runs its last
pair first (on a ragged K the masked physical tail), then all remaining pairs read
from aligned column zero, so the hot loop stays unmasked. No input tensor is
physically padded to K_PAD. The
kernel needs a TN problem (x row-major, w row-major (N, K)) with leading strides that
are multiples of 8 (16-byte loads and stores) and K a multiple of 16.

Stream-K. Let ``P`` be the resident program count, ``T`` the output tile count, and
``S=T%P`` the tail tiles. The first tiles run in a persistent loop; the tail tiles are
split in units of a K-pair (two BLOCK_K steps). A tile-aligned split is uniform and
always uses exactly ``S * split_n`` programs. Two-tile and spread use the full resident
grid. For ``P=8`` and eight K-pairs per tile, each bracket below is one program, each
letter is one K-pair from that tile, and ``|`` marks a tile boundary within a program::

    one-tile (S=2, split_n=4): [AA][AA][AA][AA] [BB][BB][BB][BB]
    spread   (S=5), pid 0..3: [AAAAA][AAA|BB][BBBBB][B|CCCC]
                    pid 4..7: [CCCC|D][DDDDD][DD|EEE][EEEEE]

DP assigns each tail tile whole to one program. Two-tile uses spread's end-to-end
assignment over the final ``P+S`` tiles. Two-tile and spread process their ranges in
reverse tile order.

Required Stream-K controls, with representative values:

* ``policy="dp"`` keeps whole tiles, ``"one_tile"`` makes aligned per-tile
  splits, ``"two_tile"`` distributes the final ``P+S`` tiles end to end, and
  ``"spread"`` distributes the ``S>P/2`` tail tiles end to end.
* ``split_n=4`` assigns four programs to each tail tile (``one_tile`` only).
* ``fixup="owner"`` gathers one-tile partials on one program; ``"rs"`` has the
  programs reduce-scatter output slices. Fixup does not change the K-pair
  assignment above.
* ``exchange="wt"`` uses write-through partials; ``"l2"`` keeps the exchange
  XCD-local (``one_tile`` with ``rs`` only) and requires a tile's programs to
  fit on one XCD.
* ``GROUP_SIZE_M=4`` sets the output-tile grouping directly.

The last persistent trip prefetches the program's first Stream-K pair. At
256x256x64 this raises the end-to-end builds' (two-tile, spread) VGPR spills from
7 to 17 on aligned shapes, but the spills stay out of the K loops, and no shape
timed is slower: 0.965-1.003 of the time with a discarded last prefetch, over
five tiles, with bias and ragged edges (results/last_prefetch*.md, WORKLOG
entry 9).

Tile order: the persistent tiles are the first total_tiles - S tile ids, with v9's
XCD remap and GROUP_SIZE_M swizzle applied over those ids alone, and the Stream-K
tiles are the last S ids. This keeps the Stream-K tail separate from the persistent
mapping, which a configurable XCD mapping can replace later. Against v9's order
(remap over all tiles, Stream-K tiles at the top of each XCD block) it is faster on
two-tile with one trip, but up to about 1% slower on some 256x256x64 shapes and more
on smaller tiles with LLC-cold inputs (WORKLOG entries 10, 11 and 13).
"""

import math
import os

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import cdna4 as ttgl_cdna4
from triton.runtime.jit import constexpr_function

from .hardware import streamk_topology
from .runtime import streamk_workspace
from .selection import StreamKConfig

NUM_WARPS = 4


@constexpr_function
def compute_gload_layout(shared_layout, load_contig, num_warps, threads_per_warp=64):
    """The coalesced global-load linear layout that matches a PaddedSharedLayout.

    Partitions the shared layout's offset bases into register, lane and warp bases so
    that each warp writes contiguous LDS offsets (what CoalesceAsyncCopy derives for a
    padded encoding). At 256x256x64 this reproduces the tutorial's hand-written
    layouts exactly; for the other tiles it follows BLOCK_M, BLOCK_N and BLOCK_K.
    """
    bases = [list(b) for b in shared_layout.offset_bases]
    rank = len(shared_layout.shape)
    n_reg = int(math.log2(load_contig))
    n_lane = int(math.log2(threads_per_warp))
    n_warp = int(math.log2(num_warps))
    reg = bases[:n_reg]
    lane = bases[n_reg : n_reg + n_lane]
    warp = bases[n_reg + n_lane : n_reg + n_lane + n_warp]
    while len(warp) < n_warp:  # zero-pad (broadcast) if we ran out of bases
        warp.append([0] * rank)
    reg += bases[n_reg + n_lane + n_warp :]
    return gl.DistributedLinearLayout(
        reg_bases=reg,
        lane_bases=lane,
        warp_bases=warp,
        block_bases=[],
        shape=list(shared_layout.shape),
    )


@gluon.jit
def even_split_start(i, TOTAL, PARTS: gl.constexpr):
    """First item of part i when TOTAL items are split over PARTS parts as evenly as
    possible: every part gets TOTAL // PARTS items, the first TOTAL % PARTS one more.
    Part i covers [even_split_start(i), even_split_start(i + 1))."""
    return i * (TOTAL // PARTS) + gl.minimum(i, TOTAL % PARTS)


@gluon.jit
def even_split_size(i, TOTAL, PARTS: gl.constexpr):
    """Items in part i of even_split_start's split."""
    return TOTAL // PARTS + (i < TOTAL % PARTS).to(gl.int32)


@gluon.jit
def round_robin_to_xcd_blocked(i, count, NUM_XCDS: gl.constexpr):
    """Id i of count, placed round-robin (it runs on XCD i % NUM_XCDS), renumbered so
    that each XCD's ids form one contiguous block of an even split of count."""
    return even_split_start(i % NUM_XCDS, count, NUM_XCDS) + i // NUM_XCDS


@gluon.jit
def _get_pids(
    pid,
    M,
    N,
    BM: gl.constexpr,
    BN: gl.constexpr,
    GRID_MN,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
):
    """XCD-aware PID remapping + GROUP_SIZE_M swizzle. Active at any grid_mn.

    pid is a virtual pid of the persistent loop (program id + i NUM_PROGRAMS); with
    NUM_XCDS 1 it is a tile id and only the swizzle applies."""
    num_pid_m = gl.cdiv(M, BM)
    num_pid_n = gl.cdiv(N, BN)

    if NUM_XCDS != 1:
        pid = round_robin_to_xcd_blocked(pid, GRID_MN, NUM_XCDS)

    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    return pid_m, pid_n


@gluon.jit
def _gemm_a16w16_compute_bound_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    p_ptr,
    locks_ptr,
    sync_ptr,
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
    ADD_BIAS: gl.constexpr,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
    EVEN_K: gl.constexpr,
    STREAMK_TILES: gl.constexpr,
    SPLIT_N: gl.constexpr,
    REDUCE_SCATTER: gl.constexpr,
    RS_NB: gl.constexpr,
    RS_U: gl.constexpr,
    RS_R: gl.constexpr,
    END_TO_END: gl.constexpr,
    L2: gl.constexpr = False,
    XCD_SPLIT: gl.constexpr = 1,
    HAS_FULL: gl.constexpr = True,
    GRID: gl.constexpr = 0,
    OWNER_QPB: gl.constexpr = 1,
):
    """
    Stream-K on top of v9_beyond_hotloop: persistent full tiles followed by a uniform
    or end-to-end Stream-K tail.

    Keeps v9's slice-both-M-and-N design (4 quadrant accumulators, unrolled by 2,
    pre-computed load offsets, XCD-aware PID remapping + GROUP_SIZE_M swizzling) and
    adds:

    1. A persistent loop over the first total_full_tiles = total_tiles - STREAMK_TILES
       tile ids, XCD-remapped over those ids alone: each tile's epilogue prefetches
       the first K-step pair of the program's next tile. The last trip's epilogue
       prefetches the first pair of the program's first Stream-K segment, which then
       skips its prologue (masked off when the program has no segment). Without a
       Stream-K tail it loads the last pair of tile total_full_tiles - 1, which is
       discarded.
    2. A Stream-K tail over the last STREAMK_TILES tile ids, split in units of a
       K-step pair, whose partials are combined by an owner or a reduce-scatter
       ending. Every program of the Stream-K phase must be resident at once: the
       endings wait on each other.
    """
    start = gl.program_id(axis=0)
    num_tiles_m = gl.cdiv(M, BLOCK_M)
    num_tiles_n = gl.cdiv(N, BLOCK_N)
    total_tiles = num_tiles_m * num_tiles_n
    total_full_tiles = total_tiles - STREAMK_TILES
    strides = (stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn)

    # Every layout follows the tile: the shared layouts come from the dot-operand
    # layouts and the global-load layouts from the shared layouts.
    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 2]
    )
    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfmaLayout, k_width=8
    )
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfmaLayout, k_width=8
    )
    sharedLayoutA: gl.constexpr = ttgl_cdna4.compute_efficient_padded_shared_layout(
        dotOpLayoutA, [BLOCK_M // 2, BLOCK_K], a_ptr.dtype.element_ty
    )
    sharedLayoutB: gl.constexpr = ttgl_cdna4.compute_efficient_padded_shared_layout(
        dotOpLayoutB, [BLOCK_K, BLOCK_N // 2], b_ptr.dtype.element_ty
    )
    gLoadLayoutA: gl.constexpr = compute_gload_layout(sharedLayoutA, 8, gl.num_warps())
    gLoadLayoutB: gl.constexpr = compute_gload_layout(sharedLayoutB, 8, gl.num_warps())

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
    smem = (smemA_top, smemA_bot, smemB_left, smemB_right)

    offs_am = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, gLoadLayoutA))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutA))

    offs_bn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gLoadLayoutB))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadLayoutB))

    # Row and column terms of the offsets. The row strides are multiples of 8
    # elements (the wrapper checks), which keeps every 8-element group of a row
    # 16-byte aligned; the hint says so, because Triton's argument specialization
    # only records strides that are multiples of 16.
    a_rows_top = gl.multiple_of(offs_am * stride_am, 8)
    b_cols_left = gl.multiple_of(offs_bn * stride_bn, 8)

    # One half-tile offset tensor per operand, shared by both halves: the bottom /
    # right half (a_bot_shift / b_right_shift) and the odd K-step (BLOCK_K *
    # stride_ak / stride_bk) are shifts in the scalar base; a ragged tile replaces
    # them with its clamped offsets (a_load_addressing). Offsets are relative to the
    # tile, whose base is 64-bit.
    a_offsets = a_rows_top[:, None] + offs_ak[None, :] * stride_ak
    b_offsets = offs_bk[:, None] * stride_bk + b_cols_left[None, :]

    a_bot_shift = BLOCK_M // 2 * stride_am
    b_right_shift = BLOCK_N // 2 * stride_bn
    kofs = (a_offsets, a_bot_shift, b_offsets, b_right_shift)

    # A ragged K is virtually padded to whole K-step pairs; every tile runs its last
    # pair first (on a ragged K partial and masked), so the other pairs stay aligned to
    # the row's start.
    if EVEN_K:
        gl.static_assert(K % (2 * BLOCK_K) == 0)
        K_PAD: gl.constexpr = K
    else:
        K_PAD: gl.constexpr = gl.cdiv(K, 2 * BLOCK_K) * (2 * BLOCK_K)
    K_RES: gl.constexpr = K_PAD - K
    iterMax = gl.cdiv(K_PAD, BLOCK_K)

    gl.assume(iterMax > 3)
    PAIRS_PER_TILE: gl.constexpr = K_PAD // (2 * BLOCK_K)
    SK_PREFETCH: gl.constexpr = HAS_FULL and STREAMK_TILES > 0

    # Persistent phase. Pair zero is already in flight on entry to each tile;
    # its epilogue prefetches pair zero of the next virtual tile.
    if HAS_FULL:
        # The last trip's prefetch target: the first pair of the program's first
        # Stream-K segment, as quadrant_segment's prologue loads it.
        sk_target = None
        if SK_PREFETCH:
            sk_has, sk_tile, sk_k0 = streamk_first_segment(
                total_full_tiles,
                STREAMK_TILES,
                PAIRS_PER_TILE,
                SPLIT_N,
                END_TO_END,
                L2,
                XCD_SPLIT,
                NUM_PROGRAMS,
                NUM_XCDS,
            )
            sk_m, sk_n = _get_pids(
                sk_tile, M, N, BLOCK_M, BLOCK_N, 0, 1, GROUP_SIZE_M
            )
            sk_col = gl.where(
                sk_k0 == 0, K_PAD - 2 * BLOCK_K, (sk_k0 - 2) * BLOCK_K
            ).to(gl.int64)
            sk_khi = gl.where(sk_k0 == 0, 2 * BLOCK_K - K_RES, 2 * BLOCK_K) * sk_has
            sk_target = (sk_m, sk_n, sk_col, sk_khi)
        first_m, first_n = _get_pids(
            start,
            M,
            N,
            BLOCK_M,
            BLOCK_N,
            total_full_tiles,
            NUM_XCDS,
            GROUP_SIZE_M,
        )
        a_first = a_ptr + first_m.to(gl.int64) * BLOCK_M * stride_am
        b_first = b_ptr + first_n.to(gl.int64) * BLOCK_N * stride_bn
        a_addr = a_load_addressing(
            first_m,
            a_offsets,
            a_bot_shift,
            M,
            stride_am,
            stride_ak,
            BLOCK_M,
            BLOCK_K,
            gLoadLayoutA,
            EVEN_M,
        )
        b_addr = b_load_addressing(
            first_n,
            b_offsets,
            b_right_shift,
            N,
            stride_bk,
            stride_bn,
            BLOCK_N,
            BLOCK_K,
            gLoadLayoutB,
            EVEN_N,
        )
        if EVEN_K:
            mask_a0, mask_b0, mask_a1, mask_b1 = None, None, None, None
        else:
            mask_a0, mask_b0, mask_a1, mask_b1 = tail_pair_masks(
                2 * BLOCK_K - K_RES,
                K_RES,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                gLoadLayoutA,
                gLoadLayoutB,
                True,
            )
        a_top, b_left = load_first_pair(
            a_first + (K_PAD - 2 * BLOCK_K) * stride_ak,
            b_first + (K_PAD - 2 * BLOCK_K) * stride_bk,
            smem,
            a_addr,
            b_addr,
            dotOpLayoutA,
            dotOpLayoutB,
            mask_a0,
            mask_b0,
            mask_a1,
            mask_b1,
        )
        tile_addr = (a_addr, b_addr)
        for vpid in range(start, total_full_tiles, NUM_PROGRAMS):
            a_top, b_left, tile_addr = persistent_tile(
                vpid,
                a_top,
                b_left,
                tile_addr,
                a_ptr,
                b_ptr,
                c_ptr,
                bias_ptr,
                smem,
                kofs,
                strides,
                M,
                N,
                K_PAD,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                NUM_PROGRAMS,
                NUM_XCDS,
                GROUP_SIZE_M,
                ADD_BIAS,
                mfmaLayout,
                gLoadLayoutA,
                gLoadLayoutB,
                EVEN_M,
                EVEN_N,
                EVEN_K,
                K_RES,
                total_full_tiles,
                sk_target,
                SK_PREFETCH,
            )

    # The Stream-K phase picks up the last trip's prefetch.
    if not SK_PREFETCH:
        gl.amd.cdna4.async_copy.wait_group(0)
    if not HAS_FULL:
        a_top = 0
        b_left = 0

    if END_TO_END and STREAMK_TILES > 0:
        # Two-tile and spread always use the full resident grid.
        SK_PAIRS: gl.constexpr = STREAMK_TILES * PAIRS_PER_TILE
        L: gl.constexpr = SK_PAIRS // NUM_PROGRAMS
        R: gl.constexpr = SK_PAIRS % NUM_PROGRAMS
        gl.static_assert(NUM_PROGRAMS <= SK_PAIRS)
        rank = streamk_rank(NUM_PROGRAMS, NUM_XCDS)
        sk_start = even_split_start(rank, SK_PAIRS, NUM_PROGRAMS)
        cur = even_split_start(rank + 1, SK_PAIRS, NUM_PROGRAMS)
        # Segments run last to first (STREAMK_VARIANTS.md, concept 6): each program
        # starts at k-step 0 of its last tile, so an XCD's programs stay in step and
        # share A / B in L2 (forward order drops two-tile's hits from 68% to 9%), and
        # it publishes its partial before the owned tile's wait.
        if SK_PREFETCH:
            # The first segment, whose first pair the last persistent trip loaded.
            t = (cur - 1) // PAIRS_PER_TILE
            tile_start = t * PAIRS_PER_TILE
            seg_start = gl.maximum(tile_start, sk_start)
            is_owner = (cur == tile_start + PAIRS_PER_TILE).to(gl.int32)
            n_peers = (rank - streamk_owner_of(tile_start, L, R)) * is_owner
            quadrant_segment(
                total_full_tiles + t,
                (seg_start - tile_start) * 2,
                cur - seg_start,
                1 - is_owner,
                n_peers,
                rank,
                0,
                0,
                0,
                a_ptr,
                b_ptr,
                c_ptr,
                bias_ptr,
                p_ptr,
                locks_ptr,
                sync_ptr,
                smem,
                kofs,
                strides,
                M,
                N,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                GROUP_SIZE_M,
                ADD_BIAS,
                -1,
                False,
                1,
                1,
                False,
                NUM_PROGRAMS,
                NUM_XCDS,
                mfmaLayout,
                gLoadLayoutA,
                gLoadLayoutB,
                EVEN_M,
                EVEN_N,
                EVEN_K,
                K_PAD,
                K_RES,
                True,
                OWNER_QPB,
                0,
                True,
                a_top,
                b_left,
            )
            cur = seg_start
        while cur > sk_start:
            t = (cur - 1) // PAIRS_PER_TILE
            tile_start = t * PAIRS_PER_TILE
            seg_start = gl.maximum(tile_start, sk_start)
            is_owner = (cur == tile_start + PAIRS_PER_TILE).to(gl.int32)
            n_peers = (rank - streamk_owner_of(tile_start, L, R)) * is_owner
            quadrant_segment(
                total_full_tiles + t,
                (seg_start - tile_start) * 2,
                cur - seg_start,
                1 - is_owner,
                n_peers,
                rank,
                0,
                0,
                0,
                a_ptr,
                b_ptr,
                c_ptr,
                bias_ptr,
                p_ptr,
                locks_ptr,
                sync_ptr,
                smem,
                kofs,
                strides,
                M,
                N,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                GROUP_SIZE_M,
                ADD_BIAS,
                -1,
                False,
                1,
                1,
                False,
                NUM_PROGRAMS,
                NUM_XCDS,
                mfmaLayout,
                gLoadLayoutA,
                gLoadLayoutB,
                EVEN_M,
                EVEN_N,
                EVEN_K,
                K_PAD,
                K_RES,
                True,
                OWNER_QPB,
                0,
            )
            cur = seg_start
    elif STREAMK_TILES > 0:
        # Uniform tile-aligned split: every tile has exactly SPLIT_N programs.
        WORKING: gl.constexpr = STREAMK_TILES * SPLIT_N
        NBLK: gl.constexpr = BLOCK_M * BLOCK_N // 1024
        gl.static_assert(SPLIT_N <= PAIRS_PER_TILE)
        gl.static_assert(RS_R > 0 and RS_R <= SPLIT_N)
        gl.static_assert(
            2 * RS_R > SPLIT_N or RS_R == NBLK,
            "RS_R does not match SPLIT_N",
        )
        gl.static_assert((RS_R & (RS_R - 1)) == 0)
        gl.static_assert(RS_NB == 1 or RS_NB * RS_R <= NBLK)
        if L2:
            gl.static_assert(GRID == 0)
            rank = streamk_rank(NUM_PROGRAMS, NUM_XCDS)
            t, j, n_tile = xcd_local_chunk_of(
                rank,
                STREAMK_TILES,
                NUM_PROGRAMS // NUM_XCDS,
                NUM_XCDS,
                XCD_SPLIT,
            )
            active = n_tile > 0
        else:
            if GRID > 0:
                rank = streamk_rank(GRID, NUM_XCDS)
            else:
                rank = streamk_rank(WORKING, NUM_XCDS, IDLE=True)
            active = rank < WORKING
            t = rank // SPLIT_N
            j = rank % SPLIT_N
            n_tile = SPLIT_N
        if active:
            is_partial = (j != 0).to(gl.int32)
            n_peers = (SPLIT_N - 1) * (1 - is_partial)
            OWNER_MAX_PEERS: gl.constexpr = SPLIT_N - 1 if SPLIT_N <= 8 else 0
            quadrant_segment(
                total_full_tiles + t,
                even_split_start(j, PAIRS_PER_TILE, SPLIT_N) * 2,
                even_split_size(j, PAIRS_PER_TILE, SPLIT_N),
                is_partial,
                n_peers,
                rank,
                j,
                n_tile,
                RS_R,
                a_ptr,
                b_ptr,
                c_ptr,
                bias_ptr,
                p_ptr,
                locks_ptr,
                sync_ptr,
                smem,
                kofs,
                strides,
                M,
                N,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                GROUP_SIZE_M,
                ADD_BIAS,
                1,
                REDUCE_SCATTER,
                RS_NB,
                RS_U,
                L2,
                NUM_PROGRAMS,
                NUM_XCDS,
                mfmaLayout,
                gLoadLayoutA,
                gLoadLayoutB,
                EVEN_M,
                EVEN_N,
                EVEN_K,
                K_PAD,
                K_RES,
                False,
                OWNER_QPB,
                OWNER_MAX_PEERS,
                SK_PREFETCH,
                a_top,
                b_left,
            )


# ---------------------------------------------------------------------------------
# v9's pipeline as functions, shared by the persistent tiles and the Stream-K
# segments.
#
# accs is (acc_tl, acc_bl, acc_tr, acc_br) and smem is (smemA_top, smemA_bot,
# smemB_left, smemB_right). An operand's load addressing is a_addr =
# (a_top_offsets, a_bot_offsets, a_bot_shift, a_odd_shift) and b_addr =
# (b_left_offsets, b_right_offsets, b_right_shift, b_odd_shift) (a_load_addressing /
# b_load_addressing): each half loads from base + shift + offsets, the shifts living
# in the scalar base.
# ---------------------------------------------------------------------------------


@gluon.jit
def clamped_a_offsets(
    m_last,
    stride_am,
    stride_ak,
    BLOCK_M: gl.constexpr,
    BLOCK_K: gl.constexpr,
    gLoadLayoutA: gl.constexpr,
):
    """A's top and bottom load offsets on a ragged tile, relative to the tile. Rows past
    m_last (the tile's last row inside the matrix) are clamped to it: the clamped rows
    load valid data (duplicates), compute rows nobody keeps, and the masked store drops
    them. Clamping keeps the buffer offsets inside one tile's range."""
    offs_am = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, gLoadLayoutA))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutA))
    rows_top = gl.minimum(offs_am, m_last)
    rows_bot = gl.minimum(offs_am + BLOCK_M // 2, m_last)
    a_rows_top = gl.multiple_of(rows_top * stride_am, 8)
    a_rows_bot = gl.multiple_of(rows_bot * stride_am, 8)
    a_top_offsets = a_rows_top[:, None] + offs_ak[None, :] * stride_ak
    a_bot_offsets = a_rows_bot[:, None] + offs_ak[None, :] * stride_ak
    return a_top_offsets, a_bot_offsets


@gluon.jit
def clamped_b_offsets(
    n_last,
    stride_bk,
    stride_bn,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    gLoadLayoutB: gl.constexpr,
):
    """B's left and right load offsets on a ragged tile, columns past n_last clamped
    (clamped_a_offsets)."""
    offs_bn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gLoadLayoutB))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadLayoutB))
    cols_left = gl.minimum(offs_bn, n_last)
    cols_right = gl.minimum(offs_bn + BLOCK_N // 2, n_last)
    b_cols_left = gl.multiple_of(cols_left * stride_bn, 8)
    b_cols_right = gl.multiple_of(cols_right * stride_bn, 8)
    b_left_offsets = offs_bk[:, None] * stride_bk + b_cols_left[None, :]
    b_right_offsets = offs_bk[:, None] * stride_bk + b_cols_right[None, :]
    return b_left_offsets, b_right_offsets


@gluon.jit
def a_load_addressing(
    m_idx,
    a_offsets,
    a_bot_shift,
    M,
    stride_am,
    stride_ak,
    BLOCK_M: gl.constexpr,
    BLOCK_K: gl.constexpr,
    gLoadLayoutA: gl.constexpr,
    EVEN_M: gl.constexpr,
):
    """Tile m_idx's a_addr = (top_offs, bot_offs, bot_shift, odd_shift): the bottom
    half loads from base + bot_shift + bot_offs and the odd K-step adds odd_shift.
    EVEN_M: both halves share the half-tile offsets a_offsets. Otherwise the tile's
    clamped offsets, which differ per lane, so bot_shift is 0."""
    if EVEN_M:
        top, bot, bot_shift = a_offsets, a_offsets, a_bot_shift
    else:
        top, bot = clamped_a_offsets(
            M - 1 - m_idx * BLOCK_M, stride_am, stride_ak, BLOCK_M, BLOCK_K, gLoadLayoutA
        )
        bot_shift = 0
    return top, bot, bot_shift, BLOCK_K * stride_ak


@gluon.jit
def b_load_addressing(
    n_idx,
    b_offsets,
    b_right_shift,
    N,
    stride_bk,
    stride_bn,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    gLoadLayoutB: gl.constexpr,
    EVEN_N: gl.constexpr,
):
    """Tile n_idx's b_addr = (left_offs, right_offs, right_shift, odd_shift)
    (a_load_addressing)."""
    if EVEN_N:
        left, right, right_shift = b_offsets, b_offsets, b_right_shift
    else:
        left, right = clamped_b_offsets(
            N - 1 - n_idx * BLOCK_N, stride_bk, stride_bn, BLOCK_N, BLOCK_K, gLoadLayoutB
        )
        right_shift = 0
    return left, right, right_shift, BLOCK_K * stride_bk


@gluon.jit
def k_tail_mask(
    k_hi,
    STEP: gl.constexpr,
    BLOCK_K: gl.constexpr,
    layout: gl.constexpr,
    K_FIRST: gl.constexpr,
):
    """Load mask of K-step STEP (0 or 1) of a half-tile in the last pair, whose first
    k_hi of 2 BLOCK_K K values are inside the matrix. K_FIRST: B's [BLOCK_K, 1], else
    A's [1, BLOCK_K]; the load broadcasts it over the half-tile."""
    if K_FIRST:
        k = gl.arange(0, BLOCK_K, gl.SliceLayout(1, layout))[:, None]
    else:
        k = gl.arange(0, BLOCK_K, gl.SliceLayout(0, layout))[None, :]
    return k + STEP * BLOCK_K < k_hi


@gluon.jit
def tail_pair_masks(
    k_hi,
    K_RES: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    gLoadLayoutA: gl.constexpr,
    gLoadLayoutB: gl.constexpr,
    TAIL: gl.constexpr = False,
):
    """(mask_a0, mask_b0, mask_a1, mask_b1) for load_first_pair of a pair whose first
    k_hi K values are in the matrix. K-step 1 of the last pair always reaches past K,
    K-step 0 only when K_RES > BLOCK_K; a mask that can never drop a lane is None.
    TAIL: k_hi is the last pair's 2 BLOCK_K - K_RES, so with K_RES >= BLOCK_K K-step 1
    drops every lane and its masks are constant."""
    mask_a0, mask_b0 = None, None
    if K_RES > BLOCK_K:
        mask_a0 = k_tail_mask(k_hi, 0, BLOCK_K, gLoadLayoutA, False)
        mask_b0 = k_tail_mask(k_hi, 0, BLOCK_K, gLoadLayoutB, True)
    if TAIL and K_RES >= BLOCK_K:
        mask_a1 = gl.full([BLOCK_M // 2, BLOCK_K], 0, gl.int1, gLoadLayoutA)
        mask_b1 = gl.full([BLOCK_K, BLOCK_N // 2], 0, gl.int1, gLoadLayoutB)
    else:
        mask_a1 = k_tail_mask(k_hi, 1, BLOCK_K, gLoadLayoutA, False)
        mask_b1 = k_tail_mask(k_hi, 1, BLOCK_K, gLoadLayoutB, True)
    return mask_a0, mask_b0, mask_a1, mask_b1


@gluon.jit
def load_first_pair(
    a_base,
    b_base,
    smem,
    a_addr,
    b_addr,
    dotOpLayoutA: gl.constexpr,
    dotOpLayoutB: gl.constexpr,
    mask_a0=None,
    mask_b0=None,
    mask_a1=None,
    mask_b1=None,
):
    """Prologue — same as v9: the two K-steps at a_base / b_base into buffers 0 and 1,
    and the first one's a_top / b_left. mask_*: load masks of A's and B's K-step 0 and
    1 (ragged K), None for unmasked."""
    smemA_top, smemA_bot, smemB_left, smemB_right = smem
    a_top_offsets, a_bot_offsets, a_bot_shift, a_odd_shift = a_addr
    b_left_offsets, b_right_offsets, b_right_shift, b_odd_shift = b_addr

    g_idx = 0
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_left.index(g_idx), b_base, b_left_offsets, mask=mask_b0
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_top.index(g_idx), a_base, a_top_offsets, mask=mask_a0
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_bot.index(g_idx), a_base + a_bot_shift, a_bot_offsets, mask=mask_a0
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_right.index(g_idx), b_base + b_right_shift, b_right_offsets, mask=mask_b0
    )
    gl.amd.cdna4.async_copy.commit_group()

    g_idx = 1
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_left.index(g_idx), b_base + b_odd_shift, b_left_offsets, mask=mask_b1
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_top.index(g_idx), a_base + a_odd_shift, a_top_offsets, mask=mask_a1
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_bot.index(g_idx),
        a_base + a_bot_shift + a_odd_shift,
        a_bot_offsets,
        mask=mask_a1,
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_right.index(g_idx),
        b_base + b_right_shift + b_odd_shift,
        b_right_offsets,
        mask=mask_b1,
    )
    gl.amd.cdna4.async_copy.commit_group()

    gl.amd.cdna4.async_copy.wait_group(6)
    b_left = smemB_left.index(0).load(dotOpLayoutB)
    a_top = smemA_top.index(0).load(dotOpLayoutA)
    return a_top, b_left


@gluon.jit
def k_step_pair(
    accs,
    a_top,
    b_left,
    smem,
    a_base,
    b_base,
    a_addr,
    b_addr,
    dotOpLayoutA: gl.constexpr,
    dotOpLayoutB: gl.constexpr,
    SKIP_ODD: gl.constexpr = False,
):
    """One iteration of v9's main loop, two K-steps: consumes a_top / b_left (in
    registers) and buffers 0 and 1, prefetches the two K-steps at a_base / b_base into
    the same buffers, and returns the accumulators and the next a_top / b_left. Every
    wait keeps 5 groups of async copies in flight. Constant-zero accumulators make the
    first four MFMAs zero-C MFMAs. SKIP_ODD: buffer 1 holds only K padding, so its
    MFMAs are dropped; its waits and the prefetches stay."""
    acc_tl, acc_bl, acc_tr, acc_br = accs
    smemA_top, smemA_bot, smemB_left, smemB_right = smem
    a_top_offsets, a_bot_offsets, a_bot_shift, a_odd_shift = a_addr
    b_left_offsets, b_right_offsets, b_right_shift, b_odd_shift = b_addr

    ## =============================================================
    ## Sub-iteration 0: consume buffer 0, prefetch into buffer 0
    ## =============================================================

    ########################################
    ## Region 0: C_tl = DOT(a_top, b_left)
    ########################################
    acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")

    gl.amd.cdna4.async_copy.wait_group(5)
    a_bot = smemA_bot.index(0).load(dotOpLayoutA)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_left.index(0), b_base, b_left_offsets
    )
    gl.amd.cdna4.async_copy.commit_group()

    ########################################
    ## Region 1: C_bl = DOT(a_bot, b_left)
    ########################################
    acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")

    gl.amd.cdna4.async_copy.wait_group(5)
    b_right = smemB_right.index(0).load(dotOpLayoutB)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_top.index(0), a_base, a_top_offsets
    )
    gl.amd.cdna4.async_copy.commit_group()

    ########################################
    ## Region 2: C_tr = DOT(a_top, b_right)
    ########################################
    acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")

    gl.amd.cdna4.async_copy.wait_group(5)
    b_left = smemB_left.index(1).load(dotOpLayoutB)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_bot.index(0), a_base + a_bot_shift, a_bot_offsets
    )
    gl.amd.cdna4.async_copy.commit_group()

    ########################################
    ## Region 3: C_br = DOT(a_bot, b_right)
    ########################################
    acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")

    gl.amd.cdna4.async_copy.wait_group(5)
    a_top = smemA_top.index(1).load(dotOpLayoutA)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_right.index(0), b_base + b_right_shift, b_right_offsets
    )
    gl.amd.cdna4.async_copy.commit_group()

    ## =============================================================
    ## Loop unroll: Sub-iteration 1: consume buffer 1, prefetch
    ## into buffer 1. The odd K-step's loads add a_odd_shift / b_odd_shift to the base.
    ## =============================================================

    ########################################
    ## Region 0: C_tl = DOT(a_top, b_left)
    ########################################
    if not SKIP_ODD:
        acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")

    gl.amd.cdna4.async_copy.wait_group(5)
    a_bot = smemA_bot.index(1).load(dotOpLayoutA)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_left.index(1), b_base + b_odd_shift, b_left_offsets
    )
    gl.amd.cdna4.async_copy.commit_group()

    ########################################
    ## Region 1: C_bl = DOT(a_bot, b_left)
    ########################################
    if not SKIP_ODD:
        acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")

    gl.amd.cdna4.async_copy.wait_group(5)
    b_right = smemB_right.index(1).load(dotOpLayoutB)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_top.index(1), a_base + a_odd_shift, a_top_offsets
    )
    gl.amd.cdna4.async_copy.commit_group()

    ########################################
    ## Region 2: C_tr = DOT(a_top, b_right)
    ########################################
    if not SKIP_ODD:
        acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")

    gl.amd.cdna4.async_copy.wait_group(5)
    b_left = smemB_left.index(0).load(dotOpLayoutB)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_bot.index(1), a_base + a_bot_shift + a_odd_shift, a_bot_offsets
    )
    gl.amd.cdna4.async_copy.commit_group()

    ########################################
    ## Region 3: C_br = DOT(a_bot, b_right)
    ########################################
    if not SKIP_ODD:
        acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")

    gl.amd.cdna4.async_copy.wait_group(5)
    a_top = smemA_top.index(0).load(dotOpLayoutA)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_right.index(1), b_base + b_right_shift + b_odd_shift, b_right_offsets
    )
    gl.amd.cdna4.async_copy.commit_group()

    return (acc_tl, acc_bl, acc_tr, acc_br), a_top, b_left


@gluon.jit
def init_acc(
    bias_ptr,
    pid_n,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    mfmaLayout: gl.constexpr,
    ADD_BIAS: gl.constexpr,
):
    """The four fp32 accumulators (tl, bl, tr, br) of a tile, starting from the tile's
    bias[N] (or zero).

    Adding the bias here rather than before the C stores keeps it out of the epilogue,
    where the stores already hold the register peak; the epilogue add spills. Without
    bias the accumulators are a constant zero, which a peeled first MFMA folds into an
    inline-0 C operand.
    """
    acc_l = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfmaLayout)
    acc_r = acc_l
    if ADD_BIAS:
        # Scalar base + 32-bit lane offsets: no per-lane 64-bit addresses.
        offs_bias = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, mfmaLayout))
        bias_base = bias_ptr + pid_n * BLOCK_N
        bias_l = gl.amd.cdna3.buffer_load(ptr=bias_base, offsets=offs_bias)
        bias_r = gl.amd.cdna3.buffer_load(
            ptr=bias_base + BLOCK_N // 2, offsets=offs_bias
        )
        acc_l = acc_l + bias_l.to(gl.float32)[None, :]
        acc_r = acc_r + bias_r.to(gl.float32)[None, :]
    return acc_l, acc_l, acc_r, acc_r


@gluon.jit
def init_acc_edge(
    bias_ptr,
    pid_n,
    n_left,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    mfmaLayout: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    EVEN_N: gl.constexpr,
):
    """init_acc, with the bias loaded only for the tile's first n_left columns on a
    ragged N."""
    if ADD_BIAS and not EVEN_N:
        acc_l = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfmaLayout)
        offs_bias = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, mfmaLayout))
        bias_base = bias_ptr + pid_n * BLOCK_N
        bias_l = gl.amd.cdna3.buffer_load(
            ptr=bias_base, offsets=offs_bias, mask=offs_bias < n_left, other=0.0
        )
        bias_r = gl.amd.cdna3.buffer_load(
            ptr=bias_base + BLOCK_N // 2,
            offsets=offs_bias,
            mask=offs_bias < n_left - BLOCK_N // 2,
            other=0.0,
        )
        acc_r = acc_l + bias_r.to(gl.float32)[None, :]
        acc_l = acc_l + bias_l.to(gl.float32)[None, :]
        tl, bl, tr, br = acc_l, acc_l, acc_r, acc_r
    else:
        tl, bl, tr, br = init_acc(
            bias_ptr, pid_n, BLOCK_M, BLOCK_N, mfmaLayout, ADD_BIAS
        )
    return tl, bl, tr, br


@gluon.jit
def persistent_tile(
    vpid,
    a_top,
    b_left,
    tile_addr,
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    smem,
    kofs,
    strides,
    M,
    N,
    K: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    mfmaLayout: gl.constexpr,
    gLoadLayoutA: gl.constexpr,
    gLoadLayoutB: gl.constexpr,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
    EVEN_K: gl.constexpr,
    K_RES: gl.constexpr,
    total_full_tiles,
    sk_target,
    SK_PREFETCH: gl.constexpr,
):
    """One full-K tile of the persistent loop, for virtual pid vpid: v9's main loop and
    epilogue, the epilogue prefetching the next tile.

    SK_PREFETCH: on the program's last trip the prefetch loads sk_target = (m, n,
    column, k_hi), the first pair of the program's first Stream-K segment, keeping its
    first k_hi K values (0: masked off). Without it the last trip's prefetch stays
    unmasked and is discarded: masks cost more than the loads
    (results/last_prefetch.md).

    Expects the tile's first pair in flight in buffers 0 and 1 and its first operands in
    a_top / b_left; prefetches the same for vpid + NUM_PROGRAMS (clamped to the last
    persistent tile, so every trip issues the same instructions and wait counts),
    or on the last trip the Stream-K target (SK_PREFETCH), and returns that
    tile's a_top / b_left and load addressing. tile_addr is this tile's (a_addr,
    b_addr) (a_load_addressing / b_load_addressing).

    The pair in flight is the tile's last pair (K is K_PAD here; on a ragged K the
    pair is partial), so the main loop starts at column 0, and the next tile's
    prefetch loads that tile's last pair, masked on a ragged K."""
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn = strides
    a_offsets, a_bot_shift, b_offsets, b_right_shift = kofs
    a_addr, b_addr = tile_addr
    smemA_top, smemA_bot, smemB_left, smemB_right = smem
    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfmaLayout, k_width=8
    )
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfmaLayout, k_width=8
    )
    iterMax = gl.cdiv(K, BLOCK_K)

    pid_m, pid_n = _get_pids(
        vpid, M, N, BLOCK_M, BLOCK_N, total_full_tiles, NUM_XCDS, GROUP_SIZE_M
    )
    next_m, next_n = _get_pids(
        min(vpid + NUM_PROGRAMS, total_full_tiles - 1),
        M,
        N,
        BLOCK_M,
        BLOCK_N,
        total_full_tiles,
        NUM_XCDS,
        GROUP_SIZE_M,
    )
    has_next = vpid + NUM_PROGRAMS < total_full_tiles
    if SK_PREFETCH:
        sk_m, sk_n, sk_col, sk_khi = sk_target
        next_m = gl.where(has_next, next_m, sk_m)
        next_n = gl.where(has_next, next_n, sk_n)
        next_col = gl.where(has_next, K - 2 * BLOCK_K, sk_col)
    a_next = a_ptr + next_m.to(gl.int64) * BLOCK_M * stride_am
    b_next = b_ptr + next_n.to(gl.int64) * BLOCK_N * stride_bn

    # The last pair is in flight, so the first prefetch is column 0.
    a_base = a_ptr + pid_m.to(gl.int64) * BLOCK_M * stride_am
    b_base = b_ptr + pid_n.to(gl.int64) * BLOCK_N * stride_bn
    if SK_PREFETCH:
        a_next += next_col * stride_ak
        b_next += next_col * stride_bk
    else:
        a_next += (K - 2 * BLOCK_K) * stride_ak
        b_next += (K - 2 * BLOCK_K) * stride_bk
    m_left = M - pid_m * BLOCK_M
    n_left = gl.multiple_of(N - pid_n * BLOCK_N, 8)

    ## Main loop — same as v9, its first iteration peeled: without bias, the
    ## constant-zero accumulator becomes the MFMA's inline-0 C operand, so no AGPRs
    ## are zeroed per tile (a loop-carried zero would be materialised).
    accs = init_acc_edge(
        bias_ptr, pid_n, n_left, BLOCK_M, BLOCK_N, mfmaLayout, ADD_BIAS, EVEN_N
    )
    accs, a_top, b_left = k_step_pair(
        accs,
        a_top,
        b_left,
        smem,
        a_base,
        b_base,
        a_addr,
        b_addr,
        dotOpLayoutA,
        dotOpLayoutB,
        (not EVEN_K) and K_RES >= BLOCK_K,
    )
    a_base += BLOCK_K * stride_ak * 2
    b_base += BLOCK_K * stride_bk * 2
    for k in range(2, iterMax - 2, 2):
        accs, a_top, b_left = k_step_pair(
            accs,
            a_top,
            b_left,
            smem,
            a_base,
            b_base,
            a_addr,
            b_addr,
            dotOpLayoutA,
            dotOpLayoutB,
        )
        a_base += BLOCK_K * stride_ak * 2
        b_base += BLOCK_K * stride_bk * 2
    acc_tl, acc_bl, acc_tr, acc_br = accs

    # The next tile's load offsets, for its prefetch below and for its own trip.
    na_addr = a_load_addressing(
        next_m,
        a_offsets,
        a_bot_shift,
        M,
        stride_am,
        stride_ak,
        BLOCK_M,
        BLOCK_K,
        gLoadLayoutA,
        EVEN_M,
    )
    nb_addr = b_load_addressing(
        next_n,
        b_offsets,
        b_right_shift,
        N,
        stride_bk,
        stride_bn,
        BLOCK_N,
        BLOCK_K,
        gLoadLayoutB,
        EVEN_N,
    )
    na_top, na_bot, na_bot_shift, na_odd_shift = na_addr
    nb_left, nb_right, nb_right_shift, nb_odd_shift = nb_addr
    # Computed here rather than with a_next, so they are not live across the K loop.
    if SK_PREFETCH:
        k_hi = gl.where(has_next, 2 * BLOCK_K - K_RES, sk_khi)
        mask_a0 = k_tail_mask(k_hi, 0, BLOCK_K, gLoadLayoutA, False)
        mask_b0 = k_tail_mask(k_hi, 0, BLOCK_K, gLoadLayoutB, True)
        mask_a1 = k_tail_mask(k_hi, 1, BLOCK_K, gLoadLayoutA, False)
        mask_b1 = k_tail_mask(k_hi, 1, BLOCK_K, gLoadLayoutB, True)
    elif EVEN_K:
        mask_a0, mask_b0, mask_a1, mask_b1 = None, None, None, None
    else:
        mask_a0, mask_b0, mask_a1, mask_b1 = tail_pair_masks(
            2 * BLOCK_K - K_RES,
            K_RES,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            gLoadLayoutA,
            gLoadLayoutB,
            True,
        )

    ## Epilogue: 4-quadrant stores with natural-pipeline ordering, as in v9, each
    ## async copy prefetching the next tile's first pair.

    # Store layout sized to the half tile: 8 contiguous elements per thread keep the
    # 16-byte store, and one warp spans exactly BLOCK_N // 2 columns ([4, 16] lanes at
    # BLOCK_N = 256, the tutorial's constant).
    STORE_CONTIG: gl.constexpr = 8
    STORE_LANES_N: gl.constexpr = (BLOCK_N // 2) // STORE_CONTIG
    STORE_LANES_M: gl.constexpr = 64 // STORE_LANES_N
    gStoreLayoutC: gl.constexpr = gl.BlockedLayout(
        [1, STORE_CONTIG], [STORE_LANES_M, STORE_LANES_N], [4, 1], [1, 0]
    )

    offs_cm = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gStoreLayoutC))
    # The tile's row goes in the 64-bit scalar base, so the 32-bit offsets only span
    # BLOCK_M rows. Its column stays in the vector offsets, which keeps them
    # tile-dependent: the compiler can't hoist them above the tile loop and keep them
    # live (and spilled) through the K loop. The quadrant shifts go in the scalar base
    # too, so all four stores share one offset tensor. On a ragged M the rows are
    # clamped to the tile's last one for the same reason (only rows the store masks
    # drop change). The row term carries the loads' multiple_of hint and n_left is
    # hinted a multiple of 8 (the wrapper checks N), so each thread's 8-column group
    # keeps one predicate and one 16-byte store when stride_cm is not a multiple of 16.
    c_tl_base = c_ptr + pid_m.to(gl.int64) * BLOCK_M * stride_cm
    col = pid_n * BLOCK_N + offs_cn
    if EVEN_M:
        c_rows = gl.multiple_of(stride_cm * offs_cm, 8)
    else:
        c_rows = gl.multiple_of(stride_cm * gl.minimum(offs_cm, m_left - 1), 8)
    c_offsets = c_rows[:, None] + stride_cn * col[None, :]
    c_tr_base = c_tl_base + BLOCK_N * stride_cn // 2
    c_bl_base = c_tl_base + BLOCK_M * stride_cm // 2
    c_br_base = c_bl_base + BLOCK_N * stride_cn // 2
    if EVEN_M and EVEN_N:
        mask_tl, mask_tr, mask_bl, mask_br = None, None, None, None
    else:
        # The ragged last tile stores only the rows and columns inside C.
        in_top = offs_cm[:, None] < m_left
        in_bot = offs_cm[:, None] + BLOCK_M // 2 < m_left
        if EVEN_N:
            mask_tl, mask_tr = in_top, in_top
            mask_bl, mask_br = in_bot, in_bot
        else:
            in_left = offs_cn[None, :] < n_left
            in_right = offs_cn[None, :] + BLOCK_N // 2 < n_left
            mask_tl = in_top & in_left
            mask_tr = in_top & in_right
            mask_bl = in_bot & in_left
            mask_br = in_bot & in_right

    ## Iter iterMax - 2: same 4-region pattern as main loop, each AC prefetching the
    ## next tile's K-step 0 into the buffer the main loop would refill here, so every
    ## wait keeps 5 groups in flight.
    acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    l_idx = (iterMax - 2) % 2
    a_bot = smemA_bot.index(l_idx).load(dotOpLayoutA)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_left.index(0), b_next, nb_left, mask=mask_b0
    )
    gl.amd.cdna4.async_copy.commit_group()

    acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    b_right = smemB_right.index(l_idx).load(dotOpLayoutB)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_top.index(0), a_next, na_top, mask=mask_a0
    )
    gl.amd.cdna4.async_copy.commit_group()

    acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    g_idx = 1 - l_idx
    b_left = smemB_left.index(g_idx).load(dotOpLayoutB)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_bot.index(0), a_next + na_bot_shift, na_bot, mask=mask_a0
    )
    gl.amd.cdna4.async_copy.commit_group()

    acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    a_top = smemA_top.index(g_idx).load(dotOpLayoutA)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_right.index(0), b_next + nb_right_shift, nb_right, mask=mask_b0
    )
    gl.amd.cdna4.async_copy.commit_group()

    ## Iter iterMax - 1
    ## Natural-pipeline epilogue: each store follows its MFMA with one
    ## MFMA cycle of gap, yielding uniform MFMA-store interleaving. The ACs
    ## prefetch the next tile's K-step 1.
    acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    a_bot = smemA_bot.index(g_idx).load(dotOpLayoutA)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_left.index(1), b_next + nb_odd_shift, nb_left, mask=mask_b1
    )
    gl.amd.cdna4.async_copy.commit_group()

    acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    b_right = smemB_right.index(g_idx).load(dotOpLayoutB)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_top.index(1), a_next + na_odd_shift, na_top, mask=mask_a1
    )
    gl.amd.cdna4.async_copy.commit_group()

    c_tl = acc_tl.to(c_ptr.dtype.element_ty)
    c_tl = gl.convert_layout(c_tl, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(
        ptr=c_tl_base, offsets=c_offsets, stored_value=c_tl, mask=mask_tl
    )

    acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_bot.index(1), a_next + na_bot_shift + na_odd_shift, na_bot, mask=mask_a1
    )
    gl.amd.cdna4.async_copy.commit_group()

    c_bl = acc_bl.to(c_ptr.dtype.element_ty)
    c_bl = gl.convert_layout(c_bl, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(
        ptr=c_bl_base, offsets=c_offsets, stored_value=c_bl, mask=mask_bl
    )

    acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")

    # The next tile's K-step 0 operands (prefetched at iterMax - 2), read while the
    # last MFMAs and stores drain; this tile's operand registers are all dead after
    # acc_br. Both reads are deferred to here rather than interleaved as in the main
    # loop: reading b_left after acc_tr keeps it live across the c_bl store, which
    # costs 24-36 AGPR moves per tile and, on a ragged M and N, an extra spill.
    gl.amd.cdna4.async_copy.wait_group(5)
    b_left = smemB_left.index(0).load(dotOpLayoutB)
    a_top = smemA_top.index(0).load(dotOpLayoutA)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_right.index(1), b_next + nb_right_shift + nb_odd_shift, nb_right, mask=mask_b1
    )
    gl.amd.cdna4.async_copy.commit_group()

    c_tr = acc_tr.to(c_ptr.dtype.element_ty)
    c_tr = gl.convert_layout(c_tr, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(
        ptr=c_tr_base, offsets=c_offsets, stored_value=c_tr, mask=mask_tr
    )

    c_br = acc_br.to(c_ptr.dtype.element_ty)
    c_br = gl.convert_layout(c_br, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(
        ptr=c_br_base, offsets=c_offsets, stored_value=c_br, mask=mask_br
    )

    return a_top, b_left, (na_addr, nb_addr)


# ---------------------------------------------------------------------------------
# Stream-K: programs and tiles
# ---------------------------------------------------------------------------------
# The persistent loop covers the XCD-remapped tile ids [0, total_full_tiles) (_get_pids
# with GRID_MN = total_full_tiles), so Stream-K tile t is tile total_full_tiles + t.


@gluon.jit
def streamk_rank(N: gl.constexpr, NUM_XCDS: gl.constexpr, IDLE: gl.constexpr = False):
    """Rank of this program among the N programs taking part in Stream-K: an
    XCD-blocked id, so each XCD's ranks are contiguous (hardware pid p runs on XCD
    p % NUM_XCDS; the first N % NUM_XCDS XCDs hold one rank more). The participants
    are hardware pids 0 .. N-1, so when fewer than the launched programs take part the
    idle ones are spread over the XCDs instead of packed onto the last XCDs. IDLE says
    the launch has programs beyond N; they get rank N."""
    pid = gl.program_id(axis=0)
    rank = pid
    if NUM_XCDS != 1:
        rank = round_robin_to_xcd_blocked(pid, N, NUM_XCDS)
    if IDLE:
        rank = gl.where(pid < N, rank, N)
    return rank


@gluon.jit
def streamk_first_segment(
    total_full_tiles,
    STREAMK_TILES: gl.constexpr,
    PAIRS_PER_TILE: gl.constexpr,
    SPLIT_N: gl.constexpr,
    END_TO_END: gl.constexpr,
    L2: gl.constexpr,
    XCD_SPLIT: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_XCDS: gl.constexpr,
):
    """(has, tile id, k0) of the first segment the Stream-K phase runs on this program
    (the kernel's segment assignment, full-grid launches only); has is 0 for an idle
    program, whose tile and k0 are then tile 0's last pair."""
    if END_TO_END:
        SK_PAIRS: gl.constexpr = STREAMK_TILES * PAIRS_PER_TILE
        rank = streamk_rank(NUM_PROGRAMS, NUM_XCDS)
        sk_start = even_split_start(rank, SK_PAIRS, NUM_PROGRAMS)
        cur = even_split_start(rank + 1, SK_PAIRS, NUM_PROGRAMS)
        t = (cur - 1) // PAIRS_PER_TILE
        tile_start = t * PAIRS_PER_TILE
        k0 = (gl.maximum(tile_start, sk_start) - tile_start) * 2
        has = 1
    else:
        if L2:
            rank = streamk_rank(NUM_PROGRAMS, NUM_XCDS)
            t, j, n_tile = xcd_local_chunk_of(
                rank, STREAMK_TILES, NUM_PROGRAMS // NUM_XCDS, NUM_XCDS, XCD_SPLIT
            )
            has = (n_tile > 0).to(gl.int32)
        else:
            rank = streamk_rank(STREAMK_TILES * SPLIT_N, NUM_XCDS, IDLE=True)
            has = (rank < STREAMK_TILES * SPLIT_N).to(gl.int32)
            t = rank // SPLIT_N
            j = rank % SPLIT_N
        k0 = even_split_start(j, PAIRS_PER_TILE, SPLIT_N) * 2 * has
        t = t * has
    return has, total_full_tiles + t, k0


@gluon.jit
def xcd_local_chunk_of(
    rank,
    S: gl.constexpr,
    PER: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    N: gl.constexpr,
):
    """(tile, chunk, programs on the tile) of the program with this rank in the
    XCD-local split (l2), with 0 programs if it idles. XCD x = rank // PER owns part x
    of an even split of the S Stream-K tiles over the XCDs; its local program l works
    on its tile l // N as chunk l % N, so a tile's N programs are on one XCD. The host
    checks ceil(S / NUM_XCDS) N <= PER."""
    x = rank // PER
    local = rank % PER
    tl = local // N
    t = even_split_start(x, S, NUM_XCDS) + tl
    n = N * (tl < even_split_size(x, S, NUM_XCDS)).to(gl.int32)
    return t, local % N, n


@gluon.jit
def streamk_owner_of(i, L: gl.constexpr, R: gl.constexpr):
    """End-to-end split: the program whose range holds pair i, when the first R
    programs hold L + 1 pairs and the rest L."""
    head: gl.constexpr = R * (L + 1)
    if L == 0:
        owner = i // (L + 1)
    else:
        owner = i // (L + 1)
        if i >= head:
            owner = R + (i - head) // L
    return owner


# ---------------------------------------------------------------------------------
# Stream-K: synchronisation
#
# Partials are stored write-through (.wt) and drained in every wave before the flag
# or barrier arrival that publishes them, and read with .cv loads, which miss L1; so
# every atomic is relaxed, with compiler fences where an acquire would be. Flags and
# counts are re-armed inside the launch (zero between launches).
# ---------------------------------------------------------------------------------


@gluon.jit
def _one():
    return gl.zeros([1], gl.int32, gl.BlockedLayout([1], [64], [gl.num_warps()], [0]))


@gluon.jit
def drain_and_barrier():
    """Every wave waits for all its vector memory ops (stores and async copies: vmcnt
    is one counter), then a workgroup barrier. A flag set after this is ordered after
    all the waves' partial stores."""
    gl.inline_asm_elementwise(
        "s_waitcnt vmcnt(0)\ns_barrier\nv_mov_b32 $0, $1",
        "=v,v,~{memory}",
        [_one()],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def asm_fence(code: gl.constexpr):
    """Inline asm with a memory clobber: the compiler moves no memory access across
    it."""
    gl.inline_asm_elementwise(
        code + "\nv_mov_b32 $0, $1",
        "=v,v,~{memory}",
        [_one()],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def tile_barrier(sync_ptr, b, n, wait):
    """Arrive on the reduce-scatter barrier of the tile whose first slot is b and, if
    wait, wait until all n of its programs have published their partials.

    sync[2b] counts arrivals and sync[2b + 1] is a generation. The last arriver resets
    the count and bumps the generation, which re-arms the barrier for the next launch.
    Scalar atomics run once per workgroup and broadcast their result."""
    cnt_ptr = sync_ptr + 2 * b
    gen_ptr = cnt_ptr + 1
    g0 = gl.atomic_add(gen_ptr, 0, sem="relaxed", scope="gpu")
    # If the last arrival and the bump overtook this read, the program would wait for
    # a generation that never comes.
    asm_fence("s_waitcnt vmcnt(0)")
    arrived = gl.atomic_add(cnt_ptr, 1, sem="relaxed", scope="gpu")
    if arrived == n - 1:
        gl.atomic_xchg(cnt_ptr, 0, sem="relaxed", scope="gpu")
        gl.atomic_xchg(gen_ptr, g0 + 1, sem="relaxed", scope="gpu")
    elif wait:
        while gl.atomic_add(gen_ptr, 0, sem="relaxed", scope="gpu") == g0:
            pass
    asm_fence("s_nop 0")  # no partial read moves above the wait


@gluon.jit
def l2_tile_barrier(
    sync_ptr,
    b,
    n,
    readers,
    j,
    is_reader,
    NUM_PROGRAMS: gl.constexpr,
    NUM_XCDS: gl.constexpr,
):
    """Publish to each reader's private row, then let readers reset their row.

    Layout. P = NUM_PROGRAMS, PER = P / NUM_XCDS (32 words, one 128-byte
    line). A split tile's programs are slots b .. b+n-1 (chunk j = slot - b);
    chunks j < r (readers) also reduce. Each slot reserves PER flag words,
    so the tile owns the n*PER words of its own slots:

      tile_sync (P = 256, PER = 32), word offsets:

      0           512         544               704            896
      | wt count, | slot 0    | slot 1 ...      | slots 6..11  | slot 12 ...
      | gen (2P)  | 32 words  |                 | tile b = 6   |
                  ^ flag region at 2P           ^ base = 2P + b*PER

    Within the tile, word i*n + j is writer j's flag for reader i. Example,
    n = 6, r = 4, b = 6, P = 256 (base = 512 + 6*32 = 704):

      writer j:             0   1   2   3   4   5
      reader 0  row 0*n:  [ 0][ 1][ 2][ 3][ 4][ 5]
      reader 1  row 1*n:  [ 6][ 7][ 8][ 9][10][11]
      reader 2  row 2*n:  [12][13][14][15][16][17]
      reader 3  row 3*n:  [18][19][20][21][22][23]

      writer 4 stores 1 down column 4: offsets 4, 10, 16, 22 (words 708 ..)
      reader 2 polls, then zeroes, row 2: offsets 12 .. 17 (words 716 .. 721)

    In the code j is always this program's chunk, and a lane index covers
    the other side in one instruction:
      store:  flags + m*n + j     lane m = reader i, j = this writer
      poll:   row + lane          row = flags + j*n (this reader), lane = writer
      reset:  row + m             lane m = writer

    Each word has one writer and one reader, so plain stores set the flags
    and the reader alone resets its row: the region is zero between launches.
    r*n <= n*PER keeps a tile inside its own region (the next tile starts at
    (b+n)*PER); base is line-aligned, so the grid fits one line while
    r*n <= 32 (n <= 7).

    Every wave polls the row on its own, but only wave 0 stores and resets:
    a wave that reset the row while another wave of the same program still
    polled it would leave that wave spinning, and a lagging wave's duplicate
    1 could land after the reset. The workgroup barrier after the poll keeps
    the reset behind every wave's last poll."""
    PER: gl.constexpr = NUM_PROGRAMS // NUM_XCDS
    flags = sync_ptr + 2 * NUM_PROGRAMS + b * PER
    FL: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    # Wave w holds elements 64 w .. 64 w + 63, so only wave 0 has m < 64.
    m = gl.arange(0, 64 * gl.num_warps(), FL)
    # Every wave holds a copy of elements 0 .. 63.
    lane = gl.arange(0, 64, FL)

    gl.store(
        flags + m * n + j,
        m * 0 + 1,
        mask=m < readers,
    )
    if is_reader:
        row = flags + j * n
        done = 0
        while done == 0:
            asm_fence("s_nop 0")  # keeps the poll inside the loop
            f = gl.amd.cdna3.buffer_load(
                ptr=row,
                offsets=lane,
                mask=lane < n,
                other=1,
                cache=".cg",
            )
            done = (gl.min(f, axis=0) == 1).to(gl.int32)
        gl.barrier()  # every wave has seen the row before the reset
        gl.store(row + m, m * 0, mask=m < n)
    asm_fence("s_nop 0")  # no partial read moves above the wait


# ---------------------------------------------------------------------------------
# Stream-K: partials and endings
#
# A program's slot in P is one fp32 tile, four quadrants of QUAD = HALF_M HALF_N
# elements, each stored straight from the MFMA layout (lane_contiguous_offsets) and
# read back by the owner in the same layout. Reduce-scatter reads it as NBLK =
# BLOCK_M BLOCK_N / 1024 blocks of 1024 elements, one per (quadrant, register
# vector), each a 32x32 square of the tile.
# ---------------------------------------------------------------------------------


@gluon.jit
def edge_mask(
    m_left,
    n_left,
    ROWS: gl.constexpr,
    COLS: gl.constexpr,
    layout: gl.constexpr,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
):
    """In-bounds mask of a ROWS x COLS block of C with m_left rows and n_left columns
    before the matrix edge (either may exceed the block). On a single ragged
    dimension it is [ROWS, 1] or [1, COLS]; the store broadcasts it over the block."""
    if not EVEN_M:
        r = gl.arange(0, ROWS, gl.SliceLayout(1, layout))[:, None]
    if not EVEN_N:
        c = gl.arange(0, COLS, gl.SliceLayout(0, layout))[None, :]
    if not EVEN_M and not EVEN_N:
        mask = (r < m_left) & (c < n_left)
    elif not EVEN_M:
        mask = r < m_left
    else:
        mask = c < n_left
    return mask


@gluon.jit
def store_c(
    acc,
    c_base,
    c_offsets,
    out_dtype: gl.constexpr,
    gStoreLayoutC: gl.constexpr,
    m_left=0,
    n_left=0,
    EVEN_M: gl.constexpr = True,
    EVEN_N: gl.constexpr = True,
):
    """Convert a quadrant to C's type and store layout and store it, masked to its first
    m_left rows / n_left columns on a ragged M / N."""
    c = acc.to(out_dtype)
    c = gl.convert_layout(c, layout=gStoreLayoutC)
    if EVEN_M and EVEN_N:
        gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_offsets, stored_value=c)
    else:
        mask = edge_mask(
            m_left, n_left, c.shape[0], c.shape[1], gStoreLayoutC, EVEN_M, EVEN_N
        )
        gl.amd.cdna3.buffer_store(
            ptr=c_base, offsets=c_offsets, stored_value=c, mask=mask
        )


@constexpr_function
def lane_map_matches(ll, half_m, half_n):
    """True if a HALF_M x HALF_N quadrant's MFMA layout (as a linear layout) has the
    bases lane_contiguous_offsets assumes: registers col 1, 2, then the column blocks
    32 .. HALF_N / 2, then the row blocks 32 .. HALF_M / 2; lanes row 1, 2, 4, 8 and
    col 4, 8; warps col 16 and row 16."""
    reg = (
        [[0, 1], [0, 2]]
        + [[0, c] for c in (32, 64) if c < half_n]
        + [[r, 0] for r in (32, 64) if r < half_m]
    )
    return (
        [list(b) for b in ll.reg_bases] == reg
        and [list(b) for b in ll.lane_bases]
        == [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]]
        and [list(b) for b in ll.warp_bases] == [[0, 16], [16, 0]]
    )


@gluon.jit
def lane_contiguous_offsets(
    HALF_M: gl.constexpr, HALF_N: gl.constexpr, mfmaLayout: gl.constexpr
):
    """Element offsets of a partial quadrant in its slot, in the MFMA layout: [VQ
    register vectors][4 warps][64 lanes][4 fp32], VQ = (HALF_M / 32) (HALF_N / 32), the
    column blocks before the row blocks.

    Lane l of a wave writes elements 4l .. 4l + 3 of its 1 KiB block, so one
    16-byte-per-lane access covers 8 whole cache lines (row-major addressing would
    spread it over 16 rows). The map is a bit permutation of (row, col) that follows
    the layout's bases, checked at compile time."""
    gl.static_assert(
        lane_map_matches(
            gl.to_linear_layout(mfmaLayout, [HALF_M, HALF_N]), HALF_M, HALF_N
        ),
        "the MFMA layout's bases are not the order the partial map assumes",
    )
    VN: gl.constexpr = HALF_N // 32
    r = gl.arange(0, HALF_M, gl.SliceLayout(1, mfmaLayout))
    c = gl.arange(0, HALF_N, gl.SliceLayout(0, mfmaLayout))
    # The column term is c plus terms constant over each 4 columns, so axis analysis
    # still sees 4 contiguous, aligned elements (16-byte accesses); written with c % 4
    # or shifts and masks, the same map compiles to single-dword accesses. Vector v =
    # column block + VN row block.
    f = (r % 16) * 4 + (r // 16 % 2) * 512
    if HALF_M == 128:
        f = f + (r // 32 % 2) * (VN * 1024) + (r // 64) * (2 * VN * 1024)
    elif HALF_M == 64:
        f = f + (r // 32) * (VN * 1024)
    g = c + (c // 4 % 4) * 60 + (c // 16 % 2) * 240
    if HALF_N == 128:
        g = g + (c // 32 % 2) * 992 + (c // 64) * 1984
    elif HALF_N == 64:
        g = g + (c // 32) * 992
    return f[:, None] + g[None, :]


@gluon.jit
def store_partial(
    acc,
    q: gl.constexpr,
    p_ptr,
    rank,
    p_offsets,
    HALF_M: gl.constexpr,
    HALF_N: gl.constexpr,
    CACHE: gl.constexpr = ".wt",
):
    """Quadrant q of a partial to slot P[rank]: write-through, or the default policy
    for the l2 exchange."""
    QUAD: gl.constexpr = HALF_M * HALF_N
    SLOT: gl.constexpr = 4 * QUAD
    gl.amd.cdna3.buffer_store(
        stored_value=acc,
        ptr=p_ptr + rank.to(gl.int64) * SLOT + q * QUAD,
        offsets=p_offsets,
        cache=CACHE,
    )


@gluon.jit
def owner_store(
    acc,
    psum,
    q: gl.constexpr,
    bias_ptr,
    pid_n,
    c_base,
    c_offsets,
    BLOCK_N: gl.constexpr,
    HALF_N: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    out_dtype: gl.constexpr,
    gStoreLayoutC: gl.constexpr,
    mfmaLayout: gl.constexpr,
    m_left,
    n_left,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
):
    """Quadrant q of C: the accumulator plus psum (the peers' partials) and the bias.
    Contributors' partials exclude the bias, so it is added once, here."""
    HALF_M: gl.constexpr = acc.shape[0]
    if ADD_BIAS:
        offs_pn = gl.arange(0, HALF_N, gl.SliceLayout(0, mfmaLayout))
        if EVEN_N:
            bias = gl.amd.cdna3.buffer_load(
                ptr=bias_ptr + pid_n * BLOCK_N + (q // 2) * HALF_N, offsets=offs_pn
            )
        else:
            bias = gl.amd.cdna3.buffer_load(
                ptr=bias_ptr + pid_n * BLOCK_N + (q // 2) * HALF_N,
                offsets=offs_pn,
                mask=offs_pn < n_left - (q // 2) * HALF_N,
                other=0.0,
            )
        psum = psum + bias.to(gl.float32)[None, :]
    store_c(
        acc + psum,
        c_base,
        c_offsets,
        out_dtype,
        gStoreLayoutC,
        m_left - (q % 2) * HALF_M,
        n_left - (q // 2) * HALF_N,
        EVEN_M,
        EVEN_N,
    )


@gluon.jit
def peer_sums(
    p_ptr,
    rank,
    n_peers,
    p_offsets,
    z,
    q0: gl.constexpr,
    NQ: gl.constexpr,
    PEER_STEP: gl.constexpr,
    HALF_M: gl.constexpr,
    HALF_N: gl.constexpr,
    MAX_PEERS: gl.constexpr,
):
    """The sums of quadrants q0 .. q0 + NQ - 1 (NQ = 1, 2 or 4) over the partials of
    peers rank + PEER_STEP i, i = 1 .. n_peers, as four tensors (the last 4 - NQ stay z,
    zeros); a peer's NQ quadrants are one batch of loads.
    - MAX_PEERS > 0: a compile-time loop over MAX_PEERS peers, the loads of peers past
      n_peers masked off, so that several peers' batches can be in flight at once.
    - Otherwise a run-time loop, each batch's loads written before its adds: written
      load, add, load, add, the scheduler reuses one quadrant's registers for the next
      and waits for each load in turn."""
    QUAD: gl.constexpr = HALF_M * HALF_N
    SLOT: gl.constexpr = 4 * QUAD
    s0, s1, s2, s3 = z, z, z, z
    if MAX_PEERS > 0:
        for j in gl.static_range(MAX_PEERS):
            peer = (rank + PEER_STEP * (1 + j)).to(gl.int64)
            live = j < n_peers
            s0 += gl.amd.cdna3.buffer_load(
                ptr=p_ptr + peer * SLOT + q0 * QUAD,
                offsets=p_offsets,
                mask=live,
                other=0.0,
                cache=".cv",
            ).to(gl.float32)
            if NQ > 1:
                s1 += gl.amd.cdna3.buffer_load(
                    ptr=p_ptr + peer * SLOT + (q0 + 1) * QUAD,
                    offsets=p_offsets,
                    mask=live,
                    other=0.0,
                    cache=".cv",
                ).to(gl.float32)
            if NQ > 2:
                s2 += gl.amd.cdna3.buffer_load(
                    ptr=p_ptr + peer * SLOT + (q0 + 2) * QUAD,
                    offsets=p_offsets,
                    mask=live,
                    other=0.0,
                    cache=".cv",
                ).to(gl.float32)
                s3 += gl.amd.cdna3.buffer_load(
                    ptr=p_ptr + peer * SLOT + (q0 + 3) * QUAD,
                    offsets=p_offsets,
                    mask=live,
                    other=0.0,
                    cache=".cv",
                ).to(gl.float32)
    else:
        for j in range(n_peers):
            peer = (rank + PEER_STEP * (1 + j)).to(gl.int64)
            x0 = gl.amd.cdna3.buffer_load(
                ptr=p_ptr + peer * SLOT + q0 * QUAD, offsets=p_offsets, cache=".cv"
            ).to(gl.float32)
            if NQ == 1:
                s0 += x0
            elif NQ == 2:
                x1 = gl.amd.cdna3.buffer_load(
                    ptr=p_ptr + peer * SLOT + (q0 + 1) * QUAD,
                    offsets=p_offsets,
                    cache=".cv",
                ).to(gl.float32)
                s0 += x0
                s1 += x1
            else:
                x1 = gl.amd.cdna3.buffer_load(
                    ptr=p_ptr + peer * SLOT + (q0 + 1) * QUAD,
                    offsets=p_offsets,
                    cache=".cv",
                ).to(gl.float32)
                x2 = gl.amd.cdna3.buffer_load(
                    ptr=p_ptr + peer * SLOT + (q0 + 2) * QUAD,
                    offsets=p_offsets,
                    cache=".cv",
                ).to(gl.float32)
                x3 = gl.amd.cdna3.buffer_load(
                    ptr=p_ptr + peer * SLOT + (q0 + 3) * QUAD,
                    offsets=p_offsets,
                    cache=".cv",
                ).to(gl.float32)
                s0 += x0
                s1 += x1
                s2 += x2
                s3 += x3
    return s0, s1, s2, s3


@gluon.jit
def owner_ending(
    accs,
    is_partial,
    n_peers,
    rank,
    pid_n,
    bias_ptr,
    p_ptr,
    locks_ptr,
    c_bases,
    c_offsets,
    m_left,
    n_left,
    BLOCK_N: gl.constexpr,
    PEER_STEP: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    out_dtype: gl.constexpr,
    gStoreLayoutC: gl.constexpr,
    mfmaLayout: gl.constexpr,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
    QPB: gl.constexpr,
    MAX_PEERS: gl.constexpr,
):
    """The owner ending of a segment. A contributor (is_partial) stores its four
    quadrants to P[rank], drains and raises locks[rank]. The owner waits for its
    n_peers peers' flags (rank + PEER_STEP i), reads their partials QPB quadrants at a
    time (16 fp32 loads per wave: 4 at 128x128, 2 at 256x128 / 128x256, 1 at 256x256;
    peer_sums), adds each batch's sums into the accumulator once, outside the peer loop
    (a loop or branch that carries the AGPR accumulator spills), and stores that batch
    of C before the next batch's reads, so only one batch's sums are live. Each partial
    has one reader, so the owner lowers its peers' flags for the next launch; the
    kernel boundary orders that against it."""
    HALF_M: gl.constexpr = accs[0].shape[0]
    HALF_N: gl.constexpr = accs[0].shape[1]
    for pj in range(n_peers):
        while (
            gl.atomic_add(
                locks_ptr + rank + PEER_STEP * (1 + pj), 0, sem="relaxed", scope="gpu"
            )
            == 0
        ):
            pass
    if n_peers > 0:
        asm_fence("s_nop 0")

    # With a run-time peer loop over one quadrant per batch (256x256: end to end, or
    # above 8 programs per tile) a contributor stores each quadrant just before its
    # batch: storing all four first is about 1% slower on two-tile there.
    STORE_EACH: gl.constexpr = QPB == 1 and MAX_PEERS == 0
    p_offsets = lane_contiguous_offsets(HALF_M, HALF_N, mfmaLayout)
    z = gl.zeros((HALF_M, HALF_N), gl.float32, mfmaLayout)
    if not STORE_EACH:
        if is_partial != 0:
            for q in gl.static_range(4):
                store_partial(accs[q], q, p_ptr, rank, p_offsets, HALF_M, HALF_N)
    for q0 in gl.static_range(0, 4, QPB):
        if STORE_EACH:
            if is_partial != 0:
                store_partial(accs[q0], q0, p_ptr, rank, p_offsets, HALF_M, HALF_N)
        sums = peer_sums(
            p_ptr,
            rank,
            n_peers,
            p_offsets,
            z,
            q0,
            QPB,
            PEER_STEP,
            HALF_M,
            HALF_N,
            MAX_PEERS,
        )
        if is_partial == 0:
            for i in gl.static_range(QPB):
                owner_store(
                    accs[q0 + i],
                    sums[i],
                    q0 + i,
                    bias_ptr,
                    pid_n,
                    c_bases[q0 + i],
                    c_offsets,
                    BLOCK_N,
                    HALF_N,
                    ADD_BIAS,
                    out_dtype,
                    gStoreLayoutC,
                    mfmaLayout,
                    m_left,
                    n_left,
                    EVEN_M,
                    EVEN_N,
                )

    if is_partial != 0:
        # The flag may only become visible after all four waves' partial stores have
        # completed.
        drain_and_barrier()
        gl.atomic_xchg(locks_ptr + rank, 1, sem="relaxed", scope="gpu")
    for pj in range(n_peers):
        gl.atomic_xchg(
            locks_ptr + rank + PEER_STEP * (1 + pj), 0, sem="relaxed", scope="gpu"
        )


@gluon.jit
def block_col(v, VN: gl.constexpr):
    """Column of register vector (32x32 block) v in its quadrant, VN blocks across."""
    col = 0
    if VN == 4:
        col = (v % 2) * 32 + (v // 2 % 2) * 64
    elif VN == 2:
        col = (v % 2) * 32
    return col


@gluon.jit
def to_band(x, G: gl.constexpr, VN: gl.constexpr = 4):
    """(G, 32, 32) aligned blocks -> their 2-D band, VN blocks across a quadrant. Block
    index bits from the lowest: the column blocks, then the row blocks. VN = 4: G = 1,
    2, 4: 32 x 32 G; 8, 16: 8 G x 128."""
    if VN == 4:
        if G == 1:
            y = gl.reshape(x, [32, 32])
        elif G == 2:
            y = gl.reshape(gl.permute(x, [1, 0, 2]), [32, 64])
        elif G == 4:
            y = gl.reshape(
                gl.permute(gl.reshape(x, [2, 2, 32, 32]), [2, 0, 1, 3]), [32, 128]
            )
        elif G == 8:
            y = gl.reshape(
                gl.permute(gl.reshape(x, [2, 2, 2, 32, 32]), [0, 3, 1, 2, 4]),
                [64, 128],
            )
        else:
            y = gl.reshape(
                gl.permute(gl.reshape(x, [2, 2, 2, 2, 32, 32]), [0, 1, 4, 2, 3, 5]),
                [128, 128],
            )
    else:
        CB: gl.constexpr = min(G, VN)
        RB: gl.constexpr = G // CB
        y = gl.reshape(
            gl.permute(gl.reshape(x, [RB, CB, 32, 32]), [0, 2, 1, 3]),
            [RB * 32, CB * 32],
        )
    return y


@gluon.jit
def band_pos(
    g0,
    layout: gl.constexpr,
    ROWS: gl.constexpr,
    COLS: gl.constexpr,
    HALF_M: gl.constexpr = 128,
    HALF_N: gl.constexpr = 128,
):
    """Tile rows and columns of the band of the aligned blocks g0 .. g0 + G - 1."""
    VN: gl.constexpr = HALF_N // 32
    VQ: gl.constexpr = HALF_M * HALF_N // 1024
    r = gl.arange(0, ROWS, gl.SliceLayout(1, layout))[:, None]
    c = gl.arange(0, COLS, gl.SliceLayout(0, layout))[None, :]
    q = g0 // VQ
    v0 = g0 % VQ
    row0 = (v0 // VN) * 32
    col0 = block_col(v0, VN)
    return (q % 2) * HALF_M + row0 + r, (q // 2) * HALF_N + col0 + c


@gluon.jit
def reduce_scatter_slice(
    p_ptr,
    bias_ptr,
    c_tile,
    b,
    j,
    n,
    readers,
    pid_n,
    stride_cm,
    stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    NB: gl.constexpr,
    U: gl.constexpr,
    out_dtype: gl.constexpr,
    LOAD_CACHE: gl.constexpr,
    m_left,
    n_left,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
):
    """Reader j of `readers` (a power of two) of a tile split n ways: sums blocks
    [NBLK j / readers, NBLK (j + 1) / readers) over the n partials in slots P[b .. b + n
    - 1], adds the bias and stores them to C. NB blocks (NB <= VQ, so a group stays in
    one quadrant) at a time, U partials of a group per trip, so NB U loads per lane are
    in flight. Each group is reshaped to its 2-D band and stored along C's rows, 16
    bytes per lane (8 at NB = 1)."""
    HALF_M: gl.constexpr = BLOCK_M // 2
    HALF_N: gl.constexpr = BLOCK_N // 2
    VN: gl.constexpr = HALF_N // 32
    VQ: gl.constexpr = VN * (HALF_M // 32)
    NBLK: gl.constexpr = 4 * VQ
    gl.static_assert(NB <= VQ, "a reduce-scatter group must stay inside one quadrant")
    SLOT: gl.constexpr = BLOCK_M * BLOCK_N
    blk_lo = (NBLK * j) // readers
    blk_hi = (NBLK * (j + 1)) // readers
    # Rows before columns in lane order: lane l gets row l % 16 and columns 4 (l // 16)
    # .. + 3, which in_blk puts at 4 l .. 4 l + 3, so each 4-lane group reads 64
    # contiguous bytes. Columns first, lanes 0-3 hit 4 lines 256 bytes apart and the L1
    # stalls on tag conflicts (reads take twice as long).
    L3: gl.constexpr = gl.BlockedLayout([1, 1, 4], [1, 16, 4], [1, 2, 2], [1, 2, 0])
    i = gl.arange(0, NB, gl.SliceLayout(1, gl.SliceLayout(2, L3)))[:, None, None]
    r = gl.arange(0, 32, gl.SliceLayout(0, gl.SliceLayout(2, L3)))[None, :, None]
    c = gl.arange(0, 32, gl.SliceLayout(0, gl.SliceLayout(1, L3)))[None, None, :]
    # lane_contiguous_offsets restricted to one block: rows 0-31 and columns 0-31.
    in_blk = (r % 16) * 4 + (r // 16) * 512 + c + (c // 4 % 4) * 60 + (c // 16) * 240
    cb = gl.arange(0, 32, gl.SliceLayout(0, gl.SliceLayout(1, L3)))[None, :]
    ib = gl.arange(0, NB, gl.SliceLayout(1, gl.SliceLayout(1, L3)))[:, None]
    # The band: min(NB, VN) blocks across, the rest down.
    ROWS: gl.constexpr = 32 * (NB // min(NB, VN))
    COLS: gl.constexpr = 32 * min(NB, VN)
    if COLS == 128:
        SL: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
    elif COLS == 64:
        SL: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])
    else:
        SL: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [4, 1], [1, 0])
    for g in range(blk_lo, blk_hi, NB):
        blk = g + i
        valid = blk < blk_hi
        p_off = blk * 1024 + in_blk
        s = gl.zeros((NB, 32, 32), gl.float32, L3)
        for p0 in range(0, n, U):
            for u in gl.static_range(U):
                slot = p_ptr + (b + p0 + u).to(gl.int64) * SLOT
                s += gl.amd.cdna3.buffer_load(
                    ptr=slot,
                    offsets=p_off,
                    mask=valid & (p0 + u < n),
                    other=0.0,
                    cache=LOAD_CACHE,
                ).to(gl.float32)
        if ADD_BIAS:
            # Block q VQ + v: quadrant q (tl, bl, tr, br); register vector v's bits are
            # the column blocks (32, 64), then the row blocks.
            qb = (g + ib) // VQ
            vb = (g + ib) % VQ
            col_b = (qb // 2) * HALF_N
            if VN >= 2:
                col_b = col_b + (vb % 2) * 32
            if VN == 4:
                col_b = col_b + (vb // 2 % 2) * 64
            col_b = col_b + cb
            bias_mask = (g + ib) < blk_hi
            if not EVEN_N:
                bias_mask = bias_mask & (col_b < n_left)
            bias = gl.amd.cdna3.buffer_load(
                ptr=bias_ptr + pid_n * BLOCK_N, offsets=col_b, mask=bias_mask, other=0.0
            )
            s = s + bias.to(gl.float32)[:, None, :]
        cs = gl.convert_layout(to_band(s.to(out_dtype), NB, VN), SL)
        trow, tcol = band_pos(g, SL, ROWS, COLS, HALF_M, HALF_N)
        if EVEN_M and EVEN_N:
            gl.amd.cdna3.buffer_store(
                stored_value=cs, ptr=c_tile, offsets=trow * stride_cm + tcol * stride_cn
            )
        else:
            gl.amd.cdna3.buffer_store(
                stored_value=cs,
                ptr=c_tile,
                offsets=trow * stride_cm + tcol * stride_cn,
                mask=(trow < m_left) & (tcol < n_left),
            )


@gluon.jit
def rs_ending(
    accs,
    rank,
    j,
    n_tile,
    r_tile,
    pid_n,
    bias_ptr,
    p_ptr,
    sync_ptr,
    c_bases,
    c_offsets,
    stride_cm,
    stride_cn,
    m_left,
    n_left,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    RS_NB: gl.constexpr,
    RS_U: gl.constexpr,
    out_dtype: gl.constexpr,
    gStoreLayoutC: gl.constexpr,
    mfmaLayout: gl.constexpr,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
):
    """The reduce-scatter ending of tile-aligned segment j of n_tile (its slots are
    rank - j .. + n_tile - 1). A whole tile stores C directly. A split tile's programs
    all store their partial and arrive on the tile's barrier; the first r_tile wait on
    it and each reduces its slice (reduce_scatter_slice)."""
    HALF_M: gl.constexpr = BLOCK_M // 2
    HALF_N: gl.constexpr = BLOCK_N // 2
    is_split = gl.to_tensor(n_tile > 1).to(gl.int32)
    p_offsets = lane_contiguous_offsets(HALF_M, HALF_N, mfmaLayout)
    z = gl.zeros((HALF_M, HALF_N), gl.float32, mfmaLayout)
    for q in gl.static_range(4):
        if is_split != 0:
            store_partial(accs[q], q, p_ptr, rank, p_offsets, HALF_M, HALF_N)
        if is_split == 0:
            owner_store(
                accs[q],
                z,
                q,
                bias_ptr,
                pid_n,
                c_bases[q],
                c_offsets,
                BLOCK_N,
                HALF_N,
                ADD_BIAS,
                out_dtype,
                gStoreLayoutC,
                mfmaLayout,
                m_left,
                n_left,
                EVEN_M,
                EVEN_N,
            )
    if n_tile > 1:
        # The arrival may only become visible after all four waves' partial stores have
        # completed.
        drain_and_barrier()
        is_reader = j < r_tile
        tile_barrier(sync_ptr, rank - j, n_tile, is_reader)
        if is_reader:
            reduce_scatter_slice(
                p_ptr,
                bias_ptr,
                c_bases[0] + pid_n * BLOCK_N * stride_cn,
                rank - j,
                j,
                n_tile,
                r_tile,
                pid_n,
                stride_cm,
                stride_cn,
                BLOCK_M,
                BLOCK_N,
                ADD_BIAS,
                RS_NB,
                RS_U,
                out_dtype,
                ".cv",
                m_left,
                n_left,
                EVEN_M,
                EVEN_N,
            )


@gluon.jit
def l2_rs_ending(
    accs,
    rank,
    j,
    n_tile,
    r_tile,
    pid_n,
    bias_ptr,
    p_ptr,
    sync_ptr,
    c_bases,
    c_offsets,
    stride_cm,
    stride_cn,
    m_left,
    n_left,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    RS_NB: gl.constexpr,
    RS_U: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    out_dtype: gl.constexpr,
    gStoreLayoutC: gl.constexpr,
    mfmaLayout: gl.constexpr,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
):
    """XCD-local reduce-scatter ending with graph-replay-safe flag rows: rs_ending
    with the partials kept in the XCD's L2 (default-policy stores, .cg loads)."""
    HALF_M: gl.constexpr = BLOCK_M // 2
    HALF_N: gl.constexpr = BLOCK_N // 2
    is_split = gl.to_tensor(n_tile > 1).to(gl.int32)
    p_offsets = lane_contiguous_offsets(HALF_M, HALF_N, mfmaLayout)
    z = gl.zeros((HALF_M, HALF_N), gl.float32, mfmaLayout)
    for q in gl.static_range(4):
        if is_split != 0:
            store_partial(accs[q], q, p_ptr, rank, p_offsets, HALF_M, HALF_N, "")
        if is_split == 0:
            owner_store(
                accs[q],
                z,
                q,
                bias_ptr,
                pid_n,
                c_bases[q],
                c_offsets,
                BLOCK_N,
                HALF_N,
                ADD_BIAS,
                out_dtype,
                gStoreLayoutC,
                mfmaLayout,
                m_left,
                n_left,
                EVEN_M,
                EVEN_N,
            )
    if n_tile > 1:
        drain_and_barrier()
        is_reader = j < r_tile
        l2_tile_barrier(
            sync_ptr, rank - j, n_tile, r_tile, j, is_reader, NUM_PROGRAMS, NUM_XCDS
        )
        if is_reader:
            reduce_scatter_slice(
                p_ptr,
                bias_ptr,
                c_bases[0] + pid_n * BLOCK_N * stride_cn,
                rank - j,
                j,
                n_tile,
                r_tile,
                pid_n,
                stride_cm,
                stride_cn,
                BLOCK_M,
                BLOCK_N,
                ADD_BIAS,
                RS_NB,
                RS_U,
                out_dtype,
                ".cg",
                m_left,
                n_left,
                EVEN_M,
                EVEN_N,
            )


@gluon.jit
def quadrant_segment(
    tile_id,
    k0,
    n_pairs,
    is_partial,
    n_peers,
    rank,
    j,
    n_tile,
    r_tile,
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    p_ptr,
    locks_ptr,
    sync_ptr,
    smem,
    kofs,
    strides,
    M,
    N,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    PEER_STEP: gl.constexpr,
    REDUCE_SCATTER: gl.constexpr,
    RS_NB: gl.constexpr,
    RS_U: gl.constexpr,
    L2: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    mfmaLayout: gl.constexpr,
    gLoadLayoutA: gl.constexpr,
    gLoadLayoutB: gl.constexpr,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
    EVEN_K: gl.constexpr,
    K_PAD: gl.constexpr,
    K_RES: gl.constexpr,
    SEG_LOOP: gl.constexpr,
    OWNER_QPB: gl.constexpr,
    OWNER_MAX_PEERS: gl.constexpr,
    PREFETCHED: gl.constexpr = False,
    a_top_in=None,
    b_left_in=None,
):
    """K-steps [k0, k0 + 2 n_pairs) of tile tile_id (n_pairs >= 1), from its own
    prologue, then the segment's ending: XCD-local reduce-scatter (l2_rs_ending, L2),
    reduce-scatter (rs_ending) or the owner ending (owner_ending). rank is the
    program's workspace slot, j its chunk (role 0 owns the tile), n_tile / r_tile the
    tile's programs and readers.

    Pair p of a tile (K-steps 2p, 2p + 1) is at columns 2 BLOCK_K (p - 1) for p >= 1,
    and pair 0 is the last pair, at K_PAD - 2 BLOCK_K, loaded on a ragged K with the
    lanes past K masked. So only the segment
    that starts at k0 = 0 masks, in its prologue. Every pair runs through the
    main-loop body; the last one's prefetch reloads the last pair (clamped), which the
    drain discards.

    SEG_LOOP: the caller runs segments in a loop (the end-to-end path), so the C row
    offsets are made tile-dependent (clamped to the tile's last row); otherwise the
    compiler hoists them above the loop and holds them, spilled, across it.

    PREFETCHED: the first pair is already in flight in the state load_first_pair
    leaves (the last persistent trip's epilogue loaded it), with its first operands in
    a_top_in / b_left_in."""
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn = strides
    a_offsets, a_bot_shift, b_offsets, b_right_shift = kofs
    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfmaLayout, k_width=8
    )
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfmaLayout, k_width=8
    )
    pid_m, pid_n = _get_pids(tile_id, M, N, BLOCK_M, BLOCK_N, 0, 1, GROUP_SIZE_M)
    m_left = M - pid_m * BLOCK_M
    n_left = gl.multiple_of(N - pid_n * BLOCK_N, 8)

    a_addr = a_load_addressing(
        pid_m,
        a_offsets,
        a_bot_shift,
        M,
        stride_am,
        stride_ak,
        BLOCK_M,
        BLOCK_K,
        gLoadLayoutA,
        EVEN_M,
    )
    b_addr = b_load_addressing(
        pid_n,
        b_offsets,
        b_right_shift,
        N,
        stride_bk,
        stride_bn,
        BLOCK_N,
        BLOCK_K,
        gLoadLayoutB,
        EVEN_N,
    )

    # Pair i of the segment (i >= 1) at a_tile + 2 BLOCK_K i; its first pair at a_tile,
    # or for k0 = 0 the last pair (partial and masked at K on a ragged K).
    a_row = a_ptr + pid_m.to(gl.int64) * BLOCK_M * stride_am
    b_row = b_ptr + pid_n.to(gl.int64) * BLOCK_N * stride_bn
    a_tile = a_row + (k0 - 2).to(gl.int64) * BLOCK_K * stride_ak
    b_tile = b_row + (k0 - 2).to(gl.int64) * BLOCK_K * stride_bk
    if PREFETCHED:
        a_top, b_left = a_top_in, b_left_in
    else:
        first_col = gl.where(
            k0 == 0, K_PAD - 2 * BLOCK_K, (k0 - 2) * BLOCK_K
        ).to(gl.int64)
        if EVEN_K:
            mask_a0, mask_b0, mask_a1, mask_b1 = None, None, None, None
        else:
            k_hi = gl.where(k0 == 0, 2 * BLOCK_K - K_RES, 2 * BLOCK_K)
            mask_a0, mask_b0, mask_a1, mask_b1 = tail_pair_masks(
                k_hi, K_RES, BLOCK_M, BLOCK_N, BLOCK_K, gLoadLayoutA, gLoadLayoutB
            )
        a_top, b_left = load_first_pair(
            a_row + first_col * stride_ak,
            b_row + first_col * stride_bk,
            smem,
            a_addr,
            b_addr,
            dotOpLayoutA,
            dotOpLayoutB,
            mask_a0,
            mask_b0,
            mask_a1,
            mask_b1,
        )
    # The last pair's clamped prefetch must stay inside the row: for k0 = 0, a_tile
    # itself is a pair before it.
    nxt_lo = (k0 == 0).to(gl.int64) * 2

    # The first pair is peeled: the constant-zero accumulator becomes the MFMA's
    # inline-0 C operand, so no AGPRs are zeroed per segment (a loop-carried zero
    # would be materialised).
    accs = init_acc(bias_ptr, pid_n, BLOCK_M, BLOCK_N, mfmaLayout, False)
    nxt = gl.maximum(gl.minimum(1, n_pairs - 1).to(gl.int64) * 2, nxt_lo)
    accs, a_top, b_left = k_step_pair(
        accs,
        a_top,
        b_left,
        smem,
        a_tile + nxt * BLOCK_K * stride_ak,
        b_tile + nxt * BLOCK_K * stride_bk,
        a_addr,
        b_addr,
        dotOpLayoutA,
        dotOpLayoutB,
    )
    for i in range(1, n_pairs):
        nxt = gl.maximum(gl.minimum(i + 1, n_pairs - 1).to(gl.int64) * 2, nxt_lo)
        accs, a_top, b_left = k_step_pair(
            accs,
            a_top,
            b_left,
            smem,
            a_tile + nxt * BLOCK_K * stride_ak,
            b_tile + nxt * BLOCK_K * stride_bk,
            a_addr,
            b_addr,
            dotOpLayoutA,
            dotOpLayoutB,
        )
    # The clamped prefetch is still in flight, and the next segment's prologue
    # overwrites the buffers other waves may still be reading.
    drain_and_barrier()

    STORE_CONTIG: gl.constexpr = 8
    STORE_LANES_N: gl.constexpr = (BLOCK_N // 2) // STORE_CONTIG
    STORE_LANES_M: gl.constexpr = 64 // STORE_LANES_N
    gStoreLayoutC: gl.constexpr = gl.BlockedLayout(
        [1, STORE_CONTIG], [STORE_LANES_M, STORE_LANES_N], [4, 1], [1, 0]
    )
    out_dtype: gl.constexpr = c_ptr.dtype.element_ty
    offs_cm = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gStoreLayoutC))
    c_tl_base = c_ptr + pid_m.to(gl.int64) * BLOCK_M * stride_cm
    col = pid_n * BLOCK_N + offs_cn
    if SEG_LOOP:
        c_rows = gl.multiple_of(stride_cm * gl.minimum(offs_cm, m_left - 1), 8)
    else:
        c_rows = gl.multiple_of(stride_cm * offs_cm, 8)
    c_offsets = c_rows[:, None] + stride_cn * col[None, :]
    c_tr_base = c_tl_base + BLOCK_N * stride_cn // 2
    c_bl_base = c_tl_base + BLOCK_M * stride_cm // 2
    c_br_base = c_bl_base + BLOCK_N * stride_cn // 2
    c_bases = (c_tl_base, c_bl_base, c_tr_base, c_br_base)

    if L2:
        l2_rs_ending(
            accs,
            rank,
            j,
            n_tile,
            r_tile,
            pid_n,
            bias_ptr,
            p_ptr,
            sync_ptr,
            c_bases,
            c_offsets,
            stride_cm,
            stride_cn,
            m_left,
            n_left,
            BLOCK_M,
            BLOCK_N,
            ADD_BIAS,
            RS_NB,
            RS_U,
            NUM_PROGRAMS,
            NUM_XCDS,
            out_dtype,
            gStoreLayoutC,
            mfmaLayout,
            EVEN_M,
            EVEN_N,
        )
    elif REDUCE_SCATTER:
        rs_ending(
            accs,
            rank,
            j,
            n_tile,
            r_tile,
            pid_n,
            bias_ptr,
            p_ptr,
            sync_ptr,
            c_bases,
            c_offsets,
            stride_cm,
            stride_cn,
            m_left,
            n_left,
            BLOCK_M,
            BLOCK_N,
            ADD_BIAS,
            RS_NB,
            RS_U,
            out_dtype,
            gStoreLayoutC,
            mfmaLayout,
            EVEN_M,
            EVEN_N,
        )
    else:
        owner_ending(
            accs,
            is_partial,
            n_peers,
            rank,
            pid_n,
            bias_ptr,
            p_ptr,
            locks_ptr,
            c_bases,
            c_offsets,
            m_left,
            n_left,
            BLOCK_N,
            PEER_STEP,
            ADD_BIAS,
            out_dtype,
            gStoreLayoutC,
            mfmaLayout,
            EVEN_M,
            EVEN_N,
            OWNER_QPB,
            OWNER_MAX_PEERS,
        )


MFMA_SCHEDULE_HINT = "mfma-schedule"


def schedule_hint():
    """The schedule_hint the kernel launches with: the MFMA scheduler, unless
    AITER_MFMA_SCHED=0 (an unscheduled A/B run)."""
    return "" if os.environ.get("AITER_MFMA_SCHED", "1") == "0" else MFMA_SCHEDULE_HINT


def _validate_tensors(a, b, c, bias, tile):
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("A and B must be rank-2 tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError("incompatible GEMM dimensions")
    if a.device != b.device or c.device != a.device:
        raise ValueError("A, B and C must be on the same device")
    if a.dtype != b.dtype or c.dtype != a.dtype:
        raise ValueError("A, B and C must have the same dtype")
    if a.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("v30 supports fp16 and bf16 inputs")
    if a.stride(1) != 1:
        raise ValueError("A must be row-major/K-contiguous")
    if b.stride(0) != 1:
        raise ValueError("B must be column-major/K-contiguous")
    if c.stride(1) != 1:
        raise ValueError("C must be row-major")
    if a.stride(0) % 8 or b.stride(1) % 8 or c.stride(0) % 8:
        raise ValueError("A, B and C leading strides must be multiples of 8 elements")
    if any(tensor.data_ptr() % 16 for tensor in (a, b, c)):
        raise ValueError("A, B and C data pointers must be 16-byte aligned")
    if bias is not None:
        if bias.device != a.device or bias.dtype != a.dtype:
            raise ValueError("bias must match A's device and dtype")
        if bias.shape != (b.shape[1],) or not bias.is_contiguous():
            raise ValueError("bias must be contiguous with shape (N,)")
    strides = (*a.stride(), *b.stride(), *c.stride())
    if any(stride < 0 for stride in strides):
        raise ValueError("negative strides are unsupported")

    def fits(*spans, tensor):
        return sum(spans) * tensor.element_size() < 2**31 - 2

    if not fits(tile.block_m * a.stride(0), 2 * tile.block_k * a.stride(1), tensor=a):
        raise ValueError("A's tile-local 32-bit byte offsets overflow")
    if not fits(2 * tile.block_k * b.stride(0), tile.block_n * b.stride(1), tensor=b):
        raise ValueError("B's tile-local 32-bit byte offsets overflow")
    if not fits(tile.block_m * c.stride(0), c.shape[1] * c.stride(1), tensor=c):
        raise ValueError("C's tile-local 32-bit byte offsets overflow")


def _launch_constants(M, N, K, config, num_programs, num_xcds):
    """Parse an exact config and derive only shape/topology constants."""
    config = StreamKConfig.from_dict(config)
    tile = config.tile
    total_tiles = triton.cdiv(M, tile.block_m) * triton.cdiv(N, tile.block_n)
    k_pad = triton.cdiv(K, 2 * tile.block_k) * 2 * tile.block_k
    pairs = k_pad // (2 * tile.block_k)
    leftover = total_tiles % num_programs

    streamk_tiles = 0 if config.policy == "dp" else leftover
    if config.policy == "two_tile":
        if leftover == 0 or total_tiles < num_programs:
            raise ValueError(
                "two_tile requires a partial wave after at least one full wave"
            )
        streamk_tiles += num_programs
    if config.policy in ("one_tile", "spread") and leftover == 0:
        raise ValueError(f"{config.policy} requires leftover tiles")
    if config.policy == "spread" and 2 * leftover <= num_programs:
        raise ValueError(
            "spread requires more than half of the resident programs "
            "as leftover tiles"
        )

    end_to_end = config.policy in ("two_tile", "spread")
    l2 = config.exchange == "l2"
    xcd_split = 1
    if config.policy == "one_tile":
        split_n = config.split_n
        split_cap = min(num_programs // streamk_tiles, pairs)
        if split_cap < 2:
            raise ValueError(
                "shape leaves fewer than two programs/pairs per Stream-K tile"
            )
        if split_n > split_cap:
            raise ValueError(
                f"split_n={split_n} exceeds this shape's maximum {split_cap}"
            )
        if l2:
            per_xcd_tiles = triton.cdiv(streamk_tiles, num_xcds)
            xcd_split = max(
                1,
                min(
                    split_n,
                    pairs,
                    (num_programs // num_xcds) // per_xcd_tiles,
                ),
            )
            if xcd_split != split_n:
                raise ValueError(
                    f"split_n={split_n} cannot be placed within one XCD; "
                    f"maximum is {xcd_split}"
                )
        streamk_programs = streamk_tiles * split_n
    elif end_to_end:
        split_n = 1
        streamk_programs = num_programs
        if streamk_tiles * pairs < streamk_programs:
            raise ValueError(
                f"{config.policy} has fewer pair units than "
                f"the {num_programs} programs"
            )
    else:
        split_n = 1
        streamk_programs = 0

    compact = (
        streamk_tiles > 0
        and streamk_tiles == total_tiles
        and not l2
        and streamk_programs < num_programs
    )
    # Data-parallel launches have no Stream-K phase to keep resident, so a grid of
    # fewer tiles than programs launches only the programs with a tile.
    if compact:
        grid = streamk_programs
    elif streamk_tiles == 0:
        grid = min(num_programs, total_tiles)
    else:
        grid = num_programs
    rs_split = split_n if config.policy == "one_tile" else 1
    readers = min(1 << (rs_split.bit_length() - 1), tile.blocks)
    blocks_per_group = max(
        1,
        min(
            tile.blocks // readers,
            16,
            tile.quadrant_blocks,
        ),
    )
    return config, {
        "TOTAL_TILES": total_tiles,
        "STREAMK_TILES": streamk_tiles,
        "STREAMK_PROGRAMS": streamk_programs,
        "RESOLVED_SPLIT_N": split_n,
        "END_TO_END": end_to_end,
        "L2": l2,
        "XCD_SPLIT": xcd_split,
        "COMPACT": compact,
        "GRID": grid,
        "RS_NB": blocks_per_group,
        "RS_U": max(16 // blocks_per_group, 1),
        "RS_R": readers,
    }


def gemm_a16w16(
    x,
    w,
    bias=None,
    dtype=torch.bfloat16,
    y=None,
    config=None,
    persistent_partials=False,
):
    """Run ``Y = X @ W.T (+ bias)`` with an explicit Stream-K ``config``.

    ``persistent_partials`` keeps one full-grid fp32 partial buffer sized for
    the largest tile per (device, stream) instead of allocating per call.
    """
    if x.ndim != 2 or w.ndim != 2:
        raise ValueError("x and w must be rank-2 tensors")
    M, K = x.shape
    N, w_k = w.shape
    if K != w_k:
        raise ValueError("x and w must have the same K dimension")
    if M <= 0 or N <= 0:
        raise ValueError("v30 requires positive M and N")
    if K % 16:
        raise ValueError(f"K={K} must be a multiple of 16")
    if N % 8:
        raise ValueError(f"N={N} must be a multiple of 8")
    if dtype != x.dtype:
        raise ValueError(f"v30 writes x's dtype {x.dtype}, not {dtype}")
    if config is None:
        raise ValueError("v30 needs an explicit Stream-K config")
    num_programs, num_xcds = streamk_topology()
    config, launch = _launch_constants(M, N, K, config, num_programs, num_xcds)
    tile = config.tile
    k_pad = triton.cdiv(K, 2 * tile.block_k) * 2 * tile.block_k
    if k_pad < 4 * tile.block_k:
        raise ValueError(f"K={K} needs at least {4 * tile.block_k} after pair padding")
    if y is None:
        y = torch.empty((M, N), device=x.device, dtype=dtype)
    elif y.shape != (M, N):
        raise ValueError(f"y must have shape {(M, N)}")
    b = w.T  # (K, N) view, K contiguous: the layout the kernel loads
    _validate_tensors(x, b, y, bias, tile)

    # Each ragged dimension specializes its own clamped loads and masked stores.
    even_m = M % tile.block_m == 0
    even_n = N % tile.block_n == 0
    even_k = K == k_pad
    partials, locks, sync = streamk_workspace(
        x.device,
        num_programs,
        num_xcds,
        tile.block_m,
        tile.block_n,
        launch["STREAMK_PROGRAMS"],
        launch["L2"],
        persistent_partials,
    )
    owner_qpb = min(4, 16 // max(1, tile.block_m * tile.block_n // 4096))
    total_full_tiles = launch["TOTAL_TILES"] - launch["STREAMK_TILES"]
    has_full = total_full_tiles > 0
    # The persistent prologue loads program_id's first tile unclamped.
    assert not has_full or launch["GRID"] <= total_full_tiles
    _gemm_a16w16_compute_bound_kernel[(launch["GRID"], 1)](
        x,
        b,
        y,
        bias,
        partials,
        locks,
        sync,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        b.stride(0),
        b.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_M=tile.block_m,
        BLOCK_N=tile.block_n,
        BLOCK_K=tile.block_k,
        NUM_PROGRAMS=num_programs,
        NUM_XCDS=num_xcds,
        GROUP_SIZE_M=config.group_size_m,
        ADD_BIAS=bias is not None,
        EVEN_M=even_m,
        EVEN_N=even_n,
        EVEN_K=even_k,
        STREAMK_TILES=launch["STREAMK_TILES"],
        SPLIT_N=launch["RESOLVED_SPLIT_N"],
        REDUCE_SCATTER=config.fixup == "rs",
        RS_NB=launch["RS_NB"],
        RS_U=launch["RS_U"],
        RS_R=launch["RS_R"],
        END_TO_END=launch["END_TO_END"],
        L2=launch["L2"],
        XCD_SPLIT=launch["XCD_SPLIT"],
        HAS_FULL=has_full,
        GRID=launch["GRID"] if launch["COMPACT"] else 0,
        OWNER_QPB=owner_qpb,
        num_warps=NUM_WARPS,
        schedule_hint=schedule_hint(),
    )
    return y
