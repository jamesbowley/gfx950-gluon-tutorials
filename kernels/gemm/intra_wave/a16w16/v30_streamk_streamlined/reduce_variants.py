# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Experiment: two alternatives to the one-tile reduce-scatter ending (WORKLOG entry 15).

A copy of v30's kernel restricted to the persistent phase and the uniform one-tile
tail (wt exchange), with the ending chosen by FIXUP:

* ``rs`` (0): v30's reduce-scatter ending, unchanged (the control).
* ``tree`` (1): a radix-R tree over the tile's n chunks. Chunk j's children are
  j + i R^k (i = 1 .. R-1) for every level k with j % R^(k+1) == 0 and j + i R^k < n.
  A chunk waits for its children's flags, adds their partials to its accumulator
  and, unless it is chunk 0, stores the sum as its own partial and raises its flag
  (each chunk sends once, so one lock per program). Chunk 0 stores C. Receivers
  lower their children's flags, as the owner ending does.
* ``rs_all`` (3, WORKLOG entry 16): rs with R = min(blocks, pow2_floor(P / S))
  readers per tile instead of at most n. The E = R - n extra readers of each tile
  are programs the tail leaves idle (hardware pids from S n, more programs on a
  compact launch); they arrive on the tile's barrier at once, so it counts n + E.
* ``store_only`` (2): every chunk stores its partial and exits. ``separate`` is
  this GEMM followed by _streamk_reduce_kernel; alone it is a lower bound on any
  ending (its C tail tiles are never written).

Every other helper is imported from matmul_kernel, so the K loop, the persistent
phase and the partial layout are the production ones.
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import cdna4 as ttgl_cdna4

from .hardware import streamk_topology
from .matmul_kernel import (
    NUM_WARPS,
    _launch_constants,
    _validate_tensors,
    a_load_addressing,
    asm_fence,
    b_load_addressing,
    coalesced_store_layout,
    compute_gload_layout,
    drain_and_barrier,
    even_split_size,
    even_split_start,
    group_m_tile_coords,
    init_acc,
    k_step_pair,
    lane_contiguous_offsets,
    load_first_pair,
    owner_store,
    persistent_tile,
    reduce_scatter_slice,
    rs_ending,
    schedule_hint,
    store_partial,
    streamk_first_segment,
    tail_pair_masks,
    vpid_tile_coords,
    xcd_blocked_rank,
)
from .runtime import streamk_workspace
from .selection import reduce_scatter_config

FIXUPS = {
    "rs": 0,
    "tree": 1,
    "store_only": 2,
    "separate": 2,
    "reduce_only": 2,
    "rs_all": 3,
}


@gluon.jit
def _gemm_variant_kernel(
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
    RS_NB: gl.constexpr,
    RS_U: gl.constexpr,
    RS_R: gl.constexpr,
    FIXUP: gl.constexpr,
    TREE_B: gl.constexpr,
    TREE_LEVELS: gl.constexpr,
    TREE_STATIC: gl.constexpr,
    STORE_CACHE: gl.constexpr,
    HAS_FULL: gl.constexpr = True,
    GRID: gl.constexpr = 0,
    OWNER_QPB: gl.constexpr = 1,
    RS_ALL_E: gl.constexpr = 0,
    POLL_SLEEP: gl.constexpr = 0,
    BAR_STRIDE: gl.constexpr = 0,
):
    """_gemm_a16w16_compute_bound_kernel's persistent phase and uniform one-tile tail,
    with the tail's ending chosen by FIXUP (module docstring)."""
    gl.static_assert(STREAMK_TILES > 0)
    start = gl.program_id(axis=0)
    num_tiles_m = gl.cdiv(M, BLOCK_M)
    num_tiles_n = gl.cdiv(N, BLOCK_N)
    total_tiles = num_tiles_m * num_tiles_n
    total_full_tiles = total_tiles - STREAMK_TILES
    strides = (stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn)

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
    a_rows_top = gl.multiple_of(offs_am * stride_am, 8)
    b_cols_left = gl.multiple_of(offs_bn * stride_bn, 8)
    a_offsets = a_rows_top[:, None] + offs_ak[None, :] * stride_ak
    b_offsets = offs_bk[:, None] * stride_bk + b_cols_left[None, :]
    a_bot_shift = BLOCK_M // 2 * stride_am
    b_right_shift = BLOCK_N // 2 * stride_bn
    kofs = (a_offsets, a_bot_shift, b_offsets, b_right_shift)

    if EVEN_K:
        gl.static_assert(K % (2 * BLOCK_K) == 0)
        K_PAD: gl.constexpr = K
    else:
        K_PAD: gl.constexpr = gl.cdiv(K, 2 * BLOCK_K) * (2 * BLOCK_K)
    K_RES: gl.constexpr = K_PAD - K
    iterMax = gl.cdiv(K_PAD, BLOCK_K)

    gl.assume(iterMax > 3)
    PAIRS_PER_TILE: gl.constexpr = K_PAD // (2 * BLOCK_K)
    SK_PREFETCH: gl.constexpr = HAS_FULL

    if HAS_FULL:
        sk_has, sk_tile, sk_k0 = streamk_first_segment(
            total_full_tiles,
            STREAMK_TILES,
            PAIRS_PER_TILE,
            SPLIT_N,
            False,
            False,
            NUM_PROGRAMS,
            NUM_XCDS,
        )
        sk_m, sk_n = group_m_tile_coords(sk_tile, M, N, BLOCK_M, BLOCK_N, GROUP_SIZE_M)
        sk_col = gl.where(
            sk_k0 == 0, K_PAD - 2 * BLOCK_K, (sk_k0 - 2) * BLOCK_K
        ).to(gl.int64)
        sk_khi = gl.where(sk_k0 == 0, 2 * BLOCK_K - K_RES, 2 * BLOCK_K) * sk_has
        sk_target = (sk_m, sk_n, sk_col, sk_khi)
        first_m, first_n = vpid_tile_coords(
            start, M, N, BLOCK_M, BLOCK_N, total_full_tiles, NUM_XCDS, GROUP_SIZE_M
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
    else:
        gl.amd.cdna4.async_copy.wait_group(0)
        a_top = 0
        b_left = 0

    WORKING: gl.constexpr = STREAMK_TILES * SPLIT_N
    NBLK: gl.constexpr = BLOCK_M * BLOCK_N // 1024
    gl.static_assert(SPLIT_N <= PAIRS_PER_TILE)
    if FIXUP == 3:
        # rs_all: RS_R readers per tile, the tile's own min(SPLIT_N, RS_R) and RS_ALL_E
        # idle programs.
        gl.static_assert(RS_R <= NBLK)
        gl.static_assert(RS_ALL_E == 0 or RS_R == SPLIT_N + RS_ALL_E)
    else:
        gl.static_assert(RS_R > 0 and RS_R <= SPLIT_N)
        gl.static_assert(
            2 * RS_R > SPLIT_N or RS_R == NBLK, "RS_R does not match SPLIT_N"
        )
    gl.static_assert((RS_R & (RS_R - 1)) == 0)
    gl.static_assert(RS_NB == 1 or RS_NB * RS_R <= NBLK)
    if GRID > 0 and FIXUP != 3:
        rank = xcd_blocked_rank(GRID, NUM_XCDS)
    else:
        rank = xcd_blocked_rank(WORKING, NUM_XCDS, IDLE=True)
    active = rank < WORKING
    t = rank // SPLIT_N
    j = rank % SPLIT_N
    if active:
        quadrant_segment_variant(
            total_full_tiles + t,
            even_split_start(j, PAIRS_PER_TILE, SPLIT_N) * 2,
            even_split_size(j, PAIRS_PER_TILE, SPLIT_N),
            rank,
            j,
            SPLIT_N,
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
            RS_NB,
            RS_U,
            mfmaLayout,
            gLoadLayoutA,
            gLoadLayoutB,
            EVEN_M,
            EVEN_N,
            EVEN_K,
            K_PAD,
            K_RES,
            OWNER_QPB,
            FIXUP,
            TREE_B,
            TREE_LEVELS,
            TREE_STATIC,
            STORE_CACHE,
            RS_ALL_E,
            POLL_SLEEP,
            BAR_STRIDE,
            SK_PREFETCH,
            a_top,
            b_left,
        )
    if FIXUP == 3 and RS_ALL_E > 0:
        # Programs without a segment: hardware pids WORKING .. (IDLE ranks).
        e = gl.program_id(axis=0) - WORKING
        if (e >= 0) & (e < STREAMK_TILES * RS_ALL_E):
            # The last trip's prefetch (unused here) may still be in flight into the
            # LDS the slice's layout conversions use.
            gl.amd.cdna4.async_copy.wait_group(0)
            rs_all_extra_reader(
                e // RS_ALL_E,
                SPLIT_N + e % RS_ALL_E,
                total_full_tiles,
                p_ptr,
                bias_ptr,
                c_ptr,
                sync_ptr,
                M,
                N,
                stride_cm,
                stride_cn,
                BLOCK_M,
                BLOCK_N,
                GROUP_SIZE_M,
                SPLIT_N,
                RS_R,
                RS_ALL_E,
                RS_NB,
                RS_U,
                ADD_BIAS,
                EVEN_M,
                EVEN_N,
                POLL_SLEEP,
                BAR_STRIDE,
            )


@gluon.jit
def tile_barrier_poll(sync_ptr, b, n, wait, SLEEP: gl.constexpr):
    """matmul_kernel.tile_barrier, with SLEEP 1 / 2 / 3 an s_sleep of 1 / 8 / 32
    (x 64 clocks) in the poll."""
    cnt_ptr = sync_ptr + 2 * b
    gen_ptr = cnt_ptr + 1
    g0 = gl.atomic_add(gen_ptr, 0, sem="relaxed", scope="gpu")
    asm_fence("s_waitcnt vmcnt(0)")
    arrived = gl.atomic_add(cnt_ptr, 1, sem="relaxed", scope="gpu")
    if arrived == n - 1:
        gl.atomic_xchg(cnt_ptr, 0, sem="relaxed", scope="gpu")
        gl.atomic_xchg(gen_ptr, g0 + 1, sem="relaxed", scope="gpu")
    elif wait:
        while gl.atomic_add(gen_ptr, 0, sem="relaxed", scope="gpu") == g0:
            if SLEEP == 1:
                asm_fence("s_sleep 1")
            elif SLEEP == 2:
                asm_fence("s_sleep 8")
            elif SLEEP == 3:
                asm_fence("s_sleep 32")
    asm_fence("s_nop 0")  # no partial read moves above the wait


@gluon.jit
def rs_all_ending(
    accs,
    rank,
    j,
    pid_n,
    bias_ptr,
    p_ptr,
    sync_ptr,
    c_tl_base,
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
    mfmaLayout: gl.constexpr,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
    SPLIT_N: gl.constexpr,
    R: gl.constexpr,
    E: gl.constexpr,
    POLL_SLEEP: gl.constexpr,
    BAR_STRIDE: gl.constexpr,
):
    """rs_ending of chunk j with R readers per tile: the tile's barrier also counts its
    E extra readers, and chunk j reads slice j if j < R. Tile t's barrier is slot
    t BAR_STRIDE (0: t SPLIT_N, rs's slot)."""
    HALF_M: gl.constexpr = BLOCK_M // 2
    HALF_N: gl.constexpr = BLOCK_N // 2
    p_offsets = lane_contiguous_offsets(HALF_M, HALF_N, mfmaLayout)
    for q in gl.static_range(4):
        store_partial(accs[q], q, p_ptr, rank, p_offsets, HALF_M, HALF_N)
    drain_and_barrier()
    is_reader = j < R
    BS: gl.constexpr = SPLIT_N if BAR_STRIDE == 0 else BAR_STRIDE
    tile_barrier_poll(
        sync_ptr, (rank - j) // SPLIT_N * BS, SPLIT_N + E, is_reader, POLL_SLEEP
    )
    if is_reader:
        reduce_scatter_slice(
            p_ptr,
            bias_ptr,
            c_tl_base + pid_n * BLOCK_N * stride_cn,
            rank - j,
            j,
            SPLIT_N,
            R,
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
def rs_all_extra_reader(
    t,
    jr,
    total_full_tiles,
    p_ptr,
    bias_ptr,
    c_ptr,
    sync_ptr,
    M,
    N,
    stride_cm,
    stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    SPLIT_N: gl.constexpr,
    R: gl.constexpr,
    E: gl.constexpr,
    NB: gl.constexpr,
    U: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
    POLL_SLEEP: gl.constexpr,
    BAR_STRIDE: gl.constexpr,
):
    """An idle program as reader jr (SPLIT_N <= jr < R) of Stream-K tile t: it arrives
    on the tile's barrier at once (every waiter arrives, so none can read the
    generation after the bump), waits for the producers and reduces its slice."""
    BS: gl.constexpr = SPLIT_N if BAR_STRIDE == 0 else BAR_STRIDE
    tile_barrier_poll(sync_ptr, t * BS, SPLIT_N + E, True, POLL_SLEEP)
    pid_m, pid_n = group_m_tile_coords(
        total_full_tiles + t, M, N, BLOCK_M, BLOCK_N, GROUP_SIZE_M
    )
    m_left = M - pid_m * BLOCK_M
    n_left = gl.multiple_of(N - pid_n * BLOCK_N, 8)
    c_tile = (
        c_ptr
        + pid_m.to(gl.int64) * BLOCK_M * stride_cm
        + pid_n.to(gl.int64) * BLOCK_N * stride_cn
    )
    reduce_scatter_slice(
        p_ptr,
        bias_ptr,
        c_tile,
        t * SPLIT_N,
        jr,
        SPLIT_N,
        R,
        pid_n,
        stride_cm,
        stride_cn,
        BLOCK_M,
        BLOCK_N,
        ADD_BIAS,
        NB,
        U,
        c_ptr.dtype.element_ty,
        ".cv",
        m_left,
        n_left,
        EVEN_M,
        EVEN_N,
    )


@gluon.jit
def tree_child_live(j, k, off, n: gl.constexpr, B: gl.constexpr):
    """Whether chunk j + off (off = i R^k at level k, R = 2^B) is a child of chunk j
    among n chunks."""
    return ((j & ((1 << (B * (k + 1))) - 1)) == 0) & (j + off < n)


@gluon.jit
def tree_wait(locks_ptr, rank, j, k, off, n: gl.constexpr, B: gl.constexpr):
    if tree_child_live(j, k, off, n, B):
        while gl.atomic_add(locks_ptr + rank + off, 0, sem="relaxed", scope="gpu") == 0:
            pass


@gluon.jit
def tree_child_sums(
    p_ptr, rank, j, k, off, p_offsets, s, q0: gl.constexpr, QPB: gl.constexpr,
    QUAD: gl.constexpr, n: gl.constexpr, B: gl.constexpr,
):
    """s (QPB quadrant sums from q0) plus child rank + off's quadrants, masked off
    when it is not a live child."""
    SLOT: gl.constexpr = 4 * QUAD
    live = tree_child_live(j, k, off, n, B)
    base = p_ptr + (rank + off).to(gl.int64) * SLOT
    s0, s1, s2, s3 = s
    s0 += gl.amd.cdna3.buffer_load(
        ptr=base + q0 * QUAD, offsets=p_offsets, mask=live, other=0.0, cache=".cv"
    )
    if QPB > 1:
        s1 += gl.amd.cdna3.buffer_load(
            ptr=base + (q0 + 1) * QUAD,
            offsets=p_offsets,
            mask=live,
            other=0.0,
            cache=".cv",
        )
    if QPB > 2:
        s2 += gl.amd.cdna3.buffer_load(
            ptr=base + (q0 + 2) * QUAD,
            offsets=p_offsets,
            mask=live,
            other=0.0,
            cache=".cv",
        )
        s3 += gl.amd.cdna3.buffer_load(
            ptr=base + (q0 + 3) * QUAD,
            offsets=p_offsets,
            mask=live,
            other=0.0,
            cache=".cv",
        )
    return s0, s1, s2, s3


@gluon.jit
def tree_ending(
    accs,
    rank,
    j,
    pid_n,
    bias_ptr,
    p_ptr,
    locks_ptr,
    c_bases,
    c_offsets,
    m_left,
    n_left,
    BLOCK_N: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    out_dtype: gl.constexpr,
    gStoreLayoutC: gl.constexpr,
    mfmaLayout: gl.constexpr,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
    QPB: gl.constexpr,
    SPLIT_N: gl.constexpr,
    TREE_B: gl.constexpr,
    TREE_LEVELS: gl.constexpr,
    STATIC: gl.constexpr,
):
    """The tree ending of chunk j of SPLIT_N (slot rank; module docstring), radix
    2^TREE_B over TREE_LEVELS levels. Children are read QPB quadrants at a time, as
    owner_ending reads its peers: STATIC unrolls them with masked loads (several
    children's loads in flight), otherwise a run-time loop reads one at a time."""
    HALF_M: gl.constexpr = accs[0].shape[0]
    HALF_N: gl.constexpr = accs[0].shape[1]
    QUAD: gl.constexpr = HALF_M * HALF_N
    PER: gl.constexpr = (1 << TREE_B) - 1
    NCH: gl.constexpr = TREE_LEVELS * PER

    if STATIC:
        for c in gl.static_range(NCH):
            tree_wait(
                locks_ptr, rank, j, c // PER, (c % PER + 1) << (TREE_B * (c // PER)),
                SPLIT_N, TREE_B,
            )
    else:
        for c in range(NCH):
            k = c // PER
            tree_wait(
                locks_ptr, rank, j, k, (c % PER + 1) << (TREE_B * k), SPLIT_N, TREE_B
            )
    asm_fence("s_nop 0")  # no partial read moves above the waits

    p_offsets = lane_contiguous_offsets(HALF_M, HALF_N, mfmaLayout)
    z = gl.zeros((HALF_M, HALF_N), gl.float32, mfmaLayout)
    for q0 in gl.static_range(0, 4, QPB):
        sums = (z, z, z, z)
        if STATIC:
            for c in gl.static_range(NCH):
                sums = tree_child_sums(
                    p_ptr, rank, j, c // PER, (c % PER + 1) << (TREE_B * (c // PER)),
                    p_offsets, sums, q0, QPB, QUAD, SPLIT_N, TREE_B,
                )
        else:
            for c in range(NCH):
                k = c // PER
                sums = tree_child_sums(
                    p_ptr, rank, j, k, (c % PER + 1) << (TREE_B * k),
                    p_offsets, sums, q0, QPB, QUAD, SPLIT_N, TREE_B,
                )
        for i in gl.static_range(QPB):
            if j == 0:
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
            else:
                store_partial(
                    accs[q0 + i] + sums[i], q0 + i, p_ptr, rank, p_offsets, HALF_M, HALF_N
                )

    if j != 0:
        # The flag may only become visible after all four waves' partial stores have
        # completed.
        drain_and_barrier()
        gl.atomic_xchg(locks_ptr + rank, 1, sem="relaxed", scope="gpu")
    for c in range(NCH):
        k = c // PER
        off = (c % PER + 1) << (TREE_B * k)
        if tree_child_live(j, k, off, SPLIT_N, TREE_B):
            gl.atomic_xchg(locks_ptr + rank + off, 0, sem="relaxed", scope="gpu")


@gluon.jit
def store_only_ending(
    accs,
    rank,
    p_ptr,
    mfmaLayout: gl.constexpr,
    CACHE: gl.constexpr,
):
    """Every chunk stores its partial to P[rank]; the kernel boundary publishes it."""
    HALF_M: gl.constexpr = accs[0].shape[0]
    HALF_N: gl.constexpr = accs[0].shape[1]
    p_offsets = lane_contiguous_offsets(HALF_M, HALF_N, mfmaLayout)
    for q in gl.static_range(4):
        store_partial(accs[q], q, p_ptr, rank, p_offsets, HALF_M, HALF_N, CACHE)


@gluon.jit
def quadrant_segment_variant(
    tile_id,
    k0,
    n_pairs,
    rank,
    j,
    n_tile: gl.constexpr,
    r_tile: gl.constexpr,
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
    RS_NB: gl.constexpr,
    RS_U: gl.constexpr,
    mfmaLayout: gl.constexpr,
    gLoadLayoutA: gl.constexpr,
    gLoadLayoutB: gl.constexpr,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
    EVEN_K: gl.constexpr,
    K_PAD: gl.constexpr,
    K_RES: gl.constexpr,
    OWNER_QPB: gl.constexpr,
    FIXUP: gl.constexpr,
    TREE_B: gl.constexpr,
    TREE_LEVELS: gl.constexpr,
    TREE_STATIC: gl.constexpr,
    STORE_CACHE: gl.constexpr,
    RS_ALL_E: gl.constexpr,
    POLL_SLEEP: gl.constexpr,
    BAR_STRIDE: gl.constexpr,
    PREFETCHED: gl.constexpr = False,
    a_top_in=None,
    b_left_in=None,
):
    """quadrant_segment of the uniform one-tile path (SEG_LOOP False), with the ending
    chosen by FIXUP."""
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn = strides
    a_offsets, a_bot_shift, b_offsets, b_right_shift = kofs
    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfmaLayout, k_width=8
    )
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfmaLayout, k_width=8
    )
    pid_m, pid_n = group_m_tile_coords(tile_id, M, N, BLOCK_M, BLOCK_N, GROUP_SIZE_M)
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
    nxt_lo = (k0 == 0).to(gl.int64) * 2

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
    drain_and_barrier()

    out_dtype: gl.constexpr = c_ptr.dtype.element_ty
    gStoreLayoutC: gl.constexpr = coalesced_store_layout(
        BLOCK_M // 2, BLOCK_N // 2, out_dtype.primitive_bitwidth // 8, gl.num_warps()
    )
    offs_cm = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gStoreLayoutC))
    c_tl_base = c_ptr + pid_m.to(gl.int64) * BLOCK_M * stride_cm
    col = pid_n * BLOCK_N + offs_cn
    c_rows = gl.multiple_of(stride_cm * offs_cm, 8)
    c_offsets = c_rows[:, None] + stride_cn * col[None, :]
    c_tr_base = c_tl_base + BLOCK_N * stride_cn // 2
    c_bl_base = c_tl_base + BLOCK_M * stride_cm // 2
    c_br_base = c_bl_base + BLOCK_N * stride_cn // 2
    c_bases = (c_tl_base, c_bl_base, c_tr_base, c_br_base)

    if FIXUP == 0:
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
    elif FIXUP == 1:
        tree_ending(
            accs,
            rank,
            j,
            pid_n,
            bias_ptr,
            p_ptr,
            locks_ptr,
            c_bases,
            c_offsets,
            m_left,
            n_left,
            BLOCK_N,
            ADD_BIAS,
            out_dtype,
            gStoreLayoutC,
            mfmaLayout,
            EVEN_M,
            EVEN_N,
            OWNER_QPB,
            n_tile,
            TREE_B,
            TREE_LEVELS,
            TREE_STATIC,
        )
    elif FIXUP == 2:
        store_only_ending(accs, rank, p_ptr, mfmaLayout, STORE_CACHE)
    else:
        rs_all_ending(
            accs,
            rank,
            j,
            pid_n,
            bias_ptr,
            p_ptr,
            sync_ptr,
            c_tl_base,
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
            mfmaLayout,
            EVEN_M,
            EVEN_N,
            n_tile,
            r_tile,
            RS_ALL_E,
            POLL_SLEEP,
            BAR_STRIDE,
        )


@gluon.jit
def _streamk_reduce_kernel(
    p_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    stride_cm,
    stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    STREAMK_TILES: gl.constexpr,
    SPLIT_N: gl.constexpr,
    READERS: gl.constexpr,
    NB: gl.constexpr,
    U: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
):
    """Program pid reduces slice pid % READERS of Stream-K tile pid // READERS over its
    SPLIT_N partials (slots t SPLIT_N ..), adds the bias and stores C."""
    pid = gl.program_id(axis=0)
    t = pid // READERS
    j = pid % READERS
    total_full_tiles = gl.cdiv(M, BLOCK_M) * gl.cdiv(N, BLOCK_N) - STREAMK_TILES
    pid_m, pid_n = group_m_tile_coords(
        total_full_tiles + t, M, N, BLOCK_M, BLOCK_N, GROUP_SIZE_M
    )
    m_left = M - pid_m * BLOCK_M
    n_left = gl.multiple_of(N - pid_n * BLOCK_N, 8)
    c_tile = (
        c_ptr
        + pid_m.to(gl.int64) * BLOCK_M * stride_cm
        + pid_n.to(gl.int64) * BLOCK_N * stride_cn
    )
    reduce_scatter_slice(
        p_ptr,
        bias_ptr,
        c_tile,
        t * SPLIT_N,
        j,
        SPLIT_N,
        READERS,
        pid_n,
        stride_cm,
        stride_cn,
        BLOCK_M,
        BLOCK_N,
        ADD_BIAS,
        NB,
        U,
        c_ptr.dtype.element_ty,
        "",
        m_left,
        n_left,
        EVEN_M,
        EVEN_N,
    )


def _pow2_ceil(x):
    return 1 << max(x - 1, 0).bit_length()


def reduce_launch(tile, streamk_tiles, num_programs, readers=None):
    """(READERS, NB, U) of the separate reduction. Default: the fewest power-of-two
    readers per tile (at most the tile's 32x32 blocks) that give every resident
    program a slice."""
    nblk = tile.blocks
    if readers is None:
        readers = min(nblk, _pow2_ceil(triton.cdiv(num_programs, streamk_tiles)))
    if readers & (readers - 1) or not 1 <= readers <= nblk:
        raise ValueError(f"reduce readers {readers} must be a power of two <= {nblk}")
    nb = max(1, min(nblk // readers, 16, tile.quadrant_blocks))
    return readers, nb, max(16 // nb, 1)


def rs_all_readers(tile, streamk_tiles, split_n, num_programs, readers=None):
    """(R, NB, U, E) of rs_all: R readers per tile, a power of two of at most the
    tile's 32x32 blocks, with R S <= num_programs and never fewer than rs's; E = R - n
    idle programs join each tile when R > n. ``readers`` overrides R (rs's own
    count gives rs with rs_all's barrier)."""
    rs = reduce_scatter_config(tile, split_n)
    pow2_floor = 1 << ((num_programs // streamk_tiles).bit_length() - 1)
    if readers is None:
        readers = max(min(tile.blocks, pow2_floor), rs.readers)
    _, nb, u = reduce_launch(tile, streamk_tiles, num_programs, readers)
    return readers, nb, u, readers - min(split_n, readers)


def gemm_a16w16_variant(
    x,
    w,
    bias=None,
    y=None,
    config=None,
    variant="rs",
    tree_radix=2,
    tree_static=None,
    reduce_readers=None,
    store_cache=".wt",
    persistent_partials=False,
    poll_sleep=0,
    spread_barriers=None,
    rs_all_r=None,
):
    """``Y = X @ W.T (+ bias)`` for a one-tile reduce-scatter ``config`` with the tail
    ending replaced by ``variant``: "rs", "rs_all" (rs with the idle programs as extra
    readers, rs_all_readers; ``poll_sleep`` 1-3 puts an s_sleep in the barrier poll;
    ``spread_barriers``, default for rs_all, gives each tile's barrier its own cache
    line; ``rs_all_r`` overrides its reader count),
    "tree", "separate" (a store-only GEMM, then _streamk_reduce_kernel), or for timing
    only "store_only" (no reduction; C's
    Stream-K tiles are not written) and "reduce_only" (separate's reduction alone,
    over whatever the partial buffer holds)."""
    if variant not in FIXUPS:
        raise ValueError(f"unknown variant {variant!r}")
    M, K = x.shape
    N = w.shape[0]
    num_programs, num_xcds = streamk_topology()
    config, launch = _launch_constants(M, N, K, config, num_programs, num_xcds)
    if config.policy != "one_tile" or config.fixup != "rs" or config.exchange != "wt":
        raise ValueError("variants take a one_tile / rs / wt config")
    tile = config.tile
    k_pad = triton.cdiv(K, 2 * tile.block_k) * 2 * tile.block_k
    if k_pad < 4 * tile.block_k:
        raise ValueError(f"K={K} needs at least {4 * tile.block_k} after pair padding")
    if y is None:
        y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    b = w.T
    _validate_tensors(x, b, y, bias, tile)
    even_m = M % tile.block_m == 0
    even_n = N % tile.block_n == 0
    partials, locks, sync = streamk_workspace(
        x.device,
        num_programs,
        num_xcds,
        tile.block_m,
        tile.block_n,
        launch["STREAMK_PROGRAMS"],
        False,
        persistent_partials,
    )
    owner_qpb = min(4, 16 // max(1, tile.block_m * tile.block_n // 4096))
    has_full = launch["TOTAL_TILES"] > launch["STREAMK_TILES"]
    assert not has_full or num_programs <= launch["TOTAL_TILES"] - launch["STREAMK_TILES"]
    split_n = launch["RESOLVED_SPLIT_N"]
    if tree_radix < 2 or tree_radix & (tree_radix - 1):
        raise ValueError(f"tree_radix={tree_radix} must be a power of two >= 2")
    tree_b = tree_radix.bit_length() - 1
    tree_levels = triton.cdiv((split_n - 1).bit_length(), tree_b)
    if tree_static is None:
        tree_static = tree_levels * (tree_radix - 1) <= 7
    # The pre-entry-18 rs constants: the tile's own power-of-two readers, and a
    # compact grid of S n (production's launch dict now carries rs_all's).
    S = launch["STREAMK_TILES"]
    own = reduce_scatter_config(tile, split_n)
    rs_r, rs_nb, rs_u = own.readers, own.blocks_per_group, own.partials_per_trip
    compact = S == launch["TOTAL_TILES"] and S * split_n < num_programs
    grid = S * split_n if compact else num_programs
    rs_all_e = 0
    bar_stride = 0
    if spread_barriers is None:
        spread_barriers = variant == "rs_all"
    if spread_barriers:
        # One 128-byte line (16 slots of 2 words) per tile's barrier where it fits:
        # packed at rs's slot t n, rs_all's extra readers poll the line other tiles
        # arrive on (1.19x at S = 8, split 2; WORKLOG entry 16).
        bar_stride = max(split_n, min(16, num_programs // launch["STREAMK_TILES"]))
    if variant == "rs_all":
        rs_r, rs_nb, rs_u, rs_all_e = rs_all_readers(
            tile, S, split_n, num_programs, rs_all_r
        )
        assert S * max(split_n, rs_r) <= num_programs
        if compact:
            grid = S * (split_n + rs_all_e)
    if variant != "reduce_only":
        _gemm_variant_kernel[(grid, 1)](
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
            EVEN_K=K == k_pad,
            STREAMK_TILES=launch["STREAMK_TILES"],
            SPLIT_N=split_n,
            RS_NB=rs_nb,
            RS_U=rs_u,
            RS_R=rs_r,
            FIXUP=FIXUPS[variant],
            TREE_B=tree_b,
            TREE_LEVELS=tree_levels,
            TREE_STATIC=tree_static,
            STORE_CACHE=store_cache,
            HAS_FULL=has_full,
            GRID=grid if compact else 0,
            OWNER_QPB=owner_qpb,
            RS_ALL_E=rs_all_e,
            POLL_SLEEP=poll_sleep,
            BAR_STRIDE=bar_stride,
            num_warps=NUM_WARPS,
            schedule_hint=schedule_hint(),
        )
    if variant in ("separate", "reduce_only"):
        readers, nb, u = reduce_launch(
            tile, launch["STREAMK_TILES"], num_programs, reduce_readers
        )
        _streamk_reduce_kernel[(launch["STREAMK_TILES"] * readers, 1)](
            partials,
            bias,
            y,
            M,
            N,
            y.stride(0),
            y.stride(1),
            BLOCK_M=tile.block_m,
            BLOCK_N=tile.block_n,
            GROUP_SIZE_M=config.group_size_m,
            STREAMK_TILES=launch["STREAMK_TILES"],
            SPLIT_N=split_n,
            READERS=readers,
            NB=nb,
            U=u,
            ADD_BIAS=bias is not None,
            EVEN_M=even_m,
            EVEN_N=even_n,
            num_warps=NUM_WARPS,
        )
    return y
