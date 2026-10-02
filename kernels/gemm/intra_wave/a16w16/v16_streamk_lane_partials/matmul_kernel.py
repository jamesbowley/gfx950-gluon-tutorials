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
from common import init_acc
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
def streamk_compact_pid(spid, SNP: gl.constexpr, NUM_PROGRAMS: gl.constexpr, NUM_XCDS: gl.constexpr,
                        BALANCE_XCDS: gl.constexpr):
    """Stream-K id of XCD-mapped program spid when only SNP programs take part, or NUM_PROGRAMS if
    it idles. With BALANCE_XCDS, XCD x keeps its first SNP // NUM_XCDS (+1 for the first
    SNP % NUM_XCDS XCDs) programs, numbered contiguously as spids are, so the stream-K phase still
    runs on every XCD's L2; otherwise the first SNP spids take part."""
    if BALANCE_XCDS:
        PER: gl.constexpr = NUM_PROGRAMS // NUM_XCDS
        BASE: gl.constexpr = SNP // NUM_XCDS
        TALL: gl.constexpr = SNP % NUM_XCDS
        xcd = spid // PER
        local = spid % PER
        active = local < BASE + (xcd < TALL).to(gl.int32)
        cid = gl.where(active, xcd * BASE + gl.minimum(xcd, TALL) + local, NUM_PROGRAMS)
    else:
        cid = gl.where(spid < SNP, spid, NUM_PROGRAMS)
    return cid


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
def streamk_tile_id(t, total_tiles, total_full_tiles, NUM_XCDS: gl.constexpr, TILE_ORDER_V9: gl.constexpr):
    """Tile id (before the GROUP_SIZE_M swizzle) of stream-K tile t.

    The stream-K tiles are the ones the persistent loop leaves over. Stream-K numbers its
    programs with spid, which is already grouped by XCD, so t (which follows spid) must not be
    grouped again: consecutive t have to be neighbouring tiles.

    Split order: the loop covers tile ids below total_full_tiles, so tile t is
    total_full_tiles + t. v9 order: the loop covers xcd_remap_tiles(v) for v < total_full_tiles,
    which leaves the top of every XCD's block of tile ids (local index >= total_full_tiles / 8);
    tile t is the t-th smallest of those.
    """
    if TILE_ORDER_V9:
        per = (total_tiles + NUM_XCDS - 1) // NUM_XCDS
        tall = total_tiles % NUM_XCDS
        tall = gl.where(tall == 0, NUM_XCDS, tall)
        l0 = total_full_tiles // NUM_XCDS
        # Leftover tiles per block: per - l0 in the first `tall` blocks, one fewer in the rest.
        n_tall = per - l0
        n_short = gl.maximum(per - 1 - l0, 1)
        head = tall * n_tall
        tile_id = (t // n_tall) * per + l0 + t % n_tall
        if t >= head:
            t2 = t - head
            tile_id = tall * per + (t2 // n_short) * (per - 1) + l0 + t2 % n_short
    else:
        tile_id = total_full_tiles + t
    return tile_id



@gluon.jit
def k_step_pair(
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
    dotOpLayoutA: gl.constexpr,
    dotOpLayoutB: gl.constexpr,
):
    """One unrolled main-loop iteration (two K-steps), same as v8.

    Consumes a_top / b_left (already in registers) plus buffers 0 and 1, prefetches the two
    K-steps at a_base / b_base into the same buffers, and returns the next a_top / b_left.
    Passing constant zeros as the accumulators turns the first four MFMAs into zero-C MFMAs.
    """

    ## =============================================================
    ## Sub-iteration 0: consume buffer 0, prefetch into buffer 0
    ## =============================================================

    ########################################
    ## Region 0: C_tl = DOT(a_top, b_left)
    ########################################
    acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")

    gl.amd.cdna4.async_copy.wait_group(5)
    a_bot = smemA_bot.index(0).load(dotOpLayoutA)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(0), b_base, b_offsets)
    gl.amd.cdna4.async_copy.commit_group()

    ########################################
    ## Region 1: C_bl = DOT(a_bot, b_left)
    ########################################
    acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")

    gl.amd.cdna4.async_copy.wait_group(5)
    b_right = smemB_right.index(0).load(dotOpLayoutB)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_top.index(0), a_base, a_offsets)
    gl.amd.cdna4.async_copy.commit_group()

    ########################################
    ## Region 2: C_tr = DOT(a_top, b_right)
    ########################################
    acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")

    gl.amd.cdna4.async_copy.wait_group(5)
    b_left = smemB_left.index(1).load(dotOpLayoutB)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_bot.index(0), a_base + a_half, a_offsets)
    gl.amd.cdna4.async_copy.commit_group()

    ########################################
    ## Region 3: C_br = DOT(a_bot, b_right)
    ########################################
    acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")

    gl.amd.cdna4.async_copy.wait_group(5)
    a_top = smemA_top.index(1).load(dotOpLayoutA)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_right.index(0), b_base + b_half, b_offsets)
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

    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(1), b_base, b_offsets_next)
    gl.amd.cdna4.async_copy.commit_group()

    ########################################
    ## Region 1: C_bl = DOT(a_bot, b_left)
    ########################################
    acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")

    gl.amd.cdna4.async_copy.wait_group(5)
    b_right = smemB_right.index(1).load(dotOpLayoutB)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_top.index(1), a_base, a_offsets_next)
    gl.amd.cdna4.async_copy.commit_group()

    ########################################
    ## Region 2: C_tr = DOT(a_top, b_right)
    ########################################
    acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")

    gl.amd.cdna4.async_copy.wait_group(5)
    b_left = smemB_left.index(0).load(dotOpLayoutB)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_bot.index(1), a_base + a_half, a_offsets_next
    )
    gl.amd.cdna4.async_copy.commit_group()

    ########################################
    ## Region 3: C_br = DOT(a_bot, b_right)
    ########################################
    acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")

    gl.amd.cdna4.async_copy.wait_group(5)
    a_top = smemA_top.index(0).load(dotOpLayoutA)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_right.index(1), b_base + b_half, b_offsets_next
    )
    gl.amd.cdna4.async_copy.commit_group()

    return acc_tl, acc_bl, acc_tr, acc_br, a_top, b_left


@gluon.jit
def load_first_pair(
    a_base,
    b_base,
    smemA_top,
    smemA_bot,
    smemB_left,
    smemB_right,
    a_offsets,
    b_offsets,
    a_half,
    b_half,
    a_offsets_next,
    b_offsets_next,
    dotOpLayoutA: gl.constexpr,
    dotOpLayoutB: gl.constexpr,
):
    """Pipeline prologue: the two K-steps at a_base / b_base into buffers 0 and 1, and the
    first one's a_top / b_left operands."""
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(0), b_base, b_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_top.index(0), a_base, a_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_bot.index(0), a_base + a_half, a_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_right.index(0), b_base + b_half, b_offsets)
    gl.amd.cdna4.async_copy.commit_group()

    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(1), b_base, b_offsets_next)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_top.index(1), a_base, a_offsets_next)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_bot.index(1), a_base + a_half, a_offsets_next
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_right.index(1), b_base + b_half, b_offsets_next
    )
    gl.amd.cdna4.async_copy.commit_group()

    gl.amd.cdna4.async_copy.wait_group(6)
    b_left = smemB_left.index(0).load(dotOpLayoutB)
    a_top = smemA_top.index(0).load(dotOpLayoutA)
    return a_top, b_left


@gluon.jit
def _one():
    return gl.zeros([1], gl.int32, gl.BlockedLayout([1], [64], [gl.num_warps()], [0]))


@gluon.jit
def drain_and_barrier():
    """Every wave waits for all its vector memory ops (stores and async copies alike: vmcnt is
    one counter), then a workgroup barrier. A release flag store after this orders all four
    waves' partial stores before the flag."""
    gl.inline_asm_elementwise(
        "s_waitcnt vmcnt(0)\ns_barrier\nv_mov_b32 $0, $1", "=v,v,~{memory}", [_one()],
        dtype=gl.int32, is_pure=False, pack=1,
    )


@gluon.jit
def store_c(acc, c_base, c_offsets, out_dtype: gl.constexpr, gStoreLayoutC: gl.constexpr):
    c = acc.to(out_dtype)
    c = gl.convert_layout(c, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_offsets, stored_value=c)


@gluon.jit
def lane_contiguous_offsets(HALF_M: gl.constexpr, HALF_N: gl.constexpr, mfmaLayout: gl.constexpr):
    """Element offsets of a 128x128 partial quadrant in its workspace slot, in the accumulator's
    MFMA layout: [16 register vectors][4 warps][64 lanes][4 fp32].

    Lane l of a wave writes elements 4l..4l+3 of its 1 KiB block, so one 16-byte-per-lane
    store or load covers 8 whole cache lines; row-major addressing (row * 128 + col) spread
    it over 16 rows, half a cache line each. The map is a bit permutation of (row, col) that
    follows the layout's bases: registers col 1, 2, 32, 64 and row 32, 64; lanes row 1, 2,
    4, 8 and col 4, 8; warps col 16 and row 16. The workspace only has to agree between the
    contributor and the owner, which hold the same layout.
    """
    gl.static_assert(HALF_M == 128 and HALF_N == 128, "bit map is for 128x128 quadrants")
    r = gl.arange(0, HALF_M, gl.SliceLayout(1, mfmaLayout))
    c = gl.arange(0, HALF_N, gl.SliceLayout(0, mfmaLayout))
    # The column term is c plus terms constant over each 4 columns, so axis analysis still sees
    # 4 contiguous, aligned elements (16-byte accesses). Written as c % 4 + ... or with shifts
    # and masks, the same map compiles to single-dword accesses.
    f = (r % 16) * 4 + (r // 16 % 2) * 512 + (r // 32 % 2) * 4096 + (r // 64) * 8192
    g = c + (c // 4 % 4) * 60 + (c // 16 % 2) * 240 + (c // 32 % 2) * 992 + (c // 64) * 1984
    return f[:, None] + g[None, :]


@gluon.jit
def fixup_quadrant(
    acc,
    q: gl.constexpr,
    p_ptr,
    bias_ptr,
    spid,
    pid_n,
    is_partial,
    n_peers,
    c_base,
    c_offsets,
    BLOCK_N: gl.constexpr,
    HALF_M: gl.constexpr,
    HALF_N: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    out_dtype: gl.constexpr,
    gStoreLayoutC: gl.constexpr,
    mfmaLayout: gl.constexpr,
):
    """Quadrant q (tl, bl, tr, br) of a stream-K segment's accumulator. A contributor stores it
    to its slot P[spid]; the owner adds the n_peers following programs' partials and the bias,
    and stores C.

    Partials are 128x128 fp32 quadrants stored straight from the MFMA layout, with no layout
    conversion, and read back into the same layout. Both roles run the same code with
    run-time guards: a branch or loop that carries the AGPR accumulator spills, so the peers
    are summed in VGPRs and added in once.
    """
    QUAD: gl.constexpr = HALF_M * HALF_N
    offs_pn = gl.arange(0, HALF_N, gl.SliceLayout(0, mfmaLayout))
    p_offsets = lane_contiguous_offsets(HALF_M, HALF_N, mfmaLayout)
    if is_partial != 0:
        gl.amd.cdna3.buffer_store(
            stored_value=acc, ptr=p_ptr + (spid.to(gl.int64) * 4 + q) * QUAD, offsets=p_offsets,
            cache=".wt",
        )
    psum = gl.zeros((HALF_M, HALF_N), gl.float32, mfmaLayout)
    for j in range(n_peers):
        peer = (spid + 1 + j).to(gl.int64)
        psum += gl.amd.cdna3.buffer_load(
            ptr=p_ptr + (peer * 4 + q) * QUAD, offsets=p_offsets, cache=".cv"
        )
    if is_partial == 0:
        if ADD_BIAS:
            # Only the owner adds the bias, once; contributors' partials exclude it.
            bias = gl.amd.cdna3.buffer_load(
                ptr=bias_ptr + pid_n * BLOCK_N + (q // 2) * HALF_N, offsets=offs_pn
            )
            psum = psum + bias.to(gl.float32)[None, :]
        store_c(acc + psum, c_base, c_offsets, out_dtype, gStoreLayoutC)


@gluon.jit
def streamk_owner_of(i, L: gl.constexpr, R: gl.constexpr):
    """Stream-K program whose range contains pair i: the first R programs own L + 1 pairs,
    the rest L."""
    head: gl.constexpr = R * (L + 1)
    if L == 0:
        owner = i // (L + 1)
    else:
        owner = i // (L + 1)
        if i >= head:
            owner = R + (i - head) // L
    return owner


@gluon.jit
def streamk_segment(
    tile_id,
    k0,
    n_pairs,
    is_partial,
    n_peers,
    spid,
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    p_ptr,
    locks_ptr,
    smemA_top,
    smemA_bot,
    smemB_left,
    smemB_right,
    a_offsets,
    b_offsets,
    a_half,
    b_half,
    a_offsets_next,
    b_offsets_next,
    num_tiles_m,
    num_tiles_n,
    total_tiles,
    stride_ak,
    stride_bk,
    stride_am,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    TILE_ORDER_V9: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    mfmaLayout: gl.constexpr,
    dotOpLayoutA: gl.constexpr,
    dotOpLayoutB: gl.constexpr,
):
    """K-steps [k0, k0 + 2 * n_pairs) of tile `tile_id`, n_pairs >= 1, then the segment's
    ending: publish the partial (is_partial) or collect n_peers partials and store C.

    Starts with its own prologue and runs every pair through the main-loop body; the last
    pair's prefetch reloads the last pair (clamped), which the drain discards.
    """
    pid_m, pid_n = get_group_m_tile_ids(tile_id, num_tiles_m, num_tiles_n, GROUP_SIZE_M)

    a_tile = a_ptr + pid_m.to(gl.int64) * BLOCK_M * stride_am + k0.to(gl.int64) * BLOCK_K * stride_ak
    b_tile = b_ptr + pid_n.to(gl.int64) * BLOCK_N * stride_bn + k0.to(gl.int64) * BLOCK_K * stride_bk
    a_top, b_left = load_first_pair(
        a_tile, b_tile, smemA_top, smemA_bot, smemB_left, smemB_right,
        a_offsets, b_offsets, a_half, b_half, a_offsets_next, b_offsets_next,
        dotOpLayoutA, dotOpLayoutB,
    )

    acc_tl, acc_bl, acc_tr, acc_br = init_acc(bias_ptr, pid_n, BLOCK_M, BLOCK_N, mfmaLayout, False)
    for i in range(n_pairs):
        nxt = gl.minimum(i + 1, n_pairs - 1).to(gl.int64) * 2
        acc_tl, acc_bl, acc_tr, acc_br, a_top, b_left = k_step_pair(
            acc_tl, acc_bl, acc_tr, acc_br, a_top, b_left,
            smemA_top, smemA_bot, smemB_left, smemB_right,
            a_tile + nxt * BLOCK_K * stride_ak, b_tile + nxt * BLOCK_K * stride_bk,
            a_offsets, b_offsets, a_half, b_half, a_offsets_next, b_offsets_next,
            dotOpLayoutA, dotOpLayoutB,
        )
    # The clamped prefetch is still in flight, and the next segment's prologue overwrites the
    # buffers other waves may still be reading.
    drain_and_barrier()

    gStoreLayoutC: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
    out_dtype: gl.constexpr = a_ptr.dtype.element_ty
    offs_cm = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gStoreLayoutC))
    c_tl_base = c_ptr + pid_m.to(gl.int64) * BLOCK_M * stride_cm
    col = pid_n * BLOCK_N + offs_cn
    c_offsets = stride_cm * offs_cm[:, None] + stride_cn * col[None, :]
    c_tr_base = c_tl_base + BLOCK_N * stride_cn // 2
    c_bl_base = c_tl_base + BLOCK_M * stride_cm // 2
    c_br_base = c_bl_base + BLOCK_N * stride_cn // 2

    # Owner: every peer's flag before the first partial read. Poll relaxed and acquire once:
    # a GPU-scope acquire invalidates the XCD's L2 under the neighbours' K loops.
    for j in range(n_peers):
        while gl.atomic_add(locks_ptr + spid + 1 + j, 0, sem="relaxed", scope="gpu") == 0:
            pass
    if n_peers > 0:
        gl.atomic_add(locks_ptr + spid, 0, sem="acquire", scope="gpu")

    fixup_quadrant(
        acc_tl, 0, p_ptr, bias_ptr, spid, pid_n, is_partial, n_peers, c_tl_base, c_offsets,
        BLOCK_N, BLOCK_M // 2, BLOCK_N // 2, ADD_BIAS, out_dtype, gStoreLayoutC, mfmaLayout,
    )
    fixup_quadrant(
        acc_bl, 1, p_ptr, bias_ptr, spid, pid_n, is_partial, n_peers, c_bl_base, c_offsets,
        BLOCK_N, BLOCK_M // 2, BLOCK_N // 2, ADD_BIAS, out_dtype, gStoreLayoutC, mfmaLayout,
    )
    fixup_quadrant(
        acc_tr, 2, p_ptr, bias_ptr, spid, pid_n, is_partial, n_peers, c_tr_base, c_offsets,
        BLOCK_N, BLOCK_M // 2, BLOCK_N // 2, ADD_BIAS, out_dtype, gStoreLayoutC, mfmaLayout,
    )
    fixup_quadrant(
        acc_br, 3, p_ptr, bias_ptr, spid, pid_n, is_partial, n_peers, c_br_base, c_offsets,
        BLOCK_N, BLOCK_M // 2, BLOCK_N // 2, ADD_BIAS, out_dtype, gStoreLayoutC, mfmaLayout,
    )

    if is_partial != 0:
        # Release: the flag may only become visible after all four waves' partial stores.
        drain_and_barrier()
        gl.atomic_xchg(locks_ptr + spid, 1, sem="release", scope="gpu")
    # Each partial has exactly one reader, so the owner re-arms its peers' flags for the next
    # launch; the kernel boundary orders these stores against it.
    for j in range(n_peers):
        gl.atomic_xchg(locks_ptr + spid + 1 + j, 0, sem="relaxed", scope="gpu")


@gluon.jit
def persistent_tile(
    vpid,
    a_top,
    b_left,
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    smemA_top,
    smemA_bot,
    smemB_left,
    smemB_right,
    a_offsets,
    b_offsets,
    a_half,
    b_half,
    a_offsets_next,
    b_offsets_next,
    num_tiles_m,
    num_tiles_n,
    total_tiles,
    stride_ak,
    stride_bk,
    stride_am,
    stride_bn,
    stride_cm,
    stride_cn,
    K: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    TILE_ORDER_V9: gl.constexpr,
    MASK_TAIL_PREFETCH: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    mfmaLayout: gl.constexpr,
    dotOpLayoutA: gl.constexpr,
    dotOpLayoutB: gl.constexpr,
):
    """One full-K output tile for virtual pid `vpid`: v13's persistent-loop body.

    Expects the tile's K-steps 0 and 1 in flight in buffers 0 and 1 and its K-step 0
    operands in a_top / b_left. Prefetches the same for vpid + NUM_PROGRAMS (clamped to the
    last tile) and returns that tile's a_top / b_left.
    """
    iterMax = gl.cdiv(K, BLOCK_K)

    tile_id = persistent_tile_id(vpid, total_tiles, NUM_XCDS, TILE_ORDER_V9)

    pid_m, pid_n = get_group_m_tile_ids(tile_id, num_tiles_m, num_tiles_n, GROUP_SIZE_M)

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

    # K-steps 0 and 1 are already in flight, so the first prefetch is K-step 2.
    a_base = a_ptr + pid_m.to(gl.int64) * BLOCK_M * stride_am + BLOCK_K * stride_ak * 2
    b_base = b_ptr + pid_n.to(gl.int64) * BLOCK_N * stride_bn + BLOCK_K * stride_bk * 2

    ## K-steps 0 and 1, peeled: without bias, the constant-zero accumulator becomes the
    ## MFMA's inline-0 C operand, so no AGPRs are zeroed per tile. A loop-carried zero would
    ## be materialized.
    acc_tl, acc_bl, acc_tr, acc_br = init_acc(
        bias_ptr, pid_n, BLOCK_M, BLOCK_N, mfmaLayout, ADD_BIAS
    )
    acc_tl, acc_bl, acc_tr, acc_br, a_top, b_left = k_step_pair(
        acc_tl, acc_bl, acc_tr, acc_br, a_top, b_left,
        smemA_top, smemA_bot, smemB_left, smemB_right, a_base, b_base,
        a_offsets, b_offsets, a_half, b_half, a_offsets_next, b_offsets_next,
        dotOpLayoutA, dotOpLayoutB,
    )
    a_base += BLOCK_K * stride_ak * 2
    b_base += BLOCK_K * stride_bk * 2

    ## Main loop — same as v8
    for k in range(2, iterMax - 2, 2):
        acc_tl, acc_bl, acc_tr, acc_br, a_top, b_left = k_step_pair(
            acc_tl, acc_bl, acc_tr, acc_br, a_top, b_left,
            smemA_top, smemA_bot, smemB_left, smemB_right, a_base, b_base,
            a_offsets, b_offsets, a_half, b_half, a_offsets_next, b_offsets_next,
            dotOpLayoutA, dotOpLayoutB,
        )
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
    acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    l_idx = (iterMax - 2) % 2
    a_bot = smemA_bot.index(l_idx).load(dotOpLayoutA)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_left.index(0), b_next, b_offsets, mask=has_next
    )
    gl.amd.cdna4.async_copy.commit_group()

    acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    b_right = smemB_right.index(l_idx).load(dotOpLayoutB)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_top.index(0), a_next, a_offsets, mask=has_next
    )
    gl.amd.cdna4.async_copy.commit_group()

    acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    g_idx = 1 - l_idx
    b_left = smemB_left.index(g_idx).load(dotOpLayoutB)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_bot.index(0), a_next + a_half, a_offsets, mask=has_next
    )
    gl.amd.cdna4.async_copy.commit_group()

    acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    a_top = smemA_top.index(g_idx).load(dotOpLayoutA)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_right.index(0), b_next + b_half, b_offsets, mask=has_next
    )
    gl.amd.cdna4.async_copy.commit_group()

    ## Iter iterMax - 1: prefetch the next tile's K-step 1.
    ## Natural-pipeline epilogue: each store follows its MFMA with one
    ## MFMA cycle of gap, yielding uniform MFMA-store interleaving.
    acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    a_bot = smemA_bot.index(g_idx).load(dotOpLayoutA)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_left.index(1), b_next, b_offsets_next, mask=has_next
    )
    gl.amd.cdna4.async_copy.commit_group()

    acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    b_right = smemB_right.index(g_idx).load(dotOpLayoutB)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_top.index(1), a_next, a_offsets_next, mask=has_next
    )
    gl.amd.cdna4.async_copy.commit_group()

    c_tl = acc_tl.to(a_ptr.dtype.element_ty)
    c_tl = gl.convert_layout(c_tl, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(ptr=c_tl_base, offsets=c_offsets, stored_value=c_tl)

    acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")

    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_bot.index(1), a_next + a_half, a_offsets_next, mask=has_next
    )
    gl.amd.cdna4.async_copy.commit_group()

    c_bl = acc_bl.to(a_ptr.dtype.element_ty)
    c_bl = gl.convert_layout(c_bl, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(ptr=c_bl_base, offsets=c_offsets, stored_value=c_bl)

    acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")

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

    return a_top, b_left


@gluon.jit
def v16_streamk_lane_partials(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    p_ptr,
    locks_ptr,
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
    STREAMK_TILES: gl.constexpr,
    STREAMK_NUM_PROGRAMS: gl.constexpr,
    BALANCE_XCDS: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,  #
    TILE_ORDER_V9: gl.constexpr,
    MASK_TAIL_PREFETCH: gl.constexpr,
    ADD_BIAS: gl.constexpr,
):
    """
    v14 with TensorAtlas's one-tile stream-K tail (DP + one-tile SK). The persistent loop
    covers total_tiles - STREAMK_TILES full tiles; the last STREAMK_TILES tiles' k-steps are
    laid end to end and split evenly over all NUM_PROGRAMS programs, in units of 2 k-steps so
    every segment fits the pair pipeline. The program that computes a tile's k-step 0 owns it:
    it adds its peers' partials (the next consecutive pids) and the bias, and stores C. Every
    other segment stores its fp32 partial to P[pid] and raises locks[pid] with a release.
    STREAMK_NUM_PROGRAMS programs take part in the stream-K phase and the rest idle
    (streamk_compact_pid; spread evenly over the XCDs if BALANCE_XCDS).

    v16: as v15, with the partials laid out lane-contiguous in P instead of row-major
    (lane_contiguous_offsets).
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

    # The peeled pair, the 2x-unrolled loop and the 2-step epilogue always leave buffer 0 as
    # the one to read next, which only holds for an even number of full K-steps.
    gl.static_assert(K % (2 * BLOCK_K) == 0, "K must be a multiple of 2 * BLOCK_K")
    iterMax = gl.cdiv(K, BLOCK_K)
    gl.assume(iterMax > 3)

    ## Prologue, first tile only: every later tile's first two K-steps are prefetched by the
    ## previous tile's epilogue. Programs with no tile still load a valid one, then skip the loop.
    first_id = persistent_tile_id(min(start, total_tiles - 1), total_tiles, NUM_XCDS, TILE_ORDER_V9)
    tile_m, tile_n = get_group_m_tile_ids(first_id, num_tiles_m, num_tiles_n, GROUP_SIZE_M)
    a_first = a_ptr + tile_m.to(gl.int64) * BLOCK_M * stride_am
    b_first = b_ptr + tile_n.to(gl.int64) * BLOCK_N * stride_bn

    # First tile's K-steps 0-1; later tiles get them from the previous tile's epilogue.
    a_top, b_left = load_first_pair(
        a_first, b_first, smemA_top, smemA_bot, smemB_left, smemB_right,
        a_offsets, b_offsets, a_half, b_half, a_offsets_next, b_offsets_next,
        dotOpLayoutA, dotOpLayoutB,
    )

    # Full tiles. The last trip's prefetch (clamped to total_tiles - 1) is unused; the
    # stream-K phase drains it and starts every segment with its own prologue.
    total_full_tiles = total_tiles - STREAMK_TILES
    for vpid in range(start, total_full_tiles, NUM_PROGRAMS):
        a_top, b_left = persistent_tile(
            vpid, a_top, b_left, a_ptr, b_ptr, c_ptr, bias_ptr,
            smemA_top, smemA_bot, smemB_left, smemB_right,
            a_offsets, b_offsets, a_half, b_half, a_offsets_next, b_offsets_next,
            num_tiles_m, num_tiles_n, total_tiles,
            stride_ak, stride_bk, stride_am, stride_bn, stride_cm, stride_cn,
            K, BLOCK_M, BLOCK_N, BLOCK_K, NUM_PROGRAMS, NUM_XCDS, GROUP_SIZE_M,
            TILE_ORDER_V9, MASK_TAIL_PREFETCH, ADD_BIAS,
            mfmaLayout, dotOpLayoutA, dotOpLayoutB,
        )

    # Drain the last trip's redundant prefetch.
    gl.amd.cdna4.async_copy.wait_group(0)

    if STREAMK_TILES > 0:
        # Stream-K iteration space: the STREAMK_TILES tiles' pairs of k-steps laid end to end.
        # Program spid owns pairs [sk_start, sk_end): the first R programs get L + 1, the rest L.
        PAIRS_PER_TILE: gl.constexpr = K // (2 * BLOCK_K)
        SK_PAIRS: gl.constexpr = STREAMK_TILES * PAIRS_PER_TILE
        L: gl.constexpr = SK_PAIRS // STREAMK_NUM_PROGRAMS
        R: gl.constexpr = SK_PAIRS % STREAMK_NUM_PROGRAMS
        # Stream-K's pid is always the XCD-mapped one, so a tile's peers share its XCD.
        spid = streamk_compact_pid(
            get_logical_chiplet_mapped_pids(NUM_PROGRAMS, NUM_XCDS), STREAMK_NUM_PROGRAMS, NUM_PROGRAMS,
            NUM_XCDS, BALANCE_XCDS,
        )
        sk_start = gl.minimum(spid * L + gl.minimum(spid, R), SK_PAIRS)
        sk_end = gl.minimum((spid + 1) * L + gl.minimum(spid + 1, R), SK_PAIRS)
        # Segments: the range cut at tile boundaries. Only the first can be a partial (it may
        # start mid-tile); an owner segment that needs peers is always the last.
        while sk_start < sk_end:
            t = sk_start // PAIRS_PER_TILE
            tile_start = t * PAIRS_PER_TILE
            tile_end = tile_start + PAIRS_PER_TILE
            seg_end = gl.minimum(tile_end, sk_end)
            is_partial = (sk_start != tile_start).to(gl.int32)
            n_peers = (streamk_owner_of(tile_end - 1, L, R) - spid) * (1 - is_partial)
            streamk_segment(
                streamk_tile_id(t, total_tiles, total_full_tiles, NUM_XCDS, TILE_ORDER_V9),
                (sk_start - tile_start) * 2, seg_end - sk_start,
                is_partial, n_peers, spid,
                a_ptr, b_ptr, c_ptr, bias_ptr, p_ptr, locks_ptr,
                smemA_top, smemA_bot, smemB_left, smemB_right,
                a_offsets, b_offsets, a_half, b_half, a_offsets_next, b_offsets_next,
                num_tiles_m, num_tiles_n, total_tiles,
                stride_ak, stride_bk, stride_am, stride_bn, stride_cm, stride_cn,
                BLOCK_M, BLOCK_N, BLOCK_K, NUM_XCDS, GROUP_SIZE_M, TILE_ORDER_V9, ADD_BIAS,
                mfmaLayout, dotOpLayoutA, dotOpLayoutB,
            )
            sk_start = seg_end


# Per device: the fp32 partial slots (one BLOCK_M x BLOCK_N tile per program) and one flag per
# program. Owners reset the flags they consume, so the flags are zero between launches.
_WORKSPACE = {}


def streamk_workspace(device, num_programs, block_m, block_n):
    key = (str(device), num_programs, block_m, block_n)
    if key not in _WORKSPACE:
        # Every partial fully overwrites its slot before its flag goes up, so P needs no init.
        P = torch.empty(num_programs * block_m * block_n, device=device, dtype=torch.float32)
        locks = torch.zeros(num_programs, device=device, dtype=torch.int32)
        _WORKSPACE[key] = (P, locks)
    return _WORKSPACE[key]


def matmul(a, b, c=None, bias=None):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    assert bias is None or (bias.shape == (N,) and bias.is_contiguous()), "bias must be contiguous (N,)"
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
    # Programs launched, which is also the persistent stride; it need not equal the CU count.
    NUM_PROGRAMS = 256
    # Programs that take part in the stream-K portion (STREAMK_NUM_PROGRAMS, default all); the
    # rest idle, spread evenly over the XCDs unless STREAMK_BALANCE_XCDS=0. The fixup waits need
    # every program resident at once, so the grid must fit in one wave.
    STREAMK_NUM_PROGRAMS = int(os.environ.get("STREAMK_NUM_PROGRAMS", NUM_PROGRAMS))
    BALANCE_XCDS = os.environ.get("STREAMK_BALANCE_XCDS", "1") == "1"
    # DP + one-tile SK: the partial last wave. Two-tile SK + DP would add NUM_PROGRAMS when
    # there are at least two full waves.
    total_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    STREAMK_TILES = total_tiles % NUM_PROGRAMS
    assert 0 <= STREAMK_TILES <= STREAMK_NUM_PROGRAMS <= NUM_PROGRAMS
    assert (total_tiles - STREAMK_TILES) % NUM_PROGRAMS == 0
    P, locks = streamk_workspace(a.device, NUM_PROGRAMS, BLOCK_M, BLOCK_N)
    grid = (NUM_PROGRAMS, 1)
    NUM_XCDS = 8
    GROUP_SIZE_M = 4
    v16_streamk_lane_partials[grid](
        a,
        b,
        c,  #
        bias,
        P,
        locks,
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
        NUM_PROGRAMS=NUM_PROGRAMS,
        STREAMK_TILES=STREAMK_TILES,
        STREAMK_NUM_PROGRAMS=STREAMK_NUM_PROGRAMS,
        BALANCE_XCDS=BALANCE_XCDS,
        NUM_XCDS=NUM_XCDS,
        GROUP_SIZE_M=GROUP_SIZE_M,
        # Tiles are walked in v9's order; PERSISTENT_TILE_ORDER=split selects the split order.
        TILE_ORDER_V9=os.environ.get("PERSISTENT_TILE_ORDER", "v9") == "v9",
        # The masked prefetch costs about 12 VGPRs, which here pushes the kernel into spills,
        # so the last trip loads the clamped tile unmasked.
        MASK_TAIL_PREFETCH=False,
        ADD_BIAS=bias is not None,
        num_warps=num_warps,
    )
    return c
