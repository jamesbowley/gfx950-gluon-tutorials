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

import functools
import math
import os

import triton
import torch
# from common import get_pids
from bias import init_acc
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from v9_ragged.matmul_kernel import matmul as v9_matmul

# Extra elements after each program's partial slot in P (STREAMK_P_PAD bytes of fp32, default 0).
# Unpadded, every slot starts at a multiple of 256 KB (128 KB with bf16 partials), so the n
# partials one reduce-scatter lane reads are a slot apart. Read once at import: the kernel and
# the workspace must agree.
P_SLOT_PAD = gl.constexpr(int(os.environ.get("STREAMK_P_PAD", "0")) // 4)
# STREAMK_PARTIALS values: the partials' type and map in P.
PARTIALS_CODES = {"fp32": 0, "bf16": 1, "bf16_pairs": 2}


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
    a_offsets_bot,
    b_offsets_bot,
    a_offsets_bot_next,
    b_offsets_bot_next,
    a_odd,
    b_odd,
):
    """One unrolled main-loop iteration (two K-steps), same as v8.

    Consumes a_top / b_left (already in registers) plus buffers 0 and 1, prefetches the two
    K-steps at a_base / b_base into the same buffers, and returns the next a_top / b_left.
    Passing constant zeros as the accumulators turns the first four MFMAs into zero-C MFMAs.

    The bottom / right halves load at base + half with *_bot offsets, the odd K-step at
    base + odd with *_next offsets (operand_offsets builds both sets).
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

    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_bot.index(0), a_base + a_half, a_offsets_bot)
    gl.amd.cdna4.async_copy.commit_group()

    ########################################
    ## Region 3: C_br = DOT(a_bot, b_right)
    ########################################
    acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")

    gl.amd.cdna4.async_copy.wait_group(5)
    a_top = smemA_top.index(1).load(dotOpLayoutA)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_right.index(0), b_base + b_half, b_offsets_bot)
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

    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(1), b_base + b_odd, b_offsets_next)
    gl.amd.cdna4.async_copy.commit_group()

    ########################################
    ## Region 1: C_bl = DOT(a_bot, b_left)
    ########################################
    acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")

    gl.amd.cdna4.async_copy.wait_group(5)
    b_right = smemB_right.index(1).load(dotOpLayoutB)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_top.index(1), a_base + a_odd, a_offsets_next)
    gl.amd.cdna4.async_copy.commit_group()

    ########################################
    ## Region 2: C_tr = DOT(a_top, b_right)
    ########################################
    acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")

    gl.amd.cdna4.async_copy.wait_group(5)
    b_left = smemB_left.index(0).load(dotOpLayoutB)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_bot.index(1), a_base + a_half + a_odd, a_offsets_bot_next
    )
    gl.amd.cdna4.async_copy.commit_group()

    ########################################
    ## Region 3: C_br = DOT(a_bot, b_right)
    ########################################
    acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")

    gl.amd.cdna4.async_copy.wait_group(5)
    a_top = smemA_top.index(0).load(dotOpLayoutA)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_right.index(1), b_base + b_half + b_odd, b_offsets_bot_next
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
    a_offsets_bot,
    b_offsets_bot,
    a_offsets_bot_next,
    b_offsets_bot_next,
    a_odd,
    b_odd,
    ma0=None,
    mb0=None,
    ma1=None,
    mb1=None,
):
    """Pipeline prologue: the two K-steps at a_base / b_base into buffers 0 and 1, and the
    first one's a_top / b_left operands. Offsets as in k_step_pair. ma* / mb* (EDGE_K): load masks
    of A's and B's K-step 0 and 1, None for unmasked."""
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(0), b_base, b_offsets, mask=mb0)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_top.index(0), a_base, a_offsets, mask=ma0)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_bot.index(0), a_base + a_half, a_offsets_bot, mask=ma0)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_right.index(0), b_base + b_half, b_offsets_bot, mask=mb0)
    gl.amd.cdna4.async_copy.commit_group()

    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(1), b_base + b_odd, b_offsets_next, mask=mb1)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_top.index(1), a_base + a_odd, a_offsets_next, mask=ma1)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_bot.index(1), a_base + a_half + a_odd, a_offsets_bot_next, mask=ma1
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_right.index(1), b_base + b_half + b_odd, b_offsets_bot_next, mask=mb1
    )
    gl.amd.cdna4.async_copy.commit_group()

    gl.amd.cdna4.async_copy.wait_group(6)
    b_left = smemB_left.index(0).load(dotOpLayoutB)
    a_top = smemA_top.index(0).load(dotOpLayoutA)
    return a_top, b_left


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
def tail_pair_masks(k_hi, K_RES: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
                    gLoadLayoutA: gl.constexpr, gLoadLayoutB: gl.constexpr):
    """(ma0, mb0, ma1, mb1) for load_first_pair of a pair whose first k_hi K values are in the matrix.
    K-step 1 of the last pair always reaches past K, K-step 0 only when K_RES > BLOCK_K; a mask that
    can never drop a lane is None."""
    ma0, mb0 = None, None
    if K_RES > BLOCK_K:
        ma0 = k_tail_mask(k_hi, 0, BLOCK_M // 2, BLOCK_K, gLoadLayoutA, False)
        mb0 = k_tail_mask(k_hi, 0, BLOCK_N // 2, BLOCK_K, gLoadLayoutB, True)
    ma1 = k_tail_mask(k_hi, 1, BLOCK_M // 2, BLOCK_K, gLoadLayoutA, False)
    mb1 = k_tail_mask(k_hi, 1, BLOCK_N // 2, BLOCK_K, gLoadLayoutB, True)
    return ma0, mb0, ma1, mb1


@gluon.jit
def clamped_tile_offsets(lim, stride_r, stride_k, HALF: gl.constexpr, BLOCK_K: gl.constexpr,
                         layout: gl.constexpr, A_SIDE: gl.constexpr):
    """Per-tile load offsets of the two halves (top / bot of A, left / right of B) of a tile whose
    rows (A) or columns (B) past lim, the tile's last in-bounds one, are clamped to it. Clamped
    lanes read valid memory and only feed C elements that the store masks drop. The tile's
    position stays in the 64-bit scalar base, so the offsets span one tile as in v24."""
    if A_SIDE:
        r = gl.arange(0, HALF, gl.SliceLayout(1, layout))
        k = gl.arange(0, BLOCK_K, gl.SliceLayout(0, layout))
        top = gl.minimum(r, lim)[:, None] * stride_r + k[None, :] * stride_k
        bot = gl.minimum(r + HALF, lim)[:, None] * stride_r + k[None, :] * stride_k
    else:
        r = gl.arange(0, HALF, gl.SliceLayout(0, layout))
        k = gl.arange(0, BLOCK_K, gl.SliceLayout(1, layout))
        top = k[:, None] * stride_k + gl.minimum(r, lim)[None, :] * stride_r
        bot = k[:, None] * stride_k + gl.minimum(r + HALF, lim)[None, :] * stride_r
    return top, bot


@gluon.jit
def operand_offsets(top, bot, offsets, offsets_next, half, odd, EDGE: gl.constexpr):
    """(offsets, offsets_next, offsets_bot, offsets_bot_next, half, odd) for k_step_pair and
    load_first_pair. Without EDGE: v24's kernel-wide offsets, the second half one scalar shift
    `half` away and the odd K-step in its own offsets. With EDGE: the tile's clamped top / bot
    offsets, and the odd K-step a scalar shift `odd` away, so a tile still holds four offset
    tensors per operand pair."""
    if EDGE:
        o0, o1, o2, o3, h, d = top, top, bot, bot, 0, odd
    else:
        o0, o1, o2, o3, h, d = offsets, offsets_next, offsets, offsets_next, half, 0
    return o0, o1, o2, o3, h, d


@gluon.jit
def tile_origin(pid, extent, BLOCK: gl.constexpr, SHIFT: gl.constexpr):
    """First row (column) of tile pid: pid * BLOCK, or with SHIFT at most extent - BLOCK, so that a
    ragged last tile ends at the matrix edge (it overlaps the tile before it)."""
    o = pid * BLOCK
    if SHIFT:
        o = gl.minimum(o, extent - BLOCK)
    return o


@gluon.jit
def edge_mask(rows_left, cols_left, ROWS: gl.constexpr, COLS: gl.constexpr, layout: gl.constexpr,
              EDGE_M: gl.constexpr, EDGE_N: gl.constexpr, rows_skip=0, cols_skip=0, SHIFT: gl.constexpr = False):
    """In-bounds mask of a ROWS x COLS block of C that has rows_left rows and cols_left columns
    before the matrix edge (either may exceed the block). SHIFT: the block's first rows_skip rows and
    cols_skip columns belong to the tile before (a shifted edge tile), the rest are its own."""
    r = gl.arange(0, ROWS, gl.SliceLayout(1, layout))[:, None]
    c = gl.arange(0, COLS, gl.SliceLayout(0, layout))[None, :]
    if SHIFT:
        mask = (r >= rows_skip) & (c >= cols_skip)
    elif EDGE_M and EDGE_N:
        mask = (r < rows_left) & (c < cols_left)
    elif EDGE_M:
        mask = (r < rows_left) & (c >= 0)
    else:
        mask = (r >= 0) & (c < cols_left)
    return mask


@gluon.jit
def _one():
    return gl.zeros([1], gl.int32, gl.BlockedLayout([1], [64], [gl.num_warps()], [0]))


@gluon.jit
def drain_and_barrier():
    """Every wave waits for all its vector memory ops (stores and async copies alike: vmcnt is
    one counter), then a workgroup barrier. A flag or counter set after this is ordered after
    all four waves' partial stores."""
    gl.inline_asm_elementwise(
        "s_waitcnt vmcnt(0)\ns_barrier\nv_mov_b32 $0, $1", "=v,v,~{memory}", [_one()],
        dtype=gl.int32, is_pure=False, pack=1,
    )


@gluon.jit
def asm_fence(code: gl.constexpr):
    """Inline asm with a memory clobber: the compiler moves no memory access across it."""
    gl.inline_asm_elementwise(
        code + "\nv_mov_b32 $0, $1", "=v,v,~{memory}", [_one()], dtype=gl.int32, is_pure=False, pack=1,
    )


@gluon.jit
def store_c(acc, c_base, c_offsets, out_dtype: gl.constexpr, gStoreLayoutC: gl.constexpr,
            rows_left=0, cols_left=0, EDGE_M: gl.constexpr = False, EDGE_N: gl.constexpr = False,
            rows_skip=0, cols_skip=0, SHIFT: gl.constexpr = False):
    """Store a quadrant of C; with EDGE_M / EDGE_N only its first rows_left rows / cols_left
    columns."""
    c = acc.to(out_dtype)
    c = gl.convert_layout(c, layout=gStoreLayoutC)
    store_edge(c, c_base, c_offsets, gStoreLayoutC, rows_left, cols_left, EDGE_M, EDGE_N, rows_skip, cols_skip,
               SHIFT)


@gluon.jit
def store_edge(c, c_base, c_offsets, layout: gl.constexpr, rows_left, cols_left, EDGE_M: gl.constexpr,
               EDGE_N: gl.constexpr, rows_skip=0, cols_skip=0, SHIFT: gl.constexpr = False):
    """buffer_store of a block of C, masked to its first rows_left rows / cols_left columns with
    EDGE_M / EDGE_N."""
    if EDGE_M or EDGE_N or SHIFT:
        mask = edge_mask(rows_left, cols_left, c.shape[0], c.shape[1], layout, EDGE_M, EDGE_N, rows_skip, cols_skip,
                         SHIFT)
        gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_offsets, stored_value=c, mask=mask)
    else:
        gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_offsets, stored_value=c)


@gluon.jit
def init_acc_edge(bias_ptr, n0, cols_left, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, MFMA: gl.constexpr,
                  ADD_BIAS: gl.constexpr, EDGE_N: gl.constexpr, SHIFT: gl.constexpr = False):
    """bias.init_acc, with the bias loaded only for the tile's first cols_left columns under
    EDGE_N."""
    if ADD_BIAS and EDGE_N:
        acc_l = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, MFMA)
        offs_bias = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, MFMA))
        bias_base = bias_ptr + n0
        bias_l = gl.amd.cdna3.buffer_load(ptr=bias_base, offsets=offs_bias, mask=offs_bias < cols_left, other=0.0)
        bias_r = gl.amd.cdna3.buffer_load(ptr=bias_base + BLOCK_N // 2, offsets=offs_bias,
                                          mask=offs_bias < cols_left - BLOCK_N // 2, other=0.0)
        acc_r = acc_l + bias_r.to(gl.float32)[None, :]
        acc_l = acc_l + bias_l.to(gl.float32)[None, :]
        tl, bl, tr, br = acc_l, acc_l, acc_r, acc_r
    elif ADD_BIAS and SHIFT:
        acc_l = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, MFMA)
        offs_bias = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, MFMA))
        bias_l = gl.amd.cdna3.buffer_load(ptr=bias_ptr + n0, offsets=offs_bias)
        bias_r = gl.amd.cdna3.buffer_load(ptr=bias_ptr + n0 + BLOCK_N // 2, offsets=offs_bias)
        acc_r = acc_l + bias_r.to(gl.float32)[None, :]
        acc_l = acc_l + bias_l.to(gl.float32)[None, :]
        tl, bl, tr, br = acc_l, acc_l, acc_r, acc_r
    else:
        tl, bl, tr, br = init_acc(bias_ptr, n0 // BLOCK_N, BLOCK_M, BLOCK_N, MFMA, ADD_BIAS)
    return tl, bl, tr, br


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


@gluon.constexpr_function
def pair_layout(rows):
    """The MFMA layout of a (rows, 128) band of a quadrant viewed as (rows, col 64, col 4 .. 16,
    col 1, 2, 32): the last dimension is a lane's first 8 registers, columns c .. c + 3 of two
    blocks 32 columns apart."""
    reg = [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 4], [0, 1, 0, 0]]
    if rows >= 64:
        reg.append([32, 0, 0, 0])
    if rows >= 128:
        reg.append([64, 0, 0, 0])
    return gl.DistributedLinearLayout(
        reg_bases=reg,
        lane_bases=[[1, 0, 0, 0], [2, 0, 0, 0], [4, 0, 0, 0], [8, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0]],
        warp_bases=[[0, 0, 4, 0], [16, 0, 0, 0]], block_bases=[], shape=[rows, 2, 8, 8])


@gluon.jit
def to_pairs(x, ROWS: gl.constexpr):
    """(ROWS, 128) in the MFMA layout -> (ROWS, 2, 8, 8) in pair_layout(ROWS); no data moves."""
    y = gl.permute(gl.reshape(x, [ROWS, 2, 2, 8, 4]), [0, 1, 3, 2, 4])
    return gl.convert_layout(gl.reshape(y, [ROWS, 2, 8, 8]), pair_layout(ROWS), assert_trivial=True)


@gluon.jit
def from_pairs(y, ROWS: gl.constexpr, mfmaLayout: gl.constexpr):
    x = gl.permute(gl.reshape(y, [ROWS, 2, 8, 2, 4]), [0, 1, 3, 2, 4])
    return gl.convert_layout(gl.reshape(x, [ROWS, 128]), mfmaLayout, assert_trivial=True)


@gluon.jit
def pair_offsets(qrow0, ROWS: gl.constexpr):
    """Element offsets in a quadrant of the bf16 pair map, for quadrant rows qrow0 .. + ROWS:
    [8 register-vector pairs][4 warps][64 lanes][8 bf16], so lane l of a wave holds elements
    8 l .. 8 l + 7 (16 bytes) of its 1 KiB."""
    L4: gl.constexpr = pair_layout(ROWS)
    d0 = gl.arange(0, ROWS, gl.SliceLayout(1, gl.SliceLayout(2, gl.SliceLayout(3, L4))))[:, None, None, None]
    d1 = gl.arange(0, 2, gl.SliceLayout(0, gl.SliceLayout(2, gl.SliceLayout(3, L4))))[None, :, None, None]
    d2 = gl.arange(0, 8, gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(3, L4))))[None, None, :, None]
    e = gl.arange(0, 8, gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, L4))))[None, None, None, :]
    row = qrow0 + d0
    vp = d1 + 2 * (row // 32 % 2) + 4 * (row // 64)
    w = d2 // 4 + 2 * (row // 16 % 2)
    lane = row % 16 + 16 * (d2 % 4)
    return ((vp * 4 + w) * 64 + lane) * 8 + e


@gluon.jit
def to_band(x, G: gl.constexpr):
    """(G, 32, 32) aligned blocks -> their 2-D band (G = 1, 2, 4: 32 x 32 G; 8, 16: 8 G x 128).
    Block index bits from the lowest: col 32, col 64, row 32, row 64."""
    if G == 1:
        y = gl.reshape(x, [32, 32])
    elif G == 2:
        y = gl.reshape(gl.permute(x, [1, 0, 2]), [32, 64])
    elif G == 4:
        y = gl.reshape(gl.permute(gl.reshape(x, [2, 2, 32, 32]), [2, 0, 1, 3]), [32, 128])
    elif G == 8:
        y = gl.reshape(gl.permute(gl.reshape(x, [2, 2, 2, 32, 32]), [0, 3, 1, 2, 4]), [64, 128])
    else:
        y = gl.reshape(gl.permute(gl.reshape(x, [2, 2, 2, 2, 32, 32]), [0, 1, 4, 2, 3, 5]), [128, 128])
    return y


@gluon.jit
def band_pos(g0, layout: gl.constexpr, ROWS: gl.constexpr, COLS: gl.constexpr):
    """Tile rows and columns of the band of the aligned blocks g0 .. g0 + G - 1."""
    r = gl.arange(0, ROWS, gl.SliceLayout(1, layout))[:, None]
    c = gl.arange(0, COLS, gl.SliceLayout(0, layout))[None, :]
    q = g0 // 16
    v0 = g0 % 16
    row0 = (v0 // 4) * 32
    col0 = (v0 % 2) * 32 + (v0 // 2 % 2) * 64
    return (q % 2) * 128 + row0 + r, (q // 2) * 128 + col0 + c


@gluon.jit
def fixup_quadrant(
    acc,
    q: gl.constexpr,
    p_ptr,
    bias_ptr,
    spid,
    n0,
    is_partial,
    n_peers,
    c_base,
    c_offsets,
    PEER_STEP: gl.constexpr,
    BLOCK_N: gl.constexpr,
    HALF_M: gl.constexpr,
    HALF_N: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    PARTIALS: gl.constexpr,
    out_dtype: gl.constexpr,
    gStoreLayoutC: gl.constexpr,
    mfmaLayout: gl.constexpr,
    STORE_CACHE: gl.constexpr = ".wt",
    rows_left=0,
    cols_left=0,
    EDGE_M: gl.constexpr = False,
    EDGE_N: gl.constexpr = False,
    rows_skip=0,
    cols_skip=0,
    SHIFT: gl.constexpr = False,
):
    """Quadrant q (tl, bl, tr, br) of a stream-K segment's accumulator. A contributor stores it
    to its slot P[spid] (cache modifier STORE_CACHE: write-through, or the default policy for the
    l2 exchange); the owner adds the partials of its n_peers peers, spid + PEER_STEP * i
    for i = 1 .. n_peers (the following programs for PEER_STEP = 1, the preceding ones for -1),
    and the bias, and stores C.

    Partials are 128x128 quadrants stored straight from the MFMA layout, with no layout
    conversion, and read back into the same layout. PARTIALS 0: fp32, 16 bytes per lane
    (lane_contiguous_offsets); 1: bf16 in the same map, 8 bytes per lane; 2: bf16 pairs, a
    lane's 8 registers of two blocks 32 columns apart as 16 bytes (pair_offsets). The peers are
    always summed in fp32. Both roles run the same code with run-time guards: a branch or loop
    that carries the AGPR accumulator spills, so the peers are summed in VGPRs and added in once.

    rows_left / cols_left (EDGE_M / EDGE_N): the tile's rows and columns before the edge of C;
    partials stay whole quadrants, only the C store and the bias load are masked.
    """
    QUAD: gl.constexpr = HALF_M * HALF_N
    SLOT: gl.constexpr = 4 * QUAD + P_SLOT_PAD
    offs_pn = gl.arange(0, HALF_N, gl.SliceLayout(0, mfmaLayout))
    if PARTIALS == 2:
        p_offsets = pair_offsets(0, HALF_M)
    else:
        p_offsets = lane_contiguous_offsets(HALF_M, HALF_N, mfmaLayout)
    if is_partial != 0:
        if PARTIALS == 0:
            part = acc
        elif PARTIALS == 1:
            part = acc.to(gl.bfloat16)
        else:
            part = to_pairs(acc.to(gl.bfloat16), HALF_M)
        gl.amd.cdna3.buffer_store(
            stored_value=part, ptr=p_ptr + spid.to(gl.int64) * SLOT + q * QUAD, offsets=p_offsets,
            cache=STORE_CACHE,
        )
    if PARTIALS == 2:
        pp = gl.zeros((HALF_M, 2, 8, 8), gl.float32, pair_layout(HALF_M))
        for j in range(n_peers):
            peer = (spid + PEER_STEP * (1 + j)).to(gl.int64)
            pp += gl.amd.cdna3.buffer_load(
                ptr=p_ptr + peer * SLOT + q * QUAD, offsets=p_offsets, cache=".cv"
            ).to(gl.float32)
        psum = from_pairs(pp, HALF_M, mfmaLayout)
    else:
        psum = gl.zeros((HALF_M, HALF_N), gl.float32, mfmaLayout)
        for j in range(n_peers):
            peer = (spid + PEER_STEP * (1 + j)).to(gl.int64)
            psum += gl.amd.cdna3.buffer_load(
                ptr=p_ptr + peer * SLOT + q * QUAD, offsets=p_offsets, cache=".cv"
            ).to(gl.float32)
    if is_partial == 0:
        if ADD_BIAS:
            # Only the owner adds the bias, once; contributors' partials exclude it.
            if EDGE_N:
                bias = gl.amd.cdna3.buffer_load(
                    ptr=bias_ptr + n0 + (q // 2) * HALF_N, offsets=offs_pn,
                    mask=offs_pn < cols_left - (q // 2) * HALF_N, other=0.0,
                )
            else:
                bias = gl.amd.cdna3.buffer_load(
                    ptr=bias_ptr + n0 + (q // 2) * HALF_N, offsets=offs_pn
                )
            psum = psum + bias.to(gl.float32)[None, :]
        store_c(acc + psum, c_base, c_offsets, out_dtype, gStoreLayoutC,
                rows_left - (q % 2) * HALF_M, cols_left - (q // 2) * HALF_N, EDGE_M, EDGE_N,
                rows_skip - (q % 2) * HALF_M, cols_skip - (q // 2) * HALF_N, SHIFT)


@gluon.jit
def atomic_quadrant(
    acc,
    q: gl.constexpr,
    bias_ptr,
    n0,
    first,
    c_base,
    c_offsets,
    BLOCK_N: gl.constexpr,
    HALF_N: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    out_dtype: gl.constexpr,
    atomLayoutC: gl.constexpr,
    mfmaLayout: gl.constexpr,
    rows_left=0,
    cols_left=0,
    EDGE_M: gl.constexpr = False,
    EDGE_N: gl.constexpr = False,
    rows_skip=0,
    cols_skip=0,
    SHIFT: gl.constexpr = False,
):
    """Quadrant q of a split tile's partial, added into C (zeroed by the tile's programs) with
    packed 16-bit atomics (v22): every program of the tile adds its own, in any order. first
    (1.0 on role 0, else 0.0) scales the bias, so it is added once with no branch around the
    accumulator.

    The layout gives each lane 2 adjacent elements (one packed dword) and a wave's 64 lanes one
    row of the quadrant, so an instruction covers 256 contiguous bytes: straight from the MFMA
    layout the same adds are 3x slower (experiments/streamk_atomics)."""
    if ADD_BIAS:
        offs_pn = gl.arange(0, HALF_N, gl.SliceLayout(0, mfmaLayout))
        if EDGE_N:
            bias = gl.amd.cdna3.buffer_load(ptr=bias_ptr + n0 + (q // 2) * HALF_N, offsets=offs_pn,
                                            mask=offs_pn < cols_left - (q // 2) * HALF_N, other=0.0)
        else:
            bias = gl.amd.cdna3.buffer_load(ptr=bias_ptr + n0 + (q // 2) * HALF_N, offsets=offs_pn)
        acc = acc + bias.to(gl.float32)[None, :] * first
    v = gl.convert_layout(acc.to(out_dtype), atomLayoutC)
    if EDGE_M or EDGE_N or SHIFT:
        HALF_M: gl.constexpr = acc.shape[0]
        mask = edge_mask(rows_left - (q % 2) * HALF_M, cols_left - (q // 2) * HALF_N, HALF_M, HALF_N,
                         atomLayoutC, EDGE_M, EDGE_N, rows_skip - (q % 2) * HALF_M, cols_skip - (q // 2) * HALF_N,
                         SHIFT)
        gl.amd.cdna4.buffer_atomic_add(c_base, c_offsets, v, mask=mask, sem="relaxed", scope="gpu")
    else:
        gl.amd.cdna4.buffer_atomic_add(c_base, c_offsets, v, sem="relaxed", scope="gpu")


@gluon.jit
def zero_tile_rows(c_ptr, m0, n0, j, n_tile, stride_cm, stride_cn, BLOCK_M: gl.constexpr,
                   BLOCK_N: gl.constexpr, M=0, N=0, EDGE_M: gl.constexpr = False, EDGE_N: gl.constexpr = False,
                   rows_skip=0, cols_skip=0, SHIFT: gl.constexpr = False):
    """Role j of n_tile zeroes its share of the tile's rows of C, write-through (.wt), so that the
    zeros reach memory and invalidate other XCDs' copies before the tile's counter moves. With
    EDGE_M / EDGE_N only rows below M and columns below N."""
    ZR: gl.constexpr = 8
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 64], [gl.num_warps(), 1], [1, 0])
    rows = gl.arange(0, ZR, gl.SliceLayout(1, layout))
    cols_rel = gl.arange(0, BLOCK_N, gl.SliceLayout(0, layout))
    cols = n0 + cols_rel
    per = (BLOCK_M + n_tile - 1) // n_tile
    r0 = j * per
    r1 = gl.minimum(r0 + per, BLOCK_M)
    if EDGE_M:
        r1 = gl.minimum(r1, M - m0)
    if SHIFT:
        r0 = gl.maximum(r0, rows_skip)
    zero = gl.zeros([ZR, BLOCK_N], c_ptr.dtype.element_ty, layout)
    base = c_ptr + m0.to(gl.int64) * stride_cm
    for r in range(r0, r1, ZR):
        rr = r + rows
        offs = rr[:, None] * stride_cm + stride_cn * cols[None, :]
        if SHIFT:
            mask = (rr[:, None] < r1) & (cols_rel[None, :] >= cols_skip)
        elif EDGE_N:
            mask = (rr[:, None] < r1) & (cols[None, :] < N)
        else:
            mask = (rr[:, None] < r1) & (cols[None, :] >= 0)
        gl.amd.cdna3.buffer_store(stored_value=zero, ptr=base, offsets=offs, mask=mask, cache=".wt")


@gluon.jit
def streamk_chunk_of(spid, EXTRA: gl.constexpr, N_HI: gl.constexpr, N_LO: gl.constexpr,
                     R_HI: gl.constexpr, R_LO: gl.constexpr):
    """(stream-K tile, chunk index, programs on that tile, reduce-scatter readers of that tile)
    of program spid: the first EXTRA tiles get N_HI consecutive programs each (R_HI of them
    read), the rest N_LO (R_LO)."""
    head: gl.constexpr = EXTRA * N_HI
    t = spid // N_HI
    j = spid % N_HI
    n = spid * 0 + N_HI
    r = spid * 0 + R_HI
    if spid >= head:
        t = EXTRA + (spid - head) // N_LO
        j = (spid - head) % N_LO
        n = spid * 0 + N_LO
        r = spid * 0 + R_LO
    return t, j, n, r


@gluon.jit
def streamk_chunk_major_of(spid, S: gl.constexpr, EXTRA: gl.constexpr, N_HI: gl.constexpr, N_LO: gl.constexpr,
                           R_HI: gl.constexpr, R_LO: gl.constexpr):
    """As streamk_chunk_of, in v18's chunk-major order: program spid runs work item
    w = j * S + t (chunk j of tile t). Every tile has chunks 0 .. N_LO - 1; the first EXTRA
    tiles' chunk N_LO comes after all the others. Consecutive spids, and so one XCD, hold the
    same chunk of neighbouring tiles, which read the same k-steps at the same time."""
    t = spid % S
    j = spid // S
    if spid >= N_LO * S:
        t = spid - N_LO * S
        j = spid * 0 + N_LO
    n = spid * 0 + N_LO
    r = spid * 0 + R_LO
    if t < EXTRA:
        n = spid * 0 + N_HI
        r = spid * 0 + R_HI
    return t, j, n, r


@gluon.jit
def streamk_slot(t, i, EXTRA: gl.constexpr, N_HI: gl.constexpr, N_LO: gl.constexpr):
    """Workspace slot of role i of stream-K tile t: the spid that role would have in the
    tile-major order, so a tile's slots, flags and barrier are b .. b + n - 1 with b its first
    slot, whichever program runs the role."""
    b = t * N_HI
    if t >= EXTRA:
        b = EXTRA * N_HI + (t - EXTRA) * N_LO
    return b + i


@gluon.jit
def xcd_local_chunk_of(spid, S: gl.constexpr, PER: gl.constexpr, NUM_XCDS: gl.constexpr, N: gl.constexpr):
    """(stream-K tile, chunk index, programs on that tile) of XCD-mapped program spid in v21's
    XCD-local split, with 0 programs if it idles. XCD x = spid // PER owns S // NUM_XCDS
    consecutive stream-K tiles (one more for x < S % NUM_XCDS); its local slot l = spid % PER
    works on its tile l // N as chunk l % N, so a tile's N programs are consecutive spids on
    one XCD. The host checks ceil(S / NUM_XCDS) * N <= PER."""
    LO: gl.constexpr = S // NUM_XCDS
    TALL: gl.constexpr = S % NUM_XCDS
    x = spid // PER
    local = spid % PER
    tl = local // N
    t = x * LO + gl.minimum(x, TALL) + tl
    n = N * (tl < LO + (x < TALL).to(gl.int32)).to(gl.int32)
    return t, local % N, n


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
def tile_barrier(sync_ptr, b, n, wait, slot, epoch, NUM_PROGRAMS: gl.constexpr, L2: gl.constexpr):
    """Arrive on the barrier of the tile whose first slot is b and, if wait, wait until all n
    of its programs have published their partials (a program that reads no slice only
    arrives). sync[2b] counts arrivals and sync[2b + 1] is a generation: the last arriver
    resets the count and bumps the generation, so the barrier is re-armed for the next launch
    without host state. Scalar atomics run once per workgroup and broadcast their result.

    Every atomic is relaxed, so no buffer_wbl2 / buffer_inv sc1 is emitted. The caller wrote
    its partial through (.wt) and drained, so it is in memory before the arrival; the slice
    reads it with .cv loads, which miss L1. s_waitcnt keeps the generation read ahead of the
    arrival, so it is always the old one.

    L2 (STREAMK_EXCHANGE=l2, v21; every program of the tile on one XCD): no atomics. The program
    stores the launch's epoch to its flag sync[2 NUM_PROGRAMS + slot], then polls the tile's n
    flags with sc0 nt loads, which miss L1 and hit the L2 that every program of the tile shares;
    the flags never leave it until the kernel ends. Each flag has one writer and the host bumps
    the epoch every launch, so nothing needs re-arming. The slice then reads the partials with
    sc0 nt loads too, which always miss L1.
    """
    if L2:
        flags = sync_ptr + 2 * NUM_PROGRAMS
        one = _one()
        gl.store(flags + slot + one * 0, epoch + one * 0)
        if wait:
            FL: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
            m = gl.arange(0, 64, FL)
            mask = m < n
            done = 0
            while done == 0:
                asm_fence("s_nop 0")  # keeps the poll inside the loop
                f = gl.amd.cdna3.buffer_load(ptr=flags + b, offsets=m, mask=mask, other=epoch, cache=".cg")
                done = (gl.min(f, axis=0) == epoch).to(gl.int32)
        asm_fence("s_nop 0")  # no partial read moves above the wait
    else:
        cnt_ptr = sync_ptr + 2 * b
        gen_ptr = cnt_ptr + 1
        g0 = gl.atomic_add(gen_ptr, 0, sem="relaxed", scope="gpu")
        # If the last arrival and the bump overtook this read, the program would wait for a
        # generation that never comes.
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
def reduce_scatter_slice(
    p_ptr,
    bias_ptr,
    c_tile,
    b,
    j,
    n,
    readers,
    n0,
    stride_cm,
    stride_cn,
    BLOCK_N: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    NB: gl.constexpr,
    U: gl.constexpr,
    BAND: gl.constexpr,
    PARTIALS: gl.constexpr,
    out_dtype: gl.constexpr,
    mfmaLayout: gl.constexpr,
    LOAD_CACHE: gl.constexpr = ".cv",
    rows_left=0,
    cols_left=0,
    EDGE_M: gl.constexpr = False,
    EDGE_N: gl.constexpr = False,
    rows_skip=0,
    cols_skip=0,
    SHIFT: gl.constexpr = False,
):
    """Reader j's share of a tile split n ways, of which the first `readers` programs reduce:
    sum its blocks over the n partials in slots P[b .. b + n - 1] (loaded with LOAD_CACHE: .cv,
    or .cg for the l2 exchange), add the bias, and store them to C.

    A slot is 64 blocks of 1024 elements, one per (quadrant, register vector) of the MFMA layout
    (fixup_quadrant's maps); each block is a 32x32 square of the tile. Reader j takes blocks
    [64 j / readers, 64 (j + 1) / readers), NB at a time, and reads U partials of each group
    per trip, so NB * U loads per lane are in flight together. Sums are fp32.

    With BAND (readers and NB powers of two, so every group is NB whole, aligned blocks), a
    group is reshaped to its 2-D band and converted to a row store layout (2 LDS rounds at 128
    columns): 16 bytes per lane along a C row (8 at NB = 1). Without, the store is v20's, from
    the slice layout, 8 bytes per lane, 4 neighbouring lanes on 4 rows. bf16 pairs with BAND and
    NB >= 4 read the band in the pair layout (16 bytes per lane) and convert from the MFMA layout
    like the owner's store; otherwise every map is read in the slice layout.

    With EDGE_M / EDGE_N the C stores keep the tile's first rows_left rows and cols_left columns,
    and the bias loads its first cols_left columns.
    """
    SLOT: gl.constexpr = 4 * 128 * 128 + P_SLOT_PAD
    blk_lo = (64 * j) // readers
    blk_hi = (64 * (j + 1)) // readers
    if PARTIALS == 2 and BAND and NB >= 4:
        ROWS: gl.constexpr = NB * 8
        gStoreLayoutC: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
        offs_bn = gl.arange(0, 128, gl.SliceLayout(0, mfmaLayout))
        for g in range(blk_lo, blk_hi, NB):
            # Aligned blocks g .. g + NB - 1 are quadrant rows (g % 16) * 8 .. + ROWS, all columns.
            q = g // 16
            off = q * 16384 + pair_offsets((g % 16) * 8, ROWS)
            s4 = gl.zeros((ROWS, 2, 8, 8), gl.float32, pair_layout(ROWS))
            for p0 in range(0, n, U):
                for u in gl.static_range(U):
                    slot = p_ptr + (b + p0 + u).to(gl.int64) * SLOT
                    s4 += gl.amd.cdna3.buffer_load(
                        ptr=slot, offsets=off, mask=(off >= 0) & (p0 + u < n), other=0.0, cache=LOAD_CACHE
                    ).to(gl.float32)
            s2 = from_pairs(s4, ROWS, mfmaLayout)
            if ADD_BIAS:
                if EDGE_N:
                    bias = gl.amd.cdna3.buffer_load(ptr=bias_ptr + n0 + (q // 2) * 128, offsets=offs_bn,
                                                    mask=offs_bn < cols_left - (q // 2) * 128, other=0.0)
                else:
                    bias = gl.amd.cdna3.buffer_load(ptr=bias_ptr + n0 + (q // 2) * 128, offsets=offs_bn)
                s2 = s2 + bias.to(gl.float32)[None, :]
            cs = gl.convert_layout(s2.to(out_dtype), gStoreLayoutC)
            trow, tcol = band_pos(g, gStoreLayoutC, ROWS, 128)
            if SHIFT:
                gl.amd.cdna3.buffer_store(stored_value=cs, ptr=c_tile, offsets=trow * stride_cm + tcol * stride_cn,
                                          mask=(trow >= rows_skip) & (tcol >= cols_skip))
            elif EDGE_M or EDGE_N:
                gl.amd.cdna3.buffer_store(stored_value=cs, ptr=c_tile, offsets=trow * stride_cm + tcol * stride_cn,
                                          mask=(trow < rows_left) & (tcol < cols_left))
            else:
                gl.amd.cdna3.buffer_store(stored_value=cs, ptr=c_tile, offsets=trow * stride_cm + tcol * stride_cn)
    else:
        # Rows before columns in lane order: lane l gets row l % 16 and columns 4 (l // 16) .. + 3,
        # which in_blk puts at 4 l .. 4 l + 3 (fp32 and bf16 maps), so each 4-lane group reads 64
        # contiguous bytes. Columns first, lanes 0-3 hit 4 lines 256 bytes apart and the L1 stalls
        # on tag conflicts (reads take twice as long).
        L3: gl.constexpr = gl.BlockedLayout([1, 1, 4], [1, 16, 4], [1, 2, 2], [1, 2, 0])
        i = gl.arange(0, NB, gl.SliceLayout(1, gl.SliceLayout(2, L3)))[:, None, None]
        r = gl.arange(0, 32, gl.SliceLayout(0, gl.SliceLayout(2, L3)))[None, :, None]
        c = gl.arange(0, 32, gl.SliceLayout(0, gl.SliceLayout(1, L3)))[None, None, :]
        if PARTIALS == 2:
            # pair_offsets restricted to one block; the block's bits are added below.
            in_blk = (r % 16) * 8 + (r // 16) * 1024 + c + (c // 4 % 4) * 124 + (c // 16) * 496
        else:
            # lane_contiguous_offsets restricted to one block: rows 0-31 and columns 0-31.
            in_blk = (r % 16) * 4 + (r // 16) * 512 + c + (c // 4 % 4) * 60 + (c // 16) * 240
        cb = gl.arange(0, 32, gl.SliceLayout(0, gl.SliceLayout(1, L3)))[None, :]
        ib = gl.arange(0, NB, gl.SliceLayout(1, gl.SliceLayout(1, L3)))[:, None]
        for g in range(blk_lo, blk_hi, NB):
            blk = g + i
            valid = (blk < blk_hi) & (in_blk >= 0)
            # Block q * 16 + v: quadrant q (tl, bl, tr, br), and the register vector v's bits are
            # column 32, column 64, row 32, row 64.
            q = blk // 16
            v = blk % 16
            if PARTIALS == 2:
                p_off = q * 16384 + (v % 2) * 4 + (v // 2 % 2) * 2048 + (v // 4 % 2) * 4096 + (v // 8) * 8192 + in_blk
            else:
                p_off = blk * 1024 + in_blk
            s = gl.zeros((NB, 32, 32), gl.float32, L3)
            for p0 in range(0, n, U):
                for u in gl.static_range(U):
                    slot = p_ptr + (b + p0 + u).to(gl.int64) * SLOT
                    s += gl.amd.cdna3.buffer_load(
                        ptr=slot, offsets=p_off, mask=valid & (p0 + u < n), other=0.0, cache=LOAD_CACHE
                    ).to(gl.float32)
            if ADD_BIAS:
                qb = (g + ib) // 16
                vb = (g + ib) % 16
                col_b = (qb // 2) * 128 + (vb % 2) * 32 + (vb // 2 % 2) * 64 + cb
                bias_mask = (g + ib) < blk_hi
                if EDGE_N:
                    bias_mask = bias_mask & (col_b < cols_left)
                bias = gl.amd.cdna3.buffer_load(
                    ptr=bias_ptr + n0, offsets=col_b, mask=bias_mask, other=0.0
                )
                s = s + bias.to(gl.float32)[:, None, :]
            if BAND:
                ROWS: gl.constexpr = 32 if NB <= 4 else NB * 8
                COLS: gl.constexpr = 32 * NB if NB <= 4 else 128
                if COLS == 128:
                    SL: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
                elif COLS == 64:
                    SL: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])
                else:
                    SL: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [4, 1], [1, 0])
                cs = gl.convert_layout(to_band(s.to(out_dtype), NB), SL)
                trow, tcol = band_pos(g, SL, ROWS, COLS)
                if SHIFT:
                    gl.amd.cdna3.buffer_store(stored_value=cs, ptr=c_tile, offsets=trow * stride_cm + tcol * stride_cn,
                                              mask=(trow >= rows_skip) & (tcol >= cols_skip))
                elif EDGE_M or EDGE_N:
                    gl.amd.cdna3.buffer_store(stored_value=cs, ptr=c_tile, offsets=trow * stride_cm + tcol * stride_cn,
                                              mask=(trow < rows_left) & (tcol < cols_left))
                else:
                    gl.amd.cdna3.buffer_store(stored_value=cs, ptr=c_tile, offsets=trow * stride_cm + tcol * stride_cn)
            else:
                row = (q % 2) * 128 + (v // 4 % 2) * 32 + (v // 8) * 64 + r
                col = (q // 2) * 128 + (v % 2) * 32 + (v // 2 % 2) * 64 + c
                store_mask = valid
                if SHIFT:
                    store_mask = store_mask & (row >= rows_skip) & (col >= cols_skip)
                elif EDGE_M or EDGE_N:
                    store_mask = store_mask & (row < rows_left) & (col < cols_left)
                gl.amd.cdna3.buffer_store(
                    stored_value=s.to(out_dtype), ptr=c_tile, offsets=row * stride_cm + col * stride_cn,
                    mask=store_mask,
                )


@gluon.jit
def streamk_segment(
    tile_id,
    k0,
    n_pairs,
    is_partial,
    n_peers,
    spid,
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
    PEER_STEP: gl.constexpr,
    REDUCE_SCATTER: gl.constexpr,
    RS_NB: gl.constexpr,
    RS_U: gl.constexpr,
    RS_BAND: gl.constexpr,
    PARTIALS: gl.constexpr,
    mfmaLayout: gl.constexpr,
    dotOpLayoutA: gl.constexpr,
    dotOpLayoutB: gl.constexpr,
    ATOMIC_C: gl.constexpr,
    L2: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    epoch,
    M,
    N,
    EDGE_M: gl.constexpr,
    EDGE_N: gl.constexpr,
    gLoadLayoutA: gl.constexpr,
    gLoadLayoutB: gl.constexpr,
    EDGE_K: gl.constexpr = False,
    K_PAD: gl.constexpr = 0,
    K_RES: gl.constexpr = 0,
    SHIFT: gl.constexpr = False,
):
    """K-steps [k0, k0 + 2 * n_pairs) of tile `tile_id`, n_pairs >= 1, then the segment's
    ending: publish the partial (is_partial) or collect n_peers partials and store C.

    EDGE_M / EDGE_N (v25): the tile's loads use clamped offsets (clamped_tile_offsets) and its C
    stores, zeroing and bias loads are masked to M x N. Partials stay whole tiles.

    EDGE_K (v26): pair p of the tile (K-steps 2p, 2p + 1) lies at columns 2 BLOCK_K (p - 1) for p >= 1,
    and pair 0 is the last, partial pair, at K_PAD - 2 BLOCK_K, loaded with the lanes past K masked.
    So only the segment that starts at k0 = 0 masks, in its prologue, and every main-loop read starts
    at a whole pair from the row's start.

    spid is the program's workspace slot and j its role in the tile (streamk_slot): the same as
    its stream-K id and chunk index, except under chunk-major placement.

    With REDUCE_SCATTER (tile-aligned one-tile only) the ending is instead: if the tile is
    split (n_tile > 1), every program publishes and arrives on the tile's barrier; the first
    r_tile of them wait on it and each reduces its slice (reduce_scatter_slice); the others
    are done. A whole tile stores C directly. L2 keeps the partials and the barrier's flags in
    the XCD's L2 (default-policy stores, sc0 nt loads, tile_barrier's epoch flags).

    With ATOMIC_C (tile-aligned one-tile only, v22's atomic_iz) every program adds its partial
    into C with packed 16-bit atomics (atomic_quadrant): no workspace, flags, barrier or slice.
    Before its K loop each program zeroes 1/n of the tile's rows and arrives on the tile's counter
    (sync[3 NUM_PROGRAMS + 2b]); before its adds it waits for all n arrivals, which the K loop has
    almost always covered. A second count (the word after it) of programs past their adds lets the
    last one re-arm both for the next launch. A whole tile (n_tile = 1) takes the same path, one add
    into zero. The counts have their own words: the reduce-scatter barrier's generation at
    sync[2b + 1] only ever goes up, so it cannot double as a count that starts at zero.

    Starts with its own prologue and runs every pair through the main-loop body; the last
    pair's prefetch reloads the last pair (clamped), which the drain discards.
    """
    pid_m, pid_n = get_group_m_tile_ids(tile_id, num_tiles_m, num_tiles_n, GROUP_SIZE_M)
    m0 = tile_origin(pid_m, M, BLOCK_M, SHIFT)
    n0 = tile_origin(pid_n, N, BLOCK_N, SHIFT)
    skip_m = pid_m * BLOCK_M - m0
    skip_n = pid_n * BLOCK_N - n0
    rows_left = M - m0
    cols_left = N - n0
    if ATOMIC_C:
        zero_tile_rows(c_ptr, m0, n0, j, n_tile, stride_cm, stride_cn, BLOCK_M, BLOCK_N, M, N, EDGE_M, EDGE_N,
                       skip_m, skip_n, SHIFT)
        # The zeros must have reached memory (write-through, drained in every wave) before the
        # arrival: the tile's other programs add into those lines as soon as they see n arrivals.
        drain_and_barrier()
        gl.atomic_add(sync_ptr + 3 * NUM_PROGRAMS + 2 * (spid - j), 1, sem="relaxed", scope="gpu")

    a_top_off, a_bot_off = a_offsets, a_offsets
    if EDGE_M:
        a_top_off, a_bot_off = clamped_tile_offsets(rows_left - 1, stride_am, stride_ak, BLOCK_M // 2, BLOCK_K,
                                                    gLoadLayoutA, True)
    b_top_off, b_bot_off = b_offsets, b_offsets
    if EDGE_N:
        b_top_off, b_bot_off = clamped_tile_offsets(cols_left - 1, stride_bn, stride_bk, BLOCK_N // 2, BLOCK_K,
                                                    gLoadLayoutB, False)
    a0, a1, a2, a3, ah, ad = operand_offsets(a_top_off, a_bot_off, a_offsets, a_offsets_next, a_half,
                                             BLOCK_K * stride_ak, EDGE_M)
    b0, b1, b2, b3, bh, bd = operand_offsets(b_top_off, b_bot_off, b_offsets, b_offsets_next, b_half,
                                             BLOCK_K * stride_bk, EDGE_N)

    if EDGE_K:
        # Pair i of the segment (i >= 1) at a_tile + 2 BLOCK_K i; its first pair at a_tile, or for
        # k0 = 0 the last, partial pair, masked at K.
        a_row = a_ptr + m0.to(gl.int64) * stride_am
        b_row = b_ptr + n0.to(gl.int64) * stride_bn
        a_tile = a_row + (k0 - 2).to(gl.int64) * BLOCK_K * stride_ak
        b_tile = b_row + (k0 - 2).to(gl.int64) * BLOCK_K * stride_bk
        first_col = gl.where(k0 == 0, K_PAD - 2 * BLOCK_K, (k0 - 2) * BLOCK_K).to(gl.int64)
        k_hi = gl.where(k0 == 0, 2 * BLOCK_K - K_RES, 2 * BLOCK_K)
        ma0, mb0, ma1, mb1 = tail_pair_masks(k_hi, K_RES, BLOCK_M, BLOCK_N, BLOCK_K, gLoadLayoutA, gLoadLayoutB)
        a_top, b_left = load_first_pair(
            a_row + first_col * stride_ak, b_row + first_col * stride_bk, smemA_top, smemA_bot, smemB_left,
            smemB_right, a0, b0, ah, bh, a1, b1,
            dotOpLayoutA, dotOpLayoutB, a2, b2, a3, b3, ad, bd, ma0, mb0, ma1, mb1,
        )
        # The last pair's clamped prefetch must stay inside the row: for k0 = 0, a_tile itself is a
        # pair before it.
        nxt_lo = (k0 == 0).to(gl.int64) * 2
    else:
        a_tile = a_ptr + m0.to(gl.int64) * stride_am + k0.to(gl.int64) * BLOCK_K * stride_ak
        b_tile = b_ptr + n0.to(gl.int64) * stride_bn + k0.to(gl.int64) * BLOCK_K * stride_bk
        a_top, b_left = load_first_pair(
            a_tile, b_tile, smemA_top, smemA_bot, smemB_left, smemB_right,
            a0, b0, ah, bh, a1, b1,
            dotOpLayoutA, dotOpLayoutB, a2, b2, a3, b3, ad, bd,
        )

    acc_tl, acc_bl, acc_tr, acc_br = init_acc(bias_ptr, pid_n, BLOCK_M, BLOCK_N, mfmaLayout, False)
    for i in range(n_pairs):
        nxt = gl.minimum(i + 1, n_pairs - 1).to(gl.int64) * 2
        if EDGE_K:
            nxt = gl.maximum(nxt, nxt_lo)
        acc_tl, acc_bl, acc_tr, acc_br, a_top, b_left = k_step_pair(
            acc_tl, acc_bl, acc_tr, acc_br, a_top, b_left,
            smemA_top, smemA_bot, smemB_left, smemB_right,
            a_tile + nxt * BLOCK_K * stride_ak, b_tile + nxt * BLOCK_K * stride_bk,
            a0, b0, ah, bh, a1, b1,
            dotOpLayoutA, dotOpLayoutB, a2, b2, a3, b3, ad, bd,
        )
    # The clamped prefetch is still in flight, and the next segment's prologue overwrites the
    # buffers other waves may still be reading.
    drain_and_barrier()

    gStoreLayoutC: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
    out_dtype: gl.constexpr = c_ptr.dtype.element_ty
    offs_cm = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gStoreLayoutC))
    c_tl_base = c_ptr + m0.to(gl.int64) * stride_cm
    col = n0 + offs_cn
    c_offsets = stride_cm * offs_cm[:, None] + stride_cn * col[None, :]
    c_tr_base = c_tl_base + BLOCK_N * stride_cn // 2
    c_bl_base = c_tl_base + BLOCK_M * stride_cm // 2
    c_br_base = c_bl_base + BLOCK_N * stride_cn // 2

    if ATOMIC_C:
        atomLayoutC: gl.constexpr = gl.BlockedLayout([1, 2], [1, 64], [4, 1], [1, 0])
        offs_am = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, atomLayoutC))
        offs_an = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, atomLayoutC))
        a_col = n0 + offs_an
        atom_offsets = stride_cm * offs_am[:, None] + stride_cn * a_col[None, :]
        first = (j == 0).to(gl.float32)
        zero_cnt = sync_ptr + 3 * NUM_PROGRAMS + 2 * (spid - j)
        while gl.atomic_add(zero_cnt, 0, sem="relaxed", scope="gpu") < n_tile:
            pass
        atomic_quadrant(acc_tl, 0, bias_ptr, n0, first, c_tl_base, atom_offsets, BLOCK_N, BLOCK_N // 2,
                        ADD_BIAS, out_dtype, atomLayoutC, mfmaLayout, rows_left, cols_left, EDGE_M, EDGE_N,
                        skip_m, skip_n, SHIFT)
        atomic_quadrant(acc_bl, 1, bias_ptr, n0, first, c_bl_base, atom_offsets, BLOCK_N, BLOCK_N // 2,
                        ADD_BIAS, out_dtype, atomLayoutC, mfmaLayout, rows_left, cols_left, EDGE_M, EDGE_N,
                        skip_m, skip_n, SHIFT)
        atomic_quadrant(acc_tr, 2, bias_ptr, n0, first, c_tr_base, atom_offsets, BLOCK_N, BLOCK_N // 2,
                        ADD_BIAS, out_dtype, atomLayoutC, mfmaLayout, rows_left, cols_left, EDGE_M, EDGE_N,
                        skip_m, skip_n, SHIFT)
        atomic_quadrant(acc_br, 3, bias_ptr, n0, first, c_br_base, atom_offsets, BLOCK_N, BLOCK_N // 2,
                        ADD_BIAS, out_dtype, atomLayoutC, mfmaLayout, rows_left, cols_left, EDGE_M, EDGE_N,
                        skip_m, skip_n, SHIFT)
        # Every program counts itself past its wait; the last re-arms both counts. The kernel
        # boundary orders these resets before the next launch.
        done = gl.atomic_add(zero_cnt + 1, 1, sem="relaxed", scope="gpu")
        if done == n_tile - 1:
            gl.atomic_xchg(zero_cnt, 0, sem="relaxed", scope="gpu")
            gl.atomic_xchg(zero_cnt + 1, 0, sem="relaxed", scope="gpu")
    elif REDUCE_SCATTER:
        # The four quadrants go to P[spid] if the tile is split, else straight to C (with
        # n_peers = 0 fixup_quadrant reads nothing back). The accumulator is dead afterwards.
        # With L2 the partials stay in this XCD's L2 (no cache bits); otherwise they are written
        # through.
        RS_STORE: gl.constexpr = "" if L2 else ".wt"
        RS_LOAD: gl.constexpr = ".cg" if L2 else ".cv"
        is_split = (n_tile > 1).to(gl.int32)
        fixup_quadrant(
            acc_tl, 0, p_ptr, bias_ptr, spid, n0, is_split, 0, c_tl_base, c_offsets,
            PEER_STEP, BLOCK_N, BLOCK_M // 2, BLOCK_N // 2, ADD_BIAS, PARTIALS, out_dtype, gStoreLayoutC, mfmaLayout,
            RS_STORE, rows_left, cols_left, EDGE_M, EDGE_N, skip_m, skip_n, SHIFT,
        )
        fixup_quadrant(
            acc_bl, 1, p_ptr, bias_ptr, spid, n0, is_split, 0, c_bl_base, c_offsets,
            PEER_STEP, BLOCK_N, BLOCK_M // 2, BLOCK_N // 2, ADD_BIAS, PARTIALS, out_dtype, gStoreLayoutC, mfmaLayout,
            RS_STORE, rows_left, cols_left, EDGE_M, EDGE_N, skip_m, skip_n, SHIFT,
        )
        fixup_quadrant(
            acc_tr, 2, p_ptr, bias_ptr, spid, n0, is_split, 0, c_tr_base, c_offsets,
            PEER_STEP, BLOCK_N, BLOCK_M // 2, BLOCK_N // 2, ADD_BIAS, PARTIALS, out_dtype, gStoreLayoutC, mfmaLayout,
            RS_STORE, rows_left, cols_left, EDGE_M, EDGE_N, skip_m, skip_n, SHIFT,
        )
        fixup_quadrant(
            acc_br, 3, p_ptr, bias_ptr, spid, n0, is_split, 0, c_br_base, c_offsets,
            PEER_STEP, BLOCK_N, BLOCK_M // 2, BLOCK_N // 2, ADD_BIAS, PARTIALS, out_dtype, gStoreLayoutC, mfmaLayout,
            RS_STORE, rows_left, cols_left, EDGE_M, EDGE_N, skip_m, skip_n, SHIFT,
        )
        if n_tile > 1:
            # The arrival may only become visible after all four waves' partial stores have completed.
            drain_and_barrier()
            is_reader = j < r_tile
            tile_barrier(sync_ptr, spid - j, n_tile, is_reader, spid, epoch, NUM_PROGRAMS, L2)
            if is_reader:
                reduce_scatter_slice(
                    p_ptr, bias_ptr, c_tl_base + n0 * stride_cn, spid - j, j, n_tile, r_tile,
                    n0, stride_cm, stride_cn, BLOCK_N, ADD_BIAS, RS_NB, RS_U, RS_BAND, PARTIALS, out_dtype,
                    mfmaLayout, RS_LOAD, rows_left, cols_left, EDGE_M, EDGE_N, skip_m, skip_n, SHIFT,
                )
    else:
        # Owner: every peer's flag before the first partial read. The partials are written
        # through, so the drain before each relaxed flag waited for them to reach memory, and
        # the .cv reads miss L1: no acquire, only a compiler fence so no read moves up.
        for pj in range(n_peers):
            while gl.atomic_add(locks_ptr + spid + PEER_STEP * (1 + pj), 0, sem="relaxed", scope="gpu") == 0:
                pass
        if n_peers > 0:
            asm_fence("s_nop 0")

        fixup_quadrant(
            acc_tl, 0, p_ptr, bias_ptr, spid, n0, is_partial, n_peers, c_tl_base, c_offsets,
            PEER_STEP, BLOCK_N, BLOCK_M // 2, BLOCK_N // 2, ADD_BIAS, PARTIALS, out_dtype, gStoreLayoutC, mfmaLayout,
            ".wt", rows_left, cols_left, EDGE_M, EDGE_N, skip_m, skip_n, SHIFT,
        )
        fixup_quadrant(
            acc_bl, 1, p_ptr, bias_ptr, spid, n0, is_partial, n_peers, c_bl_base, c_offsets,
            PEER_STEP, BLOCK_N, BLOCK_M // 2, BLOCK_N // 2, ADD_BIAS, PARTIALS, out_dtype, gStoreLayoutC, mfmaLayout,
            ".wt", rows_left, cols_left, EDGE_M, EDGE_N, skip_m, skip_n, SHIFT,
        )
        fixup_quadrant(
            acc_tr, 2, p_ptr, bias_ptr, spid, n0, is_partial, n_peers, c_tr_base, c_offsets,
            PEER_STEP, BLOCK_N, BLOCK_M // 2, BLOCK_N // 2, ADD_BIAS, PARTIALS, out_dtype, gStoreLayoutC, mfmaLayout,
            ".wt", rows_left, cols_left, EDGE_M, EDGE_N, skip_m, skip_n, SHIFT,
        )
        fixup_quadrant(
            acc_br, 3, p_ptr, bias_ptr, spid, n0, is_partial, n_peers, c_br_base, c_offsets,
            PEER_STEP, BLOCK_N, BLOCK_M // 2, BLOCK_N // 2, ADD_BIAS, PARTIALS, out_dtype, gStoreLayoutC, mfmaLayout,
            ".wt", rows_left, cols_left, EDGE_M, EDGE_N, skip_m, skip_n, SHIFT,
        )

        if is_partial != 0:
            # The flag may only become visible after all four waves' partial stores have completed.
            drain_and_barrier()
            gl.atomic_xchg(locks_ptr + spid, 1, sem="relaxed", scope="gpu")
        # Each partial has exactly one reader, so the owner re-arms its peers' flags for the next
        # launch; the kernel boundary orders these stores against it.
        for pj in range(n_peers):
            gl.atomic_xchg(locks_ptr + spid + PEER_STEP * (1 + pj), 0, sem="relaxed", scope="gpu")


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
    a_cur_top,
    a_cur_bot,
    b_cur_top,
    b_cur_bot,
    M,
    N,
    EDGE_M: gl.constexpr,
    EDGE_N: gl.constexpr,
    gLoadLayoutA: gl.constexpr,
    gLoadLayoutB: gl.constexpr,
    EDGE_K: gl.constexpr = False,
    K_RES: gl.constexpr = 0,
    SHIFT: gl.constexpr = False,
):
    """One full-K output tile for virtual pid `vpid`: v13's persistent-loop body.

    Expects the tile's K-steps 0 and 1 in flight in buffers 0 and 1 and its K-step 0
    operands in a_top / b_left. Prefetches the same for vpid + NUM_PROGRAMS (clamped to the
    last tile) and returns that tile's a_top / b_left.

    EDGE_M / EDGE_N (v25): a_cur_* / b_cur_* are this tile's clamped load offsets; the next
    tile's are built for its prefetch and returned with a_top / b_left, and the C stores and the
    bias are masked to M x N. Without them the a_cur_* / b_cur_* pass through unused.

    EDGE_K (v26): K is K_PAD here. The tile's first pair (in flight) is its last, partial pair, so the
    main loop starts at column 0, and the next tile's prefetch loads that tile's last pair, masked.
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
    a_next = a_ptr + tile_origin(next_m, M, BLOCK_M, SHIFT).to(gl.int64) * stride_am
    b_next = b_ptr + tile_origin(next_n, N, BLOCK_N, SHIFT).to(gl.int64) * stride_bn
    m0 = tile_origin(pid_m, M, BLOCK_M, SHIFT)
    n0 = tile_origin(pid_n, N, BLOCK_N, SHIFT)
    skip_m = pid_m * BLOCK_M - m0
    skip_n = pid_n * BLOCK_N - n0

    # K-steps 0 and 1 are already in flight, so the first prefetch is K-step 2 (with EDGE_K, the
    # pair in flight is the last one and the first prefetch is column 0).
    if EDGE_K:
        a_base = a_ptr + m0.to(gl.int64) * stride_am
        b_base = b_ptr + n0.to(gl.int64) * stride_bn
        a_next += (K - 2 * BLOCK_K) * stride_ak
        b_next += (K - 2 * BLOCK_K) * stride_bk
    else:
        a_base = a_ptr + m0.to(gl.int64) * stride_am + BLOCK_K * stride_ak * 2
        b_base = b_ptr + n0.to(gl.int64) * stride_bn + BLOCK_K * stride_bk * 2
    # The next tile's prefetch masks: MASK_TAIL_PREFETCH's has_next, and with EDGE_K the last pair's.
    if EDGE_K:
        pa0, pb0, pa1, pb1 = tail_pair_masks(2 * BLOCK_K - K_RES, K_RES, BLOCK_M, BLOCK_N, BLOCK_K,
                                             gLoadLayoutA, gLoadLayoutB)
        if MASK_TAIL_PREFETCH:
            pa1 = pa1 & has_next
            pb1 = pb1 & has_next
            if K_RES > BLOCK_K:
                pa0 = pa0 & has_next
                pb0 = pb0 & has_next
            else:
                pa0 = has_next
                pb0 = has_next
    else:
        pa0 = has_next
        pb0 = has_next
        pa1 = has_next
        pb1 = has_next
    rows_left = M - m0
    cols_left = N - n0
    a0, a1, a2, a3, ah, ad = operand_offsets(a_cur_top, a_cur_bot, a_offsets, a_offsets_next, a_half,
                                             BLOCK_K * stride_ak, EDGE_M)
    b0, b1, b2, b3, bh, bd = operand_offsets(b_cur_top, b_cur_bot, b_offsets, b_offsets_next, b_half,
                                             BLOCK_K * stride_bk, EDGE_N)

    ## K-steps 0 and 1, peeled: without bias, the constant-zero accumulator becomes the
    ## MFMA's inline-0 C operand, so no AGPRs are zeroed per tile. A loop-carried zero would
    ## be materialized.
    acc_tl, acc_bl, acc_tr, acc_br = init_acc_edge(
        bias_ptr, n0, cols_left, BLOCK_M, BLOCK_N, mfmaLayout, ADD_BIAS, EDGE_N, SHIFT
    )
    acc_tl, acc_bl, acc_tr, acc_br, a_top, b_left = k_step_pair(
        acc_tl, acc_bl, acc_tr, acc_br, a_top, b_left,
        smemA_top, smemA_bot, smemB_left, smemB_right, a_base, b_base,
        a0, b0, ah, bh, a1, b1,
        dotOpLayoutA, dotOpLayoutB, a2, b2, a3, b3, ad, bd,
    )
    a_base += BLOCK_K * stride_ak * 2
    b_base += BLOCK_K * stride_bk * 2

    ## Main loop — same as v8
    for k in range(2, iterMax - 2, 2):
        acc_tl, acc_bl, acc_tr, acc_br, a_top, b_left = k_step_pair(
            acc_tl, acc_bl, acc_tr, acc_br, a_top, b_left,
            smemA_top, smemA_bot, smemB_left, smemB_right, a_base, b_base,
            a0, b0, ah, bh, a1, b1,
            dotOpLayoutA, dotOpLayoutB, a2, b2, a3, b3, ad, bd,
        )
        a_base += BLOCK_K * stride_ak * 2
        b_base += BLOCK_K * stride_bk * 2

    # The next tile's load offsets, for its prefetch below and for its own trip.
    na_top, na_bot = a_cur_top, a_cur_bot
    if EDGE_M:
        na_top, na_bot = clamped_tile_offsets(M - 1 - next_m * BLOCK_M, stride_am, stride_ak, BLOCK_M // 2, BLOCK_K,
                                              gLoadLayoutA, True)
    nb_top, nb_bot = b_cur_top, b_cur_bot
    if EDGE_N:
        nb_top, nb_bot = clamped_tile_offsets(N - 1 - next_n * BLOCK_N, stride_bn, stride_bk, BLOCK_N // 2, BLOCK_K,
                                              gLoadLayoutB, False)
    na0, na1, na2, na3, nah, nad = operand_offsets(na_top, na_bot, a_offsets, a_offsets_next, a_half,
                                                   BLOCK_K * stride_ak, EDGE_M)
    nb0, nb1, nb2, nb3, nbh, nbd = operand_offsets(nb_top, nb_bot, b_offsets, b_offsets_next, b_half,
                                                   BLOCK_K * stride_bk, EDGE_N)

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
    c_tl_base = c_ptr + m0.to(gl.int64) * stride_cm
    col = n0 + offs_cn
    if EDGE_M:
        c_offsets = stride_cm * gl.minimum(offs_cm, rows_left - 1)[:, None] + stride_cn * col[None, :]
    else:
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
        smemB_left.index(0), b_next, nb0, mask=pb0
    )
    gl.amd.cdna4.async_copy.commit_group()

    acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    b_right = smemB_right.index(l_idx).load(dotOpLayoutB)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_top.index(0), a_next, na0, mask=pa0
    )
    gl.amd.cdna4.async_copy.commit_group()

    acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    g_idx = 1 - l_idx
    b_left = smemB_left.index(g_idx).load(dotOpLayoutB)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_bot.index(0), a_next + nah, na2, mask=pa0
    )
    gl.amd.cdna4.async_copy.commit_group()

    acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    a_top = smemA_top.index(g_idx).load(dotOpLayoutA)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_right.index(0), b_next + nbh, nb2, mask=pb0
    )
    gl.amd.cdna4.async_copy.commit_group()

    ## Iter iterMax - 1: prefetch the next tile's K-step 1.
    ## Natural-pipeline epilogue: each store follows its MFMA with one
    ## MFMA cycle of gap, yielding uniform MFMA-store interleaving.
    acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    a_bot = smemA_bot.index(g_idx).load(dotOpLayoutA)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_left.index(1), b_next + nbd, nb1, mask=pb1
    )
    gl.amd.cdna4.async_copy.commit_group()

    acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    b_right = smemB_right.index(g_idx).load(dotOpLayoutB)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_top.index(1), a_next + nad, na1, mask=pa1
    )
    gl.amd.cdna4.async_copy.commit_group()

    c_tl = acc_tl.to(c_ptr.dtype.element_ty)
    c_tl = gl.convert_layout(c_tl, layout=gStoreLayoutC)
    store_edge(c_tl, c_tl_base, c_offsets, gStoreLayoutC, rows_left, cols_left, EDGE_M, EDGE_N, skip_m, skip_n,
               SHIFT)

    acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")

    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_bot.index(1), a_next + nah + nad, na3, mask=pa1
    )
    gl.amd.cdna4.async_copy.commit_group()

    c_bl = acc_bl.to(c_ptr.dtype.element_ty)
    c_bl = gl.convert_layout(c_bl, layout=gStoreLayoutC)
    store_edge(c_bl, c_bl_base, c_offsets, gStoreLayoutC, rows_left - BLOCK_M // 2, cols_left, EDGE_M, EDGE_N,
               skip_m - BLOCK_M // 2, skip_n, SHIFT)

    acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")

    # Next tile's K-step 0 operands (prefetched in iter iterMax - 2), read while the last
    # MFMAs and stores drain; this tile's operand registers are all dead after acc_br.
    gl.amd.cdna4.async_copy.wait_group(5)
    b_left = smemB_left.index(0).load(dotOpLayoutB)
    a_top = smemA_top.index(0).load(dotOpLayoutA)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_right.index(1), b_next + nbh + nbd, nb3, mask=pb1
    )
    gl.amd.cdna4.async_copy.commit_group()

    c_tr = acc_tr.to(c_ptr.dtype.element_ty)
    c_tr = gl.convert_layout(c_tr, layout=gStoreLayoutC)
    store_edge(c_tr, c_tr_base, c_offsets, gStoreLayoutC, rows_left, cols_left - BLOCK_N // 2, EDGE_M, EDGE_N,
               skip_m, skip_n - BLOCK_N // 2, SHIFT)

    c_br = acc_br.to(c_ptr.dtype.element_ty)
    c_br = gl.convert_layout(c_br, layout=gStoreLayoutC)
    store_edge(c_br, c_br_base, c_offsets, gStoreLayoutC, rows_left - BLOCK_M // 2, cols_left - BLOCK_N // 2, EDGE_M, EDGE_N,
               skip_m - BLOCK_M // 2, skip_n - BLOCK_N // 2, SHIFT)

    return a_top, b_left, na_top, na_bot, nb_top, nb_bot


@gluon.jit(do_not_specialize=["epoch"])
def v26_streamk_ragged_k(
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
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,  #
    epoch,
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
    REDUCE_SCATTER: gl.constexpr,
    RS_NB: gl.constexpr,
    RS_U: gl.constexpr,
    RS_R_HI: gl.constexpr,
    RS_R_LO: gl.constexpr,
    RS_BAND: gl.constexpr,
    PARTIALS: gl.constexpr,
    END_TO_END: gl.constexpr,
    TILE_LIMIT: gl.constexpr = 0,
    ATOMIC_C: gl.constexpr = False,
    L2: gl.constexpr = False,
    XCD_LOCAL: gl.constexpr = False,
    XCD_SPLIT: gl.constexpr = 1,
    CHUNK_MAJOR: gl.constexpr = False,
    EDGE_M: gl.constexpr = False,
    EDGE_N: gl.constexpr = False,
    EDGE_K: gl.constexpr = False,
    SHIFT: gl.constexpr = False,
):
    """
    v14 with TensorAtlas's one-tile stream-K tail (DP + one-tile SK). The persistent loop
    covers total_tiles - STREAMK_TILES full tiles; the last STREAMK_TILES tiles' k-steps are
    laid end to end and split evenly over all NUM_PROGRAMS programs, in units of 2 k-steps so
    every segment fits the pair pipeline. The program that computes a tile's k-step 0 owns it:
    it adds its peers' partials (the next consecutive pids) and the bias, and stores C. Every
    other segment stores its fp32 partial to P[pid] and raises locks[pid] (relaxed).
    STREAMK_NUM_PROGRAMS programs take part in the stream-K phase and the rest idle
    (streamk_compact_pid; spread evenly over the XCDs if BALANCE_XCDS).

    v16: as v15, with the partials laid out lane-contiguous in P instead of row-major
    (lane_contiguous_offsets).

    v17: tile-aligned cuts. Instead of laying the stream-K tiles' pairs end to end, every
    program works inside one tile: tile t gets NUM_PROGRAMS // STREAMK_TILES consecutive
    programs (the first NUM_PROGRAMS % STREAMK_TILES tiles one more, at most one per pair),
    and splits its pairs evenly among them. Tiles with the same program count have the same
    chunk boundaries, so their chunk j covers the same k-steps at the same time. Each program
    runs one segment; chunk 0 owns the tile.

    v19: reversed two-tile. With at least one full wave, the host moves one more wave into
    stream-K (STREAMK_TILES = total % NUM_PROGRAMS + NUM_PROGRAMS): ranges are laid end to end as
    in v15, every tile has at most one peer, and each program processes its segments last to
    first, so all programs start at k-step 0 and stay within a fixed lag of each other. Without
    a full wave, v17's tile-aligned one-tile split runs unchanged.

    v20: reduce-scatter fixup for the tile-aligned one-tile split (REDUCE_SCATTER). Every
    program of a split tile publishes its partial, waits on the tile's self-resetting barrier
    (sync_ptr), then sums 1/n of the tile from all n partials and stores that slice of C, so
    the collection is spread over the tile's programs instead of serialised on chunk 0. The
    host picks one-tile or two-tile (STREAMK_POLICY); the two-tile path is v19's. With
    STREAMK_POLICY=capped the host also caps the one-tile split (STREAMK_NUM_PROGRAMS = S * n)
    and never picks two-tile.

    v23: the reduce-scatter ending with what experiments/rs_slice_layout found.
    - Readers: of a tile's n programs only the first RS_R_HI / RS_R_LO (tiles with N_HI / N_LO
      programs) reduce; the host passes the largest power of two <= n (STREAMK_RS_READERS=pow2,
      the default) or n (all, v20). The others publish, arrive on the barrier and finish.
    - Band C store (RS_BAND, whenever every reader count is a power of two): each group of
      RS_NB aligned blocks is reshaped to its 2-D band and converted to a row store layout
      before the C store, instead of v20's store straight from the slice layout.
    - PARTIALS (STREAMK_PARTIALS): 0 fp32 (v20), 1 bf16 in v20's map, 2 bf16 register pairs
      (16 bytes per lane), for both endings; sums stay fp32.

    END_TO_END selects v19's end-to-end reversed path for the stream-K tiles; without it they
    get the tile-aligned split. The host sets it for two-tile, and for STREAMK_POLICY=spread,
    which runs the same path on the one-tile tiles (an experiment for S > 128).

    TILE_LIMIT (an experiment, 0 = off): run as a GEMM of the grid's first TILE_LIMIT tile ids,
    with the tile positions still taken from the full grid; the other tiles of C are not written.

    v24: v23 with three earlier options, each for the tile-aligned one-tile split only, so that
    every ending, placement and partial type runs in one kernel. With all of them off the kernel
    is v23's.
    - ATOMIC_C (STREAMK_FIXUP=atomic_iz, v22): every program adds its partial into C with packed
      16-bit atomics, the tile zeroed inside this launch (streamk_segment).
    - L2 (STREAMK_EXCHANGE=l2, v21): reduce-scatter partials and barrier flags through the XCD's
      L2, epoch flags instead of atomics (tile_barrier); needs every program of a tile on one
      XCD, which XCD_LOCAL (v21's XCD-local split, XCD_SPLIT programs per tile) provides.
    - CHUNK_MAJOR (STREAMK_PLACEMENT=chunk, v18): program w runs chunk j = w // S of tile
      t = w % S, so an XCD holds the same chunk of many tiles; the tile's role 0 (owner, first
      slice, bias) rotates to chunk t % n so that roles are spread over the XCDs. The workspace
      stays tile-major: role i of tile t uses slot streamk_slot(t, i), so the endings read the
      same slots, flags and barriers as under tile-major placement.
    epoch (L2 only) goes up by one every launch.

    v25: M and N need not be multiples of the tile. EDGE_M: every tile's A rows past M are clamped
    to its last in-bounds one (clamped_tile_offsets; the tile position stays in the scalar base,
    the odd K-step becomes a scalar shift), and the C stores and the atomic ending's zeroing and
    adds are masked to M rows. EDGE_N: the same for B's columns and N, plus the bias loads. Partials
    stay whole tiles. With both off the kernel is v24's. The host sets both whenever either
    dimension is ragged: with both on every build fits without spills, with one on some spill.

    v26: K need not be a multiple of 2 BLOCK_K. EDGE_K (the host sets it when it is not): K is rounded
    up to K_PAD, a whole number of K-step pairs, and every tile's last, partial pair goes first, as
    its pair 0: loaded with the lanes past K masked to zero by the kernel prologue, the persistent
    epilogue's next-tile prefetch and the stream-K segment that starts at k0 = 0. Every other pair
    p is at columns 2 BLOCK_K (p - 1), so the main loops are unchanged and their reads stay aligned
    to the row's start (shifting the whole loop by K_PAD - K misaligns them with the 128-byte lines:
    55% slower in v9_ragged at K_PAD - K = 16). With EDGE_K off the kernel is v25's.
    """

    # Programs step through virtual pids start, start + NUM_PROGRAMS, ...; persistent_tile_id
    # turns each into a tile id in either the split or the v9 order.
    if TILE_ORDER_V9:
        start = gl.program_id(axis=0)
    else:
        start = get_logical_chiplet_mapped_pids(NUM_PROGRAMS, NUM_XCDS)
    num_tiles_m = gl.cdiv(M, BLOCK_M)
    num_tiles_n = gl.cdiv(N, BLOCK_N)
    if TILE_LIMIT > 0:
        total_tiles = num_tiles_m * 0 + TILE_LIMIT
    else:
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
    # the one to read next, which only holds for an even number of full K-steps; EDGE_K rounds K up.
    K_PAD: gl.constexpr = (K + 2 * BLOCK_K - 1) // (2 * BLOCK_K) * (2 * BLOCK_K) if EDGE_K else K
    K_RES: gl.constexpr = K_PAD - K
    gl.static_assert(K_PAD % (2 * BLOCK_K) == 0, "K must be a multiple of 2 * BLOCK_K (or EDGE_K)")
    iterMax = gl.cdiv(K_PAD, BLOCK_K)
    gl.assume(iterMax > 3)

    ## Prologue, first tile only: every later tile's first two K-steps are prefetched by the
    ## previous tile's epilogue. Programs with no tile still load a valid one, then skip the loop.
    first_id = persistent_tile_id(min(start, total_tiles - 1), total_tiles, NUM_XCDS, TILE_ORDER_V9)
    tile_m, tile_n = get_group_m_tile_ids(first_id, num_tiles_m, num_tiles_n, GROUP_SIZE_M)
    a_first = a_ptr + tile_origin(tile_m, M, BLOCK_M, SHIFT).to(gl.int64) * stride_am
    b_first = b_ptr + tile_origin(tile_n, N, BLOCK_N, SHIFT).to(gl.int64) * stride_bn

    # The persistent tile's clamped load offsets (EDGE_M / EDGE_N), carried from trip to trip.
    a_cur_top, a_cur_bot = a_offsets, a_offsets
    if EDGE_M:
        a_cur_top, a_cur_bot = clamped_tile_offsets(M - 1 - tile_m * BLOCK_M, stride_am, stride_ak, BLOCK_M // 2,
                                                    BLOCK_K, gLoadLayoutA, True)
    b_cur_top, b_cur_bot = b_offsets, b_offsets
    if EDGE_N:
        b_cur_top, b_cur_bot = clamped_tile_offsets(N - 1 - tile_n * BLOCK_N, stride_bn, stride_bk, BLOCK_N // 2,
                                                    BLOCK_K, gLoadLayoutB, False)
    a0, a1, a2, a3, ah, ad = operand_offsets(a_cur_top, a_cur_bot, a_offsets, a_offsets_next, a_half,
                                             BLOCK_K * stride_ak, EDGE_M)
    b0, b1, b2, b3, bh, bd = operand_offsets(b_cur_top, b_cur_bot, b_offsets, b_offsets_next, b_half,
                                             BLOCK_K * stride_bk, EDGE_N)

    # First tile's K-steps 0-1; later tiles get them from the previous tile's epilogue. With EDGE_K
    # these are its last, partial pair.
    if EDGE_K:
        ma0, mb0, ma1, mb1 = tail_pair_masks(2 * BLOCK_K - K_RES, K_RES, BLOCK_M, BLOCK_N, BLOCK_K,
                                             gLoadLayoutA, gLoadLayoutB)
        a_top, b_left = load_first_pair(
            a_first + (K_PAD - 2 * BLOCK_K) * stride_ak, b_first + (K_PAD - 2 * BLOCK_K) * stride_bk,
            smemA_top, smemA_bot, smemB_left, smemB_right,
            a0, b0, ah, bh, a1, b1,
            dotOpLayoutA, dotOpLayoutB, a2, b2, a3, b3, ad, bd, ma0, mb0, ma1, mb1,
        )
    else:
        a_top, b_left = load_first_pair(
            a_first, b_first, smemA_top, smemA_bot, smemB_left, smemB_right,
            a0, b0, ah, bh, a1, b1,
            dotOpLayoutA, dotOpLayoutB, a2, b2, a3, b3, ad, bd,
        )

    # Full tiles. The last trip's prefetch (clamped to total_tiles - 1) is unused; the
    # stream-K phase drains it and starts every segment with its own prologue.
    total_full_tiles = total_tiles - STREAMK_TILES
    for vpid in range(start, total_full_tiles, NUM_PROGRAMS):
        a_top, b_left, a_cur_top, a_cur_bot, b_cur_top, b_cur_bot = persistent_tile(
            vpid, a_top, b_left, a_ptr, b_ptr, c_ptr, bias_ptr,
            smemA_top, smemA_bot, smemB_left, smemB_right,
            a_offsets, b_offsets, a_half, b_half, a_offsets_next, b_offsets_next,
            num_tiles_m, num_tiles_n, total_tiles,
            stride_ak, stride_bk, stride_am, stride_bn, stride_cm, stride_cn,
            K_PAD, BLOCK_M, BLOCK_N, BLOCK_K, NUM_PROGRAMS, NUM_XCDS, GROUP_SIZE_M,
            TILE_ORDER_V9, MASK_TAIL_PREFETCH, ADD_BIAS,
            mfmaLayout, dotOpLayoutA, dotOpLayoutB,
            a_cur_top, a_cur_bot, b_cur_top, b_cur_bot, M, N, EDGE_M, EDGE_N, gLoadLayoutA, gLoadLayoutB,
            EDGE_K, K_RES, SHIFT,
        )

    # Drain the last trip's redundant prefetch.
    gl.amd.cdna4.async_copy.wait_group(0)

    if END_TO_END and STREAMK_TILES > 0:
        # Two-tile: the host moved one full wave into stream-K, so every program's range is about
        # one tile long plus d = PAIRS_PER_TILE * (STREAMK_TILES - NUM_PROGRAMS) / NUM_PROGRAMS pairs, and
        # every tile has at most one peer. Spread (STREAMK_TILES < NUM_PROGRAMS): ranges are
        # shorter than a tile, so a tile can have up to ceil(PAIRS_PER_TILE / L) programs (at most
        # 3 for STREAMK_TILES > NUM_PROGRAMS / 2), all below its owner. The ranges are laid end to end in pairs of k-steps as in
        # v15, but each program processes its segments last to first (reversed order): the head
        # of its last tile (k-step 0 upward, its only partial) first, then any whole tile, then
        # the tail of its first tile, which it owns. So every program starts at k-step 0, the
        # programs of an XCD read k-slices at most d apart, and an owner's peer is the preceding
        # program, which processed that tile's head first.
        PAIRS_PER_TILE: gl.constexpr = K_PAD // (2 * BLOCK_K)
        SK_PAIRS: gl.constexpr = STREAMK_TILES * PAIRS_PER_TILE
        L: gl.constexpr = SK_PAIRS // STREAMK_NUM_PROGRAMS
        R: gl.constexpr = SK_PAIRS % STREAMK_NUM_PROGRAMS
        spid = streamk_compact_pid(
            get_logical_chiplet_mapped_pids(NUM_PROGRAMS, NUM_XCDS), STREAMK_NUM_PROGRAMS, NUM_PROGRAMS,
            NUM_XCDS, BALANCE_XCDS,
        )
        sk_start = gl.minimum(spid * L + gl.minimum(spid, R), SK_PAIRS)
        cur = gl.minimum((spid + 1) * L + gl.minimum(spid + 1, R), SK_PAIRS)
        # Waits point only to lower spids, and a program publishes its partial before it waits.
        while cur > sk_start:
            t = (cur - 1) // PAIRS_PER_TILE
            tile_start = t * PAIRS_PER_TILE
            seg_start = gl.maximum(tile_start, sk_start)
            # Holding the tile's last pair makes this program the owner; otherwise this is the
            # head piece, the program's partial.
            is_owner = (cur == tile_start + PAIRS_PER_TILE).to(gl.int32)
            n_peers = (spid - streamk_owner_of(tile_start, L, R)) * is_owner
            streamk_segment(
                streamk_tile_id(t, total_tiles, total_full_tiles, NUM_XCDS, TILE_ORDER_V9),
                (seg_start - tile_start) * 2, cur - seg_start,
                1 - is_owner, n_peers, spid, 0, 0, 0,
                a_ptr, b_ptr, c_ptr, bias_ptr, p_ptr, locks_ptr, sync_ptr,
                smemA_top, smemA_bot, smemB_left, smemB_right,
                a_offsets, b_offsets, a_half, b_half, a_offsets_next, b_offsets_next,
                num_tiles_m, num_tiles_n, total_tiles,
                stride_ak, stride_bk, stride_am, stride_bn, stride_cm, stride_cn,
                BLOCK_M, BLOCK_N, BLOCK_K, NUM_XCDS, GROUP_SIZE_M, TILE_ORDER_V9, ADD_BIAS,
                -1, False, 1, 1, False, PARTIALS, mfmaLayout, dotOpLayoutA, dotOpLayoutB,
                False, False, NUM_PROGRAMS, epoch, M, N, EDGE_M, EDGE_N, gLoadLayoutA, gLoadLayoutB, EDGE_K, K_PAD, K_RES, SHIFT,
            )
            cur = seg_start
    elif STREAMK_TILES > 0:
        # Tile-aligned split: tile t gets N_HI (first EXTRA tiles) or N_LO consecutive programs,
        # at most one per pair of k-steps; programs beyond SK_PROGRAMS have no work.
        PAIRS_PER_TILE: gl.constexpr = K_PAD // (2 * BLOCK_K)
        N_LO: gl.constexpr = min(STREAMK_NUM_PROGRAMS // STREAMK_TILES, PAIRS_PER_TILE)
        N_HI: gl.constexpr = min(STREAMK_NUM_PROGRAMS // STREAMK_TILES + 1, PAIRS_PER_TILE)
        EXTRA: gl.constexpr = STREAMK_NUM_PROGRAMS % STREAMK_TILES if N_HI > N_LO else 0
        SK_PROGRAMS: gl.constexpr = EXTRA * N_HI + (STREAMK_TILES - EXTRA) * N_LO
        # The host's reader counts (rs_split) must match these program counts, and RS_NB must
        # cut every reader's slice into whole groups for the band store.
        gl.static_assert(0 < RS_R_LO and RS_R_LO <= N_LO and 2 * RS_R_LO > N_LO, "RS_R_LO does not match N_LO")
        gl.static_assert(0 < RS_R_HI and RS_R_HI <= N_HI and 2 * RS_R_HI > N_HI, "RS_R_HI does not match N_HI")
        R_MAX: gl.constexpr = RS_R_HI if EXTRA > 0 else RS_R_LO
        gl.static_assert(
            not RS_BAND or ((RS_R_LO & (RS_R_LO - 1)) == 0 and (R_MAX & (R_MAX - 1)) == 0
                            and (RS_NB == 1 or RS_NB * R_MAX <= 64)),
            "the band store needs power-of-two readers and whole groups",
        )
        if XCD_LOCAL:
            # Every tile's XCD_SPLIT programs are consecutive spids of one XCD (the host sets
            # STREAMK_NUM_PROGRAMS = S * XCD_SPLIT, so N_LO = XCD_SPLIT and EXTRA = 0).
            spid = get_logical_chiplet_mapped_pids(NUM_PROGRAMS, NUM_XCDS)
            t, j, n = xcd_local_chunk_of(spid, STREAMK_TILES, NUM_PROGRAMS // NUM_XCDS, NUM_XCDS, XCD_SPLIT)
            active = n > 0
        else:
            # Stream-K's pid is always the XCD-mapped one, so a tile's peers share its XCD.
            spid = streamk_compact_pid(
                get_logical_chiplet_mapped_pids(NUM_PROGRAMS, NUM_XCDS), STREAMK_NUM_PROGRAMS, NUM_PROGRAMS,
                NUM_XCDS, BALANCE_XCDS,
            )
            active = spid < SK_PROGRAMS
        if active:
            if XCD_LOCAL:
                readers = n * 0 + RS_R_LO
                role = j
                slot = spid
            elif CHUNK_MAJOR:
                t, j, n, readers = streamk_chunk_major_of(spid, STREAMK_TILES, EXTRA, N_HI, N_LO, RS_R_HI, RS_R_LO)
                # Role 0 (owner, first slice, bias) rotates over the chunks, so that it is not
                # every tile's chunk 0, which chunk-major puts on the same XCDs.
                role = (j + n - t % n) % n
                slot = streamk_slot(t, role, EXTRA, N_HI, N_LO)
            else:
                t, j, n, readers = streamk_chunk_of(spid, EXTRA, N_HI, N_LO, RS_R_HI, RS_R_LO)
                role = j
                slot = spid
            # Chunk j of n: the first PAIRS_PER_TILE % n chunks are one pair longer.
            q = PAIRS_PER_TILE // n
            r = PAIRS_PER_TILE % n
            is_partial = (role != 0).to(gl.int32)
            # Role 0 owns the tile; its peers are the next n - 1 slots.
            n_peers = (n - 1) * (1 - is_partial)
            streamk_segment(
                streamk_tile_id(t, total_tiles, total_full_tiles, NUM_XCDS, TILE_ORDER_V9),
                (j * q + gl.minimum(j, r)) * 2, q + (j < r).to(gl.int32),
                is_partial, n_peers, slot, role, n, readers,
                a_ptr, b_ptr, c_ptr, bias_ptr, p_ptr, locks_ptr, sync_ptr,
                smemA_top, smemA_bot, smemB_left, smemB_right,
                a_offsets, b_offsets, a_half, b_half, a_offsets_next, b_offsets_next,
                num_tiles_m, num_tiles_n, total_tiles,
                stride_ak, stride_bk, stride_am, stride_bn, stride_cm, stride_cn,
                BLOCK_M, BLOCK_N, BLOCK_K, NUM_XCDS, GROUP_SIZE_M, TILE_ORDER_V9, ADD_BIAS,
                1, REDUCE_SCATTER, RS_NB, RS_U, RS_BAND, PARTIALS, mfmaLayout, dotOpLayoutA, dotOpLayoutB,
                ATOMIC_C, L2, NUM_PROGRAMS, epoch, M, N, EDGE_M, EDGE_N, gLoadLayoutA, gLoadLayoutB, EDGE_K, K_PAD, K_RES, SHIFT,
            )


# Per device: the partial slots (one BLOCK_M x BLOCK_N fp32 tile per program; bf16 partials use
# the first half of each slot's bytes, through a bf16 view of P), one flag per
# program, and tile_sync: a (count, generation) pair per program for the reduce-scatter tile
# barriers, indexed by a tile's first slot, then one L2 flag per program (STREAMK_EXCHANGE=l2), then
# a (zeroed, done) pair of counts per program for the atomic ending, indexed like the barriers.
# Owners reset the flags they consume and the last arriver resets a barrier's count (the last
# adder both zero counts), so flags and counts are zero between launches; generations only ever
# go up. The L2 flags hold the epoch of the last launch that wrote them; the epoch goes up by one
# every launch.
_WORKSPACE = {}


def streamk_workspace(device, num_programs, block_m, block_n):
    key = (str(device), num_programs, block_m, block_n)
    if key not in _WORKSPACE:
        # Every partial fully overwrites its slot before its flag goes up, so P needs no init.
        P = torch.empty(num_programs * (block_m * block_n + P_SLOT_PAD.value), device=device,
                        dtype=torch.float32)
        locks = torch.zeros(num_programs, device=device, dtype=torch.int32)
        tile_sync = torch.zeros(5 * num_programs, device=device, dtype=torch.int32)
        _WORKSPACE[key] = (P, locks, tile_sync, [0])
    return _WORKSPACE[key]


def next_epoch(counter):
    """The next launch's epoch: 1, 2, ..., never 0 (the L2 flags' initial value), within int32."""
    counter[0] = counter[0] % (2**31 - 2) + 1
    return counter[0]


def xcd_local_split(streamk_tiles, n_cap, pairs_per_tile, num_programs, num_xcds):
    """Programs per tile in the XCD-local split (v21): n_cap, at most one per pair, and few enough
    that ceil(S / num_xcds) tiles fit in one XCD's num_programs // num_xcds programs."""
    per_xcd_tiles = -(-streamk_tiles // num_xcds)
    return max(1, min(n_cap, pairs_per_tile, (num_programs // num_xcds) // per_xcd_tiles))


def tile_groups(streamk_tiles, streamk_programs, pairs_per_tile, num_programs, num_xcds, balance, xcd_local,
                xcd_split):
    """The XCD-mapped spids of every stream-K tile's programs in the tile-major tile-aligned split
    (the l2 exchange does not take chunk-major placement), as the kernel assigns them
    (streamk_compact_pid + streamk_chunk_of, or xcd_local_chunk_of), in slot order."""
    per = num_programs // num_xcds
    S = streamk_tiles
    groups = [[] for _ in range(S)]
    if xcd_local:
        lo, tall = S // num_xcds, S % num_xcds
        for spid in range(num_programs):
            x, local = spid // per, spid % per
            tl = local // xcd_split
            if tl < lo + (x < tall):
                groups[x * lo + min(x, tall) + tl].append(spid)
        return groups
    n_lo = min(streamk_programs // S, pairs_per_tile)
    n_hi = min(streamk_programs // S + 1, pairs_per_tile)
    extra = streamk_programs % S if n_hi > n_lo else 0
    sk_programs = extra * n_hi + (S - extra) * n_lo
    base, tall = streamk_programs // num_xcds, streamk_programs % num_xcds
    for spid in range(num_programs):
        if balance:
            x, local = spid // per, spid % per
            cid = x * base + min(x, tall) + local if local < base + (x < tall) else num_programs
        else:
            cid = spid if spid < streamk_programs else num_programs
        if cid >= sk_programs:
            continue
        head = extra * n_hi
        t = cid // n_hi if cid < head else extra + (cid - head) // n_lo
        # The kernel's slots and barrier are indexed by cid, consecutive within a tile.
        groups[t].append((cid, spid))
    return [[s for _, s in sorted(g)] for g in groups]


@functools.lru_cache(maxsize=None)
def first_straddling_tile(*args):
    """The spids of the first stream-K tile whose programs span XCDs (tile_groups' arguments),
    or None. Cached: the replay is a Python loop over every program, which costs more than a
    short kernel when it runs on every launch."""
    num_programs, num_xcds = args[3], args[4]
    per = num_programs // num_xcds
    for g in tile_groups(*args):
        if len({s // per for s in g}) > 1:
            return g
    return None


def pow2_floor(x):
    return 1 << (x.bit_length() - 1)


def rs_split(streamk_tiles, num_programs, pairs_per_tile, pow2_readers, loads_in_flight=16):
    """The reduce-scatter shape of the tile-aligned one-tile split, as a dict:
    - n_hi, n_lo, extra: programs per tile, as the kernel's N_HI, N_LO and EXTRA;
    - r_hi, r_lo: the readers of those tiles: the largest power of two <= n with pow2_readers,
      else n (v20);
    - band: every reader count in use is a power of two, so slices are whole aligned groups;
    - nb, u: blocks per group and partials per trip of reduce_scatter_slice, nb * u loads in
      flight. With band, nb = 64 / (largest reader count), at most loads_in_flight, so every slice
      is whole groups; without, v20's: enough for the largest slice (the fewest readers).
    Whole tiles (n = 1) do not reduce."""
    s = max(streamk_tiles, 1)
    n_lo = min(num_programs // s, pairs_per_tile)
    n_hi = min(num_programs // s + 1, pairs_per_tile)
    extra = num_programs % s if n_hi > n_lo else 0
    readers = pow2_floor if pow2_readers else (lambda n: n)
    r_hi, r_lo = readers(max(n_hi, 1)), readers(max(n_lo, 1))
    r_max = r_hi if extra > 0 else r_lo
    band = all(r & (r - 1) == 0 for r in ((r_hi, r_lo) if extra > 0 else (r_lo,)))
    if band:
        nb = max(1, min(64 // r_max, loads_in_flight))
    else:
        nb = min(triton.next_power_of_2(triton.cdiv(64, max(r_lo, 2))), loads_in_flight)
    return dict(n_hi=n_hi, n_lo=n_lo, extra=extra, r_hi=r_hi, r_lo=r_lo, band=band, nb=nb,
                u=max(loads_in_flight // nb, 1))


def streamk_policy(total_tiles, iters_per_tile, num_programs, streamk_programs=None):
    """(policy, fixup) for STREAMK_POLICY=auto (v20 WORKLOG entry 2; the default until entry 7)
    and its STREAMK_FIXUP=auto.

    With S = total_tiles % num_programs leftover tiles, n = num_programs // S programs per
    tile in the one-tile split, and d = iters_per_tile * S / num_programs k-steps of two-tile
    lag (also one-tile's chunk length):
    - S > num_programs / 2: data-parallel. Most leftover tiles would run whole anyway, so they
      set the tail and the split ones only add traffic (v17 entry 3).
    - A full wave to borrow, n >= 7 and d <= 18: two-tile (v19). Every tile has at most one
      peer and the programs of an XCD stay within d k-steps of each other.
    - Otherwise one-tile, with the reduce-scatter fixup if every tile is split at least 8 ways
      (its cost is about flat, the owner's grows with n) and the owner's below that.
    """
    s = total_tiles % num_programs
    if s == 0:
        return "dp", "rs"
    if 2 * s > num_programs:
        return "dp", "rs"
    n_lo = min((streamk_programs or num_programs) // s, iters_per_tile // 2)
    fixup = "rs" if n_lo >= 8 else "owner"
    if total_tiles >= num_programs and num_programs // s >= 7 and iters_per_tile * s <= 18 * num_programs:
        return "two_tile", fixup
    return "one_tile", fixup


def capped_policy(total_tiles, iters_per_tile, num_programs):
    """(policy, fixup, streamk_programs) for STREAMK_POLICY=capped, the default (v20 WORKLOG
    entries 5 and 7): no two-tile, and the one-tile split capped to n programs per leftover
    tile. streamk_programs is S * n, or None for all programs.

    With S = total_tiles % num_programs and k = iters_per_tile:
    - S = 0: data-parallel.
    - S > num_programs / 2: spread if k >= 512 (entry 3), otherwise data-parallel.
    - (num_programs - S) * k < 20 * num_programs: data-parallel. The idle share of the
      data-parallel last wave is under about 20 k-steps, less than any split's fixed cost.
    - num_programs // S >= 8: one-tile reduce-scatter with n = min(128 // S, max(8, k // 8)).
    - Otherwise one-tile owner with n = round(sqrt(k / c)), c = max(2.2, S / 5) k-steps per
      serial partial read, at most num_programs // S.
    - n = 1 runs data-parallel: one program per leftover tile is the data-parallel last wave's
      work plus stream-K's overhead, and it lost on every measured shape (entry 7).
    The coefficients are fits to the split-count sweeps (experiments/streamk_split_count and
    experiments/streamk_version_tables).
    """
    s = total_tiles % num_programs
    if s == 0:
        return "dp", "rs", None
    if 2 * s > num_programs:
        return ("spread" if iters_per_tile >= 512 else "dp"), "rs", None
    if (num_programs - s) * iters_per_tile < 20 * num_programs:
        return "dp", "rs", None
    if num_programs // s >= 8:
        fixup, n = "rs", min(num_programs // 2 // s, max(8, iters_per_tile // 8))
    else:
        fixup = "owner"
        n = max(1, min(num_programs // s, round(math.sqrt(iters_per_tile / max(2.2, s / 5)))))
    if n == 1:
        return "dp", "rs", None
    return "one_tile", fixup, (None if n >= num_programs // s else s * n)


# STREAMK_POLICY=split_rule (v20 WORKLOG entry 9): the fastest configuration of each cell of the
# experiments/streamk_split_rule grid (cold rocprof, 256 programs), by full-wave class (0, 1-3,
# 4 or more), leftover tiles S (SPLIT_RULE_S) and k-tiles (SPLIT_RULE_K, one column each).
# dp: data-parallel; 2T: two-tile; rsN / owN: one-tile on N programs per leftover tile with the
# reduce-scatter / owner ending. Not the default: it beat capped on off-grid shapes timed cold but
# lost on the real-model shapes timed back to back.
SPLIT_RULE_S = (1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 48, 64, 96, 128)
SPLIT_RULE_K = (16, 20, 24, 28, 32, 40, 48, 64, 96, 128, 192, 256)
SPLIT_RULE_TABLE = {
    0: {
        1: " rs8  rs8 rs12  rs8 rs16 rs16 rs24 rs16 rs32 rs32 rs32 rs32",
        2: " rs8  rs8 rs12  rs8 rs16 rs16 rs24 rs16 rs32 rs32 rs32 rs32",
        4: " rs8  rs8 rs12  rs8 rs16 rs12 rs24 rs16 rs16 rs16 rs32 rs32",
        6: " rs8  rs5 rs12  rs8 rs16 rs12  rs8 rs16 rs16 rs16 rs16 rs16",
        8: " rs8  rs6 rs12  rs8  rs8  rs8  rs8 rs16 rs16 rs16 rs16 rs16",
        12: " rs8  rs5  rs8  rs8  rs8  rs8  rs8  rs8 rs12  rs8 rs12 rs16",
        16: "  dp  rs5  rs6  rs8  rs8  rs5  rs8  rs8  rs8  rs8 rs12 rs12",
        20: "  dp  rs5  rs6  rs5  rs5  rs5  rs8  rs8  rs8  rs8  rs8  rs8",
        24: "  dp  rs5  rs4  rs5  rs4  rs5  rs6  rs8  rs8  rs8  rs8  rs8",
        28: "  dp  ow3  rs4  rs5  rs4  rs5  rs4  rs5  rs8  rs8  rs8  rs8",
        32: "  dp  ow3  rs4  rs4  rs4  rs5  rs4  rs5  rs6  rs6  rs8  rs8",
        40: "  dp  ow3  ow3  ow2  ow3  rs4  rs4  ow4  rs6  rs5  rs6  rs6",
        48: "  dp  ow3  ow3  ow2  ow3  ow4  ow4  ow4  ow4  ow4  ow5  ow5",
        64: "  dp  ow3  ow2  ow2  ow3  ow2  ow3  ow3  ow3  ow4  ow4  ow4",
        96: "  dp   dp  ow2  ow2  ow2  ow2  ow2  ow2  ow2  ow2  ow2  ow2",
        128: "  dp   dp   dp   dp   dp  ow2  ow2  ow2  ow2  ow2  ow2  ow2",
    },
    1: {
        1: "  2T  rs8 rs12   2T rs16 rs16 rs24 rs32 rs12 rs32 rs32 rs32",
        2: "  2T  rs8 rs12 rs12 rs16 rs16 rs16 rs32 rs32 rs32 rs32 rs32",
        4: "  dp  rs8 rs12  rs8 rs16 rs16 rs16 rs16 rs16 rs12 rs32 rs32",
        6: " rs8  rs8 rs12  rs8 rs16  rs8  rs8 rs16  rs6 rs16 rs16 rs12",
        8: "  dp  rs8  rs8  rs8  rs8  rs8  rs8  rs8  rs8  rs8  rs8 rs16",
        12: "  dp   dp  rs8  rs8  rs8  rs6  rs8  rs5  rs8  rs8 rs12 rs12",
        16: "  dp   dp  rs4  rs5  rs4  rs5  rs6  rs6  rs6  rs8  rs6  rs8",
        20: "  dp   dp  rs4  ow3  rs4  rs4  rs4   2T  rs8  rs6  rs6  rs8",
        24: "  dp   dp  ow3  rs4  rs4  rs4  rs4   2T  rs4  rs5  rs6  rs8",
        28: "  dp   dp   dp  ow2   2T  rs4  rs4  rs4  rs5  ow4  rs6  rs8",
        32: "  dp   dp  ow3  rs4  ow3  ow3  rs4   2T  rs4  rs5  ow5  rs8",
        40: "  dp   dp   dp  ow2  ow3  ow2  ow3  ow3  rs4  ow3  rs6  rs6",
        48: "  dp   dp  ow2  ow2  ow2  ow3  ow3  ow3  ow4  ow4  ow4  ow5",
        64: "  dp   dp  ow2  ow2  ow3  ow2  ow3  ow3  ow3  ow3  ow4  ow4",
        96: "  dp   dp   dp   dp   dp  ow2  ow2  ow2  ow2  ow2  ow2  ow2",
        128: "  dp   dp   dp   dp   dp   dp   dp  ow2  ow2  ow2  ow2  ow2",
    },
    4: {
        1: "  2T   2T   2T   2T   2T   2T  rs8 rs32 rs32 rs32 rs32 rs32",
        2: "  2T  rs8 rs12   2T   2T rs16 rs24 rs32 rs32 rs32 rs32 rs32",
        4: " rs8  rs8 rs12 rs12  rs8  rs8  rs5 rs16 rs16 rs16 rs32 rs32",
        6: "  2T  rs6 rs12  rs8  rs5  rs5 rs16 rs16 rs16 rs16 rs16 rs24",
        8: " rs8  rs8  rs8  rs8  rs8  rs8 rs16 rs16 rs16 rs16 rs16 rs16",
        12: " rs8  rs8  rs8  rs8  rs6  rs8  rs8  rs4  rs8  rs8 rs12  rs8",
        16: " rs4  rs8  rs4  rs5  rs4  rs4  rs6  rs8  rs8  rs8  rs8  rs8",
        20: "  dp  rs4  rs4  rs4  rs4  rs4  rs4  ow4  rs8   2T  rs8   2T",
        24: "  dp   dp   dp  rs5  rs5  rs5  rs4  rs8  rs8  rs8   2T   2T",
        28: "  dp   dp   dp  rs4  ow3  rs5  ow6  rs5  rs6  rs6  rs8  rs6",
        32: "  dp  ow3  rs4  rs4  ow3  rs4  rs4  ow4  ow3  rs6  rs6  rs6",
        40: "  dp   dp  ow3  ow2  ow3  ow3  ow3  rs4  ow4  rs5  rs5  rs5",
        48: "  dp   dp   dp  ow3  ow3  ow3  ow4  rs4  ow4  rs4  rs4  rs5",
        64: "  dp   dp   dp   dp  ow2  ow2  ow2  ow2  ow2  ow3  ow4  rs4",
        96: "  dp   dp   dp   dp   dp  ow2  ow2  ow2  ow2  ow2  ow2  ow2",
        128: "  dp   dp   dp   dp   dp  ow2   dp  ow2  ow2  ow2  ow2  ow2",
    },
}


def split_rule_policy(total_tiles, iters_per_tile, num_programs):
    """(policy, fixup, streamk_programs) for STREAMK_POLICY=split_rule: the SPLIT_RULE_TABLE cell
    at or below S and k-tiles (the smallest if there is none below). S = 0 is data-parallel and
    S > num_programs / 2 follows capped_policy."""
    s = total_tiles % num_programs
    waves = total_tiles // num_programs
    if s == 0:
        return "dp", "rs", None
    if 2 * s > num_programs:
        return capped_policy(total_tiles, iters_per_tile, num_programs)
    wclass = 0 if waves == 0 else (1 if waves < 4 else 4)
    s_cell = max((x for x in SPLIT_RULE_S if x <= s), default=SPLIT_RULE_S[0])
    k_col = max((i for i, x in enumerate(SPLIT_RULE_K) if x <= iters_per_tile), default=0)
    cell = SPLIT_RULE_TABLE[wclass][s_cell].split()[k_col]
    if cell == "dp" or (cell == "2T" and waves == 0):
        return "dp", "rs", None
    if cell == "2T":
        return "two_tile", "rs", None
    fixup = "rs" if cell.startswith("rs") else "owner"
    n = min(int(cell[2:]), num_programs // s, iters_per_tile // 2)
    if n < 2:
        return "dp", "rs", None
    return "one_tile", fixup, (None if n >= num_programs // s else s * n)


def matmul(a, b, c=None, bias=None):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.stride(1) == 1, "Matrix A must be row-major (its row stride may be padded)"
    M, K = a.shape
    K, N = b.shape
    assert bias is None or (bias.shape == (N,) and bias.is_contiguous()), "bias must be contiguous (N,)"
    BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 64
    # K runs in pairs of K-steps, at least two; other K are rounded up (EDGE_K, v26). K must keep
    # 16-byte rows: Triton only knows a stride is aligned when it is a multiple of 16.
    assert K % 16 == 0, f"K={K} must be a multiple of 16 (pad A and B)"
    K_PAD = triton.cdiv(K, 2 * BLOCK_K) * 2 * BLOCK_K
    assert K_PAD >= 4 * BLOCK_K, f"K={K} needs at least {4 * BLOCK_K} after rounding up to {2 * BLOCK_K}"
    EDGE_K = K != K_PAD
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
    env_programs = os.environ.get("STREAMK_NUM_PROGRAMS")
    STREAMK_NUM_PROGRAMS = int(env_programs or NUM_PROGRAMS)
    BALANCE_XCDS = os.environ.get("STREAMK_BALANCE_XCDS", "1") == "1"
    # STREAMK_POLICY: dp is data-parallel only (the persistent loop also runs the partial last
    # wave); one_tile is DP + one-tile SK with v17's tile-aligned split; two_tile moves one full
    # wave into stream-K as well, when there is one to borrow; spread (an experiment) runs
    # two-tile's end-to-end reversed split on the one-tile tiles; capped (default) is
    # capped_policy's choice, which also sets the program count unless STREAMK_NUM_PROGRAMS is
    # given; split_rule is split_rule_policy's table, likewise; auto is streamk_policy's choice.
    # When one of these rules picks dp and the tiles fit in one wave, the call goes to v9_ragged
    # (v9 with the same edge handling, v9's code on aligned shapes), which gains nothing from being
    # persistent there; an explicit dp stays on the persistent loop.
    # STREAMK_FIXUP picks the one-tile split's ending: rs reduce-scatters the partials, owner is
    # v17's serial collection on role 0, atomic_iz adds every partial into C with packed 16-bit
    # atomics, the tile zeroed inside the GEMM (v22), auto (default) is the policy rule's choice.
    # STREAMK_RS_READERS: pow2 (default) lets the largest power of two of a tile's programs
    # reduce, all lets every program reduce (v20). STREAMK_PARTIALS: fp32 (default), bf16 or
    # bf16_pairs, for the rs and owner endings.
    # STREAMK_EXCHANGE: wt (default) writes the partials through and syncs with relaxed atomics;
    # l2 (rs only, v21) keeps every tile's programs on one XCD and exchanges the partials and the
    # barrier's flags through its L2. STREAMK_XCD_LOCAL (default: on for l2) selects v21's
    # XCD-local split, with n = STREAMK_NUM_PROGRAMS // S programs per tile, capped so that every
    # XCD's tiles fit in its programs.
    # STREAMK_PLACEMENT: tile (default) gives a tile consecutive stream-K ids; chunk (v18) gives
    # consecutive ids the same chunk of neighbouring tiles, with role 0 rotating.
    pow2_readers = os.environ.get("STREAMK_RS_READERS", "pow2")
    partials = os.environ.get("STREAMK_PARTIALS", "fp32")
    exchange = os.environ.get("STREAMK_EXCHANGE", "wt")
    placement = os.environ.get("STREAMK_PLACEMENT", "tile")
    assert pow2_readers in ("pow2", "all"), pow2_readers
    assert partials in PARTIALS_CODES, partials
    assert exchange in ("wt", "l2"), exchange
    assert placement in ("tile", "chunk"), placement
    total_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    # GEMM_TILE_LIMIT (an experiment): compute only the first T tile ids, as a T-tile GEMM.
    TILE_LIMIT = int(os.environ.get("GEMM_TILE_LIMIT", "0"))
    assert 0 <= TILE_LIMIT <= total_tiles
    total_tiles = TILE_LIMIT or total_tiles
    requested = os.environ.get("STREAMK_POLICY", "capped")
    policy = requested
    fixup = os.environ.get("STREAMK_FIXUP", "auto")
    assert policy in ("dp", "one_tile", "two_tile", "spread", "auto", "capped", "split_rule"), policy
    assert fixup in ("rs", "owner", "atomic_iz", "auto"), fixup
    if policy in ("capped", "split_rule"):
        rule = capped_policy if policy == "capped" else split_rule_policy
        policy, rule_fixup, capped_programs = rule(total_tiles, K_PAD // BLOCK_K, NUM_PROGRAMS)
        if env_programs is None and capped_programs is not None:
            STREAMK_NUM_PROGRAMS = capped_programs
    else:
        auto_policy, rule_fixup = streamk_policy(total_tiles, K_PAD // BLOCK_K, NUM_PROGRAMS, STREAMK_NUM_PROGRAMS)
        policy = auto_policy if policy == "auto" else policy
    fixup = rule_fixup if fixup == "auto" else fixup
    # Ragged M / N: clamped loads and masked stores (v25), on both dimensions if either is ragged:
    # with only one on, the register allocation spills on some shapes, with both it does not.
    # STREAMK_FORCE_EDGE=1 turns them on for any shape, to measure their cost where they change
    # nothing.
    EDGE_M = EDGE_N = os.environ.get("STREAMK_FORCE_EDGE") == "1" or M % BLOCK_M != 0 or N % BLOCK_N != 0
    SHIFT = (EDGE_M and M >= BLOCK_M and N >= BLOCK_N and os.environ.get("STREAMK_EDGE_MODE", "shift") == "shift")
    if SHIFT:
        EDGE_M = EDGE_N = False
    if requested != "dp" and policy == "dp" and total_tiles <= NUM_PROGRAMS and not TILE_LIMIT:
        return v9_matmul(a, b, c, bias)
    STREAMK_TILES = 0 if policy == "dp" else total_tiles % NUM_PROGRAMS
    if policy == "two_tile" and STREAMK_TILES > 0 and total_tiles >= NUM_PROGRAMS:
        STREAMK_TILES += NUM_PROGRAMS
    END_TO_END = policy == "spread" or STREAMK_TILES > NUM_PROGRAMS
    assert 0 < STREAMK_NUM_PROGRAMS <= NUM_PROGRAMS
    # The tile-aligned split needs a program per tile; the end-to-end one does not.
    assert END_TO_END or STREAMK_TILES <= STREAMK_NUM_PROGRAMS
    assert STREAMK_TILES == 0 or (total_tiles - STREAMK_TILES) % NUM_PROGRAMS == 0
    NUM_XCDS = 8
    GROUP_SIZE_M = 4
    PAIRS_PER_TILE = K_PAD // (2 * BLOCK_K)
    tile_aligned = STREAMK_TILES > 0 and not END_TO_END
    ATOMIC_C = fixup == "atomic_iz" and STREAMK_TILES > 0
    if ATOMIC_C:
        assert not END_TO_END, "STREAMK_FIXUP=atomic_iz needs the tile-aligned one-tile split"
        assert c.dtype in (torch.bfloat16, torch.float16), "packed atomics need 16-bit C"
        assert partials == "fp32", "STREAMK_FIXUP=atomic_iz has no partials (STREAMK_PARTIALS)"
    L2 = exchange == "l2" and tile_aligned
    if exchange == "l2":
        assert fixup == "rs" or STREAMK_TILES == 0, "STREAMK_EXCHANGE=l2 is for the reduce-scatter ending"
        assert not END_TO_END, "STREAMK_EXCHANGE=l2 needs the tile-aligned one-tile split"
        assert placement == "tile", "STREAMK_EXCHANGE=l2 needs a tile's programs on one XCD, not chunk-major"
    CHUNK_MAJOR = placement == "chunk" and tile_aligned
    if placement == "chunk":
        assert not END_TO_END, "STREAMK_PLACEMENT=chunk needs the tile-aligned one-tile split"
    XCD_LOCAL = L2 and os.environ.get("STREAMK_XCD_LOCAL", "1") == "1"
    XCD_SPLIT = 1
    if XCD_LOCAL:
        XCD_SPLIT = xcd_local_split(STREAMK_TILES, STREAMK_NUM_PROGRAMS // STREAMK_TILES, PAIRS_PER_TILE,
                                    NUM_PROGRAMS, NUM_XCDS)
        # The kernel's tile-aligned constants then give every tile XCD_SPLIT programs.
        STREAMK_NUM_PROGRAMS = STREAMK_TILES * XCD_SPLIT
    if L2 and os.environ.get("STREAMK_UNSAFE_PLACEMENT") != "1":
        # Partials and flags stay in one XCD's L2: a tile whose programs span two XCDs would
        # read stale partials or wait forever. STREAMK_UNSAFE_PLACEMENT=1 skips this check
        # (the stress test's negative control).
        straddler = first_straddling_tile(STREAMK_TILES, STREAMK_NUM_PROGRAMS, PAIRS_PER_TILE, NUM_PROGRAMS,
                                          NUM_XCDS, BALANCE_XCDS, XCD_LOCAL, XCD_SPLIT)
        assert straddler is None, f"STREAMK_EXCHANGE=l2: a tile spans XCDs ({straddler})"
    rs = rs_split(STREAMK_TILES, STREAMK_NUM_PROGRAMS, PAIRS_PER_TILE, pow2_readers == "pow2")
    P, locks, tile_sync, epochs = streamk_workspace(a.device, NUM_PROGRAMS, BLOCK_M, BLOCK_N)
    if partials != "fp32":
        P = P.view(torch.bfloat16)
    grid = (NUM_PROGRAMS, 1)
    v26_streamk_ragged_k[grid](
        a,
        b,
        c,  #
        bias,
        P,
        locks,
        tile_sync,
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        next_epoch(epochs) if L2 else 0,
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
        REDUCE_SCATTER=fixup == "rs",
        END_TO_END=END_TO_END,
        RS_NB=rs["nb"],
        RS_U=rs["u"],
        RS_R_HI=rs["r_hi"],
        RS_R_LO=rs["r_lo"],
        RS_BAND=rs["band"],
        PARTIALS=PARTIALS_CODES[partials],
        TILE_LIMIT=TILE_LIMIT,
        ATOMIC_C=ATOMIC_C,
        L2=L2,
        XCD_LOCAL=XCD_LOCAL,
        XCD_SPLIT=XCD_SPLIT,
        CHUNK_MAJOR=CHUNK_MAJOR,
        EDGE_M=EDGE_M,
        EDGE_N=EDGE_N,
        EDGE_K=EDGE_K,
        SHIFT=SHIFT,
        num_warps=num_warps,
    )
    return c
