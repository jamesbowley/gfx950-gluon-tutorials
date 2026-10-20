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
"""v29_streamk_compact: C = A B (+ bias) on gfx950, data-parallel plus stream-K, one kernel.

v28 with only the paths that runs take (what was removed and why: ../DROPPED.md, v29 WORKLOG §1).

Work, in two phases:
1. Persistent data-parallel loop. Program p runs tiles p, p + NUM_PROGRAMS, ... of the first
   total_tiles - STREAMK_TILES tiles (v9's order: XCD-remapped tile ids, GROUP_SIZE_M swizzle), each
   tile's first K-step pair prefetched by the previous tile's epilogue.
2. Stream-K on the STREAMK_TILES leftover tiles, split in units of a K-step pair, one of:
   - tile-aligned one-tile split: tile t gets N_HI (the first EXTRA tiles) or N_LO consecutive programs, at
     most one per pair, each running one chunk of the tile's K. Chunk 0 is role 0.
   - end to end (two-tile: a full wave moved into stream-K as well; spread: the leftover tiles alone): the
     tiles' pairs laid end to end and cut evenly; each program runs its pieces last to first, so every
     program starts at K-step 0 and owns the tile whose last pair it holds.

A split tile's partials (fp32, the MFMA layout's registers stored lane-contiguous) are combined by:
- owner: the others store their partial to P[slot] write-through, drain and raise locks[slot]; the owner
  waits for its peers' flags, sums their partials into its accumulator, adds the bias, stores C and lowers
  the flags again;
- reduce-scatter (tile-aligned only): every program stores its partial and arrives on the tile's barrier;
  the first R (the largest power of two <= n) each sum 1/R of the tile's 32x32 blocks over all n partials
  and store them to C.
Exchange: partials written through with relaxed gpu-scope flags and barriers (wt), or, for reduce-scatter
only, every tile's programs on one XCD with the partials and epoch flags kept in its L2 (l2).

Every program of the stream-K phase must be resident at once: the endings wait on each other.

Tiles: four-quadrant tiles (BM, BN in 64, 128, 256; BK 64 or 128) with v9_tile's layouts, 256 programs of 4
waves; small tiles (a side of 32 or less) with v9_tile_small's strip loop, 1024 / W programs of W waves at
one wave per SIMD, a small BN run as C^T = B^T A^T.

Ragged shapes: EDGE_M / EDGE_N clamp the loads to the last in-bounds row / column and mask the stores and
the bias. EDGE_K rounds K up to whole pairs (K_PAD) and runs every tile's last, partial pair first, its
lanes past K masked, so the other pairs stay aligned to the row's start.
"""

import math
import os

import triton
import torch
from bias import init_acc
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from v9_tile import matmul_kernel as v9_tile
from v9_tile.matmul_kernel import (LDS_PER_CU, group_size_m, is_small, mfma_layout, parse_spec, small_config,
                                   small_lds_bytes, small_mfma_layout, small_store_layout, spec_options,
                                   store_layout, strip_load_layout, strip_shared_layout, swizzled_k,
                                   tile_load_layout, tile_shared_layout)


# ---------------------------------------------------------------------------------------------------------
# Programs and tiles
# ---------------------------------------------------------------------------------------------------------

@gluon.jit
def get_group_m_tile_ids(tile_ids, num_tiles_m, num_tiles_n, GROUP_SIZE_M: gl.constexpr):
    """(pid_m, pid_n) of a tile id under the GROUP_SIZE_M swizzle."""
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
def get_logical_chiplet_mapped_pids(NUM_PROGRAMS: gl.constexpr, NUM_XCDS: gl.constexpr):
    """The program id renumbered so that each XCD's programs are consecutive (hardware pid p runs on XCD
    p % NUM_XCDS). The first NUM_PROGRAMS % NUM_XCDS XCDs ("tall") hold one program more."""
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
            pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid
    return pid


@gluon.jit
def streamk_compact_pid(spid, SNP: gl.constexpr, NUM_PROGRAMS: gl.constexpr, NUM_XCDS: gl.constexpr):
    """Stream-K id of XCD-mapped program spid when SNP programs take part, or NUM_PROGRAMS if it idles.
    XCD x keeps its first SNP // NUM_XCDS programs (one more for the first SNP % NUM_XCDS XCDs), numbered
    contiguously as spids are, so the idle programs are spread over the XCDs."""
    PER: gl.constexpr = NUM_PROGRAMS // NUM_XCDS
    BASE: gl.constexpr = SNP // NUM_XCDS
    TALL: gl.constexpr = SNP % NUM_XCDS
    xcd = spid // PER
    local = spid % PER
    active = local < BASE + (xcd < TALL).to(gl.int32)
    return gl.where(active, xcd * BASE + gl.minimum(xcd, TALL) + local, NUM_PROGRAMS)


@gluon.jit
def xcd_remap_tiles(vpid, count, NUM_XCDS: gl.constexpr):
    """Tile id (before the GROUP_SIZE_M swizzle) of persistent virtual pid vpid = pid + i NUM_PROGRAMS:
    XCD x owns a contiguous 1/NUM_XCDS of the `count` tile ids (v9's get_pids)."""
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
def streamk_tile_id(t, total_tiles, total_full_tiles, NUM_XCDS: gl.constexpr):
    """Tile id of stream-K tile t: the persistent loop covers xcd_remap_tiles(v) for v < total_full_tiles,
    which leaves the top of every XCD's block of tile ids, and tile t is the t-th smallest of those. t follows
    the stream-K ids, which are already grouped by XCD, so consecutive t are neighbouring tiles."""
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
    return tile_id


@gluon.jit
def streamk_chunk_of(spid, EXTRA: gl.constexpr, N_HI: gl.constexpr, N_LO: gl.constexpr,
                     R_HI: gl.constexpr, R_LO: gl.constexpr):
    """(tile, chunk, programs on the tile, reduce-scatter readers of the tile) of stream-K program spid in
    the tile-aligned split: the first EXTRA tiles get N_HI consecutive programs (R_HI readers), the rest
    N_LO (R_LO)."""
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
def xcd_local_chunk_of(spid, S: gl.constexpr, PER: gl.constexpr, NUM_XCDS: gl.constexpr, N: gl.constexpr):
    """(tile, chunk, programs on the tile) of XCD-mapped program spid in the XCD-local split (l2), with 0
    programs if it idles. XCD x = spid // PER owns S // NUM_XCDS consecutive stream-K tiles (one more for
    x < S % NUM_XCDS); its local program l works on its tile l // N as chunk l % N, so a tile's N programs
    are on one XCD. The host checks ceil(S / NUM_XCDS) N <= PER."""
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
    """End-to-end split: the program whose range holds pair i, when the first R programs hold L + 1
    pairs and the rest L."""
    head: gl.constexpr = R * (L + 1)
    if L == 0:
        owner = i // (L + 1)
    else:
        owner = i // (L + 1)
        if i >= head:
            owner = R + (i - head) // L
    return owner


# ---------------------------------------------------------------------------------------------------------
# Quadrant tiles: the main loop
#
# A tile is four quadrants (tl, bl, tr, br) of HALF_M x HALF_N, each with its own fp32 accumulator: accs is
# (acc_tl, acc_bl, acc_tr, acc_br). LDS holds two buffers of each operand half: smem is (A top, A bottom,
# B left, B right). An operand's offsets are aop / bop = (offsets, offsets_next, offsets_bot,
# offsets_bot_next, half, odd) (operand_offsets): even and odd K-steps, top and bottom halves, the half-tile
# and odd-K-step shifts that live in the scalar base.
# ---------------------------------------------------------------------------------------------------------

@gluon.jit
def k_step_pair(accs, a_top, b_left, smem, a_base, b_base, aop, bop, dotOpLayoutA: gl.constexpr,
                dotOpLayoutB: gl.constexpr):
    """One main-loop iteration, two K-steps: consumes a_top / b_left (in registers) and buffers 0 and 1,
    prefetches the two K-steps at a_base / b_base into the same buffers, and returns the accumulators and
    the next a_top / b_left. Every wait keeps 5 groups of async copies in flight. Constant-zero
    accumulators make the first four MFMAs zero-C MFMAs."""
    acc_tl, acc_bl, acc_tr, acc_br = accs
    smemA_top, smemA_bot, smemB_left, smemB_right = smem
    a_offsets, a_offsets_next, a_offsets_bot, a_offsets_bot_next, a_half, a_odd = aop
    b_offsets, b_offsets_next, b_offsets_bot, b_offsets_bot_next, b_half, b_odd = bop

    # Even K-step: consume buffer 0, prefetch into buffer 0.
    acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    a_bot = smemA_bot.index(0).load(dotOpLayoutA)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(0), b_base, b_offsets)
    gl.amd.cdna4.async_copy.commit_group()

    acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    b_right = smemB_right.index(0).load(dotOpLayoutB)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_top.index(0), a_base, a_offsets)
    gl.amd.cdna4.async_copy.commit_group()

    acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    b_left = smemB_left.index(1).load(dotOpLayoutB)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_bot.index(0), a_base + a_half, a_offsets_bot)
    gl.amd.cdna4.async_copy.commit_group()

    acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    a_top = smemA_top.index(1).load(dotOpLayoutA)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_right.index(0), b_base + b_half, b_offsets_bot)
    gl.amd.cdna4.async_copy.commit_group()

    # Odd K-step: consume buffer 1, prefetch into buffer 1.
    acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    a_bot = smemA_bot.index(1).load(dotOpLayoutA)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(1), b_base + b_odd, b_offsets_next)
    gl.amd.cdna4.async_copy.commit_group()

    acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    b_right = smemB_right.index(1).load(dotOpLayoutB)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_top.index(1), a_base + a_odd, a_offsets_next)
    gl.amd.cdna4.async_copy.commit_group()

    acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    b_left = smemB_left.index(0).load(dotOpLayoutB)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_bot.index(1), a_base + a_half + a_odd, a_offsets_bot_next
    )
    gl.amd.cdna4.async_copy.commit_group()

    acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    a_top = smemA_top.index(0).load(dotOpLayoutA)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_right.index(1), b_base + b_half + b_odd, b_offsets_bot_next
    )
    gl.amd.cdna4.async_copy.commit_group()

    return (acc_tl, acc_bl, acc_tr, acc_br), a_top, b_left


@gluon.jit
def load_first_pair(a_base, b_base, smem, aop, bop, dotOpLayoutA: gl.constexpr, dotOpLayoutB: gl.constexpr,
                    ma0=None, mb0=None, ma1=None, mb1=None):
    """Pipeline prologue: the two K-steps at a_base / b_base into buffers 0 and 1, and the first one's
    a_top / b_left. ma* / mb*: load masks of A's and B's K-step 0 and 1 (EDGE_K), None for unmasked."""
    smemA_top, smemA_bot, smemB_left, smemB_right = smem
    a_offsets, a_offsets_next, a_offsets_bot, a_offsets_bot_next, a_half, a_odd = aop
    b_offsets, b_offsets_next, b_offsets_bot, b_offsets_bot_next, b_half, b_odd = bop
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


# ---------------------------------------------------------------------------------------------------------
# Ragged edges
# ---------------------------------------------------------------------------------------------------------

@gluon.jit
def k_tail_mask(k_hi, STEP: gl.constexpr, ROWS: gl.constexpr, BLOCK_K: gl.constexpr, layout: gl.constexpr,
                K_FIRST: gl.constexpr):
    """Load mask of K-step STEP (0 or 1) of a half-tile in the last pair, whose first k_hi of 2 BLOCK_K K
    values are inside the matrix. K_FIRST: B's [BLOCK_K, ROWS] half-tile, else A's [ROWS, BLOCK_K]."""
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
    K-step 1 of the last pair always reaches past K, K-step 0 only when K_RES > BLOCK_K; a mask that can
    never drop a lane is None."""
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
    """Load offsets of a tile's two halves (A's top / bottom or B's left / right) with the rows (A) or
    columns (B) past lim, the tile's last in-bounds one, clamped to it. Clamped lanes read valid memory
    and only feed C elements that the store masks drop. The tile's position stays in the 64-bit scalar
    base, so the offsets span one tile."""
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
    """An operand's (offsets, offsets_next, offsets_bot, offsets_bot_next, half, odd). Without EDGE: the
    kernel-wide offsets, the second half a scalar shift `half` away and the odd K-step in its own offsets.
    With EDGE: the tile's clamped top / bot offsets, and the odd K-step a scalar shift `odd` away, so a
    tile still holds four offset tensors per operand pair."""
    if EDGE:
        o0, o1, o2, o3, h, d = top, top, bot, bot, 0, odd
    else:
        o0, o1, o2, o3, h, d = offsets, offsets_next, offsets, offsets_next, half, 0
    return o0, o1, o2, o3, h, d


@gluon.jit
def edge_mask(rows_left, cols_left, ROWS: gl.constexpr, COLS: gl.constexpr, layout: gl.constexpr,
              EDGE_M: gl.constexpr, EDGE_N: gl.constexpr):
    """In-bounds mask of a ROWS x COLS block of C with rows_left rows and cols_left columns before the
    matrix edge (either may exceed the block)."""
    r = gl.arange(0, ROWS, gl.SliceLayout(1, layout))[:, None]
    c = gl.arange(0, COLS, gl.SliceLayout(0, layout))[None, :]
    if EDGE_M and EDGE_N:
        mask = (r < rows_left) & (c < cols_left)
    elif EDGE_M:
        mask = (r < rows_left) & (c >= 0)
    else:
        mask = (r >= 0) & (c < cols_left)
    return mask


@gluon.jit
def store_edge(c, c_base, c_offsets, layout: gl.constexpr, rows_left, cols_left, EDGE_M: gl.constexpr,
               EDGE_N: gl.constexpr):
    """buffer_store of a block of C, masked to its first rows_left rows / cols_left columns under EDGE_M /
    EDGE_N."""
    if EDGE_M or EDGE_N:
        mask = edge_mask(rows_left, cols_left, c.shape[0], c.shape[1], layout, EDGE_M, EDGE_N)
        gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_offsets, stored_value=c, mask=mask)
    else:
        gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_offsets, stored_value=c)


@gluon.jit
def store_c(acc, c_base, c_offsets, out_dtype: gl.constexpr, gStoreLayoutC: gl.constexpr,
            rows_left=0, cols_left=0, EDGE_M: gl.constexpr = False, EDGE_N: gl.constexpr = False):
    """Convert a quadrant to C's type and store layout and store it (store_edge)."""
    c = acc.to(out_dtype)
    c = gl.convert_layout(c, layout=gStoreLayoutC)
    store_edge(c, c_base, c_offsets, gStoreLayoutC, rows_left, cols_left, EDGE_M, EDGE_N)


@gluon.jit
def init_acc_edge(bias_ptr, pid_n, cols_left, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, MFMA: gl.constexpr,
                  ADD_BIAS: gl.constexpr, EDGE_N: gl.constexpr):
    """bias.init_acc, with the bias loaded only for the tile's first cols_left columns under EDGE_N."""
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


# ---------------------------------------------------------------------------------------------------------
# Synchronisation
#
# Partials are stored write-through (.wt) and drained in every wave before the flag or barrier arrival that
# publishes them, and read with .cv loads, which miss L1; so every atomic is relaxed, with compiler fences
# where an acquire would be. Flags and counts are re-armed inside the launch (zero between launches).
# ---------------------------------------------------------------------------------------------------------

@gluon.jit
def _one():
    return gl.zeros([1], gl.int32, gl.BlockedLayout([1], [64], [gl.num_warps()], [0]))


@gluon.jit
def drain_and_barrier():
    """Every wave waits for all its vector memory ops (stores and async copies: vmcnt is one counter),
    then a workgroup barrier. A flag set after this is ordered after all the waves' partial stores."""
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
def tile_barrier(sync_ptr, b, n, wait, slot, epoch, NUM_PROGRAMS: gl.constexpr, L2: gl.constexpr):
    """Arrive on the reduce-scatter barrier of the tile whose first slot is b and, if wait, wait until all
    n of its programs have published their partials.

    wt: sync[2b] counts arrivals and sync[2b + 1] is a generation. The last arriver resets the count and
    bumps the generation, which re-arms the barrier for the next launch. Scalar atomics run once per
    workgroup and broadcast their result.

    l2 (every program of the tile on one XCD): no atomics. The program stores the launch's epoch to its flag
    sync[2 NUM_PROGRAMS + slot], then polls the tile's n flags with .cg loads, which miss L1 and hit the
    XCD's L2. Each flag has one writer and the host bumps the epoch every launch, so nothing is re-armed."""
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
        # If the last arrival and the bump overtook this read, the program would wait for a generation
        # that never comes.
        asm_fence("s_waitcnt vmcnt(0)")
        arrived = gl.atomic_add(cnt_ptr, 1, sem="relaxed", scope="gpu")
        if arrived == n - 1:
            gl.atomic_xchg(cnt_ptr, 0, sem="relaxed", scope="gpu")
            gl.atomic_xchg(gen_ptr, g0 + 1, sem="relaxed", scope="gpu")
        elif wait:
            while gl.atomic_add(gen_ptr, 0, sem="relaxed", scope="gpu") == g0:
                pass
        asm_fence("s_nop 0")  # no partial read moves above the wait


# ---------------------------------------------------------------------------------------------------------
# Quadrant tiles: partials and endings
#
# A program's slot in P is one fp32 tile, four quadrants of QUAD = HALF_M HALF_N elements, each stored
# straight from the MFMA layout (lane_contiguous_offsets) and read back by the owner in the same layout.
# Reduce-scatter reads it as NBLK = BLOCK_M BLOCK_N / 1024 blocks of 1024 elements, one per (quadrant,
# register vector), each a 32x32 square of the tile.
# ---------------------------------------------------------------------------------------------------------

@gluon.constexpr_function
def lane_map_matches(ll, half_m, half_n):
    """True if a HALF_M x HALF_N quadrant's MFMA layout (as a linear layout) has the bases
    lane_contiguous_offsets assumes: registers col 1, 2, then the column blocks 32 .. HALF_N / 2, then
    the row blocks 32 .. HALF_M / 2; lanes row 1, 2, 4, 8 and col 4, 8; warps col 16 and row 16."""
    reg = [[0, 1], [0, 2]] + [[0, c] for c in (32, 64) if c < half_n] + [[r, 0] for r in (32, 64) if r < half_m]
    return ([list(b) for b in ll.reg_bases] == reg
            and [list(b) for b in ll.lane_bases] == [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]]
            and [list(b) for b in ll.warp_bases] == [[0, 16], [16, 0]])


@gluon.jit
def lane_contiguous_offsets(HALF_M: gl.constexpr, HALF_N: gl.constexpr, mfmaLayout: gl.constexpr):
    """Element offsets of a partial quadrant in its slot, in the MFMA layout: [VQ register vectors][4 warps]
    [64 lanes][4 fp32], VQ = (HALF_M / 32) (HALF_N / 32), the column blocks before the row blocks.

    Lane l of a wave writes elements 4l .. 4l + 3 of its 1 KiB block, so one 16-byte-per-lane access
    covers 8 whole cache lines (row-major addressing would spread it over 16 rows). The map is a bit
    permutation of (row, col) that follows the layout's bases, checked at compile time."""
    gl.static_assert(lane_map_matches(gl.to_linear_layout(mfmaLayout, [HALF_M, HALF_N]), HALF_M, HALF_N),
                     "the MFMA layout's bases are not the order the partial map assumes")
    VN: gl.constexpr = HALF_N // 32
    r = gl.arange(0, HALF_M, gl.SliceLayout(1, mfmaLayout))
    c = gl.arange(0, HALF_N, gl.SliceLayout(0, mfmaLayout))
    # The column term is c plus terms constant over each 4 columns, so axis analysis still sees 4
    # contiguous, aligned elements (16-byte accesses); written with c % 4 or shifts and masks, the same map
    # compiles to single-dword accesses. Vector v = column block + VN row block.
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
def store_partial(acc, q: gl.constexpr, p_ptr, spid, p_offsets, HALF_M: gl.constexpr, HALF_N: gl.constexpr,
                  CACHE: gl.constexpr = ".wt"):
    """Quadrant q of a partial to slot P[spid]: write-through, or the default policy for the l2 exchange."""
    QUAD: gl.constexpr = HALF_M * HALF_N
    SLOT: gl.constexpr = 4 * QUAD
    gl.amd.cdna3.buffer_store(stored_value=acc, ptr=p_ptr + spid.to(gl.int64) * SLOT + q * QUAD,
                              offsets=p_offsets, cache=CACHE)


@gluon.jit
def owner_store(acc, psum, q: gl.constexpr, bias_ptr, pid_n, c_base, c_offsets, BLOCK_N: gl.constexpr,
                HALF_N: gl.constexpr, ADD_BIAS: gl.constexpr, out_dtype: gl.constexpr, gStoreLayoutC: gl.constexpr,
                mfmaLayout: gl.constexpr, rows_left, cols_left, EDGE_M: gl.constexpr, EDGE_N: gl.constexpr):
    """Quadrant q of C: the accumulator plus psum (the peers' partials) and the bias. Contributors'
    partials exclude the bias, so it is added once, here."""
    HALF_M: gl.constexpr = acc.shape[0]
    if ADD_BIAS:
        offs_pn = gl.arange(0, HALF_N, gl.SliceLayout(0, mfmaLayout))
        if EDGE_N:
            bias = gl.amd.cdna3.buffer_load(
                ptr=bias_ptr + pid_n * BLOCK_N + (q // 2) * HALF_N, offsets=offs_pn,
                mask=offs_pn < cols_left - (q // 2) * HALF_N, other=0.0,
            )
        else:
            bias = gl.amd.cdna3.buffer_load(ptr=bias_ptr + pid_n * BLOCK_N + (q // 2) * HALF_N, offsets=offs_pn)
        psum = psum + bias.to(gl.float32)[None, :]
    store_c(acc + psum, c_base, c_offsets, out_dtype, gStoreLayoutC,
            rows_left - (q % 2) * HALF_M, cols_left - (q // 2) * HALF_N, EDGE_M, EDGE_N)


@gluon.jit
def peer_sums(p_ptr, spid, n_peers, p_offsets, z, q0: gl.constexpr, NQ: gl.constexpr, PEER_STEP: gl.constexpr,
              HALF_M: gl.constexpr, HALF_N: gl.constexpr, MAX_PEERS: gl.constexpr):
    """The sums of quadrants q0 .. q0 + NQ - 1 (NQ = 1, 2 or 4) over the partials of peers spid + PEER_STEP i,
    i = 1 .. n_peers, as four tensors (the last 4 - NQ stay z, zeros); a peer's NQ quadrants are one batch
    of loads.
    - MAX_PEERS > 0: a compile-time loop over MAX_PEERS peers, the loads of peers past n_peers masked off,
      so that several peers' batches can be in flight at once.
    - Otherwise a run-time loop, each batch's loads written before its adds: written load, add, load, add,
      the scheduler reuses one quadrant's registers for the next and waits for each load in turn."""
    QUAD: gl.constexpr = HALF_M * HALF_N
    SLOT: gl.constexpr = 4 * QUAD
    s0, s1, s2, s3 = z, z, z, z
    if MAX_PEERS > 0:
        for j in gl.static_range(MAX_PEERS):
            peer = (spid + PEER_STEP * (1 + j)).to(gl.int64)
            live = j < n_peers
            s0 += gl.amd.cdna3.buffer_load(ptr=p_ptr + peer * SLOT + q0 * QUAD, offsets=p_offsets, mask=live,
                                           other=0.0, cache=".cv").to(gl.float32)
            if NQ > 1:
                s1 += gl.amd.cdna3.buffer_load(ptr=p_ptr + peer * SLOT + (q0 + 1) * QUAD, offsets=p_offsets,
                                               mask=live, other=0.0, cache=".cv").to(gl.float32)
            if NQ > 2:
                s2 += gl.amd.cdna3.buffer_load(ptr=p_ptr + peer * SLOT + (q0 + 2) * QUAD, offsets=p_offsets,
                                               mask=live, other=0.0, cache=".cv").to(gl.float32)
                s3 += gl.amd.cdna3.buffer_load(ptr=p_ptr + peer * SLOT + (q0 + 3) * QUAD, offsets=p_offsets,
                                               mask=live, other=0.0, cache=".cv").to(gl.float32)
    else:
        for j in range(n_peers):
            peer = (spid + PEER_STEP * (1 + j)).to(gl.int64)
            x0 = gl.amd.cdna3.buffer_load(ptr=p_ptr + peer * SLOT + q0 * QUAD, offsets=p_offsets,
                                          cache=".cv").to(gl.float32)
            if NQ == 1:
                s0 += x0
            elif NQ == 2:
                x1 = gl.amd.cdna3.buffer_load(ptr=p_ptr + peer * SLOT + (q0 + 1) * QUAD, offsets=p_offsets,
                                              cache=".cv").to(gl.float32)
                s0 += x0
                s1 += x1
            else:
                x1 = gl.amd.cdna3.buffer_load(ptr=p_ptr + peer * SLOT + (q0 + 1) * QUAD, offsets=p_offsets,
                                              cache=".cv").to(gl.float32)
                x2 = gl.amd.cdna3.buffer_load(ptr=p_ptr + peer * SLOT + (q0 + 2) * QUAD, offsets=p_offsets,
                                              cache=".cv").to(gl.float32)
                x3 = gl.amd.cdna3.buffer_load(ptr=p_ptr + peer * SLOT + (q0 + 3) * QUAD, offsets=p_offsets,
                                              cache=".cv").to(gl.float32)
                s0 += x0
                s1 += x1
                s2 += x2
                s3 += x3
    return s0, s1, s2, s3


@gluon.jit
def owner_ending(accs, is_partial, n_peers, spid, pid_n, bias_ptr, p_ptr, locks_ptr, c_bases, c_offsets,
                 rows_left, cols_left, BLOCK_N: gl.constexpr, PEER_STEP: gl.constexpr, ADD_BIAS: gl.constexpr,
                 out_dtype: gl.constexpr, gStoreLayoutC: gl.constexpr, mfmaLayout: gl.constexpr,
                 EDGE_M: gl.constexpr, EDGE_N: gl.constexpr, QPB: gl.constexpr, MAX_PEERS: gl.constexpr):
    """The owner ending of a segment. A contributor (is_partial) stores its four quadrants to P[spid],
    drains and raises locks[spid]. The owner waits for its n_peers peers' flags (spid + PEER_STEP i), reads
    their partials QPB quadrants at a time (16 fp32 loads per wave: 4 at 128x128, 2 at 256x128 / 128x256, 1
    at 256x256; peer_sums), adds each batch's sums into the accumulator once, outside the peer loop (a loop
    or branch that carries the AGPR accumulator spills), and stores that batch of C before the next batch's
    reads, so only one batch's sums are live. Each partial has one reader, so the owner lowers its peers'
    flags for the next launch; the kernel boundary orders that against it."""
    HALF_M: gl.constexpr = accs[0].shape[0]
    HALF_N: gl.constexpr = accs[0].shape[1]
    for pj in range(n_peers):
        while gl.atomic_add(locks_ptr + spid + PEER_STEP * (1 + pj), 0, sem="relaxed", scope="gpu") == 0:
            pass
    if n_peers > 0:
        asm_fence("s_nop 0")

    # With a run-time peer loop over one quadrant per batch (256x256: end to end, or above 8 programs per tile)
    # a contributor stores each quadrant just before its batch: storing all four first is about 1% slower on
    # two-tile there (v29 WORKLOG §1).
    STORE_EACH: gl.constexpr = QPB == 1 and MAX_PEERS == 0
    p_offsets = lane_contiguous_offsets(HALF_M, HALF_N, mfmaLayout)
    z = gl.zeros((HALF_M, HALF_N), gl.float32, mfmaLayout)
    if not STORE_EACH:
        if is_partial != 0:
            for q in gl.static_range(4):
                store_partial(accs[q], q, p_ptr, spid, p_offsets, HALF_M, HALF_N)
    for q0 in gl.static_range(0, 4, QPB):
        if STORE_EACH:
            if is_partial != 0:
                store_partial(accs[q0], q0, p_ptr, spid, p_offsets, HALF_M, HALF_N)
        sums = peer_sums(p_ptr, spid, n_peers, p_offsets, z, q0, QPB, PEER_STEP, HALF_M, HALF_N, MAX_PEERS)
        if is_partial == 0:
            for i in gl.static_range(QPB):
                owner_store(accs[q0 + i], sums[i], q0 + i, bias_ptr, pid_n, c_bases[q0 + i], c_offsets, BLOCK_N,
                            HALF_N, ADD_BIAS, out_dtype, gStoreLayoutC, mfmaLayout, rows_left, cols_left, EDGE_M,
                            EDGE_N)

    if is_partial != 0:
        # The flag may only become visible after all four waves' partial stores have completed.
        drain_and_barrier()
        gl.atomic_xchg(locks_ptr + spid, 1, sem="relaxed", scope="gpu")
    for pj in range(n_peers):
        gl.atomic_xchg(locks_ptr + spid + PEER_STEP * (1 + pj), 0, sem="relaxed", scope="gpu")


@gluon.jit
def block_col(v, VN: gl.constexpr):
    """Column of register vector (32x32 block) v in its quadrant, VN blocks across."""
    col = v * 0
    if VN == 4:
        col = (v % 2) * 32 + (v // 2 % 2) * 64
    elif VN == 2:
        col = (v % 2) * 32
    return col


@gluon.jit
def to_band(x, G: gl.constexpr, VN: gl.constexpr = 4):
    """(G, 32, 32) aligned blocks -> their 2-D band, VN blocks across a quadrant. Block index bits from the
    lowest: the column blocks, then the row blocks. VN = 4: G = 1, 2, 4: 32 x 32 G; 8, 16: 8 G x 128."""
    if VN == 4:
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
    else:
        CB: gl.constexpr = min(G, VN)
        RB: gl.constexpr = G // CB
        y = gl.reshape(gl.permute(gl.reshape(x, [RB, CB, 32, 32]), [0, 2, 1, 3]), [RB * 32, CB * 32])
    return y


@gluon.jit
def band_pos(g0, layout: gl.constexpr, ROWS: gl.constexpr, COLS: gl.constexpr, HALF_M: gl.constexpr = 128,
             HALF_N: gl.constexpr = 128):
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
def reduce_scatter_slice(p_ptr, bias_ptr, c_tile, b, j, n, readers, pid_n, stride_cm, stride_cn,
                         BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, ADD_BIAS: gl.constexpr, NB: gl.constexpr,
                         U: gl.constexpr, out_dtype: gl.constexpr, LOAD_CACHE: gl.constexpr, rows_left, cols_left,
                         EDGE_M: gl.constexpr, EDGE_N: gl.constexpr):
    """Reader j of `readers` (a power of two) of a tile split n ways: sums blocks [NBLK j / readers,
    NBLK (j + 1) / readers) over the n partials in slots P[b .. b + n - 1], adds the bias and stores them to
    C. NB blocks (NB <= VQ, so a group stays in one quadrant) at a time, U partials of a group per trip, so
    NB U loads per lane are in flight. Each group is reshaped to its 2-D band and stored along C's rows, 16
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
    # Rows before columns in lane order: lane l gets row l % 16 and columns 4 (l // 16) .. + 3, which in_blk
    # puts at 4 l .. 4 l + 3, so each 4-lane group reads 64 contiguous bytes. Columns first, lanes 0-3 hit 4
    # lines 256 bytes apart and the L1 stalls on tag conflicts (reads take twice as long).
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
        valid = (blk < blk_hi) & (in_blk >= 0)
        p_off = blk * 1024 + in_blk
        s = gl.zeros((NB, 32, 32), gl.float32, L3)
        for p0 in range(0, n, U):
            for u in gl.static_range(U):
                slot = p_ptr + (b + p0 + u).to(gl.int64) * SLOT
                s += gl.amd.cdna3.buffer_load(
                    ptr=slot, offsets=p_off, mask=valid & (p0 + u < n), other=0.0, cache=LOAD_CACHE
                ).to(gl.float32)
        if ADD_BIAS:
            # Block q VQ + v: quadrant q (tl, bl, tr, br); register vector v's bits are the column blocks
            # (32, 64), then the row blocks.
            qb = (g + ib) // VQ
            vb = (g + ib) % VQ
            col_b = (qb // 2) * HALF_N
            if VN >= 2:
                col_b = col_b + (vb % 2) * 32
            if VN == 4:
                col_b = col_b + (vb // 2 % 2) * 64
            col_b = col_b + cb
            bias_mask = (g + ib) < blk_hi
            if EDGE_N:
                bias_mask = bias_mask & (col_b < cols_left)
            bias = gl.amd.cdna3.buffer_load(
                ptr=bias_ptr + pid_n * BLOCK_N, offsets=col_b, mask=bias_mask, other=0.0
            )
            s = s + bias.to(gl.float32)[:, None, :]
        cs = gl.convert_layout(to_band(s.to(out_dtype), NB, VN), SL)
        trow, tcol = band_pos(g, SL, ROWS, COLS, HALF_M, HALF_N)
        if EDGE_M or EDGE_N:
            gl.amd.cdna3.buffer_store(stored_value=cs, ptr=c_tile, offsets=trow * stride_cm + tcol * stride_cn,
                                      mask=(trow < rows_left) & (tcol < cols_left))
        else:
            gl.amd.cdna3.buffer_store(stored_value=cs, ptr=c_tile, offsets=trow * stride_cm + tcol * stride_cn)


@gluon.jit
def rs_ending(accs, spid, j, n_tile, r_tile, pid_n, bias_ptr, p_ptr, sync_ptr, epoch, c_bases, c_offsets,
              stride_cm, stride_cn, rows_left, cols_left, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
              ADD_BIAS: gl.constexpr, RS_NB: gl.constexpr, RS_U: gl.constexpr, L2: gl.constexpr,
              NUM_PROGRAMS: gl.constexpr, out_dtype: gl.constexpr, gStoreLayoutC: gl.constexpr,
              mfmaLayout: gl.constexpr, EDGE_M: gl.constexpr, EDGE_N: gl.constexpr):
    """The reduce-scatter ending of tile-aligned segment j of n_tile (its slots are spid - j .. + n_tile - 1).
    A whole tile stores C directly. A split tile's programs all store their partial and arrive on the
    tile's barrier; the first r_tile wait on it and each reduces its slice (reduce_scatter_slice). Under
    l2 the partials stay in the XCD's L2 (default-policy stores, .cg loads)."""
    HALF_M: gl.constexpr = BLOCK_M // 2
    HALF_N: gl.constexpr = BLOCK_N // 2
    RS_STORE: gl.constexpr = "" if L2 else ".wt"
    RS_LOAD: gl.constexpr = ".cg" if L2 else ".cv"
    is_split = (n_tile > 1).to(gl.int32)
    p_offsets = lane_contiguous_offsets(HALF_M, HALF_N, mfmaLayout)
    z = gl.zeros((HALF_M, HALF_N), gl.float32, mfmaLayout)
    for q in gl.static_range(4):
        if is_split != 0:
            store_partial(accs[q], q, p_ptr, spid, p_offsets, HALF_M, HALF_N, RS_STORE)
        if is_split == 0:
            owner_store(accs[q], z, q, bias_ptr, pid_n, c_bases[q], c_offsets, BLOCK_N, HALF_N, ADD_BIAS,
                        out_dtype, gStoreLayoutC, mfmaLayout, rows_left, cols_left, EDGE_M, EDGE_N)
    if n_tile > 1:
        # The arrival may only become visible after all four waves' partial stores have completed.
        drain_and_barrier()
        is_reader = j < r_tile
        tile_barrier(sync_ptr, spid - j, n_tile, is_reader, spid, epoch, NUM_PROGRAMS, L2)
        if is_reader:
            reduce_scatter_slice(
                p_ptr, bias_ptr, c_bases[0] + pid_n * BLOCK_N * stride_cn, spid - j, j, n_tile, r_tile, pid_n,
                stride_cm, stride_cn, BLOCK_M, BLOCK_N, ADD_BIAS, RS_NB, RS_U, out_dtype, RS_LOAD, rows_left,
                cols_left, EDGE_M, EDGE_N,
            )


# ---------------------------------------------------------------------------------------------------------
# Quadrant tiles: a stream-K segment and a persistent tile
# ---------------------------------------------------------------------------------------------------------

@gluon.jit
def quadrant_segment(
    tile_id, k0, n_pairs, is_partial, n_peers, spid, j, n_tile, r_tile,
    a_ptr, b_ptr, c_ptr, bias_ptr, p_ptr, locks_ptr, sync_ptr, smem, kofs,
    num_tiles_m, num_tiles_n, strides, M, N, epoch,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr, GROUP_SIZE_M: gl.constexpr,
    ADD_BIAS: gl.constexpr, PEER_STEP: gl.constexpr, REDUCE_SCATTER: gl.constexpr, RS_NB: gl.constexpr,
    RS_U: gl.constexpr, L2: gl.constexpr, NUM_PROGRAMS: gl.constexpr, mfmaLayout: gl.constexpr,
    gLoadLayoutA: gl.constexpr, gLoadLayoutB: gl.constexpr, EDGE_M: gl.constexpr, EDGE_N: gl.constexpr,
    EDGE_K: gl.constexpr, K_PAD: gl.constexpr, K_RES: gl.constexpr, SEG_LOOP: gl.constexpr,
    OWNER_QPB: gl.constexpr, OWNER_MAX_PEERS: gl.constexpr,
):
    """K-steps [k0, k0 + 2 n_pairs) of tile tile_id (n_pairs >= 1), from its own prologue, then the
    segment's ending: reduce-scatter (rs_ending) or the owner ending (owner_ending). spid is the program's
    workspace slot, j its chunk (role 0 owns the tile), n_tile / r_tile the tile's programs and readers.

    Under EDGE_K pair p of a tile (K-steps 2p, 2p + 1) is at columns 2 BLOCK_K (p - 1) for p >= 1, and pair
    0 is the last, partial pair, at K_PAD - 2 BLOCK_K, loaded with the lanes past K masked. So only the
    segment that starts at k0 = 0 masks, in its prologue. Every pair runs through the main-loop body; the
    last one's prefetch reloads the last pair (clamped), which the drain discards.

    SEG_LOOP: the caller runs segments in a loop (the end-to-end path), so the C row offsets are made
    tile-dependent (clamped to the tile's last row); otherwise the compiler hoists them above the loop and
    holds them, spilled, across it."""
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn = strides
    a_offsets, a_offsets_next, a_half, b_offsets, b_offsets_next, b_half = kofs
    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfmaLayout, k_width=8)
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfmaLayout, k_width=8)
    pid_m, pid_n = get_group_m_tile_ids(tile_id, num_tiles_m, num_tiles_n, GROUP_SIZE_M)
    rows_left = M - pid_m * BLOCK_M
    cols_left = N - pid_n * BLOCK_N

    a_top_off, a_bot_off = a_offsets, a_offsets
    if EDGE_M:
        a_top_off, a_bot_off = clamped_tile_offsets(rows_left - 1, stride_am, stride_ak, BLOCK_M // 2, BLOCK_K,
                                                    gLoadLayoutA, True)
    b_top_off, b_bot_off = b_offsets, b_offsets
    if EDGE_N:
        b_top_off, b_bot_off = clamped_tile_offsets(cols_left - 1, stride_bn, stride_bk, BLOCK_N // 2, BLOCK_K,
                                                    gLoadLayoutB, False)
    aop = operand_offsets(a_top_off, a_bot_off, a_offsets, a_offsets_next, a_half, BLOCK_K * stride_ak, EDGE_M)
    bop = operand_offsets(b_top_off, b_bot_off, b_offsets, b_offsets_next, b_half, BLOCK_K * stride_bk, EDGE_N)

    if EDGE_K:
        # Pair i of the segment (i >= 1) at a_tile + 2 BLOCK_K i; its first pair at a_tile, or for k0 = 0
        # the last, partial pair, masked at K.
        a_row = a_ptr + pid_m.to(gl.int64) * BLOCK_M * stride_am
        b_row = b_ptr + pid_n.to(gl.int64) * BLOCK_N * stride_bn
        a_tile = a_row + (k0 - 2).to(gl.int64) * BLOCK_K * stride_ak
        b_tile = b_row + (k0 - 2).to(gl.int64) * BLOCK_K * stride_bk
        first_col = gl.where(k0 == 0, K_PAD - 2 * BLOCK_K, (k0 - 2) * BLOCK_K).to(gl.int64)
        k_hi = gl.where(k0 == 0, 2 * BLOCK_K - K_RES, 2 * BLOCK_K)
        ma0, mb0, ma1, mb1 = tail_pair_masks(k_hi, K_RES, BLOCK_M, BLOCK_N, BLOCK_K, gLoadLayoutA, gLoadLayoutB)
        a_top, b_left = load_first_pair(a_row + first_col * stride_ak, b_row + first_col * stride_bk, smem, aop,
                                        bop, dotOpLayoutA, dotOpLayoutB, ma0, mb0, ma1, mb1)
        # The last pair's clamped prefetch must stay inside the row: for k0 = 0, a_tile itself is a pair
        # before it.
        nxt_lo = (k0 == 0).to(gl.int64) * 2
    else:
        a_tile = a_ptr + pid_m.to(gl.int64) * BLOCK_M * stride_am + k0.to(gl.int64) * BLOCK_K * stride_ak
        b_tile = b_ptr + pid_n.to(gl.int64) * BLOCK_N * stride_bn + k0.to(gl.int64) * BLOCK_K * stride_bk
        a_top, b_left = load_first_pair(a_tile, b_tile, smem, aop, bop, dotOpLayoutA, dotOpLayoutB)

    accs = init_acc(bias_ptr, pid_n, BLOCK_M, BLOCK_N, mfmaLayout, False)
    for i in range(n_pairs):
        nxt = gl.minimum(i + 1, n_pairs - 1).to(gl.int64) * 2
        if EDGE_K:
            nxt = gl.maximum(nxt, nxt_lo)
        accs, a_top, b_left = k_step_pair(accs, a_top, b_left, smem, a_tile + nxt * BLOCK_K * stride_ak,
                                          b_tile + nxt * BLOCK_K * stride_bk, aop, bop, dotOpLayoutA,
                                          dotOpLayoutB)
    # The clamped prefetch is still in flight, and the next segment's prologue overwrites the buffers other
    # waves may still be reading.
    drain_and_barrier()

    gStoreLayoutC: gl.constexpr = store_layout(BLOCK_N // 2)
    out_dtype: gl.constexpr = c_ptr.dtype.element_ty
    offs_cm = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gStoreLayoutC))
    c_tl_base = c_ptr + pid_m.to(gl.int64) * BLOCK_M * stride_cm
    col = pid_n * BLOCK_N + offs_cn
    if SEG_LOOP:
        c_offsets = stride_cm * gl.minimum(offs_cm, rows_left - 1)[:, None] + stride_cn * col[None, :]
    else:
        c_offsets = stride_cm * offs_cm[:, None] + stride_cn * col[None, :]
    c_tr_base = c_tl_base + BLOCK_N * stride_cn // 2
    c_bl_base = c_tl_base + BLOCK_M * stride_cm // 2
    c_br_base = c_bl_base + BLOCK_N * stride_cn // 2
    c_bases = (c_tl_base, c_bl_base, c_tr_base, c_br_base)

    if REDUCE_SCATTER:
        rs_ending(accs, spid, j, n_tile, r_tile, pid_n, bias_ptr, p_ptr, sync_ptr, epoch, c_bases, c_offsets,
                  stride_cm, stride_cn, rows_left, cols_left, BLOCK_M, BLOCK_N, ADD_BIAS, RS_NB, RS_U, L2,
                  NUM_PROGRAMS, out_dtype, gStoreLayoutC, mfmaLayout, EDGE_M, EDGE_N)
    else:
        owner_ending(accs, is_partial, n_peers, spid, pid_n, bias_ptr, p_ptr, locks_ptr, c_bases, c_offsets,
                     rows_left, cols_left, BLOCK_N, PEER_STEP, ADD_BIAS, out_dtype, gStoreLayoutC, mfmaLayout,
                     EDGE_M, EDGE_N, OWNER_QPB, OWNER_MAX_PEERS)


@gluon.jit
def persistent_tile(
    vpid, a_top, b_left, cur, a_ptr, b_ptr, c_ptr, bias_ptr, smem, kofs,
    num_tiles_m, num_tiles_n, total_tiles, strides, M, N,
    K: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr, NUM_XCDS: gl.constexpr, GROUP_SIZE_M: gl.constexpr, ADD_BIAS: gl.constexpr,
    mfmaLayout: gl.constexpr, gLoadLayoutA: gl.constexpr, gLoadLayoutB: gl.constexpr,
    EDGE_M: gl.constexpr, EDGE_N: gl.constexpr, EDGE_K: gl.constexpr, K_RES: gl.constexpr,
):
    """One full-K tile of the persistent loop, for virtual pid vpid.

    Expects the tile's first pair in flight in buffers 0 and 1 and its first operands in a_top / b_left;
    prefetches the same for vpid + NUM_PROGRAMS (clamped to the last tile, so every trip issues the same
    instructions and wait counts) and returns that tile's a_top / b_left and load offsets. cur is this
    tile's clamped load offsets (a_top, a_bot, b_top, b_bot), used under EDGE_M / EDGE_N.

    Under EDGE_K (K is K_PAD here) the pair in flight is the tile's last, partial pair, so the main loop
    starts at column 0, and the next tile's prefetch loads that tile's last pair, masked."""
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn = strides
    a_offsets, a_offsets_next, a_half, b_offsets, b_offsets_next, b_half = kofs
    a_cur_top, a_cur_bot, b_cur_top, b_cur_bot = cur
    smemA_top, smemA_bot, smemB_left, smemB_right = smem
    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfmaLayout, k_width=8)
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfmaLayout, k_width=8)
    iterMax = gl.cdiv(K, BLOCK_K)

    tile_id = xcd_remap_tiles(vpid, total_tiles, NUM_XCDS)
    pid_m, pid_n = get_group_m_tile_ids(tile_id, num_tiles_m, num_tiles_n, GROUP_SIZE_M)
    next_id = xcd_remap_tiles(min(vpid + NUM_PROGRAMS, total_tiles - 1), total_tiles, NUM_XCDS)
    next_m, next_n = get_group_m_tile_ids(next_id, num_tiles_m, num_tiles_n, GROUP_SIZE_M)
    a_next = a_ptr + next_m.to(gl.int64) * BLOCK_M * stride_am
    b_next = b_ptr + next_n.to(gl.int64) * BLOCK_N * stride_bn

    # The first pair is in flight, so the first prefetch is K-step 2 (under EDGE_K, column 0).
    if EDGE_K:
        a_base = a_ptr + pid_m.to(gl.int64) * BLOCK_M * stride_am
        b_base = b_ptr + pid_n.to(gl.int64) * BLOCK_N * stride_bn
        a_next += (K - 2 * BLOCK_K) * stride_ak
        b_next += (K - 2 * BLOCK_K) * stride_bk
        pa0, pb0, pa1, pb1 = tail_pair_masks(2 * BLOCK_K - K_RES, K_RES, BLOCK_M, BLOCK_N, BLOCK_K,
                                             gLoadLayoutA, gLoadLayoutB)
    else:
        a_base = a_ptr + pid_m.to(gl.int64) * BLOCK_M * stride_am + BLOCK_K * stride_ak * 2
        b_base = b_ptr + pid_n.to(gl.int64) * BLOCK_N * stride_bn + BLOCK_K * stride_bk * 2
        pa0, pb0, pa1, pb1 = None, None, None, None
    rows_left = M - pid_m * BLOCK_M
    cols_left = N - pid_n * BLOCK_N
    aop = operand_offsets(a_cur_top, a_cur_bot, a_offsets, a_offsets_next, a_half, BLOCK_K * stride_ak, EDGE_M)
    bop = operand_offsets(b_cur_top, b_cur_bot, b_offsets, b_offsets_next, b_half, BLOCK_K * stride_bk, EDGE_N)

    # The first pair, peeled: without bias, the constant-zero accumulator becomes the MFMA's inline-0 C
    # operand, so no AGPRs are zeroed per tile (a loop-carried zero would be materialised).
    accs = init_acc_edge(bias_ptr, pid_n, cols_left, BLOCK_M, BLOCK_N, mfmaLayout, ADD_BIAS, EDGE_N)
    accs, a_top, b_left = k_step_pair(accs, a_top, b_left, smem, a_base, b_base, aop, bop, dotOpLayoutA,
                                      dotOpLayoutB)
    a_base += BLOCK_K * stride_ak * 2
    b_base += BLOCK_K * stride_bk * 2
    for k in range(2, iterMax - 2, 2):
        accs, a_top, b_left = k_step_pair(accs, a_top, b_left, smem, a_base, b_base, aop, bop, dotOpLayoutA,
                                          dotOpLayoutB)
        a_base += BLOCK_K * stride_ak * 2
        b_base += BLOCK_K * stride_bk * 2
    acc_tl, acc_bl, acc_tr, acc_br = accs

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

    # The tile's row goes in the 64-bit scalar base, so the 32-bit offsets only span BLOCK_M rows. Its column
    # stays in the vector offsets, which keeps them tile-dependent: the compiler can't hoist them above the
    # tile loop and keep them live (and spilled) through the K loop. The quadrant shifts go in the scalar base
    # too, so all four stores share one offset tensor. Under EDGE_M the rows are clamped to the tile's last
    # one for the same reason (only rows the store masks drop change).
    gStoreLayoutC: gl.constexpr = store_layout(BLOCK_N // 2)
    offs_cm = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gStoreLayoutC))
    c_tl_base = c_ptr + pid_m.to(gl.int64) * BLOCK_M * stride_cm
    col = pid_n * BLOCK_N + offs_cn
    if EDGE_M:
        c_offsets = stride_cm * gl.minimum(offs_cm, rows_left - 1)[:, None] + stride_cn * col[None, :]
    else:
        c_offsets = stride_cm * offs_cm[:, None] + stride_cn * col[None, :]
    c_tr_base = c_tl_base + BLOCK_N * stride_cn // 2
    c_bl_base = c_tl_base + BLOCK_M * stride_cm // 2
    c_br_base = c_bl_base + BLOCK_N * stride_cn // 2

    # K-step iterMax - 2: the main loop's four regions, each async copy prefetching the next tile's K-step 0
    # into the buffer the main loop would refill here, so every wait keeps 5 groups in flight.
    acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    l_idx = (iterMax - 2) % 2
    a_bot = smemA_bot.index(l_idx).load(dotOpLayoutA)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(0), b_next, nb0, mask=pb0)
    gl.amd.cdna4.async_copy.commit_group()

    acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    b_right = smemB_right.index(l_idx).load(dotOpLayoutB)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_top.index(0), a_next, na0, mask=pa0)
    gl.amd.cdna4.async_copy.commit_group()

    acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    g_idx = 1 - l_idx
    b_left = smemB_left.index(g_idx).load(dotOpLayoutB)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_bot.index(0), a_next + nah, na2, mask=pa0)
    gl.amd.cdna4.async_copy.commit_group()

    acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    a_top = smemA_top.index(g_idx).load(dotOpLayoutA)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_right.index(0), b_next + nbh, nb2, mask=pb0)
    gl.amd.cdna4.async_copy.commit_group()

    # K-step iterMax - 1: prefetch the next tile's K-step 1; each C store follows its quadrant's last MFMA.
    acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    a_bot = smemA_bot.index(g_idx).load(dotOpLayoutA)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(1), b_next + nbd, nb1, mask=pb1)
    gl.amd.cdna4.async_copy.commit_group()

    acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    b_right = smemB_right.index(g_idx).load(dotOpLayoutB)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_top.index(1), a_next + nad, na1, mask=pa1)
    gl.amd.cdna4.async_copy.commit_group()

    c_tl = acc_tl.to(c_ptr.dtype.element_ty)
    c_tl = gl.convert_layout(c_tl, layout=gStoreLayoutC)
    store_edge(c_tl, c_tl_base, c_offsets, gStoreLayoutC, rows_left, cols_left, EDGE_M, EDGE_N)

    acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_bot.index(1), a_next + nah + nad, na3, mask=pa1)
    gl.amd.cdna4.async_copy.commit_group()

    c_bl = acc_bl.to(c_ptr.dtype.element_ty)
    c_bl = gl.convert_layout(c_bl, layout=gStoreLayoutC)
    store_edge(c_bl, c_bl_base, c_offsets, gStoreLayoutC, rows_left - BLOCK_M // 2, cols_left, EDGE_M, EDGE_N)

    acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")

    # The next tile's K-step 0 operands (prefetched at iterMax - 2), read while the last MFMAs and stores
    # drain; this tile's operand registers are all dead after acc_br.
    gl.amd.cdna4.async_copy.wait_group(5)
    b_left = smemB_left.index(0).load(dotOpLayoutB)
    a_top = smemA_top.index(0).load(dotOpLayoutA)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_right.index(1), b_next + nbh + nbd, nb3, mask=pb1)
    gl.amd.cdna4.async_copy.commit_group()

    c_tr = acc_tr.to(c_ptr.dtype.element_ty)
    c_tr = gl.convert_layout(c_tr, layout=gStoreLayoutC)
    store_edge(c_tr, c_tr_base, c_offsets, gStoreLayoutC, rows_left, cols_left - BLOCK_N // 2, EDGE_M, EDGE_N)

    c_br = acc_br.to(c_ptr.dtype.element_ty)
    c_br = gl.convert_layout(c_br, layout=gStoreLayoutC)
    store_edge(c_br, c_br_base, c_offsets, gStoreLayoutC, rows_left - BLOCK_M // 2, cols_left - BLOCK_N // 2,
               EDGE_M, EDGE_N)

    return a_top, b_left, (na_top, na_bot, nb_top, nb_bot)


# ---------------------------------------------------------------------------------------------------------
# Small tiles (a side of 32 or less): v9_tile_small's strip loop, W waves side by side along N, A padded to
# MP = 16 or 32 rows by masked lanes. smem is (A strips, B strips, and their swizzled read views). The
# accumulator is staged through the free B strip (lds_convert) for every layout change, and a partial is the
# tile's [MP, BLOCK_N] accumulator, row-major, 16 bytes per lane.
# ---------------------------------------------------------------------------------------------------------

@gluon.constexpr_function
def small_strip_rows(block_m):
    """A small tile's A strip: BLOCK_M rounded up to the 16-row MFMA."""
    return (block_m + 15) // 16 * 16


@gluon.constexpr_function
def small_vec_layout(rows, cols, num_warps, vec, warps_on_rows=False):
    """vec elements per lane along N, the warps side by side along N (as the MFMA layout's) or down the
    rows."""
    if warps_on_rows:
        tn = min(64, cols // vec)
        return gl.BlockedLayout([1, vec], [64 // tn, tn], [num_warps, 1], [1, 0])
    tn = max(1, min(64, cols // (vec * num_warps)))
    return gl.BlockedLayout([1, vec], [64 // tn, tn], [1, num_warps], [1, 0])


@gluon.constexpr_function
def staging_layout(rows, cols, elem_bytes):
    """Row-major [rows, cols] for lds_convert, 16-byte chunks XOR-swizzled by row so that the 16 rows a
    wave's MFMA-layout lanes write land in different banks (a padded layout cannot alias the strips'
    swizzled one)."""
    row_bytes = cols * elem_bytes
    return gl.SwizzledSharedLayout(vec=16 // elem_bytes, per_phase=max(1, 256 // row_bytes),
                                   max_phase=min(16, row_bytes // 16), order=[1, 0])


@gluon.jit
def lds_convert(x, layout: gl.constexpr, smem):
    """x in `layout`, through `smem`, a strip buffer that is free (the K loop is drained): convert_layout
    would take its own LDS scratch on top of the strips, which the persistent loop keeps live, and can push
    the LDS past 4 / W workgroups per CU."""
    ROWS: gl.constexpr = x.shape[0]
    COLS: gl.constexpr = x.shape[1]
    view = smem.reinterpret(x.dtype, [ROWS, COLS], staging_layout(ROWS, COLS, x.dtype.primitive_bitwidth // 8))
    view.store(x)
    return view.load(layout)


@gluon.jit
def small_bias(bias_ptr, pid_m, pid_n, col0, layout: gl.constexpr, ROWS: gl.constexpr, COLS: gl.constexpr,
               BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, C_T: gl.constexpr, rows_left, cols_left,
               EDGE_N: gl.constexpr):
    """The bias of a [ROWS, COLS] block of a small tile at column col0, in fp32, ready to add: along N, or
    along M under C_T, masked to the tile and the matrix."""
    if C_T:
        r = gl.arange(0, ROWS, gl.SliceLayout(1, layout))
        bias = gl.amd.cdna3.buffer_load(ptr=bias_ptr + pid_m * BLOCK_M, offsets=r,
                                        mask=(r < BLOCK_M) & (r < rows_left), other=0.0)
        out = bias.to(gl.float32)[:, None]
    else:
        c = gl.arange(0, COLS, gl.SliceLayout(0, layout))
        if EDGE_N:
            bias = gl.amd.cdna3.buffer_load(ptr=bias_ptr + pid_n * BLOCK_N + col0, offsets=c,
                                            mask=c < cols_left - col0, other=0.0)
        else:
            bias = gl.amd.cdna3.buffer_load(ptr=bias_ptr + pid_n * BLOCK_N + col0, offsets=c)
        out = bias.to(gl.float32)[None, :]
    return out


@gluon.jit
def store_small_block(v, c_ptr, pid_m, pid_n, col0, smem, stride_cm, stride_cn, BLOCK_M: gl.constexpr,
                      BLOCK_N: gl.constexpr, NUM_WARPS: gl.constexpr, C_T: gl.constexpr, rows_left, cols_left,
                      EDGE_M: gl.constexpr, EDGE_N: gl.constexpr):
    """Store a [MP, COLS] block of a small tile, at column col0, to C: converted (lds_convert) to
    v9_tile_small's store layout, 8 values per lane along C's contiguous dimension, and masked to the
    tile's BLOCK_M rows and the matrix."""
    ROWS: gl.constexpr = v.shape[0]
    COLS: gl.constexpr = v.shape[1]
    SL: gl.constexpr = small_store_layout(ROWS, COLS, NUM_WARPS, C_T)
    cv = lds_convert(v, SL, smem)
    rr = gl.arange(0, ROWS, gl.SliceLayout(1, SL))
    cc = gl.arange(0, COLS, gl.SliceLayout(0, SL))
    c_base = c_ptr + pid_m.to(gl.int64) * BLOCK_M * stride_cm + (pid_n.to(gl.int64) * BLOCK_N + col0) * stride_cn
    offs = stride_cm * rr[:, None] + stride_cn * cc[None, :]
    if EDGE_M or EDGE_N or ROWS != BLOCK_M:
        mr = rr < BLOCK_M
        if EDGE_M:
            mr = mr & (rr < rows_left)
        mc = cc < COLS
        if EDGE_N:
            mc = cc < cols_left - col0
        gl.amd.cdna3.buffer_store(ptr=c_base, offsets=offs, stored_value=cv, mask=mr[:, None] & mc[None, :])
    else:
        gl.amd.cdna3.buffer_store(ptr=c_base, offsets=offs, stored_value=cv)


@gluon.jit
def reduce_scatter_rows(p_ptr, bias_ptr, c_ptr, pid_m, pid_n, b, j, n, readers, smem, stride_cm, stride_cn,
                        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, ADD_BIAS: gl.constexpr, U: gl.constexpr,
                        NUM_WARPS: gl.constexpr, C_T: gl.constexpr, LOAD_CACHE: gl.constexpr, rows_left,
                        cols_left, EDGE_M: gl.constexpr, EDGE_N: gl.constexpr):
    """reduce_scatter_slice for a small tile: reader j of `readers` sums the 64-column blocks
    [NCB j / readers, NCB (j + 1) / readers) of the tile (NCB = BLOCK_N / 64, or one block of BLOCK_N < 64)
    over the n partials in slots P[b .. b + n - 1], U partials per trip, adds the bias and stores them."""
    MP: gl.constexpr = small_strip_rows(BLOCK_M)
    SLOT: gl.constexpr = MP * BLOCK_N
    CB: gl.constexpr = min(64, BLOCK_N)
    NCB: gl.constexpr = BLOCK_N // CB
    RL: gl.constexpr = small_vec_layout(MP, CB, NUM_WARPS, 4, True)
    r = gl.arange(0, MP, gl.SliceLayout(1, RL))[:, None]
    c = gl.arange(0, CB, gl.SliceLayout(0, RL))[None, :]
    blk_lo = (NCB * j) // readers
    blk_hi = (NCB * (j + 1)) // readers
    for g in range(blk_lo, blk_hi):
        off = r * BLOCK_N + g * CB + c
        s = gl.zeros((MP, CB), gl.float32, RL)
        for p0 in range(0, n, U):
            for u in gl.static_range(U):
                slot = p_ptr + (b + p0 + u).to(gl.int64) * SLOT
                s += gl.amd.cdna3.buffer_load(ptr=slot, offsets=off, mask=(off >= 0) & (p0 + u < n), other=0.0,
                                              cache=LOAD_CACHE).to(gl.float32)
        if ADD_BIAS:
            s = s + small_bias(bias_ptr, pid_m, pid_n, g * CB, RL, MP, CB, BLOCK_M, BLOCK_N, C_T, rows_left, cols_left,
                               EDGE_N)
        store_small_block(s.to(c_ptr.dtype.element_ty), c_ptr, pid_m, pid_n, g * CB, smem, stride_cm, stride_cn,
                          BLOCK_M, BLOCK_N, NUM_WARPS, C_T, rows_left, cols_left, EDGE_M, EDGE_N)


@gluon.jit
def small_segment(
    tile_id, k0, n_pairs, is_partial, n_peers, spid, j, n_tile, r_tile,
    a_ptr, b_ptr, c_ptr, bias_ptr, p_ptr, locks_ptr, sync_ptr, smem,
    num_tiles_m, num_tiles_n, strides, M, N, epoch,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr, GROUP_SIZE_M: gl.constexpr,
    ADD_BIAS: gl.constexpr, PEER_STEP: gl.constexpr, REDUCE_SCATTER: gl.constexpr, RS_U: gl.constexpr,
    L2: gl.constexpr, NUM_PROGRAMS: gl.constexpr, NUM_WARPS: gl.constexpr, C_T: gl.constexpr,
    EDGE_M: gl.constexpr, EDGE_N: gl.constexpr, EDGE_K: gl.constexpr, K_PAD: gl.constexpr, K_RES: gl.constexpr,
    WHOLE: gl.constexpr = False,
):
    """K-steps [k0, k0 + 2 n_pairs) of small tile tile_id (n_pairs >= 1) in the two-stage strip loop (one
    region per K-step), then the segment's ending, as quadrant_segment's:
    - WHOLE (the persistent loop's tiles): add the bias and store C;
    - REDUCE_SCATTER: a split tile publishes its partial, arrives on the tile's barrier, and its first
      r_tile programs reduce 64-column blocks (reduce_scatter_rows); a whole tile stores C;
    - otherwise the owner ending.
    Ragged M / N as v9_tile_small (masked A rows, clamped B columns); EDGE_K's pair order as
    quadrant_segment's, the k0 = 0 segment masking at K on the swizzled column each lane loads. No
    cross-tile prefetch."""
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn = strides
    smemA, smemB, smemA_rd, smemB_rd = smem
    MP: gl.constexpr = small_strip_rows(BLOCK_M)
    MASK_A: gl.constexpr = EDGE_M or MP != BLOCK_M
    gLoadLayoutA: gl.constexpr = strip_load_layout(MP, BLOCK_K, False, NUM_WARPS)
    gLoadLayoutB: gl.constexpr = strip_load_layout(BLOCK_N, BLOCK_K, True, NUM_WARPS)
    mfmaLayout: gl.constexpr = small_mfma_layout(False, NUM_WARPS)
    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfmaLayout, k_width=8)
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfmaLayout, k_width=8)
    out_dtype: gl.constexpr = c_ptr.dtype.element_ty

    pid_m, pid_n = get_group_m_tile_ids(tile_id, num_tiles_m, num_tiles_n, GROUP_SIZE_M)
    rows_left = M - pid_m * BLOCK_M
    cols_left = N - pid_n * BLOCK_N

    offs_am = gl.arange(0, MP, gl.SliceLayout(1, gLoadLayoutA))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutA))
    offs_bn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, gLoadLayoutB))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadLayoutB))
    a_k = swizzled_k(offs_ak[None, :], offs_am[:, None], BLOCK_K)
    b_k = swizzled_k(offs_bk[:, None], offs_bn[None, :], BLOCK_K)
    a_offsets = offs_am[:, None] * stride_am + a_k * stride_ak
    if EDGE_N:
        b_offsets = b_k * stride_bk + gl.minimum(offs_bn, cols_left - 1)[None, :] * stride_bn
    else:
        b_offsets = b_k * stride_bk + offs_bn[None, :] * stride_bn
    mA = None
    if MASK_A:
        a_rows = offs_am < BLOCK_M
        if EDGE_M:
            a_rows = a_rows & (offs_am < rows_left)
        mA = a_rows[:, None] & (offs_ak < BLOCK_K)[None, :]

    a_row = a_ptr + pid_m.to(gl.int64) * BLOCK_M * stride_am
    b_row = b_ptr + pid_n.to(gl.int64) * BLOCK_N * stride_bn
    if EDGE_K:
        # Pair i >= 1 of the segment at columns (k0 - 2 + 2 i) BLOCK_K; its first pair there too, or for
        # k0 = 0 the last, partial pair, masked at K (k_hi of its 2 BLOCK_K columns inside).
        first = gl.where(k0 == 0, K_PAD - 2 * BLOCK_K, (k0 - 2) * BLOCK_K).to(gl.int64)
        k_hi = gl.where(k0 == 0, 2 * BLOCK_K - K_RES, 2 * BLOCK_K)
        for s in gl.static_range(2):
            a_s = a_row + (first + s * BLOCK_K) * stride_ak
            b_s = b_row + (first + s * BLOCK_K) * stride_bk
            if s == 1 or K_RES > BLOCK_K:
                mA_s = (a_k + s * BLOCK_K < k_hi) & (offs_am >= 0)[:, None]
                if MASK_A:
                    mA_s = mA_s & mA
                gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(s), a_s, a_offsets, mask=mA_s)
                gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB.index(s), b_s, b_offsets,
                                                              mask=b_k + s * BLOCK_K < k_hi)
            else:
                gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(s), a_s, a_offsets, mask=mA)
                gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB.index(s), b_s, b_offsets)
            gl.amd.cdna4.async_copy.commit_group()
        a_base = a_row + k0.to(gl.int64) * BLOCK_K * stride_ak
        b_base = b_row + k0.to(gl.int64) * BLOCK_K * stride_bk
    else:
        a_base = a_row + k0.to(gl.int64) * BLOCK_K * stride_ak
        b_base = b_row + k0.to(gl.int64) * BLOCK_K * stride_bk
        for s in gl.static_range(2):
            gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(s), a_base, a_offsets, mask=mA)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB.index(s), b_base, b_offsets)
            gl.amd.cdna4.async_copy.commit_group()
            a_base += BLOCK_K * stride_ak
            b_base += BLOCK_K * stride_bk

    gl.amd.cdna4.async_copy.wait_group(1)
    a = smemA_rd.index(0).load(dotOpLayoutA)
    b = smemB_rd.index(0).load(dotOpLayoutB)
    acc = gl.zeros((MP, BLOCK_N), gl.float32, mfmaLayout)
    for i in range(n_pairs - 1):
        for s in gl.static_range(2):
            gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(s), a_base, a_offsets, mask=mA)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB.index(s), b_base, b_offsets)
            gl.amd.cdna4.async_copy.commit_group()

            acc = gl.amd.cdna3.mfma(a, b, acc, cd_regclass="a")

            gl.amd.cdna4.async_copy.wait_group(1)
            a = smemA_rd.index(1 - s).load(dotOpLayoutA)
            b = smemB_rd.index(1 - s).load(dotOpLayoutB)

            a_base += BLOCK_K * stride_ak
            b_base += BLOCK_K * stride_bk

    acc = gl.amd.cdna3.mfma(a, b, acc, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(0)
    a = smemA_rd.index(1).load(dotOpLayoutA)
    b = smemB_rd.index(1).load(dotOpLayoutB)
    acc = gl.amd.cdna3.mfma(a, b, acc, cd_regclass="a")
    # Every wave is done with the strips before the ending stages through them, and before the next
    # segment's prologue refills them.
    drain_and_barrier()

    PL: gl.constexpr = small_vec_layout(MP, BLOCK_N, NUM_WARPS, 4)
    SLOT: gl.constexpr = MP * BLOCK_N
    p_r = gl.arange(0, MP, gl.SliceLayout(1, PL))
    p_c = gl.arange(0, BLOCK_N, gl.SliceLayout(0, PL))
    p_offsets = p_r[:, None] * BLOCK_N + p_c[None, :]
    if WHOLE:
        if ADD_BIAS:
            acc = acc + small_bias(bias_ptr, pid_m, pid_n, 0, mfmaLayout, MP, BLOCK_N, BLOCK_M, BLOCK_N, C_T,
                                   rows_left, cols_left, EDGE_N)
        store_small_block(acc.to(out_dtype), c_ptr, pid_m, pid_n, 0, smemB, stride_cm, stride_cn, BLOCK_M, BLOCK_N,
                          NUM_WARPS, C_T, rows_left, cols_left, EDGE_M, EDGE_N)
    elif REDUCE_SCATTER:
        RS_STORE: gl.constexpr = "" if L2 else ".wt"
        RS_LOAD: gl.constexpr = ".cg" if L2 else ".cv"
        if n_tile > 1:
            x = lds_convert(acc, PL, smemB)
            gl.amd.cdna3.buffer_store(stored_value=x, ptr=p_ptr + spid.to(gl.int64) * SLOT, offsets=p_offsets,
                                      cache=RS_STORE)
            drain_and_barrier()
            is_reader = j < r_tile
            tile_barrier(sync_ptr, spid - j, n_tile, is_reader, spid, epoch, NUM_PROGRAMS, L2)
            if is_reader:
                reduce_scatter_rows(p_ptr, bias_ptr, c_ptr, pid_m, pid_n, spid - j, j, n_tile, r_tile, smemB,
                                    stride_cm, stride_cn, BLOCK_M, BLOCK_N, ADD_BIAS, RS_U, NUM_WARPS, C_T,
                                    RS_LOAD, rows_left, cols_left, EDGE_M, EDGE_N)
        else:
            if ADD_BIAS:
                acc = acc + small_bias(bias_ptr, pid_m, pid_n, 0, mfmaLayout, MP, BLOCK_N, BLOCK_M, BLOCK_N, C_T,
                                       rows_left, cols_left, EDGE_N)
            store_small_block(acc.to(out_dtype), c_ptr, pid_m, pid_n, 0, smemB, stride_cm, stride_cn, BLOCK_M,
                              BLOCK_N, NUM_WARPS, C_T, rows_left, cols_left, EDGE_M, EDGE_N)
    else:
        # atomic_poll's result needs no LDS broadcast, unlike a polled scalar atomic_add: inside the
        # end-to-end segment loop the strips stay live, and a 32x128x128 tile's fill its LDS share.
        for pj in range(n_peers):
            gl.atomic_poll(locks_ptr + spid + PEER_STEP * (1 + pj), 1, sem="relaxed", scope="gpu")
        if n_peers > 0:
            asm_fence("s_nop 0")
        x = lds_convert(acc, PL, smemB)
        if is_partial != 0:
            gl.amd.cdna3.buffer_store(stored_value=x, ptr=p_ptr + spid.to(gl.int64) * SLOT, offsets=p_offsets,
                                      cache=".wt")
        psum = gl.zeros((MP, BLOCK_N), gl.float32, PL)
        for pj in range(n_peers):
            peer = (spid + PEER_STEP * (1 + pj)).to(gl.int64)
            psum += gl.amd.cdna3.buffer_load(ptr=p_ptr + peer * SLOT, offsets=p_offsets, cache=".cv").to(gl.float32)
        if is_partial == 0:
            if ADD_BIAS:
                psum = psum + small_bias(bias_ptr, pid_m, pid_n, 0, PL, MP, BLOCK_N, BLOCK_M, BLOCK_N, C_T,
                                         rows_left, cols_left, EDGE_N)
            store_small_block((x + psum).to(out_dtype), c_ptr, pid_m, pid_n, 0, smemB, stride_cm, stride_cn,
                              BLOCK_M, BLOCK_N, NUM_WARPS, C_T, rows_left, cols_left, EDGE_M, EDGE_N)
        if is_partial != 0:
            drain_and_barrier()
            gl.atomic_xchg(locks_ptr + spid, 1, sem="relaxed", scope="gpu")
        for pj in range(n_peers):
            gl.atomic_xchg(locks_ptr + spid + PEER_STEP * (1 + pj), 0, sem="relaxed", scope="gpu")


@gluon.jit
def streamk_segment(
    tile_id, k0, n_pairs, is_partial, n_peers, spid, j, n_tile, r_tile,
    a_ptr, b_ptr, c_ptr, bias_ptr, p_ptr, locks_ptr, sync_ptr, smem, kofs,
    num_tiles_m, num_tiles_n, strides, M, N, epoch,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr, GROUP_SIZE_M: gl.constexpr,
    ADD_BIAS: gl.constexpr, PEER_STEP: gl.constexpr, REDUCE_SCATTER: gl.constexpr, RS_NB: gl.constexpr,
    RS_U: gl.constexpr, L2: gl.constexpr, NUM_PROGRAMS: gl.constexpr, mfmaLayout: gl.constexpr,
    gLoadLayoutA: gl.constexpr, gLoadLayoutB: gl.constexpr, EDGE_M: gl.constexpr, EDGE_N: gl.constexpr,
    EDGE_K: gl.constexpr, K_PAD: gl.constexpr, K_RES: gl.constexpr, SEG_LOOP: gl.constexpr,
    OWNER_QPB: gl.constexpr, OWNER_MAX_PEERS: gl.constexpr, SMALL: gl.constexpr, NUM_WARPS: gl.constexpr,
    C_T: gl.constexpr,
):
    """A stream-K segment of either body: quadrant_segment or, with SMALL, small_segment (which has no
    kernel-wide offsets, kofs, and builds its own layouts)."""
    if SMALL:
        small_segment(
            tile_id, k0, n_pairs, is_partial, n_peers, spid, j, n_tile, r_tile,
            a_ptr, b_ptr, c_ptr, bias_ptr, p_ptr, locks_ptr, sync_ptr, smem,
            num_tiles_m, num_tiles_n, strides, M, N, epoch,
            BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, ADD_BIAS, PEER_STEP, REDUCE_SCATTER, RS_U, L2, NUM_PROGRAMS,
            NUM_WARPS, C_T, EDGE_M, EDGE_N, EDGE_K, K_PAD, K_RES,
        )
    else:
        quadrant_segment(
            tile_id, k0, n_pairs, is_partial, n_peers, spid, j, n_tile, r_tile,
            a_ptr, b_ptr, c_ptr, bias_ptr, p_ptr, locks_ptr, sync_ptr, smem, kofs,
            num_tiles_m, num_tiles_n, strides, M, N, epoch,
            BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, ADD_BIAS, PEER_STEP, REDUCE_SCATTER, RS_NB, RS_U, L2,
            NUM_PROGRAMS, mfmaLayout, gLoadLayoutA, gLoadLayoutB, EDGE_M, EDGE_N, EDGE_K, K_PAD, K_RES, SEG_LOOP,
            OWNER_QPB, OWNER_MAX_PEERS,
        )


# ---------------------------------------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------------------------------------

@gluon.jit(do_not_specialize=["epoch"])
def v29_streamk_compact(
    a_ptr, b_ptr, c_ptr, bias_ptr, p_ptr, locks_ptr, sync_ptr,
    M, N, K: gl.constexpr,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    epoch,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    STREAMK_TILES: gl.constexpr,
    STREAMK_NUM_PROGRAMS: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    REDUCE_SCATTER: gl.constexpr,
    RS_NB: gl.constexpr,
    RS_U: gl.constexpr,
    RS_R_HI: gl.constexpr,
    RS_R_LO: gl.constexpr,
    END_TO_END: gl.constexpr,
    L2: gl.constexpr = False,
    XCD_SPLIT: gl.constexpr = 1,
    EDGE_M: gl.constexpr = False,
    EDGE_N: gl.constexpr = False,
    EDGE_K: gl.constexpr = False,
    HAS_FULL: gl.constexpr = True,
    SMALL: gl.constexpr = False,
    NUM_WARPS: gl.constexpr = 4,
    C_T: gl.constexpr = False,
    GRID: gl.constexpr = 0,
    OWNER_QPB: gl.constexpr = 1,
):
    """C = A B (+ bias): the persistent loop over the first total_tiles - STREAMK_TILES tiles, then the
    stream-K phase on the rest (the module docstring).

    - STREAMK_NUM_PROGRAMS of the NUM_PROGRAMS programs take part in the stream-K phase, the idle ones
      spread over the XCDs (streamk_compact_pid). With no full tiles the host may launch only those
      (GRID = STREAMK_NUM_PROGRAMS); NUM_PROGRAMS keeps the workspace layout.
    - END_TO_END: the end-to-end reversed split; otherwise the tile-aligned split, with the reduce-scatter
      ending under REDUCE_SCATTER (RS_R_HI / RS_R_LO readers on tiles with N_HI / N_LO programs, RS_NB
      blocks per group, RS_U partials per trip) and the owner ending otherwise (OWNER_QPB quadrants per
      batch). L2: the l2 exchange, on the XCD-local split of XCD_SPLIT programs per tile; epoch goes up by
      one every launch.
    - HAS_FULL = False (host, ragged shapes with no full tile): the persistent loop and its prologue are
      not compiled; their registers would cost the stream-K phase spills.
    - SMALL: the small-tile body, NUM_WARPS waves, C_T for a swapped tile."""
    start = gl.program_id(axis=0)
    num_tiles_m = gl.cdiv(M, BLOCK_M)
    num_tiles_n = gl.cdiv(N, BLOCK_N)
    total_tiles = num_tiles_m * num_tiles_n
    strides = (stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn)

    if SMALL:
        # v9_tile_small's strips: A padded to MP rows, written row by row and read through a swizzled view;
        # the LDS pads leave room for exactly 4 / NUM_WARPS workgroups per CU.
        MP: gl.constexpr = small_strip_rows(BLOCK_M)
        smemA = gl.allocate_shared_memory(a_ptr.dtype.element_ty, [2, MP, BLOCK_K],
                                          strip_shared_layout(BLOCK_K, False, False))
        smemB = gl.allocate_shared_memory(b_ptr.dtype.element_ty, [2, BLOCK_K, BLOCK_N],
                                          strip_shared_layout(BLOCK_K, True, False))
        smemA_rd = smemA.reinterpret(a_ptr.dtype.element_ty, [2, MP, BLOCK_K], strip_shared_layout(BLOCK_K, False, True))
        smemB_rd = smemB.reinterpret(b_ptr.dtype.element_ty, [2, BLOCK_K, BLOCK_N], strip_shared_layout(BLOCK_K, True, True))
        smem = (smemA, smemB, smemA_rd, smemB_rd)
        kofs = None
        mfmaLayout: gl.constexpr = small_mfma_layout(False, NUM_WARPS)
        gLoadLayoutA: gl.constexpr = strip_load_layout(MP, BLOCK_K, False, NUM_WARPS)
        gLoadLayoutB: gl.constexpr = strip_load_layout(BLOCK_N, BLOCK_K, True, NUM_WARPS)
    else:
        # v9_tile's derived layouts: Triton's padded shared layout for each MFMA operand half, and the
        # global-load layouts cut from it. Two buffers of each half.
        sharedLayoutA: gl.constexpr = tile_shared_layout(BLOCK_M // 2, BLOCK_K, False, False, a_ptr.dtype.element_ty)
        sharedLayoutB: gl.constexpr = tile_shared_layout(BLOCK_N // 2, BLOCK_K, True, False, b_ptr.dtype.element_ty)
        gLoadLayoutA: gl.constexpr = tile_load_layout(sharedLayoutA)
        gLoadLayoutB: gl.constexpr = tile_load_layout(sharedLayoutB)
        mfmaLayout: gl.constexpr = mfma_layout(False)
        smem = (
            gl.allocate_shared_memory(a_ptr.dtype.element_ty, [2, BLOCK_M // 2, BLOCK_K], sharedLayoutA),
            gl.allocate_shared_memory(a_ptr.dtype.element_ty, [2, BLOCK_M // 2, BLOCK_K], sharedLayoutA),
            gl.allocate_shared_memory(b_ptr.dtype.element_ty, [2, BLOCK_K, BLOCK_N // 2], sharedLayoutB),
            gl.allocate_shared_memory(b_ptr.dtype.element_ty, [2, BLOCK_K, BLOCK_N // 2], sharedLayoutB),
        )
        offs_am = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, gLoadLayoutA))
        offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutA))
        offs_bn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gLoadLayoutB))
        offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadLayoutB))
        # Two offset tensors per operand, even and odd K-steps. The half-tile shift is uniform, so it goes in
        # the scalar base instead of 4 more offset tensors held in VGPRs for the whole kernel.
        a_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
        b_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        a_half = BLOCK_M // 2 * stride_am
        b_half = BLOCK_N // 2 * stride_bn
        kofs = (a_offsets, a_offsets + BLOCK_K * stride_ak, a_half, b_offsets, b_offsets + BLOCK_K * stride_bk, b_half)

    # The peeled pair, the 2x-unrolled loop and the 2-step epilogue always leave buffer 0 as the one to read
    # next, which only holds for an even number of K-steps; EDGE_K rounds K up.
    K_PAD: gl.constexpr = (K + 2 * BLOCK_K - 1) // (2 * BLOCK_K) * (2 * BLOCK_K) if EDGE_K else K
    K_RES: gl.constexpr = K_PAD - K
    gl.static_assert(K_PAD % (2 * BLOCK_K) == 0, "K must be a multiple of 2 * BLOCK_K (or EDGE_K)")
    iterMax = gl.cdiv(K_PAD, BLOCK_K)
    gl.assume(iterMax > 3)

    # Phase 1: the persistent data-parallel loop.
    if SMALL:
        # Whole tiles, one segment each, stored straight to C; no cross-tile prefetch.
        total_full_tiles = total_tiles - STREAMK_TILES
        zero = start * 0
        for vpid in range(start, total_full_tiles, NUM_PROGRAMS):
            small_segment(
                xcd_remap_tiles(vpid, total_tiles, NUM_XCDS), zero, K_PAD // (2 * BLOCK_K),
                zero, zero, zero, zero, zero + 1, zero + 1,
                a_ptr, b_ptr, c_ptr, bias_ptr, p_ptr, locks_ptr, sync_ptr, smem,
                num_tiles_m, num_tiles_n, strides, M, N, epoch,
                BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, ADD_BIAS, 1, False, 1, False, NUM_PROGRAMS, NUM_WARPS, C_T,
                EDGE_M, EDGE_N, EDGE_K, K_PAD, K_RES, WHOLE=True,
            )
    elif HAS_FULL:
        a_offsets, a_offsets_next, a_half, b_offsets, b_offsets_next, b_half = kofs
        dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfmaLayout, k_width=8)
        dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfmaLayout, k_width=8)
        # The first tile's prologue; every later tile's first pair is prefetched by the previous tile's
        # epilogue. Programs with no tile still load a valid one, then skip the loop.
        first_id = xcd_remap_tiles(min(start, total_tiles - 1), total_tiles, NUM_XCDS)
        tile_m, tile_n = get_group_m_tile_ids(first_id, num_tiles_m, num_tiles_n, GROUP_SIZE_M)
        a_first = a_ptr + tile_m.to(gl.int64) * BLOCK_M * stride_am
        b_first = b_ptr + tile_n.to(gl.int64) * BLOCK_N * stride_bn
        # The persistent tile's clamped load offsets (EDGE_M / EDGE_N), carried from trip to trip.
        a_cur_top, a_cur_bot = a_offsets, a_offsets
        if EDGE_M:
            a_cur_top, a_cur_bot = clamped_tile_offsets(M - 1 - tile_m * BLOCK_M, stride_am, stride_ak, BLOCK_M // 2,
                                                        BLOCK_K, gLoadLayoutA, True)
        b_cur_top, b_cur_bot = b_offsets, b_offsets
        if EDGE_N:
            b_cur_top, b_cur_bot = clamped_tile_offsets(N - 1 - tile_n * BLOCK_N, stride_bn, stride_bk, BLOCK_N // 2,
                                                        BLOCK_K, gLoadLayoutB, False)
        aop = operand_offsets(a_cur_top, a_cur_bot, a_offsets, a_offsets_next, a_half, BLOCK_K * stride_ak, EDGE_M)
        bop = operand_offsets(b_cur_top, b_cur_bot, b_offsets, b_offsets_next, b_half, BLOCK_K * stride_bk, EDGE_N)
        if EDGE_K:
            # The first pair loaded is the tile's last, partial one.
            ma0, mb0, ma1, mb1 = tail_pair_masks(2 * BLOCK_K - K_RES, K_RES, BLOCK_M, BLOCK_N, BLOCK_K,
                                                 gLoadLayoutA, gLoadLayoutB)
            a_top, b_left = load_first_pair(a_first + (K_PAD - 2 * BLOCK_K) * stride_ak,
                                            b_first + (K_PAD - 2 * BLOCK_K) * stride_bk, smem, aop, bop,
                                            dotOpLayoutA, dotOpLayoutB, ma0, mb0, ma1, mb1)
        else:
            a_top, b_left = load_first_pair(a_first, b_first, smem, aop, bop, dotOpLayoutA, dotOpLayoutB)

        # The last trip's prefetch (clamped to total_tiles - 1) is unused: the stream-K phase drains it and
        # starts every segment with its own prologue.
        total_full_tiles = total_tiles - STREAMK_TILES
        tile_offs = (a_cur_top, a_cur_bot, b_cur_top, b_cur_bot)
        for vpid in range(start, total_full_tiles, NUM_PROGRAMS):
            a_top, b_left, tile_offs = persistent_tile(
                vpid, a_top, b_left, tile_offs, a_ptr, b_ptr, c_ptr, bias_ptr, smem, kofs,
                num_tiles_m, num_tiles_n, total_tiles, strides, M, N,
                K_PAD, BLOCK_M, BLOCK_N, BLOCK_K, NUM_PROGRAMS, NUM_XCDS, GROUP_SIZE_M, ADD_BIAS,
                mfmaLayout, gLoadLayoutA, gLoadLayoutB, EDGE_M, EDGE_N, EDGE_K, K_RES,
            )
    else:
        total_full_tiles = total_tiles - STREAMK_TILES

    gl.amd.cdna4.async_copy.wait_group(0)

    # Phase 2: stream-K. The program ids below are always the XCD-mapped ones, so a tile's consecutive
    # programs share an XCD.
    gl.static_assert(GRID == 0 or GRID == STREAMK_NUM_PROGRAMS, "a compact grid is STREAMK_NUM_PROGRAMS")
    PAIRS_PER_TILE: gl.constexpr = K_PAD // (2 * BLOCK_K)
    if END_TO_END and STREAMK_TILES > 0:
        # The ranges are laid end to end in pairs (the first R programs one pair longer) and each program runs
        # its pieces last to first: the head of its last tile (from K-step 0, its only partial), any whole
        # tile, then the tail of its first tile, which it owns. So every program starts at K-step 0, and an
        # owner's peers are the preceding programs, which ran that tile's head first. Two-tile: every program's
        # range is about one tile long and every tile has at most one peer; spread: ranges are shorter than a
        # tile, so a tile can have up to ceil(PAIRS_PER_TILE / L) programs.
        SK_PAIRS: gl.constexpr = STREAMK_TILES * PAIRS_PER_TILE
        L: gl.constexpr = SK_PAIRS // STREAMK_NUM_PROGRAMS
        R: gl.constexpr = SK_PAIRS % STREAMK_NUM_PROGRAMS
        gl.static_assert(STREAMK_NUM_PROGRAMS <= SK_PAIRS, "the host passes at most one program per pair")
        if GRID > 0:
            spid = get_logical_chiplet_mapped_pids(GRID, NUM_XCDS)
        else:
            spid = streamk_compact_pid(get_logical_chiplet_mapped_pids(NUM_PROGRAMS, NUM_XCDS), STREAMK_NUM_PROGRAMS,
                                       NUM_PROGRAMS, NUM_XCDS)
        sk_start = gl.minimum(spid * L + gl.minimum(spid, R), SK_PAIRS)
        cur = gl.minimum((spid + 1) * L + gl.minimum(spid + 1, R), SK_PAIRS)
        # Waits point only to lower spids, and a program publishes its partial before it waits.
        while cur > sk_start:
            t = (cur - 1) // PAIRS_PER_TILE
            tile_start = t * PAIRS_PER_TILE
            seg_start = gl.maximum(tile_start, sk_start)
            # Holding the tile's last pair makes this program the owner; otherwise this is its head piece.
            is_owner = (cur == tile_start + PAIRS_PER_TILE).to(gl.int32)
            n_peers = (spid - streamk_owner_of(tile_start, L, R)) * is_owner
            streamk_segment(
                streamk_tile_id(t, total_tiles, total_full_tiles, NUM_XCDS),
                (seg_start - tile_start) * 2, cur - seg_start, 1 - is_owner, n_peers, spid, 0, 0, 0,
                a_ptr, b_ptr, c_ptr, bias_ptr, p_ptr, locks_ptr, sync_ptr, smem, kofs,
                num_tiles_m, num_tiles_n, strides, M, N, epoch,
                BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, ADD_BIAS, -1, False, 1, 1, False, NUM_PROGRAMS,
                mfmaLayout, gLoadLayoutA, gLoadLayoutB, EDGE_M, EDGE_N, EDGE_K, K_PAD, K_RES, True,
                OWNER_QPB, 0, SMALL, NUM_WARPS, C_T,
            )
            cur = seg_start
    elif STREAMK_TILES > 0:
        # Tile-aligned split: tile t gets N_HI (the first EXTRA tiles) or N_LO consecutive programs, at most
        # one per pair; chunk j of n covers pairs [j q + min(j, r), + q + (j < r)), q, r = divmod(pairs, n).
        N_LO: gl.constexpr = min(STREAMK_NUM_PROGRAMS // STREAMK_TILES, PAIRS_PER_TILE)
        N_HI: gl.constexpr = min(STREAMK_NUM_PROGRAMS // STREAMK_TILES + 1, PAIRS_PER_TILE)
        EXTRA: gl.constexpr = STREAMK_NUM_PROGRAMS % STREAMK_TILES if N_HI > N_LO else 0
        SK_PROGRAMS: gl.constexpr = EXTRA * N_HI + (STREAMK_TILES - EXTRA) * N_LO
        gl.static_assert(SK_PROGRAMS == STREAMK_NUM_PROGRAMS, "the host passes the working programs")
        # The owner's peer loop at compile time, over the most peers a tile has, up to 7.
        N_MAX: gl.constexpr = N_HI if EXTRA > 0 else N_LO
        OWNER_MAX_PEERS: gl.constexpr = N_MAX - 1 if N_MAX <= 8 else 0
        # The host's reader counts (rs_split): powers of two, at most n and at most the tile's NBLK
        # reduce-scatter blocks (32x32 squares, or a small tile's 64-column blocks), and RS_NB cuts every
        # reader's slice into whole groups.
        NBLK: gl.constexpr = BLOCK_N // min(64, BLOCK_N) if SMALL else BLOCK_M * BLOCK_N // 1024
        R_MAX: gl.constexpr = RS_R_HI if EXTRA > 0 else RS_R_LO
        gl.static_assert(0 < RS_R_LO and RS_R_LO <= N_LO and (2 * RS_R_LO > N_LO or RS_R_LO == NBLK),
                         "RS_R_LO does not match N_LO")
        gl.static_assert(0 < RS_R_HI and RS_R_HI <= N_HI and (2 * RS_R_HI > N_HI or RS_R_HI == NBLK),
                         "RS_R_HI does not match N_HI")
        gl.static_assert((RS_R_LO & (RS_R_LO - 1)) == 0 and (R_MAX & (R_MAX - 1)) == 0
                         and (RS_NB == 1 or RS_NB * R_MAX <= NBLK),
                         "reduce-scatter needs power-of-two readers and whole groups")
        if L2:
            # Every tile's XCD_SPLIT programs are consecutive spids of one XCD (the host sets
            # STREAMK_NUM_PROGRAMS = S XCD_SPLIT, so N_LO = XCD_SPLIT and EXTRA = 0).
            gl.static_assert(GRID == 0, "the XCD-local split runs on the full grid")
            spid = get_logical_chiplet_mapped_pids(NUM_PROGRAMS, NUM_XCDS)
            t, j, n = xcd_local_chunk_of(spid, STREAMK_TILES, NUM_PROGRAMS // NUM_XCDS, NUM_XCDS, XCD_SPLIT)
            readers = n * 0 + RS_R_LO
            active = n > 0
        else:
            if GRID > 0:
                spid = get_logical_chiplet_mapped_pids(GRID, NUM_XCDS)
            else:
                spid = streamk_compact_pid(get_logical_chiplet_mapped_pids(NUM_PROGRAMS, NUM_XCDS),
                                           STREAMK_NUM_PROGRAMS, NUM_PROGRAMS, NUM_XCDS)
            active = spid < SK_PROGRAMS
        if active:
            if not L2:
                t, j, n, readers = streamk_chunk_of(spid, EXTRA, N_HI, N_LO, RS_R_HI, RS_R_LO)
            q = PAIRS_PER_TILE // n
            r = PAIRS_PER_TILE % n
            # Chunk 0 owns the tile; its peers are the next n - 1 slots.
            is_partial = (j != 0).to(gl.int32)
            n_peers = (n - 1) * (1 - is_partial)
            streamk_segment(
                streamk_tile_id(t, total_tiles, total_full_tiles, NUM_XCDS),
                (j * q + gl.minimum(j, r)) * 2, q + (j < r).to(gl.int32), is_partial, n_peers, spid, j, n, readers,
                a_ptr, b_ptr, c_ptr, bias_ptr, p_ptr, locks_ptr, sync_ptr, smem, kofs,
                num_tiles_m, num_tiles_n, strides, M, N, epoch,
                BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, ADD_BIAS, 1, REDUCE_SCATTER, RS_NB, RS_U, L2, NUM_PROGRAMS,
                mfmaLayout, gLoadLayoutA, gLoadLayoutB, EDGE_M, EDGE_N, EDGE_K, K_PAD, K_RES, False,
                OWNER_QPB, OWNER_MAX_PEERS, SMALL, NUM_WARPS, C_T,
            )


# ---------------------------------------------------------------------------------------------------------
# Host: workspace, rules, tiles, launch
# ---------------------------------------------------------------------------------------------------------

# Per device and tile geometry: the partial slots (one BLOCK_M x BLOCK_N fp32 tile per program, a small tile's
# MP x BLOCK_N), one owner flag per program, and tile_sync: a (count, generation) pair per program for the
# reduce-scatter barriers, indexed by a tile's first slot, then one l2 epoch flag per program. Owners lower
# the flags they consume and the last arriver resets a barrier's count, so flags and counts are zero between
# launches; generations only go up. The l2 flags hold the epoch of the last launch that wrote them.
_WORKSPACE = {}


def streamk_workspace(device, num_programs, block_m, block_n):
    key = (str(device), num_programs, block_m, block_n)
    if key not in _WORKSPACE:
        # Every partial fully overwrites its slot before its flag goes up, so P needs no init.
        P = torch.empty(num_programs * block_m * block_n, device=device, dtype=torch.float32)
        locks = torch.zeros(num_programs, device=device, dtype=torch.int32)
        tile_sync = torch.zeros(3 * num_programs, device=device, dtype=torch.int32)
        _WORKSPACE[key] = (P, locks, tile_sync, [0])
    return _WORKSPACE[key]


def next_epoch(counter):
    """The next launch's epoch: 1, 2, ..., never 0 (the l2 flags' initial value), within int32."""
    counter[0] = counter[0] % (2**31 - 2) + 1
    return counter[0]


def xcd_local_split(streamk_tiles, n_cap, pairs_per_tile, num_programs, num_xcds):
    """Programs per tile in the XCD-local split (l2): n_cap, at most one per pair, and few enough that
    ceil(S / num_xcds) tiles fit in one XCD's num_programs // num_xcds programs."""
    per_xcd_tiles = -(-streamk_tiles // num_xcds)
    return max(1, min(n_cap, pairs_per_tile, (num_programs // num_xcds) // per_xcd_tiles))


def pow2_floor(x):
    return 1 << (x.bit_length() - 1)


def rs_split(streamk_tiles, num_programs, pairs_per_tile, loads_in_flight=16, blocks=64, max_nb=16):
    """The reduce-scatter shape of the tile-aligned split, as a dict:
    - n_hi, n_lo, extra: programs per tile, as the kernel's N_HI, N_LO and EXTRA;
    - r_hi, r_lo: the readers of those tiles, the largest power of two <= n, at most `blocks` (the tile's
      reduce-scatter blocks: BM BN / 1024, or BN / 64 for a small tile);
    - nb, u: blocks per group and partials per trip of reduce_scatter_slice, nb u loads in flight: nb =
      blocks / (the largest reader count), at most loads_in_flight and max_nb (a quadrant's blocks), so
      every slice is whole groups.
    Whole tiles (n = 1) do not reduce."""
    s = max(streamk_tiles, 1)
    n_lo = min(num_programs // s, pairs_per_tile)
    n_hi = min(num_programs // s + 1, pairs_per_tile)
    extra = num_programs % s if n_hi > n_lo else 0
    r_hi, r_lo = min(pow2_floor(max(n_hi, 1)), blocks), min(pow2_floor(max(n_lo, 1)), blocks)
    r_max = r_hi if extra > 0 else r_lo
    nb = max(1, min(blocks // r_max, loads_in_flight, max_nb))
    return dict(n_hi=n_hi, n_lo=n_lo, extra=extra, r_hi=r_hi, r_lo=r_lo, nb=nb, u=max(loads_in_flight // nb, 1))


def capped_policy(total_tiles, iters_per_tile, num_programs):
    """(policy, fixup, streamk_programs) for STREAMK_POLICY=capped, the default (v20 WORKLOG entries 5 and
    7): no two-tile, and the one-tile split capped to n programs per leftover tile. streamk_programs is S n,
    or None for all programs.

    With S = total_tiles % num_programs and k = iters_per_tile:
    - S = 0: data-parallel.
    - S > num_programs / 2: spread if (num_programs - S) k >= 10493, otherwise data-parallel (the
      data-parallel last wave's idle K-steps).
    - Fewer tiles than programs:
      - (num_programs - S) k < 14 num_programs: data-parallel.
      - num_programs // S >= 4: one-tile reduce-scatter with n = min(192 // S, round(k / 4)).
      - Otherwise one-tile owner with n = round(sqrt(k / 1.5)).
      Both at most num_programs // S and k // 2; a split at that cap runs on all programs.
    - One wave or more:
      - (num_programs - S) k < 20 num_programs: data-parallel (the data-parallel last wave's idle share
        is under about 20 K-steps, less than any split's fixed cost).
      - num_programs // S >= 8: one-tile reduce-scatter with n = min(128 // S, max(8, k // 8)).
      - Otherwise one-tile owner with n = round(sqrt(k / c)), c = max(2.2, S / 5) K-steps per serial
        partial read, at most num_programs // S.
    - n = 1 runs data-parallel: one program per leftover tile is the data-parallel last wave's work plus
      stream-K's overhead."""
    s = total_tiles % num_programs
    if s == 0:
        return "dp", "rs", None
    if 2 * s > num_programs:
        return ("spread" if (num_programs - s) * iters_per_tile >= 10493 else "dp"), "rs", None
    if total_tiles < num_programs:
        if (num_programs - s) * iters_per_tile < 14 * num_programs:
            return "dp", "rs", None
        cap = min(num_programs // s, iters_per_tile // 2)
        if num_programs // s >= 4:
            fixup, n = "rs", min(cap, 192 // s, round(iters_per_tile / 4))
        else:
            fixup, n = "owner", min(cap, round(math.sqrt(iters_per_tile / 1.5)))
        n = max(1, n)
        if n == 1:
            return "dp", "rs", None
        return "one_tile", fixup, (None if n >= cap else s * n)
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


def default_fixup(total_tiles, iters_per_tile, num_programs, streamk_programs):
    """The ending for STREAMK_FIXUP=auto when STREAMK_POLICY names a method: reduce-scatter if every leftover
    tile is split at least 8 ways (its cost is about flat in n, the owner's grows with n), else owner."""
    s = total_tiles % num_programs
    if s == 0 or 2 * s > num_programs:
        return "rs"
    return "rs" if min(streamk_programs // s, iters_per_tile // 2) >= 8 else "owner"


DEFAULT_TILE = "256x256x64"
QUADRANT_SIDES = (64, 128, 256)
LAST_LAUNCH = {}  # the last launch's kernel, tile, policy and grid, for the checks
# WARMUP = True: matmul compiles the kernel (warmup) and returns without launching anything, so that the
# checks can compile cases in parallel and launch them one at a time per GPU.
WARMUP = False
_RESIDENCY = {}  # small builds whose LDS was checked: shared bytes per constexpr configuration
# auto_tile's candidates: the REFIT sweep's quadrant tiles (v27 WORKLOG §15).
AUTO_TILES = ("256x256x64", "128x256x64", "256x128x64", "128x128x128", "128x128x64", "64x256x64", "256x64x64",
              "64x128x128", "128x64x128", "64x128x64", "128x64x64", "64x64x128", "64x64x64")
AUTO_MIN_TILES = 128
AUTO_HALVE_K = 4096


def tile_spec(tile=None):
    """The tile spec in use: the argument, else STREAMK_TILE, else 256x256x64 (v9_tile's format); "auto" is
    resolved per shape by auto_tile."""
    return tile or os.environ.get("STREAMK_TILE", DEFAULT_TILE)


def auto_tile(M, N, K):
    """The tile for tile="auto" (v27 WORKLOG §15). With a wave or more of 256x256 tiles, or M < 512, 256x256x64.
    Below a wave, the largest-area quadrant tile that makes at least AUTO_MIN_TILES tiles (ties: the fewest
    tiles, then BK = 128), else 64x64x128. If that tile makes exactly 128 tiles (half a wave) and
    K <= AUTO_HALVE_K, the half-area tile that makes exactly 256 instead: with that little K stream-K cannot
    fill the idle half."""
    def count(spec):
        bm, bn = parse_spec(spec)[:2]
        return triton.cdiv(M, bm) * triton.cdiv(N, bn)

    def area(spec):
        return parse_spec(spec)[0] * parse_spec(spec)[1]

    if M < 512 or count(DEFAULT_TILE) >= 256:
        return DEFAULT_TILE
    ok = [t for t in AUTO_TILES if count(t) >= AUTO_MIN_TILES]
    if not ok:
        return "64x64x128"
    big = max(area(t) for t in ok)
    pick = min((t for t in ok if area(t) == big), key=lambda t: (count(t), parse_spec(t)[2] != 128))
    if count(pick) == 128 and K <= AUTO_HALVE_K:
        halves = [t for t in AUTO_TILES if 2 * area(t) == area(pick) and count(t) == 256]
        if halves:
            pick = min(halves, key=lambda t: parse_spec(t)[2] != 128)
    return pick


def tile_geometry(spec):
    """What the kernel runs for a tile spec, after the host's checks, as a dict: small; the kernel's block_m,
    block_n, block_k (a small tile's after the swap); w, waves per workgroup; swap; num_programs; slot, the
    (rows, columns) of a partial; blocks, the reduce-scatter blocks (BM BN / 1024 32x32 squares, or BN / 64
    column blocks of a small tile); vq, a quadrant's blocks."""
    bm, bn, bk, regions, w, pin = parse_spec(spec)
    opts = spec_options(spec)
    assert regions == 4, f"tile {spec}: r2 (v9_tile_2r) is not a stream-K tile"
    if is_small(bm, bn):
        assert not opts["mfma4"], f"tile {spec}: m4 is a v9_tile comparison, not a stream-K option"
        assert opts["waves_per_eu"] in (0, 1), f"tile {spec}: small tiles run one wave per SIMD (e1, implied)"
        km, kn, bk, w, swap = small_config(bm, bn, bk, w)
        assert kn % (16 * w) == 0, f"tile {spec}: {w} waves over {kn} columns would replicate the tile"
        mp = (km + 15) // 16 * 16
        assert mp * (kn + 4) <= bk * kn, f"tile {spec}: BK = {bk} is too small to stage the accumulator in the B strips"
        assert mp >= w * (64 // min(64, bk // 8)), (
            f"tile {spec}: the {mp}-row A strip at BK = {bk} is less than one load per wave for {w} waves; use a "
            "larger BK")
        # Every program must be resident (the endings wait on each other): one wave per SIMD and LDS for
        # 4 / W workgroups per CU.
        assert small_lds_bytes(km, kn, bk) <= LDS_PER_CU // (4 // w), (
            f"tile {spec}: {small_lds_bytes(km, kn, bk)} B of LDS, so fewer than {4 // w} workgroups of {w} waves "
            "fit in a CU; give it more waves (wW)")
        return dict(small=True, block_m=km, block_n=kn, block_k=bk, w=w, swap=swap, num_programs=256 * 4 // w,
                    slot=(mp, kn), blocks=kn // min(64, kn), vq=1)
    assert w is None and not pin, f"tile {spec}: wW and p are for tiles with a side of 32 or less"
    assert bm in QUADRANT_SIDES and bn in QUADRANT_SIDES and bk in (64, 128), (
        f"tile {spec}: four-quadrant tiles have BM, BN in {QUADRANT_SIDES} and BK in (64, 128)")
    lds = 2 * (bm + bn) * bk * 2 * (1 + 16 / 512)
    assert lds <= LDS_PER_CU, f"tile {spec}: two stages take {lds / 1024:.0f} KB of LDS, more than a CU's 160"
    return dict(small=False, block_m=bm, block_n=bn, block_k=bk, w=4, swap=False, num_programs=256, slot=(bm, bn),
                blocks=bm * bn // 1024, vq=bm * bn // 4096)


def fallback_spec(spec, geo):
    """The v9_tile spec of the data-parallel fallback: the same tile, one wave per SIMD if small (e1)."""
    if not geo["small"]:
        return spec
    bm, bn, bk, _, w, _ = parse_spec(spec)
    return f"{bm}x{bn}x{bk}" + (f"w{w}" if w else "") + "e1"


def check_residency(args, kwargs, grid, per_cu):
    """Compile a small-tile build without launching it (once per configuration) and check that its LDS leaves
    room for per_cu = 4 / W workgroups per CU: with fewer, not every program is resident and the endings'
    waits never return."""
    key = (tuple(sorted((k, str(v)) for k, v in kwargs.items())), args[9], str(args[0].dtype), args[3] is None)
    if key not in _RESIDENCY:
        k = v29_streamk_compact.warmup(*args, grid=grid, **kwargs)
        assert k.metadata.shared <= LDS_PER_CU // per_cu, (
            f"the build takes {k.metadata.shared} B of LDS: fewer than {per_cu} workgroups fit in a CU")
        _RESIDENCY[key] = k.metadata.shared
    return _RESIDENCY[key]


def matmul(a, b, c=None, bias=None, tile=None, schedule_hint="mfma-schedule"):
    """C = A B (+ bias) on `tile` (tile_spec: the argument, STREAMK_TILE, or 256x256x64; "auto": auto_tile).

    Environment:
    - STREAMK_POLICY: capped (default; capped_policy picks dp, one-tile with its ending and program count, or
      spread), or one method: dp (the persistent loop only), one_tile (the tile-aligned split), two_tile (one
      full wave moved into stream-K as well, when there is one), spread (end to end on the leftover tiles).
      When capped picks dp and the tiles fit in one wave, the call goes to v9_tile.matmul on the same tile;
      an explicit dp stays on the persistent loop.
    - STREAMK_FIXUP: the tile-aligned split's ending, auto (the rule's), rs or owner.
    - STREAMK_NUM_PROGRAMS: the programs in the stream-K phase (default: capped's, else all).
    - STREAMK_EXCHANGE: wt (default) or l2 (reduce-scatter on the tile-aligned split only; every tile's
      programs on one XCD).
    - STREAMK_FORCE_EDGE=1: the ragged-edge code on any shape, to measure its cost.
    Quadrant tiles run 256 programs of 4 waves; small tiles v9_tile_small's (KM, KN, BK, W, swap), 1024 / W
    programs of W waves at one wave per SIMD (a swapped tile runs C^T = B^T A^T). With no full tiles only the
    stream-K phase's working programs are launched (the kernel's GRID)."""
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.stride(1) == 1, "Matrix A must be row-major (its row stride may be padded)"
    M, K = a.shape
    K, N = b.shape
    assert bias is None or (bias.shape == (N,) and bias.is_contiguous()), "bias must be contiguous (N,)"
    spec = tile_spec(tile)
    if spec == "auto":
        spec = auto_tile(M, N, K)
    geo = tile_geometry(spec)
    SMALL = geo["small"]
    BLOCK_M, BLOCK_N, BLOCK_K, W = geo["block_m"], geo["block_n"], geo["block_k"], geo["w"]
    # K runs in pairs of K-steps, at least two; other K are rounded up (EDGE_K). K must keep 16-byte rows:
    # Triton only knows a stride is aligned when it is a multiple of 16.
    assert K % 16 == 0, f"K={K} must be a multiple of 16 (pad A and B)"
    K_PAD = triton.cdiv(K, 2 * BLOCK_K) * 2 * BLOCK_K
    assert K_PAD >= 4 * BLOCK_K, f"K={K} needs at least {4 * BLOCK_K} after rounding up to {2 * BLOCK_K}"
    EDGE_K = K != K_PAD
    if c is None:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    a_in, b_in, c_in = a, b, c
    if geo["swap"]:
        a, b, c = b.T, a.T, c.T
        M, N = N, M
    if SMALL:
        assert a.stride(1) == 1 and b.stride(0) == 1, "A and B must be K-contiguous for a small tile"
    # Buffer offsets are 32-bit bytes. Tile positions, half-tile shifts and the K advance live in the 64-bit
    # scalar bases, so the per-lane offsets only span one tile (plus C's width).
    def fits(*spans, t):
        return sum(spans) * t.element_size() < 2**31 - 2

    assert fits(BLOCK_M * a.stride(0), 2 * BLOCK_K * a.stride(1), t=a), "A strides too large"
    assert fits(2 * BLOCK_K * b.stride(0), BLOCK_N * b.stride(1), t=b), "B strides too large"
    assert fits(BLOCK_M * c.stride(0), N * c.stride(1), t=c), "C strides too large"

    # Programs launched, which is also the persistent stride. The endings' waits need every stream-K program
    # resident at once, so the grid is one wave.
    NUM_PROGRAMS = geo["num_programs"]
    NUM_XCDS = 8
    env_programs = os.environ.get("STREAMK_NUM_PROGRAMS")
    STREAMK_NUM_PROGRAMS = int(env_programs or NUM_PROGRAMS)
    requested = os.environ.get("STREAMK_POLICY", "capped")
    fixup = os.environ.get("STREAMK_FIXUP", "auto")
    exchange = os.environ.get("STREAMK_EXCHANGE", "wt")
    assert requested in ("capped", "dp", "one_tile", "two_tile", "spread"), requested
    assert fixup in ("auto", "rs", "owner"), fixup
    assert exchange in ("wt", "l2"), exchange
    total_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    iters_per_tile = K_PAD // BLOCK_K
    if requested == "capped":
        policy, rule_fixup, capped_programs = capped_policy(total_tiles, iters_per_tile, NUM_PROGRAMS)
        if env_programs is None and capped_programs is not None:
            STREAMK_NUM_PROGRAMS = capped_programs
    else:
        policy = requested
        rule_fixup = default_fixup(total_tiles, iters_per_tile, NUM_PROGRAMS, STREAMK_NUM_PROGRAMS)
    fixup = rule_fixup if fixup == "auto" else fixup
    # Ragged M / N: clamped loads and masked stores, on both dimensions if either is ragged (with only one on,
    # some shapes spill; with both, none do). Small tiles mask rows and columns separately.
    force_edge = os.environ.get("STREAMK_FORCE_EDGE") == "1"
    EDGE_M = EDGE_N = force_edge or M % BLOCK_M != 0 or N % BLOCK_N != 0
    if SMALL:
        EDGE_M, EDGE_N = force_edge or M % BLOCK_M != 0, force_edge or N % BLOCK_N != 0
    if requested != "dp" and policy == "dp" and total_tiles <= NUM_PROGRAMS:
        LAST_LAUNCH.clear()
        LAST_LAUNCH.update(kernel="v9_tile", tile=fallback_spec(spec, geo))
        if WARMUP:
            return c_in
        return v9_tile.matmul(a_in, b_in, c_in, bias, tile=fallback_spec(spec, geo), schedule_hint=schedule_hint)

    STREAMK_TILES = 0 if policy == "dp" else total_tiles % NUM_PROGRAMS
    if policy == "two_tile" and STREAMK_TILES > 0 and total_tiles >= NUM_PROGRAMS:
        STREAMK_TILES += NUM_PROGRAMS
    END_TO_END = policy == "spread" or STREAMK_TILES > NUM_PROGRAMS
    assert 0 < STREAMK_NUM_PROGRAMS <= NUM_PROGRAMS
    # The tile-aligned split needs a program per tile; the end-to-end one does not.
    assert END_TO_END or STREAMK_TILES <= STREAMK_NUM_PROGRAMS
    assert STREAMK_TILES == 0 or (total_tiles - STREAMK_TILES) % NUM_PROGRAMS == 0
    # v9_tile's footprint rule for GROUP_SIZE_M; a small tile holds 4 / W programs per CU, a quadrant tile 1.
    GROUP_SIZE_M = group_size_m(BLOCK_M, BLOCK_N, triton.cdiv(M, BLOCK_M), total_tiles, NUM_XCDS,
                                tiles_per_cu=4 // W if SMALL else 1)
    PAIRS_PER_TILE = K_PAD // (2 * BLOCK_K)
    tile_aligned = STREAMK_TILES > 0 and not END_TO_END
    L2 = exchange == "l2" and tile_aligned
    XCD_SPLIT = 1
    if exchange == "l2":
        assert fixup == "rs" or STREAMK_TILES == 0, "STREAMK_EXCHANGE=l2 is for the reduce-scatter ending"
        assert not END_TO_END, "STREAMK_EXCHANGE=l2 needs the tile-aligned one-tile split"
    if L2:
        # Partials and flags stay in one XCD's L2, so every tile's programs must be on one XCD: the
        # XCD-local split gives each XCD ceil(S / 8) tiles of XCD_SPLIT programs.
        XCD_SPLIT = xcd_local_split(STREAMK_TILES, STREAMK_NUM_PROGRAMS // STREAMK_TILES, PAIRS_PER_TILE,
                                    NUM_PROGRAMS, NUM_XCDS)
        assert -(-STREAMK_TILES // NUM_XCDS) * XCD_SPLIT <= NUM_PROGRAMS // NUM_XCDS, "a tile would span XCDs"
        STREAMK_NUM_PROGRAMS = STREAMK_TILES * XCD_SPLIT
    requested_programs = STREAMK_NUM_PROGRAMS
    if STREAMK_TILES > 0:
        # At most one program per pair has work in either split, so passing the working count gives the same
        # split, and the XCD balancing then spreads the programs that work instead of leaving XCDs idle.
        STREAMK_NUM_PROGRAMS = min(STREAMK_NUM_PROGRAMS, STREAMK_TILES * PAIRS_PER_TILE)
    # With no full tiles, launch only the working programs: they are hardware ids 0 .. STREAMK_NUM_PROGRAMS - 1
    # of the full grid, with the same stream-K ids and XCDs. The XCD-local split needs the full grid's per-XCD
    # program counts.
    compact = STREAMK_TILES > 0 and STREAMK_TILES == total_tiles and not L2 and STREAMK_NUM_PROGRAMS < NUM_PROGRAMS
    GRID = STREAMK_NUM_PROGRAMS if compact else 0
    if SMALL:
        rs = rs_split(STREAMK_TILES, STREAMK_NUM_PROGRAMS, PAIRS_PER_TILE, blocks=geo["blocks"], max_nb=1)
        # reduce_scatter_rows reads U partials of a 64-column block per trip, about 16 loads of 16 bytes per
        # lane in flight.
        loads = max(1, geo["slot"][0] * min(64, BLOCK_N) * 4 // (16 * 64 * W))
        rs["u"] = max(1, 16 // loads)
    else:
        rs = rs_split(STREAMK_TILES, STREAMK_NUM_PROGRAMS, PAIRS_PER_TILE, blocks=geo["blocks"], max_nb=geo["vq"])
    P, locks, tile_sync, epochs = streamk_workspace(a.device, NUM_PROGRAMS, *geo["slot"])
    grid = (GRID or NUM_PROGRAMS, 1)
    args = (a, b, c, bias, P, locks, tile_sync, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1), next_epoch(epochs) if L2 else 0)
    kwargs = dict(
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        NUM_PROGRAMS=NUM_PROGRAMS,
        STREAMK_TILES=STREAMK_TILES,
        STREAMK_NUM_PROGRAMS=STREAMK_NUM_PROGRAMS,
        NUM_XCDS=NUM_XCDS,
        GROUP_SIZE_M=GROUP_SIZE_M,
        ADD_BIAS=bias is not None,
        REDUCE_SCATTER=fixup == "rs",
        RS_NB=rs["nb"],
        RS_U=rs["u"],
        RS_R_HI=rs["r_hi"],
        RS_R_LO=rs["r_lo"],
        END_TO_END=END_TO_END,
        L2=L2,
        XCD_SPLIT=XCD_SPLIT,
        EDGE_M=EDGE_M,
        EDGE_N=EDGE_N,
        EDGE_K=EDGE_K,
        # Without full tiles (less than one wave, all stream-K) the persistent loop is not compiled; only for
        # ragged shapes, so aligned shapes compile as before.
        HAS_FULL=total_tiles > STREAMK_TILES or not (EDGE_M or EDGE_N or EDGE_K),
        GRID=GRID,
        schedule_hint=schedule_hint,
        num_warps=W,
    )
    shared = None
    if SMALL:
        # One wave per SIMD: LLVM's register budget for one wave lets no second one in.
        kwargs.update(SMALL=True, NUM_WARPS=W, C_T=geo["swap"], waves_per_eu=1)
        shared = check_residency(args, kwargs, grid, 4 // W)
    else:
        # The owner reads a peer's partial 16 fp32 loads per wave at a time: 4 quadrants at 128x128, 2 at
        # 256x128 / 128x256, 1 at 256x256 (the kernel's default).
        qpb = min(4, 16 // max(1, BLOCK_M * BLOCK_N // 4096))
        if qpb > 1:
            kwargs["OWNER_QPB"] = qpb
    if WARMUP:
        v29_streamk_compact.warmup(*args, grid=grid, **kwargs)
        return c_in
    v29_streamk_compact[grid](*args, **kwargs)
    LAST_LAUNCH.clear()
    LAST_LAUNCH.update(kernel="v29_streamk_compact", tile=spec, small=SMALL, num_programs=NUM_PROGRAMS, shared=shared,
                       policy=policy, fixup=fixup, streamk_tiles=STREAMK_TILES, streamk_programs=STREAMK_NUM_PROGRAMS,
                       requested_programs=requested_programs, grid=grid[0], compact=compact)
    return c_in
