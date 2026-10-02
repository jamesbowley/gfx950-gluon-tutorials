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
"""v14 (a16w16, 256x256x64, 4 warps) with knobs that add stream-K's data movement to a
data-parallel GEMM, so its cost can be measured as a slope.

The kernel is v14's persistent loop for shapes whose tiles are a whole number of waves
(`STREAMK_TILES = 0`). Two things are generalised:

- A tile runs as `SEGMENTS` consecutive K segments. Each segment is v14's tile body over
  `K // SEGMENTS` k-steps starting at k-step `k_start`, and its epilogue prefetches the next
  segment's first two k-steps (the same tile, or the next tile after the last segment).
  Every segment restarts the accumulator, as a stream-K segment does.
- Every segment ends with an epilogue whose code is picked by the constexpr MODE (see MODES)
  and whose work is chosen at run time per program: store the fp32 accumulator to `n_store`
  workspace slots, add `n_peers` partials read from the workspace, store C, and publish a
  flag after the partial stores.

Partials are 4 quadrants of 128x128 fp32 in the accumulator's MFMA layout (no layout
conversion), row-major or lane-contiguous (P_LAYOUT), stored with `.wt` and read with `.cv`
by default (STORE_CACHE / READ_CACHE). Flags use a GPU-scope release (after every wave has
drained its stores) or a relaxed store; owners poll relaxed and acquire once.

Which programs do what is picked by `sel_mod` / `sel_inv` on the XCD-mapped pid `spid`
(32 consecutive spids per XCD, the stream-K pid space):
selected = (spid % sel_mod == 0) != sel_inv.
"""

import triton
import torch
from common import init_acc
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 64
NUM_PROGRAMS = 256
NUM_XCDS = 8
GROUP_SIZE_M = 4
# Workspace slots per program: store sweeps write up to MAX_REPS distinct partials.
MAX_REPS = 4

# What segment endings can do (a constexpr, so each build holds only the code it measures):
# BASE: v14 (C store only). STORE / SWITCH: C store interleaved with the last MFMAs, then
# partial stores and a flag. FIXUP: contributors store a partial and a flag; owners wait,
# add their peers' partials and store C after the last MFMA.
MODES = {"base": 0, "store": 1, "fixup": 2, "switch": 3}
MODE_BASE = gl.constexpr(0)
MODE_FIXUP = gl.constexpr(2)


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
    if TILE_ORDER_V9:
        tile_id = xcd_remap_tiles(vpid, total_tiles, NUM_XCDS)
    else:
        tile_id = vpid
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
    """One unrolled main-loop iteration (two K-steps), unchanged from v14."""
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
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_bot.index(0), a_base + a_half, a_offsets)
    gl.amd.cdna4.async_copy.commit_group()

    acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    a_top = smemA_top.index(1).load(dotOpLayoutA)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_right.index(0), b_base + b_half, b_offsets)
    gl.amd.cdna4.async_copy.commit_group()

    acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    a_bot = smemA_bot.index(1).load(dotOpLayoutA)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(1), b_base, b_offsets_next)
    gl.amd.cdna4.async_copy.commit_group()

    acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    b_right = smemB_right.index(1).load(dotOpLayoutB)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_top.index(1), a_base, a_offsets_next)
    gl.amd.cdna4.async_copy.commit_group()

    acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    b_left = smemB_left.index(0).load(dotOpLayoutB)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_bot.index(1), a_base + a_half, a_offsets_next
    )
    gl.amd.cdna4.async_copy.commit_group()

    acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    a_top = smemA_top.index(0).load(dotOpLayoutA)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_right.index(1), b_base + b_half, b_offsets_next
    )
    gl.amd.cdna4.async_copy.commit_group()

    return acc_tl, acc_bl, acc_tr, acc_br, a_top, b_left


@gluon.jit
def _one():
    return gl.zeros([1], gl.int32, gl.BlockedLayout([1], [64], [gl.num_warps()], [0]))


@gluon.jit
def drain_and_barrier():
    """Every wave waits for all its vector memory ops (stores and in-flight prefetches alike:
    vmcnt is one counter), then a workgroup barrier. A release flag store after this orders
    all four waves' partial stores before the flag."""
    gl.inline_asm_elementwise(
        "s_waitcnt vmcnt(0)\ns_barrier\nv_mov_b32 $0, $1", "=v,v,~{memory}", [_one()],
        dtype=gl.int32, is_pure=False, pack=1,
    )


@gluon.jit
def slot_quadrant(p_ptr, slot, q: gl.constexpr, QUAD: gl.constexpr):
    return p_ptr + (slot.to(gl.int64) * 4 + q) * QUAD


@gluon.jit
def store_c(acc, c_base, c_offsets, out_dtype: gl.constexpr, gStoreLayoutC: gl.constexpr):
    c = acc.to(out_dtype)
    c = gl.convert_layout(c, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_offsets, stored_value=c)


@gluon.jit
def quad_offsets(mfmaLayout: gl.constexpr, HALF_M: gl.constexpr, HALF_N: gl.constexpr,
                 P_LAYOUT: gl.constexpr):
    """Element offsets of a 128x128 quadrant in its workspace slot, in the accumulator's layout.

    P_LAYOUT 0: row-major. Each lane holds 4 consecutive columns of 16 rows, so one 16-byte
    store instruction writes 64 bytes into each of 16 rows (16 half cache lines).
    P_LAYOUT 1: lane-contiguous, [16 vectors][4 warps][64 lanes][4 fp32], so one instruction
    writes 1 KiB contiguously per wave. The address is a bit permutation of (row, col) that
    follows the MFMA layout's bases:
      registers: col 1, 2, 32, 64 and row 32, 64; lanes: row 1, 2, 4, 8 and col 4, 8;
      warps: col 16, row 16.
    """
    offs_pm = gl.arange(0, HALF_M, gl.SliceLayout(1, mfmaLayout))
    offs_pn = gl.arange(0, HALF_N, gl.SliceLayout(0, mfmaLayout))
    if P_LAYOUT == 0:
        offs = offs_pm[:, None] * HALF_N + offs_pn[None, :]
    else:
        gl.static_assert(HALF_M == 128 and HALF_N == 128, "bit map is for 128x128 quadrants")
        # The column term is c plus terms constant over each 4 columns, so axis analysis still
        # sees 4 contiguous, aligned elements (16-byte accesses). Written as c % 4 + ... or with
        # shifts and masks, the same map compiles to single-dword accesses.
        r = offs_pm
        c = offs_pn
        f = (r % 16) * 4 + (r // 16 % 2) * 512 + (r // 32 % 2) * 4096 + (r // 64) * 8192
        g = c + (c // 4 % 4) * 60 + (c // 16 % 2) * 240 + (c // 32 % 2) * 992 + (c // 64) * 1984
        offs = f[:, None] + g[None, :]
    return offs


@gluon.jit
def store_partials(
    acc,
    q: gl.constexpr,
    p_ptr,
    spid,
    seg,
    n_store,
    mfmaLayout: gl.constexpr,
    MAX_REPS: gl.constexpr,
    QUAD: gl.constexpr,
    HALF_M: gl.constexpr,
    HALF_N: gl.constexpr,
    STORE_CACHE: gl.constexpr,
    P_LAYOUT: gl.constexpr,
):
    """This quadrant into n_store (<= MAX_REPS) distinct slots of this program."""
    p_offsets = quad_offsets(mfmaLayout, HALF_M, HALF_N, P_LAYOUT)
    # Unrolled and guarded: a run-time loop around stores of the AGPR accumulator spills.
    for r in gl.static_range(MAX_REPS):
        if r < n_store:
            slot = spid * MAX_REPS + (seg + r) % MAX_REPS
            gl.amd.cdna3.buffer_store(
                stored_value=acc, ptr=slot_quadrant(p_ptr, slot, q, QUAD), offsets=p_offsets,
                cache=STORE_CACHE,
            )


@gluon.jit
def fixup_quadrant(
    acc,
    q: gl.constexpr,
    p_ptr,
    spid,
    n_store,
    n_peers,
    write_c,
    c_base,
    c_offsets,
    out_dtype: gl.constexpr,
    gStoreLayoutC: gl.constexpr,
    mfmaLayout: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    MAX_REPS: gl.constexpr,
    QUAD: gl.constexpr,
    HALF_M: gl.constexpr,
    HALF_N: gl.constexpr,
    STORE_CACHE: gl.constexpr,
    READ_CACHE: gl.constexpr,
    P_LAYOUT: gl.constexpr,
    READ_DEPTH: gl.constexpr,
):
    """Contributor: store this quadrant as the partial. Owner: add the n_peers following
    programs' partials and store C. READ_DEPTH 2 issues the next peer's loads before adding
    the current one (64 more VGPRs), so two partials' loads are in flight."""
    p_offsets = quad_offsets(mfmaLayout, HALF_M, HALF_N, P_LAYOUT)
    if n_store > 0:
        gl.amd.cdna3.buffer_store(
            stored_value=acc, ptr=slot_quadrant(p_ptr, spid * MAX_REPS, q, QUAD), offsets=p_offsets,
            cache=STORE_CACHE,
        )
    # Peers are summed in VGPRs and added unconditionally (zero without peers): a loop or a
    # branch that carries the AGPR accumulator spills.
    psum = gl.zeros((HALF_M, HALF_N), gl.float32, mfmaLayout)
    if READ_DEPTH == 1:
        for j in range(n_peers):
            peer = (spid + 1 + j) % NUM_PROGRAMS
            psum += gl.amd.cdna3.buffer_load(
                ptr=slot_quadrant(p_ptr, peer * MAX_REPS, q, QUAD), offsets=p_offsets, cache=READ_CACHE
            )
    else:
        if n_peers > 0:
            cur = gl.amd.cdna3.buffer_load(
                ptr=slot_quadrant(p_ptr, ((spid + 1) % NUM_PROGRAMS) * MAX_REPS, q, QUAD),
                offsets=p_offsets, cache=READ_CACHE,
            )
            for j in range(1, n_peers):
                peer = (spid + 1 + j) % NUM_PROGRAMS
                nxt = gl.amd.cdna3.buffer_load(
                    ptr=slot_quadrant(p_ptr, peer * MAX_REPS, q, QUAD), offsets=p_offsets, cache=READ_CACHE
                )
                psum += cur
                cur = nxt
            psum += cur
    if write_c != 0:
        store_c(acc + psum, c_base, c_offsets, out_dtype, gStoreLayoutC)


@gluon.jit
def publish(locks_ptr, spid, release):
    if release != 0:
        drain_and_barrier()
        gl.atomic_xchg(locks_ptr + spid, 1, sem="release", scope="gpu")
    else:
        gl.atomic_xchg(locks_ptr + spid, 1, sem="relaxed", scope="gpu")


@gluon.jit
def segment(
    vpid,
    k_start,
    next_vpid,
    next_k,
    a_top,
    b_left,
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    p_ptr,
    locks_ptr,
    spid,
    seg,
    n_store,
    n_peers,
    write_c,
    release,
    reset_flags,
    poll,
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
    KSEG: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    TILE_ORDER_V9: gl.constexpr,
    MAX_REPS: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    MODE: gl.constexpr,
    STORE_CACHE: gl.constexpr,
    READ_CACHE: gl.constexpr,
    P_LAYOUT: gl.constexpr,
    READ_DEPTH: gl.constexpr,
    mfmaLayout: gl.constexpr,
    dotOpLayoutA: gl.constexpr,
    dotOpLayoutB: gl.constexpr,
):
    """k-steps [k_start, k_start + KSEG / BLOCK_K) of tile `vpid`: v14's tile body.

    Expects this segment's first two k-steps in flight in buffers 0 and 1 and its first
    k-step's operands in a_top / b_left. Prefetches the same for k-step next_k of tile
    next_vpid and returns that segment's a_top / b_left.
    """
    iterMax = KSEG // BLOCK_K

    tile_id = persistent_tile_id(vpid, total_tiles, NUM_XCDS, TILE_ORDER_V9)
    pid_m, pid_n = get_group_m_tile_ids(tile_id, num_tiles_m, num_tiles_n, GROUP_SIZE_M)

    next_id = persistent_tile_id(next_vpid, total_tiles, NUM_XCDS, TILE_ORDER_V9)
    next_m, next_n = get_group_m_tile_ids(next_id, num_tiles_m, num_tiles_n, GROUP_SIZE_M)
    a_next = a_ptr + next_m.to(gl.int64) * BLOCK_M * stride_am + next_k.to(gl.int64) * BLOCK_K * stride_ak
    b_next = b_ptr + next_n.to(gl.int64) * BLOCK_N * stride_bn + next_k.to(gl.int64) * BLOCK_K * stride_bk
    has_next = None

    k2 = (k_start + 2).to(gl.int64)
    a_base = a_ptr + pid_m.to(gl.int64) * BLOCK_M * stride_am + k2 * BLOCK_K * stride_ak
    b_base = b_ptr + pid_n.to(gl.int64) * BLOCK_N * stride_bn + k2 * BLOCK_K * stride_bk

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

    for k in range(2, iterMax - 2, 2):
        acc_tl, acc_bl, acc_tr, acc_br, a_top, b_left = k_step_pair(
            acc_tl, acc_bl, acc_tr, acc_br, a_top, b_left,
            smemA_top, smemA_bot, smemB_left, smemB_right, a_base, b_base,
            a_offsets, b_offsets, a_half, b_half, a_offsets_next, b_offsets_next,
            dotOpLayoutA, dotOpLayoutB,
        )
        a_base += BLOCK_K * stride_ak * 2
        b_base += BLOCK_K * stride_bk * 2

    gStoreLayoutC: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
    QUAD: gl.constexpr = BLOCK_M // 2 * BLOCK_N // 2
    out_dtype: gl.constexpr = a_ptr.dtype.element_ty

    offs_cm = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gStoreLayoutC))
    c_tl_base = c_ptr + pid_m.to(gl.int64) * BLOCK_M * stride_cm
    col = pid_n * BLOCK_N + offs_cn
    c_offsets = stride_cm * offs_cm[:, None] + stride_cn * col[None, :]
    c_tr_base = c_tl_base + BLOCK_N * stride_cn // 2
    c_bl_base = c_tl_base + BLOCK_M * stride_cm // 2
    c_br_base = c_bl_base + BLOCK_N * stride_cn // 2

    # C goes out interleaved with the last MFMAs as in v14, except in MODE_FIXUP, where
    # peers' partials are added first. Everything else happens after the last MFMA.
    C_EARLY: gl.constexpr = MODE != MODE_FIXUP

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

    if C_EARLY and write_c != 0:
        store_c(acc_tl, c_tl_base, c_offsets, out_dtype, gStoreLayoutC)

    acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_bot.index(1), a_next + a_half, a_offsets_next, mask=has_next
    )
    gl.amd.cdna4.async_copy.commit_group()

    if C_EARLY and write_c != 0:
        store_c(acc_bl, c_bl_base, c_offsets, out_dtype, gStoreLayoutC)

    acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")

    gl.amd.cdna4.async_copy.wait_group(5)
    b_left = smemB_left.index(0).load(dotOpLayoutB)
    a_top = smemA_top.index(0).load(dotOpLayoutA)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_right.index(1), b_next + b_half, b_offsets_next, mask=has_next
    )
    gl.amd.cdna4.async_copy.commit_group()

    if C_EARLY and write_c != 0:
        store_c(acc_tr, c_tr_base, c_offsets, out_dtype, gStoreLayoutC)
        store_c(acc_br, c_br_base, c_offsets, out_dtype, gStoreLayoutC)

    if MODE == MODE_FIXUP:
        # Owner: every peer's flag before the first partial read. Poll relaxed and acquire once:
        # every GPU-scope acquire invalidates the XCD's L2 under the neighbours' K loops.
        for j in range(n_peers * poll):
            peer = (spid + 1 + j) % NUM_PROGRAMS
            while gl.atomic_add(locks_ptr + peer, 0, sem="relaxed", scope="gpu") == 0:
                pass
        if n_peers * poll > 0:
            gl.atomic_add(locks_ptr + spid, 0, sem="acquire", scope="gpu")
        fixup_quadrant(
            acc_tl, 0, p_ptr, spid, n_store, n_peers, write_c, c_tl_base, c_offsets,
            out_dtype, gStoreLayoutC, mfmaLayout, NUM_PROGRAMS, MAX_REPS, QUAD, BLOCK_M // 2, BLOCK_N // 2,
            STORE_CACHE, READ_CACHE, P_LAYOUT, READ_DEPTH,
        )
        fixup_quadrant(
            acc_bl, 1, p_ptr, spid, n_store, n_peers, write_c, c_bl_base, c_offsets,
            out_dtype, gStoreLayoutC, mfmaLayout, NUM_PROGRAMS, MAX_REPS, QUAD, BLOCK_M // 2, BLOCK_N // 2,
            STORE_CACHE, READ_CACHE, P_LAYOUT, READ_DEPTH,
        )
        fixup_quadrant(
            acc_tr, 2, p_ptr, spid, n_store, n_peers, write_c, c_tr_base, c_offsets,
            out_dtype, gStoreLayoutC, mfmaLayout, NUM_PROGRAMS, MAX_REPS, QUAD, BLOCK_M // 2, BLOCK_N // 2,
            STORE_CACHE, READ_CACHE, P_LAYOUT, READ_DEPTH,
        )
        fixup_quadrant(
            acc_br, 3, p_ptr, spid, n_store, n_peers, write_c, c_br_base, c_offsets,
            out_dtype, gStoreLayoutC, mfmaLayout, NUM_PROGRAMS, MAX_REPS, QUAD, BLOCK_M // 2, BLOCK_N // 2,
            STORE_CACHE, READ_CACHE, P_LAYOUT, READ_DEPTH,
        )
        if n_store > 0:
            publish(locks_ptr, spid, release)
        # Each partial has one reader, so the owner re-arms the flags for the next launch.
        if reset_flags != 0:
            for j in range(n_peers):
                gl.atomic_xchg(locks_ptr + (spid + 1 + j) % NUM_PROGRAMS, 0, sem="relaxed", scope="gpu")
    elif MODE != MODE_BASE:
        if n_store > 0:
            store_partials(acc_tl, 0, p_ptr, spid, seg, n_store, mfmaLayout, MAX_REPS, QUAD, BLOCK_M // 2, BLOCK_N // 2,
                           STORE_CACHE, P_LAYOUT)
            store_partials(acc_bl, 1, p_ptr, spid, seg, n_store, mfmaLayout, MAX_REPS, QUAD, BLOCK_M // 2, BLOCK_N // 2,
                           STORE_CACHE, P_LAYOUT)
            store_partials(acc_tr, 2, p_ptr, spid, seg, n_store, mfmaLayout, MAX_REPS, QUAD, BLOCK_M // 2, BLOCK_N // 2,
                           STORE_CACHE, P_LAYOUT)
            store_partials(acc_br, 3, p_ptr, spid, seg, n_store, mfmaLayout, MAX_REPS, QUAD, BLOCK_M // 2, BLOCK_N // 2,
                           STORE_CACHE, P_LAYOUT)
            publish(locks_ptr, spid, release)

    return a_top, b_left


@gluon.jit(do_not_specialize=[
    "sel_mod", "sel_inv", "fin_store_s", "fin_peers_s", "fin_c_s", "fin_store_u", "fin_peers_u",
    "fin_c_u", "sw_store_s", "sw_c_s", "release", "reset_flags", "poll",
])
def streamk_costs(
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
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    sel_mod,
    sel_inv,
    fin_store_s,
    fin_peers_s,
    fin_c_s,
    fin_store_u,
    fin_peers_u,
    fin_c_u,
    sw_store_s,
    sw_c_s,
    release,
    reset_flags,
    poll,
    SEGMENTS: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    TILE_ORDER_V9: gl.constexpr,
    MAX_REPS: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    MODE: gl.constexpr,
    STORE_CACHE: gl.constexpr,
    READ_CACHE: gl.constexpr,
    P_LAYOUT: gl.constexpr,
    READ_DEPTH: gl.constexpr,
):
    """Per-program roles (selected / unselected programs, see module docstring):

    - last segment's epilogue: store fin_store_* partials, read fin_peers_* partials,
      store C if fin_c_*;
    - every earlier segment boundary: selected programs store sw_store_s partials and C if
      sw_c_s; unselected programs just continue the pipeline.
    Flags are published with a release (release=1) or relaxed store; owners reset the flags
    they consumed if reset_flags.
    """
    if TILE_ORDER_V9:
        start = gl.program_id(axis=0)
    else:
        start = get_logical_chiplet_mapped_pids(NUM_PROGRAMS, NUM_XCDS)
    spid = get_logical_chiplet_mapped_pids(NUM_PROGRAMS, NUM_XCDS)
    sel = (((spid % sel_mod) == 0).to(gl.int32) != sel_inv).to(gl.int32)
    fin_store = fin_store_u + sel * (fin_store_s - fin_store_u)
    fin_peers = fin_peers_u + sel * (fin_peers_s - fin_peers_u)
    fin_c = fin_c_u + sel * (fin_c_s - fin_c_u)
    sw_store = sel * sw_store_s
    sw_c = sel * sw_c_s

    num_tiles_m = gl.cdiv(M, BLOCK_M)
    num_tiles_n = gl.cdiv(N, BLOCK_N)
    total_tiles = num_tiles_m * num_tiles_n

    gLoadLayoutA: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0]],
        lane_bases=[[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [64, 0]],
        warp_bases=[[1, 0], [2, 0]],
        block_bases=[],
        shape=[BLOCK_M // 2, BLOCK_K],
    )
    gLoadLayoutB: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 4], [0, 8]],
        lane_bases=[[8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 64]],
        warp_bases=[[0, 1], [0, 2]],
        block_bases=[],
        shape=[BLOCK_K, BLOCK_N // 2],
    )
    sharedLayoutA: gl.constexpr = gl.PaddedSharedLayout(
        [[512, 16]],
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [64, 0],
         [1, 0], [2, 0], [4, 0], [8, 0]],
        [],
        [BLOCK_M // 2, BLOCK_K],
    )
    sharedLayoutB: gl.constexpr = gl.PaddedSharedLayout(
        [[512, 16]],
        [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 64],
         [0, 1], [0, 2], [0, 4], [0, 8]],
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

    KSEG: gl.constexpr = K // SEGMENTS
    SEG_STEPS: gl.constexpr = KSEG // BLOCK_K
    gl.static_assert(KSEG % (2 * BLOCK_K) == 0, "K // SEGMENTS must be a multiple of 2 * BLOCK_K")
    gl.static_assert(SEG_STEPS > 3, "segments need at least 4 k-steps")

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

    gl.amd.cdna4.async_copy.wait_group(6)
    b_left = smemB_left.index(0).load(dotOpLayoutB)
    a_top = smemA_top.index(0).load(dotOpLayoutA)

    for vpid in range(start, total_tiles, NUM_PROGRAMS):
        next_tile = min(vpid + NUM_PROGRAMS, total_tiles - 1)
        for seg in range(SEGMENTS):
            last = (seg == SEGMENTS - 1).to(gl.int32)
            next_vpid = vpid + last * (next_tile - vpid)
            next_k = (1 - last) * (seg + 1) * SEG_STEPS
            n_store = sw_store + last * (fin_store - sw_store)
            write_c = sw_c + last * (fin_c - sw_c)
            n_peers = last * fin_peers
            a_top, b_left = segment(
                vpid, seg * SEG_STEPS, next_vpid, next_k, a_top, b_left,
                a_ptr, b_ptr, c_ptr, bias_ptr, p_ptr, locks_ptr,
                spid, seg, n_store, n_peers, write_c, release, reset_flags, poll,
                smemA_top, smemA_bot, smemB_left, smemB_right,
                a_offsets, b_offsets, a_half, b_half, a_offsets_next, b_offsets_next,
                num_tiles_m, num_tiles_n, total_tiles,
                stride_ak, stride_bk, stride_am, stride_bn, stride_cm, stride_cn,
                KSEG, BLOCK_M, BLOCK_N, BLOCK_K, NUM_PROGRAMS, NUM_XCDS, GROUP_SIZE_M,
                TILE_ORDER_V9, MAX_REPS, ADD_BIAS, MODE, STORE_CACHE, READ_CACHE, P_LAYOUT, READ_DEPTH,
                mfmaLayout, dotOpLayoutA, dotOpLayoutB,
            )

    gl.amd.cdna4.async_copy.wait_group(0)


def make_workspace(device="cuda"):
    """fp32 partial slots (NUM_PROGRAMS x MAX_REPS x BLOCK_M x BLOCK_N) and one flag per program."""
    P = torch.zeros(NUM_PROGRAMS * MAX_REPS * BLOCK_M * BLOCK_N, device=device, dtype=torch.float32)
    locks = torch.zeros(NUM_PROGRAMS, device=device, dtype=torch.int32)
    return P, locks


def launcher(a, b, c, P, locks, mode="base", segments=1, sel_mod=1, sel_inv=0,
             fin=(0, 0, 1), fin_u=(0, 0, 1), sw=(0, 0), release=1, reset_flags=0, poll=1,
             store_cache=".wt", read_cache=".cv", p_layout=0, read_depth=1):
    """Returns a zero-argument launch of one configuration. fin / fin_u are
    (n_store, n_peers, write_c) for the last segment of selected / unselected programs; sw is
    (n_store, write_c) at earlier segment boundaries of selected programs."""
    M, K = a.shape
    _, N = b.shape
    total_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    assert total_tiles % NUM_PROGRAMS == 0, "only whole waves (STREAMK_TILES = 0)"
    assert K % (2 * BLOCK_K * segments) == 0
    args = (
        a, b, c, None, P, locks, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        sel_mod, sel_inv, *fin, *fin_u, *sw, release, reset_flags, poll,
    )
    kwargs = dict(
        SEGMENTS=segments, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        NUM_PROGRAMS=NUM_PROGRAMS, NUM_XCDS=NUM_XCDS, GROUP_SIZE_M=GROUP_SIZE_M,
        TILE_ORDER_V9=True, MAX_REPS=MAX_REPS, ADD_BIAS=False, MODE=MODES[mode],
        STORE_CACHE=store_cache, READ_CACHE=read_cache, P_LAYOUT=p_layout, READ_DEPTH=read_depth,
        num_warps=4,
    )
    grid = (NUM_PROGRAMS, 1)

    def launch():
        return streamk_costs[grid](*args, **kwargs)

    return launch


def tile_of_spid(M, N):
    """(pid_m, pid_n) of the single tile each spid computes, for one-wave shapes in v9 order."""
    tm, tn = M // BLOCK_M, N // BLOCK_N
    total = tm * tn
    assert total == NUM_PROGRAMS
    out = {}
    per = NUM_PROGRAMS // NUM_XCDS
    for p in range(NUM_PROGRAMS):
        spid = (p % NUM_XCDS) * per + p // NUM_XCDS
        tile = (p % NUM_XCDS) * (total // NUM_XCDS) + p // NUM_XCDS
        group = tile // (GROUP_SIZE_M * tn)
        first = group * GROUP_SIZE_M
        gsz = min(tm - first, GROUP_SIZE_M)
        out[spid] = (first + (tile % (GROUP_SIZE_M * tn)) % gsz, (tile % (GROUP_SIZE_M * tn)) // gsz)
    return out
