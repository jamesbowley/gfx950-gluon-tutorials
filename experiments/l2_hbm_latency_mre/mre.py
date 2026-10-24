#!/usr/bin/env python3
"""L2 vs HBM load round trips on MI355X, from ATT and in-kernel clocks.

    python mre.py --gpu 0 gemm [--att] [--pmc]     GEMM tile pattern, 256 workgroups
    python mre.py --gpu 0 mix --check               validate the hit-fraction burst (one CU)
    python mre.py --gpu 0 mix --sweep               hit fraction x scope, plus the split test
    python mre.py --gpu 0 race --check              solo control: HBM pass, then L2 pass
    python mre.py --gpu 0 race --sweep              second request for a line still in flight

--gpu selects the physical GPU: it is written to HIP_VISIBLE_DEVICES before torch is imported.
Loads in gemm and mix are buffer_load_dwordx4 ... lds (1 KB per wave), issued without waiting.
Once a CU holds the TCP's 32 plus the queue's ~12, the next one cannot leave the wave until a
slot frees, so depart(n) - depart(n - 44) over the CU is the round trip of load n - 44 (D_44).
"""

import argparse
import collections
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(HERE, "results")
TMP = os.path.join(HERE, "tmp")
SHIM = os.path.join(REPO, "scripts", "triton_deepbind_shim")

GRID = 256
NUM_XCDS = 8
HOT_CHUNKS = 128  # 512 KB per XCD: past the 32 KB L1, small enough that cold traffic leaves it in L2
CHUNK_ELEMS = 2048  # fp16 elements in one workgroup-wide load: 4 waves x 1 KB
GROUP = 16  # loads per unrolled group in the mix burst
ATT_BUFFER = "0x60000000"
COUNTERS = ["TCC_HIT_sum", "TCC_MISS_sum", "TCC_EA0_RDREQ_sum", "TCC_TAG_STALL_sum"]


def _gpu(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_known_args(argv)[0].gpu


os.environ["HIP_VISIBLE_DEVICES"] = str(_gpu(sys.argv[1:]))
os.environ.setdefault("ROCPROF_ATT_LIBRARY_PATH", "/opt/rocm/lib/")
os.environ["PYTHONPATH"] = SHIM + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
sys.path.insert(0, SHIM)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from triton.experimental import gluon  # noqa: E402
from triton.experimental.gluon import language as gl  # noqa: E402
from triton.experimental.gluon.language.amd.cdna4 import async_copy, buffer_load  # noqa: E402

# ---------------------------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------------------------


@gluon.jit
def _one(NW: gl.constexpr):
    return gl.zeros([1], gl.int32, gl.BlockedLayout([1], [64], [NW], [0]))


@gluon.jit
def _raw_reg(NW: gl.constexpr, REG: gl.constexpr):
    # Not pure: every wave reads its own register (the SIMD field differs per wave).
    return gl.inline_asm_elementwise("s_getreg_b32 $0, hwreg(" + REG + ")", "=s,v", [_one(NW)], dtype=gl.int32,
                                     is_pure=False, pack=1)


@gluon.jit
def _reg(NW: gl.constexpr, REG: gl.constexpr):
    return gl.max(_raw_reg(NW, REG), axis=0)


@gluon.jit
def _realtime(NW: gl.constexpr):
    return gl.max(gl.inline_asm_elementwise("s_memrealtime $0\ns_waitcnt lgkmcnt(0)", "=s,v,~{memory}", [_one(NW)],
                                            dtype=gl.int64, is_pure=False, pack=1), axis=0)


@gluon.jit
def _memtime(NW: gl.constexpr):
    return gl.max(gl.inline_asm_elementwise("s_memtime $0\ns_waitcnt lgkmcnt(0)", "=s,v,~{memory}", [_one(NW)],
                                            dtype=gl.int64, is_pure=False, pack=1), axis=0)


@gluon.jit
def _after_load(NW: gl.constexpr):
    # The memory clobber keeps the load above; vmcnt(0) makes the clocks the load's return.
    t, r = gl.inline_asm_elementwise("s_waitcnt vmcnt(0)\ns_memtime $0\ns_memrealtime $1\ns_waitcnt lgkmcnt(0)",
                                     "=s,=s,v,~{memory}", [_one(NW)], dtype=(gl.int64, gl.int64), is_pure=False,
                                     pack=1)
    return gl.max(t, axis=0), gl.max(r, axis=0)


@gluon.jit
def _nop(NW: gl.constexpr):
    gl.inline_asm_elementwise("s_nop 0", "=s,v,~{memory}", [_one(NW)], dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def _record(rec, pid, row, col, NW: gl.constexpr):
    """rec[pid*16 + 0..5] = pid, row, col, xcc, se, cu; rec[pid*16 + 8 + w] = HW_ID of wave w."""
    small: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])
    xcc = _reg(NW, "HW_REG_XCC_ID") & 15
    hw = _reg(NW, "HW_REG_HW_ID")
    lane = gl.arange(0, 64 * NW, small)
    v = gl.where(lane == 0, pid, gl.where(lane == 1, row, gl.where(lane == 2, col, gl.where(
        lane == 3, xcc, gl.where(lane == 4, (hw >> 13) & 7, (hw >> 8) & 15)))))
    gl.store(rec + pid * 16 + lane, v, mask=lane < 6)
    gl.store(rec + pid * 16 + 8 + lane // 64, _raw_reg(NW, "HW_REG_HW_ID"), mask=(lane % 64) == 0)
    return xcc, (hw >> 13) & 7, (hw >> 8) & 15


@gluon.jit
def _lds_pad(ONE_PER_CU: gl.constexpr, NW: gl.constexpr):
    """96 KB of LDS (one workgroup per CU) or 65 KB (two). Shapes must be powers of two."""
    small: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])
    sl: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    pa = gl.allocate_shared_memory(gl.int8, [65536], sl)
    pb = gl.allocate_shared_memory(gl.int8, [32768 if ONE_PER_CU else 1024], sl)
    pa.slice(0, 64 * NW).store(gl.zeros([64 * NW], gl.int8, small))
    pb.slice(0, 64 * NW).store(gl.zeros([64 * NW], gl.int8, small))
    return pa, pb


@gluon.jit
def _pad_sink(pad, NW: gl.constexpr):
    small: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])
    pa, pb = pad
    return gl.sum(pa.slice(0, 64 * NW).load(small).to(gl.int32), axis=0) + \
        gl.sum(pb.slice(0, 64 * NW).load(small).to(gl.int32), axis=0)


# ---------------------------------------------------------------------------------------------
# GEMM tile pattern: 16x16 tiles of 256x256 fp16, BK = 64. Per K step each workgroup issues its
# 32 KB A panel and 32 KB B panel (16 workgroup-wide loads of 4 KB) and never waits in the loop.
# ---------------------------------------------------------------------------------------------


@gluon.jit
def gemm_kernel(a_ptr, b_ptr, out, rec, n_k, stride_am, stride_bk, block_4x8):
    NW: gl.constexpr = gl.num_warps()
    a_lay: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])
    b_lay: gl.constexpr = gl.BlockedLayout([1, 8], [2, 32], [4, 1], [1, 0])
    smem_a = gl.allocate_shared_memory(gl.float16, [32, 64], gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0]))
    smem_b = gl.allocate_shared_memory(gl.float16, [8, 256], gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0]))
    pad = _lds_pad(True, NW)
    pid = gl.program_id(0)
    xcd = pid % 8
    local = pid // 8
    # 2x16: each XCD takes 32 consecutive row-major tiles. 4x8: a 4-row x 8-column block per XCD.
    if block_4x8:
        row = (xcd // 2) * 4 + (local % 4)
        col = (xcd % 2) * 8 + (local // 4)
    else:
        row = (xcd * 32 + local) // 16
        col = (xcd * 32 + local) % 16
    _record(rec, pid, row, col, NW)

    a_off = gl.arange(0, 32, gl.SliceLayout(1, a_lay))[:, None] * stride_am + \
        gl.arange(0, 64, gl.SliceLayout(0, a_lay))[None, :]
    b_off = gl.arange(0, 8, gl.SliceLayout(1, b_lay))[:, None] * stride_bk + \
        gl.arange(0, 256, gl.SliceLayout(0, b_lay))[None, :]
    a_tile = a_ptr + row.to(gl.int64) * 256 * stride_am
    b_tile = b_ptr + col.to(gl.int64) * 256
    for ks in range(n_k):
        ak = a_tile + ks.to(gl.int64) * 64
        bk = b_tile + ks.to(gl.int64) * 64 * stride_bk
        for i in gl.static_range(8):
            async_copy.buffer_load_to_shared(smem_a, ak + i * 32 * stride_am, a_off)
        _nop(NW)
        for i in gl.static_range(8):
            async_copy.buffer_load_to_shared(smem_b, bk + i * 8 * stride_bk, b_off)
        _nop(NW)
    async_copy.commit_group()
    async_copy.wait_group(0)
    a = async_copy.load_shared_relaxed(smem_a, a_lay).to(gl.float32)
    b = async_copy.load_shared_relaxed(smem_b, b_lay).to(gl.float32)
    acc = gl.sum(gl.sum(a, axis=1), axis=0) + gl.sum(gl.sum(b, axis=1), axis=0)
    gl.store(out + pid, acc + _pad_sink(pad, NW).to(gl.float32))


# ---------------------------------------------------------------------------------------------
# Hit-fraction burst. Prime HOT chunks per XCD (in L2 after this), wait, then n_groups x 16 loads
# per workgroup with no wait. Interleave: load j of each group is hot when j % 4 < h4. Split:
# waves 0 and 1 (rows 0-63 of the tile) are hot, waves 2 and 3 cold. Cold loads read an address
# nothing else touches (HBM after the host flush). The address is chosen with a select, not a
# branch: a branch around each load makes the compiler put an s_barrier in front of it.
# scope 0: every CU. 1: only XCD ox. 2: only the CU (ox, os_, oc).
# ---------------------------------------------------------------------------------------------


NUM_XCDS_C = gl.constexpr(NUM_XCDS)


@gluon.jit
def _vmcnt0(NW: gl.constexpr):
    # A plain wait: Triton's async_wait here would make it put an s_barrier in the burst loop.
    gl.inline_asm_elementwise("s_waitcnt vmcnt(0)", "=s,v,~{memory}", [_one(NW)], dtype=gl.int32, is_pure=False,
                              pack=1)


@gluon.jit
def mix_kernel(buf, out, rec, n_groups, scope, ox, os_, oc, cold_per_wg, H4: gl.constexpr, SPLIT: gl.constexpr,
               HOT: gl.constexpr):
    NW: gl.constexpr = gl.num_warps()
    lay: gl.constexpr = gl.BlockedLayout([1, 8], [32, 2], [4, 1], [1, 0])
    sh: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    smem_p = gl.allocate_shared_memory(gl.float16, [128, 16], sh)
    smem = gl.allocate_shared_memory(gl.float16, [128, 16], sh)
    pad = _lds_pad(True, NW)
    pid = gl.program_id(0)
    xcc, se, cu = _record(rec, pid, 0, 0, NW)
    on = pid >= 0
    if scope == 1:
        on = xcc == ox
    if scope == 2:
        on = (xcc == ox) & (se == os_) & (cu == oc)
    rows = gl.arange(0, 128, gl.SliceLayout(1, lay))[:, None]
    local = rows * 16 + gl.arange(0, 16, gl.SliceLayout(0, lay))[None, :]
    hot_ptr = buf + xcc * (HOT * 2048)
    cold_ptr = buf + NUM_XCDS_C * HOT * 2048 + pid * cold_per_wg * 2048
    acc = gl.zeros([128, 16], gl.float32, lay)
    if on:
        # Each wave primes its own 1 KB of every hot chunk and waits for it: those lines are in L2.
        for c in range(HOT):
            async_copy.buffer_load_to_shared(smem_p, hot_ptr + c * 2048, local)
        _vmcnt0(NW)
        hg = (cu * 7) & (HOT - 1)
        for g in range(n_groups):
            for j in gl.static_range(16):
                hot = hot_ptr + ((hg + j * 17) & (HOT - 1)) * 2048
                cold = cold_ptr + (g * 16 + j) * 2048
                if SPLIT:
                    async_copy.buffer_load_to_shared(smem, buf, local + gl.where(
                        rows < 64, ((hg + j * 17) & (HOT - 1)) * 2048 + xcc * (HOT * 2048),
                        NUM_XCDS_C * HOT * 2048 + (pid * cold_per_wg + g * 16 + j) * 2048))
                elif (j % 4) < H4:
                    async_copy.buffer_load_to_shared(smem, hot, local)
                else:
                    async_copy.buffer_load_to_shared(smem, cold, local)
            hg = (hg + 16 * 17) & (HOT - 1)
            _nop(NW)
        async_copy.commit_group()
        async_copy.wait_group(0)
        acc = async_copy.load_shared_relaxed(smem, lay).to(gl.float32) + \
            async_copy.load_shared_relaxed(smem_p, lay).to(gl.float32)
    gl.store(out + pid, gl.sum(gl.sum(acc, axis=1), axis=0) + _pad_sink(pad, NW).to(gl.float32))


# ---------------------------------------------------------------------------------------------
# Controlled depth. After every load, s_waitcnt vmcnt(D): a wave holds at most D + 1 loads, and the
# wait after load i ends when load i - D returns, which gives that load's round trip from issue.
# TIER 0: fast loads hit a 512 KB region per XCD primed into L2 (as mix_kernel's hot region).
# TIER 1: fast loads re-read a region primed earlier in the same kernel and since pushed out of L2 by
#         16 MB per XCD of other reads, so they come from the memory-side cache.
# Load j of each group of 16 is fast when j % 4 < H4, else a cold address (HBM).
# ---------------------------------------------------------------------------------------------


@gluon.constexpr_function
def _vm_asm(n):
    return f"s_waitcnt vmcnt({n})"


@gluon.jit
def _vmwait(N: gl.constexpr, NW: gl.constexpr):
    gl.inline_asm_elementwise(_vm_asm(N), "=s,v,~{memory}", [_one(NW)], dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def _sleep(NW: gl.constexpr):
    gl.inline_asm_elementwise("s_sleep 127\ns_mov_b32 $0, 0", "=s,v,~{memory}", [_one(NW)], dtype=gl.int32,
                              is_pure=False, pack=1)


@gluon.jit
def _mfma_until(deadline, pid, NW: gl.constexpr):
    """MFMA only, no memory traffic, until s_memrealtime passes deadline. (Polling a flag in memory instead would
    put a stream of device-scope atomics on the memory path being measured.) Hashed bf16 operands: constant ones
    toggle too few bits to draw the power that brings the shader clock down. Returns a value to keep results live."""
    mlayout: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                                  warps_per_cta=[2, NW // 2])
    la: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mlayout, k_width=8)
    lb: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mlayout, k_width=8)
    ra = gl.arange(0, 64, gl.SliceLayout(1, la))[:, None] * 64 + gl.arange(0, 64, gl.SliceLayout(0, la))[None, :]
    rb = gl.arange(0, 64, gl.SliceLayout(1, lb))[:, None] * 64 + gl.arange(0, 64, gl.SliceLayout(0, lb))[None, :]
    ma = ((((ra * 1103515245 + pid * 12345) >> 9) & 1023).to(gl.float32) * (1.0 / 512) - 1.0).to(gl.bfloat16)
    mb = ((((rb * 22695477 + pid * 7919) >> 9) & 1023).to(gl.float32) * (1.0 / 512) - 1.0).to(gl.bfloat16)
    m0 = gl.zeros([64, 64], gl.float32, mlayout)
    m1 = gl.zeros([64, 64], gl.float32, mlayout)
    m2 = gl.zeros([64, 64], gl.float32, mlayout)
    m3 = gl.zeros([64, 64], gl.float32, mlayout)
    go = 1
    it = pid * 0
    while go > 0:
        for _u in range(32):  # four independent accumulators keep the matrix core busy
            m0 = gl.amd.cdna3.mfma(ma, mb, m0)
            m1 = gl.amd.cdna3.mfma(ma, mb, m1)
            m2 = gl.amd.cdna3.mfma(ma, mb, m2)
            m3 = gl.amd.cdna3.mfma(ma, mb, m3)
        it += 1
        go = gl.where((_realtime(NW) < deadline) & (it < 2000000), 1, 0)
    return gl.sum(gl.sum(m0 + m1 + m2 + m3, axis=1), axis=0)


@gluon.jit
def depth_kernel(buf, out, rec, clk, n_groups, scope, ox, os_, oc, fast_base, fast_per_wg, cold_base, cold_per_wg,
                 co_ticks, co_n, settle_ticks,
                 H4: gl.constexpr, D: gl.constexpr, TIER: gl.constexpr, HOT: gl.constexpr, CO: gl.constexpr = 1):
    """co_n > 0 (scope 2 only): co_n of the other workgroups run MFMA-only loops while the measured CU works; the
    measured CU first sleeps settle_ticks (100 MHz) so the shader clock settles under that load."""
    NW: gl.constexpr = gl.num_warps()
    lay: gl.constexpr = gl.BlockedLayout([1, 8], [32, 2], [4, 1], [1, 0])
    sh: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    small: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])
    smem_p = gl.allocate_shared_memory(gl.float16, [128, 16], sh)
    smem = gl.allocate_shared_memory(gl.float16, [128, 16], sh)
    pad = _lds_pad(True, NW)
    pid = gl.program_id(0)
    xcc, se, cu = _record(rec, pid, 0, 0, NW)
    on = pid >= 0
    if scope == 1:
        on = xcc == ox
    if scope == 2:
        on = (xcc == ox) & (se == os_) & (cu == oc)
    rows = gl.arange(0, 128, gl.SliceLayout(1, lay))[:, None]
    local = rows * 16 + gl.arange(0, 16, gl.SliceLayout(0, lay))[None, :]
    STRIDE: gl.constexpr = 17 if TIER == 0 else 1
    if TIER == 0:
        fast_ptr = buf + xcc * (HOT * 2048)
        fg = (cu * 7) & (fast_per_wg - 1)
    else:
        region = gl.where(scope == 2, 0, pid)
        fast_ptr = buf + fast_base + region * fast_per_wg * 2048
        fg = cu * 0
    cold_ptr = buf + cold_base + pid * cold_per_wg * 2048
    acc = gl.zeros([128, 16], gl.float32, lay)
    # Co-runners (co_n > 0, scope 2) run MFMA until co_ticks after their start. CO 1: on the first co_n workgroups
    # (spread over all XCDs). CO 2: on the measured XCD's other 31 CUs only.
    if CO == 2:
        if (co_n > 0) & (xcc == ox) & (on == 0):
            gl.store(out + pid, _mfma_until(_realtime(NW) + co_ticks, pid, NW))
    else:
        if (co_n > 0) & (pid < co_n + 1) & (on == 0):
            gl.store(out + pid, _mfma_until(_realtime(NW) + co_ticks, pid, NW))
    if on:
        s0 = _realtime(NW)
        if co_n > 0:
            go = 1
            while go > 0:
                _sleep(NW)
                go = gl.where(_realtime(NW) < s0 + settle_ticks, 1, 0)
        for c in range(fast_per_wg):
            async_copy.buffer_load_to_shared(smem_p, fast_ptr + c * 2048, local)
        _vmcnt0(NW)
        for g in range(n_groups):
            for j in gl.static_range(16):
                if (j % 4) < H4:
                    async_copy.buffer_load_to_shared(smem, fast_ptr + ((fg + j * STRIDE) & (fast_per_wg - 1)) * 2048,
                                                     local)
                else:
                    async_copy.buffer_load_to_shared(smem, cold_ptr + (g * 16 + j) * 2048, local)
                if D < 63:
                    _vmwait(D, NW)
            fg = (fg + 16 * STRIDE) & (fast_per_wg - 1)
            _nop(NW)
        async_copy.commit_group()
        async_copy.wait_group(0)
        # Clock after the burst (an s_memrealtime before it makes the trace decoder lose the loads).
        c0 = _memtime(NW)
        r0 = _realtime(NW)
        go = 1
        while go > 0:
            go = gl.where(_realtime(NW) < r0 + 2000, 1, 0)
        c1 = _memtime(NW)
        r1 = _realtime(NW)
        k = gl.arange(0, 64 * NW, small).to(gl.int64)
        gl.store(clk + pid * 8 + k, gl.where(k == 0, c0, gl.where(k == 1, r0, gl.where(k == 2, c1, gl.where(
            k == 3, r1, s0)))), mask=k < 5)
        acc = async_copy.load_shared_relaxed(smem, lay).to(gl.float32) + \
            async_copy.load_shared_relaxed(smem_p, lay).to(gl.float32)
        gl.store(out + pid, gl.sum(gl.sum(acc, axis=1), axis=0) + _pad_sink(pad, NW).to(gl.float32))


# ---------------------------------------------------------------------------------------------
# Race. One wave per workgroup, 64 KB of LDS each, grid 512: two workgroups per CU. The first two
# workgroups to arrive on the requester CUs take the roles (atomic counters). Each requester reads
# line i (1 KB) at realtime t0 + i * period (+ delta for the follower), one load at a time, and
# records the cycles from just before the load to its return.
# place 0: both on CU L. 1: follower on another CU of the same XCD. 2: on another XCD. 3: solo.
# ---------------------------------------------------------------------------------------------


@gluon.jit
def clock_kernel(out):
    gl.store(out, _realtime(1))


@gluon.jit
def race_kernel(src, samples, ctr, n_lines, passes, t0, delta, period, place, lx, ls, lc, fx, fs, fc):
    NW: gl.constexpr = gl.num_warps()
    small: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])
    lay: gl.constexpr = gl.BlockedLayout([4], [64], [NW], [0])
    pad = _lds_pad(False, NW)
    xcc = _reg(NW, "HW_REG_XCC_ID") & 15
    hw = _reg(NW, "HW_REG_HW_ID")
    se = (hw >> 13) & 7
    cu = (hw >> 8) & 15
    on_l = (xcc == lx) & (se == ls) & (cu == lc)
    on_f = (xcc == fx) & (se == fs) & (cu == fc) & (place != 0) & (place != 3)
    role = xcc * 0 - 1
    if on_l:
        r = gl.atomic_add(ctr, 1)
        role = gl.where(r == 0, 0, gl.where((r == 1) & (place == 0), 1, -1))
    if on_f:
        r = gl.atomic_add(ctr + 1, 1)
        role = gl.where(r == 0, 1, -1)
    offs = gl.arange(0, 256 * NW, lay)
    kk = gl.arange(0, 64 * NW, small).to(gl.int64)
    if role >= 0:
        extra = gl.where(role == 1, delta, 0).to(gl.int64)
        hwv = hw.to(gl.int64) + xcc.to(gl.int64) * 65536 * 65536
        gl.store(samples + role * 8 + kk, hwv, mask=kk == 0)
        n = n_lines * passes
        sink = gl.zeros([256 * NW], gl.float32, lay)
        for s in range(n):
            deadline = t0 + s.to(gl.int64) * period + extra
            go = 1
            while go > 0:
                go = gl.where(_realtime(NW) < deadline, 1, 0)
            c0 = _memtime(NW)
            r0 = _realtime(NW)
            x = buffer_load(src + (s % n_lines).to(gl.int64) * 256, offs)
            c1, r1 = _after_load(NW)
            sink += x
            v = gl.where(kk == 0, c1 - c0, gl.where(kk == 1, c0, gl.where(kk == 2, r0, r1)))
            gl.store(samples + 16 + (role * n + s) * 4 + kk, v, mask=kk < 4)
        gl.store(samples + 4 + role, gl.sum(sink, axis=0).to(gl.int64))
    gl.store(ctr + 2, _pad_sink(pad, NW))


# ---------------------------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------------------------

_FLUSH = None


def flush():
    """Read 1 GiB: evicts L2 and the 256 MB memory-side cache, so the next first touch is HBM."""
    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.zeros(1 << 28, dtype=torch.float32, device="cuda")
    float(_FLUSH.sum())
    torch.cuda.synchronize()


def read_rec(rec, n):
    r = rec.view(n, 16).cpu().tolist()
    return [dict(pid=x[0], row=x[1], col=x[2], xcc=x[3], se=x[4], cu=x[5],
                 wave_simd=[(h >> 4) & 3 for h in x[8:12]]) for x in r if x[0] >= 0]


def smi():
    p = subprocess.run(["rocm-smi", "-d", "0", "--showclocks"], capture_output=True, text=True)
    return p.stdout


def save(name, obj):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name + ".json"), "w") as f:
        json.dump(obj, f, indent=1)
    print("wrote results/" + name + ".json", flush=True)


# --- worker: runs launches described by a spec, each after a flush -------------------------------


def launch_gemm(c):
    k = c["k"]
    a = torch.ones((4096, k), dtype=torch.float16, device="cuda")
    b = torch.ones((k, 4096), dtype=torch.float16, device="cuda")
    out = torch.empty(GRID, dtype=torch.float32, device="cuda")
    rec = torch.full((GRID * 16,), -1, dtype=torch.int32, device="cuda")
    blk = int(c["block"] == "4x8")
    gemm_kernel.warmup(a, b, out, rec, k // 64, a.stride(0), b.stride(0), blk, num_warps=4, grid=(GRID,))
    flush()
    if c.get("fresh"):
        # The first version of this experiment: inputs written after the flush.
        a.fill_(1)
        b.fill_(1)
        torch.cuda.synchronize()
    if c.get("hot"):
        # Hot: the same launch once before, so the 32 MB of inputs sit in the memory-side cache (and the
        # last K steps in L2). This dispatch is also traced; callers keep the last one.
        gemm_kernel[(GRID,)](a, b, out, rec, k // 64, a.stride(0), b.stride(0), blk, num_warps=4)
        torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    gemm_kernel[(GRID,)](a, b, out, rec, k // 64, a.stride(0), b.stride(0), blk, num_warps=4)
    e1.record()
    torch.cuda.synchronize()
    return dict(us=e0.elapsed_time(e1) * 1000, rec=read_rec(rec, GRID))


_MIX_BUF = {}


def launch_mix(c):
    cold_per_wg = c["groups"] * GROUP
    n = NUM_XCDS * HOT_CHUNKS * CHUNK_ELEMS + GRID * cold_per_wg * CHUNK_ELEMS
    if n not in _MIX_BUF:
        _MIX_BUF.clear()
        _MIX_BUF[n] = torch.ones(n, dtype=torch.float16, device="cuda")
    buf = _MIX_BUF[n]
    out = torch.empty(GRID, dtype=torch.float32, device="cuda")
    rec = torch.full((GRID * 16,), -1, dtype=torch.int32, device="cuda")
    args = (buf, out, rec, c["groups"], c["scope"], c["x"], c["se"], c["cu"], cold_per_wg)
    kw = dict(H4=c["h4"], SPLIT=bool(c["split"]), HOT=HOT_CHUNKS, num_warps=4)
    mix_kernel.warmup(*args, **kw, grid=(GRID,))
    flush()  # after the buffer was written: nothing of it is left in L2 or the memory-side cache
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    k = mix_kernel[(GRID,)](*args, **kw)
    e1.record()
    torch.cuda.synchronize()
    return dict(us=e0.elapsed_time(e1) * 1000, rec=read_rec(rec, GRID), asm=k.asm["amdgcn"])


LLC_CHUNKS = {2: 4096, 1: 128, 0: 128}  # per workgroup: 16 MB for one CU, 512 KB each (16 MB per XCD) otherwise
_DEPTH_BUF = {}


def depth_layout(c):
    cold_per_wg = c["groups"] * GROUP
    fast_per_wg = HOT_CHUNKS if c["tier"] == 0 else LLC_CHUNKS[c["scope"]]
    fast_base = NUM_XCDS * HOT_CHUNKS * CHUNK_ELEMS
    cold_base = fast_base + GRID * 128 * CHUNK_ELEMS  # the largest fast region: 256 x 512 KB (16 MB < this)
    return fast_per_wg, fast_base, cold_base, cold_per_wg, cold_base + GRID * cold_per_wg * CHUNK_ELEMS


def launch_depth(c):
    fast_per_wg, fast_base, cold_base, cold_per_wg, n = depth_layout(c)
    if n not in _DEPTH_BUF:
        _DEPTH_BUF.clear()
        _DEPTH_BUF[n] = torch.ones(n, dtype=torch.float16, device="cuda")
    buf = _DEPTH_BUF[n]
    out = torch.empty(GRID, dtype=torch.float32, device="cuda")
    rec = torch.full((GRID * 16,), -1, dtype=torch.int32, device="cuda")
    clk = torch.zeros(GRID * 8, dtype=torch.int64, device="cuda")
    settle = int(c.get("settle_ms", 0) * 1e5)
    co_ticks = settle + int(c.get("co_extra_ms", 30) * 1e5)  # co-runners outlast the settle time plus the burst
    args = (buf, out, rec, clk, c["groups"], c["scope"], c["x"], c["se"], c["cu"], fast_base, fast_per_wg, cold_base,
            cold_per_wg, co_ticks, c.get("co", 0), settle)
    kw = dict(H4=c["h4"], D=c["d"], TIER=c["tier"], HOT=HOT_CHUNKS, CO=c.get("co_mode", 1), num_warps=4)
    depth_kernel.warmup(*args, **kw, grid=(GRID,))
    flush()
    k = depth_kernel[(GRID,)](*args, **kw)
    torch.cuda.synchronize()
    ck = clk.view(GRID, 8).cpu().tolist()
    ghz = [(x[2] - x[0]) / (x[3] - x[1]) / 10.0 for x in ck if x[3] > x[1]]
    # The measured CU's burst must end before the co-runners stop (they started about when it did).
    used = [(x[3] - x[4]) / 1e5 for x in ck if x[3] > x[1]]
    return dict(rec=read_rec(rec, GRID), kernel_ghz=float(np.median(ghz)) if ghz else None,
                measured_ms=max(used) if used else None, co_ms=co_ticks / 1e5 if c.get("co") else 0,
                asm=k.asm["amdgcn"])


def launch_race(c):
    n = c["lines"]
    src = torch.ones(n * 256, dtype=torch.float32, device="cuda")
    samples = torch.zeros(16 + 2 * n * c["passes"] * 4, dtype=torch.int64, device="cuda")
    ctr = torch.zeros(4, dtype=torch.int32, device="cuda")
    clk = torch.zeros(1, dtype=torch.int64, device="cuda")
    L, F = c["leader"], c["follower"]
    tail = (c["delta"], c["period"], c["place"], L["xcc"], L["se"], L["cu"], F["xcc"], F["se"], F["cu"])
    big = 1 << 40  # same specialisation as the real t0
    race_kernel.warmup(src, samples, ctr, n, c["passes"], big, *tail, num_warps=1, grid=(2 * GRID,))
    ctr.fill_(1000)  # a launch in which no workgroup takes a role: loads the module
    race_kernel[(2 * GRID,)](src, samples, ctr, n, c["passes"], big, *tail, num_warps=1)
    ctr.zero_()
    clock_kernel[(1,)](clk, num_warps=1)
    flush()  # after src was written
    clock_kernel[(1,)](clk, num_warps=1)
    t0 = (int(clk.item()) + 100000) & ~15  # 1 ms from now; a multiple of 16 like `big`, so no recompile
    race_kernel[(2 * GRID,)](src, samples, ctr, n, c["passes"], t0, *tail, num_warps=1)
    torch.cuda.synchronize()
    s = samples.cpu().numpy()
    body = s[16:].reshape(2, n * c["passes"], 4)
    return dict(t0=t0, counts=ctr.cpu().tolist(), hw=[int(s[0]), int(s[8])],
                dt=body[:, :, 0].tolist(), c0=body[:, :, 1].tolist(), r0=body[:, :, 2].tolist())


def worker(spec_path, dump):
    spec = json.load(open(spec_path))
    fn = dict(gemm=launch_gemm, mix=launch_mix, race=launch_race, depth=launch_depth)[spec["kind"]]
    res = [fn(c) for c in spec["configs"]]
    for r in res:
        r.pop("asm", None) if not spec.get("keep_asm") else None
    json.dump(res, open(dump, "w"))


def _env():
    return {k: v for k, v in os.environ.items() if not k.startswith("ROCPROFILER_REGISTER")}


def _spec(out, kind, configs, keep_asm=False):
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    path = os.path.join(out, "spec.json")
    json.dump(dict(kind=kind, configs=configs, keep_asm=keep_asm), open(path, "w"))
    return path, os.path.join(out, "dump.json")


def _worker_cmd(spec, dump):
    return [sys.executable, os.path.abspath(__file__), "--gpu", os.environ["HIP_VISIBLE_DEVICES"], "worker",
            "--spec", spec, "--dump", dump]


def run_plain(name, kind, configs, keep_asm=False):
    spec, dump = _spec(os.path.join(TMP, name), kind, configs, keep_asm)
    subprocess.run(_worker_cmd(spec, dump), check=True, env=_env())
    return json.load(open(dump))


def run_att(name, kind, config, regex, cu, se_mask="0xF"):
    out = os.path.join(TMP, name)
    spec, dump = _spec(out, kind, [config])
    att = os.path.join(out, "att.json")
    json.dump({"jobs": [{"kernel_include_regex": regex, "kernel_exclude_regex": "", "advanced_thread_trace": True,
                         "att_target_cu": cu, "att_shader_engine_mask": se_mask, "att_simd_select": "0xF",
                         "att_buffer_size": ATT_BUFFER}]}, open(att, "w"))
    with open(os.path.join(out, "rocprof.log"), "w") as log:
        subprocess.run(["rocprofv3", "--att", "-i", att, "-d", out, "--", *_worker_cmd(spec, dump)], env=_env(),
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    ui = [u for u in glob.glob(os.path.join(out, "**", "ui_*"), recursive=True) if os.path.isdir(u)]
    if not ui:
        raise SystemExit(f"{name}: no decoded trace, see {out}/rocprof.log")
    ui.sort(key=lambda u: int(re.search(r"dispatch_(\d+)", u).group(1)))
    return ui[-1], json.load(open(dump))[0]  # the last dispatch: the measured one


def run_pmc(name, kind, configs, regex, per_config=1):
    """Counters of the last dispatch of each config (race launches a no-op first), in launch order."""
    out = os.path.join(TMP, name)
    spec, dump = _spec(out, kind, configs)
    with open(os.path.join(out, "rocprof.log"), "w") as log:
        subprocess.run(["rocprofv3", "--pmc", *COUNTERS, "--kernel-include-regex", regex, "-d", out, "-o", "run",
                        "--output-format", "csv", "--", *_worker_cmd(spec, dump)], env=_env(), stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    by = collections.defaultdict(dict)
    for f in glob.glob(os.path.join(out, "**", "*counter_collection.csv"), recursive=True):
        for r in csv.DictReader(open(f)):
            by[int(r["Dispatch_Id"])][r["Counter_Name"]] = by[int(r["Dispatch_Id"])].get(r["Counter_Name"], 0) + \
                float(r["Counter_Value"])
    rows = [by[d] for d in sorted(by)]
    assert len(rows) == per_config * len(configs), (name, len(rows), len(configs))
    return rows[per_config - 1::per_config], json.load(open(dump))


# --- ATT reduction ------------------------------------------------------------------------------


def load_att(ui):
    code = json.load(open(os.path.join(ui, "code.json")))["code"]
    text = {c[2]: c[0] for c in code}
    rt = json.load(open(os.path.join(ui, "realtime.json")))
    ghz = [(p[-1][0] - p[0][0]) / max(1, p[-1][1] - p[0][1]) / 10.0 for k, p in rt.items()
           if k.startswith("SE") and len(p) >= 2]
    waves = []
    for f in sorted(glob.glob(os.path.join(ui, "se*_sm*_sl*_wv*.json"))):
        m = re.search(r"se(\d+)_sm(\d+)_sl(\d+)_wv(\d+)", os.path.basename(f))
        d = json.load(open(f))
        if d.get("wave", {}).get("instructions"):
            waves.append(dict(se=int(m.group(1)), simd=int(m.group(2)), ins=d["wave"]["instructions"]))
    return text, float(np.median(ghz)) if ghz else None, waves


def wave_loads(w, text):
    """(departure, issue stall) of each buffer_load, in the wave's order."""
    return [(c + s, s) for c, _t, s, _d, i in w["ins"] if text.get(i, "").startswith("buffer_load")]


def reduce_cu(loads, kinds):
    """loads: CU-wide (departure, stall, kind, simd), any order. D_k, stalled share, rate, D_44 by kind."""
    loads = sorted(loads)
    dep = np.array([x[0] for x in loads], dtype=float)
    stall = np.array([x[1] for x in loads], dtype=float)
    stalled = np.nonzero(stall >= 16)[0]
    dk = {}
    for k in range(1, 97):
        ok = stalled[stalled >= k]
        if len(ok) >= 8:
            dk[k] = float(np.median(dep[ok] - dep[ok - k]))
    by_kind = {}
    for kind in kinds:
        ok = [n for n in stalled if n >= 44 and loads[n - 44][2] == kind]
        by_kind[kind] = dict(n=len(ok), d44=float(np.median([dep[n] - dep[n - 44] for n in ok])) if ok else None)
    gaps = np.diff(dep)
    quarters = []
    for q in range(4):
        a, b = q * len(loads) // 4, (q + 1) * len(loads) // 4
        if b - a > 1:
            quarters.append(dict(stalled=float((stall[a:b] >= 16).mean()),
                                 bytes_per_clk=1024.0 * (b - a - 1) / max(1.0, dep[b - 1] - dep[a])))
    return dict(n=len(loads), stalled=float(len(stalled) / max(1, len(loads))), dk=dk, d44=dk.get(44),
                quarters=quarters,
                gap_mean=float(gaps.mean()) if len(gaps) else None, gap_p50=float(np.median(gaps)) if len(gaps) else None,
                bytes_per_clk=1024.0 / gaps.mean() if len(gaps) else None, by_kind=by_kind)


def per_wave(waves_loads):
    out = []
    for simd, kind, loads in waves_loads:
        dep = np.array([x[0] for x in loads], dtype=float)
        st = np.array([x[1] for x in loads], dtype=float)
        out.append(dict(simd=simd, kind=kind, n=len(loads), stalled=float((st >= 16).mean()) if len(st) else None,
                        gap_mean=float(np.diff(dep).mean()) if len(dep) > 1 else None))
    return out


# --- experiments ----------------------------------------------------------------------------------


def discover():
    """The CUs every role uses, from where 256 workgroups of the mix kernel land."""
    r = run_plain("discover", "mix", [dict(groups=1, h4=0, split=False, scope=2, x=99, se=0, cu=0)])[0]
    by = collections.defaultdict(list)
    for x in r["rec"]:
        by[x["xcc"]].append(x)
    assert all(len(by[x]) == 32 for x in range(8)), {x: len(by[x]) for x in by}
    assert len({(x["xcc"], x["se"], x["cu"]) for x in r["rec"]}) == 256, "two workgroups on one CU"
    # ATT traces att_target_cu = 0 in every SE. CU 0 is present on SE0 of XCD 0 on this GPU.
    tgt = next(x for x in by[0] if x["se"] == 0 and x["cu"] == 0)
    same = next(x for x in by[0] if x["se"] != 0)
    other = next(x for x in by[1] if x["se"] == 0 and x["cu"] == 0)
    pick = lambda x: dict(xcc=x["xcc"], se=x["se"], cu=x["cu"])  # noqa: E731
    return dict(target=pick(tgt), same_xcd=pick(same), other_xcd=pick(other))


def exp_gemm(args):
    c = dict(k=args.k, block=args.block, hot=args.hot, fresh=args.fresh)
    name = f"gemm_{args.block}_K{args.k}" + ("_hot" if args.hot else "") + ("_fresh" if args.fresh else "")
    out = dict(config=c)
    if args.att:
        ui, run = run_att(name, "gemm", c, "gemm_kernel", 0)
        text, ghz, waves = load_att(ui)
        tiles = {x["se"]: (x["row"], x["col"]) for x in run["rec"] if x["xcc"] == 0 and x["cu"] == 0}
        cus = collections.defaultdict(list)
        for w in waves:
            cus[w["se"]] += [(d, s, "A" if i % 16 < 8 else "B", w["simd"]) for i, (d, s) in enumerate(wave_loads(w, text))]
        out["att"] = dict(ghz=ghz, us=run["us"], cus={str(se): dict(tile=tiles.get(se), **reduce_cu(l, ["A", "B"]))
                                                        for se, l in cus.items()})
    if args.pmc:
        rows, runs = run_pmc(name + "_pmc", "gemm", [c, c], "gemm_kernel", per_config=2 if args.hot else 1)
        p = rows[-1]
        req_bytes = GRID * (args.k // 64) * 64 * 1024
        out["pmc"] = dict(counters=p, us=runs[-1]["us"], requested_TBps=req_bytes / runs[-1]["us"] / 1e6,
                          l2_hit=p["TCC_HIT_sum"] / (p["TCC_HIT_sum"] + p["TCC_MISS_sum"]),
                          ea_read_TBps=p["TCC_EA0_RDREQ_sum"] * 128 / runs[-1]["us"] / 1e6)
    save(name, out)
    return out


def mix_config(t, h4, split, scope, groups):
    return dict(groups=groups, h4=h4, split=split, scope=scope, x=t["xcc"], se=t["se"], cu=t["cu"])


def mix_att(name, t, c):
    ui, run = run_att(name, "mix", c, "mix_kernel", 0)
    text, ghz, waves = load_att(ui)
    me = next(x for x in run["rec"] if (x["xcc"], x["se"], x["cu"]) == (t["xcc"], t["se"], t["cu"]))
    hot_simd = set(me["wave_simd"][:2])  # waves 0 and 1 hold rows 0-63
    loads, waves_loads = [], []
    for w in waves:
        if w["se"] != t["se"]:
            continue
        burst = wave_loads(w, text)[HOT_CHUNKS:]
        if c["split"]:
            kind_of = lambda i: "hot" if w["simd"] in hot_simd else "cold"  # noqa: E731
        else:
            kind_of = lambda i: "hot" if (i % 4) < c["h4"] else "cold"  # noqa: E731
        lw = [(d, s, kind_of(i), w["simd"]) for i, (d, s) in enumerate(burst)]
        loads += lw
        waves_loads.append((w["simd"], "hot" if w["simd"] in hot_simd else "cold", lw))
    res = reduce_cu(loads, ["hot", "cold"])
    res.update(ghz=ghz, us=run["us"], waves=per_wave(waves_loads), hot_simd=sorted(hot_simd))
    return res


def mix_pmc(name, t, c):
    """Counters of the burst alone: the run with the burst minus the same run without it."""
    c0 = dict(c, groups=0)
    rows, _ = run_pmc(name, "mix", [c0, c, c0, c], "mix_kernel")
    d = {k: rows[3][k] - rows[2][k] for k in COUNTERS}
    hits, miss = d["TCC_HIT_sum"], d["TCC_MISS_sum"]
    active = {0: GRID, 1: 32, 2: 1}[c["scope"]]
    loads = active * c["groups"] * GROUP
    cold = loads * (0.5 if c["split"] else (4 - c["h4"]) / 4)
    return dict(delta=d, l2_hit=hits / max(1, hits + miss), expected_hit=1 - cold / loads,
                ea_bytes=d["TCC_EA0_RDREQ_sum"] * 128, expected_cold_bytes=cold * 4096)


def check_asm(t):
    r = run_plain("mix_asm", "mix", [mix_config(t, 2, False, 2, 4), mix_config(t, 2, True, 2, 4)], keep_asm=True)
    for name, x in zip(("interleave", "split"), r):
        lines = x["asm"].splitlines()
        open(os.path.join(RESULTS, f"mix_{name}.s"), "w").write(x["asm"])
        for h in [i for i, ln in enumerate(lines) if "Loop Header" in ln]:
            label = lines[h].split(":")[0]
            end = next(i for i in range(h + 1, len(lines)) if lines[i].strip().startswith("s_cbranch")
                       and lines[i].split()[-1] == label)
            body = [ln.strip() for ln in lines[h:end + 1] if ln.strip() and not ln.strip().startswith((".", ";"))]
            print(f"{name} loop {label}: {sum(ln.startswith('buffer_load') for ln in body)} loads, "
                  f"{len(body)} instructions, {sum(ln.startswith('s_barrier') for ln in body)} s_barrier")


def exp_mix(args):
    t = discover()["target"]
    os.makedirs(RESULTS, exist_ok=True)
    if args.check:
        check_asm(t)
        out = {}
        for h4 in (0, 4):
            c = mix_config(t, h4, False, 2, args.groups)
            name = f"mix_check_h{h4}"
            out[name] = dict(att=mix_att(name, t, c), pmc=mix_pmc(name + "_pmc", t, c))
            a, p = out[name]["att"], out[name]["pmc"]
            print(f"{name}: stalled {a['stalled']:.0%}, D_16 {a['dk'].get(16)}, D_44 {a['d44']}, "
                  f"{a['bytes_per_clk']:.1f} B/clk at {a['ghz']:.2f} GHz; L2 hit {p['l2_hit']:.2f} "
                  f"(expected {p['expected_hit']:.2f}), memory {p['ea_bytes'] / 1e6:.2f} MB "
                  f"(cold {p['expected_cold_bytes'] / 1e6:.2f} MB)")
        save("mix_check", out)
        return
    out = {}
    for scope in (2, 1, 0):
        for split, h4s in ((False, (0, 1, 2, 3, 4)), (True, (2,))):
            for h4 in h4s:
                c = mix_config(t, h4, split, scope, args.groups)
                name = f"mix_{'split' if split else 'int'}_h{h4}_s{scope}"
                out[name] = dict(config=c, att=mix_att(name, t, c), pmc=mix_pmc(name + "_pmc", t, c))
                a, p = out[name]["att"], out[name]["pmc"]
                print(f"{name}: stalled {a['stalled']:.0%}, D_44 {a['d44']}, by kind "
                      f"{ {k: v['d44'] for k, v in a['by_kind'].items()} }, {a['bytes_per_clk']:.1f} B/clk, "
                      f"L2 hit {p['l2_hit']:.2f} (exp {p['expected_hit']:.2f})", flush=True)
                save("mix_sweep", out)


VMCNT = re.compile(r"vmcnt\((\d+)\)")


def depth_config(t, tier, d, h4, scope, groups=16):
    return dict(tier=tier, d=d, h4=h4, scope=scope, groups=groups, x=t["xcc"], se=t["se"], cu=t["cu"])


def reduce_depth(ui, t, c):
    """Per burst load of the target CU: departure, issue stall, and return from the vmcnt wait that needed it.
    A wait that stalled >= 16 cycles ended at that load's return ("exact"); otherwise its end is an upper bound."""
    text, ghz, waves = load_att(ui)
    P = depth_layout(c)[0]  # prime loads per wave
    loads = []
    for w in waves:
        if w["se"] != t["se"]:
            continue
        ops, ret, resolved = [], {}, 0
        for clock, _ty, stall, dur, idx in w["ins"]:
            s = text.get(idx, "")
            if s.startswith("buffer_load"):
                ops.append((clock + stall, stall))
            elif s.startswith("s_waitcnt") and (m := VMCNT.search(s)):
                need = len(ops) - int(m.group(1))
                for i in range(resolved, need):
                    ret[i] = (clock + dur, stall >= 16 and i == need - 1)
                resolved = max(resolved, need)
        for i, (dep, st) in enumerate(ops[P:]):
            end, exact = ret.get(P + i, (None, False))
            loads.append(dict(dep=dep, stall=st, end=end, exact=exact, wave=w["simd"],
                              kind="fast" if (i % 4) < c["h4"] else "cold"))
    loads = [x for x in loads if x["end"] is not None]
    if len(loads) < 32:
        return dict(ghz=ghz, n=len(loads))
    loads.sort(key=lambda x: x["dep"])
    lo, hi = len(loads) // 10, len(loads) - len(loads) // 10  # middle 80%: no ramp-up or drain
    mid = loads[lo:hi]
    dep = np.array([x["dep"] for x in loads], dtype=float)
    end = np.array([x["end"] for x in loads], dtype=float)
    order = np.sort(end)
    occ = [int(np.searchsorted(dep, x["dep"], side="right") - np.searchsorted(order, x["dep"], side="right"))
           for x in mid]

    def rt(sel):
        v = [x["end"] - x["dep"] for x in sel if x["exact"]]
        return dict(n=len(v), p10=float(np.percentile(v, 10)) if v else None, p50=float(np.median(v)) if v else None,
                    p90=float(np.percentile(v, 90)) if v else None)

    span = mid[-1]["dep"] - mid[0]["dep"]
    rate = (len(mid) - 1) / span if span > 0 else None  # loads per cycle, CU-wide
    mean_l = float(np.mean([x["end"] - x["dep"] for x in mid]))
    return dict(ghz=ghz, n=len(loads), exact_share=float(np.mean([x["exact"] for x in mid])),
                rt=rt(mid), rt_fast=rt([x for x in mid if x["kind"] == "fast"]),
                rt_cold=rt([x for x in mid if x["kind"] == "cold"]),
                issue_stalled=float(np.mean([x["stall"] >= 16 for x in mid])),
                issue_stall_p90=float(np.percentile([x["stall"] for x in mid], 90)),
                occ_p50=float(np.median(occ)), occ_p90=float(np.percentile(occ, 90)), occ_max=int(max(occ)),
                bytes_per_clk=rate * 1024 if rate else None, little_kb=rate * mean_l if rate else None,
                mean_l=mean_l)


def depth_att(name, t, c):
    # With co-runners, trace only the measured CU's shader engine: the other traced CU would be a co-runner.
    mask = "0x%x" % (1 << t["se"]) if c.get("co") else "0xF"
    ui, run = run_att(name, "depth", c, "depth_kernel", t["cu"], mask)
    r = reduce_depth(ui, t, c)
    r.update(kernel_ghz=run["kernel_ghz"], ui=ui, measured_ms=run["measured_ms"], co_ms=run["co_ms"])
    return r


def depth_pmc(name, t, c):
    c0 = dict(c, groups=0)
    rows, _ = run_pmc(name, "depth", [c0, c, c0, c], "depth_kernel")
    d = {k: rows[3][k] - rows[2][k] for k in COUNTERS}
    hits, miss = d["TCC_HIT_sum"], d["TCC_MISS_sum"]
    active = {0: GRID, 1: 32, 2: 1}[c["scope"]]
    loads = active * c["groups"] * GROUP
    fast = loads * c["h4"] / 4
    return dict(l2_hit=hits / max(1, hits + miss), ea_bytes=d["TCC_EA0_RDREQ_sum"] * 128,
                burst_bytes=loads * 4096, expected_l2_hit=(fast / loads) if c["tier"] == 0 else 0.0)


def check_depth_asm(t):
    r = run_plain("depth_asm", "depth", [depth_config(t, 0, 4, 2, 2), depth_config(t, 1, 11, 4, 2)], keep_asm=True)
    for name, x in zip(("L2_D4", "LLC_D11"), r):
        lines = x["asm"].splitlines()
        open(os.path.join(RESULTS, f"depth_{name}.s"), "w").write(x["asm"])
        for h in [i for i, ln in enumerate(lines) if "Loop Header" in ln]:
            label = lines[h].split(":")[0]
            end = next(i for i in range(h + 1, len(lines)) if lines[i].strip().startswith("s_cbranch")
                       and lines[i].split()[-1] == label)
            body = [ln.strip() for ln in lines[h:end + 1] if ln.strip() and not ln.strip().startswith((".", ";"))]
            nl = sum(ln.startswith("buffer_load") for ln in body)
            waits = [ln for ln in body if ln.startswith("s_waitcnt") and "vmcnt" in ln]
            print(f"depth {name} loop {label}: {nl} loads, {len(waits)} vmcnt waits {sorted(set(waits))[:3]}, "
                  f"{sum(ln.startswith('s_barrier') for ln in body)} s_barrier, {len(body)} instructions")


TIERS = {0: "L2", 1: "LLC", 2: "HBM"}  # tier 2 = all cold: tier 0 with h4 = 0


def depth_point(t, tier, d, scope, h4=None):
    if tier == 2:
        return depth_config(t, 0, d, 0, scope)
    return depth_config(t, tier, d, 4 if h4 is None else h4, scope)


def exp_depth(args):
    t = discover()["target"]
    os.makedirs(RESULTS, exist_ok=True)
    if args.check:
        check_depth_asm(t)
        out = {}
        for tier in (0, 1, 2):
            for scope in (2, 0):
                c = depth_point(t, tier, 16, scope)
                p = depth_pmc(f"depth_check_pmc_{TIERS[tier]}_s{scope}", t, c)
                out[f"pmc_{TIERS[tier]}_s{scope}"] = p
                print(f"counters {TIERS[tier]} scope {scope}: L2 hit {p['l2_hit']:.2f} (expected "
                      f"{p['expected_l2_hit']:.2f}), memory {p['ea_bytes'] / 1e6:.2f} MB of {p['burst_bytes'] / 1e6:.2f} MB",
                      flush=True)
        for tier in (0, 1, 2):
            for d in (0, 2, 16):
                a = depth_att(f"depth_check_{TIERS[tier]}_d{d}", t, depth_point(t, tier, d, 2))
                out[f"att_{TIERS[tier]}_d{d}"] = a
                print(f"{TIERS[tier]} D={d} one CU: exact {a['exact_share']:.0%}, round trip p50 {a['rt']['p50']} "
                      f"(p10 {a['rt']['p10']}, p90 {a['rt']['p90']}) cyc, in flight p50 {a['occ_p50']} (max "
                      f"{a['occ_max']}, expected {4 * (d + 1)}), issue stalled {a['issue_stalled']:.0%}, "
                      f"{a['bytes_per_clk']:.1f} B/clk, Little {a['little_kb']:.1f}, ATT {a['ghz']:.2f} GHz, "
                      f"kernel {a['kernel_ghz']:.2f} GHz", flush=True)
        save("depth_check", out)
        return
    out = {}
    pts = [(tier, d, scope, None) for scope in (2, 0) for tier in (0, 1, 2) for d in DEPTHS]
    pts += [(tier, d, scope, 2) for scope in (2, 0) for tier in (0, 1) for d in (4, 11, 16)]
    for tier, d, scope, h4 in pts:
        c = depth_point(t, tier, d, scope, h4)
        name = f"depth_{TIERS[tier]}{'' if h4 is None else '_mix50'}_d{d}_s{scope}"
        a = depth_att(name, t, c)
        a.pop("ui", None)
        out[name] = dict(config=c, att=a)
        print(f"{name}: exact {a.get('exact_share', 0):.0%}, rt p50 {a.get('rt', {}).get('p50')} (fast "
              f"{a.get('rt_fast', {}).get('p50')}, cold {a.get('rt_cold', {}).get('p50')}), in flight "
              f"{a.get('occ_p50')} (max {a.get('occ_max')}), issue stalled {a.get('issue_stalled', 0):.0%}, "
              f"{a.get('bytes_per_clk') or 0:.1f} B/clk, ATT {a['ghz']:.2f} GHz", flush=True)
        save("depth_sweep", out)


DEPTHS = (0, 1, 2, 4, 6, 8, 10, 11, 12, 14, 16)
CORUN = {"none": None, "xcd": 2, "all": 1}  # co-runner placement -> CO mode; MFMA only, no memory traffic


def exp_corun(args):
    """The depth sweep on one CU at the shader clocks MFMA co-runners give it: none (~2.39 GHz), the measured XCD's
    other 31 CUs (~2.24), all 255 other CUs (~2.20). The measured CU sleeps 50 ms first so the clock settles."""
    t = discover()["target"]
    os.makedirs(RESULTS, exist_ok=True)

    def cfg(tier, d, h4, co):
        c = depth_point(t, tier, d, 2, h4)
        if CORUN[co]:
            c.update(co=255, co_mode=CORUN[co], settle_ms=50)
        return c

    # The co-runners must add no memory traffic: counters of the burst with and without them.
    pm = {}
    for co in ("none", "all"):
        p = depth_pmc(f"corun_pmc_{co}", t, cfg(2, 16, None, co))
        pm[co] = p
        print(f"counters, HBM D=16, co-runners {co}: L2 hit {p['l2_hit']:.3f}, memory {p['ea_bytes'] / 1e6:.3f} MB "
              f"of {p['burst_bytes'] / 1e6:.3f} MB", flush=True)
    out = dict(pmc=pm, rows={})
    pts = [(tier, d, None) for tier in (0, 1, 2) for d in (0, 2, 4, 8, 10, 12, 16)]
    pts += [(tier, d, 2) for tier in (0, 1) for d in (4, 10)]
    for co in ("none", "xcd", "all"):
        for tier, d, h4 in pts:
            name = f"corun_{co}_{TIERS[tier]}{'' if h4 is None else '_mix50'}_d{d}"
            a = depth_att(name, t, cfg(tier, d, h4, co))
            a.pop("ui", None)
            out["rows"][name] = dict(co=co, tier=TIERS[tier], mix=h4 is not None, d=d, att=a)
            rt = a.get("rt", {})
            print(f"{name}: clock {a['kernel_ghz']:.3f} GHz (ATT {a['ghz']:.3f}); rt p50 {rt.get('p50')} cyc = "
                  f"{(rt.get('p50') or 0) / a['kernel_ghz']:.0f} ns (fast {a.get('rt_fast', {}).get('p50')}, cold "
                  f"{a.get('rt_cold', {}).get('p50')}); exact {a.get('exact_share', 0):.0%}; in flight "
                  f"{a.get('occ_p50')}; issue stalled {a.get('issue_stalled', 0):.0%}; "
                  f"{a.get('bytes_per_clk') or 0:.1f} B/clk", flush=True)
            save("corun_sweep", out)


def clock_pass(name, kind, configs, regex):
    """GRBM_GUI_ACTIVE / 8 XCDs / kernel time per dispatch (reads ~6% above the ATT clock)."""
    out = os.path.join(TMP, name)
    spec, dump = _spec(out, kind, configs)
    with open(os.path.join(out, "rocprof.log"), "w") as log:
        subprocess.run(["rocprofv3", "--pmc", "GRBM_GUI_ACTIVE", "--kernel-trace", "--kernel-include-regex", regex,
                        "-d", out, "-o", "run", "--output-format", "csv", "--", *_worker_cmd(spec, dump)], env=_env(),
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    pm, kt = collections.defaultdict(float), {}
    for f in glob.glob(os.path.join(out, "**", "*counter_collection.csv"), recursive=True):
        for r in csv.DictReader(open(f)):
            pm[int(r["Dispatch_Id"])] += float(r["Counter_Value"])
    for f in glob.glob(os.path.join(out, "**", "*kernel_trace.csv"), recursive=True):
        for r in csv.DictReader(open(f)):
            if re.search(regex, r["Kernel_Name"]):
                kt[int(r["Dispatch_Id"])] = int(r["End_Timestamp"]) - int(r["Start_Timestamp"])
    return [pm[i] / 8 / kt[i] for i in sorted(kt) if i in pm]


def exp_suite(args):
    """Everything for one clock setting, into results/clock_<label>/."""
    before = smi()
    roles = discover()
    t = roles["target"]
    race_args = argparse.Namespace(check=True, sweep=False)
    exp_race(race_args)
    exp_depth(argparse.Namespace(check=False))
    exp_mix(argparse.Namespace(check=False, sweep=True, groups=16))
    for fresh in (False, True):
        exp_gemm(argparse.Namespace(k=2048, block="2x16", att=True, pmc=True, hot=False, fresh=fresh))
    clocks = dict(
        depth_hbm_d16_all=clock_pass("clock_depth", "depth", [depth_point(t, 2, 16, 0)] * 4, "depth_kernel"),
        gemm_cold=clock_pass("clock_gemm", "gemm", [dict(k=8192, block="2x16", hot=False, fresh=False)] * 4,
                             "gemm_kernel"))
    save("suite_clock", dict(label=args.label, smi_before=before, smi_after=smi(), grbm_ghz=clocks))
    print("GRBM GHz:", {k: [round(x, 2) for x in v] for k, v in clocks.items()})
    print("smi before:\n" + before + "\nsmi after:\n" + smi())


def race_config(roles, place, delta, lines=32, passes=1, period=2000):
    f = {0: roles["target"], 1: roles["same_xcd"], 2: roles["other_xcd"], 3: roles["target"]}[place]
    return dict(place=place, delta=delta, lines=lines, passes=passes, period=period, leader=roles["target"],
                follower=f)


def race_summary(r, c):
    n = c["lines"] * c["passes"]
    dt = np.array(r["dt"], dtype=float)
    r0 = np.array(r["r0"], dtype=float)
    c0 = np.array(r["c0"], dtype=float)
    ok = dt[0] > 0
    ghz = float(np.median(np.diff(c0[0][ok]) / np.diff(r0[0][ok]) / 10.0)) if ok.sum() > 2 else None
    out = dict(counts=r["counts"], ghz=ghz, leader=[float(np.median(dt[0][p * c["lines"]:(p + 1) * c["lines"]]))
                                                    for p in range(c["passes"])])
    if c["place"] != 3:
        okf = dt[1] > 0
        out["follower_n"] = int(okf.sum())
        if okf.any():
            gap = (r0[1] - r0[0])[okf & ok]
            out.update(follower=float(np.median(dt[1][okf])), follower_p10=float(np.percentile(dt[1][okf], 10)),
                       follower_p90=float(np.percentile(dt[1][okf], 90)), gap_ticks=float(np.median(gap)),
                       gap_cycles=float(np.median(gap)) * 10 * ghz if ghz else None)
    late = (r0[0][0] - r["t0"]) if ok[0] else None
    out["leader_late_ticks"] = float(late) if late is not None else None
    return out


def exp_race(args):
    roles = discover()
    print("roles", roles)
    if args.check:
        # 64 lines (64 KB, past the 32 KB L1): pass 0 HBM, pass 1 L2. 16 lines: pass 1 is an L1 hit.
        cs = [race_config(roles, 3, 0, lines=64, passes=2), race_config(roles, 3, 0, lines=16, passes=2)]
        ss = [race_summary(r, c) for c, r in zip(cs, run_plain("race_check", "race", cs))]
        for s, tiers in zip(ss, (("HBM", "L2"), ("HBM", "L1"))):
            print(f"solo: {tiers[0]} {s['leader'][0]:.0f} cyc, {tiers[1]} {s['leader'][1]:.0f} cyc at "
                  f"{s['ghz']:.2f} GHz (in-kernel), started {s['leader_late_ticks']} ticks after t0")
        save("race_check", dict(configs=cs, summary=ss, tiers=dict(HBM=ss[0]["leader"][0], L2=ss[0]["leader"][1],
                                                                    L1=ss[1]["leader"][1], ghz=ss[0]["ghz"])))
        return
    deltas = [0, 2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 70, 100]
    out = dict(roles=roles, rows=[])
    for place in (1, 0, 2):
        cs = [race_config(roles, place, d) for d in deltas]
        for c, r in zip(cs, run_plain(f"race_p{place}", "race", cs)):
            s = race_summary(r, c)
            out["rows"].append(dict(place=place, delta=c["delta"], **s))
            print(f"place {place} delta {c['delta']:3d} ticks: leader {s['leader'][0]:.0f}, follower "
                  f"{s.get('follower')} cyc (p10 {s.get('follower_p10')}, p90 {s.get('follower_p90')}), gap "
                  f"{s.get('gap_cycles')} cyc, counts {s['counts']}", flush=True)
    pm = [race_config(roles, 3, 0), race_config(roles, 1, 0), race_config(roles, 1, 100), race_config(roles, 0, 0),
          race_config(roles, 2, 0)]
    rows, runs = run_pmc("race_pmc", "race", pm, "race_kernel", per_config=2)
    out["pmc"] = []
    for c, p, r in zip(pm, rows, runs):
        s = race_summary(r, c)
        out["pmc"].append(dict(place=c["place"], delta=c["delta"], counters=p, follower=s.get("follower"),
                               leader=s["leader"][0]))
        print(f"pmc place {c['place']} delta {c['delta']}: {p}  leader {s['leader'][0]:.0f} follower "
              f"{s.get('follower')}", flush=True)
    out["smi"] = smi()
    save("race_sweep", out)


def plot(_args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = json.load(open(os.path.join(RESULTS, "race_sweep.json")))
    fig, ax = plt.subplots(figsize=(7, 4))
    for place, label in ((1, "other CU, same XCD"), (0, "same CU"), (2, "other XCD")):
        pts = sorted((x["gap_cycles"], x["follower"]) for x in r["rows"] if x["place"] == place and x.get("gap_cycles")
                     is not None)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=label)
    lead = float(np.median([x["leader"][0] for x in r["rows"]]))
    g = np.linspace(0, 2500, 50)
    ax.plot(g, lead - g, "k--", lw=0.8, label=f"leader's round trip ({lead:.0f}) minus the gap")
    ax.set_ylim(0, 1500)
    ax.set_xlabel("follower's load issued this many cycles after the leader's")
    ax.set_ylabel("follower's load latency, cycles")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "race.png"), dpi=120)
    m = json.load(open(os.path.join(RESULTS, "mix_sweep.json")))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for scope, label in ((2, "one CU"), (1, "XCD 0 (32 CUs)"), (0, "all 256 CUs")):
        hs = [0, 1, 2, 3, 4]
        pts = [m[f"mix_int_h{h}_s{scope}"]["att"] for h in hs]
        axes[0].plot([h / 4 for h in hs], [p["d44"] for p in pts], "o-", label=label)
        axes[1].plot([h / 4 for h in hs], [p["bytes_per_clk"] for p in pts], "o-", label=label)
    axes[0].set_ylabel("D_44: round trip from issue, cycles")
    axes[1].set_ylabel("one CU's load rate, B/cycle")
    axes[1].axhline(64, color="grey", ls=":", lw=0.8)
    for a in axes:
        a.set_xlabel("L2 hit fraction (counters agree to 0.02)")
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "mix.png"), dpi=120)
    print("wrote results/race.png, results/mix.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--spec", required=True)
    w.add_argument("--dump", required=True)
    g = sub.add_parser("gemm")
    g.add_argument("--block", default="2x16", choices=["2x16", "4x8"])
    g.add_argument("--k", type=int, default=2048)
    g.add_argument("--att", action="store_true")
    g.add_argument("--pmc", action="store_true")
    g.add_argument("--hot", action="store_true", help="inputs left in the memory-side cache by a previous launch")
    g.add_argument("--fresh", action="store_true", help="inputs written after the flush")
    m = sub.add_parser("mix")
    m.add_argument("--check", action="store_true")
    m.add_argument("--sweep", action="store_true")
    m.add_argument("--groups", type=int, default=16, help="16 loads each: 256 burst loads per wave")
    r = sub.add_parser("race")
    r.add_argument("--check", action="store_true")
    r.add_argument("--sweep", action="store_true")
    dp = sub.add_parser("depth")
    dp.add_argument("--check", action="store_true")
    dp.add_argument("--sweep", action="store_true")
    sub.add_parser("suite")
    sub.add_parser("corun")
    sub.add_parser("plot")
    p.add_argument("--label", default=None, help="write to results/clock_<label>/ and tmp/clock_<label>/")
    args = p.parse_args()
    global RESULTS, TMP
    if args.label:
        RESULTS = os.path.join(HERE, "results", f"clock_{args.label}")
        TMP = os.path.join(HERE, "tmp", f"clock_{args.label}")
    os.makedirs(RESULTS, exist_ok=True)
    if args.cmd == "worker":
        worker(args.spec, args.dump)
    else:
        print("physical GPU", os.environ["HIP_VISIBLE_DEVICES"], "results in", RESULTS, flush=True)
        dict(gemm=exp_gemm, mix=exp_mix, race=exp_race, plot=plot, depth=exp_depth, suite=exp_suite, corun=exp_corun)[args.cmd](args)


if __name__ == "__main__":
    main()
