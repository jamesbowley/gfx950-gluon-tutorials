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
"""Stage 3: what the correct publish / consume sequences cost (see README.md).

    HIP_VISIBLE_DEVICES=1 python costs.py --sweep bits ops handshake arrive
    HIP_VISIBLE_DEVICES=1 python costs.py --sweep handshake arrive --warm

- ops: cache_op_kernel. On every XCD, slot 0 times one cache operation (buffer_wbl2 sc1 or
  buffer_inv sc1, each followed by s_waitcnt vmcnt(0)) while the XCD's other 31 CUs have just
  written D MiB (dirty in L2) or read W MiB (valid in L2). After the operation the 31 CUs read
  their data again; with the time of that re-read against a run without the operation, the
  damage to the neighbours.
- handshake: handshake_kernel. One 256 KiB fp32 partial per XCD, published by slot 0 and
  consumed by slot 1 of the next XCD (or the same XCD), in v20's lane-contiguous order (16 bytes
  per lane, 1 KiB per wave instruction), with a relaxed gpu-scope flag between them. Each
  correct sequence, with and without D MiB of other dirty data on the writer's XCD.
- arrive: arrive_kernel. A last-arriver reduction (aiter's _sum_splits_on_xcd, v20's arrival count):
  groups of n programs store their partials and add 1 to a counter; the last sums. Each arrival
  scope and order, with the group on one XCD or spread over XCDs, for 8 KiB and 256 KiB partials.
- bits: store_kernel. The 256 KiB store alone with each set of cache bits, written with inline
  asm (dwordx2, the widest store an asm operand can carry here), to compare sc1 with sc0 sc1.

Timestamps are s_memrealtime (100 MHz, one clock for the device).
"""

import argparse
import os
import statistics
import time

import triton  # before torch: torch's HIP pulls in ROCm's shared libLLVM, which breaks libtriton's
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

NUM_PROGRAMS = 256
SLOT = 64 * 1024  # fp32 elements of one 256 KiB partial
MIB = 2**20
_SLOT = gl.constexpr(SLOT)


@gluon.jit
def _one():
    return gl.zeros([1], gl.int32, gl.BlockedLayout([1], [64], [gl.num_warps()], [0]))


@gluon.jit
def _asm(code: gl.constexpr):
    gl.inline_asm_elementwise(code + "\nv_mov_b32 $0, $1", "=v,v,~{memory}", [_one()], dtype=gl.int32,
                              is_pure=False, pack=1)


@gluon.jit
def _now():
    return gl.max(gl.inline_asm_elementwise("s_memrealtime $0\ns_waitcnt lgkmcnt(0)", "=s,v,~{memory}", [_one()],
                                            dtype=gl.int64, is_pure=False, pack=1), axis=0)


@gluon.jit
def _stamp():
    """Every wave's memory operations complete, then the workgroup meets, then the clock."""
    return gl.max(gl.inline_asm_elementwise("s_waitcnt vmcnt(0)\ns_barrier\ns_memrealtime $0\ns_waitcnt lgkmcnt(0)",
                                            "=s,v,~{memory}", [_one()], dtype=gl.int64, is_pure=False, pack=1), axis=0)


@gluon.jit
def _getreg(REG: gl.constexpr):
    return gl.max(gl.inline_asm_elementwise("s_getreg_b32 $0, hwreg(" + REG + ")", "=s,v", [_one()],
                                            dtype=gl.int32, is_pure=True, pack=1), axis=0)


@gluon.jit
def _grid_sync(cnt_ptr):
    _asm("s_waitcnt vmcnt(0)\ns_barrier")
    gl.atomic_add(cnt_ptr, 1, sem="relaxed", scope="gpu")
    while gl.atomic_add(cnt_ptr, 0, sem="relaxed", scope="gpu") < 256:
        pass
    _asm("s_nop 0")


@gluon.jit
def _keep(x):
    gl.inline_asm_elementwise("; keep", "=v,0", [x], dtype=x.dtype, is_pure=False, pack=1)


@gluon.jit
def _lds_hold(L: gl.constexpr):
    """128 KiB of LDS (of 160 KiB per CU): one workgroup per CU."""
    lds = gl.allocate_shared_memory(gl.int8, [128 * 1024], gl.SwizzledSharedLayout(1, 1, 1, [0]))
    lds.slice(0, 64 * gl.num_warps()).store(gl.zeros([64 * gl.num_warps()], gl.int8, L))
    return lds


@gluon.jit
def _write_lines(base, n_elems, val, L: gl.constexpr):
    offs = gl.arange(0, 1024, L)
    v = gl.zeros([1024], gl.float32, L) + val
    for t in range(0, n_elems, 1024):
        gl.amd.cdna4.buffer_store(stored_value=v, ptr=base + t, offsets=offs)


@gluon.jit
def _read_lines(base, n_elems, L: gl.constexpr):
    offs = gl.arange(0, 1024, L)
    s = gl.zeros([1024], gl.float32, L)
    for t in range(0, n_elems, 1024):
        s += gl.amd.cdna4.buffer_load(ptr=base + t, offsets=offs)
    return s


# ---------------------------------------------------------------------------------------------
# ops: one cache operation against the XCD's dirty or valid lines
# ---------------------------------------------------------------------------------------------


@gluon.jit
def cache_op_kernel(buf_ptr, cnt_ptr, rec_ptr, elems, MODE: gl.constexpr, OP: gl.constexpr, ALL: gl.constexpr):
    """MODE 0: the 31 helpers of each XCD write `elems` fp32 each (default stores); MODE 1: they read
    them twice (default loads). Then slot 0 (or, with ALL, every program) issues OP: 0 nothing,
    1 buffer_wbl2 sc1, 2 buffer_inv sc1. Then the helpers read their data once more."""
    L: gl.constexpr = gl.BlockedLayout([4], [64], [gl.num_warps()], [0])
    pid = gl.program_id(0)
    slot = pid // 8
    lds = _lds_hold(L)
    helper = slot > 0
    base = buf_ptr + pid.to(gl.int64) * elems
    if helper:
        if MODE == 0:
            _write_lines(base, elems, 1.0, L)
        else:
            _keep(_read_lines(base, elems, L))
            _keep(_read_lines(base, elems, L))
    _grid_sync(cnt_ptr)
    t0 = _stamp()
    t1 = t0
    if (slot == 0) | ALL:
        if OP == 1:
            _asm("buffer_wbl2 sc1\ns_waitcnt vmcnt(0)")
        elif OP == 2:
            _asm("buffer_inv sc1\ns_waitcnt vmcnt(0)")
        t1 = _stamp()
    _grid_sync(cnt_ptr + 1)
    t2 = _stamp()
    if helper:
        _keep(_read_lines(base, elems, L))
    t3 = _stamp()
    z = _one()
    rec = rec_ptr + pid * 8 + z
    gl.store(rec, (t1 - t0) + z)
    gl.store(rec + 1, (t3 - t2) + z)
    gl.store(rec + 2, (_getreg("HW_REG_XCC_ID") & 15).to(gl.int64) + z)
    gl.store(rec + 3, gl.sum(lds.slice(0, 64 * gl.num_warps()).load(L).to(gl.int64), axis=0) + z)


# ---------------------------------------------------------------------------------------------
# handshake: publish a 256 KiB partial, consume it on another CU
# ---------------------------------------------------------------------------------------------


@gluon.jit
def _flag_set(ptr, epoch, FLAG: gl.constexpr):
    if FLAG == 0:
        gl.atomic_xchg(ptr, epoch, sem="relaxed", scope="gpu")
    elif FLAG == 3:
        gl.atomic_xchg(ptr, epoch, sem="release", scope="gpu")
    else:
        v = gl.zeros([1], gl.int32, gl.BlockedLayout([1], [64], [gl.num_warps()], [0])) + epoch
        gl.amd.cdna4.buffer_store(stored_value=v, ptr=ptr, offsets=_one(), cache="" if FLAG == 1 else ".wt")


@gluon.jit
def _flag_get(ptr, FLAG: gl.constexpr):
    if FLAG == 0:
        return gl.atomic_add(ptr, 0, sem="relaxed", scope="gpu")
    elif FLAG == 3:
        return gl.atomic_add(ptr, 0, sem="relaxed", scope="gpu")
    else:
        _asm("s_nop 0")  # compiler fence: re-read every iteration
        f = gl.amd.cdna4.buffer_load(ptr=ptr, offsets=_one(), cache=".cg" if FLAG == 1 else ".cv")
        return gl.max(f, axis=0)


@gluon.jit
def handshake_kernel(p_ptr, flag_ptr, cnt_ptr, rec_ptr, dirty_ptr, epoch, place, dirty_elems, n_dirty, nw, timeout,
                     STORE: gl.constexpr, LOAD: gl.constexpr, WF: gl.constexpr, RF: gl.constexpr,
                     FLAG: gl.constexpr):
    """Slots 0 .. nw - 1 of every XCD write their partials; slot nw + i of the next XCD (place 2) or of
    the same XCD (place 1) reads writer i's. Slots 2 nw .. 2 nw + n_dirty - 1 dirty dirty_elems fp32 each
    first. WF: 0 nothing,
    1 s_waitcnt vmcnt(0), 2 buffer_wbl2 sc1 + s_waitcnt vmcnt(0), before the workgroup barrier and the
    flag. RF: 0 nothing, 1 buffer_inv sc1 in every wave, 2 an acquire atomic load (the compiler's acquire),
    after the flag and before the reads. FLAG: 0 a relaxed gpu-scope atomic swap, polled with relaxed
    gpu-scope atomic loads (sc1); 1 a plain default store, polled with sc0 nt loads (v21 `l2`, one XCD only);
    2 a `.wt` store, polled with `.cv` loads (the reference kernel); 3 as 0, but the swap is a gpu-scope
    release (the compiler's release). The spin gives up after `timeout` ticks and the record says so."""
    L: gl.constexpr = gl.BlockedLayout([4], [64], [gl.num_warps()], [0])
    pid = gl.program_id(0)
    xcd = pid % 8
    slot = pid // 8
    lds = _lds_hold(L)
    if place == 2:
        pair_r = (xcd + 7) % 8
    else:
        pair_r = xcd
    if (slot >= 2 * nw) & (slot < 2 * nw + n_dirty):
        _write_lines(dirty_ptr + pid.to(gl.int64) * dirty_elems, dirty_elems, 1.0, L)
    _grid_sync(cnt_ptr)
    offs = gl.arange(0, 1024, L)
    t0 = _stamp()
    t1 = t0
    t2 = t0
    t3 = t0
    bad = 0
    if slot < nw:
        widx = xcd * nw + slot
        slot_ptr = p_ptr + widx.to(gl.int64) * _SLOT
        v = gl.zeros([1024], gl.float32, L) + epoch.to(gl.float32)
        for r in range(64):
            gl.amd.cdna4.buffer_store(stored_value=v, ptr=slot_ptr + r * 1024, offsets=offs, cache=STORE)
        if WF == 1:
            _asm("s_waitcnt vmcnt(0)")
        elif WF == 2:
            _asm("buffer_wbl2 sc1\ns_waitcnt vmcnt(0)")
        _asm("s_barrier")
        t1 = _now()
        _flag_set(flag_ptr + widx * 32, epoch, FLAG)
    timed_out = 0
    if (slot >= nw) & (slot < 2 * nw):
        ridx = pair_r * nw + slot - nw
        deadline = t0 + timeout
        f = _flag_get(flag_ptr + ridx * 32, FLAG)
        while (f != epoch) & (_now() < deadline):
            f = _flag_get(flag_ptr + ridx * 32, FLAG)
        timed_out = (f != epoch).to(gl.int32)
        _asm("s_nop 0")
        t2 = _now()
        if RF == 1:
            _asm("s_waitcnt vmcnt(0)\nbuffer_inv sc1")
        elif RF == 2:
            gl.atomic_add(flag_ptr + ridx * 32, 0, sem="acquire", scope="gpu")
            _asm("s_nop 0")
        slot_ptr = p_ptr + ridx.to(gl.int64) * _SLOT
        nbad = gl.zeros([1024], gl.int32, L)
        for r in range(64):
            x = gl.amd.cdna4.buffer_load(ptr=slot_ptr + r * 1024, offsets=offs, cache=LOAD)
            nbad += (x != epoch.to(gl.float32)).to(gl.int32)
        bad = gl.sum(nbad, axis=0)
        t3 = _stamp()
    z = _one()
    rec = rec_ptr + pid * 8 + z
    gl.store(rec, t0 + z)
    gl.store(rec + 1, t1 + z)
    gl.store(rec + 2, t2 + z)
    gl.store(rec + 3, t3 + z)
    gl.store(rec + 4, bad.to(gl.int64) + z)
    gl.store(rec + 5, (_getreg("HW_REG_XCC_ID") & 15).to(gl.int64) + z)
    gl.store(rec + 6, gl.sum(lds.slice(0, 64 * gl.num_warps()).load(L).to(gl.int64), axis=0) + z)
    gl.store(rec + 7, timed_out.to(gl.int64) + z)


# ---------------------------------------------------------------------------------------------
# arrive: a last-arriver reduction (aiter's _sum_splits_on_xcd, v20's arrival count)
# ---------------------------------------------------------------------------------------------


@gluon.jit
def arrive_kernel(p_ptr, gcnt_ptr, cnt_ptr, rec_ptr, epoch, n, place, slot_elems, STORE: gl.constexpr,
                  LOAD: gl.constexpr, WF: gl.constexpr, ARR: gl.constexpr):
    """Groups of n programs: every member stores a partial of slot_elems fp32 (all = epoch) with STORE,
    applies WF (0 nothing, 1 s_waitcnt vmcnt(0), 2 buffer_wbl2 sc1 + s_waitcnt vmcnt(0)), meets its
    workgroup at s_barrier, and adds 1 to the group's counter with ARR: 0 relaxed gpu, 1 acq_rel cta
    (aiter), 2 acq_rel gpu (the compiler's release + acquire), 3 relaxed cta. The member that sees n - 1
    sums the group's n partials with LOAD, counts wrong sums and re-arms the counter. place 1: a group is
    n programs of one XCD; place 2: n consecutive pids, spread over min(n, 8) XCDs. Nobody spins, so a
    broken scheme gives wrong sums or no last arriver, never a hang."""
    L: gl.constexpr = gl.BlockedLayout([4], [64], [gl.num_warps()], [0])
    pid = gl.program_id(0)
    xcd = pid % 8
    slot = pid // 8
    lds = _lds_hold(L)
    if place == 1:
        group = xcd * (32 // n) + slot // n
        member = slot % n
    else:
        group = pid // n
        member = pid % n
    offs = gl.arange(0, 1024, L)
    _grid_sync(gcnt_ptr)
    t0 = _stamp()
    base = p_ptr + (group * n).to(gl.int64) * slot_elems
    v = gl.zeros([1024], gl.float32, L) + epoch.to(gl.float32)
    for t in range(0, slot_elems, 1024):
        gl.amd.cdna4.buffer_store(stored_value=v, ptr=base + member.to(gl.int64) * slot_elems + t, offsets=offs,
                                  cache=STORE)
    if WF == 1:
        _asm("s_waitcnt vmcnt(0)")
    elif WF == 2:
        _asm("buffer_wbl2 sc1\ns_waitcnt vmcnt(0)")
    _asm("s_barrier")
    t1 = _now()
    if ARR == 0:
        old = gl.atomic_add(cnt_ptr + group * 32, 1, sem="relaxed", scope="gpu")
    elif ARR == 1:
        old = gl.atomic_add(cnt_ptr + group * 32, 1, sem="acq_rel", scope="cta")
    elif ARR == 2:
        old = gl.atomic_add(cnt_ptr + group * 32, 1, sem="acq_rel", scope="gpu")
    else:
        old = gl.atomic_add(cnt_ptr + group * 32, 1, sem="relaxed", scope="cta")
    _asm("s_nop 0")  # compiler fence: no partial load moves above the arrival
    t2 = _now()
    t3 = t2
    bad = 0
    last = old == n - 1
    if last:
        expect = n.to(gl.float32) * epoch.to(gl.float32)
        nbad = gl.zeros([1024], gl.int32, L)
        for t in range(0, slot_elems, 1024):
            s = gl.zeros([1024], gl.float32, L)
            for m in range(n):
                s += gl.amd.cdna4.buffer_load(ptr=base + m * slot_elems + t, offsets=offs, cache=LOAD)
            nbad += (s != expect).to(gl.int32)
        bad = gl.sum(nbad, axis=0)
        gl.store(cnt_ptr + group * 32, 0)
        t3 = _stamp()
    z = _one()
    rec = rec_ptr + pid * 8 + z
    gl.store(rec, t0 + z)
    gl.store(rec + 1, t1 + z)
    gl.store(rec + 2, t2 + z)
    gl.store(rec + 3, t3 + z)
    gl.store(rec + 4, bad.to(gl.int64) + z)
    gl.store(rec + 5, last.to(gl.int64) + z)
    gl.store(rec + 6, (_getreg("HW_REG_XCC_ID") & 15).to(gl.int64) + z)
    gl.store(rec + 7, group.to(gl.int64) + z + gl.sum(lds.slice(0, 64 * gl.num_warps()).load(L).to(gl.int64),
                                                       axis=0) * 0)


# ---------------------------------------------------------------------------------------------
# bits: the 256 KiB store alone, each set of cache bits
# ---------------------------------------------------------------------------------------------


@gluon.jit
def store_kernel(dst_addr, cnt_ptr, rec_ptr, n_active, BITS: gl.constexpr):
    """Programs with slot < n_active (n_active per XCD) store 256 KiB each with dwordx2 asm stores and
    BITS, then drain; the time from the start stamp to the drain."""
    L: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    pid = gl.program_id(0)
    slot = pid // 8
    _lds_hold(gl.BlockedLayout([4], [64], [gl.num_warps()], [0]))
    lane = gl.arange(0, 64 * gl.num_warps(), L)
    _grid_sync(cnt_ptr)
    t0 = _stamp()
    if slot < n_active:
        base = dst_addr + pid.to(gl.int64) * (_SLOT * 4) + lane.to(gl.int64) * 8
        val = lane.to(gl.int64) * 0 + 1
        for r in range(_SLOT * 4 // (8 * 64 * gl.num_warps())):
            gl.inline_asm_elementwise("global_store_dwordx2 $1, $2, off " + BITS + "\nv_mov_b32 $0, 0",
                                      "=v,v,v,~{memory}", [base + r * (8 * 64 * gl.num_warps()), val],
                                      dtype=gl.int32, is_pure=False, pack=1)
    t1 = _stamp()
    gl.store(rec_ptr + pid + _one(), (t1 - t0) + _one().to(gl.int64) * 0)


# ---------------------------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------------------------


class Bufs:

    def __init__(self, device="cuda"):
        self.buf = torch.zeros(64 * MIB // 4, device=device, dtype=torch.float32)
        self.p = torch.zeros(8 * 16 * SLOT, device=device, dtype=torch.float32)
        self.dirty = torch.zeros(64 * MIB // 4, device=device, dtype=torch.float32)
        self.st = torch.zeros(NUM_PROGRAMS * SLOT, device=device, dtype=torch.float32)
        self.flags = torch.zeros(8 * 16 * 32, device=device, dtype=torch.int32)
        self.cnt = torch.zeros(8, device=device, dtype=torch.int32)
        self.rec = torch.zeros(NUM_PROGRAMS * 8, device=device, dtype=torch.int64)
        self.acnt = torch.zeros(NUM_PROGRAMS * 32, device=device, dtype=torch.int32)
        self.flush_buf = torch.ones(1024 * MIB // 4, device=device, dtype=torch.float32)
        self.epoch = 100
        self.warm = False

    def flush(self):
        self.flush_buf.sum()


def med(xs):
    return statistics.median(xs) if xs else float("nan")


OP_NAMES = {0: "none", 1: "`buffer_wbl2 sc1`", 2: "`buffer_inv sc1`"}


def run_op(b, mode, op, mib, all_=False, launches=20):
    """Median us of the operation on slot 0 (or every program with all_), and of the helpers' re-read."""
    elems = max(1024, mib * MIB // 4 // 31 // 1024 * 1024)
    ops, rereads = [], []
    for i in range(launches + 1):
        b.flush()
        b.cnt.zero_()
        cache_op_kernel[(NUM_PROGRAMS, )](b.buf, b.cnt, b.rec, elems, MODE=mode, OP=op, ALL=all_, num_warps=4)
        torch.cuda.synchronize()
        if i == 0:
            continue
        r = b.rec.view(NUM_PROGRAMS, 8).cpu()
        slot = torch.arange(NUM_PROGRAMS) // 8
        sel = slot == 0 if not all_ else slot >= 0
        ops += (r[sel, 0].double() / 100).tolist()
        rereads += (r[slot > 0, 1].double() / 100).tolist()
    return med(ops), med(rereads), elems * 4 * 31 / MIB


def sweep_ops(b):
    print("Every XCD at once: its 31 helper CUs first write (dirty) or twice read (valid) the given MiB in total, "
          "then slot 0 (or all 32 programs) issues the operation and `s_waitcnt vmcnt(0)`, then the helpers read "
          "their data again. us, medians over 20 launches x 8 XCDs; a 1 GiB read before each launch.\n")
    print("| XCD's lines | MiB per XCD | operation | issued by | op us | helpers' re-read us | re-read vs none |")
    print("|---|---|---|---|---|---|---|")
    for mode, what in ((0, "dirty"), (1, "valid")):
        for mib in (0, 1, 2, 4):
            base = None
            for op in (0, 1, 2):
                for all_ in ((False, True) if op else (False, )):
                    o, rr, real = run_op(b, mode, op, mib, all_)
                    if op == 0:
                        base = rr
                    print(f"| {what} | {real:.2f} | {OP_NAMES[op]} | {'all 32 CUs' if all_ else 'one CU'} | "
                          f"{o:.2f} | {rr:.2f} | {rr - base:+.2f} |", flush=True)


HANDSHAKES = [
    # name, place, STORE, LOAD, WF, RF, FLAG
    ("`.wt` stores, drain, atomic flag, `.cv` reads (v21 `wt`)", 2, ".wt", ".cv", 1, 0, 0),
    ("`.wt` stores, drain, atomic flag, `.cg` reads", 2, ".wt", ".cg", 1, 0, 0),
    ("`.wt` stores, drain, `.wt` flag polled with `.cv`, `.cv` reads (reference kernel)", 2, ".wt", ".cv", 1, 0, 2),
    ("default stores, agent release, atomic flag, `.cg` reads", 2, "", ".cg", 2, 0, 0),
    ("default stores, agent release, atomic flag, agent acquire, default reads", 2, "", "", 2, 1, 0),
    ("default stores, drain, atomic flag, `.cg` reads", 1, "", ".cg", 1, 0, 0),
    ("default stores, drain, plain flag polled with `.cg`, `.cg` reads (v21 `l2`)", 1, "", ".cg", 1, 0, 1),
    ("default stores, drain, atomic flag, agent acquire, default reads", 1, "", "", 1, 1, 0),
    ("`.wt` stores, drain, atomic flag, `.cv` reads", 1, ".wt", ".cv", 1, 0, 0),
    ("`.wt` stores, drain, `.wt` flag polled with `.cv`, `.cv` reads", 1, ".wt", ".cv", 1, 0, 2),
]


def run_handshake(b, place, store, load, wf, rf, dirty_mib, nw=1, flag=0, launches=30):
    n_dirty = 16 if dirty_mib else 0
    dirty_elems = dirty_mib * MIB // 4 // 16 if dirty_mib else 1024
    pub, flg, read, total, bad, timeouts, xccs = [], [], [], [], 0, 0, set()
    for i in range(launches + 1):
        if not b.warm:
            b.flush()
        b.cnt.zero_()
        b.epoch += 1
        b.last_k = handshake_kernel[(NUM_PROGRAMS, )](b.p, b.flags, b.cnt, b.rec, b.dirty, b.epoch, place,
                                                      dirty_elems, n_dirty, nw, 50000, STORE=store, LOAD=load, WF=wf,
                                                      RF=rf, FLAG=flag, num_warps=4)
        torch.cuda.synchronize()
        if i == 0:
            continue
        r = b.rec.view(NUM_PROGRAMS, 8).cpu()
        for x in range(8):
            for j in range(nw):
                w = r[j * 8 + x]
                rd = r[(nw + j) * 8 + ((x + 1) % 8 if place == 2 else x)]
                if rd[7]:
                    timeouts += 1
                    continue
                pub.append((w[1] - w[0]).item() / 100)
                flg.append((rd[2] - w[1]).item() / 100)
                read.append((rd[3] - rd[2]).item() / 100)
                total.append((rd[3] - w[0]).item() / 100)
                bad += int(rd[4])
                xccs.add((int(w[5]), int(rd[5])))
    ok = all((wx != rx) == (place == 2) for wx, rx in xccs)
    note = f"{bad}" + (f", {timeouts} timeouts" if timeouts else "") + ("" if ok else " (PLACEMENT WRONG)")
    return med(pub), med(flg), med(read), med(total), note


def cache_note(b):
    return ("back to back, no flush between launches (warm)" if b.warm else
            "a 1 GiB read before each launch (cold)")


def sweep_handshake(b):
    print("256 KiB partials published by `writers` CUs of every XCD at once and each consumed by one CU of the next "
          "XCD (other XCD) or of the same XCD. Writers and readers start at the same stamp. us, medians over 30 "
          f"launches x all pairs; {cache_note(b)}. 'dirty' = 16 other CUs of every XCD wrote that many MiB in total "
          "(default stores) just before. 'bad' counts wrong values the readers saw (all launches). The atomic flag "
          "is a relaxed `gpu`-scope swap polled with relaxed `gpu`-scope atomic loads.\n")
    print("| sequence | placement | writers per XCD | dirty MiB per XCD | publish us | flag seen after us | read us | "
          "total us | bad |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, place, store, load, wf, rf, flag in HANDSHAKES:
        for nw, dirty in ((1, 0), (1, 2), (16, 0)):
            p, f, r, t, bad = run_handshake(b, place, store, load, wf, rf, dirty, nw=nw, flag=flag)
            print(f"| {name} | {'other XCD' if place == 2 else 'same XCD'} | {nw} | {dirty} | {p:.2f} | {f:.2f} | "
                  f"{r:.2f} | {t:.2f} | {bad} |", flush=True)


HANDSHAKES_RELEASE = [
    # name, place, STORE, LOAD, WF, RF, FLAG
    ("`.wt` stores, drain, atomic flag, `.cv` reads", 2, ".wt", ".cv", 1, 0, 0),
    ("default stores, drain, release flag (compiler), `.cg` reads", 2, "", ".cg", 1, 0, 3),
    ("default stores, drain, release flag, acquire (compiler), default reads", 2, "", "", 1, 2, 3),
    ("default stores, `buffer_wbl2 sc1` in every wave, atomic flag, `.cg` reads", 2, "", ".cg", 2, 0, 0),
    ("default stores, drain, atomic flag, acquire (compiler), default reads", 1, "", "", 1, 2, 0),
]


def sweep_handshake_release(b):
    """The handshake with the compiler's own release and acquire (one per CU) instead of fences in every wave."""
    print("As the handshake sweep, with the release and acquire the compiler emits for `sem=\"release\"` on the flag "
          "swap and `sem=\"acquire\"` on one atomic load after the spin, instead of `buffer_wbl2 sc1` / "
          "`buffer_inv sc1` written in every wave. Every wave still drains and meets the others at `s_barrier` "
          f"before the flag. us, medians over 30 launches x all pairs; {cache_note(b)}. 'fences in the code' counts "
          "the `buffer_wbl2` / `buffer_inv` instructions in the compiled kernel. The compiler's sit in the block "
          "that only the program's first lane executes, so they run once per CU; the hand-written one runs in all "
          "4 waves.\n")
    print("| sequence | placement | writers per XCD | publish us | flag seen after us | read us | total us | bad | "
          "fences in the code |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, place, store, load, wf, rf, flag in HANDSHAKES_RELEASE:
        for nw in (1, 16):
            p, f, r, t, bad = run_handshake(b, place, store, load, wf, rf, 0, nw=nw, flag=flag)
            asm = b.last_k.asm["amdgcn"].splitlines()
            wbl2 = sum(1 for x in asm if x.strip().startswith("buffer_wbl2"))
            inv = sum(1 for x in asm if x.strip().startswith("buffer_inv"))
            print(f"| {name} | {'other XCD' if place == 2 else 'same XCD'} | {nw} | {p:.2f} | {f:.2f} | {r:.2f} | "
                  f"{t:.2f} | {bad} | {wbl2} `buffer_wbl2`, {inv} `buffer_inv` |", flush=True)


ARRIVES = [
    # name, place, STORE, LOAD, WF, ARR, status
    ("aiter: default stores, drain, `acq_rel` `cta` arrival, `.cv` sum", 1, "", ".cv", 1, 1, "not guaranteed"),
    ("default stores, drain, relaxed `cta` arrival, `.cv` sum", 1, "", ".cv", 1, 3, "not guaranteed"),
    ("default stores, drain, relaxed `gpu` arrival, `.cv` sum", 1, "", ".cv", 1, 0, "one XCD, checked placement"),
    ("default stores, drain, relaxed `gpu` arrival, `.cg` sum", 1, "", ".cg", 1, 0, "one XCD, checked placement"),
    ("`.wt` stores, drain, relaxed `gpu` arrival, `.cv` sum", 1, ".wt", ".cv", 1, 0, "any placement"),
    ("default stores, drain, `acq_rel` `gpu` arrival, default sum", 1, "", "", 1, 2, "documented"),
    ("`.wt` stores, drain, relaxed `gpu` arrival, `.cv` sum", 2, ".wt", ".cv", 1, 0, "any placement"),
    ("default stores, drain, `acq_rel` `gpu` arrival, default sum", 2, "", "", 1, 2, "documented"),
    ("`.wt` stores, drain, `acq_rel` `cta` arrival, `.cv` sum", 2, ".wt", ".cv", 1, 1, "not guaranteed"),
    ("aiter's sequence with the group spread over XCDs", 2, "", ".cv", 1, 1, "negative"),
]


def run_arrive(b, place, store, load, wf, arr, n, slot_elems, launches=30):
    pub, seen, summ, total = [], [], [], []
    bad = broken = misplaced = 0
    groups = NUM_PROGRAMS // n
    for i in range(launches + 1):
        if not b.warm:
            b.flush()
        b.cnt.zero_()
        b.acnt.zero_()
        b.epoch += 1
        arrive_kernel[(NUM_PROGRAMS, )](b.st, b.cnt, b.acnt, b.rec, b.epoch, n, place, slot_elems, STORE=store,
                                        LOAD=load, WF=wf, ARR=arr, num_warps=4)
        torch.cuda.synchronize()
        if i == 0:
            continue
        r = b.rec.view(NUM_PROGRAMS, 8).cpu()
        for g in range(groups):
            mem = r[r[:, 7] == g]
            lasts = mem[mem[:, 5] == 1]
            nx = len(set(mem[:, 6].tolist()))
            if nx != (1 if place == 1 else min(n, 8)):
                misplaced += 1
            if len(mem) != n or len(lasts) != 1:
                broken += 1
                continue
            last = lasts[0]
            t0 = mem[:, 0].min()
            pub += ((mem[:, 1] - mem[:, 0]).double() / 100).tolist()
            seen.append((last[2] - t0).item() / 100)
            summ.append((last[3] - last[2]).item() / 100)
            total.append((last[3] - t0).item() / 100)
            bad += int(last[4] > 0)
    checks = f"{bad}" + (f", {broken} groups without one last arriver" if broken else "") + (
        f", {misplaced} misplaced" if misplaced else "")
    return med(pub), med(seen), med(summ), med(total), checks, launches * groups


def sweep_arrive(b):
    print("Last-arriver reductions over all 256 CUs: groups of n programs, each storing a partial of the given size, "
          "draining, meeting its workgroup at `s_barrier` and adding 1 to the group's counter; the program that "
          "sees n - 1 sums the n partials. 'One XCD': each group on one XCD (as aiter's `_tile_on_xcd`); 'spread': "
          f"n consecutive pids on min(n, 8) XCDs. us, medians over 30 launches x groups; {cache_note(b)}. "
          "'last arrival' is from the common start stamp to the last member's atomic returning; 'sum' is the last "
          "member reading and adding the n partials. 'wrong' counts groups whose sum was wrong (all launches).\n")
    print("| sequence | placement | status | n | partial KiB | publish us | last arrival us | sum us | total us | "
          "wrong / groups |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, place, store, load, wf, arr, status in ARRIVES:
        for n in (4, 16):
            for elems in (2048, SLOT):
                p, a, s, t, chk, ng = run_arrive(b, place, store, load, wf, arr, n, elems)
                print(f"| {name} | {'one XCD' if place == 1 else 'spread'} | {status} | {n} | {elems * 4 // 1024} | "
                      f"{p:.2f} | {a:.2f} | {s:.2f} | {t:.2f} | {chk} / {ng} |", flush=True)


def sweep_bits(b, launches=20):
    print("256 KiB per CU with inline-asm `global_store_dwordx2` (8 bytes per lane), then a drain. One CU per XCD, "
          "or 16 per XCD at once. us from the start stamp to the drain, medians over launches x CUs.\n")
    print("| bits | CUs per XCD | us | GB/s per CU |")
    print("|---|---|---|---|")
    for n in (1, 16):
        for bits in ("", "sc0 nt", "sc1", "sc0 sc1"):
            ts = []
            for i in range(launches + 1):
                b.flush()
                b.cnt.zero_()
                store_kernel[(NUM_PROGRAMS, )](b.st.data_ptr(), b.cnt, b.rec, n, BITS=bits, num_warps=4)
                torch.cuda.synchronize()
                if i == 0:
                    continue
                r = b.rec[:NUM_PROGRAMS].cpu()
                slot = torch.arange(NUM_PROGRAMS) // 8
                ts += (r[slot < n].double() / 100).tolist()
            us = med(ts)
            print(f"| {('`' + bits + '`') if bits else 'none'} | {n} | {us:.2f} | {SLOT * 4 / us / 1e3:.0f} |", flush=True)


SWEEPS = {"ops": sweep_ops, "handshake": sweep_handshake, "handshake_release": sweep_handshake_release,
          "arrive": sweep_arrive, "bits": sweep_bits}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", nargs="+", default=list(SWEEPS), choices=list(SWEEPS))
    p.add_argument("--warm", action="store_true", help="handshake / arrive: no 1 GiB read between launches")
    args = p.parse_args()
    print(f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', 'unset')}, {torch.cuda.get_device_name()}, "
          f"triton {triton.__version__}, {time.strftime('%Y-%m-%d %H:%M')}")
    b = Bufs()
    b.warm = args.warm
    for s in args.sweep:
        print(f"\n## {s}\n", flush=True)
        SWEEPS[s](b)


if __name__ == "__main__":
    main()
