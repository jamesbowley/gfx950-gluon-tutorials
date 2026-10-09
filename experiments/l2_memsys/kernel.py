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
"""Microbenchmarks of the gfx950 L2: retention by cache policy, reads of lines written into L2,
where a read is served from, and the write-back of dirty lines at kernel end.

Three kernels:

- memsys_kernel (M1, M2). The active programs (one XCD's 32, or all 256) run a fixed sequence
  of phases, each switched on by a constexpr and each followed by a drain and a grid sync:
  PRE (read one dword of every line of the program's own region), W (write the region with
  cache modifier STORE and pattern WPAT, twice if TWICE), R1 (read a region with LOAD1), P
  (read a separate pollution buffer), R2 (read a region with LOAD2). Reads cover every byte of
  the region of the program ROT places further on, so with ROT = 1 another CU reads what a CU
  wrote. A timestamp after every sync gives each phase's time.
- chase_kernel (M3). Program 0 (one wave, on XCD 0) follows a chain of dependent loads, one
  per 128-byte line, and times them. Before it starts, helper programs set the chain up, in
  the same launch, according to MODE (see CHASE_MODES).
- dirty_kernel, follow_kernel (M4). Every program writes its share of a buffer with STORE and
  stamps its first and last instruction; a tiny follower runs right after.

Timestamps are s_memrealtime (100 MHz, one clock for the device), taken after s_waitcnt
vmcnt(0) and a workgroup barrier.
"""

import triton  # before torch: torch's HIP pulls in ROCm's shared libLLVM, which breaks libtriton's
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

NUM_PROGRAMS = gl.constexpr(256)
NUM_XCDS = gl.constexpr(8)
TILE = gl.constexpr(4096)  # fp32 per memsys loop trip (16 KiB)
REC = gl.constexpr(8)  # int64 stamps per program

WPATS = {"full": 0, "half": 1, "dword": 2}
RPATS = {"full": 0, "first_half": 1, "second_half": 2, "dword": 3}
CHASE_MODES = {
    "hbm": 0,  # nothing; the host evicts L2 and the LLC first
    "l2": 1,  # the chaser reads the chain once first, so it is in its XCD's L2
    "llc": 2,  # XCD 1 reads the chain first: it is in the LLC, not in XCD 0's L2
    "written": 3,  # XCD 0's other programs write the chain (no cache bits), full lines
    "written_pre": 4,  # as written, after reading every line of the chain first
    "written_wt": 5,  # XCD 0's other programs write the chain with .wt
    "remote_wt": 6,  # XCD 1 writes the chain with .wt (v20's path); chase with .cv
    "written_cs": 7,  # XCD 0's other programs write the chain with .cs (sc0 nt)
    "llc_written": 8,  # XCD 1 reads the chain (old copy now in the LLC), then XCD 0's others write it
}


@gluon.jit
def _one():
    return gl.zeros([1], gl.int32, gl.BlockedLayout([1], [64], [gl.num_warps()], [0]))


@gluon.jit
def _asm(code: gl.constexpr):
    gl.inline_asm_elementwise(code + "\nv_mov_b32 $0, $1", "=v,v,~{memory}", [_one()], dtype=gl.int32,
                              is_pure=False, pack=1)


@gluon.jit
def _stamp():
    return gl.inline_asm_elementwise(
        "s_waitcnt vmcnt(0)\ns_barrier\ns_memrealtime $0\ns_waitcnt lgkmcnt(0)", "=s,v,~{memory}", [_one()],
        dtype=gl.int64, is_pure=False, pack=1,
    )


@gluon.jit
def _xcc():
    x = gl.inline_asm_elementwise("s_getreg_b32 $0, hwreg(HW_REG_XCC_ID)", "=s,v", [_one()], dtype=gl.int32,
                                  is_pure=True, pack=1)
    return (x & 15).to(gl.int64)


@gluon.jit
def _keep(x):
    """Side-effecting no-op on every register of x: the loads that feed x complete here and are
    not removed."""
    gl.inline_asm_elementwise("; keep", "=v,0", [x], dtype=x.dtype, is_pure=False, pack=1)


@gluon.jit
def _grid_sync(cnt_ptr, n):
    """All n participants arrive (relaxed device-scope atomics, performed past L2), then go on.
    The caller has drained its own memory operations. The host zeroes the counter."""
    _asm("s_waitcnt vmcnt(0)\ns_barrier")
    gl.atomic_add(cnt_ptr, 1, sem="relaxed", scope="gpu")
    while gl.atomic_add(cnt_ptr, 0, sem="relaxed", scope="gpu") < n:
        pass
    _asm("s_nop 0")


@gluon.jit
def _write_region(base, n_elems, val, STORE: gl.constexpr, WPAT: gl.constexpr):
    L: gl.constexpr = gl.BlockedLayout([4], [64], [gl.num_warps()], [0])
    for t in range(0, n_elems, TILE):
        if WPAT == 0:
            # every byte: each wave instruction covers 1 KiB, 8 whole lines
            offs = gl.arange(0, TILE, L)
            gl.amd.cdna4.buffer_store(stored_value=gl.zeros([TILE], gl.float32, L) + val, ptr=base + t, offsets=offs,
                                      cache=STORE)
        elif WPAT == 1:
            # the first 64 bytes of every 128-byte line (idx plus a term constant over 16)
            idx = gl.arange(0, TILE // 2, L)
            offs = idx + (idx // 16) * 16
            gl.amd.cdna4.buffer_store(stored_value=gl.zeros([TILE // 2], gl.float32, L) + val, ptr=base + t,
                                      offsets=offs, cache=STORE)
        else:
            # one dword per line
            L1: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
            idx = gl.arange(0, TILE // 32, L1)
            gl.amd.cdna4.buffer_store(stored_value=gl.zeros([TILE // 32], gl.float32, L1) + val, ptr=base + t,
                                      offsets=idx * 32, cache=STORE)


@gluon.jit
def _read_region(base, n_elems, LOAD: gl.constexpr, REV: gl.constexpr = False, RPAT: gl.constexpr = 0):
    """RPAT 0: every byte, front to back, or back to front (newest written first) with REV.
    RPAT 1 / 2: only the first / second 64 bytes of every line. RPAT 3: one dword per line."""
    L: gl.constexpr = gl.BlockedLayout([4], [64], [gl.num_warps()], [0])
    if RPAT == 0:
        s = gl.zeros([TILE], gl.float32, L)
        offs = gl.arange(0, TILE, L)
        for t in range(0, n_elems, TILE):
            if REV:
                s += gl.amd.cdna4.buffer_load(ptr=base + (n_elems - TILE - t), offsets=offs, cache=LOAD)
            else:
                s += gl.amd.cdna4.buffer_load(ptr=base + t, offsets=offs, cache=LOAD)
    elif RPAT == 3:
        L1: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
        s = gl.zeros([TILE // 32], gl.float32, L1)
        idx = gl.arange(0, TILE // 32, L1) * 32
        for t in range(0, n_elems, TILE):
            s += gl.amd.cdna4.buffer_load(ptr=base + t, offsets=idx, cache=LOAD)
    else:
        s = gl.zeros([TILE // 2], gl.float32, L)
        idx = gl.arange(0, TILE // 2, L)
        offs = idx + (idx // 16) * 16 + (RPAT - 1) * 16
        for t in range(0, n_elems, TILE):
            s += gl.amd.cdna4.buffer_load(ptr=base + t, offsets=offs, cache=LOAD)
    return gl.sum(s, axis=0)


@gluon.jit
def _touch_region(base, n_elems):
    """One dword of every 128-byte line."""
    L1: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    s = gl.zeros([TILE // 32], gl.float32, L1)
    idx = gl.arange(0, TILE // 32, L1)
    for t in range(0, n_elems, TILE):
        s += gl.amd.cdna4.buffer_load(ptr=base + t, offsets=idx * 32)
    return gl.sum(s, axis=0)


@gluon.jit
def memsys_kernel(
    dst_ptr,
    pol_ptr,
    out_ptr,
    cnt_ptr,
    rec_ptr,
    chunk,
    pol_chunk,
    ACTIVE_XCD: gl.constexpr,
    PRE: gl.constexpr,
    W: gl.constexpr,
    R1: gl.constexpr,
    P: gl.constexpr,
    R2: gl.constexpr,
    STORE: gl.constexpr,
    LOAD1: gl.constexpr,
    LOAD2: gl.constexpr,
    WPAT: gl.constexpr,
    TWICE: gl.constexpr,
    ROT: gl.constexpr,
    REV: gl.constexpr,
    RPAT: gl.constexpr,
):
    pid = gl.program_id(0)
    if ACTIVE_XCD >= 0:
        active = pid % 8 == ACTIVE_XCD
        local = pid // 8
        n_act: gl.constexpr = NUM_PROGRAMS // NUM_XCDS
    else:
        active = pid >= 0
        local = pid
        n_act: gl.constexpr = NUM_PROGRAMS
    if active:
        own = dst_ptr + local.to(gl.int64) * chunk
        other = dst_ptr + ((local + ROT) % n_act).to(gl.int64) * chunk
        acc = gl.zeros([1], gl.float32, gl.BlockedLayout([1], [64], [gl.num_warps()], [0]))
        rec = rec_ptr + pid * REC + _one()
        gl.store(rec + 7, _xcc())
        _grid_sync(cnt_ptr, n_act)
        gl.store(rec, _stamp())
        if PRE:
            acc += _touch_region(own, chunk)
        _grid_sync(cnt_ptr + 1, n_act)
        gl.store(rec + 1, _stamp())
        if W:
            _write_region(own, chunk, local.to(gl.float32) + 1.0, STORE, WPAT)
            if TWICE:
                _write_region(own, chunk, local.to(gl.float32) + 1.0, STORE, WPAT)
        _grid_sync(cnt_ptr + 2, n_act)
        gl.store(rec + 2, _stamp())
        if R1:
            acc += _read_region(other, chunk, LOAD1)
        _grid_sync(cnt_ptr + 3, n_act)
        gl.store(rec + 3, _stamp())
        if P:
            acc += _read_region(pol_ptr + local.to(gl.int64) * pol_chunk, pol_chunk, "")
        _grid_sync(cnt_ptr + 4, n_act)
        gl.store(rec + 4, _stamp())
        if R2:
            acc += _read_region(other, chunk, LOAD2, REV, RPAT)
        _grid_sync(cnt_ptr + 5, n_act)
        gl.store(rec + 5, _stamp())
        gl.store(out_ptr + pid + _one() * 0, acc)


@gluon.jit
def _copy_lines(dst, src, lo, hi, STORE: gl.constexpr, PRE: gl.constexpr):
    """Copy int32 elements [lo, hi) from src to dst with STORE, whole lines; PRE reads one dword
    of every destination line first."""
    L: gl.constexpr = gl.BlockedLayout([4], [64], [gl.num_warps()], [0])
    offs = gl.arange(0, 1024, L)
    if PRE:
        L1: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
        idx = gl.arange(0, 32, L1) * 32
        s = gl.zeros([32], gl.int32, L1)
        for t in range(lo, hi, 1024):
            s += gl.amd.cdna4.buffer_load(ptr=dst + t, offsets=idx, mask=t + idx < hi, other=0)
        _keep(s)
    for t in range(lo, hi, 1024):
        v = gl.amd.cdna4.buffer_load(ptr=src + t, offsets=offs, mask=t + offs < hi, other=0)
        gl.amd.cdna4.buffer_store(stored_value=v, ptr=dst + t, offsets=offs, mask=t + offs < hi, cache=STORE)


@gluon.jit
def chase_kernel(
    chain_ptr,
    src_ptr,
    out_ptr,
    cnt_ptr,
    rec_ptr,
    n_elems,
    steps,
    start,
    MODE: gl.constexpr,
    CHASE_LOAD: gl.constexpr,
):
    """Program 0 chases; the helpers are XCD 0's other 31 programs or XCD 1's 32."""
    pid = gl.program_id(0)
    xcd = pid % 8
    local = pid // 8
    LOCAL_HELPERS: gl.constexpr = MODE == 3 or MODE == 4 or MODE == 5 or MODE == 7
    REMOTE_HELPERS: gl.constexpr = MODE == 2 or MODE == 6
    if LOCAL_HELPERS:
        n_help: gl.constexpr = 31
        is_helper = (xcd == 0) & (pid != 0)
        h = local - 1
    elif REMOTE_HELPERS:
        n_help: gl.constexpr = 32
        is_helper = xcd == 1
        h = local
    elif MODE == 8:
        n_help: gl.constexpr = 31  # the writers; the readers are ordered through cnt[1]
        is_helper = pid < 0
        h = local
    else:
        n_help: gl.constexpr = 0
        is_helper = pid < 0
        h = local
    one = _one()
    if MODE == 8:
        # Two stages: XCD 1 reads the chain (into the LLC, not into XCD 0's L2), then XCD 0's
        # other programs write it (no cache bits). cnt[1] orders the stages.
        lines8 = n_elems // 32
        if xcd == 1:
            lo = (local * lines8 // 32) * 32
            hi = ((local + 1) * lines8 // 32) * 32
            L8: gl.constexpr = gl.BlockedLayout([4], [64], [gl.num_warps()], [0])
            offs8 = gl.arange(0, 1024, L8)
            s8 = gl.zeros([1024], gl.int32, L8)
            for t in range(lo, hi, 1024):
                s8 += gl.amd.cdna4.buffer_load(ptr=chain_ptr + t, offsets=offs8, mask=t + offs8 < hi, other=0)
            gl.store(out_ptr + pid + one * 0, gl.sum(s8, axis=0) + one * 0)
            _asm("s_waitcnt vmcnt(0)\ns_barrier")
            gl.atomic_add(cnt_ptr + 1, 1, sem="relaxed", scope="gpu")
        if (xcd == 0) & (pid != 0):
            while gl.atomic_add(cnt_ptr + 1, 0, sem="relaxed", scope="gpu") < 32:
                pass
            _asm("s_nop 0")
            lo = ((local - 1) * lines8 // 31) * 32
            hi = (local * lines8 // 31) * 32
            _copy_lines(chain_ptr, src_ptr, lo, hi, "", False)
            _asm("s_waitcnt vmcnt(0)\ns_barrier")
            gl.atomic_add(cnt_ptr, 1, sem="relaxed", scope="gpu")
    if is_helper:
        lines = n_elems // 32
        lo = (h * lines // n_help) * 32
        hi = ((h + 1) * lines // n_help) * 32
        if MODE == 2:
            L: gl.constexpr = gl.BlockedLayout([4], [64], [gl.num_warps()], [0])
            offs = gl.arange(0, 1024, L)
            s = gl.zeros([1024], gl.int32, L)
            for t in range(lo, hi, 1024):
                s += gl.amd.cdna4.buffer_load(ptr=chain_ptr + t, offsets=offs, mask=t + offs < hi, other=0)
            gl.store(out_ptr + pid + one * 0, gl.sum(s, axis=0) + one * 0)
        elif MODE == 3:
            _copy_lines(chain_ptr, src_ptr, lo, hi, "", False)
        elif MODE == 4:
            _copy_lines(chain_ptr, src_ptr, lo, hi, "", True)
        elif MODE == 7:
            _copy_lines(chain_ptr, src_ptr, lo, hi, ".cs", False)
        else:
            _copy_lines(chain_ptr, src_ptr, lo, hi, ".wt", False)
        _asm("s_waitcnt vmcnt(0)\ns_barrier")
        gl.atomic_add(cnt_ptr, 1, sem="relaxed", scope="gpu")
    if pid == 0:
        rec = rec_ptr + _one()
        gl.store(rec + 3, _xcc())
        if MODE == 1:
            L: gl.constexpr = gl.BlockedLayout([4], [64], [gl.num_warps()], [0])
            offs = gl.arange(0, 1024, L)
            s = gl.zeros([1024], gl.int32, L)
            for t in range(0, n_elems, 1024):
                s += gl.amd.cdna4.buffer_load(ptr=chain_ptr + t, offsets=offs, mask=t + offs < n_elems, other=0)
            gl.store(out_ptr + 1 + one * 0, gl.sum(s, axis=0) + one * 0)
        while gl.atomic_add(cnt_ptr, 0, sem="relaxed", scope="gpu") < n_help:
            pass
        # Compiler fence only. The chase's sc0 nt loads miss L1 (ISA Table 49); the one
        # default-load case reads lines this CU never read before.
        _asm("s_nop 0")
        idx = one * 0 + start
        t0 = _stamp()
        for _ in range(steps):
            idx = gl.amd.cdna4.buffer_load(ptr=chain_ptr, offsets=idx * 32, cache=CHASE_LOAD)
        t1 = _stamp()
        gl.store(rec, t0)
        gl.store(rec + 1, t1)
        gl.store(rec + 2, idx.to(gl.int64))


@gluon.jit
def dirty_kernel(dst_ptr, rec_ptr, chunk, STORE: gl.constexpr):
    L: gl.constexpr = gl.BlockedLayout([4], [64], [gl.num_warps()], [0])
    pid = gl.program_id(0)
    rec = rec_ptr + pid * 2 + _one()
    t0 = _stamp()
    base = dst_ptr + pid.to(gl.int64) * chunk
    offs = gl.arange(0, 1024, L)
    val = gl.zeros([1024], gl.float32, L) + pid.to(gl.float32)
    for t in range(0, chunk, 1024):
        gl.amd.cdna4.buffer_store(stored_value=val, ptr=base + t, offsets=offs, cache=STORE)
    t1 = _stamp()
    gl.store(rec, t0)
    gl.store(rec + 1, t1)


@gluon.jit
def follow_kernel(rec_ptr):
    pid = gl.program_id(0)
    gl.store(rec_ptr + pid + _one(), _stamp())


# ---------------------------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------------------------

MIB = 2**20


class Buffers:
    def __init__(self, dst_mib=64, pol_mib=64, device="cuda"):
        self.dst = torch.zeros(dst_mib * MIB // 4, device=device, dtype=torch.float32)
        self.pol = torch.ones(pol_mib * MIB // 4, device=device, dtype=torch.float32)
        self.out = torch.zeros(int(NUM_PROGRAMS), device=device, dtype=torch.float32)
        self.cnt = torch.zeros(16, device=device, dtype=torch.int32)
        self.rec = torch.zeros(int(NUM_PROGRAMS) * int(REC), device=device, dtype=torch.int64)
        self.flush_buf = torch.ones(1024 * MIB // 4, device=device, dtype=torch.float32)

    def flush(self):
        """Read 1 GiB: evicts L2 and the 256 MB LLC, leaves both clean."""
        self.flush_buf.sum()


def memsys(bufs, f_mib, e_mib=0, xcd=0, pre=False, w=True, r1=False, p=False, r2=True, store="", load1="",
           load2="", wpat="full", twice=False, rot=1, rev=False, rpat="full"):
    """One memsys_kernel launch. f_mib: the region per XCD (xcd >= 0) or in total (xcd = -1);
    e_mib: the pollution read per XCD (or in total); rev: the second read goes back to front;
    rpat: what the second read covers (full, first_half, second_half, dword)."""
    n_act = 32 if xcd >= 0 else int(NUM_PROGRAMS)
    chunk = f_mib * MIB // 4 // n_act
    pol_chunk = e_mib * MIB // 4 // n_act
    assert chunk % int(TILE) == 0 and pol_chunk % int(TILE) == 0, (chunk, pol_chunk)
    assert chunk * n_act <= bufs.dst.numel() and pol_chunk * n_act <= bufs.pol.numel()
    bufs.cnt.zero_()
    return memsys_kernel[(int(NUM_PROGRAMS), )](
        bufs.dst, bufs.pol, bufs.out, bufs.cnt, bufs.rec, chunk, max(pol_chunk, int(TILE)),
        ACTIVE_XCD=xcd, PRE=pre, W=w, R1=r1, P=p and e_mib > 0, R2=r2, STORE=store, LOAD1=load1, LOAD2=load2,
        WPAT=WPATS[wpat], TWICE=twice, ROT=rot, REV=rev, RPAT=RPATS[rpat], num_warps=4,
    )


def memsys_phases(bufs, xcd=0):
    """Per phase (pre, write, read1, pollute, read2): the median over active programs of the
    time from the previous sync to this one, in us; and the XCC ids of the active programs."""
    r = bufs.rec.view(int(NUM_PROGRAMS), int(REC)).cpu()
    rows = [p for p in range(int(NUM_PROGRAMS)) if xcd < 0 or p % 8 == xcd]
    t = r[rows, :6].double() * 0.01
    ph = (t[:, 1:] - t[:, :-1]).median(dim=0).values.tolist()
    return ph, sorted(set(int(x) for x in r[rows, 7]))


class Chain:
    """A random cycle over n_lines 128-byte lines: element 32 l holds the next line."""

    def __init__(self, n_lines=8192, seed=0, device="cuda"):
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n_lines, generator=g)
        nxt = torch.empty(n_lines, dtype=torch.int32)
        nxt[perm] = torch.roll(perm, -1).to(torch.int32)
        data = torch.zeros(n_lines * 32, dtype=torch.int32)
        data[0::32] = nxt
        self.n_lines = n_lines
        self.start = int(perm[0])
        self.src = data.to(device)
        self.chain = data.to(device)
        self.out = torch.zeros(int(NUM_PROGRAMS), device=device, dtype=torch.int32)
        self.cnt = torch.zeros(4, device=device, dtype=torch.int32)
        self.rec = torch.zeros(4, device=device, dtype=torch.int64)
        self.perm = perm

    def expected_end(self, steps):
        p = self.perm.tolist()
        return p[steps % len(p)]


def chase(ch, mode, steps=4096, load=".cg"):
    ch.cnt.zero_()
    return chase_kernel[(int(NUM_PROGRAMS), )](
        ch.chain, ch.src, ch.out, ch.cnt, ch.rec, ch.n_lines * 32, steps, ch.start,
        MODE=CHASE_MODES[mode], CHASE_LOAD=load, num_warps=1,
    )


def chase_result(ch, steps):
    """(ns per load, final line == expected, XCC id of the chaser)."""
    r = ch.rec.cpu().tolist()
    return (r[1] - r[0]) * 10.0 / steps, r[2] == ch.expected_end(steps), r[3]


def dirty(dst, rec, d_mib, store):
    chunk = d_mib * MIB // 4 // int(NUM_PROGRAMS)
    assert chunk % 1024 == 0
    return dirty_kernel[(int(NUM_PROGRAMS), )](dst, rec, chunk, STORE=store, num_warps=4)


def follow(rec):
    return follow_kernel[(int(NUM_PROGRAMS), )](rec, num_warps=4)
