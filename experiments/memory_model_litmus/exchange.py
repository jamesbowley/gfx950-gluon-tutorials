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
"""How the ways of exchanging stream-K partials scale with bandwidth: a model and a measurement.

    HIP_VISIBLE_DEVICES=1 python exchange.py            # results/exchange.md; then make_figures.py

W writers (pids 0 .. W - 1, so W / 8 per XCD) each hold a partial of P bytes (P = 256 KiB for a
256x256 fp32 accumulator, 128 KiB for 128x256 fp32 or 256x256 bf16) and exchange it in one of four
ways, all with the placement-independent sequence (`.wt` stores, drain, relaxed gpu-scope flag or
counter, `.cv` loads):
- handoff: W more CUs (pids W .. 2W - 1) each read one writer's partial (1:1, e.g. two-tile
  stream-K, where a tile has one peer);
- reduce-scatter: groups of n writers; after all n have arrived, each reads 1/n of every partial;
- owner: groups of n; member 0 keeps its partial and reads the other n - 1 in order, spinning on
  each one's flag;
- last arriver: groups of n; all write and arrive on a counter, the last reads the other n - 1.
The partials are read and checked against the launch's epoch (a real ending would add them). With
C_STORE the consumers then store their share of a bf16 C tile (128 KiB per 256 KiB partial; each
reduce-scatter member its 1/n slice), timed as a phase of its own.

Three atomic endings replace the exchange (split sweep only; no workspace, flag or read):
- atomic bf16: every member adds its partial into its group's zeroed bf16 C tile with
  buffer_atomic_pk_add_bf16 (two packed bf16 per lane, 64 lanes on 256 contiguous bytes);
- atomic fp32: the same into a zeroed fp32 tile with buffer_atomic_add_f32 (one fp32 per lane);
- atomic fp32 + finalize: the fp32 adds, a drain and a relaxed gpu counter; the last arriver reads
  the fp32 tile with `.cv` loads and stores the bf16 C tile.
Member m adds m + 1, so every sum (n (n + 1) / 2) is exact in bf16 for n <= 16. The values are
generated in the atomics' own layout: the convert_layout a real kernel needs (MFMA layout to the
coalesced one, through LDS) is not included. The host zeroes the tiles before the cache flush, in
a launch of its own, and times it separately.

The model (model_us) bounds each phase by the slowest of three limits: one CU's rate, one XCD's
share of the fabric, and the chip; it measures nothing itself. The rates come from this experiment's
1-writer handshake and from experiments/hbm_bandwidth_xcd.
"""

import argparse
import json
import os
import statistics
import time

import triton  # before torch: torch's HIP pulls in ROCm's shared libLLVM, which breaks libtriton's
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from costs import _asm, _grid_sync, _lds_hold, _now, _one, _stamp, _getreg

NUM_PROGRAMS = 256
KIB = 1024
GEMM_STAMPS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "kernels", "gemm", "intra_wave",
                           "a16w16", "v20_streamk_reduce_scatter", "results", "fixup_timing_limited.jsonl")
PATTERNS = {"handoff": 0, "reduce-scatter": 1, "owner": 2, "last arriver": 3, "atomic bf16": 4, "atomic fp32": 5,
            "atomic fp32 + finalize": 6}
EXCHANGES = ("handoff", "reduce-scatter", "owner", "last arriver")
ATOMICS = ("atomic bf16", "atomic fp32", "atomic fp32 + finalize")
REC = 10  # int64 fields per program in rec

# ---------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------

RATES = dict(
    w_cu=85e9,  # one CU, write-through 16 B per lane: 256 KiB in 3.1 us (costs.md, 1-writer handshake)
    r_cu=72e9,  # one CU (4 waves, 16 loads of 16 B per lane in flight) reading a 256 KiB partial from the
    # LLC: 3.7 us in the 32-writer handoff, which is per-CU-bound (75 GB/s with 32 in flight; the 110 GB/s
    # of hbm_bandwidth_xcd needs more waves per CU)
    w_xcd=0.925e12,  # writes per XCD when all XCDs write: 1.85 TB/s per XCD pair (hbm_bandwidth_xcd)
    r_xcd=1.35e12,  # reads per XCD from the MALL (hbm_bandwidth_xcd)
    r_chip=7.5e12,  # reads, whole chip, from the MALL (hbm_bandwidth_xcd)
    sync=0.6e-6,  # flag or counter round trip (costs.md)
)


RATES["w_xcd1"] = 1.30e12  # writes, one XCD alone (hbm_bandwidth_xcd)
RATES["w_pair"] = 1.85e12  # writes, per XCD pair 0+1, 2+3, 4+5, 6+7 (hbm_bandwidth_xcd)
RATES["w_chip"] = 7.4e12  # writes, whole chip
# Coalesced atomics, results unused (experiments/streamk_atomics): one CU and the whole chip. No per-XCD
# limit was measured, so the model has none. The bf16 per-CU rate includes streamk_atomics' convert_layout
# through LDS, which this kernel does not do.
RATES["a_cu_bf16"] = 27e9  # 128 KiB of pk_add in 4.8 us
RATES["a_cu_f32"] = 36e9  # 256 KiB of fp32 adds in 7.2 us
RATES["a_chip_bf16"] = 1.32e12
RATES["a_chip_f32"] = 1.35e12


def members(W, n, place):
    """[(group, member, XCD)] of the writers, and [(writer index, XCD)] of a handoff's readers, as
    exchange_kernel places them (pid % 8 is the XCD)."""
    ws, rs = [], []
    if place == 1:
        gpx = (W + 8 * n - 1) // (8 * n)
        xu = W // (n * gpx)
        for x in range(xu):
            for slot in range(gpx * n):
                ws.append((x * gpx + slot // n, slot % n, x))
                rs.append(((x * gpx + slot // n) * n + slot % n, x))
    else:
        ws = [(pid // n, pid % n, pid % 8) for pid in range(W)]
        rs = [(w, (W + w) % 8) for w in range(W)]
    return ws, rs


def placement(pattern, W, n, P, place=1):
    """Per program: (XCD, bytes written, bytes read). The owner rotates (member g % n of group g); the last
    arriver is taken to be any member of its group, so its reads are shared evenly over the group."""
    ws, rs = members(W, n, place)
    if pattern == "handoff":
        return [(x, P, 0.0) for _, _, x in ws] + [(x, 0.0, P) for _, x in rs]
    progs = []
    for g, m, x in ws:
        if pattern == "atomic bf16":
            progs.append((x, P / 2, 0.0))
        elif pattern == "atomic fp32":
            progs.append((x, P, 0.0))
        elif pattern == "atomic fp32 + finalize":
            progs.append((x, P, P / n))
        elif pattern == "reduce-scatter":
            progs.append((x, P, P))
        elif pattern == "owner":
            own = m == g % n
            progs.append((x, 0.0 if own else P, (n - 1) * P if own else 0.0))
        else:
            progs.append((x, P, (n - 1) * P / n))
    return progs


def phase_time(per_cu, per_xcd, total, cu_rate, xcd_limits, chip):
    """The slowest of one CU, the XCD-level limits (a list of (bytes, rate)) and the chip; and its name."""
    t = {"CU": per_cu / cu_rate, "chip": total / chip}
    t["XCD link"] = max((b / r for b, r in xcd_limits), default=0.0)
    lim = max(t, key=t.get)
    return t[lim], lim


def model_us(pattern, W, P, n=16, r=RATES, phases=False, place=1, pdtype="fp32"):
    """(predicted us, write limit, read limit, bytes written, bytes read) for one exchange; with phases,
    (publish us, sync us, read us) instead. Each phase is bounded by the slowest of: one CU's rate, every
    XCD's link (writes: 1.30 TB/s per XCD and 1.85 TB/s per XCD pair; reads: 1.35 TB/s per XCD), and the chip,
    with the bytes counted per XCD for the actual placement. The atomic endings' publish is the slowest of
    one CU's and the chip's atomic rate; 'atomic bf16' and 'atomic fp32' have no sync or read. pdtype bf16:
    the exchanges' partials are P / 2 bytes, at the same rates."""
    if pdtype == "bf16" and pattern in EXCHANGES:
        P = P / 2
    progs = placement(pattern, W, n, P, place)
    wx, rx = [0.0] * 8, [0.0] * 8
    for x, wb_, rb_ in progs:
        wx[x] += wb_
        rx[x] += rb_
    wb, rb = sum(wx), sum(rx)
    w_cu = max(p[1] for p in progs)
    r_cu = max(p[2] for p in progs) if pattern not in ("last arriver", "atomic fp32 + finalize") else \
        (n - 1) * P if pattern == "last arriver" else P
    sync = r["sync"]
    if pattern in ATOMICS:
        kind = "bf16" if pattern == "atomic bf16" else "f32"
        tw, lw = phase_time(w_cu, 0, wb, r["a_cu_" + kind], [], r["a_chip_" + kind])
        if pattern != "atomic fp32 + finalize":
            sync, rb = 0.0, 0.0
    else:
        wl = [(wx[x], r["w_xcd1"]) for x in range(8)] + [(wx[2 * k] + wx[2 * k + 1], r["w_pair"]) for k in range(4)]
        tw, lw = phase_time(w_cu, 0, wb, r["w_cu"], wl, r["w_chip"])
    tr, lr = phase_time(r_cu, 0, rb, r["r_cu"], [(rx[x], r["r_xcd"]) for x in range(8)], r["r_chip"])
    if phases:
        return tw * 1e6, sync * 1e6, tr * 1e6
    if rb == 0:
        lr = "-"
    return (tw + sync + tr) * 1e6, lw, lr, wb, rb


# ---------------------------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------------------------




@gluon.jit
def _check(ptr, slot_elems, expect, L: gl.constexpr, UNROLL: gl.constexpr, LOADC: gl.constexpr,
           BF16: gl.constexpr = 0):
    """Read one whole partial with `.cv` loads, UNROLL independent 4 KiB loads per step so that they are in
    flight together (with one load per step the loop is latency-bound: about 270 ns per 4 KiB from the
    LLC, 15 GB/s; 8 in flight give 58 GB/s, 16 give 71, 32 give 76); count the values that are not
    `expect`. BF16: a bf16 partial, 8 per lane (still 16 bytes per lane and 4 KiB per load)."""
    if BF16:
        LB: gl.constexpr = gl.BlockedLayout([8], [64], [4], [0])
        offs = gl.arange(0, 2048, LB)
        nbad = gl.zeros([2048], gl.int32, LB)
        for t in range(0, slot_elems, 2048 * UNROLL):
            for u in gl.static_range(UNROLL):
                x = gl.amd.cdna4.buffer_load(ptr=ptr + t + u * 2048, offsets=offs, cache=LOADC)
                nbad += (x.to(gl.float32) != expect).to(gl.int32)
    else:
        offs = gl.arange(0, 1024, L)
        nbad = gl.zeros([1024], gl.int32, L)
        for t in range(0, slot_elems, 1024 * UNROLL):
            for u in gl.static_range(UNROLL):
                x = gl.amd.cdna4.buffer_load(ptr=ptr + t + u * 1024, offsets=offs, cache=LOADC)
                nbad += (x != expect).to(gl.int32)
    return gl.sum(nbad, axis=0)


@gluon.jit
def _atomic_tile(ptr, elems, val, BF16: gl.constexpr, REPEAT: gl.constexpr):
    """Add val to every element of a tile with relaxed gpu-scope atomics whose results are unused (the
    no-return form). Coalesced: for each instruction a wave's 64 lanes cover 256 contiguous bytes, two
    packed bf16 or one fp32 per lane. The arange sizes assume 4 warps: a smaller tensor would be
    replicated over lanes and add twice. REPEAT 2 is the double-add control."""
    if BF16:
        LA: gl.constexpr = gl.BlockedLayout([2], [64], [4], [0])
        offs = gl.arange(0, 512, LA)
        v = (gl.zeros([512], gl.float32, LA) + val).to(gl.bfloat16)
        for _r in gl.static_range(REPEAT):
            for t in range(0, elems, 512):
                gl.amd.cdna4.buffer_atomic_add(ptr + t, offs, v, sem="relaxed", scope="gpu")
    else:
        LA: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
        offs = gl.arange(0, 256, LA)
        v = gl.zeros([256], gl.float32, LA) + val
        for _r in gl.static_range(REPEAT):
            for t in range(0, elems, 256):
                gl.amd.cdna4.buffer_atomic_add(ptr + t, offs, v, sem="relaxed", scope="gpu")


@gluon.jit
def _store_c(ptr, elems, val):
    """Default bf16 stores of val over elems elements, 16 bytes per lane (the 4-warp layout covers 2048)."""
    LC: gl.constexpr = gl.BlockedLayout([8], [64], [4], [0])
    offs = gl.arange(0, 2048, LC)
    v = (gl.zeros([2048], gl.float32, LC) + val).to(gl.bfloat16)
    for t in range(0, elems, 2048):
        gl.amd.cdna4.buffer_store(stored_value=v, ptr=ptr + t, offsets=offs)


@gluon.jit
def _check_slice(p_ptr, g, m, stride, expect, N: gl.constexpr, SLICE: gl.constexpr, L: gl.constexpr,
                 LOADC: gl.constexpr, BF16: gl.constexpr = 0):
    """Sum slice m (SLICE elements) over the group's N partials in fp32, the N loads of a step in flight
    together; count the sums that are not N * expect. SLICE is a constexpr so that the slice start m * SLICE
    is known to be aligned: with a runtime slice size the loads are not vectorised (buffer_load_dword, 4x
    slower). BF16: bf16 partials, 8 per lane."""
    if BF16:
        LB: gl.constexpr = gl.BlockedLayout([8], [64], [4], [0])
        offs = gl.arange(0, 2048, LB)
        nbad = gl.zeros([2048], gl.int32, LB)
        lo = m * SLICE
        for t in range(0, SLICE, 2048):
            s = gl.zeros([2048], gl.float32, LB)
            for j in gl.static_range(N):
                s += gl.amd.cdna4.buffer_load(ptr=p_ptr + (g * N + j).to(gl.int64) * stride + lo + t, offsets=offs,
                                              cache=LOADC).to(gl.float32)
            nbad += (s != N * expect).to(gl.int32)
    else:
        offs = gl.arange(0, 1024, L)
        nbad = gl.zeros([1024], gl.int32, L)
        lo = m * SLICE
        for t in range(0, SLICE, 1024):
            s = gl.zeros([1024], gl.float32, L)
            for j in gl.static_range(N):
                s += gl.amd.cdna4.buffer_load(ptr=p_ptr + (g * N + j).to(gl.int64) * stride + lo + t,
                                              offsets=offs, cache=LOADC)
            nbad += (s != N * expect).to(gl.int32)
    return gl.sum(nbad, axis=0)


@gluon.jit
def _flag_set(ptr, epoch, SYNC: gl.constexpr):
    if SYNC == 1:
        gl.atomic_xchg(ptr, epoch, sem="release", scope="gpu")
    else:
        gl.atomic_xchg(ptr, epoch, sem="relaxed", scope="gpu")


@gluon.jit
def _wait(ptr, val, deadline, AT_LEAST: gl.constexpr, SYNC: gl.constexpr):
    """Spin with relaxed gpu-scope atomic loads until *ptr == val (or >= val); SYNC 1 then acquires once.
    Returns 1 if the deadline passed first."""
    f = gl.atomic_add(ptr, 0, sem="relaxed", scope="gpu")
    if AT_LEAST:
        while (f < val) & (_now() < deadline):
            f = gl.atomic_add(ptr, 0, sem="relaxed", scope="gpu")
        miss = (f < val).to(gl.int32)
    else:
        while (f != val) & (_now() < deadline):
            f = gl.atomic_add(ptr, 0, sem="relaxed", scope="gpu")
        miss = (f != val).to(gl.int32)
    if (SYNC == 1) | (SYNC == 3):
        gl.atomic_add(ptr, 0, sem="acquire", scope="gpu")
    _asm("s_nop 0")  # compiler fence: no partial load moves above the wait
    return miss


@gluon.jit
def _arrive(ptr, SYNC: gl.constexpr):
    if SYNC == 1:
        old = gl.atomic_add(ptr, 1, sem="acq_rel", scope="gpu")
    else:
        old = gl.atomic_add(ptr, 1, sem="relaxed", scope="gpu")
    _asm("s_nop 0")
    return old


@gluon.jit
def exchange_kernel(p_ptr, p16_ptr, flag_ptr, cnt_ptr, gcnt_ptr, rec_ptr, c16_ptr, c32_ptr, epoch, W, slot_elems,
                    stride, timeout, N: gl.constexpr, PATTERN: gl.constexpr, SYNC: gl.constexpr = 0,
                    STORE: gl.constexpr = ".wt", LOADC: gl.constexpr = ".cv", SLICE: gl.constexpr = 4096,
                    UNROLL: gl.constexpr = 16, PLACE: gl.constexpr = 1, C_STORE: gl.constexpr = 0,
                    REPEAT: gl.constexpr = 1, PART_BF16: gl.constexpr = 0, FLAGS_FIRST: gl.constexpr = 0):
    """SYNC 0: `.wt` stores, relaxed flags / counters, `.cv` loads. SYNC 1: default stores, release flags /
    acq_rel counters, an acquire after each wait, default loads (the compiler's own sequence: one
    buffer_wbl2 sc1 per CU, in the wave that executes the atomic). SYNC 3: the same, but the release is an
    inline buffer_wbl2 sc1 in every wave before the barrier, and the flag / counter is relaxed. SYNC 2:
    default stores, relaxed flags / counters, `.cg` loads (v21 `l2`; needs PLACE 1). Every wave drains its
    stores and meets the others at s_barrier before the flag or arrival: a release covers only the wave that
    issues the atomic.
    PLACE 1: every group on one XCD, as stream-K's chiplet swizzle places a tile's consecutive logical pids
    (groups fill an XCD's slots, then the next XCD; a handoff's readers sit on the writers' XCD). PLACE 0:
    groups of consecutive hardware pids, so one member per XCD (a reference: not how stream-K places them).
    PATTERN 4-6 are the atomic endings (module docstring); they ignore STORE and use SYNC only for the
    finalize counter. C_STORE 1: every consumer stores its share of the bf16 C tile after its reads.
    PART_BF16 1: the exchanges' partials are bf16 (p16_ptr, the same buffer); a tile is still slot_elems
    elements, so the partials move half the bytes and C is unchanged. FLAGS_FIRST 1: the owner waits for
    every peer's flag before its first read, as v20's GEMM does, instead of before each peer's read; its
    sync phase is then the whole flag wait and its read phase the reads alone. stride is in elements of the partial's
    type. They hold epoch % 200 + 1, exact in bf16 and different in consecutive launches (bf16(epoch) is not:
    neighbouring epochs round to the same value)."""
    L: gl.constexpr = gl.BlockedLayout([4], [64], [gl.num_warps()], [0])
    if PART_BF16:
        pbase = p16_ptr
        pexpect = (epoch % 200 + 1).to(gl.float32)
    else:
        pbase = p_ptr
        pexpect = epoch.to(gl.float32)
    pid = gl.program_id(0)
    xcd = pid % 8
    slot = pid // 8
    lds = _lds_hold(L)
    offs = gl.arange(0, 1024, L)
    n = N
    if PLACE == 1:
        gpx = (W + 8 * N - 1) // (8 * N)  # groups per XCD
        xu = W // (N * gpx)  # XCDs used
        is_w = (xcd < xu) & (slot < gpx * N)
        g = xcd * gpx + slot // N
        m = slot % N
        rs = slot - gpx * N
        is_r = (xcd < xu) & (rs >= 0) & (rs < gpx * N)
        ridx = (xcd * gpx + rs // N) * N + rs % N
    else:
        is_w = pid < W
        g = pid // N
        m = pid % N
        is_r = (pid >= W) & (pid < 2 * W)
        ridx = pid - W
    widx = g * N + m
    om = g % N  # the owner rotates through the members, so that owners land on every XCD
    writes = is_w
    if PATTERN == 2:
        writes = is_w & (m != om)  # the owner keeps its own partial
    _grid_sync(gcnt_ptr)
    t0 = _stamp()
    deadline = t0 + timeout
    tile = g.to(gl.int64) * slot_elems  # the group's C tile, in elements
    if writes:
        if PATTERN == 4:
            _atomic_tile(c16_ptr + tile, slot_elems, (m + 1).to(gl.float32), True, REPEAT)
        elif PATTERN >= 5:
            _atomic_tile(c32_ptr + tile, slot_elems, (m + 1).to(gl.float32), False, REPEAT)
        elif PART_BF16:
            LB: gl.constexpr = gl.BlockedLayout([8], [64], [4], [0])
            offs16 = gl.arange(0, 2048, LB)
            base = p16_ptr + widx.to(gl.int64) * stride
            v16 = (gl.zeros([2048], gl.float32, LB) + pexpect).to(gl.bfloat16)
            for t in range(0, slot_elems, 2048):
                gl.amd.cdna4.buffer_store(stored_value=v16, ptr=base + t, offsets=offs16, cache=STORE)
            if SYNC == 3:
                _asm("buffer_wbl2 sc1\ns_waitcnt vmcnt(0)")
        else:
            base = p_ptr + widx.to(gl.int64) * stride
            v = gl.zeros([1024], gl.float32, L) + epoch.to(gl.float32)
            for t in range(0, slot_elems, 1024):
                gl.amd.cdna4.buffer_store(stored_value=v, ptr=base + t, offsets=offs, cache=STORE)
            if SYNC == 3:
                _asm("buffer_wbl2 sc1\ns_waitcnt vmcnt(0)")  # an agent release in every wave, by hand
        _asm("s_waitcnt vmcnt(0)\ns_barrier")
    t1 = _now()
    t2 = t1
    bad = 0
    timed_out = 0
    consumer = 0
    c_off = tile
    c_elems = slot_elems
    if PATTERN == 0:
        if writes:
            _flag_set(flag_ptr + widx * 32, epoch, SYNC)
        if is_r:
            consumer = 1
            timed_out = _wait(flag_ptr + ridx * 32, epoch, deadline, False, SYNC)
            t2 = _now()
            bad = _check(pbase + ridx.to(gl.int64) * stride, slot_elems, pexpect, L, UNROLL, LOADC, PART_BF16)
            c_off = ridx.to(gl.int64) * slot_elems
    elif PATTERN == 1:
        if is_w:
            consumer = 1
            _arrive(cnt_ptr + g * 32, SYNC)
            timed_out = _wait(cnt_ptr + g * 32, n, deadline, True, SYNC)
            t2 = _now()
            bad = _check_slice(pbase, g, m, stride, pexpect, N, SLICE, L, LOADC, PART_BF16)
            c_off = tile + m * SLICE
            c_elems = SLICE
    elif PATTERN == 2:
        if writes:
            _flag_set(flag_ptr + widx * 32, epoch, SYNC)
        if is_w & (m == om):
            consumer = 1
            if FLAGS_FIRST:
                for k in range(1, n):
                    peer = g * n + (om + k) % n
                    timed_out += _wait(flag_ptr + peer * 32, epoch, deadline, False, SYNC)
                t2 = _now()
                for k in range(1, n):
                    peer = g * n + (om + k) % n
                    bad += _check(pbase + peer.to(gl.int64) * stride, slot_elems, pexpect, L, UNROLL, LOADC,
                                  PART_BF16)
            else:
                for k in range(1, n):
                    peer = g * n + (om + k) % n
                    timed_out += _wait(flag_ptr + peer * 32, epoch, deadline, False, SYNC)
                    if k == 1:
                        t2 = _now()
                    bad += _check(pbase + peer.to(gl.int64) * stride, slot_elems, pexpect, L, UNROLL, LOADC,
                                  PART_BF16)
    elif PATTERN == 3:
        if is_w:
            old = _arrive(cnt_ptr + g * 32, SYNC)
            t2 = _now()
            if old == n - 1:
                consumer = 1
                for j in range(n):
                    if j != m:
                        bad += _check(pbase + (g * n + j).to(gl.int64) * stride, slot_elems, pexpect, L, UNROLL,
                                      LOADC, PART_BF16)
    elif PATTERN == 6:
        if is_w:
            old = _arrive(cnt_ptr + g * 32, SYNC)
            t2 = _now()
            if old == n - 1:
                consumer = 1
                bad = _check(c32_ptr + tile, slot_elems, (N * (N + 1) // 2) * 1.0, L, UNROLL, LOADC)
    else:
        if is_w:
            consumer = 1  # the atomics are the whole ending
    tc = _stamp()
    if ((C_STORE == 1) & (PATTERN <= 3)) | (PATTERN == 6):
        if consumer == 1:
            cval = epoch.to(gl.float32) * N
            if PATTERN == 6:
                cval = (N * (N + 1) // 2) * 1.0 + epoch.to(gl.float32) * 0.0
            _store_c(c16_ptr + c_off, c_elems, cval)
    t3 = _stamp()
    z = _one()
    rec = rec_ptr + pid * 10 + z
    gl.store(rec, t0 + z)
    gl.store(rec + 1, t1 + z)
    gl.store(rec + 2, t2 + z)
    gl.store(rec + 3, t3 + z)
    gl.store(rec + 4, bad.to(gl.int64) + z)
    gl.store(rec + 5, (consumer + 2 * writes.to(gl.int32)).to(gl.int64) + z)
    gl.store(rec + 6, timed_out.to(gl.int64) + z)
    gid = g
    if PATTERN == 0:
        gid = gl.where(is_r, ridx, widx)  # a handoff "group" is one writer and its reader
    gl.store(rec + 7, (_getreg("HW_REG_XCC_ID") & 15).to(gl.int64) + 16 * gid.to(gl.int64) + z +
             gl.sum(lds.slice(0, 64 * gl.num_warps()).load(L).to(gl.int64), axis=0) * 0)
    gl.store(rec + 8, tc + z)


# ---------------------------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------------------------


class Bufs:

    def __init__(self, device="cuda"):
        self.p = torch.zeros(NUM_PROGRAMS * 260 * KIB // 4, device=device, dtype=torch.float32)
        self.p16 = self.p.view(torch.bfloat16)  # the same workspace, for bf16 partials
        self.flags = torch.zeros(NUM_PROGRAMS * 32, device=device, dtype=torch.int32)
        self.cnt = torch.zeros(NUM_PROGRAMS * 32, device=device, dtype=torch.int32)
        self.gcnt = torch.zeros(8, device=device, dtype=torch.int32)
        self.rec = torch.zeros(NUM_PROGRAMS * REC, device=device, dtype=torch.int64)
        self.c16 = torch.zeros(NUM_PROGRAMS * 64 * KIB, device=device, dtype=torch.bfloat16)  # bf16 C tiles
        self.c32 = torch.zeros(NUM_PROGRAMS * 64 * KIB, device=device, dtype=torch.float32)  # fp32 atomic tiles
        self.flush_buf = torch.ones(1024 * 2**20 // 4, device=device, dtype=torch.float32)
        self.epoch = 1000


def med(xs):
    return statistics.median(xs) if xs else float("nan")


SYNCS = {
    # name: (SYNC, STORE, LOADC)
    "wt": (0, ".wt", ".cv"),
    "acq_rel": (1, "", ""),
    "l2": (2, "", ".cg"),
    "wave_release": (3, "", ""),
}
SYNC_NAMES = {
    "wt": "`.wt` stores, relaxed sync, `.cv` loads (correct for any placement)",
    "acq_rel": "default stores, `gpu` release / acquire, default loads (correct for any placement)",
    "l2": "default stores, relaxed sync, `.cg` loads (correct only with the group on one XCD)",
    "wave_release": "as `acq_rel`, but the release is `buffer_wbl2 sc1` in every wave before the barrier "
                    "and the flag / counter is relaxed",
}


def launch(b, pattern, W, P, n=16, sync="wt", place=1, pad=0, unroll=16, loadc=None, cstore=False, repeat=1,
           pdtype="fp32", flags_first=False):
    """One launch after the cache flush; returns the compiled kernel. P is the fp32 partial's size, so a tile
    is P / 4 elements whatever pdtype; bf16 partials are half those bytes."""
    code, store, lc = SYNCS[sync]
    return exchange_kernel[(NUM_PROGRAMS, )](b.p, b.p16, b.flags, b.cnt, b.gcnt, b.rec, b.c16, b.c32, b.epoch, W,
                                             P // 4, P // 4 + pad, 100000, N=n, PATTERN=PATTERNS[pattern], SYNC=code,
                                             STORE=store, LOADC=lc if loadc is None else loadc, SLICE=P // 4 // n,
                                             UNROLL=unroll, PLACE=place, C_STORE=int(cstore), REPEAT=repeat,
                                             PART_BF16=int(pdtype == "bf16"), FLAGS_FIRST=int(flags_first),
                                             num_warps=4)


def n_groups(W, n, place):
    return max(g for g, _, _ in members(W, n, place)[0]) + 1


def atomic_tiles(b, pattern, W, P, n, place):
    """(buffer, tiles) that the atomic ending accumulates into: the groups' tiles, P / 4 elements each."""
    buf = b.c16 if pattern == "atomic bf16" else b.c32
    ng = n_groups(W, n, place)
    return buf[:ng * (P // 4)].view(ng, P // 4)


def wrong_tiles(b, pattern, W, P, n, place):
    """Elements of the groups' tiles that are not n (n + 1) / 2 (and, after the finalize, of its bf16 C)."""
    want = n * (n + 1) // 2
    bad = int((atomic_tiles(b, pattern, W, P, n, place) != want).sum())
    if pattern == "atomic fp32 + finalize":
        ng = n_groups(W, n, place)
        bad += int((b.c16[:ng * (P // 4)] != want).sum())
    return bad


def run(b, pattern, W, P, n=16, sync="wt", place=1, launches=30, pad=0, unroll=16, loadc=None, phases=False,
        cstore=False, repeat=1, pdtype="fp32", flags_first=False):
    """Medians over launches: (makespan us, writers' publish us, consumers' read us, wrong values, timeouts),
    and with phases the medians over groups of the critical path's (publish, sync, read, C store) us and the
    median time to zero the atomic endings' tiles (a launch of its own, before the cache flush; else 0)."""
    span, pub, rd = [], [], []
    ph_pub, ph_sync, ph_read, ph_c, zero = [], [], [], [], []
    bad = tos = 0
    atomic = pattern in ATOMICS
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for i in range(launches + 1):
        if atomic:
            tiles = atomic_tiles(b, pattern, W, P, n, place)
            torch.cuda.synchronize()
            e0.record()
            tiles.zero_()
            e1.record()
            torch.cuda.synchronize()
            if i > 0:
                zero.append(e0.elapsed_time(e1) * 1000)
            if pattern == "atomic fp32 + finalize":
                b.c16.zero_()
        b.flush_buf.sum()
        b.gcnt.zero_()
        b.cnt.zero_()
        b.epoch += 1
        launch(b, pattern, W, P, n, sync, place, pad, unroll, loadc, cstore, repeat, pdtype, flags_first)
        torch.cuda.synchronize()
        if i == 0:
            continue
        if atomic:
            bad += wrong_tiles(b, pattern, W, P, n, place)
        r = b.rec.view(NUM_PROGRAMS, REC).cpu()
        role = r[:, 5]
        t0 = r[:, 0].min()
        active = (role & 3) != 0
        span.append(float(r[active, 3].max() - t0) / 100)
        pub += (r[(role & 2) != 0, 1] - r[(role & 2) != 0, 0]).double().div(100).tolist()
        cons = (role & 1) != 0
        rd += (r[cons, 3] - r[cons, 2]).double().div(100).tolist()
        bad += int(r[:, 4].sum())
        tos += int(r[:, 6].sum())
        # The critical path of every group: until its last writer has published, then until the consumer
        # that finishes last has passed its wait, then its reads, then its C store (from the stamp tc after
        # its reads; about 0.1 us without a C store). The four segments add up to that consumer's end.
        gid = r[:, 7] // 16
        for gv in gid[active].unique().tolist():
            mem = r[active & (gid == gv)]
            w, c = mem[(mem[:, 5] & 2) != 0], mem[(mem[:, 5] & 1) != 0]
            if len(w) == 0 or len(c) == 0:
                continue
            crit = c[c[:, 3].argmax()]
            pe = float(w[:, 1].max() - t0) / 100
            t2, t3, tc = float(crit[2] - t0) / 100, float(crit[3] - t0) / 100, float(crit[8] - t0) / 100
            ph_pub.append(pe)
            ph_sync.append(max(0.0, t2 - pe))
            ph_read.append(max(0.0, tc - max(t2, pe)))
            ph_c.append(t3 - max(tc, t2, pe))
    if phases:
        return (med(span), med(pub), med(rd), bad, tos, (med(ph_pub), med(ph_sync), med(ph_read), med(ph_c)),
                med(zero) if zero else 0.0)
    return med(span), med(pub), med(rd), bad, tos


def atomic_check(b):
    """The atomic endings' instructions, and their sums with and without a double add (a control that must
    fail every element)."""
    print("\n## atomic_check\n")
    print("Atomic instructions in each ending's compiled code (distinct forms; `sc0` would be the returning "
          "form), then the tiles after 3 launches with every partial added once and, as a control, twice "
          "(REPEAT 2). W = 256, 256 KiB partials, a 1 GiB read before each launch.\n")
    print("| pattern | atomic instructions | returning form |")
    print("|---|---|---|")
    for pattern in ATOMICS:
        k = launch(b, pattern, 256, 256 * KIB, 16)
        torch.cuda.synchronize()
        ins = sorted({" ".join(line.split()[:1] + line.split()[4:]) for line in k.asm["amdgcn"].splitlines()
                      if "buffer_atomic" in line})
        ret = any(" sc0" in line for line in k.asm["amdgcn"].splitlines() if "buffer_atomic" in line)
        print(f"| {pattern} | {'; '.join(f'`{i}`' for i in ins)} | {'yes' if ret else 'no'} |")
    print("\n| pattern | placement | n | adds | elements checked | wrong | expected wrong |")
    print("|---|---|---|---|---|---|---|")
    for pattern in ATOMICS:
        for place in (1, 0):
            for n in (4, 16):
                for repeat in (1, 2):
                    *_, bad, _ = run(b, pattern, 256, 256 * KIB, n=n, place=place, launches=3, repeat=repeat)
                    tile = n_groups(256, n, place) * 64 * KIB
                    # finalize: the fp32 tiles, the last arrivers' reads of them, and the bf16 C they store
                    # (a constant, so the control leaves it right)
                    per, per_bad = (3 * tile, 2 * tile) if pattern == "atomic fp32 + finalize" else (tile, tile)
                    pname = "one XCD" if place == 1 else "spread"
                    print(f"| {pattern} | {pname} | {n} | {'once' if repeat == 1 else 'twice'} | {3 * per} | {bad} | "
                          f"{0 if repeat == 1 else 3 * per_bad} |", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", nargs="+", default=["exchange", "split", "s8"],
                   choices=["exchange", "split", "s8", "sn", "sn_owner", "atomic_check"])
    args = p.parse_args()
    print(f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', 'unset')}, {torch.cuda.get_device_name()}, "
          f"triton {triton.__version__}, {time.strftime('%Y-%m-%d %H:%M')}\n")
    b = Bufs()
    if "exchange" in args.sweep:
        print("## exchange\n")
        print("W writers exchange partials of P KiB in groups of n = 16, with one of four synchronisation methods "
              "(sync): " + "; ".join(f"`{k}` = {v}" for k, v in SYNC_NAMES.items()) + ". Placement: 'one XCD' puts "
              "each group on one XCD, as stream-K's chiplet swizzle places a tile's programs (groups fill an XCD, "
              "then the next); 'spread' (a reference, write-through only) uses consecutive hardware pids, one "
              "member per XCD. Loads with 16 in flight per CU, a 1 GiB read before each launch. Measured: medians "
              "over 30 launches of the makespan (first stamp to the last program's end), the writers' publish time "
              "and the consumers' read time. Model (`wt`): the slowest of the CU, XCD-link and chip limits per "
              "phase, bytes counted per XCD for the placement (exchange.py, model_us). MB moved is writes + "
              "reads.\n")
        print("| pattern | sync | placement | W | P KiB | MB written | MB read | measured us | model us | write limit | "
              "read limit | publish us | read us | achieved TB/s | wrong | timeouts |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for pattern in EXCHANGES:
            for sync, place in [(k, 1) for k in SYNCS] + [("wt", 0)]:
                for P in (128 * KIB, 256 * KIB):
                    for W in (16, 32, 64, 128, 256):
                        if pattern == "handoff" and W > 128:
                            continue
                        span, pub, rd, bad, tos = run(b, pattern, W, P, sync=sync, place=place)
                        mus, lw, lr, wb, rb = model_us(pattern, W, P, place=place)
                        model = f"{mus:.2f} | {lw} | {lr}" if sync == "wt" else "- | - | -"
                        pname = "one XCD" if place == 1 else "spread"
                        print(f"| {pattern} | {sync} | {pname} | {W} | {P // KIB} | {wb / 1e6:.1f} | {rb / 1e6:.1f} | "
                              f"{span:.2f} | {model} | {pub:.2f} | {rd:.2f} | {(wb + rb) / span / 1e6:.2f} | {bad} | "
                              f"{tos} |", flush=True)
    if "split" in args.sweep:
        print("\n## split\n")
        print("Split count, 256x256 tiles (256 KiB fp32 or 128 KiB bf16 partials). Two ways to vary n: 'tiles 16' keeps 16 tiles, so W = 16 n "
              "writers; 'CUs 256' keeps all 256 CUs, so 256 / n tiles (as in the GEMM tail, where every CU takes "
              "part). Placement 'one XCD' (as stream-K places a tile) and, for reference, 'spread'. The exchanges "
              "(reduce-scatter, owner, last arriver) use `wt` synchronisation and then store their share of a bf16 "
              "C tile (128 KiB per tile). The atomic endings add every partial into a zeroed tile instead: bf16 "
              "`pk_add` into C, fp32 into a workspace, or fp32 then a last arriver that reads the sum with `.cv` "
              "and stores bf16 C ('+ finalize'); member m adds m + 1 and every tile is checked. The phases are "
              "medians over groups and launches of each group's critical path, each phase taken separately: "
              "publish (until the group's last writer has published; for the atomic endings, its atomics have "
              "completed), sync (until the consumer that finishes last has passed its wait: flag, counter or "
              "arrival), read (its reads), C store (its bf16 C store). Zero: the median time of the separate "
              "launch that zeroes the atomic endings' tiles (CUDA events, before the cache flush), not part of "
              "the measured total. Partials: the exchanges' partial type; bf16 partials (128 KiB, 8 per lane, still "
              "`.wt` / `.cv` and 16 bytes per lane) are summed in fp32 and C is the same 128 KiB bf16 tile. Model "
              "phases from exchange.model_us (no C store, no zeroing).\n")
        print("| pattern | partials | setup | placement | n | W | measured us | publish us | sync us | read us | "
              "C store us | zero us | model publish | model sync | model read | model us | wrong | timeouts |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        cases = [(p, "fp32") for p in ("reduce-scatter", "owner", "last arriver")] + \
            [(p, "bf16") for p in ("reduce-scatter", "owner", "last arriver")] + [(p, "-") for p in ATOMICS]
        for place in (1, 0):
            for setup in ("tiles 16", "CUs 256"):
                for pattern, pdt in cases:
                    for n in (2, 4, 8, 16):
                        W = 16 * n if setup == "tiles 16" else 256
                        pd = "bf16" if pdt == "bf16" else "fp32"
                        span, _, _, bad, tos, (pp, ps, pr, pc), zus = run(b, pattern, W, 256 * KIB, n=n, place=place,
                                                                          phases=True, cstore=True, pdtype=pd)
                        mp, ms_, mr = model_us(pattern, W, 256 * KIB, n=n, phases=True, place=place, pdtype=pd)
                        pname = "one XCD" if place == 1 else "spread"
                        print(f"| {pattern} | {pdt} | {setup} | {pname} | {n} | {W} | {span:.2f} | {pp:.2f} | "
                              f"{ps:.2f} | {pr:.2f} | {pc:.2f} | {zus:.2f} | {mp:.2f} | {ms_:.2f} | {mr:.2f} | "
                              f"{mp + ms_ + mr:.2f} | {bad} | {tos} |", flush=True)
    if "atomic_check" in args.sweep:
        atomic_check(b)
    if "s8" in args.sweep:
        print("\n## s8\n")
        print("S = 8 tiles, each split n ways (W = 8 n writers), 256 KiB partials, `wt` synchronisation, each "
              "group on one XCD: the exchange costs that split_model.py combines with the GEMM's MAC time. Phases "
              "as in the split table.\n")
        print("| pattern | n | W | measured us | publish us | sync us | read us | model us | wrong | timeouts |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for pattern in ("reduce-scatter", "owner", "last arriver"):
            for n in (2, 4, 8, 16, 32):
                W = 8 * n
                span, _, _, bad, tos, (pp, ps, pr, _), _ = run(b, pattern, W, 256 * KIB, n=n, place=1, phases=True)
                mus = model_us(pattern, W, 256 * KIB, n=n, place=1)[0]
                print(f"| {pattern} | {n} | {W} | {span:.2f} | {pp:.2f} | {ps:.2f} | {pr:.2f} | {mus:.2f} | {bad} | "
                      f"{tos} |", flush=True)
    if "sn" in args.sweep:
        print("\n## sn\n")
        print("S tiles, each split n ways (W = S n writers), for every S of the split-count sweep: 256 KiB "
              "partials, `wt` synchronisation (v20's data path: `.wt` stores, `.cv` loads; v20's barrier is "
              "acquire-release), each group on one XCD. Groups fill an XCD's 32 slots, then the next XCD, so "
              "ceil(S / 8) groups share an XCD; where S is not a multiple of that, fewer than S groups run "
              "(S = 38: 35). n is a power of 2 up to 32: the slice must be a whole number of 4 KiB steps and a "
              "group must fit one XCD. Phases as in the split table.\n")
        print("| pattern | S | n | W | groups | measured us | publish us | sync us | read us | model us | wrong | "
              "timeouts |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for S in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 38, 48, 64, 96, 128):
            for n in (2, 4, 8, 16, 32):
                W = S * n
                gpx = (W + 8 * n - 1) // (8 * n)
                if W > NUM_PROGRAMS or gpx * n > NUM_PROGRAMS // 8:
                    continue
                groups = (W // (n * gpx)) * gpx
                for pattern in ("reduce-scatter", "owner"):
                    span, _, _, bad, tos, (pp, ps, pr, _), _ = run(b, pattern, W, 256 * KIB, n=n, place=1,
                                                                   phases=True)
                    mus = model_us(pattern, W, 256 * KIB, n=n, place=1)[0]
                    print(f"| {pattern} | {S} | {n} | {W} | {groups} | {span:.2f} | {pp:.2f} | {ps:.2f} | {pr:.2f} | "
                          f"{mus:.2f} | {bad} | {tos} |", flush=True)
    if "sn_owner" in args.sweep:
        sn_owner(b)


def sn_owner(b):
    """The owner exchange of the sn sweep in two orders: interleaved (wait for a peer's flag, read it, next)
    and flags first (every peer's flag, then every read), against v20's stamped owner ending."""
    print("\n## sn_owner\n")
    print("The owner exchange of the sn table (256 KiB partials, `wt` synchronisation, each group on one XCD, "
          "30 launches after a 1 GiB read each) in two orders. Interleaved, as in sn: the owner waits for a "
          "peer's flag, reads that peer's partial, then moves to the next; its sync phase ends at the first "
          "flag and its read phase holds the later flag waits. Flags first, as v20's GEMM: the owner waits for "
          "every peer's flag, then reads every partial; its sync phase is the whole flag wait and its read phase "
          "the reads alone. Read per peer: read / (n - 1).\n")
    print("| S | n | W | groups | order | measured us | publish us | sync us | read us | read per peer us | "
          "wrong | timeouts |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    meas = {}
    for S in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 38, 48, 64, 96, 128):
        for n in (2, 4, 8, 16, 32):
            W = S * n
            gpx = (W + 8 * n - 1) // (8 * n)
            if W > NUM_PROGRAMS or gpx * n > NUM_PROGRAMS // 8:
                continue
            groups = (W // (n * gpx)) * gpx
            for ff in (False, True):
                span, _, _, bad, tos, (pp, ps, pr, _), _ = run(b, "owner", W, 256 * KIB, n=n, place=1, phases=True,
                                                               flags_first=ff)
                meas[(S, n, ff)] = (span, pr)
                print(f"| {S} | {n} | {W} | {groups} | {'flags first' if ff else 'interleaved'} | {span:.2f} | "
                      f"{pp:.2f} | {ps:.2f} | {pr:.2f} | {pr / (n - 1):.2f} | {bad} | {tos} |", flush=True)

    gemm = {}
    if os.path.exists(GEMM_STAMPS):
        with open(GEMM_STAMPS) as f:
            for line in f:
                r = json.loads(line)
                if r["fixup"] == "owner" and r["wrong"] == 0:
                    gemm.setdefault((r["S"], r["n"]), []).append(r["ending"])
    print("\nCost per extra peer, us: the change in the measured time between the two largest n of each S, divided "
          "by the change in n - 1. GEMM: v20's stamped owner ending (MAC done to C stored, medians over the "
          "owners; fixup_timing.py --tile-limit on 4096x8192, the mean over K = 4096 and 16384), the same two n; "
          "'-' where it was not stamped.\n")
    print("| S | n | interleaved | flags first | interleaved, read only | flags first, read only | GEMM owner ending |")
    print("|---|---|---|---|---|---|---|")
    for S in sorted({s for s, _, _ in meas}):
        ns = sorted({n for s, n, _ in meas if s == S})
        if len(ns) < 2:
            continue
        lo, hi = ns[-2], ns[-1]

        def slope(f):
            return (f(hi) - f(lo)) / (hi - lo)

        cells = [slope(lambda n, ff=ff, i=i: meas[(S, n, ff)][i]) for i in (0, 1) for ff in (False, True)]
        g = (f"{slope(lambda n: statistics.mean(gemm[(S, n)])):.2f}" if (S, lo) in gemm and (S, hi) in gemm
             else "-")
        print(f"| {S} | {lo} to {hi} | " + " | ".join(f"{c:.2f}" for c in cells) + f" | {g} |", flush=True)


if __name__ == "__main__":
    main()
