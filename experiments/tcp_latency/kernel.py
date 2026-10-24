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
"""Load-latency and loads-in-flight microbenchmark kernels for MI355X (gfx950).

`bw_kernel` and its placement machinery are ported from experiments/hbm_bandwidth_xcd (branch
wip_streamk): every workgroup maps its program id to a logical worker id (or exits early), so the set
of CUs that move data can be chosen independently of the grid.

`probe_kernel` times single 1 KB buffer_load_dwordx4 round trips on one wave, optionally while other
workgroups stream.
"""

from dataclasses import dataclass, field

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna4 import buffer_load, buffer_store

NUM_XCDS = 8
CUS_PER_XCD = 32
NUM_CUS = NUM_XCDS * CUS_PER_XCD

OP_READ, OP_WRITE, OP_COPY = 0, 1, 2
OPS = {"read": OP_READ, "write": OP_WRITE, "copy": OP_COPY}

MODE_RR, MODE_PACKED, MODE_SUBSET, MODE_HWSEL = 0, 1, 2, 3
MODES = {"rr": MODE_RR, "packed": MODE_PACKED, "subset": MODE_SUBSET, "hwsel": MODE_HWSEL}

PAT_CHUNKED, PAT_GRID = 0, 1
PATTERNS = {"chunked": PAT_CHUNKED, "grid": PAT_GRID}

# Per-workgroup record: xcc_id, hw_id, t_start, t_end (100 MHz), worker, active, clk_start, clk_end (shader clock)
REC_FIELDS = 8
# CU key = xcc * 256 + se * 32 + sh * 16 + cu, decoded from HW_REG_XCC_ID / HW_REG_HW_ID
HWSEL_KEYS = NUM_XCDS * 256

# LDS per workgroup that allows exactly WPC workgroups per CU (160 KB LDS on gfx950).
# Split into two power-of-two allocations because shared memory shapes must be powers of two.
LDS_FOR_WPC = {1: (65536, 32768), 2: (65536, 0), 3: (32768, 16384), 4: (32768, 4096), 5: (32768, 0),
               6: (16384, 8192), 8: (16384, 2048)}

_NUM_XCDS = gl.constexpr(NUM_XCDS)
_CUS_PER_XCD = gl.constexpr(CUS_PER_XCD)
_REC_FIELDS = gl.constexpr(REC_FIELDS)
_HWSEL_KEYS = gl.constexpr(HWSEL_KEYS)


# The helpers below operate on 1-element tensors so that reducing them to a scalar needs no cross-lane
# traffic: the "=s" asm result stays in an SGPR, which LLVM treats as uniform.


@gluon.jit
def _one(NW: gl.constexpr):
    return gl.zeros([1], gl.int32, gl.BlockedLayout([1], [64], [NW], [0]))


@gluon.jit
def _getreg(NW: gl.constexpr, REG: gl.constexpr):
    return gl.max(
        gl.inline_asm_elementwise("s_getreg_b32 $0, hwreg(" + REG + ")", "=s,v", [_one(NW)], dtype=gl.int32,
                                  is_pure=True, pack=1), axis=0)


@gluon.jit
def _uniform(x, NW: gl.constexpr):
    # Atomic results come back through LDS, so LLVM can't prove them wave-uniform and would wrap every
    # buffer op addressed from them in a waterfall loop.
    return gl.max(
        gl.inline_asm_elementwise("v_readfirstlane_b32 $0, $1", "=s,v", [_one(NW) + x], dtype=gl.int32,
                                  is_pure=True, pack=1), axis=0)


@gluon.constexpr_function
def _pin_constraints(P):
    return ",".join(["=v"] * P + [str(i) for i in range(P)])


@gluon.jit
def _pin(x, P: gl.constexpr):
    # Empty side-effecting asm that takes all P registers of a thread's tile at once: loads can't be moved
    # past it and the tile can't be consumed element-by-element, so every load of the tile is in flight together.
    return gl.inline_asm_elementwise("; pin", _pin_constraints(P), [x], dtype=gl.float32, is_pure=False, pack=P)


@gluon.jit
def _timestamps(NW: gl.constexpr, WAIT_MEM: gl.constexpr):
    # s_memrealtime is a constant 100 MHz clock, s_memtime counts shader clocks; their ratio gives SCLK.
    if WAIT_MEM:
        pre: gl.constexpr = "s_waitcnt vmcnt(0)\ns_barrier\n"
    else:
        pre: gl.constexpr = "s_barrier\n"
    rt, ck = gl.inline_asm_elementwise(pre + "s_memrealtime $0\ns_memtime $1\ns_waitcnt lgkmcnt(0)",
                                       "=s,=s,v,~{memory}", [_one(NW)], dtype=(gl.int64, gl.int64), is_pure=False,
                                       pack=1)
    return gl.max(rt, axis=0), gl.max(ck, axis=0)


@gluon.jit
def bw_kernel(
    src,
    dst,
    out,
    rec,
    sel,
    counters,
    W,
    n_iter,
    wrap,
    rank_table,
    per_xcd,
    OP: gl.constexpr,
    MODE: gl.constexpr,
    PATTERN: gl.constexpr,
    VEC: gl.constexpr,
    UNROLL: gl.constexpr,
    WPC: gl.constexpr,
    LDS_A: gl.constexpr,
    LDS_B: gl.constexpr,
    RECORD: gl.constexpr,
    CACHE_LD: gl.constexpr,
    CACHE_ST: gl.constexpr,
):
    NW: gl.constexpr = gl.num_warps()
    BLOCK: gl.constexpr = VEC * 64 * NW
    TILE: gl.constexpr = BLOCK * UNROLL
    small: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])
    layout: gl.constexpr = gl.BlockedLayout([VEC], [64], [NW], [0])
    slayout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])

    pid = gl.program_id(0)
    xcd = pid % _NUM_XCDS
    slot = pid // _NUM_XCDS
    lane = gl.arange(0, 64 * NW, small)

    if RECORD or MODE == 3:
        xcc = _getreg(NW, "HW_REG_XCC_ID") & 15
        hw = _getreg(NW, "HW_REG_HW_ID")
    else:
        xcc = xcd
        hw = pid * 0

    if MODE == 0:
        worker = pid
        active = pid < W
    elif MODE == 1:
        worker = xcd * (_CUS_PER_XCD * WPC) + slot
        active = worker < W
    elif MODE == 2:
        rank = (rank_table >> (4 * xcd).to(gl.int64)).to(gl.int32) & 15
        worker = rank * per_xcd + slot
        active = (rank != 15) & (slot < per_xcd)
    else:
        key = xcc * 256 + ((hw >> 13) & 7) * 32 + ((hw >> 12) & 1) * 16 + ((hw >> 8) & 15)
        cap = gl.load(sel + key)
        worker = pid * 0 - 1
        if cap > 0:
            s = gl.atomic_add(counters + key, 1)
            if s < cap:
                worker = gl.atomic_add(counters + _HWSEL_KEYS, 1)
        worker = _uniform(worker, NW)
        active = (worker >= 0) & (worker < W)

    if RECORD:
        k = lane.to(gl.int64)
        v = gl.where(k == 0, xcc.to(gl.int64), gl.where(k == 1, hw.to(gl.int64), gl.where(k == 4, worker.to(gl.int64),
                                                                                        active.to(gl.int64))))
        gl.store(rec + pid * _REC_FIELDS + k, v, mask=(k == 0) | (k == 1) | (k == 4) | (k == 5))

    if active:
        # Occupancy control: both LDS buffers stay live for the whole kernel so the allocator can't alias them.
        lds_a = gl.allocate_shared_memory(gl.int8, [LDS_A], slayout)
        lds_a.slice(0, 64 * NW).store(gl.zeros([64 * NW], gl.int8, small))
        if LDS_B > 0:
            lds_b = gl.allocate_shared_memory(gl.int8, [LDS_B], slayout)
            lds_b.slice(0, 64 * NW).store(gl.zeros([64 * NW], gl.int8, small))

        if RECORD:
            t0, c0 = _timestamps(NW, False)

        offs = gl.arange(0, TILE, layout)
        w64 = worker.to(gl.int64)
        acc = gl.zeros([TILE], gl.float32, layout)
        # Iteration i touches tile j = i % wrap: with wrap < n_iter the worker re-reads its footprint n_iter / wrap
        # times. j is carried and reset rather than computed with %, which is a software divide on the SALU.
        if PATTERN == 0:
            t_first = w64 * wrap
            t_step = 1
        else:
            t_first = w64
            t_step = w64 * 0 + W
        j = pid * 0
        if OP == 1:
            wval = offs.to(gl.float32)
            for i in range(n_iter):
                buffer_store(wval, dst + (t_first + j * t_step) * TILE, offs, cache=CACHE_ST)
                j += 1
                j = gl.where(j == wrap, 0, j)
        else:
            # Register double-buffering: tile i+1 is issued before tile i is consumed.
            cur_base = t_first * TILE
            x = buffer_load(src + cur_base, offs, cache=CACHE_LD)
            for i in range(1, n_iter):
                j += 1
                j = gl.where(j == wrap, 0, j)
                nxt_base = (t_first + j * t_step) * TILE
                xn = buffer_load(src + nxt_base, offs, cache=CACHE_LD)
                x = _pin(x, VEC * UNROLL)
                if OP == 0:
                    acc += x
                else:
                    buffer_store(x, dst + cur_base, offs, cache=CACHE_ST)
                x = xn
                cur_base = nxt_base
            if OP == 0:
                acc += x
            else:
                buffer_store(x, dst + cur_base, offs, cache=CACHE_ST)

        if RECORD:
            t1, c1 = _timestamps(NW, True)
            k = lane.to(gl.int64)
            v = gl.where(k == 2, t0, gl.where(k == 3, t1, gl.where(k == 6, c0, c1)))
            gl.store(rec + pid * _REC_FIELDS + k, v, mask=(k == 2) | (k == 3) | (k == 6) | (k == 7))

        keep = gl.sum(lds_a.slice(0, 64 * NW).load(small).to(gl.int32), axis=0)
        if LDS_B > 0:
            keep += gl.sum(lds_b.slice(0, 64 * NW).load(small).to(gl.int32), axis=0)
        gl.store(out + worker, gl.sum(acc, axis=0) + keep.to(gl.float32))


@dataclass
class BWConfig:
    op: str = "read"
    mode: str = "rr"
    pattern: str = "chunked"
    W: int = 256  # active workgroups
    n_iter: int = 1  # tiles per worker, summed over all passes
    passes: int = 1  # times each worker re-reads its footprint of n_iter / passes tiles
    num_warps: int = 8
    unroll: int = 4
    vec: int = 4  # fp32 elements per thread per load -> 16 B
    wpc: int = 1  # max workgroups per CU (LDS occupancy limit)
    plain_grid: bool = False  # MODE_RR only: launch exactly W workgroups instead of 256 * wpc
    xcd_mask: tuple = ()  # MODE_SUBSET: XCDs to use
    per_xcd: int = 32  # MODE_SUBSET: workgroups per XCD
    hwsel_caps: dict = field(default_factory=dict)  # MODE_HWSEL: {cu_key: max active workgroups on that CU}
    hwsel_oversub: int = 2  # MODE_HWSEL: grid = 256 * wpc * oversub candidates
    cache_ld: str = ""
    cache_st: str = ""
    record: bool = True

    @property
    def tile_elems(self):
        return self.vec * 64 * self.num_warps * self.unroll

    @property
    def wrap(self):
        assert self.n_iter % self.passes == 0, "n_iter must be a multiple of passes"
        return self.n_iter // self.passes

    @property
    def elems(self):
        """Elements moved, counting every pass."""
        return self.W * self.n_iter * self.tile_elems

    @property
    def footprint_elems(self):
        """Distinct elements touched (buffer size needed)."""
        return self.W * self.wrap * self.tile_elems

    @property
    def bytes_moved(self):
        """Read + write bytes over all passes (unique HBM traffic when passes == 1)."""
        return self.elems * 4 * (2 if self.op == "copy" else 1)

    @property
    def grid(self):
        if self.mode == "rr" and self.plain_grid:
            return self.W
        if self.mode == "hwsel":
            return NUM_CUS * self.wpc * self.hwsel_oversub
        return NUM_CUS * self.wpc


class Buffers:
    """Device buffers sized for the largest config. src/dst are fp32."""

    def __init__(self, max_elems, dst_elems=None, device="cuda"):
        self.src = torch.ones(max_elems, dtype=torch.float32, device=device)
        self.dst = torch.empty(max_elems if dst_elems is None else dst_elems, dtype=torch.float32, device=device)
        # Up to 8 workgroups per CU with HWSEL's 2x oversubscribed grid.
        self.out = torch.empty(NUM_CUS * 16, dtype=torch.float32, device=device)
        self.rec = torch.zeros(NUM_CUS * 16 * REC_FIELDS, dtype=torch.int64, device=device)
        self.sel = torch.zeros(HWSEL_KEYS, dtype=torch.int32, device=device)
        self.counters = torch.zeros(HWSEL_KEYS + 1, dtype=torch.int32, device=device)
        self._sel_key = None


def _rank_table(xcd_mask):
    table = 0
    ranks = {x: r for r, x in enumerate(sorted(xcd_mask))}
    for x in range(NUM_XCDS):
        table |= (ranks.get(x, 15) & 15) << (4 * x)
    return table


def prepare(cfg: BWConfig, bufs: Buffers):
    """Returns a zero-arg callable that launches cfg. `launch.pre` must run before each launch (HWSEL counter reset)."""
    need = cfg.footprint_elems
    assert need <= bufs.src.numel(), f"config needs {need} elems, buffer has {bufs.src.numel()}"
    if cfg.op != "read":
        assert need <= bufs.dst.numel(), f"config needs {need} elems, dst has {bufs.dst.numel()}"
    lds_a, lds_b = LDS_FOR_WPC[cfg.wpc]
    if cfg.mode == "hwsel":
        key = tuple(sorted(cfg.hwsel_caps.items()))
        if bufs._sel_key != key:
            bufs.sel.zero_()
            for k, c in cfg.hwsel_caps.items():
                bufs.sel[k] = c
            bufs._sel_key = key
    rank_table = _rank_table(cfg.xcd_mask) if cfg.mode == "subset" else 0
    grid = (cfg.grid, )
    args = (bufs.src, bufs.dst, bufs.out, bufs.rec, bufs.sel, bufs.counters, cfg.W, cfg.n_iter, cfg.wrap,
            rank_table, cfg.per_xcd)
    kwargs = dict(OP=OPS[cfg.op], MODE=MODES[cfg.mode], PATTERN=PATTERNS[cfg.pattern], VEC=cfg.vec,
                  UNROLL=cfg.unroll, WPC=cfg.wpc, LDS_A=lds_a, LDS_B=lds_b, RECORD=cfg.record,
                  CACHE_LD=cfg.cache_ld, CACHE_ST=cfg.cache_st, num_warps=cfg.num_warps)
    counters = bufs.counters

    def launch():
        bw_kernel[grid](*args, **kwargs)

    launch.pre = counters.zero_ if cfg.mode == "hwsel" else (lambda: None)
    return launch


def read_records(cfg: BWConfig, bufs: Buffers):
    """Per-workgroup records of the last launch, as a dict of CPU int64 tensors."""
    r = bufs.rec[:cfg.grid * REC_FIELDS].view(cfg.grid, REC_FIELDS).cpu()
    hw = r[:, 1]
    xcc = r[:, 0]
    return dict(
        pid=torch.arange(cfg.grid),
        xcc=xcc,
        se=(hw >> 13) & 7,
        sh=(hw >> 12) & 1,
        cu=(hw >> 8) & 15,
        cu_key=xcc * 256 + ((hw >> 13) & 7) * 32 + ((hw >> 12) & 1) * 16 + ((hw >> 8) & 15),
        t0=r[:, 2],
        t1=r[:, 3],
        worker=r[:, 4],
        active=r[:, 5].bool(),
        c0=r[:, 6],
        c1=r[:, 7],
    )


@gluon.jit
def att_stream_kernel(src, out, n_iter, wrap, UNROLL: gl.constexpr, LDS_A: gl.constexpr, LDS_B: gl.constexpr,
                      XCD0_ONLY: gl.constexpr):
    """bw_kernel's read loop without placement, records or atomics, for ATT: the trace decoder loses its place
    in bw_kernel's HWSEL branches ("Stitch Incomplete") and maps the rest of the trace to line 0. With XCD0_ONLY,
    only the workgroups on XCD 0 (pid % 8 == 0) read: every CU of XCD 0, the traced ones included, is busy while
    the rest of the chip is idle."""
    NW: gl.constexpr = gl.num_warps()
    VEC: gl.constexpr = 4
    TILE: gl.constexpr = VEC * 64 * NW * UNROLL
    small: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])
    layout: gl.constexpr = gl.BlockedLayout([VEC], [64], [NW], [0])
    slayout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    pid = gl.program_id(0)
    if XCD0_ONLY:
        if pid % 8 != 0:
            return
    lds_a = gl.allocate_shared_memory(gl.int8, [LDS_A], slayout)
    lds_a.slice(0, 64 * NW).store(gl.zeros([64 * NW], gl.int8, small))
    if LDS_B > 0:
        lds_b = gl.allocate_shared_memory(gl.int8, [LDS_B], slayout)
        lds_b.slice(0, 64 * NW).store(gl.zeros([64 * NW], gl.int8, small))
    offs = gl.arange(0, TILE, layout)
    base = pid.to(gl.int64) * wrap
    acc = gl.zeros([TILE], gl.float32, layout)
    j = pid * 0
    x = buffer_load(src + base * TILE, offs)
    for i in range(1, n_iter):
        j += 1
        j = gl.where(j == wrap, 0, j)
        xn = buffer_load(src + (base + j) * TILE, offs)
        x = _pin(x, VEC * UNROLL)
        acc += x
        x = xn
    acc += x
    keep = gl.sum(lds_a.slice(0, 64 * NW).load(small).to(gl.int32), axis=0)
    gl.store(out + pid, gl.sum(acc, axis=0) + keep.to(gl.float32))


# ----------------------------------------------------------------------------------------------
# Latency probe: workgroup 0, wave 0 walks a region of 1 KB tiles one load at a time; workgroups
# 1..W stream reads until the probe is done.
# ----------------------------------------------------------------------------------------------
PROBE_TILE = 256  # fp32 elements: 64 lanes x 16 B, one buffer_load_dwordx4 per wave = 1 KB
# Probe record: xcc_id, hw_id, rt_start, rt_end (100 MHz), clk_start, clk_end (shader clock)
PROBE_REC = 6
_PROBE_TILE = gl.constexpr(PROBE_TILE)


@gluon.jit
def _sample(NW: gl.constexpr):
    # s_memrealtime and s_memtime without their lgkmcnt wait, so the load that follows issues right after the
    # samples. The registers are only valid after _sample_after's lgkmcnt(0), which takes them as tied operands.
    return gl.inline_asm_elementwise("s_waitcnt vmcnt(0)\ns_memrealtime $0\ns_memtime $1", "=s,=s,v,~{memory}",
                                     [_one(NW)], dtype=(gl.int64, gl.int64), is_pure=False, pack=1)


@gluon.jit
def _sample_after(r0, t0):
    t1, r0, t0 = gl.inline_asm_elementwise("s_waitcnt vmcnt(0)\ns_memtime $0\ns_waitcnt lgkmcnt(0)",
                                           "=s,=s,=s,1,2,~{memory}", [r0, t0], dtype=(gl.int64, gl.int64, gl.int64),
                                           is_pure=False, pack=1)
    return gl.max(r0, axis=0), gl.max(t0, axis=0), gl.max(t1, axis=0)


@gluon.jit
def _sleep(NW: gl.constexpr):
    gl.inline_asm_elementwise("s_sleep 127\ns_mov_b32 $0, 0", "=s,v,~{memory}", [_one(NW)], dtype=gl.int32,
                              is_pure=False, pack=1)


@gluon.jit
def probe_kernel(
    src,
    samples,
    prec,
    pout,
    n_steps,
    region,
    start,
    n_sleep,
    ssrc,
    srec,
    flag,
    W,
    s_wrap,
    s_max,
    UNROLL: gl.constexpr,
    LDS_A: gl.constexpr,
    LDS_B: gl.constexpr,
    STREAM_MFMA: gl.constexpr = False,
):
    """samples[3i] = shader cycles from just before probe load i issues to just after it returns; samples[3i+1],
    samples[3i+2] = s_memtime and s_memrealtime (100 MHz) just before it issues, for the local clock.

    Probe loads walk tiles start, start+1, ..., start+region-1 and wrap, so pass p of the region starts at
    sample p * region. Streamers (pid 1..W) re-read their s_wrap tiles of ssrc, double-buffered in registers,
    until the probe sets *flag or s_max iterations; srec gets (iterations, rt_start, rt_end) per streamer.
    With STREAM_MFMA they run MFMAs on registers instead and load nothing.
    """
    NW: gl.constexpr = gl.num_warps()
    VEC: gl.constexpr = 4
    small: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])
    layout: gl.constexpr = gl.BlockedLayout([VEC], [64], [NW], [0])
    slayout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    pid = gl.program_id(0)
    lane = gl.arange(0, 64 * NW, small)
    if pid <= W:
        # One workgroup per CU, as in bw_kernel.
        lds_a = gl.allocate_shared_memory(gl.int8, [LDS_A], slayout)
        lds_a.slice(0, 64 * NW).store(gl.zeros([64 * NW], gl.int8, small))
        if LDS_B > 0:
            lds_b = gl.allocate_shared_memory(gl.int8, [LDS_B], slayout)
            lds_b.slice(0, 64 * NW).store(gl.zeros([64 * NW], gl.int8, small))
        keep = gl.sum(lds_a.slice(0, 64 * NW).load(small).to(gl.int32), axis=0)
        if LDS_B > 0:
            keep += gl.sum(lds_b.slice(0, 64 * NW).load(small).to(gl.int32), axis=0)

        if pid == 0:
            offs = gl.arange(0, _PROBE_TILE * NW, layout)
            acc = gl.zeros([_PROBE_TILE * NW], gl.float32, layout)
            xcc = _getreg(NW, "HW_REG_XCC_ID") & 15
            hw = _getreg(NW, "HW_REG_HW_ID")
            for _ in range(n_sleep):
                _sleep(NW)
            rt0, ck0 = _timestamps(NW, False)
            j = start
            r = pid * 0
            for i in range(n_steps):
                r0, t0 = _sample(NW)
                if NW == 1:
                    x = buffer_load(src + j.to(gl.int64) * _PROBE_TILE, offs)
                else:
                    # Only wave 0 loads; the other waves of the workgroup issue fully masked loads.
                    x = buffer_load(src + j.to(gl.int64) * _PROBE_TILE, offs, mask=offs < _PROBE_TILE)
                ra, a, b = _sample_after(r0, t0)
                acc += x
                k3 = lane.to(gl.int64)
                gl.store(samples + i * 3 + k3, gl.where(k3 == 0, b - a, gl.where(k3 == 1, a, ra)), mask=k3 < 3)
                j += 1
                r += 1
                j = gl.where(r == region, start, j)
                r = gl.where(r == region, 0, r)
            rt1, ck1 = _timestamps(NW, True)
            gl.atomic_add(flag, 1)
            k = lane.to(gl.int64)
            v = gl.where(k == 0, xcc.to(gl.int64), gl.where(k == 1, hw.to(gl.int64), gl.where(
                k == 2, rt0, gl.where(k == 3, rt1, gl.where(k == 4, ck0, ck1)))))
            gl.store(prec + k, v, mask=k < 6)
            gl.store(pout, gl.sum(acc, axis=0) + keep.to(gl.float32))
        elif STREAM_MFMA:
            # Matrix cores only, no memory traffic: draws power, so the clock drops under the power cap.
            mlayout: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                                          warps_per_cta=[2, NW // 2])
            # Hashed operands: constant ones barely toggle the datapath and draw too little power to move the clock.
            la: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mlayout, k_width=8)
            lb: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mlayout, k_width=8)
            ra = gl.arange(0, 64, gl.SliceLayout(1, la))[:, None] * 64 + gl.arange(0, 64, gl.SliceLayout(0, la))[None, :]
            rb = gl.arange(0, 64, gl.SliceLayout(1, lb))[:, None] * 64 + gl.arange(0, 64, gl.SliceLayout(0, lb))[None, :]
            ma = (((ra * 1103515245 + pid * 12345) >> 9) & 1023).to(gl.float32) * (1.0 / 512) - 1.0
            mb = (((rb * 22695477 + pid * 7919) >> 9) & 1023).to(gl.float32) * (1.0 / 512) - 1.0
            ma = ma.to(gl.bfloat16)
            mb = mb.to(gl.bfloat16)
            m0 = gl.zeros([64, 64], gl.float32, mlayout)
            m1 = gl.zeros([64, 64], gl.float32, mlayout)
            st0, sc0 = _timestamps(NW, False)
            it = pid * 0
            go = pid * 0 + 1
            while go > 0:
                for _u in range(64):
                    m0 = gl.amd.cdna3.mfma(ma, mb, m0)
                    m1 = gl.amd.cdna3.mfma(ma, mb, m1)
                it += 64
                f = gl.atomic_add(flag, 0)
                go = gl.where((f == 0) & (it < s_max), 1, 0)
            st1, sc1 = _timestamps(NW, True)
            sk = lane.to(gl.int64)
            sv = gl.where(sk == 0, it.to(gl.int64) * 0, gl.where(sk == 1, st0, st1))
            gl.store(srec + pid * 3 + sk, sv, mask=sk < 3)
            gl.store(pout + pid, gl.sum(gl.sum(m0 + m1, axis=1), axis=0) + keep.to(gl.float32))
        else:
            TILE: gl.constexpr = VEC * 64 * NW * UNROLL
            soffs = gl.arange(0, TILE, layout)
            sacc = gl.zeros([TILE], gl.float32, layout)
            base = (pid - 1).to(gl.int64) * s_wrap
            st0, sc0 = _timestamps(NW, False)
            sj = pid * 0
            sx = buffer_load(ssrc + base * TILE, soffs)
            it = pid * 0
            go = pid * 0 + 1
            while go > 0:
                for _u in range(64):
                    sj += 1
                    sj = gl.where(sj == s_wrap, 0, sj)
                    sxn = buffer_load(ssrc + (base + sj) * TILE, soffs)
                    sx = _pin(sx, VEC * UNROLL)
                    sacc += sx
                    sx = sxn
                it += 64
                f = gl.atomic_add(flag, 0)
                go = gl.where((f == 0) & (it < s_max), 1, 0)
            sacc += sx
            st1, sc1 = _timestamps(NW, True)
            sk = lane.to(gl.int64)
            sv = gl.where(sk == 0, it.to(gl.int64), gl.where(sk == 1, st0, st1))
            gl.store(srec + pid * 3 + sk, sv, mask=sk < 3)
            gl.store(pout + pid, gl.sum(sacc, axis=0) + keep.to(gl.float32))
