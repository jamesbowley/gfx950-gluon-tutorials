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
"""A synthetic reduce-scatter stream-K ending, to test exchanging partials through one XCD's L2.

Groups of n programs stand in for the n programs of a split tile. Every program holds a 256x256
fp32 "accumulator" in v14's MFMA layout (src slot + epoch, so every launch's partials differ),
stores it to its slot of P, meets its group at a tile barrier, then sums its 1/n of the tile
over the group's n partials and stores that slice of the group's output.

PLACE puts a group's programs on one XCD (0: pid % 8 is the XCD, local slot pid // 8) or
spreads them over the XCDs (1: consecutive pids). The same programs are active either way:
the first T * n local slots of every XCD, T groups per XCD.

The exchange is chosen by constexprs:
- STORE_CACHE / LOAD_CACHE: Triton cache modifiers of the partial store and the slice loads
  ("" is no cache bits; on CDNA4 ".wt" is sc0 sc1, ".cv" sc0 sc1, ".cg" and ".cs" loads sc0 nt).
- SYNC: 0 is v20's tile barrier (acq_rel arrival, release generation, acquire after the
  spin: buffer_wbl2 / buffer_inv sc1); 1 is the same counter with relaxed atomics only, the
  generation read ordered before the arrival by s_waitcnt; 2 is a flag per program holding
  the launch's epoch, polled with sc0 loads (no atomics, so it stays in one XCD's L2).
- L1_INV: buffer_inv sc0 after the barrier. Meant as an L1 invalidate, but the CDNA4 ISA
  (Table 52) defines buffer_inv sc0 as a NOP outside TG-split mode, so it does nothing here.
  The sc0 nt (.cg) and sc0 sc1 (.cv) loads miss L1 by themselves (Table 49); the variant with
  default loads (l2_def) was correct only because the reading CU had no stale copy in L1.
- PREREAD: load the program's own slot (its old contents) before the clock starts.

s_memrealtime stamps (100 MHz), each after s_waitcnt vmcnt(0) and a workgroup barrier: start
(after a grid barrier), stored, synced, done.
"""

import triton  # before torch: torch's HIP pulls in ROCm's shared libLLVM, which breaks libtriton's
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

NUM_PROGRAMS = 256
NUM_XCDS = 8
PER_XCD = NUM_PROGRAMS // NUM_XCDS
NUM_WARPS = 4
QUAD = 128 * 128
SLOT = 4 * QUAD
REC_FIELDS = 6  # start, stored, synced, done (100 MHz ticks), xcc id, group

_QUAD = gl.constexpr(QUAD)
_SLOT = gl.constexpr(SLOT)
_REC_FIELDS = gl.constexpr(REC_FIELDS)

VARIANTS = {
    # name: (STORE_CACHE, LOAD_CACHE, SYNC, L1_INV[, PREREAD])
    "mem": (".wt", ".cv", 0, 0),  # v20 today
    "wt_relaxed": (".wt", ".cv", 1, 0),  # write-through partials, relaxed barrier
    "l2_cg": ("", ".cg", 1, 1),  # default store, sc0 nt loads
    "l2_def": ("", "", 1, 1),  # default store, default loads (after the no-op buffer_inv sc0)
    "l2_cv": ("", ".cv", 1, 1),  # default store, sc0 sc1 loads
    "l2_cg_noinv": ("", ".cg", 1, 0),  # as l2_cg without the (no-op) buffer_inv sc0
    "l2_nts": (".cs", ".cg", 1, 1),  # sc0 nt partial store, sc0 nt loads
    "l2_pre": ("", ".cg", 1, 1, 1),  # l2_cg after reading the own slot into L2 before the ending
    "l2_flags": ("", ".cg", 2, 1),  # l2_cg with the flag-array barrier
    "l2_flags_pre": ("", ".cg", 2, 1, 1),  # l2_flags after the pre-read
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
def _pin(x):
    """Side-effecting no-op on every register of x, so the loads that feed x are kept."""
    return gl.inline_asm_elementwise("; pin", "=v,=v,=v,=v,0,1,2,3", [x], dtype=gl.float32, is_pure=False,
                                     pack=4)


@gluon.jit
def lane_offsets(mfmaLayout: gl.constexpr):
    """v20's lane-contiguous partial map: [16 register vectors][4 warps][64 lanes][4 fp32]."""
    r = gl.arange(0, 128, gl.SliceLayout(1, mfmaLayout))
    c = gl.arange(0, 128, gl.SliceLayout(0, mfmaLayout))
    f = (r % 16) * 4 + (r // 16 % 2) * 512 + (r // 32 % 2) * 4096 + (r // 64) * 8192
    g = c + (c // 4 % 4) * 60 + (c // 16 % 2) * 240 + (c // 32 % 2) * 992 + (c // 64) * 1984
    return f[:, None] + g[None, :]


@gluon.jit
def member_pid(g, j, T, n, PLACE: gl.constexpr):
    if PLACE == 0:
        return ((g % T) * n + j) * 8 + g // T
    else:
        return g * n + j


@gluon.jit
def flag_barrier(flags_ptr, pid, g, T, n, epoch, PLACE: gl.constexpr):
    """Every program stores the launch's epoch to its own flag, then polls its group's n flags
    with sc0 loads (L1 miss, L2 hit) until all of them hold it. No atomics: on one XCD the flags
    live in that XCD's L2. The epoch comes from the host, so nothing needs re-arming."""
    one = _one()
    gl.store(flags_ptr + pid + one * 0, epoch + one * 0)
    FL: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    m = gl.arange(0, 32, FL)
    mask = m < n
    members = gl.where(mask, member_pid(g, m, T, n, PLACE), 0)
    done = 0
    while done == 0:
        _asm("s_nop 0")  # keeps the poll inside the loop
        f = gl.amd.cdna4.buffer_load(ptr=flags_ptr, offsets=members, mask=mask, other=epoch, cache=".cg")
        done = (gl.min(f, axis=0) == epoch).to(gl.int32)


@gluon.jit
def tile_barrier(sync_ptr, g, n, RELAXED: gl.constexpr):
    cnt_ptr = sync_ptr + 2 * g
    gen_ptr = cnt_ptr + 1
    if RELAXED:
        g0 = gl.atomic_add(gen_ptr, 0, sem="relaxed", scope="gpu")
        # The generation read must complete before the arrival: if the last arrival and the
        # bump overtook it, this program would wait for a generation that never comes.
        _asm("s_waitcnt vmcnt(0)")
        arrived = gl.atomic_add(cnt_ptr, 1, sem="relaxed", scope="gpu")
        if arrived == n - 1:
            gl.atomic_xchg(cnt_ptr, 0, sem="relaxed", scope="gpu")
            gl.atomic_xchg(gen_ptr, g0 + 1, sem="relaxed", scope="gpu")
        else:
            while gl.atomic_add(gen_ptr, 0, sem="relaxed", scope="gpu") == g0:
                pass
    else:
        g0 = gl.atomic_add(gen_ptr, 0, sem="relaxed", scope="gpu")
        arrived = gl.atomic_add(cnt_ptr, 1, sem="acq_rel", scope="gpu")
        if arrived == n - 1:
            gl.atomic_xchg(cnt_ptr, 0, sem="relaxed", scope="gpu")
            gl.atomic_xchg(gen_ptr, g0 + 1, sem="release", scope="gpu")
        else:
            while gl.atomic_add(gen_ptr, 0, sem="relaxed", scope="gpu") == g0:
                pass
            gl.atomic_add(gen_ptr, 0, sem="acquire", scope="gpu")


@gluon.jit
def slice_sum(p_ptr, out_ptr, g, j, T, n, PLACE: gl.constexpr, LOAD_CACHE: gl.constexpr, NB: gl.constexpr):
    """Program j's blocks [64 j / n, 64 (j + 1) / n) of 1024 fp32, summed over the group's n slots."""
    L2D: gl.constexpr = gl.BlockedLayout([1, 4], [1, 64], [1, 4], [1, 0])
    i = gl.arange(0, NB, gl.SliceLayout(1, L2D))[:, None]
    e = gl.arange(0, 1024, gl.SliceLayout(0, L2D))[None, :]
    lo = (64 * j) // n
    hi = (64 * (j + 1)) // n
    for b0 in range(lo, hi, NB):
        blk = b0 + i
        valid = (blk < hi) & (e >= 0)
        off = blk * 1024 + e
        s = gl.zeros((NB, 1024), gl.float32, L2D)
        for m in range(n):
            slot = p_ptr + member_pid(g, m, T, n, PLACE).to(gl.int64) * _SLOT
            s += gl.amd.cdna4.buffer_load(ptr=slot, offsets=off, mask=valid, other=0.0, cache=LOAD_CACHE)
        gl.amd.cdna4.buffer_store(stored_value=s, ptr=out_ptr + g.to(gl.int64) * _SLOT, offsets=off, mask=valid)


@gluon.jit
def probe_kernel(
    src_ptr,
    p_ptr,
    out_ptr,
    sync_ptr,
    flags_ptr,
    start_ptr,
    rec_ptr,
    epoch,
    T,
    n,
    PLACE: gl.constexpr,
    STORE_CACHE: gl.constexpr,
    LOAD_CACHE: gl.constexpr,
    SYNC: gl.constexpr,
    L1_INV: gl.constexpr,
    NB: gl.constexpr,
    SLICE: gl.constexpr,
    PREREAD: gl.constexpr,
):
    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 2]
    )
    pid = gl.program_id(0)
    xcd = pid % 8
    local = pid // 8
    if local < T * n:
        if PLACE == 0:
            g = xcd * T + local // n
            j = local % n
        else:
            g = pid // n
            j = pid % n
        offs = lane_offsets(mfmaLayout)
        src = src_ptr + pid.to(gl.int64) * _SLOT
        e = epoch.to(gl.float32)
        a0 = gl.amd.cdna4.buffer_load(ptr=src, offsets=offs) + e
        a1 = gl.amd.cdna4.buffer_load(ptr=src + _QUAD, offsets=offs) + e
        a2 = gl.amd.cdna4.buffer_load(ptr=src + 2 * _QUAD, offsets=offs) + e
        a3 = gl.amd.cdna4.buffer_load(ptr=src + 3 * _QUAD, offsets=offs) + e
        if PREREAD:
            # Read the slot's old contents into L2 before the clock starts (in the GEMM, under
            # the MAC loop), so the partial store writes into valid lines.
            own = p_ptr + pid.to(gl.int64) * _SLOT
            for q in gl.static_range(4):
                _pin(gl.amd.cdna4.buffer_load(ptr=own + q * _QUAD, offsets=offs))
        _asm("s_waitcnt vmcnt(0)\ns_barrier")

        # Grid barrier, so the groups run their endings together; the host zeroes the counter.
        gl.atomic_add(start_ptr, 1, sem="relaxed", scope="gpu")
        while gl.atomic_add(start_ptr, 0, sem="relaxed", scope="gpu") < 8 * T * n:
            pass

        t0 = _stamp()
        slot = p_ptr + pid.to(gl.int64) * _SLOT
        gl.amd.cdna4.buffer_store(stored_value=a0, ptr=slot, offsets=offs, cache=STORE_CACHE)
        gl.amd.cdna4.buffer_store(stored_value=a1, ptr=slot + _QUAD, offsets=offs, cache=STORE_CACHE)
        gl.amd.cdna4.buffer_store(stored_value=a2, ptr=slot + 2 * _QUAD, offsets=offs, cache=STORE_CACHE)
        gl.amd.cdna4.buffer_store(stored_value=a3, ptr=slot + 3 * _QUAD, offsets=offs, cache=STORE_CACHE)
        t1 = _stamp()  # every wave's stores completed, then a workgroup barrier
        if SYNC == 2:
            flag_barrier(flags_ptr, pid, g, T, n, epoch, PLACE)
        else:
            tile_barrier(sync_ptr, g, n, SYNC)
        if L1_INV:
            _asm("buffer_inv sc0\ns_waitcnt vmcnt(0)")
        t2 = _stamp()
        if SLICE:
            slice_sum(p_ptr, out_ptr, g, j, T, n, PLACE, LOAD_CACHE, NB)
        t3 = _stamp()

        rec = rec_ptr + pid * _REC_FIELDS + _one()
        gl.store(rec, t0)
        gl.store(rec + 1, t1)
        gl.store(rec + 2, t2)
        gl.store(rec + 3, t3)
        gl.store(rec + 4, _xcc())
        gl.store(rec + 5, g.to(gl.int64) + _one().to(gl.int64) * 0)


class Workspace:
    def __init__(self, device="cuda"):
        self.src = torch.zeros(NUM_PROGRAMS * SLOT, device=device, dtype=torch.float32)
        self.P = torch.zeros(NUM_PROGRAMS * SLOT, device=device, dtype=torch.float32)
        self.out = torch.zeros(NUM_PROGRAMS * SLOT, device=device, dtype=torch.float32)
        self.sync = torch.zeros(2 * NUM_PROGRAMS, device=device, dtype=torch.int32)
        self.flags = torch.zeros(NUM_PROGRAMS, device=device, dtype=torch.int32)
        self.start = torch.zeros(1, device=device, dtype=torch.int32)
        self.rec = torch.zeros(NUM_PROGRAMS * REC_FIELDS, device=device, dtype=torch.int64)
        self.epoch = 100  # run-time arguments equal to 1 are specialised as constants


def launch(ws, variant, T, n, place, nb=8, slice_=True):
    """One launch with the next epoch. Returns the compiled kernel. slice_=False skips the
    slice (store and barrier only), to tell the store's memory traffic from the read's."""
    assert T * n <= PER_XCD
    sc, lc, sync, inv, *pre = VARIANTS[variant]
    ws.start.zero_()
    ws.epoch += 1
    return probe_kernel[(NUM_PROGRAMS, )](
        ws.src, ws.P, ws.out, ws.sync, ws.flags, ws.start, ws.rec, ws.epoch, T, n,
        PLACE=place, STORE_CACHE=sc, LOAD_CACHE=lc, SYNC=sync, L1_INV=inv, NB=nb, SLICE=slice_,
        PREREAD=bool(pre and pre[0]), num_warps=NUM_WARPS,
    )


def groups(T, n, place):
    """Per group, its member pids."""
    G = NUM_XCDS * T
    if place == 0:
        return [[((g % T) * n + j) * 8 + g // T for j in range(n)] for g in range(G)]
    return [[g * n + j for j in range(n)] for g in range(G)]


def expected(ws, T, n, place):
    """(G, SLOT) expected outputs for the current epoch."""
    src = ws.src.view(NUM_PROGRAMS, SLOT)
    return torch.stack([src[m].sum(0) for m in groups(T, n, place)]) + n * float(ws.epoch)


def records(ws, T, n):
    """Active programs' records as an (A, REC_FIELDS) int64 CPU tensor, A = 8 T n."""
    r = ws.rec.view(NUM_PROGRAMS, REC_FIELDS)[:8 * T * n].cpu()
    return r
