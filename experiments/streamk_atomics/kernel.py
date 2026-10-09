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
"""What one stream-K tile ending costs when it is an atomic add instead of a store or a read.

Every active program holds a synthetic 256x256 fp32 "accumulator" in v14's MFMA layout (four
128x128 quadrants, 4 warps), loaded from `src` before the clock starts. It then runs one ending,
chosen by MODE, and stamps s_memrealtime (100 MHz) before and after it, each stamp after
s_waitcnt vmcnt(0) and a workgroup barrier, so the interval covers issue and completion.

Programs pid < n_active take part, so round-robin dispatch spreads them over the XCDs; the
others exit. Active program i works on tile i // writers, so consecutive programs (on different
XCDs) add into the same tile. All active programs meet at a grid barrier before the first stamp,
so they start the ending together.
"""

import triton  # before torch: torch's HIP pulls in ROCm's shared libLLVM, which breaks libtriton's
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

NUM_PROGRAMS = 256
NUM_WARPS = 4
TILE = 256
QUAD = 128 * 128
TILE_ELEMS = TILE * TILE
C_N = 4096  # C is a 4096x4096 bf16 matrix: 16 x 16 tiles of 256x256
REC_FIELDS = 4  # t_start, t_mid, t_end (100 MHz ticks), xcc id

MODES = {
    "store": 0,  # fp32 partial store, lane-contiguous (the stream-K contributor today)
    "load": 1,  # fp32 partial read from a host-filled slot, lane-contiguous (the owner today)
    "atom_lane": 2,  # fp32 atomic add, lane-contiguous map (4 consecutive dwords per lane)
    "atom_coal": 3,  # fp32 atomic add, coalesced map (64 lanes on 64 consecutive dwords)
    "c_store": 4,  # bf16 C store, v14's layout (8 bf16 per lane)
    "c_atom_acc": 5,  # bf16 pk_add into row-major C straight from the MFMA layout
    "c_atom_coal": 6,  # bf16 pk_add into row-major C after convert_layout to 2 bf16 per lane, 64 lanes per row
    "pair": 7,  # writer 0: c_store + release flag; writer 1: wait, then c_atom_coal
    "owner": 8,  # read one host-filled fp32 partial, add, store C: what "pair"'s writer 1 replaces
    "pair_today": 9,  # writer 0: store + release flag; writer 1: wait, then owner (today's one-peer fixup)
}

_QUAD = gl.constexpr(QUAD)
_C_N = gl.constexpr(C_N)
_REC_FIELDS = gl.constexpr(REC_FIELDS)


@gluon.jit
def _one():
    return gl.zeros([1], gl.int32, gl.BlockedLayout([1], [64], [gl.num_warps()], [0]))


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
def _drain_and_barrier():
    gl.inline_asm_elementwise(
        "s_waitcnt vmcnt(0)\ns_barrier\nv_mov_b32 $0, $1", "=v,v,~{memory}", [_one()],
        dtype=gl.int32, is_pure=False, pack=1,
    )


@gluon.jit
def _pin(x):
    """Side-effecting no-op on every register of x: the loads that feed x must complete here and
    cannot be sunk into a later branch."""
    return gl.inline_asm_elementwise("; pin", "=v,=v,=v,=v,0,1,2,3", [x], dtype=gl.float32, is_pure=False,
                                     pack=4)


@gluon.jit
def lane_offsets(mfmaLayout: gl.constexpr):
    """[16 register vectors][4 warps][64 lanes][4 fp32]: streamk_costs' lane-contiguous partial.
    The column term is c plus terms constant over 4 columns, so stores stay 16 bytes wide."""
    r = gl.arange(0, 128, gl.SliceLayout(1, mfmaLayout))
    c = gl.arange(0, 128, gl.SliceLayout(0, mfmaLayout))
    f = (r % 16) * 4 + (r // 16 % 2) * 512 + (r // 32 % 2) * 4096 + (r // 64) * 8192
    g = c + (c // 4 % 4) * 60 + (c // 16 % 2) * 240 + (c // 32 % 2) * 992 + (c // 64) * 1984
    return f[:, None] + g[None, :]


@gluon.jit
def coal_offsets(mfmaLayout: gl.constexpr):
    """[64 registers][4 warps][64 lanes] fp32: for every register, a wave's 64 lanes cover 256
    contiguous bytes. Bases: lanes row 1, 2, 4, 8, col 4, 8; warps col 16, row 16; registers col
    1, 2, 32, 64, row 32, 64."""
    r = gl.arange(0, 128, gl.SliceLayout(1, mfmaLayout))
    c = gl.arange(0, 128, gl.SliceLayout(0, mfmaLayout))
    f = (r % 16) + (r // 16 % 2) * 128 + (r // 32 % 2) * 4096 + (r // 64) * 8192
    g = (c // 4 % 4) * 16 + (c // 16 % 2) * 64 + (c % 4) * 256 + (c // 32 % 2) * 1024 + (c // 64) * 2048
    return f[:, None] + g[None, :]


@gluon.jit
def c_offsets(layout: gl.constexpr):
    r = gl.arange(0, 128, gl.SliceLayout(1, layout))
    c = gl.arange(0, 128, gl.SliceLayout(0, layout))
    return r[:, None] * _C_N + c[None, :]


@gluon.jit
def c_quadrant(c_ptr, t, q: gl.constexpr):
    """Quadrant q (v14 order tl, bl, tr, br) of 256x256 tile t of C."""
    row = (t // 16) * 256 + (q % 2) * 128
    col = (t % 16) * 256 + (q // 2) * 128
    return c_ptr + row.to(gl.int64) * _C_N + col


@gluon.jit
def c_store(acc, c_ptr, t, q: gl.constexpr):
    gStoreLayoutC: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
    v = gl.convert_layout(acc.to(gl.bfloat16), gStoreLayoutC)
    gl.amd.cdna4.buffer_store(stored_value=v, ptr=c_quadrant(c_ptr, t, q), offsets=c_offsets(gStoreLayoutC))


@gluon.jit
def c_atom_coal(acc, c_ptr, t, q: gl.constexpr):
    atomLayoutC: gl.constexpr = gl.BlockedLayout([1, 2], [1, 64], [4, 1], [1, 0])
    v = gl.convert_layout(acc.to(gl.bfloat16), atomLayoutC)
    gl.amd.cdna4.buffer_atomic_add(c_quadrant(c_ptr, t, q), c_offsets(atomLayoutC), v, sem="relaxed",
                                   scope="gpu")


@gluon.jit
def ending(acc, q: gl.constexpr, MODE: gl.constexpr, aidx, t, p_ptr, w_ptr, c_ptr, sink, mfmaLayout: gl.constexpr):
    if MODE == 0:
        gl.amd.cdna4.buffer_store(stored_value=acc, ptr=p_ptr + (aidx.to(gl.int64) * 4 + q) * _QUAD,
                                  offsets=lane_offsets(mfmaLayout))
    elif MODE == 1:
        x = gl.amd.cdna4.buffer_load(ptr=p_ptr + (aidx.to(gl.int64) * 4 + q) * _QUAD, offsets=lane_offsets(mfmaLayout))
        x = _pin(x)
        if sink != 0:
            gl.amd.cdna4.buffer_store(stored_value=acc + x, ptr=w_ptr + (aidx.to(gl.int64) * 4 + q) * _QUAD,
                                      offsets=lane_offsets(mfmaLayout))
    elif MODE == 2:
        gl.amd.cdna4.buffer_atomic_add(w_ptr + (t.to(gl.int64) * 4 + q) * _QUAD, lane_offsets(mfmaLayout), acc,
                                       sem="relaxed", scope="gpu")
    elif MODE == 3:
        gl.amd.cdna4.buffer_atomic_add(w_ptr + (t.to(gl.int64) * 4 + q) * _QUAD, coal_offsets(mfmaLayout), acc,
                                       sem="relaxed", scope="gpu")
    elif MODE == 4:
        c_store(acc, c_ptr, t, q)
    elif MODE == 5:
        gl.amd.cdna4.buffer_atomic_add(c_quadrant(c_ptr, t, q), c_offsets(mfmaLayout), acc.to(gl.bfloat16),
                                       sem="relaxed", scope="gpu")
    elif MODE == 6:
        c_atom_coal(acc, c_ptr, t, q)
    else:
        x = gl.amd.cdna4.buffer_load(ptr=p_ptr + (aidx.to(gl.int64) * 4 + q) * _QUAD, offsets=lane_offsets(mfmaLayout))
        c_store(acc + x, c_ptr, t, q)


@gluon.jit
def atomics_kernel(
    src_ptr,
    p_ptr,
    w_ptr,
    c_ptr,
    flags_ptr,
    counter_ptr,
    rec_ptr,
    n_active,
    writers,
    sink,
    MODE: gl.constexpr,
):
    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 2]
    )
    pid = gl.program_id(0)
    if pid < n_active:
        aidx = pid
        t = aidx // writers
        j = aidx % writers
        src = src_ptr + aidx.to(gl.int64) * (4 * _QUAD)
        offs = lane_offsets(mfmaLayout)
        a0 = gl.amd.cdna4.buffer_load(ptr=src, offsets=offs)
        a1 = gl.amd.cdna4.buffer_load(ptr=src + _QUAD, offsets=offs)
        a2 = gl.amd.cdna4.buffer_load(ptr=src + 2 * _QUAD, offsets=offs)
        a3 = gl.amd.cdna4.buffer_load(ptr=src + 3 * _QUAD, offsets=offs)
        _drain_and_barrier()

        gl.atomic_add(counter_ptr, 1, sem="relaxed", scope="gpu")
        while gl.atomic_add(counter_ptr, 0, sem="relaxed", scope="gpu") < n_active:
            pass

        t0 = _stamp()
        t1 = t0
        if MODE == 7:
            if j == 0:
                c_store(a0, c_ptr, t, 0)
                c_store(a1, c_ptr, t, 1)
                c_store(a2, c_ptr, t, 2)
                c_store(a3, c_ptr, t, 3)
                _drain_and_barrier()
                gl.atomic_xchg(flags_ptr + t, 1, sem="release", scope="gpu")
                t1 = _stamp()
            else:
                while gl.atomic_add(flags_ptr + t, 0, sem="relaxed", scope="gpu") == 0:
                    pass
                gl.atomic_add(flags_ptr + t, 0, sem="acquire", scope="gpu")
                t1 = _stamp()
                c_atom_coal(a0, c_ptr, t, 0)
                c_atom_coal(a1, c_ptr, t, 1)
                c_atom_coal(a2, c_ptr, t, 2)
                c_atom_coal(a3, c_ptr, t, 3)
        elif MODE == 9:
            if j == 0:
                ending(a0, 0, 0, aidx, t, p_ptr, w_ptr, c_ptr, sink, mfmaLayout)
                ending(a1, 1, 0, aidx, t, p_ptr, w_ptr, c_ptr, sink, mfmaLayout)
                ending(a2, 2, 0, aidx, t, p_ptr, w_ptr, c_ptr, sink, mfmaLayout)
                ending(a3, 3, 0, aidx, t, p_ptr, w_ptr, c_ptr, sink, mfmaLayout)
                _drain_and_barrier()
                gl.atomic_xchg(flags_ptr + t, 1, sem="release", scope="gpu")
                t1 = _stamp()
            else:
                while gl.atomic_add(flags_ptr + t, 0, sem="relaxed", scope="gpu") == 0:
                    pass
                gl.atomic_add(flags_ptr + t, 0, sem="acquire", scope="gpu")
                t1 = _stamp()
                peer = aidx - 1
                ending(a0, 0, 8, peer, t, p_ptr, w_ptr, c_ptr, sink, mfmaLayout)
                ending(a1, 1, 8, peer, t, p_ptr, w_ptr, c_ptr, sink, mfmaLayout)
                ending(a2, 2, 8, peer, t, p_ptr, w_ptr, c_ptr, sink, mfmaLayout)
                ending(a3, 3, 8, peer, t, p_ptr, w_ptr, c_ptr, sink, mfmaLayout)
        else:
            ending(a0, 0, MODE, aidx, t, p_ptr, w_ptr, c_ptr, sink, mfmaLayout)
            ending(a1, 1, MODE, aidx, t, p_ptr, w_ptr, c_ptr, sink, mfmaLayout)
            ending(a2, 2, MODE, aidx, t, p_ptr, w_ptr, c_ptr, sink, mfmaLayout)
            ending(a3, 3, MODE, aidx, t, p_ptr, w_ptr, c_ptr, sink, mfmaLayout)
        t2 = _stamp()

        rec = rec_ptr + pid * _REC_FIELDS + _one()
        gl.store(rec, t0)
        gl.store(rec + 1, t1)
        gl.store(rec + 2, t2)
        gl.store(rec + 3, _xcc())


class Workspace:
    """Buffers for up to NUM_PROGRAMS active programs and NUM_PROGRAMS tiles."""

    def __init__(self, device="cuda"):
        self.src = torch.zeros(NUM_PROGRAMS * TILE_ELEMS, device=device, dtype=torch.float32)
        self.P = torch.zeros(NUM_PROGRAMS * TILE_ELEMS, device=device, dtype=torch.float32)
        self.W = torch.zeros(NUM_PROGRAMS * TILE_ELEMS, device=device, dtype=torch.float32)
        self.C = torch.zeros((C_N, C_N), device=device, dtype=torch.bfloat16)
        self.flags = torch.zeros(NUM_PROGRAMS, device=device, dtype=torch.int32)
        self.counter = torch.zeros(1, device=device, dtype=torch.int32)
        self.rec = torch.zeros(NUM_PROGRAMS * REC_FIELDS, device=device, dtype=torch.int64)


def launch(ws, mode, n_active, writers, sink=0):
    """Re-arms the grid barrier and flags, then runs one launch. Returns the compiled kernel."""
    ws.counter.zero_()
    ws.flags.zero_()
    return atomics_kernel[(NUM_PROGRAMS, )](
        ws.src, ws.P, ws.W, ws.C, ws.flags, ws.counter, ws.rec, n_active, writers, sink,
        MODE=MODES[mode], num_warps=NUM_WARPS,
    )


def records(ws, n_active):
    """Per active program: (start, mid, end) in us relative to the earliest start, and the XCC id."""
    r = ws.rec.view(NUM_PROGRAMS, REC_FIELDS)[:n_active].cpu()
    t0 = int(r[:, 0].min())
    times = (r[:, :3] - t0).double() * 0.01
    return times, r[:, 3].tolist()


def lane_index():
    """Row-major (row, col) of a 128x128 quadrant -> offset in lane_offsets' map."""
    r = torch.arange(128)[:, None]
    c = torch.arange(128)[None, :]
    return (r % 16) * 4 + (r // 16 % 2) * 512 + (r // 32 % 2) * 4096 + (r // 64) * 8192 + \
        c % 4 + (c // 4 % 4) * 64 + (c // 16 % 2) * 256 + (c // 32 % 2) * 1024 + (c // 64) * 2048


def coal_index():
    r = torch.arange(128)[:, None]
    c = torch.arange(128)[None, :]
    return (r % 16) + (r // 16 % 2) * 128 + (r // 32 % 2) * 4096 + (r // 64) * 8192 + \
        (c // 4 % 4) * 16 + (c // 16 % 2) * 64 + (c % 4) * 256 + (c // 32 % 2) * 1024 + (c // 64) * 2048


def tile_from_slot(slot, index):
    """A 256x256 row-major tile from one 4-quadrant slot stored in the given quadrant map."""
    out = torch.empty((TILE, TILE), dtype=slot.dtype, device=slot.device)
    idx = index.to(slot.device)
    for q in range(4):
        qr, qc = (q % 2) * 128, (q // 2) * 128
        out[qr:qr + 128, qc:qc + 128] = slot[q * QUAD:(q + 1) * QUAD][idx]
    return out


def slot_from_tile(tile, index):
    slot = torch.empty(TILE_ELEMS, dtype=tile.dtype, device=tile.device)
    idx = index.to(tile.device).flatten()
    for q in range(4):
        qr, qc = (q % 2) * 128, (q // 2) * 128
        slot[q * QUAD:(q + 1) * QUAD][idx] = tile[qr:qr + 128, qc:qc + 128].flatten()
    return slot


def c_tile(C, t):
    return C[(t // 16) * TILE:(t // 16 + 1) * TILE, (t % 16) * TILE:(t % 16 + 1) * TILE]
