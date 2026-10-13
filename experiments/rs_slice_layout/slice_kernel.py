"""A synthetic one-tile reduce-scatter ending with four ways to read the slice and store C.

W = S n programs, n per 256x256 tile, every tile's programs on one XCD (as stream-K places them).
Each program holds a 256x256 fp32 accumulator in v20's MFMA layout (four 128x128 quadrants);
member m's value at (row, col) is (row * 256 + col) % 7 + m + 1, so every sum is an exact bf16
integer that depends on the position. The ending, as v20's reduce-scatter (`.wt` stores, drain,
relaxed gpu counter, `.cv` loads):
1. publish: store the partial to the program's slot, drain, s_barrier;
2. sync: add 1 to the tile's counter, spin until it reaches n;
3. read: load slice j (blocks [64 j / n, 64 (j + 1) / n)) of the n partials and sum them in fp32;
4. C store: convert to bf16 and store the slice to C (row-major, rows C_N elements apart).
Stamps (s_memrealtime after s_waitcnt vmcnt(0) and s_barrier) between the steps.

VARIANT
- 0 v20: lane-contiguous partials from the MFMA layout; the slice is v20's (NB, 32, 32) L3 tensor;
  its C store writes 8 bytes per lane, 4 neighbouring lanes on 4 rows.
- 1 v20_old_cvt: the same reads, then convert_layout to the columns-first L3 (v21's dropped attempt).
- 2 v20_store2d: the same reads, reshaped to the slice's 2-D band and converted to gStoreLayoutC
  (16 bytes per lane, 16 lanes on 256 contiguous bytes of a row).
- 3 mfma_read: the slice loaded as a 2-D band in the MFMA layout (the owner's map), summed, converted
  to gStoreLayoutC like the owner's store_c.
- 4 writer_cvt: the writer converts each quadrant to W_ROW (4 fp32 = 16 bytes per lane, 32 lanes on a
  512-byte row) and stores it row-major; the reader loads its band of that row-major partial in
  gStoreLayoutC (8 fp32 per lane) and stores bf16 C with no conversion.
- 5 writer_cvt8: the same, but the writer converts to gStoreLayoutC itself: 8 fp32 = 32 bytes per lane,
  so each 16-byte store instruction leaves 16-byte holes between neighbouring lanes.
"""

import triton  # before torch
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

import kernel as K

NUM_PROGRAMS = 256
SLOT = 4 * 128 * 128  # fp32 elements of one partial
C_N = K.C_N
REC = 8
VARIANTS = {"v20": 0, "v20_old_cvt": 1, "v20_store2d": 2, "mfma_read": 3, "writer_cvt": 4, "writer_cvt8": 5}
W_ROW = gl.BlockedLayout([1, 4], [2, 32], [4, 1], [1, 0])

_NUM_PROGRAMS = gl.constexpr(NUM_PROGRAMS)
_REC = gl.constexpr(REC)
_SLOT = gl.constexpr(SLOT)
_C_N = gl.constexpr(C_N)
_MFMA = gl.constexpr(K.MFMA)
_STORE2D = gl.constexpr(K.STORE2D)
_L3 = gl.constexpr(K.L3)
_L3_OLD = gl.constexpr(K.L3_OLD)
_W_ROW = gl.constexpr(W_ROW)


@gluon.jit
def _one():
    return gl.zeros([1], gl.int32, gl.BlockedLayout([1], [64], [gl.num_warps()], [0]))


@gluon.jit
def _stamp():
    return gl.max(gl.inline_asm_elementwise("s_waitcnt vmcnt(0)\ns_barrier\ns_memrealtime $0\ns_waitcnt lgkmcnt(0)",
                                            "=s,v,~{memory}", [_one()], dtype=gl.int64, is_pure=False, pack=1), axis=0)


@gluon.jit
def _drain():
    gl.inline_asm_elementwise("s_waitcnt vmcnt(0)\ns_barrier\nv_mov_b32 $0, $1", "=v,v,~{memory}", [_one()],
                              dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def _lane_off(qrow, c):
    """v20's lane_contiguous_offsets for quadrant rows qrow and columns c (any 2-D layout)."""
    f = (qrow % 16) * 4 + (qrow // 16 % 2) * 512 + (qrow // 32 % 2) * 4096 + (qrow // 64) * 8192
    g = c + (c // 4 % 4) * 60 + (c // 16 % 2) * 240 + (c // 32 % 2) * 992 + (c // 64) * 1984
    return f + g


@gluon.jit
def _acc_quadrant(q: gl.constexpr, m, layout: gl.constexpr):
    r = gl.arange(0, 128, gl.SliceLayout(1, layout))[:, None]
    c = gl.arange(0, 128, gl.SliceLayout(0, layout))[None, :]
    row = (q % 2) * 128 + r
    col = (q // 2) * 128 + c
    return ((row * 256 + col) % 7 + m + 1).to(gl.float32), r, c


@gluon.jit
def _publish(slot_ptr, m, VARIANT: gl.constexpr):
    for q in gl.static_range(4):
        acc, r, c = _acc_quadrant(q, m, _MFMA)
        if VARIANT == 4:
            accs = gl.convert_layout(acc, _W_ROW)
            rs = gl.arange(0, 128, gl.SliceLayout(1, _W_ROW))[:, None]
            cs = gl.arange(0, 128, gl.SliceLayout(0, _W_ROW))[None, :]
            gl.amd.cdna4.buffer_store(stored_value=accs, ptr=slot_ptr + q * 16384, offsets=rs * 128 + cs, cache=".wt")
        elif VARIANT == 5:
            accs = gl.convert_layout(acc, _STORE2D)
            rs = gl.arange(0, 128, gl.SliceLayout(1, _STORE2D))[:, None]
            cs = gl.arange(0, 128, gl.SliceLayout(0, _STORE2D))[None, :]
            gl.amd.cdna4.buffer_store(stored_value=accs, ptr=slot_ptr + q * 16384, offsets=rs * 128 + cs, cache=".wt")
        else:
            gl.amd.cdna4.buffer_store(stored_value=acc, ptr=slot_ptr + q * 16384, offsets=_lane_off(r, c), cache=".wt")


@gluon.jit
def _band(layout: gl.constexpr, b0, ROWS: gl.constexpr):
    """Slice rows and columns as a ROWS x 128 band: quadrant, quadrant row, tile row, tile column."""
    r = gl.arange(0, ROWS, gl.SliceLayout(1, layout))[:, None]
    c = gl.arange(0, 128, gl.SliceLayout(0, layout))[None, :]
    qq = b0 // 16 + r // 128
    qrow = (b0 % 16) * 8 + r % 128
    return qq, qrow, c, (qq % 2) * 128 + qrow, (qq // 2) * 128 + c


@gluon.jit
def _to_band(x, NB: gl.constexpr):
    """(NB, 32, 32) blocks of one slice -> its (NB * 8, 128) band. Block index bits from the lowest:
    col 32, col 64, row 32, row 64, row 128 (the second quadrant)."""
    if NB == 4:
        y = gl.permute(gl.reshape(x, [2, 2, 32, 32]), [2, 0, 1, 3])
    elif NB == 8:
        y = gl.permute(gl.reshape(x, [2, 2, 2, 32, 32]), [0, 3, 1, 2, 4])
    elif NB == 16:
        y = gl.permute(gl.reshape(x, [2, 2, 2, 2, 32, 32]), [0, 1, 4, 2, 3, 5])
    else:
        y = gl.permute(gl.reshape(x, [2, 2, 2, 2, 2, 32, 32]), [0, 1, 2, 5, 3, 4, 6])
    return gl.reshape(y, [NB * 8, 128])


@gluon.jit
def rs_kernel(p_ptr, c_ptr, cnt_ptr, gcnt_ptr, rec_ptr, W, N: gl.constexpr, VARIANT: gl.constexpr):
    pid = gl.program_id(0)
    xcd = pid % 8
    s = pid // 8
    gpx = (W + 8 * N - 1) // (8 * N)
    xu = W // (N * gpx)
    is_w = (xcd < xu) & (s < gpx * N)
    g = xcd * gpx + s // N
    m = s % N
    NB: gl.constexpr = 64 // N
    ROWS: gl.constexpr = NB * 8
    b0 = m * NB
    tile_c = c_ptr + ((g // 16) * 256).to(gl.int64) * _C_N + (g % 16) * 256
    gl.atomic_add(gcnt_ptr, 1, sem="relaxed", scope="gpu")
    while gl.atomic_add(gcnt_ptr, 0, sem="relaxed", scope="gpu") < _NUM_PROGRAMS:
        pass
    t0 = _stamp()
    t1 = t0
    t2 = t0
    t3 = t0
    if is_w:
        _publish(p_ptr + (g * N + m).to(gl.int64) * _SLOT, m, VARIANT)
        _drain()
        t1 = _stamp()
        gl.atomic_add(cnt_ptr + g * 32, 1, sem="relaxed", scope="gpu")
        while gl.atomic_add(cnt_ptr + g * 32, 0, sem="relaxed", scope="gpu") < N:
            pass
        gl.inline_asm_elementwise("s_nop 0\nv_mov_b32 $0, $1", "=v,v,~{memory}", [_one()], dtype=gl.int32,
                                  is_pure=False, pack=1)
        t2 = _stamp()
        if VARIANT <= 2:
            v = gl.arange(0, NB, gl.SliceLayout(1, gl.SliceLayout(2, _L3)))[:, None, None]
            r = gl.arange(0, 32, gl.SliceLayout(0, gl.SliceLayout(2, _L3)))[None, :, None]
            c = gl.arange(0, 32, gl.SliceLayout(0, gl.SliceLayout(1, _L3)))[None, None, :]
            blk = b0 + v
            in_blk = (r % 16) * 4 + (r // 16) * 512 + c + (c // 4 % 4) * 60 + (c // 16) * 240
            ssum = gl.zeros([NB, 32, 32], gl.float32, _L3)
            for j in gl.static_range(N):
                ssum += gl.amd.cdna4.buffer_load(ptr=p_ptr + (g * N + j).to(gl.int64) * _SLOT,
                                                 offsets=blk * 1024 + in_blk, cache=".cv")
            t3 = _stamp()
            cb = ssum.to(gl.bfloat16)
            if VARIANT == 0:
                q = blk // 16
                vv = blk % 16
                row = (q % 2) * 128 + (vv // 4 % 2) * 32 + (vv // 8) * 64 + r
                col = (q // 2) * 128 + (vv % 2) * 32 + (vv // 2 % 2) * 64 + c
                gl.amd.cdna4.buffer_store(stored_value=cb, ptr=tile_c, offsets=row * _C_N + col)
            elif VARIANT == 1:
                co = gl.convert_layout(cb, _L3_OLD)
                vo = gl.arange(0, NB, gl.SliceLayout(1, gl.SliceLayout(2, _L3_OLD)))[:, None, None]
                ro = gl.arange(0, 32, gl.SliceLayout(0, gl.SliceLayout(2, _L3_OLD)))[None, :, None]
                cc = gl.arange(0, 32, gl.SliceLayout(0, gl.SliceLayout(1, _L3_OLD)))[None, None, :]
                q = (b0 + vo) // 16
                vv = (b0 + vo) % 16
                row = (q % 2) * 128 + (vv // 4 % 2) * 32 + (vv // 8) * 64 + ro
                col = (q // 2) * 128 + (vv % 2) * 32 + (vv // 2 % 2) * 64 + cc
                gl.amd.cdna4.buffer_store(stored_value=co, ptr=tile_c, offsets=row * _C_N + col)
            else:
                cs = gl.convert_layout(_to_band(cb, NB), _STORE2D)
                _, _, _, trow, tcol = _band(_STORE2D, b0, ROWS)
                gl.amd.cdna4.buffer_store(stored_value=cs, ptr=tile_c, offsets=trow * _C_N + tcol)
        elif VARIANT == 3:
            qq, qrow, c, _, _ = _band(_MFMA, b0, ROWS)
            off = qq * 16384 + _lane_off(qrow, c)
            ssum = gl.zeros([ROWS, 128], gl.float32, _MFMA)
            for j in gl.static_range(N):
                ssum += gl.amd.cdna4.buffer_load(ptr=p_ptr + (g * N + j).to(gl.int64) * _SLOT, offsets=off,
                                                 cache=".cv")
            t3 = _stamp()
            cs = gl.convert_layout(ssum.to(gl.bfloat16), _STORE2D)
            _, _, _, trow, tcol = _band(_STORE2D, b0, ROWS)
            gl.amd.cdna4.buffer_store(stored_value=cs, ptr=tile_c, offsets=trow * _C_N + tcol)
        else:
            qq, qrow, c, trow, tcol = _band(_STORE2D, b0, ROWS)
            off = qq * 16384 + qrow * 128 + c
            ssum = gl.zeros([ROWS, 128], gl.float32, _STORE2D)
            for j in gl.static_range(N):
                ssum += gl.amd.cdna4.buffer_load(ptr=p_ptr + (g * N + j).to(gl.int64) * _SLOT, offsets=off,
                                                 cache=".cv")
            t3 = _stamp()
            gl.amd.cdna4.buffer_store(stored_value=ssum.to(gl.bfloat16), ptr=tile_c, offsets=trow * _C_N + tcol)
    t4 = _stamp()
    z = _one().to(gl.int64)
    rec = rec_ptr + pid * _REC + z
    gl.store(rec, t0 + z)
    gl.store(rec + 1, t1 + z)
    gl.store(rec + 2, t2 + z)
    gl.store(rec + 3, t3 + z)
    gl.store(rec + 4, t4 + z)
    gl.store(rec + 5, is_w.to(gl.int64) + z)
    gl.store(rec + 6, g.to(gl.int64) + z)


def expected_tile(n):
    r = torch.arange(256)[:, None]
    c = torch.arange(256)[None, :]
    return (n * ((r * 256 + c) % 7) + n * (n + 1) // 2).to(torch.bfloat16)


def groups(W, n):
    gpx = (W + 8 * n - 1) // (8 * n)
    return (W // (n * gpx)) * gpx
