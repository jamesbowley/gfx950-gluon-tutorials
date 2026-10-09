"""Reduce-scatter with any split count, a chosen number of readers, and fp32 or bf16 partials.

As slice_kernel.py (synthetic accumulator, `.wt` partials, drain, relaxed gpu counter, `.cv` loads,
bf16 C), with:
- N programs per tile publish; the first R of them read: reader j sums blocks
  [64 j / R, 64 (j + 1) / R) of all N partials, in groups of G = min(16, next_pow2(ceil(64 / R)))
  blocks, as v20's reduce_scatter_slice does (masked where R does not divide 64). R = N is v20; R =
  the largest power of two <= N leaves N - R programs that only publish.
- N <= 32: every tile's programs on one XCD; N = 64: 64 consecutive programs, spread over the XCDs.

STORE (how the reader stores C):
- 0 plain: from v20's L3 slice layout, 8 bytes per lane, 4 neighbouring lanes on 4 rows (v20).
- 1 block: convert each group's (G, 32, 32) blocks to S3 (8 bf16 per lane along a block row, 64 bytes
  per row per 4-lane group); works for any group, so for any R.
- 2 band: reshape the group to its 2-D band and convert to a row store layout (16 bytes per lane);
  needs aligned groups, so R a power of two.
- 3 mfma: load the band directly in the MFMA layout, convert like v20's owner store (G >= 4).

PART (the partial's type and map; PART 1 and 2 need STORE 3):
- 0 fp32, v20's lane-contiguous map: 16 bytes per lane, 1 KiB per wave instruction.
- 1 bf16, the same map: 4 bf16 = 8 bytes per lane (dwordx2), 512 bytes per wave instruction.
- 2 bf16 pairs: a lane's 8 registers (columns c .. c + 3 of two blocks 32 columns apart) stored as
  16 contiguous bytes, 1 KiB per wave instruction; the reader loads them back in the same layout and
  reshapes to the MFMA layout for free.
"""

import triton  # before torch
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

import kernel as K

NUM_PROGRAMS = 256
SLOT = 4 * 128 * 128  # elements of one partial (fp32 or bf16)
C_N = K.C_N
REC = 8
STORES = {"plain": 0, "block": 1, "band": 2, "mfma": 3}
PARTS = {"fp32": 0, "bf16": 1, "bf16_pairs": 2}

_NUM_PROGRAMS = gl.constexpr(NUM_PROGRAMS)
_REC = gl.constexpr(REC)
_SLOT = gl.constexpr(SLOT)
_C_N = gl.constexpr(C_N)
_MFMA = gl.constexpr(K.MFMA)
_STORE2D = gl.constexpr(K.STORE2D)
_L3 = gl.constexpr(K.L3)
_S3 = gl.constexpr(K.S3)
_ST64 = gl.constexpr(gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0]))  # 32 x 64: 8 lanes x 16 B per row
_ST32 = gl.constexpr(gl.BlockedLayout([1, 4], [8, 8], [4, 1], [1, 0]))  # 32 x 32: 8 lanes x 8 B per row


@gluon.constexpr_function
def pair_layout(rows):
    """The MFMA layout of a (rows, 128) band viewed as (rows, col 64, col 4 .. 16, col 1, 2, 32): the
    last dimension is a lane's first 8 registers."""
    reg = [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 4], [0, 1, 0, 0]]
    if rows >= 64:
        reg.append([32, 0, 0, 0])
    if rows >= 128:
        reg.append([64, 0, 0, 0])
    return gl.DistributedLinearLayout(
        reg_bases=reg, lane_bases=[[1, 0, 0, 0], [2, 0, 0, 0], [4, 0, 0, 0], [8, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0]],
        warp_bases=[[0, 0, 4, 0], [16, 0, 0, 0]], block_bases=[], shape=[rows, 2, 8, 8])


@gluon.jit
def _one():
    return gl.zeros([1], gl.int32, gl.BlockedLayout([1], [64], [gl.num_warps()], [0]))


@gluon.jit
def _stamp():
    return gl.max(gl.inline_asm_elementwise("s_waitcnt vmcnt(0)\ns_barrier\ns_memrealtime $0\ns_waitcnt lgkmcnt(0)",
                                            "=s,v,~{memory}", [_one()], dtype=gl.int64, is_pure=False, pack=1), axis=0)


@gluon.jit
def _fence():
    gl.inline_asm_elementwise("s_nop 0\nv_mov_b32 $0, $1", "=v,v,~{memory}", [_one()], dtype=gl.int32,
                              is_pure=False, pack=1)


@gluon.jit
def _lane_off(qrow, c):
    """v20's lane_contiguous_offsets (elements in a quadrant)."""
    f = (qrow % 16) * 4 + (qrow // 16 % 2) * 512 + (qrow // 32 % 2) * 4096 + (qrow // 64) * 8192
    g = c + (c // 4 % 4) * 60 + (c // 16 % 2) * 240 + (c // 32 % 2) * 992 + (c // 64) * 1984
    return f + g


@gluon.jit
def _pairs(x, ROWS: gl.constexpr):
    """(ROWS, 128) in the MFMA layout -> (ROWS, 2, 8, 8) in pair_layout(ROWS), no data movement."""
    y = gl.permute(gl.reshape(x, [ROWS, 2, 2, 8, 4]), [0, 1, 3, 2, 4])
    return gl.convert_layout(gl.reshape(y, [ROWS, 2, 8, 8]), pair_layout(ROWS), assert_trivial=True)


@gluon.jit
def _unpairs(y, ROWS: gl.constexpr):
    x = gl.permute(gl.reshape(y, [ROWS, 2, 8, 2, 4]), [0, 1, 3, 2, 4])
    return gl.convert_layout(gl.reshape(x, [ROWS, 128]), _MFMA, assert_trivial=True)


@gluon.jit
def _pair_off(qrow0, ROWS: gl.constexpr):
    """Element offsets in a quadrant of the bf16 pair map, for quadrant rows qrow0 .. + ROWS:
    [8 register-vector pairs][4 warps][64 lanes][8 bf16], so lane l of a wave holds bytes 16 l .. + 15."""
    L4: gl.constexpr = pair_layout(ROWS)
    d0 = gl.arange(0, ROWS, gl.SliceLayout(1, gl.SliceLayout(2, gl.SliceLayout(3, L4))))[:, None, None, None]
    d1 = gl.arange(0, 2, gl.SliceLayout(0, gl.SliceLayout(2, gl.SliceLayout(3, L4))))[None, :, None, None]
    d2 = gl.arange(0, 8, gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(3, L4))))[None, None, :, None]
    e = gl.arange(0, 8, gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, L4))))[None, None, None, :]
    row = qrow0 + d0
    vp = d1 + 2 * (row // 32 % 2) + 4 * (row // 64)
    w = d2 // 4 + 2 * (row // 16 % 2)
    lane = row % 16 + 16 * (d2 % 4)
    return ((vp * 4 + w) * 64 + lane) * 8 + e


@gluon.jit
def _acc_quadrant(q: gl.constexpr, m):
    r = gl.arange(0, 128, gl.SliceLayout(1, _MFMA))[:, None]
    c = gl.arange(0, 128, gl.SliceLayout(0, _MFMA))[None, :]
    row = (q % 2) * 128 + r
    col = (q // 2) * 128 + c
    return ((row * 256 + col) % 7 + m + 1).to(gl.float32), r, c


@gluon.jit
def _publish(p_ptr, p16_ptr, widx, m, PART: gl.constexpr):
    for q in gl.static_range(4):
        acc, r, c = _acc_quadrant(q, m)
        base = widx.to(gl.int64) * _SLOT + q * 16384
        if PART == 0:
            gl.amd.cdna4.buffer_store(stored_value=acc, ptr=p_ptr + base, offsets=_lane_off(r, c), cache=".wt")
        elif PART == 1:
            gl.amd.cdna4.buffer_store(stored_value=acc.to(gl.bfloat16), ptr=p16_ptr + base, offsets=_lane_off(r, c),
                                      cache=".wt")
        else:
            gl.amd.cdna4.buffer_store(stored_value=_pairs(acc.to(gl.bfloat16), 128), ptr=p16_ptr + base,
                                      offsets=_pair_off(0, 128), cache=".wt")


@gluon.jit
def _band_pos(g0, layout: gl.constexpr, ROWS: gl.constexpr, COLS: gl.constexpr):
    """Tile rows and columns of the band of the G aligned blocks starting at block g0."""
    r = gl.arange(0, ROWS, gl.SliceLayout(1, layout))[:, None]
    c = gl.arange(0, COLS, gl.SliceLayout(0, layout))[None, :]
    q = g0 // 16
    v0 = g0 % 16
    row0 = (v0 // 4) * 32
    col0 = (v0 % 2) * 32 + (v0 // 2 % 2) * 64
    return (q % 2) * 128 + row0 + r, (q // 2) * 128 + col0 + c


@gluon.jit
def _to_band(x, G: gl.constexpr):
    """(G, 32, 32) aligned blocks -> their 2-D band. Block bits from the lowest: col 32, col 64, row 32,
    row 64."""
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
    return y


@gluon.jit
def _read_l3(p_ptr, g, g0, hi, N: gl.constexpr, G: gl.constexpr):
    v = gl.arange(0, G, gl.SliceLayout(1, gl.SliceLayout(2, _L3)))[:, None, None]
    r = gl.arange(0, 32, gl.SliceLayout(0, gl.SliceLayout(2, _L3)))[None, :, None]
    c = gl.arange(0, 32, gl.SliceLayout(0, gl.SliceLayout(1, _L3)))[None, None, :]
    blk = g0 + v
    valid = (blk < hi) & (r >= 0) & (c >= 0)
    in_blk = (r % 16) * 4 + (r // 16) * 512 + c + (c // 4 % 4) * 60 + (c // 16) * 240
    off = blk * 1024 + in_blk
    s = gl.zeros([G, 32, 32], gl.float32, _L3)
    for j0 in gl.static_range(0, N, 4):  # 4 partials' loads issued before their adds, like v20's U per trip
        x0 = gl.amd.cdna4.buffer_load(ptr=p_ptr + (g * N + j0).to(gl.int64) * _SLOT, offsets=off, mask=valid,
                                      other=0.0, cache=".cv")
        x1 = gl.zeros([G, 32, 32], gl.float32, _L3)
        x2 = gl.zeros([G, 32, 32], gl.float32, _L3)
        x3 = gl.zeros([G, 32, 32], gl.float32, _L3)
        if j0 + 1 < N:
            x1 = gl.amd.cdna4.buffer_load(ptr=p_ptr + (g * N + j0 + 1).to(gl.int64) * _SLOT, offsets=off, mask=valid,
                                          other=0.0, cache=".cv")
        if j0 + 2 < N:
            x2 = gl.amd.cdna4.buffer_load(ptr=p_ptr + (g * N + j0 + 2).to(gl.int64) * _SLOT, offsets=off, mask=valid,
                                          other=0.0, cache=".cv")
        if j0 + 3 < N:
            x3 = gl.amd.cdna4.buffer_load(ptr=p_ptr + (g * N + j0 + 3).to(gl.int64) * _SLOT, offsets=off, mask=valid,
                                          other=0.0, cache=".cv")
        s += (x0 + x1) + (x2 + x3)
    return s


@gluon.jit
def _ld(base, idx, off, BF16: gl.constexpr):
    x = gl.amd.cdna4.buffer_load(ptr=base + idx.to(gl.int64) * _SLOT, offsets=off, cache=".cv")
    if BF16:
        x = x.to(gl.float32)
    return x


@gluon.jit
def _sum_n(base, g, off, zero, N: gl.constexpr, BF16: gl.constexpr):
    """Sum the N partials of group g at off, 4 partials' loads issued before their adds."""
    s = zero
    for j0 in gl.static_range(0, N, 4):
        x0 = _ld(base, g * N + j0, off, BF16)
        x1 = zero
        x2 = zero
        x3 = zero
        if j0 + 1 < N:
            x1 = _ld(base, g * N + j0 + 1, off, BF16)
        if j0 + 2 < N:
            x2 = _ld(base, g * N + j0 + 2, off, BF16)
        if j0 + 3 < N:
            x3 = _ld(base, g * N + j0 + 3, off, BF16)
        s += (x0 + x1) + (x2 + x3)
    return s


@gluon.jit
def _blocks_pos(g0, layout: gl.constexpr, G: gl.constexpr):
    v = gl.arange(0, G, gl.SliceLayout(1, gl.SliceLayout(2, layout)))[:, None, None]
    r = gl.arange(0, 32, gl.SliceLayout(0, gl.SliceLayout(2, layout)))[None, :, None]
    c = gl.arange(0, 32, gl.SliceLayout(0, gl.SliceLayout(1, layout)))[None, None, :]
    blk = g0 + v
    q = blk // 16
    vv = blk % 16
    row = (q % 2) * 128 + (vv // 4 % 2) * 32 + (vv // 8) * 64 + r
    col = (q // 2) * 128 + (vv % 2) * 32 + (vv // 2 % 2) * 64 + c
    return blk, row, col


@gluon.jit
def _slice(p_ptr, p16_ptr, tile_c, g, j, R: gl.constexpr, N: gl.constexpr, G: gl.constexpr,
           STORE: gl.constexpr, PART: gl.constexpr):
    lo = (64 * j) // R
    hi = (64 * (j + 1)) // R
    ROWS: gl.constexpr = 32 if G <= 4 else G * 8
    COLS: gl.constexpr = 32 * G if G <= 4 else 128
    for g0 in range(lo, hi, G):
        if STORE <= 2:
            s = _read_l3(p_ptr, g, g0, hi, N, G)
            cb = s.to(gl.bfloat16)
            if STORE == 0:
                blk, row, col = _blocks_pos(g0, _L3, G)
                gl.amd.cdna4.buffer_store(stored_value=cb, ptr=tile_c, offsets=row * _C_N + col, mask=blk < hi)
            elif STORE == 1:
                cs = gl.convert_layout(cb, _S3)
                blk, row, col = _blocks_pos(g0, _S3, G)
                gl.amd.cdna4.buffer_store(stored_value=cs, ptr=tile_c, offsets=row * _C_N + col, mask=blk < hi)
            else:
                if COLS == 128:
                    SL: gl.constexpr = _STORE2D
                elif COLS == 64:
                    SL: gl.constexpr = _ST64
                else:
                    SL: gl.constexpr = _ST32
                cs = gl.convert_layout(_to_band(cb, G), SL)
                trow, tcol = _band_pos(g0, SL, ROWS, COLS)
                gl.amd.cdna4.buffer_store(stored_value=cs, ptr=tile_c, offsets=trow * _C_N + tcol)
        else:
            q = g0 // 16
            qrow0 = (g0 % 16) * 8
            if PART == 2:
                off = q * 16384 + _pair_off(qrow0, ROWS)
                sp = _sum_n(p16_ptr, g, off, gl.zeros([ROWS, 2, 8, 8], gl.float32, pair_layout(ROWS)), N, True)
                s2 = _unpairs(sp, ROWS)
            else:
                r = gl.arange(0, ROWS, gl.SliceLayout(1, _MFMA))[:, None]
                c = gl.arange(0, 128, gl.SliceLayout(0, _MFMA))[None, :]
                off = q * 16384 + _lane_off(qrow0 + r, c)
                zero = gl.zeros([ROWS, 128], gl.float32, _MFMA)
                if PART == 0:
                    s2 = _sum_n(p_ptr, g, off, zero, N, False)
                else:
                    s2 = _sum_n(p16_ptr, g, off, zero, N, True)
            cs = gl.convert_layout(s2.to(gl.bfloat16), _STORE2D)
            trow, tcol = _band_pos(g0, _STORE2D, ROWS, 128)
            gl.amd.cdna4.buffer_store(stored_value=cs, ptr=tile_c, offsets=trow * _C_N + tcol)


@gluon.jit
def split_kernel(p_ptr, p16_ptr, c_ptr, cnt_ptr, gcnt_ptr, rec_ptr, W, N: gl.constexpr, R: gl.constexpr,
                 G: gl.constexpr, STORE: gl.constexpr, PART: gl.constexpr):
    pid = gl.program_id(0)
    if N <= 32:
        xcd = pid % 8
        s = pid // 8
        gpx = (W + 8 * N - 1) // (8 * N)
        xu = W // (N * gpx)
        is_w = (xcd < xu) & (s < gpx * N)
        g = xcd * gpx + s // N
        m = s % N
    else:
        is_w = pid < W
        g = pid // N
        m = pid % N
    tile_c = c_ptr + ((g // 16) * 256).to(gl.int64) * _C_N + (g % 16) * 256
    gl.atomic_add(gcnt_ptr, 1, sem="relaxed", scope="gpu")
    while gl.atomic_add(gcnt_ptr, 0, sem="relaxed", scope="gpu") < _NUM_PROGRAMS:
        pass
    t0 = _stamp()
    t1 = t0
    t2 = t0
    is_r = is_w & (m < R)
    if is_w:
        _publish(p_ptr, p16_ptr, g * N + m, m, PART)
        t1 = _stamp()
        gl.atomic_add(cnt_ptr + g * 32, 1, sem="relaxed", scope="gpu")
        if m < R:
            while gl.atomic_add(cnt_ptr + g * 32, 0, sem="relaxed", scope="gpu") < N:
                pass
            _fence()
            t2 = _stamp()
            _slice(p_ptr, p16_ptr, tile_c, g, m, R, N, G, STORE, PART)
    t3 = _stamp()
    z = _one().to(gl.int64)
    rec = rec_ptr + pid * _REC + z
    gl.store(rec, t0 + z)
    gl.store(rec + 1, t1 + z)
    gl.store(rec + 2, t2 + z)
    gl.store(rec + 3, t3 + z)
    gl.store(rec + 4, is_w.to(gl.int64) + 2 * is_r.to(gl.int64) + z)
    gl.store(rec + 5, g.to(gl.int64) + z)


def group_size(R):
    nb = -(-64 // R)
    return min(16, 1 << (nb - 1).bit_length())


def setup_W(setup, N):
    if N > 32:
        return NUM_PROGRAMS
    if setup == "tiles 16":
        return 16 * N if 16 * N <= NUM_PROGRAMS else None
    return 8 * (32 // N) * N


def groups(W, N):
    if N > 32:
        return W // N
    gpx = (W + 8 * N - 1) // (8 * N)
    return (W // (N * gpx)) * gpx


def expected_tile(n):
    r = torch.arange(256)[:, None]
    c = torch.arange(256)[None, :]
    return (n * ((r * 256 + c) % 7) + n * (n + 1) // 2).to(torch.bfloat16)
