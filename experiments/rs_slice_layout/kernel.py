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
"""Layouts of v20's reduce-scatter slice and what converting between them costs.

A 128x128 quadrant of the accumulator, in v20's MFMA layout (AMDMFMALayout(4, [16, 16, 32],
transposed, warps [2, 2])), is the linear layout
    registers col 1, 2, 32, 64, row 32, 64;  lanes row 1, 2, 4, 8, col 4, 8;  warps col 16, row 16.
v20's slice holds 16 of its 32x32 blocks as a (16, 32, 32) tensor in
    L3 = BlockedLayout([1, 1, 4], [1, 16, 4], [1, 2, 2], [1, 2, 0]),
whose bases, after reshaping the block index into (row 64, row 32, col 64, col 32), are the same
except the warps: row 16, col 16. A BlockedLayout has one `order` for lanes and warps, so it
cannot put lanes rows first and warps columns first; L3M, a DistributedLinearLayout, can.

probe_kernel: one quadrant's worth of bf16 values (exact small integers) is made in a source
layout, converted REPS times (each iteration adds the loop counter first, so no conversion can be
hoisted), and the last result is stored row-major (C's row stride) so the host can check that the
conversion kept every value at its (row, col).
"""

import triton  # before torch: torch's HIP pulls in ROCm's shared libLLVM, which breaks libtriton's
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

C_N = 4096  # C's row stride in elements (8192 bytes of bf16)
_C_N = gl.constexpr(C_N)

MFMA = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 2])
STORE2D = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])  # v20's gStoreLayoutC: 8 bf16 per lane
L3 = gl.BlockedLayout([1, 1, 4], [1, 16, 4], [1, 2, 2], [1, 2, 0])  # v20's slice layout
L3_OLD = gl.BlockedLayout([1, 1, 4], [1, 16, 4], [1, 2, 2], [2, 1, 0])  # before v20 entry 8: columns first
S3 = gl.BlockedLayout([1, 1, 8], [1, 16, 4], [2, 2, 1], [2, 1, 0])  # 3-D store layout: 8 bf16 per lane along a row
# gStoreLayoutC written in 3-D: lanes along columns, then across blocks 0-3 (block bits 0, 1 are col 32, 64)
S3_ROW = gl.BlockedLayout([1, 1, 8], [4, 4, 4], [1, 4, 1], [2, 0, 1])
L3M = gl.DistributedLinearLayout(  # L3 with MFMA's warp order
    reg_bases=[[0, 0, 1], [0, 0, 2], [1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0]],
    lane_bases=[[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 0, 4], [0, 0, 8]],
    warp_bases=[[0, 0, 16], [0, 16, 0]], block_bases=[], shape=[16, 32, 32])

_MFMA = gl.constexpr(MFMA)
_STORE2D = gl.constexpr(STORE2D)
_L3 = gl.constexpr(L3)
_L3_OLD = gl.constexpr(L3_OLD)
_S3 = gl.constexpr(S3)
_L3M = gl.constexpr(L3M)
_S3_ROW = gl.constexpr(S3_ROW)

CASES = {
    # name: (CASE, what it is)
    "mfma_add": (0, "MFMA, no conversion (loop baseline)"),
    "mfma_to_store": (1, "MFMA -> gStoreLayoutC (v20's owner / data-parallel C store)"),
    "l3_add": (2, "L3, no conversion (loop baseline)"),
    "l3_to_s3": (3, "L3 -> 3-D store layout, 8 bf16 per lane (a coalesced slice store)"),
    "l3_to_l3old": (4, "L3 -> columns-first L3 (v21's dropped attempt)"),
    "l3_to_mfma": (5, "L3 reshaped to 128x128 -> MFMA (warp order differs)"),
    "l3_to_mfma_to_store": (6, "L3 reshaped -> MFMA -> gStoreLayoutC"),
    "l3m_to_mfma": (7, "L3M reshaped -> MFMA, assert_trivial=True"),
    "l3m_to_mfma_to_store": (8, "L3M reshaped -> MFMA (trivial) -> gStoreLayoutC"),
    "l3_to_store2d": (9, "L3 reshaped -> gStoreLayoutC directly"),
    "l3_to_s3row": (10, "L3 -> gStoreLayoutC's mapping written as a 3-D layout (no reshape)"),
}


@gluon.jit
def _now():
    one = gl.zeros([1], gl.int32, gl.BlockedLayout([1], [64], [gl.num_warps()], [0]))
    return gl.max(gl.inline_asm_elementwise("s_waitcnt vmcnt(0) lgkmcnt(0)\ns_barrier\ns_memrealtime $0\n"
                                            "s_waitcnt lgkmcnt(0)", "=s,v,~{memory}", [one], dtype=gl.int64,
                                            is_pure=False, pack=1), axis=0)


@gluon.jit
def _val(row, col):
    """The value at (row, col) of a quadrant: an exact bf16 integer."""
    return ((row * 128 + col) % 251).to(gl.float32)


@gluon.jit
def _quad2d(layout: gl.constexpr):
    r = gl.arange(0, 128, gl.SliceLayout(1, layout))[:, None]
    c = gl.arange(0, 128, gl.SliceLayout(0, layout))[None, :]
    return r, c


@gluon.jit
def _blocks3d(layout: gl.constexpr):
    """(row, col) in the quadrant of every element of a (16, 32, 32) block tensor, v20's block map:
    block v adds col 32 (bit 0), col 64 (bit 1), row 32 (bit 2), row 64 (bit 3)."""
    v = gl.arange(0, 16, gl.SliceLayout(1, gl.SliceLayout(2, layout)))[:, None, None]
    r = gl.arange(0, 32, gl.SliceLayout(0, gl.SliceLayout(2, layout)))[None, :, None]
    c = gl.arange(0, 32, gl.SliceLayout(0, gl.SliceLayout(1, layout)))[None, None, :]
    row = (v // 4 % 2) * 32 + (v // 8) * 64 + r
    col = (v % 2) * 32 + (v // 2 % 2) * 64 + c
    return row, col


@gluon.jit
def _to2d(x):
    """(16, 32, 32) blocks -> the (128, 128) quadrant: block index bits (row 64, row 32, col 64, col 32)."""
    y = gl.reshape(x, [2, 2, 2, 2, 32, 32])
    y = gl.permute(y, [0, 1, 4, 2, 3, 5])
    return gl.reshape(y, [128, 128])


@gluon.jit
def _store2d(out_ptr, x, layout: gl.constexpr):
    r, c = _quad2d(layout)
    gl.amd.cdna4.buffer_store(stored_value=x, ptr=out_ptr, offsets=r * _C_N + c)


@gluon.jit
def _store3d(out_ptr, x, layout: gl.constexpr):
    row, col = _blocks3d(layout)
    gl.amd.cdna4.buffer_store(stored_value=x, ptr=out_ptr, offsets=row * _C_N + col)


@gluon.jit
def probe_kernel(out_ptr, rec_ptr, REPS, CASE: gl.constexpr):
    if CASE <= 1:
        r, c = _quad2d(_MFMA)
        x = _val(r, c)
        acc = gl.zeros([128, 128], gl.float32, _STORE2D if CASE == 1 else _MFMA).to(gl.bfloat16)
    elif CASE <= 6:
        row, col = _blocks3d(_L3)
        x = _val(row, col)
        if CASE == 2:
            acc = gl.zeros([16, 32, 32], gl.float32, _L3).to(gl.bfloat16)
        elif CASE == 3:
            acc = gl.zeros([16, 32, 32], gl.float32, _S3).to(gl.bfloat16)
        elif CASE == 4:
            acc = gl.zeros([16, 32, 32], gl.float32, _L3_OLD).to(gl.bfloat16)
        elif CASE == 5:
            acc = gl.zeros([128, 128], gl.float32, _MFMA).to(gl.bfloat16)
        else:
            acc = gl.zeros([128, 128], gl.float32, _STORE2D).to(gl.bfloat16)
    elif CASE <= 8:
        row, col = _blocks3d(_L3M)
        x = _val(row, col)
        if CASE == 7:
            acc = gl.zeros([128, 128], gl.float32, _MFMA).to(gl.bfloat16)
        else:
            acc = gl.zeros([128, 128], gl.float32, _STORE2D).to(gl.bfloat16)
    elif CASE == 9:
        row, col = _blocks3d(_L3)
        x = _val(row, col)
        acc = gl.zeros([128, 128], gl.float32, _STORE2D).to(gl.bfloat16)
    else:
        row, col = _blocks3d(_L3)
        x = _val(row, col)
        acc = gl.zeros([16, 32, 32], gl.float32, _S3_ROW).to(gl.bfloat16)
    t0 = _now()
    for i in range(REPS):
        y = (x + (i - REPS + 1) * 1.0).to(gl.bfloat16)  # never hoisted; the values only matter at i = REPS - 1
        if CASE == 0:
            z = y
        elif CASE == 1:
            z = gl.convert_layout(y, _STORE2D)
        elif CASE == 2:
            z = y
        elif CASE == 3:
            z = gl.convert_layout(y, _S3)
        elif CASE == 4:
            z = gl.convert_layout(y, _L3_OLD)
        elif CASE == 5:
            z = gl.convert_layout(_to2d(y), _MFMA)
        elif CASE == 6:
            z = gl.convert_layout(gl.convert_layout(_to2d(y), _MFMA), _STORE2D)
        elif CASE == 7:
            z = gl.convert_layout(_to2d(y), _MFMA, assert_trivial=True)
        elif CASE == 8:
            z = gl.convert_layout(gl.convert_layout(_to2d(y), _MFMA, assert_trivial=True), _STORE2D)
        elif CASE == 9:
            z = gl.convert_layout(_to2d(y), _STORE2D)
        else:
            z = gl.convert_layout(y, _S3_ROW)
        acc = gl.where(i == REPS - 1, z, acc + z * 0.0)  # keeps every iteration live
    t1 = _now()
    if CASE == 0:
        _store2d(out_ptr, acc, _MFMA)
    elif CASE == 1:
        _store2d(out_ptr, acc, _STORE2D)
    elif CASE == 2:
        _store3d(out_ptr, acc, _L3)
    elif CASE == 3:
        _store3d(out_ptr, acc, _S3)
    elif CASE == 4:
        _store3d(out_ptr, acc, _L3_OLD)
    elif CASE == 5:
        _store2d(out_ptr, acc, _MFMA)
    elif CASE == 7:
        _store2d(out_ptr, acc, _MFMA)
    elif CASE == 10:
        _store3d(out_ptr, acc, _S3_ROW)
    else:
        _store2d(out_ptr, acc, _STORE2D)
    one = gl.zeros([1], gl.int64, gl.BlockedLayout([1], [64], [gl.num_warps()], [0]))
    gl.store(rec_ptr + one, t0 + one)
    gl.store(rec_ptr + 1 + one, t1 + one)


def expected_quadrant():
    r = torch.arange(128)[:, None]
    c = torch.arange(128)[None, :]
    return ((r * 128 + c) % 251).to(torch.bfloat16)
