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

"""
`init_acc`: the a16w16 accumulators of a 256x256 tile, starting from an optional bias[N].

Used by the 4-wave a16w16 kernels beyond the tutorial (v9_tile, v9_ragged, v10 on). bench_ext.py
puts this directory on `sys.path`, so a kernel can simply `from bias import init_acc`.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def init_acc(
    bias_ptr,
    pid_n,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    MFMA: gl.constexpr,
    ADD_BIAS: gl.constexpr,
):
    """The four 128x128 fp32 accumulators (tl, bl, tr, br) of a 256x256 tile, starting
    from the tile's bias[N] (or zero).

    Adding the bias here rather than before the C stores keeps it out of the epilogue,
    where the stores already hold the register peak; the epilogue add spills. Without
    bias the accumulators are a constant zero, which a peeled first MFMA folds into an
    inline-0 C operand (v13).
    """
    acc_l = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, MFMA)
    acc_r = acc_l
    if ADD_BIAS:
        # Scalar base + 32-bit lane offsets: no per-lane 64-bit addresses.
        offs_bias = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, MFMA))
        bias_base = bias_ptr + pid_n * BLOCK_N
        bias_l = gl.amd.cdna3.buffer_load(ptr=bias_base, offsets=offs_bias)
        bias_r = gl.amd.cdna3.buffer_load(ptr=bias_base + BLOCK_N // 2, offsets=offs_bias)
        acc_l = acc_l + bias_l.to(gl.float32)[None, :]
        acc_r = acc_r + bias_r.to(gl.float32)[None, :]
    return acc_l, acc_l, acc_r, acc_r
