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
"""Row padding for benchmark operands.

A row stride that is a multiple of 512 bytes makes the rows that many workgroups touch at once
camp on a few memory channels, and how badly depends on where the buffer lands in physical
memory (experiments/placement_bimodality/PARTITION_CAMPING.md). Following AMD's MI300 / MI350
tuning guide and vLLM's ROCm weight padding, such rows get 256 extra bytes: the tensor is
allocated (rows, cols + pad) and used through its (rows, cols) view.
"""

import torch

PAD_BYTES = 256
STRIDE_MULTIPLE = 512


def padded_width(cols, dtype):
    """Allocated row length in elements: cols, plus PAD_BYTES when the row is a multiple of
    STRIDE_MULTIPLE bytes."""
    esize = torch.empty((), dtype=dtype).element_size()
    return cols + PAD_BYTES // esize if (cols * esize) % STRIDE_MULTIPLE == 0 else cols


def empty_rows(rows, cols, dtype, device, pad=True):
    """An uninitialised (rows, cols) tensor whose row stride is padded_width(cols) if pad."""
    width = padded_width(cols, dtype) if pad else cols
    return torch.empty((rows, width), dtype=dtype, device=device)[:, :cols]
