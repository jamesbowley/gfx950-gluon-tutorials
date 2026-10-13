"""Print the linear layouts of v20's MFMA accumulator quadrant and of its reduce-scatter slice (L3),
and of L3 after reshaping its (16, 32, 32) blocks into the 128x128 quadrant (no GPU work; compiles one
kernel). Set TRITON_ALWAYS_COMPILE=1 to print again after the first run.

    HIP_VISIBLE_DEVICES=1 TRITON_ALWAYS_COMPILE=1 python show_layouts.py
"""

import kernel as K  # imports triton before torch
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

_MFMA = gl.constexpr(K.MFMA)
_L3 = gl.constexpr(K.L3)
_L3M = gl.constexpr(K.L3M)
_L3_OLD = gl.constexpr(K.L3_OLD)
_S3 = gl.constexpr(K.S3)
_STORE2D = gl.constexpr(K.STORE2D)
_S3_ROW = gl.constexpr(K.S3_ROW)


@gluon.jit
def show(out):
    gl.static_print("MFMA (128, 128):", gl.to_linear_layout(_MFMA, [128, 128]))
    gl.static_print("L3 (16, 32, 32):", gl.to_linear_layout(_L3, [16, 32, 32]))
    gl.static_print("L3 reshaped to (128, 128):", K._to2d(gl.zeros([16, 32, 32], gl.float32, _L3)).type.layout)
    gl.static_print("L3M reshaped to (128, 128):", K._to2d(gl.zeros([16, 32, 32], gl.float32, _L3M)).type.layout)
    gl.static_print("L3_OLD (16, 32, 32):", gl.to_linear_layout(_L3_OLD, [16, 32, 32]))
    gl.static_print("S3 (16, 32, 32):", gl.to_linear_layout(_S3, [16, 32, 32]))
    gl.static_print("gStoreLayoutC (128, 128):", gl.to_linear_layout(_STORE2D, [128, 128]))
    gl.static_print("S3_ROW (16, 32, 32):", gl.to_linear_layout(_S3_ROW, [16, 32, 32]))


if __name__ == "__main__":
    show[(1, )](torch.zeros(1, device="cuda"), num_warps=4)
