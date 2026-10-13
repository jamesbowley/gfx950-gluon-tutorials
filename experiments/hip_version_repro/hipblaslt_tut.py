"""F.linear (hipBLASLt) TFLOPS for one shape, with the tutorial benchmark's inputs.

    run_<env>.sh python hipblaslt_tut.py M N K
"""
import os
import sys

import torch
import torch.nn.functional as F
import triton

from op_tests.triton_tests.gemm.basic.test_gemm_a16w16 import generate_gemm_a16w16_inputs

assert os.environ.get("HIP_VISIBLE_DEVICES") == "0" and torch.cuda.device_count() == 1
M, N, K = (int(v) for v in sys.argv[1:4])
x, w, bias, _, y = generate_gemm_a16w16_inputs(
    M, N, K, torch.bfloat16, layout="TN", output=True, bias=True
)
ms = triton.testing.do_bench_cudagraph(lambda: F.linear(x, w, bias))
print(f"RESULT hipblaslt {M} {N} {K} {2.0 * M * N * K / ms * 1e-9:.1f}")
