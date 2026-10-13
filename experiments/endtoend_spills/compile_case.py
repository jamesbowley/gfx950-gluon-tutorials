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
"""One build of a kernel variant (a file in variants/, a copy of v26's matmul_kernel.py): run it once on
MxNxK, check C against torch, and print the stream-K kernel's registers, spills and where its scratch ops
and SGPR lane spills are, by loop region (regions.py) and by source line (the .loc before each op).

    python experiments/endtoend_spills/compile_case.py variants/base.py 4352 4096 8192

The STREAMK_* / GEMM_TILE_LIMIT options come from the environment (CASE_BIAS=1 adds a bias); run it with a
fresh TRITON_CACHE_DIR (survey.py does). With CASE_KEEP=path the .amdgcn is copied there.
"""

import glob
import importlib.util
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
A16 = os.path.join(REPO, "kernels", "gemm", "intra_wave", "a16w16")
sys.path.insert(0, A16)
sys.path.insert(0, HERE)

import bench  # noqa: E402,F401  (plugin hooks, path to common.py)
import torch  # noqa: E402
from regions import analyse  # noqa: E402

TAG = "CASE_RESULT "


def main():
    path, M, N, K = sys.argv[1], *map(int, sys.argv[2:5])
    path = path if os.path.isabs(path) else os.path.join(HERE, path)
    spec = importlib.util.spec_from_file_location("variant", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    torch.manual_seed(0)
    a = torch.rand((M, K), device="cuda", dtype=torch.bfloat16) - 0.5
    b = torch.rand((N, K), device="cuda", dtype=torch.bfloat16).T - 0.5
    if os.environ.get("CASE_BIAS") == "1":
        bias = torch.rand((N,), device="cuda", dtype=torch.bfloat16) - 0.5
        c = mod.matmul(a, b, bias=bias)
        ref = (torch.matmul(a.float(), b.float()) + bias.float()).to(torch.bfloat16)
    else:
        c = mod.matmul(a, b)
        ref = torch.matmul(a, b)
    # GEMM_TILE_LIMIT builds write only some tiles of C: registers only, no check.
    correct = None if os.environ.get("GEMM_TILE_LIMIT") else bool(torch.allclose(c, ref, atol=1e-1, rtol=1e-2))
    files = [f for f in glob.glob(os.path.join(os.environ["TRITON_CACHE_DIR"], "**", "*.amdgcn"), recursive=True)
             if "v9_ragged" not in os.path.basename(f)]
    if len(files) != 1:
        print(TAG + json.dumps({"correct": correct, "error": f"{len(files)} stream-K .amdgcn files"}))
        return
    if os.environ.get("CASE_KEEP"):
        shutil.copy(files[0], os.environ["CASE_KEEP"])
    print(TAG + json.dumps({"correct": correct, **analyse(files[0])}), flush=True)


if __name__ == "__main__":
    main()
