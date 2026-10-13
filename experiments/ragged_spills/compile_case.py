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
MxNxK, check C against torch, and print the stream-K kernel's registers, spills and where the scratch ops
are, by region (check_spills.parse_amdgcn) and by source line (the .loc before each scratch op).

    python experiments/ragged_spills/compile_case.py variants/base.py 4300 4000 8080

The STREAMK_* options come from the environment (CASE_BIAS=1 adds a bias); run it with a fresh TRITON_CACHE_DIR (survey.py does).
"""

import collections
import glob
import importlib.util
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
A16 = os.path.join(REPO, "kernels", "gemm", "intra_wave", "a16w16")
sys.path.insert(0, A16)
sys.path.insert(0, os.path.join(REPO, "experiments", "aiter_bias_spills"))

import bench  # noqa: E402,F401  (plugin hooks, path to common.py)
import torch  # noqa: E402
from check_spills import parse_amdgcn  # noqa: E402

TAG = "CASE_RESULT "


def spill_lines(path):
    """Counter of 'file:line' of the .loc in force at each scratch op."""
    files, cur, out = {}, None, collections.Counter()
    for ln in open(path).read().splitlines():
        s = ln.strip()
        m = re.match(r'\.file\s+(\d+)\s+"([^"]*)"(?:\s+"([^"]*)")?', s)
        if m:
            files[m.group(1)] = os.path.basename(m.group(3) or m.group(2))
            continue
        m = re.match(r"\.loc\s+(\d+)\s+(\d+)", s)
        if m:
            cur = f"{files.get(m.group(1), m.group(1))}:{m.group(2)}"
            continue
        if re.match(r"scratch_(store|load)_", s):
            out[cur] += 1
    return out


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
    correct = bool(torch.allclose(c, ref, atol=1e-1, rtol=1e-2))
    files = [f for f in glob.glob(os.path.join(os.environ["TRITON_CACHE_DIR"], "**", "*.amdgcn"), recursive=True)
             if "v9_ragged" not in os.path.basename(f)]
    if len(files) != 1:
        print(TAG + json.dumps({"correct": correct, "error": f"{len(files)} stream-K .amdgcn files"}))
        return
    regs = parse_amdgcn(files[0])
    regs["spill_lines"] = dict(spill_lines(files[0]).most_common(12))
    print(TAG + json.dumps({"correct": correct, **regs}), flush=True)


if __name__ == "__main__":
    main()
