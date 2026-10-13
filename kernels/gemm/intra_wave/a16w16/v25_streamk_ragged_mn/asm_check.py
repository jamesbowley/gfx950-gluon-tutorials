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
"""v25's compiled code: with EDGE_M / EDGE_N off it must be v24's; with them on, its registers.

    cd kernels/gemm/intra_wave/a16w16
    HIP_VISIBLE_DEVICES=0 python v25_streamk_ragged_mn/asm_check.py > v25_streamk_ragged_mn/results/asm_check.md

Every configuration of v24's asm check (v23's and v24's own options), on 4096x8192xK with
GEMM_TILE_LIMIT = 256 + S:
1. Built by v24 and by v25 with the edge flags off: the instruction streams (kernel name replaced,
   comments, directives and argument metadata dropped) and the register and spill counts must match.
2. Built by v25 with STREAMK_FORCE_EDGE=1 (both flags on): VGPRs and spills, against (1).
"""

import difflib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import bench_ext as bench  # noqa: E402,F401  (plugin hooks; imports triton before torch)
import torch  # noqa: E402
from v24_streamk_combined import asm_check as a24  # noqa: E402
from v24_streamk_combined import matmul_kernel as v24  # noqa: E402
from v25_streamk_ragged_mn import matmul_kernel as v25  # noqa: E402

M, N = a24.M, a24.N
CASES = a24.SAME + a24.OWN


def compile_case(mod, K, S, env, force_edge=False):
    for k in a24.ENV_KEYS + ("STREAMK_FORCE_EDGE",):
        os.environ.pop(k, None)
    os.environ.update(env)
    os.environ["GEMM_TILE_LIMIT"] = str(256 + S)
    if force_edge:
        os.environ["STREAMK_FORCE_EDGE"] = "1"
    fn = mod.v24_streamk_combined if mod is v24 else mod.v25_streamk_ragged_mn
    for c in fn.device_caches.values():
        c[0].clear()
    a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16).T
    mod.matmul(a, b)
    torch.cuda.synchronize()
    ks = [k for c in fn.device_caches.values() for k in c[0].values()]
    assert len(ks) == 1, len(ks)
    return ks[0]


def changed_lines(x, y):
    return sum(1 for d in difflib.unified_diff(x, y, lineterm="", n=0)
               if d[:1] in "+-" and not d.startswith(("+++", "---")))


def main():
    ok = True
    print("# v25 asm check\n")
    print(f"{M}x{N}xK, GEMM_TILE_LIMIT = 256 + S. Flags off: v25 against v24 (kernel name replaced, comments and "
          "directives dropped). Forced: v25 with STREAMK_FORCE_EDGE=1.\n")
    print("| K | S | configuration | v24 VGPRs / spills | v25 off VGPRs / spills | instruction lines changed | "
          "v25 forced VGPRs / spills | result |")
    print("|---|---|---|---|---|---|---|---|")
    for K, S, env in CASES:
        k24 = compile_case(v24, K, S, env)
        k25 = compile_case(v25, K, S, env)
        kf = compile_case(v25, K, S, env, force_edge=True)
        i24 = a24.instructions(k24.asm["amdgcn"], "v24_streamk_combined")
        i25 = a24.instructions(k25.asm["amdgcn"], "v25_streamk_ragged_mn")
        diff = changed_lines(i24, i25)
        same = diff == 0 and (k24.n_regs, k24.n_spills) == (k25.n_regs, k25.n_spills)
        ok &= same
        print(f"| {K} | {S} | {a24.label(env)} | {k24.n_regs} / {k24.n_spills} | {k25.n_regs} / {k25.n_spills} | "
              f"{diff} | {kf.n_regs} / {kf.n_spills} | {'identical' if same else 'FAIL'} |", flush=True)
    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
