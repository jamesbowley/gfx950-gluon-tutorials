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
"""v26's compiled code: with EDGE_K off it must be v25's; with it on, its registers.

    cd kernels/gemm/intra_wave/a16w16
    HIP_VISIBLE_DEVICES=0 python v26_streamk_ragged_k/asm_check.py > v26_streamk_ragged_k/results/asm_check.md

Every configuration of v24's asm check (v23's and v24's own options), on 4096x8192xK with
GEMM_TILE_LIMIT = 256 + S:
1. Built by v25 and by v26 on K (aligned, EDGE_K off): the instruction streams (kernel name replaced,
   comments, directives and argument metadata dropped) and the register and spill counts must match,
   except for the end-to-end builds (two_tile, spread), which entry 3 changes.
2. Built by v26 on K - 16 and K - 112 (EDGE_K on: K-step 1 of the last pair masked, and both), and on
   K - 112 with STREAMK_FORCE_EDGE=1 (all three edges): VGPRs and spills.
"""

import difflib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import bench_ext as bench  # noqa: E402,F401  (plugin hooks; imports triton before torch)
import torch  # noqa: E402
from v24_streamk_combined import asm_check as a24  # noqa: E402
from v25_streamk_ragged_mn import matmul_kernel as v25  # noqa: E402
from v26_streamk_ragged_k import matmul_kernel as v26  # noqa: E402

M, N = a24.M, a24.N
CASES = a24.SAME + a24.OWN


def compile_case(mod, K, S, env, force_edge=False):
    for k in a24.ENV_KEYS + ("STREAMK_FORCE_EDGE",):
        os.environ.pop(k, None)
    os.environ.update(env)
    os.environ["GEMM_TILE_LIMIT"] = str(256 + S)
    if force_edge:
        os.environ["STREAMK_FORCE_EDGE"] = "1"
    fn = mod.v25_streamk_ragged_mn if mod is v25 else mod.v26_streamk_ragged_k
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
    print("# v26 asm check\n")
    print(f"{M}x{N}xK, GEMM_TILE_LIMIT = 256 + S. Aligned K: v26 against v25 (kernel name replaced, comments and "
          "directives dropped). K - 16, K - 112: v26 with EDGE_K; forced: also STREAMK_FORCE_EDGE=1.\n")
    print("| K | S | configuration | v25 VGPRs / spills | v26 VGPRs / spills | instruction lines changed | "
          "K - 16 | K - 112 | K - 112 forced | result |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for K, S, env in CASES:
        k25 = compile_case(v25, K, S, env)
        k26 = compile_case(v26, K, S, env)
        r16 = compile_case(v26, K - 16, S, env)
        r112 = compile_case(v26, K - 112, S, env)
        rf = compile_case(v26, K - 112, S, env, force_edge=True)
        diff = changed_lines(a24.instructions(k25.asm["amdgcn"], "v25_streamk_ragged_mn"),
                             a24.instructions(k26.asm["amdgcn"], "v26_streamk_ragged_k"))
        same = diff == 0 and (k25.n_regs, k25.n_spills) == (k26.n_regs, k26.n_spills)
        # Entry 3 changes the end-to-end segment loop (two_tile, spread) only.
        end_to_end = env.get("STREAMK_POLICY") in ("two_tile", "spread")
        ok &= same or end_to_end
        result = "identical" if same else ("changed (end-to-end, entry 3)" if end_to_end else "FAIL")
        print(f"| {K} | {S} | {a24.label(env)} | {k25.n_regs} / {k25.n_spills} | {k26.n_regs} / {k26.n_spills} | "
              f"{diff} | {r16.n_regs} / {r16.n_spills} | {r112.n_regs} / {r112.n_spills} | "
              f"{rf.n_regs} / {rf.n_spills} | {result} |", flush=True)
    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
