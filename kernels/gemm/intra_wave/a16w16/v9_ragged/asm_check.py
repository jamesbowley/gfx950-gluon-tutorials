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
"""v9_ragged's compiled code: with EDGE_M / EDGE_N off it must be v9_beyond_hotloop's.

    cd kernels/gemm/intra_wave/a16w16
    HIP_VISIBLE_DEVICES=0 python v9_ragged/asm_check.py > v9_ragged/results/asm_check.md

Each case is built by both (one launch). The instruction streams (kernel name replaced, comments,
directives and argument metadata dropped) and the register and spill counts must match. The
ragged rows show v9_ragged's own build (both flags on) for reference, and the hook rows its
GEMM_TILE_LIMIT build against the same shape without the limit.
"""

import difflib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import bench  # noqa: E402,F401  (plugin hooks; imports triton before torch)
import torch  # noqa: E402
from v9_beyond_hotloop import matmul_kernel as v9  # noqa: E402
from v9_ragged import matmul_kernel as v9r  # noqa: E402

# (M, N, K, bias)
SAME = [(4096, 4096, 8192, False), (4096, 4096, 8192, True), (2048, 4096, 4096, False), (256, 8192, 8192, True)]
RAGGED = [(4000, 4000, 8192, False), (4000, 4000, 8192, True), (128, 8192, 8192, False),
          (4096, 4096, 8176, False), (4096, 4096, 8080, False), (4000, 4000, 8080, True)]
# (M, N, K, T): the GEMM_TILE_LIMIT test hook (sweep G), against the same shape without it.
LIMITED = [(4096, 8192, 8192, 40), (4096, 8192, 8192, 400)]


def build(mod, fn, M, N, K, bias):
    for c in fn.device_caches.values():
        c[0].clear()
    a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16).T
    bias_n = torch.randn((N,), device="cuda", dtype=torch.bfloat16) if bias else None
    mod.matmul(a, b, bias=bias_n)
    torch.cuda.synchronize()
    ks = [k for c in fn.device_caches.values() for k in c[0].values()]
    assert len(ks) == 1, len(ks)
    return ks[0]


def instructions(asm, name):
    out = []
    for ln in asm.splitlines():
        if ln.strip().startswith(".amdgpu_metadata"):
            break
        ln = ln.split(";")[0].strip().replace(name, "KERNEL")
        if ln and not ln.startswith("."):
            out.append(ln)
    return out


def changed_lines(x, y):
    return sum(1 for d in difflib.unified_diff(x, y, lineterm="", n=0)
               if d[:1] in "+-" and not d.startswith(("+++", "---")))


def main():
    ok = True
    print("# v9_ragged asm check\n")
    print("| shape | bias | v9 VGPRs / spills | v9_ragged VGPRs / spills | instruction lines changed | result |")
    print("|---|---|---|---|---|---|")
    for M, N, K, bias in SAME:
        k9 = build(v9, v9.v9_beyond_hotloop, M, N, K, bias)
        kr = build(v9r, v9r.v9_ragged, M, N, K, bias)
        diff = changed_lines(instructions(k9.asm["amdgcn"], "v9_beyond_hotloop"),
                             instructions(kr.asm["amdgcn"], "v9_ragged"))
        same = diff == 0 and (k9.n_regs, k9.n_spills) == (kr.n_regs, kr.n_spills)
        ok &= same
        print(f"| {M}x{N}x{K} | {bias} | {k9.n_regs} / {k9.n_spills} | {kr.n_regs} / {kr.n_spills} | {diff} | "
              f"{'identical' if same else 'FAIL'} |", flush=True)
    for M, N, K, bias in RAGGED:
        kr = build(v9r, v9r.v9_ragged, M, N, K, bias)
        print(f"| {M}x{N}x{K} (ragged) | {bias} | - | {kr.n_regs} / {kr.n_spills} | - | edge build |", flush=True)
    for M, N, K, T in LIMITED:
        k0 = build(v9r, v9r.v9_ragged, M, N, K, False)
        os.environ["GEMM_TILE_LIMIT"] = str(T)
        try:
            kt = build(v9r, v9r.v9_ragged, M, N, K, False)
        finally:
            os.environ.pop("GEMM_TILE_LIMIT")
        diff = changed_lines(instructions(k0.asm["amdgcn"], "v9_ragged"), instructions(kt.asm["amdgcn"], "v9_ragged"))
        print(f"| {M}x{N}x{K}, GEMM_TILE_LIMIT = {T} | False | v9_ragged unlimited: {k0.n_regs} / {k0.n_spills} | "
              f"{kt.n_regs} / {kt.n_spills} | {diff} | hook build |", flush=True)
    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
