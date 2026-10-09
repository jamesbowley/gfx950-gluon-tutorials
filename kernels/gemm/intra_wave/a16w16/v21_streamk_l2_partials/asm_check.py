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
"""Cache bits and cache maintenance in v21's compiled kernel, per STREAMK_PARTIALS mode.

    cd kernels/gemm/intra_wave/a16w16
    HIP_VISIBLE_DEVICES=2 python v21_streamk_l2_partials/asm_check.py [MxNxK]

Builds the one-tile reduce-scatter kernel for the shape (default 4352x4096x8192) in each mode
and counts, over the whole kernel, the vector memory instructions by opcode and cache bits
(sc0, sc1, nt), buffer_wbl2 / buffer_inv, and the atomics. Then checks what each mode must
emit: mem has v20's buffer_wbl2 sc1 and buffer_inv sc1 and write-through (sc0 sc1) partial
stores; wt has the same partial stores and loads but no cache maintenance; l2 has no
write-through store, no system-scope load, no buffer_wbl2 and no buffer_inv at all, and its
slice loads are sc0 nt.
"""

import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import bench  # noqa: E402  (plugin hooks; imports triton before torch)
import torch  # noqa: E402
from v21_streamk_l2_partials import matmul_kernel as mk  # noqa: E402

OPS = ("buffer_store", "buffer_load", "global_store", "global_load", "buffer_wbl2", "buffer_inv", "global_atomic",
       "buffer_atomic", "flat_")


def compiled_kernels():
    out = []
    for cache in mk.v21_streamk_l2_partials.device_caches.values():
        out.extend(cache[0].values())
    return out


def census(asm):
    c = Counter()
    for ln in asm.splitlines():
        ln = ln.split(";")[0].strip()
        if not ln.startswith(OPS):
            continue
        words = re.split(r"[\s,]+", ln)
        bits = [b for b in ("sc0", "sc1", "nt") if b in words]
        c[" ".join([words[0]] + bits)] += 1
    return c


def main():
    shape = sys.argv[1] if len(sys.argv) > 1 else "4352x4096x8192"
    M, N, K = (int(x) for x in shape.split("x"))
    a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16).T
    ok = True
    print(f"{shape}, one-tile reduce-scatter (capped rule). Whole-kernel counts of vector memory "
          "instructions by cache bits.\n")
    for mode in ("mem", "wt", "l2"):
        os.environ["STREAMK_PARTIALS"] = mode
        before = {id(k) for k in compiled_kernels()}
        mk.matmul(a, b)
        torch.cuda.synchronize()
        new = [k for k in compiled_kernels() if id(k) not in before]
        k = new[-1] if new else compiled_kernels()[-1]
        c = census(k.asm["amdgcn"])
        print(f"## {mode}: {k.n_regs} VGPRs, {k.n_spills} spills\n")
        for key in sorted(c):
            print(f"- `{key}`: {c[key]}")
        wt_stores = sum(v for x, v in c.items() if x.startswith("buffer_store") and "sc0 sc1" in x)
        sys_loads = sum(v for x, v in c.items() if x.startswith("buffer_load") and "sc0 sc1" in x)
        nt_loads = sum(v for x, v in c.items() if x.startswith("buffer_load") and x.endswith("sc0 nt"))
        wbl2 = sum(v for x, v in c.items() if x.startswith("buffer_wbl2"))
        inv_sc1 = sum(v for x, v in c.items() if x.startswith("buffer_inv") and "sc1" in x)
        inv_sc0 = c.get("buffer_inv sc0", 0)
        if mode == "mem":
            checks = {"write-through partial stores": wt_stores > 0, "buffer_wbl2 present": wbl2 > 0,
                      "buffer_inv sc1 present": inv_sc1 > 0}
        elif mode == "wt":
            checks = {"write-through partial stores": wt_stores > 0, "no buffer_wbl2": wbl2 == 0,
                      "no buffer_inv sc1": inv_sc1 == 0}
        else:
            checks = {"no write-through store": wt_stores == 0, "no sc0 sc1 load": sys_loads == 0,
                      "no buffer_wbl2": wbl2 == 0, "no buffer_inv sc1": inv_sc1 == 0,
                      "no buffer_inv": inv_sc0 == 0, "sc0 nt slice loads": nt_loads > 0}
        print()
        for name, good in checks.items():
            ok &= good
            print(f"- check {name}: {'ok' if good else 'FAIL'}")
        print()
    if not ok:
        raise SystemExit("asm check failed")


if __name__ == "__main__":
    main()
