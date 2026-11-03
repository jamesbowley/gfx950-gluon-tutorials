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
"""v9_kreverse's compiled code against v9_beyond_hotloop.

    source v9_kreverse/env.sh
    python v9_kreverse/asm_check.py > v9_kreverse/results/asm_check.md

K_ORDER = 0 must be v9's instruction stream. K_ORDER = 1 and 2 may differ only in the scalar
set-up and in the operand of the per-pair pointer step, so their hot loops must have v9's
instruction count and mix. The hot loop is the body of the backward branch with the MFMAs.
"""

import collections
import difflib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import bench_ext as bench  # noqa: E402,F401  (plugin hooks; imports triton before torch)
import torch  # noqa: E402
from v9_beyond_hotloop import matmul_kernel as v9  # noqa: E402
from v9_kreverse import matmul_kernel as v9k  # noqa: E402

SHAPE = (8192, 8192, 8192)


def build(mod, fn, **kwargs):
    for c in fn.device_caches.values():
        c[0].clear()
    M, N, K = SHAPE
    a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16).T
    mod.matmul(a, b, **kwargs)
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
        if ln and (not ln.startswith(".") or ln.endswith(":")):
            out.append(ln)
    return out


def hot_loop(ins):
    """The body of the backward branch whose body holds the most MFMAs."""
    labels = {ln[:-1]: i for i, ln in enumerate(ins) if ln.endswith(":")}
    best = []
    for i, ln in enumerate(ins):
        m = re.match(r"s_cbranch_\w+\s+(\S+)", ln) or re.match(r"s_branch\s+(\S+)", ln)
        if m and m.group(1) in labels and labels[m.group(1)] < i:
            body = [x for x in ins[labels[m.group(1)] + 1 : i + 1] if not x.endswith(":")]
            if sum(x.startswith("v_mfma") for x in body) > sum(x.startswith("v_mfma") for x in best):
                best = body
    return best


def mix(body):
    classes = collections.Counter()
    for x in body:
        op = x.split()[0]
        for prefix in ("v_mfma", "buffer_load", "ds_read", "ds_write", "s_waitcnt", "s_barrier", "v_", "s_"):
            if op.startswith(prefix):
                classes[prefix] += 1
                break
        else:
            classes["other"] += 1
    return classes


def changed_lines(x, y):
    return [d for d in difflib.unified_diff(x, y, lineterm="", n=0)
            if d[:1] in "+-" and not d.startswith(("+++", "---"))]


def main():
    k9 = build(v9, v9.v9_beyond_hotloop)
    i9 = instructions(k9.asm["amdgcn"], "v9_beyond_hotloop")
    l9 = hot_loop(i9)
    ok = True
    print(f"# v9_kreverse asm check ({'x'.join(map(str, SHAPE))}, bf16)\n")
    print("| build | VGPRs / spills | SGPRs | instruction lines changed vs v9 | hot-loop instructions | "
          "hot-loop mix (v_mfma, buffer_load, ds_read, v_*, s_*) | result |")
    print("|---|---|---|---|---|---|---|")
    m9 = mix(l9)
    row = lambda m: f"{m['v_mfma']}, {m['buffer_load']}, {m['ds_read']}, {m['v_']}, {m['s_']}"  # noqa: E731
    print(f"| v9_beyond_hotloop | {k9.n_regs} / {k9.n_spills} | {k9.metadata.num_sgprs if hasattr(k9.metadata, 'num_sgprs') else '-'} "
          f"| - | {len(l9)} | {row(m9)} | reference |")
    diffs = {}
    for k_order in (0, 1, 2):
        kk = build(v9k, v9k.v9_kreverse, k_order=k_order)
        ik = instructions(kk.asm["amdgcn"], "v9_kreverse")
        lk = hot_loop(ik)
        d = changed_lines(i9, ik)
        diffs[k_order] = changed_lines(l9, lk)
        mk = mix(lk)
        if k_order == 0:
            good = not d and (kk.n_regs, kk.n_spills) == (k9.n_regs, k9.n_spills)
            verdict = "identical" if good else "FAIL"
        else:
            good = len(lk) == len(l9) and mk == m9 and kk.n_spills == k9.n_spills
            verdict = "same loop size and mix" if good else "FAIL"
        ok &= good
        print(f"| K_ORDER={k_order} | {kk.n_regs} / {kk.n_spills} | "
              f"{kk.metadata.num_sgprs if hasattr(kk.metadata, 'num_sgprs') else '-'} | {len(d)} | {len(lk)} | "
              f"{row(mk)} | {verdict} |", flush=True)
    for k_order in (1, 2):
        print(f"\nHot-loop lines that differ from v9, K_ORDER={k_order}:\n\n```")
        print("\n".join(diffs[k_order]) or "(none)")
        print("```")
    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
