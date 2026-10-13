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
"""Registers and spills of kernel variants over cases, one compile_case.py process (fresh Triton cache)
per (variant, case), in parallel.

    cd kernels/gemm/intra_wave/a16w16
    HIP_VISIBLE_DEVICES=3 python ../../../../experiments/endtoend_spills/survey.py base,crow \\
        4352x4096x8192:STREAMK_POLICY=two_tile --lines --mem

Variants are files in variants/ (without .py); a case is MxNxK[:NAME=VALUE...]. Scratch ops are given as
stores / loads per region (compile_case.parse_regions): out = once per program, L1 = persistent tile loop,
L2 = end-to-end segment loop (pre / k / post: before, in, after its K loop). SGPR lanes: v_writelane
count (each SGPR spill is one; v_readlane also reads them back, and has other uses). --mem adds the buffer
loads / stores by width after the K loops (a variant that scalarises the C or partial accesses shows it there);
--lines the source lines of the scratch ops.
"""

import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = "CASE_RESULT "


def run(variant, case):
    shape, *env = case.split(":")
    M, N, K = shape.split("x")
    e = {**os.environ, "TRITON_CACHE_DIR": tempfile.mkdtemp(prefix="ee_"), **dict(x.split("=", 1) for x in env)}
    p = subprocess.run([sys.executable, os.path.join(HERE, "compile_case.py"), f"variants/{variant}.py", M, N, K],
                       env=e, capture_output=True, text=True)
    for ln in p.stdout.splitlines():
        if ln.startswith(TAG):
            return json.loads(ln[len(TAG):])
    return {"error": (p.stderr or p.stdout).strip().splitlines()[-1:] or [f"exit {p.returncode}"]}


def fmt_ops(ops):
    parts = []
    for r, c in ops.items():
        if c.get("st") or c.get("ld"):
            parts.append(f"{r.replace('outside', 'out')} {c.get('st', 0)}/{c.get('ld', 0)}")
    return ", ".join(parts) or "-"


def main():
    variants = sys.argv[1].split(",")
    cases = [a for a in sys.argv[2:] if not a.startswith("--")]
    lines = "--lines" in sys.argv
    mem = "--mem" in sys.argv
    jobs = [(v, c) for c in cases for v in variants]
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda j: run(*j), jobs))
    print("| case | variant | correct | VGPR | VGPR spill | SGPR spill | scratch ops (stores / loads) | SGPR lanes |"
          + (" ending buffer ops |" if mem else "") + (" spill sites |" if lines else ""))
    print("|---|---|---|---|---|---|---|---|" + ("---|" if mem else "") + ("---|" if lines else ""))
    for (v, c), r in zip(jobs, results):
        if "error" in r:
            print(f"| {c} | {v} | error: {r['error']} | | | | | |" + (" |" if mem else "") + (" |" if lines else ""))
            continue
        ok = {True: "yes", False: "NO", None: "-"}[r["correct"]]
        wl = sum(x.get("wl", 0) for x in r["ops"].values())
        row = (f"| {c} | {v} | {ok} | {r.get('vgpr_count')} | {r.get('vgpr_spill')} | {r.get('sgpr_spill')} | "
               f"{fmt_ops(r['ops'])} | {wl} |")
        if mem:
            row += " " + ", ".join(f"{k} {n}" for k, n in r.get("ending_mem", {}).items()) + " |"
        if lines:
            row += " " + ", ".join(f"{k} x{n}" for k, n in list(r["sites"].items())[:8]) + " |"
        print(row, flush=True)


if __name__ == "__main__":
    main()
