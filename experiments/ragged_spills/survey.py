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

    HIP_VISIBLE_DEVICES=4 python experiments/ragged_spills/survey.py base,hoist 4300x4000x8080 \\
        4300x4000x8080:STREAMK_POLICY=dp --lines

Variants are files in variants/ (without .py); a case is MxNxK[:NAME=VALUE...].
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
    e = {**os.environ, "TRITON_CACHE_DIR": tempfile.mkdtemp(prefix="rs_"), **dict(x.split("=", 1) for x in env)}
    p = subprocess.run([sys.executable, os.path.join(HERE, "compile_case.py"), f"variants/{variant}.py", M, N, K],
                       env=e, capture_output=True, text=True)
    for ln in p.stdout.splitlines():
        if ln.startswith(TAG):
            return json.loads(ln[len(TAG):])
    return {"error": (p.stderr or p.stdout).strip().splitlines()[-1:] or [f"exit {p.returncode}"]}


def main():
    variants = sys.argv[1].split(",")
    cases = [a for a in sys.argv[2:] if not a.startswith("--")]
    lines = "--lines" in sys.argv
    jobs = [(v, c) for c in cases for v in variants]
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda j: run(*j), jobs))
    print("| case | variant | correct | VGPR | VGPR spill | SGPR spill | scratch ops (kloop/tile/other) |"
          + (" spill sites |" if lines else ""))
    print("|---|---|---|---|---|---|---|" + ("---|" if lines else ""))
    for (v, c), r in zip(jobs, results):
        if "error" in r:
            print(f"| {c} | {v} | error: {r['error']} | | | | |" + (" |" if lines else ""))
            continue
        s = r.get("spill_ops", {})
        row = (f"| {c} | {v} | {'yes' if r['correct'] else 'NO'} | {r.get('vgpr_count')} | {r.get('vgpr_spill')} | "
               f"{r.get('sgpr_spill')} | {s.get('kloop')}/{s.get('tile_loop')}/{s.get('outside')} |")
        if lines:
            row += " " + ", ".join(f"{k} x{n}" for k, n in r.get("spill_lines", {}).items()) + " |"
        print(row, flush=True)


if __name__ == "__main__":
    main()
