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
"""Cache maintenance in every stream-K version's compiled kernel, per ending.

    cd kernels/gemm/intra_wave/a16w16
    HIP_VISIBLE_DEVICES=0 python v21_streamk_l2_partials/sync_asm_check.py
    HIP_VISIBLE_DEVICES=0 python v21_streamk_l2_partials/sync_asm_check.py --expect none   # must pass
    HIP_VISIBLE_DEVICES=0 python v21_streamk_l2_partials/sync_asm_check.py --expect some   # release/acquire builds

Each spec (version plus STREAMK_* variables, as in version_spec.py) is compiled in its own
process with a fresh TRITON_CACHE_DIR on 4352x4096x8192 (16 leftover tiles), and the version's
own .amdgcn is scanned for buffer_wbl2, buffer_inv, write-through (sc0 sc1) buffer stores and
the atomics. With write-through partials and relaxed flags and counters, no buffer_wbl2 or
buffer_inv may appear anywhere in the kernel.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
A16 = os.path.dirname(HERE)
SPECS = [
    "15", "16", "17", "18", "19",
    "20:STREAMK_POLICY=one_tile+STREAMK_FIXUP=owner", "20:STREAMK_POLICY=one_tile+STREAMK_FIXUP=rs",
    "21:STREAMK_POLICY=one_tile+STREAMK_FIXUP=owner", "21:STREAMK_POLICY=one_tile+STREAMK_FIXUP=rs",
    "21:STREAMK_POLICY=one_tile+STREAMK_FIXUP=rs+STREAMK_PARTIALS=mem",
    "22:STREAMK_POLICY=one_tile+STREAMK_FIXUP=owner", "22:STREAMK_POLICY=one_tile+STREAMK_FIXUP=rs",
    "22:STREAMK_POLICY=one_tile+STREAMK_FIXUP=atomic_iz",
]
TAG = "SYNC_ASM "


def census(path):
    c = Counter()
    for ln in open(path):
        ln = ln.split(";")[0].strip()
        if ln.startswith(("buffer_wbl2", "buffer_inv")):
            c[ln] += 1
        elif ln.startswith(("global_atomic", "buffer_atomic")):
            c["atomic"] += 1
        elif ln.startswith("buffer_store") and re.search(r"\bsc0 sc1\b", ln):
            c["wt store"] += 1
    return c


def worker(spec):
    sys.path.insert(0, A16)
    import bench_ext as bench  # noqa: F401  (plugin hooks; imports triton before torch)
    import importlib

    import torch
    from v21_streamk_l2_partials import version_spec

    v, env = version_spec.parse(spec)
    os.environ.update(env)
    mk = importlib.import_module(f"{bench.VERSION_MAP[v]}.matmul_kernel")
    M, N, K = 4352, 4096, 8192
    a = torch.rand((M, K), device="cuda", dtype=torch.bfloat16) - 0.5
    b = torch.rand((N, K), device="cuda", dtype=torch.bfloat16).T - 0.5
    mk.matmul(a, b)
    torch.cuda.synchronize()
    files = glob.glob(os.path.join(os.environ["TRITON_CACHE_DIR"], "**", f"{bench.VERSION_MAP[v]}.amdgcn"),
                      recursive=True)
    total = Counter()
    for f in files:
        total += census(f)
    print(TAG + json.dumps({"files": len(files), **total}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--specs", nargs="+", default=SPECS)
    parser.add_argument("--expect", choices=["none", "some"],
                        help="none: no buffer_wbl2 / buffer_inv anywhere (except STREAMK_PARTIALS=mem)")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return

    sys.path.insert(0, A16)
    from v21_streamk_l2_partials import version_spec

    print("| spec | buffer_wbl2 | buffer_inv | atomics | write-through stores | check |")
    print("|---|---|---|---|---|---|")
    ok = True
    for spec in args.specs:
        env = {**os.environ, "TRITON_CACHE_DIR": tempfile.mkdtemp(prefix="syncasm_")}
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", spec], cwd=A16, env=env,
                           capture_output=True, text=True)
        line = next((x for x in p.stdout.splitlines() if x.startswith(TAG)), None)
        if line is None or json.loads(line[len(TAG):])["files"] == 0:
            ok = False
            print(f"| {version_spec.label(spec)} | error: {(p.stderr or p.stdout).strip().splitlines()[-1:]} | | | | FAIL |")
            continue
        c = json.loads(line[len(TAG):])
        wbl2 = sum(x for k, x in c.items() if k.startswith("buffer_wbl2"))
        inv = sum(x for k, x in c.items() if k.startswith("buffer_inv"))
        # v21's mem mode keeps the compiler's release/acquire; the atomic ending never had it.
        expect = "some" if "STREAMK_PARTIALS=mem" in spec else "none" if "atomic" in spec else args.expect
        good = expect is None or ((wbl2 + inv) == 0) == (expect == "none")
        ok &= good
        print(f"| {version_spec.label(spec)} | {wbl2} | {inv} | {c.get('atomic', 0)} | {c.get('wt store', 0)} | "
              f"{'ok' if good else 'FAIL'} |", flush=True)
    if not ok:
        raise SystemExit("sync asm check failed")


if __name__ == "__main__":
    main()
