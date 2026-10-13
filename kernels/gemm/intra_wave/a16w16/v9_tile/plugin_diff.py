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
"""Compile kernels through the amdgcnas hook, running an old and the current peephole on the same
input assembly, and report the compiles whose output differs.

    cp plugins/amdgcnas/amdgcnas_ext.py /some/old_amdgcnas_ext.py      # before editing the peephole
    cd kernels/gemm/intra_wave/a16w16 && source v9_tile/env.sh
    TRITON_CACHE_DIR=$(mktemp -d) python v9_tile/plugin_diff.py /some/old_amdgcnas_ext.py 7 8 9 9t:256x128 ...

Each argument is a bench_ext.py version (with ":TILE" for 9t), compiled for fp16 and bf16, with and
without bias, through bench.test_correctness. A fresh TRITON_CACHE_DIR makes every kernel compile.
DUMP=dir writes the input and both outputs of every compile that differs.
"""
import importlib
import importlib.util
import os
import sys

sys.path.insert(0, os.getcwd())
import bench_ext as bench  # noqa: E402  (installs the hook, imports amdgcnas_ext)
import amdgcnas_ext  # noqa: E402

spec = importlib.util.spec_from_file_location("amdgcnas_old", sys.argv[1])
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
new_fn = amdgcnas_ext.amdgcn_as
results = []


def both(text, verbose=False):
    a, b = old.amdgcn_as(text, verbose), new_fn(text, verbose)
    name = next((l.split()[1] for l in text.splitlines() if l.strip().startswith(".globl")), "?")
    results.append((name, a == b))
    if a != b and os.environ.get("DUMP"):
        d = os.path.join(os.environ["DUMP"], f"{name}_{len(results)}")
        os.makedirs(d, exist_ok=True)
        for fn, t in (("in.s", text), ("old.s", a), ("new.s", b)):
            open(os.path.join(d, fn), "w").write(t)
    return b


amdgcnas_ext.amdgcn_as = both
for arg in sys.argv[2:]:
    ver, _, tile = arg.partition(":")
    ver = int(ver) if ver.isdigit() else ver
    if tile:
        os.environ["V9_TILE"] = tile
    else:
        os.environ.pop("V9_TILE", None)
    mod = importlib.import_module(f"{bench.VERSION_MAP[ver]}.matmul_kernel")
    importlib.reload(mod)
    n0 = len(results)
    for dt in ("fp16", "bf16"):
        for bias in (False, True):
            try:
                bench.test_correctness(mod.matmul, dt, [(4096, 4096, 8192)], bench.VERSION_MAP[ver], bias=bias)
            except TypeError:
                pass
    new = results[n0:]
    print(f"{arg}: {len(new)} compiles, {sum(not s for _, s in new)} differ "
          f"{sorted({n for n, s in new if not s})}", flush=True)
