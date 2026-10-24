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
"""Step 1's gate: v9_tile_small with EDGE_N and EDGE_K off compiles to the code it did before.

    source v27_streamk_tile/env.sh
    python v27_streamk_tile/v9_small_gate.py dump before    # before the change
    python v27_streamk_tile/v9_small_gate.py dump after     # after it
    python v27_streamk_tile/v9_small_gate.py diff before after > v27_streamk_tile/results/v9_small_gate.md

`dump` compiles every small tile of v9_tile/results/asm_small.md (4096x4096x8192 bf16, as
v9_tile/asm_check.py), each in its own Triton cache, and keeps the .amdgcn under tmp/<name>/. `diff`
compares the instruction streams (comments, directives and metadata dropped) and the register and
spill counts.
"""

import glob
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
A16 = os.path.dirname(HERE)
TILES = ["16x256x128", "16x256x64w2", "16x128x128", "16x64x128", "16x64x64", "16x64x256", "16x32x128",
         "16x16x128", "16x16x256", "32x32x128", "32x64x128", "32x128x128", "32x256x64", "8x256x128",
         "1x256x128", "1x64x256", "256x16x128", "16x16x128p", "16x64x64p", "16x256x128p"]
COMPILE = r"""
import os, sys; sys.path.insert(0, %r)
if os.environ.get("GATE_SRC"):
    sys.path.insert(0, os.environ["GATE_SRC"])
import bench_ext as bench, torch
from v9_tile import matmul_kernel as mk
M, N, K = 4096, 4096, 8192
a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
b = bench.make_b(N, K, torch.bfloat16, False, lambda t: t.normal_())
c = bench.make_c(M, N, torch.bfloat16, False)
mk.matmul(a, b, c, tile=%r)
torch.cuda.synchronize()
"""


def dump(name):
    out = os.path.join(HERE, "tmp", "v9_small_gate", name)
    os.makedirs(out, exist_ok=True)
    for tile in TILES:
        cache = os.path.join(HERE, "tmp", "v9_small_gate", "cache", tile)
        shutil.rmtree(cache, ignore_errors=True)
        env = {**os.environ, "TRITON_CACHE_DIR": cache}
        subprocess.run([sys.executable, "-c", COMPILE % (A16, tile)], env=env, check=True, capture_output=True)
        for ext in ("amdgcn", "ttgir", "llir"):
            src = glob.glob(os.path.join(cache, "*", f"v9_tile_small.{ext}"))[0]
            shutil.copy(src, os.path.join(out, f"{tile}.{ext}"))
        print(tile, flush=True)


def instructions(path):
    out = []
    for ln in open(path):
        if ln.strip().startswith(".amdgpu_metadata"):
            break
        ln = ln.split(";")[0].strip()
        if ln and not ln.startswith("."):
            out.append(ln)
    return out


def regs(path):
    text = open(path).read()
    return tuple(int(re.search(rf"\.{k}:\s*(\d+)", text).group(1)) for k in ("vgpr_count", "agpr_count",
                                                                             "vgpr_spill_count"))


def diff(a, b):
    import difflib
    da, db = (os.path.join(HERE, "tmp", "v9_small_gate", x) for x in (a, b))
    ok = True
    print(f"v9_tile_small, 4096x4096x8192 bf16, `{a}` against `{b}` (instruction lines, comments and directives "
          "dropped).\n")
    print("| tile | VGPRs / AGPRs / spills, before | after | instructions | lines changed | result |")
    print("|---|---|---|---|---|---|")
    for tile in TILES:
        pa, pb = os.path.join(da, f"{tile}.amdgcn"), os.path.join(db, f"{tile}.amdgcn")
        ia, ib = instructions(pa), instructions(pb)
        n = sum(1 for d in difflib.unified_diff(ia, ib, lineterm="", n=0)
                if d[:1] in "+-" and not d.startswith(("+++", "---")))
        same = n == 0 and regs(pa) == regs(pb)
        ok &= same
        print(f"| {tile} | {' / '.join(map(str, regs(pa)))} | {' / '.join(map(str, regs(pb)))} | {len(ia)} | {n} | "
              f"{'identical' if same else 'FAIL'} |")
    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump(sys.argv[2])
    else:
        diff(sys.argv[2], sys.argv[3])
