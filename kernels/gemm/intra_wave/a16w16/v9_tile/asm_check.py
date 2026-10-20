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
"""Hot-loop summary of v9_tile per tile, under the environment's plugin stack.

    source v9_tile/env.sh
    python v9_tile/asm_check.py 256x256,256x128,128x256 > v9_tile/results/asm_check.md
    python v9_tile/asm_check.py 256x240,256x224 --grid16    # each tile on its 16x16 grid of tiles

Compiles each tile (one launch, 4096x4096x8192 bf16 or the smallest multiple of the tile) into
its own Triton cache, then reads the K loop (the block that branches back to itself and holds
the most MFMAs) out of the .amdgcn: MFMAs, buffer loads, LDS reads, waits, barriers, AGPR copies
and scratch, plus the spacing of the memory ops between MFMAs. One loop iteration is two
k-steps (v9 unrolls by 2).
"""

import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
A16 = os.path.dirname(HERE)
sys.path[:0] = [A16, os.path.join(A16, "..", "..", "utils")]
from v9_tile import matmul_kernel as mk  # noqa: E402

COMPILE = r"""
import sys; sys.path.insert(0, %r)
import bench_ext as bench, torch
from v9_tile import matmul_kernel as mk
M, N, K = %d, %d, %d
a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
b = bench.make_b(N, K, torch.bfloat16, False, lambda t: t.normal_())
c = bench.make_c(M, N, torch.bfloat16, False)
mk.matmul(a, b, c, tile=%r)
torch.cuda.synchronize()
"""


def compile_tile(tile, K=8192, M=4096, N=4096):
    cache = os.path.join(HERE, "tmp", "asm_cache", tile)
    subprocess.run(["rm", "-rf", cache])
    env = {**os.environ, "TRITON_CACHE_DIR": cache}
    subprocess.run([sys.executable, "-c", COMPILE % (A16, M, N, K, tile)], env=env, check=True,
                   capture_output=True)
    return glob.glob(os.path.join(cache, "*", f"{kernel_name(tile)}.amdgcn"))[0]


def kernel_name(tile):
    bm, bn, _, regions = mk.parse_tile(tile)
    return "v9_tile_2r" if regions == 2 else "v9_tile_small" if mk.is_small(bm, bn, regions) else "v9_tile"


def waves_per_wg(tile):
    bm, bn, bk, regions, w, _ = mk.parse_spec(tile)
    return mk.small_config(bm, bn, bk, w)[3] if mk.is_small(bm, bn, regions) else 4


def loop_body(path):
    lines = [l.split(";")[0].strip() for l in open(path)]
    blocks, cur, name = [], [], None
    for l in lines:
        if re.match(r"^\.?L\w+:$|^\.LBB\w+:$", l):
            if name is not None:
                blocks.append((name, cur))
            name, cur = l[:-1], []
        elif l and not l.startswith("."):
            cur.append(l)
            if l.startswith(("s_cbranch", "s_branch")) and name is not None:
                blocks.append((name, cur))
                name, cur = "<fallthrough>", []
    if name is not None:
        blocks.append((name, cur))
    loops = [(n, b) for n, b in blocks if b and re.search(r"s_cbranch\w*\s+" + re.escape(n) + r"$", b[-1])]
    if loops:
        return max(loops, key=lambda nb: sum(i.startswith("v_mfma") for i in nb[1]))[1]
    return rotated_loop_body(open(path).read().splitlines())


def rotated_loop_body(raw):
    """A loop LLVM rotated into several blocks (an exit test mid-body, the back-edge through a flow
    block): the instructions from the "Loop Header" label to the last branch back to it or to a label
    just above it."""
    head = next(i for i, l in enumerate(raw) if "Loop Header" in l)
    labels = {l.split(":")[0].strip(): i for i, l in enumerate(raw) if re.match(r"^\.LBB\w+:", l.strip())}
    back = max(i for i, l in enumerate(raw)
               if re.match(r"\s*s_(c)?branch\w*\s+(\.LBB\w+)", l)
               and head - 8 <= labels.get(l.split()[-1], -1) <= head and i > head)
    body = [l.split(";")[0].strip() for l in raw[head + 1:back + 1]]
    return [l for l in body if l and not l.startswith(".") and not re.match(r"^\.?L\w+:$", l)]


def meta(path):
    text = open(path).read()
    get = lambda k: int(re.search(rf"\.{k}:\s*(\d+)", text).group(1))  # noqa: E731
    regs = int(re.search(r"\.amdhsa_next_free_vgpr\s+(\d+)", text).group(1))  # arch + acc VGPRs
    lds = json.load(open(path.removesuffix(".amdgcn") + ".json"))["shared"]
    return get("vgpr_count"), get("agpr_count"), get("vgpr_spill_count"), regs, lds


def wg_per_cu(regs, lds, w):
    """Workgroups a CU can hold: 160 KB of LDS, 8 waves per SIMD, 512 registers per lane per SIMD
    (allocated in 8s); the W waves of a workgroup land on W different SIMDs."""
    by_lds = 160 * 1024 // lds if lds else 99
    by_regs = 512 // (-(-regs // 8) * 8) * 4 // w
    return min(by_lds, 8 * 4 // w, by_regs)


def summarize(tile, grid16=False):
    bm, bn = mk.parse_tile(tile)[:2]
    path = compile_tile(tile, M=16 * bm, N=16 * bn) if grid16 else compile_tile(tile)
    body = loop_body(path)
    ops = [i.split()[0] for i in body]
    count = lambda p: sum(o.startswith(p) for o in ops)  # noqa: E731
    mfma_idx = [k for k, o in enumerate(ops) if o.startswith("v_mfma")]
    # longest run of non-MFMA instructions between two MFMAs
    gaps = [b - a - 1 for a, b in zip(mfma_idx, mfma_idx[1:])]
    # MFMAs between consecutive buffer loads
    n_mfma, per_load, last = 0, [], None
    for o in ops:
        if o.startswith("v_mfma"):
            n_mfma += 1
        elif o.startswith("buffer_load"):
            if last is not None:
                per_load.append(n_mfma - last)
            last = n_mfma
    vgpr, agpr, spill, regs, lds = meta(path)
    w = waves_per_wg(tile)
    return {
        "tile": tile, "vgpr": vgpr, "agpr": agpr, "spill": spill, "w": w, "lds": f"{lds / 1024:.1f}",
        "wgs": wg_per_cu(regs, lds, w),
        "mfma": count("v_mfma"), "bload": count("buffer_load"), "dsread": count("ds_read"),
        "wait": count("s_waitcnt"), "barrier": count("s_barrier"), "nop": count("s_nop"),
        "acc": count("v_accvgpr"), "scratch": count("scratch_"), "insts": len(ops),
        "maxgap": max(gaps) if gaps else 0,
        "per_load": f"{min(per_load)}-{max(per_load)}" if per_load else "-",
    }


def main():
    tiles = sys.argv[1].split(",")
    grid16 = "--grid16" in sys.argv  # compile each tile on (16 BM) x (16 BN), so banded tiles run without EDGE
    # matmul() always launches with schedule_hint="mfma-schedule" (the "llir" config since v3.0).
    stack = "llir" + ("+amdgcnas" if os.environ.get("TRITON_AMDGCNAS_PLUGIN") else "")
    shape = "16 BM x 16 BN" if grid16 else "4096x4096"
    print(f"Stack: `{stack}`, {shape} x 8192, one loop iteration = two k-steps.\n")
    print("| tile | waves | VGPRs | AGPRs | spills | LDS KB | WGs per CU | loop insts | MFMAs | buffer loads | "
          "ds_reads | s_waitcnt | s_barrier | s_nop | v_accvgpr | scratch | longest non-MFMA run | "
          "MFMAs between buffer loads |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in tiles:
        r = summarize(t, grid16)
        print("| {tile} | {w} | {vgpr} | {agpr} | {spill} | {lds} | {wgs} | {insts} | {mfma} | {bload} | {dsread} | "
              "{wait} | {barrier} | {nop} | {acc} | {scratch} | {maxgap} | {per_load} |".format(**r), flush=True)


if __name__ == "__main__":
    main()
