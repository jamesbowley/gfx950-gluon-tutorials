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
"""The compiled reduce-scatter slice and partial stores of v23, against v20's.

    cd kernels/gemm/intra_wave/a16w16
    HIP_VISIBLE_DEVICES=1 python v23_streamk_rs_band/asm_check.py > v23_streamk_rs_band/results/asm_check.md

Builds the one-tile kernel on 4096x8192x16384 with GEMM_TILE_LIMIT = 256 + S for split counts that
give every slice group size (NB = 16, 8, 4, 2, 1) and for each STREAMK_PARTIALS value, and counts the
instructions of the slice: the code from the tile barrier's last fence (`s_nop 0`) to `s_endpgm`.
The owner ending is counted over the whole kernel. Checks:
- no VGPR spills or scratch;
- the slice's C stores are `buffer_store_dwordx4` (`dwordx2` at NB = 1), and v20's are `dwordx2`;
- one layout conversion per group: LDS bytes written = bytes read = 8 NB per lane (the group's bf16
  values), in at most 2 rounds (4 `s_barrier`), 8 `ds_write_b128` + 8 `ds_read_b128` at NB = 16;
- the slice loads: `dwordx4` for fp32, `dwordx2` for bf16, `dwordx4` for bf16 pairs at NB >= 4;
- the partial stores (write-through, `sc0 sc1`): `dwordx4` for fp32 and bf16 pairs, `dwordx2` for bf16.
"""

import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import bench  # noqa: E402  (plugin hooks; imports triton before torch)
import torch  # noqa: E402
from v20_streamk_reduce_scatter import matmul_kernel as v20  # noqa: E402
from v23_streamk_rs_band import matmul_kernel as v23  # noqa: E402

M, N, K = 4096, 8192, 16384
# (S, n): readers R = n here (powers of two), NB = min(16, 64 / R).
SPLITS = [(4, 4), (8, 8), (16, 16), (4, 32), (2, 64), (16, 12)]
LDS_BYTES = {"ds_write_b128": 16, "ds_read_b128": 16, "ds_write_b64": 8, "ds_read_b64": 8, "ds_read2_b64": 16,
             "ds_write2_b64": 16, "ds_write_b32": 4, "ds_read_b32": 4, "ds_write2_b32": 8, "ds_read2_b32": 8,
             "ds_write_b16": 2, "ds_read_u16": 2}


def kernel_fn(mod):
    return mod.v23_streamk_rs_band if mod is v23 else mod.v20_streamk_reduce_scatter


def compile_case(mod, S, n, fixup, partials):
    env = dict(STREAMK_POLICY="one_tile", STREAMK_FIXUP=fixup, STREAMK_NUM_PROGRAMS=str(S * n),
               GEMM_TILE_LIMIT=str(256 + S), STREAMK_PARTIALS=partials, STREAMK_RS_READERS="pow2")
    os.environ.update(env)
    fn = kernel_fn(mod)
    # Empty the in-memory cache, so the launch loads (or compiles) exactly this case's kernel.
    for c in fn.device_caches.values():
        c[0].clear()
    a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16).T
    mod.matmul(a, b)
    torch.cuda.synchronize()
    ks = [k for c in fn.device_caches.values() for k in c[0].values()]
    assert len(ks) == 1, len(ks)
    return ks[0]


def slice_region(asm):
    lines = asm.splitlines()
    end = next(i for i, ln in enumerate(lines) if ln.strip().startswith("s_endpgm"))
    fences = [i for i, ln in enumerate(lines[:end]) if ln.strip() == "s_nop 0" and "ASMSTART" in lines[i - 1]]
    return lines[fences[-1]:end]


def census(lines):
    c = Counter()
    for ln in lines:
        ln = ln.split(";")[0].strip()
        if not ln or ln.startswith("."):
            continue
        words = re.split(r"[\s,]+", ln)
        op = words[0]
        if op.startswith(("buffer_", "global_")):
            bits = [b for b in ("sc0", "sc1", "nt") if b in words]
            c[" ".join([op] + bits)] += 1
        elif op.startswith("ds_") or op == "s_barrier":
            c[op] += 1
    return c


def fmt(c):
    return ", ".join(f"`{k}` {v}" for k, v in sorted(c.items()))


def main():
    ok = True
    print(f"# v23 asm check\n\n{M}x{N}x{K}, GEMM_TILE_LIMIT = 256 + S, STREAMK_NUM_PROGRAMS = S n, "
          "STREAMK_RS_READERS=pow2.\n")
    print("## Reduce-scatter slice (tile barrier's last fence to s_endpgm)\n")
    print("| version | S | n | NB | partials | VGPRs | spills | slice instructions | checks |")
    print("|---|---|---|---|---|---|---|---|---|")
    for S, n in SPLITS:
        nb = v23.rs_split(S, S * n, K // 128, True)["nb"]
        for mod, partials in [(v20, "fp32"), (v23, "fp32"), (v23, "bf16"), (v23, "bf16_pairs")]:
            k = compile_case(mod, S, n, "rs", partials)
            c = census(slice_region(k.asm["amdgcn"]))
            stores = {x: v for x, v in c.items() if x.startswith("buffer_store")}
            loads = {x.split()[0]: v for x, v in c.items() if x.startswith("buffer_load")}
            wr = sum(LDS_BYTES.get(x, 0) * v for x, v in c.items() if x.startswith("ds_write"))
            rd = sum(LDS_BYTES.get(x, 0) * v for x, v in c.items() if x.startswith("ds_read"))
            checks = {"no spills": k.n_spills == 0}
            if mod is v20:
                checks["C store dwordx2"] = set(stores) == {"buffer_store_dwordx2"}
            else:
                want = "buffer_store_dwordx2" if nb == 1 else "buffer_store_dwordx4"
                checks[f"C store {want.split('_')[-1]}"] = set(stores) == {want}
                checks["one conversion"] = wr == rd == 8 * nb and c.get("s_barrier", 0) <= 4
                if nb == 16:
                    checks["8 + 8 b128"] = c.get("ds_write_b128") == 8 and c.get("ds_read_b128") == 8
                want_ld = {"fp32": "buffer_load_dwordx4", "bf16": "buffer_load_dwordx2",
                           "bf16_pairs": "buffer_load_dwordx4" if nb >= 4 else "buffer_load_dwordx2"}[partials]
                checks[f"loads {want_ld.split('_')[-1]}"] = set(loads) == {want_ld}
            ok &= all(checks.values())
            bad = [x for x, g in checks.items() if not g]
            print(f"| v{23 if mod is v23 else 20} | {S} | {n} | {nb} | {partials} | {k.n_regs} | {k.n_spills} | {fmt(c)} | "
                  f"{'ok' if not bad else 'FAIL: ' + ', '.join(bad)} |", flush=True)
    print("\n## Partial stores and owner reads (whole kernel, write-through stores and .cv loads)\n")
    print("| version | ending | S | n | partials | VGPRs | spills | partial stores (sc0 sc1) | partial loads (sc0 sc1) | checks |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for fixup, S, n in (("owner", 32, 4), ("rs", 16, 16)):
        for mod, partials in [(v20, "fp32"), (v23, "fp32"), (v23, "bf16"), (v23, "bf16_pairs")]:
            k = compile_case(mod, S, n, fixup, partials)
            c = census(k.asm["amdgcn"].splitlines())
            st = {x.split()[0]: v for x, v in c.items() if x.startswith("buffer_store") and x.endswith("sc0 sc1")}
            ld = {x.split()[0]: v for x, v in c.items() if x.startswith("buffer_load") and x.endswith("sc0 sc1")}
            want = "buffer_store_dwordx2" if partials == "bf16" else "buffer_store_dwordx4"
            checks = {"no spills": k.n_spills == 0, f"partial stores {want.split('_')[-1]}": set(st) == {want}}
            if fixup == "owner":
                want_ld = "buffer_load_dwordx2" if partials == "bf16" else "buffer_load_dwordx4"
                checks[f"owner loads {want_ld.split('_')[-1]}"] = set(ld) == {want_ld}
            ok &= all(checks.values())
            bad = [x for x, g in checks.items() if not g]
            print(f"| v{23 if mod is v23 else 20} | {fixup} | {S} | {n} | {partials} | {k.n_regs} | {k.n_spills} | "
                  f"{fmt(st)} | {fmt(ld)} | {'ok' if not bad else 'FAIL: ' + ', '.join(bad)} |", flush=True)
    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
