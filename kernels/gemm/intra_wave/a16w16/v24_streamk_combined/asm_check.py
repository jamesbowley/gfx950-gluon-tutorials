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
"""v24's compiled code: with its own options off it must be v23's, and each option's build must be
free of spills.

    cd kernels/gemm/intra_wave/a16w16
    HIP_VISIBLE_DEVICES=0 python v24_streamk_combined/asm_check.py > v24_streamk_combined/results/asm_check.md

1. Against v23. Every v23 configuration below is built by both versions, on 4096x8192xK with
   GEMM_TILE_LIMIT = 256 + S. The TTGIR must be identical apart from the signature, which has
   v24's epoch argument (value names and locations normalised). The instruction streams are
   compared too (kernel name replaced, comments, directives and the argument metadata dropped);
   LLVM may order an equivalent integer expression differently because of the extra argument.
2. v24's options. Each combination of ending, partial type, exchange and placement that the host
   accepts, at split counts with and without uneven tiles (EXTRA > 0), is built and checked:
   - no VGPR spills or scratch;
   - atomic_iz: `buffer_atomic_pk_add_bf16` adds, no partial loads, and no write-through stores
     but the zeroing's (`dwordx2`);
   - l2: partial stores with no cache bits and slice loads `sc0 nt`, no write-through;
   - chunk-major: builds whatever the ending.
"""

import difflib
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import bench_ext as bench  # noqa: E402  (plugin hooks; imports triton before torch)
import torch  # noqa: E402
from v23_streamk_rs_band import matmul_kernel as v23  # noqa: E402
from v24_streamk_combined import matmul_kernel as v24  # noqa: E402

M, N = 4096, 8192
ENV_KEYS = ("STREAMK_POLICY", "STREAMK_FIXUP", "STREAMK_NUM_PROGRAMS", "GEMM_TILE_LIMIT", "STREAMK_PARTIALS",
            "STREAMK_RS_READERS", "STREAMK_EXCHANGE", "STREAMK_PLACEMENT", "STREAMK_BALANCE_XCDS")
# (K, S, env): v23 configurations, built by v23 and v24.
SAME = [
    (8192, 8, dict(STREAMK_POLICY="one_tile", STREAMK_FIXUP="rs", STREAMK_NUM_PROGRAMS="128")),
    (8192, 8, dict(STREAMK_POLICY="one_tile", STREAMK_FIXUP="rs", STREAMK_NUM_PROGRAMS="100")),
    (8192, 38, dict(STREAMK_POLICY="one_tile", STREAMK_FIXUP="owner", STREAMK_NUM_PROGRAMS="152")),
    (4096, 16, dict(STREAMK_POLICY="one_tile", STREAMK_FIXUP="rs", STREAMK_NUM_PROGRAMS="128",
                    STREAMK_PARTIALS="bf16_pairs")),
    (4096, 16, dict(STREAMK_POLICY="one_tile", STREAMK_FIXUP="owner", STREAMK_NUM_PROGRAMS="64",
                    STREAMK_PARTIALS="bf16")),
    (8192, 24, dict(STREAMK_POLICY="one_tile", STREAMK_FIXUP="rs", STREAMK_NUM_PROGRAMS="144",
                    STREAMK_RS_READERS="all")),
    (8192, 16, dict(STREAMK_POLICY="two_tile")),
    (16384, 192, dict(STREAMK_POLICY="spread")),
    (8192, 16, dict(STREAMK_POLICY="dp")),
    (8192, 16, dict()),
]
# (K, S, programs per tile n or None for S n not a multiple, env): v24's own options.
OWN = []
for _fx, _parts in (("owner", ("fp32", "bf16_pairs")), ("rs", ("fp32", "bf16_pairs")), ("atomic_iz", ("fp32",))):
    for _pl in ("tile", "chunk"):
        for _p in _parts:
            for _snp in ("128", "100"):
                OWN.append((8192, 8, dict(STREAMK_POLICY="one_tile", STREAMK_FIXUP=_fx, STREAMK_NUM_PROGRAMS=_snp,
                                          STREAMK_PARTIALS=_p, STREAMK_PLACEMENT=_pl)))
for _p in ("fp32", "bf16_pairs"):
    for _snp in ("128", "64"):
        OWN.append((8192, 16, dict(STREAMK_POLICY="one_tile", STREAMK_FIXUP="rs", STREAMK_NUM_PROGRAMS=_snp,
                                   STREAMK_PARTIALS=_p, STREAMK_EXCHANGE="l2")))


def compile_case(mod, K, S, env):
    for k in ENV_KEYS:
        os.environ.pop(k, None)
    os.environ.update(env)
    os.environ["GEMM_TILE_LIMIT"] = str(256 + S)
    fn = mod.v23_streamk_rs_band if mod is v23 else mod.v24_streamk_combined
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


def instructions(asm, name):
    """The instruction lines, up to the metadata (which lists the kernel arguments)."""
    out = []
    for ln in asm.splitlines():
        if ln.strip().startswith(".amdgpu_metadata"):
            break
        ln = ln.split(";")[0].strip().replace(name, "KERNEL")
        if ln and not ln.startswith("."):
            out.append(ln)
    return out


def ttgir_body(ttgir):
    """The TTGIR with value names, block labels and locations normalised, without the signature."""
    out = []
    for ln in ttgir.splitlines():
        if "tt.func public" in ln or ln.lstrip().startswith("#loc"):
            continue
        ln = re.sub(r"loc\(#loc\d*\)", "", ln)
        ln = re.sub(r"#loc\d+", "#loc", ln)
        ln = re.sub(r"%[\w.#]+", "%v", ln)
        ln = re.sub(r"\^bb\d+", "^bb", ln).rstrip()
        if ln.strip():
            out.append(ln)
    return out


def changed_lines(x, y):
    return sum(1 for d in difflib.unified_diff(x, y, lineterm="", n=0)
               if d[:1] in "+-" and not d.startswith(("+++", "---")))


def census(asm):
    c = Counter()
    for ln in asm.splitlines():
        ln = ln.split(";")[0].strip()
        if not ln or ln.startswith("."):
            continue
        words = re.split(r"[\s,]+", ln)
        op = words[0]
        if op.startswith(("buffer_", "global_")):
            bits = [b for b in ("sc0", "sc1", "nt") if b in words]
            c[" ".join([op] + bits)] += 1
    return c


def label(env):
    return ", ".join(f"{k.removeprefix('STREAMK_')}={v}" for k, v in env.items()) or "(defaults)"


def main():
    ok = True
    print("# v24 asm check\n")
    print(f"{M}x{N}xK, GEMM_TILE_LIMIT = 256 + S.\n")
    print("## Against v23 (same configuration, kernel name replaced, comments and directives dropped)\n")
    print("| K | S | configuration | v23 VGPRs / spills | v24 VGPRs / spills | TTGIR lines changed | instructions | "
          "instruction lines changed | result |")
    print("|---|---|---|---|---|---|---|---|---|")
    for K, S, env in SAME:
        k23 = compile_case(v23, K, S, env)
        k24 = compile_case(v24, K, S, env)
        ir = changed_lines(ttgir_body(k23.asm["ttgir"]), ttgir_body(k24.asm["ttgir"]))
        i23 = instructions(k23.asm["amdgcn"], "v23_streamk_rs_band")
        i24 = instructions(k24.asm["amdgcn"], "v24_streamk_combined")
        diff = changed_lines(i23, i24)
        same = ir == 0 and (k23.n_regs, k23.n_spills) == (k24.n_regs, k24.n_spills)
        ok &= same
        res = ("identical" if diff == 0 else "same TTGIR, LLVM order differs") if same else "FAIL"
        print(f"| {K} | {S} | {label(env)} | {k23.n_regs} / {k23.n_spills} | {k24.n_regs} / {k24.n_spills} | {ir} | "
              f"{len(i23)} / {len(i24)} | {diff} | {res} |", flush=True)
    print("\n## v24's options\n")
    print("| K | S | configuration | VGPRs | spills | atomic adds | partial stores | partial and slice loads | "
          "checks |")
    print("|---|---|---|---|---|---|---|---|---|")
    for K, S, env in OWN:
        k = compile_case(v24, K, S, env)
        c = census(k.asm["amdgcn"])
        atom = {x: v for x, v in c.items() if x.startswith("buffer_atomic_pk_add")}
        # Partial stores: write-through (sc0 sc1) unless l2; slice and owner loads: .cv (sc0 sc1) or .cg (sc0 nt).
        wt_st = {x: v for x, v in c.items() if x.startswith("buffer_store") and x.endswith("sc0 sc1")}
        st = {x: v for x, v in c.items() if x.startswith("buffer_store") and x.split()[0] in
              ("buffer_store_dwordx4", "buffer_store_dwordx2") and len(x.split()) == 1}
        ld = {x: v for x, v in c.items() if x.startswith("buffer_load") and (x.endswith("sc0 sc1") or x.endswith("sc0 nt"))}
        checks = {"no spills": k.n_spills == 0}
        fx = env["STREAMK_FIXUP"]
        if fx == "atomic_iz":
            checks["pk_add adds"] = bool(atom)
            checks["no partials"] = not ld and set(wt_st) <= {"buffer_store_dwordx2 sc0 sc1"}
        elif env.get("STREAMK_EXCHANGE") == "l2":
            checks["no write-through partials"] = not any(x.split()[0] in ("buffer_store_dwordx4", "buffer_store_dwordx2")
                                                          for x in wt_st)
            checks["slice loads sc0 nt"] = bool(ld) and all(x.endswith("sc0 nt") for x in ld)
        else:
            checks["write-through partials"] = bool(wt_st)
        ok &= all(checks.values())
        bad = [x for x, g in checks.items() if not g]
        fmt = lambda d: ", ".join(f"`{x}` {v}" for x, v in sorted(d.items())) or "-"  # noqa: E731
        stores = wt_st if env.get("STREAMK_EXCHANGE") != "l2" else st
        print(f"| {K} | {S} | {label(env)} | {k.n_regs} | {k.n_spills} | {fmt(atom)} | {fmt(stores)} | {fmt(ld)} | "
              f"{'ok' if not bad else 'FAIL: ' + ', '.join(bad)} |", flush=True)
    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
