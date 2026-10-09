#!/usr/bin/env python3
"""Accumulator copies under llc option variants, on .llir saved by compile_stats.py --keep.

    python llc_variants.py builds/v13.llir [builds/v12.llir ...]

The .llir is first run through the same IR cleanup as Triton's in-process codegen
(separate-const-offset-from-gep, early-cse), then llc with Triton's target options. llc
reproduces Triton's copy counts closely (v13: 312 here, 324 in-process; v12: 0 in both).
"""

import os
import subprocess
import sys
import tempfile

from compile_stats import asm_stats

BIN = "/root/.triton/llvm/llvm-b010a18d-ubuntu-x64-1/bin"
BASE = ["-mtriple=amdgcn-amd-amdhsa", "-mcpu=gfx950", "-O3", "-fp-contract=fast", "-relocation-model=pic"]
VARIANTS = {
    "baseline": [],
    "enable-local-reassign": ["-enable-local-reassign"],
    "greedy-reverse-local-assignment": ["-greedy-reverse-local-assignment"],
    "greedy-regclass-priority-trumps-globalness": ["-greedy-regclass-priority-trumps-globalness"],
    "exhaustive-register-search": ["-exhaustive-register-search"],
    "lcr-max-depth=20,lcr-max-interf=64": ["-lcr-max-depth=20", "-lcr-max-interf=64"],
    "split-spill-mode=size": ["-split-spill-mode=size"],
    "split-spill-mode=speed": ["-split-spill-mode=speed"],
    "amdgpu-mfma-vgpr-form=0": ["-amdgpu-mfma-vgpr-form=0"],
    "amdgpu-mfma-vgpr-form=1": ["-amdgpu-mfma-vgpr-form=1"],
    "enable-misched=0": ["-enable-misched=0"],
}


def main():
    args = sys.argv[1:]
    only = None
    if "--only" in args:
        i = args.index("--only")
        only = args[i + 1].split(",")
        del args[i:i + 2]
    print("| input | llc options | accvgpr_mov | s_nop | accvgpr_read | accvgpr_write | vgpr_count | agpr_count | spills |")
    print("|---|---|---|---|---|---|---|---|---|")
    for llir in args:
        bc = tempfile.mktemp(suffix=".bc")
        subprocess.run([f"{BIN}/opt", "-passes=separate-const-offset-from-gep,early-cse", llir, "-o", bc], check=True)
        for name, opts in VARIANTS.items():
            if only and name not in only:
                continue
            s = tempfile.mktemp(suffix=".s")
            p = subprocess.run([f"{BIN}/llc", *BASE, *opts, bc, "-o", s], capture_output=True, text=True)
            if p.returncode:
                print(f"| {os.path.basename(llir)} | {name} | failed: {p.stderr.strip()[:80]} |||||||")
                continue
            st = asm_stats(open(s).read())
            print(f"| {os.path.basename(llir)} | {name} | {st['accvgpr_mov']} | {st['s_nop']} | {st['accvgpr_read']} | "
                  f"{st['accvgpr_write']} | {st['vgpr_count']} | {st['agpr_count']} | {st['vgpr_spill_count']} |", flush=True)
            os.remove(s)
        os.remove(bc)


if __name__ == "__main__":
    main()
