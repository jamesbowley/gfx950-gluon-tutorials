"""Stages hook: build the amdgcn stage with the pinned llc and extra flags instead of Triton's
in-process codegen, so LLVM options that Triton v2.2 cannot set (e.g. -amdgpu-mfma-vgpr-form=0)
can be tried on real kernels.

    PAC_LLC_FLAGS="-amdgpu-mfma-vgpr-form=0" python bench_llc.py --version 13 ...

The LLIR scheduler plugin runs at OptimizerLast in make_llir, so its effect is already in the
.llir this stage receives. With TRITON_AMDGCNAS_PLUGIN set, the amdgcnas peephole runs on llc's
output, as it does on Triton's. llc differs slightly from Triton's in-process codegen (v13 without
plugins: 312 accumulator copies against 324), so compare llc builds with llc builds.
"""

import hashlib
import os
import re
import subprocess
import tempfile

LLC = "/root/.triton/llvm/llvm-b010a18d-ubuntu-x64-1/bin/llc"
BASE = ["-mtriple=amdgcn-amd-amdhsa", "-mcpu=gfx950", "-O3", "-fp-contract=fast", "-relocation-model=pic"]


def _flags():
    return os.environ.get("PAC_LLC_FLAGS", "").split()


def _amdgcnas():
    return bool(os.environ.get("TRITON_AMDGCNAS_PLUGIN"))


def get_key():
    key = "pac-llc-stage:" + " ".join(BASE + _flags()) + open(__file__).read()
    if _amdgcnas():
        import amdgcnas_plugin
        key += amdgcnas_plugin.get_key()
    return key


def inspect_stages_hook(self=None, stages=None, options=None, language=None, capability=None):
    key = get_key()
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    if stages is None or "amdgcn" not in stages:
        return key, digest

    def make_amdgcn(src, metadata):
        names = re.findall(r"define amdgpu_kernel void @([a-zA-Z_][a-zA-Z0-9_]*)", src)
        assert len(names) == 1
        metadata["name"] = names[0]
        with tempfile.TemporaryDirectory() as d:
            ll, s = os.path.join(d, "k.ll"), os.path.join(d, "k.s")
            open(ll, "w").write(src)
            # Under rocprofv3 the process LD_PRELOADs rocprofiler-sdk, which drags ROCm's libLLVM into
            # llc and aborts it; llc needs none of the profiler environment.
            env = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD" and not k.startswith("ROCP")}
            subprocess.run([LLC, *BASE, *_flags(), ll, "-o", s], check=True, env=env)
            asm = open(s).read()
        if _amdgcnas():
            import amdgcnas_ext
            asm = amdgcnas_ext.amdgcn_as(asm, False)
        return asm

    stages["amdgcn"] = make_amdgcn
    return key, digest
