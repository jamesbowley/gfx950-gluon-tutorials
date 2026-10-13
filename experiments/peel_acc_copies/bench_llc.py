#!/usr/bin/env python3
"""kernels/gemm/intra_wave/a16w16/bench.py with the amdgcn stage built by llc_stage.py.

    PAC_LLC_FLAGS="-amdgpu-mfma-vgpr-form=0" python bench_llc.py --version 13 --K 1024 --dtype bf16 --rocprof
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
A16 = os.path.join(HERE, "..", "..", "kernels", "gemm", "intra_wave", "a16w16")
sys.path[:0] = [A16, HERE]

import bench_ext as bench  # noqa: E402  (sets up paths, dlopen flags and the amdgcnas import)
import llc_stage  # noqa: E402
from triton import knobs  # noqa: E402

knobs.runtime.add_stages_inspection_hook = llc_stage.inspect_stages_hook

if __name__ == "__main__":
    bench.main()
