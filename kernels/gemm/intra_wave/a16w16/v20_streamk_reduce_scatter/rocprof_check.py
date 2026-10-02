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
"""Cold-cache kernel time and warm L2 counters, per shape and version.

    cd kernels/gemm/intra_wave/a16w16
    HIP_VISIBLE_DEVICES=4 python v20_streamk_reduce_scatter/rocprof_check.py 4352x4096x8192,3328x5120x8192 13,17,19,20
    HIP_VISIBLE_DEVICES=4 python v20_streamk_reduce_scatter/rocprof_check.py 4352x4096x8192 13,20 --no-l2

- Kernel time: rocprofv3 --kernel-trace over `bench.py --rocprof` (1000 launches over a rotating
  512 MB of inputs), median dispatch duration.
- L2: rocprofv3 --pmc TCC_HIT_sum TCC_MISS_sum TCC_EA0_RDREQ_sum over 10 launches on the same
  inputs (warm), median of launches 3-10; hit rate is HIT / (HIT + MISS).

Versions are specs as in version_spec.py (for example 20:STREAMK_POLICY=spread); without
one, v20 runs with whatever STREAMK_POLICY / STREAMK_FIXUP the environment sets.
"""
import csv
import glob
import os
import subprocess
import sys
import tempfile

A16 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, A16)
from v20_streamk_reduce_scatter import version_spec  # noqa: E402

NAMES = {9: "v9_beyond_hotloop", 10: "v10_persistant", 11: "v11_persistant_overlap_global",
         12: "v12_persistant_overlap_lds", 13: "v13_persistant_peel_acc", 14: "v14_streamk",
         15: "v15_streamk_onetile", 16: "v16_streamk_lane_partials",
         17: "v17_streamk_tile_aligned", 18: "v18_streamk_chunk_major", 19: "v19_streamk_two_tile_reversed",
         20: "v20_streamk_reduce_scatter"}

PMC_SCRIPT = r'''
import sys, os
sys.path.insert(0, "%s")
import bench, torch, importlib
mod = importlib.import_module(bench.VERSION_MAP[%d] + ".matmul_kernel")
M, N, K = %d, %d, %d
a = torch.rand((M, K), device="cuda", dtype=torch.bfloat16) - 0.5
b = torch.rand((N, K), device="cuda", dtype=torch.bfloat16).T - 0.5
for _ in range(10):
    mod.matmul(a, b)
torch.cuda.synchronize()
'''


def run(cmd, env):
    p = subprocess.run(cmd, cwd=A16, env=env, capture_output=True, text=True)
    if p.returncode:
        print(p.stderr[-2000:], file=sys.stderr)
    return p


def kernel_rows(d, name):
    rows = []
    for f in glob.glob(os.path.join(d, "**", "*.csv"), recursive=True):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                if name in r.get("Kernel_Name", ""):
                    rows.append(r)
    return rows


def cold_us(v, M, N, K, env):
    d = tempfile.mkdtemp(prefix="rp_")
    run(["rocprofv3", "--kernel-trace", "--kernel-include-regex", NAMES[v], "-d", d, "-o", "run",
         "--output-format", "csv", "--", sys.executable, "bench.py", "--version", str(v), "--M", str(M),
         "--N", str(N), "--K", str(K), "--dtype", "bf16", "--rocprof"], env)
    rows = [r for r in kernel_rows(d, NAMES[v]) if "End_Timestamp" in r]
    t = sorted((int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3 for r in rows)
    return t[len(t) // 2] if t else float("nan")


def l2(v, M, N, K, env):
    d = tempfile.mkdtemp(prefix="rp_")
    script = os.path.join(d, "pmc.py")
    with open(script, "w") as f:
        f.write(PMC_SCRIPT % (A16, v, M, N, K))
    run(["rocprofv3", "--pmc", "TCC_HIT_sum", "TCC_MISS_sum", "TCC_EA0_RDREQ_sum", "--kernel-include-regex",
         NAMES[v], "-d", d, "-o", "run", "--output-format", "csv", "--", sys.executable, script], env)
    by_disp = {}
    for r in kernel_rows(d, NAMES[v]):
        if "Counter_Name" in r:
            by_disp.setdefault(int(r["Dispatch_Id"]), {})[r["Counter_Name"]] = float(r["Counter_Value"])
    disp = sorted(by_disp)[2:]
    if not disp:
        return float("nan"), float("nan")
    hits = sorted(by_disp[x]["TCC_HIT_sum"] / (by_disp[x]["TCC_HIT_sum"] + by_disp[x]["TCC_MISS_sum"]) for x in disp)
    reads = sorted(by_disp[x]["TCC_EA0_RDREQ_sum"] for x in disp)
    return hits[len(hits) // 2], reads[len(reads) // 2]


def main():
    shapes = [tuple(int(x) for x in s.split("x")) for s in sys.argv[1].split(",")]
    versions = sys.argv[2].split(",")
    with_l2 = "--no-l2" not in sys.argv[3:]
    print("| shape | version | cold kernel time (us) | L2 hit rate | memory read requests |")
    print("|---|---|---|---|---|")
    for M, N, K in shapes:
        for spec in versions:
            v, extra = version_spec.parse(spec)
            env = {**os.environ, **extra}
            us = cold_us(v, M, N, K, env)
            hit, reads = l2(v, M, N, K, env) if with_l2 else (float("nan"), float("nan"))
            print(f"| {M}x{N}x{K} | {version_spec.label(spec)} | {us:.1f} | {100 * hit:.1f}% | {reads / 1e6:.2f}M |",
                  flush=True)


if __name__ == "__main__":
    main()
