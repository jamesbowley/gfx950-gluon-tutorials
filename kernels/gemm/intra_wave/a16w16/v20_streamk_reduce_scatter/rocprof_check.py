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
    HIP_VISIBLE_DEVICES=4 python v20_streamk_reduce_scatter/rocprof_check.py 4096x4096x1024 9,13 --warm --no-pad

- Kernel time: rocprofv3 --kernel-trace over `bench.py --rocprof` (1000 launches over a rotating
  512 MB of inputs, or over one copy with --warm), median dispatch duration. B and C rows are
  padded as bench.py does unless --no-pad.
- L2: rocprofv3 --pmc TCC_HIT_sum TCC_MISS_sum TCC_EA0_RDREQ_sum over 10 launches on the same
  inputs (warm), median of launches 3-10; hit rate is HIT / (HIT + MISS).

Versions are specs as in version_spec.py (for example 20:STREAMK_POLICY=spread); without
one, v20 runs with whatever STREAMK_POLICY / STREAMK_FIXUP the environment sets.
"""
import csv
import glob
import os
import re
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
         20: "v20_streamk_reduce_scatter", 22: "v22_streamk_atomic_c", 23: "v23_streamk_rs_band",
         24: "v24_streamk_combined", 25: "v25_streamk_ragged_mn", 26: "v26_streamk_ragged_k", "9r": "v9_ragged"}
# v20-v24 hand data-parallel shapes of at most one wave to v9 (v25 and v26 to v9_ragged), so their
# traces can hold that kernel.
DELEGATES = {20: (9,), 21: (9,), 22: (9,), 23: (9,), 24: (9,), 25: ("9r",), 26: ("9r",)}
# Kernels a version launches before its GEMM in the same call (v22's atomic ending zeroes the
# stream-K tiles of C first); cold_detail times the call from their start.
PRE_KERNELS = {22: ("v22_zero_tiles",)}


def kernel_regex(v):
    """Kernel names version v can launch, as a regex for rocprofv3 and kernel_rows."""
    return "|".join([NAMES[x] for x in (v, *DELEGATES.get(v, ()))] + list(PRE_KERNELS.get(v, ())))

PMC_SCRIPT = r'''
import sys, os
sys.path.insert(0, "%s")
import bench, torch, importlib
mod = importlib.import_module(bench.VERSION_MAP[%d] + ".matmul_kernel")
M, N, K, pad = %d, %d, %d, %s
a = torch.rand((M, K), device="cuda", dtype=torch.bfloat16) - 0.5
b = bench.make_b(N, K, torch.bfloat16, pad, lambda t: t.uniform_(-0.5, 0.5))
c = bench.make_c(M, N, torch.bfloat16, pad)
for _ in range(10):
    mod.matmul(a, b, c)
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
                if re.search(name, r.get("Kernel_Name", "")):
                    rows.append(r)
    return rows


def cold_detail(v, M, N, K, env, bench_flags=()):
    """Medians over the dispatches, us: {"us": the call, "gemm_us": the GEMM kernel alone,
    "pre_us": the kernels launched before it (PRE_KERNELS)}. With pre-kernels in the trace, a call
    runs from the start of the pre-kernel dispatched just before the GEMM to the GEMM's end, so it
    includes the gap between the two launches; otherwise it is the GEMM's duration."""
    d = tempfile.mkdtemp(prefix="rp_")
    run(["rocprofv3", "--kernel-trace", "--kernel-include-regex", kernel_regex(v), "-d", d, "-o", "run",
         "--output-format", "csv", "--", sys.executable, "bench.py", "--version", str(v), "--M", str(M),
         "--N", str(N), "--K", str(K), "--dtype", "bf16", "--rocprof", *bench_flags], env)
    rows = sorted((r for r in kernel_rows(d, kernel_regex(v)) if "End_Timestamp" in r),
                  key=lambda r: int(r["Start_Timestamp"]))
    pre_names = PRE_KERNELS.get(v, ())

    def is_pre(r):
        return any(p in r["Kernel_Name"] for p in pre_names)

    def med(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2] if xs else float("nan")

    calls, gemm, pre, last_pre = [], [], [], None
    for r in rows:
        s, e = int(r["Start_Timestamp"]), int(r["End_Timestamp"])
        if is_pre(r):
            last_pre = s
            pre.append((e - s) / 1e3)
            continue
        gemm.append((e - s) / 1e3)
        calls.append((e - (last_pre if last_pre is not None else s)) / 1e3)
        last_pre = None
    return {"us": med(calls), "gemm_us": med(gemm), "pre_us": med(pre)}


def cold_us(v, M, N, K, env, bench_flags=()):
    return cold_detail(v, M, N, K, env, bench_flags)["us"]


def l2(v, M, N, K, env, pad=True):
    d = tempfile.mkdtemp(prefix="rp_")
    script = os.path.join(d, "pmc.py")
    with open(script, "w") as f:
        f.write(PMC_SCRIPT % (A16, v, M, N, K, pad))
    run(["rocprofv3", "--pmc", "TCC_HIT_sum", "TCC_MISS_sum", "TCC_EA0_RDREQ_sum", "--kernel-include-regex",
         kernel_regex(v), "-d", d, "-o", "run", "--output-format", "csv", "--", sys.executable, script], env)
    by_disp = {}
    for r in kernel_rows(d, kernel_regex(v)):
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
    flags = sys.argv[3:]
    with_l2 = "--no-l2" not in flags
    pad = "--no-pad" not in flags
    bench_flags = (["--rotating-buffer-size", "0"] if "--warm" in flags else []) + ([] if pad else ["--no-pad"])
    cache = "warm" if "--warm" in flags else "cold"
    print(f"| shape | version | {cache} kernel time (us) | L2 hit rate | memory read requests |")
    print("|---|---|---|---|---|")
    for M, N, K in shapes:
        for spec in versions:
            v, extra = version_spec.parse(spec)
            env = {**os.environ, **extra}
            us = cold_us(v, M, N, K, env, bench_flags)
            hit, reads = l2(v, M, N, K, env, pad) if with_l2 else (float("nan"), float("nan"))
            print(f"| {M}x{N}x{K} | {version_spec.label(spec)} | {us:.1f} | {100 * hit:.1f}% | {reads / 1e6:.2f}M |",
                  flush=True)


if __name__ == "__main__":
    main()
