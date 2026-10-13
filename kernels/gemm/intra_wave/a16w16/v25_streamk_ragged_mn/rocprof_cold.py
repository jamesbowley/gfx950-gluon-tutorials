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
"""Cold kernel-only times for the version keys v20's rocprof_check does not know (25, 26, 9r, 9t).

    cd kernels/gemm/intra_wave/a16w16
    HIP_VISIBLE_DEVICES=0 python v25_streamk_ragged_mn/rocprof_cold.py 4300x4000x8192:25 \
        4352x4096x8192:25:STREAMK_FORCE_EDGE=1 4000x4000x8192:9r > v25_streamk_ragged_mn/results/rocprof_cold.md

Each case is SHAPE:VERSION[:NAME=VALUE...], one rocprofv3 --kernel-trace run of bench.py --rocprof
(1000 launches over a rotating 512 MB, B and C rows padded as bench.py does), median dispatch
duration, as rocprof_check.cold_us. The kernel column names what ran (v25 hands one-wave
data-parallel shapes to v9_ragged).
"""

import os
import statistics
import sys
import tempfile

A16 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, A16)
sys.path.insert(0, os.path.join(A16, "v20_streamk_reduce_scatter"))
import rocprof_check as rc  # noqa: E402

REGEX = {"9": "v9_beyond_hotloop", "9r": "v9_ragged", "9t": "v9_tile",
         "24": "v24_streamk_combined|v9_beyond_hotloop", "25": "v25_streamk_ragged_mn|v9_ragged",
         "26": "v26_streamk_ragged_k|v9_ragged"}


def cold(v, M, N, K, extra):
    d = tempfile.mkdtemp(prefix="rp_")
    rc.run(["rocprofv3", "--kernel-trace", "--kernel-include-regex", REGEX[v], "-d", d, "-o", "run",
            "--output-format", "csv", "--", sys.executable, "bench.py", "--version", v, "--M", str(M),
            "--N", str(N), "--K", str(K), "--dtype", "bf16", "--rocprof"], {**os.environ, **extra})
    rows = [r for r in rc.kernel_rows(d, REGEX[v]) if "End_Timestamp" in r]
    names = sorted({r["Kernel_Name"].split("(")[0] for r in rows})
    us = statistics.median((int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3 for r in rows)
    return us, names


def main():
    print("| shape | version | env | cold kernel us | kernel |")
    print("|---|---|---|---|---|")
    for case in sys.argv[1:]:
        shape, v, *env = case.split(":")
        M, N, K = map(int, shape.split("x"))
        us, names = cold(v, M, N, K, dict(kv.split("=", 1) for kv in env))
        print(f"| {shape} | {v} | {' '.join(env) or '-'} | {us:.1f} | {','.join(names)} |", flush=True)


if __name__ == "__main__":
    main()
