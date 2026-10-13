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
"""Cold kernel-only median (rocprofv3 --kernel-trace over stride_bench.py) per case V:MxNxK:S[:ENV=VAL].

    cd kernels/gemm/intra_wave/a16w16
    HIP_VISIBLE_DEVICES=1 python v26_streamk_ragged_k/stride_rp.py 26:4352x4096x8192:8256 26:4352x4096x8176:8256

S is the row stride of A and of B (as an (N, K) tensor), the same for every K of a comparison, so a
ragged K and its rounded-up K read rows that start at the same 128-byte alignment and only the
kernel differs. With S = K the rows of a K that is not a multiple of 64 do not start on a line.
"""
import os
import statistics
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.getcwd(), "v20_streamk_reduce_scatter"))
import rocprof_check as rc  # noqa: E402

REGEX = "v9_beyond_hotloop|v9_ragged|v9_tile|v24_streamk_combined|v25_streamk_ragged_mn|v26_streamk_ragged_k"
HERE = os.path.dirname(os.path.abspath(__file__))
print("| version | shape | row stride | env | cold kernel us |")
print("|---|---|---|---|---|")
for case in sys.argv[1:]:
    v, shape, S, *env = case.split(":")
    M, N, K = shape.split("x")
    d = tempfile.mkdtemp(prefix="srp_")
    p = subprocess.run(["rocprofv3", "--kernel-trace", "--kernel-include-regex", REGEX, "-d", d, "-o", "run",
                        "--output-format", "csv", "--", sys.executable, os.path.join(HERE, "stride_bench.py"),
                        v, M, N, K, S], env={**os.environ, **dict(e.split("=", 1) for e in env)},
                       capture_output=True, text=True)
    rows = [r for r in rc.kernel_rows(d, REGEX) if "End_Timestamp" in r]
    if p.returncode or not rows:
        print(f"| {v} | {shape} | {S} | {' '.join(env) or '-'} | FAILED: {(p.stderr or '')[-200:]!r} |")
        continue
    us = statistics.median((int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3 for r in rows[1:])
    print(f"| {v} | {shape} | {S} | {' '.join(env) or '-'} | {us:.1f} |", flush=True)
