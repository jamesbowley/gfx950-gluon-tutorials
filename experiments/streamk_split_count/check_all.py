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
"""Stress-check STREAMK_NUM_PROGRAMS in every stream-K version (v15-v20), using each version's
own check_streamk.py (rotating inputs, every result checked, flags re-armed).

    HIP_VISIBLE_DEVICES=3 python experiments/streamk_split_count/check_all.py

For each shape the programs per tile n are 1, 2 and the default (all 256 programs), so
STREAMK_NUM_PROGRAMS is S, 2 S and unset. v9 order, with and without bias, 10 launches. v20 also
runs its spread and two-tile policies, both endings, one shape with STREAMK_BALANCE_XCDS=0 and
two small-K shapes.
"""

import os
import subprocess
import sys

A16 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "kernels", "gemm", "intra_wave", "a16w16")
VERSIONS = ["v15_streamk_onetile", "v16_streamk_lane_partials", "v17_streamk_tile_aligned",
            "v18_streamk_chunk_major", "v19_streamk_two_tile_reversed", "v20_streamk_reduce_scatter"]
SHAPES = [(3328, 5120, 8192), (4352, 4096, 8192), (3584, 5376, 8192), (4096, 5120, 8192)]
SMALL_K = [(4352, 4096, 512), (4352, 4096, 1024), (3584, 5376, 1024)]


def leftover(M, N):
    return (M // 256) * (N // 256) % 256


def run(version, shapes, env_extra, args):
    env = {**os.environ, **env_extra}
    cmd = [sys.executable, "-W", "ignore", os.path.join(version, "check_streamk.py"), "--shapes",
           *["x".join(map(str, s)) for s in shapes], "--launches", "10", "--orders", "v9", *args]
    p = subprocess.run(cmd, cwd=A16, env=env, capture_output=True, text=True, timeout=7200)
    rows = [l for l in p.stdout.splitlines() if l.startswith("| ") and "---" not in l and "shape" not in l]
    ok = p.returncode == 0 and "PASS" in p.stdout
    return ok, rows, (p.stderr or p.stdout).strip().splitlines()[-3:]


def main():
    failures = 0
    print("| version | STREAMK_NUM_PROGRAMS | other env | shape | policy / fixup / order / bias / launches / wrong "
          "results / flags left up |")
    print("|---|---|---|---|---|")
    for version in VERSIONS:
        for shape in SHAPES:
            S = leftover(*shape[:2])
            for snp in (S, 2 * S, None):
                env = {} if snp is None else {"STREAMK_NUM_PROGRAMS": str(snp)}
                extra = []
                if version.startswith("v20"):
                    extra = ["--policies", "one_tile", "spread", "two_tile", "--fixups", "rs", "owner"]
                ok, rows, tail = run(version, [shape], env, extra)
                failures += not ok
                for r in rows or [f"| {shape} | error: {' / '.join(tail)} |"]:
                    print(f"| {version} | {snp or 'default'} | | {r.strip('| ')} |", flush=True)
    v20 = VERSIONS[-1]
    for env, shapes, extra in (
        ({"STREAMK_NUM_PROGRAMS": "76", "STREAMK_BALANCE_XCDS": "0"}, [SHAPES[2]],
         ["--policies", "one_tile", "spread", "--fixups", "rs", "owner"]),
        ({}, SMALL_K, ["--policies", "one_tile", "spread", "two_tile", "--fixups", "rs", "owner"]),
        ({"STREAMK_NUM_PROGRAMS": "32"}, SMALL_K[:2], ["--policies", "one_tile", "spread", "--fixups", "rs", "owner"]),
    ):
        ok, rows, tail = run(v20, shapes, env, extra)
        failures += not ok
        label = " ".join(f"{k}={v}" for k, v in env.items())
        for r in rows or [f"| error: {' / '.join(tail)} |"]:
            print(f"| {v20} | {env.get('STREAMK_NUM_PROGRAMS', 'default')} | {label} | {r.strip('| ')} |", flush=True)
    print("\nPASS" if failures == 0 else f"\nFAIL ({failures} runs)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
