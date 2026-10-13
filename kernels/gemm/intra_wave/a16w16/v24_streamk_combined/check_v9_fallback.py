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
"""Which kernel v24 runs when a host rule picks data-parallel, and whether C is right (v20's check).

    cd kernels/gemm/intra_wave/a16w16
    HIP_VISIBLE_DEVICES=0 python v24_streamk_combined/check_v9_fallback.py

For every shape, policy and bias setting, counts the calls to v9's matmul and checks them against
the rule: v9 exactly when capped / split_rule / auto picks dp and the tiles fit in one wave (an
explicit STREAMK_POLICY=dp never). Checks C against an fp32 torch reference with bench_ext.py's
tolerance. Needs the same stack as regress.py.
"""

import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import bench_ext as bench  # noqa: E402  (plugin hooks and the path to common.py)
import torch  # noqa: E402
import triton  # noqa: E402

SHAPES = [
    (4096, 4096, 8192),  # 256 tiles, S = 0
    (3840, 4096, 8192),  # 240 tiles, S > 128 below 512 k-tiles
    (256, 49152, 8192),  # 192 tiles, skinny
    (2048, 2048, 1024),  # 64 tiles: capped's minimum-K cut, auto splits
    (3840, 4096, 1024),  # 240 tiles at 16 k-tiles
    (4352, 4096, 8192),  # 272 tiles, one-tile split
    (4352, 4096, 1024),  # 272 tiles, dp with more than one wave
]
POLICIES = {24: ("capped", "split_rule", "auto", "dp")}


def rule_policy(mk, policy, tiles, k_tiles):
    if policy == "capped":
        return mk.capped_policy(tiles, k_tiles, 256)[0]
    if policy == "split_rule":
        return mk.split_rule_policy(tiles, k_tiles, 256)[0]
    if policy == "auto":
        return mk.streamk_policy(tiles, k_tiles, 256)[0]
    return None


def main():
    failures = 0
    print("| version | shape | policy | bias | runs v9 | expected | matches torch |")
    print("|---|---|---|---|---|---|---|")
    for v, policies in POLICIES.items():
        mk = importlib.import_module(bench.VERSION_MAP[v] + ".matmul_kernel")
        calls = []
        v9_matmul = mk.v9_matmul

        def counting(*args, **kwargs):
            calls.append(1)
            return v9_matmul(*args, **kwargs)

        mk.v9_matmul = counting
        for M, N, K in SHAPES:
            tiles = triton.cdiv(M, 256) * triton.cdiv(N, 256)
            for policy in policies:
                os.environ["STREAMK_POLICY"] = policy
                want = rule_policy(mk, policy, tiles, K // 64) == "dp" and tiles <= 256
                for with_bias in (False, True):
                    torch.manual_seed(0)
                    a = torch.rand((M, K), device="cuda", dtype=torch.bfloat16) - 0.5
                    b = bench.make_b(N, K, torch.bfloat16, False, lambda t: t.uniform_(-0.5, 0.5))
                    c = bench.make_c(M, N, torch.bfloat16, False)
                    bias = torch.rand((N,), device="cuda", dtype=torch.bfloat16) - 0.5 if with_bias else None
                    calls.clear()
                    out = mk.matmul(a, b, c, bias)
                    ref = a.float() @ b.float() + (bias.float() if with_bias else 0)
                    ok = torch.allclose(out, ref.to(torch.bfloat16), atol=1e-1, rtol=1e-2)
                    ran_v9 = bool(calls)
                    failures += not ok or ran_v9 != want
                    print(f"| v{v} | {M}x{N}x{K} | {policy} | {int(with_bias)} | {int(ran_v9)} | {int(want)} | {int(ok)} |",
                          flush=True)
        mk.v9_matmul = v9_matmul
    os.environ.pop("STREAMK_POLICY", None)
    print(f"\n{failures} failures")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
