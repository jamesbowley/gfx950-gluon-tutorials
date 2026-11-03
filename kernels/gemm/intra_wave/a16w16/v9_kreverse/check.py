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
"""Correctness of v9_kreverse for every K order on the study shapes, against torch and v9.

    source v9_kreverse/env.sh
    python v9_kreverse/check.py > v9_kreverse/results/check.md

K_ORDER = 0 must match v9 bit for bit. The backward orders sum the k-steps in another order, so
they are checked against torch with bench_ext's tolerance and their largest difference from v9 is
reported. An order probe then checks which dispatch waves actually run backward.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import bench_ext as bench  # noqa: E402,F401  (plugin hooks; imports triton before torch)
import torch  # noqa: E402
from v9_beyond_hotloop import matmul_kernel as v9  # noqa: E402
from v9_kreverse import matmul_kernel as v9k  # noqa: E402
from v9_kreverse.shapes import SHAPES  # noqa: E402


def wave_of_tile(M, N, BM=256, BN=256, NX=8, GM=4, wave_wgs=256):
    """Dispatch wave of each output tile under get_pids (grids that divide by NX)."""
    npm, npn = M // BM, N // BN
    G = npm * npn
    w = torch.zeros(npm, npn, dtype=torch.int)
    for pid in range(G):
        p = (pid % NX) * (G // NX) + pid // NX
        g = GM * npn
        w[(p // g) * GM + (p % g) % GM, (p % g) // GM] = pid // wave_wgs
    return w


def order_probe(M=8192, N=8192, K=8192):
    """Which tiles run backward. The k = 0 product is +2^60 and the k = K - 1 product -2^60, the
    others 1: the 1s summed before the big terms are lost, so C keeps only the 1s summed after
    both, about 0 forward and about 120 backward."""
    a = torch.ones((M, K), device="cuda", dtype=torch.bfloat16)
    a[:, 0], a[:, -1] = 2.0**30, -(2.0**30)
    bt = torch.ones((N, K), device="cuda", dtype=torch.bfloat16)
    bt[:, 0], bt[:, -1] = 2.0**30, 2.0**30
    waves = wave_of_tile(M, N, wave_wgs=torch.cuda.get_device_properties(0).multi_processor_count)
    ok = True
    print(f"\n## K order probe ({M}x{N}x{K}): C per tile, by dispatch wave\n")
    print("| K_ORDER | " + " | ".join(f"wave {w}" for w in range(int(waves.max()) + 1)) + " | result |")
    print("|---|" + "---|" * (int(waves.max()) + 1) + "---|")
    for k_order in (0, 1, 2):
        c = v9k.matmul(a, bt.T, k_order=k_order).float()
        per_tile = c.reshape(M // 256, 256, N // 256, 256)[:, 0, :, 0].cpu()
        cells, good = [], True
        for w in range(int(waves.max()) + 1):
            vals = per_tile[waves == w].unique().tolist()
            backward = k_order == 2 or (k_order == 1 and w % 2 == 1)
            good &= len(vals) == 1 and (vals[0] > 60) == backward
            cells.append("/".join(f"{v:g}" for v in vals) + (" (backward)" if backward else ""))
        ok &= good
        print(f"| {k_order} | " + " | ".join(cells) + f" | {'ok' if good else 'FAIL'} |", flush=True)
    return ok


def main():
    ok = True
    print("# v9_kreverse correctness (bf16)\n")
    print("| shape | K_ORDER | matches torch | max abs diff vs v9 | result |")
    print("|---|---|---|---|---|")
    for M, N, K in SHAPES:
        torch.manual_seed(0)
        a = torch.rand((M, K), device="cuda", dtype=torch.bfloat16) - 0.5
        b = (torch.rand((N, K), device="cuda", dtype=torch.bfloat16) - 0.5).T
        ref = torch.matmul(a, b)
        c9 = v9.matmul(a, b)
        for k_order in (0, 1, 2):
            c = v9k.matmul(a, b, k_order=k_order)
            torch.cuda.synchronize()
            close = torch.allclose(c, ref, atol=1e-1, rtol=1e-2)
            d9 = (c.float() - c9.float()).abs().max().item()
            good = close and (d9 == 0 if k_order == 0 else True)
            ok &= good
            print(f"| {M}x{N}x{K} | {k_order} | {close} | {d9:.3g} | {'ok' if good else 'FAIL'} |", flush=True)
            del c
        del a, b, ref, c9
        torch.cuda.empty_cache()
    print("\nThe backward orders also match v9 exactly here: these bf16 inputs sum almost exactly in fp32,")
    print("so the order of the k-steps does not show. The probe below makes it show.")
    ok &= order_probe()
    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
