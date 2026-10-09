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
"""How accurate is v22's atomic ending? Errors of the stream-K tiles against an fp32 reference.

    HIP_VISIBLE_DEVICES=0 python experiments/streamk_split_count/atomic_accuracy.py > \
        experiments/streamk_split_count/results/atomic_accuracy.md
    HIP_VISIBLE_DEVICES=0 python experiments/streamk_split_count/atomic_accuracy.py --v23 > \
        experiments/streamk_split_count/results/atomic_accuracy_v23.md

On sweep C's matrix (4096x8192xK, GEMM_TILE_LIMIT = 256 + S), bf16 inputs uniform in [-0.5, 0.5),
bf16 C. Only the S stream-K tiles are compared (check_tile_limit.streamk_tiles). For each case:
- data-parallel (v13, one rounding of the fp32 sum), v20's reduce-scatter (one rounding) and v22's
  atomic ending (each of the n programs adds its partial into bf16 C, so n roundings);
- the largest and mean absolute error, the mean error against that of one bf16 rounding of the
  reference, and the share of elements outside the repository's check tolerance (atol 0.1,
  rtol 0.01);
- for the atomic ending, the share of elements that differ bitwise between two launches on the
  same inputs (the order of the atomic additions is not fixed).
--v23 adds v23's bf16 partials (STREAMK_PARTIALS=bf16 and bf16_pairs) with both endings: each
partial is rounded to bf16 once, summed in fp32 and rounded once more. For bf16 pairs, the share of
elements that differ bitwise from bf16 in the same ending (the same values in another map, so 0).
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
A16 = os.path.join(HERE, "..", "..", "kernels", "gemm", "intra_wave", "a16w16")
sys.path.insert(0, A16)
sys.path.insert(0, HERE)
os.chdir(A16)

import bench  # noqa: E402  (plugin hooks and the path to common.py)
import torch  # noqa: E402

import sweep  # noqa: E402
from check_tile_limit import streamk_tiles  # noqa: E402
from v13_persistant_peel_acc.matmul_kernel import matmul as v13  # noqa: E402
from v20_streamk_reduce_scatter.matmul_kernel import matmul as v20  # noqa: E402
from v22_streamk_atomic_c.matmul_kernel import matmul as v22  # noqa: E402
from v23_streamk_rs_band.matmul_kernel import matmul as v23  # noqa: E402

S_LIST = (1, 8, 32, 128)
K_LIST = (2048, 4096, 8192, 16384)
KEYS = ("STREAMK_POLICY", "STREAMK_FIXUP", "STREAMK_NUM_PROGRAMS", "GEMM_TILE_LIMIT", "STREAMK_PARTIALS",
        "STREAMK_RS_READERS")


def setenv(**kv):
    for k in KEYS:
        os.environ.pop(k, None)
    os.environ.update({k: str(v) for k, v in kv.items() if v is not None})


def run(fn, a, b, **env):
    setenv(**env)
    c = bench.make_c(a.shape[0], b.shape[1], torch.bfloat16, True)
    c.fill_(float("nan"))
    fn(a, b, c)
    torch.cuda.synchronize()
    return c


def stats(c, ref, mask):
    x, r = c.float()[mask], ref[mask]
    err = (x - r).abs()
    rnd = (r.to(torch.bfloat16).float() - r).abs().mean()  # one bf16 rounding of the reference
    bad = err > 0.1 + 0.01 * r.abs()
    return dict(max=float(err.max()), mean=float(err.mean()), rel=float(err.mean() / rnd),
                bad=float(bad.float().mean()), nan=int(torch.isnan(x).sum()))


def main():
    with_v23 = "--v23" in sys.argv
    M, N = sweep.MATRIX_C
    print("# Accuracy of the atomic ending (v22)\n")
    print("Stream-K tiles of 4096x8192xK with GEMM_TILE_LIMIT = 256 + S against an fp32 reference: largest and mean "
          "absolute error, the mean error over that of one bf16 rounding of the reference, the share of elements "
          "outside atol 0.1 + rtol 0.01 (the repository's check), and for the atomic ending the share of elements "
          "that differ bitwise between two launches on the same inputs. n = d is the default split (256 // S "
          "per tile).\n")
    print("| S | K | config | n | max err | mean err | mean err / one rounding | outside tolerance | "
          "run-to-run differing (atomic), or differing from bf16 (pairs) |")
    print("|---|---|---|---|---|---|---|---|---|")
    tm, tn = M // 256, N // 256
    for K in K_LIST:
        torch.manual_seed(0)
        a = torch.rand((M, K), device=bench.DEVICE, dtype=torch.bfloat16) - 0.5
        b = bench.make_b(N, K, torch.bfloat16, True, lambda t: t.uniform_(-0.5, 0.5))
        ref = a.float() @ b.float()
        for S in S_LIST:
            T = 256 + S
            sk = torch.tensor(streamk_tiles(T, S, tm, tn), device=bench.DEVICE)
            mask = sk.repeat_interleave(256, 0).repeat_interleave(256, 1)
            c = run(v13, a, b, GEMM_TILE_LIMIT=T)
            s = stats(c, ref, mask)
            print(f"| {S} | {K} | data-parallel (v13) | - | {s['max']:.3f} | {s['mean']:.4f} | {s['rel']:.2f} | "
                  f"{100 * s['bad']:.3f}% | - |", flush=True)
            for n in [x for x in sweep.splits(S, K, sweep.N_C) if x >= 2] + ["default"]:
                snp = None if n == "default" else S * n
                configs = [("reduce-scatter (v20)", v20, "rs", None), ("atomic (v22)", v22, "atomic", None)]
                if with_v23:
                    configs += [(f"{fx}, {p} partials (v23)", v23, fx, p) for fx in ("rs", "owner")
                                for p in ("bf16", "bf16_pairs")]
                prev = {}
                for label, fn, fixup, partials in configs:
                    env = dict(STREAMK_POLICY="one_tile", STREAMK_FIXUP=fixup, STREAMK_NUM_PROGRAMS=snp,
                               GEMM_TILE_LIMIT=T, STREAMK_PARTIALS=partials)
                    c = run(fn, a, b, **env)
                    s = stats(c, ref, mask)
                    diff = "-"
                    if fixup == "atomic":
                        c2 = run(fn, a, b, **env)
                        same = (c.view(torch.int16) == c2.view(torch.int16))[mask]
                        diff = f"{100 * (1 - same.float().mean()):.2f}%"
                    elif partials == "bf16":
                        prev[fixup] = c
                    elif partials == "bf16_pairs":
                        same = (c.view(torch.int16) == prev[fixup].view(torch.int16))[mask]
                        diff = f"{100 * (1 - same.float().mean()):.2f}% vs bf16"
                    print(f"| {S} | {K} | {label} | {'d' if n == 'default' else n} | {s['max']:.3f} | "
                          f"{s['mean']:.4f} | {s['rel']:.2f} | {100 * s['bad']:.3f}% | {diff} |", flush=True)


if __name__ == "__main__":
    main()
