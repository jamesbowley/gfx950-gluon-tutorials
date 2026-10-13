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
"""Stream-K stress check for v23: back-to-back launches on rotating inputs, every result
checked, then the flags and the tile barriers' counts must all be re-armed (zero).

    cd kernels/gemm/intra_wave/a16w16
    HIP_VISIBLE_DEVICES=2 python v23_streamk_rs_band/check_streamk.py
    HIP_VISIBLE_DEVICES=2 python v23_streamk_rs_band/check_streamk.py --partials bf16_pairs --readers pow2
    HIP_VISIBLE_DEVICES=2 python v23_streamk_rs_band/check_streamk.py --shapes 4352x4096x8192 --launches 100
    HIP_VISIBLE_DEVICES=2 python v23_streamk_rs_band/check_streamk.py --stale-count

As v20's, over every STREAMK_PARTIALS value (--partials) and, for the reduce-scatter ending,
STREAMK_RS_READERS (--readers). bf16 partials round every partial once more, which puts a few
elements of 2-way splits outside bench.test_correctness's tolerance, so with bf16 partials each
256x256 tile is held to an fp32 reference by its mean error instead (check_tile_limit.py's test for
the atomic ending: at most 1 + sqrt(n) times that of one bf16 rounding, n = programs per tile); a
missing, doubled or stale partial moves a tile's mean error by about a partial's size. The largest
error over all launches is reported too.

regress.py checks one launch on fresh workspace. A stale flag (never reset, or reset too
late) lets an owner read a partial before this launch's contributor has written it; a stale
barrier count lets a tile's programs leave the barrier before all its partials are written.
With inputs that change every launch those partials are the previous launch's, and C is
wrong. --stale-count presets tile 0's barrier count to n - 1 before every launch (one-tile
policy), so the check must fail: the tile's first arriver takes itself for the last and
releases the others before their partials are written, and later arrivals can wait for a
generation that never comes. Expect wrong results or a hang (reported after --timeout
seconds). A stale count of 1 is not a useful test: the tile's arrivals land so close
together that the early release almost never overtakes the last partial. Each configuration runs in its own process, because the policy and the fixup are read
from the environment. Needs the same stack as regress.py.
"""

import argparse
import itertools
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

N_INPUTS = 3
RESULT_TAG = "CHECK_RESULT "


def tile0_programs(M, N, K):
    """Programs on one-tile stream-K tile 0 (the kernel's N_HI if EXTRA > 0, else N_LO)."""
    s = (M // 256) * (N // 256) % 256
    pairs = K // 128
    n_lo, n_hi = min(256 // s, pairs), min(256 // s + 1, pairs)
    return n_hi if n_hi > n_lo and 256 % s else n_lo


def check_shape(M, N, K, bias, launches, stale_count, partials):
    import bench_ext as bench  # noqa: F401  (plugin hooks and the path to common.py)
    import torch

    from v23_streamk_rs_band import matmul_kernel

    dtype = torch.bfloat16
    by_mean = partials != "fp32"
    inputs = []
    for _ in range(N_INPUTS):
        a = torch.rand((M, K), device=bench.DEVICE, dtype=dtype) - 0.5
        b = torch.rand((N, K), device=bench.DEVICE, dtype=dtype).T - 0.5
        bias_n = torch.rand((N,), device=bench.DEVICE, dtype=dtype) - 0.5 if bias else None
        ref32 = None
        if bias:
            ref32 = torch.matmul(a.float(), b.float()) + bias_n.float()
            ref = ref32.to(dtype)
        else:
            ref = torch.matmul(a, b)
        if by_mean:
            ref32 = ref32 if ref32 is not None else torch.matmul(a.float(), b.float())
            rnd = (ref32.to(dtype).float() - ref32).abs().view(M // 256, 256, N // 256, 256).mean((1, 3))
            ref = (ref32, rnd)
        inputs.append((a, b, bias_n, ref))
    n_max = min(256 // max((M // 256) * (N // 256) % 256, 1) + 1, K // 128)
    outs = [torch.empty((M, N), device=bench.DEVICE, dtype=dtype) for _ in range(N_INPUTS)]
    _, locks, tile_sync = matmul_kernel.streamk_workspace(bench.DEVICE, 256, 256, 256)

    bad = 0
    max_err = 0.0
    for i in range(launches):
        a, b, bias_n, ref = inputs[i % N_INPUTS]
        c = outs[i % N_INPUTS]
        c.fill_(float("nan"))
        if stale_count:
            # Tile 0's barrier (first spid 0) thinks all but one of its programs have arrived.
            tile_sync[0] = tile0_programs(M, N, K) - 1
        matmul_kernel.matmul(a, b, c, bias=bias_n)
        if by_mean:
            ref32, rnd = ref
            err = (c.float() - ref32).abs()
            tile_err = err.view(M // 256, 256, N // 256, 256).mean((1, 3))
            bad += bool(torch.isnan(c).any() or (tile_err > (1 + n_max ** 0.5) * rnd + 1e-3).any())
            max_err = max(max_err, float(err.max()))
        else:
            # Same tolerances as bench.test_correctness.
            if not torch.allclose(c, ref, atol=1e-1, rtol=1e-2):
                bad += 1
            max_err = max(max_err, float((c.float() - ref.float()).abs().max()))
    torch.cuda.synchronize()
    flags_up = int((locks != 0).sum()) + int((tile_sync[0::2] != 0).sum())
    return bad, flags_up, max_err


def run_case(case, timeout):
    env = dict(os.environ)
    env["PERSISTENT_TILE_ORDER"] = case["order"]
    env["STREAMK_POLICY"] = case["policy"]
    env["STREAMK_FIXUP"] = case["fixup"]
    env["STREAMK_PARTIALS"] = case["partials"]
    env["STREAMK_RS_READERS"] = case["readers"]
    try:
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--worker", json.dumps(case)],
            env=env, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"no result after {timeout} s (hang)"}
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_TAG):
            return json.loads(line[len(RESULT_TAG):])
    tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
    return {"error": " / ".join(tail) or f"exit {proc.returncode}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shapes", nargs="*", help="MxNxK shapes (default: regress.STREAMK_SHAPES)")
    parser.add_argument("--launches", type=int, default=30)
    parser.add_argument("--orders", nargs="+", default=["v9", "split"], choices=["v9", "split"])
    parser.add_argument("--bias", nargs="+", type=int, default=[0, 1], choices=[0, 1])
    parser.add_argument(
        "--policies", nargs="+", default=["one_tile", "two_tile"],
        choices=["dp", "one_tile", "two_tile", "spread", "auto", "capped", "split_rule"]
    )
    parser.add_argument("--fixups", nargs="+", default=["rs", "owner"], choices=["rs", "owner", "auto"])
    parser.add_argument("--partials", nargs="+", default=["fp32", "bf16", "bf16_pairs"],
                        choices=["fp32", "bf16", "bf16_pairs"])
    parser.add_argument("--readers", nargs="+", default=["pow2", "all"], choices=["pow2", "all"],
                        help="STREAMK_RS_READERS values for the reduce-scatter ending")
    parser.add_argument("--stale-count", action="store_true", help="Preset a barrier count (must FAIL)")
    parser.add_argument("--timeout", type=int, default=1800, help="Seconds per configuration")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        case = json.loads(args.worker)
        M, N, K = case["shape"]
        bad, flags_up, max_err = check_shape(M, N, K, bool(case["bias"]), case["launches"], case["stale_count"],
                                             case["partials"])
        print(RESULT_TAG + json.dumps({"bad": bad, "flags_up": flags_up, "max_err": max_err}), flush=True)
        return

    import regress

    shapes = [regress.parse_shape(s) for s in args.shapes] if args.shapes else regress.STREAMK_SHAPES
    failures = 0
    print("| shape | policy | fixup | partials | readers | tile order | bias | launches | wrong results | "
          "flags or counts left up | largest error |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for M, N, K in shapes:
        for policy in args.policies:
            # The fixup only applies to the tile-aligned one-tile path; capped and split_rule pick their own.
            fixups = {"two_tile": ["rs"], "spread": ["rs"], "capped": ["auto"], "split_rule": ["auto"]}.get(policy, args.fixups)
            for fixup in fixups:
                # Only the one-tile reduce-scatter ending has readers to choose.
                readers = args.readers if (policy, fixup) == ("one_tile", "rs") else args.readers[:1]
                for partials, rd, order, bias in itertools.product(args.partials, readers, args.orders, args.bias):
                    case = {
                        "shape": [M, N, K], "policy": policy, "fixup": fixup, "partials": partials,
                        "readers": rd, "order": order, "bias": bias, "launches": args.launches,
                        "stale_count": args.stale_count,
                    }
                    r = run_case(case, args.timeout)
                    if "error" in r:
                        failures += 1
                        res = f"error: {r['error']} | |"
                    else:
                        failures += r["bad"] + r["flags_up"]
                        res = f"{r['bad']} | {r['flags_up']} | {r['max_err']:.3f}"
                    print(
                        f"| {M}x{N}x{K} | {policy} | {fixup} | {partials} | {rd} | {order} | {bias} | "
                        f"{args.launches} | {res} |",
                        flush=True,
                    )
    print("\nPASS" if failures == 0 else "\nFAIL")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
