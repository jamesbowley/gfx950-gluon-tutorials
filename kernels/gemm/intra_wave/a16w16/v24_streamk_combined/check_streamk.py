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
"""Stream-K stress check for v24: back-to-back launches on rotating inputs, every result
checked, then the flags and the tile barriers' (and zero counters') counts must all be re-armed.

    cd kernels/gemm/intra_wave/a16w16
    HIP_VISIBLE_DEVICES=0 python v24_streamk_combined/check_streamk.py
    HIP_VISIBLE_DEVICES=0 python v24_streamk_combined/check_streamk.py --stale-count
    HIP_VISIBLE_DEVICES=0 python v24_streamk_combined/check_streamk.py --unsafe-placement

As v23's, over every valid combination of STREAMK_FIXUP (rs, owner, atomic_iz), STREAMK_PARTIALS,
STREAMK_RS_READERS, STREAMK_PLACEMENT (tile, chunk) and STREAMK_EXCHANGE (wt, l2), on four shapes
(SHAPES) with and without bias: the combinations
the host rejects (atomic_iz with partials or l2, l2 with anything but tile-major reduce-scatter,
chunk-major with an end-to-end split) are skipped. The one-tile cases cap the split
(STREAMK_NUM_PROGRAMS = S n), at n = 4 and at an n that leaves uneven tiles (EXTRA > 0), and also
run the default (all programs). bf16 partials and the atomic ending round more than once, so their
256x256 tiles are held to an fp32 reference by their mean error (at most 1 + sqrt(n) times one bf16
rounding's, n = programs per tile); everything else to bench.test_correctness's tolerance.

Negative controls, which must fail:
- --stale-count presets tile 0's barrier count to n - 1 before every launch (one-tile
  reduce-scatter), so its first arriver takes itself for the last and releases the others before
  their partials are written (wrong results or a hang).
- --unsafe-placement runs STREAMK_EXCHANGE=l2 with the balanced placement (STREAMK_XCD_LOCAL=0) and
  the host's locality assertion off, on 3328x5120x8192 (S = 4, 16 programs per tile, 8 per XCD),
  where a tile's programs span two XCDs: a partial or a flag in one XCD's L2 is invisible to the
  other (wrong results or a hang).
Each configuration runs in its own process, because the options are read from the environment.
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
NEGATIVE_SHAPES = [(3328, 5120, 8192)]
# One per property of the tail: S = 16 (even split), 33 (uneven tiles, EXTRA > 0), 4 (many programs
# per tile, 64 at the default) and 128 (2 per tile), all with one full wave or more.
SHAPES = [(4352, 4096, 8192), (4352, 4352, 8192), (3328, 5120, 8192), (4096, 6144, 8192)]
OPTION_KEYS = ("STREAMK_POLICY", "STREAMK_FIXUP", "STREAMK_PARTIALS", "STREAMK_RS_READERS", "STREAMK_PLACEMENT",
               "STREAMK_EXCHANGE", "STREAMK_NUM_PROGRAMS", "STREAMK_XCD_LOCAL", "STREAMK_UNSAFE_PLACEMENT",
               "PERSISTENT_TILE_ORDER")


def leftover(M, N):
    return (M // 256) * (N // 256) % 256


def tile0_programs(M, N, K, snp):
    """Programs on one-tile stream-K tile 0 (the kernel's N_HI if EXTRA > 0, else N_LO)."""
    s = leftover(M, N)
    pairs = K // 128
    n_lo, n_hi = min(snp // s, pairs), min(snp // s + 1, pairs)
    return n_hi if n_hi > n_lo and snp % s else n_lo


def split_counts(M, N, K):
    """The STREAMK_NUM_PROGRAMS values of a shape's one-tile cases: S n at n = 4, an uneven split
    (S n + S / 2, so half the tiles get one more program), and the default (None, all programs)."""
    s = leftover(M, N)
    out = []
    for snp in (4 * s, 4 * s + s // 2):
        if s and 2 * s <= snp <= 256 and snp not in out:
            out.append(snp)
    return out + [None]


def check_shape(M, N, K, bias, launches, stale_count, by_mean, snp):
    import bench_ext as bench  # noqa: F401  (plugin hooks and the path to common.py)
    import torch

    from v24_streamk_combined import matmul_kernel

    dtype = torch.bfloat16
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
    n_max = min((snp or 256) // max(leftover(M, N), 1) + 1, K // 128)
    outs = [torch.empty((M, N), device=bench.DEVICE, dtype=dtype) for _ in range(N_INPUTS)]
    _, locks, tile_sync, _ = matmul_kernel.streamk_workspace(bench.DEVICE, 256, 256, 256)

    bad = 0
    max_err = 0.0
    for i in range(launches):
        a, b, bias_n, ref = inputs[i % N_INPUTS]
        c = outs[i % N_INPUTS]
        c.fill_(float("nan"))
        if stale_count:
            # Tile 0's barrier (first slot 0) thinks all but one of its programs have arrived.
            tile_sync[0] = tile0_programs(M, N, K, snp or 256) - 1
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
    # tile_sync: the barriers' (count, generation) pairs, the L2 epoch flags (which hold epochs),
    # then the atomic ending's two zero counts per tile; the counts must all be zero.
    flags_up = int((locks != 0).sum()) + int((tile_sync[:512:2] != 0).sum()) + int((tile_sync[768:] != 0).sum())
    return bad, flags_up, max_err


def run_case(case, timeout):
    env = dict(os.environ)
    for k in OPTION_KEYS:
        env.pop(k, None)
    env.update(case["env"])
    try:
        proc = subprocess.run(
            [sys.executable, "-W", "ignore", os.path.abspath(__file__), "--worker", json.dumps(case)],
            env=env, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"no result after {timeout} s (hang)"}
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_TAG):
            return json.loads(line[len(RESULT_TAG):])
    tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
    return {"error": " / ".join(tail) or f"exit {proc.returncode}"}


def configurations(args):
    """(env, by_mean) of every valid configuration, without the program count."""
    out = []
    for policy in args.policies:
        if policy != "one_tile":
            for partials in args.partials:
                out.append(({"STREAMK_POLICY": policy, "STREAMK_PARTIALS": partials}, partials != "fp32"))
            continue
        for fixup, partials, readers, placement, exchange in itertools.product(
                args.fixups, args.partials, args.readers, args.placements, args.exchanges):
            if fixup != "rs" and readers != args.readers[0]:
                continue  # only reduce-scatter has readers to choose
            if fixup == "atomic_iz" and (partials != "fp32" or exchange != "wt"):
                continue
            if exchange == "l2" and (fixup != "rs" or placement != "tile"):
                continue
            env = {"STREAMK_POLICY": "one_tile", "STREAMK_FIXUP": fixup, "STREAMK_PARTIALS": partials,
                   "STREAMK_RS_READERS": readers, "STREAMK_PLACEMENT": placement, "STREAMK_EXCHANGE": exchange}
            out.append((env, partials != "fp32" or fixup == "atomic_iz"))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shapes", nargs="*", help="MxNxK shapes (default: SHAPES)")
    parser.add_argument("--launches", type=int, default=30)
    parser.add_argument("--orders", nargs="+", default=["v9"], choices=["v9", "split"])
    parser.add_argument("--bias", nargs="+", type=int, default=[0, 1], choices=[0, 1])
    parser.add_argument("--policies", nargs="+", default=["one_tile", "two_tile"],
                        choices=["dp", "one_tile", "two_tile", "spread", "capped"])
    parser.add_argument("--fixups", nargs="+", default=["rs", "owner", "atomic_iz"], choices=["rs", "owner", "atomic_iz"])
    # bf16 shares bf16_pairs' code paths but its map, which v24 does not change (v23's check covers it).
    parser.add_argument("--partials", nargs="+", default=["fp32", "bf16_pairs"],
                        choices=["fp32", "bf16", "bf16_pairs"])
    parser.add_argument("--readers", nargs="+", default=["pow2", "all"], choices=["pow2", "all"])
    parser.add_argument("--placements", nargs="+", default=["tile", "chunk"], choices=["tile", "chunk"])
    parser.add_argument("--exchanges", nargs="+", default=["wt", "l2"], choices=["wt", "l2"])
    parser.add_argument("--stale-count", action="store_true", help="Preset a barrier count (must FAIL)")
    parser.add_argument("--unsafe-placement", action="store_true", help="l2 with tiles across XCDs (must FAIL)")
    parser.add_argument("--timeout", type=int, default=600, help="Seconds per configuration")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        case = json.loads(args.worker)
        M, N, K = case["shape"]
        bad, flags_up, max_err = check_shape(M, N, K, bool(case["bias"]), case["launches"], case["stale_count"],
                                             case["by_mean"], case["snp"])
        print(RESULT_TAG + json.dumps({"bad": bad, "flags_up": flags_up, "max_err": max_err}), flush=True)
        return

    import regress

    negative = args.stale_count or args.unsafe_placement
    if args.stale_count:
        args.policies, args.fixups, args.exchanges = ["one_tile"], ["rs"], ["wt"]
    if args.unsafe_placement:
        args.policies, args.fixups, args.exchanges, args.placements = ["one_tile"], ["rs"], ["l2"], ["tile"]
    if args.shapes:
        shapes = [regress.parse_shape(s) for s in args.shapes]
    else:
        shapes = NEGATIVE_SHAPES if args.unsafe_placement else SHAPES
    failures = 0
    print("| shape | configuration | STREAMK_NUM_PROGRAMS | tile order | bias | launches | wrong results | "
          "flags or counts left up | largest error |")
    print("|---|---|---|---|---|---|---|---|---|")
    for M, N, K in shapes:
        for env, by_mean in configurations(args):
            if env["STREAMK_POLICY"] != "one_tile":
                snps = [None]
            elif args.unsafe_placement:
                snps = [4 * leftover(M, N)]
            else:
                snps = split_counts(M, N, K)
            for snp, order, bias in itertools.product(snps, args.orders, args.bias):
                case_env = {**env, "PERSISTENT_TILE_ORDER": order}
                if snp is not None:
                    case_env["STREAMK_NUM_PROGRAMS"] = str(snp)
                if args.unsafe_placement:
                    case_env |= {"STREAMK_XCD_LOCAL": "0", "STREAMK_UNSAFE_PLACEMENT": "1"}
                case = {"shape": [M, N, K], "env": case_env, "by_mean": by_mean, "snp": snp, "bias": bias,
                        "launches": args.launches, "stale_count": args.stale_count}
                r = run_case(case, args.timeout)
                if "error" in r:
                    failures += 1
                    res = f"error: {r['error']} | |"
                else:
                    failures += r["bad"] + r["flags_up"]
                    res = f"{r['bad']} | {r['flags_up']} | {r['max_err']:.3f}"
                label = ", ".join(f"{k.removeprefix('STREAMK_')}={v}" for k, v in env.items())
                print(f"| {M}x{N}x{K} | {label} | {snp or 'default'} | {order} | {bias} | {args.launches} | {res} |",
                      flush=True)
    if negative:
        print("\nFAIL, as required (the negative control was detected)" if failures
              else "\nPASS: the negative control did NOT fail, so this check cannot be trusted")
        sys.exit(0 if failures else 1)
    print("\nPASS" if failures == 0 else "\nFAIL")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
