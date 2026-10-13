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
"""Stream-K stress check for v21: back-to-back launches on rotating inputs, every result
checked, per STREAMK_PARTIALS mode.

    cd kernels/gemm/intra_wave/a16w16
    HIP_VISIBLE_DEVICES=2 python v21_streamk_l2_partials/check_streamk.py
    HIP_VISIBLE_DEVICES=2 python v21_streamk_l2_partials/check_streamk.py --partials l2 --launches 200
    HIP_VISIBLE_DEVICES=2 python v21_streamk_l2_partials/check_streamk.py --unsafe-placement

After the launches every mode must leave the atomic flags and the barrier counts at zero, and
in l2 mode every reduce-scatter program's L2 flag must hold the last launch's epoch. One extra
launch with STREAMK_RECORD_XCC=1 records the hardware XCD (HW_REG_XCC_ID) of every partial
slot, and the "tiles on 2+ XCDs" column counts reduce-scatter tiles whose programs ran on more
than one XCD. In l2 mode it must be 0.

--unsafe-placement is the negative control. It runs STREAMK_PARTIALS=l2 with v20's balanced
placement (STREAMK_XCD_LOCAL=0) and the host's locality assertion off, on shapes where that
placement puts a tile's programs on two XCDs (3328x5120x8192: S = 4, 16 programs per tile, 8
per XCD). A partial or a flag written into one XCD's L2 is then invisible to the other XCD,
so the check must FAIL: wrong results, or a hang (the flag never arrives), reported after
--timeout seconds.

Each configuration runs in its own process, because the modes are read from the environment.
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

N_INPUTS = 3
RESULT_TAG = "CHECK_RESULT "
PLAN_SHAPES = [(3072, 5632, 8192), (4352, 4096, 8192), (4096, 4608, 8192), (3328, 5120, 8192)]
NEGATIVE_SHAPES = [(3328, 5120, 8192), (1536, 11008, 8192)]


def reduce_scatter(M, N, K, mk):
    """Whether the host picks the tile-aligned reduce-scatter ending (the only one with L2
    flags) under the environment's policy and fixup."""
    total = -(-M // 256) * -(-N // 256)
    policy, fixup = os.environ.get("STREAMK_POLICY", "capped"), os.environ.get("STREAMK_FIXUP", "auto")
    if policy == "capped":
        policy, rule_fixup, _ = mk.capped_policy(total, K // 64, 256)
    else:
        auto_policy, rule_fixup = mk.streamk_policy(total, K // 64, 256)
        policy = auto_policy if policy == "auto" else policy
    fixup = rule_fixup if fixup == "auto" else fixup
    return policy == "one_tile" and fixup == "rs" and total % 256 > 0


def check_shape(M, N, K, bias, launches):
    import bench_ext as bench  # noqa: F401  (plugin hooks, the path to common.py; imports triton before torch)
    import torch

    from v21_streamk_l2_partials import matmul_kernel

    dtype = torch.bfloat16
    inputs = []
    for _ in range(N_INPUTS):
        a = torch.rand((M, K), device=bench.DEVICE, dtype=dtype) - 0.5
        b = torch.rand((N, K), device=bench.DEVICE, dtype=dtype).T - 0.5
        bias_n = torch.rand((N,), device=bench.DEVICE, dtype=dtype) - 0.5 if bias else None
        if bias:
            ref = (torch.matmul(a.float(), b.float()) + bias_n.float()).to(dtype)
        else:
            ref = torch.matmul(a, b)
        inputs.append((a, b, bias_n, ref))
    outs = [torch.empty((M, N), device=bench.DEVICE, dtype=dtype) for _ in range(N_INPUTS)]
    P = 256
    _, locks, tile_sync, epochs = matmul_kernel.streamk_workspace(bench.DEVICE, P, 256, 256)

    bad = 0
    for i in range(launches):
        a, b, bias_n, ref = inputs[i % N_INPUTS]
        c = outs[i % N_INPUTS]
        c.fill_(float("nan"))
        matmul_kernel.matmul(a, b, c, bias=bias_n)
        # Same tolerances as bench.test_correctness.
        if not torch.allclose(c, ref, atol=1e-1, rtol=1e-2):
            bad += 1
    torch.cuda.synchronize()
    flags_up = int((locks != 0).sum()) + int((tile_sync[0:2 * P:2] != 0).sum())
    # L2 flags: every slot that took part in the last launch holds its epoch; others hold an
    # older epoch or 0, never a newer one.
    l2_flags = tile_sync[2 * P:3 * P]
    stale = int((l2_flags > epochs[0]).sum())

    # One recorded launch: (tile + 1, XCC id) per partial slot.
    tile_sync[3 * P:5 * P].zero_()
    os.environ["STREAMK_RECORD_XCC"] = "1"
    a, b, bias_n, ref = inputs[0]
    matmul_kernel.matmul(a, b, outs[0], bias=bias_n)
    torch.cuda.synchronize()
    os.environ.pop("STREAMK_RECORD_XCC")
    rec = tile_sync[3 * P:5 * P].view(P, 2).cpu().tolist()
    by_tile = {}
    for t1, x in rec:
        if t1:
            by_tile.setdefault(t1 - 1, set()).add(x)
    split = {t: xs for t, xs in by_tile.items() if sum(r[0] == t + 1 for r in rec) > 1}
    straddling = sum(len(xs) > 1 for xs in split.values())
    xcds = len({x for xs in by_tile.values() for x in xs})
    if os.environ.get("STREAMK_PARTIALS") == "l2" and reduce_scatter(M, N, K, matmul_kernel):
        for slot, (t1, _) in enumerate(rec):
            if t1 and sum(r[0] == t1 for r in rec) > 1 and int(l2_flags[slot]) != epochs[0]:
                stale += 1
    return {"bad": bad, "flags_up": flags_up, "stale_l2_flags": stale, "straddling": straddling,
            "split_tiles": len(split), "xcds": xcds}


def run_case(case, timeout):
    env = dict(os.environ)
    env["STREAMK_POLICY"] = case["policy"]
    env["STREAMK_FIXUP"] = case["fixup"]
    env["STREAMK_PARTIALS"] = case["partials"]
    env["PERSISTENT_TILE_ORDER"] = case["order"]
    if case["unsafe"]:
        env["STREAMK_XCD_LOCAL"] = "0"
        env["STREAMK_UNSAFE_PLACEMENT"] = "1"
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
    parser.add_argument("--shapes", nargs="*", help="MxNxK shapes (default: the plan's four, then regress.STREAMK_SHAPES)")
    parser.add_argument("--launches", type=int, default=30)
    parser.add_argument("--partials", nargs="+", default=["mem", "wt", "l2"], choices=["mem", "wt", "l2"])
    parser.add_argument("--policies", nargs="+", default=["capped"],
                        choices=["dp", "one_tile", "two_tile", "spread", "auto", "capped"])
    parser.add_argument("--fixups", nargs="+", default=["rs"], choices=["rs", "owner", "auto"])
    parser.add_argument("--orders", nargs="+", default=["v9"], choices=["v9", "split"])
    parser.add_argument("--bias", nargs="+", type=int, default=[0, 1], choices=[0, 1])
    parser.add_argument("--unsafe-placement", action="store_true", help="Negative control (must FAIL)")
    parser.add_argument("--timeout", type=int, default=1800, help="Seconds per configuration")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        case = json.loads(args.worker)
        M, N, K = case["shape"]
        r = check_shape(M, N, K, bool(case["bias"]), case["launches"])
        print(RESULT_TAG + json.dumps(r), flush=True)
        return

    import regress

    if args.shapes:
        shapes = [regress.parse_shape(s) for s in args.shapes]
    elif args.unsafe_placement:
        shapes = NEGATIVE_SHAPES
    else:
        shapes = PLAN_SHAPES + [s for s in regress.STREAMK_SHAPES if s not in PLAN_SHAPES]
    partials = ["l2"] if args.unsafe_placement else args.partials
    failures = 0
    print("| shape | policy | fixup | partials | order | bias | launches | wrong results | atomic flags or counts "
          "left up | stale L2 flags | split tiles | tiles on 2+ XCDs | XCDs used |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for M, N, K in shapes:
        for policy in args.policies:
            fixups = {"two_tile": ["rs"], "spread": ["rs"], "capped": ["auto"]}.get(policy, args.fixups)
            for fixup in fixups:
                for p in partials:
                    for order in args.orders:
                        for bias in args.bias:
                            case = {"shape": [M, N, K], "policy": policy, "fixup": fixup, "partials": p,
                                    "order": order, "bias": bias, "launches": args.launches,
                                    "unsafe": args.unsafe_placement}
                            r = run_case(case, args.timeout)
                            if "error" in r:
                                failures += 1
                                res = f"error: {r['error']} | | | | |"
                            else:
                                failures += r["bad"] + r["flags_up"] + r["stale_l2_flags"]
                                if p == "l2" and not args.unsafe_placement:
                                    failures += r["straddling"]
                                res = (f"{r['bad']} | {r['flags_up']} | {r['stale_l2_flags']} | {r['split_tiles']} | "
                                       f"{r['straddling']} | {r['xcds']}")
                            print(f"| {M}x{N}x{K} | {policy} | {fixup} | {p} | {order} | {bias} | {args.launches} | "
                                  f"{res} |", flush=True)
    if args.unsafe_placement:
        print("\nFAIL, as required (the negative control detected the straddling placement)" if failures
              else "\nPASS: the negative control did NOT fail, so this check cannot be trusted")
        sys.exit(0 if failures else 1)
    print("\nPASS" if failures == 0 else "\nFAIL")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
