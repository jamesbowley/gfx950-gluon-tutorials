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
"""Stream-K stress check for v29, per tile: back-to-back launches on rotating inputs, every result checked,
then every owner flag and barrier count must be re-armed (zero).

    source v29_streamk_compact/env.sh
    python v29_streamk_compact/check_streamk.py --tile 128x256
    python v29_streamk_compact/check_streamk.py --tile 16x256x128          # the small set
    python v29_streamk_compact/check_streamk.py --tile 128x128x128 --shapes 4300x4000x8176
    python v29_streamk_compact/check_streamk.py --tile 256x128 --stale-count
    python v29_streamk_compact/check_streamk.py --tile 128x256 --no-full     # where the compact launch applies

C is a window of a larger buffer (256 more rows and columns) filled with a sentinel, and a launch also fails if it
writes anything outside the window. Results are held to bench.test_correctness's tolerance (atol 0.1, rtol 0.01).

Configurations: one_tile with every STREAMK_FIXUP (rs, owner) and STREAMK_EXCHANGE (wt, l2; l2 with rs only),
two_tile and spread, each with and without bias. The one-tile cases cap the split (STREAMK_NUM_PROGRAMS = S n) at
n = 4 and at an n that leaves uneven tiles (EXTRA > 0), and also run the default (all programs).

Shapes (default):
- quadrant tiles: S = 16, 33, 4 and 128 after a full wave of 256 tiles, plus two ragged ones (fewer tiles than a
  wave with M, N and K ragged; S = 16 with all three ragged);
- small tiles: SMALL_SHAPES (M = 1-32, N = 4000-8192, K = 4080-8192, every tile stream-K), plus a full wave of
  small tiles with ragged edges; mirrored (M and N swapped) for a swapped tile.
- --no-full: shapes with no full tile, so the one-tile and spread splits on fewer programs than the tile's
  NUM_PROGRAMS take the compact launch (spread then also runs at the one-tile program counts).

Negative control, which must fail: --stale-count presets tile 0's barrier count to n - 1 before every launch
(one-tile reduce-scatter), so its first arriver takes itself for the last and releases the others before their
partials are written (wrong results or a hang).

Each configuration runs in its own process, because the options are read from the environment. --jobs processes
run at once: each compiles its kernel without launching it (matmul_kernel.WARMUP), then takes the GPU's lock (a
file lock per HIP_VISIBLE_DEVICES, shared with v27's and v28's checks) for its launches, so no two stream-K kernels
ever share the GPU (their programs could not all be resident).
"""

import argparse
import fcntl
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

N_INPUTS = 3
RESULT_TAG = "CHECK_RESULT "
# (rows, columns) of tiles: S = 16 (even split), 33 (uneven tiles, EXTRA > 0), 4 (many programs per tile) and
# 128 (2 per tile), all with one full wave or more of 256 programs.
QUADRANT_GRIDS = [(17, 16), (17, 17), (13, 20), (16, 24)]
SMALL_SHAPES = [(16, 4096, 8192), (30, 4000, 4080), (1, 8192, 4096), (32, 4000, 8192)]
OPTION_KEYS = ("STREAMK_POLICY", "STREAMK_FIXUP", "STREAMK_EXCHANGE", "STREAMK_NUM_PROGRAMS", "STREAMK_TILE")


def kernel_module():
    import bench_ext  # noqa: F401  (paths)
    import importlib

    return importlib.import_module("v29_streamk_compact.matmul_kernel")


def quadrant_shapes(geo):
    bm, bn = geo["block_m"], geo["block_n"]
    shapes = [(mt * bm, nt * bn, 8192) for mt, nt in QUADRANT_GRIDS]
    # 1000x2000x8080 (32 tiles, all stream-K) and 4300x4000x8176 (S = 16) at 256x256, scaled to the tile.
    return shapes + [(4 * bm - bm * 3 // 32, 8 * bn - bn * 3 // 16, 8080),
                     (17 * bm - bm * 13 // 64, 16 * bn - bn * 3 // 8, 8176)]


def small_shapes(geo):
    """SMALL_SHAPES, plus a full wave of small tiles and S = 16 more with ragged M, N and K."""
    km, kn, np_ = geo["block_m"], geo["block_n"], geo["num_programs"]
    cols = 4096 // kn
    rows = -(-(np_ + 16) // cols)
    full = (rows * km - km // 2 if km > 1 else rows, cols * kn - kn // 4, 8080)
    shapes = SMALL_SHAPES + [full]
    return [(n, m, k) for m, n, k in shapes] if geo["swap"] else shapes


def kernel_dims(geo, M, N):
    return (N, M) if geo["swap"] else (M, N)


def grid_shapes(geo):
    """Shapes with fewer tiles than the tile's programs (no full tile): the compact launch's cases."""
    bm, bn, bk, np_ = geo["block_m"], geo["block_n"], geo["block_k"], geo["num_programs"]
    if geo["small"]:
        shapes = list(SMALL_SHAPES) + [(3, 4000, 4080), (1, 8192, 4 * bk)]
        shapes = [(n, m, k) for m, n, k in shapes] if geo["swap"] else shapes
    else:
        shapes = [(3 * bm, 11 * bn, 8192), quadrant_shapes(geo)[4], (3 * bm, 11 * bn, 8 * bk)]
    out = []
    for M, N, K in shapes:
        m, n = kernel_dims(geo, M, N)
        tiles = -(-m // bm) * -(-n // bn)
        if 0 < tiles < np_:
            out.append((M, N, K))
    return out


def leftover(geo, M, N):
    m, n = kernel_dims(geo, M, N)
    return (-(-m // geo["block_m"])) * (-(-n // geo["block_n"])) % geo["num_programs"]


def tile0_programs(geo, M, N, K, snp):
    """Programs on one-tile stream-K tile 0 (the kernel's N_HI if EXTRA > 0, else N_LO)."""
    s = leftover(geo, M, N)
    pairs = -(-K // (2 * geo["block_k"]))
    n_lo, n_hi = min(snp // s, pairs), min(snp // s + 1, pairs)
    return n_hi if n_hi > n_lo and snp % s else n_lo


def split_counts(geo, M, N, K):
    """The STREAMK_NUM_PROGRAMS values of a shape's one-tile cases: S n at n = 4, an uneven split (S n + S / 2,
    so half the tiles get one more program), and the default (None, all programs)."""
    s = leftover(geo, M, N)
    out = []
    for snp in (4 * s, 4 * s + s // 2):
        if s and 2 * s <= snp <= geo["num_programs"] and snp not in out:
            out.append(snp)
    return out + [None]


def check_shape(tile, M, N, K, bias, launches, stale_count, snp):
    import bench_ext as bench
    import torch

    mk = kernel_module()
    geo = mk.tile_geometry(tile)
    np_ = geo["num_programs"]
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
    # Compile without launching (in parallel with the other jobs), then hold the GPU for the launches.
    mk.WARMUP = True
    mk.matmul(inputs[0][0], inputs[0][1], bias=inputs[0][2], tile=tile)
    mk.WARMUP = False
    torch.cuda.synchronize()
    lock = open(f"/tmp/v27_check_gpu{os.environ.get('HIP_VISIBLE_DEVICES', 'all')}.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    # C windows of guard buffers: 256 more rows and columns, all holding the sentinel.
    SENTINEL = 1234.0
    bufs = [torch.empty((M + 256, N + 256), device=bench.DEVICE, dtype=dtype) for _ in range(N_INPUTS)]
    outs = [buf[:M, :N] for buf in bufs]
    guard = torch.ones((M + 256, N + 256), device=bench.DEVICE, dtype=torch.bool)
    guard[:M, :N] = False
    _, locks, tile_sync, _ = mk.streamk_workspace(bench.DEVICE, np_, *geo["slot"])

    bad = 0
    max_err = 0.0
    for i in range(launches):
        a, b, bias_n, ref = inputs[i % N_INPUTS]
        c = outs[i % N_INPUTS]
        buf = bufs[i % N_INPUTS]
        buf.fill_(SENTINEL)
        c.fill_(float("nan"))
        if stale_count:
            # Tile 0's barrier (first slot 0) thinks all but one of its programs have arrived.
            tile_sync[0] = tile0_programs(geo, M, N, K, snp or np_) - 1
        mk.matmul(a, b, c, bias=bias_n, tile=tile)
        if not torch.allclose(c, ref, atol=1e-1, rtol=1e-2):
            bad += 1
        max_err = max(max_err, float((c.float() - ref.float()).abs().max()))
        bad += bool((buf[guard] != SENTINEL).any())
    torch.cuda.synchronize()
    # tile_sync: the barriers' (count, generation) pairs, then the l2 epoch flags (which hold epochs); the owner
    # flags and the counts must all be zero.
    flags_up = int((locks != 0).sum()) + int((tile_sync[:2 * np_:2] != 0).sum())
    fcntl.flock(lock, fcntl.LOCK_UN)
    kernel = mk.LAST_LAUNCH.get("kernel", "?")
    grid = mk.LAST_LAUNCH.get("grid")
    return bad, flags_up, max_err, kernel + (f" (grid {grid})" if grid and grid != np_ else "")


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
    """The env of every valid configuration, without the program count."""
    out = []
    for policy in args.policies:
        if policy != "one_tile":
            out.append({"STREAMK_POLICY": policy})
            continue
        for fixup, exchange in itertools.product(args.fixups, args.exchanges):
            if exchange == "l2" and fixup != "rs":
                continue
            out.append({"STREAMK_POLICY": "one_tile", "STREAMK_FIXUP": fixup, "STREAMK_EXCHANGE": exchange})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tile", default="256x256x64", help="v9_tile spec (STREAMK_TILE); default 256x256x64")
    parser.add_argument("--shapes", nargs="*", help="MxNxK shapes (default: the tile's set)")
    parser.add_argument("--launches", type=int, default=10)
    parser.add_argument("--bias", nargs="+", type=int, default=[0, 1], choices=[0, 1])
    parser.add_argument("--policies", nargs="+", default=["one_tile", "two_tile", "spread"],
                        choices=["dp", "one_tile", "two_tile", "spread", "capped"])
    parser.add_argument("--fixups", nargs="+", default=["rs", "owner"], choices=["rs", "owner"])
    parser.add_argument("--exchanges", nargs="+", default=["wt", "l2"], choices=["wt", "l2"])
    parser.add_argument("--stale-count", action="store_true", help="Preset a barrier count (must FAIL)")
    parser.add_argument("--no-full", action="store_true",
                        help="only shapes with no full tiles, where the compact launch applies; spread also at the "
                             "one-tile program counts")
    parser.add_argument("--timeout", type=int, default=600, help="Seconds per configuration")
    parser.add_argument("--jobs", type=int, default=8, help="Configurations compiled at once")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        case = json.loads(args.worker)
        M, N, K = case["shape"]
        bad, flags_up, max_err, kernel = check_shape(case["tile"], M, N, K, bool(case["bias"]), case["launches"],
                                                     case["stale_count"], case["snp"])
        print(RESULT_TAG + json.dumps({"bad": bad, "flags_up": flags_up, "max_err": max_err, "kernel": kernel}),
              flush=True)
        return

    import regress

    if args.stale_count:
        args.policies, args.fixups, args.exchanges = ["one_tile"], ["rs"], ["wt"]
    geo = kernel_module().tile_geometry(args.tile)
    if args.shapes:
        shapes = [regress.parse_shape(s) for s in args.shapes]
    elif args.no_full:
        shapes = grid_shapes(geo)
    else:
        shapes = small_shapes(geo) if geo["small"] else quadrant_shapes(geo)
    failures = 0
    print(f"tile {args.tile}: kernel BLOCK {geo['block_m']}x{geo['block_n']}x{geo['block_k']}, {geo['w']} waves, "
          f"{geo['num_programs']} programs{', swapped (C^T = B^T A^T)' if geo['swap'] else ''}\n")
    print("| shape | configuration | STREAMK_NUM_PROGRAMS | bias | kernel | launches | wrong results | "
          "flags or counts left up | largest error |")
    print("|---|---|---|---|---|---|---|---|---|")
    cases = []
    for M, N, K in shapes:
        for env in configurations(args):
            if env["STREAMK_POLICY"] == "one_tile" or (env["STREAMK_POLICY"] == "spread" and args.no_full):
                snps = split_counts(geo, M, N, K)
            else:
                snps = [None]
            for snp, bias in itertools.product(snps, args.bias):
                case_env = {**env, "STREAMK_TILE": args.tile}
                if snp is not None:
                    case_env["STREAMK_NUM_PROGRAMS"] = str(snp)
                cases.append(({"tile": args.tile, "shape": [M, N, K], "env": case_env, "snp": snp, "bias": bias,
                               "launches": args.launches, "stale_count": args.stale_count}, env))
    with ThreadPoolExecutor(1 if args.stale_count else args.jobs) as ex:
        for (case, env), r in zip(cases, ex.map(lambda ce: run_case(ce[0], args.timeout), cases)):
            M, N, K = case["shape"]
            if "error" in r:
                failures += 1
                res = f"| {args.launches} | error: {r['error']} | |"
            else:
                failures += r["bad"] + r["flags_up"]
                res = f"{r['kernel']} | {args.launches} | {r['bad']} | {r['flags_up']} | {r['max_err']:.3f}"
            label = ", ".join(f"{k.removeprefix('STREAMK_')}={v}" for k, v in env.items())
            print(f"| {M}x{N}x{K} | {label} | {case['snp'] or 'default'} | {case['bias']} | {res} |", flush=True)
    if args.stale_count:
        print("\nFAIL, as required (the negative control was detected)" if failures
              else "\nPASS: the negative control did NOT fail, so this check cannot be trusted")
        sys.exit(0 if failures else 1)
    print("\nPASS" if failures == 0 else "\nFAIL")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
