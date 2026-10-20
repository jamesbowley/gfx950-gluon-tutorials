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
"""Stream-K stress check for v27, per tile: back-to-back launches on rotating inputs, every result
checked, then the flags and the tile barriers' (and zero counters') counts must all be re-armed.

    source v28_streamk_owner/env.sh
    python v28_streamk_owner/check_streamk.py --tile 128x256
    python v28_streamk_owner/check_streamk.py --tile 16x256x128          # the small set
    python v28_streamk_owner/check_streamk.py --tile 128x128x128 --shapes 4300x4000x8176
    python v28_streamk_owner/check_streamk.py --tile 256x128 --stale-count
    python v28_streamk_owner/check_streamk.py --unsafe-placement
    python v28_streamk_owner/check_streamk.py --tile 128x256 --no-full     # where the compact launch applies

v26's check with --tile (STREAMK_TILE in every worker). C is a window of a larger buffer (256 more rows
and columns) filled with a sentinel, and a launch also fails if it writes anything outside the window.

Over every valid combination of STREAMK_FIXUP (rs, owner, atomic_iz), STREAMK_PARTIALS,
STREAMK_RS_READERS, STREAMK_PLACEMENT (tile, chunk) and STREAMK_EXCHANGE (wt, l2), with and without
bias: the combinations the host rejects (atomic_iz with partials or l2, l2 with anything but
tile-major reduce-scatter, chunk-major with an end-to-end split, bf16_pairs on a small tile or BN = 64,
atomic_iz with an odd N, whose packed pairs cannot be aligned) are skipped. The one-tile cases cap the split (STREAMK_NUM_PROGRAMS = S n), at n = 4 and at an n that
leaves uneven tiles (EXTRA > 0), and also run the default (all programs). bf16 partials and the atomic
ending round more than once, so their tiles are held to an fp32 reference by their mean error (at most
1 + sqrt(n) times one bf16 rounding's, n = programs per tile), over the tile widened along its long side to
at least 1024 in-bounds elements; everything else to bench.test_correctness's tolerance.

Shapes (default):
- quadrant tiles: v26's four S cases scaled to the tile (quadrant_shapes: S = 16, 33, 4 and 128 after
  a full wave of 256 tiles), plus the two ragged ones (fewer tiles than a wave with M, N and K ragged;
  S = 16 with all three ragged);
- small tiles: SMALL_SHAPES (M = 1-32, N = 4000-8192, K = 4080-8192, every tile stream-K), plus one
  with a full wave of small tiles and ragged edges; mirrored (M and N swapped) for a swapped tile.
- --no-full: check_grid.py's shapes, none with a full tile, so the one-tile and spread splits on fewer
  programs than the tile's NUM_PROGRAMS take the compact launch (spread then also runs at the one-tile
  program counts). The kernel column shows the launched grid when it is not NUM_PROGRAMS.
- --persistent-programs P: STREAMK_PERSISTENT_PROGRAMS=P with STREAMK_POLICY=dp (fp32) on the quadrant shapes,
  plus 1.5 waves of P programs aligned and about 2.2 waves with M, N and K ragged (persistent_shapes).

Negative controls, which must fail:
- --stale-count presets tile 0's barrier count to n - 1 before every launch (one-tile
  reduce-scatter), so its first arriver takes itself for the last and releases the others before
  their partials are written (wrong results or a hang).
- --unsafe-placement runs STREAMK_EXCHANGE=l2 with the balanced placement (STREAMK_XCD_LOCAL=0) and
  the host's locality assertion off, on 3328x5120x8192 at 256x256 (S = 4, 16 programs per tile, 8 per
  XCD), where a tile's programs span two XCDs: a partial or a flag in one XCD's L2 is invisible to the
  other (wrong results or a hang).
Each configuration runs in its own process, because the options are read from the environment. --jobs
processes run at once: each compiles its kernel without launching it (matmul_kernel.WARMUP), then takes the
GPU's lock (a file lock per HIP_VISIBLE_DEVICES) for its launches, so no two stream-K kernels ever share
the GPU (their programs could not all be resident).
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
NEGATIVE_SHAPES = [(3328, 5120, 8192)]
# (rows, columns) of tiles: S = 16 (even split), 33 (uneven tiles, EXTRA > 0), 4 (many programs per
# tile) and 128 (2 per tile), all with one full wave or more of 256 programs; at 256x256, v26's SHAPES.
QUADRANT_GRIDS = [(17, 16), (17, 17), (13, 20), (16, 24)]
SMALL_SHAPES = [(16, 4096, 8192), (30, 4000, 4080), (1, 8192, 4096), (32, 4000, 8192)]
OPTION_KEYS = ("STREAMK_POLICY", "STREAMK_FIXUP", "STREAMK_PARTIALS", "STREAMK_RS_READERS", "STREAMK_PLACEMENT",
               "STREAMK_EXCHANGE", "STREAMK_NUM_PROGRAMS", "STREAMK_XCD_LOCAL", "STREAMK_UNSAFE_PLACEMENT",
               "PERSISTENT_TILE_ORDER", "STREAMK_TILE", "STREAMK_PERSISTENT_PROGRAMS")


def geometry(tile):
    sys.path.insert(0, HERE)
    import bench_ext  # noqa: F401  (paths)
    import importlib

    matmul_kernel = importlib.import_module(os.environ.get("V28_MATMUL_MODULE", "v28_streamk_owner.matmul_kernel"))

    return matmul_kernel.tile_geometry(tile), matmul_kernel.parse_spec(tile)


def quadrant_shapes(geo):
    bm, bn = geo["block_m"], geo["block_n"]
    shapes = [(mt * bm, nt * bn, 8192) for mt, nt in QUADRANT_GRIDS]
    # v26's ragged pair, scaled: 1000x2000x8080 (32 tiles, all stream-K) and 4300x4000x8176 (S = 16).
    return shapes + [(4 * bm - bm * 3 // 32, 8 * bn - bn * 3 // 16, 8080),
                     (17 * bm - bm * 13 // 64, 16 * bn - bn * 3 // 8, 8176)]


def persistent_shapes(geo, programs):
    """--persistent-programs: quadrant_shapes, plus 1.5 waves of the programs aligned and about 2.2 with M, N and K
    ragged, so that programs take two and three trips of the persistent loop."""
    bm, bn = geo["block_m"], geo["block_n"]
    nt = 32
    mt_a, mt_r = -(-3 * programs // (2 * nt)), -(-(2 * programs + 37) // nt) + 1
    return quadrant_shapes(geo) + [(mt_a * bm, nt * bn, 4096),
                                   (mt_r * bm - bm * 3 // 32, nt * bn - bn * 3 // 16, 4096 - 112)]


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


def leftover(geo, M, N):
    m, n = kernel_dims(geo, M, N)
    return (-(-m // geo["block_m"])) * (-(-n // geo["block_n"])) % geo["num_programs"]


def tile_mean(x, bm, bn):
    """Mean of x over each bm x bn tile, the edge tiles over their in-bounds elements."""
    import torch
    import torch.nn.functional as F

    M, N = x.shape
    mt, nt = -(-M // bm), -(-N // bn)
    pad = (0, nt * bn - N, 0, mt * bm - M)
    s = F.pad(x, pad).view(mt, bm, nt, bn).sum((1, 3))
    cnt = F.pad(torch.ones_like(x), pad).view(mt, bm, nt, bn).sum((1, 3))
    return s / cnt


def tile0_programs(geo, M, N, K, snp):
    """Programs on one-tile stream-K tile 0 (the kernel's N_HI if EXTRA > 0, else N_LO)."""
    s = leftover(geo, M, N)
    pairs = -(-K // (2 * geo["block_k"]))
    n_lo, n_hi = min(snp // s, pairs), min(snp // s + 1, pairs)
    return n_hi if n_hi > n_lo and snp % s else n_lo


def split_counts(geo, M, N, K):
    """The STREAMK_NUM_PROGRAMS values of a shape's one-tile cases: S n at n = 4, an uneven split
    (S n + S / 2, so half the tiles get one more program), and the default (None, all programs)."""
    s = leftover(geo, M, N)
    out = []
    for snp in (4 * s, 4 * s + s // 2):
        if s and 2 * s <= snp <= geo["num_programs"] and snp not in out:
            out.append(snp)
    return out + [None]


def check_shape(tile, M, N, K, bias, launches, stale_count, by_mean, snp):
    import bench_ext as bench  # noqa: F401  (paths)
    import torch

    import importlib

    matmul_kernel = importlib.import_module(os.environ.get("V28_MATMUL_MODULE", "v28_streamk_owner.matmul_kernel"))

    geo = matmul_kernel.tile_geometry(tile)
    bm, bn = matmul_kernel.parse_spec(tile)[:2]
    np_ = int(os.environ.get("STREAMK_PERSISTENT_PROGRAMS") or geo["num_programs"])
    # The mean-error check's region: the tile, widened along its long side until it holds at least 1024
    # in-bounds elements (a tile of one row would otherwise be judged on 64 samples).
    reps = -(-1024 // (min(M, bm) * min(N, bn)))
    bm, bn = (bm, bn * reps) if bn >= bm else (bm * reps, bn)
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
            rnd = tile_mean((ref32.to(dtype).float() - ref32).abs(), bm, bn)
            ref = (ref32, rnd)
        inputs.append((a, b, bias_n, ref))
    n_max = min((snp or np_) // max(leftover(geo, M, N), 1) + 1, -(-K // (2 * geo["block_k"])))
    # Compile without launching (in parallel with the other jobs), then hold the GPU for the launches.
    matmul_kernel.WARMUP = True
    matmul_kernel.matmul(inputs[0][0], inputs[0][1], bias=inputs[0][2], tile=tile)
    matmul_kernel.WARMUP = False
    torch.cuda.synchronize()
    lock = open(f"/tmp/v27_check_gpu{os.environ.get('HIP_VISIBLE_DEVICES', 'all')}.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    # C windows of guard buffers: 256 more rows and columns, all holding the sentinel.
    SENTINEL = 1234.0
    bufs = [torch.empty((M + 256, N + 256), device=bench.DEVICE, dtype=dtype) for _ in range(N_INPUTS)]
    outs = [buf[:M, :N] for buf in bufs]
    guard = torch.ones((M + 256, N + 256), device=bench.DEVICE, dtype=torch.bool)
    guard[:M, :N] = False
    _, locks, tile_sync, _ = matmul_kernel.streamk_workspace(bench.DEVICE, np_, *geo["slot"])

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
        matmul_kernel.matmul(a, b, c, bias=bias_n, tile=tile)
        if by_mean:
            ref32, rnd = ref
            err = (c.float() - ref32).abs()
            tile_err = tile_mean(err, bm, bn)
            bad += bool(torch.isnan(c).any() or (tile_err > (1 + n_max ** 0.5) * rnd + 1e-3).any())
            max_err = max(max_err, float(err.max()))
        else:
            # Same tolerances as bench.test_correctness.
            if not torch.allclose(c, ref, atol=1e-1, rtol=1e-2):
                bad += 1
            max_err = max(max_err, float((c.float() - ref.float()).abs().max()))
        bad += bool((buf[guard] != SENTINEL).any())
    torch.cuda.synchronize()
    # tile_sync: the barriers' (count, generation) pairs, the L2 epoch flags (which hold epochs),
    # then the atomic ending's two zero counts per tile; the counts must all be zero.
    flags_up = (int((locks != 0).sum()) + int((tile_sync[:2 * np_:2] != 0).sum())
                + int((tile_sync[3 * np_:] != 0).sum()))
    fcntl.flock(lock, fcntl.LOCK_UN)
    kernel = matmul_kernel.LAST_LAUNCH.get("kernel", "?")
    grid = matmul_kernel.LAST_LAUNCH.get("grid")
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


def configurations(args, geo):
    """(env, by_mean) of every valid configuration, without the program count."""
    out = []
    partials = [p for p in args.partials if p != "bf16_pairs" or (not geo["small"] and geo["block_n"] >= 128)]
    for policy in args.policies:
        if policy != "one_tile":
            for p in partials:
                out.append(({"STREAMK_POLICY": policy, "STREAMK_PARTIALS": p}, p != "fp32"))
            continue
        for fixup, p, readers, placement, exchange in itertools.product(
                args.fixups, partials, args.readers, args.placements, args.exchanges):
            if fixup != "rs" and readers != args.readers[0]:
                continue  # only reduce-scatter has readers to choose
            if fixup == "atomic_iz" and (p != "fp32" or exchange != "wt"):
                continue
            if exchange == "l2" and (fixup != "rs" or placement != "tile"):
                continue
            env = {"STREAMK_POLICY": "one_tile", "STREAMK_FIXUP": fixup, "STREAMK_PARTIALS": p,
                   "STREAMK_RS_READERS": readers, "STREAMK_PLACEMENT": placement, "STREAMK_EXCHANGE": exchange}
            out.append((env, p != "fp32" or fixup == "atomic_iz"))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tile", default="256x256x64", help="v9_tile spec (STREAMK_TILE); default 256x256x64")
    parser.add_argument("--shapes", nargs="*", help="MxNxK shapes (default: the tile's set)")
    parser.add_argument("--launches", type=int, default=10)
    parser.add_argument("--orders", nargs="+", default=["v9"], choices=["v9", "split"])
    parser.add_argument("--bias", nargs="+", type=int, default=[0, 1], choices=[0, 1])
    parser.add_argument("--policies", nargs="+", default=["one_tile", "two_tile", "spread"],
                        choices=["dp", "one_tile", "two_tile", "spread", "capped"])
    parser.add_argument("--fixups", nargs="+", default=["rs", "owner", "atomic_iz"], choices=["rs", "owner", "atomic_iz"])
    parser.add_argument("--partials", nargs="+", default=["fp32", "bf16", "bf16_pairs"],
                        choices=["fp32", "bf16", "bf16_pairs"])
    parser.add_argument("--readers", nargs="+", default=["pow2", "all"], choices=["pow2", "all"])
    parser.add_argument("--placements", nargs="+", default=["tile", "chunk"], choices=["tile", "chunk"])
    parser.add_argument("--exchanges", nargs="+", default=["wt", "l2"], choices=["wt", "l2"])
    parser.add_argument("--stale-count", action="store_true", help="Preset a barrier count (must FAIL)")
    parser.add_argument("--unsafe-placement", action="store_true", help="l2 with tiles across XCDs (must FAIL)")
    parser.add_argument("--no-full", action="store_true",
                        help="only shapes with no full tiles (check_grid.grid_shapes), where the compact launch "
                             "applies; spread also at the one-tile program counts")
    parser.add_argument("--persistent-programs", type=int,
                        help="STREAMK_PERSISTENT_PROGRAMS: the dp policy only (fp32), on persistent_shapes")
    parser.add_argument("--timeout", type=int, default=600, help="Seconds per configuration")
    parser.add_argument("--jobs", type=int, default=8, help="Configurations compiled at once")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        case = json.loads(args.worker)
        M, N, K = case["shape"]
        bad, flags_up, max_err, kernel = check_shape(case["tile"], M, N, K, bool(case["bias"]), case["launches"],
                                                     case["stale_count"], case["by_mean"], case["snp"])
        print(RESULT_TAG + json.dumps({"bad": bad, "flags_up": flags_up, "max_err": max_err, "kernel": kernel}),
              flush=True)
        return

    import regress

    negative = args.stale_count or args.unsafe_placement
    if args.stale_count:
        args.policies, args.fixups, args.exchanges = ["one_tile"], ["rs"], ["wt"]
    if args.unsafe_placement:
        args.policies, args.fixups, args.exchanges, args.placements = ["one_tile"], ["rs"], ["l2"], ["tile"]
        args.tile = "256x256x64"
    if args.persistent_programs:
        args.policies, args.partials = ["dp"], ["fp32"]
    geo, _ = geometry(args.tile)
    if args.shapes:
        shapes = [regress.parse_shape(s) for s in args.shapes]
    elif args.persistent_programs:
        shapes = persistent_shapes(geo, args.persistent_programs)
    elif args.unsafe_placement:
        shapes = NEGATIVE_SHAPES
    elif args.no_full:
        from v28_streamk_owner.check_grid import grid_shapes

        shapes = grid_shapes(geo)
    else:
        shapes = small_shapes(geo) if geo["small"] else quadrant_shapes(geo)
    failures = 0
    print(f"tile {args.tile}: kernel BLOCK {geo['block_m']}x{geo['block_n']}x{geo['block_k']}, {geo['w']} waves, "
          f"{args.persistent_programs or geo['num_programs']} programs"
          f"{', swapped (C^T = B^T A^T)' if geo['swap'] else ''}, "
          f"STREAMK_OWNER_OPT={os.environ.get('STREAMK_OWNER_OPT', '') or '(none)'}\n")
    print("| shape | configuration | STREAMK_NUM_PROGRAMS | tile order | bias | kernel | launches | wrong results | "
          "flags or counts left up | largest error |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    cases = []
    for M, N, K in shapes:
        for env, by_mean in configurations(args, geo):
            if env.get("STREAMK_FIXUP") == "atomic_iz" and N % 2:
                continue  # packed 16-bit pairs need an even N (and C row stride): the host rejects it
            if env["STREAMK_POLICY"] == "spread" and args.no_full:
                snps = split_counts(geo, M, N, K)
            elif env["STREAMK_POLICY"] != "one_tile":
                snps = [None]
            elif args.unsafe_placement:
                snps = [4 * leftover(geo, M, N)]
            else:
                snps = split_counts(geo, M, N, K)
            for snp, order, bias in itertools.product(snps, args.orders, args.bias):
                case_env = {**env, "PERSISTENT_TILE_ORDER": order, "STREAMK_TILE": args.tile}
                if snp is not None:
                    case_env["STREAMK_NUM_PROGRAMS"] = str(snp)
                if args.unsafe_placement:
                    case_env |= {"STREAMK_XCD_LOCAL": "0", "STREAMK_UNSAFE_PLACEMENT": "1"}
                if args.persistent_programs:
                    case_env["STREAMK_PERSISTENT_PROGRAMS"] = str(args.persistent_programs)
                cases.append(({"tile": args.tile, "shape": [M, N, K], "env": case_env, "by_mean": by_mean,
                               "snp": snp, "bias": bias, "launches": args.launches, "stale_count": args.stale_count},
                              env))
    with ThreadPoolExecutor(1 if negative else args.jobs) as ex:
        for (case, env), r in zip(cases, ex.map(lambda ce: run_case(ce[0], args.timeout), cases)):
            M, N, K = case["shape"]
            if "error" in r:
                failures += 1
                res = f"| {args.launches} | error: {r['error']} | |"
            else:
                failures += r["bad"] + r["flags_up"]
                res = f"{r['kernel']} | {args.launches} | {r['bad']} | {r['flags_up']} | {r['max_err']:.3f}"
            label = ", ".join(f"{k.removeprefix('STREAMK_')}={v}" for k, v in env.items())
            print(f"| {M}x{N}x{K} | {label} | {case['snp'] or 'default'} | {case['env']['PERSISTENT_TILE_ORDER']} | "
                  f"{case['bias']} | {res} |", flush=True)
    if negative:
        print("\nFAIL, as required (the negative control was detected)" if failures
              else "\nPASS: the negative control did NOT fail, so this check cannot be trusted")
        sys.exit(0 if failures else 1)
    print("\nPASS" if failures == 0 else "\nFAIL")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
