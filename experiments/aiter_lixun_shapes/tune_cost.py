#!/usr/bin/env python3
"""Estimated cost of tuning v28 exhaustively on the origami run's shapes, and with the reductions of prune_eval.py.

    source $A16/v28_streamk_owner/env.sh; export HIP_VISIBLE_DEVICES=3; python $AIS/tune_cost.py

The full space per shape is WORST10's (sweep.WORST10_KINDS: dp, pdp, pdp{P}, cap, rs / ow at REFIT_N and all, each
also with bf16 and bf16-pair partials, 2t, sp, all) at every candidate tile of make_origami_picks (34 specs), less
what sweep.valid rejects and tiles with fewer than two pairs of k-steps; bf16 partials only on bf16_reduce shapes.
A point's time per round is the REFIT fit, 0.08 s plus its launches (sweep.shape_iters) at the kernel time, taken as
1.5x the origami run's best v28 time for the shape (the average point is slower than the best). Two rounds, as WORST10.
"""

import collections
import csv
import importlib
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
A16 = os.path.abspath(os.path.join(HERE, "..", "..", "kernels", "gemm", "intra_wave", "a16w16"))
sys.path.insert(0, A16)
sweep = importlib.import_module("v28_streamk_owner.sweep")
sys.path.insert(0, HERE)
from make_origami_picks import candidate_tiles  # noqa: E402
import origami_report  # noqa: E402

OVERHEAD_S = 0.08
ROUNDS = 2
POINT_SLOWDOWN = 1.5


def pow2(x):
    v = 1
    while v < x:
        v *= 2
    return v


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--overhead", type=float, default=OVERHEAD_S, help="seconds per point outside its launches")
    ap.add_argument("--iters", type=int, default=0, help="launches per point (0: sweep.shape_iters)")
    ap.add_argument("--share", type=float, default=1.0, help="share of the two-round cost kept (prune_eval pipeline)")
    ap.add_argument("--process-s", type=float, default=0.0, help="seconds to start and end one rocprofv3 process")
    ap.add_argument("--per-process", type=int, default=60, help="points timed per rocprofv3 process")
    ap.add_argument("--max-pad", type=float, default=0, help="drop tiles with a side over this many times the shape's "
                    "side rounded up to a power of two (at least 16); 0: keep every tile")
    args = ap.parse_args()
    rows = {(x["M"], x["N"], x["K"]): x for x in origami_report.load(
        [os.path.join(HERE, "results", n) for n in ("details_origami_rot2048.csv", "details_origami_redo.csv")],
        os.path.join(HERE, "picks_origami.csv"), os.path.join(HERE, "shapes_origami.csv")) if x["slots"]}
    tiles = candidate_tiles()
    fp32_kinds = [k for k in sweep.WORST10_KINDS if not (sweep.split_kind(k) and sweep.split_kind(k)[2] != "fp32")]
    pts, secs, by_scope = [], [], collections.defaultdict(lambda: [0, 0.0, 0])
    for s in csv.DictReader(open(os.path.join(HERE, "shapes_origami.csv"))):
        M, N, K = int(s["M"]), int(s["N"]), int(s["K"])
        x = rows.get((M, N, K))
        if not x:
            continue
        kinds = sweep.WORST10_KINDS if s["bf16_reduce"] else fp32_kinds
        n = 0
        for t in tiles:
            if sweep.shape_info(t, M, N, K)["pairs"] < 2:
                continue
            bm, bn = (int(v) for v in t.split("x")[:2])
            if args.max_pad and (bm > args.max_pad * max(16, pow2(M)) or bn > args.max_pad * max(16, pow2(N))):
                continue
            n += sum(1 for k in kinds if sweep.valid(k, t, M, N, K))
        t_us = POINT_SLOWDOWN * x["ours"] * 1e3
        iters = args.iters or sweep.shape_iters(M, N, K)
        per_round = n * (args.overhead + iters * t_us * 1e-6 + args.process_s / args.per_process)
        pts.append(n)
        secs.append(ROUNDS * per_round * args.share)
        b = by_scope[s["scope"]]
        b[0] += 1
        b[1] += ROUNDS * per_round * args.share
        b[2] += n
    print(f"{len(pts)} shapes; points per shape median {statistics.median(pts):.0f}, mean {statistics.mean(pts):.0f}, "
          f"max {max(pts)}; total {sum(pts):,} points")
    print(f"GPU time: median {statistics.median(secs) / 60:.1f} min per shape, mean {statistics.mean(secs) / 60:.1f}, "
          f"90th {sorted(secs)[int(0.9 * len(secs))] / 60:.1f}; total {sum(secs) / 3600:.0f} GPU-hours")
    for sc, (c, sec, n) in sorted(by_scope.items()):
        print(f"  {sc}: {c} shapes, {n / c:.0f} points and {sec / c / 60:.1f} min per shape, {sec / 3600:.0f} GPU-hours")


if __name__ == "__main__":
    main()
