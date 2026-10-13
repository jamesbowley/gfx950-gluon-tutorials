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
"""Predicted length of v15's stream-K tail for the regress.py STREAMK_SHAPES, from v14's
lockstep schedule model (v14_streamk/streamk_diagrams.py).

    python v15_streamk_onetile/streamk_predict.py

v15 hands out work in pairs of k-steps, so the model runs in pair units (half the k-steps
per tile, half the costs) and the result is converted back to k-steps. The tail is the
stream-K phase alone: every program starts it at t = 0 and it ends when the last owner has
read its last partial. Not modelled: the segment prologue (about 1.2 k-steps per segment),
the C store and spills.

Costs, in k-steps, from experiments/streamk_costs:
- row-major (v15): store 7 (the 6-8 burst when every contributor publishes at once, 4.1
  when spread out), read 4.9;
- lane-contiguous (doc section 9 "measured"): store 6 burst / 1.5 spread, read 2.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "v14_streamk"))
sys.path.insert(0, os.path.dirname(HERE))

import streamk_diagrams as sd  # noqa: E402

COSTS = {
    "row-major": dict(burst_store=7.0, spread_store=4.1, read=4.9),
    "lane": dict(burst_store=6.0, spread_store=1.5, read=2.0),
}


def pair_problem(M, N, K):
    """Problem.from_shape in 2-k-step units."""
    pr = sd.Problem.from_shape(M, N, K)
    return sd.Problem(pr.name, pr.num_programs, pr.streamk_tiles, pr.iters_per_tile // 2,
                      pr.full_tiles, pr.num_pid_m, pr.num_pid_n, pr.group_size_m, pr.pids_per_xcd)


def predict(M, N, K):
    pr = pair_problem(M, N, K)
    sched = sd.build_schedule(pr, sd.ORIGINAL)
    sd.self_check(sched)
    peers = [len(s.peers) for s in sd.reducers(sched)]
    crossing = sum(len(ss) > 1 for ss in sched.segs.values())
    # Every contributor publishes in the same few microseconds when programs cover less than
    # a tile each, which is the one-tile regime all these shapes are in.
    tails = {}
    for name, c in COSTS.items():
        s = sd.build_schedule(pr, sd.ORIGINAL, c["burst_store"] / 2, c["read"] / 2)
        tails[name] = 2 * sd.finish_time(s)
    return dict(pr=pr, longest=2 * (pr.L + (pr.R > 0)), peers=(min(peers), max(peers)),
                crossing=crossing, tails=tails)


def main():
    import regress

    print("| shape | full waves | SK tiles | longest range (k-steps) | peers per owner | "
          "crossing programs | tail, row-major (k-steps) | tail, lane (k-steps) |")
    print("|---|---|---|---|---|---|---|---|")
    for M, N, K in regress.STREAMK_SHAPES:
        p = predict(M, N, K)
        pr = p["pr"]
        print(f"| {M}x{N}x{K} | {pr.full_tiles // pr.num_programs} | {pr.streamk_tiles} | "
              f"{p['longest']} | {p['peers'][0]}-{p['peers'][1]} | {p['crossing']} | "
              f"{p['tails']['row-major']:.1f} | {p['tails']['lane']:.1f} |")


if __name__ == "__main__":
    main()
