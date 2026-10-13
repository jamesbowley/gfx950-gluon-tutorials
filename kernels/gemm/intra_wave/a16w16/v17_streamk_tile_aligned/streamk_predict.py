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
"""Predicted stream-K tail of v17 (tile-aligned cuts) against v16 (TensorAtlas partition),
both with lane-contiguous costs, for the regress.py STREAMK_SHAPES.

    python v17_streamk_tile_aligned/streamk_predict.py

Same model and units as v15's streamk_predict.py: 2-k-step units, lockstep, the tail from
the start of the stream-K phase, with store 6 (burst) and read 2 k-steps. The model assumes
L2 reuse is unaffected, which is what tile alignment is meant to restore.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "v15_streamk_onetile"))
sys.path.insert(0, os.path.dirname(HERE))

import streamk_predict as v15  # noqa: E402  (puts v14_streamk on the path)
import streamk_diagrams as sd  # noqa: E402

LANE = v15.COSTS["lane"]


def aligned(M, N, K):
    """Chunk lengths (k-steps) per stream-K tile and the tail, for v17's split."""
    pr = v15.pair_problem(M, N, K)
    tiles = sd.aligned_split(pr.streamk_tiles, pr.num_programs, pr.iters_per_tile)
    tail = 2 * max(sd.serial_tile(ch, LANE["burst_store"] / 2, LANE["read"] / 2) for ch in tiles)
    counts = sorted({len(ch) for ch in tiles})
    longest = 2 * max(max(ch) for ch in tiles)
    return dict(counts=counts, longest=longest, tail=tail, programs=sum(len(ch) for ch in tiles))


def main():
    import regress

    print("| shape | SK tiles | programs per tile | programs used | longest chunk (k-steps) | "
          "v16 tail, lane (k-steps) | v17 tail, lane (k-steps) |")
    print("|---|---|---|---|---|---|---|")
    for M, N, K in regress.STREAMK_SHAPES:
        p16 = v15.predict(M, N, K)
        p17 = aligned(M, N, K)
        print(f"| {M}x{N}x{K} | {p16['pr'].streamk_tiles} | {'-'.join(map(str, p17['counts']))} | "
              f"{p17['programs']} | {p17['longest']} | {p16['tails']['lane']:.1f} | {p17['tail']:.1f} |")


if __name__ == "__main__":
    main()
