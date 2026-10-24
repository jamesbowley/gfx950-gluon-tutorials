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
"""The REFIT rule: a (tile, configuration) for a shape, from the REFIT sweep's merged timings by nearest neighbours.

    from v27_streamk_tile import refit_rule
    kind, tile = refit_rule.pick(4096, 8192, 4096)          # a sweep.py kind and a v27 tile spec
    python v27_streamk_tile/refit.py init --study heldout --rule v27_streamk_tile/refit_rule.py

The training table (results/refit_rule_table.json, written by tmp/fit_rule.py) holds, per REFIT shape, every timed
(tile, kind)'s merged time over the shape's best. For a shape (M, N, K): the k nearest REFIT shapes of its regime
(compute-bound M >= 512, else memory-bound) in the variant's features, then, among the (tile, kind) valid at the
shape (sweep.valid), the one with the smallest mean log ratio over those neighbours; a candidate a neighbour did not
time (not valid there) counts as MISSING_RATIO. The variant per regime is the one tmp/fit_rule.py chose by leaving
shapes out (the table's "chosen").
"""

import functools
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sweep  # noqa: E402

TABLE = os.path.join(HERE, "results", "refit_rule_table.json")
MISSING_RATIO = 1.5


def regime(M):
    return "compute-bound" if M >= 512 else "memory-bound"


def k_bin(K):
    return round(math.log2(K))


def leftover(M, N, bm, bn, programs):
    """Leftover tiles over programs at a BM x BN tile."""
    return (-(-M // bm) * -(-N // bn)) % programs / programs


def f_mnk(sh):
    return tuple(math.log2(x) for x in sh)


def f_tile(sh):
    """log M, N, K, plus the leftover fraction of a wave at the regime's most common winning tile class (256x256 on 256
    programs; 16x64 on 1024 for memory-bound shapes), weighted 3."""
    M, N, K = sh
    bm, bn, p = (256, 256, 256) if regime(M) == "compute-bound" else (16, 64, 1024)
    return f_mnk(sh) + (3 * leftover(M, N, bm, bn, p),)


VARIANTS = {f"log MNK, k={k}": (f_mnk, k) for k in (1, 3, 5, 8)}
VARIANTS.update({f"tile coordinates, k={k}": (f_tile, k) for k in (3, 5, 8)})


@functools.lru_cache(None)
def valid(kind, tile, M, N, K):
    return sweep.valid(kind, tile, M, N, K)


def knn_pick(sh, train, variant, only=None):
    """(tile, kind) for shape sh from train {shape: {(tile, kind): ratio}}; only: restrict to that kind."""
    feat, k = variant
    x = feat(sh)
    near = sorted(train, key=lambda o: (sum((a - b) ** 2 for a, b in zip(x, feat(o))), o))[:k]
    cands = sorted({c for o in near for c in train[o] if (only is None or c[1] == only) and valid(c[1], c[0], *sh)})
    return min(cands, key=lambda c: sum(math.log(train[o].get(c, MISSING_RATIO)) for o in near))


@functools.lru_cache(None)
def table():
    t = json.load(open(TABLE))
    shapes = {g: {tuple(int(v) for v in s.split("x")): {tuple(c.split("|")): r for c, r in d.items()}
                  for s, d in t["shapes"][g].items()} for g in t["shapes"]}
    return t["chosen"], shapes


def pick(M, N, K):
    """(kind, tile) for C = A B with A M x K, B K x N."""
    chosen, shapes = table()
    g = regime(M)
    tile, kind = knn_pick((M, N, K), shapes[g], VARIANTS[chosen[g]])
    return kind, tile
