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
"""v27 with the REFIT rule choosing the tile and the stream-K configuration per shape: one call.

    source v27_streamk_tile/env.sh          # the v3.0 stack v27 was checked and timed on
    from v27_streamk_tile import refit_matmul
    c = refit_matmul.matmul(a, b)            # a: M x K, b: K x N (b may be a transposed N x K view), c: M x N
    refit_matmul.choice(M, N, K)             # what it runs: {"tile", "kind", "kernel", "options"}

    python v27_streamk_tile/refit_matmul.py picks results/heldout_picks.csv   # the choice for every AITER shape
    python v27_streamk_tile/refit_matmul.py check                             # a few shapes against torch

refit_rule.pick(M, N, K) gives a sweep.py kind and a tile (nearest neighbours over the REFIT sweep's timings,
results/refit_rule_table.json); the choice is cached per shape. The kind becomes v27's options as sweep.py timed it
(sweep.kind_env): cap runs v26's host rule (STREAMK_POLICY=capped, the default) at the chosen tile; pdp / pdp{P} the
persistent data-parallel loop (STREAMK_POLICY=dp, STREAMK_PERSISTENT_PROGRAMS); rs{n} / ow{n} the one-tile split
with n programs per leftover tile; 2t two-tile; sp spread; dp goes to v9_tile on the tile (matmul_kernel.fallback_spec).
The options are set in a private copy of the environment (sweep.v27_module), never with setenv, which races with
profilers' threads.
"""

import functools
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import refit_rule  # noqa: E402
import sweep  # noqa: E402

OPTION_KEYS = ("STREAMK_POLICY", "STREAMK_NUM_PROGRAMS", "STREAMK_FIXUP", "STREAMK_PARTIALS", "STREAMK_COMPACT_GRID",
               "STREAMK_PERSISTENT_PROGRAMS", "STREAMK_TILE")


@functools.lru_cache(None)
def choice(M, N, K):
    """{"tile", "kind", "kernel" (v27_streamk_tile or v9_tile), "options" (v27's STREAMK_* settings)} for the shape."""
    kind, tile = refit_rule.pick(M, N, K)
    if kind == "dp":
        spec = sweep.v27_module().fallback_spec(tile, sweep.v27_module().tile_geometry(tile))
        return {"tile": spec, "kind": kind, "kernel": "v9_tile", "options": {}}
    return {"tile": tile, "kind": kind, "kernel": "v27_streamk_tile", "options": sweep.kind_env(kind, tile, M, N, K)}


def matmul(a, b, c=None, bias=None):
    """C = A B (+ bias) with the REFIT rule's tile and configuration for (M, N, K)."""
    import importlib

    M, K = a.shape
    N = b.shape[1]
    ch = choice(M, N, K)
    if ch["kernel"] == "v9_tile":
        return importlib.import_module("v9_tile.matmul_kernel").matmul(a, b, c, bias, tile=ch["tile"])
    v27 = sweep.v27_module()
    opts = v27.os.environ
    for k in OPTION_KEYS:
        opts.pop(k, None)
    opts.update(ch["options"])
    try:
        return v27.matmul(a, b, c, bias, tile=ch["tile"])
    finally:
        for k in ch["options"]:
            opts.pop(k, None)


def write_picks(path):
    """The choice for every distinct AITER shape (sweep.AITER_SHAPES_CSV), as CSV."""
    import csv
    import json

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["M", "N", "K", "kind", "tile", "kernel", "options"])
        for M, N, K in sweep.heldout_shapes():
            ch = choice(M, N, K)
            w.writerow([M, N, K, ch["kind"], ch["tile"], ch["kernel"], json.dumps(ch["options"], sort_keys=True)])
    print(f"{len(sweep.heldout_shapes())} shapes -> {path}")


def check():
    """A few AITER shapes, one per kind the rule picks there, against torch (bf16 and fp32 output)."""
    import bench_ext  # noqa: F401  (paths)
    import torch

    seen, cases = set(), []
    for sh in sweep.heldout_shapes():
        k = choice(*sh)["kind"]
        key = k[:2] if k[:2] in ("rs", "ow") else k
        if key not in seen and sh[0] * sh[1] * sh[2] <= 2**36:
            seen.add(key)
            cases.append(sh)
    for M, N, K in cases:
        for dt in (torch.bfloat16, torch.float32):
            a = torch.rand((M, K), device="cuda", dtype=torch.bfloat16) - 0.5
            b = (torch.rand((N, K), device="cuda", dtype=torch.bfloat16) - 0.5).T
            c = torch.empty((M, N), device="cuda", dtype=dt)
            matmul(a, b, c)
            ref = a.float() @ b.float()
            ok = torch.allclose(c.float(), ref.to(dt).float(), atol=1e-1, rtol=1e-2)
            print(f"{M}x{N}x{K} {str(dt)[6:]}: {choice(M, N, K)} {'ok' if ok else 'WRONG'}", flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["picks"]:
        write_picks(sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "results", "heldout_picks.csv"))
    else:
        check()
