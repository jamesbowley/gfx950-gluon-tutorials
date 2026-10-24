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
"""v29 against v28, timed point by point in one process on one GPU.

    source v29_streamk_compact/env.sh
    python v29_streamk_compact/timing.py > v29_streamk_compact/results/timing.md

hbl_current.py's method (v28 WORKLOG §6): per (point, version) a CUDA graph of 100 launches over 2 GB of rotating
A, B and C copies, replayed REPLAYS times per round between two events; ROUNDS rounds, each in a new shuffled order;
the median over rounds. Each point's result is checked against torch first. B is the (K, N) view of a contiguous
(N, K) tensor. The points: the owner-ending anchor (1024x1024x7168 at 128x128x128), the builds whose code v29 changed
(asm_diff.md: 256x256 owner on 16 programs per tile, two-tile and spread, 128x128x64 reduce-scatter with bias), and
capped's default at 4096^3 (v9_tile in both: a control).
"""

import argparse
import json
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

GRAPH_LEN = 100
ROTATING_MB = 2048
MODULES = {"v28": "v28_streamk_owner.matmul_kernel", "v29": "v29_streamk_compact.matmul_kernel"}
ENV_KEYS = ("STREAMK_POLICY", "STREAMK_FIXUP", "STREAMK_NUM_PROGRAMS", "STREAMK_EXCHANGE")
# (label, tile, (M, N, K), env, bias)
POINTS = [
    ("anchor owall", "128x128x128", (1024, 1024, 7168), dict(STREAMK_POLICY="one_tile", STREAMK_FIXUP="owner"), 0),
    ("anchor ow3", "128x128x128", (1024, 1024, 7168),
     dict(STREAMK_POLICY="one_tile", STREAMK_FIXUP="owner", STREAMK_NUM_PROGRAMS="192"), 0),
    ("anchor ow2", "128x128x128", (1024, 1024, 7168),
     dict(STREAMK_POLICY="one_tile", STREAMK_FIXUP="owner", STREAMK_NUM_PROGRAMS="128"), 0),
    ("anchor rsall", "128x128x128", (1024, 1024, 7168), dict(STREAMK_POLICY="one_tile", STREAMK_FIXUP="rs"), 0),
    ("anchor rsall l2", "128x128x128", (1024, 1024, 7168),
     dict(STREAMK_POLICY="one_tile", STREAMK_FIXUP="rs", STREAMK_EXCHANGE="l2"), 0),
    ("256 ow16", "256x256x64", (4096, 4352, 8192),
     dict(STREAMK_POLICY="one_tile", STREAMK_FIXUP="owner", STREAMK_NUM_PROGRAMS="256"), 0),
    ("256 2t", "256x256x64", (4096, 4608, 8192), dict(STREAMK_POLICY="two_tile"), 0),
    ("256 2t+b", "256x256x64", (4096, 4608, 8192), dict(STREAMK_POLICY="two_tile"), 1),
    ("256 sp", "256x256x64", (4096, 4608, 8192), dict(STREAMK_POLICY="spread"), 0),
    ("256 ragged sp+b", "256x256x64", (4072, 4560, 8176), dict(STREAMK_POLICY="spread"), 1),
    ("128x64 rsall+b", "128x128x64", (2048, 2304, 8192), dict(STREAMK_POLICY="one_tile", STREAMK_FIXUP="rs"), 1),
    ("cap 4096^3", "256x256x64", (4096, 4096, 4096), {}, 0),
]


def runner(mod, tile, env, bias):
    def run(a, b, c):
        for k in ENV_KEYS:
            os.environ.pop(k, None)
        os.environ.update(env)
        mod.matmul(a, b, c, bias=bias, tile=tile)
    return run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--replays", type=int, default=5, help="graph replays per round (100 launches each)")
    ap.add_argument("--points", nargs="*", help="only these labels")
    args = ap.parse_args()
    import importlib
    import fcntl

    import bench_ext as bench
    import torch

    mods = {v: importlib.import_module(m) for v, m in MODULES.items()}
    points = [p for p in POINTS if not args.points or p[0] in args.points]
    dt = torch.bfloat16
    graphs, info, keep = {}, {}, []
    for label, tile, (M, N, K), env, has_bias in points:
        a = torch.rand((M, K), device="cuda", dtype=dt) - 0.5
        b = bench.make_b(N, K, dt, False, lambda t: t.uniform_(-0.5, 0.5))
        bias = torch.rand((N,), device="cuda", dtype=dt) - 0.5 if has_bias else None
        ref = (a.float() @ b.float() + (bias.float() if has_bias else 0)).to(dt)
        a_l, b_l, c_l, copies = bench.gen_rotating_tensors(M, N, K, dt, ROTATING_MB)
        # The graphs read these copies (and the bias) on every replay: they must outlive the loop.
        keep.append((a_l, b_l, c_l, bias))
        for v, mod in mods.items():
            fn = runner(mod, tile, env, bias)
            c = bench.make_c(M, N, dt, False)
            fn(a, b, c)
            torch.cuda.synchronize()
            assert torch.allclose(c, ref, atol=1e-1, rtol=1e-2), f"{label} {v}: wrong result"
            info[(label, v)] = {k: mod.LAST_LAUNCH.get(k) for k in ("kernel", "policy", "fixup", "streamk_programs")}
            for i in range(copies):
                fn(a_l[i], b_l[i], c_l[i])
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for i in range(GRAPH_LEN):
                    fn(a_l[i % copies], b_l[i % copies], c_l[i % copies])
            for _ in range(3):
                g.replay()
            torch.cuda.synchronize()
            graphs[(label, v)] = g

    lock = open(f"/tmp/v27_check_gpu{os.environ.get('HIP_VISIBLE_DEVICES', 'all')}.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    times = {k: [] for k in graphs}
    rng = random.Random("v29 timing")
    for _ in range(args.rounds):
        order = list(graphs)
        rng.shuffle(order)
        for key in order:
            g = graphs[key]
            g.replay()
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(args.replays):
                g.replay()
            e.record()
            torch.cuda.synchronize()
            times[key].append(s.elapsed_time(e) * 1e3 / (args.replays * GRAPH_LEN))
    fcntl.flock(lock, fcntl.LOCK_UN)

    print(f"# v29 against v28: {args.rounds} rounds of {args.replays} x {GRAPH_LEN} launches, median us per launch\n")
    print("| point | tile | shape | launched (v29) | v28 us | v29 us | v29 / v28 | spread v28 / v29 |")
    print("|---|---|---|---|---|---|---|---|")
    out = {}
    for label, tile, shp, env, has_bias in points:
        m28, m29 = (statistics.median(times[(label, v)]) for v in ("v28", "v29"))
        sp = [max(times[(label, v)]) / min(times[(label, v)]) - 1 for v in ("v28", "v29")]
        li = info[(label, "v29")]
        launched = f"{li['kernel']}" + (f" {li['policy']} / {li['fixup']} / {li['streamk_programs']}"
                                        if li["kernel"] != "v9_tile" else "")
        print(f"| {label} | {tile} | {'x'.join(map(str, shp))} | {launched} | {m28:.2f} | {m29:.2f} | "
              f"{m29 / m28:.3f} | {100 * sp[0]:.1f}% / {100 * sp[1]:.1f}% |")
        out[label] = dict(v28=times[(label, "v28")], v29=times[(label, "v29")], launched=info[(label, "v29")])
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "timing.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
