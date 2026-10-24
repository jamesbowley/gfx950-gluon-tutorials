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
"""The installed hipBLASLt (torch.matmul in /opt/venv) against v28's picks, in one process on one GPU.

    source v28_streamk_owner/env.sh                      # HIP 1 = amd-smi GPU 0
    python v28_streamk_owner/hbl_current.py --tag def1
    TENSILE_STREAMK_FIXED_GRID=128 python v28_streamk_owner/hbl_current.py --tag g128

Every method is timed the same way: a CUDA graph of 100 launches over 2 GB of rotating A, B and C copies (each launch
reads copies last touched about 60 launches earlier, so caches are cold), replayed REPLAYS times per round between two
events; the time per launch is the round's elapsed time over its launches, so it includes any launch gaps (hipBLASLt's
second kernel among them). ROUNDS rounds, each in a new shuffled order of methods; the median over rounds. Each
method's result is checked against torch first. B is the (K, N) view of a contiguous (N, K) tensor for every method,
as the sweeps run it. torch.profiler then lists the kernels each method launches and their device times.

V28_MATMUL_MODULE=v28_streamk_owner.experiments.matmul_kernel_variants times the archived experiment kernel and its
STREAMK_OWNER_OPT picks (as WORKLOG sections 6 and 8 did); without it, v28 itself (dp, rsall, owall, ow3, ow2, auto).

Methods: hbl (torch.matmul); dp (v9_tile at 128x128x128, one program per tile: the main-loop control against
TENSILE_STREAMK_FIXED_GRID=64) and v28 one-tile picks at 128x128x128 (rsall, owall, owall+b, owall+s, ow3, ow3+b, ow2,
ow2+b, ow2+s, and owall / ow3 / ow2 with bs; sweep.py's kinds); auto and auto+b (v28 as deployed, tile="auto" and the capped policy, without and with
STREAMK_OWNER_OPT=b). Results: results/hbl_current/<tag>.json.
"""

import argparse
import json
import os
import random
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sweep  # noqa: E402

OUT = os.path.join(HERE, "results", "hbl_current")
SHAPE = (1024, 1024, 7168)
TILE = "128x128x128"
ARCHIVE = "v28_streamk_owner.experiments.matmul_kernel_variants"
# The STREAMK_OWNER_OPT picks (sections 6 and 8 of the WORKLOG) need the archived experiment kernel
# (V28_MATMUL_MODULE=ARCHIVE); v28 itself is their bs.
VARIANT_PICKS = ("dp", "rsall", "owall", "owall+b", "owall+s", "owall+bs", "ow3", "ow3+b", "ow3+bs", "ow2", "ow2+b",
         "ow2+s", "ow2+bs")
PICKS = VARIANT_PICKS if os.environ.get("V28_MATMUL_MODULE") == ARCHIVE else ("dp", "rsall", "owall", "ow3", "ow2")
GRAPH_LEN = 100
ROTATING_MB = 2048


def methods(M, N, K):
    import torch

    v28 = sweep.v27_module()
    out = {"hbl": lambda a, b, c: torch.matmul(a, b, out=c)}
    for kind in PICKS:
        out[kind] = sweep.runner(kind, TILE, M, N, K)
    opts = v28.os.environ

    def auto(var):
        def run(a, b, c):
            if var:
                opts["STREAMK_OWNER_OPT"] = var
            v28.matmul(a, b, c, tile="auto")
            opts.pop("STREAMK_OWNER_OPT", None)
        return run

    out["auto"] = auto("")
    if os.environ.get("V28_MATMUL_MODULE") == ARCHIVE:
        out["auto+b"] = auto("b")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--replays", type=int, default=10, help="graph replays per round (100 launches each)")
    args = ap.parse_args()
    import bench_ext as bench
    import torch

    assert os.environ.get("HIP_VISIBLE_DEVICES") == "1", "this study runs on HIP 1 (amd-smi GPU 0) only"
    props = torch.cuda.get_device_properties(0)
    assert props.pci_bus_id == 0x05, f"HIP 1 is on PCI bus {props.pci_bus_id:#x}, expected 0x05"
    M, N, K = SHAPE
    dt = torch.bfloat16
    fns = methods(M, N, K)
    v28 = sweep.v27_module()

    # Correctness and what each method launched.
    a = torch.rand((M, K), device="cuda", dtype=dt) - 0.5
    b = bench.make_b(N, K, dt, False, lambda t: t.uniform_(-0.5, 0.5))
    ref = (a.float() @ b.float()).to(dt)
    launched, correct = {}, {}
    for name, fn in fns.items():
        c = bench.make_c(M, N, dt, False)
        v28.LAST_LAUNCH.clear()
        fn(a, b, c)
        torch.cuda.synchronize()
        correct[name] = bool(torch.allclose(c, ref, atol=1e-1, rtol=1e-2))
        launched[name] = {k: v28.LAST_LAUNCH.get(k) for k in ("kernel", "tile", "policy", "fixup", "streamk_tiles",
                                                              "streamk_programs", "owner_opt")} if name != "hbl" else {}
    assert all(correct.values()), f"wrong results: {[n for n, ok in correct.items() if not ok]}"

    # Graphs over rotating copies, one per method, built once.
    graphs = {}
    for name, fn in fns.items():
        a_l, b_l, c_l, copies = bench.gen_rotating_tensors(M, N, K, dt, ROTATING_MB)
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
        graphs[name] = (g, (a_l, b_l, c_l), copies)

    lock = open(f"/tmp/v27_check_gpu{os.environ['HIP_VISIBLE_DEVICES']}.lock", "w")
    import fcntl

    fcntl.flock(lock, fcntl.LOCK_EX)
    times = {n: [] for n in fns}
    rng = random.Random(f"hbl|{args.tag}")
    for rnd in range(args.rounds):
        order = list(fns)
        rng.shuffle(order)
        for name in order:
            g = graphs[name][0]
            g.replay()
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(args.replays):
                g.replay()
            e.record()
            torch.cuda.synchronize()
            times[name].append(s.elapsed_time(e) * 1e3 / (args.replays * GRAPH_LEN))
    fcntl.flock(lock, fcntl.LOCK_UN)

    # The kernels each method launches and their device times (cold copies, 20 calls).
    from torch.profiler import ProfilerActivity, profile

    kernels = {}
    for name, fn in fns.items():
        _, (a_l, b_l, c_l), copies = graphs[name]
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for i in range(20):
                fn(a_l[i % copies], b_l[i % copies], c_l[i % copies])
            torch.cuda.synchronize()
        evs = [e for e in prof.key_averages() if e.device_type.name == "CUDA" and e.count]
        kernels[name] = [dict(name=e.key, calls=e.count,
                              avg_us=round(getattr(e, "device_time_total", 0) / e.count, 2)) for e in evs]

    flops = 2 * M * N * K
    res = dict(tag=args.tag, time=time.strftime("%Y-%m-%d %H:%M:%S"), shape=SHAPE, tile=TILE,
               env={k: os.environ.get(k) for k in ("HIP_VISIBLE_DEVICES", "TENSILE_STREAMK_FIXED_GRID")},
               torch=torch.__version__, hip=torch.version.hip, gpu_pci_bus=props.pci_bus_id, rounds=args.rounds,
               launches_per_round=args.replays * GRAPH_LEN, correct=correct, launched=launched, kernels=kernels,
               us={n: [round(x, 3) for x in t] for n, t in times.items()},
               us_median={n: round(statistics.median(t), 3) for n, t in times.items()},
               tflops={n: round(flops / (statistics.median(t) * 1e-6) / 1e12, 1) for n, t in times.items()},
               spread={n: round(max(t) / min(t) - 1, 4) for n, t in times.items()})
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, f"{args.tag}.json"), "w"), indent=1)
    hb = res["us_median"]["hbl"]
    print(f"{args.tag}: FIXED_GRID={res['env']['TENSILE_STREAMK_FIXED_GRID']}")
    for n in fns:
        print(f"  {n:9s} {res['us_median'][n]:8.2f} us {res['tflops'][n]:7.1f} TFLOPS  vs hbl {hb / res['us_median'][n]:.3f}"
              f"  spread {100 * res['spread'][n]:.1f}%  {launched[n].get('tile') or ''} {launched[n].get('fixup') or ''}"
              f" {launched[n].get('streamk_programs') or ''}")
    for k in kernels["hbl"]:
        print("  hbl kernel:", k)


if __name__ == "__main__":
    main()
