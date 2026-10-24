"""v30's graph-safe l2 flags against v29's epoch flags and v30's wt exchange.

    source v29_streamk_compact/env.sh
    python v30_streamk_streamlined/timing.py > v30_streamk_streamlined/results/timing.md

v29 timing.py's method: per (point, variant) a CUDA graph of 100 launches
over 2 GB of rotating A, B and C copies, replayed REPLAYS times per round
between two events; ROUNDS rounds, each in a new shuffled order; the median
over rounds. Every variant's result is checked against torch first.

The 100 launches of one graph cycle through different copies, and v29's
host passes a new epoch to each captured launch, so v29's l2 exchange is
correct in this harness and serves as the reference. All variants use the
split that v29's XCD-local rule resolves for the point.
"""

import argparse
import fcntl
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

GRAPH_LEN = 100
ROTATING_MB = 2048
VARIANTS = ("v29 l2", "v30 l2", "v29 wt", "v30 wt")
# (label, tile, (M, N, K))
POINTS = [
    ("anchor rsall", "128x128x128", (1024, 1024, 7168)),
    ("S=8", "256x256x64", (3072, 5632, 8192)),
    ("S=16", "256x256x64", (4352, 4096, 8192)),
    ("S=32", "256x256x64", (4096, 4608, 8192)),
]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--replays", type=int, default=5)
    ap.add_argument("--points", nargs="*", help="only these labels")
    args = ap.parse_args()

    import bench_ext as bench
    import torch
    import v29_streamk_compact.matmul_kernel as v29
    from v30_streamk_streamlined.hardware import query_device_config
    from v30_streamk_streamlined.heuristics import select_plan
    from v30_streamk_streamlined.matmul_kernel import gemm_a16w16 as v30_matmul
    from v30_streamk_streamlined.selection import TileSpec

    device = query_device_config()
    points = [p for p in POINTS if not args.points or p[0] in args.points]
    dt = torch.bfloat16
    graphs, splits, keep = {}, {}, []
    for label, tile, (M, N, K) in points:
        a = torch.rand((M, K), device="cuda", dtype=dt) - 0.5
        b = bench.make_b(N, K, dt, False, lambda t: t.uniform_(-0.5, 0.5))
        ref = (a.float() @ b.float()).to(dt)
        a_l, b_l, c_l, copies = bench.gen_rotating_tensors(
            M, N, K, dt, ROTATING_MB
        )
        keep.append((a_l, b_l, c_l))

        env = dict(
            STREAMK_POLICY="one_tile",
            STREAMK_FIXUP="rs",
            STREAMK_EXCHANGE="l2",
        )

        def run_v29(a_, b_, c_):
            os.environ.pop("STREAMK_NUM_PROGRAMS", None)
            os.environ.update(env)
            v29.matmul(a_, b_, c_, tile=tile)

        c = bench.make_c(M, N, dt, False)
        run_v29(a, b, c)
        torch.cuda.synchronize()
        launch = v29.LAST_LAUNCH
        split = launch["streamk_programs"] // launch["streamk_tiles"]
        splits[label] = (launch["streamk_tiles"], split)
        spec = TileSpec(*(int(x) for x in tile.split("x")))

        def v30_runner(exchange):
            config = select_plan(
                M,
                N,
                K,
                device=device,
                tile=spec,
                policy="one_tile",
                split_n=split,
                fixup="rs",
                exchange=exchange,
            )

            def run(a_, b_, c_):
                v30_matmul(a_, b_.T, y=c_, config=config)

            return run

        wt_env = dict(
            env,
            STREAMK_EXCHANGE="wt",
            STREAMK_NUM_PROGRAMS=str(launch["streamk_programs"]),
        )

        def run_v29_wt(a_, b_, c_):
            os.environ.update(wt_env)
            v29.matmul(a_, b_, c_, tile=tile)

        runners = {
            "v29 l2": run_v29,
            "v30 l2": v30_runner("l2"),
            "v29 wt": run_v29_wt,
            "v30 wt": v30_runner("wt"),
        }
        for variant, fn in runners.items():
            c = bench.make_c(M, N, dt, False)
            fn(a, b, c)
            torch.cuda.synchronize()
            assert torch.allclose(c, ref, atol=1e-1, rtol=1e-2), (
                f"{label} {variant}: wrong result"
            )
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
            for i in range(copies):
                check = (a_l[i].float() @ b_l[i].float()).to(dt)
                assert torch.allclose(c_l[i], check, atol=1e-1, rtol=1e-2), (
                    f"{label} {variant}: wrong result after replay"
                )
            graphs[(label, variant)] = g

    gpu = os.environ.get("HIP_VISIBLE_DEVICES", "all")
    lock = open(f"/tmp/v27_check_gpu{gpu}.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    times = {k: [] for k in graphs}
    rng = random.Random("v30 timing")
    for _ in range(args.rounds):
        order = list(graphs)
        rng.shuffle(order)
        for key in order:
            g = graphs[key]
            g.replay()
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(args.replays):
                g.replay()
            e.record()
            torch.cuda.synchronize()
            times[key].append(
                s.elapsed_time(e) * 1e3 / (args.replays * GRAPH_LEN)
            )
    fcntl.flock(lock, fcntl.LOCK_UN)

    print(
        f"# v30 l2 flags: {args.rounds} rounds of {args.replays} x "
        f"{GRAPH_LEN} launches, median us per launch, GPU {gpu}\n"
    )
    print(
        "| point | tile | shape | S | n | v29 l2 us | v30 l2 us | v29 wt us "
        "| v30 wt us | v30 l2 / v29 l2 | v30 wt / v29 wt | v30 l2 / v30 wt "
        "| spread (max/min - 1) |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for label, tile, shape in points:
        med = {v: statistics.median(times[(label, v)]) for v in VARIANTS}
        spread = "/".join(
            f"{100 * (max(times[(label, v)]) / min(times[(label, v)]) - 1):.1f}%"
            for v in VARIANTS
        )
        s, n = splits[label]
        print(
            f"| {label} | {tile} | {'x'.join(map(str, shape))} | {s} | {n} | "
            f"{med['v29 l2']:.2f} | {med['v30 l2']:.2f} | {med['v29 wt']:.2f} | "
            f"{med['v30 wt']:.2f} | {med['v30 l2'] / med['v29 l2']:.3f} | "
            f"{med['v30 wt'] / med['v29 wt']:.3f} | "
            f"{med['v30 l2'] / med['v30 wt']:.3f} | {spread} |"
        )


if __name__ == "__main__":
    main()
