"""rs_all's barrier knobs on a few points (WORKLOG entry 16).

    GPU=g python v30_streamk_streamlined/timing_rs_all_barrier.py --shard g --shards 8

timing_reduce_variants.py's graph method. Variants: rs, sep, rs spread (rs's
readers with each tile's barrier words in their own 128-byte line), and rs_all with
the poll's s_sleep (0, 1, 8, 32), its barriers at rs's slot t n or spread. Prints
one markdown row per point: rs us, then each variant / rs in the order above.
"""

import argparse
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

GRAPH_LEN = 100
ROTATING_MB = 2048
POINTS = [
    ("256x256x64", (3072, 5632, 2048), 2),  # S = 8, E = 30: rs_all 1.19x
    ("256x256x64", (3072, 5632, 8192), 2),
    ("128x128x128", (2048, 2176, 2048), 2),  # S = 16, E = 14: 1.15x
    ("256x256x64", (4096, 4352, 2048), 2),  # S = 16, E = 14
    ("256x256x64", (4096, 4608, 2048), 2),  # S = 32, E = 6: rs_all 0.97x
    ("256x256x64", (4096, 4608, 2048), 4),  # S = 32, E = 4
    ("256x256x64", (3072, 5632, 2048), 8),  # S = 8, E = 24
    ("128x128x128", (2048, 2560, 2048), 2),  # S = 64, E = 2
]
SLEEPS = {0: "", 1: " s1", 2: " s8", 3: " s32"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--replays", type=int, default=5)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    args = ap.parse_args()

    import bench_ext as bench
    import torch
    from v30_streamk_streamlined.hardware import query_device_config
    from v30_streamk_streamlined.heuristics import select_plan
    from v30_streamk_streamlined.reduce_variants import gemm_a16w16_variant
    from v30_streamk_streamlined.selection import TileSpec

    device = query_device_config()
    dt = torch.bfloat16
    for spec, (M, N, K), n in POINTS[args.shard :: args.shards]:
        tile = TileSpec(*(int(x) for x in spec.split("x")))
        cfg = select_plan(
            M, N, K, device=device, tile=tile, policy="one_tile", split_n=n, fixup="rs"
        )

        def var(variant, **kw):
            return lambda a, b, c: gemm_a16w16_variant(
                a, b.T, y=c, config=cfg, variant=variant, **kw
            )

        from v30_streamk_streamlined.selection import reduce_scatter_config

        own = reduce_scatter_config(tile, n).readers
        runners = {
            "rs": var("rs"),
            "sep": var("separate"),
            "rs spread": var("rs_all", rs_all_r=own, spread_barriers=True),
        }
        for spread in (False, True):
            for s, tag in SLEEPS.items():
                name = "rs_all" + (" spread" if spread else "") + tag
                runners[name] = var("rs_all", poll_sleep=s, spread_barriers=spread)
        a = torch.rand((M, K), device="cuda", dtype=dt) - 0.5
        b = bench.make_b(N, K, dt, False, lambda t: t.uniform_(-0.5, 0.5))
        ref = (a.float() @ b.float()).to(dt)
        a_l, b_l, c_l, copies = bench.gen_rotating_tensors(M, N, K, dt, ROTATING_MB)
        graphs = {}
        for name, fn in runners.items():
            c = bench.make_c(M, N, dt, False)
            fn(a, b, c)
            torch.cuda.synchronize()
            assert torch.allclose(c, ref, atol=1e-1, rtol=1e-2), name
            for i in range(copies):
                fn(a_l[i], b_l[i], c_l[i])
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for i in range(GRAPH_LEN):
                    fn(a_l[i % copies], b_l[i % copies], c_l[i % copies])
            g.replay()
            torch.cuda.synchronize()
            graphs[name] = g
        times = {k: [] for k in graphs}
        rng = random.Random(f"rs_all barrier {spec} {M} {N} {K} {n}")
        for _ in range(args.rounds):
            order = list(graphs)
            rng.shuffle(order)
            for key in order:
                graphs[key].replay()
                torch.cuda.synchronize()
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                for _ in range(args.replays):
                    graphs[key].replay()
                e.record()
                torch.cuda.synchronize()
                times[key].append(s.elapsed_time(e) * 1e3 / (args.replays * GRAPH_LEN))
        med = {k: statistics.median(v) for k, v in times.items()}
        rs = med["rs"]
        cells = " | ".join(f"{med[k] / rs:.3f}" for k in runners)
        print(f"| {spec} | {M}x{N}x{K} | {n} | {rs:.2f} | {cells} |", flush=True)
        del graphs, a_l, b_l, c_l
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
