"""Tree and separate-kernel reductions against the in-kernel reduce-scatter (WORKLOG 15).

    source v29_streamk_compact/env.sh
    for g in 0 1 2 3; do
        GPU=$g python v30_streamk_streamlined/timing_reduce_variants.py --set focused \
            --shard $g --shards 4 --out /tmp/rv/focused_$g.jsonl &
    done; wait
    python v30_streamk_streamlined/timing_reduce_variants.py --report /tmp/rv/focused_*.jsonl \
        > v30_streamk_streamlined/results/reduce_variants_focused.md

timing.py's method, one point at a time: per (point, variant) a CUDA graph of 100
calls over 2 GB of rotating A, B and C copies, replayed REPLAYS times per round
between two events; ROUNDS rounds, each in a new shuffled order over the point's
variants; the median over rounds. ``--eager`` also times the same 100 calls
launched from Python (host launch cost included). Every variant that writes C is
checked against torch first.

Variants (one-tile, wt exchange, the point's split):
  prod rs / prod owner  production gemm_a16w16, fixup rs / owner (prod rs is rs_all
                        since WORKLOG entry 18)
  rs                    reduce_variants' copy of the pre-entry-18 rs (the ratio base)
  tree2 / tree4         radix-2 / radix-4 tree (tree4 only from split 8), children
                        unrolled (several in flight) up to 7, else a run-time loop
  tree2 / tree4 loop    the same with the run-time loop (one child at a time; the
                        unrolled form spills at 256x256)
  sep                   store-only GEMM + _streamk_reduce_kernel, default readers
  sep R                 the same with the reduce-scatter's RS_R readers per tile
  rs all                rs with the idle programs as extra readers (R per tile; only
                        when E = R - n > 0), each tile's barrier in its own line
  rs all packed         the same with the barriers at rs's slots t n (shared lines)
  store_only            the GEMM without any reduction (a lower bound; wrong C)
  reduce_only           sep's reduce kernel alone
"""

import argparse
import fcntl
import json
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

GRAPH_LEN = 100
ROTATING_MB = 2048
TILES = ("64x64x64", "128x128x128", "256x128x64", "128x256x64", "256x256x64")
# Tile grids of one full wave plus S leftover tiles (256 programs).
WAVE_GRIDS = {8: (12, 22), 16: (16, 17), 32: (16, 18), 64: (16, 20)}


def tile_of(spec):
    from v30_streamk_streamlined.selection import TileSpec

    return TileSpec(*(int(x) for x in spec.split("x")))


def splits(tile, M, N, K, S, num_programs=256):
    pairs = -(-K // (2 * tile.block_k))
    cap = min(num_programs // S, pairs)
    return [n for n in (2, 4, 8, 16, 32) if n <= cap]


def wave_points(tiles, Ss, Ks, tag=""):
    points = []
    for spec in tiles:
        tile = tile_of(spec)
        for S in Ss:
            tm, tn = WAVE_GRIDS[S]
            M, N = tm * tile.block_m, tn * tile.block_n
            for K in Ks:
                for n in splits(tile, M, N, K, S):
                    points.append((f"{tag}S={S}", spec, (M, N, K), n, False))
    return points


def point_set(name):
    if name == "focused":
        points = [
            ("anchor rsall", "128x128x128", (1024, 1024, 7168), 4, False),
            ("t.py S=8", "256x256x64", (3072, 5632, 8192), 32, False),
            ("t.py S=16", "256x256x64", (4352, 4096, 8192), 16, False),
            ("t.py S=32", "256x256x64", (4096, 4608, 8192), 8, False),
            ("compact", "256x256x64", (2048, 2048, 8192), 2, False),
            ("compact", "256x256x64", (2048, 2048, 8192), 4, False),
        ]
        points += wave_points(("256x256x64", "128x128x128"), (8, 16, 32, 64), (2048, 8192))
    elif name == "wide":
        points = wave_points(TILES, (16, 32), (4096, 16384))
        for spec in TILES:
            t = tile_of(spec)
            shape = (17 * t.block_m - t.block_m // 8, 16 * t.block_n - t.block_n // 8, 8192)
            for n in (4, 8):
                points.append(("ragged+bias", spec, shape, n, True))
    else:
        raise ValueError(name)
    seen, out = set(), []
    for p in points:
        key = (p[1], p[2], p[3], p[4])
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def time_point(point, args, device):
    import bench_ext as bench
    import torch
    from v30_streamk_streamlined.heuristics import select_plan
    from v30_streamk_streamlined.matmul_kernel import gemm_a16w16
    from v30_streamk_streamlined.reduce_variants import (
        gemm_a16w16_variant,
        reduce_launch,
        rs_all_readers,
    )
    from v30_streamk_streamlined.selection import reduce_scatter_config, resolve_config

    label, spec, (M, N, K), n, with_bias = point
    tile = tile_of(spec)
    dt = torch.bfloat16
    rs_cfg = select_plan(
        M, N, K, device=device, tile=tile, policy="one_tile", split_n=n, fixup="rs"
    )
    owner_cfg = select_plan(
        M, N, K, device=device, tile=tile, policy="one_tile", split_n=n, fixup="owner"
    )
    launch = resolve_config(M, N, K, config=rs_cfg, device=device)
    S = launch["STREAMK_TILES"]
    readers, _, _ = reduce_launch(tile, S, device.num_programs)
    # The old rs's own readers (production's RS_R is rs_all's since entry 18).
    rs_r = reduce_scatter_config(tile, n).readers

    def prod(cfg):
        return lambda a, b, c, bias: gemm_a16w16(a, b.T, bias, y=c, config=cfg)

    def var(variant, **kw):
        return lambda a, b, c, bias: gemm_a16w16_variant(
            a, b.T, bias, y=c, config=rs_cfg, variant=variant, **kw
        )

    runners = {
        "prod rs": (prod(rs_cfg), True),
        "prod owner": (prod(owner_cfg), True),
        "rs": (var("rs"), True),
        "tree2": (var("tree", tree_radix=2), True),
        "tree2 loop": (var("tree", tree_radix=2, tree_static=False), True),
    }
    if n >= 8:
        runners["tree4"] = (var("tree", tree_radix=4), True)
        if n <= 16:
            runners["tree4 loop"] = (var("tree", tree_radix=4, tree_static=False), True)
    runners["sep"] = (var("separate"), True)
    if rs_r != readers:
        runners["sep R"] = (var("separate", reduce_readers=rs_r), True)
    _, _, _, rs_all_e = rs_all_readers(tile, S, n, device.num_programs)
    rs_all_r = rs_r
    if rs_all_e > 0:
        rs_all_r = n + rs_all_e
        runners["rs all"] = (var("rs_all"), True)
        runners["rs all packed"] = (var("rs_all", spread_barriers=False), True)
    runners["store_only"] = (var("store_only"), False)
    runners["reduce_only"] = (var("reduce_only"), False)

    a = torch.rand((M, K), device="cuda", dtype=dt) - 0.5
    b = bench.make_b(N, K, dt, False, lambda t: t.uniform_(-0.5, 0.5))
    bias = (torch.rand((N,), device="cuda", dtype=dt) - 0.5) if with_bias else None
    ref = a.float() @ b.float()
    if bias is not None:
        ref += bias.float()
    ref = ref.to(dt)
    a_l, b_l, c_l, copies = bench.gen_rotating_tensors(M, N, K, dt, ROTATING_MB)
    for name, (fn, checked) in runners.items():
        c = bench.make_c(M, N, dt, False)
        fn(a, b, c, bias)
        torch.cuda.synchronize()
        if checked:
            assert torch.allclose(c, ref, atol=1e-1, rtol=1e-2), f"{label} {name}: wrong"

    graphs = {}
    for name, (fn, _) in runners.items():
        for i in range(copies):
            fn(a_l[i], b_l[i], c_l[i], bias)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for i in range(GRAPH_LEN):
                fn(a_l[i % copies], b_l[i % copies], c_l[i % copies], bias)
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        graphs[name] = g

    def eager(fn):
        def go():
            for i in range(GRAPH_LEN):
                fn(a_l[i % copies], b_l[i % copies], c_l[i % copies], bias)

        return go

    timed = {("graph", k): g.replay for k, g in graphs.items()}
    if args.eager:
        timed.update({("eager", k): eager(fn) for k, (fn, _) in runners.items()})
    times = {k: [] for k in timed}
    rng = random.Random(f"reduce variants {point}")
    for _ in range(args.rounds):
        order = list(timed)
        rng.shuffle(order)
        for key in order:
            go = timed[key]
            go()
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(args.replays):
                go()
            e.record()
            torch.cuda.synchronize()
            times[key].append(s.elapsed_time(e) * 1e3 / (args.replays * GRAPH_LEN))
    del graphs, a_l, b_l, c_l
    torch.cuda.empty_cache()
    return dict(
        label=label,
        tile=spec,
        shape=[M, N, K],
        S=S,
        n=n,
        bias=with_bias,
        readers=readers,
        rs_r=rs_r,
        rs_all_r=rs_all_r,
        rs_all_e=rs_all_e,
        median={f"{m}|{k}": statistics.median(v) for (m, k), v in times.items()},
        spread={
            f"{m}|{k}": max(v) / min(v) - 1 for (m, k), v in times.items()
        },
    )


def report(paths):
    rows = []
    for path in paths:
        with open(path) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    tile_rank = {t: i for i, t in enumerate(TILES)}
    rows.sort(key=lambda r: (tile_rank[r["tile"]], r["label"], r["shape"][2], r["n"]))
    modes = sorted({k.split("|")[0] for r in rows for k in r["median"]})
    cols = [
        "prod rs",
        "prod owner",
        "tree2",
        "tree2 loop",
        "tree4",
        "tree4 loop",
        "sep",
        "sep R",
        "rs all",
        "rs all packed",
        "store_only",
    ]
    for mode in modes:
        print(f"## {mode}: median us per call, ratios to the copied rs\n")
        print(
            "| point | tile | shape | S | n | S n / 256 | R sep / all / rs | rs us | "
            + " | ".join(f"{c} / rs" for c in cols)
            + " | budget us | reduce_only us | sep gap us | max spread |"
        )
        print("|" + "---|" * (12 + len(cols)))
        for r in rows:
            med = {k.split("|")[1]: v for k, v in r["median"].items() if k.startswith(mode)}
            spr = [v for k, v in r["spread"].items() if k.startswith(mode)]
            rs = med["rs"]
            ratios = [f"{med[c] / rs:.3f}" if c in med else "" for c in cols]
            budget = rs - med["store_only"]
            gap = med["sep"] - med["store_only"] - med["reduce_only"]
            print(
                f"| {r['label']}{' bias' if r['bias'] else ''} | {r['tile']} | "
                f"{'x'.join(map(str, r['shape']))} | {r['S']} | {r['n']} | "
                f"{r['S'] * r['n'] / 256:.2f} | "
                f"{r['readers']} / {r.get('rs_all_r', r['rs_r'])} / {r['rs_r']} | "
                f"{rs:.2f} | " + " | ".join(ratios)
                + f" | {budget:.2f} | {med['reduce_only']:.2f} | {gap:.2f} | "
                f"{100 * max(spr):.1f}% |"
            )
        print()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--set", default="focused", choices=("focused", "wide"))
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--replays", type=int, default=5)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--out")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--report", nargs="*")
    args = ap.parse_args()
    if args.report:
        report(args.report)
        return
    points = point_set(args.set)
    if args.list:
        for p in points:
            print(p)
        print(len(points), "points")
        return
    points = points[args.shard :: args.shards]

    from v30_streamk_streamlined.hardware import query_device_config

    device = query_device_config()
    gpu = os.environ.get("HIP_VISIBLE_DEVICES", "all")
    lock = open(f"/tmp/v27_check_gpu{gpu}.lock", "w")
    out = open(args.out, "a") if args.out else sys.stdout
    for point in points:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            row = time_point(point, args, device)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
        out.write(json.dumps(row) + "\n")
        out.flush()
        print(f"done {point}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
