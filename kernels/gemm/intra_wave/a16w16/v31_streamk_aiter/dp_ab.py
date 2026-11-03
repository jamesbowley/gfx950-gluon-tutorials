"""v31's data-parallel launch with one tile per program (a one-trip persistent loop)
against v9 (v9_ref.py).

    python v31_streamk_aiter/dp_ab.py asm  > v31_streamk_aiter/results/dp_asm.md
    python v31_streamk_aiter/dp_ab.py time > v31_streamk_aiter/results/dp_timing.md

asm: VGPRs, spills, SGPRs and the async-copy (LDS) loads of each build. time: each
variant as a graph of GRAPH_LEN launches over rotating inputs (cold caches), replayed
in shuffled rounds; median us per launch. v9_ref supports BLOCK_K 64 only.
"""

import argparse
import fcntl
import os
import random
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

GRAPH_LEN = 100
ROTATING_MB = 2048
TILES = ((256, 256), (256, 128), (128, 256), (128, 128), (64, 64))
VARIANTS = ("v9", "v31")


def shapes(bm, bn):
    """(label, M, N, K, bias): one full wave, a ragged 3/4 wave, ragged K, bias."""
    out = []
    for K in (1024, 4096, 8192):
        out += [
            ("wave", 16 * bm, 16 * bn, K, False),
            ("ragged", 12 * bm - bm // 8, 16 * bn - 8, K, False),
            ("kres", 16 * bm, 16 * bn, K - 48, False),
            ("bias", 16 * bm, 16 * bn, K, True),
        ]
    return out


def v9_module():
    import aiter.ops.triton.utils.gemm_config_utils as g

    # v9_ref picks a default tile through pick_gemm_tile, which only the v9 AITER
    # branch has; every call here names the tile.
    if not hasattr(g, "pick_gemm_tile"):
        g.pick_gemm_tile = lambda *args, **kwargs: None
    from v31_streamk_aiter import v9_ref

    return v9_ref


def runners(bm, bn, bk=64):
    from v31_streamk_aiter import matmul_kernel as mk

    v9 = v9_module()
    config = {
        "BLOCK_M": bm,
        "BLOCK_N": bn,
        "BLOCK_K": bk,
        "GROUP_SIZE_M": 4,
        "STREAMK_POLICY": "dp",
        "STREAMK_FIXUP": "owner",
        "SPLIT_N": None,
        "STREAMK_EXCHANGE": "wt",
    }
    return {
        "v9": lambda x, w, y, bias: v9.gemm_a16w16_compute_bound(x, w, y, bias, tile=(bm, bn)),
        "v31": lambda x, w, y, bias: mk.gemm_a16w16(x, w, bias, dtype=y.dtype, y=y, config=config),
    }


class Capture:
    def __init__(self, module, name):
        self.module, self.name = module, name
        self.kernel = getattr(module, name)
        self.compiled = None

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def run(*args, **kwargs):
            self.compiled = launch(*args, **kwargs)
            return self.compiled

        return run

    def __enter__(self):
        setattr(self.module, self.name, self)
        return self

    def __exit__(self, *exc):
        setattr(self.module, self.name, self.kernel)


def asm_mode():
    import torch
    from v31_streamk_aiter import matmul_kernel as mk

    v9 = v9_module()
    print("# v31 one-tile-per-program dp (one-trip persistent loop) against v9\n")
    print("| tile | shape | variant | VGPRs | spills | SGPRs | async LDS loads |")
    print("|---|---|---|---|---|---|---|")
    for bm, bn in TILES:
        for label, M, N, K, bias in (shapes(bm, bn)[4 + 0], shapes(bm, bn)[4 + 1], shapes(bm, bn)[4 + 3]):
            x = torch.rand((M, K), device="cuda", dtype=torch.bfloat16) - 0.5
            w = torch.rand((N, K), device="cuda", dtype=torch.bfloat16) - 0.5
            b = torch.rand((N,), device="cuda", dtype=torch.bfloat16) if bias else None
            for variant, run in runners(bm, bn).items():
                module = v9 if variant == "v9" else mk
                with Capture(module, "_gemm_a16w16_compute_bound_kernel") as cap:
                    run(x, w, torch.empty((M, N), device="cuda", dtype=torch.bfloat16), b)
                asm = cap.compiled.asm["amdgcn"]
                vgprs = re.search(r"\.vgpr_count:\s+(\d+)", asm).group(1)
                spills = re.search(r"\.vgpr_spill_count:\s+(\d+)", asm).group(1)
                sgprs = re.search(r"\.sgpr_count:\s+(\d+)", asm).group(1)
                loads = len(re.findall(r"^\s*buffer_load_dword\S*.*\blds\b", asm, re.MULTILINE))
                print(f"| {bm}x{bn}x64 | {label} {M}x{N}x{K} | {variant} | {vgprs} | {spills} | {sgprs} | {loads} |")
            torch.cuda.synchronize()


def time_mode(rounds, replays):
    import bench_ext as bench
    import torch

    dt = torch.bfloat16
    graphs, keep, points = {}, [], []
    for bm, bn in TILES:
        fns = runners(bm, bn)
        for label, M, N, K, with_bias in shapes(bm, bn):
            point = (f"{bm}x{bn}x64", label, M, N, K)
            points.append(point)
            x = torch.rand((M, K), device="cuda", dtype=dt) - 0.5
            w = (torch.rand((N, K), device="cuda", dtype=dt) - 0.5)
            bias = (torch.rand((N,), device="cuda", dtype=dt) - 0.5) if with_bias else None
            ref = x.float() @ w.float().T
            if bias is not None:
                ref += bias.float()
            ref = ref.to(dt)
            a_l, b_l, c_l, copies = bench.gen_rotating_tensors(M, N, K, dt, ROTATING_MB)
            keep.append((a_l, b_l, c_l, bias))
            for variant, fn in fns.items():
                y = torch.empty((M, N), device="cuda", dtype=dt)
                fn(x, w, y, bias)
                torch.cuda.synchronize()
                assert torch.allclose(y, ref, atol=1e-1, rtol=1e-2), f"{point} {variant}: wrong result"
                for i in range(copies):
                    fn(a_l[i], b_l[i].T, c_l[i], bias)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for i in range(GRAPH_LEN):
                        fn(a_l[i % copies], b_l[i % copies].T, c_l[i % copies], bias)
                for _ in range(3):
                    g.replay()
                torch.cuda.synchronize()
                graphs[(point, variant)] = g
    gpu = os.environ.get("HIP_VISIBLE_DEVICES", "all")
    lock = open(f"/tmp/v27_check_gpu{gpu}.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    times = {k: [] for k in graphs}
    rng = random.Random("v31 dp")
    for _ in range(rounds):
        order = list(graphs)
        rng.shuffle(order)
        for key in order:
            g = graphs[key]
            g.replay()
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(replays):
                g.replay()
            e.record()
            torch.cuda.synchronize()
            times[key].append(s.elapsed_time(e) * 1e3 / (replays * GRAPH_LEN))
    fcntl.flock(lock, fcntl.LOCK_UN)
    print(f"# v31 one-tile-per-program dp: {rounds} rounds of {replays} x {GRAPH_LEN} launches, "
          f"median us per launch, HIP {gpu}\n")
    print("| tile | case | shape | v9 us | v31 us | v31 / v9 | spread |")
    print("|---|---|---|---|---|---|---|")
    ratios_v9 = []
    for point in points:
        tile, label, M, N, K = point
        med = {v: statistics.median(times[(point, v)]) for v in VARIANTS}
        spread = max(100 * (max(times[(point, v)]) / min(times[(point, v)]) - 1) for v in VARIANTS)
        ratios_v9.append(med["v31"] / med["v9"])
        print(f"| {tile} | {label} | {M}x{N}x{K} | {med['v9']:.2f} | {med['v31']:.2f} | "
              f"{med['v31'] / med['v9']:.3f} | {spread:.1f}% |")
    gm = lambda r: statistics.geometric_mean(r)
    print(f"\nv31 / v9: geomean {gm(ratios_v9):.4f}, min {min(ratios_v9):.3f}, max {max(ratios_v9):.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["asm", "time"])
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--replays", type=int, default=5)
    args = ap.parse_args()
    if args.mode == "asm":
        asm_mode()
    else:
        time_mode(args.rounds, args.replays)


if __name__ == "__main__":
    main()
