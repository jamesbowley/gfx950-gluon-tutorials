#!/usr/bin/env python3
"""TFLOPS of the persistent a16w16 kernels with bias: original epilogue bias vs bias in the
accumulator init, on shapes from lixun_aiter_losses.csv with bias switched on.

    PYTHONPATH=<triton gfx950-tutorial-v2.2>/python:<aiter> HIP_VISIBLE_DEVICES=3 \
    python perf_bias.py [--shapes 12] [--rounds 3]

"orig" is orig/gemm_a16w16_persistent_orig.py (AITER's port before the change), "new" is
the file in the AITER checkout. Same bootstrap as check_spills.py's m1 stack; timing is
triton.testing.do_bench_cudagraph, `--rounds` interleaved rounds, median (as in
run_aiter_compare.py). Rows are appended to results/<tag>.csv.
"""

import argparse
import csv
import importlib.util
import os
import statistics

import check_spills

HERE = check_spills.HERE
SHAPES_CSV = os.path.join(HERE, "..", "aiter_256x256_loss_investigation", "lixun_aiter_losses.csv")
ORIG = os.path.join(HERE, "orig", "gemm_a16w16_persistent_orig.py")
# Spread over the few-tile / partial-wave / full-wave buckets and K.
DEFAULT_SHAPES = [
    (2048, 2304, 16384),
    (16384, 512, 8192),
    (16384, 1024, 7168),
    (4096, 8192, 1024),
    (8192, 5120, 25600),
    (8192, 8192, 8192),
    (10240, 8448, 7168),
    (16384, 8192, 1024),
    (32768, 5120, 640),
    (32768, 7168, 1792),
    (16384, 16384, 4096),
    (32768, 8192, 7168),
]
KERNELS = ["v10", "v11", "v12", "v13"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", type=int, default=len(DEFAULT_SHAPES))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--shape", nargs="+", default=None, help="MxNxK, instead of the default list")
    ap.add_argument("--tag", default="perf_bias", help="results/<tag>.csv")
    args = ap.parse_args()
    out_csv = os.path.join(HERE, "results", f"{args.tag}.csv")
    shapes = (
        [tuple(int(v) for v in s.split("x")) for s in args.shape]
        if args.shape
        else DEFAULT_SHAPES[: args.shapes]
    )

    check_spills._bare_aiter_package()
    check_spills._deepbind_triton_and_aiter_plugin()
    import torch
    import triton

    from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic import gemm_a16w16_persistent as new
    from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.gemm_a16w16 import (
        gemm_a16w16_compute_bound,
    )

    spec = importlib.util.spec_from_file_location("gemm_a16w16_persistent_orig", ORIG)
    orig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orig)

    with open(SHAPES_CSV, newline="") as f:
        outs = {(int(r["M"]), int(r["N"]), int(r["K"])): r["out"].strip() for r in csv.DictReader(f)}

    done = set()
    if os.path.exists(out_csv):
        with open(out_csv, newline="") as f:
            done = {(int(r["M"]), int(r["N"]), int(r["K"])) for r in csv.DictReader(f)}

    for M, N, K in shapes:
        if (M, N, K) in done:
            continue
        otype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[outs.get((M, N, K), "bfloat16")]
        torch.manual_seed(0)
        x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
        w = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
        bias = torch.randn((N,), dtype=torch.bfloat16, device="cuda")
        y = torch.empty((M, N), dtype=otype, device="cuda")

        fns = {
            "v9_nobias": lambda: gemm_a16w16_compute_bound(x, w, y, None),
            "v9_bias": lambda: gemm_a16w16_compute_bound(x, w, y, bias),
        }
        for k in KERNELS:
            key = f"compute_bound_{k}"
            fns[f"{k}_nobias"] = lambda f=new._PERSISTENT_KERNEL_MAP[key]: f(x, w, y, None)
            fns[f"{k}_bias_orig"] = lambda f=orig._PERSISTENT_KERNEL_MAP[key]: f(x, w, y, bias)
            fns[f"{k}_bias_new"] = lambda f=new._PERSISTENT_KERNEL_MAP[key]: f(x, w, y, bias)

        ref = torch.nn.functional.linear(x.float(), w.float(), bias.float()).to(otype).float()
        ok = {}
        for name, fn in fns.items():
            if name.endswith("nobias"):
                continue
            out = fn().float()
            torch.cuda.synchronize()
            ok[name] = bool(torch.allclose(out, ref, atol=1e-1, rtol=1e-2))
        del ref

        times = {name: [] for name in fns}
        for _ in range(args.rounds):
            for name, fn in fns.items():
                times[name].append(triton.testing.do_bench_cudagraph(fn))
        flops = 2.0 * M * N * K
        row = dict(M=M, N=N, K=K, out=str(otype).split(".")[-1], tiles=(M // 256) * (N // 256))
        for name, t in times.items():
            row[name] = round(flops / statistics.median(t) * 1e-9, 1)
        row["all_bias_correct"] = all(ok.values())
        new_file = not os.path.exists(out_csv)
        with open(out_csv, "a", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(row))
            if new_file:
                wr.writeheader()
            wr.writerow(row)
        print(
            f"{M}x{N}x{K}: "
            + " ".join(f"{k}: {row[f'{k}_nobias']}/{row[f'{k}_bias_orig']}/{row[f'{k}_bias_new']}" for k in KERNELS)
            + f" v9: {row['v9_nobias']}/{row['v9_bias']} correct={row['all_bias_correct']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
