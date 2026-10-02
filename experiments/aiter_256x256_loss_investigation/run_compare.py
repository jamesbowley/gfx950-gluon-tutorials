#!/usr/bin/env python3
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
"""v9 and the persistent v10-v13 (v9 tile order) on the shapes in lixun_aiter_losses.csv.

    HIP_VISIBLE_DEVICES=3 python run_compare.py --mode time --versions 9
    HIP_VISIBLE_DEVICES=3 python run_compare.py --mode time --warm --versions 9
    HIP_VISIBLE_DEVICES=3 python run_compare.py --mode time --versions 10 11 12 13
    HIP_VISIBLE_DEVICES=3 python run_compare.py --mode counters --versions 9 10 11 12 13
    python run_compare.py --summarize

Each (shape, version) is one subprocess running bench.py's own correctness check and rocprof loop
(1000 dispatches over rotating tensors) under rocprofv3. Time is the mean of the final 100
dispatches. --warm passes a zero rotating buffer, so every dispatch reuses one tensor set, like
do_bench without a cache flush.

The persistent kernels' matmul() launches 256 programs whatever the tile count. The worker
replaces the kernel handle in the imported module so the launch uses min(256, tiles) programs,
leaving the kernel sources untouched. Compiles go to ./cache so they never share a Triton cache
with other jobs.
"""

import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
A16W16 = os.path.join(REPO, "kernels", "gemm", "intra_wave", "a16w16")
SHAPES_CSV = os.path.join(HERE, "lixun_aiter_losses.csv")
RESULTS = os.path.join(HERE, "results")
CONFIG = "llir+force-agpr+amdgcnas"
PMC = ["TCC_HIT_sum", "TCC_MISS_sum", "TCC_REQ_sum"]
PERSISTENT = (10, 11, 12, 13)

sys.path.insert(0, os.path.join(REPO, "scripts"))
from run_perf_table import CONFIG_ENV, VERSION_MAP, avg_kernel_time_ns, find_kernel_trace_csv  # noqa: E402


def load_shapes():
    with open(SHAPES_CSV, newline="") as f:
        return [
            dict(
                M=int(r["M"]),
                N=int(r["N"]),
                K=int(r["K"]),
                tiles=int(r["tiles"]),
                out=r["out"],
                gluon=float(r["gluon_tflops"]),
                aiter=float(r["aiter_pick_tflops"]),
                aiter_pick=r["aiter_pick"],
            )
            for r in csv.DictReader(f)
        ]


# ---------------------------------------------------------------- worker (runs under rocprofv3)


class ClampedGrid:
    """Stands in for a persistent kernel handle: launches min(NUM_PROGRAMS, tiles) programs."""

    def __init__(self, kernel):
        self.kernel = kernel

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            M, N = args[3], args[4]
            tiles = -(-M // kwargs["BLOCK_M"]) * -(-N // kwargs["BLOCK_N"])
            assert grid[0] == kwargs["NUM_PROGRAMS"], (grid, kwargs["NUM_PROGRAMS"])
            programs = min(kwargs["NUM_PROGRAMS"], tiles)
            kwargs["NUM_PROGRAMS"] = programs
            return self.kernel[(programs,) + tuple(grid[1:])](*args, **kwargs)

        return launch


def worker(args):
    sys.path.insert(0, A16W16)
    import importlib

    import bench  # installs the llir/amdgcnas hooks from the environment

    version_dir = VERSION_MAP[args.version]
    module = importlib.import_module(f"{version_dir}.matmul_kernel")
    if args.version in PERSISTENT:
        setattr(module, version_dir, ClampedGrid(getattr(module, version_dir)))
    sizes = [(args.M, args.N, args.K)]
    bench.test_correctness(module.matmul, "bf16", sizes, version_dir)
    bench.run_rocprof_iterations(
        module.matmul,
        ["bf16"],
        sizes,
        version_dir,
        n_iters=args.iters,
        rotating_buffer_size_mb=args.rotating_buffer_size,
    )


# ---------------------------------------------------------------- driver


def run_env():
    env = os.environ.copy()
    for key in (
        "LLVM_PASS_PLUGIN_PATH",
        "LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE",
        "TRITON_FORCE_MFMA_AGPR",
        "TRITON_AMDGCNAS_PLUGIN",
        "TRITON_ENABLE_LLIR_SCHED",
        "TRITON_ENABLE_AMDGCN_AS",
    ):
        env.pop(key, None)
    env.update(CONFIG_ENV[CONFIG])
    env["PERSISTENT_TILE_ORDER"] = "v9"
    env["TRITON_CACHE_DIR"] = os.path.join(HERE, "cache")
    env["AMD_SERIALIZE_KERNEL"] = "3"
    return env


def profile(profiler_args, version, shape, iters, rotating_mb, log_path):
    """Run one worker under rocprofv3; return (output dir, correctness string)."""
    out_dir = tempfile.mkdtemp(prefix="rocprof_", dir=RESULTS)
    worker_cmd = [
        sys.executable, os.path.abspath(__file__), "--worker",
        "--version", str(version), "--M", str(shape["M"]), "--N", str(shape["N"]),
        "--K", str(shape["K"]), "--iters", str(iters), "--rotating-buffer-size", str(rotating_mb),
    ]
    cmd = ["rocprofv3", *profiler_args, "--kernel-include-regex", VERSION_MAP[version],
           "-d", out_dir, "--output-format", "csv", "--", *worker_cmd]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=run_env(), cwd=A16W16)
    with open(log_path, "w") as f:
        f.write(proc.stdout + "\n" + proc.stderr)
    if "match" in proc.stdout:
        correct = "yes"
    elif "differ" in proc.stdout:
        correct = "NO"
    else:
        correct = "?"
    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-3:]
        print(f"    FAILED (exit {proc.returncode}): " + " | ".join(tail), flush=True)
        shutil.rmtree(out_dir, ignore_errors=True)
        return None, correct
    return out_dir, correct


def time_one(version, shape, warm):
    tag = f"v{version}_{shape['M']}x{shape['N']}x{shape['K']}_{'warm' if warm else 'cold'}"
    out_dir, correct = profile(["--kernel-trace"], version, shape, 1000, 0 if warm else 512,
                               os.path.join(RESULTS, "logs", f"time_{tag}.log"))
    if out_dir is None:
        return None
    csv_path = find_kernel_trace_csv(out_dir)
    avg_ns, count = avg_kernel_time_ns(csv_path, VERSION_MAP[version], last_n=100) if csv_path else (None, 0)
    shutil.rmtree(out_dir, ignore_errors=True)
    if avg_ns is None:
        print("    no matching dispatches", flush=True)
        return None
    flops = 2 * shape["M"] * shape["N"] * shape["K"]
    return dict(time_us=round(avg_ns / 1e3, 3), tflops=round(flops / avg_ns * 1e-3, 1),
                dispatches=count, correct=correct)


def counters_one(version, shape, iters):
    tag = f"v{version}_{shape['M']}x{shape['N']}x{shape['K']}"
    out_dir, correct = profile(["--pmc", *PMC], version, shape, iters, 512,
                               os.path.join(RESULTS, "logs", f"counters_{tag}.log"))
    if out_dir is None:
        return None
    per_dispatch = defaultdict(dict)
    for path in glob.glob(os.path.join(out_dir, "**", "*counter_collection.csv"), recursive=True):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                per_dispatch[int(r["Dispatch_Id"])][r["Counter_Name"]] = float(r["Counter_Value"])
    shutil.rmtree(out_dir, ignore_errors=True)
    # Drop the correctness and warmup dispatches.
    ids = sorted(per_dispatch)[2:]
    if not ids:
        print("    no matching dispatches", flush=True)
        return None
    vals = {c: sum(per_dispatch[i].get(c, 0.0) for i in ids) / len(ids) for c in PMC}
    hit, miss = vals["TCC_HIT_sum"], vals["TCC_MISS_sum"]
    return dict(l2_hit_rate=round(hit / max(1.0, hit + miss), 4),
                **{c: int(v) for c, v in vals.items()}, dispatches=len(ids), correct=correct)


def result_path(mode, warm):
    return os.path.join(RESULTS, "counters.csv" if mode == "counters" else
                        f"timing_{'warm' if warm else 'cold'}.csv")


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def append_row(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def drive(args):
    os.makedirs(os.path.join(RESULTS, "logs"), exist_ok=True)
    path = result_path(args.mode, args.warm)
    done = {(r["version"], r["M"], r["N"], r["K"]) for r in read_rows(path)}
    shapes = load_shapes()
    if args.shapes:
        shapes = shapes[: args.shapes]
    for version in args.versions:
        for i, s in enumerate(shapes):
            key = (str(version), str(s["M"]), str(s["N"]), str(s["K"]))
            if key in done:
                continue
            print(f"[{args.mode}{' warm' if args.warm else ''}] v{version} "
                  f"{s['M']}x{s['N']}x{s['K']} ({i + 1}/{len(shapes)})", flush=True)
            if args.mode == "time":
                res = time_one(version, s, args.warm)
            else:
                res = counters_one(version, s, args.counter_iters)
            if res is None:
                continue
            print(f"    {res}", flush=True)
            append_row(path, dict(version=version, M=s["M"], N=s["N"], K=s["K"],
                                  tiles=s["tiles"], **res))


# ---------------------------------------------------------------- summary


def summarize():
    shapes = load_shapes()
    cold = {(int(r["version"]), int(r["M"]), int(r["N"]), int(r["K"])): r
            for r in read_rows(result_path("time", False))}
    warm = {(int(r["version"]), int(r["M"]), int(r["N"]), int(r["K"])): r
            for r in read_rows(result_path("time", True))}
    ctr = {(int(r["version"]), int(r["M"]), int(r["N"]), int(r["K"])): r
           for r in read_rows(result_path("counters", False))}

    def tf(table, v, s):
        r = table.get((v, s["M"], s["N"], s["K"]))
        return float(r["tflops"]) if r else None

    def fmt(x, spec="{:.0f}"):
        return spec.format(x) if x is not None else "-"

    def pct(a, b):
        return f"{100 * (a / b - 1):+.1f}%" if a is not None and b else "-"

    lines = [
        "| Shape | Tiles | Waves | CSV gluon | CSV aiter | v9 warm | v9 cold | v10 | v11 | v12 | v13 "
        "| Best persistent vs v9 | v9 warm vs CSV gluon | Best cold vs CSV aiter | L2 hit v9 / v10 / v11 / v12 / v13 | Wrong |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in sorted(shapes, key=lambda s: s["tiles"]):
        v9 = tf(cold, 9, s)
        pers = {v: tf(cold, v, s) for v in PERSISTENT}
        best_p = max((x for x in pers.values() if x is not None), default=None)
        best = max((x for x in [v9, best_p] if x is not None), default=None)
        hits = [ctr.get((v, s["M"], s["N"], s["K"])) for v in (9, *PERSISTENT)]
        hit_str = " / ".join(f"{100 * float(h['l2_hit_rate']):.1f}" if h else "-" for h in hits)
        wrong = [f"v{r['version']}" for t in (cold, warm) for r in t.values()
                 if (int(r["M"]), int(r["N"]), int(r["K"])) == (s["M"], s["N"], s["K"])
                 and r["correct"] != "yes"]
        shape = f"{s['M']}x{s['N']}x{s['K']}" + (" (fp32 out in CSV)" if s["out"] != "bfloat16" else "")
        lines.append(
            f"| {shape} | {s['tiles']} | {s['tiles'] / 256:.2f} | {s['gluon']:.0f} | {s['aiter']:.0f} "
            f"| {fmt(tf(warm, 9, s))} | {fmt(v9)} | " + " | ".join(fmt(pers[v]) for v in PERSISTENT)
            + f" | {pct(best_p, v9)} | {pct(tf(warm, 9, s), s['gluon'])} | {pct(best, s['aiter'])} "
            f"| {hit_str} | {' '.join(sorted(set(wrong))) or '-'} |"
        )
    text = "\n".join(lines) + "\n"
    with open(os.path.join(RESULTS, "summary.md"), "w") as f:
        f.write(text)
    print(text)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--mode", choices=("time", "counters"), default="time")
    p.add_argument("--warm", action="store_true", help="time with one tensor set (no cache clearing)")
    p.add_argument("--versions", type=int, nargs="+", default=[9, *PERSISTENT])
    p.add_argument("--shapes", type=int, default=None, help="only the first N shapes of the CSV")
    p.add_argument("--counter-iters", type=int, default=200)
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--version", type=int)
    p.add_argument("--M", type=int)
    p.add_argument("--N", type=int)
    p.add_argument("--K", type=int)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--rotating-buffer-size", type=int, default=512)
    args = p.parse_args()
    if args.worker:
        worker(args)
    elif args.summarize:
        summarize()
    else:
        drive(args)


if __name__ == "__main__":
    main()
