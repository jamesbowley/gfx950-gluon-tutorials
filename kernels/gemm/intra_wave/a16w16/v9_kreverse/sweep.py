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
"""Interleaved K-order measurements of v9_kreverse (run.py) on the study shapes.

    source v9_kreverse/env.sh
    python v9_kreverse/sweep.py trace     --reps 5 --out v9_kreverse/results/trace.csv
    python v9_kreverse/sweep.py counters  --out v9_kreverse/results/counters.csv
    python v9_kreverse/sweep.py telemetry --reps 3 --out v9_kreverse/results/telemetry.csv

Every repetition visits each shape once and runs the three K orders back to back, starting from
a different order each time, so drift and position effects spread over the orders.

trace:     rocprofv3 --kernel-trace around 1000 launches; mean of the last 100 dispatches.
counters:  rocprofv3 --pmc around 20 launches; mean over launches 5 and on.
telemetry: a sustained run (run.py --seconds) with amd-smi energy, power, clock and UMC activity.
"""

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from shapes import SHAPES  # noqa: E402

RUN = os.path.join(HERE, "run.py")
KERNEL = "v9_kreverse"
COUNTER_SETS = [
    ["TCC_HIT_sum", "TCC_MISS_sum", "TCC_EA0_RDREQ_sum", "TCC_EA0_RDREQ_DRAM_sum"],
    ["TCC_EA0_RDREQ_LEVEL_sum", "TCC_EA0_RDREQ_sum", "TCC_EA0_RDREQ_DRAM_CREDIT_STALL_sum", "GRBM_GUI_ACTIVE"],
]


def shape_str(s):
    return "x".join(map(str, s))


def orders(rep):
    return [(rep + j) % 3 for j in range(3)]


def find(trace_dir, suffix):
    files = glob.glob(os.path.join(trace_dir, "**", f"*_{suffix}.csv"), recursive=True)
    if not files:
        raise RuntimeError(f"no *_{suffix}.csv under {trace_dir}")
    return max(files, key=os.path.getmtime)


def rocprof(args, tmp, shape, korder, iters, extra):
    d = os.path.join(tmp, f"{shape_str(shape)}_k{korder}")
    shutil.rmtree(d, ignore_errors=True)
    cmd = ["rocprofv3", *extra, "--output-format", "csv", "-d", d, "--",
           sys.executable, RUN, "--shape", shape_str(shape), "--korder", str(korder),
           "--iters", str(iters), "--rotating-mb", str(args.rotating_mb)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return d


def trace(args):
    rows = []
    for rep in range(args.reps):
        for shape in SHAPES:
            for korder in orders(rep):
                d = rocprof(args, args.tmp, shape, korder, 1000, ["--kernel-trace"])
                durs = []
                with open(find(d, "kernel_trace")) as f:
                    for r in csv.DictReader(f):
                        if KERNEL in r["Kernel_Name"]:
                            durs.append(int(r["End_Timestamp"]) - int(r["Start_Timestamp"]))
                us = sum(durs[-100:]) / len(durs[-100:]) / 1e3
                M, N, K = shape
                row = {"rep": rep, "shape": shape_str(shape), "korder": korder, "us": round(us, 3),
                       "tflops": round(2 * M * N * K / (us * 1e-6) / 1e12, 2), "launches": len(durs)}
                rows.append(row)
                print(row, flush=True)
                write(args.out, rows)


def counters(args):
    rows = []
    for shape in SHAPES:
        for korder in (0, 1, 2):
            row = {"shape": shape_str(shape), "korder": korder}
            for i, cs in enumerate(COUNTER_SETS):
                d = rocprof(args, os.path.join(args.tmp, f"set{i}"), shape, korder, 20, ["--pmc", *cs])
                vals = {}
                with open(find(d, "counter_collection")) as f:
                    for r in csv.DictReader(f):
                        if KERNEL in r["Kernel_Name"]:
                            vals.setdefault(r["Counter_Name"], {}).setdefault(r["Dispatch_Id"], 0.0)
                            vals[r["Counter_Name"]][r["Dispatch_Id"]] += float(r["Counter_Value"])
                for name, per in vals.items():
                    v = [per[k] for k in sorted(per, key=int)][5:]
                    row[name] = round(sum(v) / len(v))
            rows.append(row)
            print(row, flush=True)
            write(args.out, rows)


def telemetry(args):
    rows = []
    out_json = os.path.join(args.tmp, "telemetry.json")
    os.makedirs(args.tmp, exist_ok=True)
    for rep in range(args.reps):
        for shape in SHAPES:
            for korder in orders(rep):
                subprocess.run([sys.executable, RUN, "--shape", shape_str(shape), "--korder", str(korder),
                                "--seconds", str(args.seconds), "--warmup-seconds", str(args.warmup),
                                "--rotating-mb", str(args.rotating_mb), "--telemetry", out_json],
                               check=True, stdout=subprocess.DEVNULL)
                with open(out_json) as f:
                    row = {"rep": rep, **json.load(f)}
                rows.append(row)
                print(row, flush=True)
                write(args.out, rows)


def write(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["trace", "counters", "telemetry"])
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--out", required=True)
    p.add_argument("--tmp", default=os.path.join(HERE, "tmp"))
    p.add_argument("--rotating-mb", type=int, default=4096)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--warmup", type=float, default=4.0)
    p.add_argument("--shapes", default=None, help="comma-separated MxNxK list instead of shapes.SHAPES")
    args = p.parse_args()
    if args.shapes:
        SHAPES[:] = [tuple(map(int, s.split("x"))) for s in args.shapes.split(",")]
    args.tmp = os.path.join(args.tmp, args.mode)
    {"trace": trace, "counters": counters, "telemetry": telemetry}[args.mode](args)


if __name__ == "__main__":
    main()
