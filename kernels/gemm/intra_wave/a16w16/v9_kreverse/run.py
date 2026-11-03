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
"""One v9_kreverse workload: rotating (a, b, c) copies, launched back to back.

    python v9_kreverse/run.py --shape 8192x8192x16384 --korder 1 --iters 1000      # under rocprofv3
    python v9_kreverse/run.py --shape 8192x8192x16384 --korder 1 --seconds 15 \
        --telemetry out.json                                                       # power / clock

--telemetry samples `amd-smi metric` for the amd-smi device $SMI_GPU during a sustained run (after
a warm-up of the same length as --warmup-seconds) and writes the energy per launch, mean socket
power, mean GFX clock, mean UMC (HBM) activity and the sustained TFLOPS as JSON.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import bench_ext as bench  # noqa: E402  (plugin hooks; imports triton before torch)
import torch  # noqa: E402
from v9_kreverse import matmul_kernel as v9k  # noqa: E402


def smi(*flags):
    out = subprocess.run(["amd-smi", "metric", "-g", os.environ.get("SMI_GPU", "1"), *flags, "--json"],
                         capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    return data["gpu_data"][0] if isinstance(data, dict) else data[0]


def sample():
    d = smi("-p", "-c", "-u")
    clks = [v["clk"]["value"] for k, v in d["clock"].items() if k.startswith("gfx_") and isinstance(v, dict)
            and isinstance(v.get("clk"), dict) and isinstance(v["clk"].get("value"), (int, float))]
    return {
        "power_w": d["power"]["socket_power"]["value"],
        "gfx_mhz": sum(clks) / len(clks),
        "umc_pct": d["usage"]["umc_activity"]["value"],
        "gfx_pct": d["usage"]["gfx_activity"]["value"],
    }


def energy_j():
    return smi("-E")["energy"]["total_energy_consumption"]["value"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", required=True, help="MxNxK")
    p.add_argument("--korder", type=int, required=True, choices=[0, 1, 2])
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--rotating-mb", type=int, default=4096)
    p.add_argument("--seconds", type=float, default=None, help="sustained run instead of --iters")
    p.add_argument("--warmup-seconds", type=float, default=5.0)
    p.add_argument("--telemetry", default=None, help="JSON output for the sustained run")
    args = p.parse_args()
    M, N, K = map(int, args.shape.split("x"))

    a_l, b_l, c_l, n = bench.gen_rotating_tensors(M, N, K, torch.bfloat16, args.rotating_mb)
    launch = lambda i: v9k.matmul(a_l[i % n], b_l[i % n], c_l[i % n], k_order=args.korder)  # noqa: E731
    launch(0)
    torch.cuda.synchronize()

    if args.seconds is None:
        for i in range(args.iters):
            launch(i)
        torch.cuda.synchronize()
        print(f"{args.shape} K_ORDER={args.korder}: {args.iters} launches, {n} rotating copies")
        return

    # Calibrate a batch of about 0.2 s so the host never runs dry and the loop rarely syncs.
    t0 = time.time()
    for i in range(10):
        launch(i)
    torch.cuda.synchronize()
    batch = max(10, int(0.2 / ((time.time() - t0) / 10)))

    end = time.time() + args.warmup_seconds
    i = 0
    while time.time() < end:
        for _ in range(batch):
            launch(i)
            i += 1
        torch.cuda.synchronize()

    samples, stop = [], threading.Event()

    def sampler():
        while not stop.is_set():
            try:
                samples.append(sample())
            except Exception as e:  # noqa: BLE001
                print(f"telemetry sample failed: {e}", file=sys.stderr)

    e0 = energy_j()
    start_ev, end_ev = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    start_ev.record()
    launches = 0
    end = time.time() + args.seconds
    while time.time() < end:
        for _ in range(batch):
            launch(i)
            i += 1
        launches += batch
        torch.cuda.synchronize()
    end_ev.record()
    torch.cuda.synchronize()
    stop.set()
    th.join()
    e1 = energy_j()
    ms = start_ev.elapsed_time(end_ev)

    mean = lambda k: sum(s[k] for s in samples) / len(samples)  # noqa: E731
    res = {
        "shape": args.shape,
        "korder": args.korder,
        "launches": launches,
        "rotating_copies": n,
        "ms_per_launch": ms / launches,
        "tflops": 2 * M * N * K * launches / (ms * 1e-3) / 1e12,
        "energy_j_per_launch": (e1 - e0) / launches,
        "power_w": mean("power_w"),
        "gfx_mhz": mean("gfx_mhz"),
        "umc_pct": mean("umc_pct"),
        "gfx_pct": mean("gfx_pct"),
        "samples": len(samples),
    }
    with open(args.telemetry, "w") as f:
        json.dump(res, f)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
