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
"""Energy per launch and average socket power for any versions, from amd-smi's energy counter
(v17's power_check.py with a version list).

    cd kernels/gemm/intra_wave/a16w16
    HIP_VISIBLE_DEVICES=2 SMI_GPU=2 python v20_streamk_reduce_scatter/power_check.py 4096x7168x32768 13,16,20:STREAMK_POLICY=spread

Each run launches the kernel back to back on the same inputs for 15 s; energy per launch is
the change in amd-smi's total-energy counter divided by the launches, and average power is
that energy over the wall time. SMI_GPU must be the amd-smi index of the device
HIP_VISIBLE_DEVICES selects (check that power rises under load). Versions are specs as in
version_spec.py. Order: idle, every version, every version again, idle, to expose drift.
"""

import importlib
import json
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench  # noqa: E402,F401  (plugin hooks)
import torch  # noqa: E402

from v20_streamk_reduce_scatter import version_spec  # noqa: E402

GPU = os.environ.get("SMI_GPU", "2")
M, N, K = map(int, sys.argv[1].split("x"))
SPECS = sys.argv[2].split(",")
SECONDS = 15.0


def smi():
    out = subprocess.run(["amd-smi", "metric", "-g", GPU, "-E", "-p", "--json"], capture_output=True, text=True).stdout
    d = json.loads(out)["gpu_data"][0]
    return d["energy"]["total_energy_consumption"]["value"], d["power"]["socket_power"]["value"]


a = torch.rand((M, K), device="cuda", dtype=torch.bfloat16) - 0.5
b = torch.rand((N, K), device="cuda", dtype=torch.bfloat16).T - 0.5
c = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
runs = {}
for spec in SPECS:
    v, env = version_spec.parse(spec)
    runs[spec] = (importlib.import_module(f"{bench.VERSION_MAP[v]}.matmul_kernel").matmul, env)
    with version_spec.environment(env):
        for _ in range(20):
            runs[spec][0](a, b, c)
torch.cuda.synchronize()


def run(label, f, env):
    samples, stop = [], threading.Event()

    def sampler():
        while not stop.is_set():
            samples.append(smi()[1])
            time.sleep(0.2)

    e0, _ = smi()
    t0 = time.perf_counter()
    th = threading.Thread(target=sampler)
    th.start()
    n = 0
    with version_spec.environment(env):
        if f is None:
            time.sleep(SECONDS)
        else:
            while time.perf_counter() - t0 < SECONDS:
                for _ in range(20):
                    f(a, b, c)
                n += 20
                torch.cuda.synchronize()
    t1 = time.perf_counter()
    stop.set()
    th.join()
    e1, _ = smi()
    dt, de = t1 - t0, e1 - e0
    us = dt / n * 1e6 if n else 0
    per = de / n * 1e3 if n else 0
    pk = sorted(samples)[len(samples) // 2] if samples else 0
    print(f"| {label} | {n} | {us:.1f} | {de / dt:.0f} | {pk} | {per:.1f} |", flush=True)


print(f"shape {M}x{N}x{K}, amd-smi GPU {GPU}, {SECONDS:.0f} s per run\n")
print("| run | launches | wall us per launch | average power (W, energy / time) | median sampled socket power (W) "
      "| energy per launch (mJ) |")
print("|---|---|---|---|---|---|")
run("idle", None, {})
for rep in range(2):
    for spec in SPECS:
        run(f"{version_spec.label(spec)} (rep {rep + 1})", *runs[spec])
run("idle", None, {})
