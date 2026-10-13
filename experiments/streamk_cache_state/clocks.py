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
"""Shader clock and board power while a GEMM and one mode's work loop, from sysfs.

    export PYTHONPATH=<repo>/scripts/triton_deepbind_shim:<triton_gfx950-tutorial-v2.2>/python
    HIP_VISIBLE_DEVICES=0 python experiments/streamk_cache_state/clocks.py 5632x3072x6144,18432x14592x8192

For each (version, mode) the GEMM and the mode's work (modes.py) loop for about 2 s while a
thread reads the device's hwmon freq1_input (sclk) and power1_input about every millisecond.
The SMU updates these on its own period, so they give the clock the GPU settles at under each
mode, not per-kernel values. Rows go to results/clocks.jsonl.
"""

import glob
import importlib
import json
import os
import statistics
import sys
import threading
import time

import modes as M

LOOP_S = 2.0
MODES = ["nothing", "sleep", "busy", "clean_cold", "clean_warm", "clean_warm_idle", "clean_warm_busy", "dirty_cold"]


def hwmon(bus):
    d = glob.glob(f"/sys/bus/pci/devices/0000:{bus:02x}:00.0/hwmon/hwmon*")[0]
    return os.path.join(d, "freq1_input"), os.path.join(d, "power1_input")


def sample(paths, stop, out):
    fq, pw = paths
    while not stop.is_set():
        with open(fq) as f:
            mhz = int(f.read()) / 1e6
        with open(pw) as f:
            w = int(f.read()) / 1e6
        out.append((mhz, w))
        time.sleep(0.001)


def main():
    import bench_ext as bench  # noqa: F401
    import torch
    from v20_streamk_reduce_scatter import version_spec

    paths = hwmon(torch.cuda.get_device_properties(0).pci_bus_id)
    for shape in sys.argv[1].split(","):
        Mm, N, K = (int(x) for x in shape.split("x"))
        a_src = torch.rand((Mm, K), device="cuda", dtype=torch.bfloat16) - 0.5
        a = a_src.clone()
        b = torch.rand((N, K), device="cuda", dtype=torch.bfloat16).T - 0.5
        c = torch.empty((Mm, N), device="cuda", dtype=torch.bfloat16)
        between = M.make_modes(a, b, a_src)
        for spec in M.VERSIONS:
            v, env = version_spec.parse(spec)
            mod = importlib.import_module(bench.VERSION_MAP[v] + ".matmul_kernel")
            with version_spec.environment(env):
                for m in MODES:
                    fn = between[m]
                    for _ in range(10):
                        fn()
                        mod.matmul(a, b, c)
                    torch.cuda.synchronize()
                    stop, out = threading.Event(), []
                    th = threading.Thread(target=sample, args=(paths, stop, out))
                    th.start()
                    t_end = time.time() + LOOP_S
                    while time.time() < t_end:
                        for _ in range(20):
                            fn()
                            mod.matmul(a, b, c)
                        torch.cuda.synchronize()
                    stop.set()
                    th.join()
                    mhz = sorted(x for x, _ in out)
                    row = dict(shape=shape, version=spec, mode=m, sclk_med=statistics.median(mhz),
                               sclk_p10=mhz[len(mhz) // 10], power_med=statistics.median(w for _, w in out),
                               samples=len(out))
                    with open(os.path.join(M.RESULTS, "clocks.jsonl"), "a") as f:
                        f.write(json.dumps(row) + "\n")
                    print(row, flush=True)


if __name__ == "__main__":
    main()
