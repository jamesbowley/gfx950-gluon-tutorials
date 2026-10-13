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
"""GEMM kernel time after different work between launches, from rocprofv3 timestamps.

    export PYTHONPATH=<repo>/scripts/triton_deepbind_shim:<triton_gfx950-tutorial-v2.2>/python
    HIP_VISIBLE_DEVICES=0 python experiments/streamk_cache_state/modes.py trace 5632x3072x6144
    HIP_VISIBLE_DEVICES=0 python experiments/streamk_cache_state/modes.py pmc 5632x3072x6144
    python experiments/streamk_cache_state/modes.py report          # results/modes.md

v20's between_launches.py times each GEMM with CUDA events. For kernels under about 200 us the
start event then fires on an idle GPU before matmul()'s host code has launched the kernel, so
its "nothing" and "zero 64 MB" columns include host time. Here one process per shape runs every
version and mode in turn (10 warm-up and 100 timed launches each) under rocprofv3, and only
the GEMM dispatches' own start and end timestamps are used. The gap from the previous
dispatch's end to the GEMM's start is recorded too: it is the time the GPU sat idle.

Modes (the work between two GEMMs):
- nothing, zero64 (zero 64 MB), sleep (torch.cuda._sleep, about 100 us): as between_launches.py;
- clean_cold: read 256 MB (clean caches, inputs evicted); dirty_cold: zero 256 MB (do_bench);
- clean_warm / dirty_warm: the same, then read A and B (inputs back in the caches);
- *_idle: the same four, then the sleep;
- busy: an ALU-only kernel on every CU, no memory traffic, for about 400 us, so that the GPU is
  still busy when the GEMM is launched (the recorded gap shows whether it was);
- clean_warm_busy: clean_warm, then the busy kernel (clean_warm_idle with the GPU kept busy);
- write_a: read 256 MB, then write A from a copy, as an activation kernel before a GEMM would
  (A warm and dirty, B cold).

`pmc` runs the same plan with 3 warm-up and 10 counted launches per (version, mode) under
rocprofv3 --pmc, in three passes of counters (PASSES). rocprofv3 serialises dispatches to
count them; the mode's work still runs before each GEMM.
"""

import csv
import glob
import json
import os
import statistics
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
A16 = os.path.join(REPO, "kernels", "gemm", "intra_wave", "a16w16")
RESULTS = os.path.join(HERE, "results")
sys.path.insert(0, A16)

VERSIONS = ["13", "20:STREAMK_POLICY=capped", "20:STREAMK_POLICY=two_tile"]
MODES = ["nothing", "zero64", "sleep", "busy", "clean_cold", "clean_warm", "dirty_cold", "dirty_warm",
         "clean_cold_idle", "clean_warm_idle", "dirty_cold_idle", "dirty_warm_idle", "clean_warm_busy",
         "write_a"]
SHAPES = ["5632x3072x6144", "5120x3584x3072", "8192x2304x6144", "2048x9216x6144", "7680x8960x6144",
          "6144x11008x6144", "18432x14592x8192", "10752x12544x8192"]
PASSES = [
    ["TCC_HIT_sum", "TCC_MISS_sum", "TCC_EA0_RDREQ_sum", "TCC_EA0_WRREQ_sum"],
    ["TCC_EA0_RDREQ_DRAM_sum", "TCC_EA0_WRREQ_DRAM_sum", "TCC_ALL_TC_OP_WB_WRITEBACK_sum", "TCC_NORMAL_WRITEBACK_sum"],
    ["TCC_EA0_RDREQ_DRAM_CREDIT_STALL_sum", "TCC_EA0_WRREQ_DRAM_CREDIT_STALL_sum", "GRBM_GUI_ACTIVE",
     "TCC_ALL_TC_OP_INV_EVICT_sum"],
]
TRACE_LAUNCHES = (10, 100)
PMC_LAUNCHES = (3, 10)
SLEEP_CYCLES = 200000
BUSY_US = 400


# ---------------------------------------------------------------- worker


def make_modes(a, b, a_src):
    """{mode: callable} for the work between two GEMMs."""
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def alu_busy(out_ptr, iters):
        x = tl.program_id(0).to(tl.float32) + tl.arange(0, 256).to(tl.float32)
        for _ in range(iters):
            x = x * 1.0000001 + 0.5
        tl.store(out_ptr + tl.program_id(0) * 256 + tl.arange(0, 256), x)

    big = torch.empty(256 * 2**20 // 4, device="cuda", dtype=torch.int32)
    small = torch.empty(64 * 2**20 // 4, device="cuda", dtype=torch.int32)
    busy_out = torch.empty(1024 * 256, device="cuda", dtype=torch.float32)

    def sleep():
        torch.cuda._sleep(SLEEP_CYCLES)

    # The busy kernel runs BUSY_US at full clock: it must outlast the host time of its own launch
    # and of matmul(), or the GPU idles before the GEMM anyway. (The sleep counts shader clocks,
    # so its length depends on the clock and cannot be the reference.)
    def length(fn):
        for _ in range(20):
            fn()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(10):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) * 100

    # Calibrate from a run long enough (about 1 ms) that the kernel, not its launch, sets the time.
    iters = 128000
    iters = max(1, int(iters * BUSY_US / length(lambda: alu_busy[(1024,)](busy_out, iters, num_warps=4))))

    def busy():
        alu_busy[(1024,)](busy_out, iters, num_warps=4)

    def warm():
        a.sum()
        b.sum()

    return {
        "nothing": lambda: None,
        "zero64": lambda: small.zero_(),
        "sleep": sleep,
        "busy": busy,
        "clean_cold": lambda: big.sum(),
        "clean_warm": lambda: (big.sum(), warm()),
        "dirty_cold": lambda: big.zero_(),
        "dirty_warm": lambda: (big.zero_(), warm()),
        "clean_cold_idle": lambda: (big.sum(), sleep()),
        "clean_warm_idle": lambda: (big.sum(), warm(), sleep()),
        "dirty_cold_idle": lambda: (big.zero_(), sleep()),
        "dirty_warm_idle": lambda: (big.zero_(), warm(), sleep()),
        "clean_warm_busy": lambda: (big.sum(), warm(), busy()),
        "write_a": lambda: (big.sum(), a.copy_(a_src)),
    }


def worker(shape, versions, modes, warmup, launches):
    import importlib

    import bench_ext as bench  # noqa: F401  (plugin hooks and the path to common.py)
    import torch
    from v20_streamk_reduce_scatter import version_spec

    M, N, K = (int(x) for x in shape.split("x"))
    a_src = torch.rand((M, K), device="cuda", dtype=torch.bfloat16) - 0.5
    a = a_src.clone()
    b = torch.rand((N, K), device="cuda", dtype=torch.bfloat16).T - 0.5
    c = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    between = make_modes(a, b, a_src)
    for spec in versions:
        v, env = version_spec.parse(spec)
        mod = importlib.import_module(bench.VERSION_MAP[v] + ".matmul_kernel")
        with version_spec.environment(env):
            mod.matmul(a, b, c)
            torch.cuda.synchronize()
            for m in modes:
                fn = between[m]
                for _ in range(warmup + launches):
                    fn()
                    mod.matmul(a, b, c)
                torch.cuda.synchronize()


# ---------------------------------------------------------------- driver


def gemm_names():
    from v20_streamk_reduce_scatter import rocprof_check

    return rocprof_check.NAMES


def csv_rows(d, key):
    rows = []
    for f in glob.glob(os.path.join(d, "**", "*.csv"), recursive=True):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                if key in r:
                    rows.append(r)
    return rows


def profile(args, shape, versions, modes, warmup, launches, out_dir):
    cmd = ["rocprofv3", *args, "-d", out_dir, "-o", "run", "--output-format", "csv", "--",
           sys.executable, os.path.abspath(__file__), "worker", shape, ",".join(versions), ",".join(modes),
           str(warmup), str(launches)]
    p = subprocess.run(cmd, cwd=A16, capture_output=True, text=True)
    if p.returncode:
        sys.exit(f"rocprofv3 failed for {shape}:\n{p.stderr[-3000:]}")


def blocks(gemm_rows, versions, modes, warmup, launches):
    """The GEMM dispatches in order, cut into one block per (version, mode); each version's
    first (untimed) launch comes before its blocks."""
    per = warmup + launches
    need = len(versions) * (1 + len(modes) * per)
    assert len(gemm_rows) == need, (len(gemm_rows), need)
    out, i = {}, 0
    for spec in versions:
        i += 1
        for m in modes:
            out[(spec, m)] = gemm_rows[i + warmup:i + per]
            i += per
    return out


def is_gemm(name):
    return any(n in name for n in gemm_names().values())


def trace(shape, versions=VERSIONS, modes=MODES):
    warmup, launches = TRACE_LAUNCHES
    d = tempfile.mkdtemp(prefix="cs_trace_")
    profile(["--kernel-trace"], shape, versions, modes, warmup, launches, d)
    rows = [r for r in csv_rows(d, "Kernel_Name") if "Start_Timestamp" in r]
    rows.sort(key=lambda r: int(r["Start_Timestamp"]))
    prev_end, prev_us, gemm = None, float("nan"), []
    for r in rows:
        s, e = int(r["Start_Timestamp"]), int(r["End_Timestamp"])
        if is_gemm(r["Kernel_Name"]):
            gemm.append(dict(us=(e - s) / 1e3, gap=(s - prev_end) / 1e3 if prev_end else float("nan"), prev=prev_us))
        prev_end = e if prev_end is None else max(prev_end, e)
        prev_us = (e - s) / 1e3
    out = []
    for (spec, m), rs in blocks(gemm, versions, modes, warmup, launches).items():
        us = sorted(r["us"] for r in rs)
        # gap: GPU idle before the GEMM; prev_us: the kernel just before it (the GEMM before
        # it in "nothing", else the last kernel of the mode's work).
        out.append(dict(shape=shape, version=spec, mode=m, us=statistics.median(us), p10=us[len(us) // 10],
                        p90=us[len(us) * 9 // 10], gap=statistics.median(r["gap"] for r in rs),
                        prev_us=statistics.median(r["prev"] for r in rs)))
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "trace.jsonl"), "a") as f:
        f.writelines(json.dumps(r) + "\n" for r in out)
    print(f"{shape}: {len(out)} (version, mode) medians", flush=True)


def pmc(shape, versions=VERSIONS, modes=MODES):
    warmup, launches = PMC_LAUNCHES
    out = {}
    for counters in PASSES:
        d = tempfile.mkdtemp(prefix="cs_pmc_")
        profile(["--pmc", *counters], shape, versions, modes, warmup, launches, d)
        by_disp = {}
        for r in csv_rows(d, "Counter_Name"):
            if is_gemm(r["Kernel_Name"]):
                by_disp.setdefault(int(r["Dispatch_Id"]), {})[r["Counter_Name"]] = float(r["Counter_Value"])
        gemm = [by_disp[k] for k in sorted(by_disp)]
        for key, rs in blocks(gemm, versions, modes, warmup, launches).items():
            for cn in counters:
                out.setdefault(key, {})[cn] = statistics.median(r[cn] for r in rs)
    with open(os.path.join(RESULTS, "pmc.jsonl"), "a") as f:
        for (spec, m), cs in out.items():
            f.write(json.dumps(dict(shape=shape, version=spec, mode=m, **cs)) + "\n")
    print(f"{shape}: counters for {len(out)} (version, mode)", flush=True)


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "worker":
        worker(sys.argv[2], sys.argv[3].split(","), sys.argv[4].split(","), int(sys.argv[5]), int(sys.argv[6]))
    elif what == "trace":
        for s in sys.argv[2].split(","):
            trace(s)
    elif what == "pmc":
        for s in sys.argv[2].split(","):
            pmc(s)
    elif what == "report":
        import report

        report.main()
    else:
        sys.exit(f"unknown mode {what}")
