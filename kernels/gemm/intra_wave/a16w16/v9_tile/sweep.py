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
"""Timing, counters and ATT for v9_tile over a list of (tile, M, N, K) points.

    source v9_tile/env.sh            # GPU 2, llir+amdgcnas
    python v9_tile/sweep.py timing  --points A --out v9_tile/results/a_timing.csv
    python v9_tile/sweep.py counters --points A --out v9_tile/results/a_counters.csv
    python v9_tile/sweep.py att      --points A --out v9_tile/results/a_att.csv

A point is "TILE:MxNxK" (TILE as in V9_TILE, "v9" for v9_beyond_hotloop itself); --points takes
a comma-separated list or the name of a preset in POINTS. All runs are bf16 with B and C rows
padded as bench.py pads them.

- timing: one rocprofv3 --kernel-trace process over every point. Each point is checked against
  torch, then launched 1000 times over a rotating 512 MB of inputs (cold caches); the median and
  mean of the last 100 dispatches are kept, and the median idle gap between them. From Python a
  launch takes ~135 us, so shorter kernels leave the GPU idle between launches, and an idle GPU
  boosts. --graph replays the launches from a HIP graph instead: back to back, no gap, the clock a
  long stream of these kernels settles at.
- counters: two rocprofv3 --pmc passes (counter sets below), 20 cold launches per point, median
  of launches 5-20. GRBM_GUI_ACTIVE / 8 XCDs is the kernel's length in shader cycles (it agrees
  with SQ_CYCLES / 32 SEs). MFMA busy = SQ_VALU_MFMA_BUSY_CYCLES (the MFMA work, a fixed count) /
  (kernel cycles x 1024 SIMDs): the whole kernel, idle CUs, prologue and epilogue included, unlike
  ATT's main-loop MFMA efficiency. Counter collection slows the kernel and changes the clock
  (pmc_clock_ghz), so the clock of the timing run is estimated as kernel cycles / timed duration.
- clock: no profiler. Each point runs back to back (HIP graph replays) for --seconds (after a 1 s
  idle gap) while a
  thread polls the device's hwmon (freq1_input = graphics clock, power1_input = socket power)
  every 20 ms; the last 70% of the window is averaged. Also gives the sustained TFLOPS.
- att: one rocprofv3 --att per point (CU 0, the 10th of 12 cold launches), reduced by
  scripts/process_json.py: main-loop MFMA efficiency, and the prologue / loop / epilogue shares
  of a wave's time.
"""

import argparse
import csv
import glob
import json
import os
import re
import shutil
import statistics as S
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
A16 = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(A16, "..", "..", "..", ".."))
TMP = os.path.join(HERE, "tmp")

KS = (1024, 2048, 4096, 8192, 16384)
FULL_WAVE = {"256x256": (4096, 4096), "256x128": (4096, 2048), "128x256": (2048, 4096),
             "128x128": (2048, 2048), "128x128x128": (2048, 2048)}
SAME_SHAPE = [(4096, n, k) for k in (4096, 8192) for n in (1024, 2048, 3072)] + [
    (16384, 512, 8192), (1024, 2304, 16384), (2048, 2304, 16384)]
POINTS = {
    "A": [f"{t}:{m}x{n}x{k}" for t in ("256x256", "256x128", "128x256") for k in KS
          for m, n in [FULL_WAVE[t]]],
    "A128": [f"{t}:2048x2048x{k}" for t in ("128x128", "128x128r2", "128x128x128") for k in KS],
    "B": [f"{t}:{m}x{n}x{k}" for m, n, k in SAME_SHAPE for t in ("256x256", "256x128", "128x256")],
    "B128": [f"{t}:{m}x{n}x{k}" for m, n, k in SAME_SHAPE for t in ("128x128", "128x128r2", "128x128x128")],
}
COUNTER_SETS = [
    ["GRBM_GUI_ACTIVE", "SQ_VALU_MFMA_BUSY_CYCLES", "SQ_BUSY_CU_CYCLES"],
    ["TCC_HIT_sum", "TCC_MISS_sum", "TCC_EA0_RDREQ_sum", "TCC_EA0_RDREQ_DRAM_sum"],
]
GRAPH_LEN = 100
ROTATING_MB = 512  # --rotating-mb; 0 reuses one set of inputs (warm caches)
NUM_SIMDS = 1024
NUM_XCDS = 8  # GRBM_GUI_ACTIVE comes back summed over the 8 XCDs


def parse_point(p):
    tile, shape = p.split(":")
    M, N, K = (int(x) for x in shape.split("x"))
    return tile, M, N, K


KERNELS = ("v9_tile", "v9_tile_2r", "v9_beyond_hotloop")


def kernel_name(tile):
    return "v9_beyond_hotloop" if tile == "v9" else "v9_tile_2r" if tile.endswith("r2") else "v9_tile"


def tile_dims(tile):
    if tile == "v9":
        return 256, 256, 64
    dims = [int(x) for x in tile.removesuffix("r2").split("x")]
    return tuple(dims + [64] * (3 - len(dims)))


# ---------------------------------------------------------------------------------------------
# worker: runs inside rocprofv3
# ---------------------------------------------------------------------------------------------
def worker(args):
    sys.path.insert(0, A16)
    import bench  # RTLD_GLOBAL, amdgcnas hook, utils path
    import torch
    from v9_tile import matmul_kernel as mk
    import importlib

    v9 = importlib.import_module("v9_beyond_hotloop.matmul_kernel").matmul
    manifest = []
    for p in args.points:
        tile, M, N, K = parse_point(p)
        run = (lambda a, b, c: v9(a, b, c)) if tile == "v9" else (lambda a, b, c: mk.matmul(a, b, c, tile=tile))
        dt = torch.bfloat16
        a = torch.rand((M, K), device="cuda", dtype=dt) - 0.5
        b = bench.make_b(N, K, dt, True, lambda t: t.uniform_(-0.5, 0.5))
        c = bench.make_c(M, N, dt, True)
        run(a, b, c)
        ref = (a.float() @ b.float()).to(dt)
        correct = bool(torch.allclose(c, ref, atol=1e-1, rtol=1e-2))
        del a, b, c, ref
        a_l, b_l, c_l, copies = bench.gen_rotating_tensors(M, N, K, dt, args.rotating_mb)
        if args.graph:
            # One HIP graph of GRAPH_LEN launches over the rotating copies, replayed: the launches
            # run back to back, with none of the host's ~135 us per-launch gap.
            for i in range(copies):
                run(a_l[i], b_l[i], c_l[i])
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for i in range(GRAPH_LEN):
                    run(a_l[i % copies], b_l[i % copies], c_l[i % copies])
            for _ in range(args.iters // GRAPH_LEN):
                g.replay()
            torch.cuda.synchronize()
            dispatches = 1 + copies + (args.iters // GRAPH_LEN) * GRAPH_LEN
            del g
        else:
            for i in range(args.iters):
                run(a_l[i % copies], b_l[i % copies], c_l[i % copies])
            torch.cuda.synchronize()
            dispatches = 1 + args.iters
        del a_l, b_l, c_l
        torch.cuda.empty_cache()
        manifest.append({"point": p, "dispatches": dispatches, "correct": correct})
        print(f"{p} copies={copies} correct={correct}", flush=True)
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=1)


def run_worker(prof_args, points, iters, out_dir, extra_env=None, graph=False):
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    manifest = os.path.join(out_dir, "manifest.json")
    regex = "|".join(sorted({kernel_name(parse_point(p)[0]) for p in points}))
    cmd = ["rocprofv3", *prof_args, "--kernel-include-regex", regex, "-d", out_dir, "-o", "run",
           "--output-format", "csv", "--", sys.executable, os.path.abspath(__file__), "worker",
           "--iters", str(iters), "--manifest", manifest, "--points", ",".join(points),
           "--rotating-mb", str(ROTATING_MB)] + (["--graph"] if graph else [])
    p = subprocess.run(cmd, cwd=A16, env={**os.environ, **(extra_env or {})}, capture_output=True, text=True)
    if p.returncode:
        print(p.stdout[-3000:], p.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"rocprofv3 failed ({p.returncode})")
    return json.load(open(manifest))


def chunks(rows, manifest):
    """Split dispatch-ordered rows into per-point lists, checking the counts."""
    expected = sum(m["dispatches"] for m in manifest)
    assert len(rows) == expected, f"{len(rows)} dispatches in the trace, expected {expected}"
    out, pos = [], 0
    for m in manifest:
        out.append(rows[pos : pos + m["dispatches"]])
        pos += m["dispatches"]
    return out


def read_csv(out_dir, suffix):
    """The GEMM kernels' rows (rocprofv3's include regex does not filter the torch kernels out)."""
    files = glob.glob(os.path.join(out_dir, "**", f"*{suffix}"), recursive=True)
    assert len(files) == 1, f"expected one *{suffix} in {out_dir}, found {files}"
    return [r for r in csv.DictReader(open(files[0])) if r["Kernel_Name"] in KERNELS]


def base_row(p, correct):
    tile, M, N, K = parse_point(p)
    BM, BN, BK = tile_dims(tile)
    return {"tile": tile, "M": M, "N": N, "K": K, "tiles": (M // BM) * (N // BN), "correct": correct}


# ---------------------------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------------------------
def timing(args):
    out_dir = os.path.join(TMP, "sweep_timing" + ("_graph" if args.graph else ""))
    manifest = run_worker(["--kernel-trace"], args.points, 1000, out_dir, graph=args.graph)
    rows = read_csv(out_dir, "kernel_trace.csv")
    rows.sort(key=lambda r: int(r["Dispatch_Id"]))
    res = []
    for m, chunk in zip(manifest, chunks(rows, manifest)):
        tail = chunk[-100:]
        us = [(int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3 for r in tail]
        gaps = [(int(b["Start_Timestamp"]) - int(a["End_Timestamp"])) / 1e3 for a, b in zip(tail, tail[1:])]
        r = base_row(m["point"], m["correct"])
        med = S.median(us)
        r.update({"us_median": round(med, 3), "us_mean": round(S.mean(us), 3),
                  "tflops": round(2 * r["M"] * r["N"] * r["K"] / (med * 1e-6) / 1e12, 1),
                  "gap_us": round(S.median(gaps), 1)})
        res.append(r)
    return res


def counters(args):
    res = None
    for i, cset in enumerate(COUNTER_SETS):
        out_dir = os.path.join(TMP, f"sweep_counters_{i}")
        manifest = run_worker(["--kernel-trace", "--pmc", *cset], args.points, 20, out_dir)
        trace = sorted(read_csv(out_dir, "kernel_trace.csv"), key=lambda r: int(r["Dispatch_Id"]))
        dur = {int(r["Dispatch_Id"]): int(r["End_Timestamp"]) - int(r["Start_Timestamp"]) for r in trace}
        vals = {}
        for r in read_csv(out_dir, "counter_collection.csv"):
            vals.setdefault(int(r["Dispatch_Id"]), {})
            vals[int(r["Dispatch_Id"])][r["Counter_Name"]] = vals[int(r["Dispatch_Id"])].get(
                r["Counter_Name"], 0.0) + float(r["Counter_Value"])
        ids = sorted(vals)
        per_point = chunks(ids, manifest)
        if res is None:
            res = [base_row(m["point"], m["correct"]) for m in manifest]
        for r, disp in zip(res, per_point):
            disp = disp[5:]
            med = lambda f: S.median(f(vals[d], dur[d]) for d in disp)  # noqa: E731
            if i == 0:
                r["kcycles"] = round(med(lambda v, t: v["GRBM_GUI_ACTIVE"] / NUM_XCDS / 1e3), 1)
                r["mfma_busy"] = round(med(lambda v, t: v["SQ_VALU_MFMA_BUSY_CYCLES"] /
                                           (v["GRBM_GUI_ACTIVE"] / NUM_XCDS * NUM_SIMDS)), 4)
                r["pmc_us"] = round(med(lambda v, t: t / 1e3), 2)
                r["pmc_clock_ghz"] = round(med(lambda v, t: v["GRBM_GUI_ACTIVE"] / NUM_XCDS / t), 3)
            else:
                r["l2_hit"] = round(med(lambda v, t: v["TCC_HIT_sum"] / (v["TCC_HIT_sum"] + v["TCC_MISS_sum"])), 4)
                r["l2_req_M"] = round(med(lambda v, t: (v["TCC_HIT_sum"] + v["TCC_MISS_sum"]) / 1e6), 3)
                r["ea_rdreq_M"] = round(med(lambda v, t: v["TCC_EA0_RDREQ_sum"] / 1e6), 3)
                r["ea_rdreq_dram_M"] = round(med(lambda v, t: v["TCC_EA0_RDREQ_DRAM_sum"] / 1e6), 3)
    return res


def clock_worker(args):
    """Each point back to back for args.seconds over rotating inputs; wall times to the manifest."""
    import time
    sys.path.insert(0, A16)
    import bench  # noqa: F401
    import torch
    from v9_tile import matmul_kernel as mk
    import importlib

    v9 = importlib.import_module("v9_beyond_hotloop.matmul_kernel").matmul
    out = []
    for p in args.points:
        tile, M, N, K = parse_point(p)
        run = (lambda a, b, c: v9(a, b, c)) if tile == "v9" else (lambda a, b, c: mk.matmul(a, b, c, tile=tile))
        a_l, b_l, c_l, copies = bench.gen_rotating_tensors(M, N, K, torch.bfloat16, args.rotating_mb)
        for i in range(copies):
            run(a_l[i], b_l[i], c_l[i])
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for i in range(GRAPH_LEN):
                run(a_l[i % copies], b_l[i % copies], c_l[i % copies])
        torch.cuda.synchronize()
        time.sleep(1.0)  # idle gap so every point starts from the same state
        t0, n = time.time(), 0
        while time.time() - t0 < args.seconds:
            for _ in range(5):
                g.replay()
            n += 5 * GRAPH_LEN
            torch.cuda.synchronize()
        t1 = time.time()
        out.append({"point": p, "t0": t0, "t1": t1, "launches": n})
        del a_l, b_l, c_l, g
        torch.cuda.empty_cache()
    json.dump(out, open(args.manifest, "w"), indent=1)


def hwmon_dir():
    """hwmon of the device HIP_VISIBLE_DEVICES selects (matched by PCI bus, single-GPU runs only)."""
    import torch
    bus = torch.cuda.get_device_properties(0).pci_bus_id
    for card in glob.glob("/sys/class/drm/card*/device"):
        if os.path.basename(os.path.realpath(card)).endswith(f":{bus:02x}:00.0"):
            return glob.glob(os.path.join(card, "hwmon", "hwmon*"))[0]
    raise SystemExit(f"no hwmon for PCI bus {bus:#x}")


def clock(args):
    import threading
    import time
    hw = hwmon_dir()
    samples, stop = [], threading.Event()

    def poll():
        while not stop.is_set():
            f = int(open(os.path.join(hw, "freq1_input")).read()) / 1e9
            w = int(open(os.path.join(hw, "power1_input")).read()) / 1e6
            samples.append((time.time(), f, w))
            time.sleep(0.02)

    th = threading.Thread(target=poll, daemon=True)
    th.start()
    manifest = os.path.join(TMP, "sweep_clock.json")
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "clock_worker", "--points", ",".join(args.points),
                        "--manifest", manifest, "--seconds", str(args.seconds), "--rotating-mb", str(ROTATING_MB)],
                       cwd=A16, capture_output=True, text=True)
    stop.set()
    th.join()
    if p.returncode:
        print(p.stderr[-3000:], file=sys.stderr)
        raise SystemExit("clock worker failed")
    res = []
    for m in json.load(open(manifest)):
        r = base_row(m["point"], None)
        r.pop("correct")
        lo = m["t0"] + 0.3 * (m["t1"] - m["t0"])
        win = [s for s in samples if lo <= s[0] <= m["t1"]]
        r.update({"sustained_tflops": round(2 * r["M"] * r["N"] * r["K"] * m["launches"] / (m["t1"] - m["t0"]) / 1e12, 1),
                  "sclk_ghz": round(S.mean(s[1] for s in win), 3), "power_w": round(S.mean(s[2] for s in win)),
                  "samples": len(win)})
        res.append(r)
    return res


ATT_JOB = {"jobs": [{"kernel_include_regex": "", "kernel_exclude_regex": "", "kernel_iteration_range": "[10]",
                     "advanced_thread_trace": True, "att_target_cu": 0, "att_shader_engine_mask": "0xF",
                     "att_simd_select": "0xF", "att_buffer_size": "0x60000000"}]}


def att(args):
    res = []
    env = {**os.environ, "ROCPROF_ATT_LIBRARY_PATH": os.environ.get("ROCPROF_ATT_LIBRARY_PATH", "/opt/rocm/lib/")}
    for p in args.points:
        tile, M, N, K = parse_point(p)
        out_dir = os.path.join(TMP, "sweep_att", p.replace(":", "_"))
        shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir)
        job = json.loads(json.dumps(ATT_JOB))
        job["jobs"][0]["kernel_include_regex"] = kernel_name(tile)
        cfg = os.path.join(out_dir, "att.json")
        json.dump(job, open(cfg, "w"))
        manifest = os.path.join(out_dir, "manifest.json")
        cmd = ["rocprofv3", "--att", "-i", cfg, "-d", out_dir, "--", sys.executable, os.path.abspath(__file__),
               "worker", "--iters", "11", "--manifest", manifest, "--points", p, "--rotating-mb", str(ROTATING_MB)]
        pr = subprocess.run(cmd, cwd=A16, env=env, capture_output=True, text=True)
        row = base_row(p, None)
        ui = glob.glob(os.path.join(out_dir, "**", "ui_*"), recursive=True)
        if pr.returncode or not ui:
            print(f"{p}: ATT failed\n{pr.stderr[-2000:]}", file=sys.stderr)
            res.append(row)
            continue
        row["correct"] = json.load(open(manifest))[0]["correct"]
        pj = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "process_json.py"), ui[0]],
                            capture_output=True, text=True)
        try:
            j = json.loads(pj.stdout)
        except json.JSONDecodeError:
            print(f"{p}: process_json failed\n{pj.stderr[-2000:]}", file=sys.stderr)
            res.append(row)
            continue
        row.update({"att_mfma_eff": float(j["mfma efficiency"].rstrip("%")) / 100,
                    "mfma_in_loop": j["mfma_count_in_loop"],
                    "iter_cycles": round(j["average_iteration_duration"], 1),
                    "pro_ratio": j["pro_ratio"], "loop_ratio": j["loop_ratio"], "epi_ratio": j["epi_ratio"],
                    "prologue_cycles": round(j["average_prologue_duration"]),
                    "epilogue_cycles": round(j["average_epilogue_duration"])})
        print(f"{p}: {j['mfma efficiency']}", flush=True)
        res.append(row)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("mode", choices=["timing", "counters", "att", "clock", "worker", "clock_worker"])
    ap.add_argument("--points", required=True)
    ap.add_argument("--out")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--rotating-mb", type=int, default=512)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--graph", action="store_true", help="timing: launch from a HIP graph, back to back")
    ap.add_argument("--manifest")
    args = ap.parse_args()
    args.points = POINTS.get(args.points) or [p for p in args.points.split(",") if p]
    global ROTATING_MB
    ROTATING_MB = args.rotating_mb
    if args.mode in ("worker", "clock_worker"):
        (worker if args.mode == "worker" else clock_worker)(args)
        return
    res = {"timing": timing, "counters": counters, "att": att, "clock": clock}[args.mode](args)
    keys = list(dict.fromkeys(k for r in res for k in r))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(res)
    print("| " + " | ".join(keys) + " |")
    print("|" + "---|" * len(keys))
    for r in res:
        print("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")


if __name__ == "__main__":
    main()
