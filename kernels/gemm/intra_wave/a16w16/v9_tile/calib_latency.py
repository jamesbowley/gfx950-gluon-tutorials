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
"""Calibrate sweep.py's latency counters (sets 5-7) on kernels whose latency is known (WORKLOG §16).

    source v9_tile/env.sh; unset TRITON_AMDGCNAS_PLUGIN
    python v9_tile/calib_latency.py --out v9_tile/results/latency_calib.csv

Runs experiments/tcp_latency's kernels under each latency counter set:

- probe_kernel: one wave, one 1 KB load in flight, timed per load with s_memtime. A 1 MB region read
  32 times (L2 after the first pass), a 16 MB region read 8 times (MALL), a 64 MB region read once
  after a flush (HBM). The counters' mean latency is compared with the probe's own mean.
- bw_kernel: one workgroup of 8 waves x 4 loads (64 KB nominally in flight), double-buffered in
  registers, on one CU (HWSEL) or every CU, from HBM (cold, one pass) or L2 (a 1 MB footprint read 64
  times). One CU from HBM moves ~26 B/clk (tcp_latency README): the window's 32 KB at ~1250 cycles
  TCP-side, or 44 KB at ~1720 from issue.
"""

import argparse
import csv
import glob
import json
import os
import statistics as S
import subprocess
import sys
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sweep  # noqa: E402

REPO = sweep.REPO
TCP_LAT = os.path.join(REPO, "experiments", "tcp_latency")
KERNELS = ("probe_kernel", "bw_kernel")
REPS = 3
# (label, region tiles of 1 KB, passes, flush first)
PROBE_RUNS = [("probe_L2", 1024, 32, True), ("probe_MALL", 16384, 8, True), ("probe_HBM", 65536, 1, True)]
# (label, scope, tier, per-CU footprint KB, passes)
STREAM_RUNS = [("stream_1cu_HBM", "1cu", "HBM", 32 * 1024, 1), ("stream_1cu_L2", "1cu", "L2", 1024, 64),
               ("stream_256cu_HBM", "256cu", "HBM", 4 * 1024, 1), ("stream_256cu_L2", "256cu", "L2", 64, 64)]


def worker(args):
    sys.path.insert(0, TCP_LAT)
    import bench as tb
    import kernel as tk
    import numpy as np
    import torch

    labels, info = [], {}

    def note(label, d=None):
        labels.append(label)
        if d is not None:
            info.setdefault(label, []).append(d)

    probe = tb.Probe()
    flush = tb.Flusher()
    for label, region, passes, do_flush in PROBE_RUNS:
        for _ in range(REPS):
            if do_flush:
                flush()
                note("flush")
            torch.cuda.synchronize()
            s, out = probe.launch(region, passes, num_warps=1)
            c = s[:, 0].astype(np.float64)
            later = c[region:] if passes > 1 else c
            note(label, {"probe_mean_cyc": float(c.mean()), "probe_mean_cyc_tier": float(later.mean()),
                         "probe_p50_cyc_tier": float(np.median(later)), "probe_ghz": out["probe_ghz"],
                         "loads": len(c)})
    del probe
    torch.cuda.empty_cache()

    bufs = tk.Buffers(2 * tb.GB)
    per_xcd = tb.discover_cus(bufs)
    note("discover")
    one = per_xcd[0][0]
    for label, scope, tier, per_cu_kb, passes in STREAM_RUNS:
        cus = [one] if scope == "1cu" else [k for v in per_xcd.values() for k in v]
        cfg = tk.BWConfig(op="read", mode="hwsel", W=len(cus), num_warps=8, unroll=4, wpc=1,
                          hwsel_caps={k: 1 for k in cus})
        wrap = max(1, per_cu_kb * tb.KB // (cfg.tile_elems * 4))
        cfg = replace(cfg, n_iter=wrap * passes, passes=passes)
        launch = tk.prepare(cfg, bufs)
        for _ in range(REPS):
            flush()
            note("flush")
            launch.pre()
            launch()
            torch.cuda.synchronize()
            rec = tb.analyse_records(cfg, bufs)
            bpc = cfg.bytes_moved / (rec["kernel_span_us"] * 1e-6) / (rec["sclk_mhz"] * 1e6) / rec["n_cus"]
            note(label, {"cus": rec["n_cus"], "stream_bpc_per_cu": bpc, "stream_sclk_mhz": rec["sclk_mhz"],
                         "nominal_kb": 2 * cfg.num_warps * cfg.unroll})
    json.dump({"labels": labels, "info": info}, open(args.manifest, "w"), indent=1)


def run_set(i, out_dir):
    cset = sweep.COUNTER_SETS[i]
    d = os.path.join(out_dir, f"set{i}")
    manifest = os.path.join(d, "manifest.json")
    os.makedirs(d, exist_ok=True)
    cmd = ["rocprofv3", "--kernel-trace", "--pmc", *cset, "--kernel-include-regex", "|".join(KERNELS), "-d", d,
           "-o", "run", "--output-format", "csv", "--", sys.executable, os.path.abspath(__file__), "worker",
           "--manifest", manifest]
    p = subprocess.run(cmd, cwd=TCP_LAT, capture_output=True, text=True)
    if p.returncode:
        print(p.stdout[-3000:], p.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"rocprofv3 failed ({p.returncode})")
    m = json.load(open(manifest))
    keep = lambda r: any(k in r["Kernel_Name"] for k in KERNELS)  # noqa: E731
    trace = [r for r in csv.DictReader(open(glob.glob(os.path.join(d, "**", "*kernel_trace.csv"), recursive=True)[0]))
             if keep(r)]
    dur = {int(r["Dispatch_Id"]): int(r["End_Timestamp"]) - int(r["Start_Timestamp"]) for r in trace}
    vals = {}
    for r in csv.DictReader(open(glob.glob(os.path.join(d, "**", "*counter_collection.csv"), recursive=True)[0])):
        if keep(r):
            vals.setdefault(int(r["Dispatch_Id"]), {})
            vals[int(r["Dispatch_Id"])][r["Counter_Name"]] = vals[int(r["Dispatch_Id"])].get(
                r["Counter_Name"], 0.0) + float(r["Counter_Value"])
    ids = sorted(vals)
    assert len(ids) == len(m["labels"]), f"{len(ids)} dispatches, {len(m['labels'])} labels"
    per_label = {}
    for d_id, label in zip(ids, m["labels"]):
        per_label.setdefault(label, []).append(d_id)
    return per_label, vals, dur, m["info"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("mode", nargs="?", default="run", choices=["run", "worker"])
    ap.add_argument("--manifest")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "latency_calib.csv"))
    ap.add_argument("--tmp", default=os.path.join(HERE, "tmp", "calib_latency"))
    args = ap.parse_args()
    if args.mode == "worker":
        worker(args)
        return
    rows = {}
    for i in (1, 5, 6, 7):
        per_label, vals, dur, info = run_set(i, args.tmp)
        for label, ids in per_label.items():
            if label in ("flush", "discover"):
                continue
            r = rows.setdefault(label, {"run": label})
            side = info[label]
            for k in side[0]:
                r[k] = round(S.median(x[k] for x in side), 3)
            cus = side[0].get("cus", 1)
            if i == 1:
                r["l2_hit"] = round(S.median(vals[d]["TCC_HIT_sum"] / max(1.0, vals[d]["TCC_HIT_sum"] +
                                                                           vals[d]["TCC_MISS_sum"]) for d in ids), 4)
                continue
            cols = [sweep.latency_columns(i, vals[d], dur[d], cus) for d in ids]
            for k in cols[0]:
                r[k] = round(S.median(c[k] for c in cols), 4)
    res = list(rows.values())
    keys = list(dict.fromkeys(k for r in res for k in r))
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
