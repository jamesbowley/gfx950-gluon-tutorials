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
"""Markdown tables from sweep.py's CSVs.

    python v9_kreverse/report.py v9_kreverse/results > v9_kreverse/results/report.md

Each table gives, per shape, the K_ORDER = 0 value and the change of K_ORDER = 1 and 2 against it.
For the repeated sweeps (trace, telemetry) the change is the mean of the per-repetition changes,
with their min..max range: each repetition ran the three orders back to back.
"""

import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shapes import SHAPES  # noqa: E402

SHAPE_ORDER = ["x".join(map(str, s)) for s in SHAPES] + ["8192x8192x32768"]


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def pct(x, base):
    return 100 * (x - base) / base


def repeated(rows, metric, title, fmt="{:.1f}"):
    print(f"\n## {title}\n")
    print(f"| shape | K_ORDER=0 {metric} | K_ORDER=1 vs 0 | K_ORDER=2 vs 0 | reps |")
    print("|---|---|---|---|---|")
    for shape in SHAPE_ORDER:
        by = {}
        for r in rows:
            if r["shape"] == shape:
                by.setdefault(r["rep"], {})[int(r["korder"])] = float(r[metric])
        reps = [v for v in by.values() if len(v) == 3]
        if not reps:
            continue
        base = statistics.mean(v[0] for v in reps)
        cells = []
        for k in (1, 2):
            d = [pct(v[k], v[0]) for v in reps]
            cells.append(f"{statistics.mean(d):+.2f}% ({min(d):+.2f}..{max(d):+.2f})")
        print(f"| {shape} | {fmt.format(base)} | {cells[0]} | {cells[1]} | {len(reps)} |")


def counters(rows):
    print("\n## L2 counters (one rocprofv3 --pmc run per point, mean of launches 5-19)\n")
    print("Latency is TCC_EA0_RDREQ_LEVEL / TCC_EA0_RDREQ: by Little's law the mean L2-miss read "
          "latency in cycles, lower when more of the misses hit the MALL instead of HBM.\n")
    print("| shape | K_ORDER | TCC_MISS | vs 0 | TCC_EA0_RDREQ | vs 0 | L2 hit rate | miss latency (cycles) | vs 0 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for shape in SHAPE_ORDER:
        by = {int(r["korder"]): {k: float(v) for k, v in r.items() if k not in ("shape", "korder")}
              for r in rows if r["shape"] == shape}
        if 0 not in by:
            continue
        lat = {k: v["TCC_EA0_RDREQ_LEVEL_sum"] / v["TCC_EA0_RDREQ_sum"] for k, v in by.items()}
        for k in sorted(by):
            v = by[k]
            hit = v["TCC_HIT_sum"] / (v["TCC_HIT_sum"] + v["TCC_MISS_sum"])
            rel = lambda name: "" if k == 0 else f"{pct(v[name], by[0][name]):+.2f}%"  # noqa: E731
            dl = "" if k == 0 else f"{pct(lat[k], lat[0]):+.2f}%"
            print(f"| {shape} | {k} | {v['TCC_MISS_sum']:.0f} | {rel('TCC_MISS_sum')} | "
                  f"{v['TCC_EA0_RDREQ_sum']:.0f} | {rel('TCC_EA0_RDREQ_sum')} | {hit:.3f} | {lat[k]:.0f} | {dl} |")


def main():
    d = sys.argv[1]
    suffix = sys.argv[2] if len(sys.argv) > 2 else ""
    print(f"# v9_kreverse results{' (' + suffix + ')' if suffix else ''} (bf16, physical GPU 1 = BDF 0000:15:00.0)")
    tag = lambda name: os.path.join(d, f"{name}{'_' + suffix if suffix else ''}.csv")  # noqa: E731
    rows = load(tag("trace"))
    if rows:
        repeated(rows, "tflops", "TFLOPS (rocprofv3 kernel trace, 1000 launches, mean of the last 100)")
    rows = load(tag("telemetry"))
    if rows:
        repeated(rows, "tflops", "Sustained TFLOPS (10 s back-to-back after a 4 s warm-up)")
        repeated(rows, "energy_j_per_launch", "Energy per launch, J (amd-smi energy accumulator)", "{:.3f}")
        repeated(rows, "power_w", "Mean socket power, W")
        repeated(rows, "gfx_mhz", "Mean GFX clock, MHz", "{:.0f}")
        repeated(rows, "umc_pct", "Mean UMC (HBM) activity, %", "{:.2f}")
    for name in ("counters", "counters_b", "counters_c"):
        rows = load(tag(name)) if name == "counters" else load(os.path.join(d, f"{name}.csv"))
        if rows and (name == "counters" or suffix):
            print(f"\n### {name}")
            counters(rows)


if __name__ == "__main__":
    main()
