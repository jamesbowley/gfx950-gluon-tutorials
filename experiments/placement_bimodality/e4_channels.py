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
"""E4: per-channel and per-XCD L2-to-memory counters for C at fast and slow offsets.

    HIP_VISIBLE_DEVICES=2 python e4_channels.py [--offsets 16,48,80,144,176,208]

Runs `place.py launch` (one 1 GB buffer, A/B at 900 MB, C at each offset, --n launches each)
under rocprofv3 --pmc with derived counters that select one TCC channel (summed over XCDs) or
one XCD (summed over channels). Kernels are serialised, so this compares request counts,
latencies and stalls, not times. Medians over the launches of each offset.
"""

import argparse
import csv
import glob
import os
import statistics
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
EVENTS = {"WR": "TCC_EA0_WRREQ", "RD": "TCC_EA0_RDREQ", "WRL": "TCC_EA0_WRREQ_LEVEL",
          "RDL": "TCC_EA0_RDREQ_LEVEL", "WRS": "TCC_EA0_WRREQ_DRAM_CREDIT_STALL"}


def write_yaml(path):
    lines = ["rocprofiler-sdk:", "  counters-schema-version: 1", "  counters:"]
    for tag, ev in EVENTS.items():
        for dim, n, label in (("INSTANCE", 16, "CH"), ("XCC", 8, "XCC")):
            for i in range(n):
                lines += [f"  - name: PB_{tag}_{label}{i}", f"    description: {ev} {label} {i}",
                          "    properties: []", "    definitions:", "    - architectures:", "      - gfx950",
                          f"      expression: reduce(select({ev},[DIMENSION_{dim}=[{i}]]),sum)"]
    open(path, "w").write("\n".join(lines) + "\n")


def collect(counters, args, yaml):
    d = tempfile.mkdtemp(prefix="pbch_")
    cmd = ["rocprofv3", "-E", yaml, "--pmc", *counters, "--kernel-include-regex", "v9_", "-d", d, "-o", "run",
           "--output-format", "csv", "--", sys.executable, os.path.join(HERE, "place.py"), "launch", "--n",
           str(args.n), "--offsets", args.offsets, "--fixed", "900"]
    subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, check=True)
    by = {}
    for f in glob.glob(os.path.join(d, "**", "*counter_collection.csv"), recursive=True):
        for r in csv.DictReader(open(f)):
            by.setdefault(int(r["Dispatch_Id"]), {})[r["Counter_Name"]] = float(r["Counter_Value"])
    ids = sorted(by)
    out = {}
    for k, off in enumerate(args.offsets.split(",")):
        grp = ids[k * args.n:(k + 1) * args.n][2:]
        out[off] = {c: statistics.median(by[i][c] for i in grp) for c in counters}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--offsets", default="16,48,80,144,176,208")
    p.add_argument("--n", type=int, default=10)
    args = p.parse_args()
    yaml = os.path.join(HERE, "results", "e4_counters.yaml")
    write_yaml(yaml)
    res = {o: {} for o in args.offsets.split(",")}
    for tag in EVENTS:
        for label, n in (("CH", 16), ("XCC", 8)):
            for o, r in collect([f"PB_{tag}_{label}{i}" for i in range(n)], args, yaml).items():
                res[o].update(r)
    offs = args.offsets.split(",")

    def tab(title, label, n, fn, fmt):
        print(f"\n{title}\n")
        print("| C offset (MB) | " + " | ".join(f"{label}{i}" for i in range(n)) + " |")
        print("|---" * (n + 1) + "|")
        for o in offs:
            print(f"| {o} | " + " | ".join(fmt(fn(res[o], label, i)) for i in range(n)) + " |")

    lat = lambda tag: lambda r, label, i: r[f"PB_{tag}L_{label}{i}"] / max(r[f"PB_{tag}_{label}{i}"], 1)
    cnt = lambda tag: lambda r, label, i: r[f"PB_{tag}_{label}{i}"]
    for label, n in (("CH", 16), ("XCC", 8)):
        tab(f"Write requests per {label}", label, n, cnt("WR"), lambda x: f"{x / 1e3:.1f}k")
        tab(f"Read requests per {label}", label, n, cnt("RD"), lambda x: f"{x / 1e3:.1f}k")
        tab(f"Average write latency per {label} (cycles)", label, n, lat("WR"), lambda x: f"{x:.0f}")
        tab(f"Average read latency per {label} (cycles)", label, n, lat("RD"), lambda x: f"{x:.0f}")
        tab(f"Write DRAM-credit stall cycles per {label}", label, n, cnt("WRS"), lambda x: f"{x / 1e3:.1f}k")


if __name__ == "__main__":
    main()
