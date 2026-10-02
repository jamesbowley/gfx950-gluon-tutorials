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
"""Hardware-counter validation of the cache tiers (sweeps G and J) with rocprofv3.

    HIP_VISIBLE_DEVICES=6 python pmc.py --results-dir results/gpu6

Each point is launched `reps` times back-to-back via `bench.py --point`; the first dispatch is dropped and the
rest averaged. Writes <results-dir>/pmc.csv.
"""

import argparse
import csv
import glob
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

GROUPS = [
    ["TCP_TOTAL_CACHE_ACCESSES_sum", "TCP_TCC_READ_REQ_sum", "TCC_REQ_sum", "TCC_HIT_sum", "TCC_MISS_sum"],
    ["TCC_EA0_RDREQ_sum", "TCC_EA0_RDREQ_32B_sum", "TCC_EA0_RDREQ_64B_sum", "TCC_EA0_RDREQ_128B_sum"],
    ["TCC_EA0_WRREQ_sum", "TCC_EA0_WRREQ_64B_sum", "TCC_WRITE_sum", "TCC_WRITEBACK_sum", "TCP_TCC_WRITE_REQ_sum"],
]

# (point, passes, reps)
POINTS = [
    ("G:read_32KB", 16, 3),
    ("G:read_48KB", 16, 3),
    ("G:read_64KB", 16, 3),
    ("G:read_128KB", 16, 3),
    ("G:read_192KB", 16, 3),
    ("G:read_512KB", 16, 3),
    ("G:read_4096KB", 2, 3),
    ("G:write_32KB", 16, 3),
    ("G:write_48KB", 16, 3),
    ("G:write_64KB", 16, 3),
    ("G:write_512KB", 16, 3),
    ("G:write_4096KB", 2, 3),
    ("J:L2_64KB", 1, 6),
    ("J:MALL_512KB", 1, 6),
]


def collect(point, passes, reps, counters):
    """Mean per-dispatch counter values, dropping the first dispatch."""
    with tempfile.TemporaryDirectory() as d:
        cmd = ["rocprofv3", "--pmc", *counters, "--kernel-include-regex", "bw_kernel", "-d", d, "-o", "run",
               "--output-format", "csv", "--", sys.executable,
               os.path.join(HERE, "bench.py"), "--point", point, "--passes", str(passes), "--reps", str(reps)]
        subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=HERE)
        per_dispatch = defaultdict(dict)
        for f in glob.glob(os.path.join(d, "**", "*counter_collection.csv"), recursive=True):
            with open(f) as fh:
                for r in csv.DictReader(fh):
                    per_dispatch[int(r["Dispatch_Id"])][r["Counter_Name"]] = float(r["Counter_Value"])
    ids = sorted(per_dispatch)[1:]
    return {c: sum(per_dispatch[i].get(c, 0.0) for i in ids) / len(ids) for c in counters}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=os.path.join(HERE, "results"))
    args = p.parse_args()

    sys.path.insert(0, HERE)
    from bench import POINTS as CFGS, with_passes  # noqa: E402

    rows = []
    for point, passes, reps in POINTS:
        sweep, tag = point.split(":", 1)
        cfg = with_passes(dict(CFGS[sweep]())[tag], passes)
        vals = {}
        for g in GROUPS:
            vals.update(collect(point, passes, reps, g))
        nbytes = cfg.bytes_moved
        lines = nbytes / 128
        rd_bytes = 32 * vals["TCC_EA0_RDREQ_32B_sum"] + 64 * vals["TCC_EA0_RDREQ_64B_sum"] + \
            128 * vals["TCC_EA0_RDREQ_128B_sum"]
        wr_bytes = 64 * vals["TCC_EA0_WRREQ_64B_sum"] + 32 * (vals["TCC_EA0_WRREQ_sum"] - vals["TCC_EA0_WRREQ_64B_sum"])
        hit, miss = vals["TCC_HIT_sum"], vals["TCC_MISS_sum"]
        row = dict(point=point, passes=passes, footprint_mb=cfg.footprint_elems * 4 / 2**20, bytes=nbytes,
                   l2_req_per_line=round(vals["TCC_REQ_sum"] / lines, 3),
                   l1_to_l2_rd_per_line=round(vals["TCP_TCC_READ_REQ_sum"] / lines, 3),
                   l1_to_l2_wr_per_line=round(vals["TCP_TCC_WRITE_REQ_sum"] / lines, 3),
                   l2_hit_rate=round(hit / max(1.0, hit + miss), 4),
                   fabric_rd_frac=round(rd_bytes / nbytes, 4), fabric_wr_frac=round(wr_bytes / nbytes, 4),
                   l2_writeback_per_line=round(vals["TCC_WRITEBACK_sum"] / lines, 4))
        row.update({k: int(v) for k, v in vals.items()})
        rows.append(row)
        print({k: row[k] for k in list(row)[:12]}, flush=True)

    os.makedirs(args.results_dir, exist_ok=True)
    path = os.path.join(args.results_dir, "pmc.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", path)


if __name__ == "__main__":
    main()
