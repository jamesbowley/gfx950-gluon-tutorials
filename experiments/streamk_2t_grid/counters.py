"""L2 hit rate (TCC_HIT_sum / (TCC_HIT_sum + TCC_MISS_sum)) per point, over the last
100 dispatches of experiments/streamk_v31_rule's sweep worker (cold rotating inputs,
graph replay), under rocprofv3 --pmc.

    source kernels/gemm/intra_wave/a16w16/v31_streamk_aiter/env.sh; cd $REPO
    python experiments/streamk_2t_grid/counters.py --out experiments/streamk_2t_grid/results/counters.md
"""

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.join(HERE, "..", "streamk_v31_rule", "sweep.py")
KERNEL = "_gemm_a16w16_compute_bound_kernel"
TILE = "256x256x64"
POINTS = [f"{k}:{TILE}:3840x5120x8192" for k in ("dp", "ow2", "rs4", "2t", "2tg240", "2tg224", "2tg200")]
POINTS += [f"{k}:{TILE}:4096x6144x8192" for k in ("dp", "ow2", "2t", "2tg232")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", default=",".join(POINTS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    points = args.points.split(",")
    out_dir = os.path.join(HERE, "tmp", "counters")
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    manifest = os.path.join(out_dir, "manifest.json")
    env = dict(os.environ)
    env.setdefault("TRITON_CACHE_DIR", os.path.join(HERE, "..", "streamk_v31_rule", "tmp", "triton"))
    subprocess.run(["rocprofv3", "--pmc", "TCC_HIT_sum", "TCC_MISS_sum", "--kernel-include-regex", KERNEL,
                    "-d", out_dir, "-o", "run", "--output-format", "csv", "--", sys.executable, SWEEP,
                    "worker", "--manifest", manifest, "--points", ",".join(points)], env=env, check=True)
    man = json.load(open(manifest))
    f = glob.glob(os.path.join(out_dir, "**", "*counter_collection.csv"), recursive=True)[0]
    per = defaultdict(dict)
    for r in csv.DictReader(open(f)):
        if r["Kernel_Name"].startswith(KERNEL):
            per[int(r["Dispatch_Id"])][r["Counter_Name"]] = float(r["Counter_Value"])
    ids = sorted(per)
    expected = sum(m["dispatches"] for m in man)
    if len(ids) != expected:
        sys.exit(f"{len(ids)} dispatches with counters, expected {expected}")
    lines = ["| point | correct | L2 hit % | hits (M) | misses (M) |", "|---|---|---|---|---|"]
    pos = 0
    for m in man:
        chunk = ids[pos:pos + m["dispatches"]][-100:]
        pos += m["dispatches"]
        hit = sum(per[i]["TCC_HIT_sum"] for i in chunk) / len(chunk)
        miss = sum(per[i]["TCC_MISS_sum"] for i in chunk) / len(chunk)
        lines.append(f"| {m['point']} | {m['correct']} | {100 * hit / (hit + miss):.1f} | "
                     f"{hit / 1e6:.2f} | {miss / 1e6:.2f} |")
    open(args.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
