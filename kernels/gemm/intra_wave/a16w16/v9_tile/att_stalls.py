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
"""Where a v9_tile loop iteration goes: ATT stall cycles in the K loop, by instruction class.

    python v9_tile/att_stalls.py v9_tile/tmp/sweep_att/256x128_4096x2048x8192 [...]

Reads code.json (per-instruction Hit, Latency, Stall summed over the traced waves) from the
sweep's ATT output, keeps the loop body (the instructions with the back-edge's hit count), and
reports per loop iteration and wave: the cycles issued (sum of latency), the stall cycles split
by what the wave was waiting on, and the most stalled instructions.
"""

import collections
import glob
import json
import os
import sys


def classify(isa):
    op = isa.split()[0]
    if op.startswith("v_mfma"):
        return "mfma"
    if op == "s_waitcnt":
        if "vmcnt" in isa and "lgkmcnt" in isa:
            return "s_waitcnt vmcnt+lgkmcnt"
        return "s_waitcnt " + ("vmcnt" if "vmcnt" in isa else "lgkmcnt")
    if op.startswith("buffer_load"):
        return "buffer_load"
    if op.startswith("ds_read"):
        return "ds_read"
    if op == "s_barrier":
        return "s_barrier"
    return "other"


def analyze(d):
    ui = glob.glob(os.path.join(d, "**", "ui_*"), recursive=True)[0]
    code = json.load(open(os.path.join(ui, "code.json")))["code"]
    branches = [c for c in code if c[0].startswith("s_cbranch")]
    loop_hit = max(c[6] for c in branches)
    body = [c for c in code if c[6] == loop_hit and loop_hit > 0]
    waves = len(glob.glob(os.path.join(ui, "se*_sm*_sl0_wv*.json")))
    iters = loop_hit / max(1, waves)
    by = collections.defaultdict(lambda: [0, 0])
    for c in body:
        k = classify(c[0])
        by[k][0] += c[7]
        by[k][1] += c[8]
    norm = loop_hit  # per (iteration x wave)
    top = sorted(body, key=lambda c: -c[8])[:6]
    return {"iters_per_wave": iters, "by": {k: (v[0] / norm, v[1] / norm) for k, v in by.items()},
            "total_latency": sum(c[7] for c in body) / norm, "total_stall": sum(c[8] for c in body) / norm,
            "top": [(c[0], c[8] / norm) for c in top]}


def main():
    for d in sys.argv[1:]:
        r = analyze(d)
        print(f"### {os.path.basename(d.rstrip('/'))}\n")
        print(f"Per loop iteration (2 k-steps) and wave: {r['total_latency']:.0f} cycles issued "
              f"(sum of instruction latency), {r['total_stall']:.0f} of them stalled.\n")
        print("| class | latency cycles | stall cycles |")
        print("|---|---|---|")
        for k, (lat, st) in sorted(r["by"].items(), key=lambda kv: -kv[1][1]):
            print(f"| {k} | {lat:.0f} | {st:.0f} |")
        print("\nMost stalled instructions:\n")
        for isa, st in r["top"]:
            print(f"- `{isa}`: {st:.0f}")
        print()


if __name__ == "__main__":
    main()
