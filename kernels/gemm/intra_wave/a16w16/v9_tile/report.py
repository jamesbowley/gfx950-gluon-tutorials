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
"""Merge the sweep CSVs of one measurement into markdown tables.

    python v9_tile/report.py a      # results/a_{timing,timing_graph,clock,att,counters}.csv
    python v9_tile/report.py b

Derived columns:
- e2e MFMA util = back-to-back TFLOPS / (peak FLOP per clock x measured clock), peak per clock
  = 256 CUs x 4 SIMDs x 16384 FLOP / 16 cycles = 1.048576 TFLOP per GHz. The whole kernel in
  time, idle CUs and prologue/epilogue included.
- B/ns per CU = global bytes per k-step / time per k-step, where the k-step time comes from the
  back-to-back kernel time and the active CUs (prologue and epilogue included, so a slight
  underestimate).
- roofline = min(peak at the measured clock, compulsory intensity x 8 TB/s).
"""

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
PEAK_PER_GHZ = 1048.576  # TFLOPS per GHz
HBM_TBS = 8.0


def load(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return {}
    return {(r["tile"], int(r["M"]), int(r["N"]), int(r["K"])): r for r in csv.DictReader(open(path))}


def dims(tile):
    d = [int(x) for x in tile.removesuffix("r2").split("x")] if tile != "v9" else [256, 256]
    return d[0], d[1], d[2] if len(d) > 2 else 64


def f(x, fmt="{:.0f}"):
    return fmt.format(float(x)) if x not in (None, "") else "-"


def merged(prefix):
    t = load(f"{prefix}_timing.csv")
    g = load(f"{prefix}_timing_graph.csv")
    c = load(f"{prefix}_clock.csv")
    a = load(f"{prefix}_att.csv")
    p = load(f"{prefix}_counters.csv")
    keys = list(dict.fromkeys(list(g) + list(t)))
    rows = []
    for k in keys:
        tile, M, N, K = k
        BM, BN, BK = dims(tile)
        r = {"tile": tile, "M": M, "N": N, "K": K, "tiles": (M // BM) * (N // BN)}
        r["iso"] = t.get(k, {}).get("tflops")
        r["b2b"] = g.get(k, {}).get("tflops")
        r["b2b_us"] = g.get(k, {}).get("us_median")
        r["sclk"] = c.get(k, {}).get("sclk_ghz")
        r["power"] = c.get(k, {}).get("power_w")
        r["att"] = a.get(k, {}).get("att_mfma_eff")
        r["pro"] = a.get(k, {}).get("pro_ratio")
        r["epi"] = a.get(k, {}).get("epi_ratio")
        r["l2"] = p.get(k, {}).get("l2_hit")
        r["mfma_busy_pmc"] = p.get(k, {}).get("mfma_busy")
        if r["b2b"] and r["sclk"]:
            r["util"] = float(r["b2b"]) / (PEAK_PER_GHZ * float(r["sclk"]))
            ai = 2 * M * N * K / (2 * (M * K + N * K + M * N))
            r["roof"] = min(PEAK_PER_GHZ * float(r["sclk"]), ai * HBM_TBS)
        if r["b2b_us"]:
            waves = -(-r["tiles"] // 256)
            ns_per_kstep = float(r["b2b_us"]) * 1e3 / waves / (K // BK)
            r["bns"] = (BM + BN) * BK * 2 / ns_per_kstep
        rows.append(r)
    return rows


def table(rows, ref_tile="256x256"):
    ref = {(r["M"], r["N"], r["K"]): r for r in rows if r["tile"] == ref_tile}
    print("| tile | M x N x K | tiles | TFLOPS (isolated) | TFLOPS (back to back) | vs 256x256 same shape | clock GHz | "
          "power W | e2e MFMA util | ATT loop MFMA eff | prologue / epilogue | L2 hit | B/ns per CU |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        rr = ref.get((r["M"], r["N"], r["K"]))
        rel = f"{float(r['b2b']) / float(rr['b2b']):.0%}" if rr and r["b2b"] and rr["b2b"] and rr is not r else "-"
        print(f"| {r['tile']} | {r['M']}x{r['N']}x{r['K']} | {r['tiles']} | {f(r['iso'])} | {f(r['b2b'])} | {rel} | "
              f"{f(r['sclk'], '{:.2f}')} | {f(r['power'])} | {f(r.get('util'), '{:.0%}')} | {f(r['att'], '{:.1%}')} | "
              f"{r['pro'] or '-'} / {r['epi'] or '-'} | {f(r['l2'], '{:.1%}')} | {f(r.get('bns'), '{:.0f}')} |")


def main():
    prefix = sys.argv[1]
    table(merged(prefix))


if __name__ == "__main__":
    main()
