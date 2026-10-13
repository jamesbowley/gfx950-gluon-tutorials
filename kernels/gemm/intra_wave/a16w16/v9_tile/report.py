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
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
LAYOUT_ATT_DIR = os.path.join(HERE, "tmp", "layout_ab", "sweep_att")
PEAK_PER_GHZ = 1048.576  # TFLOPS per GHz
HBM_TBS = 8.0


def load(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return {}
    return {(r["tile"], int(r["M"]), int(r["N"]), int(r["K"])): r for r in csv.DictReader(open(path))}


def dims(tile):
    if tile == "v9":
        return 256, 256, 64
    d = [int(x) for x in re.match(r"^(\d+)x(\d+)(?:x(\d+))?", tile).groups(default="64")]
    return d[0], d[1], d[2]


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


def att_stall_split(tile, M, N, K):
    """att_stalls.analyze on a LAYOUT_AB point's ATT run: per loop iteration and wave, the stall cycles
    on vmcnt, on buffer_load issue, on lgkmcnt + ds_read, on s_barrier, and the cycles issued."""
    import att_stalls

    d = os.path.join(LAYOUT_ATT_DIR, f"{tile}_{M}x{N}x{K}")
    if not glob.glob(os.path.join(d, "**", "ui_*"), recursive=True):
        return None
    r = att_stalls.analyze(d)
    s = {k: v[1] for k, v in r["by"].items()}
    return (s.get("s_waitcnt vmcnt", 0) + s.get("s_waitcnt vmcnt+lgkmcnt", 0), s.get("buffer_load", 0),
            s.get("s_waitcnt lgkmcnt", 0) + s.get("ds_read", 0), s.get("s_barrier", 0), r["total_latency"])


def layout_table(prefix="layout_ab"):
    """WORKLOG §15's A/B, from its CSVs: each tile with the former hand-built layouts (no suffix) and
    with aiter's derived ones ("d", now the only layouts), side by side."""
    rows = {(r["tile"], r["M"], r["N"], r["K"]): r for r in merged(prefix)}
    p = load(f"{prefix}_counters.csv")
    print("| tile | M x N x K | layout | TFLOPS (back to back) | d / ours | clock GHz | power W | ATT loop MFMA eff | "
          "stalls per iteration: vmcnt / buffer_load / lgkm+ds_read / barrier (of issued) | LDS conflict cycles "
          "per access | LDS conflict, % of CU cycles | L2 hit |")
    print("|" + "---|" * 12)
    for (tile, M, N, K), r in rows.items():
        if tile.endswith("d"):
            continue
        base = tile.removesuffix("h")
        pair = (base + "d", M, N, K) in rows
        for t, name in ((tile, "ours"), (base + "d", "aiter (d)")) if pair else ((tile, "both (identical)"),):
            x = rows.get((t, M, N, K), {})
            c = p.get((t, M, N, K), {})
            st = att_stall_split(t, M, N, K)
            sp = f"{st[0]:.0f} / {st[1]:.0f} / {st[2]:.0f} / {st[3]:.0f} ({st[4]:.0f})" if st else "-"
            rel = "-"
            if name == "aiter (d)" and x.get("b2b") and r.get("b2b"):
                rel = f"{float(x['b2b']) / float(r['b2b']):.3f}"
            print(f"| {base} | {M}x{N}x{K} | {name} | {f(x.get('b2b'))} | {rel} | {f(x.get('sclk'), '{:.2f}')} | "
                  f"{f(x.get('power'))} | {f(x.get('att'), '{:.1%}')} | {sp} | "
                  f"{f(c.get('lds_conflict_per_access'), '{:.3f}')} | {f(c.get('lds_conflict_pct'), '{:.1%}')} | "
                  f"{f(x.get('l2'), '{:.1%}')} |")


def main():
    prefix = sys.argv[1]
    if prefix == "layout_ab":
        layout_table(prefix)
        return
    table(merged(prefix))


if __name__ == "__main__":
    main()
