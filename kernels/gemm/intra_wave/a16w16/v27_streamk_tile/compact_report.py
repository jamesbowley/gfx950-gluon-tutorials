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
"""The compact launch's timing A/B (WORKLOG §13) from results/compact.csv (sweep.py COMPACT, run_prelim.sh with
PRESET=COMPACT), to results/compact_report.md.

    python v27_streamk_tile/compact_report.py

Every point was timed four times in a row: full grid, compact, full, compact. A mode's time is the mean of its
two; its repeat difference |a - b| / min(a, b). The noise band is the 90th percentile of all repeat differences
(both modes, every point). A point is beyond noise when compact / full differs from 1 by more than the band.
Points are grouped by set (MB, WQ, AITER) and by whether the compact launch applied (the idle tail dropped, the
same placement) or not (full tiles, data-parallel or every program working: the same launch twice, a control).
"""

import collections
import csv
import os
import statistics as S
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from v27_streamk_tile import sweep  # noqa: E402


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main():
    rows = list(csv.DictReader(open(os.path.join(HERE, "results", "compact.csv"))))
    bad = [r for r in rows if r["correct"] != "True"]
    pts = collections.defaultdict(dict)
    for r in rows:
        if r["error"] or not r["us_median"]:
            continue
        kind, full = sweep.base_kind(r["kind"])
        pts[(kind, r["tile"], int(r["M"]), int(r["N"]), int(r["K"]))][(full, r["rep"])] = r
    sets = {p: name for name, ps in sweep.compact_sets().items() for p in ps}
    recs, diffs = [], []
    for key, m in pts.items():
        if len(m) != 4:
            continue
        t = {}
        for full in (True, False):
            a, b = float(m[(full, "0")]["us_median"]), float(m[(full, "1")]["us_median"])
            t[full] = (a + b) / 2
            diffs.append(abs(a - b) / min(a, b))
        f, c = m[(True, "0")], m[(False, "0")]
        applied = c["grid"] and f["grid"] and int(c["grid"]) < int(f["grid"])
        what = "compact" if applied else "not applied"
        recs.append(dict(key=key, set=sets.get(key, "?"), what=what, full=t[True], compact=t[False],
                         ratio=t[False] / t[True], grid=(f["grid"], c["grid"]), S=c["S"], kernel=c["kernel"]))
    band = pct(diffs, 0.9)
    out = ["# The compact launch: timing A/B\n",
           f"{len(recs)} points (of {len(pts)}), each timed full, compact, full, compact; {len(bad)} rows not correct. "
           f"Noise band (90th percentile of the {len(diffs)} repeat differences): {100 * band:.1f}% "
           f"(median {100 * S.median(diffs):.2f}%, max {100 * max(diffs):.1f}%). Ratio = compact / full time "
           "(below 1: the compact launch is faster).\n",
           "| set | compact launch | points | median ratio | best | worst | faster beyond noise | slower beyond noise |",
           "|---|---|---|---|---|---|---|---|"]
    for name in ("MB", "WQ", "AITER"):
        for what in ("compact", "not applied"):
            rs = [r["ratio"] for r in recs if r["set"] == name and r["what"] == what]
            if not rs:
                continue
            out.append(f"| {name} | {what} | {len(rs)} | {S.median(rs):.3f} | {min(rs):.3f} | {max(rs):.3f} | "
                       f"{sum(x < 1 - band for x in rs)} | {sum(x > 1 + band for x in rs)} |")
    for what in ("compact", "not applied"):
        rs = [r["ratio"] for r in recs if r["what"] == what]
        d = [abs(x - 1) for x in rs]
        out.append(f"\n{what}: {len(rs)} points, |ratio - 1| 90th percentile {100 * pct(d, 0.9):.2f}%, 95th "
                   f"{100 * pct(d, 0.95):.2f}%, max {100 * max(d):.2f}%; {sum(x > 1 + band for x in rs)} slower and "
                   f"{sum(x < 1 - band for x in rs)} faster beyond the band." if rs else "")
    out.append("\nThe not-applied points launch the same kernel twice, so their spread is the noise of comparing two "
               "separately timed modes, wider than the repeat band.")
    slow = sorted((r for r in recs if r["ratio"] > 1 + band), key=lambda r: -r["ratio"])
    out.append(f"\n{len(slow)} points slower than the noise band"
               + (" (none: PASS)." if not slow else ":"))
    applied = [r for r in recs if r["what"] != "not applied"]
    out.append("\n## Applied points, by ratio\n")
    out.append("| set | kind | tile | shape | S | grid full / compact | full us | compact us | ratio |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(applied, key=lambda r: r["ratio"]):
        kind, tile, M, N, K = r["key"]
        out.append(f"| {r['set']} | {kind} | {tile} | {M}x{N}x{K} | {r['S']} | {r['grid'][0]} / {r['grid'][1]} | "
                   f"{r['full']:.2f} | {r['compact']:.2f} | {r['ratio']:.3f} |")
    if slow:
        out.append("\n## Slower than the noise band\n")
        for r in slow:
            out.append(f"- {r['key']}: {r['ratio']:.3f} ({r['what']})")
    text = "\n".join(out) + "\n"
    open(os.path.join(HERE, "results", "compact_report.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
