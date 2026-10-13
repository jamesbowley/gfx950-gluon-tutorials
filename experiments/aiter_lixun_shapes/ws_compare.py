#!/usr/bin/env python3
"""Stream-K workspace placement test: the ws_base, ws_realloc and ws_realloc_pad sweeps (stream-K
split shapes, --pad512 --realloc --rounds 5 --v20-only).

    python ws_compare.py

- ws_base: P, locks and tile_sync allocated once per process (as in every earlier run).
- ws_realloc: a fresh workspace every round (--realloc-workspace).
- ws_realloc_pad: as ws_realloc, with 256 bytes after each 256 KB partial slot (--p-pad 256).

Per variant and layout: streamk_wip's spread between rounds, and the same for data-parallel only
(the persistent loop, which never touches P) as the control; then each variant's median time
against ws_base.
"""

import os
import statistics

from run_v20_vs_aiter import RESULTS, fnum, key, quantile, read_rows

VARIANTS = (("ws_base", "workspace fixed per process"), ("ws_realloc", "fresh workspace every round"),
            ("ws_realloc_pad", "fresh workspace every round, slots padded 256 B"))


def spread_cells(rows, col):
    xs = [x for x in (fnum(r[col]) for r in rows) if x == x]
    return (f"{statistics.median(xs):.2f}% | {quantile(xs, 0.9):.2f}% | {sum(x > 2 for x in xs)} | "
            f"{sum(x > 5 for x in xs)} | {max(xs):.1f}%")


def main():
    runs = {}
    for tag, _ in VARIANTS:
        path = os.path.join(RESULTS, f"details_{tag}.csv")
        if os.path.exists(path):
            runs[tag] = [r for r in read_rows(path) if r["v20_policy"] != "dp"]
    L = ["# Stream-K workspace placement (warm, stream-K split shapes, 5 fresh-buffer rounds)", ""]
    L += ["| Variant | Layout | Kernel | Shapes | median spread | p90 spread | over 2% | over 5% | max |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for tag, name in VARIANTS:
        rows = runs.get(tag)
        if not rows:
            continue
        for lay, lname in (("", "contiguous"), ("_pad", "padded")):
            for c, kname in (("v20", "streamk_wip (split)"), ("v20_dp", "data-parallel only (no P)")):
                L.append(f"| {name} | {lname} | {kname} | {len(rows)} | {spread_cells(rows, f'{c}{lay}_spread')} |")
    L += [""]
    base = {key(r["M"], r["N"], r["K"], r["out"]): r for r in runs.get("ws_base", [])}
    L += ["Median time against ws_base, per shape (this variant TFLOPS / ws_base TFLOPS - 1):", "",
          "| Variant | Layout | Kernel | median | p10 | p90 |", "|---|---|---|---:|---:|---:|"]
    for tag, name in VARIANTS[1:]:
        for lay, lname in (("", "contiguous"), ("_pad", "padded")):
            for c, kname in (("v20", "streamk_wip (split)"), ("v20_dp", "data-parallel only (no P)")):
                xs = []
                for r in runs.get(tag, []):
                    o = base.get(key(r["M"], r["N"], r["K"], r["out"]))
                    if o:
                        xs.append((fnum(r[f"{c}{lay}_tflops"]) / fnum(o[f"{c}{lay}_tflops"]) - 1) * 100)
                xs = [x for x in xs if x == x]
                if xs:
                    L.append(f"| {name} | {lname} | {kname} | {statistics.median(xs):+.2f}% | "
                             f"{quantile(xs, 0.1):+.1f}% | {quantile(xs, 0.9):+.1f}% |")
    L += [""]
    out = os.path.join(RESULTS, "summary_ws_compare.md")
    with open(out, "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
