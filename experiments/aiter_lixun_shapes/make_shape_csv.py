#!/usr/bin/env python3
"""One row per shape, in the spirit of supported_status_scoped.csv: the shape, the kernels each
side ran (with tiles), their TFLOPS and the gaps, warm and cold side by side.

    python make_shape_csv.py --warm results/details_v9fb_pad.csv --cold results/details_v9fb_pad_rot2048.csv \
        --out results/shapes_2026-10-06.csv

All times are the contiguous layout, as AITER calls the libraries. A --pad512 run also gets
streamk_wip's padded time, and every run with several rounds the spread between them.
Gaps are percent, positive meaning streamk_wip (or v9) is faster.
"""

import argparse
import csv

from pad_report import best_fields, times
from run_v20_vs_aiter import SHAPES_CSV, aiter_tile, fnum, key, pct, read_rows


def r1(x):
    return "" if x != x else round(x, 1)


def r2(x):
    return "" if x != x else round(x, 2)


def first_cijk(names):
    return next((k for k in (names or "").split(" | ") if "Cijk_" in k), "")


def state_cols(r):
    """The per-cache-state columns of one details row."""
    b = best_fields(r, "")
    t = times(r, "")
    pick, hbl = fnum(r["aiter_pick_tflops"]), fnum(r["hbl_tflops"])
    hbl_tile = "x".join(r["hbl_tile"].split("x")[:2])
    handoff = str(r.get("v20_v9_delegated", "")) == "True"
    label = r["v20_label"] + (", handed to v9" if handoff else "")
    kernel = r.get("v20_kernels_launched") or "v20_streamk_reduce_scatter"
    oks = [c for c in r if c.endswith("_ok") or c.endswith("_post_ok")]
    cols = dict(
        aiter_pick=r["aiter_pick"],
        aiter_kernel=r["aiter_kernel"] or first_cijk(r["hbl_kernels"]),
        aiter_tile=aiter_tile(r),
        aiter_tflops=r1(pick),
        aiter_ok=r["pick_ok"],
        hbl_tflops=r1(hbl),
        hbl_tile=hbl_tile,
        hbl_kernel=first_cijk(r.get("hbl_kernels_torch", "")),
        best_lib=b["best_lib"],
        best_tile=b["best_tile"],
        best_tflops=r1(b["best"]),
        v9_tflops=r1(t["v9"]),
        streamk_tflops=r1(t["v20"]),
        streamk_config=label,
        streamk_kernel=kernel,
        streamk_dp_tflops=r1(t["dp"]),
        streamk_vs_best_pct=r2(pct(t["v20"], b["best"])),
        streamk_vs_aiter_pct=r2(pct(t["v20"], pick)),
        streamk_vs_hbl_pct=r2(pct(t["v20"], hbl)),
        streamk_vs_v9_pct=r2(pct(t["v20"], t["v9"])),
        v9_vs_best_pct=r2(pct(t["v9"], b["best"])),
        streamk_spread_pct=r.get("v20_spread", ""),
        streamk_ok=all(str(r.get(c)) != "False" for c in oks if c.startswith("v20"))
        and str(r["v20_flags_left"]) == "0",
    )
    if "v20_pad_tflops" in r:
        tp = fnum(r["v20_pad_tflops"])
        cols.update(streamk_padded_tflops=r1(tp), streamk_padded_vs_best_pct=r2(pct(tp, b["best"])),
                    streamk_padded_spread_pct=r.get("v20_pad_spread", ""))
    if "v9new_tflops" in r:
        # --kernel v27: streamk is v27 auto; v26, v27 at 256x256x64 and the new AITER v9 from the same run.
        v26, t256, v9new = fnum(r["v26_tflops"]), fnum(r["v27t256_tflops"]), fnum(r["v9new_tflops"])
        cols.update(streamk_tile=r["v20_tile"], streamk_tile_S=r["v20_tile_S"], v26_tflops=r1(v26),
                    v26_config=r["v26_label"] + (", handed to v9" if str(r["v26_v9_delegated"]) == "True" else ""),
                    v27_256_tflops=r1(t256), v9new_tflops=r1(v9new), v9new_tile=r["v9new_tile"],
                    streamk_vs_v26_pct=r2(pct(t["v20"], v26)), v27_256_vs_v26_pct=r2(pct(t256, v26)),
                    streamk_vs_v27_256_pct=r2(pct(t["v20"], t256)), v26_vs_best_pct=r2(pct(v26, b["best"])),
                    v9new_vs_best_pct=r2(pct(v9new, b["best"])), v9new_vs_v9_pct=r2(pct(v9new, t["v9"])),
                    streamk_vs_v9new_pct=r2(pct(t["v20"], v9new)))
    return cols


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--warm", default=None, help="omit for a cold-only run")
    p.add_argument("--cold", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    shapes = {key(s["M"], s["N"], s["K"], s["out"]): s for s in read_rows(SHAPES_CSV)}
    warm = {key(r["M"], r["N"], r["K"], r["out"]): r for r in read_rows(args.warm)} if args.warm else None
    cold = {key(r["M"], r["N"], r["K"], r["out"]): r for r in read_rows(args.cold)}
    out = []
    for k, s in shapes.items():
        w = (warm if warm is not None else cold).get(k)
        if not w:
            continue
        row = dict(M=s["M"], N=s["N"], K=s["K"], bias=s["bias"], out=s["out"], src=w["src"], AI=s["AI"],
                   tiles=w["tiles"], waves=w["waves"], S=w["S"], k_steps=w["k_steps"], scope=s["scope"],
                   band=s["band"])
        if warm is not None:
            row.update({f"warm_{c}": v for c, v in state_cols(w).items()})
        if k in cold:
            row.update({f"cold_{c}": v for c, v in state_cols(cold[k]).items()})
        out.append(row)
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0]))
        wr.writeheader()
        wr.writerows(out)
    print(f"wrote {args.out}: {len(out)} shapes, {len(out[0])} columns")


if __name__ == "__main__":
    main()
