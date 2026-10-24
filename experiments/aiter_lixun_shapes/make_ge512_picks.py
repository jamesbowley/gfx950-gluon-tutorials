#!/usr/bin/env python3
"""picks_ge512_losers.csv: larger tiles and splits for the both-sides->=512 shapes v27 auto lost by more than 4%.

    source $A16/v27_streamk_tile/env.sh
    python $AIS/make_ge512_picks.py [--details $AIS/results/details_alltuned_rot2048.csv]

For each such shape (gap against AITER's pick < -4% in the all-tuned run, from the per-round time medians), the slots:
- best: v27_streamk_tile/refit_rule.pick, the REFIT nearest-neighbour rule (compute-bound regime);
- tile2, tile3, fp32: cap at 256x256x64, 256x128x64 and 128x128x128 (2026-10-17's most frequent winning tiles);
- x5, x6, x7: the one-tile reduce-scatter split at those tiles on 4 programs per leftover tile (2026-10-17's winners
  used 3-6), or the nearest valid n (3, 5, 6, 2), else all programs.
Points v27 cannot run (fewer than two pairs of k-steps, or no valid split) are left out.
"""

import argparse
import csv
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
A16 = os.path.join(HERE, "..", "..", "kernels", "gemm", "intra_wave", "a16w16")
sys.path.insert(0, os.path.join(A16, "v27_streamk_tile"))
import refit_rule  # noqa: E402
import sweep  # noqa: E402

TILES = ("256x256x64", "256x128x64", "128x128x128")
CAP_SLOTS, RS_SLOTS = ("tile2", "tile3", "fp32"), ("x5", "x6", "x7")


def ms(r, name):
    xs = [float(x) for x in (r.get(f"{name}_ms") or "").split()]
    return statistics.median(xs) if xs else float("nan")


def ours(r):
    ts = [ms(r, n) for n in ("v20",) + tuple(f"pk_{s}" for s in ("best", "tile2", "tile3", "x5"))
          if str(r.get(f"{n}_ok")) == "True"]
    ts = [t for t in ts if t == t]
    return min(ts) if ts else float("nan")


def row(M, N, K, slot, kind, tile):
    v27 = sweep.v27_module()
    if kind == "dp":
        return dict(M=M, N=N, K=K, slot=slot, kind=kind, tile=tile, runner="v9_tile",
                    spec=v27.fallback_spec(tile, v27.tile_geometry(tile)), options="{}")
    return dict(M=M, N=N, K=K, slot=slot, kind=kind, tile=tile, runner="v27", spec="",
                options=json.dumps(sweep.kind_env(kind, tile, M, N, K), sort_keys=True))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--details", default=os.path.join(HERE, "results", "details_alltuned_rot2048.csv"))
    ap.add_argument("--shapes", default=os.path.join(HERE, "shapes_all_tuned.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "picks_ge512_losers.csv"))
    args = ap.parse_args()

    losers = set()
    for r in csv.DictReader(open(args.details)):
        o, p = ours(r), ms(r, "pick")
        if r["scope"] == "both >= 512" and o == o and p / o - 1 < -0.04:
            losers.add((int(r["M"]), int(r["N"]), int(r["K"]), r["out"]))
    rows, only = [], []
    for i, s in enumerate(csv.DictReader(open(args.shapes))):
        M, N, K = int(s["M"]), int(s["N"]), int(s["K"])
        if (M, N, K, s["out"]) not in losers:
            continue
        only.append(i)
        kind, tile = refit_rule.pick(M, N, K)
        rows.append(row(M, N, K, "best", kind, tile))
        for t, cs, rs in zip(TILES, CAP_SLOTS, RS_SLOTS):
            if sweep.shape_info(t, M, N, K)["pairs"] < 2:
                continue
            rows.append(row(M, N, K, cs, "cap", t))
            split = next((f"rs{n}" for n in (4, 3, 5, 6, 2) if sweep.valid(f"rs{n}", t, M, N, K)),
                         "rsall" if sweep.valid("rsall", t, M, N, K) else None)
            if split:
                rows.append(row(M, N, K, rs, split, t))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} picks for {len(only)} shapes -> {args.out}")
    print("--only " + ",".join(map(str, only)))


if __name__ == "__main__":
    main()
