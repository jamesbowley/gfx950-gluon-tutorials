#!/usr/bin/env python3
"""picks_all_tuned.csv: the REFIT rule's short list for each shape of shapes_all_tuned.csv whose small side is < 512.

    source $A16/v27_streamk_tile/env.sh     # (cds to a16w16; any idle GPU: the import needs one, nothing launches)
    python $AIS/make_tuned_picks.py [--shapes $AIS/shapes_all_tuned.csv] [--out $AIS/picks_all_tuned.csv]

v27_streamk_tile/refit_rule.py's nearest neighbours (its table, chosen variant per regime, MISSING_RATIO), ranked
instead of argmin: every (tile, kind) valid at the shape, by the mean log time over the best of its k nearest REFIT
shapes. The slots, in run_v20_vs_aiter.py's --v27-picks format:
- best, tile2, tile3: the best-scoring kind at each of the three best-scoring distinct tiles;
- x5: cap at the best tile, when the best kind is not cap already.

Routing is by the small side, not by M as refit_rule.regime does: for N < 512 <= M the shape is scored as (N, M, K)
in the memory-bound regime and each tile spec transposed (v27 swaps a tile whose short side is BN and runs C^T); a
transposed pick that sweep.valid rejects is dropped, and with none left the shape is scored as given. v27 needs two
pairs of k-steps (K_PAD >= 4 BK, the host asserts before choosing a policy), so a v27 kind with fewer becomes dp on
v9_tile at the same tile. Shapes with both sides >= 512 get no picks: v27 auto (the harness's v20 column) is their
rule, as on the 630; only where auto's tile has fewer than two pairs of k-steps (v27 asserts there) do they get
dp on v9_tile at that tile as slot best.
"""

import argparse
import csv
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
A16 = os.path.join(HERE, "..", "..", "kernels", "gemm", "intra_wave", "a16w16")
sys.path.insert(0, os.path.join(A16, "v27_streamk_tile"))
import refit_rule  # noqa: E402
import sweep  # noqa: E402

SLOTS = ("best", "tile2", "tile3")


def transpose(tile):
    bm, bn, *rest = tile.split("x")
    return "x".join([bn, bm] + rest)


def ranked(sh):
    """[(score, tile, kind)] for shape sh in the memory-bound regime, best first (refit_rule.knn_pick's scoring)."""
    chosen, shapes = refit_rule.table()
    g = refit_rule.regime(sh[0])
    train = shapes[g]
    feat, k = refit_rule.VARIANTS[chosen[g]]
    x = feat(sh)
    near = sorted(train, key=lambda o: (sum((a - b) ** 2 for a, b in zip(x, feat(o))), o))[:k]
    cands = sorted({c for o in near for c in train[o]})
    out = [(sum(math.log(train[o].get(c, refit_rule.MISSING_RATIO)) for o in near) / len(near), c[0], c[1])
           for c in cands]
    return sorted(out)


def short_list(M, N, K):
    """[(slot, kind, tile, score)] for (M, N, K), the tile spec in the output's M x N."""
    if M >= 512 and N >= 512:
        tile = sweep.v27_module().auto_tile(M, N, K)
        return [("best", "dp", tile, float("nan"))] if sweep.shape_info(tile, M, N, K)["pairs"] < 2 else []
    swap = M >= 512
    cands = []
    for score, tile, kind in ranked((N, M, K) if swap else (M, N, K)):
        t = transpose(tile) if swap else tile
        if refit_rule.valid(kind, t, M, N, K):
            cands.append((score, t, kind))
    if not cands:
        cands = [(s, t, kd) for s, t, kd in ranked((M, N, K)) if refit_rule.valid(kd, t, M, N, K)]
    picks, tiles = [], []
    for score, tile, kind in cands:
        if tile not in tiles:
            tiles.append(tile)
            picks.append((SLOTS[len(picks)], kind, tile, score))
            if len(picks) == len(SLOTS):
                break
    if picks and picks[0][1] != "cap":
        picks.append(("x5", "cap", picks[0][2], float("nan")))
    return picks


def pick_row(M, N, K, group, slot, kind, tile, score):
    v27 = sweep.v27_module()
    geo = v27.tile_geometry(tile)
    note = ""
    if kind != "dp" and sweep.shape_info(tile, M, N, K)["pairs"] < 2:
        note, kind = f"{kind} -> dp (one pair of k-steps)", "dp"
    if kind == "dp":
        runner, spec, opts = "v9_tile", v27.fallback_spec(tile, geo), {}
    else:
        runner, spec, opts = "v27", "", sweep.kind_env(kind, tile, M, N, K)
    return dict(M=M, N=N, K=K, group=group, slot=slot, kind=kind, tile=tile, runner=runner, spec=spec,
                options=json.dumps(opts, sort_keys=True), score=f"{score:.4f}", note=note)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shapes", default=os.path.join(HERE, "shapes_all_tuned.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "picks_all_tuned.csv"))
    args = ap.parse_args()

    rows, seen = [], set()
    for s in csv.DictReader(open(args.shapes)):
        M, N, K = int(s["M"]), int(s["N"]), int(s["K"])
        if (M, N, K) in seen:
            continue
        seen.add((M, N, K))
        ran = set()
        for slot, kind, tile, score in short_list(M, N, K):
            r = pick_row(M, N, K, s["scope"], slot, kind, tile, score)
            if (r["kind"], r["tile"]) not in ran:
                ran.add((r["kind"], r["tile"]))
                rows.append(r)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    shapes = {(r["M"], r["N"], r["K"]) for r in rows}
    kinds, tiles = {}, {}
    for r in rows:
        if r["slot"] == "best":
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
            tiles[r["tile"]] = tiles.get(r["tile"], 0) + 1
    print(f"{len(rows)} picks for {len(shapes)} shapes -> {args.out}; {sum(1 for r in rows if r['note'])} moved to dp")
    print("best kinds: " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])))
    print("best tiles: " + ", ".join(f"{k} {v}" for k, v in sorted(tiles.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
