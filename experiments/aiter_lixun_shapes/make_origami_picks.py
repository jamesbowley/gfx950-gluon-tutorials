#!/usr/bin/env python3
"""picks_origami.csv: v28 at origami's tile and split, every applicable stream-K ending, plus controls.

    source $A16/v28_streamk_owner/env.sh; export HIP_VISIBLE_DEVICES=3     # host only: nothing launches
    PYTHONPATH=/home/jbowley/origami_build/site:$PYTHONPATH python $AIS/make_origami_picks.py \
        [--shapes $AIS/shapes_origami.csv] [--out $AIS/picks_origami.csv]
    ... python $AIS/make_origami_picks.py --hbl-check results/details_alltuned_rot2048.csv

Origami (rocm-libraries shared/origami at 55e6a027075, built against ROCm 7.2.4 into /home/jbowley/origami_build;
the pip origami 0.1.0 in /opt/venv predates its stream-K-aware ranking) ranks v28's candidate tiles for each shape:
- hardware from get_hardware_for_device(0), as hipBLASLt builds it;
- candidates: sweep.REFIT_TILES (the tiles that passed their REFIT stress check) and the transposes of the small ones
  (v28 swaps a tile whose short side is BN and runs C^T), less tiles with fewer than two pairs of k-steps (v28 and
  v9_tile assert there); a shape with none left is excluded;
- each config: the recommended bf16 MI, occupancy 1 for quadrant tiles (one program per CU) and 4 / W for small tiles
  (1024 / W programs), a workspace that never limits the split (v28's is sized by programs, not by the split);
  stream_k stays at its default (5), so rank_configs scores each tile under its k_split_aware grid.
compute_launch_parameters on the top tile gives the reduction hint (tree or parallel), the workgroup count and the
split factor s = ceil(workgroups / tiles).

The slots, in run_v20_vs_aiter.py's --v27-picks format, all at origami's tile:
- v9: data-parallel on v9_tile (sweep kind dp); pdp: v28's persistent data-parallel loop;
- rs, ow: fewer tiles than programs and s >= 2: the one-tile split on s programs per tile with the reduce-scatter
  and owner endings (rs{s}, ow{s}; rsall / owall where s reaches every program the tile can take); a wave or more
  with leftover tiles: rsall / owall (origami's split there is a fractional tiles-per-workgroup, not a count per
  leftover tile);
- 2t, sp: two-tile and spread, where valid and the tile makes a wave or more (below a wave origami gives a count per
  tile, which they do not take);
- rsb, owb (and rsp, owp where bf16_pairs is valid): rs / ow with bf16 (bf16-pair) partials, on shapes whose AITER
  pick reduces K in bf16 (bf16_reduce in the shapes CSV);
- cap: v28's own capped split at origami's tile (a comparison, not part of origami's pick);
- refit: the REFIT rule's single top pick (control): v27_streamk_tile/make_tuned_picks.short_list's best where a
  side is under 512, refit_rule.pick (compute-bound regime) where both are at least 512.
A (kind, tile) already in an earlier slot is not repeated: the later slot's alias column names the earlier one.
"""

import argparse
import collections
import csv
import importlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
A16 = os.path.abspath(os.path.join(HERE, "..", "..", "kernels", "gemm", "intra_wave", "a16w16"))
sys.path.insert(0, A16)
sweep = importlib.import_module("v28_streamk_owner.sweep")

import torch  # noqa: E402,F401  (loads the HIP runtime origami links against)
import origami  # noqa: E402

SLOTS = ("v9", "pdp", "rs", "ow", "2t", "sp", "rsb", "owb", "rsp", "owp", "cap", "refit")
WORKSPACE = 1 << 40


def transpose(tile):
    bm, bn, *rest = tile.split("x")
    return "x".join([bn, bm] + rest)


def candidate_tiles():
    v28 = sweep.v27_module()
    out = []
    for t in sweep.REFIT_TILES:
        for s in (t, transpose(t)):
            if s in out:
                continue
            try:
                geo = v28.tile_geometry(s)
            except AssertionError:
                continue
            if s == t or geo["small"]:
                out.append(s)
    return out


def tile_dims(tile):
    return [int(x) for x in sweep.full_spec(tile).split("x")]


class Origami:
    def __init__(self):
        self.hw = origami.get_hardware_for_device(0)
        self.dt = origami.data_type_t.BFloat16
        self.mi = self.hw.get_recommended_matrix_instruction(self.dt)
        self.v28 = sweep.v27_module()

    def problem(self, M, N, K):
        p = origami.problem_t()
        p.size = origami.dim3_t(M, N, K)
        p.batch = 1
        p.a_transpose, p.b_transpose = origami.transpose_t.T, origami.transpose_t.N
        p.a_dtype = p.b_dtype = p.c_dtype = p.d_dtype = p.mi_dtype = self.dt
        p.a_mx_block_size = p.b_mx_block_size = 0
        return p

    def config(self, tile, occupancy=None):
        bm, bn, bk = tile_dims(tile)
        if occupancy is None:
            geo = self.v28.tile_geometry(tile)
            occupancy = geo["num_programs"] // 256
        c = origami.config_t()
        c.mt = origami.dim3_t(bm, bn, bk)
        c.mi = self.mi
        c.occupancy = occupancy
        c.workspace_size = WORKSPACE
        c.workspace_size_per_elem_c = 4
        return c

    def rank(self, M, N, K, tiles, occupancy=None):
        """[(tile, latency)] best first."""
        p = self.problem(M, N, K)
        cfgs = [self.config(t, occupancy) for t in tiles]
        by_dims = collections.defaultdict(list)
        for t in tiles:
            by_dims[tuple(tile_dims(t))].append(t)
        out, used = [], collections.Counter()
        for r in origami.rank_configs(p, self.hw, cfgs, origami.origami.model_t.gemm):
            d = (r.config.mt.m, r.config.mt.n, r.config.mt.k)
            t = by_dims[d][used[d]]
            used[d] += 1
            out.append((t, r.latency))
        return out

    def launch(self, M, N, K, tile):
        """(reduction, workgroups, split factor) of origami's k_split_aware launch at tile."""
        red, wgs, _cus, _steps, split = origami.origami.compute_launch_parameters(
            self.problem(M, N, K), self.hw, self.config(tile), origami.grid_selection_t.k_split_aware)
        return str(red).split(".")[-1].lower(), int(wgs), int(split)


def split_kind(ending, s, tile, M, N, K, partials=""):
    """The valid one-tile kind for s programs per tile (None: s < 2 or no valid split), as rs{s} / ow{s}, or
    rsall / owall where s reaches every program the tile can take."""
    if s is None or s < 2:
        return None
    sh = sweep.shape_info(tile, M, N, K)
    if sh["S"] == 0:
        return None
    for kind in (f"{ending}{s}{partials}", f"{ending}all{partials}"):
        if sweep.valid(kind, tile, M, N, K):
            return kind
    return None


def refit_pick(M, N, K):
    """(kind, tile) of the REFIT control, or None."""
    sys.path.insert(0, os.path.join(A16, "v27_streamk_tile"))
    import make_tuned_picks  # noqa: F401  (imports v27's refit_rule and sweep)
    import refit_rule

    if M >= 512 and N >= 512:
        kind, tile = refit_rule.pick(M, N, K)
    else:
        sl = make_tuned_picks.short_list(M, N, K)
        if not sl:
            return None
        _, kind, tile, _ = sl[0]
    if kind != "dp" and sweep.shape_info(tile, M, N, K)["pairs"] < 2:
        kind = "dp"
    if sweep.shape_info(tile, M, N, K)["pairs"] < 2:
        return None
    return kind, tile


def n_max(kind, tile, M, N, K):
    """Most programs on one tile (the bound in the mean-error check)."""
    sh = sweep.shape_info(tile, M, N, K)
    sk = sweep.split_kind(kind)
    if not sk or sh["S"] == 0:
        return 1
    _, n, _ = sk
    if n is not None:
        return n
    return min(sh["P"] // sh["S"] + 1, sh["pairs"])


def pick_row(M, N, K, slot, kind, tile, info):
    v28 = sweep.v27_module()
    geo = v28.tile_geometry(tile)
    if kind == "dp":
        runner, spec, opts = "v9_tile", v28.fallback_spec(tile, geo), {}
    else:
        runner, spec, opts = "v27", "", sweep.kind_env(kind, tile, M, N, K)
    return dict(M=M, N=N, K=K, slot=slot, kind=kind, tile=tile, runner=runner, spec=spec,
                options=json.dumps(opts, sort_keys=True), n_max=n_max(kind, tile, M, N, K), alias="", **info)


def picks_for(o, tiles, s):
    M, N, K = int(s["M"]), int(s["N"]), int(s["K"])
    cands = [t for t in tiles if sweep.shape_info(t, M, N, K)["pairs"] >= 2]
    if not cands:
        return None, "no tile with two pairs of k-steps"
    ranked = o.rank(M, N, K, cands)
    tile = ranked[0][0]
    red, wgs, split = o.launch(M, N, K, tile)
    sh = sweep.shape_info(tile, M, N, K)
    info = dict(o_tile=tile, o_latency=f"{ranked[0][1]:.1f}", o_top3=" ".join(t for t, _ in ranked[:3]),
                o_reduction=red, o_wgs=wgs, o_split=split, o_tiles=sh["tiles"], o_S=sh["S"], o_P=sh["P"],
                o_pairs=sh["pairs"], bf16_reduce=s.get("bf16_reduce", ""))
    slots = [("v9", "dp", tile), ("pdp", "pdp", tile)]
    if sh["tiles"] < sh["P"]:
        rs, ow = split_kind("rs", split, tile, M, N, K), split_kind("ow", split, tile, M, N, K)
        bs = split
    else:
        rs = "rsall" if sh["S"] and sweep.valid("rsall", tile, M, N, K) else None
        ow = "owall" if sh["S"] and sweep.valid("owall", tile, M, N, K) else None
        bs = None
    slots += [("rs", rs, tile), ("ow", ow, tile)]
    if sh["tiles"] >= sh["P"]:
        slots += [(k, k, tile) for k in ("2t", "sp") if sweep.valid(k, tile, M, N, K)]
    if s.get("bf16_reduce"):
        for slot, base, suf in (("rsb", rs, "b"), ("owb", ow, "b"), ("rsp", rs, "p"), ("owp", ow, "p")):
            if base:
                k = base + suf
                if sweep.valid(k, tile, M, N, K):
                    slots.append((slot, k, tile))
    slots.append(("cap", "cap", tile))
    rp = refit_pick(M, N, K)
    if rp:
        slots.append(("refit", rp[0], rp[1]))
    rows, seen = [], {}
    for slot, kind, t in slots:
        if not kind:
            continue
        r = pick_row(M, N, K, slot, kind, t, info)
        if (kind, t) in seen:
            r["alias"] = seen[(kind, t)]
        else:
            seen[(kind, t)] = slot
        rows.append(r)
    return rows, ""


def hbl_check(o, details):
    """How often origami's top tile over hipBLASLt-like candidates (every tile hipBLASLt used in DETAILS, occupancy
    1) is the tile hipBLASLt ran, per scope."""
    rows = list(csv.DictReader(open(details)))
    tiles = sorted({r["hbl_tile"] for r in rows if r.get("hbl_tile")}, key=lambda t: [int(x) for x in t.split("x")])
    agree, top3, n = collections.Counter(), collections.Counter(), collections.Counter()
    for r in rows:
        if not r.get("hbl_tile"):
            continue
        M, N, K = int(r["M"]), int(r["N"]), int(r["K"])
        ranked = [t for t, _ in o.rank(M, N, K, tiles, occupancy=1)]
        for g in (r["scope"], "all"):
            n[g] += 1
            agree[g] += ranked[0] == r["hbl_tile"]
            top3[g] += r["hbl_tile"] in ranked[:3]
    print(f"{len(tiles)} hipBLASLt tiles as candidates")
    for g in sorted(n):
        print(f"{g}: {n[g]} rows, origami top 1 = hipBLASLt's tile on {agree[g]} ({agree[g] / n[g]:.0%}), "
              f"in its top 3 on {top3[g]} ({top3[g] / n[g]:.0%})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shapes", default=os.path.join(HERE, "shapes_origami.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "picks_origami.csv"))
    ap.add_argument("--excluded", default=os.path.join(HERE, "shapes_origami_no_tile.csv"))
    ap.add_argument("--hbl-check", metavar="DETAILS", help="compare origami with the hipBLASLt tiles of a details CSV")
    args = ap.parse_args()
    o = Origami()
    if args.hbl_check:
        hbl_check(o, args.hbl_check)
        return
    tiles = candidate_tiles()
    print(f"{len(tiles)} candidate tiles: {' '.join(tiles)}")
    rows, excluded = [], []
    for s in csv.DictReader(open(args.shapes)):
        got, why = picks_for(o, tiles, s)
        if got is None:
            excluded.append({**s, "reason": why})
        else:
            rows += got
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if excluded:
        with open(args.excluded, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(excluded[0]))
            w.writeheader()
            w.writerows(excluded)
    shapes = {(r["M"], r["N"], r["K"]) for r in rows}
    timed = [r for r in rows if not r["alias"]]
    print(f"{len(rows)} slots ({len(timed)} to time) for {len(shapes)} shapes -> {args.out}; "
          f"{len(excluded)} shapes excluded -> {args.excluded}")
    first = {}
    for r in rows:
        first.setdefault((r["M"], r["N"], r["K"]), r)
    for col in ("o_tile", "o_reduction", "o_split"):
        c = collections.Counter(r[col] for r in first.values())
        print(f"{col}: " + ", ".join(f"{k} {v}" for k, v in c.most_common(12)))
    c = collections.Counter(r["slot"] for r in timed)
    print("timed slots: " + ", ".join(f"{k} {c[k]}" for k in SLOTS if c[k]))


if __name__ == "__main__":
    main()
