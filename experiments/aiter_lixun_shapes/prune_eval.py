#!/usr/bin/env python3
"""How much each search-space reduction would have cost on the exhaustive v27 sweeps (REFIT, held-out, WORST10).

    python prune_eval.py [--origami ORIGAMI_RANKS.json]

For each shape, the reference is the fastest correct point of the full sweep (us_median, the merged rounds). A rule
keeps a subset of the points; its loss is the subset's fastest over the reference, minus 1. Reported per dataset and
over all: shapes, points kept (share of the full sweep), loss median, 90th percentile, max, and shapes losing more than
1% and 2%. Rounds: the same with round 1 (us_r1) deciding which points get a second round.
"""

import argparse
import collections
import csv
import json
import os
import statistics

RES = "/home/jbowley/repos/gfx950-gluon-tutorials/kernels/gemm/intra_wave/a16w16/v27_streamk_tile/results"
DATASETS = ("refit.csv", "heldout.csv", "worst10.csv")
COARSE_N = {"2", "3", "4", "6", "8", "12", "16", "all"}


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def split(kind):
    """(ending, n, partials) of a one-tile kind, else None."""
    if kind[:2] in ("rs", "ow") and kind != "all":
        p = kind[-1] if kind[-1] in "bp" else ""
        n = kind[2:len(kind) - len(p)]
        return kind[:2], n, p
    return None


def load():
    data = {}
    for f in DATASETS:
        shapes = collections.defaultdict(list)
        for r in csv.DictReader(open(os.path.join(RES, f))):
            if r["correct"] != "True" or r.get("rep", "0") != "0":
                continue
            t = fnum(r["us_median"])
            if t != t:
                continue
            r["t"], r["t1"] = t, fnum(r.get("us_r1"))
            shapes[(int(r["M"]), int(r["N"]), int(r["K"]))].append(r)
        data[f] = shapes
    return data


def config_key(r):
    return (r["tile"], r["kernel"], r["policy"], r["fixup"], r["streamk_tiles"], r["programs"], r["n"], r["n_hi"],
            r["ending"], r["grid"], (split(r["kind"]) or ("", "", ""))[2])


def evaluate(data, name, keep, fp32_only=True):
    """keep(shape, points) -> subset. fp32_only: the reference and the rule both exclude bf16 / bf16-pair partials."""
    lines, allloss, allkept, alltot = [], [], 0, 0
    for f, shapes in data.items():
        loss, kept, tot = [], 0, 0
        for sh, pts in shapes.items():
            if fp32_only:
                pts = [p for p in pts if not (split(p["kind"]) and split(p["kind"])[2])]
            if not pts:
                continue
            best = min(p["t"] for p in pts)
            sub = keep(sh, pts)
            kept += len(sub) if isinstance(sub, list) else sub[1]
            tot += len(pts)
            got = min((p["t"] for p in (sub if isinstance(sub, list) else sub[0])), default=float("inf"))
            loss.append(got / best - 1)
        allloss += loss
        allkept += kept
        alltot += tot
        lines.append(row(f.replace(".csv", ""), loss, kept, tot))
    lines.append(row("**all**", allloss, allkept, alltot))
    return f"**{name}**\n\n| dataset | shapes | points kept | loss median | 90th | max | > 1% | > 2% |\n" \
           "|---|---:|---:|---:|---:|---:|---:|---:|\n" + "\n".join(lines) + "\n"


def row(name, loss, kept, tot):
    s = sorted(loss)
    q = s[int(0.9 * (len(s) - 1))] if s else float("nan")
    return (f"| {name} | {len(s)} | {kept / tot:.0%} ({kept / max(len(s), 1):.0f}/shape) | "
            f"{statistics.median(s) * 100:.2f}% | {q * 100:.2f}% | {max(s) * 100:.2f}% | "
            f"{sum(v > 0.01 for v in s)} | {sum(v > 0.02 for v in s)} |")


def dedupe(sh, pts):
    seen = {}
    for p in pts:
        seen.setdefault(config_key(p), p)
    return list(seen.values())


def coarse_n(sh, pts):
    return [p for p in pts if not split(p["kind"]) or split(p["kind"])[1] in COARSE_N]


def rs_only(sh, pts):
    return [p for p in pts if not (split(p["kind"]) and split(p["kind"])[0] == "ow")]


def screen(delta, kinds=("dp", "cap", "rsall", "owall")):
    """Time a few kinds at every tile, keep every point at the tiles whose best screen is within delta of the best."""
    def keep(sh, pts):
        by_tile = collections.defaultdict(list)
        for p in pts:
            by_tile[p["tile"]].append(p)
        scr = {t: min((p["t1"] if p["t1"] == p["t1"] else p["t"] for p in ps if p["kind"] in kinds),
                      default=float("inf")) for t, ps in by_tile.items()}
        lo = min(scr.values())
        tiles = [t for t, v in scr.items() if v <= lo * (1 + delta)]
        sub = [p for t in tiles for p in by_tile[t]]
        cost = sum(1 for p in pts if p["kind"] in kinds) + sum(1 for p in sub if p["kind"] not in kinds)
        return sub, cost
    return keep


def origami_topk(ranks, k):
    def keep(sh, pts):
        r = ranks.get(f"{sh[0]}x{sh[1]}x{sh[2]}")
        if not r:
            return pts
        tiles = set(r[:k])
        return [p for p in pts if p["tile"] in tiles]
    return keep


def rounds(delta):
    """Every point gets round 1; only those within delta of the shape's best round-1 time get more rounds. The pick
    is the fastest by the merged time among those (the others keep only their round-1 time)."""
    def keep(sh, pts):
        lo = min(p["t1"] for p in pts if p["t1"] == p["t1"])
        close = [p for p in pts if p["t1"] == p["t1"] and p["t1"] <= lo * (1 + delta)]
        pick = min(close, key=lambda p: p["t"])
        # Against the full sweep's two rounds of every point.
        return [pick], (len(pts) + len(close)) / 2
    return keep


def pipeline(delta_tile, delta_round, kinds=("dp", "cap", "rsall", "owall")):
    """Duplicates removed, the tile screen at delta_tile, then a second round only within delta_round of round 1's
    best; cost against the full sweep's two rounds of every point."""
    scr = screen(delta_tile, kinds)

    def keep(sh, pts):
        pts = dedupe(sh, pts)
        sub, round1 = scr(sh, pts)
        lo = min(p["t1"] for p in sub if p["t1"] == p["t1"])
        close = [p for p in sub if p["t1"] == p["t1"] and p["t1"] <= lo * (1 + delta_round)]
        return [min(close, key=lambda p: p["t"])], (round1 + len(close)) / 2
    return keep


def drop(kinds):
    def keep(sh, pts):
        return [p for p in pts if not any(p["kind"] == k or (k.endswith("*") and p["kind"].startswith(k[:-1]))
                                          for k in kinds)]
    return keep


def chain(*rules):
    def keep(sh, pts):
        cost = None
        for r in rules:
            out = r(sh, pts)
            if isinstance(out, tuple):
                pts, cost = out
            else:
                pts = out
        return (pts, cost) if cost is not None else pts
    return keep


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--origami", default=None, help="JSON {MxNxK: [tiles best first]} from origami_ranks.py")
    ap.add_argument("--out", default="prune_eval.md")
    args = ap.parse_args()
    data = load()
    L = ["# Search-space reductions replayed on the exhaustive v27 sweeps", ""]
    L.append(evaluate(data, "full sweep (reference)", lambda sh, p: p))
    L.append(evaluate(data, "exact duplicates removed (same launch)", dedupe))
    L.append(evaluate(data, "n in {2, 3, 4, 6, 8, 12, 16, all}", coarse_n))
    L.append(evaluate(data, "reduce-scatter only (no owner ending)", rs_only))
    for d in (0.02, 0.05, 0.10, 0.20):
        L.append(evaluate(data, f"tile screen (dp, cap, rsall, owall at every tile; all kinds within {d:.0%})",
                          screen(d)))
    for d in (0.03, 0.05):
        L.append(evaluate(data, f"second round only within {d:.0%} of round 1's best", rounds(d)))
    if args.origami:
        ranks = json.load(open(args.origami))
        for k in (1, 3, 5, 8):
            L.append(evaluate(data, f"origami top {k} tiles, every kind", origami_topk(ranks, k)))
        L.append(evaluate(data, "origami top 5 tiles, coarse n", chain(origami_topk(ranks, 5), coarse_n)))
    L.append(evaluate(data, "screen within 10%, then coarse n", chain(screen(0.10), coarse_n)))
    L.append(evaluate(data, "no pdp512 / pdp768 / pdp1024", drop(("pdp512", "pdp768", "pdp1024"))))
    L.append(evaluate(data, "no two-tile or spread", drop(("2t", "sp"))))
    L.append(evaluate(data, "no cap or all (rules that repeat other points' launches)", drop(("cap", "all"))))
    for dt, dr in ((0.20, 0.05), (0.10, 0.05), (0.20, 0.03)):
        L.append(evaluate(data, f"pipeline: duplicates removed, tile screen {dt:.0%}, second round within {dr:.0%}",
                          pipeline(dt, dr)))
    wins = collections.Counter()
    for shapes in data.values():
        for pts in shapes.values():
            pts = [p for p in pts if not (split(p["kind"]) and split(p["kind"])[2])]
            k = min(pts, key=lambda p: p["t"])["kind"]
            wins[k if not split(k) else f"{split(k)[0]}{'all' if split(k)[1] == 'all' else 'N'}"] += 1
    L.append("**fastest kind (fp32 partials), all datasets:** " + ", ".join(f"{k} {v}" for k, v in wins.most_common()))
    with open(args.out, "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
