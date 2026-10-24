#!/usr/bin/env python3
"""Where v27 is behind AITER on the all-tuned-shapes run: counts by group and band, and the losers' list.

    python tuned_all_report.py --details results/details_alltuned_rot2048.csv \
        [--summary results/summary_alltuned.md] [--losers results/losers_all_tuned.csv]

"Ours" is the fastest correct of v27 auto (the harness's v20 column, capped at auto_tile's tile) and the REFIT
short list (pk_best, pk_tile2, pk_tile3, pk_x5). Both-sides->=512 shapes have no short list, so theirs is v27 auto
alone, as on the 630. AITER is its tuned pick as dispatched, and the best library the faster of AITER's pick and
hipBLASLt; as in pad_report.best_fields (needle.py), library timings count whatever their check says, and the rows
where AITER's pick fails it are counted (FlyDSL split-K picks with bf16 output, a few bf16 ulps off).
gap = ours / library - 1 in speed (library time / our time - 1, from the per-round *_ms medians): negative means
behind. The 4% line is about 2 sigma of the REFIT noise.
"""

import argparse
import csv
import math
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
SLOTS = ("best", "tile2", "tile3", "x5")
LINES = (0.01, 0.04, 0.10)
SCOPES = ("M < 512", "N < 512 <= M", "both >= 512")
SMALL_BANDS = ((1, 15), (16, 63), (64, 255), (256, 511), (512, 1 << 30))


def f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v


def ok(r, name):
    return str(r.get(f"{name}_ok")) == "True"


def ms(r, name):
    """The median of a column's per-round times (ms; the TFLOPS columns are rounded to 0.1, too coarse for the
    smallest shapes)."""
    xs = [float(x) for x in (r.get(f"{name}_ms") or "").split()]
    return statistics.median(xs) if xs else math.nan


def tflops(r, t):
    return 2e-9 * int(r["M"]) * int(r["N"]) * int(r["K"]) / t if t == t else math.nan


def ours(r):
    """(ms, what) of the fastest correct v27 column."""
    best = (math.nan, "")
    cands = [("v20", f"auto {r.get('v20_tile', '')} {r.get('v20_label', '')}".strip())]
    cands += [(f"pk_{s}", f"{s}: {r.get(f'pk_{s}_kind', '')} {r.get(f'pk_{s}_tile', '')}") for s in SLOTS
              if r.get(f"pk_{s}_kind")]
    for name, what in cands:
        t = ms(r, name)
        if ok(r, name) and t == t and not t >= best[0]:
            best = (t, what)
    return best


def small_band(r):
    s = min(int(r["M"]), int(r["N"]))
    for lo, hi in SMALL_BANDS:
        if lo <= s <= hi:
            return f"{lo}-{hi}" if hi < 1 << 30 else f">={lo}"


def group_order(g):
    """Bands by their lower bound, other groups by name."""
    head = g.lstrip(">=").split("-")[0]
    return (0, int(head), g) if head.isdigit() else (1, 0, g)


def tally(rows, key):
    """[(group, n, unsupported, ahead, within 1%, behind > 1% / 4% / 10%, median gap)] against key."""
    out = []
    for g in sorted({r["_g"] for r in rows}, key=group_order):
        rs = [r for r in rows if r["_g"] == g]
        gaps = [r[key] for r in rs if r[key] == r[key]]
        out.append((g, len(rs), len(rs) - len(gaps), sum(x > 0.01 for x in gaps), sum(-0.01 <= x <= 0.01 for x in gaps),
                    *(sum(x < -line for x in gaps) for line in LINES),
                    statistics.median(gaps) if gaps else math.nan))
    return out


def table(title, rows, groupf, key, label):
    for r in rows:
        r["_g"] = groupf(r)
    L = [f"**{title}**", "",
         f"| {label} | shapes | unsupported | ahead > 1% | within 1% | behind > 1% | behind > 4% | behind > 10% "
         "| median gap |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    t = tally(rows, key)
    for g, n, uns, ah, wi, b1, b4, b10, med in t:
        L.append(f"| {g} | {n} | {uns} | {ah} | {wi} | {b1} | {b4} | {b10} | {med * 100:+.1f}% |")
    gaps = [r[key] for r in rows if r[key] == r[key]]
    L.append(f"| **all** | {len(rows)} | {len(rows) - len(gaps)} | {sum(x > 0.01 for x in gaps)} | "
             f"{sum(-0.01 <= x <= 0.01 for x in gaps)} | " + " | ".join(str(sum(x < -l for x in gaps)) for l in LINES)
             + f" | {statistics.median(gaps) * 100 if gaps else math.nan:+.1f}% |")
    return L + [""]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--details", default=os.path.join(HERE, "results", "details_alltuned_rot2048.csv"))
    ap.add_argument("--summary", default=os.path.join(HERE, "results", "summary_alltuned.md"))
    ap.add_argument("--losers", default=os.path.join(HERE, "results", "losers_all_tuned.csv"))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.details)))
    for r in rows:
        r["_ours_ms"], r["_what"] = ours(r)
        pick, hbl = ms(r, "pick"), ms(r, "hbl")
        libs = [(x, name) for x, name in ((pick, f"AITER {r['aiter_pick']}"), (hbl, "hipBLASLt")) if x == x]
        r["_lib_ms"], r["_lib_name"] = min(libs) if libs else (math.nan, "")
        r["_ours"], r["_lib"] = tflops(r, r["_ours_ms"]), tflops(r, r["_lib_ms"])
        r["_gap_aiter"] = pick / r["_ours_ms"] - 1  # NaN propagates
        r["_gap_best"] = r["_lib_ms"] / r["_ours_ms"] - 1
        best = ms(r, "pk_best") if ok(r, "pk_best") else math.nan
        r["_rule_gain"] = best / r["_ours_ms"] - 1
        v9n = ms(r, "v9new") if ok(r, "v9new") else math.nan
        r["_v9new_gap"] = pick / v9n - 1

    L = [f"# v27 against AITER on all gfx950 tuned bf16 shapes ({os.path.basename(args.details)})", "",
         f"{len(rows)} rows. Ours: the fastest correct of v27 auto and the REFIT short list (both sides >= 512: v27 "
         "auto alone). gap = ours / reference - 1.", ""]
    untuned = sum(r.get("src") != "tuned" for r in rows)
    bad_pick = sum(not ok(r, "pick") for r in rows)
    bad_fly = sum(not ok(r, "pick") and r["aiter_pick"] == "flydsl" for r in rows)
    L += [f"- AITER dispatched an untuned config on {untuned} rows. Its pick failed the harness's check (allclose atol "
          f"0.1, rtol 1%) on {bad_pick} rows, {bad_fly} of them FlyDSL; those timings still count, as in earlier runs.",
          ""]
    for key, ref in (("_gap_aiter", "AITER's pick"), ("_gap_best", "the best library (AITER's pick or hipBLASLt)")):
        L += table(f"Against {ref}, by group", rows, lambda r: r["scope"], key, "group")
        L += table(f"Against {ref}, by min(M, N)", rows, small_band, key, "min(M, N)")
    L += table("Against AITER's pick, by AITER's library", rows, lambda r: r["aiter_pick"], "_gap_aiter", "AITER pick")

    win = {}
    for r in rows:
        if r["_what"]:
            k = r["_what"].split(":")[0].split()[0]
            win[k] = win.get(k, 0) + 1
    gains = [r["_rule_gain"] for r in rows if r["_rule_gain"] == r["_rule_gain"]]
    v9g = [r["_v9new_gap"] for r in rows if r["_v9new_gap"] == r["_v9new_gap"]]
    L += ["**Which of our columns was fastest:** " + ", ".join(f"{k} {v}" for k, v in sorted(win.items(), key=lambda kv: -kv[1])),
          "",
          f"**The short list over the rule's single pick (pk_best):** median {statistics.median(gains) * 100:+.1f}%, "
          f"90th percentile {sorted(gains)[int(0.9 * len(gains))] * 100:+.1f}%, {sum(g > 0.02 for g in gains)} of "
          f"{len(gains)} shapes more than 2% faster than the single pick." if gains else "", "",
          f"**New AITER v9 against AITER's pick:** median {statistics.median(v9g) * 100:+.1f}% "
          f"({len(v9g)} rows correct)." if v9g else "", ""]

    losers = sorted((r for r in rows if r["_gap_best"] == r["_gap_best"] and r["_gap_best"] < -0.01)
                    or [], key=lambda r: r["_gap_best"])
    uns = [r for r in rows if r["_ours"] != r["_ours"]]
    L += [f"**Worst 40 against the best library** (of {len(losers)} more than 1% behind; full list in "
          f"`{os.path.basename(args.losers)}`):", "",
          "| Shape (MxNxK) | out | group | best library | library TFLOPS | ours TFLOPS | ours | gap vs best | gap vs AITER |",
          "|---|---|---|---|---:|---:|---|---:|---:|"]
    for r in losers[:40]:
        L.append(f"| {r['M']}x{r['N']}x{r['K']} | {r['out']} | {r['scope']} | {r['_lib_name']} | {r['_lib']:.1f} | "
                 f"{r['_ours']:.1f} | {r['_what']} | {r['_gap_best'] * 100:+.1f}% | {r['_gap_aiter'] * 100:+.1f}% |")
    L += ["", f"**Unsupported** (no correct v27 column): {len(uns)}"
          + (": " + ", ".join(f"{r['M']}x{r['N']}x{r['K']}" for r in uns[:30]) if uns else ""), ""]
    with open(args.summary, "w") as fh:
        fh.write("\n".join(L))
    fields = ["M", "N", "K", "out", "scope", "band", "aiter_pick", "aiter_kernel", "aiter_pick_tflops", "hbl_tflops",
              "best_library", "library_tflops", "ours_tflops", "ours", "gap_best_pct", "gap_aiter_pct",
              "v20_tile", "v20_label"] + [f"pk_{s}_{c}" for s in SLOTS for c in ("kind", "tile", "tflops")]
    with open(args.losers, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in losers + uns:
            w.writerow({**r, "best_library": r["_lib_name"], "library_tflops": r["_lib"], "ours_tflops": r["_ours"],
                        "ours": r["_what"] or "unsupported",
                        "gap_best_pct": round(r["_gap_best"] * 100, 2) if r["_gap_best"] == r["_gap_best"] else "",
                        "gap_aiter_pct": round(r["_gap_aiter"] * 100, 2) if r["_gap_aiter"] == r["_gap_aiter"] else ""})
    print("\n".join(L[:60]))
    print(f"wrote {args.summary}, {args.losers} ({len(losers)} behind > 1%, {len(uns)} unsupported)")


if __name__ == "__main__":
    main()
