#!/usr/bin/env python3
"""The numbers of a run's README: streamk_wip, v9 and data-parallel against the best library, and
the per-shape change against an earlier run, corrected for the card by AITER's v9.

    python needle.py --warm results/details_v24.csv --cold results/details_v24_rot2048.csv \
        --base-warm results/details_aisched.csv --base-cold results/details_aisched_rot2048.csv

All times are contiguous. "Best library" is the faster of AITER's pick and hipBLASLt. Gaps are
percent, positive meaning the first kernel is faster; "within 1%" is a tie.
"""

import argparse
import statistics

from pad_report import best_fields, times
from run_v20_vs_aiter import fnum, key, pct, read_rows

TIE = 1.0
# --name / --ctl-name: the labels of the run's kernel and of the control's.
NAME, CTL = "v24", "v20"


def med(xs):
    xs = [x for x in xs if x == x]
    return statistics.median(xs) if xs else float("nan")


def counts(gaps):
    gaps = [g for g in gaps if g == g]
    return sum(g > TIE for g in gaps), sum(abs(g) <= TIE for g in gaps), sum(g < -TIE for g in gaps), med(gaps)


def path(r):
    if str(r.get("v20_v9_delegated")) == "True":
        return "Handed to v9"
    return "Persistent data-parallel" if r["v20_policy"] == "dp" else "Stream-K split"


def enrich(rows):
    for r in rows:
        b = best_fields(r, "")
        t = times(r, "")
        r.update(_best=b["best"], _best_lib=b["best_lib"], _best_tile=b["best_tile"], _v9=t["v9"], _v20=t["v20"],
                 _dp=t["dp"], _hbl=fnum(r["hbl_tflops"]), _pick=fnum(r["aiter_pick_tflops"]))
        r["_gap"] = pct(r["_v20"], r["_best"])
    return {key(r["M"], r["N"], r["K"], r["out"]): r for r in rows}


def headline(rows, label):
    rs = list(rows.values())
    f, w, s, m = counts([r["_gap"] for r in rs])
    under = med([r["_gap"] for r in rs if int(r["tiles"]) < 256])
    same = [r for r in rs if r["_best_tile"] == "256x256"]
    behind = sum(r["_gap"] < -TIE for r in same)
    v9 = med([pct(r["_v9"], r["_best"]) for r in rs])
    return f"| {label} | {f} | {w} | {s} | {m:+.1f}% | {under:+.1f}% | {behind} of {len(same)} | {v9:+.1f}% |"


def three_way(rows, state):
    rs = list(rows.values())
    out = []
    for name, col in ((f"{NAME} (capped)", "_v20"), ("AITER v9", "_v9"), (f"{NAME} data-parallel (forced)", "_dp")):
        f, w, s, m = counts([pct(r[col], r["_best"]) for r in rs])
        same = [r for r in rs if r["_best_tile"] == "256x256"]
        sb = sum(pct(r[col], r["_best"]) < -TIE for r in same)
        out.append(f"| {state} | {name} | {f} | {w} | {s} | {m:+.1f}% | {sb} of {len(same)} |")
    return out


def by_path(rows, base, state):
    out = []
    groups = {}
    for k, r in rows.items():
        if k in base:
            groups.setdefault(path(r), []).append((r, base[k]))
            groups.setdefault("All shapes", []).append((r, base[k]))
    for p in ("All shapes", "Handed to v9", "Stream-K split", "Persistent data-parallel"):
        ps = groups.get(p, [])
        v20 = [pct(r["_v20"], o["_v20"]) for r, o in ps]
        v9 = [pct(r["_v9"], o["_v9"]) for r, o in ps]
        corr = [((r["_v20"] / o["_v20"]) / (r["_v9"] / o["_v9"]) - 1) * 100 for r, o in ps]
        dp = [pct(r["_dp"], o["_dp"]) for r, o in ps]
        hbl = [pct(r["_hbl"], o["_hbl"]) for r, o in ps]
        pick = [pct(r["_pick"], o["_pick"]) for r, o in ps]
        f, w, s, m = counts(corr)
        out.append(f"| {p} | {state} | {len(ps)} | {med(v20):+.1f}% | {med(v9):+.1f}% | {m:+.1f}% ({f} up, {s} down) | "
                   f"{med(dp):+.1f}% | {med(hbl):+.1f}% | {med(pick):+.1f}% |")
    return out


def kind(r):
    """The configuration's kind, without its split count."""
    if str(r.get("v20_v9_delegated")) == "True":
        return "handed to v9"
    lab = r["v20_label"]
    return "reduce-scatter" if "reduce-scatter" in lab else "owner" if "owner" in lab else lab


def control(rows, ctl, state):
    """The run's kernel against the control's, timed on the same card (a --v20-only run): per path,
    and by how the configuration changed between the two (the host rules may differ)."""
    out = []
    groups = {}
    for k, r in rows.items():
        c = ctl.get(k)
        if not c or fnum(c["v20_tflops"]) != fnum(c["v20_tflops"]):
            continue
        g = pct(r["_v20"], fnum(c["v20_tflops"]))
        gdp = pct(r["_dp"], fnum(c["v20_dp_tflops"]) if c["v20_dp_tflops"] not in ("", "nan") and
                  fnum(c["v20_dp_tflops"]) == fnum(c["v20_dp_tflops"]) else fnum(c["v20_tflops"]))
        p = path(r)
        keys = ["All shapes", p]
        if p == "Stream-K split":
            keys.append("Split: " + kind(r))
        if r["v20_label"] == c["v20_label"] and kind(r) == kind(c):
            keys.append("Same configuration: " + kind(r))
        elif kind(r) == kind(c):
            keys.append(f"Changed: {kind(c)}, another split count")
        else:
            keys.append(f"Changed: {kind(c)} -> {kind(r)}")
        for kk in keys:
            groups.setdefault(kk, []).append((g, gdp))
    order = ["All shapes", "Handed to v9", "Stream-K split", "Split: reduce-scatter", "Split: owner", "Split: spread",
             "Persistent data-parallel"]
    order += sorted(k for k in groups if k.startswith("Same")) + sorted(k for k in groups if k.startswith("Changed"))
    for p in order:
        gs = groups.get(p, [])
        if not gs:
            continue
        f, w, s, m = counts([g for g, _ in gs])
        fd, wd, sd, md = counts([d for _, d in gs])
        out.append(f"| {p} | {state} | {len(gs)} | {m:+.1f}% | {f} | {w} | {s} | {md:+.1f}% |")
    return out


def flips(rows, base):
    ahead = sum(1 for k, r in rows.items() if k in base and r["_gap"] > TIE and base[k]["_gap"] < -TIE)
    behind = sum(1 for k, r in rows.items() if k in base and r["_gap"] < -TIE and base[k]["_gap"] > TIE)
    return ahead, behind


def same_tile_behind(rows, other, state, other_state):
    rs = sorted((r for r in rows.values() if r["_best_tile"] == "256x256" and r["_gap"] < -TIE), key=lambda r: r["_gap"])
    ties = sum(1 for r in rows.values() if r["_best_tile"] == "256x256" and -TIE <= r["_gap"] < 0)
    L = [f"{state}: {len(rs)} behind by more than 1%, {ties} behind by less (ties)", "",
         f"| # | Shape (MxNxK) | Output tiles | Full waves | S | k-steps | Best library | {NAME} config | Best TFLOPS | "
         f"v9 TFLOPS | {NAME} TFLOPS | {NAME} vs best | {NAME} vs v9 | {other_state}: vs best |",
         "|---:|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for i, r in enumerate(rs, 1):
        o = other.get(key(r["M"], r["N"], r["K"], r["out"]))
        cfg = r["v20_label"] + (", handed to v9" if str(r.get("v20_v9_delegated")) == "True" else "")
        L.append(f"| {i} | {r['M']}x{r['N']}x{r['K']} | {r['tiles']} | {r['waves']} | {r['S']} | {r['k_steps']} | "
                 f"{r['_best_lib'].replace('AITER ', 'AITER ')} | {cfg} | {r['_best']:.0f} | {r['_v9']:.0f} | "
                 f"{r['_v20']:.0f} | {r['_gap']:+.1f}% | {pct(r['_v20'], r['_v9']):+.1f}% | "
                 f"{o['_gap'] if o else float('nan'):+.1f}% |")
    return L


def handoff(rows, state):
    rs = [r for r in rows.values() if str(r.get("v20_v9_delegated")) == "True"]
    out = []
    for title, sel in (("all", rs), ("at 32 k-steps or fewer", [r for r in rs if int(r["k_steps"]) <= 32]),
                       ("above 32 k-steps", [r for r in rs if int(r["k_steps"]) > 32])):
        f, w, s, m = counts([pct(r["_v20"], r["_dp"]) for r in sel])
        f9, w9, s9, m9 = counts([pct(r["_v20"], r["_v9"]) for r in sel])
        out.append(f"| {state} | {len(sel)} {title} | {m:+.1f}% ({f} faster, {s} slower) | "
                   f"{m9:+.1f}% ({w9} within 1%, {f9} faster, {s9} slower) |")
    return out


def spread(rows, state):
    sp = [fnum(r["v20_spread"]) for r in rows.values()]
    sp9 = [fnum(r["v9_spread"]) for r in rows.values()]
    return (f"{state}: {NAME} median spread {med(sp):.2f}%, {sum(x > 2 for x in sp if x == x)} shapes over 2%; "
            f"v9 {med(sp9):.2f}%, {sum(x > 2 for x in sp9 if x == x)} over 2%")


def correctness(rows, state):
    bad = [r for r in rows.values() if any(str(r.get(c)) not in ("True",) for c in ("v9_ok", "v20_ok", "hbl_ok", "v20_post_ok"))
           or str(r.get("v20_dp_ok")) == "False" or str(r["v20_flags_left"]) != "0"]
    pick_bad = [r for r in rows.values() if str(r["pick_ok"]) != "True"]
    notes = [r for r in rows.values() if r["note"]]
    return f"{state}: {len(bad)} shapes with a {NAME}/v9/hipBLASLt/dp failure or flag left; AITER pick outside tolerance on {len(pick_bad)}; notes on {len(notes)}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warm", required=True)
    p.add_argument("--cold", required=True)
    p.add_argument("--base-warm", required=True)
    p.add_argument("--base-cold", required=True)
    p.add_argument("--ctl-warm", default=None, help="a --v20-only run of the baseline kernel on this card")
    p.add_argument("--ctl-cold", default=None)
    p.add_argument("--name", default="v24", help="the run's kernel, for labels")
    p.add_argument("--ctl-name", default="v20", help="the control's kernel, for labels")
    a = p.parse_args()
    global NAME, CTL
    NAME, CTL = a.name, a.ctl_name
    W, C = enrich(read_rows(a.warm)), enrich(read_rows(a.cold))
    BW, BC = enrich(read_rows(a.base_warm)), enrich(read_rows(a.base_cold))
    if a.ctl_warm:
        print(f"## {NAME} against {CTL} on the same card (this / control - 1)\n")
        print(f"| Path | Caches | Shapes | {NAME} vs {CTL}: median | faster | within 1% | slower | "
              f"data-parallel (forced) vs {CTL}'s |")
        lines = control(W, {key(r["M"], r["N"], r["K"], r["out"]): r for r in read_rows(a.ctl_warm)}, "warm")
        if a.ctl_cold:
            lines += control(C, {key(r["M"], r["N"], r["K"], r["out"]): r for r in read_rows(a.ctl_cold)}, "cold")
        print("\n".join(lines) + "\n")
    print("## Results table\n")
    print("| Run | Caches | faster | within 1% | slower | median | under one wave: median | same-tile behind | v9: median |")
    print(headline(BW, "base | warm"))
    print(headline(W, "this | warm"))
    print(headline(BC, "base | cold"))
    print(headline(C, "this | cold"))
    print("\n## Three-way against the best library\n")
    print("| Caches | Kernel | faster | within 1% | slower | median | same-tile behind |")
    print("\n".join(three_way(W, "warm") + three_way(C, "cold")))
    print("\n## Per path against the baseline (this / base - 1, median)\n")
    print(f"| Path | Caches | Shapes | {NAME} vs base | AITER v9 (card) | {NAME} corrected by v9 | dp | hipBLASLt | AITER pick |")
    print("\n".join(by_path(W, BW, "warm") + by_path(C, BC, "cold")))
    for st, R, B in (("warm", W, BW), ("cold", C, BC)):
        ah, be = flips(R, B)
        v9f = counts([pct(r["_v20"], r["_v9"]) for r in R.values()])
        print(f"\n{st}: behind->ahead {ah}, ahead->behind {be}; {NAME} vs v9: {v9f[0]} faster, {v9f[2]} slower, median {v9f[3]:+.1f}%")
    print("\n## Same-tile behind\n")
    print("\n".join(same_tile_behind(W, C, "Warm", "Cold")))
    print()
    print("\n".join(same_tile_behind(C, W, "Cold", "Warm")))
    print("\n## Hand-off\n")
    print("| Caches | Shapes | Against the persistent loop | Against AITER's v9 |")
    print("\n".join(handoff(W, "Warm") + handoff(C, "Cold")))
    print("\n## Spread and correctness\n")
    for st, R, B in (("warm", W, BW), ("cold", C, BC)):
        print(spread(R, st), "| base:", spread(B, st))
        print(correctness(R, st))


if __name__ == "__main__":
    main()
