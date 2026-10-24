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
    """other None (a cold-only run): no column for the other cache state."""
    rs = sorted((r for r in rows.values() if r["_best_tile"] == "256x256" and r["_gap"] < -TIE), key=lambda r: r["_gap"])
    ties = sum(1 for r in rows.values() if r["_best_tile"] == "256x256" and -TIE <= r["_gap"] < 0)
    tile = " v27 tile |" if "v20_tile" in next(iter(rows.values())) else ""
    L = [f"{state}: {len(rs)} behind by more than 1%, {ties} behind by less (ties)", "",
         f"| # | Shape (MxNxK) | Output tiles | Full waves | S | k-steps | Best library | {NAME} config |{tile} Best TFLOPS | "
         f"v9 TFLOPS | {NAME} TFLOPS | {NAME} vs best | {NAME} vs v9 |" + (f" {other_state}: vs best |" if other is not None else ""),
         "|---:|---|---:|---:|---:|---:|---|---|" + ("---|" if tile else "") + "---:|---:|---:|---:|---:|"
         + ("---:|" if other is not None else "")]
    for i, r in enumerate(rs, 1):
        cfg = r["v20_label"] + (", handed to v9" if str(r.get("v20_v9_delegated")) == "True" else "")
        line = (f"| {i} | {r['M']}x{r['N']}x{r['K']} | {r['tiles']} | {r['waves']} | {r['S']} | {r['k_steps']} | "
                f"{r['_best_lib'].replace('AITER ', 'AITER ')} | {cfg} |" + (f" {r['v20_tile']} |" if tile else "")
                + f" {r['_best']:.0f} | {r['_v9']:.0f} | "
                f"{r['_v20']:.0f} | {r['_gap']:+.1f}% | {pct(r['_v20'], r['_v9']):+.1f}% |")
        if other is not None:
            o = other.get(key(r["M"], r["N"], r["K"], r["out"]))
            line += f" {o['_gap'] if o else float('nan'):+.1f}% |"
        L.append(line)
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


TILE_ORDER = ("256x256x64", "128x256x64", "128x128x128", "64x128x128", "64x64x128")


def tile_group(r):
    return "v27 keeps 256x256x64" if r["v20_tile"] == "256x256x64" else "v27 picks a smaller tile"


def v27_sections(states):
    """--kernel v27 runs: v27 auto against v26 timed in the same run, split into the tile's effect (v27 auto
    against v27 at 256x256x64, both on Triton 3.9) and the compiler's (v27 at 256x256x64 against v26 on
    Triton v2.2 + llirSched); the new AITER v9 against the existing one; every kernel against the best library."""
    L = []
    for _, R, _ in states:
        for r in R.values():
            r.update(_v26=fnum(r["v26_tflops"]), _t256=fnum(r["v27t256_tflops"]), _v9new=fnum(r["v9new_tflops"]))

    def groups(R, keyf, order):
        g = {}
        for r in R.values():
            for k in ("All shapes",) + tuple(keyf(r)):
                g.setdefault(k, []).append(r)
        return [(k, g[k]) for k in ("All shapes",) + tuple(order) if k in g] + sorted(
            (k, v) for k, v in g.items() if k not in order and k != "All shapes")

    def v27_keys(r):
        p = path(r)
        return (tile_group(r), f"tile {r['v20_tile']}", p if p != "Stream-K split" else "Stream-K split: " + kind(r),
                "under one wave" if int(r["tiles"]) < 256 else "one wave or more")

    order = ("v27 keeps 256x256x64", "v27 picks a smaller tile") + tuple(f"tile {t}" for t in TILE_ORDER) + (
        "Handed to v9", "Persistent data-parallel", "Stream-K split: reduce-scatter", "Stream-K split: owner",
        "Stream-K split: spread", "under one wave", "one wave or more")
    L += [f"## {NAME} (auto tile) against v26, timed in the same run (this / v26 - 1)", "",
          "Tiles: v27 auto against v27 at 256x256x64, both on Triton 3.9. Compiler: v27 at 256x256x64 (v26's code) "
          "against v26 on Triton v2.2 + llirSched. On each shape the two ratios multiply to v27 against v26; "
          "their medians need not.", "",
          f"| Group | Caches | Shapes | {NAME} vs v26: faster | within 1% | slower | median | tiles: median | "
          "compiler: median | v27 at 256 vs v26: faster | slower |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for st, R, _ in states:
        for k, rs in groups(R, v27_keys, order):
            f, w, s, m = counts([pct(r["_v20"], r["_v26"]) for r in rs])
            ft, _, st_, mt = counts([pct(r["_v20"], r["_t256"]) for r in rs])
            fc, _, sc, mc = counts([pct(r["_t256"], r["_v26"]) for r in rs])
            L.append(f"| {k} | {st} | {len(rs)} | {f} | {w} | {s} | {m:+.1f}% | {mt:+.1f}% | {mc:+.1f}% | {fc} | {sc} |")
    L += [""]

    L += ["## Every kernel against the best library", "",
          "| Caches | Kernel | faster | within 1% | slower | median | under one wave: median | same-tile behind |",
          "|---|---|---:|---:|---:|---:|---:|---:|"]
    for st, R, _ in states:
        rs = list(R.values())
        same = [r for r in rs if r["_best_tile"] == "256x256"]
        for name, col in ((f"{NAME} auto (capped)", "_v20"), (f"{NAME} auto, data-parallel (forced)", "_dp"),
                          (f"{NAME} at 256x256x64 (capped)", "_t256"), ("v26 (capped)", "_v26"),
                          ("AITER v9, existing", "_v9"), ("AITER v9, new", "_v9new")):
            f, w, s, m = counts([pct(r[col], r["_best"]) for r in rs])
            under = med([pct(r[col], r["_best"]) for r in rs if int(r["tiles"]) < 256])
            sb = sum(pct(r[col], r["_best"]) < -TIE for r in same)
            L.append(f"| {st} | {name} | {f} | {w} | {s} | {m:+.1f}% | {under:+.1f}% | {sb} of {len(same)} |")
    L += [""]

    L += [f"## {NAME} auto against the best library, by its tile", "",
          "| Group | Caches | Shapes | faster | within 1% | slower | median | v26 vs best: median |",
          "|---|---|---:|---:|---:|---:|---:|---:|"]
    for st, R, _ in states:
        for k, rs in groups(R, v27_keys, order):
            f, w, s, m = counts([r["_gap"] for r in rs])
            L.append(f"| {k} | {st} | {len(rs)} | {f} | {w} | {s} | {m:+.1f}% | "
                     f"{med([pct(r['_v26'], r['_best']) for r in rs]):+.1f}% |")
    L += [""]

    L += ["## The new AITER v9 against the existing one (this run)", "",
          "| Group | Caches | Shapes | new faster | within 1% | new slower | median | new vs best: median | "
          "existing vs best: median |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    v9_order = tuple(f"new v9 tile {t}" for t in ("256x256", "256x128", "128x256", "128x128", "128x64", "64x128", "64x64"))
    for st, R, _ in states:
        for k, rs in groups(R, lambda r: (f"new v9 tile {r['v9new_tile']}",
                                          "under one wave" if int(r["tiles"]) < 256 else "one wave or more"),
                            v9_order + ("under one wave", "one wave or more")):
            f, w, s, m = counts([pct(r["_v9new"], r["_v9"]) for r in rs])
            L.append(f"| {k} | {st} | {len(rs)} | {f} | {w} | {s} | {m:+.1f}% | "
                     f"{med([pct(r['_v9new'], r['_best']) for r in rs]):+.1f}% | "
                     f"{med([pct(r['_v9'], r['_best']) for r in rs]):+.1f}% |")
    L += [""]

    L += [f"## {NAME} auto against the new AITER v9", "",
          "| Group | Caches | Shapes | v27 faster | within 1% | v27 slower | median |", "|---|---|---:|---:|---:|---:|---:|"]
    for st, R, _ in states:
        for k, rs in groups(R, v27_keys, order):
            f, w, s, m = counts([pct(r["_v20"], r["_v9new"]) for r in rs])
            L.append(f"| {k} | {st} | {len(rs)} | {f} | {w} | {s} | {m:+.1f}% |")
    L += [""]

    L += ["## Checks of the extra columns", ""]
    for st, R, _ in states:
        rs = list(R.values())
        bad = [r for r in rs if any(str(r.get(c)) != "True" for c in ("v9new_ok", "v26_ok", "v27t256_ok", "v26_post_ok",
                                                                       "v27t256_post_ok"))
               or str(r["v26_flags_left"]) != "0"]
        last = max(rs, key=lambda r: int(r["new_builds"] or 0))
        L.append(f"- {st}: {len(bad)} shapes with a new v9 / v26 / v27-at-256 failure or v26 flag left; builds on "
                 f"Triton 3.9 {last['new_builds']}, without sched_barriers: new v9 {last['v9new_unscheduled']}, "
                 f"v27 and v9_tile {last['v27_unscheduled']}; notes on {sum(1 for r in rs if r['note'])}.")
    return L


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warm", default=None, help="omit (with --base-warm) for a cold-only run")
    p.add_argument("--cold", required=True)
    p.add_argument("--base-warm", default=None)
    p.add_argument("--base-cold", required=True)
    p.add_argument("--ctl-warm", default=None, help="a --v20-only run of the baseline kernel on this card")
    p.add_argument("--ctl-cold", default=None)
    p.add_argument("--name", default="v24", help="the run's kernel, for labels")
    p.add_argument("--ctl-name", default="v20", help="the control's kernel, for labels")
    a = p.parse_args()
    assert (a.warm is None) == (a.base_warm is None), "--warm and --base-warm go together"
    assert a.warm or not a.ctl_warm, "--ctl-warm needs --warm"
    global NAME, CTL
    NAME, CTL = a.name, a.ctl_name
    W, C = (enrich(read_rows(a.warm)) if a.warm else None), enrich(read_rows(a.cold))
    BW, BC = (enrich(read_rows(a.base_warm)) if a.base_warm else None), enrich(read_rows(a.base_cold))
    states = ([("warm", W, BW)] if W is not None else []) + [("cold", C, BC)]
    if a.ctl_warm or a.ctl_cold:
        print(f"## {NAME} against {CTL} on the same card (this / control - 1)\n")
        print(f"| Path | Caches | Shapes | {NAME} vs {CTL}: median | faster | within 1% | slower | "
              f"data-parallel (forced) vs {CTL}'s |")
        lines = []
        for st, R, ctl in (("warm", W, a.ctl_warm), ("cold", C, a.ctl_cold)):
            if ctl:
                lines += control(R, {key(r["M"], r["N"], r["K"], r["out"]): r for r in read_rows(ctl)}, st)
        print("\n".join(lines) + "\n")
    print("## Results table\n")
    print("| Run | Caches | faster | within 1% | slower | median | under one wave: median | same-tile behind | v9: median |")
    for st, R, B in states:
        print(headline(B, f"base | {st}"))
        print(headline(R, f"this | {st}"))
    print("\n## Three-way against the best library\n")
    print("| Caches | Kernel | faster | within 1% | slower | median | same-tile behind |")
    print("\n".join(sum((three_way(R, st) for st, R, _ in states), [])))
    print("\n## Per path against the baseline (this / base - 1, median)\n")
    print(f"| Path | Caches | Shapes | {NAME} vs base | AITER v9 (card) | {NAME} corrected by v9 | dp | hipBLASLt | AITER pick |")
    print("\n".join(sum((by_path(R, B, st) for st, R, B in states), [])))
    for st, R, B in states:
        ah, be = flips(R, B)
        v9f = counts([pct(r["_v20"], r["_v9"]) for r in R.values()])
        print(f"\n{st}: behind->ahead {ah}, ahead->behind {be}; {NAME} vs v9: {v9f[0]} faster, {v9f[2]} slower, median {v9f[3]:+.1f}%")
    print("\n## Same-tile behind\n")
    for i, (st, R, _) in enumerate(states):
        other = [(s, X) for s, X, _ in states if s != st]
        if i:
            print()
        print("\n".join(same_tile_behind(R, other[0][1] if other else None, st.capitalize(),
                                         other[0][0].capitalize() if other else None)))
    print("\n## Hand-off\n")
    print("| Caches | Shapes | Against the persistent loop | Against AITER's v9 |")
    print("\n".join(sum((handoff(R, st.capitalize()) for st, R, _ in states), [])))
    print("\n## Spread and correctness\n")
    for st, R, B in states:
        print(spread(R, st), "| base:", spread(B, st))
        print(correctness(R, st))
    if "v9new_tflops" in next(iter(C.values())):
        print()
        print("\n".join(v27_sections(states)))


if __name__ == "__main__":
    main()
