#!/usr/bin/env python3
"""summary_origami.md: v28 at origami's tile and split against AITER's pick and hipBLASLt (the origami run).

    python origami_report.py [--details results/details_origami_rot2048.csv results/details_origami_redo.csv] [--picks picks_origami.csv]
        [--shapes shapes_origami.csv] [--baseline DETAILS_ALLTUNED] [--out results/summary_origami.md]

Times are the median of the per-round *_ms medians. The gap against a library is its time over ours, minus 1
(positive: we are faster). Columns:
- ours (origami): the fastest of the origami slots v9, pdp, rs, ow, 2t, sp (fp32 partials) that passed the
  harness's strict check; a slot whose (kind, tile) repeats an earlier one (alias) takes that slot's time;
- ours + bf16: on rows whose AITER pick reduces K in bf16 (bf16_reduce), also rsb, owb, rsp, owp, counted when
  their mean_ratio is at most 1 + sqrt(n_max) (check_streamk.py's bound; the strict check fails them by design);
- cap: v28's capped split at origami's tile; refit: the REFIT rule's single pick; auto: v28 auto + capped (column
  v20); best of all: the fastest of everything above.
AITER's pick counts whether or not it passes the strict check, as in every earlier run; its mean_ratio is reported.
"""

import argparse
import csv
import math
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ORIGAMI = ("v9", "pdp", "rs", "ow", "2t", "sp")
BF16 = ("rsb", "owb", "rsp", "owp")
EXTRA = ("cap", "refit")
BASELINE = "/home/jbowley/repos/aiter/streamk_study/2026-10-20_all_tuned/results/details_alltuned_rot2048.csv"


def med_ms(r, name):
    xs = [float(x) for x in (r.get(f"{name}_ms") or "").split()]
    return statistics.median(xs) if xs else float("nan")


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def ok(r, name):
    return str(r.get(f"{name}_ok")) == "True"


def gap(lib, ours):
    return lib / ours - 1 if lib == lib and ours == ours and ours > 0 else float("nan")


def fmt_pct(x):
    return "" if x != x else f"{x * 100:+.1f}%"


def band_min(M, N):
    m = min(M, N)
    return "1-15" if m < 16 else "16-63" if m < 64 else "64-255" if m < 256 else "256-511" if m < 512 else ">= 512"


BANDS = ("1-15", "16-63", "64-255", "256-511", ">= 512")


def load(details, picks, shapes):
    pk = {}
    for p in csv.DictReader(open(picks)):
        pk.setdefault((int(p["M"]), int(p["N"]), int(p["K"])), {})[p["slot"]] = p
    sh = {(int(s["M"]), int(s["N"]), int(s["K"])): s for s in csv.DictReader(open(shapes))}
    merged = {}
    for path in details:
        for r in csv.DictReader(open(path)):
            merged[(int(r["M"]), int(r["N"]), int(r["K"]))] = r
    out = []
    for r in merged.values():
        key = (int(r["M"]), int(r["N"]), int(r["K"]))
        slots = pk.get(key, {})
        s = sh.get(key, {})
        t = {}
        for slot, p in slots.items():
            src = p["alias"] or slot
            name = f"pk_{src}"
            t[slot] = dict(ms=med_ms(r, name), ok=ok(r, name), mratio=fnum(r.get(f"{name}_mratio")),
                           kind=p["kind"], tile=p["tile"], n_max=int(p["n_max"]))
        x = dict(M=key[0], N=key[1], K=key[2], scope=r["scope"], band=band_min(*key[:2]), models=s.get("models", ""),
                 bf16_reduce=s.get("bf16_reduce", ""), aiter_pick=r["aiter_pick"], aiter_kernel=r["aiter_kernel"],
                 pick=med_ms(r, "pick"), pick_ok=ok(r, "pick"), pick_mratio=fnum(r.get("pick_mratio")),
                 hbl=med_ms(r, "hbl"), hbl_ok=ok(r, "hbl"), hbl_tile=r.get("hbl_tile", ""),
                 auto=med_ms(r, "v20") if ok(r, "v20") else float("nan"), slots=t, picks=slots,
                 o_tile=next(iter(slots.values()))["o_tile"] if slots else "",
                 o_reduction=next(iter(slots.values()))["o_reduction"] if slots else "",
                 o_split=next(iter(slots.values()))["o_split"] if slots else "")

        def best(names, bf16=False):
            c = [(v["ms"], n) for n, v in t.items() if n in names and v["ms"] == v["ms"]
                 and (v["ok"] if not bf16 or n not in BF16 else v["mratio"] <= 1 + math.sqrt(v["n_max"]))]
            return min(c) if c else (float("nan"), "")

        x["ours"], x["ours_slot"] = best(ORIGAMI)
        x["ours_b"], x["ours_b_slot"] = best(ORIGAMI + BF16, bf16=True) if x["bf16_reduce"] else (x["ours"], x["ours_slot"])
        x["cap"] = best(("cap",))[0]
        x["refit"] = best(("refit",))[0]
        everything = [v for v in (x["ours_b"], x["cap"], x["refit"], x["auto"]) if v == v]
        x["all"] = min(everything) if everything else float("nan")
        x["lib"] = min(v for v in (x["pick"], x["hbl"]) if v == v) if x["pick"] == x["pick"] or x["hbl"] == x["hbl"] \
            else float("nan")
        out.append(x)
    return out


def summarize(gaps):
    g = [v for v in gaps if v == v]
    if not g:
        return dict(n=0)
    return dict(n=len(g), ahead=sum(v > 0.01 for v in g), within=sum(abs(v) <= 0.01 for v in g),
                behind=sum(v < -0.01 for v in g), b4=sum(v < -0.04 for v in g), b10=sum(v < -0.10 for v in g),
                median=statistics.median(g))


HEAD = "| {} | rows | ahead > 1% | within 1% | behind > 1% | behind > 4% | behind > 10% | median |\n" \
       "|---|---:|---:|---:|---:|---:|---:|---:|"


def table(title, groups, rows, ours_key, lib_key, extra_unsupported=None):
    lines = [HEAD.format(title)]
    for g in groups:
        rs = [x for x in rows if g[1](x)]
        s = summarize([gap(x[lib_key], x[ours_key]) for x in rs])
        if not s["n"]:
            continue
        lines.append(f"| {g[0]} | {s['n']:,} | {s['ahead']:,} | {s['within']:,} | {s['behind']:,} | {s['b4']:,} | "
                     f"{s['b10']:,} | {fmt_pct(s['median'])} |")
    return "\n".join(lines)


def by_scope():
    return [(s, lambda x, s=s: x["scope"] == s) for s in ("M < 512", "N < 512 <= M", "both >= 512")] + \
           [("**all**", lambda x: True)]


def by_band():
    return [(b, lambda x, b=b: x["band"] == b) for b in BANDS] + [("**all**", lambda x: True)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--details", nargs="+", default=[os.path.join(HERE, "results", n) for n in
                    ("details_origami_rot2048.csv", "details_origami_redo.csv")],
                    help="details CSVs; a later file's row replaces an earlier one's for the same shape")
    ap.add_argument("--picks", default=os.path.join(HERE, "picks_origami.csv"))
    ap.add_argument("--shapes", default=os.path.join(HERE, "shapes_origami.csv"))
    ap.add_argument("--baseline", default=BASELINE, help="2026-10-20's all-tuned details (v27 short list)")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "summary_origami.md"))
    ap.add_argument("--losers", default=os.path.join(HERE, "results", "losers_origami.csv"))
    args = ap.parse_args()
    rows = [x for x in load(args.details, args.picks, args.shapes) if x["slots"]]
    no_tile = [x for x in load(args.details, args.picks, args.shapes) if not x["slots"]]
    L = [f"# v28 at origami's tile and split: {len(rows):,} shapes", ""]
    L.append(f"From {', '.join(f'`{os.path.basename(d)}`' for d in args.details)}; {len(no_tile)} timed shapes have no origami pick (no tile with "
             "two pairs of k-steps) and are left out of every table.")
    L.append("")

    L += ["## Against AITER's pick", "", "Ours: the fastest origami slot (fp32 partials).", "",
          table("group", by_scope(), rows, "ours", "pick"), "",
          table("min(M, N)", by_band(), rows, "ours", "pick"), ""]
    L += ["## Against hipBLASLt (torch.mm)", "", table("group", by_scope(), rows, "ours", "hbl"), "",
          table("min(M, N)", by_band(), rows, "ours", "hbl"), ""]
    L += ["## Against the best library (AITER's pick or hipBLASLt)", "",
          table("group", by_scope(), rows, "ours", "lib"), ""]

    libs = sorted({x["aiter_pick"] for x in rows}, key=lambda l: -sum(x["aiter_pick"] == l for x in rows))
    L += ["## Against AITER's pick, by AITER's library", "",
          table("AITER pick", [(l, lambda x, l=l: x["aiter_pick"] == l) for l in libs] + [("**all**", lambda x: True)],
                rows, "ours", "pick"), ""]
    hb = [x for x in rows if x["aiter_pick"] in ("torch", "hipblaslt")]
    L += [f"## Where AITER's pick is hipBLASLt ({len(hb)} shapes)", "",
          "AITER's pick there is hipBLASLt itself (torch libtype, or hipblaslt with a tuned solution index).", "",
          table("group", by_scope(), hb, "ours", "pick"), ""]

    bf = [x for x in rows if x["bf16_reduce"]]
    L += [f"## AITER picks that reduce K in bf16 ({len(bf)} shapes)", "",
          "fp32: ours as above; bf16: with rs / ow on bf16 (bf16-pair) partials, where their mean_ratio is within "
          "1 + sqrt(n). mean_ratio is the largest 32-row region's mean error over one bf16 rounding's (1.0 for an "
          "fp32 reduction).", ""]
    classes = sorted({x["bf16_reduce"] for x in bf})
    grp = [(c, lambda x, c=c: x["bf16_reduce"] == c) for c in classes] + [("**all**", lambda x: True)]
    L += ["fp32 partials:", "", table("class", grp, bf, "ours", "pick"), "", "with bf16 partials:", "",
          table("class", grp, bf, "ours_b", "pick"), ""]
    L += ["| class | AITER pick mean_ratio (median, max) | ours bf16 slots mean_ratio (median, max) | "
          "bf16 slot fastest |", "|---|---:|---:|---:|"]
    for c in classes:
        xs = [x for x in bf if x["bf16_reduce"] == c]
        pm = [x["pick_mratio"] for x in xs if x["pick_mratio"] == x["pick_mratio"]]
        om = [v["mratio"] for x in xs for n, v in x["slots"].items() if n in BF16 and v["mratio"] == v["mratio"]]
        won = sum(x["ours_b_slot"] in BF16 for x in xs)
        L.append(f"| {c} | {statistics.median(pm):.2f}, {max(pm):.2f} | "
                 + (f"{statistics.median(om):.2f}, {max(om):.2f}" if om else "-") + f" | {won} of {len(xs)} |")
    L.append("")

    # Endings.
    L += ["## Which ending wins at origami's tile", "",
          "| slot | shapes timed | fastest on | median loss to the best slot | loss > 2% on |",
          "|---|---:|---:|---:|---:|"]
    for n in ORIGAMI:
        xs = [x for x in rows if n in x["slots"] and x["slots"][n]["ok"] and x["ours"] == x["ours"]]
        if not xs:
            continue
        loss = [x["slots"][n]["ms"] / x["ours"] - 1 for x in xs]
        L.append(f"| {n} | {len(xs):,} | {sum(x['ours_slot'] == n for x in rows):,} | "
                 f"{fmt_pct(statistics.median(loss))} | {sum(v > 0.02 for v in loss):,} |")
    both = [x for x in rows if all(n in x["slots"] and x["slots"][n]["ok"] for n in ("rs", "ow"))]
    if both:
        hint = [x["slots"]["rs" if x["o_reduction"] == "parallel" else "ow"]["ms"] for x in both]
        best2 = [min(x["slots"]["rs"]["ms"], x["slots"]["ow"]["ms"]) for x in both]
        agree = sum(h <= b * 1.0001 for h, b in zip(hint, best2))
        loss = [h / b - 1 for h, b in zip(hint, best2)]
        L += ["", f"Origami's reduction hint as the ending (parallel: reduce-scatter, tree: owner), on the {len(both):,} "
              f"shapes with both: the faster of the two on {agree:,}; following it costs a median "
              f"{fmt_pct(statistics.median(loss))} against the faster, more than 2% on {sum(v > 0.02 for v in loss):,}."]
    L.append("")

    # Selectors.
    L += ["## Selectors against AITER's pick", "",
          "| selector | rows | ahead > 1% | behind > 4% | behind > 10% | median |", "|---|---:|---:|---:|---:|---:|"]
    for name, k in (("origami (fp32)", "ours"), ("origami (+ bf16 on bf16-reducing picks)", "ours_b"),
                    ("cap at origami's tile", "cap"), ("REFIT rule's single pick", "refit"),
                    ("v28 auto + capped", "auto"), ("best of all of these", "all")):
        s = summarize([gap(x["pick"], x[k]) for x in rows])
        L.append(f"| {name} | {s['n']:,} | {s['ahead']:,} | {s['b4']:,} | {s['b10']:,} | {fmt_pct(s['median'])} |")
    L.append("")

    if args.baseline and os.path.exists(args.baseline):
        base = {}
        for r in csv.DictReader(open(args.baseline)):
            ts = [med_ms(r, n) for n in ("v20",) + tuple(f"pk_{s}" for s in ("best", "tile2", "tile3", "x5"))
                  if str(r.get(f"{n}_ok")) == "True"]
            ts = [t for t in ts if t == t]
            if ts:
                base[(int(r["M"]), int(r["N"]), int(r["K"]))] = (min(ts), med_ms(r, "pick"))
        common = [x for x in rows if (x["M"], x["N"], x["K"]) in base]
        if common:
            g_new = [gap(x["pick"], x["ours"]) for x in common]
            g_old = [gap(base[(x["M"], x["N"], x["K"])][1], base[(x["M"], x["N"], x["K"])][0]) for x in common]
            drift = [x["pick"] / base[(x["M"], x["N"], x["K"])][1] - 1 for x in common]
            s_new, s_old = summarize(g_new), summarize(g_old)
            L += ["## Against 2026-10-20 (v27, REFIT short list of up to 4 picks, or auto)", "",
                  f"On the {len(common):,} shapes in both runs, each against its own run's AITER pick (AITER's pick "
                  f"moved a median {fmt_pct(statistics.median(drift))} between the runs):", "",
                  "| | median | behind > 4% | behind > 10% |", "|---|---:|---:|---:|",
                  f"| 2026-10-20 best of short list | {fmt_pct(s_old['median'])} | {s_old['b4']:,} | {s_old['b10']:,} |",
                  f"| origami (fp32) | {fmt_pct(s_new['median'])} | {s_new['b4']:,} | {s_new['b10']:,} |", ""]

    # Origami's choices.
    L += ["## Origami's choices", "",
          "| tile | shapes | median gap against AITER's pick |", "|---|---:|---:|"]
    tiles = sorted({x["o_tile"] for x in rows}, key=lambda t: -sum(x["o_tile"] == t for x in rows))
    for t in tiles[:15]:
        xs = [x for x in rows if x["o_tile"] == t]
        g = [gap(x["pick"], x["ours"]) for x in xs if x["ours"] == x["ours"]]
        L.append(f"| {t} | {len(xs):,} | {fmt_pct(statistics.median(g)) if g else ''} |")
    L.append("")

    worst = sorted((x for x in rows if x["ours"] == x["ours"] and x["lib"] == x["lib"]),
                   key=lambda x: gap(x["lib"], x["ours"]))
    L += ["## The 40 worst shapes against the best library", "",
          "| M x N x K | group | models | AITER pick | ours (slot, tile) | gap vs pick | vs hipBLASLt | best of all vs lib |",
          "|---|---|---|---|---|---:|---:|---:|"]
    for x in worst[:40]:
        s = x["slots"][x["ours_slot"]]
        L.append(f"| {x['M']}x{x['N']}x{x['K']} | {x['scope']} | {x['models']} | {x['aiter_pick']} | "
                 f"{s['kind']} {s['tile']} | {fmt_pct(gap(x['pick'], x['ours']))} | {fmt_pct(gap(x['hbl'], x['ours']))} | "
                 f"{fmt_pct(gap(x['lib'], x['all']))} |")
    with open(args.out, "w") as f:
        f.write("\n".join(L) + "\n")
    with open(args.losers, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["M", "N", "K", "scope", "models", "bf16_reduce", "aiter_pick", "aiter_kernel", "pick_us", "hbl_us",
                    "hbl_tile", "ours_us", "ours_slot", "ours_kind", "ours_tile", "o_reduction", "o_split", "gap_pick",
                    "gap_hbl", "gap_lib", "ours_b_us", "cap_us", "refit_us", "auto_us", "all_us"])
        for x in worst:
            if gap(x["lib"], x["ours"]) >= -0.01:
                break
            s = x["slots"][x["ours_slot"]]
            w.writerow([x["M"], x["N"], x["K"], x["scope"], x["models"], x["bf16_reduce"], x["aiter_pick"],
                        x["aiter_kernel"], f"{x['pick'] * 1e3:.2f}", f"{x['hbl'] * 1e3:.2f}", x["hbl_tile"],
                        f"{x['ours'] * 1e3:.2f}", x["ours_slot"], s["kind"], s["tile"], x["o_reduction"], x["o_split"],
                        f"{gap(x['pick'], x['ours']):.4f}", f"{gap(x['hbl'], x['ours']):.4f}",
                        f"{gap(x['lib'], x['ours']):.4f}", f"{x['ours_b'] * 1e3:.2f}", f"{x['cap'] * 1e3:.2f}",
                        f"{x['refit'] * 1e3:.2f}", f"{x['auto'] * 1e3:.2f}", f"{x['all'] * 1e3:.2f}"])
    print(f"{len(rows)} shapes -> {args.out}, {args.losers}")


if __name__ == "__main__":
    main()
