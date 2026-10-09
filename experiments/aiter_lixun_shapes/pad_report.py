"""Report for a `sweep --pad512` run: every kernel on the contiguous layout and on the padded one
(each row stride that is a multiple of 512 bytes gets 256 more), with --realloc rounds.

Called from `run_v20_vs_aiter.py report` when the details CSV has the *_pad columns. Answers:
- the headline: streamk_wip (padded) against the best library, where the best library is the
  fastest valid time of AITER's pick and hipBLASLt over both layouts (a padded call that copies its
  inputs, FlyDSL's, does not count);
- the v9 delegation (v20 WORKLOG entry 10): delegated v20 against the persistent loop it replaced
  (STREAMK_POLICY=dp, timed in the same sweep) and against AITER's v9;
- the padding: spread between fresh-buffer rounds per kernel and layout, and the padded/contiguous
  ratio, with the shapes whose strides were not padded (K % 256 == 128) as a control.
"""

import os
import statistics

from run_v20_vs_aiter import GRID_ORDER, HERE, aiter_tile, fnum, grid_bucket, key, pct, quantile, read_rows

TIE = 1.0


def best_fields(r, layout):
    """Best library on the shape. layout "": contiguous only (as on Oct 5). layout "_pad": the fastest
    valid time over both layouts; a padded call that launched a copy kernel is not valid."""
    cands = [("AITER " + r["aiter_pick"], fnum(r["aiter_pick_tflops"]), aiter_tile(r), ""),
             ("hipBLASLt", fnum(r["hbl_tflops"]), "x".join(r["hbl_tile"].split("x")[:2]), "")]
    if layout == "_pad":
        if str(r["pick_pad_copy"]) != "True":
            cands.append(("AITER " + r["aiter_pick"], fnum(r["pick_pad_tflops"]), aiter_tile(r), "_pad"))
        if str(r["hbl_pad_copy"]) != "True":
            cands.append(("hipBLASLt", fnum(r["hbl_pad_tflops"]), "x".join(r["hbl_tile_pad"].split("x")[:2]), "_pad"))
    cands = [c for c in cands if c[1] == c[1]]
    lib, t, tile, lay = max(cands, key=lambda c: c[1])
    return dict(best=t, best_lib=lib, best_tile=tile, best_layout="padded" if lay else "contiguous")


def times(r, layout):
    """v9, streamk_wip forced data-parallel (the persistent loop) and streamk_wip on one layout."""
    v20 = fnum(r[f"v20{layout}_tflops"])
    dp = fnum(r[f"v20_dp{layout}_tflops"])
    return dict(v9=fnum(r[f"v9{layout}_tflops"]), dp=dp if dp == dp else v20, v20=v20)


def counts(gaps):
    gaps = [g for g in gaps if g == g]
    if not gaps:
        return "0 | 0 | 0 | -"
    return (f"{sum(g > TIE for g in gaps)} | {sum(abs(g) <= TIE for g in gaps)} | {sum(g < -TIE for g in gaps)} | "
            f"{statistics.median(gaps):+.1f}%")


def dist(xs):
    xs = [x for x in xs if x == x]
    if not xs:
        return "- | - | -"
    return f"{statistics.median(xs):+.1f}% | {quantile(xs, 0.1):+.1f}% | {quantile(xs, 0.9):+.1f}%"


def padded_strides(r):
    return int(r["K"]) % 256 == 0


SUMMARY_GROUPS = (
    ("All shapes", lambda r: True),
    ("Under 256 tiles (less than one wave)", lambda r: int(r["tiles"]) < 256),
    ("256 to 1023 tiles", lambda r: 256 <= int(r["tiles"]) < 1024),
    ("1024 or more tiles", lambda r: int(r["tiles"]) >= 1024),
    ("Best library uses 256x256", lambda r: r["_same"]),
    ("Best library uses another tile", lambda r: not r["_same"]),
)
KERNELS = (("v9", "v9"), ("data-parallel only", "dp"), ("streamk_wip", "v20"))
# name, contiguous column, padded column, copy flag
PAD_KERNELS = (("v9", "v9_tflops", "v9_pad_tflops", "v9_pad_copy"),
               ("streamk_wip", "v20_tflops", "v20_pad_tflops", None),
               ("AITER's pick", "aiter_pick_tflops", "pick_pad_tflops", "pick_pad_copy"),
               ("hipBLASLt", "hbl_tflops", "hbl_pad_tflops", "hbl_pad_copy"))
SPREAD_KERNELS = (("v9", "v9"), ("streamk_wip", "v20"), ("AITER's pick", "pick"), ("hipBLASLt", "hbl"))


def summary_table(rows, layout, best_layout=None):
    """v9 and streamk_wip on `layout` against the best library on `best_layout` (default the same)."""
    for r in rows:
        b = best_fields(r, layout if best_layout is None else best_layout)
        r["_same"] = b["best_tile"] == "256x256"
        r["_best"] = b["best"]
        r["_t"] = times(r, layout)
    head = " | ".join(f"{n}: faster | {n}: within 1% | {n}: slower | {n}: median" for n, _ in KERNELS)
    L = [f"| Shapes | Count | {head} |", "|---|---:|" + "---:|---:|---:|---:|" * len(KERNELS)]
    for name, pred in SUMMARY_GROUPS:
        sel = [r for r in rows if pred(r)]
        cells = " | ".join(counts([pct(r["_t"][c], r["_best"]) for r in sel]) for _, c in KERNELS)
        L.append(f"| {name} | {len(sel)} | {cells} |")
    return L


def oct5_losses(base_rows):
    """Oct 5's same-tile shapes where streamk_wip was more than 1% behind the best library."""
    out = []
    for o in base_rows:
        pick, hbl = fnum(o["aiter_pick_tflops"]), fnum(o["hbl_tflops"])
        tile = "x".join(o["hbl_tile"].split("x")[:2]) if hbl > pick else aiter_tile(o)
        g = pct(fnum(o["v20_tflops"]), max(pick, hbl))
        if tile == "256x256" and g < -TIE:
            out.append((g, o))
    return [o for _, o in sorted(out, key=lambda p: p[0])]


def report_pad(args, rows, info):
    a, run = info.get("aiter", {}), info.get("run", {})
    rot = run.get("rotating_mb", 0)
    n_rounds = run.get("rounds", 3)
    L = [f"# streamk_wip (v20, capped, v9 for one-wave dp) vs AITER, v9 and hipBLASLt, contiguous and "
         f"512-byte-padded strides{f' ({rot} MB rotating buffer)' if rot else ' (warm)'}", ""]
    L += [
        f"- Stack: torch {a.get('torch', '?')}, hipBLASLt {a.get('hipblaslt', '?')}, Triton {a.get('triton', '?')}; "
        f"GPU: PCI bus {a.get('gpu_pci_bus', '?')}, {a.get('gpu', '?')}; AITER {a.get('aiter_commit', '?')} on "
        f"{a.get('aiter_base', '?')}.",
        f"- streamk_wip: snapshot sha256 {run.get('v20_snapshot_sha256', '?')} (newest WORKLOG entry: "
        f"\"{run.get('v20_worklog_latest', '?').lstrip('# ')}\"); the v9 it delegates to: live sha256 "
        f"{run.get('v9_live_sha256_at_start', '?')}.",
        f"- Layouts: contiguous, and padded: every row stride (x, w, C) that is a multiple of 512 bytes gets "
        f"{run.get('pad_bytes', 256)} more (x and w: K + 128 when K % 256 == 0; C: N + 128 for bf16, N + 64 for "
        "fp32). v9, streamk_wip and hipBLASLt (`torch.mm(x, w.t(), out=y)`) get padded x, w and C; AITER's pick "
        "allocates its own C, so it gets padded x and w only.",
        f"- Timing: {n_rounds} rounds, each with fresh inputs (new buffers) when `realloc` is on "
        f"(here: {run.get('realloc', False)}); median of the rounds. Spread = (slowest - fastest round) / median.",
        "- Best library: the fastest valid time of AITER's pick and hipBLASLt; for the padded comparison over both "
        "layouts. A padded call that launches a copy kernel (FlyDSL makes its inputs contiguous) is not valid.",
        "- \"Faster\" and \"slower\" mean a gap of more than 1%.",
        "",
    ]

    copies = {}
    for r in rows:
        for n in ("pick", "v9", "hbl"):
            if str(r.get(f"{n}_pad_copy")) == "True":
                lib = r["aiter_pick"] if n == "pick" else n
                copies[(n, lib)] = copies.get((n, lib), 0) + 1
    L += ["Padded calls that copied their inputs (left out of the best library): "
          + (", ".join(f"{n} ({lib}) on {c} shapes" for (n, lib), c in sorted(copies.items())) or "none") + ".", ""]
    best_lay = {}
    for r in rows:
        b = best_fields(r, "_pad")
        best_lay[b["best_layout"]] = best_lay.get(b["best_layout"], 0) + 1
    L += ["The best library's time comes from the " + ", ".join(f"{k} layout on {v} shapes"
                                                                for k, v in sorted(best_lay.items())) + ".", ""]

    L += ["## v9 and streamk_wip padded, libraries contiguous (as AITER calls them)", ""]
    L += summary_table(rows, "_pad", "") + [""]
    L += ["## Everything contiguous (comparable with Oct 5)", ""]
    L += summary_table(rows, "") + [""]
    L += ["## v9 and streamk_wip padded, against the best library over both layouts (strictest)", ""]
    L += summary_table(rows, "_pad") + [""]
    L += ["\"Data-parallel only\" is `STREAMK_POLICY=dp`, the persistent loop without stream-K and without the v9 "
          "hand-off; where capped already runs data-parallel on the persistent loop it is streamk_wip's time.", ""]

    # ---- v9 delegation
    deleg = [r for r in rows if str(r.get("v20_v9_delegated")) == "True"]
    L += [f"## v9 delegation ({len(deleg)} shapes: capped picks data-parallel, at most 256 tiles, bf16 output)", "",
          "streamk_wip (now v9) against the persistent loop it replaced and against AITER's v9, same sweep. Positive "
          "means streamk_wip is faster.", "",
          "| Layout | Shapes | Against | faster | within 1% | slower | median | p10 | p90 |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    kgroups = (("all", lambda r: True), ("k-steps <= 32", lambda r: int(r["k_steps"]) <= 32),
               ("k-steps > 32", lambda r: int(r["k_steps"]) > 32))
    for lay, lname in (("", "contiguous"), ("_pad", "padded")):
        for gname, pred in kgroups:
            sel = [r for r in deleg if pred(r)]
            for against, col in (("persistent loop (dp)", f"v20_dp{lay}_tflops"), ("AITER's v9", f"v9{lay}_tflops")):
                gaps = [pct(fnum(r[f"v20{lay}_tflops"]), fnum(r[col])) for r in sel]
                L.append(f"| {lname} | {gname} ({len(sel)}) | {against} | {counts(gaps)} | "
                         f"{' | '.join(dist(gaps).split(' | ')[1:])} |")
    L += [""]

    base_rows = read_rows(args.baseline) if args.baseline and os.path.exists(args.baseline) else []
    if base_rows:
        base = {key(o["M"], o["N"], o["K"], o["out"]): o for o in base_rows}
        L += [f"### Against `{os.path.relpath(args.baseline, HERE)}` (contiguous, this run / that run - 1)", "",
              "| Shapes | Count | streamk_wip | v9 (AITER) | AITER's pick | hipBLASLt |", "|---|---:|---:|---:|---:|---:|"]
        for gname, pred in (("Delegated to v9 now", lambda r: str(r.get("v20_v9_delegated")) == "True"),
                            ("Not delegated", lambda r: str(r.get("v20_v9_delegated")) != "True")):
            ps = [(r, base[key(r["M"], r["N"], r["K"], r["out"])]) for r in rows if pred(r)
                  and key(r["M"], r["N"], r["K"], r["out"]) in base]
            cells = [dist([pct(fnum(r[c]), fnum(o[c])) for r, o in ps]).split(" | ")[0]
                     for c in ("v20_tflops", "v9_tflops", "aiter_pick_tflops", "hbl_tflops")]
            L.append(f"| {gname} | {len(ps)} | " + " | ".join(cells) + " |")
        L += [""]
        losses = oct5_losses(base_rows)
        by = {key(r["M"], r["N"], r["K"], r["out"]): r for r in rows}
        L += [f"### The {len(losses)} same-tile shapes where streamk_wip was behind on Oct 5", "",
              "| Shape | Tiles | k-steps | Delegated now | Oct 5: vs best | Now contiguous: vs best | "
              "Now padded: vs best | Now padded: vs v9 padded |", "|---|---:|---:|---|---:|---:|---:|---:|"]
        for o in losses:
            r = by.get(key(o["M"], o["N"], o["K"], o["out"]))
            if not r:
                continue
            g5 = pct(fnum(o["v20_tflops"]), max(fnum(o["aiter_pick_tflops"]), fnum(o["hbl_tflops"])))
            gc = pct(fnum(r["v20_tflops"]), best_fields(r, "")["best"])
            gp = pct(fnum(r["v20_pad_tflops"]), best_fields(r, "_pad")["best"])
            L.append(f"| {r['M']}x{r['N']}x{r['K']} | {r['tiles']} | {r['k_steps']} | "
                     f"{'yes' if str(r.get('v20_v9_delegated')) == 'True' else 'no'} | {g5:+.1f}% | {gc:+.1f}% | "
                     f"{gp:+.1f}% | {pct(fnum(r['v20_pad_tflops']), fnum(r['v9_pad_tflops'])):+.1f}% |")
        L += [""]

    # ---- padding
    ctrl = [r for r in rows if not padded_strides(r)]
    L += ["## Padding: does it remove the noise?", "",
          f"Spread between the {n_rounds} rounds, per kernel and layout. With `realloc` every round has new buffers, so "
          "a timing that depends on where the buffers landed shows up here. Control: the "
          f"{len(ctrl)} shapes with K % 256 == 128, whose x and w strides are not padded (only C is).", "",
          "| Kernel | Shapes | contiguous: median spread | contiguous: > 2% | contiguous: > 5% | padded: median "
          "spread | padded: > 2% | padded: > 5% |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    sgroups = (("All, strides padded", lambda r: padded_strides(r)),) + tuple(
        (f"{g}, strides padded", (lambda g: lambda r: padded_strides(r) and grid_bucket(r) == g)(g)) for g in GRID_ORDER
    ) + (("Control (K % 256 == 128)", lambda r: not padded_strides(r)),)

    def sp(sel, col):
        xs = [fnum(r.get(col)) for r in sel]
        xs = [x for x in xs if x == x]
        if not xs:
            return "- | - | -"
        return f"{statistics.median(xs):.2f}% | {sum(x > 2 for x in xs)} | {sum(x > 5 for x in xs)}"

    for kname, c in SPREAD_KERNELS:
        for gname, pred in sgroups:
            sel = [r for r in rows if pred(r) and (c != "pick" or str(r["pick_pad_copy"]) != "True")]
            L.append(f"| {kname} | {gname} ({len(sel)}) | {sp(sel, f'{c}_spread')} | {sp(sel, f'{c}_pad_spread')} |")
    L += ["", "AITER's pick leaves out (on both layouts) the shapes where its padded call copied its inputs.", ""]

    L += ["### Padded against contiguous, median time over the rounds", "",
          "Padded TFLOPS / contiguous TFLOPS - 1 (positive: padding is faster).", "",
          "| Kernel | Shapes | Count | faster | within 1% | slower | median | p10 | p90 |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    pgroups = (("All, strides padded", lambda r: padded_strides(r)),) + tuple(
        (g, (lambda g: lambda r: padded_strides(r) and grid_bucket(r) == g)(g)) for g in GRID_ORDER
    ) + (("Control (K % 256 == 128)", lambda r: not padded_strides(r)),)
    for kname, cc, pc, flag in PAD_KERNELS:
        for gname, pred in pgroups:
            sel = [r for r in rows if pred(r) and (flag is None or str(r[flag]) != "True")]
            gaps = [pct(fnum(r[pc]), fnum(r[cc])) for r in sel]
            L.append(f"| {kname} | {gname} | {len(sel)} | {counts(gaps)} | {' | '.join(dist(gaps).split(' | ')[1:])} |")
    L += [""]

    # ---- correctness
    checks = [c for c in rows[0] if c.endswith("_ok") or c.endswith("_post_ok")] if rows else []
    bad = [r for r in rows if any(str(r.get(c)) == "False" for c in checks) or str(r["v20_flags_left"]) != "0"
           or r["note"]]
    pick_bad = [r for r in bad if all(str(r.get(c)) != "False" for c in checks if not c.startswith("pick"))
                and str(r["v20_flags_left"]) == "0" and not r["note"]]
    L += ["## Correctness", "",
          f"Checks: {', '.join(checks)}; and streamk_wip's flags and counts are back at 0 after timing.", "",
          f"- Shapes with any failed check or note: {len(bad)}; of those, {len(pick_bad)} are AITER's pick alone "
          "(contiguous and/or padded) outside the tolerance."]
    for r in bad:
        if r in pick_bad:
            continue
        failed = [c for c in checks if str(r.get(c)) == "False"]
        L.append(f"  - {r['M']}x{r['N']}x{r['K']} {r['out']}: failed {failed}, flags {r['v20_flags_left']}, "
                 f"note {r['note'][:200]}")
    L += [""]

    L += ["## Appendix: every shape (TFLOPS)", "",
          "| Shape | out | tiles | streamk_wip config | v9 hand-off | streamk_wip | padded | dp only | padded | v9 | "
          "padded | AITER pick | AITER | padded | hipBLASLt | padded | best (padded comparison) |",
          "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|"]
    for r in rows:
        b = best_fields(r, "_pad")
        tc, tp = times(r, ""), times(r, "_pad")
        cp = " (copy)" if str(r["pick_pad_copy"]) == "True" else ""
        L.append(f"| {r['M']}x{r['N']}x{r['K']} | {r['out']} | {r['tiles']} | {r['v20_label']} | "
                 f"{'yes' if str(r.get('v20_v9_delegated')) == 'True' else ''} | {tc['v20']:.1f} | {tp['v20']:.1f} | "
                 f"{tc['dp']:.1f} | {tp['dp']:.1f} | {tc['v9']:.1f} | {tp['v9']:.1f} | {r['aiter_pick']} | "
                 f"{fnum(r['aiter_pick_tflops']):.1f} | {fnum(r['pick_pad_tflops']):.1f}{cp} | {fnum(r['hbl_tflops']):.1f} | "
                 f"{fnum(r['hbl_pad_tflops']):.1f} | {b['best_lib']} {b['best_layout']} {b['best']:.1f} |")
    L += [""]
    with open(args.summary, "w") as f:
        f.write("\n".join(L))
    print(f"wrote {args.summary}")
