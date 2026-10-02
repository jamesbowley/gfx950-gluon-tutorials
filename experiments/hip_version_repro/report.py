#!/usr/bin/env python3
"""Tutorial tables, the issue's scope summary per environment, and per-shape changes.

    python report.py            # writes tutorial_<env>.md and comparison.md
    python report.py --summary supported_status_scoped_issue.csv   # one CSV's scope summary
"""

import csv
import os
import re
import statistics
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ENVS = {
    "issue": "Issue (torch 2.10 + ROCm 7.2.4, as reported)",
    "rocm724": "torch 2.10.0+rocm7.2.4 (rerun here)",
    "nightly": "TheRock nightly torch 2.15.0a0 + ROCm 10.2.0a20260929",
}
TUT_SHAPES = ["4096 4096 8192", "8192 8192 8192", "4096 8192 4096", "4096 106496 16384"]
TUT_VARIANTS = [
    ("triton", "Triton backend"),
    ("gluon_noplugin", "Gluon, no plugin"),
    ("gluon_llirsched", "Gluon + llirSched"),
    ("hipblaslt", "hipBLASLt (F.linear)"),
]
ISSUE_TUT = {
    "4096 4096 8192": (1305, 1354, 1589, 1536),
    "8192 8192 8192": (1248, 1338, 1590, 1586),
    "4096 8192 4096": (1224, 1316, 1526, 1492),
    "4096 106496 16384": (1201, 1326, 1589, 1498),
}


def csv_path(env):
    return os.path.join(HERE, f"supported_status_scoped_{env}.csv")


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def pct(xs, q):
    """Linear-interpolated percentile (numpy's default)."""
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def hbl_tile(name):
    m = re.search(r"_MT(\d+x\d+)x\d+", name)
    return m[1] if m else ""


def hbl_sk(name):
    m = re.search(r"_(SK\d)_", name)
    return m[1] if m else ("Custom" if "Custom_Cijk" in name else "none")


def summary(rows):
    """The issue's 'Gluon's gap to the winner' table, as numbers."""
    out = {"measured": len(rows)}
    scope = Counter(r["scope"] for r in rows)
    out["smaller"] = scope["out: smaller tile"]
    out["splitk"] = scope["out: split-K"]
    cb = [r for r in rows if r["scope"].startswith("in:")]
    out["compute_bound"] = len(cb)
    out["gluon_wins"] = sum(r["winner"] == "gluon" for r in cb)
    for name, sc in (("other", "in: other listed tile"), ("nonpow2", "in: non-pow2 tile")):
        gaps = [float(r["gap_pct"]) for r in cb if r["scope"] == sc]
        out[name] = (len(gaps), statistics.median(gaps) if gaps else float("nan"), pct(gaps, 90))
    g256 = [float(r["gap_pct"]) for r in cb if r["scope"] == "in: 256x256" and r["winner"] != "gluon"]
    out["same256"] = (len(g256), statistics.median(g256) if g256 else float("nan"), max(g256, default=float("nan")))
    # The issue counts a gap that rounds to 0.0% as a Gluon win here (195 of 225).
    big = [r for r in cb if int(r["tiles"]) >= 512]
    out["big"] = (
        sum(float(r["gap_pct"]) == 0.0 for r in big),
        len(big),
        max((float(r["gap_pct"]) for r in big if r["winner"] != "gluon"), default=0.0),
    )
    hbl = [r for r in rows if r["winner"] == "torch:hipBLASLt"]
    out["hbl_wins"] = len(hbl)
    out["hbl_sk"] = Counter((re.search(r"_(SK\d)_", r["hbl"]) or [None, "?"])[1] for r in hbl)
    return out


def summary_md(label, s):
    o, n, g = s["other"], s["nonpow2"], s["same256"]
    sk = ", ".join(f"{k}: {v}" for k, v in sorted(s["hbl_sk"].items()))
    return [
        f"**{label}**",
        "",
        "| Shapes | Count | Gluon's gap to the winner |",
        "|---|---:|---|",
        f"| Measured | {s['measured']} | |",
        f"| Out of scope (smaller tile: {s['smaller']}; split-K: {s['splitk']}) | {s['smaller'] + s['splitk']} | |",
        f"| Compute-bound | {s['compute_bound']} | |",
        f"| Gluon wins | {s['gluon_wins']} | 0 |",
        f"| Lost to another of the eight tiles | {o[0]} | median {o[1]:.0f}%, p90 {o[2]:.0f}% |",
        f"| Lost to a non-power-of-2 tile | {n[0]} | median {n[1]:.0f}%, p90 {n[2]:.0f}% |",
        f"| Lost to another 256x256 kernel | {g[0]} | median {g[1]:.1f}% (up to {g[2]:.0f}%) |",
        "",
        f"At 512 or more output tiles Gluon wins (or ties to 0.1%) {s['big'][0]} of {s['big'][1]} "
        f"shapes; its worst loss there is {s['big'][2]:.1f}%. hipBLASLt wins {s['hbl_wins']} of the "
        f"630 shapes (stream-K mode of the winning kernel: {sk}).",
        "",
    ]


def tutorial(env):
    path = os.path.join(HERE, "logs", f"tutorial_{env}_results.log")
    vals = defaultdict(list)
    info = ""
    for line in open(path):
        if line.startswith("env="):
            info = line.strip()
        m = re.match(r"RESULT (\S+) (\d+) (\d+) (\d+) ([\d.]+)", line)
        if m:
            vals[(m[1], f"{m[2]} {m[3]} {m[4]}")].append(float(m[5]))
    med = {k: statistics.median(v) for k, v in vals.items()}
    return med, vals, info


def tutorial_md(env):
    med, vals, info = tutorial(env)
    lines = [
        f"# Tutorial shapes: {ENVS[env]}",
        "",
        f"`{info}`",
        "",
        "TFLOPS, `bench_gemm_a16w16.py` (hipBLASLt: `F.linear` with bias, `do_bench_cudagraph`), "
        "one process per measurement, median of 3 interleaved rounds, GPU 0.",
        "",
        "| M×N×K | " + " | ".join(n for _, n in TUT_VARIANTS) + " |",
        "|---|" + "---:|" * len(TUT_VARIANTS),
    ]
    for s in TUT_SHAPES:
        lines.append(
            f"| {s.replace(' ', '×')} | "
            + " | ".join(f"{med.get((v, s), float('nan')):.0f}" for v, _ in TUT_VARIANTS)
            + " |"
        )
    lines += ["", "Individual rounds:", ""]
    for s in TUT_SHAPES:
        for v, n in TUT_VARIANTS:
            lines.append(f"- {s.replace(' ', '×')} {n}: " + ", ".join(f"{x:.0f}" for x in vals.get((v, s), [])))
    return "\n".join(lines) + "\n", med


def main():
    if sys.argv[1:2] == ["--summary"]:
        print("\n".join(summary_md(sys.argv[2], summary(load(sys.argv[2])))))
        return

    tut = {}
    for env in ("rocm724", "nightly"):
        text, tut[env] = tutorial_md(env)
        with open(os.path.join(HERE, f"tutorial_{env}.md"), "w") as f:
            f.write(text)

    rows = {env: load(csv_path(env)) for env in ENVS}
    key = lambda r: (r["M"], r["N"], r["K"], r["out"])  # noqa: E731
    by = {env: {key(r): r for r in rs} for env, rs in rows.items()}
    details = {
        env: {key(r): r for r in load(os.path.join(HERE, f"details_{env}.csv"))}
        for env in ("rocm724", "nightly")
    }
    info = {
        env: dict(l.strip().split("=", 1) for l in open(os.path.join(HERE, f"envinfo_{env}.txt")))
        for env in ("rocm724", "nightly")
    }

    L = ["# Issue GEMM numbers on two hip stacks", ""]
    L += ["## Environments", "", "| | torch | HIP | hipBLASLt | GPU 0 PCI bus |", "|---|---|---|---|---|"]
    for env in ("rocm724", "nightly"):
        i = info[env]
        L.append(f"| {env} | {i['torch']} | {i['hip']} | {i['hipblaslt']} | {i['gpu_pci_bus']} |")
    L += [
        "",
        "Both use Triton `gfx950-tutorial-v2.2` (llirSched plugin loaded), the same AITER checkout "
        "and tuned CSVs, and GPU 0 only. The issue's own run used torch 2.10.0+rocm7.2.4 as well.",
        "",
    ]

    L += ["## Tutorial shapes (TFLOPS, median of 3)", ""]
    L += ["| M×N×K | Variant | Issue | rocm724 | nightly | nightly vs rocm724 |", "|---|---|---:|---:|---:|---:|"]
    for s in TUT_SHAPES:
        for j, (v, n) in enumerate(TUT_VARIANTS):
            a, b = tut["rocm724"].get((v, s), float("nan")), tut["nightly"].get((v, s), float("nan"))
            L.append(f"| {s.replace(' ', '×')} | {n} | {ISSUE_TUT[s][j]} | {a:.0f} | {b:.0f} | {100 * (b / a - 1):+.1f}% |")
    L.append("")

    L += ["## Scope summary per environment (the issue's table, recomputed)", ""]
    for env, label in ENVS.items():
        L += summary_md(label, summary(rows[env]))

    # Per-shape changes.
    L += ["## Per-shape changes", ""]
    for a, b in (("issue", "rocm724"), ("rocm724", "nightly")):
        flips = [k for k in by[a] if (by[a][k]["winner"] == "gluon") != (by[b][k]["winner"] == "gluon")]

        def margin(r):
            g, p = float(r["gluon_tflops"]), float(r["aiter_pick_tflops"])
            return 100 * abs(p / g - 1)

        real = [k for k in flips if max(margin(by[a][k]), margin(by[b][k])) > 2.0]
        L.append(f"- **{a} → {b}:** {len(flips)} shapes change between Gluon winning and AITER's pick "
                 f"winning; on {len(flips) - len(real)} of them Gluon and the pick are within 2% of each "
                 f"other in both runs (near-ties), leaving {len(real)} real flips.")
        for k in sorted(real, key=lambda k: by[a][k]["aiter_pick"]):
            ra, rb = by[a][k], by[b][k]
            L.append(f"  - {k[0]}×{k[1]}×{k[2]} {k[3]} ({ra['aiter_pick']}): {a} gluon {ra['gluon_tflops']} vs "
                     f"pick {ra['aiter_pick_tflops']}; {b} gluon {rb['gluon_tflops']} vs pick {rb['aiter_pick_tflops']}")
    hb = [k for k, r in by["issue"].items() if r["hbl"]]
    same = sum(details["rocm724"][k]["hbl_kernels"] == by["issue"][k]["hbl"] for k in hb)
    L += ["", f"- **hipBLASLt kernels, issue vs rocm724 rerun:** the issue records the hipBLASLt kernel for "
          f"its {len(hb)} hipBLASLt wins; the rerun launches the identical kernel(s) on {same} of them.", ""]

    # TFLOPS deltas by live pick libtype, with v9 drift alongside.
    L += ["### TFLOPS change by AITER pick libtype", "",
          "Median per-shape ratio. The Gluon column is the same v9 kernel in every environment, so it "
          "shows the GPU/day drift; the pick's residual is the pick ratio divided by Gluon's ratio.", ""]
    for a, b in (("issue", "rocm724"), ("rocm724", "nightly"), ("issue", "nightly")):
        L += [f"**{a} → {b}**", "", "| Pick | Shapes | Pick | Gluon v9 | Residual | Shapes with residual below −3% |",
              "|---|---:|---:|---:|---:|---:|"]
        grp = defaultdict(list)
        for k, ra in by[a].items():
            rb = by[b][k]
            if ra["aiter_pick"] != rb["aiter_pick"]:
                grp["(pick changed)"].append((ra, rb))
            else:
                grp[ra["aiter_pick"]].append((ra, rb))
        for lib, pairs in sorted(grp.items(), key=lambda t: -len(t[1])):
            pr = [float(rb["aiter_pick_tflops"]) / float(ra["aiter_pick_tflops"]) for ra, rb in pairs]
            gr = [float(rb["gluon_tflops"]) / float(ra["gluon_tflops"]) for ra, rb in pairs]
            res = [p / g for p, g in zip(pr, gr)]
            L.append(f"| {lib} | {len(pairs)} | {100 * (statistics.median(pr) - 1):+.1f}% | "
                     f"{100 * (statistics.median(gr) - 1):+.1f}% | {100 * (statistics.median(res) - 1):+.1f}% | "
                     f"{sum(r < 0.97 for r in res)} |")
        L.append("")

    # hipBLASLt tile changes on torch-picked shapes.
    L += ["### hipBLASLt kernel changes on `torch`-picked shapes (rocm724 → nightly)", ""]
    tchg = []
    for k, d in details["rocm724"].items():
        n = details["nightly"][k]
        if d["aiter_pick"] == "torch" and n["aiter_pick"] == "torch":
            ta = hbl_tile(d["hbl_kernels"]) or "?"
            tb = hbl_tile(n["hbl_kernels"]) or "?"
            pa, pb = float(d["aiter_pick_tflops"]), float(n["aiter_pick_tflops"])
            ga, gb = float(d["gluon_tflops"]), float(n["gluon_tflops"])
            tchg.append((k, ta, tb, pa, pb, ga, gb))
    same = [t for t in tchg if t[1] == t[2]]
    diff = [t for t in tchg if t[1] != t[2]]
    sks = Counter(
        (hbl_sk(details["rocm724"][k]["hbl_kernels"]), hbl_sk(details["nightly"][k]["hbl_kernels"]))
        for k, *_ in tchg
    )
    L.append(f"{len(tchg)} shapes are `torch` picks in both. hipBLASLt keeps the same macro tile on "
             f"{len(same)} and switches to a different one on {len(diff)}. Stream-K mode: "
             + ", ".join(f"{a}→{b}: {n}" for (a, b), n in sks.most_common())
             + " (hipBLASLt's SK3→SK5 rename; SK5 with hybrid mode off runs the SK3 path).")
    for name, ts in (("same kernel tile", same), ("different kernel tile", diff)):
        if ts:
            res = [(pb / pa) / (gb / ga) for _, _, _, pa, pb, ga, gb in ts]
            L.append(f"- {name}: median residual {100 * (statistics.median(res) - 1):+.1f}%, "
                     f"{sum(r < 0.97 for r in res)} below −3%, {sum(r > 1.03 for r in res)} above +3%.")
    for label, pred in (("non-power-of-2", lambda t: any(int(v) & (int(v) - 1) for v in t.split("x"))),):
        np2 = [t for t in diff if pred(t[2]) and not pred(t[1])]
        if np2:
            res = [(pb / pa) / (gb / ga) for _, _, _, pa, pb, ga, gb in np2]
            L.append(f"- power-of-2 tile → {label} tile: {len(np2)} shapes, median residual "
                     f"{100 * (statistics.median(res) - 1):+.1f}%, {sum(r < 0.97 for r in res)} below −3%, "
                     f"{sum(r > 1.03 for r in res)} above +3%.")
    L += ["", "Shapes where the macro tile changed, worst first (Change = pick TFLOPS ratio; ordering uses "
          "the drift-corrected residual):", "",
          "| Shape | Tiles | rocm724 tile | TFLOPS | nightly tile | TFLOPS | Change | Gluon change | Winner rocm724 → nightly |",
          "|---|---:|---|---:|---|---:|---:|---:|---|"]
    for k, ta, tb, pa, pb, ga, gb in sorted(diff, key=lambda t: (t[4] / t[3]) / (t[6] / t[5])):
        wa = "gluon" if by["rocm724"][k]["winner"] == "gluon" else "hipBLASLt"
        wb = "gluon" if by["nightly"][k]["winner"] == "gluon" else "hipBLASLt"
        tiles = by["rocm724"][k]["tiles"]
        shape = f"{k[0]}×{k[1]}×{k[2]}" + ("" if k[3] == "bfloat16" else " fp32")
        L.append(f"| {shape} | {tiles} | {ta} | {pa:.0f} | {tb} | {pb:.0f} | {100 * (pb / pa - 1):+.1f}% | "
                 f"{100 * (gb / ga - 1):+.1f}% | {wa} → {wb} |")
    L.append("")

    # Correctness and anomalies.
    L += ["## Correctness and anomalies", ""]
    for env in ("rocm724", "nightly"):
        d = details[env].values()
        bad = [r for r in d if r["pick_ok"] != "True" or r["gluon_ok"] != "True"]
        notes = [r for r in d if r["note"]]
        picks = Counter(r["aiter_pick"] for r in d)
        L.append(f"- **{env}:** {len(d)} shapes; {len(bad)} with an output outside the fp32 tolerance; "
                 f"{len(notes)} with errors. Live picks: " + ", ".join(f"{k} {v}" for k, v in picks.most_common()) + ".")
        for r in bad + notes:
            L.append(f"  - {r['M']}×{r['N']}×{r['K']} {r['out']}: pick_ok={r['pick_ok']} gluon_ok={r['gluon_ok']} "
                     f"pick_maxerr={r['pick_maxerr']} gluon_maxerr={r['gluon_maxerr']} {r['note']}")
    ipk = Counter(r["aiter_pick"] for r in rows["issue"])
    L.append("- **issue:** picks " + ", ".join(f"{k} {v}" for k, v in ipk.most_common()) + ".")
    L += [
        "",
        "The outputs outside tolerance are the same 29 FlyDSL picks in both environments, and none is "
        "a hipBLASLt shape. Their max error (1–4) is the same size as Gluon's on those shapes (outputs "
        "are around 100, where one bf16 step is 1). They all split K, inside the workgroup (`w…x2`) "
        "or with split-K (`ks>1`), so an extra bf16 rounding of the partial sums fails the element-wise "
        "`allclose(atol=0.1, rtol=0.01)` on outputs close to zero. The issue reported all outputs correct, "
        "so it presumably used a looser check.",
        "",
        "## Method notes",
        "",
        "- Timing, shapes, bias and output dtype follow the issue: `do_bench_cudagraph`, 2 interleaved "
        "rounds (median) of AITER's live `tuned_gemm.gemm_a16w16` pick and Gluon v9 "
        "(`backend=\"gluon\", kernel_type=\"compute_bound\"`), same inputs, fp32 reference check.",
        "- Columns are rebuilt by `derive.py`, whose rules reproduce all 630 rows of the issue's CSV "
        "(`python derive.py --validate`); the scope summary reproduces the issue's table.",
        "- hipBLASLt kernel names come from the torch profiler (the issue used rocprofv3).",
        "- The ROCm 7.2.4 runtime needs `AMD_COMGR_CACHE=0` in this container: with comgr's cache "
        "on, it compiles its blit kernels without the device libraries and every GPU call crashes. No "
        "fallback was needed; the whole stack (HIP runtime, hipBLASLt, rocBLAS, hipcc for AITER's JIT) "
        "is ROCm 7.2.4.",
        "- Both environments use their own AITER JIT and Triton cache directories, so `/opt/venv` "
        "is untouched. Everything ran on GPU 0 (PCI bus 117 in both), one environment at a time.",
        "",
    ]
    with open(os.path.join(HERE, "comparison.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
