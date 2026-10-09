"""Phase breakdown of ATT wave traces: prologue (begin -> first MFMA), MFMA span, epilogue
(last MFMA -> end), plus the instructions that wait longest in prologue and epilogue.

    python att_phases.py <att_dir> [<att_dir> ...] [--top N]
"""
import collections
import glob
import json
import os
import re
import statistics
import sys

top = 0
args = sys.argv[1:]
if "--top" in args:
    i = args.index("--top")
    top = int(args[i + 1])
    del args[i:i + 2]


def norm(t):
    t = t.split(";")[0].strip()
    if t.startswith("s_waitcnt"):
        return t
    return t.split()[0] if t else t


def analyze(d):
    ui = glob.glob(os.path.join(d, "ui_output_*"))[0]
    code = json.load(open(os.path.join(ui, "code.json")))["code"]
    rows, waits = [], {"pro": collections.Counter(), "epi": collections.Counter()}
    for f in sorted(glob.glob(os.path.join(ui, "se*_wv*.json"))):
        w = json.load(open(f))["wave"]
        ins = w["instructions"]
        mf = [k for k, x in enumerate(ins) if "mfma" in code[x[4]][0]]
        if not mf:
            continue
        b, e = w["begin"], w["end"]
        f0, f1 = ins[mf[0]][0], ins[mf[-1]][0]
        for k in range(1, len(ins)):
            gap = ins[k][0] - ins[k - 1][0]
            if k <= mf[0]:
                waits["pro"][norm(code[ins[k][4]][0])] += gap
            elif k > mf[-1]:
                waits["epi"][norm(code[ins[k][4]][0])] += gap
        rows.append(dict(total=e - b, pro=f0 - b, span=f1 - f0, epi=e - f1, n=len(ins), n_pro=mf[0],
                         n_epi=len(ins) - mf[-1] - 1, mfma=len(mf), begin=b))
    med = {k: statistics.median(r[k] for r in rows) for k in rows[0]}
    return med, waits, len(rows)


print("| trace | waves | wave cycles | prologue | MFMA span | epilogue | insts | prologue insts | epilogue insts |")
print("|---|---|---|---|---|---|---|---|---|")
detail = []
for d in args:
    m, w, n = analyze(d)
    print(f"| {os.path.basename(d)} | {n} | {m['total']:.0f} | {m['pro']:.0f} | {m['span']:.0f} | {m['epi']:.0f} | "
          f"{m['n']:.0f} | {m['n_pro']:.0f} | {m['n_epi']:.0f} |")
    detail.append((d, w, n))
if top:
    for d, w, n in detail:
        print(f"\n{os.path.basename(d)} (per wave)")
        for ph in ("pro", "epi"):
            items = ", ".join(f"{k} {v / n:.0f}" for k, v in w[ph].most_common(top))
            print(f"  {ph}: {items}")
