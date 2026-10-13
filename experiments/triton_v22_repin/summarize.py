#!/usr/bin/env python3
"""Tables for results.md from run_matrix.sh logs.

    python summarize.py compare <dir a1>,<dir a2> <dir b1>,<dir b2> [label a] [label b]
    python summarize.py bias <nobias dir1>,<dir2> <bias dir1>,<dir2>

`compare`: every (table, config, kernel) row of a and b with the mean over the given runs of
TFLOPS and MFMA efficiency, VGPRs/spills, and the relative TFLOPS change. `bias`: v9-v13
with and without bias.
"""

import os
import re
import statistics
import sys

ROW = re.compile(r"^\|\s*(\S+)\s*\|\s*(\S+)\s*\|\s*(\S+)\s*\|\s*(\S+)\s*\|\s*(\S+)\s*\|")


def parse_dir(d):
    rows = {}
    for name in sorted(os.listdir(d)):
        if not name.endswith(".log"):
            continue
        table = name[: -len(".log")].replace("_bias", "")
        config = None
        summary = False
        for line in open(os.path.join(d, name)):
            if "RESULTS SUMMARY" in line:
                summary = True
            if not summary:
                continue
            if line.startswith("Config:"):
                config = line.split(":", 1)[1].strip()
            m = ROW.match(line)
            if m and config and m.group(1) not in ("Version",) and not m.group(1).startswith("-"):
                k, tf, vg, sp, eff = m.groups()
                rows[(table, config, k)] = dict(
                    tflops=float(tf) if tf.replace(".", "").isdigit() else None,
                    vgprs=vg,
                    spills=sp,
                    eff=float(eff.rstrip("%")) if eff.endswith("%") else None,
                )
    return rows


def merged(dirs):
    runs = [parse_dir(d) for d in dirs]
    out = {}
    for key in runs[0]:
        vals = [r[key] for r in runs if key in r]
        tf = [v["tflops"] for v in vals if v["tflops"] is not None]
        ef = [v["eff"] for v in vals if v["eff"] is not None]
        out[key] = dict(
            tflops=statistics.mean(tf) if tf else None,
            eff=statistics.mean(ef) if ef else None,
            vgprs=vals[0]["vgprs"],
            spills=vals[0]["spills"],
            tf_range=(min(tf), max(tf)) if tf else None,
        )
    return out


def fmt(v, spec):
    return "FAIL" if v is None else format(v, spec)


def compare(a_dirs, b_dirs, la="before", lb="after"):
    a, b = merged(a_dirs), merged(b_dirs)
    print(
        f"| Table | Config | Kernel | {la} TFLOPS | {lb} TFLOPS | change | {la} MFMA eff | "
        f"{lb} MFMA eff | VGPRs / spills ({la} -> {lb}) |"
    )
    print("|---|---|---|---:|---:|---:|---:|---:|---|")
    for key in a:
        if key not in b:
            continue
        x, y = a[key], b[key]
        ch = (y["tflops"] / x["tflops"] - 1) * 100 if x["tflops"] and y["tflops"] else None
        regs = f"{x['vgprs']}/{x['spills']}"
        if (y["vgprs"], y["spills"]) != (x["vgprs"], x["spills"]):
            regs += f" -> **{y['vgprs']}/{y['spills']}**"
        print(
            f"| {key[0]} | {key[1]} | {key[2]} | {fmt(x['tflops'], '.0f')} | {fmt(y['tflops'], '.0f')} | "
            f"{fmt(ch, '+.1f')}% | {fmt(x['eff'], '.2f')}% | {fmt(y['eff'], '.2f')}% | {regs} |"
        )


def bias(nb_dirs, b_dirs):
    nb, b = merged(nb_dirs), merged(b_dirs)
    print(
        "| Dtype | Config | Kernel | TFLOPS no bias | TFLOPS bias | bias cost | MFMA eff no bias | "
        "MFMA eff bias | VGPRs / spills no bias | VGPRs / spills bias |"
    )
    print("|---|---|---|---:|---:|---:|---:|---:|---|---|")
    for key in b:
        x = nb.get(key)
        if x is None:
            continue
        y = b[key]
        ch = (y["tflops"] / x["tflops"] - 1) * 100 if x["tflops"] and y["tflops"] else None
        dtype = key[0].split("_")[-1]
        print(
            f"| {dtype} | {key[1]} | {key[2]} | {fmt(x['tflops'], '.0f')} | {fmt(y['tflops'], '.0f')} | "
            f"{fmt(ch, '+.1f')}% | {fmt(x['eff'], '.2f')}% | {fmt(y['eff'], '.2f')}% | "
            f"{x['vgprs']}/{x['spills']} | {y['vgprs']}/{y['spills']} |"
        )


if __name__ == "__main__":
    mode = sys.argv[1]
    dirs_a = sys.argv[2].split(",")
    dirs_b = sys.argv[3].split(",")
    if mode == "compare":
        compare(dirs_a, dirs_b, *sys.argv[4:6])
    else:
        bias(dirs_a, dirs_b)
