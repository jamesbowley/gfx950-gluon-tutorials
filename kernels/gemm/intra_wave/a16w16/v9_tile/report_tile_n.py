##############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
##############################################################################
"""WORKLOG §19's tables, figure and list of traces to investigate, from run_tile_n.sh's results.

    python v9_tile/report_tile_n.py > v9_tile/results/tile_n/report.md

Reads results/tile_n_{timing_graph,timing,clock,counters,att_r1..3}.csv, results/tile_n/att_gaps_r*.json and
results/asm_tile_n.md; writes images/tile_n/efficiency.png and images/tile_n/idle.png.

- Chip-wide MFMA efficiency = TFLOPS / (1048.576 TFLOPS per GHz x clock): for the ATT launch with its
  kernel-trace TFLOPS and realtime.json clock, and back to back with the graph TFLOPS and hwmon clock.
- Window ceiling = min(1, 59.6 B/ns x C_cyc / (B x clock)) (README §3; Q = 32 KB, L = 550 ns), with
  C_cyc = BM BN / 32 MFMA cycles and B = (BM + BN) x 64 x 2 bytes per k-step.
"""

import csv
import json
import os
import statistics as S

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
IMG = os.path.join(HERE, "images", "tile_n")
PEAK_PER_GHZ = 1048.576  # TFLOPS per GHz
WINDOW_BNS = 32768 / 550.0  # B/ns per CU
TILES = [f"256x{bn}" for bn in range(256, 127, -16)]
REPEATS = (1, 2, 3)
# flag thresholds (points of efficiency unless noted)
MODEL_GAP = 0.05
REPEAT_SPREAD = 0.03
WHOLE_VS_LOOP = 0.02
CHIP_VS_WHOLE = 0.05
ONE_INSTRUCTION = 0.03  # share of the MFMA idle time
VMCNT_SHARE = 0.05
UNATTRIBUTED = 0.20
MONOTONIC = 0.01


def load(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return {}
    return {r["tile"]: r for r in csv.DictReader(open(path))}


def num(x):
    if x in (None, ""):
        return None
    return float(str(x).rstrip("%")) / (100 if str(x).endswith("%") else 1)


def window(tile, ghz):
    bm, bn = (int(v) for v in tile.split("x")[:2])
    c_cyc, b = bm * bn / 32, (bm + bn) * 64 * 2
    return min(1.0, WINDOW_BNS * c_cyc / (b * ghz)) if ghz else None


def asm_table():
    path = os.path.join(RES, "asm_tile_n.md")
    if not os.path.exists(path):
        return {}
    lines = [l for l in open(path) if l.startswith("| 256x")]
    head = next(l for l in open(path) if l.startswith("| tile"))
    keys = [k.strip() for k in head.strip("|\n").split("|")]
    return {l.split("|")[1].strip(): dict(zip(keys, (v.strip() for v in l.strip("|\n").split("|")))) for l in lines}


def gaps():
    out = {}
    for r in REPEATS:
        path = os.path.join(RES, "tile_n", f"att_gaps_r{r}.json")
        if not os.path.exists(path):
            continue
        for name, v in json.load(open(path)).items():
            out.setdefault(name.split("_")[0], {})[r] = v
    return out


def med(xs):
    xs = [x for x in xs if x is not None]
    return S.median(xs) if xs else None


def spread(xs):
    xs = [x for x in xs if x is not None]
    return max(xs) - min(xs) if len(xs) > 1 else 0.0


def pct(x, d=1):
    return "-" if x is None else f"{x * 100:.{d}f}%"


def fmt(x, f="{:.0f}"):
    return "-" if x is None else f.format(x)


def rows():
    g, t, c, p = (load(f"tile_n_{n}.csv") for n in ("timing_graph", "timing", "clock", "counters"))
    att = {r: load(f"tile_n_att_r{r}.csv") for r in REPEATS}
    out = []
    for tile in TILES:
        a = [att[r].get(tile, {}) for r in REPEATS]
        r = {"tile": tile, "bn": int(tile.split("x")[1]), "shape": f"4096x{16 * int(tile.split('x')[1])}x8192"}
        r["loop"] = [num(x.get("att_mfma_eff")) for x in a]
        r["whole"] = [num(x.get("att_e2e_eff")) for x in a]
        r["share"] = [num(x.get("loop_ratio")) for x in a]
        r["att_ghz"] = [num(x.get("att_ghz")) for x in a]
        r["att_tflops"] = [num(x.get("att_tflops")) for x in a]
        r["att_chip"] = [num(x.get("att_chip_eff")) for x in a]
        r["ui"] = [x.get("ui") for x in a]
        r["iso"] = num(t.get(tile, {}).get("tflops"))
        r["b2b"] = num(g.get(tile, {}).get("tflops"))
        r["sclk"] = num(c.get(tile, {}).get("sclk_ghz"))
        r["power"] = num(c.get(tile, {}).get("power_w"))
        r["sustained"] = num(c.get(tile, {}).get("sustained_tflops"))
        r["chip_b2b"] = r["b2b"] / (PEAK_PER_GHZ * r["sclk"]) if r["b2b"] and r["sclk"] else None
        r["pmc_busy"] = num(p.get(tile, {}).get("mfma_busy"))
        r["l2"] = num(p.get(tile, {}).get("l2_hit"))
        r["win_att"] = window(tile, med(r["att_ghz"]))
        r["win_b2b"] = window(tile, r["sclk"])
        out.append(r)
    return out


def efficiency_table(rs):
    print("| tile | shape | ATT loop eff (median [range]) | ATT whole wave | loop x share | ATT GHz | ATT TFLOPS | "
          "ATT chip-wide | window at ATT clock | isolated TFLOPS | back-to-back TFLOPS | hwmon GHz | W | "
          "back-to-back chip-wide | counters MFMA busy | window at hwmon clock | L2 hit |")
    print("|" + "---|" * 17)
    for r in rs:
        lx = [l * s for l, s in zip(r["loop"], r["share"]) if l is not None and s is not None]
        loop = f"{pct(med(r['loop']))} [{pct(min(filter(None, r['loop']), default=None))}, " \
               f"{pct(max(filter(None, r['loop']), default=None))}]"
        whole = f"{pct(med(r['whole']))} [{pct(min(filter(None, r['whole']), default=None))}, " \
                f"{pct(max(filter(None, r['whole']), default=None))}]"
        print(f"| {r['tile']} | {r['shape']} | {loop} | {whole} | {pct(med(lx))} | {fmt(med(r['att_ghz']), '{:.2f}')} | "
              f"{fmt(med(r['att_tflops']))} | {pct(med(r['att_chip']))} | {pct(r['win_att'], 0)} | {fmt(r['iso'])} | "
              f"{fmt(r['b2b'])} | {fmt(r['sclk'], '{:.2f}')} | {fmt(r['power'])} | {pct(r['chip_b2b'])} | "
              f"{pct(r['pmc_busy'])} | {pct(r['win_b2b'], 0)} | {pct(r['l2'])} |")


CATS = ["issue", "dependency hole", "lds: ds_read stall", "lds: lgkmcnt wait", "barrier", "other stall",
        "memory: vmcnt wait", "memory: buffer_load issue stall", "mfma stall, pipe empty"]


def gap_summary(v):
    """One repeat's att_gaps result in a few numbers."""
    idle = v["idle_per_iter"] or 1
    regs = [g for g in v["regions"] if g["mfma"] >= 2]
    fit_idle = [g for g in regs if g["fits"] and g["idle"] >= 5]
    return {"window": v["window_per_iter"], "idle": v["idle_per_iter"], "loop_eff": v["loop_eff"],
            "cats": {k: v["exposed_by_category"].get(k, 0) for k in CATS}, "unattr": v["unattributed"],
            "fit_idle": f"{len(fit_idle)}/{len(regs)}", "fit_idle_cyc": sum(g["idle"] for g in fit_idle),
            "nofit": [g for g in regs if not g["fits"]],
            "skew": max((b["skew"] for b in v["barriers"]), default=0), "share_top": v["hotspots"][0]["share_of_idle"]
            if v["hotspots"] else 0, "idle_norm": idle}


def gaps_table(rs, gp):
    print("| tile | repeat | cycles / iter | MFMA idle / iter | " + " | ".join(CATS) + " | unattributed | "
          "regions that fit but idle (idle cycles) | regions that do not fit | max barrier skew |")
    print("|" + "---|" * (len(CATS) + 8))
    for r in rs:
        for rep, v in sorted(gp.get(r["tile"], {}).items()):
            s = gap_summary(v)
            print(f"| {r['tile']} | {rep} | {s['window']:.0f} | {s['idle']:.0f} | "
                  + " | ".join(f"{s['cats'][k]:.0f}" for k in CATS)
                  + f" | {s['unattr']:.0f} | {s['fit_idle']} ({s['fit_idle_cyc']:.0f}) | {len(s['nofit'])} | "
                  f"{s['skew']:.0f} |")


def asm_out(asm):
    print("| tile | loop insts | MFMAs | non-MFMA | non-MFMA per MFMA | buffer loads | ds_reads | s_waitcnt | s_nop | "
          "longest non-MFMA run | VGPRs | AGPRs | LDS KB |")
    print("|" + "---|" * 13)
    for t in TILES:
        a = asm.get(t)
        if not a:
            continue
        ins, m = int(a["loop insts"]), int(a["MFMAs"])
        print(f"| {t} | {ins} | {m} | {ins - m} | {(ins - m) / m:.2f} | {a['buffer loads']} | {a['ds_reads']} | "
              f"{a['s_waitcnt']} | {a['s_nop']} | {a['longest non-MFMA run']} | {a['VGPRs']} | {a['AGPRs']} | "
              f"{a['LDS KB']} |")


def flags(rs, gp):
    """Per trace, the rules it trips: (rule, text) pairs, with the trace's att_gaps result."""
    out = []
    loop_med = {r["tile"]: med(r["loop"]) for r in rs}
    for i, r in enumerate(rs):
        tile_reasons = []
        if spread(r["loop"]) > REPEAT_SPREAD:
            tile_reasons.append(("spread", f"repeats spread {spread(r['loop']) * 100:.1f} points "
                                           f"({', '.join(pct(x) for x in r['loop'])})"))
        if i + 1 < len(rs) and loop_med[r["tile"]] is not None and loop_med[rs[i + 1]["tile"]] is not None \
                and loop_med[r["tile"]] < loop_med[rs[i + 1]["tile"]] - MONOTONIC:
            tile_reasons.append(("monotonic", f"non-monotonic: {pct(loop_med[r['tile']])} below the narrower "
                                              f"{rs[i + 1]['tile']}'s {pct(loop_med[rs[i + 1]['tile']])}"))
        for k, rep in enumerate(REPEATS):
            v = gp.get(r["tile"], {}).get(rep)
            loop, whole, share = r["loop"][k], r["whole"][k], r["share"][k]
            ghz, chip = r["att_ghz"][k], r["att_chip"][k]
            reasons = list(tile_reasons)
            w = window(r["tile"], ghz)
            if loop is not None and w is not None and abs(loop - w) > MODEL_GAP:
                reasons.append(("model", f"loop {pct(loop)} vs window ceiling {pct(w, 0)} at {ghz:.2f} GHz "
                                         f"({'under' if loop < w else 'over'} by {abs(loop - w) * 100:.1f} points)"))
            if None not in (loop, whole, share) and whole < loop * share - WHOLE_VS_LOOP:
                reasons.append(("whole", f"whole wave {pct(whole)} < loop x share {pct(loop * share)}"))
            if None not in (chip, whole) and chip < whole - CHIP_VS_WHOLE:
                reasons.append(("chip", f"chip-wide {pct(chip)} < whole wave {pct(whole)}"))
            if v:
                idle = v["idle_per_iter"] or 1
                hot = [h for h in v["hotspots"] if h["share_of_idle"] > ONE_INSTRUCTION]
                if hot:
                    reasons.append(("one", f"{len(hot)} instruction(s) over {ONE_INSTRUCTION:.0%} of MFMA idle "
                                           f"each (largest {hot[0]['share_of_idle']:.1%})"))
                vm = v["exposed_by_category"].get("memory: vmcnt wait", 0)
                if vm / idle > VMCNT_SHARE:
                    reasons.append(("vmcnt", f"vmcnt waits {vm / idle:.0%} of MFMA idle"))
                if v["unattributed"] / idle > UNATTRIBUTED:
                    reasons.append(("unattr", f"unattributed {v['unattributed'] / idle:.0%} of MFMA idle"))
            out.append((r, rep, k, v, reasons))
    return out


def summary_line(v):
    s = gap_summary(v)
    cats = sorted(s["cats"].items(), key=lambda kv: -kv[1])
    idle = s["idle"] or 1
    top = ", ".join(f"{k} {c / idle:.0%}" for k, c in cats[:4] if c > 0)
    return (f"MFMA idle {s['idle']:.0f} of {s['window']:.0f} cycles per iteration ({top}); "
            f"{s['fit_idle']} MFMA regions fit their shadow yet idle ({s['fit_idle_cyc']:.0f} cycles); "
            f"max barrier skew {s['skew']:.0f} cycles.")


RULE_NAMES = {"model": "window model gap", "spread": "repeat spread", "whole": "whole wave under loop x share",
              "chip": "chip-wide under whole wave", "one": "one instruction over the share", "vmcnt": "vmcnt share",
              "unattr": "unattributed share", "monotonic": "non-monotonic in BN"}


def flags_out(fl, rs):
    print(f"Thresholds: window model gap {MODEL_GAP:.0%}, repeat spread {REPEAT_SPREAD:.0%}, whole wave vs loop x "
          f"share {WHOLE_VS_LOOP:.0%}, chip-wide vs whole wave {CHIP_VS_WHOLE:.0%}, one instruction "
          f"{ONE_INSTRUCTION:.0%} of MFMA idle, vmcnt {VMCNT_SHARE:.0%}, unattributed {UNATTRIBUTED:.0%}, monotonic "
          f"{MONOTONIC:.0%}.\n")
    fired = {}
    for _, _, _, _, reasons in fl:
        for rule, _ in reasons:
            fired[rule] = fired.get(rule, 0) + 1
    common = {rule for rule, n in fired.items() if n == len(fl)}
    if common:
        print("Rules every trace trips (not repeated per tile below): "
              + "; ".join(f"**{RULE_NAMES[r]}**" for r in sorted(common)) + ".")
        if "chip" in common:
            gaps_ = [r["whole"][k] - r["att_chip"][k] for r in rs for k in range(len(REPEATS))
                     if r["whole"][k] is not None and r["att_chip"][k] is not None]
            print(f"  Chip-wide efficiency of the ATT launch sits {min(gaps_) * 100:.1f}-{max(gaps_) * 100:.1f} points "
                  "under CU 0's whole-wave efficiency on every tile: the kernel lasts longer than CU 0's waves "
                  "(dispatch ramp, the slowest CUs' tail).")
        print()
    by_tile = {}
    for r, rep, k, v, reasons in fl:
        by_tile.setdefault(r["tile"], []).append((r, rep, k, v, [x for x in reasons if x[0] not in common]))
    for tile, entries in by_tile.items():
        if not any(e[4] for e in entries):
            continue
        r = entries[0][0]
        mid = sorted(entries, key=lambda e: e[0]["loop"][e[2]] or 0)[len(entries) // 2]
        print(f"#### {tile} (loop {', '.join(pct(r['loop'][e[2]]) for e in entries)}; ATT clock "
              f"{', '.join(fmt(r['att_ghz'][e[2]], '{:.2f}') for e in entries)} GHz)\n")
        for _, rep, k, v, reasons in entries:
            files = f" (wave files {', '.join(v['wave_files'])})" if v and rep == mid[1] else ""
            print(f"- Repeat {rep}: `{r['ui'][k]}`{files}")
            seen = set()
            for rule, text in reasons:
                if rule in ("spread", "monotonic") and rep != entries[0][1]:
                    continue
                if text not in seen:
                    print(f"  - {text}")
                    seen.add(text)
        v = mid[3]
        if v:
            print(f"- Repeat {mid[1]} (median): {summary_line(v)}")
            print("- Hot spots, repeat {} (exposed cycles per iteration and wave):".format(mid[1]))
            for h in v["hotspots"][:6]:
                prev = f" after `{h['prev_isa']}`" if h["hole"] > 0 else ""
                print(f"  - code index {h['idx']}: `{h['isa']}`{prev} ({h['class']}): {h['hole']:.1f} hole + "
                      f"{h['stall']:.1f} stall + {h['issue']:.1f} issue, {h['share_of_idle']:.1%} of MFMA idle")
            waits = [w for w in v["waits"] if w["stalled_per_iter"] > 0.5][:3]
            for w in waits:
                print(f"  - wait at code index {w['idx']} `{w['isa']}` ({w['kind']}): stalls {w['stall']:.0f} cycles "
                      f"{w['stalled_per_iter']:.2f}x per iteration, {w['dist_cyc']:.0f} cycles "
                      f"({w['dist_mfma']:.0f} MFMAs) after the load it waits on")
        print()


def plot(rs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(IMG, exist_ok=True)
    bn = [r["bn"] for r in rs]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    series = [("ATT hot loop", [med(r["loop"]) for r in rs], "o-"),
              ("ATT whole wave", [med(r["whole"]) for r in rs], "s-"),
              ("chip-wide, ATT launch", [med(r["att_chip"]) for r in rs], "^--"),
              ("chip-wide, back to back", [r["chip_b2b"] for r in rs], "v-"),
              ("window ceiling at hwmon clock", [r["win_b2b"] for r in rs], "k:"),
              ("window ceiling at ATT clock", [r["win_att"] for r in rs], "k-.")]
    for name, ys, style in series:
        pts = [(x, y * 100) for x, y in zip(bn, ys) if y is not None]
        if pts:
            ax.plot(*zip(*pts), style, label=name)
    lo = [min(filter(None, r["loop"]), default=None) for r in rs]
    hi = [max(filter(None, r["loop"]), default=None) for r in rs]
    pts = [(x, a * 100, b * 100) for x, a, b in zip(bn, lo, hi) if a is not None]
    if pts:
        ax.fill_between([p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts], alpha=0.15)
    ax.set_xlabel("BLOCK_N (BLOCK_M = 256, BK = 64; 256 tiles, one per CU, K = 8192)")
    ax.set_ylabel("MFMA efficiency (%)")
    ax.invert_xaxis()
    ax.set_xticks(bn)
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    for name, ys, style in (("ATT clock", [med(r["att_ghz"]) for r in rs], "r.--"),
                            ("hwmon clock, back to back", [r["sclk"] for r in rs], "m.-")):
        pts = [(x, y) for x, y in zip(bn, ys) if y is not None]
        if pts:
            ax2.plot(*zip(*pts), style, label=name, alpha=0.7)
    ax2.set_ylabel("clock (GHz)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left")
    ax.set_title("v9_tile 256xBN: MFMA efficiency and clock")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "efficiency.png"), dpi=130)
    plt.close(fig)


def plot_idle(rs, gp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    bn, bottoms = [r["bn"] for r in rs], [0.0] * len(rs)
    for cat in CATS + ["unattributed"]:
        ys = []
        for r in rs:
            reps = gp.get(r["tile"], {})
            vals = [(v["exposed_by_category"].get(cat, 0) if cat != "unattributed" else v["unattributed"])
                    / v["window_per_iter"] * 100 for v in reps.values()]
            ys.append(S.median(vals) if vals else 0)
        ax.bar([str(b) for b in bn], ys, bottom=bottoms, label=cat)
        bottoms = [a + b for a, b in zip(bottoms, ys)]
    ax.set_xlabel("BLOCK_N (BLOCK_M = 256)")
    ax.set_ylabel("MFMA idle, % of loop cycles (median of repeats)")
    ax.set_title("What the MFMAs fail to hide, per loop iteration (att_gaps.py)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "idle.png"), dpi=130)
    plt.close(fig)


def main():
    rs = rows()
    gp = gaps()
    print("### Efficiency, TFLOPS and clock\n")
    efficiency_table(rs)
    print("\n### The compiled loop (one iteration = two k-steps, per wave)\n")
    asm_out(asm_table())
    print("\n### MFMA idle per loop iteration and wave, by what was exposed (att_gaps.py)\n")
    gaps_table(rs, gp)
    print("\n### Traces to investigate\n")
    flags_out(flags(rs, gp), rs)
    plot(rs)
    if gp:
        plot_idle(rs, gp)


if __name__ == "__main__":
    main()
