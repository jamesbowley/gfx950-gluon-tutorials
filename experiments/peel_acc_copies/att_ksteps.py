"""MFMA bubbles per K-step: for each wave, gaps between consecutive MFMAs beyond the MFMA's
16 issue cycles, grouped by K-step (MFMAs / 16 K-steps), median over waves.

    python att_ksteps.py <att_dir> [...]
"""
import glob
import json
import os
import statistics
import sys

KSTEPS = 16
MFMA_CYCLES = 16


def per_wave(d):
    ui = glob.glob(os.path.join(d, "ui_output_*"))[0]
    code = json.load(open(os.path.join(ui, "code.json")))["code"]
    out = []
    for f in sorted(glob.glob(os.path.join(ui, "se*_wv*.json"))):
        ins = json.load(open(f))["wave"]["instructions"]
        t = [x[0] for x in ins if "mfma" in code[x[4]][0]]
        if not t:
            continue
        per = len(t) // KSTEPS
        bub = [0.0] * KSTEPS
        for i in range(1, len(t)):
            bub[min(i // per, KSTEPS - 1)] += max(0, t[i] - t[i - 1] - MFMA_CYCLES)
        out.append(bub)
    return [statistics.median(w[k] for w in out) for k in range(KSTEPS)]


print("| trace | " + " | ".join(f"k{k}" for k in range(KSTEPS)) + " | total |")
print("|---" * (KSTEPS + 2) + "|")
for d in sys.argv[1:]:
    b = per_wave(d)
    print(f"| {os.path.basename(d)} | " + " | ".join(f"{x:.0f}" for x in b) + f" | {sum(b):.0f} |")
