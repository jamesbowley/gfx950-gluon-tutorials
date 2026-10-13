#!/usr/bin/env python3
"""Merge the pure-hipBLASLt runs and pick the regression set.

    python analyze.py stageA    # sanity check vs the torch runs, write regress_shapes.txt
    python analyze.py final     # merge stage A + B, write hbl_pure_results.csv

Versions: hipBLASLt 1.2.2 (ROCm 7.2.4), 1.4.1 as shipped in ROCm 10.0.0 (file tag 10.0.0),
1.5.0 (TheRock nightly 10.2.0a20260929). Shapes are keyed in hipBLASLt terms (m, n, k, dtypeD);
torch F.linear M x N x K is n x m x k.
"""

import csv
import os
import re
import sys

import attribute_models as am

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
BASE, MID, NEW = "1.2.2", "10.0.0", "1.5.0"
VERS = [BASE, MID, NEW]
THRESH = 0.95  # "significant": at least 5% slower than 1.2.2


def tag(v):
    return v.replace(".", "_")


def read(path):
    rows = {}
    if not os.path.exists(path):
        return rows
    for line in open(path):
        if line.startswith("#") or line.startswith("m,n,k"):
            continue
        m, n, k, dt, mode, kernel, us, tf, cand = line.rstrip("\n").split(",")
        rows[(int(m), int(n), int(k), dt)] = dict(kernel=kernel, us=float(us), tflops=float(tf))
    return rows


def meta(path):
    return next(line[2:].strip() for line in open(path) if line.startswith("#"))


def mt(kernel):
    """Macro-tile, MFMA instruction and stream-K mode, e.g. '256x224 MI16 SK3'."""
    m = re.search(r"MT(\d+x\d+)x\d+", kernel)
    tile = m[1] if m else "?"
    mi = re.search(r"_MI(\d+)x\d+x\d+", kernel)
    mfma = f" MI{mi[1]}" if mi else ""
    if kernel.startswith("Custom"):
        return f"{tile}{mfma} (custom)"
    sk = re.search(r"_(SK\d)_", kernel)
    return f"{tile}{mfma} {sk[1]}" if sk else f"{tile}{mfma}"


def torch_key(key):
    m, n, k, dt = key
    return (str(n), str(m), str(k), "float32" if dt == "f32" else "bfloat16")


def tiles(key):
    return (key[0] // 256) * (key[1] // 256)


def stage_a():
    h = {v: read(os.path.join(HERE, f"heuristic_{v}.csv")) for v in VERS}
    for v in VERS:
        print(f"{v}: {meta(os.path.join(HERE, f'heuristic_{v}.csv'))}  shapes={len(h[v])}")
    keys = list(h[BASE])

    # Sanity check against the torch F.linear runs on GPU 0 (hbl_all_*.csv).
    tor = {}
    for env, v in (("rocm724", BASE), ("nightly", NEW)):
        tor[v] = {(r["M"], r["N"], r["K"], r["out"]): r for r in csv.DictReader(open(os.path.join(PARENT, f"hbl_all_{env}.csv")))}
    for v in (BASE, NEW):
        same = sum(mt(h[v][k]["kernel"]) == mt(tor[v][torch_key(k)]["hbl_kernels"].split(" | ")[0]) for k in keys)
        print(f"{v}: tile/MFMA/SK identical to the torch run on {same}/{len(keys)} shapes")

    reg = [k for k in keys if min(h[MID][k]["tflops"], h[NEW][k]["tflops"]) < THRESH * h[BASE][k]["tflops"]]
    imp = [k for k in keys if h[NEW][k]["tflops"] > h[BASE][k]["tflops"] / THRESH]
    print(f"regress >=5% in {MID} or {NEW}: {len(reg)} (>=256 tiles: {sum(tiles(k) >= 256 for k in reg)}); "
          f"{MID}: {sum(h[MID][k]['tflops'] < THRESH * h[BASE][k]['tflops'] for k in keys)}, "
          f"{NEW}: {sum(h[NEW][k]['tflops'] < THRESH * h[BASE][k]['tflops'] for k in keys)}; "
          f"improve >=5% in {NEW}: {len(imp)}")
    with open(os.path.join(HERE, "regress_shapes.txt"), "w") as f:
        f.write(f"# m n k dtypeD: >=5% slower than hipBLASLt 1.2.2 in ROCm {MID}'s 1.4.1 or in 1.5.0 (heuristic top-1)\n")
        for k in reg:
            f.write(f"{k[0]} {k[1]} {k[2]} {k[3]}\n")


def status(kernel):
    m = re.search(r"\[hipblasLtMatmul status (\d+)\]", kernel)
    return f"error {m[1]}" if m else ""


def final():
    h = {v: read(os.path.join(HERE, f"heuristic_{v}.csv")) for v in VERS}
    w32 = {v: read(os.path.join(HERE, f"heuristic_ws32_{v}.csv")) for v in VERS}
    b256 = {v: read(os.path.join(HERE, f"best256_{v}.csv")) for v in (MID, NEW)}
    pick = {v: read(os.path.join(HERE, f"pick256_{v}.csv")) for v in (MID, NEW)}
    out = []
    for k in h[BASE]:
        M, N, K = k[1], k[0], k[2]
        dt = "float32" if k[3] == "f32" else "bfloat16"
        base = h[BASE][k]["tflops"]
        row = dict(
            m=k[0], n=k[1], k=k[2], dtype_d=k[3], torch_MxNxK=f"{M}x{N}x{K}", output_tiles=tiles(k),
            models=am.describe(M, N, K, dt),
            **{f"tile_{tag(BASE)}": mt(h[BASE][k]["kernel"]), f"tflops_{tag(BASE)}": base},
        )
        for v in (MID, NEW):
            t = h[v][k]["tflops"]
            row[f"tile_{tag(v)}"] = mt(h[v][k]["kernel"])
            row[f"tflops_{tag(v)}"] = t
            row[f"delta_{tag(v)}_pct"] = round(100 * (t / base - 1), 1)
        row["regressed"] = min(row[f"tflops_{tag(MID)}"], row[f"tflops_{tag(NEW)}"]) < THRESH * base
        for v in (MID, NEW):
            bb, pp = b256[v].get(k), pick[v].get(k)
            row[f"best256_tile_{tag(v)}"] = mt(bb["kernel"]) if bb else ""
            row[f"best256_tflops_{tag(v)}"] = bb["tflops"] if bb and bb["tflops"] > 0 else ""
            row[f"pick256_tile_{tag(v)}"] = mt(pp["kernel"]) if pp else ""
            row[f"pick256_tflops_{tag(v)}"] = pp["tflops"] if pp and pp["tflops"] > 0 else ""
        b32 = w32[BASE][k]["tflops"]
        for v in VERS:
            r32 = w32[v][k]
            row[f"ws32_tile_{tag(v)}"] = mt(r32["kernel"])
            row[f"ws32_tflops_{tag(v)}"] = r32["tflops"]
            row[f"ws32_status_{tag(v)}"] = status(r32["kernel"])
            if v != BASE:
                ok = b32 > 0 and r32["tflops"] > 0
                row[f"ws32_delta_{tag(v)}_pct"] = round(100 * (r32["tflops"] / b32 - 1), 1) if ok else ""
        for v in VERS:
            row[f"kernel_{tag(v)}"] = h[v][k]["kernel"]
        out.append(row)
    with open(os.path.join(HERE, "hbl_pure_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    reg = [r for r in out if r["regressed"]]
    print(f"wrote {len(out)} rows, {len(reg)} regressed")
    for v in (MID, NEW):
        s = tag(v)
        okb = sum(r[f"best256_tflops_{s}"] != "" and r[f"best256_tflops_{s}"] >= THRESH * r[f"tflops_{tag(BASE)}"] for r in reg)
        okp = sum(r[f"pick256_tflops_{s}"] != "" and r[f"pick256_tflops_{s}"] >= THRESH * r[f"tflops_{tag(BASE)}"] for r in reg)
        miss = sum(r[f"best256_tflops_{s}"] == "" for r in reg)
        print(f"{v}: best MT256x256 within 5% of 1.2.2 on {okb}/{len(reg)} (missing {miss}); ANALYTICAL_GEMM_PICK within 5% on {okp}")
        errs = [r for r in out if r[f"ws32_status_{s}"]]
        slow = [r for r in out if not r[f"ws32_status_{s}"] and r[f"ws32_delta_{s}_pct"] != "" and r[f"ws32_delta_{s}_pct"] <= -5]
        print(f"32 MiB workspace, {v}: {len(errs)} hipblasLtMatmul errors, {len(slow)} shapes >=5% slower than 1.2.2 at 32 MiB")


if __name__ == "__main__":
    {"stageA": stage_a, "final": final}[sys.argv[1]]()
