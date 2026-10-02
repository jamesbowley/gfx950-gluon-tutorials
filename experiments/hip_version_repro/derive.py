#!/usr/bin/env python3
"""Build supported_status_scoped-format rows from raw per-shape measurements.

    python derive.py --validate                    # rules vs the issue CSV (must be 630/630)
    python derive.py <details.csv> <out.csv>       # sweep details -> issue-format CSV

Columns (same order as the issue's supported_status_scoped.csv):
M,N,K,bias,out,src,AI,tiles,gluon_tflops,aiter_pick_tflops,aiter_pick,aiter_kernel,
winner,gap_pct,hbl,bm,bn,sk,scope,band
"""

import csv
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ISSUE_CSV = os.path.join(HERE, "supported_status_scoped_issue.csv")
COLUMNS = [
    "M", "N", "K", "bias", "out", "src", "AI", "tiles", "gluon_tflops", "aiter_pick_tflops",
    "aiter_pick", "aiter_kernel", "winner", "gap_pct", "hbl", "bm", "bn", "sk", "scope", "band",
]
LISTED = {(256, 256), (256, 128), (128, 256), (128, 128), (256, 64), (64, 256), (128, 64), (64, 128)}
OUT_BYTES = {"bfloat16": 2, "float32": 4}


def arith_intensity(M, N, K, out):
    return round(2 * M * N * K / (2 * (M * K + N * K) + OUT_BYTES[out] * M * N))


def band(tiles):
    if tiles < 128:
        return "<128"
    if tiles < 256:
        return "128-255"
    if tiles < 512:
        return "256-511"
    return ">=512"


def main_hbl_kernel(hbl):
    names = [n.strip() for n in hbl.split("|") if n.strip()]
    return next((n for n in names if "_MT" in n), names[0] if names else "")


def tile_of(libtype, kernel, hbl):
    """(bm, bn, split_k) of the kernel AITER's pick ran, or (None, None, False) if unknown."""
    if libtype in ("torch", "hipblaslt"):
        m = re.search(r"_MT(\d+)x(\d+)x\d+", main_hbl_kernel(hbl))
        return (int(m[1]), int(m[2]), False) if m else (None, None, False)
    if libtype == "flydsl":
        m = re.search(r"_t(\d+)x(\d+)x\d+x\d+_ks(\d+)", kernel)
        return (int(m[1]), int(m[2]), int(m[3]) > 1) if m else (None, None, False)
    if libtype == "opus":
        m = re.search(r"opus_gemm_(?:[a-z_]+?_)?(\d+)x(\d+)x(\d+)x(\d+)_", kernel)
        return (int(m[2]), int(m[3]), "splitk" in kernel) if m else (None, None, False)
    if libtype == "asm":
        m = re.search(r"_tn_(\d+)x(\d+)", kernel)
        return (int(m[1]), int(m[2]), "splitk" in kernel) if m else (None, None, False)
    return None, None, False


def derive(raw):
    """raw: M,N,K,bias,out,src,gluon_tflops,aiter_pick_tflops,aiter_pick,aiter_kernel,hbl_kernels."""
    M, N, K = int(raw["M"]), int(raw["N"]), int(raw["K"])
    out = raw["out"]
    tiles = math.ceil(M / 256) * math.ceil(N / 256)
    g, a = float(raw["gluon_tflops"]), float(raw["aiter_pick_tflops"])
    lib, kern = raw["aiter_pick"], raw["aiter_kernel"]
    g_ok = g == g
    aiter_wins = a == a and (not g_ok or a > g)
    row = dict(
        M=M, N=N, K=K, bias=raw["bias"], out=out, src=raw["src"],
        AI=arith_intensity(M, N, K, out), tiles=tiles,
        gluon_tflops=f"{g:.1f}", aiter_pick_tflops=f"{a:.1f}",
        aiter_pick=lib, aiter_kernel=kern, band=band(tiles),
    )
    if aiter_wins:
        bm, bn, sk = tile_of(lib, kern, raw.get("hbl_kernels", ""))
        is_hbl = lib in ("torch", "hipblaslt")
        row.update(
            winner="torch:hipBLASLt" if lib == "torch" else f"{lib}:{kern}",
            gap_pct=f"{100 * (a / g - 1):.1f}" if g_ok else "nan",
            hbl=raw.get("hbl_kernels", "") if is_hbl else "",
            bm=bm if bm else "", bn=bn if bn else "", sk=sk,
        )
        if bm is None:
            scope = "unknown tile"
        elif (bm, bn) == (256, 256):
            scope = "in: 256x256"
        elif sk:
            scope = "out: split-K"
        elif (bm, bn) in LISTED:
            scope = "in: other listed tile"
        elif bm * bn < 8192:
            scope = "out: smaller tile"
        else:
            scope = "in: non-pow2 tile"
    else:
        row.update(winner="gluon", gap_pct="0.0", hbl="", bm=256, bn=256, sk=False)
        scope = "in: 256x256"
    row["scope"] = scope
    return {c: row[c] for c in COLUMNS}


def validate():
    rows = list(csv.DictReader(open(ISSUE_CSV, newline="")))
    checked = ["AI", "tiles", "winner", "gap_pct", "hbl", "bm", "bn", "sk", "scope", "band"]
    bad = 0
    for r in rows:
        d = derive(dict(r, hbl_kernels=r["hbl"]))
        diffs = []
        for c in checked:
            want, got = r[c], str(d[c])
            if c == "gap_pct":
                # the issue computed the gap before rounding TFLOPS to 0.1
                a, g = float(r["aiter_pick_tflops"]), float(r["gluon_tflops"])
                tol = 100 * (a / g) * (0.05 / a + 0.05 / g) + 0.1 + 1e-9
                if abs(float(want) - float(got)) > tol:
                    diffs.append((c, want, got))
            elif want != got:
                diffs.append((c, want, got))
        if diffs:
            bad += 1
            print(f"{r['M']}x{r['N']}x{r['K']} {r['out']}: {diffs}")
    print(f"{len(rows) - bad}/{len(rows)} rows reproduce the issue's derived columns")
    return bad == 0


def build(details_csv, out_csv):
    rows = [derive(r) for r in csv.DictReader(open(details_csv, newline=""))]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--validate"]:
        sys.exit(0 if validate() else 1)
    build(sys.argv[1], sys.argv[2])
