#!/usr/bin/env python3
"""shapes_all_tuned.csv: every gfx950 bf16 tuned GEMM shape of the harness's AITER worktree not in the 630.

    python make_shape_csv_all_tuned.py [--aiter /home/jbowley/repos/aiter_llirsched_wt] [--out shapes_all_tuned.csv]
    python make_shape_csv_all_tuned.py --include-scoped --bf16-only --out shapes_origami.csv \
        --bias-only shapes_origami_bias_only.csv --excluded shapes_origami_excluded.csv

Rows come from aiter/configs/bf16_tuned_gemm.csv and aiter/configs/model_configs/*bf16_tuned_gemm*.csv (gfx950,
cu_num 256, bf16 inputs, no scaleAB or bpreshuffle). One row per (M, N, K, output dtype) that has a bias=False entry:
the harness asks AITER for the bias=False config, so a shape tuned only with bias would time an untuned fallback.
Those are written to --bias-only instead. (M, N, K) in supported_status_scoped.csv are left out unless
--include-scoped. With --bf16-only, fp32-output rows go to --excluded instead.

scope is the regime by the small side: "both >= 512", "M < 512" or "N < 512 <= M". band is the 256x256 tile count
band of supported_status_scoped.csv (<128, 128-255, 256-511, >=512). aiter_kernels lists the bias=False tuned
kernels (libtype:kernelName:splitK); bf16_reduce is the class of those that reduce K in bf16 with bf16 output
(2026-10-20 addendum: they all fail the harness's fp32-reference check): flydsl_splitk (ks > 1, packed bf16 atomics),
flydsl_kwaves (two or more K-waves summed in bf16 in LDS), asm_splitk (*splitk* with splitK > 1); empty otherwise.
"""

import argparse
import csv
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DTYPES = {"torch.bfloat16": "bfloat16", "torch.float32": "float32"}


def scope(M, N):
    if M >= 512 and N >= 512:
        return "both >= 512"
    return "M < 512" if M < 512 else "N < 512 <= M"


def band(M, N):
    t = -(-M // 256) * -(-N // 256)
    return "<128" if t < 128 else "128-255" if t < 256 else "256-511" if t < 512 else ">=512"


def bf16_reduce(libtype, kernel, split_k, out):
    if out != "bfloat16":
        return ""
    if libtype == "flydsl":
        ks = re.search(r"_ks(\d+)_", kernel)
        if ks and int(ks.group(1)) > 1:
            return "flydsl_splitk"
        kw = re.search(r"_w\d+x\d+x(\d+)_", kernel)
        if kw and int(kw.group(1)) > 1:
            return "flydsl_kwaves"
    if libtype == "asm" and "splitk" in kernel and int(split_k or 1) > 1:
        return "asm_splitk"
    return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aiter", default="/home/jbowley/repos/aiter_llirsched_wt")
    ap.add_argument("--out", default=os.path.join(HERE, "shapes_all_tuned.csv"))
    ap.add_argument("--bias-only", default=os.path.join(HERE, "shapes_all_tuned_bias_only.csv"))
    ap.add_argument("--include-scoped", action="store_true", help="keep the (M, N, K) of supported_status_scoped.csv")
    ap.add_argument("--bf16-only", action="store_true", help="fp32-output rows go to --excluded")
    ap.add_argument("--excluded", default=None, help="with --bf16-only: where the fp32-output rows go")
    args = ap.parse_args()

    cfg = os.path.join(args.aiter, "aiter", "configs")
    files = [os.path.join(cfg, "bf16_tuned_gemm.csv")] + sorted(glob.glob(os.path.join(cfg, "model_configs",
                                                                                         "*bf16_tuned_gemm*.csv")))
    seen = {}  # (M, N, K, out) -> {bias values}, sources, bias=False kernels
    for f in files:
        for r in csv.DictReader(open(f)):
            if (r.get("gfx") != "gfx950" or r.get("cu_num") != "256" or r.get("dtype") != "torch.bfloat16"
                    or r.get("scaleAB") == "True" or r.get("bpreshuffle") == "True"
                    or r.get("outdtype") not in OUT_DTYPES):
                continue
            k = (int(r["M"]), int(r["N"]), int(r["K"]), OUT_DTYPES[r["outdtype"]])
            e = seen.setdefault(k, (set(), set(), set()))
            e[0].add(r["bias"])
            e[1].add(os.path.basename(f).replace("_bf16_tuned_gemm.csv", "").replace("bf16_tuned_gemm.csv", "base"))
            if r["bias"] == "False":
                e[2].add((r["libtype"], r.get("kernelName", ""), r.get("splitK", "")))

    old = {(int(r["M"]), int(r["N"]), int(r["K"]))
           for r in csv.DictReader(open(os.path.join(HERE, "supported_status_scoped.csv")))}
    rows, bias_only, excluded = [], [], []
    for (M, N, K, out), (biases, srcs, kernels) in sorted(seen.items()):
        if (M, N, K) in old and not args.include_scoped:
            continue
        row = dict(M=M, N=N, K=K, bias=False, out=out, scope=scope(M, N), band=band(M, N),
                   models=" ".join(sorted(srcs)))
        if args.include_scoped or args.bf16_only:
            row.update(scoped=(M, N, K) in old,
                       aiter_kernels=" | ".join(sorted(f"{lt}:{kn}:{sk}" for lt, kn, sk in kernels)),
                       bf16_reduce=" ".join(sorted({c for lt, kn, sk in kernels if (c := bf16_reduce(lt, kn, sk, out))})))
        if "False" not in biases:
            bias_only.append(row)
        elif args.bf16_only and out != "bfloat16":
            excluded.append(row)
        else:
            rows.append(row)
    outs = [(args.out, rows), (args.bias_only, bias_only)]
    if args.bf16_only:
        assert args.excluded, "--bf16-only needs --excluded"
        outs.append((args.excluded, excluded))
    for path, rs in outs:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rs)
    mnk = {(r["M"], r["N"], r["K"]) for r in rows}
    by = {}
    for r in rows:
        by[r["scope"]] = by.get(r["scope"], 0) + 1
    print(f"{len(rows)} rows ({len(mnk)} distinct M, N, K) -> {args.out}: "
          + ", ".join(f"{k} {v}" for k, v in sorted(by.items())))
    print(f"{len(bias_only)} rows tuned only with bias -> {args.bias_only}")
    if args.bf16_only:
        print(f"{len(excluded)} fp32-output rows -> {args.excluded}")
    if "bf16_reduce" in rows[0]:
        cls = {}
        for r in rows:
            cls[r["bf16_reduce"] or "none"] = cls.get(r["bf16_reduce"] or "none", 0) + 1
        print("bf16_reduce: " + ", ".join(f"{k} {v}" for k, v in sorted(cls.items())))


if __name__ == "__main__":
    main()
