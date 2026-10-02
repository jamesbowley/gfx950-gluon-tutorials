#!/usr/bin/env python3
"""Cross-check hbl_bench against hipBLASLt's own hipblaslt-bench (TheRock tests tarballs), GPU 3.

    python bench_check.py shapes     # regressed shapes (128 MiB) + quoted 32 MiB shapes -> bench_check.csv
    python bench_check.py issue      # run the hipblaslt-bench lines from hipblaslt_issue.md as written

Versions: ROCm 10.0.0 (hipBLASLt 1.4.1) and the 10.2.0a20260929 nightly (hipBLASLt 1.5.0).
"""

import csv
import os
import re
import shlex
import subprocess
import sys

import analyze as a

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNERS = {a.MID: "run_rocm1000_gpu3.sh", a.NEW: "run_therock_nightly_gpu3.sh"}


def bench_args(m, n, k, dt, workspace=None):
    t = "f32_r" if dt == "f32" else "bf16_r"
    args = (f"-m {m} -n {n} -k {k} --transA T --transB N --lda {k} --ldb {k} --ldc {m} --ldd {m} "
            f"--a_type bf16_r --b_type bf16_r --c_type {t} --d_type {t} --compute_type f32_r "
            f"--alpha 1 --beta 0 -j 20 -i 100 --print_kernel_info")
    if workspace:
        args += f" --workspace {workspace}"
    return args


def run(version, argline, env=None):
    cmd = [os.path.join(HERE, RUNNERS[version])]
    if env:
        cmd += ["env"] + [f"{k}={v}" for k, v in env.items()]
    cmd += ["hipblaslt-bench"] + shlex.split(argline)
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    text = p.stdout + p.stderr
    kernel = re.search(r"--kernel name:\s+(\S+)", text)
    header = re.search(r"\[0\]:(\S+)\n\s+(\S+)", text)
    tflops = us = None
    if header:
        cols, vals = header[1].split(","), header[2].split(",")
        d = dict(zip(cols, vals))
        tflops = float(d["hipblaslt-Gflops"]) / 1e3
        us = float(d["us"])
    err = ""
    if p.returncode != 0 or not header:
        tail = [l for l in text.strip().splitlines() if l.strip()][-3:]
        err = f"rc={p.returncode}: " + " | ".join(tail)[:300]
    return dict(kernel=kernel[1] if kernel else "", tflops=tflops, us=us, error=err, rc=p.returncode)


def shapes():
    reg = [tuple(l.split()) for l in open(os.path.join(HERE, "regress_shapes.txt")) if l.strip() and not l.startswith("#")]
    ref = {v: a.read(os.path.join(HERE, f"heuristic_{v}.csv")) for v in (a.MID, a.NEW)}
    ref32 = {v: a.read(os.path.join(HERE, f"heuristic_ws32_{v}.csv")) for v in (a.MID, a.NEW)}
    # 32 MiB cases: every shape where hbl_bench saw an error or a >=20% loss vs 1.2.2 at 32 MiB.
    b32 = a.read(os.path.join(HERE, f"heuristic_ws32_{a.BASE}.csv"))
    ws_keys = set()
    for v in (a.MID, a.NEW):
        for key, r in ref32[v].items():
            if a.status(r["kernel"]) or (b32[key]["tflops"] > 0 and r["tflops"] < 0.8 * b32[key]["tflops"]):
                ws_keys.add(key)
    jobs = [(tuple(int(x) for x in s[:3]) + (s[3],), None) for s in reg]
    jobs += [(key, 32 << 20) for key in sorted(ws_keys)]
    out = []
    for i, (key, ws) in enumerate(jobs):
        for v in (a.MID, a.NEW):
            r = run(v, bench_args(*key, workspace=ws))
            ours = (ref32 if ws else ref)[v][key]
            ours_kernel = ours["kernel"].split(" [")[0]
            ours_err = a.status(ours["kernel"])
            same = r["kernel"] == ours_kernel
            row = dict(
                version=v, m=key[0], n=key[1], k=key[2], dtype_d=key[3], workspace_mib=(ws or 128 << 20) >> 20,
                bench_tile=a.mt(r["kernel"]) if r["kernel"] else "", bench_tflops=r["tflops"] or "",
                bench_error=r["error"], ours_tile=a.mt(ours_kernel), ours_tflops=ours["tflops"] or "",
                ours_error=ours_err, same_kernel=same,
                tflops_ratio=round(r["tflops"] / ours["tflops"], 3) if r["tflops"] and ours["tflops"] else "",
            )
            out.append(row)
            print(f"[{i + 1}/{len(jobs)}] {v} {key} ws={row['workspace_mib']}MiB same_kernel={same} "
                  f"bench={row['bench_tflops'] and round(row['bench_tflops'], 1)} ours={row['ours_tflops']} "
                  f"{r['error'][:120]}", flush=True)
    with open(os.path.join(HERE, "bench_check.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    summarize(out)


def summarize(out=None):
    if out is None:
        out = list(csv.DictReader(open(os.path.join(HERE, "bench_check.csv"))))
    for v in (a.MID, a.NEW):
        for ws in (128, 32):
            rs = [r for r in out if r["version"] == v and int(r["workspace_mib"]) == ws]
            if not rs:
                continue
            same = sum(str(r["same_kernel"]) == "True" for r in rs)
            both_err = sum(bool(r["bench_error"]) and bool(r["ours_error"]) for r in rs)
            ok = [float(r["tflops_ratio"]) for r in rs if r["tflops_ratio"] not in ("", None)]
            within = sum(abs(x - 1) <= 0.03 for x in ok)
            print(f"{v} @ {ws} MiB: {len(rs)} shapes; same kernel {same}/{len(rs)}; TFLOPS within 3% on "
                  f"{within}/{len(ok)} (median ratio {sorted(ok)[len(ok) // 2] if ok else 'n/a'}); "
                  f"errors in both {both_err}; errors bench-only "
                  f"{sum(bool(r['bench_error']) and not r['ours_error'] for r in rs)}, ours-only "
                  f"{sum(bool(r['ours_error']) and not r['bench_error'] for r in rs)}")


def issue():
    text = open(os.path.join(HERE, "hipblaslt_issue.md")).read()
    block = re.search(r"```bash\n(.*?)```", text, re.S)[1]
    for line in block.splitlines():
        line = line.strip()
        if not line.startswith(("hipblaslt-bench", "ANALYTICAL_GEMM_PICK")):
            continue
        env = {}
        if line.startswith("ANALYTICAL_GEMM_PICK"):
            kv, line = line.split(" ", 1)
            env = dict([kv.split("=")])
        argline = line[len("hipblaslt-bench"):].strip()
        versions = [a.NEW] if env else [a.MID, a.NEW]
        for v in versions:
            r = run(v, argline, env)
            print(f"{v}: {'ANALYTICAL_GEMM_PICK ' if env else ''}{argline[:60]}... -> rc={r['rc']} "
                  f"{a.mt(r['kernel']) if r['kernel'] else '-'} {r['tflops'] and round(r['tflops'], 1)} TFLOPS {r['error'][:150]}")


if __name__ == "__main__":
    {"shapes": shapes, "issue": issue, "summary": summarize}[sys.argv[1]]()
