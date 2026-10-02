#!/usr/bin/env python3
"""Write hipblaslt_issue.md from hbl_pure_results.csv and bench_check.csv.

    python analyze.py final && python make_issue.py
"""

import csv
import os
import re
import statistics

import analyze as a

HERE = os.path.dirname(os.path.abspath(__file__))
B, M, N = a.tag(a.BASE), a.tag(a.MID), a.tag(a.NEW)
LABEL = {B: "1.2.2", M: "1.4.1 (ROCm 10.0.0)", N: "1.5.0"}


def f(x):
    return float(x) if x not in ("", None) else float("nan")


def models(s):
    """All models: exact matches first, then 'layer:' matches."""
    parts = [p.strip() for p in s.split(";")]
    exact = [p.replace(" (exact)", "") for p in parts if p.endswith("(exact)")]
    layer = [p.replace(" (layer)", "") for p in parts if p.endswith("(layer)")]
    other = [p for p in parts if not p.endswith(("(exact)", "(layer)"))]
    text = ", ".join(exact + other)
    if layer:
        text += ("; " if text else "") + "layer: " + ", ".join(layer)
    return text


def meta(v, ws=""):
    line = next(l for l in open(os.path.join(HERE, f"heuristic{ws}_{v}.csv")) if l.startswith("#"))
    return dict(kv.split("=", 1) for kv in line[2:].split())


def cell(r, s):
    t = f(r[f"tflops_{s}"])
    if s == B:
        return f"{r[f'tile_{s}']}: {t:.0f}"
    return f"{r[f'tile_{s}']}: {t:.0f} ({f(r[f'delta_{s}_pct']):+.1f}%)"


def pct(r, key):
    t, b = f(r[key]), f(r[f"tflops_{B}"])
    return f"{t:.0f} ({100 * (t / b - 1):+.1f}%)" if t == t and t > 0 else "–"


def row_big(r):
    c = [models(r["models"]), f"{r['m']}, {r['n']}, {r['k']}" + (" (fp32 D)" if r["dtype_d"] == "f32" else ""),
         r["output_tiles"], cell(r, B), cell(r, M), cell(r, N),
         pct(r, f"best256_tflops_{M}"), pct(r, f"best256_tflops_{N}"), pct(r, f"pick256_tflops_{N}")]
    return "| " + " | ".join(str(x) for x in c) + " |"


def row_small(r):
    c = [models(r["models"]), f"{r['m']}, {r['n']}, {r['k']}" + (" (fp32 D)" if r["dtype_d"] == "f32" else ""),
         r["output_tiles"], cell(r, B), cell(r, M), cell(r, N)]
    return "| " + " | ".join(str(x) for x in c) + " |"


HEADER_BIG = (
    f"| Model(s) | m, n, k | Tiles | {LABEL[B]} | {LABEL[M]} | {LABEL[N]} | Best MT256x256 in {LABEL[M]} | "
    f"Best MT256x256 in {LABEL[N]} | {LABEL[N]} + `ANALYTICAL_GEMM_PICK=256x256x64` |\n"
    "|---|---|---:|---|---|---|---|---|---|"
)
HEADER_SMALL = (
    f"| Model(s) | m, n, k | Tiles | {LABEL[B]} | {LABEL[M]} | {LABEL[N]} |\n|---|---|---:|---|---|---|"
)
LEGEND = [
    "- **m, n, k:** hipBLASLt column-major problem size, `opA=T, opB=N`, `lda=ldb=k`, `ldc=ldd=m`. For PyTorch "
    "`F.linear(x[M,K], w[N,K])`: m = N, n = M, k = K.",
    "- **Model(s):** models whose AITER tuning list (`aiter/configs/model_configs`) contains this exact GEMM; "
    "*layer* means only the weight shape matches (same m and k, different token count n).",
    "- **Tiles:** output tiles at 256x256, (m/256)·(n/256); the GPU has 256 CUs.",
    f"- **{LABEL[B]} / {LABEL[M]} / {LABEL[N]}:** heuristic top-1 (`hipblasLtMatmulAlgoGetHeuristic`, 128 MiB "
    "workspace), shown as `macro-tile MFMA stream-K: TFLOPS (change vs 1.2.2)`. MI16 / MI32 = MI16x16x1 / MI32x32x1.",
]
LEGEND_BIG = LEGEND + [
    "- **Best MT256x256:** the fastest of all MT256x256x64 solutions returned by `hipblaslt_ext::getAllAlgos`, each "
    "timed individually. Shows that the kernel still exists in that library and what it achieves.",
    f"- **{LABEL[N]} + `ANALYTICAL_GEMM_PICK=256x256x64`:** heuristic top-1 with that environment variable set, which "
    "restricts Origami's ranking to MT256x256x64 configs. It exists only in develop (#7228), not in 1.4.1.",
]


def issue_lines_output():
    """The reproduce lines as run on this machine (bench_check.py issue -> bench_check_issue_lines.log)."""
    path = os.path.join(HERE, "bench_check_issue_lines.log")
    if not os.path.exists(path):
        return []
    out = ["Output of these lines here (`hipblaslt-bench` from the TheRock tarballs, selected kernel and TFLOPS):", ""]
    for line in open(path):
        m = re.match(r"(\S+): (ANALYTICAL_GEMM_PICK )?-m (\d+) -n (\d+) -k (\d+).*?-> rc=(\d+) (.+?) ([\d.]+) TFLOPS", line)
        if not m:
            continue
        v = "1.4.1 (ROCm 10.0.0)" if m[1] == a.MID else "1.5.0"
        ws = " (32 MiB)" if (m[3], m[4], m[5]) == ("2304", "2048", "16384") else ""
        pick = " + ANALYTICAL_GEMM_PICK" if m[2] else ""
        out.append(f"- m={m[3]} n={m[4]} k={m[5]}{ws}, {v}{pick}: {m[7]}, {float(m[8]):.0f} TFLOPS")
    return out + [""]


def main():
    rows = list(csv.DictReader(open(os.path.join(HERE, "hbl_pure_results.csv"))))
    reg = [r for r in rows if r["regressed"] == "True"]
    worst = lambda r: min(f(r[f"delta_{M}_pct"]), f(r[f"delta_{N}_pct"]))  # noqa: E731
    big = sorted([r for r in reg if int(r["output_tiles"]) >= 256], key=worst)
    small = sorted([r for r in reg if int(r["output_tiles"]) < 256], key=worst)
    imp = [r for r in rows if f(r[f"delta_{N}_pct"]) >= 5]
    regM = [r for r in rows if f(r[f"delta_{M}_pct"]) <= -5]
    regN = [r for r in rows if f(r[f"delta_{N}_pct"]) <= -5]
    is256 = lambda r: r[f"tile_{B}"].startswith("256x256")  # noqa: E731
    big_sub = [r for r in big if is256(r)]
    big_other = [r for r in big if not is256(r)]
    sub256 = [r for r in reg if is256(r)]
    tile_only = lambda t: t.split()[0]  # noqa: E731
    mfma = lambda t: " ".join(t.split()[:2])  # noqa: E731
    small_new_only = [r for r in small if f(r[f"delta_{M}_pct"]) > -5]
    snm_same_tile = sum(tile_only(r[f"tile_{M}"]) == tile_only(r[f"tile_{N}"]) for r in small_new_only)
    snm_same_mfma = sum(mfma(r[f"tile_{M}"]) == mfma(r[f"tile_{N}"]) for r in small_new_only)
    m = {v: meta(v) for v in a.VERS}

    def recovered(rs, key):
        return sum(f(r[key]) == f(r[key]) and f(r[key]) >= 0.95 * f(r[f"tflops_{B}"]) for r in rs)

    err = {s: [r for r in rows if r[f"ws32_status_{s}"]] for s in (B, M, N)}
    ws_slowN = sorted(
        [r for r in rows if not r[f"ws32_status_{N}"] and r[f"ws32_delta_{N}_pct"] != "" and f(r[f"ws32_delta_{N}_pct"]) <= -20],
        key=lambda r: f(r[f"ws32_delta_{N}_pct"]),
    )
    ws_slowM = [r for r in rows if not r[f"ws32_status_{M}"] and r[f"ws32_delta_{M}_pct"] != "" and f(r[f"ws32_delta_{M}_pct"]) <= -20]
    same_ws = {v: sum(k1 == k2 for k1, k2 in zip(
        [r["kernel"].split(" [")[0] for r in a.read(os.path.join(HERE, f"heuristic_{v}.csv")).values()],
        [r["kernel"].split(" [")[0] for r in a.read(os.path.join(HERE, f"heuristic_ws32_{v}.csv")).values()])) for v in a.VERS}

    ex = next(r for r in rows if r["torch_MxNxK"] == "32768x8192x1024")
    w = next(r for r in rows if (r["m"], r["n"], r["k"]) == ("2304", "2048", "16384"))

    # hipblaslt-bench cross-check
    bc = list(csv.DictReader(open(os.path.join(HERE, "bench_check.csv"))))

    def bc_summary(v, ws, large=None):
        rs = [r for r in bc if r["version"] == v and int(r["workspace_mib"]) == ws]
        if large is not None:
            rs = [r for r in rs if ((int(r["m"]) // 256) * (int(r["n"]) // 256) >= 256) == large]
        same = sum(r["same_kernel"] == "True" for r in rs)
        ratios = sorted(float(r["tflops_ratio"]) for r in rs if r["tflops_ratio"])
        within = sum(abs(x - 1) <= 0.03 for x in ratios)
        errs = sum(bool(r["bench_error"]) and bool(r["ours_error"]) for r in rs)
        return len(rs), same, within, len(ratios), (statistics.median(ratios) if ratios else float("nan")), errs

    L = [
        "## gfx950 BF16 TN: hipBLASLt 1.4.1 (ROCm 10.0.0) and 1.5.0 heuristic selects slower solutions than 1.2.2 (ROCm 7.2.4)",
        "",
        "We observe a performance regression in the default heuristic solution selection for BF16 TN GEMM on gfx950 "
        "(MI355X) between hipBLASLt 1.2.2 (ROCm 7.2.4) and hipBLASLt 1.4.1 (ROCm 10.0.0) / 1.5.0 (develop). Could you "
        "confirm whether this is a known regression, the result of an incorrect configuration on our side (for example "
        "the workspace size, `HIPBLASLT_MATMUL_DESC_*` attributes or environment settings), or expected behaviour?",
        "",
        "For BF16 GEMM, TN, fp32 compute (`Cijk_Alik_Bljk_BBS`), the newer libraries replace MT256x256x64 SK3 on large "
        "compute-bound problems with the non-power-of-2 subtile macro-tiles added in #8604 (MT256x320, MT224x384, "
        "MT288x288, MT192x448). The only subtile reject is `K < 512` (#8626), so nothing filters them at larger K, and "
        "they are 5-13% slower. 1.5.0 also regresses many small problems (fewer than 256 output tiles) by up to 62% "
        "where 1.4.1 does not.",
        "",
        f"Example: PyTorch 32768×8192×1024 (hipBLASLt m={ex['m']}, n={ex['n']}, k={ex['k']}; Llama 70B): 1.2.2 selects "
        f"{ex[f'tile_{B}']} at {f(ex[f'tflops_{B}']):.0f} TFLOPS; 1.4.1 selects {ex[f'tile_{M}']} at "
        f"{f(ex[f'tflops_{M}']):.0f} ({f(ex[f'delta_{M}_pct']):+.1f}%); 1.5.0 selects {ex[f'tile_{N}']} at "
        f"{f(ex[f'tflops_{N}']):.0f} ({f(ex[f'delta_{N}_pct']):+.1f}%). An MT256x256x64 solution is still in both newer "
        f"libraries and runs at {f(ex[f'best256_tflops_{M}']):.0f} / {f(ex[f'best256_tflops_{N}']):.0f}.",
        "",
        "**Environment:**",
        "",
        f"- MI355X, one GPU (PCI bus {m[a.BASE]['pci_bus']}); every version measured on the same card, one after the other.",
        f"- hipBLASLt 1.2.2 (`{m[a.BASE]['rev']}`, ROCm 7.2.4 debs), 1.4.1 (`{m[a.MID]['rev']}`, TheRock ROCm 10.0.0 "
        f"gfx950 tarball), 1.5.0 (`{m[a.NEW]['rev']}`, TheRock nightly 10.2.0a20260929).",
        "- Standalone hipBLASLt benchmark (attached, no framework): `hipblasLtMatmulAlgoGetHeuristic` top-1, 128 MiB "
        "workspace (the `hipblaslt-bench` default). A/B BF16, C=D BF16 (fp32 where noted), alpha 1, beta 0, no bias. "
        "hipEvent timing, 20 warm-up + 100 timed calls, median of 3.",
        f"- Cross-checked with `hipblaslt-bench` from TheRock's tests tarballs for 1.4.1 and 1.5.0 (see Reproduce).",
        "- Problems: 630 BF16 GEMMs from AITER's per-model tuning lists.",
        "",
        "**Summary** (at least 5% slower than 1.2.2 counts as a regression):",
        "",
        f"- 1.4.1 (ROCm 10.0.0) regresses {len(regM)} of 630 problems, 1.5.0 regresses {len(regN)}; {len(reg)} regress in at least one.",
        f"- **Large problems (256 or more output tiles): {len(big)}**, {len(big_sub)} of them where 1.2.2 selects "
        f"MT256x256 SK3. 1.4.1 and 1.5.0 regress them by a similar amount (median "
        f"{statistics.median(f(r[f'delta_{M}_pct']) for r in big):+.1f}% / "
        f"{statistics.median(f(r[f'delta_{N}_pct']) for r in big):+.1f}%). Llama 70B, Llama 405B, Kimi-K3, Qwen 32B, "
        "MiniMax-M3, DeepSeek-V4.",
    ]
    if big_other:
        L.append(
            f"- The other {len(big_other)} large problems keep 1.2.2's macro-tile but select a different kernel: "
            + "; ".join(
                f"m={r['m']} n={r['n']} k={r['k']} ({models(r['models'])}): {r[f'tile_{B}']} {f(r[f'tflops_{B}']):.0f} → "
                f"{r[f'tile_{M}']} {f(r[f'tflops_{M}']):.0f} / {r[f'tile_{N}']} {f(r[f'tflops_{N}']):.0f}"
                for r in big_other
            )
            + ". An MT256x256 solution in the same libraries is within 5% of 1.2.2 on "
            + f"{recovered(big_other, f'best256_tflops_{M}')} of {len(big_other)} (1.4.1) and "
            + f"{recovered(big_other, f'best256_tflops_{N}')} of {len(big_other)} (1.5.0)."
        )
    L += [
        f"- **Small problems (fewer than 256 output tiles): {len(small)}**; {len(small_new_only)} regress in 1.5.0 only. "
        f"On {snm_same_tile} of those, 1.5.0 selects the same macro-tile as 1.4.1 ({snm_same_mfma} with the same MFMA "
        "as well), so they are not a tile-ranking change. DeepSeek-V4, Kimi / Kimi-K2 / Kimi-K3, MiniMax-M3, Llama 405B.",
        f"- **The MT256x256 solutions are still present.** On the {len(sub256)} regressed problems where 1.2.2 selects "
        f"MT256x256, the best MT256x256x64 solution is within 5% of 1.2.2 on {recovered(sub256, f'best256_tflops_{M}')} "
        f"in 1.4.1 and {recovered(sub256, f'best256_tflops_{N}')} in 1.5.0, and 1.5.0 with "
        f"`ANALYTICAL_GEMM_PICK=256x256x64` on {recovered(sub256, f'pick256_tflops_{N}')}. So this is a selection "
        "(ranking) regression, not a missing kernel. `ANALYTICAL_GEMM_PICK` was added in #7228 (2026-08-06) and is not "
        "in 1.4.1; on the released version the only runtime workaround is a per-problem "
        "`HIPBLASLT_TUNING_OVERRIDE_FILE`.",
        f"- Not all changes are regressions: 1.5.0 is at least 5% faster than 1.2.2 on {len(imp)} of 630 problems, "
        "mostly with fewer than 256 output tiles, where smaller or non-square tiles raise CU occupancy. A fix should be "
        "scoped (for example subtile rejects for saturated grids), not a revert.",
        "",
        f"### Large problems (256 or more output tiles), {len(big)}",
        "",
        *LEGEND_BIG,
        "",
        HEADER_BIG,
        *[row_big(r) for r in big],
        "",
        f"### Small problems (fewer than 256 output tiles), {len(small)}",
        "",
        "Same columns as above; the number in brackets is the TFLOPS change vs 1.2.2.",
        "",
        "<details><summary>Table</summary>",
        "",
        HEADER_SMALL,
        *[row_small(r) for r in small],
        "",
        "</details>",
        "",
        "### Smaller workspace",
        "",
        f"Same 630 problems with `HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES` = 32 MiB. The heuristic top-1 is the same "
        f"solution as at 128 MiB on {same_ws[a.BASE]} / {same_ws[a.MID]} / {same_ws[a.NEW]} of 630 problems (1.2.2 / "
        "1.4.1 / 1.5.0), i.e. the workspace limit does not change the selection, but:",
        "",
        f"- Under the 32 MiB limit the heuristic returns the same stream-K solutions with `workspaceSize = 0` (at 128 MiB "
        f"the same solution reports e.g. 85 MB for m={w['m']} n={w['n']} k={w['k']}). When `hipblasLtMatmul` is then "
        "called with the allocated 32 MiB workspace (what frameworks such as PyTorch pass), it returns "
        f"`HIPBLAS_STATUS_INTERNAL_ERROR` (6) on {len(err[B])} problems in 1.2.2, **{len(err[M])} in 1.4.1** and "
        f"{len(err[N])} in 1.5.0."
        + ("" if not err[M] else " 1.4.1 examples: " + ", ".join(
            f"m={r['m']} n={r['n']} k={r['k']} ({r[f'ws32_tile_{M}']})" for r in err[M][:5])
            + (f" and {len(err[M]) - 5} more" if len(err[M]) > 5 else "") + ".")
        + f" Passing the heuristic's reported size (0) instead, as `hipblaslt-bench` does, runs all {len(err[M])} "
        "1.4.1 cases without error, with the same kernel; `hipblaslt-bench` therefore does not show the error.",
        f"- Problems at least 20% slower than 1.2.2 at the same 32 MiB workspace: {len(ws_slowM)} in 1.4.1, "
        f"{len(ws_slowN)} in 1.5.0, e.g. "
        + "; ".join(
            f"m={r['m']} n={r['n']} k={r['k']}: {r[f'ws32_tile_{N}']} {f(r[f'ws32_tflops_{N}']):.0f} vs "
            f"{f(r[f'ws32_tflops_{B}']):.0f} ({f(r[f'ws32_delta_{N}_pct']):+.0f}%)"
            for r in ws_slowN[:4]
        )
        + ".",
        f"- m={w['m']} n={w['n']} k={w['k']}: 1.4.1 and 1.5.0 select {w[f'tile_{N}']} at both workspace sizes. At 32 "
        f"MiB: 1.2.2 {w[f'ws32_tile_{B}']} {f(w[f'ws32_tflops_{B}']):.0f}, 1.4.1 "
        + (w[f"ws32_status_{M}"].replace("error", "error status") if w[f"ws32_status_{M}"] else f"{f(w[f'ws32_tflops_{M}']):.0f}")
        + f", 1.5.0 {f(w[f'ws32_tflops_{N}']):.0f}; at 128 MiB 1.5.0 runs at {f(w[f'tflops_{N}']):.0f}. Frameworks hit "
        "this: with PyTorch's default hipBLASLt workspace, `F.linear` on this GEMM (PyTorch 2048×2304×16384) runs at "
        "~505 TFLOPS on 1.5.0 (1054 with `HIPBLASLT_WORKSPACE_SIZE=131072`), and on PyTorch 2.13 + ROCm 7.14 "
        "(hipBLASLt 1.4.1) it fails with `HIPBLAS_STATUS_INTERNAL_ERROR` and PyTorch falls back to hipBLAS.",
        "",
        "### Reproduce",
        "",
        "`hipblaslt-bench` is hipBLASLt's benchmark client (`projects/hipblaslt/clients/bench`). Each line below times a "
        "single problem and is an example; the full data set comes from the attached standalone benchmark (second block).",
        "",
    ]
    n128, s128, w128, r128, med128, _ = bc_summary(a.MID, 128)
    n128n, s128n, w128n, r128n, med128n, _ = bc_summary(a.NEW, 128)
    n32, s32, w32, r32, med32, e32 = bc_summary(a.MID, 32)
    n32n, s32n, w32n, r32n, med32n, e32n = bc_summary(a.NEW, 32)
    nb, _, wb, rb, medb, _ = bc_summary(a.MID, 128, True)
    nbn, _, wbn, rbn, medbn, _ = bc_summary(a.NEW, 128, True)
    ns, _, _, _, meds, _ = bc_summary(a.MID, 128, False)
    nsn, _, _, _, medsn, _ = bc_summary(a.NEW, 128, False)
    L += [
        "Verified with the `hipblaslt-bench` binaries from TheRock's gfx950 tests tarballs "
        "(`therock-dist-linux-gfx950-dcgpu-tests-10.0.0`, `...-tests-10.2.0a20260929`), using these exact arguments on "
        f"every regressed problem. At 128 MiB it selects the same kernel as the attached benchmark on {s128}/{n128} "
        f"(1.4.1) and {s128n}/{n128n} (1.5.0) problems. On the {nb} large problems its TFLOPS are within 3% on "
        f"{wb}/{rb} and {wbn}/{rbn} (median ratio {medb:.3f} / {medbn:.3f}); on the {ns} small problems (tens of µs "
        f"per call) it reports lower TFLOPS (median ratio {meds:.3f} / {medsn:.3f}) because it times each call "
        "individually, while the attached benchmark times 100 back-to-back calls. At 32 MiB: same kernel on "
        f"{s32}/{n32} and {s32n}/{n32n}, "
        f"and the 1.5.0 slowdowns reproduce (median TFLOPS ratio {med32n:.2f}). The 1.4.1 `INTERNAL_ERROR` needs "
        "the allocated workspace size passed to `hipblasLtMatmul` (see above), which `hipblaslt-bench` does not do; "
        "reproduce it with the attached benchmark (`HBL_WORKSPACE_KB=32768`).",
        "",
        "```bash",
    ]
    rep = [r for r in big if is256(r)][:2] + small[:1]
    for r in rep:
        t = "f32_r" if r["dtype_d"] == "f32" else "bf16_r"
        L.append(
            f"hipblaslt-bench -m {r['m']} -n {r['n']} -k {r['k']} --transA T --transB N "
            f"--lda {r['k']} --ldb {r['k']} --ldc {r['m']} --ldd {r['m']} "
            f"--a_type bf16_r --b_type bf16_r --c_type {t} --d_type {t} --compute_type f32_r "
            "--alpha 1 --beta 0 -j 20 -i 100 --print_kernel_info"
        )
    L += [
        f"# 32 MiB workspace (m={w['m']} n={w['n']} k={w['k']}):",
        f"hipblaslt-bench -m {w['m']} -n {w['n']} -k {w['k']} --transA T --transB N --lda {w['k']} --ldb {w['k']} "
        f"--ldc {w['m']} --ldd {w['m']} --a_type bf16_r --b_type bf16_r --c_type bf16_r --d_type bf16_r "
        "--compute_type f32_r --alpha 1 --beta 0 -j 20 -i 100 --workspace 33554432 --print_kernel_info",
        "# Workaround check (1.5.0 / develop only): restrict Origami's ranking to the 256x256 macro-tile",
        f"ANALYTICAL_GEMM_PICK=256x256x64 hipblaslt-bench -m {rep[0]['m']} -n {rep[0]['n']} -k {rep[0]['k']} "
        f"--transA T --transB N --lda {rep[0]['k']} --ldb {rep[0]['k']} --ldc {rep[0]['m']} --ldd {rep[0]['m']} "
        "--a_type bf16_r --b_type bf16_r --c_type bf16_r --d_type bf16_r --compute_type f32_r --alpha 1 --beta 0 "
        "-j 20 -i 100 --print_kernel_info",
        "```",
        "",
        *issue_lines_output(),
        "Full data set with the attached benchmark (`shapes_all.txt`: all 630 problems as `m n k dtypeD`; "
        "`regress_shapes.txt`: the regressed ones):",
        "",
        "```bash",
        "hipcc -O2 -std=c++17 --offload-arch=gfx950 hbl_bench.cpp -lhipblaslt -o hbl_bench",
        "./hbl_bench shapes_all.txt heuristic                                   # heuristic top-1, 128 MiB",
        "HBL_WORKSPACE_KB=32768 ./hbl_bench shapes_all.txt heuristic            # heuristic top-1, 32 MiB",
        "./hbl_bench regress_shapes.txt best256                                 # fastest MT256x256x64 solution",
        "ANALYTICAL_GEMM_PICK=256x256x64 ./hbl_bench regress_shapes.txt heuristic",
        "```",
        "",
        "### What I checked in rocm-libraries (`develop` 55e6a02)",
        "",
        "- `shared/origami/src/origami/heuristics.cpp` (HEURISTIC 3) rejects gfx950 BF16 TN subtile kernels only for "
        "`K < 512` (`key.subtile = true; key.max_k = 511`). Every regressed large problem where 1.2.2 selects "
        f"MT256x256 has k ≥ {min(int(r['k']) for r in big_sub)}.",
        "- These problems miss the Equality table and fall through to the Origami Prediction library "
        "(`Logic/asm_full/gfx950/gfx950/Origami/gfx950_Cijk_Alik_Bljk_BBS_BH_BiasSB_HAS_SAV_UserArgs.yaml`, 56 "
        "`UseSubtileImpl: true` solutions), where `rank_configs` ranks the subtile tiles above MT256x256.",
        "- No runtime flag disables subtile solutions or restores the 1.2.2 selection. `ANALYTICAL_GEMM_PICK` (develop "
        "only, #7228) forces one macro-tile globally; `HIPBLASLT_TUNING_OVERRIDE_FILE` pins per problem; "
        "`ANALYTICAL_GEMM_HEURISTICS=0` only removes the `K < 512` reject.",
        "- CHANGELOG 1.5.0 lists gfx950 Origami improvements and a subtile out-of-bounds fix; no known issue for this. No "
        "commit after #8626 (2026-06-19) changes subtile ranking. #8648 (\"Refine gfx950 BF16 TN subtile reject "
        "heuristic\") said the `K < 512` rule was \"too permissive\" and proposed rejects to \"recover the regressions\", "
        "but was closed unmerged (\"Not needed anymore\"). None of these problems are in "
        "`shared/origami/python/tests/baselines/rankings/gfx950.yaml`.",
        "",
        "**Attachments:** `hbl_pure_results.csv` (all 630 problems: selection and TFLOPS per version at 128 MiB and "
        "32 MiB, best MT256x256 and `ANALYTICAL_GEMM_PICK` results for the regressed problems, full kernel names, model "
        "tags), `bench_check.csv` (the `hipblaslt-bench` cross-check), `hbl_bench.cpp`, `shapes_all.txt`, "
        "`regress_shapes.txt`.",
    ]
    with open(os.path.join(HERE, "hipblaslt_issue.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"wrote hipblaslt_issue.md: {len(big)} large, {len(small)} small regressions")


if __name__ == "__main__":
    main()
