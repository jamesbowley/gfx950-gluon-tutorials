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
"""TFLOPS and L2 hit rate of v9-v20 (every v20 policy) on the stream-K test shapes.

    export PYTHONPATH=<repo>/scripts/triton_deepbind_shim:<triton_gfx950-tutorial-v2.2>/python
    HIP_VISIBLE_DEVICES=0 python experiments/streamk_version_tables/sweep.py main    # all versions
    HIP_VISIBLE_DEVICES=0 python experiments/streamk_version_tables/sweep.py split   # split count n
    HIP_VISIBLE_DEVICES=0 python experiments/streamk_version_tables/sweep.py repeat main 3  # median of 3
    python experiments/streamk_version_tables/sweep.py report                        # results/tables.md

- TFLOPS: `rocprofv3 --kernel-trace` over `bench.py --rocprof` (1000 launches over a rotating
  512 MB of inputs), median kernel time. bench.py checks the result against torch first; the
  row records whether it matched.
- L2: `rocprofv3 --pmc TCC_HIT_sum TCC_MISS_sum TCC_EA0_RDREQ_sum` over 10 launches on the same
  inputs (warm), median of launches 3-10 (as rocprof_check.py).
- split: STREAMK_NUM_PROGRAMS = S * n for the one-tile configurations, n programs per leftover
  tile (v15 and v16 cut end to end, so each program gets 1/n of a tile).

Rows go to results/<main|split>.jsonl, one per case; a rerun skips the cases already there.
The LLVM scheduler plugin and the amdgcnas hook are switched on here; Triton v2.2 has to be
on PYTHONPATH.
"""

import json
import os
import signal
import statistics
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
A16 = os.path.join(REPO, "kernels", "gemm", "intra_wave", "a16w16")
sys.path.insert(0, os.path.join(A16, "v20_streamk_reduce_scatter"))
import rocprof_check  # noqa: E402

RESULTS = os.path.join(HERE, "results")
TIMEOUT_S = 900
NUM_CUS = 256

# (M, N, K, class). Every dimension is a multiple of the 256x256x64 tile.
SHAPES = [
    (4096, 4096, 8192, "compute"),     # S = 0, 1 wave
    (8192, 8192, 8192, "compute"),     # S = 0, 4 waves
    (1536, 11008, 8192, "compute"),    # S = 2
    (3328, 5120, 8192, "compute"),     # S = 4
    (4352, 4096, 4096, "compute"),     # S = 16
    (4352, 4096, 8192, "compute"),     # S = 16
    (4352, 4096, 16384, "compute"),    # S = 16
    (4096, 16640, 8192, "compute"),    # S = 16, 4 waves
    (4096, 4608, 8192, "compute"),     # S = 32
    (8192, 8448, 8192, "compute"),     # S = 32, 4 waves
    (4352, 4352, 8192, "compute"),     # S = 33
    (3584, 5376, 8192, "compute"),     # S = 38
    (4096, 5120, 8192, "compute"),     # S = 64
    (4096, 6144, 8192, "compute"),     # S = 128
    (4096, 7168, 8192, "compute"),     # S = 192
    (4096, 7168, 32768, "compute"),    # S = 192, where spread wins
    (8192, 7936, 8192, "compute"),     # S = 224, 3 waves
    (4096, 7936, 8192, "compute"),     # S = 240
    (1792, 18688, 8192, "compute"),    # S = 255
    (3840, 4096, 8192, "compute"),     # 240 tiles, no full wave
    (4352, 4096, 1024, "short K"),     # S = 16, 16 k-tiles
    (4352, 4096, 2048, "short K"),     # S = 16
    (3328, 5120, 2048, "short K"),     # S = 4
    (4096, 5120, 2048, "short K"),     # S = 64
    (4096, 6144, 2048, "short K"),     # S = 128
    (4096, 7168, 1024, "short K"),     # S = 192
    (256, 65792, 8192, "skinny"),      # 257 tiles, S = 1
    (256, 67328, 4096, "skinny"),      # 263 tiles, S = 7
    (256, 32768, 8192, "skinny"),      # 128 tiles, no full wave
    (256, 49152, 8192, "skinny"),      # 192 tiles, no full wave
]

V20 = {
    "v20 auto": {},
    "v20 rs": {"STREAMK_POLICY": "one_tile", "STREAMK_FIXUP": "rs"},
    "v20 owner": {"STREAMK_POLICY": "one_tile", "STREAMK_FIXUP": "owner"},
    "v20 two_tile": {"STREAMK_POLICY": "two_tile"},
    "v20 spread": {"STREAMK_POLICY": "spread"},
    "v20 dp": {"STREAMK_POLICY": "dp"},
    "v20 capped": {"STREAMK_POLICY": "capped"},
}
MAIN = {f"v{v}": (v, {}) for v in range(9, 20)} | {k: (20, env) for k, env in V20.items()}
ONE_TILE = ["v15", "v16", "v17", "v18", "v20 owner", "v20 rs"]
N_SPLIT = (1, 2, 3, 4, 6, 8, 16)
CLEARED = ("STREAMK_NUM_PROGRAMS", "STREAMK_POLICY", "STREAMK_FIXUP", "STREAMK_BALANCE_XCDS",
           "PERSISTENT_TILE_ORDER")


def geometry(M, N, K):
    tiles = (M // 256) * (N // 256)
    return {"tiles": tiles, "waves": tiles // NUM_CUS, "S": tiles % NUM_CUS, "k_tiles": K // 64}


def shape_name(M, N, K):
    return f"{M}x{N}x{K}"


def splits(S, K):
    """n with a smaller split than the default (S * n < 256) and at least one k-step pair each."""
    return [n for n in N_SPLIT if S * n < NUM_CUS and n <= K // 128]


def cases_main():
    for M, N, K, cls in SHAPES:
        for config, (v, env) in MAIN.items():
            yield dict(M=M, N=N, K=K, cls=cls, config=config, n="default", snp=None, v=v, env=env)


def cases_split():
    for M, N, K, cls in SHAPES:
        S = geometry(M, N, K)["S"]
        if not (1 <= S <= 128 and K >= 2048):
            continue
        for n in splits(S, K):
            for config in ONE_TILE:
                v, env = MAIN[config]
                yield dict(M=M, N=N, K=K, cls=cls, config=config, n=n, snp=S * n, v=v, env=env)


def run(cmd, env):
    """(returncode, stdout); a run that exceeds TIMEOUT_S (a hung fixup wait) is killed."""
    p = subprocess.Popen(cmd, cwd=A16, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, start_new_session=True)
    try:
        out, _ = p.communicate(timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, _ = p.communicate()
        return "timeout", out
    return p.returncode, out


def cold(c, env):
    name = rocprof_check.NAMES[c["v"]]
    d = tempfile.mkdtemp(prefix="rp_")
    rc, out = run(["rocprofv3", "--kernel-trace", "--kernel-include-regex", name, "-d", d, "-o", "run",
                   "--output-format", "csv", "--", sys.executable, "bench.py", "--version", str(c["v"]),
                   "--M", str(c["M"]), "--N", str(c["N"]), "--K", str(c["K"]), "--dtype", "bf16",
                   "--rocprof"], env)
    correct = True if "Triton and Torch match" in out else False if "Triton and Torch differ" in out else None
    rows = [r for r in rocprof_check.kernel_rows(d, name) if "End_Timestamp" in r]
    us = [(int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3 for r in rows]
    error = None if rc == 0 and us else f"rc={rc}: " + " / ".join(out.strip().splitlines()[-3:])
    return (statistics.median(us) if us else None), correct, error


def l2(c, env):
    name = rocprof_check.NAMES[c["v"]]
    d = tempfile.mkdtemp(prefix="rp_")
    script = os.path.join(d, "pmc.py")
    with open(script, "w") as f:
        f.write(rocprof_check.PMC_SCRIPT % (A16, c["v"], c["M"], c["N"], c["K"]))
    run(["rocprofv3", "--pmc", "TCC_HIT_sum", "TCC_MISS_sum", "TCC_EA0_RDREQ_sum", "--kernel-include-regex",
         name, "-d", d, "-o", "run", "--output-format", "csv", "--", sys.executable, script], env)
    by_disp = {}
    for r in rocprof_check.kernel_rows(d, name):
        if "Counter_Name" in r:
            by_disp.setdefault(int(r["Dispatch_Id"]), {})[r["Counter_Name"]] = float(r["Counter_Value"])
    disp = sorted(by_disp)[2:]
    if not disp:
        return None, None
    hit = statistics.median(by_disp[x]["TCC_HIT_sum"] / (by_disp[x]["TCC_HIT_sum"] + by_disp[x]["TCC_MISS_sum"])
                            for x in disp)
    reads = statistics.median(by_disp[x]["TCC_EA0_RDREQ_sum"] for x in disp)
    return hit, reads


def key(r):
    return (r["M"], r["N"], r["K"], r["config"], str(r["n"]))


def load(which):
    path = os.path.join(RESULTS, f"{which}.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]


def sweep(which, only=None):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"{which}.jsonl")
    done = {key(r) for r in load(which)}
    cases = list(cases_main() if which == "main" else cases_split())
    if only:
        cases = [c for c in cases if shape_name(c["M"], c["N"], c["K"]) in only or c["config"] in only]
    todo = [c for c in cases if key(c) not in done]
    print(f"{which}: {len(cases)} cases, {len(todo)} to run", flush=True)
    base = {k: x for k, x in os.environ.items() if k not in CLEARED}
    base.setdefault("LLVM_PASS_PLUGIN_PATH", os.path.join(REPO, "plugins", "llir_scheduler", "libLlirSched.so"))
    base.setdefault("TRITON_AMDGCNAS_PLUGIN", "1")
    for i, c in enumerate(todo):
        env = {**base, **c["env"]}
        if c["snp"] is not None:
            env["STREAMK_NUM_PROGRAMS"] = str(c["snp"])
        us, correct, error = cold(c, env)
        hit, reads = l2(c, env) if error is None else (None, None)
        M, N, K = c["M"], c["N"], c["K"]
        row = {"shape": shape_name(M, N, K), "M": M, "N": N, "K": K, "cls": c["cls"], **geometry(M, N, K),
               "config": c["config"], "n": c["n"], "snp": c["snp"], "us": us,
               "tflops": None if us is None else 2 * M * N * K / (us * 1e-6) / 1e12,
               "l2_hit": hit, "l2_reads": reads, "correct": correct, "error": error}
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")
        tflops = "-" if us is None else f"{row['tflops']:.0f}"
        l2_s = "-" if hit is None else f"{100 * hit:.1f}%"
        print(f"[{i + 1}/{len(todo)}] {row['shape']} {row['config']} n={row['n']}: {tflops} TFLOPS, "
              f"L2 {l2_s}, correct={correct}" + (f", {error}" if error else ""), flush=True)


def repeat(which, runs, only=None):
    """Time each case again in fresh processes until it has `runs` timings; us becomes their median.

    Kernel time can differ between processes by up to 15% on the skinny shapes (with identical
    settings), so one process per case is not enough to rank close configurations.
    """
    path = os.path.join(RESULTS, f"{which}.jsonl")
    rows = load(which)
    base = {k: x for k, x in os.environ.items() if k not in CLEARED}
    base.setdefault("LLVM_PASS_PLUGIN_PATH", os.path.join(REPO, "plugins", "llir_scheduler", "libLlirSched.so"))
    base.setdefault("TRITON_AMDGCNAS_PLUGIN", "1")
    todo = [r for r in rows if r["error"] is None and len(r.get("us_runs", [r["us"]])) < runs
            and (not only or r["shape"] in only or r["config"] in only)]
    print(f"repeat {which}: {len(todo)} cases to {runs} runs", flush=True)
    for i, r in enumerate(todo):
        v, env = MAIN[r["config"]]
        env = {**base, **env}
        if r["snp"] is not None:
            env["STREAMK_NUM_PROGRAMS"] = str(r["snp"])
        us_runs = r.get("us_runs", [r["us"]])
        while len(us_runs) < runs:
            us, correct, error = cold(dict(v=v, M=r["M"], N=r["N"], K=r["K"]), env)
            if us is None:
                r["error"] = error
                break
            us_runs.append(us)
            r["correct"] = r["correct"] and correct
        r["us_runs"] = us_runs
        r["us"] = statistics.median(us_runs)
        r["tflops"] = 2 * r["M"] * r["N"] * r["K"] / (r["us"] * 1e-6) / 1e12
        with open(path + ".tmp", "w") as f:
            f.writelines(json.dumps(x) + "\n" for x in rows)
        os.replace(path + ".tmp", path)
        spread = 100 * (max(us_runs) / min(us_runs) - 1)
        print(f"[{i + 1}/{len(todo)}] {r['shape']} {r['config']} n={r['n']}: {r['tflops']:.0f} TFLOPS, "
              f"spread {spread:.1f}%", flush=True)


def v20_host_functions(*names):
    """v20's host rules (streamk_policy, capped_policy), taken from its source so that the report
    needs no Triton."""
    import ast
    import math

    path = os.path.join(A16, "v20_streamk_reduce_scatter", "matmul_kernel.py")
    tree = ast.parse(open(path).read())
    fns = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name in names]
    scope = {"math": math}
    exec(compile(ast.Module(body=fns, type_ignores=[]), path, "exec"), scope)
    return [scope[name] for name in names]


def capped_label(g, capped):
    """v20 capped's pick on a shape, e.g. "one-tile rs, n = 8"; capped = capped_policy's result."""
    policy, fixup, programs = capped
    if policy != "one_tile":
        return {"dp": "data-parallel", "spread": "spread"}[policy]
    return f"one-tile {fixup}" + (f", n = {programs // g['S']}" if programs else "")


def schedule(config, g, auto, capped):
    """The work schedule a configuration runs on a shape; auto = v20's (policy, fixup) there and
    capped = capped_policy's (policy, fixup, programs)."""
    if g["S"] == 0 or config in ("v9", "v10", "v11", "v12", "v13", "v14", "v20 dp"):
        return "data-parallel"
    if config == "v20 capped":
        return capped_label(g, capped)
    policy, fixup = auto
    if config == "v20 auto":
        config = {"dp": "v20 dp", "two_tile": "v20 two_tile", "one_tile": f"v20 {fixup}"}[policy]
        return schedule(config, g, auto, capped)
    if config in ("v19", "v20 two_tile") and g["waves"] == 0:
        # Two-tile needs a full wave to borrow; without one both fall back to the tile-aligned split.
        return "one-tile owner" if config == "v19" or fixup == "owner" else "one-tile rs"
    return {"v15": "one-tile end-to-end", "v16": "one-tile end-to-end", "v17": "one-tile owner",
            "v18": "one-tile owner", "v20 owner": "one-tile owner", "v20 rs": "one-tile rs",
            "v19": "two-tile", "v20 two_tile": "two-tile", "v20 spread": "spread"}[config]


def good(r):
    return r is not None and r["tflops"] is not None and r["correct"] is True


def t_cell(r, best=None):
    if r is None:
        return ""
    if r["correct"] is False:
        return "wrong"
    if r["tflops"] is None:
        return "err"
    s = f"{r['tflops']:.0f}"
    return f"**{s}**" if r is best else s


def l2_cell(r):
    return "" if r is None or r["l2_hit"] is None else f"{100 * r['l2_hit']:.0f}%"


def pct(a, b):
    return f"{100 * (a / b - 1):+.1f}%"


def table(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + \
           ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]


def report():
    policy, capped_policy = v20_host_functions("streamk_policy", "capped_policy")
    main_rows = {(r["shape"], r["config"]): r for r in load("main")}
    split_rows = {(r["shape"], r["config"], r["n"]): r for r in load("split")}
    shapes = [(shape_name(M, N, K), M, N, K, cls) for M, N, K, cls in SHAPES
              if any((shape_name(M, N, K), c) in main_rows for c in MAIN)]
    old = [f"v{v}" for v in range(9, 20)]
    out = ["# v9-v20 on the stream-K test shapes", "",
           "bf16, no bias, v9 tile order, GPU 0. TFLOPS from the median rocprof kernel time over 1000 "
           "launches on a rotating 512 MB of inputs (cold caches); where a case was timed in several "
           "processes (`repeat`), the median of those. L2 hit rate is TCC_HIT / (HIT + MISS) "
           "over warm launches 3-10 on the same inputs. Generated by `sweep.py report`.", ""]

    best_of, best_uncapped = {}, {}
    t1 = []
    for name, M, N, K, cls in shapes:
        g = geometry(M, N, K)
        rows = [main_rows.get((name, c)) for c in MAIN]
        ok = [r for r in rows if good(r)]
        if not ok:
            continue
        best = max(ok, key=lambda r: r["tflops"])
        best_of[name] = best
        best_uncapped[name] = max((r for r in ok if r["config"] != "v20 capped"), key=lambda r: r["tflops"])
        auto = policy(g["tiles"], g["k_tiles"], NUM_CUS)
        capped = capped_policy(g["tiles"], g["k_tiles"], NUM_CUS)
        sched = {r["config"]: schedule(r["config"], g, auto, capped) for r in ok}
        others = [r for r in ok if r["config"].split()[0] != best["config"].split()[0]]
        second = max(others, key=lambda r: r["tflops"]) if others else None
        other_sched = [r for r in ok if sched[r["config"]] != sched[best["config"]]]
        next_sched = max(other_sched, key=lambda r: r["tflops"]) if other_sched else None
        v13 = main_rows.get((name, "v13"))
        spreads = [100 * (max(r["us_runs"]) / min(r["us_runs"]) - 1) for r in ok if len(r.get("us_runs", [])) > 1]
        spread = f"{statistics.median(spreads):.1f}% / {max(spreads):.1f}%" if spreads else "-"
        d = g["k_tiles"] * g["S"] / NUM_CUS
        t1.append([name, cls, g["tiles"], g["waves"], g["S"], g["k_tiles"], f"{d:g}",
                   schedule("v20 auto", g, auto, capped), capped_label(g, capped),
                   f"{best['config']} ({sched[best['config']]})",
                   f"{second['config']}, {pct(best['tflops'], second['tflops'])}" if second else "-",
                   f"{sched[next_sched['config']]} ({next_sched['config']}), "
                   f"{pct(best['tflops'], next_sched['tflops'])}" if next_sched else "-",
                   pct(best["tflops"], v13["tflops"]) if good(v13) else "-", spread])
    out += ["## 1. Shapes and winners", "",
            "d = k-tiles * S / 256. \"Runner-up\" is the fastest configuration from another version (v20's "
            "variants count as one version); it often runs the same schedule as the best. \"Next schedule\" "
            "is the fastest configuration that runs a different schedule, which is the margin the schedule "
            "itself wins by. With S = 0 every configuration is data-parallel. \"Spread\" is the difference "
            "between the slowest and fastest of a configuration's processes, median / max over the "
            "configurations; margins below it are not significant.", ""]
    out += table(["shape", "class", "tiles", "full waves", "S", "k-tiles", "d", "v20 auto picks",
                  "v20 capped picks", "best (schedule)", "runner-up, best ahead by", "next schedule, best ahead by", "best vs v13",
                  "spread"], t1)

    def grid(configs, cell, bold):
        rows = []
        for name, *_ in shapes:
            if name in best_of:
                rows.append([name] + [cell(main_rows.get((name, c)), best_of[name]) if bold
                                      else cell(main_rows.get((name, c))) for c in configs])
        return rows

    v20_cols = list(V20)
    out += ["", "## 2. TFLOPS, v9-v19", "", f"Bold is the best over all {len(MAIN)} configurations.", ""]
    out += table(["shape"] + old, grid(old, t_cell, True))
    out += ["", "## 3. TFLOPS, v20 policies", ""]
    out += table(["shape"] + [c.removeprefix("v20 ") for c in v20_cols], grid(v20_cols, t_cell, True))
    out += ["", "## 4. L2 hit rate, v9-v19", ""]
    out += table(["shape"] + old, grid(old, l2_cell, False))
    out += ["", "## 5. L2 hit rate, v20 policies", ""]
    out += table(["shape"] + [c.removeprefix("v20 ") for c in v20_cols], grid(v20_cols, l2_cell, False))

    # Split count.
    split_shapes = [s for s in shapes if any(k[0] == s[0] for k in split_rows)]
    cond_t, cond_l2 = [], []
    full = []
    for name, M, N, K, cls in split_shapes:
        g = geometry(M, N, K)
        ns = splits(g["S"], K)
        row_t, row_l2 = [name, g["S"]], [name, g["S"]]
        best_capped = None
        cols = {}
        for c in ONE_TILE:
            entries = [(n, split_rows.get((name, c, n))) for n in ns] + [("default", main_rows.get((name, c)))]
            entries = [(n, r) for n, r in entries if r is not None]
            cols[c] = entries
            dflt = main_rows.get((name, c))
            ok = [(n, r) for n, r in entries if good(r)]
            if not ok or not good(dflt):
                row_t.append(t_cell(dflt) or "-")
                row_l2.append(l2_cell(dflt) or "-")
                continue
            bn, br = max(ok, key=lambda e: e[1]["tflops"])
            capped = [(n, r) for n, r in ok if n != "default"]
            if capped:
                cn, cr = max(capped, key=lambda e: e[1]["tflops"])
                if best_capped is None or cr["tflops"] > best_capped[2]["tflops"]:
                    best_capped = (c, cn, cr)
            if bn == "default":
                row_t.append(f"{dflt['tflops']:.0f} (default)")
                row_l2.append(l2_cell(dflt))
            else:
                row_t.append(f"{dflt['tflops']:.0f} -> {br['tflops']:.0f} (n {bn}, {pct(br['tflops'], dflt['tflops'])})")
                row_l2.append(f"{l2_cell(dflt)} -> {l2_cell(br)}")
        mb = best_uncapped.get(name)
        if best_capped and mb:
            c, bn, br = best_capped
            row_t.append(f"{c} n {bn}: {pct(br['tflops'], mb['tflops'])} vs {mb['config']}")
        else:
            row_t.append("-")
        cond_t.append(row_t)
        cond_l2.append(row_l2)

        full += ["", f"### {name} (S = {g['S']}, k-tiles = {g['k_tiles']}, default split "
                 f"{NUM_CUS // g['S']}-{-(-NUM_CUS // g['S'])} ways)", ""]
        all_n = ns + ["default"]
        best_c = {c: max((r for _, r in cols[c] if good(r)), key=lambda r: r["tflops"], default=None)
                  for c in ONE_TILE}
        rows = []
        for n in all_n:
            cells = []
            for c in ONE_TILE:
                r = next((r for m, r in cols[c] if m == n), None)
                t = t_cell(r, best_c[c])
                cells.append(f"{t} / {l2_cell(r)}" if t else "")
            rows.append([n] + cells)
        full += table(["n"] + ONE_TILE, rows)

    if cond_t:
        out += ["", "## 6. Split count (STREAMK_NUM_PROGRAMS = S * n)", "",
                "TFLOPS at the default split (all 256 programs) -> at the best n in {1, 2, 3, 4, 6, 8, 16}. "
                "The last column compares the fastest capped split (n < default) of any configuration with the "
                "main-grid best other than v20 capped.", ""]
        out += table(["shape", "S"] + ONE_TILE + ["best capped vs main best"], cond_t)
        out += ["", "L2 hit rate at the default split -> at the best n:", ""]
        out += table(["shape", "S"] + ONE_TILE, cond_l2)
        out += ["", "### Every n (TFLOPS / L2 hit rate; bold is each configuration's best)"] + full

    # Builds that should match an older version.
    out += ["", "## Same-path checks", ""]
    for a, b in (("v20 owner", "v17"), ("v20 two_tile", "v19"), ("v20 dp", "v13")):
        ratios = [main_rows[(n, a)]["tflops"] / main_rows[(n, b)]["tflops"] for n, *_ in shapes
                  if good(main_rows.get((n, a))) and good(main_rows.get((n, b)))]
        if ratios:
            out.append(f"- {a} / {b}: median {statistics.median(ratios):.3f}, range "
                       f"{min(ratios):.3f}-{max(ratios):.3f} over {len(ratios)} shapes")
    bad = [r for r in list(main_rows.values()) + list(split_rows.values()) if not good(r)]
    out += ["", f"Failed or wrong cases: {len(bad)}"]
    out += [f"- {r['shape']} {r['config']} n={r['n']}: correct={r['correct']}, {r['error']}" for r in bad]

    text = "\n".join(out) + "\n"
    with open(os.path.join(RESULTS, "tables.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode in ("main", "split"):
        sweep(mode, set(sys.argv[2:]))
    elif mode == "repeat":
        repeat(sys.argv[2], int(sys.argv[3]), set(sys.argv[4:]))
    elif mode == "report":
        report()
    else:
        sys.exit(f"unknown mode {mode}")
