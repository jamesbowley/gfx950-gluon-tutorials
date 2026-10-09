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
"""Every v20 stream-K tail configuration over a grid of leftover tiles S, k-tiles and full waves.

Run it through run.sh, which sets the stack and keeps it running after a disconnect:

    experiments/streamk_split_rule/run.sh start          # main, repeat, maps, fit
    experiments/streamk_split_rule/run.sh status
    experiments/streamk_split_rule/run.sh start validate # off-grid shapes, after the rule exists

Modes (python sweep.py MODE):
- main: every configuration of every grid shape, once. results/main.jsonl.
- repeat: each shape's top 3 configurations, capped and auto, in 2 more processes; us becomes
  the median of the 3.
- retime: the reduce-scatter rows timed before v20 WORKLOG entry 8's slice change that are within
  5% of their shape's best, once more on the current kernel (results/main.pre_lane_order.jsonl
  keeps the old file). Then repeat, maps and fit as usual.
- retime2: the second pass (retime2_selection), after the slice turned out 7-16% faster without
  a full wave.
- install: writes results/rule.json's table into v20's SPLIT_RULE_TABLE, checks the kernel's
  split_rule_policy against it on every grid shape, and drops the off-grid split_rule rows so
  that validate re-times them.
- retime2: the second pass (retime2_selection), after the slice turned out 7-16% faster without
  a full wave.
- install: writes results/rule.json's table into v20's SPLIT_RULE_TABLE, checks the kernel's
  split_rule_policy against it on every grid shape, and drops the off-grid split_rule rows so
  that validate re-times them.
- maps: winner, best-n and margin maps per wave count (results/maps.md, results/*.png).
- fit: candidate rules, cross-validated on the grid (results/fit.md, results/rule.json).
- validate: the off-grid shapes (results/validate.jsonl), every row to 3 processes, then the
  score of every rule there (results/validate.md).
- realmodel: the 630 real-model shapes through experiments/aiter_lixun_shapes/run_v20_vs_aiter.py
  (--v20-only, with the new rule as a column), then their score (results/realmodel.md).
- status: progress of the current stage.

Timing is streamk_version_tables' cold rocprof (median kernel time of 1000 launches) over a
2048 MB rotating buffer, on HIP device 0 only. Before every case the GPU must read idle.
"""

import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.environ.get("SPLIT_RULE_RESULTS", os.path.join(HERE, "results"))
TMP = os.path.join(RESULTS, "tmp")
# A comma-separated subset of shapes, for smoke tests.
ONLY = set(filter(None, os.environ.get("SPLIT_RULE_SHAPES", "").split(",")))

_spec = importlib.util.spec_from_file_location(
    "streamk_version_tables_sweep", os.path.join(REPO, "experiments", "streamk_version_tables", "sweep.py"))
vt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vt)

NUM_CUS = 256
ROTATING_MB = 2048
GPU_BDF = "0000:75:00.0"
GPU_PCI_BUS = 0x75

S_GRID = (1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 48, 64, 96, 128)
K_GRID = (16, 20, 24, 28, 32, 40, 48, 64, 96, 128, 192, 256)
WAVES = (1, 0, 4)
RS_N = (2, 3, 4, 5, 6, 8, 12, 16, 24, 32)
OWNER_N = (2, 3, 4, 5, 6, 8)
# capped, auto and the default (STREAMK_POLICY unset, which is capped) are timed in the same
# conditions; default against capped also measures process-to-process noise.
BASELINES = ("capped", "auto", "default")


def factor(tiles):
    """(M, N), multiples of 256, with M * N / 256**2 = tiles and the aspect ratio closest to 1."""
    tm = min((a for a in range(1, tiles + 1) if tiles % a == 0), key=lambda a: abs(math.log(a * a / tiles)))
    return 256 * tm, 256 * (tiles // tm)


# A wave class takes the first wave count, in this order, whose tiles factor into an aspect ratio
# of at most 2.5 (else the squarest), so that prime-ish tile counts such as 257 do not turn into
# 256 x 65792 shapes whose L2 behaviour has nothing to do with S.
WAVE_CHOICES = {0: (0,), 1: (1, 2, 3), 4: (4, 5, 6, 7)}


def aspect(M, N):
    return max(M, N) / min(M, N)


def grid_shape(wclass, S, k):
    """With full waves, the most square factoring of 256 * waves + S tiles. Without, the skinny
    real-model kind: M = 512 (256 x 256 for S = 1)."""
    if wclass == 0:
        M, N = (256, 256) if S == 1 else (512, 128 * S)
        return M, N, 64 * k
    options = [factor(NUM_CUS * w + S) for w in WAVE_CHOICES[wclass]]
    M, N = next((mn for mn in options if aspect(*mn) <= 2.5), min(options, key=lambda mn: aspect(*mn)))
    return M, N, 64 * k


def env_of(config, n, S):
    if config in ("rs", "owner"):
        return {"STREAMK_POLICY": "one_tile", "STREAMK_FIXUP": config, "STREAMK_NUM_PROGRAMS": str(S * n)}
    if config == "default":
        return {}
    return {"STREAMK_POLICY": config}


def configs_of(g):
    S, k, w = g["S"], g["k_tiles"], g["waves"]
    out = [("dp", None)] + ([("two_tile", None)] if w >= 1 else [])
    out += [("rs", n) for n in RS_N if S * n <= NUM_CUS and n <= k // 2]
    out += [("owner", n) for n in OWNER_N if S * n <= NUM_CUS and n <= k // 2]
    return out + [(b, None) for b in BASELINES]


def shape_name(M, N, K):
    return f"{M}x{N}x{K}"


def grid_cases():
    for wc in WAVES:
        for S in S_GRID:
            for k in K_GRID:
                M, N, K = grid_shape(wc, S, k)
                g = vt.geometry(M, N, K)
                assert g["waves"] in WAVE_CHOICES[wc] and (g["S"], g["k_tiles"]) == (S, k), (M, N, K, g)
                for config, n in configs_of(g):
                    yield dict(shape=shape_name(M, N, K), M=M, N=N, K=K, wclass=wc, waves=g["waves"], S=S, k=k,
                               aspect=round(aspect(M, N), 2), config=config, n=n)


# Off-grid validation: S and k between grid points, and other aspect ratios.
OFF_S = (3, 10, 14, 18, 36, 56, 80)
OFF_K = (18, 22, 36, 80, 160)


def offgrid_shapes():
    """About 48 shapes: each off-grid S with three of the off-grid k values (rotating), at 0 and
    1 full waves, plus a few at 4 waves; the full-wave ones use a 1:4 aspect instead of square."""
    shapes = []
    for i, S in enumerate(OFF_S):
        ks = [OFF_K[(i + j) % len(OFF_K)] for j in range(3)]
        for k in ks:
            if S % 2 == 0 or S == 1:
                shapes.append(grid_shape(0, S, k))
            else:
                shapes.append((256, 256 * S, 64 * k))
            tiles = NUM_CUS + S
            tm = min((a for a in range(1, tiles + 1) if tiles % a == 0), key=lambda a: abs(math.log(4 * a * a / tiles)))
            shapes.append((256 * tm, 256 * (tiles // tm), 64 * k))
        shapes.append((*factor(4 * NUM_CUS + S), 64 * ks[0]))
    seen, out = set(), []
    for M, N, K in shapes:
        if (M, N, K) not in seen:
            seen.add((M, N, K))
            out.append((M, N, K))
    return out


def offgrid_cases(rules):
    for M, N, K in offgrid_shapes():
        g = vt.geometry(M, N, K)
        for config in ("dp", "capped", "auto") + tuple(rules):
            yield dict(shape=shape_name(M, N, K), M=M, N=N, K=K, wclass=min(g["waves"], 4), waves=g["waves"],
                       S=g["S"], k=g["k_tiles"], aspect=round(aspect(M, N), 2), config=config, n=None)


# ---------------------------------------------------------------- running


def key(r):
    return (r["shape"], r["config"], r["n"])


def load(which):
    path = os.path.join(RESULTS, f"{which}.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def busy():
    with open(f"/sys/bus/pci/devices/{GPU_BDF}/gpu_busy_percent") as f:
        return int(f.read().strip())


def wait_idle():
    """The reading right after our own launch can be non-zero, so wait half a second first; if it
    is still busy, wait for 3 zero readings in a row. Returns the first reading."""
    time.sleep(0.5)
    first = busy()
    zeros, waited = (3, 0) if first == 0 else (0, 0)
    while zeros < 3:
        u = busy()
        zeros = zeros + 1 if u == 0 else 0
        if u:
            waited += 1
            if waited % 30 == 1:
                print(f"GPU {GPU_BDF} busy ({u}%), waiting", flush=True)
        time.sleep(1)
    return first


def check_gpu():
    """HIP device 0 must be the PCI bus 0x75 GPU (in a subprocess, so this one holds no context)."""
    assert os.environ.get("HIP_VISIBLE_DEVICES") == "0", "run on HIP_VISIBLE_DEVICES=0 (run.sh sets it)"
    out = subprocess.run([sys.executable, "-c", "import torch;p=torch.cuda.get_device_properties(0);"
                          "print(torch.cuda.device_count(), p.pci_bus_id)"], capture_output=True, text=True)
    count, bus = (int(x) for x in out.stdout.split())
    assert count == 1 and bus == GPU_PCI_BUS, f"HIP device 0 is on PCI bus {bus:#x} ({count} devices)"


def base_env():
    env = {k: x for k, x in os.environ.items() if k not in vt.CLEARED}
    env.setdefault("LLVM_PASS_PLUGIN_PATH", os.path.join(REPO, "plugins", "llir_scheduler", "libLlirSched.so"))
    env.setdefault("TRITON_AMDGCNAS_PLUGIN", "1")
    return env


def time_case(c, env):
    """(us, correct, error, busy) of one cold rocprof run; its rocprof directory is removed after."""
    reading = wait_idle()
    tempfile.tempdir = TMP
    os.makedirs(TMP, exist_ok=True)
    try:
        us, correct, error = vt.cold(dict(v=20, M=c["M"], N=c["N"], K=c["K"]),
                                     {**env, **env_of(c["config"], c["n"], c["S"])}, ROTATING_MB)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    return us, correct, error, reading


def tflops(c, us):
    return None if us is None else 2 * c["M"] * c["N"] * c["K"] / (us * 1e-6) / 1e12


def sweep(which, cases):
    check_gpu()
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"{which}.jsonl")
    done = {key(r) for r in load(which)}
    cases = [c for c in cases if not ONLY or c["shape"] in ONLY]
    todo = [c for c in cases if key(c) not in done]
    print(f"{which}: {len(cases)} cases, {len(todo)} to run", flush=True)
    env = base_env()
    for i, c in enumerate(todo):
        us, correct, error, reading = time_case(c, env)
        row = {**c, "us": us, "tflops": tflops(c, us), "correct": correct, "error": error, "busy": reading,
               "t": time.time()}
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")
        t = "-" if us is None else f"{row['tflops']:.0f}"
        print(f"[{i + 1}/{len(todo)}] {c['shape']} {c['config']} n={c['n']}: {t} TFLOPS, correct={correct}"
              + (f", busy {reading}%" if reading else "") + (f", {error}" if error else ""), flush=True)


def good(r):
    return r is not None and r.get("tflops") is not None and r.get("correct") is True


def repeat_selection(rows):
    """Per shape: its top 3 configurations by TFLOPS, and capped and auto."""
    by_shape = {}
    for r in rows:
        by_shape.setdefault(r["shape"], []).append(r)
    chosen = set()
    for rs in by_shape.values():
        ok = sorted((r for r in rs if good(r)), key=lambda r: -r["tflops"])
        chosen |= {key(r) for r in ok[:3]}
        chosen |= {key(r) for r in ok if r["config"] in ("capped", "auto")}
    return chosen


def repeat(which, runs, everything=False):
    """Time the chosen rows again until they have `runs` timings; us becomes their median. The
    main pass repeats repeat_selection's rows, the validation set every row."""
    check_gpu()
    path = os.path.join(RESULTS, f"{which}.jsonl")
    rows = load(which)
    chosen = {key(r) for r in rows if good(r)} if everything else repeat_selection(rows)
    todo = [r for r in rows if key(r) in chosen and len(r.get("us_runs", [r["us"]])) < runs]
    print(f"repeat {which}: {len(todo)} cases to {runs} runs", flush=True)
    env = base_env()
    for i, r in enumerate(todo):
        us_runs = r.get("us_runs", [r["us"]])
        while len(us_runs) < runs:
            us, correct, error, reading = time_case(r, env)
            if us is None:
                r["error"] = error
                break
            us_runs.append(us)
            r["correct"] = r["correct"] and correct
        r["us_runs"] = us_runs
        r["us"] = statistics.median(us_runs)
        r["tflops"] = tflops(r, r["us"])
        with open(path + ".tmp", "w") as f:
            f.writelines(json.dumps(x) + "\n" for x in rows)
        os.replace(path + ".tmp", path)
        spread = 100 * (max(us_runs) / min(us_runs) - 1)
        print(f"[{i + 1}/{len(todo)}] {r['shape']} {r['config']} n={r['n']}: {r['tflops']:.0f} TFLOPS, "
              f"spread {spread:.1f}%", flush=True)


def uses_reduce_scatter(r, policies):
    """Whether a grid row runs the reduce-scatter ending (rs, or a rule that picks it)."""
    if r["config"] == "rs":
        return True
    tiles = (r["M"] // 256) * (r["N"] // 256)
    streamk_policy, capped_policy = policies
    if r["config"] in ("capped", "default"):
        policy, fixup, _ = capped_policy(tiles, r["k"], NUM_CUS)
    elif r["config"] == "auto":
        policy, fixup = streamk_policy(tiles, r["k"], NUM_CUS, NUM_CUS)
    else:
        return False
    return policy == "one_tile" and fixup == "rs"


RETIME_MARGIN = 0.95


def retime_selection(rows):
    """Rows timed with the old reduce_scatter_slice (v20 WORKLOG entry 8) that could change a
    shape's winner: reduce-scatter rows within 5% of the shape's best (the new slice is 1-3%
    faster), without default, which duplicates capped."""
    policies = vt.v20_host_functions("streamk_policy", "capped_policy")
    best = {}
    for r in rows:
        if good(r):
            best[r["shape"]] = max(best.get(r["shape"], 0), r["tflops"])
    return {key(r) for r in rows if r["config"] != "default" and good(r) and uses_reduce_scatter(r, policies)
            and r["tflops"] >= RETIME_MARGIN * best[r["shape"]]}


def retime(which):
    """Time the selected rows again on the current kernel, once, replacing their timing and
    dropping their old repeats (the repeat stage then re-times the ones in a shape's top 3).
    The first call backs up the file to <which>.pre_lane_order.jsonl."""
    check_gpu()
    path = os.path.join(RESULTS, f"{which}.jsonl")
    backup = os.path.join(RESULTS, f"{which}.pre_lane_order.jsonl")
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
    rows = load(which)
    # Selected from the backup, so that a restart picks the same rows.
    with open(backup) as f:
        chosen = retime_selection([json.loads(line) for line in f if line.strip()])
    todo = [r for r in rows if key(r) in chosen and not r.get("retimed")]
    print(f"retime {which}: {len(chosen)} rows selected, {len(todo)} to run", flush=True)
    env = base_env()
    for i, r in enumerate(todo):
        us, correct, error, reading = time_case(r, env)
        r.update(us=us, tflops=tflops(r, us), correct=correct, error=error, busy=reading, t_retime=time.time(),
                 retimed=True, us_old=r["us"])
        r.pop("us_runs", None)
        with open(path + ".tmp", "w") as f:
            f.writelines(json.dumps(x) + "\n" for x in rows)
        os.replace(path + ".tmp", path)
        t = "-" if us is None else f"{r['tflops']:.0f}"
        print(f"[{i + 1}/{len(todo)}] {r['shape']} {r['config']} n={r['n']}: {t} TFLOPS "
              f"({pct_str(us, r['us_old'])} time), correct={correct}" + (f", {error}" if error else ""), flush=True)


def retime2_selection(rows):
    """The second retime pass. The new slice gained 7-16% on shapes with no full wave, not the
    1-3% the first pass assumed, so: every remaining reduce-scatter row without a full wave, and
    elsewhere the remaining ones within the shape's largest measured gain plus 2% of its best."""
    policies = vt.v20_host_functions("streamk_policy", "capped_policy")
    by = {}
    for r in rows:
        by.setdefault(r["shape"], []).append(r)
    chosen = set()
    for rs in by.values():
        gains = [1 - r["us"] / r["us_old"] for r in rs if r.get("retimed") and r["us"] and r.get("us_old")]
        margin = 1 + max(gains, default=0.03) + 0.02
        best = max((r["tflops"] for r in rs if good(r)), default=0)
        for r in rs:
            if r.get("retimed") or r["config"] == "default" or not good(r) or not uses_reduce_scatter(r, policies):
                continue
            if r["wclass"] == 0 or r["tflops"] * margin >= best:
                chosen.add(key(r))
    return chosen


def retime2(which):
    """Like retime, for retime2_selection. The selection is saved on the first call, so that a
    restart picks the same rows."""
    check_gpu()
    path = os.path.join(RESULTS, f"{which}.jsonl")
    keys_path = os.path.join(RESULTS, "retime2_keys.json")
    rows = load(which)
    if not os.path.exists(keys_path):
        with open(keys_path, "w") as f:
            json.dump(sorted(list(k) for k in retime2_selection(rows)), f)
    with open(keys_path) as f:
        chosen = {tuple(k) for k in json.load(f)}
    todo = [r for r in rows if key(r) in chosen and not r.get("retimed")]
    print(f"retime2 {which}: {len(chosen)} rows selected, {len(todo)} to run", flush=True)
    env = base_env()
    for i, r in enumerate(todo):
        us, correct, error, reading = time_case(r, env)
        r.update(us=us, tflops=tflops(r, us), correct=correct, error=error, busy=reading, t_retime=time.time(),
                 retimed=True, retime_pass=2, us_old=r["us"])
        r.pop("us_runs", None)
        with open(path + ".tmp", "w") as f:
            f.writelines(json.dumps(x) + "\n" for x in rows)
        os.replace(path + ".tmp", path)
        t = "-" if us is None else f"{r['tflops']:.0f}"
        print(f"[{i + 1}/{len(todo)}] {r['shape']} {r['config']} n={r['n']}: {t} TFLOPS "
              f"({pct_str(us, r['us_old'])} time), correct={correct}" + (f", {error}" if error else ""), flush=True)


def retime2_selection(rows):
    """The second retime pass. The new slice gained 7-16% on shapes with no full wave, not the
    1-3% the first pass assumed, so: every remaining reduce-scatter row without a full wave, and
    elsewhere the remaining ones within the shape's largest measured gain plus 2% of its best."""
    policies = vt.v20_host_functions("streamk_policy", "capped_policy")
    by = {}
    for r in rows:
        by.setdefault(r["shape"], []).append(r)
    chosen = set()
    for rs in by.values():
        gains = [1 - r["us"] / r["us_old"] for r in rs if r.get("retimed") and r["us"] and r.get("us_old")]
        margin = 1 + max(gains, default=0.03) + 0.02
        best = max((r["tflops"] for r in rs if good(r)), default=0)
        for r in rs:
            if r.get("retimed") or r["config"] == "default" or not good(r) or not uses_reduce_scatter(r, policies):
                continue
            if r["wclass"] == 0 or r["tflops"] * margin >= best:
                chosen.add(key(r))
    return chosen


def retime2(which):
    """Like retime, for retime2_selection. The selection is saved on the first call, so that a
    restart picks the same rows."""
    check_gpu()
    path = os.path.join(RESULTS, f"{which}.jsonl")
    keys_path = os.path.join(RESULTS, "retime2_keys.json")
    rows = load(which)
    if not os.path.exists(keys_path):
        with open(keys_path, "w") as f:
            json.dump(sorted(list(k) for k in retime2_selection(rows)), f)
    with open(keys_path) as f:
        chosen = {tuple(k) for k in json.load(f)}
    todo = [r for r in rows if key(r) in chosen and not r.get("retimed")]
    print(f"retime2 {which}: {len(chosen)} rows selected, {len(todo)} to run", flush=True)
    env = base_env()
    for i, r in enumerate(todo):
        us, correct, error, reading = time_case(r, env)
        r.update(us=us, tflops=tflops(r, us), correct=correct, error=error, busy=reading, t_retime=time.time(),
                 retimed=True, retime_pass=2, us_old=r["us"])
        r.pop("us_runs", None)
        with open(path + ".tmp", "w") as f:
            f.writelines(json.dumps(x) + "\n" for x in rows)
        os.replace(path + ".tmp", path)
        t = "-" if us is None else f"{r['tflops']:.0f}"
        print(f"[{i + 1}/{len(todo)}] {r['shape']} {r['config']} n={r['n']}: {t} TFLOPS "
              f"({pct_str(us, r['us_old'])} time), correct={correct}" + (f", {error}" if error else ""), flush=True)


def pct_str(new, old):
    return "-" if new is None or not old else f"{100 * (new / old - 1):+.1f}%"


def status():
    stage_path = os.path.join(RESULTS, "stage")
    stage = open(stage_path).read().strip() if os.path.exists(stage_path) else "-"
    print(f"stage: {stage}")
    for which, total in (("main", sum(1 for c in grid_cases() if not ONLY or c["shape"] in ONLY)), ("validate", None)):
        rows = load(which)
        if not rows and total is None:
            continue
        line = f"{which}: {len(rows)}" + (f" of {total}" if total else "") + " cases"
        ts = [r["t"] for r in rows if "t" in r]
        if len(ts) > 20 and total:
            rate = (ts[-1] - ts[-21]) / 20
            line += f", {rate:.1f} s per case recently, ETA {(total - len(rows)) * rate / 3600:.1f} h"
        print(line)
    rows = load("main")
    backup = os.path.join(RESULTS, "main.pre_lane_order.jsonl")
    if os.path.exists(backup):
        with open(backup) as f:
            n_sel = len(retime_selection([json.loads(line) for line in f if line.strip()]))
        retimed = [r for r in rows if r.get("retimed") and r.get("retime_pass", 1) == 1]
        line = f"retime: {len(retimed)} of {n_sel} rows"
        if len(retimed) > 20:
            ts = sorted(r["t_retime"] for r in retimed)
            rate = (ts[-1] - ts[-21]) / 20
            line += f", {rate:.1f} s per row recently, ETA {(n_sel - len(retimed)) * rate / 3600:.1f} h"
        print(line)
    keys_path = os.path.join(RESULTS, "retime2_keys.json")
    if os.path.exists(keys_path):
        with open(keys_path) as f:
            keys2 = {tuple(k) for k in json.load(f)}
        done2 = [r for r in rows if key(r) in keys2 and r.get("retime_pass") == 2]
        line = f"retime2: {len(done2)} of {len(keys2)} rows"
        if len(done2) > 20:
            ts = sorted(r["t_retime"] for r in done2)
            rate = (ts[-1] - ts[-21]) / 20
            line += f", {rate:.1f} s per row recently, ETA {(len(keys2) - len(done2)) * rate / 3600:.1f} h"
        print(line)
    if rows and any("us_runs" in r for r in rows):
        chosen = repeat_selection(rows)
        n_done = sum(1 for r in rows if key(r) in chosen and len(r.get("us_runs", [])) >= 3)
        print(f"repeat: {n_done} of {len(chosen)} cases at 3 runs")
    bad = [r for r in rows if not good(r)]
    print(f"failed or wrong so far: {len(bad)}")
    for r in bad[-5:]:
        print(f"  {r['shape']} {r['config']} n={r['n']}: correct={r['correct']}, {r['error']}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "main":
        sweep("main", list(grid_cases()))
    elif mode == "repeat":
        repeat("main", 3)
    elif mode == "retime":
        retime("main")
    elif mode == "retime2":
        retime2("main")
    elif mode == "retime2":
        retime2("main")
    elif mode == "status":
        status()
    elif mode == "count":
        cases = list(grid_cases())
        print(len(cases), "cases,", len({c["shape"] for c in cases}), "shapes")
    elif mode in ("maps", "fit", "install", "validate", "realmodel"):
        import analysis  # noqa: E402  (results analysis lives next to this file)

        getattr(analysis, mode)()
    else:
        sys.exit(f"unknown mode {mode}")
