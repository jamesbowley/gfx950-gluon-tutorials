#!/usr/bin/env python3
"""Does v7 still stall at its 12th buffer_load per wave, and at what clock? (v8 README §4.5)

    python v7_compare.py --gpu 0 run        # K = 8192 and 16384, hot and cold, ATT at dispatches 15 and 600
    python v7_compare.py --gpu 0 report

Each run: one process launches v7 (fp16, 4096 x 4096 x K, the MFMA scheduler alone) ITERS times, under
rocprofv3 --att (one dispatch) and --kernel-trace (every dispatch). A thread in that process reads the
GPU's current shader clock from sysfs (pp_dpm_sclk) every 2 ms while the launches run. The traced
dispatch's own average clock comes from the trace's realtime.json.

cold: rotating inputs, at least 2 GB, so each launch's A and B was last read GBs earlier.
hot:  one set of inputs reused by every launch (A + B = 128 or 256 MB, inside the 256 MB memory-side cache).
"""

import argparse
import collections
import csv
import glob
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
A16 = os.path.join(REPO, "kernels", "gemm", "intra_wave", "a16w16")
OUT = os.path.join(HERE, "results", "v7")
TMP = os.path.join(HERE, "tmp", "v7")
ITERS = 1500
BUS = {0: "0000:05:00.0"}
AMDSMI = "/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/share/amd_smi"


def sysfs_dev(bus):
    for c in glob.glob("/sys/class/drm/card*/device"):
        try:
            if f"PCI_SLOT_NAME={bus}" in open(os.path.join(c, "uevent")).read():
                return c
        except OSError:
            pass
    raise SystemExit(f"no sysfs device for {bus}")


def sclk_now(dev):
    for line in open(os.path.join(dev, "pp_dpm_sclk")):
        if "*" in line:
            return int(re.search(r"(\d+)Mhz", line).group(1))
    return None


# ------------------------------------------------------------------------------------------------
# worker: runs inside rocprofv3
# ------------------------------------------------------------------------------------------------


def worker(args):
    sys.path.insert(0, A16)
    sys.path.insert(0, os.path.join(A16, "..", "..", "utils"))
    import torch
    from v7_sliceN.matmul_kernel import matmul

    sys.path.insert(0, AMDSMI)
    import amdsmi
    amdsmi.amdsmi_init()
    h = next(x for x in amdsmi.amdsmi_get_processor_handles()
             if amdsmi.amdsmi_get_gpu_device_bdf(x).lower() == BUS[args.gpu])
    M = N = 4096
    K = args.K
    per = (M * K + K * N + M * N) * 2
    copies = max(1, (2048 << 20) // per + 1) if args.cache == "cold" else 1
    sets = [(torch.randn((M, K), device="cuda", dtype=torch.float16),
             torch.randn((N, K), device="cuda", dtype=torch.float16).T,
             torch.empty((M, N), device="cuda", dtype=torch.float16)) for _ in range(copies)]
    a, b, c = sets[0]
    ref = torch.matmul(a, b)
    ok = torch.allclose(matmul(a, b, c), ref, atol=1e-1, rtol=0)
    torch.cuda.synchronize()
    samples = []
    stop = threading.Event()

    def sample():
        # current_gfxclks: the effective shader clock of each of the 8 XCDs, from the SMU's gpu_metrics.
        while not stop.is_set():
            m = amdsmi.amdsmi_get_gpu_metrics_info(h)
            samples.append((time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns(),
                            m["current_gfxclks"], m["current_socket_power"]))
            time.sleep(0.0005)

    th = threading.Thread(target=sample)
    th.start()
    time.sleep(0.05)
    t0 = time.monotonic_ns()
    for i in range(args.iters):
        a, b, c = sets[i % copies]
        matmul(a, b, c)
    torch.cuda.synchronize()
    t1 = time.monotonic_ns()
    time.sleep(0.05)
    stop.set()
    th.join()
    run = [statistics.mean(g) for _b, t, g, _p in samples if t0 <= t <= t1]
    json.dump(dict(correct=bool(ok), copies=copies, MB=copies * per >> 20, iters=args.iters, wall_s=(t1 - t0) / 1e9,
                   gfxclk_samples=len(run), gfxclk_median=statistics.median(run) if run else None,
                   samples=samples), open(args.dump, "w"))


# ------------------------------------------------------------------------------------------------
# driver
# ------------------------------------------------------------------------------------------------


def run_one(gpu, K, cache, att_iter):
    name = f"v7_K{K}_{cache}_d{att_iter}"
    d = os.path.join(TMP, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    att = os.path.join(d, "att.json")
    json.dump({"jobs": [{"kernel_include_regex": "v7_sliceN", "kernel_exclude_regex": "",
                         "kernel_iteration_range": f"[{att_iter}]", "advanced_thread_trace": True,
                         "att_target_cu": 0, "att_shader_engine_mask": "0xF", "att_simd_select": "0xF",
                         "att_buffer_size": "0x60000000"}]}, open(att, "w"))
    env = {k: v for k, v in os.environ.items() if not k.startswith("ROCPROFILER_REGISTER")}
    env.update(HIP_VISIBLE_DEVICES=str(gpu), PYTHONPATH=os.path.join(REPO, "scripts", "triton_deepbind_shim"),
               ROCPROF_ATT_LIBRARY_PATH=env.get("ROCPROF_ATT_LIBRARY_PATH", "/opt/rocm/lib/"),
               TRITON_CACHE_DIR=os.path.join(HERE, "tmp", "v7_cache"))
    env.pop("TRITON_AMDGCNAS_PLUGIN", None)
    env.pop("LLVM_PASS_PLUGIN_PATH", None)
    dump = os.path.join(d, "run.json")
    cmd = ["rocprofv3", "--att", "-i", att, "--kernel-trace", "-d", d, "-o", "run", "--output-format", "csv", "--",
           sys.executable, os.path.abspath(__file__), "--gpu", str(gpu), "worker", "--K", str(K), "--cache", cache,
           "--iters", str(ITERS), "--dump", dump]
    with open(os.path.join(d, "rocprof.log"), "w") as log:
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    ui = [u for u in glob.glob(os.path.join(d, "**", "ui_*"), recursive=True) if os.path.isdir(u)]
    if not ui:
        raise SystemExit(f"{name}: no trace, see {d}/rocprof.log")
    res = dict(name=name, K=K, cache=cache, att_iter=att_iter, ui=ui[0], host=json.load(open(dump)))
    samples = res["host"].pop("samples")
    res["kernel_us"], windows = kernel_times(d)
    res["smu"] = smu_clocks(samples, windows, att_iter)
    res["att"] = analyse(ui[0], K)
    lat = subprocess.run([sys.executable, os.path.join(REPO, "experiments", "tcp_latency", "att_latency.py"), ui[0],
                          "--name", name, "--loop", "--exact-stall", "64", "--out", OUT], capture_output=True, text=True)
    res["att_latency"] = json.load(open(os.path.join(OUT, f"att_{name}.json")))
    mf = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "process_json.py"), ui[0]], capture_output=True,
                        text=True).stdout
    m = re.search(r'"mfma efficiency": "([\d.]+)%"', mf)
    res["mfma_eff"] = float(m.group(1)) if m else None
    json.dump(res, open(os.path.join(OUT, name + ".json"), "w"), indent=1)
    return res


def clock_pass(gpu, K, cache):
    """GRBM_GUI_ACTIVE (summed over the 8 XCDs) / 8 / kernel duration, per v7 dispatch, without ATT. On the
    load-only MRE GEMM this reads 2.49 GHz where its ATT trace says 2.35 (about 6% high)."""
    name = f"v7_K{K}_{cache}_clock"
    d = os.path.join(TMP, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    env = {k: v for k, v in os.environ.items() if not k.startswith("ROCPROFILER_REGISTER")}
    env.update(HIP_VISIBLE_DEVICES=str(gpu), PYTHONPATH=os.path.join(REPO, "scripts", "triton_deepbind_shim"),
               TRITON_CACHE_DIR=os.path.join(HERE, "tmp", "v7_cache"))
    env.pop("TRITON_AMDGCNAS_PLUGIN", None)
    env.pop("LLVM_PASS_PLUGIN_PATH", None)
    dump = os.path.join(d, "run.json")
    cmd = ["rocprofv3", "--pmc", "GRBM_GUI_ACTIVE", "--kernel-trace", "--kernel-include-regex", "v7_sliceN", "-d", d,
           "-o", "run", "--output-format", "csv", "--", sys.executable, os.path.abspath(__file__), "--gpu", str(gpu),
           "worker", "--K", str(K), "--cache", cache, "--iters", str(ITERS), "--dump", dump]
    with open(os.path.join(d, "rocprof.log"), "w") as log:
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    pm = collections.defaultdict(float)
    for f in glob.glob(os.path.join(d, "**", "*counter_collection.csv"), recursive=True):
        for r in csv.DictReader(open(f)):
            pm[int(r["Dispatch_Id"])] += float(r["Counter_Value"])
    kt = {}
    for f in glob.glob(os.path.join(d, "**", "*kernel_trace.csv"), recursive=True):
        for r in csv.DictReader(open(f)):
            if "v7_sliceN" in r["Kernel_Name"]:
                kt[int(r["Dispatch_Id"])] = int(r["End_Timestamp"]) - int(r["Start_Timestamp"])
    ids = sorted(i for i in kt if i in pm)
    ghz = [pm[i] / 8 / kt[i] for i in ids]
    us = [kt[i] / 1000 for i in ids]
    return dict(n=len(ghz), ghz_at15=ghz[15], ghz_at600=ghz[600], ghz_median_after_100=statistics.median(ghz[100:]),
                us_at15=us[15], us_at600=us[600], us_median_after_100=statistics.median(us[100:]),
                ghz_first10=[round(g, 2) for g in ghz[:10]])


def kernel_times(d):
    rows = [r for f in glob.glob(os.path.join(d, "**", "*kernel_trace.csv"), recursive=True)
            for r in csv.DictReader(open(f)) if "v7_sliceN" in r["Kernel_Name"]]
    rows.sort(key=lambda r: int(r["Dispatch_Id"]))
    win = [(int(r["Start_Timestamp"]), int(r["End_Timestamp"])) for r in rows]
    us = [(e - s) / 1000 for s, e in win]
    # Dispatch 0 is bench's correctness launch; the loop's launch i is dispatch i + 1. ATT's iteration
    # index counts every v7 dispatch from 0, so the traced one is win[att_iter].
    return dict(n=len(us), first=us[:3], at15=us[15] if len(us) > 15 else None,
                at600=us[600] if len(us) > 600 else None,
                median_after_100=statistics.median(us[100:]) if len(us) > 100 else None), win


def smu_clocks(samples, windows, att_iter):
    """SMU effective gfxclk (mean of the 8 XCDs) and socket power inside kernel windows. The kernel trace's
    clock domain is whichever of BOOTTIME / MONOTONIC brackets the dispatches with samples."""
    out = {}
    for dom, col in (("boottime", 0), ("monotonic", 1)):
        ts = [s[col] for s in samples]
        if ts and ts[0] <= windows[1][0] <= ts[-1]:
            out["domain"] = dom

            def inside(lo, hi):
                return [(statistics.mean(s[2]), s[3]) for s in samples if lo <= s[col] <= hi]

            tr = inside(*windows[att_iter]) if att_iter < len(windows) else []
            steady = [x for w in windows[100:] for x in inside(*w)]
            out["traced"] = dict(n=len(tr), mhz=statistics.median([x[0] for x in tr]) if tr else None,
                                 watts=statistics.median([x[1] for x in tr]) if tr else None)
            out["steady"] = dict(n=len(steady), mhz=statistics.median([x[0] for x in steady]) if steady else None,
                                 watts=statistics.median([x[1] for x in steady]) if steady else None)
            first = [x for w in windows[1:20] for x in inside(*w)]
            out["first_20"] = dict(n=len(first), mhz=statistics.median([x[0] for x in first]) if first else None)
            return out
    out["domain"] = None
    return out


def analyse(ui, K):
    """Per buffer_load position in the loop body (code order): stall at issue, and its issue time relative to
    the iteration's first loop load (the README's point A), over the traced waves."""
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import process_json
    code = json.load(open(os.path.join(ui, "code.json")))["code"]
    info = process_json.analyze_code(code)
    lo, hi = info["loop_first_index"], info["loop_last_index"]
    text = {c[2]: c[0] for c in code}
    loads = sorted(i for i in text if lo <= i <= hi and text[i].startswith("buffer_load"))
    pos = {i: n for n, i in enumerate(loads)}
    stall = collections.defaultdict(list)
    since_a = collections.defaultdict(list)
    rt = json.load(open(os.path.join(ui, "realtime.json")))
    ghz = [(p[-1][0] - p[0][0]) / max(1, p[-1][1] - p[0][1]) / 10.0 for k, p in rt.items() if k.startswith("SE")
           and len(p) >= 2]
    for f in glob.glob(os.path.join(ui, "se*_sm*_sl*_wv*.json")):
        ins = json.load(open(f)).get("wave", {}).get("instructions") or []
        a = None
        for clock, _t, st, _d, idx in ins:
            if idx in pos:
                if pos[idx] == 0:
                    a = clock
                stall[pos[idx]].append(st)
                if a is not None:
                    since_a[pos[idx]].append(clock + st - a)
    table = []
    for p in range(len(loads)):
        s = stall[p]
        table.append(dict(pos=p + 1, n=len(s), stalled=sum(x >= 16 for x in s) / max(1, len(s)),
                          stall_p50=statistics.median(s) if s else None, stall_p90=sorted(s)[9 * len(s) // 10] if s else None,
                          stall_max=max(s) if s else None,
                          since_a_p50=statistics.median(since_a[p]) if since_a[p] else None))
    return dict(ghz=statistics.median(ghz) if ghz else None, loads_per_iter=len(loads), positions=table)


def report():
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, "v7_K*_d*.json")))]
    print("| K | cache | traced dispatch | ATT clock of traced dispatch, GHz | GRBM clock, same dispatch index / steady, GHz "
          "| SMU gfxclk steady, MHz | kernel µs: traced / steady | loop MFMA eff | loop loads stalled >= 16 cyc at issue "
          "| vmcnt waits stalled >= 64 cyc | D_44, cyc | issue to its wait, p50 cyc |")
    print("|---|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|")
    for r in rows:
        k, a, L = r["kernel_us"], r["att"], r["att_latency"]
        clk = json.load(open(os.path.join(OUT, f"v7_K{r['K']}_{r['cache']}_clock.json")))
        it = r["att_iter"]
        traced = k["at15"] if it == 15 else k["at600"]
        g_same = clk["ghz_at15"] if it == 15 else clk["ghz_at600"]
        print(f"| {r['K']} | {r['cache']} | {it} | {a['ghz']:.2f} | {g_same:.2f} / {clk['ghz_median_after_100']:.2f} | "
              f"{r['smu'].get('steady', {}).get('mhz')} | {traced:.0f} / {k['median_after_100']:.0f} | {r['mfma_eff']}% | "
              f"{L['stretched_loads_per_cu']:.0f} / {L['loads_per_cu']:.0f} | {L['L_exact']['n']} | {L['dk_at']['44']} | "
              f"{L['L_bound']['p50']} |")
    print()
    for r in rows:
        a = r["att"]
        worst = sorted(a["positions"], key=lambda x: -(x["stall_p90"] or 0))[:4]
        print(f"{r['name']}: {a['loads_per_iter']} loads per loop iteration; most stalled positions: " +
              ", ".join(f"#{w['pos']} p50 {w['stall_p50']} p90 {w['stall_p90']} cyc ({w['stalled']:.0%} >= 16), "
                        f"issued {w['since_a_p50']} cyc after load #1" for w in worst))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--K", type=int, required=True)
    w.add_argument("--cache", required=True)
    w.add_argument("--iters", type=int, default=ITERS)
    w.add_argument("--dump", required=True)
    sub.add_parser("run")
    sub.add_parser("report")
    args = p.parse_args()
    if args.cmd == "worker":
        worker(args)
        return
    os.makedirs(OUT, exist_ok=True)
    if args.cmd == "run":
        for K in (8192, 16384):
            for cache in ("cold", "hot"):
                clk = clock_pass(args.gpu, K, cache)
                json.dump(clk, open(os.path.join(OUT, f"v7_K{K}_{cache}_clock.json"), "w"), indent=1)
                print(f"v7 K={K} {cache} clock pass: GRBM GHz dispatch 15 {clk['ghz_at15']:.2f}, 600 "
                      f"{clk['ghz_at600']:.2f}, median {clk['ghz_median_after_100']:.2f}; µs {clk['us_at15']:.0f} / "
                      f"{clk['us_at600']:.0f} / {clk['us_median_after_100']:.0f}; first 10 {clk['ghz_first10']}",
                      flush=True)
                for it in (15, 600):
                    r = run_one(args.gpu, K, cache, it)
                    print(f"{r['name']}: correct {r['host']['correct']}, trace {r['att']['ghz']:.2f} GHz, SMU "
                          f"{r['smu'].get('steady', {}).get('mhz')} MHz, MFMA {r['mfma_eff']}%, stretched "
                          f"{r['att_latency']['stretched_loads_per_cu']:.0f}/{r['att_latency']['loads_per_cu']:.0f}",
                          flush=True)
    report()


if __name__ == "__main__":
    main()
