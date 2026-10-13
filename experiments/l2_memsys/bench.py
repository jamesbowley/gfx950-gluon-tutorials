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
"""L2 memory-system microbenchmarks (see README.md).

    HIP_VISIBLE_DEVICES=1 python bench.py --sweep m1 m2 m3 m4

Counters come from rocprofv3, one process per counter group: the worker runs the whole list of
configurations in order (`reps` launches each, a 1 GiB read before every launch), and the
dispatches of the measured kernel are matched back to configurations by order. Each value is
the median over launches 2..reps, per launch. A phase's counters are the difference between
two configurations that differ only in that phase.
"""

import argparse
import csv
import glob
import json
import os
import statistics
import subprocess
import sys
import tempfile

import kernel as K  # imports triton before torch
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPS = 6
MEMSYS_COUNTERS = [
    ["TCC_HIT_sum", "TCC_MISS_sum", "TCC_EA0_RDREQ_sum", "TCC_EA0_WRREQ_sum"],
    ["TCC_STREAMING_REQ_sum", "TCC_NORMAL_EVICT_sum", "TCC_NORMAL_WRITEBACK_sum", "TCC_ALL_TC_OP_WB_WRITEBACK_sum"],
    ["TCC_READ_SECTORS_sum", "TCC_WRITE_SECTORS_sum", "TCC_PROBE_sum", "TCC_BYPASS_REQ_sum"],
]
LINES_PER_MIB = 2**20 // 128


# ---------------------------------------------------------------------------------------------
# rocprof plumbing
# ---------------------------------------------------------------------------------------------


def run_worker(task, configs, counters=None, trace=False):
    """Run `bench.py --worker` under rocprofv3. Returns the csv rows of the profiled kernels."""
    d = tempfile.mkdtemp(prefix="l2m_")
    spec = os.path.join(d, "spec.json")
    with open(spec, "w") as f:
        json.dump({"task": task, "configs": configs}, f)
    regex = {"memsys": "memsys_kernel", "dirty": "dirty_kernel|follow_kernel"}[task]
    cmd = ["rocprofv3"]
    cmd += ["--pmc", *counters] if counters else []
    cmd += ["--kernel-trace"] if trace else []
    cmd += ["--kernel-include-regex", regex, "-d", d, "-o", "run", "--output-format", "csv", "--",
            sys.executable, os.path.join(HERE, "bench.py"), "--worker", spec]
    # Once this process has loaded HIP, ROCPROFILER_REGISTER_LIBRARY is set in its environment,
    # and rocprofv3 refuses to start under it.
    env = {k: v for k, v in os.environ.items() if not k.startswith("ROCPROFILER_REGISTER")}
    p = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, env=env)
    if p.returncode:
        print(p.stderr[-3000:], file=sys.stderr)
    name = "counter_collection" if counters else "kernel_trace"
    rows = []
    for f in glob.glob(os.path.join(d, "**", f"*{name}.csv"), recursive=True):
        with open(f) as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def counters_by_config(rows, n_configs, reps=REPS):
    """{config index: {counter: median per launch}} from counter rows, dispatches in order."""
    by = {}
    for r in rows:
        by.setdefault(int(r["Dispatch_Id"]), {})[r["Counter_Name"]] = float(r["Counter_Value"])
    disp = sorted(by)
    assert len(disp) == n_configs * reps, (len(disp), n_configs, reps)
    out = {}
    for i in range(n_configs):
        ds = disp[i * reps + 1:(i + 1) * reps]
        out[i] = {c: statistics.median(by[x][c] for x in ds) for c in by[ds[0]]}
    return out


def memsys_counters(configs, groups=None):
    res = [dict() for _ in configs]
    for group in groups or MEMSYS_COUNTERS:
        got = counters_by_config(run_worker("memsys", configs, counters=group), len(configs))
        for i in range(len(configs)):
            res[i].update(got[i])
    return res


def worker(spec_path):
    with open(spec_path) as f:
        spec = json.load(f)
    if spec["task"] == "memsys":
        bufs = K.Buffers()
        for cfg in spec["configs"]:
            cfg = dict(cfg)
            flush = cfg.pop("flush", True)
            for _ in range(REPS):
                if flush:
                    bufs.flush()
                K.memsys(bufs, **cfg)
        torch.cuda.synchronize()
    elif spec["task"] == "dirty":
        dst = torch.zeros(128 * K.MIB // 4, device="cuda", dtype=torch.float32)
        flush = torch.ones(1024 * K.MIB // 4, device="cuda", dtype=torch.float32)
        rec = torch.zeros(512, device="cuda", dtype=torch.int64)
        frec = torch.zeros(256, device="cuda", dtype=torch.int64)
        for cfg in spec["configs"]:
            for _ in range(cfg.get("reps", REPS)):
                flush.sum()
                K.dirty(dst, rec, cfg["d_mib"], cfg["store"])
                K.follow(frec)
        torch.cuda.synchronize()


def med(xs):
    return statistics.median(xs)


def timed_phases(bufs, cfg, launches=10):
    """Medians over launches of memsys_phases (us), each launch after a 1 GiB read."""
    cfg = dict(cfg)
    flush = cfg.pop("flush", True)
    runs = []
    for _ in range(launches + 1):
        if flush:
            bufs.flush()
        K.memsys(bufs, **cfg)
        torch.cuda.synchronize()
        runs.append(K.memsys_phases(bufs, cfg.get("xcd", 0))[0])
    return [med(r[i] for r in runs[1:]) for i in range(5)]


def k(x):
    return f"{x / 1e3:.1f}"


# ---------------------------------------------------------------------------------------------
# M1: retention by store policy
# ---------------------------------------------------------------------------------------------


def m1():
    """Write F MiB on XCD 0 with a store policy, read E MiB of other data, read the F MiB back
    (another CU's region) newest first. The read-back's hits are the written lines still in L2.
    Front-to-back read-back thrashes LRU (each miss evicts the next line to be read), so it
    only shows all-or-nothing; results/m1_forward.md has that version."""
    stores = {"default": "", "cs (sc0 nt)": ".cs", "wt (sc0 sc1)": ".wt"}
    base = []
    for pre in (False, True):
        for f in (1, 2, 4, 8):
            for e in (0, 1, 2, 3, 4, 6):
                for sname, s in stores.items():
                    cfg = dict(f_mib=f, e_mib=e, store=s, pre=pre, p=e > 0, rev=True)
                    base.append((pre, f, e, sname, cfg))
    configs = []
    for *_, cfg in base:
        configs.append({**cfg, "r2": False})
        configs.append({**cfg, "r2": True})
    res = memsys_counters(configs)
    print("XCD 0 (32 CUs, 4 MiB of L2). Each program writes its region, every program then reads E MiB / 32 of "
          "other data, then reads back its neighbour's region, newest lines first. Read-back counters are the "
          "difference with and without the read-back; hit % = TCC_HIT / F's 128-byte lines. Evictions and "
          "write-backs are over the whole kernel without the read-back, thousands of 128-byte lines. 'pre': "
          "every line of the region was read once before it was written.\n")
    print("| pre | F MiB | E MiB | store | read-back hit % | memory reads per line | evictions | write-backs during "
          "the kernel | write-backs at kernel end |")
    print("|---|---|---|---|---|---|---|---|---|")
    for i, (pre, f, e, sname, cfg) in enumerate(base):
        a, b = res[2 * i], res[2 * i + 1]
        lines = f * LINES_PER_MIB
        hit = b["TCC_HIT_sum"] - a["TCC_HIT_sum"]
        rd = b["TCC_EA0_RDREQ_sum"] - a["TCC_EA0_RDREQ_sum"]
        print(f"| {'yes' if pre else 'no'} | {f} | {e} | {sname} | {100 * hit / lines:.0f} | {rd / lines:.2f} | "
              f"{k(a['TCC_NORMAL_EVICT_sum'])} | {k(a['TCC_NORMAL_WRITEBACK_sum'])} | "
              f"{k(a['TCC_ALL_TC_OP_WB_WRITEBACK_sum'])} |", flush=True)

    # Does an nt read demote a line? Write, read once with LOAD1, pollute, read again (default).
    base2 = []
    for e in (1, 2, 3, 4):
        for l1 in ("", ".cg"):
            cfg = dict(f_mib=2, e_mib=e, pre=True, r1=True, load1=l1, p=True, rev=True)
            base2.append((e, l1, cfg))
    configs = []
    for *_, cfg in base2:
        configs.append({**cfg, "r2": False})
        configs.append({**cfg, "r2": True})
    res = memsys_counters(configs)
    print("\nDemotion: F = 2 MiB written after a pre-read, read once with the first-read policy, E MiB of "
          "pollution, read again with the default policy, newest first. Second read's hits.\n")
    print("| E MiB | first read | second read hit % | memory reads per line |")
    print("|---|---|---|---|")
    for i, (e, l1, cfg) in enumerate(base2):
        a, b = res[2 * i], res[2 * i + 1]
        lines = 2 * LINES_PER_MIB
        hit = b["TCC_HIT_sum"] - a["TCC_HIT_sum"]
        rd = b["TCC_EA0_RDREQ_sum"] - a["TCC_EA0_RDREQ_sum"]
        print(f"| {e} | {'.cg (sc0 nt)' if l1 else 'default'} | {100 * hit / lines:.0f} | {rd / lines:.2f} |",
              flush=True)


# ---------------------------------------------------------------------------------------------
# M2: reading back lines written into L2
# ---------------------------------------------------------------------------------------------

M2_CASES = {
    "full lines": dict(),
    "full lines, read by the writing CU": dict(rot=0),
    "64-byte halves": dict(wpat="half"),
    "one dword per line": dict(wpat="dword"),
    "full lines written twice": dict(twice=True),
    "full lines after a pre-read": dict(pre=True),
    "full lines, cs (sc0 nt) store": dict(store=".cs"),
    "full lines, wt (sc0 sc1) store": dict(store=".wt"),
    "full lines, no flush (old contents in the LLC)": dict(flush=False),
}


def m2():
    """F = 2 MiB on XCD 0, no pollution. Counters of the write (W only minus nothing) and of the
    read-back (W + R2 minus W only); read-back time from the stamps."""
    configs = []
    for case in M2_CASES.values():
        cfg = dict(f_mib=2, **case)
        configs.append({**cfg, "w": False, "r2": False})
        configs.append({**cfg, "r2": False})
        configs.append(dict(cfg))
    res = memsys_counters(configs)
    bufs = K.Buffers()
    lines = 2 * LINES_PER_MIB
    print("F = 2 MiB on XCD 0 (16384 lines), a 1 GiB read before every launch unless noted. Per 128-byte line "
          "of the region; the read-back reads every byte. Times are medians of the stamped phases.\n")
    print("| write | write: L2 write sectors | write: memory writes | read-back: hits | misses | memory reads "
          "| read sectors | probes | read-back us | write us |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for i, (name, case) in enumerate(M2_CASES.items()):
        n, w, r = res[3 * i], res[3 * i + 1], res[3 * i + 2]

        def d(x, y, c):
            return (y[c] - x[c]) / lines

        ph = timed_phases(bufs, dict(f_mib=2, **case))
        print(f"| {name} | {d(n, w, 'TCC_WRITE_SECTORS_sum'):.2f} | {d(n, w, 'TCC_EA0_WRREQ_sum'):.2f} | "
              f"{d(w, r, 'TCC_HIT_sum'):.2f} | {d(w, r, 'TCC_MISS_sum'):.2f} | {d(w, r, 'TCC_EA0_RDREQ_sum'):.2f} | "
              f"{d(w, r, 'TCC_READ_SECTORS_sum'):.2f} | {d(w, r, 'TCC_PROBE_sum'):.2f} | {ph[4]:.2f} | {ph[1]:.2f} |",
              flush=True)


# ---------------------------------------------------------------------------------------------
# M3: where a read is served from (latency)
# ---------------------------------------------------------------------------------------------

M3_CASES = [
    ("hbm", ".cg", True, "reference: HBM (1 GiB read first)"),
    ("llc", ".cg", True, "reference: LLC (XCD 1 read the chain first)"),
    ("l2", ".cg", True, "reference: L2 (the chaser read the chain first)"),
    ("written", ".cg", True, "written by XCD 0's other CUs, no cache bits"),
    ("written", ".cg", False, "same, no 1 GiB read first (old contents in the LLC)"),
    ("written_cs", ".cg", True, "written by XCD 0's other CUs, .cs (sc0 nt)"),
    ("written_pre", ".cg", True, "written by XCD 0's other CUs after a pre-read"),
    ("written_wt", ".cg", True, "written by XCD 0's other CUs, .wt"),
    ("remote_wt", ".cv", True, "written by XCD 1 with .wt, read with .cv (v20's path)"),
    ("written", "", True, "written by XCD 0's other CUs, chased with default loads"),
]


def m3(launches=10, steps=4096):
    bufs = K.Buffers(dst_mib=1, pol_mib=1)
    ch = K.Chain()
    print(f"One wave on XCD 0 follows a chain of {steps} dependent loads over distinct 128-byte lines of a "
          "1 MiB buffer (a random cycle); ns per load, median of "
          f"{launches} launches. Helpers set the chain up in the same launch.\n")
    print("| case | chase load | ns per load | min | max | chain correct |")
    print("|---|---|---|---|---|---|")
    for mode, load, flush, desc in M3_CASES:
        ns, ok = [], True
        for i in range(launches + 1):
            if flush:
                bufs.flush()
            K.chase(ch, mode, steps=steps, load=load)
            torch.cuda.synchronize()
            t, good, xcc = K.chase_result(ch, steps)
            ok &= good and xcc == 0
            if i:
                ns.append(t)
        print(f"| {desc} | {load or 'default'} | {med(ns):.0f} | {min(ns):.0f} | {max(ns):.0f} | {ok} |", flush=True)


# ---------------------------------------------------------------------------------------------
# M5: what the read of a written line fetches
# ---------------------------------------------------------------------------------------------

M5_CASES = {
    "full lines written, every byte read": dict(),
    "full lines written, one dword per line read": dict(rpat="dword"),
    "first 64 B written, only those 64 B read": dict(wpat="half", rpat="first_half"),
    "first 64 B written, only the other 64 B read": dict(wpat="half", rpat="second_half"),
    "full lines written, read twice: the second read": dict(r1=True),
    "control: pre-read, full lines written, every byte read": dict(pre=True),
    "control: nothing written, every byte read": dict(w=False),
}
M5_COUNTERS = [
    ["TCC_EA0_RDREQ_sum", "TCC_EA0_RDREQ_32B_sum", "TCC_EA0_RDREQ_64B_sum", "TCC_EA0_RDREQ_128B_sum"],
    ["TCC_EA0_WRREQ_sum", "TCC_NORMAL_WRITEBACK_sum", "TCC_HIT_sum", "TCC_MISS_sum"],
    ["TCC_READ_sum", "TCC_READ_SECTORS_sum", "TCC_PROBE_sum", "TCC_EA0_RDREQ_DRAM_sum"],
]
M5_CHASE = [
    ("hbm", "reference: HBM"),
    ("llc", "reference: LLC"),
    ("l2", "reference: L2"),
    ("written", "written by XCD 0's other CUs; old copy in HBM"),
    ("llc_written", "written by XCD 0's other CUs; old copy in the LLC (XCD 1 read it first)"),
    ("written_pre", "written by XCD 0's other CUs after a pre-read"),
]


def m5(launches=10, steps=4096):
    """The read-back phase alone (with minus without), XCD 0, 2 MiB, after a 1 GiB read."""
    configs = []
    for case in M5_CASES.values():
        cfg = dict(f_mib=2, **case)
        configs.append({**cfg, "r2": False})
        configs.append(dict(cfg))
    res = memsys_counters(configs, M5_COUNTERS)
    bufs = K.Buffers()
    lines = 2 * LINES_PER_MIB
    cols = ["TCC_READ_sum", "TCC_HIT_sum", "TCC_MISS_sum", "TCC_EA0_RDREQ_sum", "TCC_EA0_RDREQ_32B_sum",
            "TCC_EA0_RDREQ_64B_sum", "TCC_EA0_RDREQ_128B_sum", "TCC_EA0_WRREQ_sum", "TCC_NORMAL_WRITEBACK_sum",
            "TCC_PROBE_sum"]
    print("XCD 0, 2 MiB region (16384 lines), a 1 GiB read before every launch. Counters of the read-back phase "
          "alone, per line of the region. L2 reads are 128-byte requests from L1; memory reads are split by "
          "request size.\n")
    print("| case | L2 reads | hits | misses | memory reads | of 32 B | of 64 B | of 128 B | memory writes "
          "| L2 write-backs | probes | read-back us |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, (name, case) in enumerate(M5_CASES.items()):
        a, b = res[2 * i], res[2 * i + 1]
        vals = [(b[c] - a[c]) / lines for c in cols]
        ph = timed_phases(bufs, dict(f_mib=2, **case))
        print(f"| {name} | " + " | ".join(f"{v:.2f}" for v in vals) + f" | {ph[4]:.2f} |", flush=True)

    ch = K.Chain()
    print(f"\nChase: one wave on XCD 0 follows {steps} dependent loads (one dword of each line), ns per load, "
          f"median of {launches} launches, a 1 GiB read before each.\n")
    print("| case | ns per load | chain correct |")
    print("|---|---|---|")
    for mode, desc in M5_CHASE:
        ns, ok = [], True
        for i in range(launches + 1):
            bufs.flush()
            K.chase(ch, mode, steps=steps)
            torch.cuda.synchronize()
            t, good, xcc = K.chase_result(ch, steps)
            ok &= good and xcc == 0
            if i:
                ns.append(t)
        print(f"| {desc} | {med(ns):.0f} | {ok} |", flush=True)


# ---------------------------------------------------------------------------------------------
# M4: the kernel-end write-back
# ---------------------------------------------------------------------------------------------


def m4(reps=20):
    sizes = [0, 4, 8, 16, 32, 64, 128]
    stores = {"default": "", "wt": ".wt", "cs": ".cs"}
    configs = [dict(d_mib=d, store=s, reps=reps) for s in stores.values() for d in sizes]
    rows = run_worker("dirty", configs, trace=True)
    rows.sort(key=lambda r: int(r["Start_Timestamp"]))
    dirty_rows = [r for r in rows if "dirty_kernel" in r["Kernel_Name"]]
    follow_rows = [r for r in rows if "follow_kernel" in r["Kernel_Name"]]
    assert len(dirty_rows) == len(configs) * reps == len(follow_rows), (len(dirty_rows), len(follow_rows))

    # In-kernel span (first stamp to last), measured in this process.
    dst = torch.zeros(128 * K.MIB // 4, device="cuda", dtype=torch.float32)
    rec = torch.zeros(512, device="cuda", dtype=torch.int64)
    flush = torch.ones(1024 * K.MIB // 4, device="cuda", dtype=torch.float32)

    def span(d, s):
        sp, ev = [], []
        for i in range(reps + 1):
            flush.sum()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            K.dirty(dst, rec, d, s)
            b.record()
            torch.cuda.synchronize()
            r = rec.view(256, 2).cpu()
            if i:
                sp.append(float(r[:, 1].max() - r[:, 0].min()) * 0.01)
                ev.append(a.elapsed_time(b) * 1e3)
        return med(sp), med(ev)

    ctr_configs = [dict(d_mib=d, store=s, reps=REPS) for s in stores.values() for d in sizes]
    ctr = {}
    for group in (["TCC_EA0_WRREQ_sum", "TCC_ALL_TC_OP_WB_WRITEBACK_sum", "TCC_NORMAL_WRITEBACK_sum",
                   "TCC_WRITE_sum"],):
        rows_c = run_worker("dirty", ctr_configs, counters=group)
        by = {}
        for r in rows_c:
            by.setdefault(int(r["Dispatch_Id"]), {})[r["Counter_Name"]] = float(r["Counter_Value"])
        disp = sorted(by)
        assert len(disp) == 2 * len(ctr_configs) * REPS, len(disp)
        for i in range(len(ctr_configs)):
            ds = disp[2 * i * REPS:2 * (i + 1) * REPS]
            dk, fk = ds[0::2][1:], ds[1::2][1:]
            ctr[i] = {"dirty": {c: med(by[x][c] for x in dk) for c in group},
                      "follow": {c: med(by[x][c] for x in fk) for c in group}}

    print("All 256 programs write D MiB in total (D / 8 per XCD, 4 MiB of L2 each), then a one-instruction "
          f"follower kernel runs. A 1 GiB read before every pair. Medians of {reps - 1} launches. Kernel time "
          "and the gap to the follower from rocprofv3's kernel trace; span is the writer's first stamp to its "
          "last (stores complete), in-kernel. Counters per launch, 64-byte write requests, thousands.\n")
    print("| store | D MiB | kernel us | span us | kernel - span us | kernel - wt kernel us | events us | gap to "
          "follower us | follower us | writer: memory writes | writer: lines written back during the kernel | "
          "writer: lines written back at kernel end | follower: memory writes |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    kernel_us = {}
    for i, cfg in enumerate(configs):
        drs = dirty_rows[i * reps + 1:(i + 1) * reps]
        kernel_us[(cfg["store"], cfg["d_mib"])] = med((int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3
                                                      for r in drs)
    for i, cfg in enumerate(configs):
        drs = dirty_rows[i * reps + 1:(i + 1) * reps]
        frs = follow_rows[i * reps + 1:(i + 1) * reps]
        kt = med((int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3 for r in drs)
        gap = med((int(f["Start_Timestamp"]) - int(r["End_Timestamp"])) / 1e3 for r, f in zip(drs, frs))
        ft = med((int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3 for r in frs)
        sp, ev = span(cfg["d_mib"], cfg["store"])
        c = ctr[i]
        sname = [n for n, s in stores.items() if s == cfg["store"]][0]
        dwt = kt - kernel_us[(".wt", cfg["d_mib"])]
        print(f"| {sname} | {cfg['d_mib']} | {kt:.1f} | {sp:.1f} | {kt - sp:.1f} | {dwt:+.1f} | {ev:.1f} | {gap:.1f} | "
              f"{ft:.1f} | {k(c['dirty']['TCC_EA0_WRREQ_sum'])} | {k(c['dirty']['TCC_NORMAL_WRITEBACK_sum'])} | "
              f"{k(c['dirty']['TCC_ALL_TC_OP_WB_WRITEBACK_sum'])} | {k(c['follow']['TCC_EA0_WRREQ_sum'])} |",
              flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", nargs="+", default=["m1", "m2", "m3", "m4", "m5"],
                   choices=["m1", "m2", "m3", "m4", "m5"])
    p.add_argument("--worker", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker:
        worker(args.worker)
        return
    print(f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', 'unset')}")
    for s in args.sweep:
        print(f"\n## {s}\n")
        {"m1": m1, "m2": m2, "m3": m3, "m4": m4, "m5": m5}[s]()


if __name__ == "__main__":
    main()
