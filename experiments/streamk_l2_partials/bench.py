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
"""Reduce-scatter partials through one XCD's L2: correctness, assembly, timing and L2 counters.

    HIP_VISIBLE_DEVICES=2 python bench.py                        # check, asm, time
    HIP_VISIBLE_DEVICES=2 python bench.py --sweep time --cold
    HIP_VISIBLE_DEVICES=2 python bench.py --sweep pmc

Configurations are (T groups per XCD, n programs per group); the partials of one XCD take
T * n * 256 KiB, against 4 MiB of L2 per XCD.
- check: every variant, both placements, --launches launches each with a new epoch; a launch
  fails if any group's output differs from the exact sum. The L2 variants must pass with the
  XCD-local placement and fail with the spread one.
- time: medians over programs and launches of each phase (store: issue to completion of the
  partial stores; sync: tile barrier; slice: reads and the slice store), in us and k-steps.
- end: CUDA-event time of the whole launch against the stamped span, wt_relaxed against
  l2_flags: the L2 path leaves the partials dirty in L2, to be written back at kernel end.
- pmc: rocprofv3 TCC counters per launch (median of launches 3-10) for the XCD-local placement.
- pmc_detail: more counters at one configuration, with and without the slice, to tell where
  the memory reads come from.
"""

import argparse
import csv
import glob
import os
import statistics
import subprocess
import sys
import tempfile

import kernel as K  # imports triton before torch
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
K_STEP_US = 1.21
CONFIGS = [(1, 4), (1, 8), (1, 16), (1, 32), (2, 4), (2, 8), (2, 16), (4, 4), (4, 8)]
TIME_VARIANTS = ["mem", "wt_relaxed", "l2_cg", "l2_def", "l2_cv", "l2_pre", "l2_flags", "l2_flags_pre"]
PMC_COUNTERS = ["TCC_HIT_sum", "TCC_MISS_sum", "TCC_EA0_RDREQ_sum", "TCC_EA0_WRREQ_sum"]

_FLUSH = None


def flush():
    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.ones(256 * 2**20, device="cuda", dtype=torch.float32)
    _FLUSH.sum()


def fill(ws, seed=0):
    """Small integers, so sums of up to 32 partials plus the epochs are exact in fp32."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    ws.src.copy_(torch.randint(-32, 32, ws.src.shape, device="cuda", generator=g).float())


def footprint(T, n):
    return T * n / 4


def check(launches):
    ws = K.Workspace()
    fill(ws)
    ok = True
    print("| T | n | MiB per XCD | variant | XCD-local: bad launches | spread: bad launches | expected |")
    print("|---|---|---|---|---|---|---|")
    for T, n in CONFIGS:
        for v in K.VARIANTS:
            # The flag barrier would hang when spread: a flag can sit in another XCD's L2.
            flags = K.VARIANTS[v][2] == 2
            bad = []
            for place in ((0, ) if flags else (0, 1)):
                b = 0
                for _ in range(launches):
                    K.launch(ws, v, T, n, place)
                    torch.cuda.synchronize()
                    out = ws.out.view(K.NUM_PROGRAMS, K.SLOT)[:K.NUM_XCDS * T]
                    b += int(not torch.equal(out, K.expected(ws, T, n, place)))
                bad.append(b)
            l2 = v.startswith("l2")
            good = bad[0] == 0 and (flags or (bad[1] > 0 if l2 else bad[1] == 0))
            ok &= good
            exp = "local ok" if flags else ("local ok, spread FAIL" if l2 else "both ok")
            spread = "not run (would hang)" if flags else f"{bad[1]}/{launches}"
            print(f"| {T} | {n} | {footprint(T, n):g} | {v} | {bad[0]}/{launches} | {spread} | "
                  f"{exp}: {'yes' if good else 'NO'} |", flush=True)
        # Placement: every XCD-local group on one XCD.
        K.launch(ws, "mem", T, n, 0)
        torch.cuda.synchronize()
        r = K.records(ws, T, n)
        xcc = {int(p): int(r[p, 4]) for p in range(r.shape[0])}
        same = all(len({xcc[p] for p in m}) == 1 for m in K.groups(T, n, 0))
        ok &= same
        print(f"| {T} | {n} | | placement | every XCD-local group on one XCC_ID: {same} | | |", flush=True)
    return ok


def asm():
    ws = K.Workspace()
    print("| variant | VGPRs | instructions with cache bits (count) |")
    print("|---|---|---|")
    for v in K.VARIANTS:
        k = K.launch(ws, v, 2, 8, 0)
        torch.cuda.synchronize()
        counts = {}
        for ln in k.asm["amdgcn"].splitlines():
            ln = ln.split(";")[0].strip()
            if not ln.startswith(("buffer_store_dwordx4", "buffer_load_dwordx4", "buffer_wbl2", "buffer_inv")):
                continue
            words = ln.replace(",", " ").split()
            key = " ".join([words[0]] + [b for b in ("sc0", "sc1", "nt") if b in words])
            counts[key] = counts.get(key, 0) + 1
        print(f"| {v} | {k.n_regs} | " + ", ".join(f"`{x}` x{c}" for x, c in counts.items()) + " |")


def run(ws, v, T, n, place, launches, cold):
    recs = []
    for i in range(launches + 1):
        if cold:
            flush()
        K.launch(ws, v, T, n, place)
        torch.cuda.synchronize()
        if i:
            r = K.records(ws, T, n)
            t = (r[:, :4] - r[:, 0].min()).double() * 0.01
            recs.append(t)
    return recs


def time(launches, cold):
    ws = K.Workspace()
    fill(ws)
    med = statistics.median
    print(f"Medians over programs and launches, us (k-steps of {K_STEP_US} us); "
          f"{'cold: a 1 GiB read before every launch' if cold else 'warm'}.\n")
    print("| T | n | MiB per XCD | variant | store | sync | slice | ending | span |")
    print("|---|---|---|---|---|---|---|---|---|")
    for T, n in CONFIGS:
        for v in TIME_VARIANTS:
            recs = run(ws, v, T, n, 0, launches, cold)
            ph = [med(float(x) for r in recs for x in (r[:, i + 1] - r[:, i])) for i in range(3)]
            end = med(float(x) for r in recs for x in (r[:, 3] - r[:, 0]))
            span = med(float(r[:, 3].max()) for r in recs)
            f = [f"{x:.2f} ({x / K_STEP_US:.2f})" for x in ph + [end, span]]
            print(f"| {T} | {n} | {footprint(T, n):g} | {v} | " + " | ".join(f) + " |", flush=True)


def end(launches):
    """Whole-launch time (CUDA events) against the in-kernel span (first start to last done):
    what the launch spends outside the stamped ending, including any write-back of dirty L2
    lines at the end of the kernel."""
    ws = K.Workspace()
    fill(ws)
    med = statistics.median
    print("| T | n | MiB per XCD | variant | launch (events) | in-kernel span | launch - span |")
    print("|---|---|---|---|---|---|---|")
    for T, n in [(1, 8), (2, 8), (1, 16), (2, 16), (4, 8)]:
        for v in ("wt_relaxed", "l2_flags"):
            ev, span = [], []
            for i in range(launches + 1):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                s.record()
                K.launch(ws, v, T, n, 0)
                e.record()
                torch.cuda.synchronize()
                if i:
                    ev.append(s.elapsed_time(e) * 1e3)
                    r = K.records(ws, T, n)
                    span.append(float(r[:, 3].max() - r[:, 0].min()) * 0.01)
            print(f"| {T} | {n} | {footprint(T, n):g} | {v} | {med(ev):.1f} | {med(span):.1f} | "
                  f"{med(ev) - med(span):.1f} |", flush=True)


def pmc_worker(v, T, n, slice_=True):
    ws = K.Workspace()
    fill(ws)
    for _ in range(10):
        K.launch(ws, v, T, n, 0, slice_=slice_)
    torch.cuda.synchronize()


def collect(counters, worker):
    """rocprofv3 counters of `bench.py --worker <worker>`, median of launches 3-10, or None."""
    d = tempfile.mkdtemp(prefix="l2p_")
    cmd = ["rocprofv3", "--pmc", *counters, "--kernel-include-regex", "probe_kernel", "-d", d, "-o", "run",
           "--output-format", "csv", "--", sys.executable, os.path.join(HERE, "bench.py"), "--worker", worker]
    p = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    if p.returncode:
        print(p.stderr[-2000:], file=sys.stderr)
    by = {}
    for f in glob.glob(os.path.join(d, "**", "*counter_collection.csv"), recursive=True):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                by.setdefault(int(r["Dispatch_Id"]), {})[r["Counter_Name"]] = float(r["Counter_Value"])
    disp = sorted(by)[2:]
    if not disp:
        return None
    return {c: statistics.median(by[x].get(c, float("nan")) for x in disp) for c in counters}


DETAIL_COUNTERS = [
    ["TCC_READ_sum", "TCC_WRITE_sum", "TCC_EA0_RDREQ_sum", "TCC_EA0_WRREQ_sum"],
    ["TCC_HIT_sum", "TCC_MISS_sum", "TCC_RW_REQ_sum", "TCC_NC_REQ_sum"],
    ["TCC_PROBE_sum", "TCC_BYPASS_REQ_sum", "TCC_STREAMING_REQ_sum", "TCC_ALL_TC_OP_WB_WRITEBACK_sum"],
]


def pmc_detail():
    """Where the L2 path's memory reads come from, at (T, n) = (1, 8): the counters with and
    without the slice (store and barrier only)."""
    names = [c for group in DETAIL_COUNTERS for c in group]
    print("(T, n) = (1, 8): 64 programs, 131K 128-byte lines of partials, 131K of src. Thousands per "
          "launch; 'no slice' skips step 4. Reads (TCC_READ, EA reads) count 128-byte requests, writes "
          "64-byte ones.\n")
    print("| variant | slice | " + " | ".join(c.removesuffix("_sum") for c in names) + " |")
    print("|---|---|" + "---|" * len(names))
    for v in ("mem", "wt_relaxed", "l2_cg", "l2_pre"):
        for slice_ in ("", ",noslice"):
            vals = {}
            for group in DETAIL_COUNTERS:
                vals.update(collect(group, f"{v},1,8{slice_}") or {})
            cells = [f"{vals[c] / 1e3:.0f}" if c in vals else "-" for c in names]
            print(f"| {v} | {'no' if slice_ else 'yes'} | " + " | ".join(cells) + " |", flush=True)


def pmc():
    print("Counters per launch, XCD-local placement, median of launches 3-10. Partial bytes = "
          "8 T n x 256 KiB (2048 128-byte lines per partial).\n")
    print("| T | n | variant | partial lines | L2 hit rate | TCC_HIT | TCC_MISS | EA read requests | EA write requests |")
    print("|---|---|---|---|---|---|---|---|---|")
    for T, n in CONFIGS:
        for v in TIME_VARIANTS:
            r = collect(PMC_COUNTERS, f"{v},{T},{n}")
            if r is None:
                print(f"| {T} | {n} | {v} | no counters | | | | | |")
                continue

            def m(c):
                return r[c]

            hit, miss = m("TCC_HIT_sum"), m("TCC_MISS_sum")
            lines = 8 * T * n * 2048
            print(f"| {T} | {n} | {v} | {lines / 1e3:.0f}K | {100 * hit / (hit + miss):.1f}% | {hit / 1e3:.0f}K | "
                  f"{miss / 1e3:.0f}K | {m('TCC_EA0_RDREQ_sum') / 1e3:.0f}K | {m('TCC_EA0_WRREQ_sum') / 1e3:.0f}K |",
                  flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", nargs="+", default=["check", "asm", "time"],
                   choices=["check", "asm", "time", "end", "pmc", "pmc_detail"])
    p.add_argument("--launches", type=int, default=20)
    p.add_argument("--cold", action="store_true")
    p.add_argument("--worker", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker:
        v, T, n, *rest = args.worker.split(",")
        pmc_worker(v, int(T), int(n), slice_=rest != ["noslice"])
        return
    print(f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', 'unset')} launches/point={args.launches}")
    if "check" in args.sweep:
        print("\n## check\n")
        if not check(args.launches):
            raise SystemExit("check failed")
    if "asm" in args.sweep:
        print("\n## asm\n")
        asm()
    if "time" in args.sweep:
        print("\n## time\n")
        time(args.launches, args.cold)
    if "end" in args.sweep:
        print("\n## end\n")
        end(args.launches)
    if "pmc" in args.sweep:
        print("\n## pmc\n")
        pmc()
    if "pmc_detail" in args.sweep:
        print("\n## pmc_detail\n")
        pmc_detail()


if __name__ == "__main__":
    main()
