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
"""Load round trips and per-CU load capacity from a decoded ATT trace (rocprofv3 --att, ui_* directory).

    python att_latency.py <ui_dir> [--name v8_K8192] [--plot]

Per wave file, instructions are [clock, type, stall, duration, code index]: an instruction reaches issue at
`clock`, leaves the wave at `clock + stall`, and the next one starts at `clock + duration`.

Round trip (per wave). vmcnt counts the wave's VMEM operations (loads and stores) in order, so the
s_waitcnt vmcnt(N) that follows the i-th operation with i = issued - N - 1 is satisfied once operation i is
back. When that wait stalled, it ended within a few cycles of operation i's return:
    L = (wait clock + duration) - (op clock + stall)       ("exact")
A wait that stalled less than --exact-stall cycles (the ATT reports a few cycles of stall on most waits)
only bounds L from above ("bound").

Capacity (per CU: the 4 traced waves of one SE share one CU and one clock). A buffer_load that stalls
before leaving the wave is waiting for a slot in the CU's VMEM queue + TCP. If those hold k loads, it leaves
when the load issued k loads earlier (counted over all 4 waves) comes back, so for the true k
    D_k = issue(n) - issue(n - k)  ~  L (the round trip of load n - k, measured from issue as above).
The script reports the median D_k for every k and the k at which it matches the trace's own median exact L
(none if no k comes within 10%: the stretched loads were paced by something else, such as the TCP's 16 cycles
per 1 KB request, which makes D_k = 16 k).

With --loop, only the hot loop counts: the loop body is scripts/process_json.py's (the back-edge's hit count),
round trips are kept when their s_waitcnt is in the body, and D_k uses only the body's loads. The CU's load
rate in the loop (loads over the union of its waves' loop spans) times L gives the KB in flight (Little's law).

Writes <out>/att_<name>.json (summary) and, with --plot, <out>/att_<name>_timeline.png and _dk.png.
"""

import argparse
import collections
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
VMCNT = re.compile(r"vmcnt\((\d+)\)")
LOAD_KB = 1  # one buffer_load_dwordx4 per wave


def is_vmem(op):
    return op.startswith(("buffer_", "global_", "flat_")) and "wbl2" not in op and "inv" not in op


def load_trace(ui):
    code = json.load(open(os.path.join(ui, "code.json")))["code"]
    if not code:
        raise SystemExit(f"{ui}: no code in code.json (the traced CUs ran none of the kernel's waves?)")
    text = {c[2]: c[0] for c in code}
    rt = json.load(open(os.path.join(ui, "realtime.json")))
    ghz = {}
    for se, pts in rt.items():
        if se.startswith("SE") and len(pts) >= 2:
            (g0, r0), (g1, r1) = pts[0], pts[-1]
            ghz[int(se[2:])] = (g1 - g0) / max(1, r1 - r0) / 10.0  # realtime counter at 100 MHz
    waves = []
    for f in sorted(glob.glob(os.path.join(ui, "se*_sm*_sl*_wv*.json"))):
        m = re.search(r"se(\d+)_sm(\d+)_sl(\d+)_wv(\d+)", os.path.basename(f))
        d = json.load(open(f))
        if "wave" not in d or not d["wave"].get("instructions"):
            continue
        waves.append(dict(se=int(m.group(1)), simd=int(m.group(2)), slot=int(m.group(3)), name=os.path.basename(f),
                          ins=d["wave"]["instructions"]))
    return text, ghz, waves, code


def loop_range(code):
    """(first, last) code index of the hot loop body, as scripts/process_json.py finds it."""
    import sys

    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import process_json

    info = process_json.analyze_code(code)
    return info["loop_first_index"], info["loop_last_index"]


def wave_events(w, text, exact_stall=16, loop=None):
    """VMEM ops (issue time, stall, is_load, in loop) and round-trip samples (op index, L, exact, wait end,
    in loop) from the vmcnt waits of one wave. Without `loop`, everything counts as in the loop."""
    ops = []
    samples = []
    resolved = 0  # ops known to be back
    in_loop = (lambda i: loop[0] <= i <= loop[1]) if loop else (lambda i: True)
    for clock, _ty, stall, dur, idx in w["ins"]:
        t = text.get(idx, "")
        op = t.split()[0] if t else ""
        if is_vmem(op):
            ops.append((clock + stall, stall, op.startswith("buffer_load") or op.startswith("global_load"),
                        in_loop(idx)))
        elif op == "s_waitcnt":
            m = VMCNT.search(t)
            if not m:
                continue
            need = len(ops) - int(m.group(1))  # ops [0, need) must be back
            if need > resolved:
                i = need - 1
                end = clock + dur
                samples.append((i, end - ops[i][0], stall >= exact_stall, end, in_loop(idx)))
                resolved = need
    return ops, samples


def loop_rate(items):
    """A CU's loads per cycle in the loop: its waves' loop loads over the union of their loop spans (first to
    last loop load), so the gap between two rounds of workgroups does not count."""
    spans, n = [], 0
    for _, ops, _ in items:
        t = [o[0] for o in ops if o[2] and o[3]]
        if len(t) >= 2:
            spans.append((min(t), max(t)))
            n += len(t)
    union, end = 0.0, -np.inf
    for a, b in sorted(spans):
        if b > end:
            union += b - max(a, end)
            end = b
    return n / union if union else None


def analyse(ui, stretch=16, exact_stall=16, kmax=72, loop=False):
    text, ghz, waves, code = load_trace(ui)
    lrange = loop_range(code) if loop else None
    per_wave = []
    by_se = collections.defaultdict(list)
    exact, bound = [], []
    for w in waves:
        ops, samples = wave_events(w, text, exact_stall, lrange)
        samples = [s for s in samples if s[4]]
        loads = [o for o in ops if o[2] and o[3]]
        per_wave.append(dict(name=w["name"], se=w["se"], vmem_ops=len(ops), loads=len(loads),
                             exact=sum(1 for s in samples if s[2]), bound=sum(1 for s in samples if not s[2])))
        exact += [s[1] for s in samples if s[2]]
        bound += [s[1] for s in samples if not s[2]]
        by_se[w["se"]].append((w, ops, samples))

    # Per CU: merged loads, stretched-issue D_k
    dk_all = collections.defaultdict(list)
    stretched_total = 0
    loads_total = 0
    rates = {}
    for se, items in by_se.items():
        rates[se] = loop_rate(items)
        merged = sorted((o[0], o[1]) for _, ops, _ in items for o in ops if o[2] and o[3])
        issue = np.array([m[0] for m in merged], dtype=np.float64)
        stall = np.array([m[1] for m in merged], dtype=np.float64)
        loads_total += len(issue)
        idx = np.nonzero(stall >= stretch)[0]
        stretched_total += len(idx)
        for k in range(8, kmax + 1):
            ok = idx[idx >= k]
            if len(ok):
                dk_all[k] += (issue[ok] - issue[ok - k]).tolist()

    def pct(a, q):
        return float(np.percentile(a, q)) if len(a) else None

    L_exact = pct(exact, 50)
    dk_med = {k: float(np.median(v)) for k, v in sorted(dk_all.items()) if v}
    k_match = None
    if L_exact is not None and dk_med:
        k_best = min(dk_med, key=lambda k: abs(dk_med[k] - L_exact))
        # No match unless D_k crosses L: loads paced by something other than free slots never reach it.
        if abs(dk_med[k_best] - L_exact) <= 0.1 * L_exact:
            k_match = k_best
    g = float(np.median(list(ghz.values()))) if ghz else None
    loads_per_wave = np.median([p["loads"] for p in per_wave]) if per_wave else 0
    rate = [r for r in rates.values() if r]
    rate = float(np.median(rate)) if rate else None

    def ns(c):
        return round(c / g, 1) if (c and g) else None

    def inflight_kb(L):
        return round(rate * L * LOAD_KB, 1) if (rate and L) else None

    L_bound = pct(bound, 50)
    return dict(
        ui=os.path.abspath(ui),
        loop_only=bool(loop),
        loop_code_range=list(lrange) if lrange else None,
        waves=len(waves),
        ses=sorted(by_se),
        ghz_realtime=round(g, 3) if g else None,
        loads_per_wave=float(loads_per_wave),
        loads_per_cu=loads_total / max(1, len(by_se)),
        stretched_loads_per_cu=stretched_total / max(1, len(by_se)),
        stretch_threshold_cyc=stretch,
        L_exact=dict(n=len(exact), p10=pct(exact, 10), p50=L_exact, p90=pct(exact, 90),
                     mean=float(np.mean(exact)) if exact else None, p50_ns=ns(L_exact)),
        L_bound=dict(n=len(bound), p10=pct(bound, 10), p50=L_bound, p90=pct(bound, 90), p50_ns=ns(L_bound)),
        load_rate_per_cu=dict(loads_per_cyc=rate, bytes_per_cyc=rate * LOAD_KB * 1024 if rate else None,
                              bytes_per_ns=rate * LOAD_KB * 1024 * g if (rate and g) else None),
        inflight_kb_per_cu=dict(at_L_exact_p50=inflight_kb(L_exact), at_L_bound_p50=inflight_kb(L_bound)),
        dk_median=dk_med,
        k_match=k_match,
        dk_at={k: dk_med.get(k) for k in (32, 44, 48)},
        per_wave=per_wave,
    ), (text, waves, by_se, exact, dk_all)


def plot(res, extra, out_prefix, window=6000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    text, waves, by_se, exact, dk_all = extra
    se = sorted(by_se)[0]
    items = sorted(by_se[se], key=lambda x: (x[0]["simd"], x[0]["slot"]))[:16]
    # Window: centred in the busiest stretch of the loop (most stretched loads in any `window` cycles).
    merged = sorted(o[0] for _, ops, _ in items for o in ops if o[2] and o[3] and o[1] >= res["stretch_threshold_cyc"])
    allops = sorted(o[0] for _, ops, _ in items for o in ops if o[3])
    if merged:
        arr = np.array(merged)
        counts = np.searchsorted(arr, arr + window) - np.arange(len(arr))
        t0 = arr[int(np.argmax(counts))] - window * 0.1
    else:
        t0 = allops[len(allops) // 2]
    t1 = t0 + window

    fig, ax = plt.subplots(figsize=(14, 1.2 + 0.9 * len(items) + 1.2))
    for row, (w, ops, samples) in enumerate(items):
        y = len(items) - row
        for clock, _ty, stall, dur, idx in w["ins"]:
            if clock + dur < t0 or clock > t1:
                continue
            t = text.get(idx, "")
            op = t.split()[0] if t else ""
            if op.startswith("v_mfma"):
                ax.add_patch(plt.Rectangle((clock, y - 0.12), dur, 0.24, color="#4c9a5a", alpha=0.35, lw=0))
            elif is_vmem(op):
                if stall > 0:
                    ax.add_patch(plt.Rectangle((clock, y - 0.3), stall, 0.6, color="#c8443c", alpha=0.6, lw=0))
                ax.plot([clock + stall] * 2, [y - 0.35, y + 0.35], color="#d9822b", lw=1.2)
            elif op == "s_waitcnt" and "vmcnt" in t and stall > 0:
                ax.add_patch(plt.Rectangle((clock, y - 0.3), dur, 0.6, color="#3b7dd8", alpha=0.6, lw=0))
        ax.text(t0 - window * 0.01, y, f"SIMD {w['simd']} wave {w['slot']}", ha="right", va="center", fontsize=8)
    # CU-wide loads issued in the trailing round trip (median exact L): the loads that can be in flight.
    L = res["L_exact"]["p50"]
    if L:
        cu_loads = np.array(sorted(o[0] for _, ops, _ in items for o in ops if o[2]))
        ts = np.linspace(t0, t1, 400)
        inflight = np.searchsorted(cu_loads, ts) - np.searchsorted(cu_loads, ts - L)
        ax2 = ax.twinx()
        ax2.plot(ts, inflight, color="black", lw=1)
        for c, s in ((32, "--"), (44, ":")):
            ax2.axhline(c, color="grey", ls=s, lw=0.8)
            ax2.text(t1, c, f" {c}", va="center", fontsize=8, color="grey")
        ax2.set_ylabel(f"CU loads (1 KB) issued in the last L = {L:.0f} cyc", fontsize=9)
        ax2.set_ylim(0, max(60, inflight.max() * 1.1))
    ax.set_xlim(t0 - window * 0.08, t1)
    ax.set_ylim(0.3, len(items) + 0.8)
    ax.set_yticks([])
    ax.set_xlabel("ATT clock (cycles)")
    ax.set_title(f"{res['name']}: SE{se} CU, {window} cycles{' (first 16 waves)' if len(by_se[se]) > 16 else ''}. "
                 f"orange = buffer_load leaves the wave, red = its issue stall (slot wait), blue = s_waitcnt vmcnt "
                 f"stall, green = MFMA", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_prefix + "_timeline.png", dpi=130)
    plt.close(fig)

    if res["dk_median"]:
        ks = sorted(res["dk_median"])
        fig, ax = plt.subplots(figsize=(7, 4))
        q = {k: np.percentile(dk_all[k], [25, 50, 75]) for k in ks}
        ax.fill_between(ks, [q[k][0] for k in ks], [q[k][2] for k in ks], alpha=0.25, label="D_k, 25-75%")
        ax.plot(ks, [q[k][1] for k in ks], label="D_k median")
        if L:
            ax.axhline(L, color="black", ls="--", lw=1, label=f"median exact L = {L:.0f}")
            lo, hi = res["L_exact"]["p10"], res["L_exact"]["p90"]
            ax.axhspan(lo, hi, color="grey", alpha=0.15, label="exact L, 10-90%")
        for c in (32, 44):
            ax.axvline(c, color="grey", lw=0.8, ls=":")
        ax.set_xlabel("k (loads back, over the CU's 4 waves)")
        ax.set_ylabel("issue(n) - issue(n-k), cycles")
        ax.set_title(f"{res['name']}: stretched buffer_loads (stall >= {res['stretch_threshold_cyc']})", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_prefix + "_dk.png", dpi=130)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ui", nargs="+", help="decoded ATT ui_* directories")
    p.add_argument("--name", nargs="*", default=None, help="one label per ui directory")
    p.add_argument("--out", default=os.path.join(HERE, "results"))
    p.add_argument("--stretch", type=int, default=16, help="issue stall (cycles) that counts as waiting for a slot")
    p.add_argument("--exact-stall", type=int, default=16,
                   help="vmcnt wait stall (cycles) for its end to count as the load's return (shorter: upper bound)")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--loop", action="store_true", help="hot loop only (round trips, D_k, load rate)")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    names = args.name or [os.path.basename(os.path.dirname(os.path.abspath(u))) for u in args.ui]
    for ui, name in zip(args.ui, names):
        res, extra = analyse(ui, stretch=args.stretch, exact_stall=args.exact_stall, loop=args.loop)
        res["name"] = name
        path = os.path.join(args.out, f"att_{name}.json")
        with open(path, "w") as f:
            json.dump(res, f, indent=1)
        le, lb = res["L_exact"], res["L_bound"]
        print(f"{name}: {res['waves']} waves, {res['loads_per_wave']:.0f} loads/wave, clock {res['ghz_realtime']} GHz; "
              f"stretched loads/CU {res['stretched_loads_per_cu']:.0f} of {res['loads_per_cu']:.0f}")
        print(f"  L exact (stalled vmcnt waits): n={le['n']} p10/50/90 = {le['p10']}/{le['p50']}/{le['p90']} cyc "
              f"({le['p50_ns']} ns); bound n={lb['n']} p50={lb['p50']}")
        print(f"  D_k medians: k=32 {res['dk_at'][32]}, k=44 {res['dk_at'][44]}, k=48 {res['dk_at'][48]}; "
              f"k matching L: {res['k_match']}")
        lr, kb = res["load_rate_per_cu"], res["inflight_kb_per_cu"]
        if lr["loads_per_cyc"]:
            print(f"  CU load rate{' in the loop' if args.loop else ''}: {lr['bytes_per_cyc']:.1f} B/clk "
                  f"({lr['bytes_per_ns']:.1f} B/ns); KB in flight at L exact p50 {kb['at_L_exact_p50']}, "
                  f"at L bound p50 {kb['at_L_bound_p50']}")
        if args.plot:
            plot(res, extra, os.path.join(args.out, f"att_{name}"))
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
