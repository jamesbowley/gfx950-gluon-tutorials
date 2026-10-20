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
"""Which instructions the MFMAs fail to hide in a v9_tile K loop, from ATT.

    python v9_tile/att_gaps.py v9_tile/tmp/tile_n/att_r1/sweep_att/256x128_4096x2048x8192 [--json out.json]

Every traced wave owns one SIMD (4 waves per workgroup, one workgroup per CU), so it owns that SIMD's
MFMA pipe. In a wave file an instruction reaches issue at `clock`, issues at `clock + stall` and the
next one starts at `clock + duration`. An MFMA keeps the pipe busy from its issue for its cycle count
(16 for v_mfma_f32_16x16x32_bf16, process_json.get_mfma_cycles). Inside the hot loop (process_json's
loop window) the script:

- builds the pipe's busy intervals; the gaps between them are the loop's MFMA idle time, so
  busy / window is the hot-loop efficiency again (`loop_eff`, for checking against process_json);
- charges each gap to the instructions whose [clock, clock + duration) covers it, split into the
  instruction's stall part and its issue part, and to the hole before an instruction whose clock comes
  after the previous instruction's end (it waited on that instruction's result, e.g. a v_add that
  computes the next ds_read's address; ATT records the hole as neither stall nor issue). What overlaps a
  busy interval is hidden; what overlaps a gap is exposed. Exposed cycles are listed by class and by code
  index (the hot spots);
- splits MFMA stall cycles into pipe-full (behind a busy pipe: free) and pipe-empty (a loss);
- per region between consecutive s_waitcnt / s_barrier instructions: the non-MFMA issue cycles
  against the MFMA shadow available (each MFMA's cycles minus its own issue cycles), and the idle
  charged to the region. Idle in a region whose work fits its shadow is placement: the schedule;
- for each stalled s_waitcnt, the load it waits on (vmcnt(N): the (N+1)-th most recent VMEM
  instruction; lgkmcnt(N): the (N+1)-th most recent LDS instruction), and the cycles and MFMAs from
  that load's issue to the wait;
- per s_barrier, the arrival skew across the waves of one CU and which SIMD arrived last.

Numbers are per loop iteration and wave, averaged over the traced waves.
"""

import argparse
import collections
import glob
import json
import os
import re
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import process_json as pj  # noqa: E402

WAIT_RE = re.compile(r"(vmcnt|lgkmcnt)\((\d+)\)")
STALLED_WAIT = 8  # cycles: a wait that stalls at least this long gets a distance
GROUP_START_SPREAD = 2000  # cycles: waves of one workgroup start within this of each other


def classify(isa):
    """As att_stalls.classify."""
    op = isa.split()[0]
    if op.startswith("v_mfma"):
        return "mfma"
    if op == "s_waitcnt":
        if "vmcnt" in isa and "lgkmcnt" in isa:
            return "s_waitcnt vmcnt+lgkmcnt"
        return "s_waitcnt " + ("vmcnt" if "vmcnt" in isa else "lgkmcnt")
    if op.startswith("buffer_load"):
        return "buffer_load"
    if op.startswith("ds_read"):
        return "ds_read"
    if op == "s_barrier":
        return "s_barrier"
    return "other"


def category(cls, part):
    """What an exposed cycle is: memory (the TCP window or a load not back), the LDS, a barrier, an MFMA
    waiting with the pipe empty, issue (the wave busy issuing other instructions), or a dependency hole
    (the instruction reached issue late because it waits on the previous one's result)."""
    if part == "issue":
        return "issue"
    if part == "hole":
        return "dependency hole"
    if cls == "buffer_load":
        return "memory: buffer_load issue stall"
    if "vmcnt" in cls:
        return "memory: vmcnt wait"
    if cls == "s_waitcnt lgkmcnt":
        return "lds: lgkmcnt wait"
    if cls == "ds_read":
        return "lds: ds_read stall"
    if cls == "s_barrier":
        return "barrier"
    if cls == "mfma":
        return "mfma stall, pipe empty"
    return "other stall"


def is_vmem(op):
    return op.startswith(("buffer_", "global_", "flat_"))


def is_lgkm(op):
    return op.startswith("ds_")


def overlap(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


def load(ui):
    code = pj.load_code_json(ui)
    info = pj.analyze_code(code)
    text = {c[2]: c[0] for c in code}
    cyc = {c[2]: pj.get_mfma_cycles(c[0]) for c in code}
    waves = []
    for path in sorted(glob.glob(os.path.join(ui, "se*_sm*_sl*_wv*.json"))):
        w = json.load(open(path))["wave"]
        if w.get("instructions"):
            se = int(re.match(r"se(\d+)", os.path.basename(path)).group(1))
            waves.append((os.path.basename(path), se, w))
    return info, text, cyc, waves


def loop_slice(ins, info):
    """The instructions of the loop window: first loop-head occurrence up to the first epilogue
    instruction (or the last loop-body instruction when the epilogue never runs)."""
    lf, ll, ef = info["loop_first_index"], info["loop_last_index"], info["epilogue_first_index"]
    start = next((i for i, x in enumerate(ins) if x[4] == lf), None)
    if start is None:
        return None, 0
    if ef is not None:
        end = next((i for i in range(start, len(ins)) if ins[i][4] == ef), len(ins))
    else:
        end = max(i for i in range(start, len(ins)) if lf <= ins[i][4] <= ll) + 1
    iters = sum(1 for x in ins[start:end] if x[4] == lf)
    return (start, end), iters


def analyze_wave(ins, info, text, cyc):
    (s, e), iters = loop_slice(ins, info)
    t0, t1 = ins[s][0], ins[e][0] if e < len(ins) else ins[e - 1][0] + ins[e - 1][3]
    # MFMA busy intervals, merged
    busy = []
    for x in ins[s:e]:
        c = cyc.get(x[4], 0)
        if c:
            b0 = x[0] + x[2]
            if busy and b0 <= busy[-1][1]:
                busy[-1][1] = max(busy[-1][1], b0 + c)
            else:
                busy.append([b0, b0 + c])
    gaps, prev = [], t0
    for b0, b1 in busy:
        if b0 > prev:
            gaps.append((prev, b0))
        prev = max(prev, b1)
    if t1 > prev:
        gaps.append((prev, t1))
    window = t1 - t0
    idle = sum(g1 - g0 for g0, g1 in gaps)

    # code index -> [stall, issue, hole] cycles in gaps; a hole is the time between the previous instruction's
    # end and this one's clock, which ATT records as neither (a VALU result used as the next ds_read's address)
    exposed = collections.defaultdict(lambda: [0, 0, 0])
    exp_pos = []  # exposed cycles of each loop-window instruction, in order
    mfma_full = mfma_empty = 0
    gi = 0
    prev_end = t0
    for x in ins[s:e]:
        c0, st, du, idx = x[0], x[2], x[3], x[4]
        h0 = min(prev_end, c0)
        while gi < len(gaps) and gaps[gi][1] <= h0:
            gi += 1
        j, ex_st, ex_is, ex_h = gi, 0, 0, 0
        while j < len(gaps) and gaps[j][0] < c0 + du:
            ex_h += overlap(h0, c0, *gaps[j])
            ex_st += overlap(c0, c0 + st, *gaps[j])
            ex_is += overlap(c0 + st, c0 + du, *gaps[j])
            j += 1
        exposed[idx][0] += ex_st
        exposed[idx][1] += ex_is
        exposed[idx][2] += ex_h
        exp_pos.append(ex_st + ex_is + ex_h)
        prev_end = max(prev_end, c0 + du)
        if cyc.get(idx, 0) and st:
            mfma_empty += ex_st
            mfma_full += st - ex_st

    # regions between consecutive s_waitcnt / s_barrier
    regions = collections.defaultdict(lambda: collections.Counter())
    ends = {}
    reg = None
    for x, ex in zip(ins[s:e], exp_pos):
        idx, st, du = x[4], x[2], x[3]
        if reg is None:
            reg = idx
        r = regions[reg]
        if cyc.get(idx, 0):
            r["mfma"] += 1
            r["shadow"] += cyc[idx] - (du - st)
        else:
            r["other"] += 1
            r["issue"] += du - st
            r["stall"] += st
        r["idle"] += ex
        op = text[idx].split()[0]
        if op in ("s_waitcnt", "s_barrier"):
            ends[reg] = idx
            r["end_stall"] += st
            reg = None

    # wait distances (whole wave, so loads issued before the loop count), and barrier arrivals
    vm, lg, nmfma = [], [], 0
    waits = collections.defaultdict(list)
    barriers = []
    for i, x in enumerate(ins):
        idx, c0, st = x[4], x[0], x[2]
        isa = text[idx]
        op = isa.split()[0]
        if cyc.get(idx, 0):
            nmfma += 1
        elif is_vmem(op):
            vm.append((c0 + st, nmfma))
        elif is_lgkm(op):
            lg.append((c0 + st, nmfma))
        elif op == "s_waitcnt" and s <= i < e and st >= STALLED_WAIT:
            for kind, n in WAIT_RE.findall(isa):
                q = vm if kind == "vmcnt" else lg
                n = int(n)
                if len(q) > n:
                    t_issue, m_issue = q[-(n + 1)]
                    waits[(idx, kind)].append((st, c0 - t_issue, nmfma - m_issue))
        elif op == "s_barrier" and s <= i < e:
            barriers.append((idx, c0, st))
    return {"iters": iters, "window": window, "idle": idle, "busy": window - idle, "exposed": exposed,
            "mfma_full": mfma_full, "mfma_empty": mfma_empty, "regions": regions, "ends": ends, "waits": waits,
            "barriers": barriers, "start": ins[0][0]}


def analyze(path):
    ui = path if os.path.basename(path.rstrip("/")).startswith("ui_") else \
        glob.glob(os.path.join(path, "**", "ui_*"), recursive=True)[0]
    info, text, cyc, waves = load(ui)
    per = []
    for name, se, w in waves:
        r = analyze_wave(w["instructions"], info, text, cyc)
        r.update(name=name, se=se, cu=w.get("cu"), simd=w.get("simd"))
        per.append(r)
    n = len(per)
    norm = sum(r["iters"] for r in per)  # iterations x waves
    idle = sum(r["idle"] for r in per) / norm
    exp = collections.defaultdict(lambda: [0.0, 0.0, 0.0])
    for r in per:
        for idx, v in r["exposed"].items():
            for k in range(3):
                exp[idx][k] += v[k] / norm
    by_class = collections.defaultdict(lambda: [0.0, 0.0, 0.0])
    by_cat = collections.Counter()
    for idx, (a, b, h) in exp.items():
        cl = classify(text[idx])
        by_class[cl][0] += a
        by_class[cl][1] += b
        by_class[cl][2] += h
        by_cat[category(cl, "stall")] += a
        by_cat[category(cl, "issue")] += b
        by_cat[category(cl, "hole")] += h
    attributed = sum(by_cat.values())
    hot = sorted(exp.items(), key=lambda kv: -sum(kv[1]))[:15]

    regs = collections.defaultdict(collections.Counter)
    ends = {}
    for r in per:
        ends.update(r["ends"])
        for k, c in r["regions"].items():
            regs[k].update(c)
    regions = []
    for k in sorted(regs):
        c = regs[k]
        f = {m: c[m] / norm for m in ("mfma", "other", "shadow", "issue", "stall", "idle", "end_stall")}
        f.update(start=k, end=ends.get(k), end_isa=text.get(ends.get(k), "(loop end)"))
        f["fits"] = f["issue"] <= f["shadow"]
        regions.append(f)

    wl = collections.defaultdict(list)
    for r in per:
        for k, v in r["waits"].items():
            wl[k].extend(v)
    waits = [{"idx": idx, "kind": kind, "isa": text[idx], "stalled_per_iter": len(v) / norm,
              "stall": sum(a for a, _, _ in v) / len(v), "dist_cyc": sum(b for _, b, _ in v) / len(v),
              "dist_mfma": sum(c for _, _, c in v) / len(v)}
             for (idx, kind), v in sorted(wl.items(), key=lambda kv: -sum(a for a, _, _ in kv[1]))]

    # barrier skew per workgroup: waves of the same SE and CU that start together (a CU that runs several
    # tiles in turn has one group per tile), matched by occurrence
    groups = collections.defaultdict(list)
    for r in sorted(per, key=lambda r: r["start"]):
        key = (r["se"], r["cu"])
        g = groups.get((key, len([k for k in groups if k[0] == key]) - 1))
        if g is None or r["start"] - g[0]["start"] > GROUP_START_SPREAD:
            groups[(key, len([k for k in groups if k[0] == key]))] = [r]
        else:
            g.append(r)
    skew = collections.defaultdict(list)
    last = collections.defaultdict(collections.Counter)
    for g in groups.values():
        if len(g) < 2:
            continue
        m = min(len(r["barriers"]) for r in g)
        for k in range(m):
            arr = [(r["barriers"][k][1], r["simd"]) for r in g]
            idx = g[0]["barriers"][k][0]
            skew[idx].append(max(a for a, _ in arr) - min(a for a, _ in arr))
            last[idx][max(arr)[1]] += 1
    barriers = [{"idx": idx, "isa": text[idx], "skew": sum(v) / len(v), "n": len(v),
                 "last_simd": dict(last[idx].most_common())} for idx, v in sorted(skew.items())]

    return {
        "ui": ui, "waves": n, "wave_files": [r["name"] for r in per], "iters_per_wave": norm / n,
        "window_per_iter": sum(r["window"] for r in per) / norm, "idle_per_iter": idle,
        "loop_eff": sum(r["busy"] for r in per) / sum(r["window"] for r in per),
        "loop_eff_per_wave": {r["name"]: r["busy"] / r["window"] for r in per},
        "mfma_stall_full": sum(r["mfma_full"] for r in per) / norm,
        "mfma_stall_empty": sum(r["mfma_empty"] for r in per) / norm,
        "exposed_by_class": {k: {"stall": v[0], "issue": v[1], "hole": v[2]} for k, v in by_class.items()},
        "exposed_by_category": dict(by_cat), "unattributed": idle - attributed,
        "hotspots": [{"idx": idx, "isa": text[idx], "prev_isa": text.get(idx - 1, ""), "class": classify(text[idx]),
                      "stall": a, "issue": b, "hole": h, "share_of_idle": (a + b + h) / idle if idle else 0}
                     for idx, (a, b, h) in hot],
        "regions": regions, "waits": waits, "barriers": barriers,
    }


def report(name, r):
    out = [f"### {name}", "",
           f"`{r['ui']}`: {r['waves']} waves, {r['iters_per_wave']:.0f} iterations each. Per iteration and wave: "
           f"{r['window_per_iter']:.0f} cycles, {r['idle_per_iter']:.0f} of them MFMA idle "
           f"(loop eff {r['loop_eff'] * 100:.1f}%); MFMA stall {r['mfma_stall_full']:.0f} pipe-full, "
           f"{r['mfma_stall_empty']:.0f} pipe-empty; unattributed {r['unattributed']:.0f}.", "",
           "| exposed as | cycles / iter | share of idle |", "|---|---|---|"]
    idle = r["idle_per_iter"] or 1
    for k, v in sorted(r["exposed_by_category"].items(), key=lambda kv: -kv[1]):
        out.append(f"| {k} | {v:.0f} | {v / idle * 100:.1f}% |")
    out += ["", "Hot spots (exposed cycles per iteration; a hole is time before the instruction reached issue, "
            "waiting on the previous one):", "",
            "| code index | instruction | previous instruction | class | hole | stall | issue | share of idle |",
            "|---|---|---|---|---|---|---|---|"]
    for h in r["hotspots"]:
        out.append(f"| {h['idx']} | `{h['isa'][:60]}` | `{h['prev_isa'][:40]}` | {h['class']} | {h['hole']:.1f} | "
                   f"{h['stall']:.1f} | {h['issue']:.1f} | {h['share_of_idle'] * 100:.1f}% |")
    out += ["", "Regions between waits / barriers (per iteration; regions with under 2 MFMAs and under 5 idle "
            "cycles left out):", "",
            "| from index | to | MFMAs | other | shadow | non-MFMA issue | fits | idle | wait/barrier stall |",
            "|---|---|---|---|---|---|---|---|---|"]
    for g in r["regions"]:
        if g["mfma"] < 2 and g["idle"] < 5:
            continue
        out.append(f"| {g['start']} | {g['end']} `{g['end_isa'][:34]}` | {g['mfma']:.0f} | {g['other']:.0f} | "
                   f"{g['shadow']:.0f} | {g['issue']:.0f} | {'yes' if g['fits'] else 'NO'} | {g['idle']:.0f} | "
                   f"{g['end_stall']:.0f} |")
    if r["waits"]:
        out += ["", f"Stalled waits (>= {STALLED_WAIT} cycles): distance from the awaited load's issue:", "",
                "| code index | wait | counter | stalls / iter | mean stall | cycles since load | MFMAs since load |",
                "|---|---|---|---|---|---|---|"]
        for w in r["waits"][:12]:
            out.append(f"| {w['idx']} | `{w['isa']}` | {w['kind']} | {w['stalled_per_iter']:.2f} | {w['stall']:.0f} | "
                       f"{w['dist_cyc']:.0f} | {w['dist_mfma']:.1f} |")
    if r["barriers"]:
        out += ["", "Barrier arrival skew across a CU's waves:", "",
                "| code index | mean skew (cycles) | last to arrive (SIMD: count) |", "|---|---|---|"]
        for b in r["barriers"]:
            out.append(f"| {b['idx']} | {b['skew']:.0f} | {b['last_simd']} |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="sweep_att point directories or ui_* directories")
    ap.add_argument("--json", help="write the results of every directory, keyed by its name")
    args = ap.parse_args()
    res = {}
    for d in args.dirs:
        name = os.path.basename(d.rstrip("/"))
        r = analyze(d)
        res[name] = r
        print(report(name, r))
    if args.json:
        json.dump(res, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
