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
"""Memory-model litmus tests (see README.md).

    HIP_VISIBLE_DEVICES=1 python bench.py --sweep check refs controls stage1 pollute stage2a stage2b stage2c
    HIP_VISIBLE_DEVICES=3 python bench.py --sweep primebits > results/primebits.md

Every sweep prints markdown; the README's tables come from results/*.md.
"""

import argparse
import os
import re
import statistics
import subprocess
import time

import kernel as K  # imports triton before torch
import torch

PLACES = {0: "same workgroup", 1: "same XCD", 2: "other XCD"}
STAGE1_BITS = ["none", "sc0nt", "sc1", "sc0sc1"]
TRITON_NAME = {"none": "default", "sc0nt": "`.cg` / `.cs`", "sc1": "(asm only)", "sc0sc1": "`.cv` / `.wt`"}


def bits_md(b):
    return "none" if b == "none" else "`" + K.BITS[b] + "`"


# ---------------------------------------------------------------------------------------------
# Running a cell and classifying what the reader saw
# ---------------------------------------------------------------------------------------------


class Stats:
    """Accumulates one cell over launches. A line is fresh (the launch's epoch), stale (what the
    reader's prime saw) or other. Line 0 of every pair is the timed read; lines 1-63 are the bulk
    read. Pairs whose timestamps break the protocol, or whose placement is wrong, are excluded and
    counted."""

    def __init__(self, place, sync, phases):
        self.place, self.sync, self.phases = place, sync, phases
        self.lines = self.fresh = self.stale = self.other = 0
        self.t_lines = self.t_fresh = self.t_stale = 0
        self.bad_time = self.bad_place = self.bad_prime = self.timeouts = self.pairs = 0
        self.dup_cu_launches = self.launches = 0
        self.lat_ns, self.lat_cyc, self.lat_rt_ns, self.empty_cyc, self.mhz, self.spins = [], [], [], [], [], []
        self.lat_ns_fresh, self.lat_ns_stale = [], []
        self.x0_older = self.x0_lines = 0
        self.gap_us = []

    def add(self, rec, x0, x1, epoch):
        f = K.FIELD
        P = rec.shape[0]
        self.launches += 1
        self.pairs += P
        wk = K.cu_key(rec[:, f["w_xcc"]], rec[:, f["w_hw"]])
        rk = K.cu_key(rec[:, f["r_xcc"]], rec[:, f["r_hw"]])
        keys = wk if self.place == 0 else torch.cat([wk, rk])
        if keys.unique().numel() != keys.numel():
            self.dup_cu_launches += 1
        same_xcc = rec[:, f["w_xcc"]] == rec[:, f["r_xcc"]]
        if self.place == 0:
            ok_place = wk == rk
        elif self.place == 1:
            ok_place = same_xcc & (wk != rk)
        else:
            ok_place = ~same_xcc
        ok = ok_place.clone()
        self.bad_place += int((~ok_place).sum())
        ok_time = torch.ones(P, dtype=torch.bool)
        if self.phases & 3 == 3:
            ok_time &= rec[:, f["r_primed"]] + 100 < rec[:, f["w_start"]]
        if self.phases & 6 == 6 and self.sync == 0:
            ok_time &= rec[:, f["w_done"]] + 500 < rec[:, f["r_start"]]
            self.gap_us += ((rec[:, f["r_start"]] - rec[:, f["w_done"]]).double() / 100).tolist()
        self.bad_time += int((ok & ~ok_time).sum())
        ok &= ok_time
        if self.sync == 1:
            to = rec[:, f["timeout"]] != 0
            self.timeouts += int((ok & to).sum())
            ok &= ~to
            self.spins += rec[ok, f["spins"]].tolist()
        obs = torch.cat([rec[:, f["timed_val"]].to(torch.int32)[:, None], x1[:, 1:]], dim=1)
        good_prime = x0 != epoch
        self.bad_prime += int((ok[:, None] & ~good_prime).sum())
        valid = ok[:, None] & good_prime
        fresh = (obs == epoch) & valid
        stale = (obs == x0) & valid & ~fresh
        n = int(valid.sum())
        self.lines += n
        self.fresh += int(fresh.sum())
        self.stale += int(stale.sum())
        self.other += n - int(fresh.sum()) - int(stale.sum())
        self.t_lines += int(valid[:, 0].sum())
        self.t_fresh += int(fresh[:, 0].sum())
        self.t_stale += int(stale[:, 0].sum())
        self.x0_lines += int(valid.sum())
        self.x0_older += int((valid & (x0 < epoch - 1)).sum())
        if self.phases & 4:
            e = torch.stack([rec[:, f["e1"]], rec[:, f["e2"]], rec[:, f["e3"]]], 1).double()
            emed = e.median(dim=1).values
            mhz = (rec[:, f["mt1"]] - rec[:, f["mt0"]]).double() / (rec[:, f["rt1"]] - rec[:, f["rt0"]]).double() * 100
            cyc = rec[:, f["mt"]].double() - emed
            ns = cyc / mhz * 1e3
            self.lat_cyc += cyc[ok].tolist()
            self.lat_ns += ns[ok].tolist()
            self.lat_rt_ns += (rec[ok, f["rt"]].double() * 10).tolist()
            self.empty_cyc += e[ok].flatten().tolist()
            self.mhz += mhz[ok].tolist()
            self.lat_ns_fresh += ns[ok & fresh[:, 0]].tolist()
            self.lat_ns_stale += ns[ok & stale[:, 0]].tolist()

    def pct(self, a, b):
        return 100.0 * a / b if b else float("nan")

    def lat(self, which="all"):
        xs = {"all": self.lat_ns, "fresh": self.lat_ns_fresh, "stale": self.lat_ns_stale}[which]
        return statistics.median(xs) if xs else float("nan")

    def verdict(self):
        if self.lines == 0:
            return "no data"
        if self.stale == 0 and self.other == 0:
            return "fresh"
        if self.fresh == 0:
            return "stale"
        return "mixed"


def run_cell(ws, launches=100, flush=True, warmup=1, two_kernel=False, **cfg):
    place = cfg.get("place", 2)
    sync = cfg.get("sync", 0)
    phases = 7 if two_kernel else cfg.get("phases", 7)
    st = Stats(place, sync, phases)
    k = None
    for i in range(warmup + launches):
        if flush:
            ws.flush()
        if two_kernel:
            K.launch(ws, **{**cfg, "phases": 3})
            k = K.launch(ws, **{**cfg, "phases": 4}, new_epoch=False)
        else:
            k = K.launch(ws, **cfg)
        torch.cuda.synchronize()
        if i >= warmup:
            st.add(*K.results(ws, place), ws.epoch)
    return st, k


def fmt_counts(st):
    return (f"{st.pct(st.fresh, st.lines):.2f} | {st.pct(st.stale, st.lines):.2f} | {st.pct(st.other, st.lines):.2f} "
            f"| {st.stale + st.other} / {st.lines}")


def fmt_checks(st):
    s = []
    if st.bad_place:
        s.append(f"{st.bad_place} misplaced")
    if st.bad_time:
        s.append(f"{st.bad_time} out of order")
    if st.bad_prime:
        s.append(f"{st.bad_prime} primed new")
    if st.timeouts:
        s.append(f"{st.timeouts} timeouts")
    if st.dup_cu_launches:
        s.append(f"{st.dup_cu_launches} launches with 2 WGs on a CU")
    return ", ".join(s) or "ok"


# ---------------------------------------------------------------------------------------------
# Assembly check
# ---------------------------------------------------------------------------------------------

RE_LD = re.compile(r"^\s*global_load_dword v\d+, v\[\d+:\d+\], off\s*(.*?)\s*$")
RE_ST = re.compile(r"^\s*global_store_dword v\[\d+:\d+\], v\d+, off\s*(.*?)\s*$")
RE_SYNC = re.compile(r"^\s*(buffer_wbl2|buffer_inv|global_atomic\w*|buffer_atomic\w*|global_load_dword v\d+, v\d+, s\[\d+:\d+\].*"
                     r"|s_waitcnt vmcnt\(0\)|s_barrier)(.*)$")


def asm_bits(k):
    """(load bits, store bits) of the experiment's own data accesses (64-bit vaddr, `off`)."""
    lds, sts = set(), set()
    for line in k.asm["amdgcn"].splitlines():
        m = RE_LD.match(line)
        if m:
            lds.add(m.group(1))
        m = RE_ST.match(line)
        if m:
            sts.add(m.group(1))
    return lds, sts


def check_bits(k, store, load):
    lds, sts = asm_bits(k)
    want_ld = {K.BITS[load]}
    want_st = {K.BITS[store], ""}
    return lds == want_ld and sts == want_st, lds, sts


# ---------------------------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------------------------


def gpu_snapshot():
    try:
        out = subprocess.run(["rocm-smi", "--showuse", "--showpidgpus"], capture_output=True, text=True).stdout
    except FileNotFoundError:
        return "rocm-smi not found"
    lines = [x for x in out.splitlines() if "GPU use" in x or "PID" in x or x.strip().isdigit()]
    return "\n".join(lines)


def sweep_check(ws):
    """Placement and occupancy over 20 launches per placement; the encodings of every stage-1 cell."""
    print("Placement: every program's XCC and CU from `HW_REG_XCC_ID` / `HW_REG_HW_ID`, 20 launches per "
          "placement, default bits.\n")
    print("| placement | pairs | writer XCDs | reader XCDs | pairs misplaced | launches with two workgroups on one CU |")
    print("|---|---|---|---|---|---|")
    f = K.FIELD
    for place in PLACES:
        st, _ = run_cell(ws, launches=20, place=place)
        rec, _, _ = K.results(ws, place)
        wx = sorted(set(rec[:, f["w_xcc"]].tolist()))
        rx = sorted(set(rec[:, f["r_xcc"]].tolist()))
        print(f"| {PLACES[place]} | {rec.shape[0]} | {wx} | {rx} | {st.bad_place} | {st.dup_cu_launches} |", flush=True)
    print("\nEncodings: the cache bits on the experiment's own `global_load_dword` / `global_store_dword` "
          "(64-bit address, `off`) in each compiled stage-1 cell. Stores include the pollution store "
          "(no bits).\n")
    print("| store | load | load bits in the asm | store bits in the asm | as named |")
    print("|---|---|---|---|---|")
    for s in STAGE1_BITS:
        for l in STAGE1_BITS:
            _, k = run_cell(ws, launches=1, warmup=0, store=s, load=l, place=2)
            ok, lds, sts = check_bits(k, s, l)
            print(f"| {bits_md(s)} | {bits_md(l)} | {sorted(lds)} | {sorted(sts)} | {ok} |", flush=True)


REF_CASES = [
    # name, cfg, flush. Tier = the text before the colon; the first case of a tier is its reference.
    ("L1: the reader read the line (default), default read", dict(place=1, phases=5, prime="none", load="none"), True),
    ("L2: another CU of the XCD read the line, default read", dict(place=1, phases=12, prime="none", load="none"),
     True),
    ("LLC: another XCD read the line, default read", dict(place=2, phases=12, prime="none", load="none"), True),
    ("HBM: 1 GiB read first, default read", dict(place=1, phases=4, load="none"), True),
    ("L2: the reader read the line (default), `sc0 nt` read", dict(place=1, phases=5, prime="none", load="sc0nt"),
     True),
    ("L2: the reader read the line (default), `sc1` read", dict(place=1, phases=5, prime="none", load="sc1"), True),
    ("L2: the reader read the line (default), `sc0 sc1` read", dict(place=1, phases=5, prime="none", load="sc0sc1"),
     True),
    ("L2: another CU of the XCD read the line, `sc1` read", dict(place=1, phases=12, prime="none", load="sc1"), True),
    ("LLC: another XCD read the line, `sc1` read", dict(place=2, phases=12, prime="none", load="sc1"), True),
    ("LLC: another XCD read the line, `sc0 sc1` read", dict(place=2, phases=12, prime="none", load="sc0sc1"), True),
    ("HBM: 1 GiB read first, `sc1` read", dict(place=1, phases=4, load="sc1"), True),
    ("HBM: 1 GiB read first, `sc0 sc1` read", dict(place=1, phases=4, load="sc0sc1"), True),
    ("(no flush) the reader read the line with `sc0 sc1`, `sc0 sc1` read",
     dict(place=1, phases=5, prime="sc0sc1", load="sc0sc1"), False),
    ("(no flush) the reader read the line with `sc1`, `sc1` read", dict(place=1, phases=5, prime="sc1", load="sc1"),
     False),
]


def references(ws, launches=50, verbose=True):
    """Reference latencies of the timed read, and the empty bracket. Returns {tier: median ns}."""
    if verbose:
        print("The timed read of line 0 (one wave-uniform line), with nothing written. Bracket-subtracted "
              "`s_memtime` cycles converted to ns with each wave's own clock (s_memtime against "
              f"s_memrealtime over the kernel); medians and 5th-95th percentiles over {launches} launches x "
              "128 pairs. `rt` is the same read timed with `s_memrealtime` (10 ns ticks, includes the "
              "s_memtime round trips).\n")
        print("| case | ns | p5-p95 ns | cycles | rt ns | clock MHz |")
        print("|---|---|---|---|---|---|")
    refs = {}
    empties, mhzs = [], []
    for name, cfg, flush in REF_CASES:
        st, _ = run_cell(ws, launches=launches, flush=flush, **cfg)
        ns = sorted(st.lat_ns)
        p5, p95 = ns[len(ns) // 20], ns[len(ns) * 19 // 20]
        if not name.startswith("("):
            refs.setdefault(name.split(":")[0], statistics.median(ns))
        empties += st.empty_cyc
        mhzs += st.mhz
        if verbose:
            print(f"| {name} | {statistics.median(ns):.0f} | {p5:.0f}-{p95:.0f} | {statistics.median(st.lat_cyc):.0f} | "
                  f"{statistics.median(st.lat_rt_ns):.0f} | {statistics.median(st.mhz):.0f} |", flush=True)
    if verbose:
        e = sorted(empties)

        def q(p):
            return e[min(len(e) - 1, int(len(e) * p))]

        mhz = statistics.median(mhzs)
        print(f"\nEmpty bracket ({len(e)} samples): median {q(0.5):.0f} cycles, p5 {q(0.05):.0f}, p25 {q(0.25):.0f}, "
              f"p75 {q(0.75):.0f}, p95 {q(0.95):.0f}, max {e[-1]:.0f}. At the median clock ({mhz:.0f} MHz) the "
              f"p5-p95 spread is {(q(0.95) - q(0.05)) / mhz * 1e3:.0f} ns.")
        gaps = sorted(refs.values())
        print(f"Smallest gap between tier medians: {min(b - a for a, b in zip(gaps, gaps[1:])):.0f} ns.")
    return refs


def tier(refs, ns):
    if ns != ns:
        return "-"
    return min(refs, key=lambda t: abs(refs[t] - ns))


def sweep_refs(ws):
    references(ws)


def sweep_controls(ws, launches=100):
    print(f"{launches} launches per row, a 1 GiB read before each. Fresh / stale / other are % of lines "
          "(64 per pair: the timed line 0 and 63 bulk lines).\n")
    print("| control | placement | expected | fresh % | stale % | other % | not fresh / lines | checks |")
    print("|---|---|---|---|---|---|---|---|")
    rows = [
        ("negative: default store, default load", dict(place=2), "stale"),
        ("negative: default store, default load", dict(place=1), "stale (L1)"),
        ("two kernels: write in one launch, read in the next, default bits", dict(place=2, two_kernel=True), "fresh"),
        ("two kernels, default bits", dict(place=1, two_kernel=True), "fresh"),
        ("two kernels, default bits", dict(place=0, two_kernel=True), "fresh"),
        ("agent release + agent acquire (asm), default bits", dict(place=2, wfence="agent", rfence="agent"), "fresh"),
        ("agent release + agent acquire (compiler), default bits", dict(place=2, wfence="agent_cc", rfence="agent_cc"),
         "fresh"),
        ("agent release + agent acquire (asm), default bits", dict(place=1, wfence="agent", rfence="agent"), "fresh"),
        ("agent release + agent acquire (asm), default bits", dict(place=0, wfence="agent", rfence="agent"), "fresh"),
    ]
    for name, cfg, exp in rows:
        st, _ = run_cell(ws, launches=launches, **cfg)
        print(f"| {name} | {PLACES[cfg['place']]} | {exp} | {fmt_counts(st)} | {fmt_checks(st)} |", flush=True)


def predict(place, store, load):
    """ISA Tables 49-50 and the L2 policy definitions, applied to a reader primed with the same bits.
    The load's path: default loads check L1, then L2, then memory; sc0 nt loads skip L1; sc1 loads skip
    L1 and bypass L2 (a hit line is written back if dirty and invalidated, then fetched from memory).
    Default and sc0 nt stores stay in the writer's L2; sc1 stores bypass L2 to the memory side."""
    store_l2 = store in ("none", "sc0nt")
    if load == "none":
        if place == 0:
            return "?"  # stale if a store does not update the CU's own L1 copy ("Miss" for stores)
        return "stale"  # the primed line in the reader's L1
    if load == "sc0nt":
        if place == 2:
            return "stale"  # the primed line in the reader's L2
        return "fresh"  # one L2: the store updated (or, bypassing, invalidated) the line
    if store_l2:
        return "stale" if place == 2 else "fresh"  # dirty in the writer's L2; same XCD: bypass writes it back
    return "fresh"


def sweep_stage1(ws, launches=100, refs=None):
    refs = refs or references(ws, launches=20, verbose=False)
    print(f"No fences. Time-separated: prime at ts, write at ts + 20 us, read at ts + 70 us. {launches} launches "
          "per cell, a 1 GiB read before each. Reference tiers (median ns): "
          + ", ".join(f"{t} {v:.0f}" for t, v in refs.items()) + ".\n")
    for place in PLACES:
        print(f"\n### {PLACES[place]}\n")
        print("| store | load | predicted | observed | fresh % | stale % | other % | not fresh / lines | timed read ns "
              "| tier | timed: fresh ns | timed: stale ns | checks |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for s in STAGE1_BITS:
            for l in STAGE1_BITS:
                st, k = run_cell(ws, launches=launches, store=s, load=l, place=place)
                ok, _, _ = check_bits(k, s, l)
                chk = fmt_checks(st) + ("" if ok else ", BITS MISMATCH")
                ns = st.lat()
                print(f"| {bits_md(s)} | {bits_md(l)} | {predict(place, s, l)} | {st.verdict()} | {fmt_counts(st)} | "
                      f"{ns:.0f} | {tier(refs, ns)} | {st.lat('fresh'):.0f} | {st.lat('stale'):.0f} | {chk} |",
                      flush=True)


def sweep_pollute(ws, launches=50):
    """Writer-side pollution: how eviction turns a stale cell into a fresh one."""
    print(f"Writer programs store E MiB per writer XCD of other lines (default bits) right after the data. "
          f"{launches} launches per cell.\n")
    print("| placement | store | load | E MiB per writer XCD | fresh % | stale % | not fresh / lines | checks |")
    print("|---|---|---|---|---|---|---|---|")
    for place, s, l in [(2, "none", "sc1"), (2, "sc0nt", "sc1"), (2, "none", "sc0sc1"), (1, "none", "none")]:
        writers_per_xcd = 32 if place == 2 else 16
        for e_mib in (0, 1, 2, 4, 8):
            pol = e_mib * 1024 // 8 // writers_per_xcd
            st, _ = run_cell(ws, launches=launches, store=s, load=l, place=place, pol_iters=pol)
            print(f"| {PLACES[place]} | {bits_md(s)} | {bits_md(l)} | {e_mib} | {st.pct(st.fresh, st.lines):.2f} | "
                  f"{st.pct(st.stale, st.lines):.2f} | {st.stale + st.other} / {st.lines} | {fmt_checks(st)} |",
                  flush=True)


def sweep_stage2a(ws, launches=50):
    print(f"Fences with the time gap kept: the fence alone has to make the data visible. {launches} launches "
          "per cell. Writer fence after the data stores; reader fence just before the read. `agent` is the "
          "inline-asm sequence LLVM emits (release `buffer_wbl2 sc1; s_waitcnt vmcnt(0)`, acquire "
          "`s_waitcnt vmcnt(0); buffer_inv sc1`); `agent_cc` is the compiler's own, from an atomic with "
          "`sem=release` / `acquire`, `scope=gpu`; `wg` is `s_waitcnt vmcnt(0)`; `inv_sc0` is "
          "`buffer_inv sc0`.\n")
    for bits in ("none", "sc0nt"):
        for place in PLACES:
            print(f"\n### {bits_md(bits)} loads and stores, {PLACES[place]}\n")
            print("| writer fence | reader fence | observed | fresh % | stale % | not fresh / lines | checks |")
            print("|---|---|---|---|---|---|---|")
            for wf in ("none", "wg", "agent", "agent_cc"):
                for rf in ("none", "inv_sc0", "agent", "agent_cc"):
                    st, _ = run_cell(ws, launches=launches, store=bits, load=bits, place=place, wfence=wf, rfence=rf)
                    print(f"| {wf} | {rf} | {st.verdict()} | {st.pct(st.fresh, st.lines):.2f} | "
                          f"{st.pct(st.stale, st.lines):.2f} | {st.stale + st.other} / {st.lines} | {fmt_checks(st)} |",
                          flush=True)


STAGE2B = [
    # placement, store, load, wfence, rfence, sync, note
    (2, "sc0sc1", "sc0sc1", "wg", "none", 1, "v21 `wt`: write-through, drain, relaxed flag, `.cv` reads"),
    (2, "sc0sc1", "sc0sc1", "none", "none", 1, "the same without the drain before the flag"),
    (2, "sc1", "sc1", "wg", "none", 1, "device scope only (`sc1`), drain, relaxed flag"),
    (2, "sc1", "sc1", "none", "none", 1, "the same without the drain"),
    (2, "none", "none", "agent", "agent", 1, "default bits, agent release and acquire fences around a relaxed flag"),
    (2, "none", "sc0nt", "agent", "none", 1, "default stores, agent release, flag, `sc0 nt` loads (no acquire)"),
    (2, "sc0sc1", "sc0nt", "wg", "none", 1, "write-through stores, drain, flag, `sc0 nt` loads"),
    (2, "none", "none", "none", "agent", 1, "default bits, acquire only (negative: no release)"),
    (2, "none", "none", "wg", "none", 1, "default bits, drain only (negative)"),
    (1, "none", "sc0nt", "wg", "none", 1, "v21 `l2`: default stores, drain, flag, `sc0 nt` reads"),
    (1, "none", "sc0nt", "none", "none", 1, "the same without the drain"),
    (1, "none", "none", "wg", "none", 1, "default loads (L1 can be stale)"),
    (1, "none", "none", "wg", "agent", 1, "default loads after an agent acquire"),
    (0, "none", "none", "wg", "none", 2, "same workgroup: drain, s_barrier, default loads"),
    (0, "none", "none", "none", "none", 2, "the same without the drain"),
    (0, "none", "sc0nt", "none", "none", 2, "no drain, `sc0 nt` loads"),
]
# The same with 2 MiB per writer XCD of other stores queued ahead of the data store (pol_first).
STRESS = [
    (2, "sc0sc1", "sc0sc1", "wg", "none", 1, "v21 `wt`, stores queued ahead"),
    (2, "sc0sc1", "sc0sc1", "none", "none", 1, "write-through, no drain, stores queued ahead"),
    (2, "sc1", "sc1", "none", "none", 1, "`sc1`, no drain, stores queued ahead"),
    (2, "none", "none", "agent", "agent", 1, "default bits, agent fences, stores queued ahead"),
    (2, "none", "sc0nt", "agent", "none", 1, "agent release, `sc0 nt` loads, stores queued ahead"),
    (2, "sc0sc1", "sc0nt", "wg", "none", 1, "write-through, drain, `sc0 nt` loads, stores queued ahead"),
    (1, "none", "sc0nt", "wg", "none", 1, "v21 `l2`, stores queued ahead"),
    (1, "none", "sc0nt", "none", "none", 1, "v21 `l2` without the drain, stores queued ahead"),
    (0, "none", "none", "none", "none", 2, "same workgroup, no drain, stores queued ahead"),
]


def sweep_stage2b(ws, launches=300):
    print("Ordering: no time gap. The writer stores, applies its fence, then sets flag[pair] = epoch with a "
          "relaxed `gpu`-scope atomic; the reader, which has been spinning since its prime, reads at once. "
          "The writer starts 20 us + 0-10 us of per-pair jitter after the grid barrier. Same workgroup: "
          f"an `s_barrier` instead of the flag. {launches} launches per row.\n")
    print("| placement | store | load | writer fence | reader fence | sync | what | observed | fresh % | "
          "not fresh / lines | median spins | checks |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    rows = [(r, 0) for r in STAGE2B] + [(r, 2) for r in STRESS]
    for (place, s, l, wf, rf, sync, note), mib in rows:
        writers_per_xcd = {0: 32, 1: 16, 2: 32}[place]
        pol = mib * 1024 // 8 // writers_per_xcd
        st, _ = run_cell(ws, launches=launches, store=s, load=l, place=place, wfence=wf, rfence=rf, sync=sync,
                         jitter_us=10.0, pol_iters=pol, pol_first=mib > 0)
        sp = statistics.median(st.spins) if st.spins else 0
        print(f"| {PLACES[place]} | {bits_md(s)} | {bits_md(l)} | {wf} | {rf} | {'flag' if sync == 1 else 'barrier'} "
              f"| {note} | {st.verdict()} | {st.pct(st.fresh, st.lines):.3f} | {st.stale + st.other} / {st.lines} | "
              f"{sp:.0f} | {fmt_checks(st)} |", flush=True)


def flag_asm(k):
    """The flag's own instructions, from the asm: everything between the data store and the read that is an
    atomic, a scalar-addressed load, a cache maintenance op or a wait."""
    out = []
    for line in k.asm["amdgcn"].splitlines():
        m = RE_SYNC.match(line)
        if m and "s_barrier" not in line:
            out.append(re.sub(r"\s+", " ", line.strip()))
    return out


def sweep_stage2c(ws, launches=100):
    print("The flag atomic's own scope and order. First the code each (scope, sem) produces; then runs.\n")
    variants = []
    for scope in ("cta", "gpu", "sys"):
        for sw, sr in (("relaxed", "relaxed"), ("release", "acquire")):
            variants.append((scope, sw, sr))
    print("| scope | writer sem | reader sem | distinct instructions of the flag path (sync-related lines in the "
          "whole kernel) |")
    print("|---|---|---|---|")
    seen = {}
    for scope, sw, sr in variants:
        _, k = run_cell(ws, launches=1, warmup=0, store="none", load="none", place=2, sync=1, fscope=scope,
                        fsem_w=sw, fsem_r=sr)
        lines = flag_asm(k)
        uniq = sorted(set(lines))
        key = tuple(lines)
        same = seen.get(key)
        seen.setdefault(key, f"{scope}/{sw}/{sr}")
        print(f"| {scope} | {sw} | {sr} | {'; '.join(f'`{u}`' for u in uniq)}"
              f"{' (identical to ' + same + ')' if same else ''} |", flush=True)

    print(f"\nRuns, {launches} launches per row, 0-10 us jitter. Data with write-through bits and a drain "
          "(visibility of the data is not in question, only the flag's), then default data bits, where the "
          "atomic's own release / acquire must also make the data visible.\n")
    print("| placement | data bits | scope | writer sem | reader sem | observed | fresh % | not fresh / lines | "
          "timeouts | median spins | checks |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for place in (1, 2):
        for data in ("sc0sc1", "none"):
            for scope, sw, sr in variants:
                wf = "wg" if data == "sc0sc1" else "none"
                st, _ = run_cell(ws, launches=launches, store=data, load=data, place=place, sync=1, wfence=wf,
                                 fscope=scope, fsem_w=sw, fsem_r=sr, jitter_us=10.0, t_timeout_us=500.0)
                sp = statistics.median(st.spins) if st.spins else 0
                print(f"| {PLACES[place]} | {bits_md(data)} | {scope} | {sw} | {sr} | {st.verdict()} | "
                      f"{st.pct(st.fresh, st.lines):.3f} | {st.stale + st.other} / {st.lines} | {st.timeouts} | {sp:.0f} "
                      f"| {fmt_checks(st)} |", flush=True)


def sweep_third(ws, launches=50):
    """Is an agent-scope cache operation local to the issuing XCD's L2, or device-wide? Writers on XCDs
    0, 2, 4, 6 store with default bits (dirty in their own L2); readers on XCDs 1, 3, 5, 7 read at 70 us
    with `sc0 nt` (L1 miss, L2 hit), no fences. At 40 us every program on one XCD issues the operation.
    Results per (writer XCD, reader XCD) pair of XCDs."""
    f = K.FIELD
    print(f"Default stores on XCDs 0/2/4/6, `sc0 nt` reads (primed) on XCDs 1/3/5/7, no fences. At ts + 40 us "
          f"every program on one XCD issues the operation (then `s_waitcnt vmcnt(0)`). {launches} launches.\n")
    print("| operation | issued on XCD | writer -> reader XCDs | relation of the issuing XCD | fresh % | "
          "not fresh / lines | op cycles (median) |")
    print("|---|---|---|---|---|---|---|")
    for third, opname in ((1, "`buffer_inv sc1`"), (2, "`buffer_wbl2 sc1`"), (0, "none")):
        for txcc in ((3, 2) if third else (3, )):
            buckets = {}
            ops = []
            for i in range(launches + 1):
                ws.flush()
                K.launch(ws, store="none", load="sc0nt", place=2, third=third, third_xcc=txcc)
                torch.cuda.synchronize()
                if i == 0:
                    continue
                rec, x0, x1 = K.results(ws, 2)
                pr = ws.prog_rec.view(K.NUM_PROGRAMS, 4).cpu()
                ops += pr[pr[:, 1] == txcc, 0].tolist()
                obs = torch.cat([rec[:, f["timed_val"]].to(torch.int32)[:, None], x1[:, 1:]], 1)
                fresh = obs == ws.epoch
                for p in range(rec.shape[0]):
                    key = (int(rec[p, f["w_xcc"]]), int(rec[p, f["r_xcc"]]))
                    b = buckets.setdefault(key, [0, 0])
                    b[0] += int(fresh[p].sum())
                    b[1] += obs.shape[1]
            for (wx, rx), (fr, n) in sorted(buckets.items()):
                if third == 0:
                    rel = "-"
                elif txcc == rx:
                    rel = "the reader's"
                elif txcc == wx:
                    rel = "the writer's"
                else:
                    rel = "neither"
                opc = statistics.median(ops) if ops else float("nan")
                print(f"| {opname} | {txcc if third else '-'} | {wx} -> {rx} | {rel} | {100 * fr / n:.2f} | {n - fr} / {n} "
                      f"| {opc:.0f} |", flush=True)


def sweep_acqgap(ws, launches=30):
    """The reader's agent acquire (buffer_inv sc1; s_waitcnt vmcnt(0)) issued `gap` us before its read
    instead of just before it."""
    print(f"The reader applies `buffer_inv sc1` + `s_waitcnt vmcnt(0)` gap us before its read at ts + 70 us "
          f"(gap 0: just before). No writer fence. {launches} launches per cell.\n")
    print("| placement | store | load | gap us | observed | fresh % | not fresh / lines | timed ns | tier |")
    print("|---|---|---|---|---|---|---|---|---|")
    refs = references(ws, launches=10, verbose=False)
    for place, s, l in [(2, "sc0sc1", "none"), (1, "none", "none"), (2, "none", "none"), (2, "none", "sc0nt")]:
        for gap in (0, 0.5, 1, 2, 5, 10, 30):
            st, _ = run_cell(ws, launches=launches, store=s, load=l, place=place, rfence="agent_wait", rgap_us=gap)
            ns = st.lat()
            print(f"| {PLACES[place]} | {bits_md(s)} | {bits_md(l)} | {gap} | {st.verdict()} | "
                  f"{st.pct(st.fresh, st.lines):.2f} | {st.stale + st.other} / {st.lines} | {ns:.0f} | {tier(refs, ns)} |",
                  flush=True)


def sweep_primebits(ws, launches=100, refs=None):
    """Write-through stores read with `sc0 nt`: does the tier of the read depend on the reader's prime bits
    (stage 1 primes with the load bits), and on whether the writer's L2 holds the line?"""
    refs = refs or references(ws, launches=20, verbose=False)
    print(f"`sc0 sc1` (`.wt`) stores, no fences, time-separated as in stage 1. The prime bits vary separately from "
          f"the load bits; 'none' primes with default loads, 'no prime' skips the prime (phases 6: an old line "
          f"then counts as other, not stale). 'writer reads too' (phases 15) has the writer read its own lines "
          f"with the prime bits at prime time. {launches} launches per cell, a 1 GiB read before each. Reference "
          "tiers (median ns): " + ", ".join(f"{t} {v:.0f}" for t, v in refs.items()) + ".\n")
    print("| placement | prime | writer reads too | load | observed | fresh % | stale % | other % | not fresh / lines | "
          "timed read ns | tier | checks |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    cells = [(place, prime, 7, "sc0nt") for place in PLACES for prime in ("sc0nt", "none", "sc0sc1")]
    cells += [(place, None, 6, load) for load in ("sc0nt", "sc1", "sc0sc1") for place in PLACES]
    cells += [(1, "sc0nt", 7, "sc0sc1"), (1, "sc0nt", 7, "sc1"), (2, "sc0nt", 15, "sc0nt"), (2, "none", 15, "sc0nt")]
    for place, prime, phases, load in cells:
        st, k = run_cell(ws, launches=launches, store="sc0sc1", load=load, prime=prime or load, place=place,
                         phases=phases)
        lds, sts = asm_bits(k)
        ok = lds == {K.BITS[load], K.BITS[prime or load]} and sts == {K.BITS["sc0sc1"], ""}
        chk = fmt_checks(st) + ("" if ok else ", BITS MISMATCH")
        ns = st.lat()
        pr = "no prime" if phases == 6 else bits_md(prime)
        print(f"| {PLACES[place]} | {pr} | {'yes' if phases & 8 else 'no'} | {bits_md(load)} | {st.verdict()} | "
              f"{fmt_counts(st)} | {ns:.0f} | {tier(refs, ns)} | {chk} |", flush=True)


SWEEPS = {
    "check": sweep_check,
    "refs": sweep_refs,
    "controls": sweep_controls,
    "stage1": sweep_stage1,
    "pollute": sweep_pollute,
    "stage2a": sweep_stage2a,
    "third": sweep_third,
    "acqgap": sweep_acqgap,
    "stage2b": sweep_stage2b,
    "stage2c": sweep_stage2c,
    "primebits": sweep_primebits,
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", nargs="+", default=list(SWEEPS), choices=list(SWEEPS))
    args = p.parse_args()
    print(f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', 'unset')}, "
          f"{torch.cuda.get_device_name()}, triton {K.triton.__version__}, {time.strftime('%Y-%m-%d %H:%M')}\n")
    print("```text\n" + gpu_snapshot() + "\n```")
    ws = K.Workspace()
    for s in args.sweep:
        print(f"\n## {s}\n", flush=True)
        SWEEPS[s](ws)


if __name__ == "__main__":
    main()
