#!/usr/bin/env python3
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

"""
Draw the stream-K phase of the a16w16 v15-v20 GEMMs from ATT traces: the data-parallel to
stream-K transition, the stream-K segments, and each segment's ending (partial store, flag
release, spin on peers' flags or the tile barrier, partial reads, C store).

plot_pipeline_schedule.py models one persistent tile's buffers from the kernel source; stream-K
segments start at any k-step and end in code that differs per version, so this script reads
the executed instruction stream instead. Every workgroup's four waves are analysed together,
because one wave polls the flags and broadcasts through LDS while the other three wait at a
barrier. Instructions are classified by mnemonic:

  buffer_load ... lds              global->LDS operand load
  buffer_store ... sc0 sc1         partial store to P (.wt)
  buffer_load ... sc0 sc1          partial or slice read from P (.cv)
  global_load_dword ... sc1        relaxed flag / barrier poll
  global_atomic_*                  flag release, tile-barrier arrival, flag re-arm
  buffer_wbl2 / buffer_inv         release / acquire
  buffer_store (no cache bits)     C store

The first n_dp * 16384 MFMAs of a wave (128 k-steps x 128 MFMAs per wave) are its data-parallel
tiles; the rest are cut into stream-K segments wherever a store, partial read, poll or atomic
falls between two MFMAs. Each traced workgroup's role (owner, contributor, reduce-scatter or a
whole tile) and k-step counts are matched against a replay of the version's partition to list
the stream-K pids it can be; the XCD is taken as SE // 4.

Capture: one rocprofv3 ATT run per att_target_cu (CU 0 is traced on the 16 even SEs, CUs 1 and
2 on all 32), with "kernel_include_regex" set to the version's kernel and
"att_shader_engine_mask": "0xFFFFFFFF", under the v10+ stack (Triton v2.2 + deepbind shim,
llirSched, amdgcnas):

    python scripts/run_att.py --att-output out python <a16w16>/bench.py --version 17 \\
        --M 4352 --N 8192 --K 8192 --dtype bf16

Each capture is a separate launch, so each one's time axis starts at its own first wave.

    python scripts/plot_streamk_schedule.py --version v17_streamk_tile_aligned \\
        --att-dirs <cu0 out> <cu1 out> <cu2 out> --out-dir <dir>
    python scripts/plot_streamk_schedule.py --compare <dir> [<dir> ...] --out ending_compare.png
"""

import argparse
import glob
import html
import json
import os
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
from plot_pipeline_schedule import KINDS, Canvas, to_png, to_svg, write_html  # noqa: E402

P = 256
NUM_XCDS = 8
MFMAS_PER_KSTEP = 128  # per wave: 4 regions x 32 MFMAs
MFMA_ISSUE = 16
LEFT = 330
RIGHT = 30

VERSIONS = {
    "v15_streamk_onetile": dict(partition="e2e", fixup="owner",
                                note="one-tile, end-to-end split, row-major partials"),
    "v16_streamk_lane_partials": dict(partition="e2e", fixup="owner",
                                      note="one-tile, end-to-end split, lane-contiguous partials"),
    "v17_streamk_tile_aligned": dict(partition="aligned", fixup="owner",
                                     note="one-tile, tile-aligned split, chunk-0 owner"),
    "v18_streamk_chunk_major": dict(partition="chunk_major", fixup="owner",
                                    note="one-tile, chunk-major split, rotating owner"),
    "v19_streamk_two_tile_reversed": dict(partition="e2e_rev", fixup="owner",
                                          note="two-tile, end-to-end split, reversed order"),
    "v20_streamk_reduce_scatter": dict(partition="aligned", fixup="rs",
                                       note="one-tile, tile-aligned split, reduce-scatter ending "
                                            "(STREAMK_POLICY=one_tile STREAMK_FIXUP=rs)"),
}  # fmt: skip

LANES = [
    ("wait", "Waits (s_waitcnt, s_barrier)", KINDS["wait"][0]),
    ("load", "Global->LDS loads", KINDS["load"][0]),
    ("read", "LDS->register reads", KINDS["read"][0]),
    ("mfma", "MFMA", KINDS["mfma"][0]),
    ("pstore", "Partial store (P, .wt)", "#db2777"),
    ("pread", "Partial / slice reads (P, .cv)", "#0d9488"),
    ("sync", "Flag poll, atomics, wbl2 / inv (any wave)", "#d97706"),
    ("store", "C stores", KINDS["store"][0]),
]
LANE_COLOR = {k: c for k, _, c in LANES}

PHASES = {
    "data-parallel tile": "#e5e7eb",
    "transition": "#fef3c7",
    "MAC": "#c4b5fd",
    "drain": "#d1d5db",
    "partial store": "#f9a8d4",
    "store drain": "#be185d",
    "spin on flags": "#fbbf24",
    "barrier wait": "#f97316",
    "collect": "#5eead4",
    "slice": "#2dd4bf",
    "C store": "#fca5a5",
    "exit": "#f3f4f6",
}
PHASE_TIPS = {
    "transition": "from the last data-parallel MFMA to the first MFMA of the next segment: "
                  "C store / partial ending, drain, and the new segment's prologue",
    "MAC": "the segment's k-step loop",
    "drain": "drain_and_barrier after the K loop (the clamped prefetch), up to the ending's "
             "first memory operation",
    "partial store": "fp32 partial to P[spid] (.wt), all four waves",
    "store drain": "s_waitcnt vmcnt(0) + barrier until the partial stores have completed, then "
                   "buffer_wbl2: the release before the flag / barrier arrival",
    "spin on flags": "relaxed polls of the peers' flags (one wave polls, the others wait at a "
                     "barrier), up to the acquire (buffer_inv)",
    "barrier wait": "tile barrier: arrival atomic, then polls of the generation, up to the "
                    "acquire (buffer_inv)",
    "collect": "the peers' partials read (.cv) and added, quadrant by quadrant, interleaved "
               "with the C stores",
    "slice": "reduce-scatter: this program's 1/n of the tile read from all n partials and "
             "stored to C",
    "C store": "C stores of a whole tile (no peers)",
    "exit": "flag release or re-arm and the end of the wave (or the start of the next segment)",
}  # fmt: skip


# --------------------------------------------------------------------------
# Partition replay (pairs of k-steps, as the kernels hand them out)
# --------------------------------------------------------------------------


def replay(scheme, fixup, total_tiles, ppt):
    """(n_dp, sk_tiles, {spid: [(k-steps, role, tile, n_peers), ...] in processing order})."""
    S = total_tiles % P
    sk = S + P if scheme == "e2e_rev" and S and total_tiles >= P else S
    n_dp = (total_tiles - sk) // P
    progs = {}
    if scheme in ("e2e", "e2e_rev"):
        L, R = sk * ppt // P, sk * ppt % P

        def owner_of(i):
            head = R * (L + 1)
            return i // (L + 1) if (L == 0 or i < head) else R + (i - head) // L

        for spid in range(P):
            lo, hi = spid * L + min(spid, R), (spid + 1) * L + min(spid + 1, R)
            segs = []
            if scheme == "e2e":
                cur = lo
                while cur < hi:
                    t = cur // ppt
                    ts, e = t * ppt, min(t * ppt + ppt, hi)
                    peers = owner_of(ts + ppt - 1) - spid if cur == ts else 0
                    role = "partial" if cur != ts else ("owner" if peers else "whole")
                    segs.append((2 * (e - cur), role, t, peers))
                    cur = e
            else:
                cur = hi
                while cur > lo:
                    t = (cur - 1) // ppt
                    ts = t * ppt
                    s = max(ts, lo)
                    owner = cur == ts + ppt
                    peers = spid - owner_of(ts) if owner else 0
                    role = "partial" if not owner else ("owner" if peers else "whole")
                    segs.append((2 * (cur - s), role, t, peers))
                    cur = s
            progs[spid] = segs
    else:
        n_lo, n_hi = min(P // S, ppt), min(P // S + 1, ppt)
        extra = P % S if n_hi > n_lo else 0
        sk_programs = extra * n_hi + (S - extra) * n_lo
        for spid in range(sk_programs):
            if scheme == "aligned":
                head = extra * n_hi
                if spid < head:
                    t, j, n = spid // n_hi, spid % n_hi, n_hi
                else:
                    t, j, n = extra + (spid - head) // n_lo, (spid - head) % n_lo, n_lo
                owner_j = 0
            else:
                t, j = spid % S, spid // S
                if spid >= n_lo * S:
                    t, j = spid - n_lo * S, n_lo
                n = n_hi if t < extra else n_lo
                owner_j = t % n
            q, r = ppt // n, ppt % n
            pairs = q + (j < r)
            if n == 1:
                role, peers = "whole", 0
            elif fixup == "rs":
                role, peers = "rs", n - 1
            else:
                role, peers = ("owner", n - 1) if j == owner_j else ("partial", 0)
            progs[spid] = [(2 * pairs, role, t, peers)]
        for spid in range(sk_programs, P):
            progs[spid] = []
    return n_dp, sk, progs


# --------------------------------------------------------------------------
# ATT trace -> workgroups
# --------------------------------------------------------------------------


def kind_of(name):
    op = name.split(None, 1)[0]
    if op.startswith("v_mfma"):
        return "mfma"
    if op.startswith("buffer_load") and name.endswith(" lds"):
        return "load"
    if op.startswith(("buffer_load", "global_load")):
        if "sc0 sc1" in name:
            return "pread"
        if op.startswith("global_load") and " sc1" in name:
            return "poll"
        return "gload"
    if op.startswith(("buffer_store", "global_store")):
        return "pstore" if "sc0 sc1" in name else "store"
    if "_atomic" in op:
        return "atomic"
    if op.startswith("buffer_wbl2"):
        return "wbl2"
    if op.startswith("buffer_inv"):
        return "inv"
    if op.startswith("s_waitcnt"):
        return "wait"
    if op.startswith("s_barrier"):
        return "barrier"
    if op.startswith("ds_read"):
        return "ds_read"
    if op.startswith("ds_write"):
        return "ds_write"
    if op == "s_endpgm":
        return "end"
    return "other"


SYNC = ("poll", "atomic", "wbl2", "inv")
SPLIT = ("pstore", "store", "pread", "poll", "atomic")


def load_capture(att_dir, tag):
    ui = sorted(glob.glob(os.path.join(att_dir, "ui_*")))
    if not ui:
        raise SystemExit(f"no ui_* directory under {att_dir}")
    code = json.load(open(os.path.join(ui[0], "code.json")))["code"]
    names = {r[2]: r[0] for r in code}
    kinds = {i: kind_of(n) for i, n in names.items()}
    groups = {}
    for f in sorted(glob.glob(os.path.join(ui[0], "se*_wv*.json"))):
        w = json.load(open(f))["wave"]
        se = int(os.path.basename(f).split("_")[0][2:])
        ins = w["instructions"]
        groups.setdefault((se, w["cu"]), []).append({
            "file": os.path.basename(f), "simd": w["simd"], "begin": w["begin"], "end": w["end"],
            "t": [x[0] for x in ins], "idx": [x[4] for x in ins],
            "k": [kinds[x[4]] for x in ins],
        })  # fmt: skip
    wgs = []
    for (se, cu), waves in sorted(groups.items()):
        waves.sort(key=lambda w: w["simd"])
        wgs.append({"capture": tag, "se": se, "cu": cu, "xcd": se // 4, "waves": waves,
                    "names": names})  # fmt: skip
    origin = min(w["begin"] for g in wgs for w in g["waves"])
    for g in wgs:
        for w in g["waves"]:
            w["t"] = [t - origin for t in w["t"]]
            w["begin"] -= origin
            w["end"] -= origin
        g["end"] = max(w["end"] for w in g["waves"])
        g["begin"] = min(w["begin"] for w in g["waves"])
    return wgs


def first_at(times, lo, hi=float("inf")):
    return next((t for t in times if lo <= t < hi), None)


def analyse_wg(g, n_dp):
    """Data-parallel tiles, stream-K segments and their ending phases (WG-wide landmarks)."""
    w0 = g["waves"][0]
    t, k = w0["t"], w0["k"]
    mfma = [i for i, x in enumerate(k) if x == "mfma"]
    n_dp_mfma = n_dp * 16384
    if len(mfma) < n_dp_mfma:
        return None
    sk = mfma[n_dp_mfma:]
    assert len(sk) % (2 * MFMAS_PER_KSTEP) == 0, (g["se"], len(sk))
    dp_last = mfma[n_dp_mfma - 1] if n_dp else None
    # Steady-state k-step from the last data-parallel tile
    ks_cycles = None
    if n_dp:
        a = t[mfma[n_dp_mfma - 16384]]
        ks_cycles = (t[mfma[n_dp_mfma - 1]] + MFMA_ISSUE - a) / 128

    runs = []
    for i in sk:
        if runs and not any(x in SPLIT for x in k[runs[-1][-1] + 1 : i]):
            runs[-1].append(i)
        else:
            runs.append([i])
    # All waves' events, by kind, as sorted times
    ev = {}
    for w in g["waves"]:
        for tt, kk in zip(w["t"], w["k"]):
            if kk in SYNC + ("pstore", "pread", "store", "load"):
                ev.setdefault(kk, []).append(tt)
    for v in ev.values():
        v.sort()

    segs = []
    prev = dp_last
    for r_i, run in enumerate(runs):
        lo = t[prev] + 1 if prev is not None else 0
        pro = next((t[i] for i in range(prev + 1 if prev is not None else 0, run[0])
                    if k[i] == "load"), t[run[0]])  # fmt: skip
        mac0, mac1 = t[run[0]], t[run[-1]] + MFMA_ISSUE
        nxt_pro = None
        if r_i + 1 < len(runs):
            nxt_pro = next((t[i] for i in range(run[-1] + 1, runs[r_i + 1][0]) if k[i] == "load"),
                           t[runs[r_i + 1][0]])  # fmt: skip
        hi = nxt_pro if nxt_pro is not None else g["end"]

        def win(kind, a=mac1, b=hi):
            return [x for x in ev.get(kind, []) if a <= x < b]

        ps, pr, cs = win("pstore"), win("pread"), win("store")
        polls, atoms, wbl2, inv = win("poll"), win("atomic"), win("wbl2"), win("inv")
        role = "rs" if ps and pr else "partial" if ps else "owner" if pr else "whole"
        marks = [("drain", mac1)]
        if ps:
            marks += [("partial store", ps[0]), ("store drain", ps[-1] + MFMA_ISSUE)]
            rel = first_at(atoms, first_at(wbl2, ps[-1]) or ps[-1])
            if role == "partial" and rel is not None:
                marks.append(("exit", rel))
            if role == "rs":
                acq = [x for x in inv if x < pr[0]] if pr else []
                if rel is not None:
                    marks.append(("barrier wait", rel))
                marks.append(("slice", acq[-1] if acq else pr[0]))
        if role == "owner":
            acq = [x for x in inv if x < pr[0]]
            if polls and polls[0] < pr[0]:
                marks.append(("spin on flags", polls[0]))
            marks.append(("collect", acq[-1] if acq else pr[0]))
        if role == "whole" and cs:
            marks.append(("C store", cs[0]))
        if role in ("owner", "rs", "whole") and cs:
            marks.append(("exit", cs[-1] + MFMA_ISSUE))
        marks.sort(key=lambda m: m[1])
        phases = []
        for (lab, a), (_, b) in zip(marks, marks[1:] + [("", hi)]):
            if b > a:
                phases.append((lab, a, b))
        segs.append({
            "ksteps": len(run) // MFMAS_PER_KSTEP, "role": role, "prologue": pro, "lo": lo,
            "mac0": mac0, "mac1": mac1, "end": hi, "phases": phases,
            "n_polls": len(polls), "n_pstore": len(ps), "n_pread": len(pr), "n_store": len(cs),
        })  # fmt: skip
        prev = run[-1]
    return {
        "ks_cycles": ks_cycles,
        "dp_tiles": [(t[mfma[i * 16384]], t[mfma[i * 16384 + 16383]] + MFMA_ISSUE)
                     for i in range(n_dp)],
        "dp_end": t[dp_last] + MFMA_ISSUE if dp_last is not None else 0,
        "segs": segs,
    }  # fmt: skip


def candidates(g, progs):
    sig = [(s["ksteps"], s["role"]) for s in g["a"]["segs"]]
    xcd = [s for s in range(g["xcd"] * 32, g["xcd"] * 32 + 32)]
    match = [s for s in xcd if [(n, r) for n, r, _, _ in progs[s]] == sig]
    anywhere = match or [s for s in progs if [(n, r) for n, r, _, _ in progs[s]] == sig]
    return match, anywhere


def fmt_spids(sp):
    if not sp:
        return "none"
    if len(sp) > 6:
        return f"{len(sp)} pids ({sp[0]}..{sp[-1]})"
    return ",".join(map(str, sp))


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------


class ScaledXMap:
    """Piecewise-linear cycles -> pixels over visible segments, with breaks between them."""

    def __init__(self, segments, scale, left=LEFT, brk=60):
        self.segs, self.scale = [], scale
        x = left
        for i, (a, b) in enumerate(segments):
            self.segs.append((a, b, x))
            x += (b - a) * scale
            if i < len(segments) - 1:
                x += brk
        self.width = x + RIGHT
        self.brk = brk

    def pieces(self, t0, t1):
        for a, b, x in self.segs:
            lo, hi = max(t0, a), min(t1, b)
            if lo < hi:
                yield x + (lo - a) * self.scale, (hi - lo) * self.scale

    def x(self, t):
        for a, b, x in self.segs:
            if a <= t <= b:
                return x + (t - a) * self.scale
        return None


def clusters(times, gap, width=MFMA_ISSUE):
    out = []
    for tt in times:
        if out and tt - out[-1][1] <= gap:
            out[-1][1] = tt + width
            out[-1][2] += 1
        else:
            out.append([tt, tt + width, 1])
    return out


def rect(cv, xm, a, b, y, h, color, tip=None, kind=None, min_w=1.0, opacity=1.0, stroke=None):
    for x, w in xm.pieces(a, b):
        cv.rect(x, y, max(w, min_w), h, color, kind=kind, tip=tip, opacity=opacity, stroke=stroke,
                sw=0.6 if stroke else 0)  # fmt: skip


def phase_bar(cv, xm, g, y, h, ks, labels=True):
    a = g["a"]
    for i, (t0, t1) in enumerate(a["dp_tiles"]):
        rect(cv, xm, t0, t1, y, h, PHASES["data-parallel tile"],
             tip=f"<b>data-parallel tile {i}</b><br>{(t1 - t0) / ks:.1f} k-steps")  # fmt: skip
    prev = a["dp_end"] if a["dp_tiles"] else None
    for si, s in enumerate(a["segs"]):
        if prev is not None:
            rect(cv, xm, prev, s["mac0"], y, h, PHASES["transition"],
                 tip=f"<b>transition into segment {si}</b><br>{(s['mac0'] - prev) / ks:.1f} "
                     f"k-steps of MFMA idle<br>{PHASE_TIPS['transition']}")  # fmt: skip
        rect(cv, xm, s["mac0"], s["mac1"], y, h, PHASES["MAC"],
             tip=f"<b>segment {si} ({s['role']}): MAC, {s['ksteps']} k-steps</b><br>"
                 f"{(s['mac1'] - s['mac0']) / ks:.1f} DP k-steps of time "
                 f"({(s['mac1'] - s['mac0']) / s['ksteps']:.0f} cycles per k-step)")  # fmt: skip
        for lab, p0, p1 in s["phases"]:
            rect(cv, xm, p0, p1, y, h, PHASES[lab],
                 tip=f"<b>{lab}</b> ({s['role']} ending)<br>{(p1 - p0) / ks:.2f} k-steps "
                     f"({p1 - p0} cycles)<br>{PHASE_TIPS.get(lab, '')}")  # fmt: skip
            if labels:
                xs = list(xm.pieces(p0, p1))
                if xs and xs[0][1] > 7 * len(lab):
                    cv.text(xs[0][0] + xs[0][1] / 2, y + h / 2 + 4, lab, size=10,
                            anchor="middle", color="#111827")  # fmt: skip
        prev = s["end"]


def legend(cv, x0, y, items, size=11):
    x = x0
    for label, color in items:
        cv.rect(x, y, 13, 13, color, stroke="#9ca3af", sw=0.5)
        cv.text(x + 18, y + 11, label, size=size)
        x += 28 + 6.2 * len(label)
    return y + 22


def axis(cv, xm, y, ref, ks, label, first_label=None):
    cv.line(xm.segs[0][2], y, xm.width - RIGHT, y, color="#374151", width=1)
    if first_label:
        cv.text(xm.segs[0][2], y + 18, first_label, size=10, color="#374151")
    for a, b, x in xm.segs[1:] if first_label else xm.segs:
        span = (b - a) / ks
        step = next(s for s in (1, 2, 5, 10, 20, 50, 100) if span / s <= 14)
        n = int((a - ref) / ks // step) + 1
        while ref + n * step * ks < b:
            tx = xm.x(ref + n * step * ks)
            if tx is not None:
                cv.line(tx, y, tx, y + 5, color="#374151", width=1)
                cv.text(tx, y + 18, f"{n * step:g}", size=10, anchor="middle", color="#374151")
            n += 1
    cv.text(xm.segs[0][2], y + 34, label, size=11, color="#4b5563")


def draw_overview(wgs, ks, title, notes):
    cv = Canvas()
    end = max(g["end"] for g in wgs)
    boundary = min(g["a"]["dp_end"] for g in wgs) if wgs[0]["a"]["dp_tiles"] else 0
    first = min(g["begin"] for g in wgs)
    scale = 1900 / (end - (boundary - 6 * ks))
    segments = [(first, first + 6 * ks), (boundary - 6 * ks, end)] if boundary else [(0, end)]
    xm = ScaledXMap(segments, scale)
    cv.text(12, 26, title, size=17, weight="bold")
    for i, n in enumerate(notes):
        cv.text(12, 46 + 16 * i, n, size=11, color="#4b5563")
    y = 52 + 16 * len(notes)
    y = legend(cv, 12, y, [(k, c) for k, c in PHASES.items()])
    y += 8
    row_h, gap = 13, 3
    top = y
    for g in wgs:
        cv.text(LEFT - 8, y + 10, g["label"], size=10, anchor="end")
        phase_bar(cv, xm, g, y, row_h, ks, labels=False)
        y += row_h + gap
    for i in range(len(xm.segs) - 1):
        a, b, x = xm.segs[i]
        bx = x + (b - a) * scale
        cv.rect(bx + 4, top - 4, xm.brk - 8, y - top + 4, "#f9fafb", stroke="#d1d5db", sw=1)
        cv.text(bx + xm.brk / 2, top + 14, "data-", size=10, anchor="middle", color="#6b7280")
        cv.text(bx + xm.brk / 2, top + 28, "parallel", size=10, anchor="middle", color="#6b7280")
    axis(cv, xm, y + 4, boundary, ks,
         "k-steps (data-parallel steady state) from the first workgroup's last data-parallel "
         "MFMA; each capture's clock starts at its first wave",
         first_label="kernel start" if len(xm.segs) > 1 else None)  # fmt: skip
    return cv, xm.width, y + 50


def draw_blocks(blocks, ks, title, notes, max_px=2300):
    """Swimlanes for (label, wg, window, ref) blocks, stacked, on one scale."""
    cv = Canvas()
    span = max(b - a for _, _, (a, b), _ in blocks)
    scale = min(0.07, max_px / span)
    cv.text(12, 26, title, size=17, weight="bold")
    for i, n in enumerate(notes):
        cv.text(12, 46 + 16 * i, n, size=11, color="#4b5563")
    y = 52 + 16 * len(notes)
    y = legend(cv, 12, y, [(lab, c) for _, lab, c in LANES])
    y = legend(cv, 12, y, [(k, c) for k, c in PHASES.items() if k != "data-parallel tile"])
    LANE_H, LANE_GAP = 22, 4
    W = 0
    for label, g, (a, b), ref in blocks:
        y += 30
        cv.text(12, y, label, size=13, weight="bold")
        y += 12
        xm = ScaledXMap([(a, b)], scale)
        W = max(W, xm.width)
        cv.text(LEFT - 8, y + 12, "phase", size=11, anchor="end")
        phase_bar(cv, xm, g, y, 18, ks)
        y += 26
        lane_y = {}
        for i, (lane, name, color) in enumerate(LANES):
            ly = y + i * (LANE_H + LANE_GAP)
            lane_y[lane] = ly
            cv.rect(LEFT, ly, (b - a) * scale, LANE_H, "#f9fafb")
            cv.text(LEFT - 8, ly + LANE_H / 2 + 4, name, size=11, anchor="end")
        draw_lanes(cv, xm, g, a, b, lane_y, LANE_H)
        y += len(LANES) * (LANE_H + LANE_GAP)
        axis(cv, xm, y + 2, ref[0], ks, ref[1])
        y += 40
    return cv, max(W, 1200), y + 10


def draw_lanes(cv, xm, g, a, b, lane_y, h):
    names = g["names"]
    w0 = g["waves"][0]
    t, k, idx = w0["t"], w0["k"], w0["idx"]
    sel = [i for i, tt in enumerate(t) if a <= tt < b]
    by = {}
    for i in sel:
        by.setdefault(k[i], []).append(i)
    # Waits: duration until the next instruction
    for i in by.get("wait", []) + by.get("barrier", []):
        d = (t[i + 1] if i + 1 < len(t) else w0["end"]) - t[i]
        if d >= 24:
            rect(cv, xm, t[i], t[i] + d, lane_y["wait"] + 2, h - 4, LANE_COLOR["wait"],
                 tip=f"<b>{html.escape(names[idx[i]])}</b><br>{d} cycles (wave {w0['simd']})")  # fmt: skip
    spec = [("mfma", "mfma", 40), ("load", "load", 64), ("ds_read", "read", 32),
            ("pstore", "pstore", 96), ("pread", "pread", 96), ("store", "store", 96)]  # fmt: skip
    for kind, lane, gap in spec:
        for c0, c1, n in clusters([t[i] for i in by.get(kind, [])], gap):
            what = {"mfma": "MFMAs", "load": "buffer_load ... lds", "ds_read": "ds_read",
                    "pstore": "partial stores (sc0 sc1)", "pread": "partial reads (sc0 sc1)",
                    "store": "C stores"}[kind]  # fmt: skip
            rect(cv, xm, c0, c1, lane_y[lane] + 2, h - 4, LANE_COLOR[lane], min_w=1.5,
                 tip=f"<b>{n} x {what}</b> (wave {w0['simd']})<br>{c0}-{c1} cycles ({c1 - c0})")  # fmt: skip
    # Sync events of every wave in the workgroup
    for w in g["waves"]:
        for tt, kk, ii in zip(w["t"], w["k"], w["idx"]):
            if a <= tt < b and kk in SYNC:
                color = {"poll": "#d97706", "atomic": "#78350f", "wbl2": "#1d4ed8",
                         "inv": "#059669"}[kk]  # fmt: skip
                hh = h - 4 if kk != "poll" else (h - 4) / 2
                rect(cv, xm, tt, tt + 8, lane_y["sync"] + 2, hh, color, min_w=2.0,
                     tip=f"<b>{html.escape(names[ii])}</b><br>wave {w['simd']}, {tt} cycles")  # fmt: skip


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def representative(wgs):
    """One workgroup per role signature (the one with the median end), most common first."""
    groups = {}
    for g in wgs:
        groups.setdefault(g["sig"], []).append(g)
    out = []
    for sig, gs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        gs = sorted(gs, key=lambda g: g["end"] - g["a"]["segs"][-1]["mac1"])
        out.append((sig, gs[len(gs) // 2], gs))
    return out


def sig_text(sig):
    return " + ".join(f"{n}-k-step {r}" for n, r in sig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--version", choices=sorted(VERSIONS))
    ap.add_argument("--att-dirs", nargs="+", help="rocprofv3 -d directories (one per capture)")
    ap.add_argument("--shape", default="4352 8192 8192", help="M N K")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--out-dir")
    ap.add_argument("--compare", nargs="+", help="output directories of earlier runs")
    ap.add_argument("--out", help="--compare: output PNG")
    args = ap.parse_args()
    if args.compare:
        return compare(args.compare, args.out)

    cfg = VERSIONS[args.version]
    M, N, K = map(int, args.shape.split())
    total_tiles = -(-M // 256) * -(-N // 256)
    ppt = K // 128
    n_dp, sk_tiles, progs = replay(cfg["partition"], cfg["fixup"], total_tiles, ppt)
    os.makedirs(args.out_dir, exist_ok=True)

    wgs = []
    for ci, d in enumerate(args.att_dirs):
        for g in load_capture(d, f"c{ci}"):
            g["a"] = analyse_wg(g, n_dp)
            if g["a"] is None or not g["a"]["segs"]:
                continue
            g["sig"] = tuple((s["ksteps"], s["role"]) for s in g["a"]["segs"])
            g["cand"], g["cand_any"] = candidates(g, progs)
            wgs.append(g)
    ks = statistics.median(g["a"]["ks_cycles"] for g in wgs) if n_dp else 2093.0
    role_order = {"owner": 0, "rs": 1, "partial": 2, "whole": 3}
    wgs.sort(key=lambda g: (role_order[g["sig"][-1][1]], g["sig"], g["capture"], g["xcd"], g["se"]))
    for g in wgs:
        roles = "+".join(f"{n}{r[0].upper()}" for n, r in g["sig"])
        cand = fmt_spids(g["cand"]) if g["cand"] else f"({fmt_spids(g['cand_any'])}, other XCD)"
        g["label"] = (f"{g['capture']} se{g['se']} cu{g['cu']} xcd{g['xcd']}  {roles}  "
                      f"spid {cand}")  # fmt: skip

    S = total_tiles % P
    shape = f"{M}x{N}x{K} {args.dtype}"
    common = [
        f"a16w16 {args.version}: {cfg['note']}. {shape}: {total_tiles} tiles = {total_tiles // P} "
        f"waves + S = {S}; {n_dp} data-parallel tile(s) per program, {sk_tiles} stream-K tiles.",
        f"ATT, {len(wgs)} traced workgroups from {len(args.att_dirs)} launches; one k-step = "
        f"{ks:.0f} cycles (the last data-parallel tile's median).",
    ]
    cv_o, W_o, H_o = draw_overview(
        wgs, ks, f"{args.version}: every traced workgroup, data-parallel to stream-K end",
        common + ["Rows: capture, SE, CU, XCD (SE // 4), segments (k-steps + Owner / Partial / "
                  "Rs / Whole), and the stream-K pids the partition allows on that XCD."],
    )  # fmt: skip

    reps = representative(wgs)
    picks = [r[1] for r in reps[:2]]
    if len(picks) == 1 and len(reps[0][2]) > 1:
        gs = reps[0][2]
        picks = [gs[0], gs[-1]]
    trans = []
    for g in picks:
        s0 = g["a"]["segs"][0]
        lo = (g["a"]["dp_end"] - 2.5 * ks) if n_dp else s0["prologue"] - ks
        trans.append((f"{g['label']}", g, (lo, s0["mac0"] + 3 * ks),
                      (g["a"]["dp_end"], "k-steps from the last data-parallel MFMA")))  # fmt: skip
    cv_t, W_t, H_t = draw_blocks(
        trans, ks, f"{args.version}: data-parallel to stream-K transition", common,
        max_px=2300,
    )  # fmt: skip
    ends = []
    for g in picks:
        segs = g["a"]["segs"]
        for si, s in enumerate(segs):
            if si + 1 < len(segs):
                ends.append((f"{g['label']}: segment {si} ending into segment {si + 1}", g,
                             (s["mac1"] - 2 * ks, segs[si + 1]["mac0"] + 2 * ks),
                             (s["mac1"], f"k-steps from segment {si}'s last MFMA")))  # fmt: skip
            else:
                ends.append((f"{g['label']}: segment {si} ending", g,
                             (s["mac1"] - 2 * ks, g["end"] + 0.3 * ks),
                             (s["mac1"], f"k-steps from segment {si}'s last MFMA")))  # fmt: skip
    cv_e, W_e, H_e = draw_blocks(ends, ks, f"{args.version}: stream-K endings", common)

    # Summary and metrics
    lines = [common[0], common[1], ""]
    metrics = {"version": args.version, "note": cfg["note"], "ks_cycles": ks, "shape": shape,
               "groups": []}  # fmt: skip
    for sig, rep, gs in reps:
        lines.append(f"{sig_text(sig)}: {len(gs)} workgroups")
        trans_idle = (
            [(g["a"]["segs"][0]["mac0"] - g["a"]["dp_end"]) / ks for g in gs] if n_dp else []
        )
        if trans_idle:
            lines.append(f"  DP -> SK MFMA idle: median {statistics.median(trans_idle):.1f} k-steps "
                         f"(min {min(trans_idle):.1f}, max {max(trans_idle):.1f})")  # fmt: skip
        grp = {"sig": sig_text(sig), "n": len(gs), "n_dp": n_dp, "segs": [],
               "dp_last": statistics.median((g["a"]["dp_tiles"][-1][1] - g["a"]["dp_tiles"][-1][0])
                                            / ks for g in gs) if n_dp else 0.0}  # fmt: skip
        for si in range(len(sig)):
            ss = [g["a"]["segs"][si] for g in gs]
            mac = [(s["mac1"] - s["mac0"]) / s["ksteps"] / ks for s in ss]
            lines.append(f"  segment {si} ({sig[si][0]} k-steps, {sig[si][1]}): MAC "
                         f"{statistics.median(mac):.2f} DP k-steps per k-step")  # fmt: skip
            labs = []
            for s in ss:
                for lab, _, _ in s["phases"]:
                    if lab not in labs:
                        labs.append(lab)
            med = {}
            for lab in labs:
                v = [sum(p1 - p0 for l2, p0, p1 in s["phases"] if l2 == lab) / ks for s in ss]
                med[lab] = (statistics.median(v), min(v), max(v))
            total = [(s["end"] - s["mac1"]) / ks for s in ss]
            lines.append("    ending: " + ", ".join(
                f"{lab} {m:.1f} [{lo:.1f}-{hi:.1f}]" for lab, (m, lo, hi) in med.items())
                + f"; total {statistics.median(total):.1f} [{min(total):.1f}-{max(total):.1f}] "
                "k-steps (median [min-max])")  # fmt: skip
            polls = [s["n_polls"] for s in ss]
            lines.append(f"    flag / barrier polls per workgroup: median {statistics.median(polls):g}; "
                         f"partial stores {ss[0]['n_pstore']}, partial reads "
                         f"{statistics.median([s['n_pread'] for s in ss]):g}, C stores "
                         f"{statistics.median([s['n_store'] for s in ss]):g} (all four waves)")  # fmt: skip
            grp["segs"].append({
                "ksteps": sig[si][0], "role": sig[si][1],
                "mac": statistics.median([(s["mac1"] - s["mac0"]) / ks for s in ss]),
                "transition": statistics.median(
                    [(s["mac0"] - (g["a"]["dp_end"] if si == 0 else g["a"]["segs"][si - 1]["end"]))
                     / ks for s, g in zip(ss, gs)]) if (n_dp or si) else 0.0,
                "phases": {lab: m for lab, (m, _, _) in med.items()},
            })  # fmt: skip
        metrics["groups"].append(grp)
        lines.append("")
    tail = [(g["end"] - g["a"]["dp_end"]) / ks for g in wgs]
    lines.append(f"stream-K phase per workgroup (last DP MFMA to wave end): median "
                 f"{statistics.median(tail):.1f}, max {max(tail):.1f} k-steps")  # fmt: skip
    lines.append("")
    lines.append("workgroups:")
    for g in wgs:
        a = g["a"]
        parts = []
        for s in a["segs"]:
            parts.append(f"[{s['ksteps']} {s['role']}: MAC {(s['mac1'] - s['mac0']) / ks:.1f}, "
                         + ", ".join(f"{lab} {(p1 - p0) / ks:.1f}" for lab, p0, p1 in s["phases"])
                         + "]")  # fmt: skip
        lines.append(f"  {g['label']}: DP end {a['dp_end'] / ks:.1f}, end {g['end'] / ks:.1f} "
                     f"k-steps; " + " ".join(parts))  # fmt: skip
    metrics["tail_median"] = statistics.median(tail)
    metrics["tail_max"] = max(tail)

    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    views = [("overview", "Overview (all traced workgroups)", to_svg(cv_o, W_o, H_o, "overview", True)),
             ("transition", "Transition", to_svg(cv_t, W_t, H_t, "transition", False)),
             ("ending", "Endings", to_svg(cv_e, W_e, H_e, "ending", False))]  # fmt: skip
    html_path = os.path.join(args.out_dir, "schedule.html")
    write_html(html_path, views, "<br>".join(html.escape(s) for s in lines[:40]))
    with open(html_path) as f:
        doc = f.read().replace("<title>v10 pipeline schedule</title>",
                               f"<title>{args.version} stream-K schedule</title>")  # fmt: skip
    with open(html_path, "w") as f:
        f.write(doc)
    to_png(cv_o, W_o, H_o, os.path.join(args.out_dir, "overview.png"), dpi=110)
    to_png(cv_t, W_t, H_t, os.path.join(args.out_dir, "transition.png"), dpi=110)
    to_png(cv_e, W_e, H_e, os.path.join(args.out_dir, "ending.png"), dpi=110)
    print("\n".join(lines[: lines.index("workgroups:")]))
    print(f"wrote {args.out_dir}")


def compare(dirs, out, max_groups=3):
    """Median critical path per role signature, from the end of the earliest data-parallel tile
    that every version shares: versions with more data-parallel tiles (one-tile) start with
    their extra tiles, so one-tile and two-tile rows cover the same work."""
    ms = [json.load(open(os.path.join(d, "metrics.json"))) for d in dirs]
    n_dp_min = min(g["n_dp"] for m in ms for g in m["groups"])
    rows = []
    for m in ms:
        short = m["version"].split("_")[0]
        groups = sorted(m["groups"], key=lambda g: -g["n"])
        total = sum(g["n"] for g in groups)
        for grp in groups[:max_groups]:
            rows.append((short, m, grp, total))
    order = list(PHASES)
    H = 0.42 * len(rows) + 2.6
    fig, ax = plt.subplots(figsize=(15, H))
    labels = []
    for r, (short, m, grp, total) in enumerate(rows):
        x = 0.0
        for _ in range(grp["n_dp"] - n_dp_min):
            ax.barh(r, grp["dp_last"], left=x, color=PHASES["data-parallel tile"], edgecolor="#374151",
                    linewidth=0.4)  # fmt: skip
            ax.text(x + grp["dp_last"] / 2, r, f"DP tile {grp['dp_last']:.0f}", ha="center",
                    va="center", fontsize=7)  # fmt: skip
            x += grp["dp_last"]
        for s in grp["segs"]:
            parts = [("transition", s["transition"]), ("MAC", s["mac"])] + list(s["phases"].items())
            for lab, v in parts:
                if v <= 0:
                    continue
                ax.barh(r, v, left=x, color=PHASES[lab], edgecolor="#374151", linewidth=0.4)
                if v > 3.5 and lab not in ("transition",):
                    ax.text(x + v / 2, r, f"{v:.0f}", ha="center", va="center", fontsize=7)
                x += v
        ax.text(x + 0.8, r, f"{x:.0f}", va="center", fontsize=8, color="#374151")
        labels.append(f"{short}: {grp['sig']}  ({grp['n']} of {total} WGs)")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"k-steps (data-parallel steady state) after {n_dp_min} data-parallel tile(s), "
                  "medians per role signature")  # fmt: skip
    handles = [plt.Rectangle((0, 0), 1, 1, color=PHASES[p]) for p in order]
    fig.legend(handles, order, ncol=len(order) // 2, fontsize=8, loc="lower center")
    ax.set_title(f"Stream-K critical path by role, {ms[0]['shape']} (ATT; at most {max_groups} "
                 "most common signatures per version)", fontsize=11)  # fmt: skip
    ax.grid(axis="x", color="#e5e7eb")
    ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0.9 / H, 1, 1))
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
