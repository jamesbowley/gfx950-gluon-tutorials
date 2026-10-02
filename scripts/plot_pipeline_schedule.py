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
Draw the software-pipeline schedule of the a16w16 v9/v10 GEMM kernel.

The schedule (which buffer each load, LDS read and MFMA region touches, and
for which K-step) comes from the kernel source. The timing comes from one
wave of an ATT trace, so every block sits at the cycle its instructions were
actually issued. Three stacked panels share the time axis:

  1. swimlane timeline: waits, global->LDS loads, LDS reads, MFMA, epilogue
     convert, C stores, setup / accumulator zero-init
  2. shared-memory buffer occupancy (in flight, ready, being read, free)
  3. pipeline stages per K-step (load, LDS read, MFMA)

Outputs a self-contained HTML file (inline SVG, hover tooltips, one-tile /
two-tile toggle, click a legend entry to highlight one operation kind) and
two PNGs.

    python scripts/plot_pipeline_schedule.py
    python scripts/plot_pipeline_schedule.py --att-dir <run dir> --out-dir <dir>
"""

import argparse
import glob
import html
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPTS_DIR)
sys.path.insert(0, SCRIPTS_DIR)
from process_json import analyze_code  # noqa: E402

A16W16 = os.path.join(REPO_ROOT, "kernels", "gemm", "intra_wave", "a16w16")
DEFAULT_ATT_DIR = os.path.join(A16W16, "tmp", "llir+amdgcnas", "v9_beyond_hotloop")
DEFAULT_SOURCE = os.path.join(A16W16, "v10_persistant", "matmul_kernel.py")
DEFAULT_OUT_DIR = os.path.join(A16W16, "v10_persistant", "images")

MFMAS_PER_REGION = 32
REGIONS_PER_KSTEP = 4
LOADS_PER_GROUP = 4
READS_PER_GROUP = 8
MFMA_ISSUE_CYCLES = 16
STORE_ISSUE_CYCLES = 16  # buffer_store_dwordx4 on a wave64 sends 1 KB out of VGPRs

OPERANDS = ["A_top", "A_bot", "B_left", "B_right"]
SHORT = {"A_top": "At", "A_bot": "Ab", "B_left": "Bl", "B_right": "Br"}
LOAD_ORDER = ["B_left", "A_top", "A_bot", "B_right"]
QUADRANTS = [("tl", "A_top", "B_left"), ("bl", "A_bot", "B_left"),
             ("tr", "A_top", "B_right"), ("br", "A_bot", "B_right")]  # fmt: skip

KINDS = {
    "wait": ("#374151", "Wait: s_waitcnt vmcnt + s_barrier"),
    "load": ("#2563eb", "Global->LDS async load (buffer_load ... lds)"),
    "read": ("#16a34a", "LDS->register read (ds_read_b128)"),
    "mfma": ("#9333ea", "MFMA region (32 x v_mfma_f32_16x16x32_f16)"),
    "convert": ("#ea580c", "Epilogue convert (cvt + LDS shuffle)"),
    "regmove": ("#0891b2", "AGPR<->VGPR move (v_accvgpr_read/write)"),
    "store": ("#dc2626", "C store (buffer_store_dwordx4)"),
    "alu": ("#9ca3af", "Setup / accumulator zero-init"),
}
STATE_COLORS = {
    "inflight": "#93c5fd",
    "ready": "#bbf7d0",
    "reading": "#16a34a",
    "free": "#f3f4f6",
    "free_end": "#fde047",
    "free_next": "#fed7aa",
}
PHASE_COLORS = {"prologue": "#e0e7ff", "loop": "#dcfce7", "epilogue": "#ffedd5"}

PX_PER_CYCLE = 0.085
LEFT = 250
RIGHT = 30
BREAK_PX = 96


# --------------------------------------------------------------------------
# Schedule from the kernel source
# --------------------------------------------------------------------------


def source_loads(num_iters):
    """(operand, kstep) for every async-copy group, in issue order."""
    loads = [(op, k) for k in (0, 1) for op in LOAD_ORDER]
    for j in range(num_iters):
        for half in (0, 1):
            loads += [(op, 2 * j + 2 + half) for op in LOAD_ORDER]
    return loads


def source_reads(num_iters, num_ksteps):
    """(operand, kstep) for every smem.index(i).load(), in program order."""
    reads = [("B_left", 0), ("A_top", 0)]
    for j in range(num_iters):
        k = 2 * j
        reads += [("A_bot", k), ("B_right", k), ("B_left", k + 1), ("A_top", k + 1),
                  ("A_bot", k + 1), ("B_right", k + 1), ("B_left", k + 2), ("A_top", k + 2)]  # fmt: skip
    k = num_ksteps - 2
    reads += [("A_bot", k), ("B_right", k), ("B_left", k + 1), ("A_top", k + 1),
              ("A_bot", k + 1), ("B_right", k + 1)]  # fmt: skip
    return reads


def source_lines(path):
    """Line numbers of each schedule-relevant call, in source order."""
    pats = {
        "load": "buffer_load_to_shared(",
        "read": ".load(dotOpLayout",
        "mfma": "cdna3.mfma(",
        "store": "buffer_store(",
        "convert": "convert_layout(",
    }
    lines = {k: [] for k in pats}
    with open(path) as f:
        text = f.read()
    for n, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        for k, p in pats.items():
            if p in line:
                lines[k].append(n)
    # A K-loop body factored into a helper defined above the kernel (v10_A_B's k_step_pair)
    # lists its 8 loads and 8 reads first; move them to main-loop position after the prologue's.
    # v14 also moves the rest of the tile body into persistent_tile (between k_step_pair and the
    # kernel): file order is K-loop body, epilogue, prologue.
    if "def persistent_tile(" in text:
        lines["load"] = lines["load"][16:24] + lines["load"][:8] + lines["load"][8:16]
        lines["read"] = lines["read"][16:18] + lines["read"][:8] + lines["read"][8:16]
    elif "def k_step_pair(" in text:
        lines["load"] = lines["load"][8:16] + lines["load"][:8] + lines["load"][16:]
        lines["read"] = lines["read"][8:10] + lines["read"][:8] + lines["read"][10:]
    return lines


# --------------------------------------------------------------------------
# ATT trace -> events
# --------------------------------------------------------------------------


def classify(name):
    if name.startswith("v_mfma"):
        return "mfma"
    if name.startswith("buffer_load") and name.endswith(" lds"):
        return "load"
    if name.startswith("buffer_store"):
        return "store"
    if name.startswith("ds_read"):
        return "ds_read"
    if name.startswith("ds_write"):
        return "ds_write"
    if name.startswith("s_waitcnt") and "vmcnt" in name:
        return "vmwait"
    if name.startswith("s_barrier"):
        return "barrier"
    # Only the fp32 -> fp16/bf16 packs of the output; v_cvt_f32_u32 etc. are integer-division math
    if name.startswith(("v_cvt_pk_f16", "v_cvt_pk_bf16", "v_cvt_f16_f32", "v_cvt_bf16_f32")):
        return "cvt"
    if name.startswith("v_accvgpr_write"):
        return "accw"
    if name.startswith("v_accvgpr_read"):
        return "accr"
    return "other"


def ev(kind, t0, t1, lane, label="", tip="", **extra):
    e = {"kind": kind, "t0": t0, "t1": max(t1, t0 + 4), "lane": lane, "label": label, "tip": tip}
    e.update(extra)
    return e


def load_run(att_dir):
    ui = glob.glob(os.path.join(att_dir, "ui_*"))
    if not ui:
        raise SystemExit(f"no ui_* directory under {att_dir}")
    code = json.load(open(os.path.join(ui[0], "code.json")))["code"]
    info = analyze_code(code)
    rows = sorted(code, key=lambda r: r[2])
    idx_at = {r[5]: r[2] for r in rows}  # byte address -> instruction index

    # Backward s_cbranch: 16-bit signed dword offset from the next instruction
    backward = []
    for r in rows:
        parts = r[0].split()
        if (r[0].startswith("s_cbranch") and len(parts) > 1 and parts[1].isdigit()
                and int(parts[1]) > 32767):  # fmt: skip
            backward.append((r[2], idx_at.get(r[5] + 4 + 4 * (int(parts[1]) - 65536))))

    # The persistent tile loop's back-edge: after the K-loop epilogue, jumping to before the K loop
    backedge = next((b for b, t in backward if b > info["epilogue_first_index"]
                     and t is not None and t < info["loop_first_index"]), None)  # fmt: skip

    # v14's peeled tail: a second copy of the tile code after the persistent loop, with its own
    # K loop (a backward branch around at least one MFMA region) at different indices.
    info_tail, tail_start = None, None
    if backedge is not None:
        mfma_idx = [r[2] for r in rows if r[0].startswith("v_mfma")]
        for b, t in backward:
            if b > backedge and t is not None and t > backedge and (
                    sum(t <= i <= b for i in mfma_idx) >= MFMAS_PER_REGION):  # fmt: skip
                info_tail = dict(info, loop_first_index=t, loop_last_index=b,
                                 epilogue_first_index=b + 1)  # fmt: skip
                tail_start = backedge + 1
                break
    return {
        "info": info,
        "info_tail": info_tail,
        "tail_start": tail_start,
        "names": {r[2]: r[0] for r in code},
        "backedge": backedge,
        "wave_files": sorted(glob.glob(os.path.join(ui[0], "se*_wv*.json"))),
    }


def same_simd_waves(run, first):
    """`first` and the later waves that ran in the same SIMD slot (next workgroups)."""
    prefix = re.sub(r"_wv\d+\.json$", "", os.path.basename(first))
    waves = [f for f in run["wave_files"] if os.path.basename(f).startswith(prefix + "_wv")]
    return sorted(waves, key=lambda f: int(re.search(r"_wv(\d+)\.json$", f).group(1)))


def build_wave_tiles(run, wave_path, lines, origin):
    """Split one wave into tiles (one per persistent-loop trip); times relative to `origin`."""
    names, info = run["names"], run["info"]
    wave = json.load(open(wave_path))["wave"]
    seq = [
        (c - origin, idx, names[idx], classify(names[idx]))
        for c, _, _, _, idx in wave["instructions"]
    ]
    wave_start, wave_end = wave["begin"] - origin, wave["end"] - origin

    # vmcnt counts loads and stores together; model in-order completion across the whole wave
    waits, vm_is_load, loads_prefix = [], [], [0]
    for i, (t, _, name, k) in enumerate(seq):
        if k in ("load", "store"):
            vm_is_load.append(k == "load")
            loads_prefix.append(loads_prefix[-1] + (k == "load"))
        if k != "vmwait":
            continue
        n = int(re.search(r"vmcnt\((\d+)\)", name).group(1))
        j = i + 1
        while j < len(seq) and j <= i + 3 and seq[j][3] != "barrier":
            j += 1
        end = seq[j + 1][0] if j < len(seq) - 1 and seq[j][3] == "barrier" else seq[i + 1][0]
        done = loads_prefix[max(0, len(vm_is_load) - n)]
        waits.append({"pos": i, "t0": t, "t1": end, "vmcnt": n, "guaranteed": done // 4,
                      "loads": loads_prefix[-1], "stores": len(vm_is_load) - loads_prefix[-1]})  # fmt: skip

    cuts = [0]
    info_tail = run.get("info_tail")
    has_tail = info_tail is not None and any(s[1] == info_tail["loop_first_index"] for s in seq)
    if run["backedge"] is not None:
        hits = [i + 1 for i, s in enumerate(seq) if s[1] == run["backedge"]]
        # The last back-edge execution falls through; with a peeled tail, the tail starts there
        cuts += hits if has_tail else hits[:-1]
    cuts.append(len(seq))
    n_trips = len(cuts) - 1
    trip_info = [info_tail if has_tail and t == n_trips - 1 else info for t in range(n_trips)]

    # K-steps 0-1 peeled out of the K loop (v10_A_B) show up as 8 extra MFMA regions per trip.
    # The peeled pair issues and reads exactly like main-loop iteration 0, so the source models
    # count it as one more iteration.
    loop_iters = round(info["num_iterations"])
    n_mfma = sum(1 for s in seq if s[3] == "mfma")
    peel = n_mfma // MFMAS_PER_REGION // n_trips - 2 * REGIONS_PER_KSTEP * (loop_iters + 1)
    assert peel in (0, 2 * REGIONS_PER_KSTEP), f"unexpected MFMA region count (extra {peel})"
    num_iters = loop_iters + (1 if peel else 0)

    # The next tile's K-step 0 reads issued in this trip's epilogue (v10_A, v10_A_B): the
    # ds_reads after the epilogue's first ds_write that hit the same buffer-0 slots (same
    # address operands) as the wave's first two reads. convert_layout's reads use other slots.
    def addr(name):
        return name.split(",", 1)[1].strip() if "," in name else ""

    ds = [i for i, s in enumerate(seq) if s[3] == "ds_read"]
    k0_addr = {addr(seq[i][2]) for i in ds[: 2 * READS_PER_GROUP]}
    def operand_reads(a, b):
        # Fallback when the epilogue's reads use other address registers (v14): operand reads
        # step through offsets 0, 64, 256, 320, ...; convert_layout's only multiples of 256.
        reads = [(i, *(addr(seq[i][2]).split(" offset:") + ["0"])[:2]) for i in range(a, b)
                 if seq[i][3] == "ds_read"]  # fmt: skip
        bases = {base for _, base, off in reads if int(off) % 256 == 64}
        return [i for i, base, _ in reads if base in bases]

    next_reads = []
    for t in range(n_trips):
        a, b = cuts[t], cuts[t + 1]
        dsw = next((i for i in range(a, b) if seq[i][3] == "ds_write"), b)
        pos = [i for i in range(dsw, b) if seq[i][3] == "ds_read" and addr(seq[i][2]) in k0_addr]
        if pos and len(pos) != 2 * READS_PER_GROUP:
            pos = operand_reads(dsw, b)
        next_reads.append([i - a for i in pos])
    epi_reads = any(next_reads)
    # The peeled tail's epilogue has none: nothing uses the operands it would read, so the
    # compiler drops those reads.
    assert not epi_reads or all(
        len(p) == 2 * READS_PER_GROUP or (not p and has_tail and t == n_trips - 1)
        for t, p in enumerate(next_reads)
    ), next_reads

    # Async-copy groups (4 loads each) in issue order, each assigned to the tile whose data it
    # carries. Without cross-tile prefetch every trip issues its own K-steps 0-1 and then the
    # main-loop K-steps. With prefetch (v10) only the first trip issues its own K-steps 0-1;
    # every trip's epilogue issues the next tile's.
    loop_src = source_loads(num_iters)[8:]
    load_pos = [i for i, s in enumerate(seq) if s[3] == "load"]
    groups = []
    for gg in range(len(load_pos) // LOADS_PER_GROUP):
        p0, p1 = load_pos[LOADS_PER_GROUP * gg], load_pos[LOADS_PER_GROUP * gg + 3]
        cover = next((w for w in waits if w["guaranteed"] > gg), None)
        groups.append({"gg": gg, "pos": p0, "t0": seq[p0][0], "t1": seq[p1][0] + 4,
                       "done": cover["t1"] if cover else wave_end})  # fmt: skip
    per_slice = [sum(cuts[t] <= g["pos"] < cuts[t + 1] for g in groups) for t in range(n_trips)]
    prefetch = per_slice[0] == 8 + len(loop_src) + 8
    order = []
    for t in range(n_trips):
        if t == 0 or not prefetch:
            order += [(t, g, "prologue") for g in range(8)]
        order += [(t, 8 + g, "loop") for g in range(len(loop_src))]
        if prefetch:
            order += [(t + 1, g, "prefetch") for g in range(8)]
    assert len(order) == len(groups), (len(order), len(groups), per_slice)
    prologue_src = source_loads(num_iters)[:8]
    for g, (owner, local, how) in zip(groups, order):
        op, k = (prologue_src + loop_src)[local]
        if how == "prologue":
            line = lines["load"][local]
        elif how == "loop":
            line = lines["load"][8 + (local - 8) % 8]
        else:
            line = lines["load"][16 + local] if len(lines["load"]) >= 24 else None
        g.update(owner=owner, g=local, op=op, k=k, buf=k % 2, how=how, line=line)
    for w in waits:
        last = groups[w["guaranteed"] - 1] if w["guaranteed"] else None
        w["newest"] = (
            f"{last['op']}[{last['buf']}] K-step {last['k']} of trip {last['owner']}"
            if last
            else "none yet"
        )

    tiles = []
    for ti in range(n_trips):
        a, b = cuts[ti], cuts[ti + 1]
        t_origin = wave_start if ti == 0 else seq[a][0]
        tile_end = wave_end if ti == n_trips - 1 else seq[b][0]
        tw = [w for w in waits if a <= w["pos"] < b]
        own = sorted((g for g in groups if g["owner"] == ti), key=lambda g: g["g"])
        issued = [g for g in groups if a <= g["pos"] < b]
        prev_k0 = None
        if epi_reads and ti > 0:
            prev_k0 = [seq[cuts[ti - 1] + i][0] for i in next_reads[ti - 1]]
        tile = build_tile(seq[a:b], lines, trip_info[ti], t_origin, tile_end, tw, own, issued, ti,
                          n_trips, peel, prev_k0, next_reads[ti])  # fmt: skip
        tile["wave_file"] = os.path.basename(wave_path)
        tile["trip"] = ti
        tile["tail"] = has_tail and ti == n_trips - 1
        tile["prefetch_mode"] = prefetch
        tiles.append(tile)
    return tiles


def build_tile(seq, lines, info, t_origin, tile_end, waits, own_groups, issued, trip, n_trips,
               peel=0, prev_k0=None, next_pos=()):  # fmt: skip
    """One persistent-loop trip. `peel`: MFMA regions of K-steps 0-1 peeled out of the K loop.
    `prev_k0`: times of this tile's K-step 0 reads, issued in the previous trip's epilogue.
    `next_pos`: positions (in `seq`) of the next tile's K-step 0 reads issued in this epilogue."""
    loop_start = next(t for t, idx, _, _ in seq if idx == info["loop_first_index"])
    epi_start = next(t for t, idx, _, _ in seq if idx == info["epilogue_first_index"])

    mfma_t = [t for t, _, _, k in seq if k == "mfma"]
    n_windows = len(mfma_t) // MFMAS_PER_REGION
    num_ksteps = n_windows // REGIONS_PER_KSTEP
    loop_iters = round(info["num_iterations"])
    num_iters = loop_iters + (1 if peel else 0)
    loop_windows = 2 * REGIONS_PER_KSTEP * loop_iters
    assert len(mfma_t) == n_windows * MFMAS_PER_REGION
    assert info["mfma_count_in_loop"] == 2 * REGIONS_PER_KSTEP * MFMAS_PER_REGION
    assert peel + loop_windows + 2 * REGIONS_PER_KSTEP == n_windows, (
        "expected 2 K-steps in the epilogue"
    )

    events = []
    win_start = [mfma_t[MFMAS_PER_REGION * w] for w in range(n_windows)]

    # MFMA regions
    for w in range(n_windows):
        k, (q, a, b) = w // REGIONS_PER_KSTEP, QUADRANTS[w % REGIONS_PER_KSTEP]
        t0 = win_start[w]
        t1 = mfma_t[MFMAS_PER_REGION * (w + 1) - 1] + MFMA_ISSUE_CYCLES
        if w < peel + loop_windows:
            line = lines["mfma"][w % 8]
            if w < peel:
                what = (f"acc_{q} = {a}[{k % 2}] x {b}[{k % 2}]" + (" + 0" if k == 0 else "")
                        + "</b><br>peeled out of the K loop" + (
                            "; the first MFMA of each 16x16 block takes an inline-0 C operand, "
                            "so no accumulator is zeroed" if k == 0 else ""))  # fmt: skip
            else:
                what = f"acc_{q} += {a}[{k % 2}] x {b}[{k % 2}]</b>"
            tip = (f"<b>MFMA region: {what}<br>K-step {k}, 32 MFMAs<br>{t0}-{t1} cycles "
                   f"({t1 - t0})<br>source line {line}")  # fmt: skip
            events.append(ev("mfma", t0, t1, "mfma", f"{q} k{k}", tip, kstep=k, window=w,
                             phase="peel" if w < peel else "loop"))  # fmt: skip
            continue
        e = w - peel - loop_windows
        n0 = MFMAS_PER_REGION * e + 1
        tip = (f"<b>Epilogue MFMAs {n0}-{n0 + MFMAS_PER_REGION - 1} (issue order)</b><br>The "
               f"compiler interleaves K-steps {num_ksteps - 2}-{num_ksteps - 1} across quadrants "
               f"and rotates accumulator registers here, so this block is not one quadrant<br>"
               f"{t0}-{t1} cycles ({t1 - t0}; {MFMAS_PER_REGION * MFMA_ISSUE_CYCLES} of them "
               f"MFMA)<br>source lines {lines['mfma'][8:]}")  # fmt: skip
        events.append(ev("mfma", t0, t1, "mfma", "32 MFMA", tip, kstep=None, window=w,
                         phase="epilogue"))  # fmt: skip

    # Global->LDS loads: `own_groups` carry this tile's data (possibly issued by the previous
    # trip's epilogue); `issued` are the groups this trip issued, drawn where they happened.
    load_groups = own_groups
    assert len(load_groups) == len(source_loads(num_iters)), len(load_groups)
    prefetch_out = {}
    for g in issued:
        nxt = g["owner"] != trip
        if nxt:
            prefetch_out.setdefault((g["op"], g["buf"]), g["t0"])
        who = ""
        if nxt:
            who = (" for the next tile" if trip + 1 < n_trips
                   else " (last trip: redundant prefetch of a clamped tile)")  # fmt: skip
        line = f"<br>source line {g['line']}" if g["line"] else ""
        tip = (f"<b>Async copy{who}: {g['op']}[{g['buf']}] &lt;- K-step {g['k']}</b><br>"
               f"4 x buffer_load_dwordx4 ... lds<br>issued {g['t0']}-{g['t1']}, guaranteed "
               f"landed by {g['done']}{line}")  # fmt: skip
        label = ("n" if nxt else "") + f"{SHORT[g['op']]}{g['buf']}"
        events.append(ev("load", g["t0"], g["t1"], "load", label, tip, kstep=g["k"], op=g["op"],
                         buf=g["buf"], prefetch=nxt))  # fmt: skip
    for w in waits:
        tip = (f"<b>s_waitcnt vmcnt({w['vmcnt']}) + s_barrier</b><br>{w['loads']} loads and "
               f"{w['stores']} stores issued so far on this wave; all but the newest {w['vmcnt']} "
               f"have completed (in-order model)<br>newest guaranteed load: {w['newest']}<br>"
               f"stalled {w['t1'] - w['t0']} cycles<br>wait_group(n) lowers to vmcnt(4n); the "
               "scheduler may move it relative to the source")  # fmt: skip
        events.append(ev("wait", w["t0"], w["t1"], "wait", "", tip))

    # LDS->register operand reads, 8 ds_read_b128 per read; later ds_reads are convert_layout
    first_dsw = next(t for t, _, _, k in seq if k == "ds_write")
    read_t = [t for t, _, _, k in seq if k == "ds_read" and t < first_dsw]
    if prev_k0 is not None:
        read_t = prev_k0 + read_t
    src_r = source_reads(num_iters, num_ksteps)
    assert len(read_t) == len(src_r) * READS_PER_GROUP, (len(read_t), len(src_r))
    group_of = {(g["op"], g["k"]): g for g in load_groups}
    reads = []
    for r, (op, k) in enumerate(src_r):
        t0, t1 = read_t[READS_PER_GROUP * r], read_t[READS_PER_GROUP * r + 7] + 8
        g = group_of[(op, k)]
        guarded = g["done"] <= t0
        if r < 2:
            line = lines["read"][r]
        elif r < 2 + 8 * num_iters:
            line = lines["read"][2 + (r - 2) % 8]
        else:
            line = lines["read"][10 + r - 2 - 8 * num_iters]
        reads.append(
            {"op": op, "buf": k % 2, "k": k, "t0": t0, "t1": t1, "g": g, "guarded": guarded}
        )
        warn = "" if guarded else (
            f"<br><span style='color:#dc2626'><b>issued before any wait guarantees this "
            f"buffer's load</b> (group {g['g'] + 1}, guaranteed at {g['done']})</span>")  # fmt: skip
        tip = (f"<b>LDS read: {op}[{k % 2}] (K-step {k}) -&gt; registers</b><br>8 x ds_read_b128<br>"
               f"{t0}-{t1} cycles<br>source line {line}{warn}")  # fmt: skip
        events.append(ev("read", t0, t1, "read", f"{SHORT[op]}{k % 2}", tip, kstep=k, op=op,
                         buf=k % 2, guarded=guarded))  # fmt: skip

    # The next tile's K-step 0 reads, issued in this epilogue (program order: B_left, A_top)
    for gi, op in enumerate(["B_left", "A_top"]):
        chunk = next_pos[READS_PER_GROUP * gi : READS_PER_GROUP * (gi + 1)]
        if not chunk:
            continue
        t0, t1 = seq[chunk[0]][0], seq[chunk[-1]][0] + 8
        who = ("the next tile" if trip + 1 < n_trips
               else "a clamped tile (last trip: redundant, like the prefetch)")  # fmt: skip
        tip = (f"<b>LDS read for {who}: {op}[0] (K-step 0) -&gt; registers</b><br>8 x ds_read_b128, "
               f"issued in this tile's epilogue<br>{t0}-{t1} cycles<br>source line "
               f"{lines['read'][len(lines['read']) - 2 + gi]}")  # fmt: skip
        events.append(ev("read", t0, t1, "read", f"n{SHORT[op]}0", tip, op=op, buf=0))
    next_set = set(next_pos)

    # Setup and accumulator zero-init
    setup_end = next(t for t, _, _, k in seq if k in ("load", "vmwait"))
    events.append(ev("alu", t_origin, setup_end, "alu", "setup",
                     "<b>Setup</b><br>kernel args, XCD remap, group-M, offsets<br>"
                     f"{t_origin}-{setup_end} cycles"))  # fmt: skip
    zero_w = zero_init_writes(seq)
    # With the peel, pre-loop AGPR writes are register moves, not zero-init; only writes of 0 count
    accw = [
        t
        for i, (t, _, _, k) in enumerate(seq)
        if k == "accw" and ((t < loop_start and not peel) or i in zero_w)
    ]
    for c in clusters_of([(t, "v_accvgpr_write") for t in accw], max_gap=128):
        events.append(ev("alu", c["t0"], c["t1"], "alu", "acc = 0",
                         f"<b>Accumulator zero-init</b><br>{c['parts']}<br>"
                         f"{c['t0']}-{c['t1']} cycles ({c['t1'] - c['t0']})"))  # fmt: skip

    # Epilogue convert (acc.to(fp16) + convert_layout through LDS), from the first v_cvt on
    first_cvt = next((t for t, _, _, k in seq if k == "cvt" and t >= epi_start), None)
    conv = [] if first_cvt is None else [(t, k) for i, (t, _, _, k) in enumerate(seq) if (
        t >= first_cvt and i not in next_set and (
            k in ("cvt", "ds_write") or (k in ("ds_read", "barrier") and t >= first_dsw)))]  # fmt: skip
    for c in clusters_of(conv):
        tip = (f"<b>Epilogue convert</b><br>{c['parts']}<br>{c['t0']}-{c['t1']} cycles<br>"
               f"acc.to(fp16) + convert_layout via LDS, source lines {lines['convert']}")  # fmt: skip
        events.append(ev("convert", c["t0"], c["t1"], "convert", "", tip))

    # AGPR<->VGPR moves in the epilogue: VALU (v_cvt) cannot read AGPRs, so finished accumulators
    # are copied out; before the first v_cvt they are register-allocator shuffles instead
    moves = [(t, k) for i, (t, _, _, k) in enumerate(seq)
             if (t >= epi_start or (peel and t < loop_start)) and k in ("accr", "accw")
             and i not in zero_w]  # fmt: skip
    for c in clusters_of(moves):
        why = (
            "copies finished accumulators to VGPRs for v_cvt (plus some register shuffling)"
            if first_cvt is not None and c["t0"] >= first_cvt
            else "register-allocator shuffling: partly accumulated tiles moved between AGPRs "
            "and VGPRs, and MFMA results written to scratch AGPRs then copied out"
        )
        tip = (f"<b>AGPR&lt;-&gt;VGPR moves</b><br>{c['parts']}<br>{c['t0']}-{c['t1']} cycles<br>"
               f"{why}")  # fmt: skip
        events.append(ev("regmove", c["t0"], c["t1"], "regmove", "", tip))

    # C stores, attributed to quadrants in program order (8 per quadrant per wave). A store's own
    # issue takes STORE_ISSUE_CYCLES; only the time beyond that before the next instruction counts
    # as a stall, split by what that next instruction is.
    stores = []
    for i, (t, _, _, k) in enumerate(seq):
        if k == "store":
            nxt = seq[i + 1] if i + 1 < len(seq) else None
            gap = nxt[0] - t if nxt else STORE_ISSUE_CYCLES
            stores.append((t, gap, nxt[3] if nxt else "end", nxt[2] if nxt else "end of trip"))
    store_excess = {"store": 0, "mfma": 0, "other": 0}
    for t, gap, nk, _ in stores:
        key = nk if nk in ("store", "mfma") else "other"
        store_excess[key] += max(0, gap - STORE_ISSUE_CYCLES)
    per_q = len(stores) // 4
    for qi, q in enumerate(["tl", "bl", "tr", "br"]):
        qs = stores[per_q * qi : per_q * (qi + 1)]
        excess = sum(max(0, g - STORE_ISSUE_CYCLES) for _, g, _, _ in qs)
        tip = (f"<b>Store c_{q}</b><br>{per_q} x buffer_store_dwordx4 (attributed in program order)"
               f"<br>{qs[0][0]}-{qs[-1][0] + STORE_ISSUE_CYCLES} cycles<br>issue {per_q} x "
               f"{STORE_ISSUE_CYCLES}, plus {excess} cycles before the next instruction beyond "
               f"that<br>source line {lines['store'][qi]}")  # fmt: skip
        events.append(ev("store", qs[0][0], qs[-1][0] + STORE_ISSUE_CYCLES, "store", f"c_{q}",
                         tip, span=True))  # fmt: skip
        for t, gap, nk, nname in qs:
            extra = max(0, gap - STORE_ISSUE_CYCLES)
            tip = (f"<b>buffer_store_dwordx4 (c_{q})</b><br>issued at {t}; next instruction "
                   f"{html.escape(nname.split(',')[0])} at +{gap}<br>a 1 KB store takes about "
                   f"{STORE_ISSUE_CYCLES} cycles to issue, so {extra} cycles are extra"
                   + ("" if extra == 0 or nk == "store" else
                      " (often the next instruction waiting on something else)"))  # fmt: skip
            events.append(ev("store", t, t + STORE_ISSUE_CYCLES, "store", "", tip, tick=True))

    return {
        "events": events, "load_groups": load_groups, "reads": reads, "waits": waits,
        "win_start": win_start, "loop_start": loop_start, "epi_start": epi_start,
        "tile_end": tile_end, "num_iters": num_iters, "num_ksteps": num_ksteps,
        "loop_windows": loop_windows, "info": info, "t_origin": t_origin,
        "loop_iters": loop_iters, "peel_windows": peel,
        "prefetch_out": prefetch_out, "store_excess": store_excess, "n_stores": len(stores),
    }  # fmt: skip


def zero_init_writes(seq):
    """Positions of v_accvgpr_write whose source is 0 (a literal, or a VGPR set by v_mov ..., 0)."""
    zero_regs, out = set(), set()
    for i, (_, _, name, k) in enumerate(seq):
        ops = [o.strip() for o in name.split(None, 1)[1].split(",")] if " " in name else []
        if k == "accw":
            if len(ops) > 1 and (ops[1] == "0" or ops[1] in zero_regs):
                out.add(i)
            continue
        if not ops or not ops[0].startswith("v"):
            continue
        if name.startswith("v_mov_b32") and len(ops) > 1 and ops[1] == "0":
            zero_regs.add(ops[0])
        else:
            zero_regs.discard(ops[0])
    return out


def clusters_of(items, max_gap=48):
    """Group (time, kind) pairs into runs separated by less than `max_gap` cycles."""
    out = []
    for t, k in items:
        if out and t - out[-1]["t1"] <= max_gap:
            out[-1]["t1"] = t + 4
            out[-1]["n"][k] = out[-1]["n"].get(k, 0) + 1
        else:
            out.append({"t0": t, "t1": t + 4, "n": {k: 1}})
    for c in out:
        c["parts"] = ", ".join(f"{v} x {k}" for k, v in sorted(c["n"].items()))
    return out


def buffer_intervals(tile):
    """Per-buffer states: in flight, ready, being read, free."""
    out = []
    reads_by = {}
    for r in tile["reads"]:
        reads_by[(r["op"], r["k"])] = r
    by_buf = {}
    for g in tile["load_groups"]:
        by_buf.setdefault((g["op"], g["buf"]), []).append(g)
    for (op, buf), groups in by_buf.items():
        for i, g in enumerate(groups):
            r = reads_by[(op, g["k"])]
            row = f"{op}[{buf}]"
            ready_at = min(g["done"], r["t0"])
            out.append((row, "inflight", g["t0"], ready_at, g["k"]))
            if g["done"] < r["t0"]:
                out.append((row, "ready", g["done"], r["t0"], g["k"]))
            out.append((row, "reading", r["t0"], r["t1"], g["k"], r["guarded"]))
            refill = tile.get("prefetch_out", {}).get((op, buf))
            if i + 1 < len(groups):
                out.append((row, "free", r["t1"], groups[i + 1]["t0"], g["k"]))
            elif refill is not None and refill > r["t1"]:
                out.append((row, "free_next", r["t1"], refill, g["k"]))
            else:
                out.append((row, "free_end", r["t1"], tile["tile_end"], g["k"]))
    return out


def stage_spans(tile):
    """Per K-step: load (issue -> guaranteed), LDS read, MFMA spans."""
    spans = {}
    for g in tile["load_groups"]:
        s = spans.setdefault(g["k"], {})
        a, b = s.get("load", (g["t0"], g["done"]))
        s["load"] = (min(a, g["t0"]), max(b, g["done"]))
    for r in tile["reads"]:
        s = spans[r["k"]]
        a, b = s.get("read", (r["t0"], r["t1"]))
        s["read"] = (min(a, r["t0"]), max(b, r["t1"]))
    K = tile["num_ksteps"]
    for e in tile["events"]:
        if e["kind"] == "mfma":
            # Epilogue MFMAs interleave the last two K-steps, so each block counts for both
            ks = [e["kstep"]] if e["phase"] in ("loop", "peel") else [K - 2, K - 1]
            for k in ks:
                s = spans[k]
                a, b = s.get("mfma", (e["t0"], e["t1"]))
                s["mfma"] = (min(a, e["t0"]), max(b, e["t1"]))
    return spans


# --------------------------------------------------------------------------
# Layout -> drawing primitives (shared by the SVG and matplotlib renderers)
# --------------------------------------------------------------------------


class Canvas:
    def __init__(self):
        self.items = []

    def rect(self, x, y, w, h, fill, kind=None, tip=None, stroke=None, sw=0.0, opacity=1.0):
        self.items.append(("rect", x, y, w, h, fill, kind, tip, stroke, sw, opacity))

    def line(self, x1, y1, x2, y2, color="#9ca3af", width=1.0, dash=False, kind=None):
        self.items.append(("line", x1, y1, x2, y2, color, width, dash, kind))

    def text(self, x, y, s, size=11, color="#111827", anchor="start", weight="normal", kind=None):
        self.items.append(("text", x, y, s, size, color, anchor, weight, kind))


class XMap:
    """Piecewise-linear cycles -> pixels over visible segments, with breaks between them."""

    def __init__(self, segments):
        self.segs = []
        x = LEFT
        for i, (a, b) in enumerate(segments):
            self.segs.append((a, b, x))
            x += (b - a) * PX_PER_CYCLE
            if i < len(segments) - 1:
                x += BREAK_PX
        self.width = x + RIGHT

    def pieces(self, t0, t1):
        for a, b, x in self.segs:
            lo, hi = max(t0, a), min(t1, b)
            if lo < hi:
                yield x + (lo - a) * PX_PER_CYCLE, (hi - lo) * PX_PER_CYCLE

    def x(self, t):
        for a, b, x in self.segs:
            if a <= t <= b:
                return x + (t - a) * PX_PER_CYCLE
        return None


def shift_tile(tile, dt, tag):
    def sh(d):
        d = dict(d)
        for key in ("t0", "t1", "done"):
            if key in d:
                d[key] += dt
        return d

    t = dict(tile)
    t["events"] = [sh(e) for e in tile["events"]]
    t["load_groups"] = [sh(g) for g in tile["load_groups"]]
    gmap = {g["g"]: g for g in t["load_groups"]}
    t["reads"] = [dict(sh(r), g=gmap[r["g"]["g"]]) for r in tile["reads"]]
    t["win_start"] = [w + dt for w in tile["win_start"]]
    for key in ("loop_start", "epi_start", "tile_end"):
        t[key] = tile[key] + dt
    t["t_origin"] = tile.get("t_origin", 0) + dt
    t["tag"] = tag
    return t


def draw_view(tiles, segments, stage_rows, title, notes, break_label, gaps=()):
    cv = Canvas()
    xm = XMap(segments)
    W = xm.width

    lanes = [
        ("wait", "Waits (vmcnt + barrier)"),
        ("load", "Global->LDS loads"),
        ("read", "LDS->register reads"),
        ("mfma", "MFMA"),
        ("convert", "Epilogue convert"),
        ("regmove", "AGPR<->VGPR moves"),
        ("store", "C stores"),
        ("alu", "Setup / acc zero-init"),
    ]
    buf_rows = [f"{op}[{b}]" for op in OPERANDS for b in (0, 1)]
    LANE_H, LANE_GAP, ROW_H, STAGE_H = 26, 6, 17, 24

    y = 0
    cv.text(12, 26, title, size=17, weight="bold")
    for i, n in enumerate(notes):
        cv.text(12, 46 + 16 * i, n, size=11, color="#4b5563")
    y = 52 + 16 * len(notes)

    # Legend
    lx = 12
    for kind, (color, label) in KINDS.items():
        cv.rect(lx, y, 14, 14, color, kind=f"legend:{kind}")
        cv.text(lx + 19, y + 11, label, size=11, kind=f"legend:{kind}")
        lx += 30 + 6.3 * len(label)
    y += 22
    lx = 12
    for state, label in [("inflight", "buffer: load in flight"), ("ready", "buffer: landed, not read"),
                         ("reading", "buffer: being read"), ("free", "buffer: free"),
                         ("free_next", "buffer: free until the next tile's prefetch"),
                         ("free_end", "buffer: free until the tile ends")]:  # fmt: skip
        cv.rect(lx, y, 14, 14, STATE_COLORS[state], stroke="#9ca3af", sw=0.5)
        cv.text(lx + 19, y + 11, label, size=11)
        lx += 30 + 6.3 * len(label)
    cv.rect(lx, y, 14, 14, "#ffffff", stroke="#dc2626", sw=2)
    cv.text(lx + 19, y + 11, "red outline: LDS read issued before a wait guarantees its data",
            size=11)  # fmt: skip
    y += 34

    top = y + 26
    panel1_h = len(lanes) * (LANE_H + LANE_GAP)
    panel2_top = top + panel1_h + 34
    panel2_h = len(buf_rows) * ROW_H
    panel3_top = panel2_top + panel2_h + 34
    panel3_h = len(stage_rows) * STAGE_H
    bottom = panel3_top + panel3_h
    H = bottom + 46

    # Phase bands across all panels
    for tile in tiles:
        tag = tile.get("tag", "")
        start = tile.get("t_origin", 0)
        bands = [("prologue", start, tile["loop_start"]), ("loop", tile["loop_start"], tile["epi_start"]),
                 ("epilogue", tile["epi_start"], tile["tile_end"])]  # fmt: skip
        for name, a, b in bands:
            for x, w in xm.pieces(a, b):
                cv.rect(x, top - 22, w, bottom - top + 22, PHASE_COLORS[name], opacity=0.55)
                label = {"prologue": "Prologue", "loop": "Main loop", "epilogue": "Epilogue"}[name]
                if w > 60:
                    cv.text(
                        x + 6, top - 7, f"{tag}{label}", size=12, weight="bold", color="#374151"
                    )

    # Breaks
    for i in range(len(xm.segs) - 1):
        a, b, x = xm.segs[i]
        bx = x + (b - a) * PX_PER_CYCLE
        cv.rect(
            bx + 4, top - 22, BREAK_PX - 8, bottom - top + 22, "#f9fafb", stroke="#d1d5db", sw=1
        )
        for j, s in enumerate(break_label):
            cv.text(
                bx + BREAK_PX / 2, top + 40 + 15 * j, s, size=11, anchor="middle", color="#6b7280"
            )

    # Iteration markers
    for tile in tiles:
        for j in range(tile["num_iters"]):
            t = tile["win_start"][8 * j]
            x = xm.x(t)
            if x is not None:
                peeled = 1 if tile.get("peel_windows") else 0
                label = "peeled k0-1" if j < peeled else f"iter {j - peeled}"
                cv.line(x, top, x, bottom, color="#6b7280", width=0.8, dash=True)
                cv.text(x + 3, bottom + 14, label, size=10, color="#6b7280")

    # Panel 1: swimlanes
    cv.text(12, top - 30, "1. Swimlane timeline", size=13, weight="bold")
    lane_y = {}
    for i, (lane, label) in enumerate(lanes):
        ly = top + i * (LANE_H + LANE_GAP)
        lane_y[lane] = ly
        cv.rect(LEFT, ly, W - LEFT - RIGHT, LANE_H, "#ffffff", opacity=0.35)
        cv.text(LEFT - 10, ly + LANE_H / 2 + 4, label, size=12, anchor="end")
    for tile in tiles:
        for e in tile["events"]:
            ly = lane_y[e["lane"]]
            color = KINDS[e["kind"]][0]
            for x, w in xm.pieces(e["t0"], e["t1"]):
                w = max(w, 1.2)
                if e.get("span"):
                    cv.rect(
                        x, ly + 4, w, LANE_H - 8, color, kind=e["kind"], tip=e["tip"], opacity=0.22
                    )
                    if w > 34:
                        cv.text(x + 3, ly + 11, e["label"], size=9, color="#7f1d1d", kind=e["kind"])
                    continue
                if e.get("tick"):
                    cv.rect(x, ly + 12, w, LANE_H - 12, color, kind=e["kind"], tip=e["tip"])
                    continue
                stroke, sw = ("#ffffff", 1.0) if e["kind"] == "mfma" else (None, 0)
                if e["kind"] == "read" and not e.get("guarded", True):
                    stroke, sw = "#dc2626", 2.0
                cv.rect(x, ly + 2, w, LANE_H - 4, color, kind=e["kind"], tip=e["tip"], stroke=stroke,
                        sw=sw)  # fmt: skip
                if e["label"] and w > 5.2 * len(e["label"]) + 3:
                    cv.text(x + w / 2, ly + LANE_H / 2 + 3, e["label"], size=9, color="#ffffff",
                            anchor="middle", kind=e["kind"])  # fmt: skip
            if e["kind"] == "wait":
                x = xm.x(e["t0"])
                if x is not None:
                    cv.line(x, top + LANE_H, x, top + panel1_h - LANE_GAP, color="#374151", width=0.5,
                            dash=True, kind="wait")  # fmt: skip

    # Idle-gap annotations
    for lane, a, b, label in gaps:
        x1, x2 = xm.x(a), xm.x(b)
        if x1 is None or x2 is None:
            continue
        gy = lane_y[lane] + LANE_H / 2
        cv.line(x1, gy, x2, gy, color="#111827", width=1.5)
        cv.line(x1, gy - 6, x1, gy + 6, color="#111827", width=1.5)
        cv.line(x2, gy - 6, x2, gy + 6, color="#111827", width=1.5)
        tw = 6.2 * len(label) + 10
        cv.rect((x1 + x2 - tw) / 2, gy - 9, tw, 17, "#ffffff", stroke="#111827", sw=1)
        cv.text((x1 + x2) / 2, gy + 4, label, size=11, anchor="middle", weight="bold")

    # Panel 2: buffer occupancy
    cv.text(12, panel2_top - 10, "2. Shared-memory buffer occupancy", size=13, weight="bold")
    row_y = {r: panel2_top + i * ROW_H for i, r in enumerate(buf_rows)}
    for r, ry in row_y.items():
        cv.text(LEFT - 10, ry + ROW_H / 2 + 4, r, size=11, anchor="end")
    for tile in tiles:
        tag = tile.get("tag", "")
        for iv in buffer_intervals(tile):
            row, state, a, b, k = iv[:5]
            guarded = iv[5] if len(iv) > 5 else True
            desc = {"inflight": f"load of K-step {k} in flight (until a wait guarantees it)",
                    "ready": f"K-step {k} landed, waiting to be read",
                    "reading": f"K-step {k} being read into registers",
                    "free": f"free after K-step {k} was read (until the next load into it)",
                    "free_next": "free until the next tile's prefetch into it",
                    "free_end": f"free from here to the end of the tile: the next tile could "
                                f"prefetch into {row} now"}[state]  # fmt: skip
            tip = f"<b>{tag}{row}</b><br>{desc}<br>{a}-{b} cycles ({b - a})"
            stroke, sw = ("#dc2626", 2.0) if state == "reading" and not guarded else (None, 0)
            for x, w in xm.pieces(a, b):
                cv.rect(x, row_y[row] + 1.5, max(w, 1), ROW_H - 3, STATE_COLORS[state], tip=tip,
                        stroke=stroke, sw=sw)  # fmt: skip

    # Panel 3: pipeline stages per K-step
    cv.text(12, panel3_top - 10, "3. Pipeline stages per K-step", size=13, weight="bold")
    for i, (tile, k, label) in enumerate(stage_rows):
        ry = panel3_top + i * STAGE_H
        cv.text(LEFT - 10, ry + STAGE_H / 2 + 4, label, size=11, anchor="end")
        spans = stage_spans(tile)[k]
        for j, (stage, color_kind, desc) in enumerate(
            [("load", "load", "global->LDS load in flight"), ("read", "read", "LDS->register reads"),
             ("mfma", "mfma", "MFMA regions")]  # fmt: skip
        ):
            if stage not in spans:
                continue
            a, b = spans[stage]
            if stage == "mfma" and k >= tile["num_ksteps"] - 2:
                desc += " (epilogue: the last two K-steps are interleaved, so both rows share "
                desc += "one span)"
            tip = f"<b>{label}: {desc}</b><br>{a}-{b} cycles ({b - a})"
            for x, w in xm.pieces(a, b):
                cv.rect(
                    x, ry + 2 + 7 * j, max(w, 1), 6, KINDS[color_kind][0], kind=color_kind, tip=tip
                )

    # Axis
    cv.line(LEFT, bottom + 2, W - RIGHT, bottom + 2, color="#374151", width=1)
    for a, b, x in xm.segs:
        step = 2000
        t = (a // step + 1) * step
        while t < b:
            tx = xm.x(t)
            cv.line(tx, bottom + 2, tx, bottom + 7, color="#374151", width=1)
            cv.text(tx, bottom + 30, f"{t / 1000:g}k", size=10, anchor="middle", color="#374151")
            t += step
    cv.text(LEFT, bottom + 44, "cycles from the start of tile i (ATT, one wave)", size=11,
            color="#4b5563")  # fmt: skip
    return cv, W, H


# --------------------------------------------------------------------------
# Renderers
# --------------------------------------------------------------------------


def to_svg(cv, W, H, view_id, visible):
    out = [f'<svg id="{view_id}" class="view" xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" '
           f'height="{H:.0f}" style="display:{"block" if visible else "none"}">']  # fmt: skip
    for it in cv.items:
        if it[0] == "rect":
            _, x, y, w, h, fill, kind, tip, stroke, sw, op = it
            cls, attrs = [], ""
            if kind and kind.startswith("legend:"):
                cls.append("legend")
                attrs += f' data-kind="{kind[7:]}"'
            elif kind:
                cls.append(f"ev k-{kind}")
            if tip:
                attrs += f' data-tip="{html.escape(tip, quote=True)}"'
            st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
            o = f' fill-opacity="{op}"' if op < 1 else ""
            c = f' class="{" ".join(cls)}"' if cls else ""
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                       f'fill="{fill}"{o}{st}{c}{attrs}/>')  # fmt: skip
        elif it[0] == "line":
            _, x1, y1, x2, y2, color, width, dash, kind = it
            d = ' stroke-dasharray="3,3"' if dash else ""
            c = f' class="ev k-{kind}"' if kind else ""
            out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                       f'stroke="{color}" stroke-width="{width}"{d}{c}/>')  # fmt: skip
        else:
            _, x, y, s, size, color, anchor, weight, kind = it
            c = ""
            if kind and kind.startswith("legend:"):
                c = f' class="legend" data-kind="{kind[7:]}"'
            elif kind:
                c = f' class="ev k-{kind}"'
            out.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" '
                       f'text-anchor="{anchor}" font-weight="{weight}"{c}>{html.escape(s)}</text>')  # fmt: skip
    out.append("</svg>")
    return "\n".join(out)


def write_html(path, views, summary):
    css_focus = "\n".join(
        f'body[data-focus="{k}"] .ev:not(.k-{k}) {{ opacity: 0.12; }}' for k in KINDS
    )
    buttons = "".join(
        f'<button data-view="{vid}" class="{"on" if i == 0 else ""}">{label}</button>'
        for i, (vid, label, _) in enumerate(views)
    )
    svgs = "\n".join(svg for _, _, svg in views)
    doc = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>v10 pipeline schedule</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; margin: 16px; color: #111827; }}
.bar {{ margin-bottom: 10px; }}
.bar button {{ font-size: 14px; padding: 6px 14px; margin-right: 6px; border: 1px solid #9ca3af;
  background: #f9fafb; border-radius: 6px; cursor: pointer; }}
.bar button.on {{ background: #1f2937; color: #fff; border-color: #1f2937; }}
.wrap {{ overflow-x: auto; border: 1px solid #e5e7eb; border-radius: 8px; }}
.legend {{ cursor: pointer; }}
svg text {{ font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; pointer-events: none; }}
svg .legend {{ pointer-events: all; }}
#tip {{ position: absolute; display: none; background: #111827; color: #f9fafb; padding: 8px 10px;
  border-radius: 6px; font-size: 13px; line-height: 1.4; max-width: 420px; pointer-events: none; z-index: 9; }}
.summary {{ font-size: 13px; color: #374151; margin: 10px 2px; line-height: 1.5; }}
{css_focus}
</style></head>
<body data-focus="">
<div class="bar">{buttons}<span style="color:#6b7280;font-size:13px">
Hover a block for details. Click a legend entry to highlight one operation kind.</span></div>
<div class="wrap">{svgs}</div>
<div class="summary">{summary}</div>
<div id="tip"></div>
<script>
const tip = document.getElementById('tip');
document.querySelectorAll('[data-tip]').forEach(el => {{
  el.addEventListener('mousemove', e => {{
    tip.innerHTML = el.dataset.tip; tip.style.display = 'block';
    tip.style.left = (e.pageX + 14) + 'px'; tip.style.top = (e.pageY + 14) + 'px';
  }});
  el.addEventListener('mouseleave', () => tip.style.display = 'none');
}});
document.querySelectorAll('.bar button').forEach(b => b.addEventListener('click', () => {{
  document.querySelectorAll('.bar button').forEach(o => o.classList.toggle('on', o === b));
  document.querySelectorAll('svg.view').forEach(s => s.style.display = s.id === b.dataset.view ? 'block' : 'none');
}}));
document.querySelectorAll('.legend').forEach(el => el.addEventListener('click', () => {{
  const k = el.dataset.kind;
  document.body.dataset.focus = document.body.dataset.focus === k ? '' : k;
}}));
</script>
</body></html>
"""
    with open(path, "w") as f:
        f.write(doc)


def to_png(cv, W, H, path, dpi=130):
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    pt = 72 / 100
    # Paint in drawing order, like SVG (matplotlib otherwise layers by artist type)
    for z, it in enumerate(cv.items):
        if it[0] == "rect":
            _, x, y, w, h, fill, kind, tip, stroke, sw, op = it
            ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, alpha=op if op < 1 else None,
                                   edgecolor=stroke or "none", linewidth=sw * pt, zorder=z))  # fmt: skip
        elif it[0] == "line":
            _, x1, y1, x2, y2, color, width, dash, kind = it
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=width * pt,
                    linestyle=(0, (3, 3)) if dash else "-", zorder=z)  # fmt: skip
        else:
            _, x, y, s, size, color, anchor, weight, kind = it
            ha = {"start": "left", "middle": "center", "end": "right"}[anchor]
            ax.text(x, y, s, fontsize=size * pt, color=color, ha=ha, va="baseline", fontweight=weight,
                    zorder=z)  # fmt: skip
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--att-dir", default=DEFAULT_ATT_DIR, help="rocprofv3 ATT run directory")
    ap.add_argument("--source", default=DEFAULT_SOURCE, help="kernel source, for line numbers")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--kernel-label", default="v9/v10", help="kernel name for the chart notes")
    ap.add_argument("--shape", default="M=N=4096 K=8192 fp16", help="problem label for the notes")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    run = load_run(args.att_dir)
    lines = source_lines(args.source)
    first = run["wave_files"][0]
    if run["info_tail"] is not None:
        tail_loop = run["info_tail"]["loop_first_index"]
        with_tail = [f for f in run["wave_files"]
                     if any(x[4] == tail_loop for x in json.load(open(f))["wave"]["instructions"])]  # fmt: skip
        if not with_tail:
            raise SystemExit("no traced wave runs the peeled tail: trace more CUs "
                             "(att_shader_engine_mask / att_target_cu)")  # fmt: skip
        first = with_tail[0]
    origin = json.load(open(first))["wave"]["begin"]
    tiles = build_wave_tiles(run, first, lines, origin)
    chain = "persistent"
    if len(tiles) == 1:
        for f in same_simd_waves(run, first)[1:]:
            tiles += build_wave_tiles(run, f, lines, origin)
        chain = "workgroups" if len(tiles) > 1 else "shifted"

    tile = tiles[0]
    tile["tag"] = ""
    ws, n_it, K = tile["win_start"], tile["num_iters"], tile["num_ksteps"]
    hidden_one = n_it - 4
    common = [
        f"a16w16 {args.kernel_label}, {args.shape}, BLOCK 256x256x64, 4 waves. Timing: ATT, "
        f"{tile['wave_file']} (one wave per SIMD); schedule labels from the kernel source.",
    ]

    # One tile: prologue + iterations 0-1, break, iterations n-2..n-1 + epilogue
    one_segments = [(tile["t_origin"], ws[16]), (ws[8 * (n_it - 2)], tile["tile_end"])]
    one_rows = [(tile, k, f"k{k}") for k in list(range(6)) + list(range(K - 4, K))]
    last_load = max(e["t1"] for e in tile["events"] if e["kind"] == "load")
    gaps1 = [("load", last_load, tile["tile_end"],
              f"no global loads for the last {tile['tile_end'] - last_load} cycles")]  # fmt: skip
    cv1, W1, H1 = draw_view([tile], one_segments, one_rows, "Current pipeline: one tile",
                            common, [f"{hidden_one}", "iterations", "not shown"], gaps1)  # fmt: skip

    def boundary_view(ta, tb, labels, title, note):
        """Last two iterations + epilogue of `ta` into the start of `tb` (same SIMD)."""
        segments = [(ta["win_start"][8 * (n_it - 2)], tb["win_start"][16])]
        rows = [(ta, k, f"{labels[0]} k{k}") for k in range(K - 4, K)] + [
            (tb, k, f"{labels[1]} k{k}") for k in range(4)
        ]
        lo, hi = segments[0]
        lds = sorted(
            (e["t0"], e["t1"])
            for t in (ta, tb)
            for e in t["events"]
            if e["kind"] == "load" and lo <= e["t0"] <= hi
        )
        load_gap, gap_a, gap_b = max(
            ((b[0] - a[1], a[1], b[0]) for a, b in zip(lds, lds[1:])), default=(0, lo, lo)
        )
        a_mfma = max(e["t1"] for e in ta["events"] if e["kind"] == "mfma")
        b_mfma = tb["win_start"][0]
        gaps = [
            ("load", gap_a, gap_b, f"longest load-lane gap {load_gap} cycles"),
            ("mfma", a_mfma, b_mfma, f"MFMA idle {b_mfma - a_mfma} cycles"),
        ]
        cv, W, H = draw_view([ta, tb], segments, rows, title, common + [note], [], gaps)
        return cv, W, H, load_gap, b_mfma - a_mfma

    # Two consecutive tiles on the same SIMD
    if chain == "shifted":
        t2 = shift_tile(tile, tile["tile_end"] - tile["t_origin"], "Tile i+1: ")
    else:
        t2 = dict(tiles[1], tag="Tile i+1: ")
    t1 = dict(tile, tag="Tile i: ")
    chain_note = {
        "persistent": "Tile i+1 is the next trip of the persistent tile loop in the same wave "
        "(measured).",
        "workgroups": "Tile i+1 is the next workgroup's wave on the same SIMD, launched by the "
        "hardware after tile i's wave exits (measured).",
        "shifted": "Tile i+1 is the same trace shifted to start where tile i ends (no second tile "
        "in this trace).",
    }[chain]
    cv2, W2, H2, load_gap, mfma_idle = boundary_view(
        t1, t2, ("tile i", "tile i+1"),
        "Current pipeline: two consecutive tiles (tile boundary)", chain_note,
    )  # fmt: skip

    # v14: the last persistent-loop tile into the peeled stream-K tail tile
    tail_view = None
    if tiles[-1].get("tail") and len(tiles) >= 2:
        ta = dict(tiles[-2], tag="Last persistent tile: ")
        tb = dict(tiles[-1], tag="Peeled tile: ")
        tail_note = ("Peeled tile: the extra full-K tile this program runs after the persistent "
                     "loop (the tail copy of the tile code), same wave (measured). The last "
                     "persistent tile's epilogue prefetches its K-steps 0-1 and reads its K-step 0.")  # fmt: skip
        tail_view = boundary_view(ta, tb, ("last", "peeled"),
                                  "Stream-K tail: last persistent tile into the peeled tile",
                                  tail_note)  # fmt: skip

    # Summary numbers
    lg, rd = tile["load_groups"], tile["reads"]
    first_wait = tile["waits"][0]
    unguarded = [r for r in rd if not r["guarded"]]
    epi = tile["epi_start"]
    free_at = {}
    for iv in buffer_intervals(tile):
        if iv[1] in ("free_end", "free_next"):
            refill = f"refilled {iv[3] - epi:+d}" if iv[1] == "free_next" else "not refilled"
            free_at[iv[0]] = (iv[2], refill)
    mfma_cycles = MFMAS_PER_REGION * MFMA_ISSUE_CYCLES * len(tile["win_start"])
    loop_mfma = MFMAS_PER_REGION * MFMA_ISSUE_CYCLES * tile["loop_windows"]
    span = tiles[-1]["tile_end"] - tiles[0]["t_origin"]
    summary_lines = [
        f"tile i: {tile['tile_end'] - tile['t_origin']} cycles; prologue "
        f"{tile['loop_start'] - tile['t_origin']}, main loop {epi - tile['loop_start']} "
        f"({tile['loop_iters']} iterations, "
        f"{(epi - tile['loop_start']) / tile['loop_iters']:.0f} per iteration), epilogue "
        f"{tile['tile_end'] - epi}",
    ]
    for i, t in enumerate(tiles):
        gap = tiles[i + 1]["t_origin"] - t["tile_end"] if i + 1 < len(tiles) else 0
        loop = t["epi_start"] - t["loop_start"]
        w0 = t["waits"][0]
        summary_lines.append(
            f"  tile {i} ({t['wave_file']}, {'peeled tail' if t.get('tail') else 'trip'} "
            f"{t['trip']}): start {t['t_origin']}, prologue "
            f"{t['loop_start'] - t['t_origin']} (first wait vmcnt({w0['vmcnt']}) stalls "
            f"{w0['t1'] - w0['t0']} with {w0['stores']} stores issued), loop {loop} "
            f"(MFMA {100 * loop_mfma / loop:.2f}%), epilogue {t['tile_end'] - t['epi_start']}, "
            f"gap to next tile {gap}"
        )
    summary_lines += [
        f"MFMA utilization over all {len(tiles)} tiles on this SIMD: "
        f"{100 * len(tiles) * mfma_cycles / span:.2f}% ({len(tiles)} x {mfma_cycles} MFMA cycles "
        f"in {span} cycles)",
        f"first load issued at {lg[0]['t0']}; the first wait releases at {first_wait['t1']} "
        f"(about {first_wait['t1'] - lg[1]['t1']} cycles after group 2 was issued)",
        f"last global->LDS load of tile i issued at {last_load}, {tile['tile_end'] - last_load} "
        "cycles before tile i ends",
        f"tile boundary ({chain}): longest load-lane gap {load_gap} cycles, MFMA idle "
        f"{mfma_idle} cycles",
        "last read of each buffer (cycles after the epilogue starts): "
        + ", ".join(
            f"{b} {free_at[b][0] - epi:+d} ({free_at[b][1]})"
            for b in sorted(free_at, key=lambda b: free_at[b][0])
        ),
        f"LDS reads issued before a wait guarantees their data: {len(unguarded)} of {len(rd)}",
        f"tile i C stores: {tile['n_stores']} x {STORE_ISSUE_CYCLES} cycles of issue; extra "
        f"before the next instruction: {tile['store_excess']['store']} when it is another store, "
        f"{tile['store_excess']['mfma']} when it is an MFMA, {tile['store_excess']['other']} "
        "otherwise",
    ]
    if tail_view is not None:
        summary_lines.append(
            f"stream-K tail boundary (last persistent tile -> peeled tile): longest load-lane gap "
            f"{tail_view[3]} cycles, MFMA idle {tail_view[4]} cycles"
        )
    summary_html = "<br>".join(html.escape(s) for s in summary_lines)

    html_path = os.path.join(args.out_dir, "pipeline_schedule.html")
    views = [("one", "One tile", to_svg(cv1, W1, H1, "one", True)),
             ("two", "Two tiles (tile boundary)", to_svg(cv2, W2, H2, "two", False))]  # fmt: skip
    if tail_view is not None:
        cv3, W3, H3 = tail_view[:3]
        views.append(("tail", "Stream-K tail (peeled tile)", to_svg(cv3, W3, H3, "tail", False)))
    write_html(html_path, views, summary_html)
    to_png(cv1, W1, H1, os.path.join(args.out_dir, "pipeline_schedule.png"))
    to_png(cv2, W2, H2, os.path.join(args.out_dir, "pipeline_schedule_2tiles.png"))
    if tail_view is not None:
        to_png(cv3, W3, H3, os.path.join(args.out_dir, "pipeline_schedule_tail.png"))
    for s in summary_lines:
        print(s)
    print(f"wrote {html_path}")


if __name__ == "__main__":
    main()
