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
"""Loop regions of a stream-K .amdgcn and the spill ops in each (no GPU, no torch).

    python experiments/endtoend_spills/regions.py a.amdgcn
    python experiments/endtoend_spills/regions.py a.amdgcn b.amdgcn

With one file: registers, spill ops by region. With two, also the instructions per region and how many of
them differ (registers renamed).
"""

import collections
import difflib
import json
import os
import re
import sys

_META = {
    "vgpr_count": r"^\s*\.vgpr_count:\s*(\d+)",
    "agpr_count": r"^\s*-?\s*\.agpr_count:\s*(\d+)",
    "vgpr_spill": r"^\s*\.vgpr_spill_count:\s*(\d+)",
    "sgpr_spill": r"^\s*\.sgpr_spill_count:\s*(\d+)",
    "arch_vgprs": r"^\s*;\s*NumVgprs:\s*(\d+)",
    "scratch": r"^\s*;\s*ScratchSize:\s*(\d+)",
}
_LABEL = re.compile(r"^(\.LBB\d+_\d+):")
_BRANCH = re.compile(r"^\s*s_(?:c)?branch\w*\s+(\.LBB\d+_\d+)")
_MFMA = re.compile(r"^\s*v_mfma")
_OPS = {
    "st": re.compile(r"^\s*scratch_store_"),
    "ld": re.compile(r"^\s*scratch_load_"),
    "wl": re.compile(r"^\s*v_writelane_b32"),
    "rl": re.compile(r"^\s*v_readlane_b32"),
}


def parse_regions(lines):
    """Region of every line. Loops are backward branches; a K loop is a loop with >= 16 MFMAs and no such
    loop inside it; an outer loop is the innermost loop around a K loop. Outer loops are numbered in code
    order (with a persistent loop and the end-to-end segment loop: L1 = tile loop, L2 = segment loop), and
    a line in one is L<n>.k (its K loop), L<n>.pre / L<n>.post (before / after the K loop); every other
    line is 'outside' (once per program)."""
    labels, ends = {}, collections.defaultdict(int)
    for i, ln in enumerate(lines):
        m = _LABEL.match(ln)
        if m:
            labels[m.group(1)] = i
        m = _BRANCH.match(ln)
        if m and m.group(1) in labels:
            ends[labels[m.group(1)]] = max(ends[labels[m.group(1)]], i)
    loops = sorted(ends.items())
    mfma = [i for i, ln in enumerate(lines) if _MFMA.match(ln)]

    def n_mfma(lo, hi):
        return sum(lo <= i <= hi for i in mfma)

    def inside(a, b):
        return b[0] <= a[0] and a[1] <= b[1] and a != b

    big = [lp for lp in loops if n_mfma(*lp) >= 16]
    kloops = [lp for lp in big if not any(inside(o, lp) for o in big)]
    outer = {}
    for k in kloops:
        around = [lp for lp in loops if inside(k, lp)]
        if around:
            o = min(around, key=lambda lp: lp[1] - lp[0])
            outer.setdefault(o, []).append(k)
    names = {o: f"L{n + 1}" for n, o in enumerate(sorted(outer))}
    region = ["outside"] * len(lines)
    for o, ks in outer.items():
        for i in range(o[0], o[1] + 1):
            if any(k[0] <= i <= k[1] for k in ks):
                region[i] = names[o] + ".k"
            elif i < min(k[0] for k in ks):
                region[i] = names[o] + ".pre"
            else:
                region[i] = names[o] + ".post"
    for k in kloops:
        if not any(k in ks for ks in outer.values()):
            for i in range(k[0], k[1] + 1):
                region[i] = "k"
    return region, {names[o]: (o[0], o[1], len(ks)) for o, ks in outer.items()}


def analyse(path):
    """Registers and spills, the outer loops, spill ops per region and per (region, op, source line), and the
    buffer loads / stores by width after the K loops (ending_mem: the C, partial and bias accesses)."""
    lines = open(path).read().splitlines()
    out = {}
    for key, pat in _META.items():
        rx = re.compile(pat)
        for ln in lines:
            m = rx.search(ln)
            if m:
                out[key] = int(m.group(1))
                break
    region, loops = parse_regions(lines)
    files, cur = {}, None
    by_region = collections.defaultdict(collections.Counter)
    sites = collections.Counter()
    mem = collections.Counter()
    for i, ln in enumerate(lines):
        m = re.match(r"\s*buffer_(load|store)_(\w+)", ln)
        if m and region[i].endswith(".post"):
            mem[f"{m.group(1)}_{m.group(2)}"] += 1
        s = ln.strip()
        m = re.match(r'\.file\s+(\d+)\s+"([^"]*)"(?:\s+"([^"]*)")?', s)
        if m:
            files[m.group(1)] = os.path.basename(m.group(3) or m.group(2))
            continue
        m = re.match(r"\.loc\s+(\d+)\s+(\d+)", s)
        if m:
            cur = f"{files.get(m.group(1), m.group(1))}:{m.group(2)}"
            continue
        for op, rx in _OPS.items():
            if rx.match(ln):
                by_region[region[i]][op] += 1
                if op in ("st", "ld"):
                    sites[f"{region[i]} {op} {cur}"] += 1
    out["loops"] = loops
    out["ops"] = {r: dict(c) for r, c in sorted(by_region.items())}
    out["sites"] = dict(sites.most_common(40))
    out["ending_mem"] = dict(sorted(mem.items()))
    return out


def region_instructions(path):
    """{region: [instruction, ...]} with comments, labels and directives dropped and registers renamed."""
    lines = open(path).read().splitlines()
    region, _ = parse_regions(lines)
    out = collections.defaultdict(list)
    for ln, r in zip(lines, region):
        s = re.sub(r";.*", "", ln).strip()
        if not s or s.startswith(".") or s.endswith(":"):
            continue
        out[r].append(re.sub(r"\b[vsa]\[\d+:\d+\]|\b[vsa]\d+\b", "R", s))
    return out


def main():
    a = analyse(sys.argv[1])
    print(json.dumps(a, indent=1))
    if len(sys.argv) > 2:
        ra, rb = region_instructions(sys.argv[1]), region_instructions(sys.argv[2])
        print("| region | instructions a / b | MFMAs a / b | lines changed |")
        print("|---|---|---|---|")
        for r in sorted(set(ra) | set(rb)):
            x, y = ra.get(r, []), rb.get(r, [])
            d = sum(1 for t in difflib.unified_diff(x, y, lineterm="", n=0)
                    if t[:1] in "+-" and not t.startswith(("+++", "---")))
            mf = (sum(i.startswith("v_mfma") for i in x), sum(i.startswith("v_mfma") for i in y))
            print(f"| {r} | {len(x)} / {len(y)} | {mf[0]} / {mf[1]} | {d} |")


if __name__ == "__main__":
    main()
