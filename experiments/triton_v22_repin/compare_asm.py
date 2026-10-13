#!/usr/bin/env python3
"""Compare two dump_asm.sh output directories kernel by kernel, ignoring debug info.

    python compare_asm.py <dir a> <dir b>

DWARF sections, .loc/.file directives and comments differ whenever source lines or the
checkout path move, so they are dropped before comparing. Prints one line per kernel
(identical, or the number of differing lines plus register metadata of both) and exits
non-zero if anything differs.
"""

import difflib
import os
import re
import sys

META = {
    "vgprs": re.compile(r"^\s*\.vgpr_count:\s*(\d+)"),
    "spills": re.compile(r"^\s*\.vgpr_spill_count:\s*(\d+)"),
}


def code(path):
    out = []
    in_debug = False
    for line in open(path):
        s = line.strip()
        if s.startswith(".section"):
            in_debug = ".debug" in s
            continue
        if in_debug or not s or s.startswith((";", ".loc", ".file", ".Ltmp", ".Lfunc", ".Ldebug")):
            continue
        if "DW_AT" in s or "debug" in s:
            continue
        out.append(re.sub(r"\s+", " ", s.split(";")[0].strip()))
    return [l for l in out if l]


def meta(path):
    found = {}
    for line in open(path):
        for key, rx in META.items():
            m = rx.match(line)
            if m and key not in found:
                found[key] = int(m.group(1))
    return found


def main():
    a, b = sys.argv[1], sys.argv[2]
    names = sorted(set(os.listdir(a)) | set(os.listdir(b)))
    names = [n for n in names if n.endswith(".amdgcn")]
    bad = 0
    for name in names:
        pa, pb = os.path.join(a, name), os.path.join(b, name)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print(f"{name}: missing in {'a' if not os.path.exists(pa) else 'b'}")
            bad += 1
            continue
        ca, cb = code(pa), code(pb)
        if ca == cb:
            print(f"{name}: identical ({meta(pa)})")
            continue
        bad += 1
        diff = [l for l in difflib.unified_diff(ca, cb, n=0, lineterm="") if l[:1] in "+-"]
        diff = [l for l in diff if not l.startswith(("+++", "---"))]
        print(f"{name}: DIFFERS, {len(diff)} lines; a {meta(pa)}, b {meta(pb)}")
    print(f"{len(names) - bad}/{len(names)} identical")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
