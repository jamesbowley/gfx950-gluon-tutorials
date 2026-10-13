#!/usr/bin/env python3
"""AGPR copy counts after every machine pass of a print-after-all dump.

    LLVM_IR_ENABLE_DUMP=1 python compile_stats.py --worker '<case>' 2> dump.txt
    python mir_copies.py dump.txt [dump2.txt ...]

Counts, per "IR Dump After" section that holds machine code: COPYs whose destination is an
AGPR (virtual register of an areg class, or a physical $agpr), how many of those copy from
another AGPR, AGPR PHIs, and V_ACCVGPR_MOV_B32 (copies after lowering). Only sections where a
count changes are printed.
"""

import re
import sys

HDR = re.compile(r"^# \*\*\* IR Dump After (.+?) \*\*\*")
AREG_DEF = re.compile(r"^\s*(%\d+(?:\.\w+)?:areg[\w]*|\$agpr\w*|%\d+(?:\.\w+)?)\s*=\s*COPY\s+(.*)$")


def scan(path):
    vclass = {}
    rows, cur, stats = [], None, None

    def flush():
        if cur is not None and stats["mir"]:
            rows.append((cur, stats))

    for line in open(path, errors="replace"):
        m = HDR.match(line)
        if m:
            flush()
            cur = m.group(1)
            stats = {"mir": False, "agpr_copy": 0, "agpr_to_agpr": 0, "agpr_phi": 0, "accvgpr_mov": 0}
            vclass = {}
            continue
        if stats is None:
            continue
        if "# Machine code for function" in line:
            stats["mir"] = True
            continue
        if not stats["mir"]:
            continue
        for v, c in re.findall(r"(%\d+)(?:\.\w+)?:(\w+)", line):
            vclass.setdefault(v, c)
        s = line.strip()
        if "V_ACCVGPR_MOV_B32" in s:
            stats["accvgpr_mov"] += 1
        if " = PHI " in s and ("areg" in s.split("=")[0] or "$agpr" in s.split("=")[0]):
            stats["agpr_phi"] += 1
        if " = COPY " not in s:
            continue
        dst, src = s.split(" = COPY ", 1)
        dst = dst.split()[-1]
        src = [w for w in src.split(",")[0].split() if w not in ("killed", "undef", "renamable", "implicit")][0]

        def is_agpr(r):
            if r.startswith("$agpr") or r.startswith("$a"):
                return r.startswith("$agpr")
            base = r.split(":")[0].split(".")[0]
            cls = r.split(":")[1] if ":" in r else vclass.get(base, "")
            return cls.startswith("areg")
        if is_agpr(dst):
            stats["agpr_copy"] += 1
            if is_agpr(src):
                stats["agpr_to_agpr"] += 1
    flush()
    return rows


def main():
    args = sys.argv[1:]
    every = "--all" in args
    args = [a for a in args if a != "--all"]
    for path in args:
        print(f"\n{path}")
        print("| # | after pass | AGPR-dest COPYs | AGPR-to-AGPR COPYs | AGPR PHIs | V_ACCVGPR_MOV |")
        print("|---|---|---|---|---|---|")
        prev = None
        for i, (name, st) in enumerate(scan(path)):
            key = (st["agpr_copy"], st["agpr_to_agpr"], st["agpr_phi"], st["accvgpr_mov"])
            if key != prev or (every and 30 <= i <= 75):
                print(f"| {i} | {name} | {key[0]} | {key[1]} | {key[2]} | {key[3]} |")
                prev = key


if __name__ == "__main__":
    main()
