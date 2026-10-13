#!/usr/bin/env python3
"""Write the MIR of chosen pass dumps (by index, as numbered by mir_copies.py) to files.

    python mir_section.py dump.txt OUT_PREFIX 55 57 58
"""

import re
import sys

HDR = re.compile(r"^# \*\*\* IR Dump After (.+?) \*\*\*")
path, prefix, want = sys.argv[1], sys.argv[2], {int(x) for x in sys.argv[3:]}
i, out, mir = -1, None, False
for line in open(path, errors="replace"):
    m = HDR.match(line)
    if m:
        if out:
            out.close()
            out = None
        mir = False
        pending = m.group(1)
        continue
    if "# Machine code for function" in line and not mir:
        mir = True
        i += 1
        if i in want:
            out = open(f"{prefix}_{i:02d}.mir", "w")
            out.write(f"# after {pending}\n")
    if out:
        out.write(line)
if out:
    out.close()
