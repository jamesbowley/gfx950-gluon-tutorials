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
"""v25 against v24 on stored sweep F and AITER test cases, before v25 rows join v24's.

    HIP_VISIBLE_DEVICES=0 python experiments/streamk_split_count/spot_v25.py   # results/spot_v25.md

Ten sweep F cases (v24 reduce-scatter, owner, two-tile, spread; GEMM_TILE_LIMIT = 256 + S, padded)
and ten AITER test cases (v24 as deployed, reduce-scatter, owner; contiguous, no limit), picked evenly
over each configuration's stored rows; one deployed case falls back to data-parallel at one wave or less
(v9 under v24, v9_ragged under v25). Each is timed now with v24 and with v25 (same STREAMK_*, the
edge flags unset, the order alternating per case), as the sweeps time (rocprof, 1000 launches over
rotating inputs, median). v25's rows can share a comparison with v24's if v25 / v24 timed now stays
within the run-to-run spread (about 1%).
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "streamk_aiter_test"))
import sweep  # noqa: E402
import test as aiter  # noqa: E402

F_PICK = {"v24 rs": 3, "v24 owner": 3, "v24 two_tile": 2, "v24 spread": 2}
AITER_PICK = {"v24 capped": 3, "v24 rs": 4, "v24 owner": 3}


def even(rows, k):
    rows = sorted(rows, key=lambda r: (r["S"], r["K"], str(r["n"]), r["M"], r["N"]))
    return [rows[int((i + 0.5) * len(rows) / k)] for i in range(k)]


def falls_back(r):
    """Whether v24 / v25 as deployed hand this AITER shape to the one-wave data-parallel fallback."""
    import rules_f

    return r["tiles"] <= 256 and rules_f.load_capped()(r["tiles"], r["K"] // 64, 256)[0] == "dp"


def cases():
    f = sweep.load("f")
    for cfg, k in F_PICK.items():
        for r in even([r for r in f if r["config"] == cfg], k):
            env = next(c["env"] for c in sweep.cases_f() if c["config"] == cfg)
            yield dict(src="sweep F", r=r, env=env, flags=(),
                       case=dict(env=env, snp=r["snp"], limit=r["limit"]), mod=sweep)
    t = aiter.load("test.jsonl")
    for cfg, k in AITER_PICK.items():
        rows = [r for r in t if r["config"] == cfg]
        picked = even(rows, k)
        if cfg == "v24 capped":
            picked = even(rows, k - 1) + even([r for r in rows if falls_back(r)], 1)
        for r in picked:
            env = {} if cfg == "v24 capped" else {"STREAMK_POLICY": "one_tile", "STREAMK_FIXUP": aiter.SPLITS[cfg]}
            yield dict(src="AITER", r=r, env=env, flags=("--no-pad",), case=dict(env=env, snp=r["snp"]), mod=aiter)


def main():
    import rocprof_check

    path = os.path.join(HERE, "results", "spot_v25.jsonl")
    done = {}
    if os.path.exists(path):
        for line in open(path):
            x = json.loads(line)
            done[(x["src"], x["M"], x["N"], x["K"], x["config"], str(x["n"]), x["v"])] = x["us"]
    out = []
    for i, c in enumerate(cases()):
        r = c["r"]
        times = {}
        for v in ((24, 25) if i % 2 == 0 else (25, 24)):
            k = (c["src"], r["M"], r["N"], r["K"], r["config"], str(r["n"]), v)
            if k not in done:
                us = rocprof_check.cold_us(v, r["M"], r["N"], r["K"], c["mod"].case_env(c["case"]), c["flags"])
                with open(path, "a") as f:
                    f.write(json.dumps(dict(src=c["src"], M=r["M"], N=r["N"], K=r["K"], S=r["S"], config=r["config"],
                                            n=r["n"], v=v, us=us)) + "\n")
                done[k] = us
            times[v] = done[k]
        out.append((c, times))
        print(f"[{i + 1}] {c['src']} {r['M']}x{r['N']}x{r['K']} S={r['S']} {r['config']} n={r['n']}: stored "
              f"{r['us']:.1f}, v24 {times[24]:.1f}, v25 {times[25]:.1f}", flush=True)
    ratios = [t[25] / t[24] for _, t in out]
    lines = ["# v25 against v24 on stored cases (spot check)\n",
             "rocprof kernel times, us (1000 launches over rotating inputs, median), HIP device 0. Sweep F cases: "
             "4096x8192xK, GEMM_TILE_LIMIT = 256 + S, padded B and C; AITER cases: contiguous, no limit. `stored`: the "
             "v24 row in sweep_f.jsonl / test.jsonl; v24 and v25 timed now, alternating which goes first. Edge flags "
             "unset (all shapes aligned).\n",
             "| source | shape | S | config | n | stored v24 | v24 now | v25 now | v25 / v24 now | v25 / stored |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for c, t in out:
        r = c["r"]
        lines.append(f"| {c['src']} | {r['M']}x{r['N']}x{r['K']} | {r['S']} | {r['config']} | {r['n']} | {r['us']:.1f} | "
                     f"{t[24]:.1f} | {t[25]:.1f} | {t[25] / t[24]:.3f} | {t[25] / r['us']:.3f} |")
    rs = sorted(ratios)
    lines.append(f"\nv25 / v24 timed now: median {rs[len(rs) // 2]:.3f}, range {rs[0]:.3f}-{rs[-1]:.3f}, "
                 f"{sum(abs(x - 1) <= 0.01 for x in rs)} of {len(rs)} within 1%.")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(HERE, "results", "spot_v25.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
