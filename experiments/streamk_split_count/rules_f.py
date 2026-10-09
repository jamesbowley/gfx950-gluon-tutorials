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
"""Host rules against sweep F (no GPU): which (configuration, n) a rule picks per (S, K), and its
kernel-time regret against the fastest measured choice.

    python experiments/streamk_split_count/rules_f.py            # results/rules_f.md, images/sweep_f_regret.png
    python experiments/streamk_split_count/rules_f.py --freeze   # also results/rules_frozen.json

--freeze writes the rules fitted on every cell, with SHA-256 checksums of the sweep data and of this
file, for held-out tests (experiments/streamk_aiter_test), which must read them and never refit.
The rules take the tile (BLOCK_M, BLOCK_N, BLOCK_K) and the program count as parameters, so the same
family can be refitted for another tile; sweep F's rows are all 256x256x64 on 256 programs.

Regret = kernel time of the rule's choice / kernel time of the fastest measured choice in the cell
among the configurations the rule can pick (S <= 128). Kernel time, not tail: a tail 20% long can be
1% of the kernel. A rule's n that was not measured is replaced by the nearest measured n (log scale),
and an n at least the default split's by the default; the report counts those cells. A separate table
gives what the configurations no rule picks (chunk-major, l2, v20's slice, two-tile) would add.

Rules:
- capped: v24's host rule as it is (capped_policy, replayed from the kernel's source); fp32.
- fitted, one rounding (data-parallel, v24 reduce-scatter, v24 owner; fp32): data-parallel if the
  idle share of the last wave, (256 - S) k / 256 k-steps, is below T; otherwise reduce-scatter if a
  tile can have at least E programs (256 // S >= E), with n = k / m (m k-steps per program), at most
  P // S; else the owner ending with n = round(sqrt(k / c)), c = max(c0, S / c1). n is at most
  256 // S and one program per pair of k-steps.
- fitted, any rounding: the same with bf16-pair partials, and the atomic ending (at the
  reduce-scatter n) instead of reduce-scatter from S >= A.
Fitting: every parameter set of the grid is scored once; the fit minimises the 90th-percentile
regret, then the worst, then the mean. Cross-validation fits with one S (or one K) left out and
scores the held-out cells.
"""

import ast
import hashlib
import itertools
import json
import math
import os
import statistics
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sweep  # noqa: E402

V24 = os.path.join(HERE, "..", "..", "kernels", "gemm", "intra_wave", "a16w16", "v24_streamk_combined",
                   "matmul_kernel.py")
# Sweep F's tile and program count; the rules take them as parameters.
TILE = (256, 256, 64)
NUM_PROGRAMS = 256
GRID = dict(T=(8, 12, 16, 20, 24, 28, 32), E=(2, 3, 4, 6, 8), m=(2, 3, 4, 6, 8), P=(64, 96, 128, 192, 256),
            c0=(1.5, 2.2, 3.0), c1=(3.0, 5.0, 1e9))
GRID_A = (16, 24, 32, 48, 64, 1e9)
CHOICES = {"one": ("v13", "v24 rs", "v24 owner"), "any": ("v13", "v24 rs bf16_pairs", "v24 owner bf16_pairs",
                                                            "v24 atomic_iz")}
ALL = {"one": ["v13", "v24 two_tile", *[c for c in sweep.SPLIT_F if c not in sweep.MULTI_ROUNDING_F]],
       "any": ["v13", "v24 two_tile", *sweep.SPLIT_F]}


def load_capped():
    """v24's capped_policy, taken from the kernel's source (needs only math, so Triton is not imported)."""
    src = open(V24).read()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "capped_policy")
    ns = {"math": math}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), V24, "exec"), ns)
    return ns["capped_policy"]


def load():
    """{(S, K): {(config, n): kernel us}} for S <= 128."""
    data = {}
    for r in sweep.load("f"):
        if r["config"] != "v13 base" and r["S"] <= 128:
            data.setdefault((r["S"], r["K"]), {})[(r["config"], r["n"])] = r["us"]
    return data


def measured(cell, cfg, n_req, S, K, tile=TILE, programs=NUM_PROGRAMS):
    """(us, n used, approximated) for config cfg at n_req in a cell, or None."""
    if cfg == "v13":
        return (cell[("v13", "-")], "-", False) if ("v13", "-") in cell else None
    ns = {n: us for (c, n), us in cell.items() if c == cfg}
    if not ns:
        return None
    full = min(programs // S, K // (2 * tile[2]))
    if n_req >= full and "default" in ns:
        return ns["default"], "default", n_req != full
    if n_req in ns:
        return ns[n_req], n_req, False
    ints = [n for n in ns if isinstance(n, int)]
    n = min(ints, key=lambda x: (abs(math.log(x) - math.log(n_req)), x))
    return ns[n], n, True


def oracle(cell, cfgs):
    v = [us for (c, _), us in cell.items() if c in cfgs]
    return min(v) if v else None


def fitted_choice(S, K, p, rounding, tile=TILE, programs=NUM_PROGRAMS):
    """(configuration, n) of the fitted rule for S leftover tiles (the leftover of the last wave, or every
    tile when there is less than a wave) and K; "v13" is data-parallel. P is in programs of a
    256-program wave and scales with the program count."""
    k = K // tile[2]
    if (programs - S) * k / programs < p["T"]:
        return "v13", 1
    cap = min(programs // S, k // 2)
    suffix = " bf16_pairs" if rounding == "any" else ""
    if programs // S >= p["E"]:
        n = max(1, min(cap, round(p["P"] * programs / NUM_PROGRAMS) // S, round(k / p["m"])))
        if rounding == "any" and S >= p["A"]:
            return "v24 atomic_iz", n
        return "v24 rs" + suffix, n
    n = max(1, min(cap, round(math.sqrt(k / max(p["c0"], S / p["c1"])))))
    return "v24 owner" + suffix, n


def capped_choice(S, K, capped, total_tiles=None, tile=TILE, programs=NUM_PROGRAMS):
    pol, fixup, snp = capped(total_tiles or programs + S, K // tile[2], programs)
    if pol == "dp":
        return "v13", 1
    n = programs if snp is None else snp // S
    return ("v24 rs" if fixup == "rs" else "v24 owner"), n


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def freeze(fits):
    """results/rules_frozen.json: the fitted rules for held-out tests, with what they were fitted on."""
    path = os.path.join(HERE, "results", "rules_frozen.json")
    out = {"fitted_on": "sweep F (4096x8192xK, GEMM_TILE_LIMIT = 256 + S, S <= 128, K = 512-16384), every cell",
           "tile": TILE, "num_programs": NUM_PROGRAMS, "frozen": time.strftime("%Y-%m-%d %H:%M"),
           "sha256": {"sweep_f.jsonl": sha256(os.path.join(HERE, "results", "sweep_f.jsonl")),
                      "rules_f.py": sha256(os.path.abspath(__file__))},
           "rules": {r: {k: (None if v >= 1e9 else v) for k, v in p.items()} for r, p in fits.items()}}
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    return path


def evaluate(data, cells, choose, cfgs):
    """{cell: (regret, approximated, (config, n))}."""
    out = {}
    for c in cells:
        cell = data[c]
        cfg, n = choose(*c)
        m = measured(cell, cfg, n, *c)
        o = oracle(cell, cfgs)
        if m is not None and o:
            out[c] = (m[0] / o, m[2], (cfg, m[1]))
    return out


def grid_regrets(data, cells, rounding):
    """(parameter sets, regret matrix [sets x cells]) for the fitted family."""
    keys = list(GRID)
    sets = [dict(zip(keys, v), A=a) for v in itertools.product(*(GRID[k] for k in keys))
            for a in (GRID_A if rounding == "any" else (1e9,))]
    cache = {}
    reg = np.empty((len(sets), len(cells)))
    for i, p in enumerate(sets):
        for j, c in enumerate(cells):
            ch = fitted_choice(*c, p, rounding)
            if (c, ch) not in cache:
                m = measured(data[c], ch[0], ch[1], *c)
                cache[(c, ch)] = m[0] / oracle(data[c], CHOICES[rounding]) if m else np.nan
            reg[i, j] = cache[(c, ch)]
    return sets, reg


def best_set(reg, cols):
    """Index of the parameter set with the smallest (90th percentile, worst, mean) regret over cols."""
    sub = np.sort(np.nan_to_num(reg[:, cols], nan=9.0), axis=1)
    p90 = sub[:, int(0.9 * (sub.shape[1] - 1))]
    key = np.lexsort((sub.mean(axis=1), sub[:, -1], p90))
    return key[0]


def cross_validate(data, cells, rounding, sets, reg, axis):
    held = {}
    for g in sorted({c[axis] for c in cells}):
        train = [j for j, c in enumerate(cells) if c[axis] != g]
        p = sets[best_set(reg, train)]
        test = [c for c in cells if c[axis] == g]
        held |= evaluate(data, test, lambda S, K: fitted_choice(S, K, p, rounding), CHOICES[rounding])
    return held


def stats(res):
    r = sorted(v[0] for v in res.values())
    q = lambda f: r[min(len(r) - 1, int(f * len(r)))]  # noqa: E731
    return (f"{len(r)} | {statistics.median(r):.3f} | {q(0.9):.3f} | {r[-1]:.3f} | {sum(x > 1.02 for x in r)} | "
            f"{sum(v[1] for v in res.values())}")


def main():
    data = load()
    cells = sorted(data)
    capped = load_capped()
    fits, rules = {}, {}
    rules["capped (v24 host rule), one rounding"] = evaluate(
        data, cells, lambda S, K: capped_choice(S, K, capped), CHOICES["one"])
    for rounding in ("one", "any"):
        sets, reg = grid_regrets(data, cells, rounding)
        p = sets[best_set(reg, list(range(len(cells))))]
        fits[rounding] = p
        label = "one rounding" if rounding == "one" else "any rounding"
        rules[f"fitted, {label}"] = evaluate(data, cells, lambda S, K, p=p, r=rounding: fitted_choice(S, K, p, r),
                                             CHOICES[rounding])
        rules[f"fitted, {label}, leave one S out"] = cross_validate(data, cells, rounding, sets, reg, 0)
        rules[f"fitted, {label}, leave one K out"] = cross_validate(data, cells, rounding, sets, reg, 1)

    out = ["# Host rules against sweep F\n",
           "Regret = kernel time of the rule's choice / kernel time of the fastest measured choice in the cell among "
           "the configurations the rule can pick: data-parallel, v24 reduce-scatter and owner with fp32 partials "
           "(`one rounding`), or data-parallel, reduce-scatter and owner with bf16-pair partials and the atomic "
           f"ending (`any rounding`). S <= 128, {len(cells)} cells of 4096x8192xK. Approximated: cells where the "
           "rule's n was not measured and the nearest measured n stood in. The leave-one-out rows fit without that "
           "S (or K) and score the held-out cells only.\n",
           "| rule | cells | median regret | 90th percentile | worst | cells more than 2% off | approximated |",
           "|---|---|---|---|---|---|---|"]
    for name, res in rules.items():
        out.append(f"| {name} | {stats(res)} |")
    fmt = lambda p: ", ".join(f"{k} = {('inf' if v >= 1e9 else v)}" for k, v in p.items())  # noqa: E731
    out.append(f"\nFitted on every cell. One rounding: {fmt({k: v for k, v in fits['one'].items() if k != 'A'})}. "
               f"Any rounding: {fmt(fits['any'])}. T: idle k-steps below which data-parallel runs; E: programs per "
               "tile from which reduce-scatter runs; m: k-steps per reduce-scatter program; P: programs the "
               "reduce-scatter split may use (n <= P // S); c0, c1: the owner's c = max(c0, S / c1); A: S from which "
               "the atomic ending replaces bf16 reduce-scatter.\n")
    out.append("\n## What the configurations no rule picks would add\n")
    out.append("Kernel time of the fastest choice among the rule's configurations / the fastest among every "
               "configuration of the same rounding class (chunk-major, l2 with the XCD-local split, v20's slice, "
               "two-tile).\n")
    out.append("| rounding | median | 90th percentile | worst | cells more than 2% |")
    out.append("|---|---|---|---|---|")
    for rounding in ("one", "any"):
        g = sorted(oracle(data[c], CHOICES[rounding]) / oracle(data[c], ALL[rounding]) for c in cells)
        out.append(f"| {rounding} | {statistics.median(g):.3f} | {g[int(0.9 * (len(g) - 1))]:.3f} | {g[-1]:.3f} | "
                   f"{sum(x > 1.02 for x in g)} |")
    for name in ("capped (v24 host rule), one rounding", "fitted, one rounding", "fitted, any rounding"):
        res = rules[name]
        worst = sorted(res.items(), key=lambda kv: -kv[1][0])[:8]
        out.append(f"\n## {name}: the worst cells\n")
        out.append("| S | K | regret | choice |")
        out.append("|---|---|---|---|")
        for (S, K), (reg_, approx, (cfg, n)) in worst:
            out.append(f"| {S} | {K} | {reg_:.3f} | {cfg} n = {n}{' (nearest measured)' if approx else ''} |")
    text = "\n".join(out) + "\n"
    with open(os.path.join(HERE, "results", "rules_f.md"), "w") as f:
        f.write(text)
    print(text)

    fig, ax = plt.subplots(figsize=(8, 5))
    hi = 1.0
    for name, res in rules.items():
        r = np.sort([v[0] for v in res.values()])
        hi = max(hi, r[int(0.97 * (len(r) - 1))])
        ax.step(r, np.arange(1, len(r) + 1) / len(r), where="post", label=name, ls="--" if "leave" in name else "-")
    ax.set_xlim(0.995, min(1.5, hi * 1.02))
    ax.set_xlabel("kernel time / fastest measured kernel time among the rule's configurations")
    ax.set_ylabel("share of cells")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("Sweep F: regret of host rules (cumulative over cells, S <= 128)", fontsize=10)
    fig.tight_layout()
    img = os.path.join(HERE, "images", "sweep_f_regret.png")
    os.makedirs(os.path.dirname(img), exist_ok=True)
    fig.savefig(img, dpi=130)
    print(img)
    if "--freeze" in sys.argv:
        print(freeze(fits))


if __name__ == "__main__":
    main()
