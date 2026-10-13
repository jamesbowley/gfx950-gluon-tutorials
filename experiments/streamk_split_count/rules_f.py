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
"""Host rules against sweeps F and G (no GPU): which (configuration, n) a rule picks per (S, K), and its
kernel-time regret against the fastest measured choice.

    python experiments/streamk_split_count/rules_f.py            # results/rules_f.md, images/sweep_[fg]_regret.png
    python experiments/streamk_split_count/rules_f.py --freeze   # also results/rules_frozen.json (version 2)
    python experiments/streamk_split_count/rules_f.py --freeze-v3   # version 3: version 2 + sweep H's spread test

--freeze writes the rules fitted on every cell, with SHA-256 checksums of the sweep data and of this
file, for held-out tests (experiments/streamk_aiter_test), which must read them and never refit.
Version 2 has two regimes: one wave or more (sweep F's constants, unchanged from version 1, which is
kept as results/rules_frozen_v1.json) and less than one wave (tiles < programs; fitted on sweep G).
The rules take the tile (BLOCK_M, BLOCK_N, BLOCK_K) and the program count as parameters, so the same
family can be refitted for another tile; sweep F's and G's rows are all 256x256x64 on 256 programs.

Sweep G (GEMM_TILE_LIMIT = S, no full wave) runs data-parallel on v9_ragged and the splits on v25 (v24's
code on aligned shapes); the rule's labels name roles there: "v13" is data-parallel, "v24" the stream-K
kernel. S > 128 (a tile-aligned split has one program per tile there) has its own branch: spread
(end-to-end reversed cuts, all programs) if k >= k_s, else data-parallel, k_s fitted per regime: on
sweep G's cells above 128 (less than one wave) and on sweep F's 15 (one wave or more, where
data-parallel is whichever of v9_ragged and v13 sweep G's full-wave cells show faster).

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
# Less than one wave (sweep G): the same family on a wider grid (sweep F's grid stays as it was, so its
# constants stay version 1's).
GRID_G = dict(T=(4, 6, 8, 12, 14, 16, 20, 24, 28, 32, 40, 48), E=(1, 2, 3, 4, 5, 6, 8), m=(2, 3, 4, 6, 8, 12, 16),
              P=(64, 96, 128, 192, 256), c0=(1.5, 2.2, 3.0, 4.0), c1=(3.0, 5.0, 1e9))
# The same family with one more cap on the reduce-scatter split, n <= Q / sqrt(S): below one wave the best n
# falls as about 32 / sqrt(S) (sweep G), which neither P // S nor k / m can follow.
GRID_G_SQRT = dict(T=(8, 12, 14, 16, 20, 24), E=(3, 4, 5, 6, 8), m=(2, 3, 4, 6, 8), P=(128, 256),
                   Q=(16, 24, 32, 40, 48, 64, 1e9), c0=(1.5, 2.2, 3.0), c1=(3.0, 5.0, 1e9))
G_ROLE = {"v9r dp": "v13", "v25 rs": "v24 rs", "v25 owner": "v24 owner"}
REGIMES = ("one_wave_or_more", "less_than_one_wave")
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


def load_g():
    """Sweep G's cells with S <= 128 (no full wave): {(S, K): {(role, n): kernel us}}, in the rule's labels."""
    data = {}
    for r in sweep.load("g"):
        if r["limit"] == r["S"] and r["S"] <= 128 and r["config"] in G_ROLE:
            data.setdefault((r["S"], r["K"]), {})[(G_ROLE[r["config"]], r["n"])] = r["us"]
    return data


def load_spread():
    """S > 128: ({regime: {(S, K): {kernel: us}}}, the data-parallel kernel above one wave, v9_ragged / v13
    kernel times over sweep G's full-wave cells). Kernels: "v9r dp", "v13", "spread". Less than one wave:
    sweep G (T = S). One wave or more: sweep F's v13 and spread (T = 256 + S), with sweep G's v9_ragged."""
    g = sweep.load("g")
    less, more = {}, {}
    for r in g:
        if r["S"] > 128 and r["limit"] == r["S"] and r["config"] in ("v9r dp", "v13", "v25 spread"):
            less.setdefault((r["S"], r["K"]), {})["spread" if r["config"] == "v25 spread" else r["config"]] = r["us"]
    full = {(r["S"], r["K"], r["config"]): r["us"] for r in g if r["limit"] == 256 + r["S"]}
    ratios = [full[(S, K, "v9r dp")] / full[(S, K, "v13")] for S, K, c in sorted(full)
              if c == "v13" and (S, K, "v9r dp") in full]
    dp_above = "v9r dp" if ratios and statistics.median(ratios) < 1 else "v13"
    for r in sweep.load("f"):
        if r["S"] > 128 and r["config"] in ("v13", "v24 spread"):
            more.setdefault((r["S"], r["K"]), {})["spread" if r["config"] == "v24 spread" else "v13"] = r["us"]
    for (S, K), cell in more.items():
        if (S, K, "v9r dp") in full:
            cell["v9r dp"] = full[(S, K, "v9r dp")]
    return {"less_than_one_wave": less, "one_wave_or_more": more}, dp_above, ratios


def spread_oracle(cell, regime):
    """The fastest of spread and the data-parallel kernels either rule can run in the regime (v9_ragged below
    one wave; v9_ragged or v13 with one wave or more)."""
    dps = ("v9r dp",) if regime == "less_than_one_wave" else ("v9r dp", "v13")
    return min([cell["spread"]] + [cell[x] for x in dps if x in cell])


def spread_evaluate(data, cells, k_s, dp, regime, tile=TILE):
    """{cell: (regret, False, (choice, "-"))} of "spread if k >= k_s, else dp"."""
    out = {}
    for S, K in cells:
        cell = data[(S, K)]
        if "spread" not in cell or dp not in cell:
            continue
        ch = "spread" if K // tile[2] >= k_s else dp
        out[(S, K)] = (cell[ch] / spread_oracle(cell, regime), False, (ch, "-"))
    return out


def fit_k_s(data, cells, dp, regime):
    """k_s among the measured k (and never), by the 90th-percentile regret, then the worst, then the mean."""
    grid = sorted({K // TILE[2] for _, K in cells}) + [1e9]
    reg = np.array([[v[0] for v in spread_evaluate(data, cells, k, dp, regime).values()] for k in grid])
    return grid[best_set(reg, list(range(reg.shape[1])))]


def load_h():
    """Sweep H by regime: {regime: {(S, K): {"dp": us, "spread": us}}}, means of the rounds; data-parallel is
    v9_ragged below one wave and v13 with one, spread is v26's."""
    t = sweep.h_means()
    out = {"less_than_one_wave": {}, "one_wave_or_more": {}}
    for (S, K, lim, cfg), us in t.items():
        regime = "less_than_one_wave" if lim == S else "one_wave_or_more"
        role = {"v9r dp": "dp", "v13": "dp", "v26 spread": "spread"}.get(cfg)
        if role:
            out[regime].setdefault((S, K), {})[role] = sum(us) / len(us)
    return out


# Spread tests for S > programs / 2, by the feature they threshold.
SPREAD_FEATURES = {"k": lambda S, k: k, "(256 - S) k": lambda S, k: (NUM_PROGRAMS - S) * k}


def threshold_evaluate(data, cells, feature, thr):
    """{cell: (regret, False, (choice, "-"))} of "spread if feature >= thr" against the faster of dp and spread."""
    f = SPREAD_FEATURES[feature]
    out = {}
    for S, K in cells:
        cell = data[(S, K)]
        ch = "spread" if f(S, K // TILE[2]) >= thr else "dp"
        out[(S, K)] = (cell[ch] / min(cell["dp"], cell["spread"]), False, (ch, "-"))
    return out


def fit_threshold(data, cells, feature):
    """The cut among the feature's measured values (and never), by the 90th-percentile regret, then the worst, then
    the mean, placed at the geometric mean of the measured values on either side of it."""
    f = SPREAD_FEATURES[feature]
    vals = sorted({f(S, K // TILE[2]) for S, K in cells})
    grid = vals + [1e9]
    reg = np.array([[v[0] for v in threshold_evaluate(data, cells, feature, t).values()] for t in grid])
    i = best_set(reg, list(range(reg.shape[1])))
    if 0 < i < len(vals):
        return round(math.sqrt(vals[i - 1] * vals[i]))
    return grid[i]


def sweep_h_sections():
    """rules_f.md's sweep H section: (lines, {regime: C}). Empty without sweep H's rows."""
    if not os.path.exists(os.path.join(HERE, "results", "sweep_h.jsonl")):
        return [], None
    data = load_h()
    out = ["\n# S > 128 on sweep H: which spread test\n",
           "Sweep H (4096x8192xK, S = 132-255, K = 2048-32768, below one wave and with one, two rounds, means): spread "
           "(v26) or data-parallel (v9_ragged below one wave, v13 with one). Regret against the faster of the two. "
           "`capped` (and version 2) spread from k >= 512; the fitted tests threshold k alone or the data-parallel last "
           "wave's idle k-steps, (256 - S) k, fitted on sweep H only (the cut between measured values that scores "
           "best, placed at the geometric mean of the values on either side), with leave-one-S-out and "
           "leave-one-K-out rows.\n"]
    fits = {}
    for regime in REGIMES:
        d = data[regime]
        cs = sorted(c for c, cell in d.items() if "dp" in cell and "spread" in cell)
        if not cs:
            continue
        res = {"capped / version 2 (k >= 512)": threshold_evaluate(d, cs, "k", 512)}
        for feat in SPREAD_FEATURES:
            thr = fit_threshold(d, cs, feat)
            if feat == "(256 - S) k":
                fits[regime] = thr
            res[f"fitted, {feat} >= {'inf' if thr >= 1e9 else thr}"] = threshold_evaluate(d, cs, feat, thr)
            for axis, name in ((0, "S"), (1, "K")):
                held = {}
                for g in sorted({c[axis] for c in cs}):
                    train = [c for c in cs if c[axis] != g]
                    held |= threshold_evaluate(d, [c for c in cs if c[axis] == g], feat, fit_threshold(d, train, feat))
                res[f"fitted, {feat}, leave one {name} out"] = held
        out.append(f"\n## {regime.replace('_', ' ')} ({len(cs)} cells)\n")
        out.append("| rule | cells | median regret | 90th percentile | worst | cells more than 2% off | approximated |")
        out.append("|---|---|---|---|---|---|---|")
        for name, r in res.items():
            out.append(f"| {name} | {stats(r)} |")
    return out, fits


def freeze_v3(c_fit):
    """results/rules_frozen.json, version 3: version 2 with the spread test for S > 128 replaced by
    (256 - S) k >= C per regime (sweep H, spread on v26). Version 2 moves to results/rules_frozen_v2.json once."""
    res = os.path.join(HERE, "results")
    path, v2_path = os.path.join(res, "rules_frozen.json"), os.path.join(res, "rules_frozen_v2.json")
    if not os.path.exists(v2_path):
        with open(path) as f:
            v2 = json.load(f)
        assert v2.get("version") == 2, "rules_frozen.json is not version 2 and rules_frozen_v2.json is missing"
        with open(v2_path, "w") as f:
            json.dump(v2, f, indent=1)
    with open(v2_path) as f:
        out = json.load(f)
    out["version"] = 3
    out["frozen"] = time.strftime("%Y-%m-%d %H:%M")
    out["fitted_on"]["spread_test"] = ("sweep H (4096x8192xK, S = 132-255, K = 2048-32768, GEMM_TILE_LIMIT = S and "
                                       "256 + S, two rounds), every cell: spread if (256 - S) k >= C, per regime")
    out["spread_kernel"] = "v26"
    out["sha256"]["sweep_h.jsonl"] = sha256(os.path.join(res, "sweep_h.jsonl"))
    out["sha256"]["rules_f.py"] = sha256(os.path.abspath(__file__))
    out["sha256_version_2"] = sha256(v2_path)
    for regime, variants in out["rules"].items():
        for p in variants.values():
            p["C"] = None if c_fit[regime] >= 1e9 else c_fit[regime]
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    return path


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
    256-program wave and scales with the program count. With Q (version 2's one_sqrt), the
    reduce-scatter n is also at most Q / sqrt(S). With C (version 3), S > programs / 2 runs spread
    when the data-parallel last wave's idle k-steps, (programs - S) k, are at least C. With k_s (version 2), S > programs / 2 runs
    spread from k >= k_s and data-parallel below; without it (version 1) such S fall through."""
    k = K // tile[2]
    if p.get("C") is not None and 2 * S > programs:
        return ("v24 spread" if (programs - S) * k >= p["C"] else "v13"), 1
    if p.get("k_s") is not None and 2 * S > programs:
        return ("v24 spread" if k >= p["k_s"] else "v13"), 1
    if (programs - S) * k / programs < p["T"]:
        return "v13", 1
    cap = min(programs // S, k // 2)
    suffix = " bf16_pairs" if rounding == "any" else ""
    if programs // S >= p["E"]:
        n = max(1, min(cap, round(p["P"] * programs / NUM_PROGRAMS) // S, round(k / p["m"])))
        if p.get("Q") is not None:
            n = max(1, min(n, round(p["Q"] / math.sqrt(S))))
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


def freeze(fits, fit_g, k_s, dp_above):
    """results/rules_frozen.json, version 2: the fitted rules for held-out tests by regime, with what they were
    fitted on. Version 1 (sweep F only) moves to results/rules_frozen_v1.json once, and sweep F's constants
    must refit to it unchanged."""
    res = os.path.join(HERE, "results")
    path, v1_path = os.path.join(res, "rules_frozen.json"), os.path.join(res, "rules_frozen_v1.json")
    if not os.path.exists(v1_path):
        with open(path) as f:
            v1 = json.load(f)
        assert v1.get("version", 1) == 1, "rules_frozen.json is not version 1 and rules_frozen_v1.json is missing"
        with open(v1_path, "w") as f:
            json.dump(v1, f, indent=1)
    with open(v1_path) as f:
        v1 = json.load(f)
    clean = lambda p: {k: (None if v >= 1e9 else v) for k, v in p.items()}  # noqa: E731
    assert v1["rules"] == {r: clean(p) for r, p in fits.items()}, "sweep F's constants no longer refit to version 1's"
    ks = {g: (None if k >= 1e9 else k) for g, k in k_s.items()}
    dp_name = {"v9r dp": "v9_ragged", "v13": "v13"}
    out = {"version": 2,
           "fitted_on": {"one_wave_or_more": "sweep F (4096x8192xK, GEMM_TILE_LIMIT = 256 + S, S <= 128, K = "
                                             "512-16384), every cell (version 1's constants); k_s on sweep F's 15 "
                                             "cells with S > 128",
                         "less_than_one_wave": "sweep G (4096x8192xK, GEMM_TILE_LIMIT = S, S <= 128, K = 512-16384, "
                                               "and 32768 at S = 128), every cell; k_s on its cells with S > 128"},
           "regime": "less_than_one_wave if tiles < num_programs, else one_wave_or_more",
           "data_parallel": {"less_than_one_wave": "v9_ragged", "one_wave_or_more": dp_name[dp_above]},
           "tile": TILE, "num_programs": NUM_PROGRAMS, "frozen": time.strftime("%Y-%m-%d %H:%M"),
           "sha256": {"sweep_f.jsonl": sha256(os.path.join(res, "sweep_f.jsonl")),
                      "sweep_g.jsonl": sha256(os.path.join(res, "sweep_g.jsonl")),
                      "rules_f.py": sha256(os.path.abspath(__file__))},
           "variants": {"one": "the rule family as version 1, constants by regime",
                        "one_sqrt": "below one wave, the family with the reduce-scatter n also at most Q / sqrt(S); "
                                    "one wave or more as `one`"},
           "rules": {"one_wave_or_more": {r: clean(p) | {"k_s": ks["one_wave_or_more"]} for r, p in fits.items()},
                     "less_than_one_wave": {r: clean(p) | {"k_s": ks["less_than_one_wave"]}
                                            for r, p in fit_g.items()}}}
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


def grid_regrets(data, cells, rounding, grid=GRID):
    """(parameter sets, regret matrix [sets x cells]) for the fitted family."""
    keys = list(grid)
    sets = [dict(zip(keys, v), A=a) for v in itertools.product(*(grid[k] for k in keys))
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
    g_out, fit_g, rules_g, k_s, dp_above = sweep_g_sections(fits["one"], capped)
    out += g_out
    h_out, c_fit = sweep_h_sections()
    out += h_out
    text = "\n".join(out) + "\n"
    with open(os.path.join(HERE, "results", "rules_f.md"), "w") as f:
        f.write(text)
    print(text)

    regret_figure(rules, "Sweep F: regret of host rules (cumulative over cells, S <= 128)", "sweep_f_regret.png")
    if rules_g:
        regret_figure(rules_g, "Sweep G (no full wave): regret of host rules (cumulative over cells, S <= 128)",
                      "sweep_g_regret.png")
    if "--freeze" in sys.argv:
        assert fit_g is not None, "--freeze needs sweep G's rows"
        with open(os.path.join(HERE, "results", "rules_frozen.json")) as f:
            assert json.load(f).get("version", 1) <= 2, "rules_frozen.json is version 3; --freeze-v3 refreezes it"
        print(freeze(fits, fit_g, k_s, dp_above))
    if "--freeze-v3" in sys.argv:
        assert c_fit, "--freeze-v3 needs sweep H's rows"
        print(freeze_v3(c_fit))


def regret_figure(rules, title, name):
    fig, ax = plt.subplots(figsize=(8, 5))
    hi = 1.0
    for label, res in rules.items():
        r = np.sort([v[0] for v in res.values()])
        hi = max(hi, r[int(0.97 * (len(r) - 1))])
        ax.step(r, np.arange(1, len(r) + 1) / len(r), where="post", label=label, ls="--" if "leave" in label else "-")
    ax.set_xlim(0.995, min(1.5, hi * 1.02))
    ax.set_xlabel("kernel time / fastest measured kernel time among the rule's configurations")
    ax.set_ylabel("share of cells")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    img = os.path.join(HERE, "images", name)
    os.makedirs(os.path.dirname(img), exist_ok=True)
    fig.savefig(img, dpi=130)
    plt.close(fig)
    print(img)


def sweep_g_sections(fit_f, capped):
    """rules_f.md's sweep G sections: (lines, {"one": the family's constants below one wave, "one_sqrt": with
    n <= Q / sqrt(S)}, {rule: evaluation} for the
    figure, {regime: k_s}, the data-parallel kernel above one wave). Empty without sweep G's rows."""
    if not os.path.exists(os.path.join(HERE, "results", "sweep_g.jsonl")):
        return [], None, {}, None, None
    data = load_g()
    cells = sorted(c for c, cell in data.items() if ("v13", "-") in cell)
    rules = {"capped (v24 / v25 host rule)": evaluate(
        data, cells, lambda S, K: capped_choice(S, K, capped, total_tiles=S), CHOICES["one"]),
        "sweep F's rule, unchanged (version 1)": evaluate(
        data, cells, lambda S, K: fitted_choice(S, K, fit_f, "one"), CHOICES["one"])}
    sets, reg = grid_regrets(data, cells, "one", GRID_G)
    fit_g = sets[best_set(reg, list(range(len(cells))))]
    rules["fitted on sweep G"] = evaluate(data, cells, lambda S, K: fitted_choice(S, K, fit_g, "one"), CHOICES["one"])
    rules["fitted on sweep G, leave one S out"] = cross_validate(data, cells, "one", sets, reg, 0)
    rules["fitted on sweep G, leave one K out"] = cross_validate(data, cells, "one", sets, reg, 1)
    sets_q, reg_q = grid_regrets(data, cells, "one", GRID_G_SQRT)
    fit_q = sets_q[best_set(reg_q, list(range(len(cells))))]
    rules["fitted on sweep G, n <= Q / sqrt(S)"] = evaluate(
        data, cells, lambda S, K: fitted_choice(S, K, fit_q, "one"), CHOICES["one"])
    rules["fitted on sweep G, n <= Q / sqrt(S), leave one S out"] = cross_validate(data, cells, "one", sets_q, reg_q, 0)
    rules["fitted on sweep G, n <= Q / sqrt(S), leave one K out"] = cross_validate(data, cells, "one", sets_q, reg_q, 1)
    fmt = lambda p: ", ".join(f"{k} = {('inf' if v >= 1e9 else v)}" for k, v in p.items() if k != "A")  # noqa: E731
    out = ["\n# Sweep G: less than one wave\n",
           "Sweep G's cells (4096x8192xK, GEMM_TILE_LIMIT = S, no full wave), S <= 128: regret against the fastest of "
           "data-parallel (v9_ragged), v25 reduce-scatter and owner (fp32) at any measured n. The capped rule is "
           "replayed with S tiles in all; sweep F's rule uses version 1's constants as frozen; `fitted on sweep G` is "
           f"the same family fitted on these {len(cells)} cells only (wider grid); `n <= Q / sqrt(S)` adds one cap to "
           "its reduce-scatter split, n <= Q / sqrt(S) (below one wave the best n falls as about 32 / sqrt(S), which "
           "the family's P // S and k / m cannot follow). Leave-one-out rows as above. Both are frozen in version 2, "
           "before the AITER test is re-scored.\n",
           "| rule | cells | median regret | 90th percentile | worst | cells more than 2% off | approximated |",
           "|---|---|---|---|---|---|---|"]
    for name, res in rules.items():
        out.append(f"| {name} | {stats(res)} |")
    out.append(f"\nFitted on every cell of sweep G (S <= 128): {fmt(fit_g)}. With n <= Q / sqrt(S): {fmt(fit_q)}. "
               f"Sweep F's (version 1): {fmt(fit_f)}.\n")
    for name in ("capped (v24 / v25 host rule)", "sweep F's rule, unchanged (version 1)", "fitted on sweep G",
                 "fitted on sweep G, n <= Q / sqrt(S)"):
        worst = sorted(rules[name].items(), key=lambda kv: -kv[1][0])[:8]
        out.append(f"\n## Sweep G, {name}: the worst cells\n")
        out.append("| S | K | regret | choice |")
        out.append("|---|---|---|---|")
        for (S, K), (reg_, approx, (cfg, n)) in worst:
            out.append(f"| {S} | {K} | {reg_:.3f} | {cfg} n = {n}{' (nearest measured)' if approx else ''} |")
    Ks = sorted({K for _, K in cells})
    short = lambda ch: "dp" if ch[0] == "v13" else f"{ch[0].removeprefix('v24 ')} {'d' if ch[1] == 'default' else ch[1]}"  # noqa: E731,E501
    for first in ("fitted on sweep G", "fitted on sweep G, n <= Q / sqrt(S)"):
        out.append(f"\n## Sweep G, the rules' choices per cell ({first} / capped, regret in brackets; dp = "
                   "data-parallel, d = default split)\n")
        out.append("| S | " + " | ".join(f"K = {K}" for K in Ks) + " |")
        out.append("|---|" + "---|" * len(Ks))
        for S in sorted({S for S, _ in cells}):
            row = []
            for K in Ks:
                a, b = rules[first].get((S, K)), rules["capped (v24 / v25 host rule)"].get((S, K))
                row.append(f"{short(a[2])} ({a[0]:.3f}) / {short(b[2])} ({b[0]:.3f})" if a and b else "-")
            out.append(f"| {S} | " + " | ".join(row) + " |")

    sp, dp_above, ratios = load_spread()
    dp_of = {"less_than_one_wave": "v9r dp", "one_wave_or_more": dp_above}
    k_s, spread_rules = {}, {}
    for regime in REGIMES:
        d = sp[regime]
        cs = sorted(c for c, cell in d.items() if "spread" in cell and dp_of[regime] in cell)
        if not cs:
            continue
        k_s[regime] = fit_k_s(d, cs, dp_of[regime], regime)
        res = {"capped (spread if k >= 512; its data-parallel kernel)": spread_evaluate(
                   d, cs, 512, "v9r dp" if regime == "less_than_one_wave" else "v13", regime),
               f"fitted, k_s = {'inf' if k_s[regime] >= 1e9 else k_s[regime]}": spread_evaluate(
                   d, cs, k_s[regime], dp_of[regime], regime)}
        for axis, name in ((0, "S"), (1, "K")):
            held = {}
            for g in sorted({c[axis] for c in cs}):
                train = [c for c in cs if c[axis] != g]
                held |= spread_evaluate(d, [c for c in cs if c[axis] == g],
                                        fit_k_s(d, train, dp_of[regime], regime), dp_of[regime], regime)
            res[f"fitted, leave one {name} out"] = held
        spread_rules[regime] = (cs, res)
    if spread_rules:
        rs = sorted(ratios)
        out.append("\n# S > 128: spread or data-parallel\n")
        out.append("A tile-aligned split has one program per tile above 128 (S n <= 256), which is data-parallel plus "
                   "overhead, so the rule's only split there is spread (end-to-end reversed cuts over all programs): "
                   "spread if k >= k_s, else data-parallel. Regret against the faster of the two. Below one wave "
                   "(sweep G, T = S) data-parallel is v9_ragged and spread is v25's. With one wave or more (sweep F's "
                   "15 cells, T = 256 + S: v13 and v24 spread) data-parallel is the faster of v9_ragged and v13 by "
                   "sweep G's full-wave cells"
                   + (f": v9_ragged / v13 median {statistics.median(rs):.3f} (range {rs[0]:.3f}-{rs[-1]:.3f}, "
                      f"{len(rs)} cells), so `{dp_above}`" if rs else f": not measured, so `{dp_above}`")
                   + ". capped's data-parallel kernel is v9_ragged below one wave (its fallback) and v13 above.\n")
        for regime, (cs, res) in spread_rules.items():
            out.append(f"\n## {regime.replace('_', ' ')} ({len(cs)} cells)\n")
            out.append("| rule | cells | median regret | 90th percentile | worst | cells more than 2% off | approximated |")
            out.append("|---|---|---|---|---|---|---|")
            for name, r in res.items():
                out.append(f"| {name} | {stats(r)} |")
            out.append("\n| S | K | data-parallel | spread | spread / data-parallel |")
            out.append("|---|---|---|---|---|")
            for S, K in cs:
                cell = sp[regime][(S, K)]
                out.append(f"| {S} | {K} | {cell[dp_of[regime]]:.1f} | {cell['spread']:.1f} | "
                           f"{cell['spread'] / cell[dp_of[regime]]:.3f} |")
    return out, {"one": fit_g, "one_sqrt": fit_q}, rules, k_s, dp_above


if __name__ == "__main__":
    main()
