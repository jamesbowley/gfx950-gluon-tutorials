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
"""The stream-K host rules on AITER's shapes, a held-out test set: nothing here is fitted.

    python experiments/streamk_aiter_test/test.py cases                       # count the cases
    HIP_VISIBLE_DEVICES=1 python experiments/streamk_aiter_test/test.py check [--jobs 3]
    HIP_VISIBLE_DEVICES=0 python experiments/streamk_aiter_test/test.py run   # results/test.jsonl
    python experiments/streamk_aiter_test/test.py report                      # results/test.md

Shapes: AITER's 630 bf16 shapes (experiments/aiter_lixun_shapes/supported_status_scoped.csv), those
with a partial last wave of 256x256 tiles (S = tiles % 256 > 0): 321 with less than one wave (all their
tiles are leftover) and 128 with one full wave or more. The rest are whole waves, with nothing to choose.

Per shape, every single-rounding choice the rules can make, timed with rocprof as sweep F was (1000
launches over a rotating 512 MB of inputs, median kernel time) but with contiguous B and C, as AITER
calls the libraries (bench.py --no-pad):
- data-parallel: v9 (one program per tile) when the shape has one wave or less, else v13
  (persistent); v9 has less overhead when no program takes a second tile;
- v24 one-tile reduce-scatter and owner (fp32 partials, tile-major), at every n with S n <= 256 and
  n <= K / 128 in sweep C's set, and the default split;
- v24 as deployed (no STREAMK_* set: its capped rule picks policy, ending and n, and hands
  data-parallel shapes of one wave or less to v9);
- the frozen fitted rule's choice (experiments/streamk_split_count/results/rules_frozen.json, fitted on
  sweep F only), added if its n is not in the set.
The fastest of these per shape is the per-shape tuned choice; a rule's loss is its kernel time over that.
Rows record the tile (256x256x64), the program count (256) and the layout, so later sweeps at other
tiles share the format.
"""

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
A16 = os.path.join(REPO, "kernels", "gemm", "intra_wave", "a16w16")
SPLIT_COUNT = os.path.join(REPO, "experiments", "streamk_split_count")
SHAPES_CSV = os.path.join(REPO, "experiments", "aiter_lixun_shapes", "supported_status_scoped.csv")
AITER_COLD = os.path.join(REPO, "experiments", "aiter_lixun_shapes", "results", "details_capped_hbl_rot2048.csv")
FROZEN = os.path.join(SPLIT_COUNT, "results", "rules_frozen.json")
RESULTS = os.path.join(HERE, "results")
sys.path.insert(0, os.path.join(A16, "v20_streamk_reduce_scatter"))
sys.path.insert(0, A16)
sys.path.insert(0, SPLIT_COUNT)

TILE = (256, 256, 64)
PROGRAMS = 256
N_SET = (1, 2, 3, 4, 6, 8, 12, 16, 32, 64)
LAYOUT = "contiguous"
TAG = "CHECK_CASE "
ENV_KEYS = ("STREAMK_NUM_PROGRAMS", "STREAMK_POLICY", "STREAMK_FIXUP", "STREAMK_BALANCE_XCDS", "GEMM_TILE_LIMIT",
            "STREAMK_PARTIALS", "STREAMK_RS_READERS", "STREAMK_EXCHANGE", "STREAMK_PLACEMENT", "STREAMK_XCD_LOCAL")
SPLITS = {"v24 rs": "rs", "v24 owner": "owner"}


def shapes():
    """(M, N, K, tiles, S) of AITER's shapes with a partial last wave, in file order."""
    seen, out = set(), []
    for r in csv.DictReader(open(SHAPES_CSV)):
        M, N, K = int(r["M"]), int(r["N"]), int(r["K"])
        tiles = -(-M // TILE[0]) * -(-N // TILE[1])
        if (M, N, K) in seen or tiles % PROGRAMS == 0:
            continue
        seen.add((M, N, K))
        out.append((M, N, K, tiles, tiles % PROGRAMS))
    return out


def frozen_rule():
    import rules_f

    fz = json.load(open(FROZEN))
    p = {k: (1e9 if v is None else v) for k, v in fz["rules"]["one"].items()}
    return (lambda S, K: rules_f.fitted_choice(S, K, p, "one", tuple(fz["tile"]), fz["num_programs"])), fz


def cases():
    choose, _ = frozen_rule()
    for M, N, K, tiles, S in shapes():
        base = dict(M=M, N=N, K=K, tiles=tiles, S=S, tile=list(TILE), programs=PROGRAMS, layout=LAYOUT)
        dp_v = 9 if tiles <= PROGRAMS else 13
        yield base | dict(config="dp", n="-", snp=None, v=dp_v, env={})
        yield base | dict(config="v24 capped", n="-", snp=None, v=24, env={})
        ns = [n for n in N_SET if S * n <= PROGRAMS and n <= K // (2 * TILE[2])]
        cfg, n_rule = choose(S, K)
        extra = {}
        if cfg != "v13" and n_rule not in ns and n_rule < min(PROGRAMS // S, K // (2 * TILE[2])):
            extra[cfg] = n_rule
        for cfg_s, fixup in SPLITS.items():
            env = {"STREAMK_POLICY": "one_tile", "STREAMK_FIXUP": fixup}
            for n in sorted(set(ns + ([extra[cfg_s]] if cfg_s in extra else []))) + ["default"]:
                yield base | dict(config=cfg_s, n=n, snp=None if n == "default" else S * n, v=24, env=env)


def key(c):
    return (c["M"], c["N"], c["K"], c["config"], str(c["n"]))


def case_env(c):
    env = {**os.environ, **c["env"]}
    for k in ENV_KEYS:
        if k not in c["env"]:
            env.pop(k, None)
    if c["snp"] is not None:
        env["STREAMK_NUM_PROGRAMS"] = str(c["snp"])
    return env


def load(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f]


def run():
    import rocprof_check

    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "test.jsonl")
    done = {key(r) for r in load("test.jsonl")}
    todo = [c for c in cases() if key(c) not in done]
    print(f"{len(todo)} cases to run", flush=True)
    todo = todo[:int(os.environ.get("SWEEP_MAX_CASES", len(todo)))]
    for i, c in enumerate(todo):
        det = rocprof_check.cold_detail(c["v"], c["M"], c["N"], c["K"], case_env(c), bench_flags=("--no-pad",))
        row = {k: c[k] for k in ("M", "N", "K", "tiles", "S", "config", "n", "snp", "tile", "programs", "layout")}
        with open(path, "a") as f:
            f.write(json.dumps(row | {"us": det["us"]}) + "\n")
        print(f"[{i + 1}/{len(todo)}] {row['M']}x{row['N']}x{row['K']} {row['config']} n={row['n']} "
              f"{det['us']:.1f} us", flush=True)


def check_worker(K, group):
    sys.path.insert(0, A16)
    os.chdir(A16)
    import importlib

    import bench  # noqa: F401  (plugin hooks and the path to common.py)
    import torch

    mods = {}
    for c in group:
        for k in ENV_KEYS:
            os.environ.pop(k, None)
        os.environ.update({k: v for k, v in case_env(c).items() if k in ENV_KEYS})
        name = {9: "v9_beyond_hotloop", 13: "v13_persistant_peel_acc", 24: "v24_streamk_combined"}[c["v"]]
        mk = mods.setdefault(name, importlib.import_module(f"{name}.matmul_kernel"))
        res = {"wrong": 0, "error": None}
        try:
            torch.manual_seed(0)
            a = torch.rand((c["M"], K), device=bench.DEVICE, dtype=torch.bfloat16) - 0.5
            # B as the timed runs (bench.py --no-pad) and AITER pass it: (N, K) weights, seen transposed.
            b = bench.make_b(c["N"], K, torch.bfloat16, False, lambda t: t.uniform_(-0.5, 0.5))
            ref = torch.matmul(a, b)
            for _ in range(2):
                out = mk.matmul(a, b)
                res["wrong"] += int(not torch.allclose(out, ref, atol=1e-1, rtol=1e-2))
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            res["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
        print(TAG + json.dumps({k: c[k] for k in ("M", "N", "K", "config", "n")} | res), flush=True)


def check(jobs):
    groups = {}
    for c in cases():
        groups.setdefault(c["K"], []).append(c)

    def launch(K):
        proc = subprocess.run([sys.executable, "-W", "ignore", os.path.abspath(__file__), "check_worker", str(K)],
                              capture_output=True, text=True)
        return K, proc.stdout, proc.returncode

    rows = []
    with ThreadPoolExecutor(jobs) as ex:
        for K, out, rc in ex.map(launch, sorted(groups, key=lambda k: -len(groups[k]))):
            got = [json.loads(l[len(TAG):]) for l in out.splitlines() if l.startswith(TAG)]
            rows += got
            print(f"K = {K}: {len(got)} of {len(groups[K])} cases, exit {rc}", file=sys.stderr, flush=True)
    bad = [r for r in rows if r["wrong"] or r["error"]]
    total = sum(len(g) for g in groups.values())
    print(f"# AITER test set: correctness ({LAYOUT} B and C)\n\n{len(rows)} of {total} cases ran, 2 launches each "
          f"against torch.matmul (atol 0.1, rtol 0.01).\n")
    for r in bad[:50]:
        print(f"- FAIL {r}")
    print("\nPASS" if not bad and len(rows) == total else "\nFAIL")


def report():
    rows = load("test.jsonl")
    _, fz = frozen_rule()
    choose, _ = frozen_rule()
    by_shape = {}
    for r in rows:
        by_shape.setdefault((r["M"], r["N"], r["K"]), {})[(r["config"], str(r["n"]))] = r
    aiter = {}
    if os.path.exists(AITER_COLD):
        for r in csv.DictReader(open(AITER_COLD)):
            if r.get("aiter_pick_tflops"):
                aiter[(int(r["M"]), int(r["N"]), int(r["K"]))] = (float(r["aiter_pick_tflops"]), r.get("aiter_pick", ""))

    def rule_row(shape, cell, S, K):
        cfg, n = choose(S, K)
        if cfg == "v13":
            return cell.get(("dp", "-"))
        full = min(PROGRAMS // S, K // (2 * TILE[2]))
        return cell.get((cfg, "default" if n >= full else str(n)))

    groups = {"less than one wave": [], "one full wave or more": []}
    detail = []
    for (M, N, K), cell in sorted(by_shape.items()):
        r0 = next(iter(cell.values()))
        tiles, S = r0["tiles"], r0["S"]
        best = min(cell.values(), key=lambda r: r["us"])
        dp = cell.get(("dp", "-"))
        cap = cell.get(("v24 capped", "-"))
        fit = rule_row((M, N, K), cell, S, K)
        if not (dp and cap and fit):
            continue
        g = "less than one wave" if tiles < PROGRAMS else "one full wave or more"
        flops = 2 * M * N * K
        rec = dict(M=M, N=N, K=K, tiles=tiles, S=S, best=best, dp=dp["us"] / best["us"], capped=cap["us"] / best["us"],
                   fitted=fit["us"] / best["us"], fit_choice=(fit["config"], fit["n"]), best_tflops=flops / best["us"] / 1e6,
                   aiter=aiter.get((M, N, K)))
        groups[g].append(rec)
        detail.append(rec)

    def q(xs, f):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(f * len(xs)))]

    out = ["# The stream-K host rules on AITER's shapes (held-out test set)\n",
           f"rocprof kernel times, {LAYOUT} B and C, 1000 launches over rotating inputs, median. Rules frozen "
           f"{fz['frozen']} from sweep F only (sha256 of sweep_f.jsonl {fz['sha256']['sweep_f.jsonl'][:12]}). Loss = "
           "kernel time / the fastest single-rounding choice measured for that shape (data-parallel, v24 one-tile "
           "reduce-scatter or owner at any n, or v24 as deployed). Data-parallel is v9 at one wave or less, v13 above.\n",
           "| shapes | count | data-parallel: median / 90th pct / worst | v24 capped (deployed) | fitted rule (frozen) | "
           "shapes where the fitted rule loses more than 2% |",
           "|---|---|---|---|---|---|"]
    for g, recs in groups.items():
        if not recs:
            continue
        cells = [f"{statistics.median([r[x] for r in recs]):.3f} / {q([r[x] for r in recs], 0.9):.3f} / "
                 f"{max(r[x] for r in recs):.3f}" for x in ("dp", "capped", "fitted")]
        out.append(f"| {g} | {len(recs)} | " + " | ".join(cells) + f" | {sum(r['fitted'] > 1.02 for r in recs)} |")
    out.append("\n## Every shape\n")
    out.append("| M | N | K | tiles | S | fastest (n) | TFLOPS | data-parallel | capped | fitted (choice) | "
               "AITER's tuned pick, TFLOPS (other tiles; do_bench timing) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in detail:
        b = r["best"]
        a = f"{r['aiter'][0]:.0f} ({r['aiter'][1]})" if r["aiter"] else "-"
        out.append(f"| {r['M']} | {r['N']} | {r['K']} | {r['tiles']} | {r['S']} | {b['config']} ({b['n']}) | "
                   f"{r['best_tflops']:.0f} | {r['dp']:.3f} | {r['capped']:.3f} | {r['fitted']:.3f} "
                   f"({r['fit_choice'][0]} {r['fit_choice'][1]}) | {a} |")
    text = "\n".join(out) + "\n"
    with open(os.path.join(RESULTS, "test.md"), "w") as f:
        f.write(text)
    print(text)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["cases", "todo", "check", "check_worker", "run", "report"])
    p.add_argument("arg", nargs="?")
    p.add_argument("--jobs", type=int, default=3)
    a = p.parse_args()
    if a.stage == "cases":
        cs = list(cases())
        sh = shapes()
        print(f"{len(sh)} shapes ({sum(t < PROGRAMS for *_, t, _ in sh)} with less than one wave), {len(cs)} cases")
    elif a.stage == "todo":
        done = {key(r) for r in load("test.jsonl")}
        print(sum(key(c) not in done for c in cases()))
    elif a.stage == "check":
        check(a.jobs)
    elif a.stage == "check_worker":
        K = int(a.arg)
        check_worker(K, [c for c in cases() if c["K"] == K])
    elif a.stage == "run":
        run()
    else:
        report()


if __name__ == "__main__":
    main()
