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
"""Spread or data-parallel for S > 128 on ragged shapes (v26), against aligned twins.

    python experiments/streamk_split_count/ragged_spread.py cases                       # count
    HIP_VISIBLE_DEVICES=1 python experiments/streamk_split_count/ragged_spread.py check  # against torch
    HIP_VISIBLE_DEVICES=0 python experiments/streamk_split_count/ragged_spread.py run    # results/ragged_spread.jsonl
    python experiments/streamk_split_count/ragged_spread.py report                       # results/ragged_spread.md

Sweeps G and H, and every AITER shape, have M and N multiples of 256 and K of 128, so v26's spread test
((256 - S) k >= 10493) was never fitted or tested with its edge builds, which spill more (EDGE_M / EDGE_N,
and more with a bias). Real shapes (no GEMM_TILE_LIMIT) of S tiles (less than one wave) or 256 + S (one full
wave first), S = 136-248, K = 4096-32768, in four variants of the same tile grid: aligned, M and N ragged
(M = 256 tm - 72, N = 256 tn - 48; v26 sets both edge flags when either is ragged), K ragged (K - 112, EDGE_K),
and all three; each with and without a bias. Per shape: v26 spread and data-parallel as deployed (v9_ragged
below one wave, v26's persistent loop with a full wave). rocprof as the sweeps (1000 launches over rotating
inputs, median, B and C rows padded as bench.py pads them), one round.
"""

import json
import math
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
A16 = os.path.join(HERE, "..", "..", "kernels", "gemm", "intra_wave", "a16w16")
sys.path.insert(0, os.path.join(A16, "v20_streamk_reduce_scatter"))
sys.path.insert(0, A16)
RESULTS = os.path.join(HERE, "results")
PATH = os.path.join(RESULTS, "ragged_spread.jsonl")
TAG = "CHECK_CASE "

S_R = (136, 160, 192, 224, 248)
K_R = (4096, 8192, 16384, 32768)
VARIANTS = {"aligned": (0, 0, 0), "MN ragged": (72, 48, 0), "K ragged": (0, 0, 112), "MNK ragged": (72, 48, 112)}
C_V3 = 10493
ENV_KEYS = ("STREAMK_NUM_PROGRAMS", "STREAMK_POLICY", "STREAMK_FIXUP", "STREAMK_BALANCE_XCDS", "GEMM_TILE_LIMIT",
            "STREAMK_PARTIALS", "STREAMK_RS_READERS", "STREAMK_EXCHANGE", "STREAMK_PLACEMENT", "STREAMK_XCD_LOCAL",
            "STREAMK_FORCE_EDGE", "V9_FORCE_EDGE")


def grid(T):
    """(tiles in M, tiles in N) with tm * tn = T, as square as possible."""
    tm = max(d for d in range(1, int(math.isqrt(T)) + 1) if T % d == 0)
    return tm, T // tm


def cases():
    for K in K_R:
        for S in S_R:
            for full in (False, True):
                tm, tn = grid(256 + S if full else S)
                for var, (dm, dn, dk) in VARIANTS.items():
                    M, N, KK = 256 * tm - dm, 256 * tn - dn, K - dk
                    for bias in (False, True):
                        base = dict(S=S, full=full, variant=var, bias=bias, M=M, N=N, K=KK, k=K // 64)
                        dp = dict(config="dp", v=26, env={"STREAMK_POLICY": "dp"}) if full else \
                            dict(config="dp", v="9r", env={})
                        yield base | dp
                        yield base | dict(config="spread", v=26, env={"STREAMK_POLICY": "spread"})


def key(c):
    return (c["M"], c["N"], c["K"], c["bias"], c["config"])


def case_env(c):
    env = {**os.environ, **c["env"]}
    for k in ENV_KEYS:
        if k not in c["env"]:
            env.pop(k, None)
    return env


def load():
    if not os.path.exists(PATH):
        return []
    with open(PATH) as f:
        return [json.loads(l) for l in f]


def todo():
    done = {key(r) for r in load()}
    return [c for c in cases() if key(c) not in done]


def run():
    import rocprof_check

    t = todo()
    print(f"{len(t)} cases to run", flush=True)
    t = t[:int(os.environ.get("SWEEP_MAX_CASES", len(t)))]
    for i, c in enumerate(t):
        us = rocprof_check.cold_us(c["v"], c["M"], c["N"], c["K"], case_env(c), ("--bias",) if c["bias"] else ())
        row = {k: c[k] for k in ("S", "full", "variant", "bias", "M", "N", "K", "k", "config")} | {"us": us}
        with open(PATH, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{i + 1}/{len(t)}] {row}", flush=True)


def check_worker(K):
    os.chdir(A16)
    import importlib

    import bench
    import torch

    mods = {}
    for c in [c for c in cases() if c["K"] == K]:
        for k in ENV_KEYS:
            os.environ.pop(k, None)
        os.environ.update(c["env"])
        name = {26: "v26_streamk_ragged_k", "9r": "v9_ragged"}[c["v"]]
        mk = mods.setdefault(name, importlib.import_module(f"{name}.matmul_kernel"))
        res = {"wrong": 0, "error": None}
        try:
            torch.manual_seed(0)
            a = torch.rand((c["M"], c["K"]), device=bench.DEVICE, dtype=torch.bfloat16) - 0.5
            b = bench.make_b(c["N"], c["K"], torch.bfloat16, True, lambda t: t.uniform_(-0.5, 0.5))
            out = bench.make_c(c["M"], c["N"], torch.bfloat16, True)
            bias = (torch.rand((c["N"],), device=bench.DEVICE, dtype=torch.bfloat16) - 0.5) if c["bias"] else None
            ref = (a.float() @ b.float() + (bias.float() if c["bias"] else 0)).to(torch.bfloat16)
            for _ in range(2):
                out.fill_(float("nan"))
                mk.matmul(a, b, out, bias=bias)
                res["wrong"] += int(not torch.allclose(out, ref, atol=1e-1, rtol=1e-2))
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            res["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
        print(TAG + json.dumps({k: c[k] for k in ("S", "full", "variant", "bias", "M", "N", "K", "config")} | res),
              flush=True)


def check(jobs=3):
    def launch(K):
        p = subprocess.run([sys.executable, "-W", "ignore", os.path.abspath(__file__), "check_worker", str(K)],
                           capture_output=True, text=True)
        return K, p.stdout, p.returncode

    rows = []
    Ks = sorted({c["K"] for c in cases()})
    with ThreadPoolExecutor(jobs) as ex:
        for K, out, rc in ex.map(launch, Ks):
            got = [json.loads(l[len(TAG):]) for l in out.splitlines() if l.startswith(TAG)]
            rows += got
            print(f"K = {K}: {len(got)} cases, exit {rc}", file=sys.stderr, flush=True)
    total = sum(1 for _ in cases())
    bad = [r for r in rows if r["wrong"] or r["error"]]
    print(f"# Ragged spread sweep: correctness\n\n{len(rows)} of {total} cases ran, 2 launches each against an fp32 "
          "torch reference (plus bias), C NaN-filled before each launch, atol 0.1, rtol 0.01.\n")
    for r in bad[:50]:
        print(f"- FAIL {r}")
    print("\nPASS" if not bad and len(rows) == total else "\nFAIL")


def report():
    rows = load()
    t = {(r["S"], r["full"], r["variant"], r["bias"], r["k"], r["config"]): r["us"] for r in rows}
    cells = sorted({(S, full, var, bias, k) for S, full, var, bias, k, _ in t})

    def ratio(c):
        a, b = t.get((*c, "spread")), t.get((*c, "dp"))
        return a / b if a and b else None

    def rule(S, k, spread_ok):
        return "spread" if spread_ok and (256 - S) * k >= C_V3 else "dp"

    out = ["# Spread or data-parallel on ragged shapes (S > 128, v26)\n", __doc__.split("\n\n", 2)[2].strip() + "\n",
           "Cells: spread / data-parallel (below 1: spread faster). Bold: v26's rule ((256 - S) k >= 10493) picks spread.\n"]
    for full in (False, True):
        for bias in (False, True):
            out.append(f"\n## {'One full wave, then S' if full else 'Less than one wave'}, "
                       f"{'with' if bias else 'without'} bias\n")
            ks = sorted({c[4] for c in cells})
            out.append("| variant | S | " + " | ".join(f"K = {64 * k}" for k in ks) + " |")
            out.append("|---|---|" + "---|" * len(ks))
            for var in VARIANTS:
                for S in S_R:
                    vals = []
                    for k in ks:
                        r = ratio((S, full, var, bias, k))
                        txt = "-" if r is None else f"{r:.3f}"
                        vals.append(f"**{txt}**" if r is not None and rule(S, k, True) == "spread" else txt)
                    out.append(f"| {var} | {S} | " + " | ".join(vals) + " |")
    out.append("\n## Rules on these cells (regret against the faster of spread and data-parallel)\n")
    out.append("`v3 rule`: spread if (256 - S) k >= 10493 for every shape. `no spread when ragged`: the same, but "
               "data-parallel whenever the variant has an edge flag set (M, N or K ragged). `no spread when M / N ragged`: "
               "data-parallel when EDGE_M / EDGE_N are on (the spilling edge build), K ragged allowed. `old (k >= 512)`: "
               "the rule before v26 entry 5.\n")
    out.append("| variant | bias | cells | rule | median | 90th percentile | worst | cells more than 2% off |")
    out.append("|---|---|---|---|---|---|---|---|")
    rules = {"v3 rule": lambda S, k, var: rule(S, k, True),
             "no spread when ragged": lambda S, k, var: rule(S, k, var == "aligned"),
             "no spread when M / N ragged": lambda S, k, var: rule(S, k, var in ("aligned", "K ragged")),
             "old (k >= 512)": lambda S, k, var: "spread" if k >= 512 else "dp"}
    for var in VARIANTS:
        for bias in (False, True):
            cs = [c for c in cells if c[2] == var and c[3] == bias and ratio(c) is not None]
            for name, f in rules.items():
                if name.startswith("no spread") and var == "aligned":
                    continue
                regs = sorted(t[(*c, f(c[0], c[4], var))] / min(t[(*c, "spread")], t[(*c, "dp")]) for c in cs)
                if not regs:
                    continue
                q = regs[min(len(regs) - 1, int(0.9 * len(regs)))]
                out.append(f"| {var} | {bias} | {len(regs)} | {name} | {statistics.median(regs):.3f} | {q:.3f} | "
                           f"{regs[-1]:.3f} | {sum(x > 1.02 for x in regs)} |")
    text = "\n".join(out) + "\n"
    with open(os.path.join(RESULTS, "ragged_spread.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "cases":
        cs = list(cases())
        print(f"{len(cs)} cases, {len(todo())} to run")
    elif stage == "todo":
        print(len(todo()))
    elif stage == "check":
        check()
    elif stage == "check_worker":
        check_worker(int(sys.argv[2]))
    elif stage == "run":
        run()
    else:
        report()
