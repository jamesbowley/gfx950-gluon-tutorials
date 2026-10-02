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
"""Split-count and small-K sweeps over the stream-K versions, timed with rocprof.

    HIP_VISIBLE_DEVICES=2 python experiments/streamk_split_count/sweep.py a   # split count, K = 8192
    HIP_VISIBLE_DEVICES=0 python experiments/streamk_split_count/sweep.py b   # K = 512-8192
    python experiments/streamk_split_count/sweep.py report                    # tables from the raw rows

Every case is one `rocprofv3 --kernel-trace` run of `bench.py --rocprof` (1000 launches over a
rotating 512 MB of inputs, median GPU kernel time), so host launch gaps do not count. The split is
set with STREAMK_NUM_PROGRAMS = S * n (n programs per leftover tile; "default" leaves it unset,
which is all 256 programs). Raw rows go to results/sweep_<a|b>.jsonl, one per case, and a run
skips the cases already there.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
A16 = os.path.join(HERE, "..", "..", "kernels", "gemm", "intra_wave", "a16w16")
sys.path.insert(0, os.path.join(A16, "v20_streamk_reduce_scatter"))
sys.path.insert(0, A16)
import rocprof_check  # noqa: E402

# Leftover tiles S -> (M, N) with at least one full wave of 256x256 tiles.
SHAPES = {4: (3328, 5120), 16: (4352, 4096), 38: (3584, 5376), 64: (4096, 5120), 128: (4096, 6144),
          192: (4096, 7168)}
ONE_TILE = {
    "v15": (15, {}),
    "v16": (16, {}),
    "v17": (17, {}),
    "v18": (18, {}),
    "v20 owner": (20, {"STREAMK_POLICY": "one_tile", "STREAMK_FIXUP": "owner"}),
    "v20 rs": (20, {"STREAMK_POLICY": "one_tile", "STREAMK_FIXUP": "rs"}),
    "v20 spread": (20, {"STREAMK_POLICY": "spread"}),
}
TWO_TILE = {"v19": (19, {}), "v20 two_tile": (20, {"STREAMK_POLICY": "two_tile"})}
BASE = {"v13": (13, {}), "v20 dp": (20, {"STREAMK_POLICY": "dp"})}
N_A = (1, 2, 3, 4, 6, 8, 16, 32, 64)
N_B = (1, 2, 3, 4, 6, 8, 16)
K_B = (512, 1024, 2048, 4096, 8192)


def splits(S, K, ns):
    """Programs per tile n that the tile-aligned split can use: S * n <= 256, n <= pairs per tile."""
    return [n for n in ns if S * n <= 256 and n <= K // 128]


def cases_a():
    K = 8192
    for S, (M, N) in SHAPES.items():
        if S == 192:
            continue
        for label, (v, env) in BASE.items():
            yield dict(S=S, M=M, N=N, K=K, config=label, n="-", snp=None, v=v, env=env)
        for n in splits(S, K, N_A) + ["default"]:
            snp = None if n == "default" else S * n
            for label, (v, env) in ONE_TILE.items():
                yield dict(S=S, M=M, N=N, K=K, config=label, n=n, snp=snp, v=v, env=env)
        if S in (16, 38):
            # Two-tile: every tile has at most one peer whatever the program count.
            for snp in (None, 224, 192, 160, 128):
                for label, (v, env) in TWO_TILE.items():
                    yield dict(S=S, M=M, N=N, K=K, config=label, n="-", snp=snp, v=v, env=env)
    # XCD balance: the idle programs on the last XCDs instead of spread over all eight.
    M, N = SHAPES[38]
    for label in ("v20 owner", "v20 rs"):
        v, env = ONE_TILE[label]
        yield dict(S=38, M=M, N=N, K=K, config=label + " unbalanced", n=2, snp=76, v=v,
                   env={**env, "STREAMK_BALANCE_XCDS": "0"})


def cases_b():
    for K in K_B:
        for S, (M, N) in SHAPES.items():
            for label, (v, env) in BASE.items():
                yield dict(S=S, M=M, N=N, K=K, config=label, n="-", snp=None, v=v, env=env)
            v, env = TWO_TILE["v20 two_tile"]
            yield dict(S=S, M=M, N=N, K=K, config="v20 two_tile", n="-", snp=None, v=v, env=env)
            for n in splits(S, K, N_B) + ["default"]:
                snp = None if n == "default" else S * n
                for label in ("v20 owner", "v20 rs", "v20 spread"):
                    v, env = ONE_TILE[label]
                    yield dict(S=S, M=M, N=N, K=K, config=label, n=n, snp=snp, v=v, env=env)


def key(c):
    return (c["S"], c["K"], c["config"], str(c["n"]), c["snp"])


def run(which):
    path = os.path.join(HERE, "results", f"sweep_{which}.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            done = {key(json.loads(l)) for l in f}
    cases = list(cases_a() if which == "a" else cases_b())
    todo = [c for c in cases if key(c) not in done]
    print(f"sweep {which}: {len(cases)} cases, {len(todo)} to run", flush=True)
    for i, c in enumerate(todo):
        env = {**os.environ, **c["env"]}
        for k in ("STREAMK_NUM_PROGRAMS", "STREAMK_POLICY", "STREAMK_FIXUP", "STREAMK_BALANCE_XCDS"):
            if k not in c["env"]:
                env.pop(k, None)
        if c["snp"] is not None:
            env["STREAMK_NUM_PROGRAMS"] = str(c["snp"])
        us = rocprof_check.cold_us(c["v"], c["M"], c["N"], c["K"], env)
        row = {k: c[k] for k in ("S", "M", "N", "K", "config", "n", "snp")} | {"us": us}
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{i + 1}/{len(todo)}] {row}", flush=True)


def load(which):
    path = os.path.join(HERE, "results", f"sweep_{which}.jsonl")
    with open(path) as f:
        return [json.loads(l) for l in f]


def report():
    out = []
    rows = load("a")
    out.append("# Sweep A: programs per leftover tile n, K = 8192 (rocprof kernel time, us)\n")
    out.append("n is the programs per leftover tile (STREAMK_NUM_PROGRAMS = S * n); \"default\" is all 256 "
               "programs. Bold is the fastest stream-K entry in each column.\n")
    for S in sorted({r["S"] for r in rows}):
        rs = [r for r in rows if r["S"] == S and r["config"] in ONE_TILE]
        base = {r["config"]: r["us"] for r in rows if r["S"] == S and r["config"] in BASE}
        configs = [c for c in ONE_TILE if any(r["config"] == c for r in rs)]
        ns = []
        for r in rs:
            if r["n"] not in ns:
                ns.append(r["n"])
        best = {c: min(r["us"] for r in rs if r["config"] == c) for c in configs}
        M, N = SHAPES[S]
        out.append(f"\n## S = {S} ({M}x{N}x8192): v13 {base.get('v13', float('nan')):.1f}, "
                   f"v20 dp {base.get('v20 dp', float('nan')):.1f}\n")
        out.append("| n | " + " | ".join(configs) + " |")
        out.append("|---|" + "---|" * len(configs))
        for n in ns:
            cells = []
            for c in configs:
                us = next((r["us"] for r in rs if r["config"] == c and r["n"] == n), None)
                cells.append("" if us is None else (f"**{us:.1f}**" if us == best[c] else f"{us:.1f}"))
            out.append(f"| {n} | " + " | ".join(cells) + " |")
        tt = [r for r in rows if r["S"] == S and r["config"] in TWO_TILE]
        if tt:
            out.append("\nTwo-tile by STREAMK_NUM_PROGRAMS:\n")
            out.append("| STREAMK_NUM_PROGRAMS | " + " | ".join(TWO_TILE) + " |")
            out.append("|---|" + "---|" * len(TWO_TILE))
            for snp in [None, 224, 192, 160, 128]:
                cells = [next((f"{r['us']:.1f}" for r in tt if r["config"] == c and r["snp"] == snp), "")
                         for c in TWO_TILE]
                out.append(f"| {snp or 256} | " + " | ".join(cells) + " |")
        ub = [r for r in rows if r["S"] == S and r["config"].endswith("unbalanced")]
        if ub:
            out.append("\nXCD balance at n = 2 (idle programs on the last XCDs instead of spread):\n")
            for r in ub:
                bal = next(x["us"] for x in rs if x["config"] == r["config"].removesuffix(" unbalanced")
                           and x["n"] == 2)
                out.append(f"- {r['config'].removesuffix(' unbalanced')}: balanced {bal:.1f}, unbalanced "
                           f"{r['us']:.1f}")
    if os.path.exists(os.path.join(HERE, "results", "sweep_b.jsonl")):
        rows = load("b")
        out.append("\n\n# Sweep B: small K (rocprof kernel time, us)\n")
        out.append("For each S and K: data-parallel (the faster of v13 and v20 dp), v20 two-tile, and each "
                   "one-tile ending at its best n (n in brackets, d = default) and at the default split.\n")
        out.append("| S | K | data-parallel | two_tile | owner best (n) | rs best (n) | spread best (n) | "
                   "owner default | rs default | spread default | best stream-K vs data-parallel |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for S in sorted({r["S"] for r in rows}):
            for K in K_B:
                rr = [r for r in rows if r["S"] == S and r["K"] == K]
                if not rr:
                    continue
                dp = min(r["us"] for r in rr if r["config"] in BASE)
                tt = next((r["us"] for r in rr if r["config"] == "v20 two_tile"), float("nan"))
                cells, best_sk = [], tt
                for c in ("v20 owner", "v20 rs", "v20 spread"):
                    cr = [r for r in rr if r["config"] == c]
                    b = min(cr, key=lambda r: r["us"])
                    best_sk = min(best_sk, b["us"])
                    cells.append(f"{b['us']:.1f} ({'d' if b['n'] == 'default' else b['n']})")
                defaults = [next((f"{r['us']:.1f}" for r in rr if r["config"] == c and r["n"] == "default"), "")
                            for c in ("v20 owner", "v20 rs", "v20 spread")]
                out.append(f"| {S} | {K} | {dp:.1f} | {tt:.1f} | " + " | ".join(cells) + " | " +
                           " | ".join(defaults) + f" | {100 * (dp / best_sk - 1):+.1f}% |")
    text = "\n".join(out) + "\n"
    with open(os.path.join(HERE, "results", "sweeps.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    {"a": lambda: run("a"), "b": lambda: run("b"), "report": report}[sys.argv[1]]()
