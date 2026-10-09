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
    HIP_VISIBLE_DEVICES=0 python experiments/streamk_split_count/sweep.py c   # S = 1-255 on one matrix
    HIP_VISIBLE_DEVICES=0 python experiments/streamk_split_count/sweep.py l2  # its L2 counters, S = 8, 38
    python experiments/streamk_split_count/sweep.py report                    # tables from the raw rows
    HIP_VISIBLE_DEVICES=1 python experiments/streamk_split_count/sweep.py e1  # v23 against v20 and v22, 2 rounds
    python experiments/streamk_split_count/sweep.py e1_report                 # results/e1_tails.md
    HIP_VISIBLE_DEVICES=0 python experiments/streamk_split_count/sweep.py f   # v24: every ending and placement
    python experiments/streamk_split_count/sweep.py f_report                  # results/sweep_f.md

Every case is one `rocprofv3 --kernel-trace` run of `bench.py --rocprof` (1000 launches over a
rotating 512 MB of inputs, median GPU kernel time), so host launch gaps do not count. The split is
set with STREAMK_NUM_PROGRAMS = S * n (n programs per leftover tile; "default" leaves it unset,
which is all 256 programs). Raw rows go to results/sweep_<a|b|c>.jsonl, one per case, and a run
skips the cases already there.

Sweep C holds the matrix fixed: every S runs on 4096x8192xK (16 x 32 tiles) with
GEMM_TILE_LIMIT = 256 + S, so the kernels compute only the first 256 + S tile ids, as a GEMM of
that many tiles (check_tile_limit.py checks every case). Every S then has one full wave, the same
full-wave tiles and the same A and B; the baseline is T = 256 on the same matrix, so the tail is
kernel(256 + S) - kernel(256). Real shapes at S = 8, 16, 38, 128 and 192 validate the limit.
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

# Sweep C: one matrix, the first 256 + S tile ids (GEMM_TILE_LIMIT).
MATRIX_C = (4096, 8192)
S_C = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 38, 48, 64, 96, 128)
K_C = (2048, 4096, 8192, 16384)
S_C_HIGH = (160, 192, 224, 240, 255)
K_C_HIGH = (8192, 16384, 32768)
N_C = (1, 2, 3, 4, 6, 8, 12, 16, 32, 64)
# The one-tile splits with n programs per tile: end-to-end cuts (v16) and tile-aligned cuts with
# the owner or the reduce-scatter ending (v20; at the default, 256 // S programs per tile).
SPLIT_C = {
    "v16": (16, {}),
    "v20 owner": (20, {"STREAMK_POLICY": "one_tile", "STREAMK_FIXUP": "owner"}),
    "v20 rs": (20, {"STREAMK_POLICY": "one_tile", "STREAMK_FIXUP": "rs"}),
    # Every program pk_adds its partial into C, zeroed by a launch of its own (timed with it).
    "v22 atomic": (22, {"STREAMK_POLICY": "one_tile", "STREAMK_FIXUP": "atomic"}),
    # The same, with the zeroing inside the GEMM behind a per-tile counter: one launch.
    "v22 atomic_iz": (22, {"STREAMK_POLICY": "one_tile", "STREAMK_FIXUP": "atomic_iz"}),
}
# Small K: below sweep C's main grid, where sweep B found no single-rounding split pays.
K_SMALL = (512, 1024)
SMALL_K_C = ("v20 owner", "v20 rs", "v22 atomic", "v22 atomic_iz")
# One XCD: the active programs are the first S n stream-K ids instead of spread evenly over the XCDs
# (STREAMK_BALANCE_XCDS=0), so a tile's programs share an XCD even when S < 8, and the tail runs on
# ceil(S n / 32) XCDs.
S_PACKED = (1, 2, 3, 4, 6, 8)
PACKED_C = ("v20 rs", "v22 atomic", "v22 atomic_iz")
# End-to-end reversed cuts with the split capped (S n programs); its default is in WHOLE_C (n "-").
SPREAD_CAPPED = {"v20 spread": (20, {"STREAMK_POLICY": "spread"})}
# One run per S and K: data-parallel, forward and reversed two-tile, end-to-end reversed cuts.
WHOLE_C = {
    "v13": (13, {}),
    "v16 two_tile": (16, {"STREAMK_POLICY": "two_tile"}),
    "v19": (19, {}),
    "v20 spread": (20, {"STREAMK_POLICY": "spread"}),
}
# Real shapes (no limit) to compare with the limited matrix, at K = 8192.
REAL_C = {8: (3072, 5632), 16: (4352, 4096), 38: (3584, 5376), 128: (4096, 6144), 192: (4096, 7168)}

# Sweep E1 (v23 WORKLOG): on sweep C's matrix, the cells where the atomic ending won in the GEMM grid
# but lost in the synthetic exchange: (S, K, n), the grid's best reduce-scatter n and its neighbours.
E1_CELLS = ((1, 16384, (32, 64)), (8, 8192, (12, 16, 32)), (16, 8192, (6, 8, 12)), (32, 4096, (3, 4, 6)),
            (64, 8192, (3, 4)), (12, 8192, (8, 12, 16)), (24, 8192, (4, 6, 8)))
_RS = {"STREAMK_POLICY": "one_tile", "STREAMK_FIXUP": "rs"}
_OWNER = {"STREAMK_POLICY": "one_tile", "STREAMK_FIXUP": "owner"}
E1_CONFIGS = {
    "v20 rs": (20, _RS),
    "v20 owner": (20, _OWNER),
    # Every program reads, as v20; only run where n is not a power of two (otherwise it is "v23 rs").
    "v23 rs all": (23, {**_RS, "STREAMK_RS_READERS": "all"}),
    "v23 rs": (23, {**_RS, "STREAMK_RS_READERS": "pow2"}),
    "v23 rs bf16": (23, {**_RS, "STREAMK_RS_READERS": "pow2", "STREAMK_PARTIALS": "bf16"}),
    "v23 rs bf16_pairs": (23, {**_RS, "STREAMK_RS_READERS": "pow2", "STREAMK_PARTIALS": "bf16_pairs"}),
    "v23 owner bf16_pairs": (23, {**_OWNER, "STREAMK_PARTIALS": "bf16_pairs"}),
    "v22 atomic": (22, {"STREAMK_POLICY": "one_tile", "STREAMK_FIXUP": "atomic"}),
    "v22 atomic_iz": (22, {"STREAMK_POLICY": "one_tile", "STREAMK_FIXUP": "atomic_iz"}),
}


# Sweep F (v24 WORKLOG): sweep C's matrix and grid with every ending, placement and partial type in
# one kernel (v24), against v13 and v20's reduce-scatter (the link to sweep C).
K_F = (512, 1024, 2048, 4096, 8192, 16384)
_V24_RS = {"STREAMK_POLICY": "one_tile", "STREAMK_FIXUP": "rs"}
_V24_OWNER = {"STREAMK_POLICY": "one_tile", "STREAMK_FIXUP": "owner"}
_V24_ATOMIC = {"STREAMK_POLICY": "one_tile", "STREAMK_FIXUP": "atomic_iz"}
SPLIT_F = {
    "v20 rs": (20, _RS),
    "v24 owner": (24, _V24_OWNER),
    "v24 owner chunk": (24, {**_V24_OWNER, "STREAMK_PLACEMENT": "chunk"}),
    "v24 owner bf16_pairs": (24, {**_V24_OWNER, "STREAMK_PARTIALS": "bf16_pairs"}),
    "v24 rs": (24, _V24_RS),
    "v24 rs chunk": (24, {**_V24_RS, "STREAMK_PLACEMENT": "chunk"}),
    "v24 rs l2": (24, {**_V24_RS, "STREAMK_EXCHANGE": "l2"}),
    "v24 rs bf16_pairs": (24, {**_V24_RS, "STREAMK_PARTIALS": "bf16_pairs"}),
    "v24 atomic_iz": (24, _V24_ATOMIC),
    "v24 atomic_iz chunk": (24, {**_V24_ATOMIC, "STREAMK_PLACEMENT": "chunk"}),
}
WHOLE_F = {"v13": (13, {}), "v24 two_tile": (24, {"STREAMK_POLICY": "two_tile"})}
# Rounds C more than once (bf16 partials: about 1.6 times; atomics: 0.5 + 0.75 sqrt(n) times).
MULTI_ROUNDING_F = ("v24 owner bf16_pairs", "v24 rs bf16_pairs", "v24 atomic_iz", "v24 atomic_iz chunk")


def l2_split_ok(S, K, n):
    """Whether the l2 exchange runs n programs per tile: its XCD-local split fits ceil(S / 8) tiles
    in an XCD's 32 programs, and caps n to do it (a capped n would repeat a smaller n's case)."""
    if n == "default":
        return True
    return min(n, K // 128, 32 // -(-S // 8)) == n


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


def cases_c(include_real=True):
    """Sweep C's cases. limit is GEMM_TILE_LIMIT (None: the real shape, every tile)."""
    M, N = MATRIX_C

    def case(S, K, config, n, snp, v, env, m=M, nn=N, limit=None):
        return dict(S=S, M=m, N=nn, K=K, config=config, n=n, snp=snp, v=v, env=env, limit=limit)

    for K in sorted(set(K_C) | set(K_C_HIGH) | set(K_SMALL)):
        # Baselines: the full wave alone, and two full waves.
        for T in (256, 512):
            yield case(T - 256, K, "v13 base", "-", None, 13, {}, limit=T)
    for K in K_SMALL:
        for S in S_C:
            yield case(S, K, "v13", "-", None, *WHOLE_C["v13"], limit=256 + S)
            for n in splits(S, K, N_C) + ["default"]:
                snp = None if n == "default" else S * n
                for label in SMALL_K_C:
                    v, env = SPLIT_C[label]
                    yield case(S, K, label, n, snp, v, env, limit=256 + S)
    for K in K_C:
        for S in S_PACKED:
            for n in [x for x in splits(S, K, N_C) if x >= 2]:
                for label in PACKED_C:
                    v, env = SPLIT_C[label]
                    yield case(S, K, label + " 1xcd", n, S * n, v, {**env, "STREAMK_BALANCE_XCDS": "0"},
                               limit=256 + S)
    for K in K_C:
        for S in S_C:
            for label, (v, env) in WHOLE_C.items():
                yield case(S, K, label, "-", None, v, env, limit=256 + S)
            for n in splits(S, K, N_C) + ["default"]:
                snp = None if n == "default" else S * n
                for label, (v, env) in SPLIT_C.items():
                    yield case(S, K, label, n, snp, v, env, limit=256 + S)
            for n in splits(S, K, N_C):
                for label, (v, env) in SPREAD_CAPPED.items():
                    yield case(S, K, label, n, S * n, v, env, limit=256 + S)
    for K in K_C_HIGH:
        for S in S_C_HIGH:
            for label, (v, env) in [*WHOLE_C.items(), ("v16", SPLIT_C["v16"])]:
                yield case(S, K, label, "-" if label != "v16" else "default", None, v, env, limit=256 + S)
    if include_real:
        for S, (m, nn) in REAL_C.items():
            labels = [*WHOLE_C, "v16"] + (["v20 owner", "v20 rs"] if S <= 128 else [])
            for label in labels:
                v, env = WHOLE_C[label] if label in WHOLE_C else SPLIT_C[label]
                n = "-" if label in WHOLE_C else "default"
                yield case(S, 8192, label, n, None, v, env, m=m, nn=nn)


def cases_f():
    """Sweep F's cases: per K the full wave alone (v13, T = 256), then for every S the whole-tile
    configurations once and the split ones at every n and the default; S > 128 only v13, two-tile and
    v24's spread (end-to-end reversed, the only split that pays there)."""
    M, N = MATRIX_C

    def case(S, K, config, n, snp, v, env):
        return dict(S=S, M=M, N=N, K=K, config=config, n=n, snp=snp, v=v, env=env, limit=256 + S)

    for K in sorted(set(K_F) | set(K_C_HIGH)):
        yield dict(S=0, M=M, N=N, K=K, config="v13 base", n="-", snp=None, v=13, env={}, limit=256)
    for K in K_F:
        for S in S_C:
            for label, (v, env) in WHOLE_F.items():
                yield case(S, K, label, "-", None, v, env)
            for n in splits(S, K, N_C) + ["default"]:
                snp = None if n == "default" else S * n
                for label, (v, env) in SPLIT_F.items():
                    if label == "v24 rs l2" and not l2_split_ok(S, K, n):
                        continue
                    yield case(S, K, label, n, snp, v, env)
    for K in K_C_HIGH:
        for S in S_C_HIGH:
            for label, (v, env) in [*WHOLE_F.items(), ("v24 spread", (24, {"STREAMK_POLICY": "spread"}))]:
                yield case(S, K, label, "-", None, v, env)


def cases_e1():
    """Sweep E1's cases for one round: the full wave alone per K, then every (S, K, n) with the
    configurations interleaved."""
    M, N = MATRIX_C
    for K in sorted({k for _, k, _ in E1_CELLS}):
        yield dict(S=0, M=M, N=N, K=K, config="v13 base", n="-", snp=None, v=13, env={}, limit=256)
    for S, K, ns in E1_CELLS:
        for n in ns:
            for label, (v, env) in E1_CONFIGS.items():
                if label == "v23 rs all" and n & (n - 1) == 0:
                    continue
                yield dict(S=S, M=M, N=N, K=K, config=label, n=n, snp=S * n, v=v, env=env, limit=256 + S)


def cases_l2():
    """L2 counters for every variant on sweep C's matrix: S = 8 and 38, K = 8192."""
    M, N = MATRIX_C
    K = 8192
    yield dict(S=0, M=M, N=N, K=K, config="v13 base", n="-", snp=None, v=13, env={}, limit=256)
    for S in (8, 38):
        for label, (v, env) in WHOLE_C.items():
            yield dict(S=S, M=M, N=N, K=K, config=label, n="-", snp=None, v=v, env=env, limit=256 + S)
        for n in [x for x in (2, 4, 8) if S * x <= 256] + ["default"]:
            for label, (v, env) in SPLIT_C.items():
                yield dict(S=S, M=M, N=N, K=K, config=label, n=n, snp=None if n == "default" else S * n, v=v,
                           env=env, limit=256 + S)


def run_l2():
    """TCC hit rate and memory read requests (rocprof_check.l2: 10 launches on the same inputs, warm)."""
    path = os.path.join(HERE, "results", "l2_c.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            done = {key(json.loads(l)) for l in f}
    todo = [c for c in cases_l2() if key(c) not in done]
    for i, c in enumerate(todo):
        hit, reads = rocprof_check.l2(c["v"], c["M"], c["N"], c["K"], case_env(c))
        row = {k: c[k] for k in ("S", "M", "N", "K", "config", "n", "snp", "limit")} | {"hit": hit, "reads": reads}
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{i + 1}/{len(todo)}] {row}", flush=True)


def key(c):
    return (c["S"], c["K"], c["config"], str(c["n"]), c["snp"], c["M"], c["N"], c.get("limit"))


# The variables a case sets; any the case does not set are removed from its environment.
ENV_KEYS = ("STREAMK_NUM_PROGRAMS", "STREAMK_POLICY", "STREAMK_FIXUP", "STREAMK_BALANCE_XCDS", "GEMM_TILE_LIMIT",
            "STREAMK_PARTIALS", "STREAMK_RS_READERS", "STREAMK_EXCHANGE", "STREAMK_PLACEMENT", "STREAMK_XCD_LOCAL")


def case_env(c):
    """The process environment for case c: its own STREAMK_* and GEMM_TILE_LIMIT, nothing inherited."""
    env = {**os.environ, **c["env"]}
    for k in ENV_KEYS:
        if k not in c["env"]:
            env.pop(k, None)
    if c["snp"] is not None:
        env["STREAMK_NUM_PROGRAMS"] = str(c["snp"])
    if c.get("limit"):
        env["GEMM_TILE_LIMIT"] = str(c["limit"])
    return env


def run(which):
    path = os.path.join(HERE, "results", f"sweep_{which}.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            done = {key(json.loads(l)) for l in f}
    cases = list({"a": cases_a, "b": cases_b, "c": cases_c, "f": cases_f}[which]())
    todo = [c for c in cases if key(c) not in done]
    print(f"sweep {which}: {len(cases)} cases, {len(todo)} to run", flush=True)
    # SWEEP_MAX_CASES: stop after that many (run_f.sh checks the GPU is still idle between chunks).
    todo = todo[:int(os.environ.get("SWEEP_MAX_CASES", len(todo)))]
    for i, c in enumerate(todo):
        det = rocprof_check.cold_detail(c["v"], c["M"], c["N"], c["K"], case_env(c))
        row = {k: c[k] for k in ("S", "M", "N", "K", "config", "n", "snp")} | {"us": det["us"]}
        if det["pre_us"] == det["pre_us"]:  # not NaN: a pre-kernel (v22's zeroing) ran
            row |= {"gemm_us": det["gemm_us"], "pre_us": det["pre_us"]}
        if c.get("limit"):
            row["limit"] = c["limit"]
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{i + 1}/{len(todo)}] {row}", flush=True)


def run_e1(rounds=2):
    """Sweep E1, `rounds` passes over every case on one GPU, alternate passes in reverse order, so each
    configuration is timed between the others twice. Rows carry their round."""
    path = os.path.join(HERE, "results", "sweep_e1.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            done = {(*key(r), r["round"]) for r in map(json.loads, f)}
    cases = list(cases_e1())
    todo = [(r, c) for r in range(1, rounds + 1) for c in (cases if r % 2 else cases[::-1])
            if (*key(c), r) not in done]
    print(f"sweep e1: {len(cases)} cases x {rounds} rounds, {len(todo)} to run", flush=True)
    for i, (rnd, c) in enumerate(todo):
        det = rocprof_check.cold_detail(c["v"], c["M"], c["N"], c["K"], case_env(c))
        row = {k: c[k] for k in ("S", "M", "N", "K", "config", "n", "snp")} | {"us": det["us"], "round": rnd,
                                                                                "limit": c["limit"]}
        if det["pre_us"] == det["pre_us"]:
            row |= {"gemm_us": det["gemm_us"], "pre_us": det["pre_us"]}
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{i + 1}/{len(todo)}] {row}", flush=True)


def report_e1():
    """results/e1_tails.md: tails (kernel - full wave alone, mean of the rounds) per cell and configuration."""
    rows = load("e1")
    rounds = sorted({r["round"] for r in rows})

    def us(S, K, config, n, rnd):
        return next((r["us"] for r in rows if r["S"] == S and r["K"] == K and r["config"] == config
                     and str(r["n"]) == str(n) and r["round"] == rnd), None)

    configs = [c for c in E1_CONFIGS if any(r["config"] == c for r in rows)]
    out = ["# Sweep E1: v23 against v20 and the atomic ending (rocprof tails, us)\n",
           f"4096x8192xK with GEMM_TILE_LIMIT = 256 + S and STREAMK_NUM_PROGRAMS = S n; 1000 launches over rotating "
           f"inputs, median; tail = kernel - the same matrix with 256 tiles (each round against its own baseline). "
           f"Mean of {len(rounds)} rounds (the second in reverse order), the two rounds in brackets. `v23 rs` reads with "
           f"the largest power of two of a tile's n programs and stores C in bands; `v23 rs all` reads with all n "
           f"(v20's split, band store only for power-of-two n, so it is run for the other n). The atomic tail includes "
           f"its zeroing launch.\n"]
    best_rows = []
    for S, K, ns in E1_CELLS:
        base = {rnd: us(0, K, "v13 base", "-", rnd) for rnd in rounds}
        out.append(f"\n## S = {S}, K = {K} (full wave alone: "
                   + " / ".join(f"{base[r]:.1f}" for r in rounds if base[r] is not None) + " us)\n")
        out.append("| config | " + " | ".join(f"n = {n}" for n in ns) + " |")
        out.append("|---|" + "---|" * len(ns))
        cell_best = {}
        for cfg in configs:
            cells = []
            for n in ns:
                t = [us(S, K, cfg, n, rnd) - base[rnd] for rnd in rounds
                     if us(S, K, cfg, n, rnd) is not None and base[rnd] is not None]
                if not t:
                    cells.append("")
                    continue
                m = sum(t) / len(t)
                if cfg not in cell_best or m < cell_best[cfg][0]:
                    cell_best[cfg] = (m, n)
                cells.append(f"{m:.1f} (" + " / ".join(f"{x:.1f}" for x in t) + ")")
            out.append(f"| {cfg} | " + " | ".join(cells) + " |")
        best_rows.append((S, K, cell_best))
    out.append("\n## Each configuration at its best n\n")
    out.append("| S | K | " + " | ".join(configs) + " |")
    out.append("|---|---|" + "---|" * len(configs))
    for S, K, cb in best_rows:
        fastest = min(v[0] for v in cb.values()) if cb else None
        cells = []
        for cfg in configs:
            if cfg not in cb:
                cells.append("")
                continue
            m, n = cb[cfg]
            txt = f"{m:.1f} ({n})"
            cells.append(f"**{txt}**" if m == fastest else txt)
        out.append(f"| {S} | {K} | " + " | ".join(cells) + " |")
    text = "\n".join(out) + "\n"
    with open(os.path.join(HERE, "results", "e1_tails.md"), "w") as f:
        f.write(text)
    print(text)


def cases_f_noise(top=3):
    """Sweep F's second round: per K the full wave alone, then in every cell (S <= 128) the `top`
    fastest (configuration, n) of round 1 and the `top` fastest that round C once, so run-to-run
    spread is measured where either choice is made."""
    rows = load("f")
    base = {r["K"]: r["us"] for r in rows if r["config"] == "v13 base"}
    by_case = {key(c): c for c in cases_f()}
    for K in sorted(base):
        yield next(c for c in cases_f() if c["config"] == "v13 base" and c["K"] == K)
    cells = {}
    for r in rows:
        if r["config"] != "v13 base" and r["S"] <= 128:
            cells.setdefault((r["S"], r["K"]), []).append(r)
    for (S, K), rr in sorted(cells.items()):
        rr = sorted(rr, key=lambda r: r["us"])
        picked = rr[:top] + [r for r in rr if r["config"] not in MULTI_ROUNDING_F][:top]
        seen = set()
        for r in picked:
            k = (r["S"], r["K"], r["config"], str(r["n"]), r["snp"], r["M"], r["N"], r.get("limit"))
            if k in by_case and k not in seen:
                seen.add(k)
                yield by_case[k]


def todo(which):
    """Cases of sweep F ("f") or its second round ("f_noise") not yet in their results file."""
    name, gen = {"f": ("sweep_f.jsonl", cases_f), "f_noise": ("sweep_f_round2.jsonl", cases_f_noise)}[which]
    path = os.path.join(HERE, "results", name)
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            done = {key(json.loads(l)) for l in f}
    return sum(key(c) not in done for c in gen())


def run_f_noise():
    """results/sweep_f_round2.jsonl: cases_f_noise timed again (resumable, SWEEP_MAX_CASES per call)."""
    path = os.path.join(HERE, "results", "sweep_f_round2.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            done = {key(json.loads(l)) for l in f}
    cases = list(cases_f_noise())
    todo = [c for c in cases if key(c) not in done]
    print(f"sweep f round 2: {len(cases)} cases, {len(todo)} to run", flush=True)
    todo = todo[:int(os.environ.get("SWEEP_MAX_CASES", len(todo)))]
    for i, c in enumerate(todo):
        det = rocprof_check.cold_detail(c["v"], c["M"], c["N"], c["K"], case_env(c))
        row = {k: c[k] for k in ("S", "M", "N", "K", "config", "n", "snp")} | {"us": det["us"], "limit": c["limit"]}
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{i + 1}/{len(todo)}] {row}", flush=True)


def report_f():
    """results/sweep_f.md: sweep F's tails (kernel - v13's full wave alone at the same K), each
    configuration at its best n, the fastest per cell, and the comparisons sweep F was run for."""
    rows = load("f")
    base = {r["K"]: r["us"] for r in rows if r["config"] == "v13 base"}
    tails = {}
    for r in rows:
        if r["config"] != "v13 base" and r["K"] in base:
            tails.setdefault((r["S"], r["K"], r["config"]), []).append((r["us"] - base[r["K"]], r["n"]))

    def best(S, K, cfg):
        """(tail, n) at the configuration's best n, or None."""
        t = tails.get((S, K, cfg))
        return min(t, key=lambda x: x[0]) if t else None

    def fmt(b):
        if b is None:
            return "-"
        n = "" if b[1] == "-" else f" ({'d' if b[1] == 'default' else b[1]})"
        return f"{b[0]:.1f}{n}"

    def ratios(a, b, cells):
        """Tail of a / tail of b, each at its best n, over the cells where both ran (positive tails)."""
        out = []
        for S, K in cells:
            x, y = best(S, K, a), best(S, K, b)
            if x and y and x[0] > 0 and y[0] > 0:
                out.append(x[0] / y[0])
        return out

    def summary(rs):
        if not rs:
            return "-"
        rs = sorted(rs)
        q = lambda p: rs[min(len(rs) - 1, int(p * len(rs)))]  # noqa: E731
        return (f"{q(0.5):.3f} ({q(0.1):.3f}-{q(0.9):.3f}), shorter in {sum(x < 1 for x in rs)} of {len(rs)}")

    cells = sorted({(S, K) for S, K, _ in tails if S <= 128})
    splits_cfg = list(SPLIT_F)
    singles = ["v13", "v24 two_tile", *[c for c in SPLIT_F if c not in MULTI_ROUNDING_F]]
    out = ["# Sweep F: every ending, placement and partial type (v24) on one matrix\n",
           f"rocprof kernel times (1000 launches over rotating inputs, median) on {MATRIX_C[0]}x{MATRIX_C[1]}xK with "
           "GEMM_TILE_LIMIT = 256 + S. Tail = kernel time minus v13's with 256 tiles at the same K (one control for "
           "every configuration). Each split configuration at its best n (STREAMK_NUM_PROGRAMS = S n; d: the default, "
           "all 256 programs), n in brackets. `chunk`: v18's chunk-major placement with the rotating role 0; `l2`: v21's "
           "XCD-local split with the partials and flags in the XCD's L2; `bf16_pairs`: v23's bf16 partials (about 1.6 "
           "roundings); `atomic_iz`: v22's bf16 pk_add into C, zeroed in the GEMM (0.5 + 0.75 sqrt(n) roundings). "
           "Raw rows: results/sweep_f.jsonl.\n",
           "## Full wave alone (v13, T = 256)\n",
           "| K | " + " | ".join(str(K) for K in sorted(base)) + " |",
           "|---|" + "---|" * len(base),
           "| us | " + " | ".join(f"{base[K]:.1f}" for K in sorted(base)) + " |\n",
           "## Tails at each configuration's best n (us)\n",
           "| S | K | " + " | ".join(["v13", "v24 two_tile", *splits_cfg]) + " | fastest | fastest, one rounding |",
           "|---|---|" + "---|" * (len(splits_cfg) + 4)]
    fastest_rows = []
    for S, K in cells:
        allc = ["v13", "v24 two_tile", *splits_cfg]
        bs = {c: best(S, K, c) for c in allc}
        have = {c: b for c, b in bs.items() if b}
        f_all = min(have, key=lambda c: have[c][0])
        f_one = min((c for c in have if c in singles), key=lambda c: have[c][0])
        dp = have.get("v13")
        sp = lambda c: f" ({dp[0] / have[c][0]:.2f}x)" if dp and have[c][0] > 0 else ""  # noqa: E731
        fastest_rows.append((S, K, f_all, have[f_all], f_one, have[f_one], dp))
        out.append(f"| {S} | {K} | " + " | ".join(fmt(bs[c]) for c in allc) +
                   f" | {f_all}{sp(f_all)} | {f_one}{sp(f_one)} |")
    out.append("\n## Comparisons (tail ratio at each one's best n: median (10th-90th percentile), and how often "
               "the first is shorter), S <= 128\n")
    out.append("| comparison | what it isolates | K | ratio |")
    out.append("|---|---|---|---|")
    comps = [
        ("v24 atomic_iz", "v24 rs bf16_pairs", "atomics against bf16 reduce-scatter, like for like"),
        ("v24 atomic_iz", "v24 rs", "atomics against fp32 reduce-scatter (one rounding)"),
        ("v24 rs bf16_pairs", "v24 rs", "bf16 partials against fp32, reduce-scatter"),
        ("v24 owner bf16_pairs", "v24 owner", "bf16 partials against fp32, owner"),
        ("v24 owner chunk", "v24 owner", "chunk-major against tile-major, owner"),
        ("v24 rs chunk", "v24 rs", "chunk-major against tile-major, reduce-scatter"),
        ("v24 atomic_iz chunk", "v24 atomic_iz", "chunk-major against tile-major, atomics"),
        ("v24 rs l2", "v24 rs", "partials through one XCD's L2 against write-through"),
        ("v24 rs", "v20 rs", "v23's band store and power-of-two readers, against v20's slice"),
        ("v24 rs", "v24 owner", "reduce-scatter against owner (both fp32)"),
        ("v24 two_tile", "v24 rs", "two-tile (v19) against one-tile reduce-scatter"),
    ]
    for a, b, what in comps:
        for Ks in [sorted({K for _, K in cells}), *[[K] for K in sorted({K for _, K in cells})]]:
            cs = [c for c in cells if c[1] in Ks]
            out.append(f"| {a} / {b} | {what} | {'all' if len(Ks) > 1 else Ks[0]} | {summary(ratios(a, b, cs))} |")
    out.append("\n## Fastest configuration per cell, against data-parallel (v13)\n")
    out.append("| S | " + " | ".join(f"K = {K}" for K in sorted({K for _, K in cells})) + " |")
    out.append("|---|" + "---|" * len({K for _, K in cells}))
    for S in sorted({S for S, _ in cells}):
        row = []
        for K in sorted({K for _, K in cells}):
            fr = next((f for f in fastest_rows if f[0] == S and f[1] == K), None)
            if not fr:
                row.append("-")
                continue
            _, _, f_all, b_all, f_one, b_one, dp = fr
            txt = f"{f_all.removeprefix('v24 ')} {fmt(b_all)}"
            if f_one != f_all:
                txt += f"; one rounding: {f_one.removeprefix('v24 ')} {fmt(b_one)}"
            row.append(txt)
        out.append(f"| {S} | " + " | ".join(row) + " |")
    high = sorted({(S, K) for S, K, _ in tails if S > 128})
    if high:
        out.append("\n## S > 128 (tails, us)\n")
        out.append("| S | K | v13 | v24 two_tile | v24 spread |")
        out.append("|---|---|---|---|---|")
        for S, K in high:
            out.append(f"| {S} | {K} | " + " | ".join(fmt(best(S, K, c)) for c in ("v13", "v24 two_tile", "v24 spread"))
                       + " |")
    text = "\n".join(out) + "\n"
    with open(os.path.join(HERE, "results", "sweep_f.md"), "w") as f:
        f.write(text)
    print(text)


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
    {"a": lambda: run("a"), "b": lambda: run("b"), "c": lambda: run("c"), "l2": run_l2, "report": report,
     "e1": run_e1, "e1_report": report_e1, "f": lambda: run("f"), "f_report": report_f,
     "f_noise": run_f_noise, "todo": lambda: print(todo(sys.argv[2]))}[sys.argv[1]]()
