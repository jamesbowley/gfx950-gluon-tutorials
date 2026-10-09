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
"""Check every GEMM_TILE_LIMIT case of sweep C before it is timed.

    HIP_VISIBLE_DEVICES=0 python experiments/streamk_split_count/check_tile_limit.py [--jobs 8]
    HIP_VISIBLE_DEVICES=0 python experiments/streamk_split_count/check_tile_limit.py --sweep e1   # sweep E1's cases

Each case (version, policy, split, K, T = 256 + S) runs 6 launches over 3 rotating inputs, with
B and C padded as bench.py pads them (so the compiled kernels are the ones the sweep times). C is
filled with NaN before every launch, and the launch must write exactly the tiles whose ids are
below T (GROUP_SIZE_M = 4 swizzle over the full 16 x 32 grid), each whole, matching torch, and
leave every other tile NaN. v22's atomic ending rounds once per program, and v23 with bf16
partials (STREAMK_PARTIALS=bf16, bf16_pairs) rounds every partial once, so their stream-K tiles
are held to an fp32 reference by their mean error instead (atomic_mean_factor). After the launches the stream-K flags and the tile barriers' counts
must be zero (re-armed). One worker process per (version, K); the workers share the GPU.
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
A16 = os.path.join(HERE, "..", "..", "kernels", "gemm", "intra_wave", "a16w16")
TAG = "CHECK_CASE "
NAMES = {13: "v13_persistant_peel_acc", 16: "v16_streamk_lane_partials", 19: "v19_streamk_two_tile_reversed",
         20: "v20_streamk_reduce_scatter", 22: "v22_streamk_atomic_c", 23: "v23_streamk_rs_band",
         24: "v24_streamk_combined"}


def atomic_mean_factor(n):
    """v22's atomic ending rounds C once per program of a tile, so its stream-K tiles are held to the
    fp32 reference by their mean error: at most this many times the mean error of one bf16 rounding
    of that reference. n roundings measured 0.5 + 0.75 sqrt(n) times (atomic_accuracy.md); a doubled
    partial measured about 700 times, and a missing or misplaced one moves a tile by a partial's size."""
    return 1 + n ** 0.5


def tile_position(t, tiles_m, tiles_n, group=4):
    """(m, n) of tile id t under the GROUP_SIZE_M swizzle (get_group_m_tile_ids)."""
    per_group = group * tiles_n
    first_m = (t // per_group) * group
    size_m = min(tiles_m - first_m, group)
    return first_m + (t % per_group) % size_m, (t % per_group) // size_m


def expected_tiles(T, tiles_m, tiles_n, group=4):
    """Boolean [tiles_m, tiles_n]: the tiles with ids below T under the GROUP_SIZE_M swizzle."""
    out = [[False] * tiles_n for _ in range(tiles_m)]
    for t in range(T):
        m, n = tile_position(t, tiles_m, tiles_n, group)
        out[m][n] = True
    return out


def streamk_tiles(T, S, tiles_m, tiles_n, group=4, num_xcds=8):
    """Boolean [tiles_m, tiles_n]: the S one-tile stream-K tiles of a T-tile GEMM in the v9 order,
    as the kernels' streamk_tile_id finds them (the top of every XCD's block of tile ids)."""
    out = [[False] * tiles_n for _ in range(tiles_m)]
    per = (T + num_xcds - 1) // num_xcds
    tall = T % num_xcds or num_xcds
    l0 = (T - S) // num_xcds
    n_tall, n_short = per - l0, max(per - 1 - l0, 1)
    head = tall * n_tall
    for t in range(S):
        if t < head:
            tid = (t // n_tall) * per + l0 + t % n_tall
        else:
            t2 = t - head
            tid = tall * per + (t2 // n_short) * (per - 1) + l0 + t2 % n_short
        m, n = tile_position(tid, tiles_m, tiles_n, group)
        out[m][n] = True
    return out


def rounds_per_partial(case):
    """Whether the case's stream-K tiles round once per program (held to the mean-error test)."""
    if case["env"].get("STREAMK_FIXUP") in ("atomic", "atomic_iz"):
        return True
    return case["env"].get("STREAMK_PARTIALS", "fp32") != "fp32"


def worker(v, K, cases):
    sys.path.insert(0, A16)
    os.chdir(A16)
    import importlib

    import bench  # noqa: F401  (plugin hooks and the path to common.py)
    import torch

    sys.path.insert(0, HERE)
    import sweep

    mk = importlib.import_module(f"{NAMES[v]}.matmul_kernel")
    M, N = sweep.MATRIX_C
    tm, tn = M // 256, N // 256
    dtype = torch.bfloat16
    inputs = []
    for _ in range(3):
        a = torch.rand((M, K), device=bench.DEVICE, dtype=dtype) - 0.5
        b = bench.make_b(N, K, dtype, True, lambda t: t.uniform_(-0.5, 0.5))
        c = bench.make_c(M, N, dtype, True)
        inputs.append((a, b, c, torch.matmul(a, b)))
    atomic = any(rounds_per_partial(c) for c in cases)
    refs32 = [a.float() @ b.float() for a, b, *_ in inputs] if atomic else []
    for case in cases:
        base_env = sweep.case_env(case)
        for k in sweep.ENV_KEYS:
            os.environ.pop(k, None)
            if k in base_env:
                os.environ[k] = base_env[k]
        want = torch.tensor(expected_tiles(case["limit"], tm, tn), device=bench.DEVICE)
        want_el = want.repeat_interleave(256, 0).repeat_interleave(256, 1)
        res = {"wrong": 0, "partial": 0, "extra_or_missing": 0, "flags_up": 0, "error": None}
        try:
            for i in range(6):
                a, b, c, ref = inputs[i % 3]
                c.fill_(float("nan"))
                mk.matmul(a, b, c)
                cc = c.contiguous()
                written = ~torch.isnan(cc).view(tm, 256, tn, 256)
                t_all, t_any = written.all(3).all(1), written.any(3).any(1)
                res["partial"] += int((t_all != t_any).sum())
                res["extra_or_missing"] += int((t_any != want).sum())
                if rounds_per_partial(case):
                    sk = torch.tensor(streamk_tiles(case["limit"], case["S"], tm, tn), device=bench.DEVICE)
                    fw_el = (want & ~sk).repeat_interleave(256, 0).repeat_interleave(256, 1)
                    r32 = refs32[i % 3]
                    tile_err = (cc.float() - r32).abs().view(tm, 256, tn, 256).mean((1, 3))
                    tile_rnd = (r32.to(dtype).float() - r32).abs().view(tm, 256, tn, 256).mean((1, 3))
                    n_max = 256 // case["S"] + 1 if case["n"] == "default" else case["n"]
                    n_max = min(n_max, K // 128)
                    bad_sk = (~(tile_err <= atomic_mean_factor(n_max) * tile_rnd + 1e-3) & sk).any()
                    bad_fw = not torch.allclose(torch.where(fw_el, cc, ref), ref, atol=1e-1, rtol=1e-2)
                    res["wrong"] += int(bool(bad_sk) or bad_fw)
                elif not torch.allclose(torch.where(want_el, cc, ref), ref, atol=1e-1, rtol=1e-2):
                    res["wrong"] += 1
            torch.cuda.synchronize()
            if v != 13:
                ws = mk.streamk_workspace(bench.DEVICE, 256, 256, 256)
                # The barrier counts are the even words of tile_sync's first 2 x 256; v24 follows them
                # with its L2 epoch flags (which hold epochs) and its atomic ending's zero counts.
                res["flags_up"] = int((ws[1] != 0).sum()) + (int((ws[2][:512:2] != 0).sum()) if len(ws) > 2 else 0)
                if v == 24:
                    res["flags_up"] += int((ws[2][768:] != 0).sum())
        except Exception as e:  # noqa: BLE001  (report and go on to the next case)
            res["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
        out = {k: case[k] for k in ("S", "K", "config", "n", "snp", "limit")} | res
        print(TAG + json.dumps(out), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--timeout", type=int, default=5400, help="seconds per worker")
    p.add_argument("--versions", nargs="*", type=int, help="only these kernel versions (default: all)")
    p.add_argument("--sweep", default="c", choices=["c", "e1", "f"], help="the sweep whose cases to check")
    p.add_argument("--worker", help=argparse.SUPPRESS)
    args = p.parse_args()
    sys.path.insert(0, HERE)
    import sweep

    all_cases = {"c": lambda: sweep.cases_c(include_real=False), "e1": sweep.cases_e1,
                 "f": sweep.cases_f}[args.sweep]()
    cases = [c for c in all_cases if args.worker or not args.versions or c["v"] in args.versions]
    groups = {}
    for c in cases:
        groups.setdefault((c["v"], c["K"]), []).append(c)
    if args.worker:
        v, K = json.loads(args.worker)
        worker(v, K, groups[(v, K)])
        return

    def launch(vk):
        try:
            proc = subprocess.run([sys.executable, "-W", "ignore", os.path.abspath(__file__), "--sweep", args.sweep,
                                   "--worker", json.dumps(list(vk))], capture_output=True, text=True,
                                  timeout=args.timeout)
        except subprocess.TimeoutExpired as e:
            out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            return vk, out, f"timeout after {args.timeout} s (hang?)"
        return vk, proc.stdout, (proc.stderr.strip().splitlines() or [""])[-1] if proc.returncode else None

    rows, problems = [], []
    order = sorted(groups, key=lambda vk: -len(groups[vk]) * vk[1])
    with ThreadPoolExecutor(args.jobs) as ex:
        for vk, out, err in ex.map(launch, order):
            got = [json.loads(l[len(TAG):]) for l in out.splitlines() if l.startswith(TAG)]
            rows += got
            if err or len(got) != len(groups[vk]):
                problems.append(f"v{vk[0]} K = {vk[1]}: {len(got)} of {len(groups[vk])} cases reported; {err}")
            print(f"v{vk[0]} K = {vk[1]}: {len(got)} cases", file=sys.stderr, flush=True)
    bad = [r for r in rows if r["error"] or r["wrong"] or r["partial"] or r["extra_or_missing"] or r["flags_up"]]
    print(f"\n# GEMM_TILE_LIMIT check (sweep {args.sweep.upper()} cases)\n")
    print(f"{len(rows)} of {len(cases)} cases ran, 6 launches each over 3 rotating inputs, on "
          f"{sweep.MATRIX_C[0]}x{sweep.MATRIX_C[1]}xK with B and C padded as bench.py pads them.\n")
    print("| config | cases | launches with wrong tiles | partially written tiles | tiles written that should "
          "not be, or missing | flags or counts left up | errors |")
    print("|---|---|---|---|---|---|---|")
    for cfg in sorted({r["config"] for r in rows}):
        rr = [r for r in rows if r["config"] == cfg]
        print(f"| {cfg} | {len(rr)} | {sum(r['wrong'] for r in rr)} | {sum(r['partial'] for r in rr)} | "
              f"{sum(r['extra_or_missing'] for r in rr)} | {sum(r['flags_up'] for r in rr)} | "
              f"{sum(bool(r['error']) for r in rr)} |")
    for r in bad[:40]:
        print(f"- FAIL {r}")
    for pr in problems:
        print(f"- PROBLEM {pr}")
    ok = not bad and not problems and len(rows) == len(cases)
    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
