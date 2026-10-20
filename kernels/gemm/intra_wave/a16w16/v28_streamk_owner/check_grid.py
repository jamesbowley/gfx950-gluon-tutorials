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
"""The compact launch against the full grid (STREAMK_COMPACT_GRID=1 / 0), per tile, on shapes with no full
tiles, where the compact launch applies.

    source v28_streamk_owner/env.sh
    python v28_streamk_owner/check_grid.py --tile 128x256 > v28_streamk_owner/results/check_grid_128x256.md

Configurations: check_streamk.py's (every ending, partial type, reader mode and placement; the l2 exchange
with the balanced placement, STREAMK_XCD_LOCAL=0, where the host takes it, and once XCD-local, which keeps
the full grid), spread (end to end) and capped (as deployed), with and without bias, at several program
counts: S n at n = 4, an uneven split, a count that is not a multiple of 8, and all programs.

Shapes: quadrant tiles: an aligned grid of 3 x 11 tiles (S = 33, K = 8192), check_streamk.py's ragged one
(32 tiles, M, N and K ragged), and the 3 x 11 grid at 4 pairs of k-steps, where the request is capped at one
program per pair (the host passes S x pairs, fewer than requested); small tiles: check_streamk.SMALL_SHAPES, 3x4000x4080 and 1x8192 at 2 pairs (mirrored for a swapped
tile), those with fewer tiles than programs.

Each configuration runs both modes on the same inputs, interleaved, `--launches` times each. Asserts:
- C is bit-identical between the modes (atomic_iz, whose add order is not deterministic, and bf16
  partials: both modes within check_streamk.py's tolerance against an fp32 reference; everything else is
  also checked against torch);
- every flag and count is re-armed after every launch of either mode;
- placement (STREAMK_RECORD_PLACEMENT=1, two recorded launches per mode): both modes pass the same
  STREAMK_NUM_PROGRAMS, and the compact grid is exactly that many programs; in either mode (balanced, not
  XCD-local) the working programs are hardware ids 0 .. STREAMK_NUM_PROGRAMS - 1, within 1 of each other per
  XCD; their (spid, tile, role, chunk) agree per hardware pid between the modes and between launches;
- hardware dispatch: in every recorded launch, workgroup p runs on XCC (p + r) % 8 for one rotation r.
  r = 0 is v21's assumption A1; r != 0 still keeps every p % 8 class on one XCD. Reported: the rotations
  seen, and per XCD the distinct CUs (from HW_REG_HW_ID) against the working programs.
Like check_streamk.py, cases compile in parallel (--jobs) and launch under a per-GPU file lock.
"""

import argparse
import collections
import fcntl
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from v28_streamk_owner import check_streamk as cs  # noqa: E402

RESULT_TAG = "GRID_RESULT "
N_INPUTS = 2
ODD_COUNTS = (165, 200)
OPTION_KEYS = cs.OPTION_KEYS + ("STREAMK_COMPACT_GRID", "STREAMK_RECORD_PLACEMENT")


def cu_key(xcc, hw):
    """XCC, SE, SH and CU from HW_REG_XCC_ID / HW_REG_HW_ID (experiments/memory_model_litmus)."""
    return xcc * 256 + ((hw >> 13) & 7) * 32 + ((hw >> 12) & 1) * 16 + ((hw >> 8) & 15)


def grid_shapes(geo):
    bm, bn, bk, np_ = geo["block_m"], geo["block_n"], geo["block_k"], geo["num_programs"]
    if geo["small"]:
        shapes = list(cs.SMALL_SHAPES) + [(3, 4000, 4080), (1, 8192, 4 * bk)]
        shapes = [(n, m, k) for m, n, k in shapes] if geo["swap"] else shapes
    else:
        ragged = cs.quadrant_shapes(geo)[4]
        shapes = [(3 * bm, 11 * bn, 8192), ragged, (3 * bm, 11 * bn, 8 * bk)]
    out = []
    for M, N, K in shapes:
        m, n = cs.kernel_dims(geo, M, N)
        tiles = -(-m // bm) * -(-n // bn)
        if 0 < tiles < np_:
            out.append((M, N, K))
    return out


def program_counts(geo, M, N):
    s = cs.leftover(geo, M, N)
    out = []
    for snp in (4 * s, 4 * s + s // 2) + ODD_COUNTS:
        if s <= snp < geo["num_programs"] and snp not in out:
            out.append(snp)
    return out + [None]


def configurations(geo):
    """(label, env, by_mean) of every configuration, without the program count."""
    args = argparse.Namespace(policies=["one_tile", "spread"], fixups=["rs", "owner", "atomic_iz"],
                              partials=["fp32", "bf16", "bf16_pairs"], readers=["pow2", "all"],
                              placements=["tile", "chunk"], exchanges=["wt", "l2"])
    out = []
    for env, by_mean in cs.configurations(args, geo):
        if env.get("STREAMK_EXCHANGE") == "l2":
            out.append(({**env, "STREAMK_XCD_LOCAL": "0"}, by_mean))
            if env["STREAMK_PARTIALS"] == "fp32" and env["STREAMK_RS_READERS"] == "pow2":
                out.append(({**env, "STREAMK_XCD_LOCAL": "1"}, by_mean))
        else:
            out.append((env, by_mean))
    out.append(({"STREAMK_POLICY": "capped"}, False))
    return out


def records(rec):
    """The working programs of a recorded launch: {pid: (spid, tile, role, a0, a1, xcc, hw)}."""
    out = {}
    for row in rec.tolist():
        if row[0] > 0:
            out[row[0] - 1] = tuple(row[1:])
    return out


def check_case(case):
    import bench_ext as bench  # noqa: F401  (paths)
    import torch

    import importlib

    mk = importlib.import_module(os.environ.get("V28_MATMUL_MODULE", "v28_streamk_owner.matmul_kernel"))

    tile, (M, N, K), bias, by_mean = case["tile"], case["shape"], case["bias"], case["by_mean"]
    geo = mk.tile_geometry(tile)
    np_ = geo["num_programs"]
    bm, bn = mk.parse_spec(tile)[:2]
    reps = -(-1024 // (min(M, bm) * min(N, bn)))
    bm, bn = (bm, bn * reps) if bn >= bm else (bm * reps, bn)
    dt = torch.bfloat16
    inputs = []
    for _ in range(N_INPUTS):
        a = torch.rand((M, K), device=bench.DEVICE, dtype=dt) - 0.5
        b = torch.rand((N, K), device=bench.DEVICE, dtype=dt).T - 0.5
        bias_n = torch.rand((N,), device=bench.DEVICE, dtype=dt) - 0.5 if bias else None
        ref32 = torch.matmul(a.float(), b.float()) + (bias_n.float() if bias else 0)
        rnd = cs.tile_mean((ref32.to(dt).float() - ref32).abs(), bm, bn)
        inputs.append((a, b, bias_n, ref32, rnd))

    def set_mode(compact, record):
        os.environ["STREAMK_COMPACT_GRID"] = str(compact)
        os.environ["STREAMK_RECORD_PLACEMENT"] = str(record)

    # Compile all four builds (mode x recorder) without launching, then hold the GPU.
    mk.WARMUP = True
    for compact, record in itertools.product((0, 1), (0, 1)):
        set_mode(compact, record)
        try:
            mk.matmul(inputs[0][0], inputs[0][1], bias=inputs[0][2], tile=tile)
        except AssertionError as e:
            if "a tile spans XCDs" in str(e):
                return {"skip": "host: l2 tile spans XCDs"}
            raise
    mk.WARMUP = False
    torch.cuda.synchronize()
    lock = open(f"/tmp/v27_check_gpu{os.environ.get('HIP_VISIBLE_DEVICES', 'all')}.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    _, locks, tile_sync, _ = mk.streamk_workspace(bench.DEVICE, np_, *geo["slot"])

    def flags_up():
        torch.cuda.synchronize()
        return (int((locks != 0).sum()) + int((tile_sync[:2 * np_:2] != 0).sum())
                + int((tile_sync[3 * np_:] != 0).sum()))

    def wrong(c, i):
        a, b, bias_n, ref32, rnd = inputs[i]
        if by_mean:
            n_max = min((mk.LAST_LAUNCH["streamk_programs"] or np_) // max(mk.LAST_LAUNCH["streamk_tiles"], 1) + 1,
                        -(-K // (2 * geo["block_k"])))
            err = cs.tile_mean((c.float() - ref32).abs(), bm, bn)
            return bool(torch.isnan(c).any() or (err > (1 + n_max ** 0.5) * rnd + 1e-3).any())
        return not torch.allclose(c, ref32.to(dt), atol=1e-1, rtol=1e-2)

    def launch(compact, record, i):
        set_mode(compact, record)
        a, b, bias_n = inputs[i][:3]
        c = torch.full((M, N), float("nan"), device=bench.DEVICE, dtype=dt)
        mk.matmul(a, b, c, bias=bias_n, tile=tile)
        launched = dict(mk.LAST_LAUNCH)
        return c, launched, wrong(c, i), flags_up()

    res = dict(mismatch=0, wrong=0, flags_up=0, launches=0, exact=True)
    info = {}
    for i in range(case["launches"]):
        out = {}
        for compact in (0, 1):
            c, launched, bad, up = launch(compact, 0, i % N_INPUTS)
            out[compact] = c
            info[compact] = launched
            res["wrong"] += bad
            res["flags_up"] += up
            res["launches"] += 1
        same = torch.equal(out[0], out[1])
        res["exact"] &= same
        if not by_mean:
            res["mismatch"] += not same
    for k in ("kernel", "grid", "compact", "requested_programs", "streamk_programs", "streamk_tiles", "policy",
              "fixup"):
        res[f"{k}_full"] = info[0].get(k)
        res[f"{k}_compact"] = info[1].get(k)
    if info[1].get("kernel") != "v28_streamk_owner":
        fcntl.flock(lock, fcntl.LOCK_UN)
        return res

    # Placement: two recorded launches per mode, interleaved.
    recs = {0: [], 1: []}
    for rep in range(2):
        for compact in (0, 1):
            c, launched, bad, up = launch(compact, 1, rep % N_INPUTS)
            res["wrong"] += bad
            res["flags_up"] += up
            recs[compact].append((records(launched["placement"].cpu()), launched["grid"]))
    fcntl.flock(lock, fcntl.LOCK_UN)

    problems = []
    applied = bool(info[1]["compact"])
    snp = info[1]["streamk_programs"]
    xcd_local = case["env"].get("STREAMK_EXCHANGE") == "l2" and case["env"].get("STREAMK_XCD_LOCAL") == "1"
    rotations = []
    cu_rows = []
    if info[0]["grid"] != info[0]["num_programs"]:
        problems.append(f"STREAMK_COMPACT_GRID=0 launched {info[0]['grid']} programs")
    if info[1]["grid"] != (snp if applied else info[1]["num_programs"]):
        problems.append(f"compact={applied}, but launched {info[1]['grid']} programs ({snp} working)")
    if info[0]["streamk_programs"] != snp:
        problems.append("the modes pass different STREAMK_NUM_PROGRAMS")
    for compact in (0, 1):
        for rec, grid in recs[compact]:
            if not rec:
                problems.append(f"mode {compact}: no records")
                continue
            # Balanced (not XCD-local), the working programs are exactly hardware ids 0 .. STREAMK_NUM_PROGRAMS - 1
            # in either mode: every stream-K program works, spread within 1 over the XCDs.
            if not xcd_local and set(rec) != set(range(snp)):
                problems.append(f"mode {compact}: working pids are not 0 .. {snp - 1} ({len(rec)}, max {max(rec)})")
            rot = {(v[5] - p) % 8 for p, v in rec.items()}
            if len(rot) != 1:
                problems.append(f"mode {compact}: XCC is not a rotation of p % 8 ({sorted(rot)})")
            rotations += sorted(rot)
            per_xcd = collections.Counter(v[5] for v in rec.values())
            cus = collections.defaultdict(set)
            per_cu = collections.Counter()
            for p, v in rec.items():
                cus[v[5]].add(cu_key(v[5], v[6]))
                per_cu[cu_key(v[5], v[6])] += 1
            cu_rows.append(dict(mode=compact, grid=grid, working=len(rec), per_xcd=[per_xcd[x] for x in range(8)],
                                cus=[len(cus[x]) for x in range(8)], max_per_cu=max(per_cu.values())))
            if not xcd_local and max(per_xcd.values()) - min(per_xcd[x] for x in range(8)) > 1:
                problems.append(f"mode {compact}: per-XCD working counts {[per_xcd[x] for x in range(8)]}")
    strip = lambda rec: {p: v[:5] for p, v in rec.items()}  # noqa: E731
    if strip(recs[0][0][0]) != strip(recs[1][0][0]):
        problems.append("the working programs' (pid, spid, tile, role, chunk) differ between the modes")
    for compact in (0, 1):
        if strip(recs[compact][0][0]) != strip(recs[compact][1][0]):
            problems.append(f"mode {compact}: placement differs between launches")
    res.update(problems=problems, rotations=rotations, cu=cu_rows)
    return res


def run_case(case, timeout):
    env = dict(os.environ)
    for k in OPTION_KEYS:
        env.pop(k, None)
    env.update(case["env"])
    try:
        proc = subprocess.run([sys.executable, "-W", "ignore", os.path.abspath(__file__), "--worker", json.dumps(case)],
                              env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": f"no result after {timeout} s (hang)"}
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_TAG):
            return json.loads(line[len(RESULT_TAG):])
    tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
    return {"error": " / ".join(tail) or f"exit {proc.returncode}"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tile", default="256x256x64")
    ap.add_argument("--shapes", nargs="*", help="MxNxK shapes (default: grid_shapes)")
    ap.add_argument("--launches", type=int, default=3, help="unrecorded launches per mode")
    ap.add_argument("--bias", nargs="+", type=int, default=[0, 1], choices=[0, 1])
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        print(RESULT_TAG + json.dumps(check_case(json.loads(args.worker))), flush=True)
        return

    import regress

    geo, _ = cs.geometry(args.tile)
    shapes = [regress.parse_shape(s) for s in args.shapes] if args.shapes else grid_shapes(geo)
    print(f"# Compact launch against the full grid: {args.tile}\n")
    print(f"Kernel BLOCK {geo['block_m']}x{geo['block_n']}x{geo['block_k']}, {geo['w']} waves, "
          f"{geo['num_programs']} programs{', swapped (C^T = B^T A^T)' if geo['swap'] else ''}. "
          f"{args.launches} launches per mode plus 2 recorded. grid: full / compact; programs: the stream-K programs "
          "the host passes (the request capped at one per pair, S x pairs), all working; C: bit-identical (or both "
          "correct, for atomic_iz and bf16 partials); XCC rotations r (workgroup p on XCC (p + r) % 8); CUs: distinct "
          "CUs / working programs per XCD, compact launch; at most per CU.\n")
    print("| shape | configuration | STREAMK_NUM_PROGRAMS | bias | grid | programs | C | wrong | "
          "flags up | rotations | CUs (compact) | per CU | result |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    cases = []
    for M, N, K in shapes:
        for env, by_mean in configurations(geo):
            if env.get("STREAMK_FIXUP") == "atomic_iz" and N % 2:
                continue
            snps = [None] if env["STREAMK_POLICY"] == "capped" else program_counts(geo, M, N)
            for snp, bias in itertools.product(snps, args.bias):
                case_env = {**env, "STREAMK_TILE": args.tile}
                if snp is not None:
                    case_env["STREAMK_NUM_PROGRAMS"] = str(snp)
                cases.append(({"tile": args.tile, "shape": [M, N, K], "env": case_env, "by_mean": by_mean,
                               "bias": bias, "launches": args.launches}, env, snp))
    stats = collections.Counter()
    rotations = collections.Counter()
    odd_grids = set()
    with ThreadPoolExecutor(args.jobs) as ex:
        for (case, env, snp), r in zip(cases, ex.map(lambda c: run_case(c[0], args.timeout), cases)):
            M, N, K = case["shape"]
            label = ", ".join(f"{k.removeprefix('STREAMK_')}={v}" for k, v in env.items())
            head = f"| {M}x{N}x{K} | {label} | {snp or 'default'} | {case['bias']} |"
            if "error" in r:
                stats["error"] += 1
                print(f"{head} | | | | | | | | | error: {r['error']} |", flush=True)
                continue
            if "skip" in r:
                stats["skip"] += 1
                print(f"{head} | | | | | | | | | skipped ({r['skip']}) |", flush=True)
                continue
            probs = r.get("problems", [])
            fail = r["mismatch"] or r["wrong"] or r["flags_up"] or probs
            stats["fail" if fail else "pass"] += 1
            stats["compact"] += bool(r["compact_compact"])
            req, used = r["requested_programs_compact"], r["streamk_programs_compact"]
            stats["capped"] += bool(r["streamk_tiles_compact"] and req != used)
            for x in r.get("rotations", []):
                rotations[x] += 1
            comp_cu = [row for row in r.get("cu", []) if row["mode"] == 1]
            if r["compact_compact"] and r["grid_compact"] % 8:
                odd_grids.add(r["grid_compact"])
            cus = per_cu = ""
            if comp_cu:
                row = comp_cu[0]
                cus = " ".join(f"{c}/{w}" for c, w in zip(row["cus"], row["per_xcd"]))
                per_cu = max(x["max_per_cu"] for x in comp_cu)
            c_cell = "identical" if r["exact"] else ("both correct" if not r["mismatch"] else f"{r['mismatch']} differ")
            programs = (f"{used}" if req == used else f"{req} → {used}") if r["streamk_tiles_compact"] else ""
            print(f"{head} {r['grid_full']} / {r['grid_compact']} | {programs} | "
                  f"{c_cell} | {r['wrong']} | {r['flags_up']} | "
                  f"{' '.join(map(str, sorted(set(r.get('rotations', [])))))} | {cus} | {per_cu} | "
                  f"{'FAIL: ' + '; '.join(probs) if fail else 'ok'} |", flush=True)
    total = sum(stats[k] for k in ("pass", "fail", "error"))
    print(f"\n{total} configurations: {stats['pass']} ok, {stats['fail']} failed, {stats['error']} errors, "
          f"{stats['skip']} skipped (host rejects); compact launch in {stats['compact']}; the request capped at one "
          f"program per pair in {stats['capped']}. XCC rotations seen (launches): "
          f"{', '.join(f'r={k}: {v}' for k, v in sorted(rotations.items()))}. Compact grids not a multiple of 8: "
          f"{', '.join(map(str, sorted(odd_grids)))}.")
    ok = stats["fail"] == 0 and stats["error"] == 0
    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
